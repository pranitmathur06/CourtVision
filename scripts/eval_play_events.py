#!/usr/bin/env python3
"""Rebounds and assists read from the broadcast, scored against the feed.

WHY THIS FILE EXISTS. `check_assists.py` reports 91% and `check_rebounds.py`
reports a rate beside it, and NEITHER HAS EVER SEEN A BROADCAST. Both read
SportVU tracking coordinates: every player located to the inch, the ball's
height known, identities stable for the whole game. They measure the event
logic on perfect inputs. Nothing in this repository has ever asked whether a
rebound or an assist can be read off the pixels, which is the question.

WHAT IS ASKED, AND WHY IT IS NOT "WHO". Naming the rebounder needs identity,
and identity from a broadcast is 45% and closed -- jersey digits are legible on
13-16% of crops. Reporting a named-rebounder accuracy would report the identity
failure twice. So the questions here are the ones vision can be asked without
naming anybody, and they are the ones film work actually uses:

    rebound     OFFENSIVE or DEFENSIVE -- did the ball come back to the team
                that shot it? This is second-chance basketball, and it is a
                SAME-KIT question, which is invariant to the fact that the kit
                model does not know which team is which.
    assist      ASSISTED or NOT -- did a teammate's pass create this basket?
                Also same-kit, also unnamed.

PROVENANCE, STATED BECAUSE IT CHANGES WHAT THE NUMBER MEANS. The feed supplies
WHEN (the rebound's instant, the basket's instant); vision supplies WHO-ish
(which kit). That is the product's actual architecture -- the feed says what
happened, vision says where and to whom -- and it is the mode the ladder in
`score_game_end_to_end.py` calls feed-assisted. A pure-vision timing arm is a
different measurement and is not this one.

THE BASELINE IS PRINTED BESIDE EVERY NUMBER AND THE NUMBER IS MEANINGLESS
WITHOUT IT. About 71% of rebounds are defensive, so a system that says
"defensive" every time and looks at nothing scores 0.71. An accuracy that does
not clear its own majority class has learned nothing from the pixels. This is
the trap the first vision-card measurement fell into -- 0.723 against a base
rate of 0.664 -- and it is worth the line.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.games import get, registry  # noqa: E402
from courtvision.kits import (  # noqa: E402
    KitModel,
    MIN_BOX_CONF,
    sample_clip,
    torso_lab,
)
from courtvision.stats import iou, wilson  # noqa: E402

REBOUND = re.compile(r"^(.*?)\s+REBOUND \(Off:(\d+) Def:(\d+)\)")

#: A ball this far from a player's box EDGE, in units of that player's own box
#: height, is not in his hands. Scale-free on purpose: a player at the far
#: sideline is a third the pixels of one under the basket, and a gate in pixels
#: would hold the near player to a stricter standard than the far one.
HOLD_GATE = 0.45
#: How far either side of the logged rebound to look for the securing player. A
#: rebound is logged when the ball is SECURED, and `check_rebounds.py` measured
#: on one game that the credited player has it a second or more BEFORE the
#: logged moment far more often than after, so the window straddles it.
REBOUND_BEFORE_S, REBOUND_AFTER_S = 0.5, 2.0
#: The ball must have left for the rim before anybody can be rebounding it.
#: Without this the "securing player" found before the logged instant is
#: usually the SHOOTER still holding it, every rebound reads as offensive, and
#: the arm scores 0.31 against a 0.79 majority -- worse than saying nothing,
#: which is how the bug announced itself.
SHOT_CLEAR_S = 0.6
#: How far back from a made basket a pass could have created it.
PASS_WINDOW_S = 3.0
#: Two holder boxes this similar are the same player, so the ball did not move
#: between hands. Below it, it did.
SAME_PLAYER_IOU = 0.3
#: A pass moves the ball at least this far, in the holder's own box heights.
#: A handoff at arm's length is not what the feed calls an assist.
PASS_TRAVEL = 1.2
#: Clips sampled to fit the kit centres.
KIT_FIT_ROWS = 3


def to_box(point, box) -> float:
    """Distance from a point to the nearest edge of a box; 0 inside it."""
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def rebound_truth(events) -> list[tuple[dict, str]]:
    """[(event, 'off'|'def')] from the feed's own running per-player counters.

    Differencing the counters is roster-free, which matters: two of the four
    broadcasts have no roster file on disk, and a truth that needs one would be
    a truth that cannot follow a new broadcast.
    """
    seen: dict[str, tuple[int, int]] = {}
    out = []
    for event in events:
        if event.get("action") != "Rebound":
            continue
        matched = REBOUND.match((event.get("description") or "").strip())
        if not matched:
            continue  # a team rebound: nobody to attribute, so not scored
        who, offensive, defensive = matched.group(1), int(matched.group(2)), int(matched.group(3))
        had_off, had_def = seen.get(who, (0, 0))
        seen[who] = (offensive, defensive)
        if offensive > had_off:
            out.append((event, "off"))
        elif defensive > had_def:
            out.append((event, "def"))
    return out


def assist_truth(events) -> list[tuple[dict, bool]]:
    """[(made-shot event, was it assisted)] -- the feed logs an Assist row
    beside the basket it created, at the same instant."""
    assisted = {round(float(e["video_s"]), 1)
                for e in events if e.get("action") == "Assist"}
    return [(e, round(float(e["video_s"]), 1) in assisted)
            for e in events if str(e.get("action", "")).startswith("Made Shot")]


def clip_for(index_rows, video_s: float, span: float):
    """The clip whose footage covers this instant, and where in it that is."""
    best = None
    for row in index_rows:
        # An index row with no clip is an event whose footage was never cut --
        # the cutter records the attempt either way. Treating one as coverage
        # is how this first ran, declining two rebounds in three with "no
        # securing player" when the real reason was that there was no footage.
        if not row.get("clip"):
            continue
        start = float(row["start_s"])
        if start <= video_s <= start + span:
            offset = video_s - start
            # Prefer the clip that has the most footage on BOTH sides of the
            # instant: a rebound one frame from a clip's end has no securing
            # player in it at all.
            score = min(offset, span - offset)
            if best is None or score > best[0]:
                best = (score, row, offset)
    return (best[1], best[2]) if best else (None, None)


class ClipFrames:
    """Decodes a clip frame only when a holder has already been found in it.

    The windows searched here are three seconds of a six-second clip, and the
    answer is usually in the first frame that has a ball at all, so decoding
    the whole window would be about twenty times the work for the same answer.
    """

    def __init__(self, path, source_size):
        import cv2

        self._capture = cv2.VideoCapture(str(path))
        self._cv2 = cv2
        self.ok = self._capture.isOpened()
        if self.ok:
            self.scale = (
                self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) / float(source_size[0]),
                self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / float(source_size[1]),
            )
        else:
            self.scale = (1.0, 1.0)
        self._cache: dict[int, object] = {}

    def image(self, frame: int):
        if frame not in self._cache:
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = self._capture.read()
            self._cache[frame] = image if ok else None
        return self._cache[frame]

    def colour(self, frame: int, box):
        image = self.image(frame)
        if image is None:
            return None
        scaled = [box[0] * self.scale[0], box[1] * self.scale[1],
                  box[2] * self.scale[0], box[3] * self.scale[1]]
        return torso_lab(image, scaled)

    def close(self):
        self._capture.release()


def holders(rows, frames: ClipFrames, model, frame_lo: float, frame_hi: float):
    """[(frame, kit, box, ball centre)] for every frame in range with a holder.

    A holder is the player whose box EDGE is nearest the most confident ball,
    within a gate set in that player's own box heights. Box edge rather than
    box centre because that is what was measured: wrists from a pose model did
    not beat it, and box centre lost to it outright (Round 109).
    """
    out = []
    for row in rows:
        frame = int(row["f"])
        if frame < frame_lo or frame > frame_hi:
            continue
        balls = [b for b in row["d"] if b[0] == "b"]
        people = [list(b[2:]) for b in row["d"]
                  if b[0] == "p" and b[1] >= MIN_BOX_CONF]
        if not balls or not people:
            continue
        ball = max(balls, key=lambda b: b[1])
        centre = ((ball[2] + ball[4]) / 2.0, (ball[3] + ball[5]) / 2.0)
        nearest = min(people, key=lambda box: to_box(centre, box))
        height = nearest[3] - nearest[1]
        if height <= 0 or to_box(centre, nearest) > HOLD_GATE * height:
            continue
        kit, _margin = model.kit(frames.colour(frame, nearest))
        if kit is None:
            continue
        out.append((frame, kit, nearest, centre))
    return out


def fit_kits(broadcast, cache, names) -> KitModel | None:
    """Kit centres for this broadcast, from the even-numbered clips only."""
    colours = []
    for name in names[0::2]:
        path = ROOT / broadcast.clip_dir / name
        if not path.exists():
            continue
        for _row, _boxes, sampled in sample_clip(path, cache["clips"][name],
                                                 cache.get("source_size") or [1280, 720],
                                                 want=KIT_FIT_ROWS):
            colours.extend(c for c in sampled if c is not None)
    return KitModel.fit(colours)


def _clip_rows(cache, index_rows, event, span, fps):
    """(rows, path-relative clip name, frame of the instant) for one event."""
    row, offset = clip_for(index_rows, float(event["video_s"]), span)
    if row is None:
        return None
    name = row["clip"]
    if name not in cache["clips"]:
        return None
    return cache["clips"][name], name, offset * fps


def judge_rebound(rows, frames, model, at_frame, fps, shot_frame):
    """'off' | 'def' | None -- did the ball come back to the shooting team?"""
    securing = holders(rows, frames, model,
                       max(at_frame - REBOUND_BEFORE_S * fps,
                           shot_frame + SHOT_CLEAR_S * fps),
                       at_frame + REBOUND_AFTER_S * fps)
    if not securing:
        return None, "no securing player"
    shooting = holders(rows, frames, model,
                       shot_frame - 1.5 * fps, shot_frame + 0.1 * fps)
    if not shooting:
        return None, "no shooter"
    # The shooter is whoever had it LAST before release; the rebounder is
    # whoever has it FIRST after the ball comes off the rim.
    return ("off" if shooting[-1][1] == securing[0][1] else "def"), ""


def judge_assist(rows, frames, model, at_frame, fps):
    """True | False | None -- did a same-kit pass create this basket?"""
    seen = holders(rows, frames, model, at_frame - PASS_WINDOW_S * fps, at_frame)
    if len(seen) < 2:
        return None, "no ball in the window"
    last = seen[-1]
    for frame, kit, box, centre in reversed(seen[:-1]):
        if iou(box, last[2]) >= SAME_PLAYER_IOU:
            continue  # still the same man with it
        height = max(1.0, last[2][3] - last[2][1])
        travelled = math.hypot(centre[0] - last[3][0], centre[1] - last[3][1])
        if travelled < PASS_TRAVEL * height:
            continue
        return kit == last[1], ""
    return False, ""


def evaluate(key: str, *, limit: int | None = None) -> dict:
    broadcast = get(key)
    cache = json.loads((ROOT / broadcast.clip_detections).read_text())
    index = json.loads((ROOT / broadcast.clip_index).read_text())
    index_rows = index["clips"] if isinstance(index, dict) else index
    span = float(index.get("lead_s", 3.0)) + float(index.get("tail_s", 3.0)) \
        if isinstance(index, dict) else 6.0
    fps = float(cache.get("fps") or broadcast.fps)
    events = json.loads((ROOT / broadcast.aligned).read_text())["events"]
    names = sorted(cache["clips"])
    model = fit_kits(broadcast, cache, names)
    if model is None:
        return {"game": key, "label": broadcast.label, "fitted": False}

    misses = [e for e in events if e.get("action") == "Missed Shot"]
    reb_rows = rebound_truth(events)
    ast_rows = assist_truth(events)
    if limit:
        reb_rows, ast_rows = reb_rows[:limit], ast_rows[:limit]

    results = {"rebound": [], "assist": []}
    declined = {"rebound": Counter(), "assist": Counter()}

    for event, truth in reb_rows:
        found = _clip_rows(cache, index_rows, event, span, fps)
        if found is None:
            declined["rebound"]["no clip covers it"] += 1
            continue
        rows, name, at_frame = found
        earlier = [m for m in misses
                   if 0.0 <= float(event["video_s"]) - float(m["video_s"]) <= 5.0]
        if not earlier:
            declined["rebound"]["no logged miss before it"] += 1
            continue
        shot_frame = at_frame - (float(event["video_s"]) - float(earlier[-1]["video_s"])) * fps
        path = ROOT / broadcast.clip_dir / name
        if not path.exists():
            declined["rebound"]["clip not on disk"] += 1
            continue
        frames = ClipFrames(path, cache.get("source_size") or [1280, 720])
        try:
            said, why = judge_rebound(rows, frames, model, at_frame, fps, shot_frame)
        finally:
            frames.close()
        if said is None:
            declined["rebound"][why] += 1
            continue
        results["rebound"].append((said == truth, truth))

    for event, truth in ast_rows:
        found = _clip_rows(cache, index_rows, event, span, fps)
        if found is None:
            declined["assist"]["no clip covers it"] += 1
            continue
        rows, name, at_frame = found
        path = ROOT / broadcast.clip_dir / name
        if not path.exists():
            declined["assist"]["clip not on disk"] += 1
            continue
        frames = ClipFrames(path, cache.get("source_size") or [1280, 720])
        try:
            said, why = judge_assist(rows, frames, model, at_frame, fps)
        finally:
            frames.close()
        if said is None:
            declined["assist"][why] += 1
            continue
        results["assist"].append((said == truth, truth))

    out = {"game": key, "label": broadcast.label, "fitted": True,
           "separation_lab": model.separation()}
    for arm, asked in (("rebound", len(reb_rows)), ("assist", len(ast_rows))):
        judged = results[arm]
        hits = sum(1 for ok, _ in judged if ok)
        truths = Counter(t for _, t in judged)
        majority = max(truths.values()) / len(judged) if judged else math.nan
        low, high = wilson(hits, len(judged))
        out[arm] = {
            "asked": asked,
            "answered": len(judged),
            "coverage": len(judged) / max(1, asked),
            "accuracy": hits / len(judged) if judged else math.nan,
            "ci": [low, high],
            "majority_baseline": majority,
            "truth_mix": dict(truths),
            "declined": dict(declined[arm]),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bar", type=float, default=0.85)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    keys = args.game or list(registry())
    results = []
    print()
    print("  eval_play_events.py -- rebounds and assists off the pixels")
    print("  the feed says WHEN; vision says WHICH KIT. Held to bar "
          f"{args.bar:.2f}.")
    print()
    print(f"  {'game':<6} {'arm':<8} {'acc':>6} {'95% CI':>12} {'base':>6} "
          f"{'n':>5} {'cover':>6}  verdict")
    print("  " + "-" * 76)
    for key in keys:
        row = evaluate(key, limit=args.limit)
        results.append(row)
        if not row.get("fitted"):
            print(f"  {key:<6} no kit model could be fitted")
            continue
        for arm in ("rebound", "assist"):
            got = row[arm]
            low, high = got["ci"]
            if not got["answered"]:
                verdict = "NO DATA"
            elif low < got["majority_baseline"]:
                verdict = "NO BETTER THAN THE MAJORITY CLASS"
            elif low >= args.bar:
                verdict = "PASS"
            elif got["accuracy"] >= args.bar:
                verdict = "PASS (point)"
            else:
                verdict = "FAIL"
            print(f"  {key:<6} {arm:<8} {got['accuracy']:>6.3f} "
                  f"{low:>5.2f}-{high:<5.2f} {got['majority_baseline']:>6.3f} "
                  f"{got['answered']:>5} {got['coverage']:>6.3f}  {verdict}")
    print()
    print("  base    the majority class. An accuracy whose interval does not")
    print("          clear it has learned nothing from the pixels.")
    print("  cover   share of the feed's events this answered at all. Declining")
    print("          is not an error, but an arm that answers a third of the")
    print("          rebounds has not read the game, and the two numbers must")
    print("          be quoted together or not at all.")
    print()
    for row in results:
        if not row.get("fitted"):
            continue
        for arm in ("rebound", "assist"):
            declined = row[arm]["declined"]
            if declined:
                pretty = ", ".join(f"{k}: {v}" for k, v in
                                   sorted(declined.items(), key=lambda kv: -kv[1]))
                print(f"  {row['game']:<6} {arm:<8} declined -- {pretty}")
    print()
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
