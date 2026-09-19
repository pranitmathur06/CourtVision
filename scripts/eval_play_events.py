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

from courtvision.candidates import HOLD_GATE, to_box  # noqa: E402
from courtvision.broadcast import SourceReader, clip_starts  # noqa: E402
from courtvision.games import get, registry  # noqa: E402
from courtvision.kits import (  # noqa: E402
    KitModel,
    MIN_BOX_CONF,
    sample_broadcast,
    torso_lab,
)
from courtvision.stats import iou, wilson  # noqa: E402

REBOUND = re.compile(r"^(.*?)\s+REBOUND \(Off:(\d+) Def:(\d+)\)")

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
#: Where the cached play-by-play lives. It carries `teamTricode` on every
#: action, which is what lets one vision attribution be replaced by a
#: game-level fact.
PBP_CACHE = "data/pbp_cache"
#: A shooter vote this one-sided is worth putting into the kit-to-team anchor.
#: Below it the window is a scramble and the vote is noise.
ANCHOR_MIN_SHARE = 0.75


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


class BroadcastFrames:
    """The SOURCE frames of one clip, for reading a torso colour.

    This used to decode the 854x480 published clip and scale every box down to
    it. The boxes are in the broadcast's own pixels and so is everything else
    in this file, so reading the broadcast removes a conversion as well as a
    resolution loss -- and the resolution loss is worth three points of a real
    metric elsewhere in this repository. See `courtvision.broadcast` for why
    the frame mapping is copied from the pipeline rather than computed.

    The frames wanted are known before any is read, so they come back in one
    forward pass over the clip's own six seconds rather than a seek apiece.
    """

    def __init__(self, reader, start_s, rows, step, wanted_frames):
        positions = {position: int(row["f"])
                     for position, row in enumerate(rows)
                     if int(row["f"]) in wanted_frames}
        self._images = {frame: image for frame, image
                        in reader.frames(start_s, positions, step)}

    def colour(self, frame, box):
        image = self._images.get(int(frame))
        return None if image is None else torso_lab(image, box)

    def close(self):
        self._images.clear()


def holders(rows, frames, model, frame_lo: float, frame_hi: float):
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


def team_codes(broadcast) -> dict[str, str]:
    """{event description: team tricode} from the cached play-by-play.

    Joined on the description rather than the clock because the two agree
    exactly -- 563 of 563 on the held-out broadcast -- and because a clock join
    would have to reproduce the period arithmetic the aligner already did.
    """
    path = ROOT / PBP_CACHE / f"{broadcast.game_id}.json"
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    actions = blob["game"]["actions"] if "game" in blob else blob
    return {(a.get("description") or "").strip(): a.get("teamTricode")
            for a in actions if a.get("teamTricode")}


def vote(seen) -> tuple[int | None, float]:
    """The kit most of a window's frames agree on, and how one-sided it was.

    `holders` returns one attribution per frame and each is right about two
    times in three. Taking the FIRST of them throws the other twenty away and
    inherits the single-frame error rate; taking the majority does not. This is
    the cheapest variance reduction available anywhere in this file.
    """
    if not seen:
        return None, 0.0
    counted = Counter(kit for _frame, kit, _box, _ball in seen)
    kit, hits = counted.most_common(1)[0]
    return kit, hits / len(seen)


def judge_rebound_anchored(rows, frames, model, at_frame, fps, shot_frame,
                           anchor, shooting_team):
    """'off' | 'def' | None, using ONE vision attribution instead of two.

    The old arm asked vision who shot AND who rebounded and compared the two
    kits, so a 0.66 attribution entered the answer twice. The feed already
    knows which team shot. All vision has to supply is the rebounder's kit,
    and the kit-to-team map is estimated once per broadcast over every made
    basket rather than per event.
    """
    if not anchor or shooting_team is None:
        return None, "no anchor"
    securing = holders(rows, frames, model,
                       max(at_frame - REBOUND_BEFORE_S * fps,
                           shot_frame + SHOT_CLEAR_S * fps),
                       at_frame + REBOUND_AFTER_S * fps)
    kit, _share = vote(securing)
    if kit is None:
        return None, "no securing player"
    team = anchor.get(kit)
    if team is None:
        return None, "kit not in the anchor"
    return ("off" if team == shooting_team else "def"), ""


def fit_kits(broadcast, cache, names, reader, starts, step) -> KitModel | None:
    """Kit centres for this broadcast, from the even-numbered clips only.

    Read from the SOURCE, like everything else here: a torso sampled off an
    854x480 clip is a coarser colour than the detector saw.
    """
    colours = []
    for name in names[0::2]:
        if name not in starts:
            continue
        for _row, _boxes, sampled in sample_broadcast(
                reader, starts[name], cache["clips"][name], step,
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
    span = (float(index.get("lead_s", 3.0)) + float(index.get("tail_s", 3.0))
            if isinstance(index, dict) else 6.0)
    fps = float(cache.get("fps") or broadcast.fps)
    events = json.loads((ROOT / broadcast.aligned).read_text())["events"]
    names = sorted(cache["clips"])
    codes = team_codes(broadcast)
    starts = clip_starts(ROOT / broadcast.clip_index)
    step = max(1, int(cache.get("step") or 2))
    reader = SourceReader(ROOT / broadcast.video)
    if not reader.ok:
        return {"game": key, "label": broadcast.label, "fitted": False}
    model = fit_kits(broadcast, cache, names, reader, starts, step)
    if model is None:
        reader.close()
        return {"game": key, "label": broadcast.label, "fitted": False}

    misses = [e for e in events if e.get("action") == "Missed Shot"]
    reb_rows = rebound_truth(events)
    ast_rows = assist_truth(events)
    if limit:
        reb_rows, ast_rows = reb_rows[:limit], ast_rows[:limit]

    results = {"rebound": [], "rebound (anchored)": [], "assist": []}
    declined = {"rebound": Counter(), "rebound (anchored)": Counter(),
                "assist": Counter()}
    contingency: Counter = Counter()

    def open_clip(event, arm):
        found = _clip_rows(cache, index_rows, event, span, fps)
        if found is None:
            declined[arm]["no clip covers it"] += 1
            return None
        rows, name, at_frame = found
        if name not in starts:
            declined[arm]["no start time for the clip"] += 1
            return None
        return rows, at_frame, name

    def open_frames(name, rows, lo, hi):
        """The SOURCE frames a window will ask about, read in one pass."""
        wanted = {int(row["f"]) for row in rows
                  if lo <= int(row["f"]) <= hi}
        return BroadcastFrames(reader, starts[name], rows, step, wanted)

    # Pass one: every made basket. Each opens its clip once and pays for two
    # things -- the assist answer, and one vote towards the kit-to-team anchor.
    for event, truth in ast_rows:
        opened = open_clip(event, "assist")
        if opened is None:
            continue
        rows, at_frame, name = opened
        frames = open_frames(name, rows, at_frame - PASS_WINDOW_S * fps,
                             at_frame + 0.2 * fps)
        try:
            said, why = judge_assist(rows, frames, model, at_frame, fps)
            shooting = holders(rows, frames, model,
                               at_frame - 1.0 * fps, at_frame + 0.1 * fps)
        finally:
            frames.close()
        kit, share = vote(shooting)
        team = codes.get((event.get("description") or "").strip())
        if kit is not None and team and share >= ANCHOR_MIN_SHARE:
            contingency[(kit, team)] += 1
        if said is None:
            declined["assist"][why] += 1
        else:
            results["assist"].append((said == truth, truth))

    # The anchor: for each kit, the team it voted for most often over the whole
    # broadcast. Estimated over some seventy baskets, so it survives a
    # per-event attribution that is right two times in three, which is the
    # entire point of moving the question here.
    anchor: dict[int, str] = {}
    for kit in (0, 1):
        options = {team: n for (k, team), n in contingency.items() if k == kit}
        if options:
            anchor[kit] = max(options, key=options.get)
    if len(set(anchor.values())) < 2:
        anchor = {}          # both kits voted for one team: no map at all

    for event, truth in reb_rows:
        prior = [m for m in misses
                 if 0.0 <= float(event["video_s"]) - float(m["video_s"]) <= 5.0]
        if not prior:
            for arm in ("rebound", "rebound (anchored)"):
                declined[arm]["no logged miss before it"] += 1
            continue
        opened = open_clip(event, "rebound")
        if opened is None:
            declined["rebound (anchored)"]["no clip covers it"] += 1
            continue
        rows, at_frame, name = opened
        shot_frame = at_frame - (float(event["video_s"])
                                 - float(prior[-1]["video_s"])) * fps
        shooting_team = codes.get((prior[-1].get("description") or "").strip())
        frames = open_frames(name, rows, shot_frame - 2.0 * fps,
                             at_frame + REBOUND_AFTER_S * fps)
        try:
            said, why = judge_rebound(rows, frames, model, at_frame, fps, shot_frame)
            anchored, anchored_why = judge_rebound_anchored(
                rows, frames, model, at_frame, fps, shot_frame,
                anchor, shooting_team)
        finally:
            frames.close()
        for arm, answer, reason in (("rebound", said, why),
                                    ("rebound (anchored)", anchored, anchored_why)):
            if answer is None:
                declined[arm][reason] += 1
            else:
                results[arm].append((answer == truth, truth))

    reader.close()
    out = {"game": key, "label": broadcast.label, "fitted": True,
           "separation_lab": model.separation(),
           "anchor": {str(k): v for k, v in anchor.items()},
           "anchor_votes": {f"{k}:{t}": n for (k, t), n in contingency.items()}}
    asked_by_arm = {"rebound": len(reb_rows), "rebound (anchored)": len(reb_rows),
                    "assist": len(ast_rows)}
    for arm, asked in asked_by_arm.items():
        judged = results[arm]
        hits = sum(1 for ok, _ in judged if ok)
        truths = Counter(t for _, t in judged)
        majority = max(truths.values()) / len(judged) if judged else math.nan
        low, high = wilson(hits, len(judged))
        per_class = {}
        for label in truths:
            rows = [ok for ok, t in judged if t == label]
            per_class[str(label)] = {
                "n": len(rows),
                "accuracy": sum(rows) / len(rows) if rows else math.nan,
            }
        out[arm] = {
            "per_class": per_class,
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
    print(f"  {'game':<6} {'arm':<18} {'acc':>6} {'95% CI':>12} {'base':>6} "
          f"{'n':>5} {'cover':>6}  verdict")
    print("  " + "-" * 86)
    for key in keys:
        row = evaluate(key, limit=args.limit)
        results.append(row)
        if not row.get("fitted"):
            print(f"  {key:<6} no kit model could be fitted")
            continue
        for arm in ("rebound", "rebound (anchored)", "assist"):
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
            print(f"  {key:<6} {arm:<18} {got['accuracy']:>6.3f} "
                  f"{low:>5.2f}-{high:<5.2f} {got['majority_baseline']:>6.3f} "
                  f"{got['answered']:>5} {got['coverage']:>6.3f}  {verdict}")
            classes = "  ".join(
                f"{label} {stat['accuracy']:.3f} (n={stat['n']})"
                for label, stat in sorted(got["per_class"].items()))
            if classes:
                print(f"  {'':<6} {'':<18} by class: {classes}")
    print()
    print("  by class  THE diagnostic. An arm that reads one class well and the")
    print("            other badly is not a weak answer, it is a different")
    print("            question being answered: a rebound arm at 0.67 on")
    print("            offensive and 0.27 on defensive is naming the SHOOTING")
    print("            team every time, and the blend hides that completely.")
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
        for arm in ("rebound", "rebound (anchored)", "assist"):
            declined = row[arm]["declined"]
            if declined:
                pretty = ", ".join(f"{k}: {v}" for k, v in
                                   sorted(declined.items(), key=lambda kv: -kv[1]))
                print(f"  {row['game']:<6} {arm:<18} declined -- {pretty}")
    print()
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
