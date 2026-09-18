"""How wrong is the floor mask? Ten players, and nobody had to label anything.

THE CONSTRAINT. From tip-off to the final buzzer there are exactly **ten players
on the court**, plus at most three referees. That is a fact about basketball, it
holds on every broadcast ever made, and it is a ground truth this project can
check itself against without a single human judgement.

`candidates.stands_on_court` decides whether a detected person's feet are on the
floor, and everything downstream depends on it: a player it rejects cannot be
the handler, cannot be tracked, and cannot anchor a ball candidate. Nothing had
ever measured it. The acceptance run on a fourth broadcast made it visible only
because the overlay line printed "players per frame p50 4" where the tracker's
own docstring says 9 to 10.

TWO ERRORS, AND THEY POINT IN OPPOSITE DIRECTIONS.

    over-keeping   more than 13 people kept is impossible, whatever the frame
                   shows. Those are bench, crowd or coaching staff that the mask
                   was built to remove and did not.

    under-keeping  cannot be counted from the constraint alone -- a tight shot
                   legitimately shows three players -- so it is measured where a
                   person has already pointed at somebody the pipeline missed.
                   See `--misses`.

WHY THIS MATTERS MORE THAN THE "KEPT SHARE". The share of detected people the
mask keeps reads 0.479 / 0.630 / 0.822 across three broadcasts, and the
temptation is to call 0.822 the good end. It is not. That broadcast keeps more
than thirteen people on 42.6% of its frames, which is the mask failing in the
other direction. The share alone cannot tell the two apart; ten players can.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.games import get, registry  # noqa: E402
from courtvision.candidates import HOLD_GATE, to_box  # noqa: E402
from courtvision.stats import iou, wilson  # noqa: E402

#: The confidence the labelling pages offered a player box at.
PLAYER_CONF = 0.35
#: Ten players. The referees are on the floor too and the detector's person
#: class does not tell them apart, so the bound is generous on purpose: any
#: frame above it is wrong for certain, and frames at or below it are not
#: claimed to be right.
ON_COURT = 10
REFEREES = 3
IMPOSSIBLE_ABOVE = ON_COURT + REFEREES


#: Two boxes overlapping this much are the same person seen by the two
#: detectors, not two people.
SAME_PERSON_IOU = 0.5


def _people_of(frame) -> list:
    """Person boxes of one frame, in the order the `on` mask indexes them."""
    return [b for b in frame["d"] if b[0] in ("p", "h")]


def distinct(boxes) -> int:
    """How many PEOPLE these boxes are.

    `p` and `h` are two different detectors run over the same frame, and both
    draw the same players. Counting the rows is counting most players twice:
    the median frame carries 13 `p` boxes and 2 `h` boxes on Finals G7, and
    treating that as fifteen people made 71% of frames "impossible" before any
    mask had a chance to be wrong. Round 110 made exactly this error in the
    tracking metric and it made the tracker look twice as good as it is; here
    it makes the mask look twice as bad.
    """
    kept: list = []
    for box in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if any(iou(box, other) >= SAME_PERSON_IOU for other in kept):
            continue
        kept.append(box)
    return len(kept)


def counts(game) -> tuple[list[int], list[int]]:
    """(detected people, kept people) per frame, from the clip detection cache."""
    if not game.clip_detections.exists():
        return [], []
    data = json.loads(game.clip_detections.read_text())
    detected, kept = [], []
    for frames in data.get("clips", {}).values():
        for frame in frames:
            people = _people_of(frame)
            mask = frame.get("on") or []
            detected.append(distinct([b[2:] for b in people]))
            # The mask is indexed against the UNDEDUPLICATED order, so it is
            # applied first and the survivors are deduplicated after.
            survivors = [b[2:] for n, b in enumerate(people)
                         if n >= len(mask) or mask[n]]
            kept.append(distinct(survivors))
    return detected, kept


def drops_the_ball_carrier(game) -> dict:
    """How often the mask throws away the man holding the ball.

    THE TEN-PLAYER RULE IS ONE-SIDED. Six of a kit on the court is impossible,
    so over-keeping has a hard bound; five people kept is merely suspicious,
    because a close-up or a dead ball genuinely has five people in frame. That
    left the mask's real error -- under-keeping -- with no label-free bound at
    all.

    The ball supplies one. Whoever is holding the ball is playing, so a mask
    that drops him is wrong with no appeal to how many people ought to be in
    shot. It needs no labels, no clock and no registration, and it is available
    on any broadcast the moment its detections are cached.

    It inherits the ball selector's own error: on a frame where the argmax ball
    is a logo, the "carrier" is whoever stands near that logo. So this is a
    LOWER bound on the mask's under-keeping, and the ball's own numbers are
    printed in the same report.
    """
    if not game.clip_detections.exists():
        return {}
    data = json.loads(game.clip_detections.read_text())
    frames = dropped = 0
    for clip in data.get("clips", {}).values():
        for frame in clip:
            mask = frame.get("on") or []
            balls = [b for b in frame["d"] if b[0] == "b"]
            people = _people_of(frame)
            if not balls or not people:
                continue
            ball = max(balls, key=lambda b: b[1])
            centre = ((ball[2] + ball[4]) / 2.0, (ball[3] + ball[5]) / 2.0)
            best = None
            for index, box in enumerate(people):
                edge = to_box(centre, box[2:])
                if best is None or edge < best[0]:
                    best = (edge, index, box)
            edge, index, box = best
            height = box[5] - box[3]
            if height <= 0 or edge > HOLD_GATE * height:
                continue
            frames += 1
            if index < len(mask) and not mask[index]:
                dropped += 1
    low, high = wilson(frames - dropped, frames)
    return {"carrier_frames": frames, "carrier_dropped": dropped,
            "carrier_kept": (frames - dropped) / frames if frames else float("nan"),
            "carrier_ci": [low, high]}


def report(key: str, detected: list[int], kept: list[int]) -> dict:
    n = len(kept)
    if not n:
        print(f"  {key:5} no clip detection cache")
        return {}
    over = sum(1 for k in kept if k > IMPOSSIBLE_ABOVE)
    low, high = wilson(over, n)
    spread = Counter(kept)
    print(f"  {key:5} {n:6} frames   detected p50 {statistics.median(detected):4.0f}"
          f"   kept p50 {statistics.median(kept):4.0f}")
    print(f"        IMPOSSIBLE (more than {IMPOSSIBLE_ABOVE} kept): "
          f"{over}/{n} = {over / n:.3f}  (95% CI {low:.3f}-{high:.3f})")
    print(f"        kept exactly {ON_COURT}: {spread[ON_COURT] / n:.3f}"
          f"   kept 5 or fewer: {sum(v for c, v in spread.items() if c <= 5) / n:.3f}"
          f"   worst frame kept {max(kept)}")
    return {"frames": n, "impossible": over, "impossible_rate": over / n,
            "kept_p50": statistics.median(kept),
            "detected_p50": statistics.median(detected), "worst": max(kept)}


def _cached_masks(game) -> dict[float, list]:
    """{frame time: the per-detection on-court mask}, keyed as `_cached_frames`."""
    data = json.loads(game.clip_detections.read_text())
    index = {c["clip"]: c for c in
             json.loads(game.clip_index.read_text())["clips"] if c.get("clip")}
    out = {}
    for clip, frames in data.get("clips", {}).items():
        start = index.get(clip, {}).get("start_s")
        if start is None:
            continue
        for frame in frames:
            for rate in {30.0, float(data.get("fps") or 30.0)}:
                out[round(float(start) + frame["f"] / rate, 1)] = frame.get("on") or []
    return out


def misses(game) -> dict:
    """Frames where a person pointed at a player the pipeline did not offer.

    Every `missing` row in the two label files carries a `handler_at` click --
    somebody saying "he is on screen and you drew no box for him". The question
    this answers is which half of the pipeline lost him: the DETECTOR never
    proposed a box there, or it did and the floor mask discarded it.

    THE ANSWER IS THE MASK, AND IT IS NOT CLOSE. Of the 71 clicks across the
    three labelled broadcasts, the detector had a confident box under the click
    in 67 and the floor mask had thrown away 42 of them. Four frames have no box
    at any confidence. The plan's standing item to "train the player detector on
    the 71 miss-clicks" was aimed at a failure that is mostly not the detector's.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_by_game import _cached_frames, _labels

    frames, _ = _cached_frames(game)
    masks = _cached_masks(game)
    rows = [r for r in (_labels(ROOT / "data/labels/handler_labels.json", game.label)
                        + _labels(ROOT / "data/labels/possession_labels.json",
                                  game.label))
            if (r.get("handler_verdict") or r.get("verdict")) == "missing"
            and r.get("handler_at")]
    if not rows:
        return {}
    detector, weak, masked, absent, scored = 0, 0, 0, 0, 0
    for row in rows:
        key = round(float(row["t"]), 1)
        dets = frames.get(key)
        if dets is None:
            continue
        scored += 1
        row = dict(row, _on=masks.get(key, []))
        x, y = row["handler_at"]
        people = [b for b in dets if b[0] in ("p", "h")]
        # The mask is stored per detection in the order the cache wrote them,
        # so `on` has to be indexed against that same list.
        # The click has to land INSIDE the box. A margin only moves cases from
        # "the mask dropped it" into "it was offered after all", and the
        # conclusion does not depend on it: 42 / 38 / 34 mask failures at 0, 10
        # and 20 px, against 4 / 2 / 0 frames with no box at any confidence.
        near = [(n, b) for n, b in enumerate(people)
                if b[2] <= x <= b[4] and b[3] <= y <= b[5]]
        if not near:
            absent += 1
            continue
        confident = [(n, b) for n, b in near if b[1] >= PLAYER_CONF]
        if not confident:
            weak += 1
            continue
        on = row.get("_on") or []
        kept = [1 for n, _ in confident if n >= len(on) or on[n]]
        if kept:
            detector += 1
        else:
            masked += 1
    return {"clicks": len(rows), "scored": scored,
            "no_box_at_all": absent, "below_confidence": weak,
            "dropped_by_the_floor_mask": masked, "offered_after_all": detector}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--misses", action="store_true",
                        help="also split the hand-clicked misses into detector "
                             "failures and floor-mask failures")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"\n  eval_court_mask.py -- ten players, plus at most {REFEREES} "
          f"referees, so more than {IMPOSSIBLE_ABOVE} kept is wrong for certain\n")
    out = {}
    for key in (args.game or list(registry())):
        game = get(key)
        out[key] = report(key, *counts(game))
        # Which mask built this cache, if it says. A cache written before
        # `remask_detections.py` existed says nothing, and that silence is
        # itself the answer: it was built with the shipped constant.
        if game.clip_detections.exists():
            built = json.loads(game.clip_detections.read_text())
            share = built.get("mask_erode_share")
            gate = built.get("mask_kit_max_lab")
            print("        built with: "
                  + ("the shipped constant (the cache records no mask)"
                     if share is None else
                     f"erosion {share:.4f} of frame height, kit gate "
                     f"{'off' if gate is None else f'{gate:.0f}'}"))
        carrier = drops_the_ball_carrier(game)
        if carrier.get("carrier_frames"):
            low, high = carrier["carrier_ci"]
            print(f"        the man holding the ball is KEPT: "
                  f"{carrier['carrier_frames'] - carrier['carrier_dropped']}"
                  f"/{carrier['carrier_frames']} = {carrier['carrier_kept']:.3f}"
                  f"  (95% CI {low:.3f}-{high:.3f})")
        out[key].update(carrier)
        if args.misses and out[key]:
            found = misses(game)
            if found:
                out[key]["misses"] = found
                print(f"        of {found['scored']} hand-clicked misses: "
                      f"NO BOX {found['no_box_at_all']}, "
                      f"below {PLAYER_CONF} confidence {found['below_confidence']}, "
                      f"DROPPED BY THE FLOOR MASK "
                      f"{found['dropped_by_the_floor_mask']}, "
                      f"offered after all {found['offered_after_all']}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
