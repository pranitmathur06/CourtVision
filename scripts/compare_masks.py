#!/usr/bin/env python3
"""Two detection caches, the same clips, both label-free bounds. Before/after.

`remask_detections.py` writes a new cache rather than overwriting the old one
precisely so this can exist. Both bounds come from the sport:

    over-keeping    more than thirteen people kept is impossible
    under-keeping   the man holding the ball must be kept

Neither needs a label, so this is the whole evidence a mask change is an
improvement, on any broadcast including one nobody has touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.candidates import HOLD_GATE, to_box  # noqa: E402
from courtvision.stats import iou, mcnemar, wilson  # noqa: E402

IMPOSSIBLE_ABOVE = 13
SAME_PERSON_IOU = 0.5
#: `clip_detect_raw.py` re-finds the floor every this many rows and reuses it
#: between, so a cache that records nothing was built at this cadence.
PIPELINE_COURT_EVERY = 3


def distinct(boxes) -> int:
    kept: list = []
    for box in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if any(iou(box, other) >= SAME_PERSON_IOU for other in kept):
            continue
        kept.append(box)
    return len(kept)


def provenance(path: Path) -> dict:
    """What built this cache's mask, as far as it records.

    A cache written before `remask_detections.py` existed records nothing,
    which is itself the answer: the shipped constant, at `clip_detect_raw.py`'s
    own cadence.
    """
    cache = json.loads(path.read_text())
    return {
        "erode_share": cache.get("mask_erode_share"),
        "kit_max_lab": cache.get("mask_kit_max_lab"),
        "fill_holes": cache.get("mask_fill_holes"),
        "court_every": cache.get("mask_court_every", PIPELINE_COURT_EVERY),
    }


def describe(got: dict) -> str:
    if got["erode_share"] is None:
        return (f"the shipped constant, floor every {got['court_every']} "
                f"(the cache records no mask)")
    gate = "off" if got["kit_max_lab"] is None else f"{got['kit_max_lab']:.0f}"
    fill = "on" if got["fill_holes"] in (None, True) else "off"
    return (f"erode {got['erode_share']:.4f}, kit gate {gate}, filling {fill}, "
            f"floor every {got['court_every']}")


def judge(path: Path):
    """(carrier kept per frame, count-ok per frame), keyed so two caches pair."""
    cache = json.loads(path.read_text())
    carrier, counts = {}, {}
    for name, rows in sorted(cache["clips"].items()):
        for row in rows:
            people = [b for b in row["d"] if b[0] in ("p", "h")]
            if not people:
                continue
            mask = row.get("on") or []
            keep = [n >= len(mask) or mask[n] for n in range(len(people))]
            key = (name, int(row["f"]))
            counts[key] = distinct([people[n][2:] for n in range(len(people))
                                    if keep[n]]) <= IMPOSSIBLE_ABOVE
            balls = [b for b in row["d"] if b[0] == "b"]
            if not balls:
                continue
            ball = max(balls, key=lambda b: b[1])
            centre = ((ball[2] + ball[4]) / 2.0, (ball[3] + ball[5]) / 2.0)
            nearest = min(range(len(people)),
                          key=lambda i: to_box(centre, people[i][2:]))
            box = people[nearest]
            height = box[5] - box[3]
            if height > 0 and to_box(centre, box[2:]) <= HOLD_GATE * height:
                carrier[key] = keep[nearest]
    return carrier, counts


def report(tag: str, before: dict, after: dict) -> None:
    shared = sorted(set(before) & set(after))
    one = [before[k] for k in shared]
    two = [after[k] for k in shared]
    for name, arm in (("before", one), ("after", two)):
        low, high = wilson(sum(arm), len(arm))
        print(f"    {tag:<14} {name:<7} {sum(arm) / max(1, len(arm)):.3f}  "
              f"({low:.3f}-{high:.3f})  n={len(arm)}")
    wins, losses, p = mcnemar(two, one)
    print(f"    {'':<14} paired  after won {wins}, lost {losses}, p = {p:.4g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    args = parser.parse_args()

    one, two = provenance(Path(args.before)), provenance(Path(args.after))
    before_carrier, before_counts = judge(Path(args.before))
    after_carrier, after_counts = judge(Path(args.after))
    print()
    print("  compare_masks.py -- the same frames, judged by the sport")
    print(f"  before {args.before}")
    print(f"         {describe(one)}")
    print(f"  after  {args.after}")
    print(f"         {describe(two)}")
    if one["court_every"] != two["court_every"]:
        print()
        print("  *** THESE WERE BUILT AT DIFFERENT FLOOR CADENCES AND THE ***")
        print("  *** COMPARISON IS NOT FAIR. The floor is found on one    ***")
        print("  *** frame and reused while the camera pans, and on one   ***")
        print("  *** broadcast going from every-1 to every-5 cost ten     ***")
        print("  *** points of kept carrier on its own. Rebuild one of    ***")
        print("  *** them at the other's cadence before reading any of    ***")
        print("  *** the numbers below.                                   ***")
    print()
    report("carrier kept", before_carrier, after_carrier)
    print()
    report("<=13 kept", before_counts, after_counts)
    print()
    print("  The two move in opposite directions by construction, so read this")
    print("  as an EXCHANGE RATE and not as a win: how many carriers the change")
    print("  buys per point of over-keeping it spends. The change is acceptable")
    print("  when the carrier side wins AND the count side stays above the")
    print("  floor fit_court_mask.py declared (0.95) -- not when the count side")
    print("  fails to lose, which it almost always will.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
