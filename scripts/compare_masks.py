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


def distinct(boxes) -> int:
    kept: list = []
    for box in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if any(iou(box, other) >= SAME_PERSON_IOU for other in kept):
            continue
        kept.append(box)
    return len(kept)


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

    before_carrier, before_counts = judge(Path(args.before))
    after_carrier, after_counts = judge(Path(args.after))
    print()
    print("  compare_masks.py -- the same frames, judged by the sport")
    print(f"  before {args.before}")
    print(f"  after  {args.after}")
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
