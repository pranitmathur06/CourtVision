"""Turn hand ball verdicts into a truth file, keeping the exclusions honest.

`make_ball_label_page.py` records three verdicts and only one of them is a
located ball. The other two are not noise to be dropped quietly:

    none      no ball in the picture. A frame the system must NOT claim a ball
              on, so it belongs in the false-alarm denominator, not the bin.
    unknown   the ball is in there and a person could not find it at 8x. It is
              excluded from BOTH sides, because scoring a system against truth
              nobody can establish measures the labeller, not the system.

The unknown COUNT is itself a headline measurement. Of 41 frames sampled this
way before, 28 came back unknown -- if that rate holds, it says more about what
this 1280x720 broadcast can support than any accuracy figure does, and it is
printed rather than buried.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def split(rows):
    """(located, absent, unknown) from verdict rows."""
    located = [r for r in rows if r.get("verdict") == "ball" and r.get("ball")]
    absent = [r for r in rows if r.get("verdict") == "none"]
    unknown = [r for r in rows if r.get("verdict") == "unknown"]
    return located, absent, unknown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--tolerance-px", type=float, default=28.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for path in args.labels:
        rows.extend(json.load(open(path))["frames"])
    located, absent, unknown = split(rows)
    total = len(located) + len(absent) + len(unknown)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"note": "hand-located balls; 'absent' frames carry no ball and belong in "
                       "the false-alarm denominator; 'unknown' frames are excluded from "
                       "both sides of the accuracy",
               "tolerance_px": args.tolerance_px,
               "unknown_count": len(unknown), "absent_count": len(absent),
               "frames": [{"t": r["t"], "ball": r["ball"],
                           "radius": r.get("radius", 0)} for r in located],
               "absent": [{"t": r["t"]} for r in absent],
               "unknown": [{"t": r["t"]} for r in unknown]}, open(out, "w"), indent=1)

    print(f"{total} frames judged")
    print(f"  {len(located):4d} ball located   <- the truth set the gate is scored on")
    print(f"  {len(absent):4d} no ball in frame")
    print(f"  {len(unknown):4d} present but not findable by eye ({len(unknown)/max(total,1):.0%})")
    if located:
        import numpy as np
        r = [x.get("radius", 0) for x in located]
        print(f"  ball radius px: min {min(r)} p50 {int(np.median(r))} max {max(r)}")
    if len(located) < 100:
        print(f"\n  NOTE: {len(located)} located balls is not yet enough to certify 95%. "
              f"At n=100 a true 0.95 measures 0.888-0.978; at n=13 it measures 0.667-0.986.")
    print(f"  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
