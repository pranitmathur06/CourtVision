"""Pick the frames whose ball the detector cannot find, for hand labelling.

Every automatic label source built here finds the ball only where it is already
found: the shot chart at the rim, tracking in free flight, gap-filling where a
neighbour already had it. The frames that matter are the opposite, and the only
way to get labels on them is for a person to look.

"Hard" is defined without knowing the answer, so this cannot be circular: a
frame is hard when the detector's most confident ball candidate is below
HARD_CONF. On such a frame the ball is either absent from the proposals or
buried in them, which is exactly the population the detector fails on.

Frames are spread across the game and rendered large with a measuring grid, two
to a sheet, so a centre can be read to a few pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HARD_CONF = 0.35


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--clock", default=None, help="restrict to the game's span")
    parser.add_argument("--hard-conf", type=float, default=HARD_CONF)
    parser.add_argument("--spacing-s", type=float, default=11.0)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = json.load(open(args.detections))["frames"]
    low, high = -1e9, 1e9
    if args.clock:
        readings = json.load(open(args.clock))["readings"]
        low, high = min(r["t"] for r in readings), max(r["t"] for r in readings)

    wanted, last = [], -1e9
    for row in rows:
        t = row["t"]
        if not (low <= t <= high) or t - last < args.spacing_s:
            continue
        balls = [b for b in row["boxes"] if b["cls"] == "ball"]
        top = max((b["conf"] for b in balls), default=0.0)
        if top >= args.hard_conf:
            continue
        wanted.append({"t": t, "top_conf": top, "n_candidates": len(balls)})
        last = t
    chosen = wanted[args.skip:args.skip + args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"detections": args.detections, "hard_conf": args.hard_conf,
               "frames": chosen}, open(out, "w"), indent=0)
    print(f"{len(wanted)} hard frames in the game; {len(chosen)} written to {out}")
    if chosen:
        print(f"  top candidate confidence: median "
              f"{np.median([c['top_conf'] for c in chosen]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
