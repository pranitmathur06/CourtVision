"""How concentrated the ball candidates are by world direction.

The fixture rule needs a threshold, and a threshold chosen after seeing the
labels is a threshold fitted to the test. This picks one from the shape of the
data instead, which the labels never enter: count, for each direction cell, how
many posed frames put a ball candidate in it, and look at the distribution.

The game ball is all over the arena during a game, so no single direction
should hold it for long. Furniture -- the spare ball on the rack at the
scorer's table -- holds exactly one direction for as long as it is in shot. If
the two are separable there is a gap in the ranked counts, and the threshold
goes in the gap. If there is no gap, the rule does not have a threshold and
should not be given one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--min-conf", type=float, default=0.10)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_rim_ball import cell_of, ray_directions

    spec = json.load(open(args.camera))
    centre = np.array(spec["centre"], np.float64)
    size = tuple(spec["size"])
    k1, k2 = spec.get("k1", 0.0), spec.get("k2", 0.0)

    cache = json.load(open(args.detections))
    by_time = {round(row["t"], 3): row for row in cache["frames"]}
    cache_times = np.array(sorted(by_time)) if by_time else np.array([])

    counts: dict[tuple[int, int], int] = {}
    examples: dict[tuple[int, int], list] = {}
    posed = 0
    for row in json.load(open(args.poses))["frames"]:
        if not row.get("params"):
            continue
        posed += 1
        t = row["t"]
        if not len(cache_times):
            continue
        j = int(np.argmin(np.abs(cache_times - t)))
        if abs(cache_times[j] - t) > 0.3:
            continue
        balls = [b for b in by_time[cache_times[j]]["boxes"]
                 if b["cls"] == "ball" and b["conf"] >= args.min_conf]
        if not balls:
            continue
        pixels = [[(b["xyxy"][0] + b["xyxy"][2]) / 2,
                   (b["xyxy"][1] + b["xyxy"][3]) / 2] for b in balls]
        for direction, pixel in zip(
                ray_directions(row["params"], centre, size, pixels, k1, k2), pixels):
            cell = cell_of(direction)
            counts[cell] = counts.get(cell, 0) + 1
            examples.setdefault(cell, []).append((t, pixel))

    if not counts:
        print("no posed frames with candidates")
        return 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    print(f"{posed} posed frames; {len(counts)} direction cells occupied\n")
    print(f"{'cell':>16}  {'frames':>6}  {'share':>6}   first seen .. last seen")
    for cell, n in ranked[:args.top]:
        times = [e[0] for e in examples[cell]]
        print(f"{str(cell):>16}  {n:6d}  {n / posed:6.1%}   "
              f"{min(times):7.0f} .. {max(times):7.0f} s")
    tail = [n for _, n in ranked]
    print(f"\nranked counts: {tail[:12]}{' ...' if len(tail) > 12 else ''}")
    gaps = [(tail[i] - tail[i + 1], i) for i in range(min(len(tail) - 1, 20))]
    if gaps:
        size_of_gap, at = max(gaps)
        print(f"largest gap {size_of_gap} between rank {at + 1} ({tail[at]}) and "
              f"rank {at + 2} ({tail[at + 1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
