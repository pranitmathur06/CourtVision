"""Why the rim and ball numbers are what they are, so the next work is chosen on evidence.

`eval_rim_and_ball.py` says whether the gate is met. This says what to do about
it, by splitting the misses into kinds that need different work:

- BALL CEILING. Was a correct candidate on offer at all? A miss with a good
  candidate present is a selection failure and costs nothing to fix but
  thought; a miss with none present is a detection failure and costs a
  re-detection pass over the whole game. The earlier round that raised
  inference size from 1280 to 2560 moved ball coverage 0.733 to 0.892 and shot
  F1 not at all -- so which of the two this is decides hours of work.
- FIXTURES. How often the fixture rule removed the ONLY correct candidate.
  A rule that buys precision by throwing away the answer is worse than no rule.
- RIM SOURCE. Whether the located rims came from the projection or from the
  detector, and what the frames with neither look like -- if the misses are
  replay and baseline cameras, the fixed camera model cannot reach them by
  construction and a free-homography projection is the lever.
- WHERE. Misses in the game against misses outside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--clock", default=None)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_rim_and_ball import TOL_BALL_WIDTHS, TOL_RIM_WIDTHS, located

    truth_rows = json.load(open(args.truth))["frames"]
    system = {r["t"]: r for r in json.load(open(args.system))["frames"]}

    span = None
    if args.clock:
        readings = json.load(open(args.clock))["readings"]
        span = (min(r["t"] for r in readings), max(r["t"] for r in readings))

    ball_visible = ball_hit = ball_reachable = fixture_cost = 0
    rim_visible = 0
    rim_by_source: dict[str, list[int]] = {}
    misses_in_game = misses_outside = 0
    missed_frames = []

    for row in truth_rows:
        got = system.get(row["t"], {})
        source = got.get("rim_source") or "none"

        for truth_rim in row.get("rim") or []:
            rim_visible += 1
            found = any(located(p, truth_rim["centre"], truth_rim["width"], TOL_RIM_WIDTHS)
                        for p in (got.get("rim") or []))
            hits, total = rim_by_source.setdefault(source, [0, 0])
            rim_by_source[source] = [hits + int(found), total + 1]
            if not found:
                missed_frames.append((row["t"], source))
                if span and span[0] <= row["t"] <= span[1]:
                    misses_in_game += 1
                else:
                    misses_outside += 1

        truth_ball = row.get("ball")
        if truth_ball:
            ball_visible += 1
            chosen = got.get("ball")
            hit = located(chosen, truth_ball["centre"], truth_ball["width"],
                          TOL_BALL_WIDTHS)
            ball_hit += int(hit)
            offered = [c for c in (got.get("candidates") or [])
                       if located(c, truth_ball["centre"], truth_ball["width"],
                                  TOL_BALL_WIDTHS)]
            ball_reachable += int(bool(offered))
            survived = [c for c in (got.get("survivors") or [])
                        if located(c, truth_ball["centre"], truth_ball["width"],
                                   TOL_BALL_WIDTHS)]
            if offered and not survived:
                fixture_cost += 1

    print(f"RIM   {rim_visible} visible")
    for source, (hits, total) in sorted(rim_by_source.items()):
        print(f"        {source:10s} {hits:4d}/{total:4d} located ({hits / max(total,1):.1%})")
    if span:
        print(f"        misses in game {misses_in_game}, outside it {misses_outside}")

    print(f"\nBALL  {ball_visible} visible")
    print(f"        located                  {ball_hit:4d} ({ball_hit / max(ball_visible,1):.1%})")
    print(f"        a right candidate existed{ball_reachable:5d} "
          f"({ball_reachable / max(ball_visible,1):.1%})   <- the ceiling")
    print(f"        selection lost it        {ball_reachable - ball_hit:4d}")
    print(f"        the fixture rule lost it {fixture_cost:4d}")
    gap = ball_visible - ball_reachable
    print(f"        no candidate at all      {gap:4d}  <- only a better detector fixes these")

    if missed_frames:
        print("\nrim misses, first 25 by time:")
        for t, source in sorted(missed_frames)[:25]:
            print(f"  {t:8.0f} s   reported from: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
