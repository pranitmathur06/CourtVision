"""Formation rates, read at the moment an offense is actually set.

Formations are the one part of tactical labelling a rule can settle. An action
is a judgement -- the literature on pick-and-roll annotation says experts
disagree on edge cases -- but a formation is a configuration at one instant, and
two people looking at the same frame agree on whether two players are at the
elbows.

That makes this the right place to start weak supervision, and the rates are
checkable against knowledge of the sport, which is what makes the labels
falsifiable rather than merely produced.
"""

from __future__ import annotations

import argparse
import collections
import glob
import math
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from check_transition import SAMPLE_HZ, possessions, stretches
from courtvision.court import BASKET
from courtvision.tactical_features import (SET_AT_SHOT_CLOCK_S, in_corners,
                                           is_horns, near_elbows)
from courtvision.tracking_data import load_game

NEAR = (BASKET[1], 25.0)
FAR = (94.0 - BASKET[1], 25.0)


def oriented(points, rim):
    """Put the attacked basket at (25, 0) so one rule serves both ends.

    SportVU runs the length of the court along x; `courtvision.court` runs it
    along y, so the axes swap as well as flip.
    """
    pts = np.asarray(points, dtype=float)
    if rim is FAR:
        return np.stack([50.0 - pts[:, 1], 94.0 - pts[:, 0]], axis=1)
    return np.stack([pts[:, 1], pts[:, 0]], axis=1)


def set_positions(frames, clocks, teams, side, at):
    """The five attackers' positions once the offense is set, or None."""
    for frame in frames:
        clock = clocks.get(frame.index)
        if clock is None or math.isnan(clock) or clock > at:
            continue
        attacking = [t for t in frame.tracks
                     if t.label != "ball" and teams.get(t.track_id) == side]
        if len(attacking) != 5:
            return None
        spots = np.array([[(t.box.x1 + t.box.x2) / 2, t.box.y2]
                          for t in attacking])
        middle = spots.mean(axis=0)
        rim = NEAR if abs(middle[0] - NEAR[0]) < abs(middle[0] - FAR[0]) else FAR
        return oriented(spots, rim)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--at", type=float, default=SET_AT_SHOT_CLOCK_S)
    args = parser.parse_args()

    tally: collections.Counter = collections.Counter()
    total = 0
    for path in sorted(glob.glob("data/tracking/*.json"))[:args.limit]:
        game = load_game(path, target_hz=SAMPLE_HZ)
        clocks = {f.index: c for f, c in zip(game.frames, game.shot_clocks)}
        for chunk in stretches(game):
            for side, frames in possessions(chunk, game.teams, clocks):
                court = set_positions(frames, clocks, game.teams, side, args.at)
                if court is None:
                    continue
                total += 1
                tally["two at the elbows"] += near_elbows(court) >= 2
                tally["two in the corners"] += in_corners(court) >= 2
                tally["horns"] += is_horns(court)
                tally["one big at an elbow"] += near_elbows(court) >= 1
        print(f"    {path.split('/')[-1]}: {total} sets", flush=True)

    print(f"\n  {total} half-court sets, read at {args.at:.0f}s on the shot clock")
    for name, count in tally.most_common():
        print(f"    {name:<22}{count:>6}{count / max(total, 1):>8.1%}")
    print("\n  For reference: a team that leans on Horns runs it on roughly a")
    print("  fifth of half-court possessions, most teams a good deal less.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
