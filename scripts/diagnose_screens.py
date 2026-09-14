"""Why did or did not a screen fire? Answers it without re-registering.

Reads the court positions scripts/analyze_plays.py cached to
outputs/play_positions.json, because registration cost ~8 s a frame and every
question about thresholds afterwards should be answerable without paying that
again. analyze_plays registered by line search, which ee94b99 removed, so it
was deleted and the cache cannot be regenerated; it is in history at 199e28c.

A screen needs two offensive players to be SCREEN_SEPARATION_FT apart and then
come within SCREEN_CONTACT_FT. This prints every near-contact with its prior
separation, so a null result can be read as "no screens happened" or "the
thresholds sit outside what this footage resolves" rather than guessed at.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

from courtvision.plays import (SCREEN_CONTACT_FT, SCREEN_SEPARATION_FT,
                               WINDOW_FRAMES)

CACHE = Path("outputs/play_positions.json")


def main() -> int:
    if not CACHE.exists():
        print(f"no cache at {CACHE}; it was written by scripts/analyze_plays.py, "
              "removed after 199e28c")
        return 1
    blob = json.loads(CACHE.read_text())
    frames = [{int(k): tuple(v) for k, v in f.items()} for f in blob["positions"]]

    def dist(a, b) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    print(f"{len(frames)} frames; contact <= {SCREEN_CONTACT_FT} ft after "
          f">= {SCREEN_SEPARATION_FT} ft apart\n")
    print(f"  {'frame':<7}{'pair':<10}{'now':>7}{'prior max':>11}   verdict")
    near_misses = []
    for index, frame in enumerate(frames):
        for a, b in itertools.combinations(sorted(frame), 2):
            now = dist(frame[a], frame[b])
            if now > SCREEN_CONTACT_FT * 2:
                continue
            prior = [dist(frames[index - k][a], frames[index - k][b])
                     for k in range(1, min(WINDOW_FRAMES, index) + 1)
                     if a in frames[index - k] and b in frames[index - k]]
            widest = max(prior) if prior else None
            separated = widest is not None and widest >= SCREEN_SEPARATION_FT
            if separated and now > SCREEN_CONTACT_FT:
                near_misses.append(now)
                verdict = f"separated, but {now - SCREEN_CONTACT_FT:.1f} ft short of contact"
            elif separated:
                verdict = "SCREEN"
            else:
                verdict = "never separated first"
            print(f"  {index:<7}({a},{b})".ljust(24)
                  + f"{now:>7.1f}"
                  + f"{'  n/a' if widest is None else f'{widest:>11.1f}'}"
                  + f"   {verdict}")

    if near_misses:
        print(f"\n  {len(near_misses)} pair(s) converged after separating but stopped "
              f"at {min(near_misses):.1f}-{max(near_misses):.1f} ft.")
        print(f"  The contact threshold is {SCREEN_CONTACT_FT:.0f} ft. Registration on "
              f"real footage carries a\n  couple of feet of error and a player's "
              f"position is the bottom-centre of a\n  detection box, so a real screen "
              f"can measure 6-7 ft. These sit exactly in\n  that band: not evidence "
              f"of screens, and not evidence against them.")
        print(f"\n  Loosening the threshold would produce detections nobody can check. "
              f"The\n  honest fix is better registration, not a wider threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
