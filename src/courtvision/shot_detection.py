"""Shots and blocks from the ball's trajectory, not from pixels.

`derived_events.derive` splits rebound from steal by asking whether a shot
preceded a possession change, so shots are load-bearing: without them every
change reads as a steal and rebound collapses. From video that meant a
classifier. From coordinates it is geometry, and the rims never move — they sit
5.25 ft from each baseline on the centre line of a 94 x 50 ft court, with the
hoop 10 ft up. Nothing needs detecting.

Both signals were measured against the official play-by-play for 0021500492
rather than assumed. In a two-second window around the 212 official shots:

    ball-to-rim distance   median  1.0 ft   (13.6 ft in random windows)
    ball peak height       median 11.0 ft   ( 6.5 ft in random windows)

Swept over both thresholds, rim <= 6.0 ft with a peak >= 8.0 ft gives
precision 0.85, recall 0.70, and 0.83x the official count.

The missing 30% is not noise: requiring the ball to REACH the rim necessarily
misses shots that were blocked before getting there. That complement is what
`blocks` looks for, and it is the only route to a class that scores BELOW
chance from pixels (-0.170 lift from a player crop).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from courtvision.types import Event, Frame

# Court-absolute rim positions on a 94 x 50 ft floor.
RIMS = ((5.25, 25.0), (88.75, 25.0))
RIM_HEIGHT_FT = 10.0

# Swept over 12 games, fitting on six and validating on the six held out, so
# the gain is not the sweep choosing its own test set. Tightening the rim
# radius from 6.0 to 4.0 and widening the merge gap to 2.5 moved held-out
# field-goal F1 from 0.849 to 0.858 — real but marginal, and recall stays near
# 0.79 whatever the thresholds. The missing fifth of field goals is airballs,
# shots blocked before the ball approaches, and stretches where the ball is
# not tracked; no threshold recovers them.
MAX_RIM_DISTANCE_FT = 4.0
MIN_PEAK_HEIGHT_FT = 8.0
# Two approaches closer together than this are one attempt being scored twice.
MIN_GAP_S = 2.5
# How far back to look for the ball's high point before it reaches the rim.
LOOKBACK_S = 1.6


def _ball_xy(frame: Frame) -> tuple[float, float] | None:
    ball = frame.ball()
    return None if ball is None else ball.box.center


def rim_distance(frame: Frame) -> float | None:
    """Feet from the ball to the nearer rim, ignoring height."""
    centre = _ball_xy(frame)
    if centre is None:
        return None
    x, y = centre
    return min(math.hypot(x - rx, y - ry) for rx, ry in RIMS)


def shots(frames: Sequence[Frame], ball_z: Sequence[float],
          max_rim_ft: float = MAX_RIM_DISTANCE_FT,
          min_peak_ft: float = MIN_PEAK_HEIGHT_FT,
          min_gap_s: float = MIN_GAP_S) -> list[Event]:
    """One event per approach of the ball to a rim, from above.

    The height floor is what separates a shot from a ball carried under the
    basket or a pass through the paint; without it precision falls from 0.85
    to 0.77 while recall barely moves.
    """
    times = [f.time_s for f in frames]
    approaches: list[list[int]] = []
    current: list[int] = []
    for index, frame in enumerate(frames):
        distance = rim_distance(frame)
        if distance is None or distance > max_rim_ft:
            continue
        if current and times[index] - times[current[-1]] > min_gap_s:
            approaches.append(current)
            current = []
        current.append(index)
    if current:
        approaches.append(current)

    events: list[Event] = []
    for group in approaches:
        start_time = times[group[0]]
        peak = 0.0
        for index in range(group[0], -1, -1):
            if start_time - times[index] > LOOKBACK_S:
                break
            height = ball_z[index] if index < len(ball_z) else float("nan")
            if height == height:
                peak = max(peak, height)
        for index in group:
            height = ball_z[index] if index < len(ball_z) else float("nan")
            if height == height:
                peak = max(peak, height)
        if peak < min_peak_ft:
            continue
        closest = min(group, key=lambda i: rim_distance(frames[i]) or 1e9)
        events.append(Event(time_s=times[closest], track_id=None, team=None,
                            action="shot", possession_change=False))
    return events


# A shot that goes IN passes through a narrow cylinder at the rim; one that
# misses does not. Measured on 616 official attempts across four games, taking
# the ball's horizontal distance to the rim at the moment it descends through
# hoop height (linear interpolation between the bracketing samples, so the
# result does not depend on the sample rate):
#
#     made   n=270   median 0.33 ft
#     miss   n=346   median 2.04 ft
#
# Swept, a 1.0 ft cylinder keeps 93% of makes and 12% of misses (precision
# 0.86). This is load-bearing for rebound, not a scoring nicety: `derive`
# treated EVERY possession change after a shot as a rebound, including the
# inbound that follows a made basket, and roughly half a game's field goals go
# in. Those inbounds were about half of all emitted rebounds.
MAKE_RADIUS_FT = 1.0


def makes(frames: Sequence[Frame], ball_z: Sequence[float],
          shot_events: Sequence[Event],
          radius_ft: float = MAKE_RADIUS_FT,
          window_s: float = 2.5) -> list[float]:
    """Times of the shots in `shot_events` whose ball passed through the hoop."""
    times = [f.time_s for f in frames]
    centres = [_ball_xy(f) for f in frames]
    made: list[float] = []
    for event in shot_events:
        best: float | None = None
        for index in range(len(times) - 1):
            if not (event.time_s - window_s <= times[index] <= event.time_s + window_s):
                continue
            high = ball_z[index] if index < len(ball_z) else float("nan")
            low = ball_z[index + 1] if index + 1 < len(ball_z) else float("nan")
            if high != high or low != low or not (high >= RIM_HEIGHT_FT >= low):
                continue
            here, next_ = centres[index], centres[index + 1]
            if here is None or next_ is None:
                continue
            # Where was the ball horizontally as it crossed hoop height?
            span = high - low
            fraction = 0.0 if span <= 0 else (high - RIM_HEIGHT_FT) / span
            x = here[0] + fraction * (next_[0] - here[0])
            y = here[1] + fraction * (next_[1] - here[1])
            distance = min(math.hypot(x - rx, y - ry) for rx, ry in RIMS)
            best = distance if best is None else min(best, distance)
        if best is not None and best <= radius_ft:
            made.append(event.time_s)
    return sorted(made)


# Block is NOT provided, and that is a measured decision rather than an
# omission. Two geometric hypotheses were tested against the 11 official blocks
# in 0021500492 and both failed:
#
#   "a blocked shot never reaches the rim" — false. Blocked shots reach it
#   anyway: median 0.4 ft from the rim, 8 of 11 within 2.3 ft. Blocks are a
#   SUBSET of shots here, not a complement, so a stopped-short detector fired
#   35 times for 11 blocks at 0.00 precision.
#
#   "a defender is at the ball while it is high" — no separation. Closest
#   player to a ball above 8.5 ft is 1.1 ft on blocks and 0.8 ft on ordinary
#   shots, because on any shot the SHOOTER is right there. Telling them apart
#   needs to know the near player is an opponent, which needs possession,
#   which needs the shot.
#
# Block is also below chance from pixels (-0.170 lift). At 11 events a game it
# is 2.6% of the action, and a detector at 0.00 precision is worse than none.
