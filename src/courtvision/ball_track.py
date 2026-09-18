"""Pick the ball out of the candidates, using the fact that a ball has a path.

The detector is not failing to see the ball. Measured on a real broadcast it
emits a mean of 14.2 ball boxes per frame — max 37 — with a median confidence
of 0.050, and 99% of frames carry more than one. There is one ball. The
pipeline took the highest-confidence box, and with fourteen candidates the
argmax is frequently the wrong object, which is why end-to-end shot detection
scored F1 0.37 while the same method on clean coordinates scored 0.885.

So this is a SELECTION problem, not a detection one, and selection has a signal
the per-frame score does not: a ball moves smoothly and a false positive
teleports. Choosing one candidate per frame to minimise

    sum over frames of  -log(confidence)  +  weight * (distance moved)

is a shortest path through the candidates, solved exactly by Viterbi. No
retraining, no new data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Between frames at 10 fps a ball travels a few tens of pixels; a jump of
# hundreds is a different object. The cost is linear rather than capped so a
# genuinely fast ball is discouraged, not forbidden.
MOVE_WEIGHT = 0.02
MISSING_COST = 6.0


def choose(candidates: Sequence[Sequence[tuple[float, float, float]]],
           move_weight: float = MOVE_WEIGHT,
           missing_cost: float = MISSING_COST
           ) -> list[tuple[float, float] | None]:
    """One ball per frame, or None where the path is better off skipping.

    `candidates[i]` is that frame's (x, y, confidence) boxes. Returns the chosen
    centre per frame.
    """
    n = len(candidates)
    if n == 0:
        return []
    # state 0 of each frame is "no ball this frame"; states 1.. are candidates.
    best: list[list[float]] = []
    back: list[list[int]] = []
    previous_costs = [0.0]
    previous_points: list[tuple[float, float] | None] = [None]
    for index in range(n):
        points: list[tuple[float, float] | None] = [None]
        emission = [missing_cost]
        for x, y, conf in candidates[index]:
            points.append((x, y))
            emission.append(-math.log(max(conf, 1e-6)))
        costs = [math.inf] * len(points)
        choice = [0] * len(points)
        for state, point in enumerate(points):
            for prior, prior_point in enumerate(previous_points):
                step = 0.0
                if point is not None and prior_point is not None:
                    step = move_weight * math.hypot(point[0] - prior_point[0],
                                                    point[1] - prior_point[1])
                total = previous_costs[prior] + step + emission[state]
                if total < costs[state]:
                    costs[state] = total
                    choice[state] = prior
        best.append(costs)
        back.append(choice)
        previous_costs = costs
        previous_points = points

    state = min(range(len(best[-1])), key=lambda s: best[-1][s])
    path = [state]
    for index in range(n - 1, 0, -1):
        state = back[index][state]
        path.append(state)
    path.reverse()
    out: list[tuple[float, float] | None] = []
    for index, state in enumerate(path):
        if state == 0:
            out.append(None)
        else:
            x, y, _ = candidates[index][state - 1]
            out.append((x, y))
    return out


# A ball in flight is the HARD case, and `choose` above is built on the belief
# that the ball is the smooth candidate. Measured, that is backwards. On the 17
# frames across three broadcasts where the ball is proposed and the per-frame
# argmax takes something else -- the exact frames a selector exists to fix --
# the real ball's nearest neighbour in an adjacent frame is a median 90.5 px
# away and the decoy's is 6.5 px, and the decoy is the smoother of the two on
# 13 of 17. The false positives are stationary things: a head, a shoe, a logo.
# The fit agreed, driving MOVE_WEIGHT to the bottom of its grid on all three
# games, which is the optimiser saying switch the prior off.
#
# So penalise ACCELERATION rather than speed. A ball flying at 90 px a frame is
# then free, and it is the stationary decoy -- which requires decelerating to a
# stop and accelerating away again -- that is expensive. Same information,
# opposite sign. That needs the two previous choices rather than one, so the
# state is a PAIR of candidates and the recursion is over pairs.
#: Cost per pixel of acceleration between consecutive steps.
BEND_WEIGHT = 0.02
#: Speed, in pixels per frame, above which a candidate is a different object
#: however well it fits a line. A ball crosses about a quarter of a 1280-wide
#: frame in a second at 15 Hz, so this is deliberately generous.
MAX_SPEED_PX = 400.0


def choose_ballistic(candidates: Sequence[Sequence[tuple[float, float, float]]],
                     bend_weight: float = BEND_WEIGHT,
                     missing_cost: float = MISSING_COST,
                     max_speed_px: float = MAX_SPEED_PX,
                     ) -> list[tuple[float, float] | None]:
    """One ball per frame, chosen by how straight its motion is.

    Exact over pairs of consecutive candidates, so the cost of a triple is the
    deviation from where a constant velocity would have put the third point.
    `None` states are allowed anywhere and simply carry `missing_cost` and no
    geometry, which is what lets the path cross a frame where the detector
    proposed nothing at all.
    """
    n = len(candidates)
    if n == 0:
        return []
    if n == 1:
        frame = candidates[0]
        if not frame:
            return [None]
        best = max(frame, key=lambda c: c[2])
        return [(best[0], best[1])]

    def points(index: int) -> list[tuple[float, float] | None]:
        return [None] + [(x, y) for x, y, _ in candidates[index]]

    def emission(index: int) -> list[float]:
        return [missing_cost] + [-math.log(max(conf, 1e-6))
                                 for _, _, conf in candidates[index]]

    first_points, second_points = points(0), points(1)
    first_cost, second_cost = emission(0), emission(1)

    # cost[(a, b)] is the best total for "state a at frame 0, state b at 1".
    costs: dict[tuple[int, int], float] = {}
    for a, point_a in enumerate(first_points):
        for b, point_b in enumerate(second_points):
            if point_a is not None and point_b is not None:
                speed = math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
                if speed > max_speed_px:
                    continue
            costs[(a, b)] = first_cost[a] + second_cost[b]

    back: list[dict[tuple[int, int], tuple[int, int]]] = []
    previous_points = (first_points, second_points)
    for index in range(2, n):
        here_points, here_cost = points(index), emission(index)
        nxt: dict[tuple[int, int], float] = {}
        pointer: dict[tuple[int, int], tuple[int, int]] = {}
        for (a, b), total in costs.items():
            point_a = previous_points[0][a]
            point_b = previous_points[1][b]
            for c, point_c in enumerate(here_points):
                bend = 0.0
                if point_b is not None and point_c is not None:
                    speed = math.hypot(point_c[0] - point_b[0], point_c[1] - point_b[1])
                    if speed > max_speed_px:
                        continue
                    if point_a is not None:
                        # Where a constant velocity would have put it.
                        predicted = (2 * point_b[0] - point_a[0],
                                     2 * point_b[1] - point_a[1])
                        bend = bend_weight * math.hypot(point_c[0] - predicted[0],
                                                        point_c[1] - predicted[1])
                value = total + bend + here_cost[c]
                key = (b, c)
                if key not in nxt or value < nxt[key]:
                    nxt[key] = value
                    pointer[key] = (a, b)
        if not nxt:
            return [None] * n
        costs = nxt
        back.append(pointer)
        previous_points = (previous_points[1], here_points)

    end = min(costs, key=lambda key: costs[key])
    path = [end[0], end[1]]
    state = end
    for pointer in reversed(back):
        state = pointer[state]
        path.insert(0, state[0])
    out: list[tuple[float, float] | None] = []
    for index, state_index in enumerate(path[:n]):
        if state_index == 0:
            out.append(None)
        else:
            x, y, _ = candidates[index][state_index - 1]
            out.append((x, y))
    return out


# WHY A THIRD SELECTOR. `choose` penalises speed and `choose_ballistic`
# penalises acceleration, and neither can separate the ball from the decoys
# this detector actually emits, because A STATIONARY OBJECT HAS ZERO
# ACCELERATION TOO. Measured on the 97 uniformly sampled windows where the ball
# is both labelled and proposed, taking each candidate's nearest neighbour in
# the adjacent frame:
#
#     frame of reference        the ball moves    a decoy moves
#     raw pixels                      11.2 px          5.0 px
#     rim-relative (the court)        10.9 px          1.8 px
#
# The decoys are things painted on or standing still on the floor -- a logo, a
# head, a shoe -- so they move with the CAMERA and not with the game. Subtract
# the camera and they stop dead, while the ball keeps its ten pixels. That
# takes the contrast from 2.2x to 6.1x, and it is the whole of the signal.
#
# The camera's motion comes free: the rim is bolted to the building, the
# detector draws it on three frames in four, and its displacement between two
# frames IS the camera's. No optical flow, no registration, no new model.
#: Court-relative pixels per step below which a candidate is suspected of being
#: part of the furniture. Sits between the two medians above.
STILL_PX = 5.0
#: Cost per pixel of stillness, under that floor.
STILL_WEIGHT = 0.12


def choose_moving(candidates: Sequence[Sequence[tuple[float, float, float]]],
                  shifts: Sequence[tuple[float, float] | None] | None = None,
                  still_px: float = STILL_PX,
                  still_weight: float = STILL_WEIGHT,
                  missing_cost: float = MISSING_COST,
                  max_speed_px: float = MAX_SPEED_PX,
                  ) -> list[tuple[float, float] | None]:
    """One ball per frame, preferring the candidate that MOVES on the court.

    `shifts[i]` is the camera's displacement from frame i-1 to frame i, or
    None where it is unknown -- in which case that step falls back to raw
    pixels, which still carries some of the signal (2.2x rather than 6.1x).
    """
    n = len(candidates)
    if n == 0:
        return []
    if shifts is None:
        shifts = [None] * n

    previous_points: list[tuple[float, float] | None] = [None]
    previous_costs = [0.0]
    back: list[list[int]] = []
    kept: list[list[tuple[float, float] | None]] = []
    for index in range(n):
        points: list[tuple[float, float] | None] = [None]
        emission = [missing_cost]
        for x, y, conf in candidates[index]:
            points.append((x, y))
            emission.append(-math.log(max(conf, 1e-6)))
        shift = shifts[index] if index < len(shifts) else None
        costs = [math.inf] * len(points)
        choice = [0] * len(points)
        for state, point in enumerate(points):
            for prior, prior_point in enumerate(previous_points):
                step = 0.0
                if point is not None and prior_point is not None:
                    expected = (prior_point[0] + (shift[0] if shift else 0.0),
                                prior_point[1] + (shift[1] if shift else 0.0))
                    moved = math.hypot(point[0] - expected[0], point[1] - expected[1])
                    if moved > max_speed_px:
                        continue
                    step = still_weight * max(0.0, still_px - moved)
                total = previous_costs[prior] + step + emission[state]
                if total < costs[state]:
                    costs[state] = total
                    choice[state] = prior
        if all(math.isinf(c) for c in costs):
            return [None] * n
        back.append(choice)
        kept.append(points)
        previous_costs, previous_points = costs, points

    state = min(range(len(previous_costs)), key=lambda s: previous_costs[s])
    path = [state]
    for index in range(n - 1, 0, -1):
        state = back[index][state]
        path.append(state)
    path.reverse()
    out: list[tuple[float, float] | None] = []
    for index, state_index in enumerate(path):
        if state_index == 0:
            out.append(None)
        else:
            x, y, _ = candidates[index][state_index - 1]
            out.append((x, y))
    return out
