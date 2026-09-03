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
