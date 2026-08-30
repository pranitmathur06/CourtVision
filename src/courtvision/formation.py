"""Offensive formation from court coordinates.

Formations, NOT plays. The distinction is the whole design.

A formation is an arrangement: where five players stand at one instant. Horns is
two bigs at the elbows and two shooters in the corners, and you can see that in
a single frame. A play is a sequence — Spain pick-and-roll is a screen, then a
back-screen on the screener, then a roll — and no snapshot contains it. Naming a
play from one frame would be guessing with extra steps.

So this reports what the geometry actually supports, and says `unknown` when the
arrangement is not one it can name. Everything here consumes COURT coordinates
from `courtvision.court`; feeding it pixels produces confident nonsense, because
the same pixel gap means different distances at different depths.

Play recognition proper still needs formation sequences over time and labelled
play types, which neither BARD nor SpaceJam carries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from courtvision.court import (
    COURT_WIDTH,
    FREE_THROW_LINE_Y,
    LANDMARKS,
    distance_to_basket_ft,
    is_three_point_attempt,
)

ELBOW_LEFT = LANDMARKS["lane_left_ft"]
ELBOW_RIGHT = LANDMARKS["lane_right_ft"]
ELBOW_TOLERANCE_FT = 5.0
CORNER_MAX_X = 9.0
CORNER_MAX_Y = 11.0
PAINT_HALF_WIDTH = 8.0
ISOLATION_CLEARANCE_FT = 14.0
MAX_OFFENSE = 7          # five, plus slack for a duplicated track


@dataclass(frozen=True)
class Formation:
    """What the arrangement supports, and why."""

    name: str
    evidence: str
    spacing_ft: float          # mean nearest-neighbour distance
    tightest_pair_ft: float

    def __str__(self) -> str:
        return f"{self.name} ({self.evidence})"


def _near(point, target, tolerance: float) -> bool:
    return float(np.hypot(point[0] - target[0], point[1] - target[1])) <= tolerance


def _in_corner(point) -> bool:
    x, y = point
    return y <= CORNER_MAX_Y and (x <= CORNER_MAX_X or x >= COURT_WIDTH - CORNER_MAX_X)


def _in_paint(point) -> bool:
    x, y = point
    return abs(x - COURT_WIDTH / 2) <= PAINT_HALF_WIDTH and 0.0 <= y <= FREE_THROW_LINE_Y


def spacing(positions: np.ndarray) -> tuple[float, float]:
    """Mean nearest-neighbour distance and the tightest pair, in feet.

    Spacing is the number commentary actually wants — "they have no room" is a
    claim about nearest neighbours, not about the centroid.
    """
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return float("nan"), float("nan")
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    return float(nearest.mean()), float(dist.min())


def classify_formation(offense: np.ndarray) -> Formation:
    """Name the arrangement of five offensive players, or decline to.

    `offense` is (N, 2) in court feet. N need not be 5 — a tracker loses players
    behind the camera constantly — but below four the question is meaningless.
    """
    pts = np.asarray(offense, dtype=np.float64).reshape(-1, 2)
    pts = pts[~np.isnan(pts).any(axis=1)]
    mean_gap, tightest = spacing(pts)

    if len(pts) < 4:
        return Formation("unknown", f"only {len(pts)} players located",
                         mean_gap, tightest)
    if len(pts) > MAX_OFFENSE:
        # Almost certainly both teams. Say so rather than answering: fed ten
        # players this returned `unknown` for every frame and reported spacing
        # of 6-8 ft, because a defender guards at three to six feet and drags
        # every nearest-neighbour distance down. Filtered to the five with the
        # ball, the same footage reads 13-20 ft, which is what NBA half-court
        # spacing actually looks like.
        return Formation("unknown",
                         f"{len(pts)} players given; this expects one team, so "
                         f"filter to the side with the ball", mean_gap, tightest)

    at_elbow = [p for p in pts
                if _near(p, ELBOW_LEFT, ELBOW_TOLERANCE_FT)
                or _near(p, ELBOW_RIGHT, ELBOW_TOLERANCE_FT)]
    left_elbow = any(_near(p, ELBOW_LEFT, ELBOW_TOLERANCE_FT) for p in pts)
    right_elbow = any(_near(p, ELBOW_RIGHT, ELBOW_TOLERANCE_FT) for p in pts)
    corners = [p for p in pts if _in_corner(p)]
    in_paint = [p for p in pts if _in_paint(p)]
    beyond_arc = [p for p in pts if is_three_point_attempt(tuple(p))]

    # Horns: both elbows occupied and both corners filled. Distinctive enough to
    # name from a snapshot, which is exactly why it is the one worth naming.
    if left_elbow and right_elbow and len(corners) >= 2:
        return Formation("horns", "both elbows occupied, two corners filled",
                         mean_gap, tightest)

    # Five-out: nobody inside the arc. A spacing choice, visible at an instant.
    if len(beyond_arc) == len(pts) and len(pts) >= 5:
        return Formation("five_out", "all five beyond the arc", mean_gap, tightest)

    # Post-up: someone on the block with the rest spaced away from them.
    if len(in_paint) == 1:
        post = in_paint[0]
        others = [p for p in pts if not np.array_equal(p, post)]
        if others and min(float(np.linalg.norm(post - p)) for p in others) >= 10.0:
            if distance_to_basket_ft([post])[0] <= 12.0:
                return Formation("post_up", "one player on the block, rest cleared out",
                                 mean_gap, tightest)

    # Isolation: the most isolated player has a wide berth and everyone else is
    # bunched away from them.
    distances = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    loneliest = int(distances.min(axis=1).argmax())
    if distances[loneliest].min() >= ISOLATION_CLEARANCE_FT:
        return Formation("isolation",
                         f"one player {distances[loneliest].min():.0f} ft from "
                         f"the nearest teammate", mean_gap, tightest)

    if at_elbow or len(corners) >= 2:
        return Formation("unknown", "partial alignment, not a formation this names",
                         mean_gap, tightest)
    return Formation("unknown", "no recognised alignment", mean_gap, tightest)
