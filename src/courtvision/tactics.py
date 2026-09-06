"""Offensive and defensive structure, from court positions.

Everything here needs positions in feet, which only became trustworthy once the
registration reached 1.72 ft median / 2.43 ft at p75. At the earlier 7-10 ft
error none of these quantities meant anything: a defender 4 ft away and one
12 ft away were indistinguishable, so "contested" could not be measured at all.

Deliberately NOT here: anything requiring player identity. Jersey OCR does not
exist in this project, so these functions describe configurations -- how many
defenders are near, how spread the offense is, whether the floor is balanced --
and never who was where. A coaching tool may say "the nearest defender was
4.2 ft away"; it must not say a name it does not know.

Team assignment is by SIDE OF BALL, not by jersey colour: the five players
nearest the basket being attacked are treated as the defense. That is an
approximation and is stated as one wherever it matters.

CALLERS MUST DETECT AT conf 0.25, NOT HIGHER. Splitting five-and-five requires
most of the ten to be found, and the detector's threshold decides that:

    conf 0.35    6 players on court    8% of frames reach 8
    conf 0.25    9 players on court   75%
    conf 0.15   14 players on court  100%   (referees and false positives)

Run at 0.35 first, this layer reported a 13.9 ft median nearest defender
against a real 4-5 ft, called 42 of 61 shots open, and inverted the
open-versus-contested relationship -- all downstream of splitting five-and-five
when only six people had been found.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from courtvision.court import BASKET, COURT_WIDTH, FREE_THROW_LINE_Y

COURT_LENGTH_FT = 94.0
# Inside this, a defender is contesting rather than merely nearby. NBA tracking
# work has used 4 ft for "tightly contested" and 6 ft for "open"; both are
# reported so a caller can choose.
CONTEST_FT = 4.0
OPEN_FT = 6.0
# The paint, for counting help defenders.
PAINT_HALF_WIDTH_FT = 8.0
PAINT_DEPTH_FT = 19.0
# Corner threes sit 22 ft from the basket, everything else 23.75.
THREE_POINT_RADIUS_FT = 23.75
CORNER_THREE_X_FT = 3.0


@dataclass(frozen=True)
class Spacing:
    """How the offense occupies the floor at one moment."""

    players: int
    width_ft: float
    depth_ft: float
    nearest_pair_ft: float
    in_paint: int


def distance_to_basket(points: np.ndarray) -> np.ndarray:
    """Feet from each court position to the basket being attacked."""
    points = np.asarray(points, dtype=float)
    return np.hypot(points[:, 0] - BASKET[0], points[:, 1] - BASKET[1])


def is_three_point_distance(point) -> bool:
    """Whether a spot is beyond the arc, corners included."""
    x, y = float(point[0]), float(point[1])
    if x <= CORNER_THREE_X_FT or x >= COURT_WIDTH - CORNER_THREE_X_FT:
        # In the corners the line is a straight 22 ft, not the arc.
        return float(np.hypot(x - BASKET[0], y - BASKET[1])) >= 22.0
    return float(np.hypot(x - BASKET[0], y - BASKET[1])) >= THREE_POINT_RADIUS_FT


def in_paint(points: np.ndarray) -> np.ndarray:
    """Mask of positions inside the painted lane."""
    points = np.asarray(points, dtype=float)
    return ((np.abs(points[:, 0] - COURT_WIDTH / 2.0) <= PAINT_HALF_WIDTH_FT)
            & (points[:, 1] >= 0.0) & (points[:, 1] <= PAINT_DEPTH_FT))


def defenders_near(spot, defenders: np.ndarray,
                   radius_ft: float = CONTEST_FT) -> int:
    """How many defenders are within `radius_ft` of a spot."""
    if defenders is None or len(defenders) == 0:
        return 0
    defenders = np.asarray(defenders, dtype=float)
    return int((np.hypot(defenders[:, 0] - spot[0],
                         defenders[:, 1] - spot[1]) <= radius_ft).sum())


def nearest_defender_ft(spot, defenders: np.ndarray) -> float | None:
    """Distance to the closest defender, or None when none are known."""
    if defenders is None or len(defenders) == 0:
        return None
    defenders = np.asarray(defenders, dtype=float)
    return float(np.hypot(defenders[:, 0] - spot[0],
                          defenders[:, 1] - spot[1]).min())


def contest_level(spot, defenders: np.ndarray) -> str:
    """'open', 'guarded' or 'contested' -- the vocabulary a coach uses."""
    nearest = nearest_defender_ft(spot, defenders)
    if nearest is None:
        return "unknown"
    if nearest >= OPEN_FT:
        return "open"
    if nearest >= CONTEST_FT:
        return "guarded"
    return "contested"


def spacing(offense: np.ndarray) -> Spacing | None:
    """How well the offense is spread.

    Width and depth are the extents actually occupied; `nearest_pair_ft` is the
    tightest pair, which is what "bunched up" means concretely.
    """
    if offense is None or len(offense) < 2:
        return None
    offense = np.asarray(offense, dtype=float)
    gaps = []
    for index in range(len(offense)):
        others = np.delete(offense, index, axis=0)
        gaps.append(np.hypot(others[:, 0] - offense[index, 0],
                             others[:, 1] - offense[index, 1]).min())
    return Spacing(
        players=len(offense),
        width_ft=float(offense[:, 0].max() - offense[:, 0].min()),
        depth_ft=float(offense[:, 1].max() - offense[:, 1].min()),
        nearest_pair_ft=float(min(gaps)),
        in_paint=int(in_paint(offense).sum()),
    )


def help_defenders(spot, defenders: np.ndarray,
                   shooter_radius_ft: float = OPEN_FT) -> int:
    """Defenders in the paint who are NOT guarding this spot.

    This is what "the help sank" means: bodies committed to the rim while the
    ball is somewhere else, which is the reason a shooter ends up open.
    """
    if defenders is None or len(defenders) == 0:
        return 0
    defenders = np.asarray(defenders, dtype=float)
    away = np.hypot(defenders[:, 0] - spot[0],
                    defenders[:, 1] - spot[1]) > shooter_radius_ft
    return int((in_paint(defenders) & away).sum())


def split_by_side(points: np.ndarray,
                  ball_spot=None) -> tuple[np.ndarray, np.ndarray]:
    """Split people into (offense, defense) WITHOUT jersey colour.

    The five nearest the attacked basket are taken as the defense. This is an
    approximation -- a rolling big or a crashing offensive rebounder breaks it
    -- and callers must treat the split as a heuristic, not a fact. It exists
    because jersey OCR does not, and it fails most often in exactly the
    situations a coach cares about, so it is reported alongside its own
    uncertainty rather than silently.
    """
    if points is None or len(points) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    points = np.asarray(points, dtype=float)
    if len(points) <= 5:
        return points, np.empty((0, 2))
    order = np.argsort(distance_to_basket(points))
    defense = points[order[:5]]
    offense = points[order[5:]]
    if ball_spot is not None and len(offense):
        # The ball handler is offense by definition; if he sorted into the
        # defensive five, move him across.
        ball_spot = np.asarray(ball_spot, dtype=float)
        gaps = np.hypot(defense[:, 0] - ball_spot[0],
                        defense[:, 1] - ball_spot[1])
        if gaps.min() < 3.0:
            index = int(np.argmin(gaps))
            offense = np.vstack([offense, defense[index]])
            defense = np.delete(defense, index, axis=0)
    return offense, defense
