"""Tactical features from tracking: derivatives, matchups, and spatial control.

Positions alone are a weak description of a play. What separates a pin-down
from two players passing each other, or a switch from a defender trailing, is
in the derivatives and in who is responsible for whom -- neither of which is
visible in a single frame of coordinates.

Three families, in the order they became necessary:

**Derivatives.** A screen is a player stopping; a cut is a player accelerating
out of a change of direction. Both are second-order and invisible to a
position-only detector, which is why screen detection here kept firing on two
players who merely drifted close together.

**Matchups by spatial control, not proximity.** Nearest-defender assignment
breaks exactly when it matters: during a switch, both defenders are near both
attackers for a beat, and the assignment flickers. Voronoi cell ownership is
stable through that, because it asks who CONTROLS the space around a player
rather than who is momentarily closest, and a switch shows up as a clean
exchange of cells rather than as noise.

**Formation.** Unlike an action, a formation is a configuration at one instant:
two players at the elbows and two in the corners is Horns whether or not
anything follows. That makes it the one part of this that can be labelled by
rule without the labels becoming a matter of judgement -- which every attempt
at rule-labelling a screen in this project turned out to be.
"""

from __future__ import annotations

import numpy as np

from courtvision.court import BASKET, COURT_WIDTH, FREE_THROW_LINE_Y

# A body's width, used as the distance at which two players are interacting.
CONTACT_FT = 5.0
# Elbow: where the free-throw line meets the lane.
ELBOW_X = ((COURT_WIDTH - 16.0) / 2, (COURT_WIDTH + 16.0) / 2)
ELBOW_TOLERANCE_FT = 6.0
# The corner is a REGION, not a point: the corner three sits a few feet inside
# the sideline and up to about fourteen from the baseline. At eight feet this
# found corner spacing on 2% of possessions, which is not basketball.
CORNER_TOLERANCE_FT = 14.0
# When to read a formation. Measured rather than chosen: corner spacing climbs
# from 11% of sets at 20 s on the shot clock to about a third by 16-14 s and
# then flattens, and the offense's width does the same (31 ft to 33 ft, then
# flat). Before that the ball has arrived but the other four have not spaced,
# and a formation rule reads an offense that does not exist yet.
SET_AT_SHOT_CLOCK_S = 14.0


def derivatives(track: np.ndarray, hz: float) -> dict[str, np.ndarray]:
    """Speed, acceleration and turn rate for one player's path.

    `track` is (n, 2) court feet. Everything is per second, so a change of
    sampling rate does not silently change what a threshold means -- a mistake
    already made once here, where a screener speed limit tuned at one rate
    admitted transition run-bys at another.
    """
    path = np.asarray(track, dtype=float).reshape(-1, 2)
    if len(path) < 3:
        zero = np.zeros(len(path))
        return dict(speed=zero, acceleration=zero, turn=zero)
    step = np.gradient(path, axis=0) * hz
    speed = np.hypot(step[:, 0], step[:, 1])
    change = np.gradient(step, axis=0) * hz
    acceleration = np.hypot(change[:, 0], change[:, 1])
    heading = np.arctan2(step[:, 1], step[:, 0])
    turn = np.abs(np.gradient(np.unwrap(heading))) * hz
    return dict(speed=speed, acceleration=acceleration, turn=turn)


def control_grid(offense: np.ndarray, defense: np.ndarray,
                 spacing_ft: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Which defender controls each patch of the offensive half.

    A Voronoi tessellation over the defenders, sampled on a grid. Returns
    (grid points, index of the controlling defender), so an offensive player's
    matchup is whoever owns the cell he stands in.
    """
    defenders = np.asarray(defense, dtype=float).reshape(-1, 2)
    if len(defenders) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    xs = np.arange(0.0, COURT_WIDTH + spacing_ft, spacing_ft)
    ys = np.arange(0.0, 50.0 + spacing_ft, spacing_ft)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    gaps = np.linalg.norm(grid[:, None, :] - defenders[None, :, :], axis=2)
    return grid, np.argmin(gaps, axis=1)


def matchups(offense: np.ndarray, defense: np.ndarray) -> np.ndarray:
    """Which defender owns the space each attacker stands in.

    Ownership rather than proximity. During a switch both defenders are close
    to both attackers for a moment and a nearest-defender assignment flickers
    between them; cell ownership changes once, cleanly, which is what makes a
    switch detectable at all.
    """
    attackers = np.asarray(offense, dtype=float).reshape(-1, 2)
    defenders = np.asarray(defense, dtype=float).reshape(-1, 2)
    if len(attackers) == 0 or len(defenders) == 0:
        return np.full(len(attackers), -1, dtype=int)
    gaps = np.linalg.norm(attackers[:, None, :] - defenders[None, :, :], axis=2)
    return np.argmin(gaps, axis=1)


def switched(before: np.ndarray, after: np.ndarray,
             pair: tuple[int, int]) -> bool:
    """Whether two defenders exchanged the men they were responsible for.

    This is the one defensive read that is a fact rather than a judgement: the
    assignment either swapped or it did not. Everything else about coverage --
    drop, hedge, ICE -- is a matter of degree.
    """
    first, second = pair
    if max(first, second) >= min(len(before), len(after)):
        return False
    return (before[first] != after[first]
            and before[second] != after[second]
            and before[first] == after[second]
            and before[second] == after[first])


def near_elbows(offense: np.ndarray) -> int:
    """How many attackers stand at the free-throw elbows."""
    points = np.asarray(offense, dtype=float).reshape(-1, 2)
    if not len(points):
        return 0
    count = 0
    for x, y in points:
        if any(np.hypot(x - ex, y - FREE_THROW_LINE_Y) <= ELBOW_TOLERANCE_FT
               for ex in ELBOW_X):
            count += 1
    return count


def in_corners(offense: np.ndarray) -> int:
    """How many attackers stand in the two corners."""
    points = np.asarray(offense, dtype=float).reshape(-1, 2)
    if not len(points):
        return 0
    count = 0
    for x, y in points:
        near_baseline = y <= CORNER_TOLERANCE_FT
        near_sideline = min(x, COURT_WIDTH - x) <= CORNER_TOLERANCE_FT
        count += bool(near_baseline and near_sideline)
    return count


def is_horns(offense: np.ndarray) -> bool:
    """Two attackers at the elbows and two in the corners.

    A formation is a configuration, not an interaction, so this is one of the
    few things in this file a rule can settle. Two people looking at the same
    frame would agree -- which is not true of a pin-down, and is why formations
    are where weak supervision here is worth trusting.
    """
    return near_elbows(offense) >= 2 and in_corners(offense) >= 2
