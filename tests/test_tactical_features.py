"""Derivatives, matchups by spatial control, and formation.

Each of these exists because a position-only description failed at something
specific, and the tests pin the specific failure rather than the happy path.
"""

import numpy as np
import pytest

from courtvision.tactical_features import (control_grid, derivatives,
                                           in_corners, is_horns, matchups,
                                           near_elbows, switched)


def test_speed_is_per_second_not_per_frame():
    """A threshold in feet per second must not change meaning with the rate."""
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    slow = derivatives(path, hz=10.0)["speed"]
    fast = derivatives(path, hz=25.0)["speed"]
    assert np.isclose(slow[1], 10.0) and np.isclose(fast[1], 25.0)


def test_a_planted_screener_shows_as_a_speed_of_zero():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                     [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    speed = derivatives(path, hz=10.0)["speed"]
    assert speed[0] > 5.0 and speed[-1] == pytest.approx(0.0)


def test_a_hard_cut_shows_as_a_turn_rate():
    """A change of direction is invisible in speed alone."""
    straight = np.array([[0.0, float(i)] for i in range(8)])
    cut = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0],
                    [1.0, 3.0], [2.0, 3.0], [3.0, 3.0], [4.0, 3.0]])
    assert derivatives(cut, hz=10.0)["turn"].max() > \
        derivatives(straight, hz=10.0)["turn"].max()


def test_derivatives_tolerate_a_track_too_short_to_differentiate():
    out = derivatives(np.array([[0.0, 0.0]]), hz=10.0)
    assert out["speed"].shape == (1,) and out["acceleration"][0] == 0.0


def test_matchup_is_by_who_owns_the_space():
    offense = np.array([[10.0, 20.0], [30.0, 20.0]])
    defense = np.array([[11.0, 21.0], [31.0, 21.0]])
    assert list(matchups(offense, defense)) == [0, 1]


def test_a_switch_is_an_exchange_not_a_drift():
    """Both assignments must swap; one defender drifting is not a switch."""
    before = np.array([0, 1])
    assert switched(before, np.array([1, 0]), (0, 1))
    assert not switched(before, np.array([1, 1]), (0, 1))
    assert not switched(before, np.array([0, 1]), (0, 1))


def test_switch_ignores_a_pair_that_is_not_present():
    assert not switched(np.array([0]), np.array([0]), (0, 3))


def test_control_grid_assigns_every_patch_to_a_defender():
    grid, owner = control_grid(np.empty((0, 2)),
                               np.array([[10.0, 10.0], [40.0, 40.0]]))
    assert len(grid) == len(owner)
    assert set(np.unique(owner)) <= {0, 1}
    # A patch beside the first defender belongs to him.
    near = int(np.argmin(np.linalg.norm(grid - np.array([10.0, 10.0]), axis=1)))
    assert owner[near] == 0


def test_control_grid_with_no_defenders_is_empty_not_an_error():
    grid, owner = control_grid(np.array([[10.0, 10.0]]), np.empty((0, 2)))
    assert len(grid) == 0 and len(owner) == 0


def test_horns_needs_both_elbows_and_both_corners():
    horns = np.array([[17.0, 19.0], [33.0, 19.0], [3.0, 5.0], [47.0, 5.0],
                      [25.0, 32.0]])
    assert is_horns(horns)
    # The same two bigs at the elbows, but nobody in the corners.
    no_corners = np.array([[17.0, 19.0], [33.0, 19.0], [12.0, 25.0],
                           [38.0, 25.0], [25.0, 32.0]])
    assert near_elbows(no_corners) >= 2 and not is_horns(no_corners)


def test_a_flat_set_is_not_horns():
    flat = np.array([[10.0, 25.0], [20.0, 25.0], [30.0, 25.0],
                     [40.0, 25.0], [25.0, 32.0]])
    assert not is_horns(flat)


def test_corners_are_baseline_and_sideline_together():
    """A player on the wing is near a sideline but is not in the corner."""
    wing = np.array([[3.0, 25.0]])
    assert in_corners(wing) == 0
    assert in_corners(np.array([[3.0, 4.0]])) == 1
