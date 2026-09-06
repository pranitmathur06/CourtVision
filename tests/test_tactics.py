"""Offensive and defensive structure from court positions."""

from __future__ import annotations

import numpy as np

from courtvision.court import BASKET
from courtvision.tactics import (contest_level, defenders_near,
                                 distance_to_basket, help_defenders, in_paint,
                                 is_three_point_distance, nearest_defender_ft,
                                 spacing, split_by_side)


def test_distance_to_basket():
    points = np.array([[25.0, 5.25], [25.0, 25.25]])
    assert np.allclose(distance_to_basket(points), [0.0, 20.0])


def test_three_point_distance_uses_the_shorter_corner_line():
    # A corner spot 22.5 ft out is a three; the same distance at the top is not.
    assert is_three_point_distance((2.0, 5.25 + 22.5))
    assert not is_three_point_distance((25.0, 5.25 + 22.5))
    assert is_three_point_distance((25.0, 5.25 + 24.0))


def test_in_paint_marks_only_the_lane():
    points = np.array([[25.0, 10.0], [25.0, 25.0], [40.0, 10.0]])
    assert list(in_paint(points)) == [True, False, False]


def test_nearest_defender_and_contest_level():
    defenders = np.array([[25.0, 12.0], [10.0, 40.0]])
    assert nearest_defender_ft((25.0, 20.0), defenders) == 8.0
    assert contest_level((25.0, 20.0), defenders) == "open"
    assert contest_level((25.0, 17.0), defenders) == "guarded"   # 5 ft
    assert contest_level((25.0, 15.0), defenders) == "contested"  # 3 ft


def test_nearest_defender_without_defenders_is_unknown():
    assert nearest_defender_ft((25.0, 20.0), np.empty((0, 2))) is None
    assert contest_level((25.0, 20.0), np.empty((0, 2))) == "unknown"


def test_defenders_near_counts_inside_the_radius():
    defenders = np.array([[25.0, 8.0], [25.0, 9.0], [25.0, 30.0]])
    assert defenders_near((25.0, 10.0), defenders, radius_ft=4.0) == 2


def test_spacing_reports_the_tightest_pair():
    offense = np.array([[5.0, 10.0], [45.0, 10.0], [25.0, 30.0], [27.0, 31.0]])
    found = spacing(offense)
    assert found.players == 4
    assert found.width_ft == 40.0
    assert found.nearest_pair_ft < 2.5, "two players standing on each other"


def test_spacing_needs_at_least_two_players():
    assert spacing(np.array([[25.0, 10.0]])) is None
    assert spacing(None) is None


def test_help_defenders_counts_paint_bodies_away_from_the_ball():
    # Two defenders sunk into the paint while the ball is out at the arc.
    defenders = np.array([[24.0, 6.0], [27.0, 9.0], [25.0, 30.0]])
    assert help_defenders((25.0, 30.0), defenders) == 2


def test_help_defenders_ignores_the_one_guarding_the_ball():
    # A defender in the paint who IS on the ball is not help.
    defenders = np.array([[25.0, 10.0]])
    assert help_defenders((25.0, 12.0), defenders) == 0


def test_split_by_side_puts_the_five_nearest_the_rim_on_defense():
    points = np.array([[25.0, 5.0], [24.0, 7.0], [26.0, 8.0], [23.0, 9.0],
                       [27.0, 10.0], [25.0, 40.0], [10.0, 42.0]])
    offense, defense = split_by_side(points)
    assert len(defense) == 5 and len(offense) == 2
    assert distance_to_basket(offense).min() > distance_to_basket(defense).max()


def test_split_by_side_moves_the_ball_handler_to_offense():
    # A handler who drove to the rim sorts into the defensive five; the ball
    # spot pulls him back across.
    points = np.array([[25.0, 5.0], [24.0, 7.0], [26.0, 8.0], [23.0, 9.0],
                       [25.0, 6.0], [25.0, 40.0], [10.0, 42.0]])
    offense, defense = split_by_side(points, ball_spot=(25.0, 6.0))
    assert len(defense) == 4 and len(offense) == 3


def test_split_by_side_handles_too_few_people():
    points = np.array([[25.0, 10.0], [25.0, 20.0]])
    offense, defense = split_by_side(points)
    assert len(offense) == 2 and len(defense) == 0
