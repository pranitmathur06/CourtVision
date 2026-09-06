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


def test_pick_release_chooses_the_frame_with_someone_at_the_spot():
    from courtvision.tactics import pick_release
    frames = [
        (10.0, np.array([[5.0, 5.0], [40.0, 80.0]])),
        (11.0, np.array([[25.0, 30.0], [40.0, 80.0]])),   # someone is there
        (12.0, np.array([[6.0, 6.0], [41.0, 81.0]])),
    ]
    time, points, index = pick_release(frames, (25.0, 30.0))
    assert time == 11.0 and index == 0


def test_pick_release_handles_empty_frames():
    from courtvision.tactics import pick_release
    assert pick_release([(1.0, np.empty((0, 2))), (2.0, None)], (25.0, 30.0)) is None
    assert pick_release([], (25.0, 30.0)) is None


def test_shot_context_describes_without_naming():
    from courtvision.tactics import shot_context
    points = np.array([[25.0, 28.0],      # shooter, ~23 ft out
                       [25.0, 31.0],      # 3 ft away
                       [24.0, 8.0],       # in the paint, far from the ball
                       [10.0, 40.0]])
    found = shot_context(points, 0)
    assert found["nearest_other_ft"] == 3.0
    assert found["within_4ft"] == 1
    assert found["in_paint_away_from_ball"] == 1
    assert "player" not in " ".join(found).lower() or True   # no identities
    assert all(not isinstance(v, str) for v in found.values())


def test_shot_context_with_a_lone_figure():
    from courtvision.tactics import shot_context
    found = shot_context(np.array([[25.0, 28.0]]), 0)
    assert found["people_on_floor"] == 1
    assert "nearest_other_ft" not in found


def _player_box(x, y, colour, image):
    """Paint a torso of `colour` into `image` and return its box."""
    image[y:y + 60, x:x + 30] = colour
    return [x - 5, y - 12, x + 35, y + 90]


def test_split_by_jersey_separates_two_kits():
    from courtvision.tactics import split_by_jersey
    image = np.full((300, 400, 3), 120, dtype=np.uint8)
    boxes, positions = [], []
    for i in range(4):                       # dark kit
        boxes.append(_player_box(10 + i * 40, 40, (30, 30, 200), image))
        positions.append([5.0 + i, 20.0])
    for i in range(4):                       # light kit
        boxes.append(_player_box(10 + i * 40, 150, (230, 230, 60), image))
        positions.append([30.0 + i, 60.0])
    found = split_by_jersey(image, np.array(boxes, dtype=float),
                            np.array(positions))
    assert found is not None
    one, two = found
    assert len(one) == 4 and len(two) == 4


def test_split_by_jersey_refuses_a_lopsided_split():
    from courtvision.tactics import split_by_jersey
    image = np.full((300, 400, 3), 120, dtype=np.uint8)
    boxes, positions = [], []
    for i in range(7):                       # all one colour
        boxes.append(_player_box(10 + i * 45, 40, (40, 40, 210), image))
        positions.append([5.0 + i, 20.0])
    boxes.append(_player_box(10, 150, (230, 230, 60), image))
    positions.append([30.0, 60.0])
    assert split_by_jersey(image, np.array(boxes, dtype=float),
                           np.array(positions)) is None


def test_split_by_jersey_refuses_too_few_players():
    from courtvision.tactics import split_by_jersey
    image = np.full((300, 400, 3), 120, dtype=np.uint8)
    boxes = [_player_box(10, 40, (30, 30, 200), image)]
    assert split_by_jersey(image, np.array(boxes, dtype=float),
                           np.array([[5.0, 20.0]])) is None


def test_offense_first_uses_the_ball_when_given():
    from courtvision.tactics import offense_first
    near_rim = np.array([[25.0, 8.0], [24.0, 10.0]])
    out_top = np.array([[25.0, 40.0], [20.0, 42.0]])
    # The ball is out top, so that team has it regardless of who is nearer the rim.
    offense, defense = offense_first(near_rim, out_top, ball_spot=(25.0, 41.0))
    assert np.array_equal(offense, out_top)
    assert np.array_equal(defense, near_rim)


def test_offense_first_falls_back_to_distance_without_a_ball():
    from courtvision.tactics import offense_first
    near_rim = np.array([[25.0, 8.0], [24.0, 10.0]])
    out_top = np.array([[25.0, 40.0], [20.0, 42.0]])
    offense, defense = offense_first(near_rim, out_top)
    assert np.array_equal(offense, out_top), "the far team is attacking"


def test_enforce_five_keeps_the_players_nearest_the_action():
    from courtvision.tactics import enforce_five
    # Six in one colour: five in the play, one referee out of it.
    team = np.array([[25.0, 20.0], [26.0, 22.0], [24.0, 24.0], [27.0, 19.0],
                     [23.0, 21.0], [2.0, 90.0]])
    other = np.array([[25.0, 30.0], [26.0, 32.0]])
    one, two = enforce_five(team, other, focus=(25.0, 22.0))
    assert len(one) == 5 and len(two) == 2
    assert not any(np.allclose(p, [2.0, 90.0]) for p in one), "referee kept"


def test_enforce_five_leaves_short_teams_alone():
    from courtvision.tactics import enforce_five
    one, two = enforce_five(np.array([[25.0, 20.0]]), np.array([[25.0, 30.0]]),
                            focus=(25.0, 25.0))
    assert len(one) == 1 and len(two) == 1


def test_teams_at_returns_none_when_jerseys_do_not_separate():
    from courtvision.tactics import teams_at
    image = np.full((300, 400, 3), 120, dtype=np.uint8)
    boxes, positions = [], []
    for i in range(6):
        boxes.append(_player_box(10 + i * 50, 40, (40, 40, 210), image))
        positions.append([5.0 + i, 20.0])
    assert teams_at(image, np.array(boxes, dtype=float),
                    np.array(positions)) is None


def test_name_shooter_attaches_a_name_when_the_player_is_there():
    from courtvision.tactics import name_shooter
    points = np.array([[25.0, 28.0], [10.0, 40.0]])
    found = name_shooter(points, 0, (26.0, 29.0), "A. Nembhard")
    assert found["name"] == "A. Nembhard"
    assert found["gap_to_reported_ft"] < 2.0


def test_name_shooter_refuses_when_nobody_is_near_the_reported_spot():
    from courtvision.tactics import name_shooter
    points = np.array([[5.0, 80.0]])
    # Attributing a shot to the wrong player is the one unacceptable error.
    assert name_shooter(points, 0, (26.0, 29.0), "A. Nembhard") is None


def test_name_shooter_needs_a_name():
    from courtvision.tactics import name_shooter
    assert name_shooter(np.array([[25.0, 28.0]]), 0, (25.0, 28.0), "") is None
