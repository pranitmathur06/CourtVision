"""Agreement across inference sizes, and what counts as a big ball."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mine_big_balls import (  # noqa: E402
    MIN_BALL_PX, agreeing_groups, box_size, centre_of,
)


def test_the_longer_side_is_the_diameter():
    # Occlusion clips a ball's box on one axis; the other still shows its size.
    assert box_size([0, 0, 50, 20]) == 50


def test_the_centre_is_the_middle():
    assert centre_of([10, 20, 30, 60]) == (20.0, 40.0)


def test_one_size_alone_is_not_enough():
    assert agreeing_groups([(320, [0, 0, 50, 50], 0.4)]) == []


def test_two_sizes_on_the_same_object_agree():
    got = agreeing_groups([(320, [0, 0, 50, 50], 0.4),
                           (416, [3, 2, 53, 52], 0.3)])
    assert len(got) == 1
    box, conf, sizes = got[0]
    assert sizes == [320, 416]
    assert conf == 0.4  # the best member's confidence, not the last one's


def test_two_hits_from_the_SAME_size_do_not_agree():
    # One size firing twice on one object is not corroboration.
    assert agreeing_groups([(320, [0, 0, 50, 50], 0.4),
                            (320, [3, 2, 53, 52], 0.3)]) == []


def test_far_apart_boxes_stay_separate_objects():
    assert agreeing_groups([(320, [0, 0, 50, 50], 0.4),
                            (416, [600, 400, 650, 450], 0.3)]) == []


def test_three_sizes_on_one_object_are_reported_once():
    got = agreeing_groups([(256, [0, 0, 50, 50], 0.2),
                           (320, [2, 2, 52, 52], 0.5),
                           (416, [1, 1, 51, 51], 0.3)])
    assert len(got) == 1 and got[0][2] == [256, 320, 416]


def test_the_size_floor_is_above_an_ordinary_broadcast_ball():
    # The native crops already cover 15-25 px; this mine is for what they miss.
    assert MIN_BALL_PX > 30


def test_a_candidate_at_the_top_middle_of_a_player_is_a_head():
    from mine_big_balls import looks_like_a_head
    player = [100, 200, 180, 400]          # 80 wide, 200 tall
    assert looks_like_a_head([130, 210, 150, 230], [player])


def test_a_ball_held_at_chest_height_is_not_a_head():
    from mine_big_balls import looks_like_a_head
    player = [100, 200, 180, 400]
    assert not looks_like_a_head([130, 290, 150, 310], [player])


def test_a_candidate_beside_a_player_is_not_a_head():
    from mine_big_balls import looks_like_a_head
    player = [100, 200, 180, 400]
    assert not looks_like_a_head([300, 210, 320, 230], [player])


def test_a_candidate_at_the_top_but_far_to_one_side_is_not_a_head():
    # A ball raised beside the head, still inside the box, is not the head.
    from mine_big_balls import looks_like_a_head
    player = [100, 200, 180, 400]
    assert not looks_like_a_head([102, 210, 112, 220], [player])


def test_no_players_means_nothing_is_a_head():
    from mine_big_balls import looks_like_a_head
    assert not looks_like_a_head([130, 210, 150, 230], [])
