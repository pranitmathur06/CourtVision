"""Smoothing, the static rim, and which ball gaps get filled."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from clip_detect_dense import fill_gaps, link, median_box, smooth_track  # noqa: E402


def test_smoothing_pulls_a_jittering_box_toward_its_neighbours():
    jitter = [[0, 0, 10, 10], [6, 0, 16, 10], [0, 0, 10, 10]]
    out = smooth_track(jitter, window=1)
    assert 0 < out[1][0] < 6          # the spike is pulled back


def test_smoothing_leaves_a_steady_box_alone():
    steady = [[0, 0, 10, 10]] * 5
    assert smooth_track(steady, window=2) == [[0, 0, 10, 10]] * 5


def test_smoothing_follows_real_movement():
    moving = [[i * 10, 0, i * 10 + 10, 10] for i in range(9)]
    out = smooth_track(moving, window=2)
    assert out[0][0] < out[4][0] < out[8][0]


def test_the_rim_is_one_box_despite_a_wild_frame():
    boxes = [[100, 50, 140, 70]] * 6 + [[600, 300, 640, 320]]
    assert median_box(boxes) == [100, 50, 140, 70]


def test_no_rim_seen_means_no_rim_drawn():
    assert median_box([]) is None


def test_a_short_ball_gap_is_interpolated():
    got = fill_gaps({0: [0, 0, 10, 10], 4: [40, 0, 50, 10]}, 10)
    assert 2 in got and got[2][0] == 20


def test_a_long_ball_gap_is_left_empty():
    # Half a second with no ball is not a ball whose position is known.
    got = fill_gaps({0: [0, 0, 10, 10], 20: [200, 0, 210, 10]}, 30, max_gap=6)
    assert set(got) == {0, 20}


def test_a_player_moving_steadily_stays_one_tracklet():
    per_frame = [[[i * 2, 0, i * 2 + 40, 90]] for i in range(10)]
    tracks = link(per_frame)
    assert len(tracks) == 1 and len(next(iter(tracks.values()))) == 10


def test_two_players_do_not_merge():
    per_frame = [[[0, 0, 40, 90], [300, 0, 340, 90]] for _ in range(5)]
    assert len(link(per_frame)) == 2


def test_the_subject_gap_fill_keeps_the_box_from_blinking():
    # The one box a viewer is watching must not disappear for three frames
    # because the detector lost the player behind someone.
    from clip_detect_dense import MAX_SUBJECT_GAP, fill_gaps
    got = fill_gaps({0: [0, 0, 40, 90], 4: [40, 0, 80, 90]}, 10, MAX_SUBJECT_GAP)
    assert sorted(got) == [0, 1, 2, 3, 4]


def test_a_long_absence_is_not_filled_in():
    from clip_detect_dense import MAX_SUBJECT_GAP, fill_gaps
    got = fill_gaps({0: [0, 0, 40, 90], 40: [40, 0, 80, 90]}, 60, MAX_SUBJECT_GAP)
    assert sorted(got) == [0, 40]


def test_a_foul_is_not_an_act_of_the_ball_handler():
    # The fouler is usually a defender who never touches the ball, so the
    # subject must be left unset rather than pointed at whoever was holding it.
    from clip_detect_dense import BALL_ACTS
    assert "foul" not in BALL_ACTS
    assert "shot" in BALL_ACTS and "made shot" in BALL_ACTS
