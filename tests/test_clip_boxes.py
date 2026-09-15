"""The ball path, the tracked rim, and the gaps that are and are not filled."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from clip_boxes import best_path, interpolate, iou, link, smooth  # noqa: E402


def box(x, y, w=20, h=20):
    return [x, y, x + w, y + h]


def test_a_single_confident_ball_is_followed():
    frames = [[(box(i * 10, 0), 0.9)] for i in range(6)]
    got = best_path(frames)
    assert len(got) == 6 and got[3][0] == 30


def test_a_confident_decoy_far_away_does_not_steal_the_path():
    # This is the measured fault: per-frame argmax teleported the box to a head
    # that briefly outscored the ball. The path must stay with the trajectory.
    frames = []
    for i in range(7):
        row = [(box(i * 10, 0), 0.45)]
        if i == 3:
            row.append((box(900, 600), 0.95))     # a head, far off the path
        frames.append(row)
    got = best_path(frames)
    assert got[3][0] == 30, "the path jumped to the decoy"


def test_a_frame_with_nothing_plausible_becomes_a_gap():
    frames = [[(box(0, 0), 0.9)], [], [(box(20, 0), 0.9)]]
    got = best_path(frames)
    assert 1 not in got and 0 in got and 2 in got


def test_a_lone_weak_candidate_miles_away_is_refused():
    # Better to claim no ball than to put the box on a spectator.
    frames = [[(box(0, 0), 0.9)], [(box(1200, 700), 0.12)], [(box(20, 0), 0.9)]]
    assert 1 not in best_path(frames)


def test_no_candidates_at_all_gives_no_path():
    assert best_path([[], [], []]) == {}


def test_a_moving_rim_keeps_one_track():
    # The camera pans: the rim slides across the picture and must stay one box.
    per_frame = [[box(300 + i * 6, 100, 40, 18)] for i in range(12)]
    tracks = link(per_frame)
    assert len(tracks) == 1


def test_smoothing_does_not_flatten_a_pan():
    moving = {i: box(300 + i * 6, 100) for i in range(12)}
    out = smooth(moving)
    assert out[0][0] < out[5][0] < out[11][0]


def test_a_short_gap_is_interpolated():
    got = interpolate({0: box(0, 0), 4: box(40, 0)})
    assert 2 in got and got[2][0] == 20


def test_a_long_gap_is_left_alone():
    got = interpolate({0: box(0, 0), 40: box(400, 0)}, max_gap=10)
    assert sorted(got) == [0, 40]


def test_overlap_is_zero_for_disjoint_boxes():
    assert iou(box(0, 0), box(500, 500)) == 0.0
