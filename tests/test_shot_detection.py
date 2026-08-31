"""Shots from ball trajectory. Thresholds were swept, not chosen.

Against the 212 official shots in 0021500492: ball-to-rim distance is 1.0 ft in
a window around a shot versus 13.6 ft at random, and peak ball height is 11.0 ft
versus 6.5 ft. Sweeping both gives precision 0.85, recall 0.70 at rim <= 6 ft
with a peak >= 8 ft.
"""

import math

from courtvision.shot_detection import (MIN_PEAK_HEIGHT_FT, RIMS,
                                        rim_distance, shots)
from courtvision.types import BALL, PLAYER, Box, Frame, Track


def _frame(index, t, ball_xy, ball_z_unused=None, players=()):
    tracks = [Track(-1, Box(ball_xy[0] - 0.4, ball_xy[1] - 0.4,
                            ball_xy[0] + 0.4, ball_xy[1] + 0.4), BALL, 1.0)]
    for pid, (x, y) in players:
        tracks.append(Track(pid, Box(x - 1, y - 3, x + 1, y + 3), PLAYER, 1.0))
    return Frame(index, t, tuple(tracks))


def _arc(start_xy, end_xy, peak_z, n=25, t0=0.0, dt=0.04):
    """A ball travelling from start to end, rising to peak_z and falling."""
    frames, zs = [], []
    for i in range(n):
        f = i / (n - 1)
        x = start_xy[0] + (end_xy[0] - start_xy[0]) * f
        y = start_xy[1] + (end_xy[1] - start_xy[1]) * f
        frames.append(_frame(i, t0 + i * dt, (x, y)))
        zs.append(peak_z * math.sin(math.pi * f))
    return frames, zs


def test_rim_distance_uses_the_nearer_rim():
    near_left = _frame(0, 0.0, (6.0, 25.0))
    assert rim_distance(near_left) < 2.0
    near_right = _frame(0, 0.0, (88.0, 25.0))
    assert rim_distance(near_right) < 2.0


def test_an_arc_into_the_rim_is_a_shot():
    frames, zs = _arc((25.0, 25.0), RIMS[0], peak_z=13.0)
    found = shots(frames, zs)
    assert len(found) == 1
    assert found[0].action == "shot"


def test_a_pass_across_the_court_is_not_a_shot():
    """It never approaches a rim, however far it travels."""
    frames, zs = _arc((20.0, 5.0), (70.0, 45.0), peak_z=7.0)
    assert shots(frames, zs) == []


def test_a_ball_carried_under_the_basket_is_not_a_shot():
    """Reaching the rim is not enough; it has to have gone up.

    Without the height floor precision drops from 0.85 to 0.77 on a real game.
    """
    frames, zs = _arc((15.0, 25.0), RIMS[0], peak_z=MIN_PEAK_HEIGHT_FT - 3.0)
    assert shots(frames, zs) == []


def test_two_approaches_close_together_are_one_attempt():
    """Overlapping windows must not score the same attempt twice."""
    a, za = _arc((25.0, 25.0), RIMS[0], peak_z=13.0, t0=0.0)
    b, zb = _arc((25.0, 25.0), RIMS[0], peak_z=13.0, t0=0.6)
    frames = a + [Frame(len(a) + i, f.time_s, f.tracks) for i, f in enumerate(b)]
    assert len(shots(frames, za + zb)) == 1


def test_two_attempts_far_apart_are_two_shots():
    a, za = _arc((25.0, 25.0), RIMS[0], peak_z=13.0, t0=0.0)
    b, zb = _arc((25.0, 25.0), RIMS[0], peak_z=13.0, t0=30.0)
    frames = a + [Frame(len(a) + i, f.time_s, f.tracks) for i, f in enumerate(b)]
    assert len(shots(frames, za + zb)) == 2


def test_missing_ball_frames_do_not_crash():
    frames, zs = _arc((25.0, 25.0), RIMS[0], peak_z=13.0)
    frames[5] = Frame(5, frames[5].time_s, tuple())
    assert len(shots(frames, zs)) == 1
