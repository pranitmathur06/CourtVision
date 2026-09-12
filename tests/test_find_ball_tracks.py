"""The smoothness test that decides whether a chain of candidates is a ball."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import find_ball_tracks as tracks  # noqa: E402


def test_a_straight_fast_path_is_a_ball():
    points = [[100.0 + 60 * i, 200.0 - 10 * i] for i in range(6)]
    assert tracks.smooth_enough(points)


def test_a_gently_bending_path_is_a_ball():
    """A shot arcs; the test must not insist on a straight line."""
    points = [[100.0 + 60 * i, 300.0 - 80 * i + 8 * i * i] for i in range(6)]
    assert tracks.smooth_enough(points)


def test_a_standing_still_chain_is_not_a_ball():
    """This is the spectator's head that four other rules could not remove."""
    points = [[100.0, 200.0] for _ in range(6)]
    assert not tracks.smooth_enough(points)


def test_a_chain_that_jumps_between_objects_is_not_a_ball():
    points = [[100.0, 200.0], [140.0, 200.0], [700.0, 90.0], [180.0, 200.0]]
    assert not tracks.smooth_enough(points)


def test_a_jogging_player_is_too_slow_to_be_a_ball_in_flight():
    """Smoothness alone labelled 42% heads and shoulders; speed is the fix."""
    points = [[100.0 + 20 * i, 200.0] for i in range(6)]
    assert not tracks.smooth_enough(points)


def test_a_zigzag_is_not_a_ball():
    points = [[100.0, 200.0], [180.0, 200.0], [100.0, 200.0], [180.0, 200.0],
              [100.0, 200.0]]
    assert not tracks.smooth_enough(points)


def test_too_short_a_chain_is_refused():
    assert not tracks.smooth_enough([[0.0, 0.0], [60.0, 0.0]])


def test_best_track_finds_the_moving_thing_among_still_ones():
    frames = []
    for i in range(6):
        frames.append(np.array([[500.0, 500.0],                 # a still head
                                [100.0 + 60 * i, 200.0]]))      # the ball
    got = tracks.best_track(frames)
    assert got is not None
    chain, points = got
    assert len(chain) == 6
    assert all(pick == 1 for _, pick in chain)


def test_best_track_returns_nothing_when_everything_is_still():
    frames = [np.array([[500.0, 500.0], [10.0, 10.0]]) for _ in range(6)]
    assert tracks.best_track(frames) is None
