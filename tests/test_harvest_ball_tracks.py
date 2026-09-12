"""The bounce counter and the two ways a chain is accepted as a ball."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from harvest_ball_tracks import (  # noqa: E402
    MIN_BOUNCE_PX, bounces, classify, dribble_like, flight_like,
)


def _dribble(cycles=3, amplitude=30.0, per_cycle=8, drift=6.0):
    """A ball bouncing while its handler walks sideways.

    A triangle between hand height and the floor; y grows downward, so the
    floor is the larger value.
    """
    points = []
    for i in range(cycles * per_cycle):
        phase = (i % per_cycle) / per_cycle
        rise = 2 * phase if phase < 0.5 else 2 * (1 - phase)
        points.append([i * drift, 300.0 + amplitude * rise])
    return np.array(points)


def test_a_flat_line_never_bounces():
    assert bounces(np.full(20, 300.0)) == 0


def test_jitter_below_the_amplitude_never_bounces():
    rng = np.random.default_rng(0)
    assert bounces(300.0 + rng.normal(0, MIN_BOUNCE_PX / 8, 40)) == 0


def test_a_bouncing_series_counts_its_reversals():
    assert bounces(_dribble()[:, 1]) >= 2


def test_a_single_rise_and_fall_is_one_reversal_not_two():
    series = np.concatenate([np.linspace(300, 240, 8), np.linspace(240, 300, 8)])
    assert bounces(series) == 1


def test_a_dribble_is_accepted_as_a_dribble():
    assert dribble_like(_dribble())
    assert classify(_dribble()) == "dribble"


def test_a_stationary_head_is_not_a_dribble():
    rng = np.random.default_rng(1)
    still = np.c_[300 + rng.normal(0, 1.5, 20), 200 + rng.normal(0, 1.5, 20)]
    assert not dribble_like(still)
    assert classify(still) is None


def test_a_player_running_smoothly_is_not_a_dribble():
    # Steady sideways travel with no vertical excursion: a shoulder, which is
    # exactly what poisoned the labels at a speed floor of 14.
    walk = np.c_[np.arange(20) * 9.0, np.full(20, 200.0)]
    assert not dribble_like(walk)
    assert classify(walk) is None


def test_a_ball_in_flight_is_accepted_as_flight():
    arc = np.c_[np.arange(8) * 60.0, 300 - np.arange(8) * 40 + np.arange(8) ** 2 * 4]
    assert flight_like(arc)
    assert classify(arc) == "flight"


def test_a_dribble_is_too_slow_to_pass_the_flight_rule():
    # The whole reason this script exists: the flight floor cannot see it.
    assert not flight_like(_dribble())


def test_a_ball_dribbled_on_the_spot_is_still_a_ball():
    # No sideways travel at all, but it bounces: a handler working the top of
    # the key. Rejecting this would throw away the population this exists for.
    on_the_spot = np.c_[np.zeros(20), 300 + np.tile([0, MIN_BOUNCE_PX + 1], 10)]
    assert dribble_like(on_the_spot)


def test_a_stationary_object_wobbling_below_the_amplitude_is_rejected():
    # The same shape with a ball-sized bounce removed: detector jitter on a
    # head, which is what this must never label.
    jitter = np.c_[np.zeros(20), 300 + np.tile([0, MIN_BOUNCE_PX / 3], 10)]
    assert not dribble_like(jitter)


def test_too_few_points_decide_nothing():
    assert classify(np.array([[0.0, 0.0], [1.0, 1.0]])) is None
