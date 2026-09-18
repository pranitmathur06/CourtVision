"""The court-motion ball selector, and the sign error it exists to correct."""

from __future__ import annotations

from courtvision.ball_track import choose, choose_ballistic, choose_moving


def _flying_ball_and_a_confident_decoy(steps: int = 7, speed: float = 12.0):
    """The measured failure: the decoy is still, brighter, and wrong."""
    return [[(100.0 + speed * t, 200.0, 0.30), (500.0, 600.0, 0.55)]
            for t in range(steps)]


def _picked_ball(path, speed: float = 12.0):
    return [p is not None and abs(p[0] - (100.0 + speed * i)) < 1.0
            for i, p in enumerate(path)]


def test_a_still_decoy_has_zero_acceleration_too():
    """Why `choose_ballistic` cannot settle this on its own.

    A constant-velocity prior is indifferent between a ball flying in a
    straight line and a logo painted on the floor, because zero velocity is a
    constant velocity. Recorded as a test so the next reader does not rebuild
    it expecting otherwise.
    """
    frames = _flying_ball_and_a_confident_decoy()
    assert not any(_picked_ball(choose_ballistic(frames)))


def test_the_smoothness_prior_prefers_the_decoy():
    """`choose` rewards holding still, which is exactly what the decoys do."""
    frames = _flying_ball_and_a_confident_decoy()
    assert not any(_picked_ball(choose(frames)))


def test_court_motion_takes_the_ball_when_the_camera_is_still():
    frames = _flying_ball_and_a_confident_decoy()
    shifts = [None] + [(0.0, 0.0)] * (len(frames) - 1)
    picked = choose_moving(frames, shifts, still_px=8.0, still_weight=0.4)
    assert sum(_picked_ball(picked)) >= len(frames) - 1


def test_the_camera_is_subtracted_before_anything_is_called_still():
    """On a pan, the decoy moves in pixels and the rim moves with it.

    Without the shift the painted floor looks as lively as the ball. With it,
    the decoy is stationary again and the ball is not. This is the whole
    argument for the rim being the frame of reference.
    """
    pan = 20.0
    frames = [[(100.0 + 12.0 * t + pan * t, 200.0, 0.30),
               (500.0 + pan * t, 600.0, 0.55)] for t in range(7)]
    shifts = [None] + [(pan, 0.0)] * 6

    def took_ball(path):
        return sum(1 for i, p in enumerate(path)
                   if p is not None and abs(p[0] - (100.0 + 32.0 * i)) < 1.0)

    assert took_ball(choose_moving(frames, shifts, still_px=8.0,
                                   still_weight=0.4)) >= 6
    assert took_ball(choose_moving(frames, None, still_px=8.0,
                                   still_weight=0.4)) == 0


def test_a_teleport_is_refused_however_confident():
    frames = [[(0.0, 0.0, 0.9)], [(5000.0, 5000.0, 0.99)], [(12.0, 0.0, 0.9)]]
    picked = choose_moving(frames, [None, (0.0, 0.0), (0.0, 0.0)],
                           max_speed_px=400.0)
    assert picked[1] is None


def test_empty_and_single_frame_windows_do_not_raise():
    assert choose_moving([]) == []
    assert choose_moving([[]]) == [None]
    assert choose_moving([[(1.0, 2.0, 0.5)]]) == [(1.0, 2.0)]
    assert choose_ballistic([]) == []
    assert choose_ballistic([[(1.0, 2.0, 0.5)]]) == [(1.0, 2.0)]


def test_a_frame_with_no_candidate_is_crossed_rather_than_ending_the_path():
    frames = [[(0.0, 0.0, 0.9)], [], [(24.0, 0.0, 0.9)]]
    picked = choose_moving(frames, [None, None, None])
    assert picked[0] is not None and picked[1] is None and picked[2] is not None
