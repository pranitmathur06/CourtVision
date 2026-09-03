"""Choosing the ball among candidates, which is where the video pipeline failed.

Measured on a real broadcast the detector emits a mean of 14.2 ball boxes per
frame with a median confidence of 0.050. Taking the argmax gave end-to-end shot
detection F1 0.37 against 0.885 for the same method on clean coordinates. The
ball was being SEEN and not SELECTED, so these pin selection.
"""

from courtvision.ball_track import choose


def test_a_smooth_faint_track_beats_a_loud_teleporting_one():
    """The whole point. A distractor six times more confident, but jumping 800
    px per frame, must lose to the ball."""
    frames = []
    for i in range(10):
        ball = (100.0 + 5 * i, 200.0, 0.10)
        noise = (900.0 if i % 2 else 100.0, 50.0, 0.60)
        frames.append([ball, noise])
    picked = choose(frames)
    assert all(p is not None and abs(p[1] - 200.0) < 1e-6 for p in picked)


def test_the_confident_candidate_wins_when_neither_moves():
    """With no motion to separate them, confidence is all there is."""
    frames = [[(100.0, 100.0, 0.2), (500.0, 500.0, 0.9)] for _ in range(6)]
    picked = choose(frames)
    assert all(p == (500.0, 500.0) for p in picked)


def test_a_frame_with_no_candidates_yields_none():
    frames = [[(10.0, 10.0, 0.5)], [], [(14.0, 10.0, 0.5)]]
    picked = choose(frames)
    assert picked[1] is None
    assert picked[0] is not None and picked[2] is not None


def test_a_lone_wild_detection_is_dropped_rather_than_followed():
    """One frame's only candidate sits 900 px away. Following it would cost two
    big jumps; the path pays `missing_cost` once instead."""
    frames = []
    for i in range(9):
        frames.append([(100.0 + 4 * i, 300.0, 0.30)])
    frames[4] = [(1000.0, 40.0, 0.30)]
    picked = choose(frames)
    assert picked[4] is None


def test_no_frames_is_not_an_error():
    assert choose([]) == []
