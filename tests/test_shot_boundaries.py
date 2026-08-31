import numpy as np

from courtvision.shot_boundaries import (MIN_SEGMENT_FRAMES, cut_frames,
                                         segments)


def _scene(value: int, n: int, jitter: int = 3) -> list[np.ndarray]:
    """n frames of one scene, with small frame-to-frame movement."""
    rng = np.random.default_rng(value)
    base = np.full((90, 160, 3), value, np.uint8)
    out = []
    for _ in range(n):
        noise = rng.integers(-jitter, jitter + 1, base.shape, dtype=np.int16)
        out.append(np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    return out


def test_finds_the_boundary_between_two_scenes():
    frames = _scene(40, 20) + _scene(210, 20)
    assert cut_frames(frames) == [20]


def test_motion_within_one_scene_is_not_a_cut():
    """Players move fast; that must not read as a camera change."""
    frames = _scene(120, 40, jitter=25)
    assert cut_frames(frames) == []


def test_segments_split_at_cuts():
    spans = segments(40, [20])
    assert spans == [(0, 20), (20, 40)]


def test_a_segment_too_short_to_hold_a_window_is_dropped():
    """Keeping it would only invite windows straddling its edges."""
    spans = segments(40, [2, 20])
    assert (0, 2) not in spans
    assert all(end - start >= MIN_SEGMENT_FRAMES for start, end in spans)


def test_no_cuts_gives_one_segment():
    assert segments(50, []) == [(0, 50)]


def test_a_single_frame_has_no_cuts():
    assert cut_frames([np.zeros((90, 160, 3), np.uint8)]) == []


def test_windows_are_planned_inside_segments_only():
    """No window may span a cut.

    plan_windows over the whole clip would place one across the boundary; given
    segments it must not.
    """
    from courtvision.action_classifier import plan_windows

    segs = [(0, 32), (32, 64)]
    planned = []
    for start, end in segs:
        for a, b in plan_windows(end - start, 16, 8):
            planned.append((start + a, start + b))
    assert planned, "segments long enough must yield windows"
    for a, b in planned:
        assert any(s <= a and b < e for s, e in segs), \
            f"window ({a},{b}) crosses a segment boundary"
