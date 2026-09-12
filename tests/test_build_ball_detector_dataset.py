"""The crop window, which must keep the ball at the size the detector will meet."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_ball_detector_dataset as data  # noqa: E402


def test_the_window_contains_the_ball():
    rng = random.Random(0)
    for _ in range(50):
        got = data.crop_window((720, 1280, 3), [600.0, 300.0, 618.0, 318.0], rng)
        assert got is not None
        x0, y0, size = got
        assert x0 <= 600 and 618 <= x0 + size
        assert y0 <= 300 and 318 <= y0 + size


def test_the_window_is_not_resized_so_the_ball_keeps_its_size():
    """The ball's whole difficulty is that it is 15-25 px; enlarging it lies."""
    rng = random.Random(0)
    _, _, size = data.crop_window((720, 1280, 3), [600.0, 300.0, 618.0, 318.0], rng)
    assert size == min(data.CROP, 720)


def test_the_window_moves_around_between_calls():
    rng = random.Random(1)
    seen = {data.crop_window((720, 1280, 3), [600.0, 300.0, 618.0, 318.0], rng)[:2]
            for _ in range(30)}
    assert len(seen) > 1


def test_a_ball_too_near_the_edge_for_a_window_is_refused():
    rng = random.Random(0)
    assert data.crop_window((720, 1280, 3), [0.0, 0.0, 4.0, 4.0], rng) is None


def test_the_window_never_leaves_the_frame():
    rng = random.Random(2)
    for _ in range(50):
        got = data.crop_window((720, 1280, 3), [1200.0, 650.0, 1218.0, 668.0], rng)
        if got is None:
            continue
        x0, y0, size = got
        assert 0 <= x0 and x0 + size <= 1280
        assert 0 <= y0 and y0 + size <= 720
