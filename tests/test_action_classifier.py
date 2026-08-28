import numpy as np

from courtvision.action_classifier import classify_windows, plan_windows
from courtvision.config import Config
from courtvision.types import ACTIONS, Frame
from tests.fixtures.synthetic import StubActionClassifier


def test_plan_windows_covers_a_clip_at_stride():
    windows = plan_windows(n_frames=32, size=16, stride=8)
    assert windows == [(0, 15), (8, 23), (16, 31)]


def test_plan_windows_bounds_are_inclusive_and_sized():
    for start, end in plan_windows(n_frames=64, size=16, stride=8):
        assert end - start + 1 == 16


def test_plan_windows_returns_nothing_for_too_short_a_clip():
    assert plan_windows(n_frames=10, size=16, stride=8) == []


def test_plan_windows_handles_exact_single_window():
    assert plan_windows(n_frames=16, size=16, stride=8) == [(0, 15)]


def test_classify_windows_produces_labelled_windows():
    config = Config()
    n = 32
    images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(n)]
    frames = [Frame(i, i / config.target_fps, ()) for i in range(n)]

    windows = classify_windows(images, frames, StubActionClassifier("shot"), config)

    assert len(windows) == 3
    assert all(w.label == "shot" for w in windows)
    assert all(w.label in ACTIONS for w in windows)
    assert windows[0].start_index == 0
    assert windows[0].end_index == 15
    assert windows[0].start_time_s == 0.0
    assert windows[1].start_time_s > windows[0].start_time_s
