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


def test_crop_player_stays_inside_the_image():
    from courtvision.action_classifier import FRAME_SIZE, crop_player
    from courtvision.types import Box

    image = np.zeros((200, 300, 3), dtype=np.uint8)
    # A box hard against the top-left corner: padding must clamp, not go negative.
    crop = crop_player(image, Box(0, 0, 30, 60))
    assert crop.shape == (FRAME_SIZE, FRAME_SIZE, 3)


def test_crop_player_selects_the_box_region():
    from courtvision.action_classifier import crop_player
    from courtvision.types import Box

    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[100:150, 100:130] = 255  # a bright patch where the player is
    crop = crop_player(image, Box(100, 100, 130, 150), margin=0.0)
    assert crop.mean() > 200  # the crop is dominated by the patch


def test_classify_windows_labels_other_when_nobody_has_the_ball():
    from courtvision.action_classifier import classify_windows

    config = Config()
    n = 32
    images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(n)]
    frames = [Frame(i, i / config.target_fps, ()) for i in range(n)]

    class ExplodingClassifier:
        def classify(self, clip):
            raise AssertionError("must not run the model when there is no holder")

    windows = classify_windows(
        images, frames, ExplodingClassifier(), config, holders=[None] * n
    )
    assert len(windows) == 3
    assert all(w.label == "other" and w.conf == 0.0 for w in windows)


def test_classify_windows_crops_to_the_holder():
    from courtvision.action_classifier import FRAME_SIZE, classify_windows
    from courtvision.types import PLAYER, Box, Track

    config = Config()
    n = 32
    images = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(n)]
    for img in images:
        img[20:80, 20:50] = 255  # holder 7 is the bright region
    frames = [
        Frame(i, i / config.target_fps,
              (Track(7, Box(20, 20, 50, 80), PLAYER, 0.9),
               Track(9, Box(150, 20, 180, 80), PLAYER, 0.9)))
        for i in range(n)
    ]

    seen = {}

    class RecordingClassifier:
        def classify(self, clip):
            seen["shape"] = clip.shape
            seen["mean"] = float(clip.mean())
            return "dribble", 0.9

    windows = classify_windows(
        images, frames, RecordingClassifier(), config, holders=[7] * n
    )
    assert seen["shape"] == (16, FRAME_SIZE, FRAME_SIZE, 3)
    # The point of cropping is that the holder fills the frame. Compare against
    # the whole-frame mean: the patch is 4.5% of the image but ~33% of the crop
    # (30x60 box padded 25% on each side), so the crop is several times brighter.
    whole_frame_mean = float(images[0].mean())
    assert seen["mean"] > 4 * whole_frame_mean
    assert all(w.label == "dribble" for w in windows)


def test_classify_windows_uses_one_batched_call():
    """All windows go to the model together, not one at a time."""
    from courtvision.action_classifier import classify_windows
    from courtvision.types import PLAYER, Box, Track

    config = Config()
    n = 40
    images = [np.zeros((120, 120, 3), dtype=np.uint8) for _ in range(n)]
    frames = [
        Frame(i, i / config.target_fps, (Track(7, Box(10, 10, 40, 90), PLAYER, 0.9),))
        for i in range(n)
    ]

    calls = {"batch": 0, "single": 0, "sizes": []}

    class BatchingClassifier:
        def classify(self, clip):
            calls["single"] += 1
            return "shot", 0.5

        def classify_batch(self, clips):
            calls["batch"] += 1
            calls["sizes"].append(len(clips))
            return [("dribble", 0.8)] * len(clips)

    windows = classify_windows(
        images, frames, BatchingClassifier(), config, holders=[7] * n
    )

    assert calls["batch"] == 1, "expected exactly one batched call"
    assert calls["single"] == 0, "must not fall back to per-clip classification"
    assert calls["sizes"] == [len(windows)]
    assert all(w.label == "dribble" and w.conf == 0.8 for w in windows)


def test_classify_windows_falls_back_when_only_classify_exists():
    """A classifier with no classify_batch still works, one clip at a time."""
    from courtvision.action_classifier import classify_windows
    from courtvision.types import PLAYER, Box, Track

    config = Config()
    n = 40
    images = [np.zeros((120, 120, 3), dtype=np.uint8) for _ in range(n)]
    frames = [
        Frame(i, i / config.target_fps, (Track(7, Box(10, 10, 40, 90), PLAYER, 0.9),))
        for i in range(n)
    ]

    class SingleOnly:
        calls = 0

        def classify(self, clip):
            SingleOnly.calls += 1
            return "pass", 0.7

    windows = classify_windows(images, frames, SingleOnly(), config, holders=[7] * n)
    assert SingleOnly.calls == len(windows)
    assert all(w.label == "pass" for w in windows)


def test_classify_windows_mixes_batched_and_no_holder_windows():
    """Windows with no holder are labelled `other` and never reach the model."""
    from courtvision.action_classifier import classify_windows
    from courtvision.types import PLAYER, Box, Track

    config = Config()
    n = 40
    images = [np.zeros((120, 120, 3), dtype=np.uint8) for _ in range(n)]
    frames = [
        Frame(i, i / config.target_fps, (Track(7, Box(10, 10, 40, 90), PLAYER, 0.9),))
        for i in range(n)
    ]
    holders = [7] * 16 + [None] * (n - 16)   # first window held, later ones not

    seen = {"n": 0}

    class Counting:
        def classify_batch(self, clips):
            seen["n"] = len(clips)
            return [("shot", 0.9)] * len(clips)

    windows = classify_windows(images, frames, Counting(), config, holders=holders)
    held = [w for w in windows if w.label == "shot"]
    unheld = [w for w in windows if w.label == "other"]
    assert seen["n"] == len(held) < len(windows)
    assert unheld and all(w.conf == 0.0 for w in unheld)
