"""The crop-and-zoom labels, on a frame where the answer is arithmetic."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_rim_dataset as data  # noqa: E402


class _NoJitter:
    def uniform(self, a, b):
        return 0.0

    def random(self):
        return 0.5

    def randrange(self, n):
        return 0


def _frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)


def test_the_box_survives_the_zoom_in_the_right_place():
    box = [600.0, 300.0, 640.0, 320.0]          # 40x20 rim at the centre-ish
    for zoom in (1.0, 4.0):
        patch, label = data.crop_around(_frame(), box, zoom, rng=_NoJitter())
        assert patch.shape == (data.OUT_SIZE, data.OUT_SIZE, 3)
        cx, cy, w, h = label
        assert 0 < cx < 1 and 0 < cy < 1
        # The rim must occupy more of the picture the further it is zoomed.
        if zoom == 1.0:
            small = w
        else:
            assert w > small * 2


def test_a_rim_too_big_for_the_crop_is_refused_rather_than_clipped():
    """A clipped box would teach the model a rim's edge is a rim."""
    box = [200.0, 200.0, 700.0, 400.0]          # wider than an 8x crop window
    patch, label = data.crop_around(_frame(), box, 8.0, rng=_NoJitter())
    assert patch is None and label is None


def test_a_negative_crop_contains_no_rim():
    box = [600.0, 300.0, 640.0, 320.0]
    empty = data.crop_without(_frame(), [box], 4.0, rng=np.random.default_rng(1))
    assert empty is None or empty.shape == (data.OUT_SIZE, data.OUT_SIZE, 3)


def test_zooming_keeps_the_label_a_valid_yolo_box():
    box = [600.0, 300.0, 640.0, 320.0]
    _, label = data.crop_around(_frame(), box, 6.0, rng=_NoJitter())
    cx, cy, w, h = label
    assert 0 <= cx - w / 2 and cx + w / 2 <= 1
    assert 0 <= cy - h / 2 and cy + h / 2 <= 1
