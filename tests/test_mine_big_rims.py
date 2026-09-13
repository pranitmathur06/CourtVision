"""The colour-and-shape queue: what it offers a labeller and what it refuses."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mine_big_rims import MIN_WIDTH_PX, ring_candidates  # noqa: E402


def _frame():
    return np.zeros((720, 1280, 3), np.uint8)


def _paint(frame, x0, y0, x1, y1, bgr=(20, 60, 220)):
    frame[y0:y1, x0:x1] = bgr
    return frame


def test_a_big_orange_flattened_blob_is_offered():
    got = ring_candidates(_paint(_frame(), 500, 300, 800, 400))
    assert got and got[0]["w"] >= 290


def test_a_small_ring_is_not_offered():
    # Distant rims are the class that already works; they are not the queue.
    assert ring_candidates(_paint(_frame(), 500, 300, 560, 330)) == []


def test_a_blue_blob_is_not_offered():
    assert ring_candidates(_paint(_frame(), 500, 300, 800, 400, (220, 60, 20))) == []


def test_the_scoreboard_band_at_the_bottom_is_ignored():
    # Permanently red, permanently there, and never a rim.
    assert ring_candidates(_paint(_frame(), 300, 640, 900, 700)) == []


def test_the_top_furniture_band_is_ignored():
    assert ring_candidates(_paint(_frame(), 300, 5, 900, 35)) == []


def test_a_long_thin_red_bar_is_refused_as_a_hoarding():
    assert ring_candidates(_paint(_frame(), 200, 350, 1100, 368)) == []


def test_a_tall_red_column_is_refused():
    assert ring_candidates(_paint(_frame(), 500, 150, 700, 600)) == []


def test_the_floor_is_above_the_rims_that_already_work():
    assert MIN_WIDTH_PX >= 120
