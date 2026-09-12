"""Tiling covers the frame, and the orange measure means what it says."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probe_ball_proposals import orange_share, tile_origins  # noqa: E402


def test_tiles_reach_the_right_and_bottom_edges():
    origins = tile_origins(1280, 720, 320, 0.3)
    assert max(x for x, _ in origins) == 1280 - 320
    assert max(y for _, y in origins) == 720 - 320


def test_tiles_overlap_so_a_ball_on_a_seam_is_whole_somewhere():
    origins = sorted({x for x, _ in tile_origins(1280, 720, 320, 0.3)})
    steps = np.diff(origins)
    assert steps.max() < 320


def test_every_pixel_is_covered_by_some_tile():
    tile = 320
    covered = np.zeros((720, 1280), bool)
    for x, y in tile_origins(1280, 720, tile, 0.3):
        covered[y:y + tile, x:x + tile] = True
    assert covered.all()


def test_a_frame_smaller_than_one_tile_still_yields_a_window():
    assert tile_origins(200, 100, 320, 0.3) == [(0, 0)]


def test_an_all_orange_patch_scores_one():
    patch = np.zeros((8, 8, 3), np.uint8)
    patch[..., 0], patch[..., 1], patch[..., 2] = 12, 200, 200
    assert orange_share(patch) == 1.0


def test_a_blue_patch_scores_zero():
    patch = np.zeros((8, 8, 3), np.uint8)
    patch[..., 0], patch[..., 1], patch[..., 2] = 110, 200, 200
    assert orange_share(patch) == 0.0


def test_orange_but_washed_out_does_not_count():
    # Right hue, no saturation: a grey smudge, which is what a blurred ball is.
    patch = np.zeros((8, 8, 3), np.uint8)
    patch[..., 0], patch[..., 1], patch[..., 2] = 12, 20, 200
    assert orange_share(patch) == 0.0


def test_an_empty_patch_is_silence_not_zero():
    assert orange_share(np.zeros((0, 4, 3), np.uint8)) is None
