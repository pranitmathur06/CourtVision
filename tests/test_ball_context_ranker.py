"""The neighbourhood patch, its centre mark, and an honest split."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_ball_context_ranker import (  # noqa: E402
    CONTEXT_PX, INPUT_PX, context_patch, split_by_time,
)


def _frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)


def test_the_patch_is_three_colour_channels():
    patch = context_patch(_frame(), (640, 360))
    assert patch.shape == (INPUT_PX, INPUT_PX, 3)


def test_two_candidates_a_few_pixels_apart_get_different_patches():
    # Centring is the only thing that says which object is being scored, so
    # neighbours in a crowd must not collapse to the same picture.
    frame = _frame()
    near = context_patch(frame, (640, 360))
    also = context_patch(frame, (658, 372))
    assert not np.allclose(near, also)


def test_a_clipped_patch_is_padded_so_the_candidate_stays_centred():
    frame = _frame()
    patch = context_patch(frame, (4, 4))
    assert patch.shape == (INPUT_PX, INPUT_PX, 3)
    # Everything up and left of the candidate fell outside the frame.
    assert patch[:INPUT_PX // 3, :INPUT_PX // 3].max() == 0.0


def test_context_is_wider_than_the_ball():
    # The whole argument for this ranker: the neighbourhood, not the blob.
    assert CONTEXT_PX >= 128


def test_colour_channels_are_scaled_to_the_unit_range():
    patch = context_patch(_frame(), (640, 360))
    assert 0.0 <= patch.min() and patch.max() <= 1.0


def test_a_candidate_off_the_frame_gets_no_patch():
    assert context_patch(_frame(), (-400, 360)) is None


def test_the_split_never_shares_a_moment():
    times = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 200.0, 201.0, 300.0])
    train, valid = split_by_time(times, valid_fraction=0.25)
    assert not set(times[train]) & set(times[valid])
    assert train.any() and valid.any()


def test_the_split_puts_the_later_game_in_validation():
    times = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0])
    train, valid = split_by_time(times, valid_fraction=0.25)
    assert times[train].max() < times[valid].min()
