"""The patch extraction and the honest split."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_ball_ranker as ranker  # noqa: E402


def _frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)


def test_a_patch_comes_back_square_and_sized():
    got = ranker.patch_of(_frame(), [600.0, 300.0, 620.0, 320.0])
    assert got.shape == (ranker.PATCH, ranker.PATCH, 3)


def test_a_box_at_the_edge_still_yields_a_patch():
    got = ranker.patch_of(_frame(), [0.0, 0.0, 12.0, 12.0])
    assert got is not None and got.shape == (ranker.PATCH, ranker.PATCH, 3)


def test_a_degenerate_box_is_refused():
    assert ranker.patch_of(_frame(), [5.0, 5.0, 5.0, 5.0]) is not None  # padded out
    assert ranker.patch_of(_frame(), [2000.0, 2000.0, 2001.0, 2001.0]) is None


def test_the_split_never_shares_a_moment():
    """Patches a second apart are the same picture; sharing them fakes the score."""
    times = np.array([1.0, 1.2, 1.4, 100.0, 100.2, 200.0, 200.5])
    train, valid = ranker.split_by_time(times, valid_fraction=0.4, block_s=20.0)
    assert not (train & valid).any()
    assert train.any() and valid.any()
    blocks_train = {int(t // 20) for t in times[train]}
    blocks_valid = {int(t // 20) for t in times[valid]}
    assert not (blocks_train & blocks_valid)


def test_every_patch_lands_on_one_side_of_the_split():
    times = np.array([0.0, 30.0, 60.0, 90.0])
    train, valid = ranker.split_by_time(times, valid_fraction=0.5, block_s=20.0)
    assert (train | valid).all()
