"""The scoring order and the interval the ball gate is read through."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_ball_selection import scored_order, wilson  # noqa: E402


def test_the_best_score_comes_first():
    assert scored_order([0, 1, 2], [0.1, 0.9, 0.5]) == [1, 2, 0]


def test_ties_keep_the_earlier_candidate():
    assert scored_order([0, 1], [0.5, 0.5]) == [0, 1]


def test_a_single_candidate_is_rank_one():
    assert scored_order([0], [0.2]) == [0]


def test_a_perfect_small_sample_still_has_a_lower_bound_below_one():
    low, high = wilson(10, 10)
    assert low < 1.0 and high == 1.0


def test_three_of_ten_spans_the_measured_ball_interval():
    low, high = wilson(3, 10)
    assert 0.10 < low < 0.12 and 0.59 < high < 0.61


def test_nothing_measured_is_not_a_claim_of_zero():
    assert wilson(0, 0) == (0.0, 1.0)


def test_a_bigger_sample_narrows_the_interval():
    narrow = wilson(30, 100)
    wide = wilson(3, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])
