"""The statistics every scorer shares, after four copies of `wilson` became one.

The copies agreed, which was luck. A scorer whose interval drifts from another
scorer's is a scorer whose numbers cannot be compared, and comparing numbers is
this project's whole method.
"""

from __future__ import annotations

import math

import pytest

from courtvision.stats import (block_bootstrap, chi_square_tail,
                               cluster_bootstrap, cochran_q, iou, mcnemar,
                               wilson)


def test_wilson_stays_inside_zero_and_one_where_the_normal_approximation_does_not():
    """At n=13 a normal approximation puts the interval outside [0, 1].

    Which is the sample size the ball was measured on for most of this
    project's life.
    """
    low, high = wilson(5, 13)
    assert 0.0 <= low < 5 / 13 < high <= 1.0
    assert wilson(13, 13)[1] == 1.0
    assert wilson(0, 13)[0] == 0.0


def test_nothing_measured_is_not_a_claim_of_zero():
    """An empty denominator is no idea, not a confident failure.

    Three of the four copies this module replaced returned (0, 0) and the
    fourth returned (0, 1). They were never consistent; consolidating them is
    what made the disagreement show up. A per-game report on a new broadcast
    will routinely have classes with no instances yet, and every one of them
    would otherwise print 0%-0%.
    """
    assert wilson(0, 0) == (0.0, 1.0)


def test_wilson_narrows_as_the_sample_grows():
    def width(n):
        low, high = wilson(int(0.6 * n), n)
        return high - low
    assert width(157) > width(600) > width(1500)


def test_mcnemar_only_counts_the_disagreements():
    """Agreements carry no information about which method is better."""
    a = [True] * 50 + [True, False]
    b = [True] * 50 + [False, True]
    assert mcnemar(a, b) == (1, 1, 1.0)


def test_mcnemar_finds_a_one_sided_result_significant():
    a = [True] * 10 + [False] * 0
    b = [False] * 10
    only_a, only_b, p = mcnemar(a, b)
    assert (only_a, only_b) == (10, 0)
    assert p < 0.01


def test_mcnemar_is_symmetric_in_its_p_value():
    a = [True, True, True, False, False]
    b = [False, False, True, True, False]
    assert mcnemar(a, b)[2] == mcnemar(b, a)[2]


def test_the_chi_square_tail_matches_a_known_table():
    """3.841 at one degree of freedom is the 0.05 point, and must land there."""
    assert chi_square_tail(3.8415, 1) == pytest.approx(0.05, abs=1e-4)
    assert chi_square_tail(5.9915, 2) == pytest.approx(0.05, abs=1e-4)
    assert chi_square_tail(7.8147, 3) == pytest.approx(0.05, abs=1e-4)
    assert chi_square_tail(0.0, 4) == 1.0


def test_cochran_q_is_quiet_when_every_game_behaves_the_same():
    same = [[True] * 6 + [False] * 4] * 3
    statistic, df, p = cochran_q(same)
    assert df == 2 and statistic == pytest.approx(0.0) and p == 1.0


def test_cochran_q_fires_when_one_game_is_different():
    """The per-game regression alarm.

    Three games of 150 frames cannot each carry an interval tight enough to see
    a five-point move, but asking whether one game DIFFERS from the others is a
    far more powerful question on the same data -- and it is the question "did
    this game regress" actually is.
    """
    spread = [[True] * 9 + [False], [True] * 5 + [False] * 5,
              [True] * 1 + [False] * 9]
    statistic, df, p = cochran_q(spread)
    assert df == 2 and statistic > 5.0 and p < 0.05


def test_a_cluster_bootstrap_is_wider_than_wilson_on_correlated_items():
    """Boxes within a frame are not independent observations.

    Ten frames of ten boxes is not a hundred draws, and a Wilson interval over
    box counts is too narrow by roughly the square root of the boxes per frame.
    """
    groups = [(10, 10)] * 5 + [(0, 10)] * 5
    low, high = cluster_bootstrap(groups, draws=4000, seed=1)
    tight_low, tight_high = wilson(50, 100)
    assert (high - low) > (tight_high - tight_low)


def test_the_block_bootstrap_resamples_whole_blocks():
    """Events near each other in a game are not independent.

    Round 65 measured what resampling individual attempts does: it reported a
    meaningless 0.375-0.456 on data whose blocks disagree wildly.
    """
    items = ([(t * 1.0, True) for t in range(120)]
             + [(120.0 + t, False) for t in range(120)])
    low, high = block_bootstrap(items, span_s=240.0, block_s=120.0,
                                draws=800, seed=0)
    assert low < 0.5 < high
    assert (high - low) > 0.3, "two blocks that disagree must give a wide answer"


def test_overlap_is_zero_for_disjoint_boxes_and_one_for_identical_ones():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
