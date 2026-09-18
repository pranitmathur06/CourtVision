"""Choosing the court erosion against two bounds the sport supplies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "fit_court_mask", ROOT / "scripts" / "fit_court_mask.py")
fit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit)


def _rows(pairs):
    """{share: {carrier_kept, over_ok}} with denominators that pass."""
    return {share: {"carrier_kept": carrier, "carrier_n": 100,
                    "over_ok": over, "over_n": 100}
            for share, (carrier, over) in pairs.items()}


def test_it_takes_the_most_carriers_among_masks_that_obey_the_rule():
    rows = _rows({0.0: (0.98, 0.80), 0.03: (0.90, 0.96), 0.0625: (0.85, 0.99)})
    assert fit.choose(rows) == 0.03


def test_a_mask_that_admits_the_crowd_is_refused_however_many_carriers_it_keeps():
    """Keeping everybody never drops the carrier. One bound alone is
    degenerate, which is why there are two."""
    rows = _rows({0.0: (1.00, 0.10), 0.0625: (0.60, 0.99)})
    assert fit.choose(rows) == 0.0625


def test_nothing_clearing_the_floor_answers_none_rather_than_guessing():
    rows = _rows({0.0: (0.9, 0.5), 0.03: (0.8, 0.6)})
    assert fit.choose(rows) is None


def test_a_tie_on_carriers_goes_to_the_larger_erosion():
    """Admitting the front row costs more downstream than it costs here."""
    rows = _rows({0.0: (0.90, 0.99), 0.03: (0.90, 0.99)})
    assert fit.choose(rows) == 0.03


def test_an_empty_denominator_is_not_treated_as_a_pass():
    rows = {0.0: {"carrier_kept": 1.0, "carrier_n": 0,
                  "over_ok": 1.0, "over_n": 0}}
    assert fit.choose(rows) is None


def test_the_floor_and_the_impossible_count_are_the_declared_ones():
    assert fit.OVER_FLOOR == 0.95
    assert fit.IMPOSSIBLE_ABOVE == 13


def test_two_detectors_on_one_player_count_once_here_too():
    same = [[100.0, 100.0, 200.0, 400.0], [104.0, 98.0, 198.0, 402.0]]
    assert fit.distinct(same) == 1
    assert fit.distinct(same + [[600.0, 100.0, 700.0, 400.0]]) == 2


def test_distance_to_a_box_is_zero_inside_it():
    assert fit.to_box((50.0, 50.0), [0.0, 0.0, 100.0, 100.0]) == 0.0
    assert fit.to_box((0.0, 50.0), [10.0, 0.0, 100.0, 100.0]) == pytest.approx(10.0)
