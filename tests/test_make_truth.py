"""The judgement format, on cases where the intended truth is obvious."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_truth  # noqa: E402


def test_ok_confirms_the_system_s_own_report_as_the_truth_point():
    out = make_truth.parse_spec("ok", [[100.0, 50.0]], 2.0, 34.0, [])
    assert out == [{"centre": [100.0, 50.0], "width": 34.0}]


def test_a_hand_point_is_scaled_from_panel_to_frame():
    out = make_truth.parse_spec("x:120,88", [], 2.0, 34.0, [])
    assert out == [{"centre": [240.0, 176.0], "width": 34.0}]


def test_a_dash_means_the_object_is_not_in_the_picture():
    assert make_truth.parse_spec("-", [[1.0, 2.0]], 2.0, 34.0, []) == []


def test_two_baskets_confirm_two_reports_in_order():
    out = make_truth.parse_spec("ok;ok", [[10.0, 10.0], [900.0, 12.0]], 2.0, 34.0, [])
    assert [o["centre"] for o in out] == [[10.0, 10.0], [900.0, 12.0]]


def test_a_confirmed_report_can_sit_beside_an_unclaimed_one():
    out = make_truth.parse_spec("ok;+x:520,90", [[10.0, 10.0]], 2.0, 34.0, [])
    assert [o["centre"] for o in out] == [[10.0, 10.0], [1040.0, 180.0]]


def test_confirming_a_report_that_was_never_made_is_an_error():
    """Otherwise a typo silently invents truth that flatters the system."""
    with pytest.raises(ValueError):
        make_truth.parse_spec("ok", [], 2.0, 34.0, [])


def test_an_unreadable_spec_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        make_truth.parse_spec("probably-fine", [], 2.0, 34.0, [])
