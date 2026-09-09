"""The painted-line geometry a registration is checked against."""

import numpy as np
import pytest

from courtvision.court_lines import (CORNER_THREE_Y, COURT_LENGTH, COURT_WIDTH,
                                     court_lines, line_mask)


def test_every_line_stays_on_the_court():
    for polyline in court_lines():
        for x, y in polyline:
            assert -0.01 <= x <= COURT_WIDTH + 0.01, (x, y)
            assert -0.01 <= y <= COURT_LENGTH + 0.01, (x, y)


def test_both_ends_are_drawn():
    """A registration on the wrong basket must match markings there too.

    Drawing only one end would let a flipped registration score well simply by
    finding no lines to disagree with at the other.
    """
    ys = [y for polyline in court_lines() for _, y in polyline]
    assert min(ys) < 1.0 and max(ys) > COURT_LENGTH - 1.0
    near = sum(1 for y in ys if y < COURT_LENGTH / 2)
    far = sum(1 for y in ys if y > COURT_LENGTH / 2)
    assert abs(near - far) < 0.05 * len(ys), "the two ends must be drawn alike"


def test_the_three_point_arc_meets_the_corner_line_where_it_should():
    """The arc and the straight corner section have to join, not merely both exist."""
    from courtvision.court import BASKET, CORNER_THREE_X, THREE_POINT_RADIUS

    gap = np.hypot(CORNER_THREE_X - BASKET[0], CORNER_THREE_Y - BASKET[1])
    assert abs(gap - THREE_POINT_RADIUS) < 0.35, (
        "the corner three's endpoint must sit on the arc's radius")


def test_the_mask_is_mostly_floor():
    pytest.importorskip("cv2")
    mask = line_mask(8.0)
    painted = (mask > 0).mean()
    assert 0.02 < painted < 0.25, (
        f"{painted:.1%} painted -- a mask this dense would score any "
        "registration well")


def test_the_mask_is_symmetric_about_half_court():
    """Pins the reason the line check cannot detect an end swap.

    This is a property of the sport, not a bug, but it is load-bearing: if a
    future change made the mask asymmetric, the end-swap blind spot documented
    in check_court_lines.py would quietly stop being true.
    """
    pytest.importorskip("cv2")
    mask = line_mask(8.0) > 0
    overlap = (mask & mask[::-1]).sum() / mask.sum()
    assert overlap > 0.95, f"only {overlap:.1%} of paint maps onto paint when flipped"
