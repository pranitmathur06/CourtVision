"""Choosing which player and moment to classify.

Every guard here exists because its absence was measured on real broadcast:
spectators and officials entering the sample, and clips cut at instants where
nothing was happening.
"""

import numpy as np
import pytest

from courtvision.candidates import kit_members, near_ball, peak_score


def _frame(colours):
    """A frame with one solid-coloured torso block per player."""
    image = np.zeros((400, 60 * len(colours), 3), dtype=np.uint8)
    boxes = []
    for i, colour in enumerate(colours):
        x = i * 60
        image[100:260, x + 10:x + 50] = colour
        boxes.append([x + 5, 80, x + 55, 300])
    return image, np.array(boxes, dtype=float)


def test_two_kits_are_both_kept():
    pytest.importorskip("cv2")
    gold, blue = (40, 200, 230), (200, 90, 40)
    image, boxes = _frame([gold] * 5 + [blue] * 5)
    assert kit_members(image, boxes).sum() == 10


def test_an_official_between_the_kits_is_dropped():
    """Grey belongs to neither cluster, which is the whole point."""
    pytest.importorskip("cv2")
    gold, blue, grey = (40, 200, 230), (200, 90, 40), (128, 128, 128)
    image, boxes = _frame([gold] * 5 + [blue] * 5 + [grey])
    keep = kit_members(image, boxes)
    assert keep[:10].all() and not keep[10]


def test_one_colour_only_is_refused_rather_than_split():
    pytest.importorskip("cv2")
    gold = (40, 200, 230)
    image, boxes = _frame([gold] * 8)
    assert not kit_members(image, boxes).any()


def test_too_few_people_is_refused():
    pytest.importorskip("cv2")
    image, boxes = _frame([(40, 200, 230)] * 3)
    assert not kit_members(image, boxes).any()


def test_near_ball_measures_from_the_feet():
    boxes = np.array([[100.0, 100.0, 150.0, 300.0],
                      [900.0, 100.0, 950.0, 300.0]])
    keep = near_ball(boxes, (120.0, 310.0), limit_px=100.0)
    assert keep[0] and not keep[1]


def test_near_ball_with_no_ball_keeps_nobody():
    boxes = np.array([[100.0, 100.0, 150.0, 300.0]])
    assert not near_ball(boxes, None).any()


def test_peak_takes_the_strongest_offset():
    """A screen lasts a second; the clip must be cut where it happens."""
    assert peak_score([0.1, 0.9, 0.2]) == pytest.approx(0.9)
    assert peak_score([None, 0.3]) == pytest.approx(0.3)
    assert peak_score([]) == 0.0


# --- on the court, not beside it -------------------------------------------
#
# Colour cannot separate players from spectators in a "blue out" arena: the
# crowd wears the home kit, so a fan in a blue shirt clusters with a player in
# a blue jersey. Position can.

def _court_frame():
    """A bright floor band across the middle, crowd above and below."""
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    image[150:300] = (150, 190, 215)       # wood: warm and bright
    return image


def test_a_player_standing_on_the_floor_is_kept():
    pytest.importorskip("cv2")
    from courtvision.candidates import court_region, stands_on_court
    image = _court_frame()
    region = court_region(image, erode_px=5)
    boxes = np.array([[280.0, 120.0, 320.0, 250.0]])   # feet at y=250, on wood
    assert stands_on_court(region, boxes)[0]


def test_someone_beyond_the_edge_is_dropped():
    pytest.importorskip("cv2")
    from courtvision.candidates import court_region, stands_on_court
    image = _court_frame()
    region = court_region(image, erode_px=5)
    boxes = np.array([[280.0, 20.0, 320.0, 140.0]])    # feet above the floor
    assert not stands_on_court(region, boxes)[0]


def test_no_court_found_keeps_nobody():
    pytest.importorskip("cv2")
    from courtvision.candidates import stands_on_court
    assert not stands_on_court(None, np.array([[0.0, 0.0, 10.0, 10.0]])).any()
