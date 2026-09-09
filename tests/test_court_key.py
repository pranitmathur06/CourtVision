"""Registration from the painted key."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.court_key import (KEY_COURT_CORNERS, court_region,
                                   key_homography, key_quad,
                                   order_key_corners)


def _synthetic_court(width: int = 640, height: int = 360):
    """Hardwood with a blue key painted on it, plus blue 'crowd' above it.

    The crowd band is the case that broke the first version: this arena's seats
    are the same hue as the paint, so a colour mask alone finds both.
    """
    cv2 = pytest.importorskip("cv2")
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2] = 15, 120, 200      # wood
    hsv[: height // 4, :, 0] = 110                                # crowd hue
    hsv[: height // 4, :, 1] = 200
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    key = np.array([[200, 150], [400, 150], [400, 300], [200, 300]])
    cv2.fillPoly(image, [key], (200, 120, 30))                    # BGR blue
    return image, key


def test_court_region_finds_the_wood_and_not_the_crowd():
    image, _ = _synthetic_court()
    region = court_region(image)
    assert region is not None
    assert region[300, 320] == 255, "the floor should be inside the region"
    assert region[10, 320] == 0, "the crowd band should not be"


def test_key_quad_finds_the_painted_key():
    image, key = _synthetic_court()
    quad = key_quad(image)
    assert quad is not None and len(quad) == 4
    found = np.array(sorted(map(tuple, quad.astype(int))))
    expected = np.array(sorted(map(tuple, key)))
    assert np.abs(found - expected).max() <= 6


def test_key_quad_ignores_blue_crowd_outside_the_court():
    cv2 = pytest.importorskip("cv2")
    image, key = _synthetic_court()
    # A large blue block in the stands must not win over the real key.
    cv2.fillPoly(image, [np.array([[10, 5], [300, 5], [300, 80], [10, 80]])],
                 (200, 120, 30))
    quad = key_quad(image)
    assert quad is not None
    assert quad[:, 1].min() > 100, "picked the crowd block, not the key"


def test_order_key_corners_puts_the_baseline_first():
    _, key = _synthetic_court()
    quad = np.array(key, dtype=np.float32)
    # A rim below the key makes the y=300 edge the baseline pair.
    ordered = order_key_corners(quad, (300.0, 340.0))
    assert ordered is not None
    assert ordered[0][1] == pytest.approx(300, abs=1)
    assert ordered[1][1] == pytest.approx(300, abs=1)
    assert ordered[2][1] == pytest.approx(150, abs=1)


def test_order_key_corners_refuses_without_a_rim():
    _, key = _synthetic_court()
    # Guessing the baseline flips the court end for end; refusing is correct.
    assert order_key_corners(np.array(key, dtype=np.float32), None) is None


def test_key_homography_maps_corners_to_their_court_coordinates():
    image, _ = _synthetic_court()
    matrix = key_homography(image, (300.0, 340.0))
    assert matrix is not None
    quad = order_key_corners(key_quad(image), (300.0, 340.0))
    homogeneous = np.hstack([quad, np.ones((4, 1))])
    projected = homogeneous @ matrix.T
    court = projected[:, :2] / projected[:, 2:3]
    assert np.abs(court - KEY_COURT_CORNERS).max() < 1.0, "within a foot"


def test_key_homography_is_none_without_a_rim():
    image, _ = _synthetic_court()
    assert key_homography(image, None) is None


def test_key_homography_is_none_on_a_frame_with_no_court():
    cv2 = pytest.importorskip("cv2")
    hsv = np.zeros((360, 640, 3), dtype=np.uint8)
    hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2] = 110, 200, 60      # all crowd
    crowd = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert key_homography(crowd, (300.0, 340.0)) is None


def test_key_rim_distance_measures_from_the_quad_centre():
    from courtvision.court_key import key_rim_distance
    quad = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert key_rim_distance(quad, (5.0, 5.0)) == 0.0
    assert key_rim_distance(quad, (5.0, 15.0)) == 10.0


def test_key_rim_distance_is_none_without_a_rim():
    from courtvision.court_key import key_rim_distance
    quad = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert key_rim_distance(quad, None) is None


def test_key_matches_rim_rejects_the_other_basket():
    from courtvision.court_key import key_matches_rim
    quad = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    # A rim 600 px away belongs to the far basket, not this key.
    assert not key_matches_rim(quad, (605.0, 5.0))
    assert key_matches_rim(quad, (100.0, 5.0))


def test_key_matches_rim_is_false_without_a_rim():
    from courtvision.court_key import key_matches_rim
    quad = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert not key_matches_rim(quad, None)


def _floor_with_paint(hue: int, size=(300, 400)):
    """Hardwood with a key painted at a given hue."""
    cv2 = pytest.importorskip("cv2")
    hsv = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2] = 15, 120, 200      # wood
    hsv[120:260, 150:330, 0] = hue                               # the key
    hsv[120:260, 150:330, 1] = 200
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_detect_paint_hue_finds_a_blue_key():
    from courtvision.court_key import detect_paint_hue
    found = detect_paint_hue([_floor_with_paint(110)] * 3)
    assert found is not None and found[0] <= 110 <= found[1]


def test_detect_paint_hue_finds_a_RED_key():
    from courtvision.court_key import detect_paint_hue
    # The arena that broke the hardcoded range: 10% of frames became 87%.
    found = detect_paint_hue([_floor_with_paint(174)] * 3)
    assert found is not None and found[0] <= 174 <= found[1]


def test_detect_paint_hue_ignores_the_wooden_floor():
    from courtvision.court_key import detect_paint_hue
    cv2 = pytest.importorskip("cv2")
    hsv = np.zeros((300, 400, 3), dtype=np.uint8)
    hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2] = 15, 120, 200
    bare = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Wood is hue 15 and must never be mistaken for paint.
    assert detect_paint_hue([bare] * 3) is None


def test_key_quad_uses_a_calibrated_hue():
    from courtvision.court_key import key_quad
    red = _floor_with_paint(174)
    assert key_quad(red) is None, "the default blue range must miss red paint"
    assert key_quad(red, None, (160, 179)) is not None


# --- Phase 0: defects that were written but wired to nothing ---------------


def test_precise_key_corners_takes_a_paint_hue_without_raising():
    """It used `paint_hue` without taking it, so any call that reached the
    body raised NameError. A blank frame returned early, which is why the
    fault stayed hidden -- this uses a frame that reaches the body.
    """
    pytest.importorskip("cv2")
    from courtvision.court_key import key_quad, precise_key_corners
    image, _ = _synthetic_court()
    quad = key_quad(image)
    assert quad is not None, "the synthetic key must be findable"
    # Reaches the body; before the fix this raised NameError, and passing the
    # keyword raised TypeError.
    precise_key_corners(image, quad)
    precise_key_corners(image, quad, paint_hue=(95, 125))


def test_the_gate_rejects_a_key_far_from_the_rim():
    """The gate is what separates p50 1.72 ft from p50 2.84 ft.

    `build_game_model` ran ungated, so a key belonging to the other basket
    registered happily. Both rims below find the same quad; only the distance
    differs.
    """
    pytest.importorskip("cv2")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from build_game_model import court_positions

    image, key = _synthetic_court()
    boxes = np.array([[220.0, 180.0, 260.0, 295.0]])
    # The synthetic key spans x 200-400, y 150-300; its centre is (300, 225).
    near_rim = (300.0, 160.0)      # just above the key: the same basket
    far_rim = (620.0, 60.0)        # across the frame: a different basket

    assert court_positions(image, near_rim, boxes, gate=True) is not None
    assert court_positions(image, far_rim, boxes, gate=True) is None
    # Ungated, the far rim registers anyway -- the behaviour before this fix.
    assert court_positions(image, far_rim, boxes, gate=False) is not None
