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
