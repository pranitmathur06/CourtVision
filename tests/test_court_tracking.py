"""Propagation must fill gaps without inventing coordinates."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.court_tracking import (pairwise_homography, propagate,
                                        segment_of)


def _textured(seed: int = 0, size: int = 400) -> np.ndarray:
    """An image with enough corner texture for ORB to key on."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (size, size), dtype=np.uint8)
    # Blur-free blocks give strong, repeatable corners.
    return np.kron(image[::8, ::8], np.ones((8, 8), dtype=np.uint8))


def test_pairwise_recovers_a_known_translation():
    cv2 = pytest.importorskip("cv2")
    base = _textured(1)
    shift = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -12.0], [0.0, 0.0, 1.0]])
    moved = cv2.warpPerspective(base, shift, (base.shape[1], base.shape[0]))

    # moved -> base, so the recovered matrix should undo the shift.
    found = pairwise_homography(moved, base)
    assert found is not None
    point = np.array([200.0, 200.0, 1.0])
    mapped = found @ point
    mapped = mapped[:2] / mapped[2]
    expected = np.array([200.0 - 25.0, 200.0 + 12.0])
    assert np.allclose(mapped, expected, atol=3.0)


def test_pairwise_refuses_unrelated_images():
    pytest.importorskip("cv2")
    assert pairwise_homography(_textured(2), _textured(9999)) is None


def test_segment_of_respects_cuts():
    assert segment_of(5, [10, 20], 30) == (0, 10)
    assert segment_of(10, [10, 20], 30) == (10, 20)
    assert segment_of(25, [10, 20], 30) == (20, 30)
    assert segment_of(3, [], 30) == (0, 30)


def test_propagate_fills_a_gap_from_one_anchor():
    cv2 = pytest.importorskip("cv2")
    base = _textured(3)
    images = [base]
    for step in range(1, 5):
        shift = np.array([[1.0, 0.0, 6.0 * step], [0.0, 1.0, 0.0], [0, 0, 1.0]])
        images.append(cv2.warpPerspective(base, shift,
                                          (base.shape[1], base.shape[0])))
    solved = {0: np.eye(3)}
    out = propagate(images, solved)
    assert set(out) == {0, 1, 2, 3, 4}
    # Frame 3 is shifted +18 px, so its court mapping must undo that.
    point = np.array([150.0, 150.0, 1.0])
    mapped = out[3] @ point
    mapped = mapped[:2] / mapped[2]
    assert np.allclose(mapped, [150.0 - 18.0, 150.0], atol=4.0)


def test_propagate_never_crosses_a_cut():
    cv2 = pytest.importorskip("cv2")
    base = _textured(4)
    same = cv2.warpPerspective(
        base, np.array([[1.0, 0, 5.0], [0, 1.0, 0], [0, 0, 1.0]]),
        (base.shape[1], base.shape[0]))
    images = [base, same, _textured(777), _textured(778)]
    out = propagate(images, {0: np.eye(3)}, cuts=[2])
    assert 1 in out, "the frame sharing the camera should be reached"
    assert 2 not in out and 3 not in out, "must not chain past a cut"


def test_propagate_keeps_a_real_solution_over_a_propagated_one():
    cv2 = pytest.importorskip("cv2")
    base = _textured(5)
    moved = cv2.warpPerspective(
        base, np.array([[1.0, 0, 4.0], [0, 1.0, 0], [0, 0, 1.0]]),
        (base.shape[1], base.shape[0]))
    marker = np.full((3, 3), 7.0)
    out = propagate([base, moved], {0: np.eye(3), 1: marker})
    assert np.array_equal(out[1], marker)


def test_propagate_without_anchors_returns_nothing():
    assert propagate([_textured(6)], {}) == {}


def test_max_chain_bounds_the_drift():
    cv2 = pytest.importorskip("cv2")
    base = _textured(7)
    images = [base]
    for step in range(1, 6):
        images.append(cv2.warpPerspective(
            base, np.array([[1.0, 0, 3.0 * step], [0, 1.0, 0], [0, 0, 1.0]]),
            (base.shape[1], base.shape[0])))
    out = propagate(images, {0: np.eye(3)}, max_chain=2)
    assert set(out) == {0, 1, 2}


def test_estimate_rig_falls_back_when_nothing_clears_the_threshold():
    from courtvision.court_tracking import estimate_rig
    # Real broadcast scores sat entirely between 0.30 and 0.45; refusing to fix
    # the rig there wasted a 943-second search.
    params = [np.array([50.0 + i, -20.0, 40.0, 25.0, 20.0, 1500.0])
              for i in range(9)]
    rig = estimate_rig(params, [0.33] * 9, min_score=0.45)
    assert rig is not None
    assert 49.0 <= rig[0] <= 58.0


def test_estimate_rig_still_needs_a_minimum_of_frames():
    from courtvision.court_tracking import estimate_rig
    assert estimate_rig([np.zeros(6)] * 2, [0.9] * 2) is None


def _hsv_image(hue: int, saturation: int, value: int, size: int = 60):
    cv2 = pytest.importorskip("cv2")
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2] = hue, saturation, value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_wood_fraction_sees_a_hardwood_floor():
    from courtvision.court_tracking import wood_fraction
    assert wood_fraction(_hsv_image(15, 120, 200)) > 0.95


def test_wood_fraction_rejects_a_dark_crowd():
    from courtvision.court_tracking import wood_fraction
    # A crowd shot measured 0.012; darkness alone must disqualify it.
    assert wood_fraction(_hsv_image(15, 120, 20)) < 0.05


def test_wood_fraction_rejects_blue_seating():
    from courtvision.court_tracking import wood_fraction
    assert wood_fraction(_hsv_image(110, 200, 200)) < 0.05


def test_has_court_gates_on_the_measured_threshold():
    from courtvision.court_tracking import has_court
    assert has_court(_hsv_image(15, 120, 200))
    assert not has_court(_hsv_image(110, 200, 200))


def test_propagate_stops_when_verification_fails():
    cv2 = pytest.importorskip("cv2")
    base = _textured(11)
    images = [base]
    for step in range(1, 5):
        images.append(cv2.warpPerspective(
            base, np.array([[1.0, 0, 4.0 * step], [0, 1.0, 0], [0, 0, 1.0]]),
            (base.shape[1], base.shape[0])))
    # Accept the first hop, then refuse: the chain must not continue past it.
    seen = []

    def verify(index, matrix):
        seen.append(index)
        return index <= 1

    out = propagate(images, {0: np.eye(3)}, verify=verify)
    assert set(out) == {0, 1}
    assert 2 in seen, "verification should have been offered the failing frame"


def test_players_on_court_accepts_an_identity_mapping():
    from courtvision.court_tracking import players_on_court
    feet = np.array([[10.0, 20.0], [25.0, 40.0], [45.0, 80.0]])
    assert players_on_court(np.eye(3), feet) == 1.0


def test_players_on_court_rejects_a_slid_court():
    from courtvision.court_tracking import players_on_court
    # A court shifted 200 ft sideways still explains lines but strands everyone.
    slid = np.array([[1.0, 0.0, 200.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    feet = np.array([[10.0, 20.0], [25.0, 40.0], [45.0, 80.0]])
    assert players_on_court(slid, feet) == 0.0


def test_plausible_positions_gates_on_the_share():
    from courtvision.court_tracking import plausible_positions
    feet = np.array([[10.0, 20.0], [25.0, 40.0], [45.0, 80.0], [300.0, 300.0]])
    assert plausible_positions(np.eye(3), feet)          # 3 of 4 on court
    assert not plausible_positions(np.eye(3), np.array([[300.0, 300.0]]))


def test_players_on_court_with_no_detections_is_a_failure():
    from courtvision.court_tracking import players_on_court
    assert players_on_court(np.eye(3), np.empty((0, 2))) == 0.0


def test_basket_offset_is_zero_for_a_perfect_fit():
    from courtvision.court_tracking import basket_offset_px
    from courtvision.court import BASKET
    # Identity: the basket's floor point projects to its own court coordinates.
    offset = basket_offset_px(np.eye(3), (BASKET[0], BASKET[1]))
    assert abs(offset) < 1e-6


def test_right_basket_rejects_a_fit_on_the_other_half():
    from courtvision.court_tracking import right_basket
    from courtvision.court import BASKET
    # The measured failure: the model lands ~900 px from the detected rim.
    shifted = np.array([[1.0, 0.0, 900.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert not right_basket(shifted, (BASKET[0], BASKET[1]))
    assert right_basket(np.eye(3), (BASKET[0], BASKET[1]))


def test_right_basket_is_false_without_a_rim():
    from courtvision.court_tracking import right_basket
    # Unverifiable is not the same as fine.
    assert not right_basket(np.eye(3), None)


def test_court_rotation_is_its_own_inverse():
    from courtvision.court_tracking import court_rotation
    rotation = court_rotation()
    assert np.allclose(rotation @ rotation, np.eye(3))


def test_court_rotation_maps_one_basket_onto_the_other():
    from courtvision.court_tracking import court_rotation
    from courtvision.court import BASKET, COURT_WIDTH
    from courtvision.court_lines import COURT_LENGTH
    assert COURT_LENGTH == 94.0   # y runs baseline to baseline; x is the 50 ft width
    near = np.array([BASKET[0], BASKET[1], 1.0])
    far = court_rotation() @ near
    assert np.allclose(far[:2], [COURT_WIDTH - BASKET[0],
                                 COURT_LENGTH - BASKET[1]])


# The basket sits on the court centreline (x = 25) and the rotation maps
# x -> 50 - x, so in COURT space its x is unchanged. Only a projection that
# sends court LENGTH to image WIDTH separates the two baskets horizontally --
# which is what a sideline broadcast camera does, and why the ~900 px split
# shows up on real footage. A test using an identity mapping is degenerate.
_SIDELINE = np.array([[0.0, 10.0, 0.0],     # image x from court y
                      [10.0, 0.0, 0.0],     # image y from court x
                      [0.0, 0.0, 1.0]])


def _rim_pixel_for(court_to_image, basket_xy):
    point = court_to_image @ np.array([basket_xy[0], basket_xy[1], 1.0])
    return (point[0] / point[2], point[1] / point[2])


def test_orient_to_rim_recovers_a_rotated_fit():
    from courtvision.court_tracking import court_rotation, orient_to_rim
    from courtvision.court import BASKET

    correct = np.linalg.inv(_SIDELINE)                 # image -> court
    rim = _rim_pixel_for(_SIDELINE, BASKET)
    rotated = court_rotation() @ correct               # the wrong-basket fit
    fixed = orient_to_rim(rotated, rim)
    assert fixed is not None
    assert np.allclose(fixed, correct)


def test_orient_to_rim_keeps_an_already_correct_fit():
    from courtvision.court_tracking import orient_to_rim
    from courtvision.court import BASKET

    correct = np.linalg.inv(_SIDELINE)
    rim = _rim_pixel_for(_SIDELINE, BASKET)
    fixed = orient_to_rim(correct, rim)
    assert fixed is not None and np.allclose(fixed, correct)


def test_orient_to_rim_refuses_when_neither_orientation_fits():
    from courtvision.court_tracking import orient_to_rim
    assert orient_to_rim(np.linalg.inv(_SIDELINE), (9000.0, 9000.0)) is None
    assert orient_to_rim(np.linalg.inv(_SIDELINE), None) is None
