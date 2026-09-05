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
