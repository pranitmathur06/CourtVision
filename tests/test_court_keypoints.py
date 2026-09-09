"""The recovered court-landmark schema, and registration from it.

The schema was not read off documentation -- the dataset ships none -- it was
recovered from the annotations and then checked against the dataset's own
mirror symmetry. These tests pin both the recovery's result and the checks that
would have caught it being wrong.
"""

import numpy as np
import pytest

from courtvision.court_keypoints import (COURT_LENGTH_FT, COURT_WIDTH_FT,
                                         FLIP_INDEX, KEYPOINTS, MIN_KEYPOINTS,
                                         homography_from_keypoints,
                                         symmetry_error)


def test_every_landmark_is_on_the_court():
    for index, (x, y) in KEYPOINTS.items():
        assert -0.01 <= x <= COURT_WIDTH_FT + 0.01, index
        assert -0.01 <= y <= COURT_LENGTH_FT + 0.01, index


def test_the_schema_satisfies_the_datasets_own_mirror_symmetry():
    """`flip_idx` is annotation metadata the schema was never fitted to.

    The first attempt at these coordinates -- bootstrapped from the painted-key
    registration -- failed this at 65 ft, which is how it was caught.
    """
    assert symmetry_error(KEYPOINTS) < 0.01


def test_landmarks_match_known_nba_geometry():
    """Spot-checks against dimensions that are facts about the sport."""
    assert KEYPOINTS[8] == (25.0, 5.25)          # the basket
    assert KEYPOINTS[22] == (25.0, 47.0)         # centre court
    assert KEYPOINTS[13] == (25.0, 19.0)         # free-throw line centre
    assert KEYPOINTS[3][0] == 17.0 and KEYPOINTS[4][0] == 33.0   # lane, 16 wide
    assert KEYPOINTS[0] == (0.0, 0.0) and KEYPOINTS[42] == (50.0, 94.0)


def test_a_perfect_view_recovers_the_court_exactly():
    pytest.importorskip("cv2")
    import cv2
    # A synthetic camera looking at the court.
    court = np.array([[0, 0], [50, 0], [50, 94], [0, 94]], dtype=np.float32)
    image = np.array([[100, 600], [900, 600], [780, 120], [220, 120]],
                     dtype=np.float32)
    court_to_image = cv2.getPerspectiveTransform(court, image)

    seen = {}
    for index, (x, y) in KEYPOINTS.items():
        p = np.array([x, y, 1.0]) @ court_to_image.T
        seen[index] = (p[0] / p[2], p[1] / p[2])

    matrix, inliers = homography_from_keypoints(seen)
    assert matrix is not None and inliers >= 30
    for index, (x, y) in KEYPOINTS.items():
        px, py = seen[index]
        q = np.array([px, py, 1.0]) @ matrix.T
        got = q[:2] / q[2]
        assert np.hypot(got[0] - x, got[1] - y) < 0.1, index


def test_occluding_the_paint_still_registers():
    """The failure that defeats the painted key: players standing in the lane.

    The key method needs one unoccluded quad, so this costs it the frame. Here
    it costs a handful of landmarks out of the ones spread across the floor.
    """
    pytest.importorskip("cv2")
    import cv2
    court = np.array([[0, 0], [50, 0], [50, 94], [0, 94]], dtype=np.float32)
    image = np.array([[100, 600], [900, 600], [780, 120], [220, 120]],
                     dtype=np.float32)
    court_to_image = cv2.getPerspectiveTransform(court, image)
    hidden = {3, 4, 12, 13, 14, 8, 17}          # the whole near lane and arc
    seen = {}
    for index, (x, y) in KEYPOINTS.items():
        if index in hidden:
            continue
        p = np.array([x, y, 1.0]) @ court_to_image.T
        seen[index] = (p[0] / p[2], p[1] / p[2])
    matrix, inliers = homography_from_keypoints(seen)
    assert matrix is not None and inliers >= 20


def test_too_few_landmarks_is_refused_not_guessed():
    pytest.importorskip("cv2")
    few = {i: (100.0 + 10 * n, 200.0 + 5 * n)
           for n, i in enumerate(list(KEYPOINTS)[:MIN_KEYPOINTS - 1])}
    matrix, inliers = homography_from_keypoints(few)
    assert matrix is None


def test_unknown_indices_are_ignored_rather_than_guessed():
    pytest.importorskip("cv2")
    import cv2
    court = np.array([[0, 0], [50, 0], [50, 94], [0, 94]], dtype=np.float32)
    image = np.array([[100, 600], [900, 600], [780, 120], [220, 120]],
                     dtype=np.float32)
    court_to_image = cv2.getPerspectiveTransform(court, image)
    seen = {}
    for index, (x, y) in KEYPOINTS.items():
        p = np.array([x, y, 1.0]) @ court_to_image.T
        seen[index] = (p[0] / p[2], p[1] / p[2])
    seen[2] = (5.0, 5.0)        # never annotated in 850 frames
    seen[47] = (9.0, 9.0)
    matrix, inliers = homography_from_keypoints(seen)
    assert matrix is not None
    assert inliers <= len(KEYPOINTS), "the unknown indices must not be used"


def test_flip_index_is_an_involution():
    """A mirror applied twice is the identity; this is what makes it a check."""
    for i, j in enumerate(FLIP_INDEX):
        assert FLIP_INDEX[j] == i
