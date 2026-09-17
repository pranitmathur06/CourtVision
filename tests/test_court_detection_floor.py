"""The court DETECTION floor, which is not the per-keypoint floor.

A pose model emits keypoints only for a detected instance, so a court scored
below the detection threshold yields no landmarks at all rather than a few. That
is why coverage was strictly bimodal, and why dropping the per-keypoint floor --
which was tried, 0.6 to 0.3 -- moved almost nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.court_keypoints import (COURT_DETECTION_CONF, KEYPOINTS,
                                         MIN_KEYPOINTS, landmark_points)

torch = pytest.importorskip("torch")


class FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = torch.tensor(np.asarray(xy, dtype=np.float32)).unsqueeze(0)
        self.conf = (None if conf is None else
                     torch.tensor(np.asarray(conf, dtype=np.float32)).unsqueeze(0))

    def __len__(self):
        return 1


class FakeResult:
    def __init__(self, xy=None, conf=None):
        self.keypoints = None if xy is None else FakeKeypoints(xy, conf)


def scene(found=MIN_KEYPOINTS, score=0.9):
    """A result placing `found` of the annotated landmarks at `score`."""
    indices = sorted(KEYPOINTS)[:found]
    xy = np.zeros((48, 2), dtype=np.float32)
    conf = np.zeros(48, dtype=np.float32)
    for n, i in enumerate(indices):
        xy[i] = (100.0 + n, 200.0 + n)
        conf[i] = score
    return FakeResult(xy, conf)


def test_the_detection_floor_is_far_below_the_library_default():
    """0.25 threw away ten to thirteen points of coverage on every broadcast.

    Measured two independent ways, neither of them annotated: coverage by
    check_registration_coverage.py, and accuracy by two registrations of one
    instant disagreeing in check_registration_consistency.py.
    """
    assert COURT_DETECTION_CONF < 0.01
    assert COURT_DETECTION_CONF > 0.0


def test_a_frame_with_no_court_instance_cannot_register():
    assert landmark_points(FakeResult(), conf=0.5) is None
    assert landmark_points(None, conf=0.5) is None


def test_too_few_confident_landmarks_is_the_same_answer_as_none():
    """A homography needs points; five of them is not 'almost'."""
    assert landmark_points(scene(found=MIN_KEYPOINTS - 1), conf=0.5) is None
    assert landmark_points(scene(found=MIN_KEYPOINTS), conf=0.5) is not None


def test_landmarks_below_the_keypoint_floor_are_dropped():
    assert landmark_points(scene(score=0.4), conf=0.5) is None
    assert landmark_points(scene(score=0.6), conf=0.5) is not None


def test_only_the_annotated_indices_count():
    """16 of the 48 are never marked in 850 frames and mean nothing."""
    xy = np.full((48, 2), 50.0, dtype=np.float32)
    conf = np.ones(48, dtype=np.float32)
    seen = landmark_points(FakeResult(xy, conf), conf=0.5)
    assert seen is not None
    assert set(seen) <= set(KEYPOINTS)
    assert len(seen) == len(KEYPOINTS)


def test_a_landmark_at_the_origin_is_a_placeholder_not_a_corner():
    xy = np.zeros((48, 2), dtype=np.float32)
    conf = np.ones(48, dtype=np.float32)
    assert landmark_points(FakeResult(xy, conf), conf=0.5) is None
