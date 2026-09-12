"""The tracker's helpers, on poses built so the right answer is known."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import track_camera as tracker  # noqa: E402
from courtvision.court_camera import (RIMS_3D, FixedCamera, look_at,  # noqa: E402
                                      ptz_matrix)

CENTRE = np.array([114.0, 42.9, 26.4])
SIZE = (1280, 720)


def _camera(k1=0.0, k2=0.0):
    return FixedCamera(CENTRE, SIZE, k1=k1, k2=k2)


def _image_to_court(target, focal=1400.0):
    params = look_at(target, focal, CENTRE)
    return np.linalg.inv(ptz_matrix(params, CENTRE, SIZE))


def test_snap_leaves_a_true_pose_where_it_is():
    """A homography that already IS a pose of this camera must survive intact."""
    pose = _image_to_court((25.0, 14.0))
    snapped, cost, params = tracker.snap(_camera(), pose, SIZE)
    assert cost < 0.5
    grid = np.array([[10.0, 10.0, 1.0], [40.0, 60.0, 1.0], [25.0, 47.0, 1.0]])
    before = grid @ np.linalg.inv(pose).T
    after = grid @ np.linalg.inv(snapped).T
    assert np.allclose(before[:, :2] / before[:, 2:3],
                       after[:, :2] / after[:, 2:3], atol=1.0)
    assert len(params) == 4


def test_snap_reports_a_cost_when_the_homography_is_not_a_pose_of_this_camera():
    """A slipped hop has to be visible as a number, not silently accepted."""
    pose = _image_to_court((25.0, 14.0))
    skewed = pose @ np.array([[1.0, 0.06, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    _, cost, _ = tracker.snap(_camera(), skewed, SIZE)
    assert cost > 1.0


def test_a_rim_behind_the_camera_is_not_reported():
    """Looking at one basket, the other must not be claimed as visible."""
    shown = tracker.project_rims(_camera(), _image_to_court((25.0, 5.25), 2400.0), SIZE)
    assert len(shown) <= 1


def test_a_projected_rim_lands_where_the_basket_is():
    camera = _camera()
    pose = _image_to_court((25.0, 14.0))
    shown = tracker.project_rims(camera, pose, SIZE)
    assert shown, "the near basket should be in this view"
    import cv2
    params = look_at((25.0, 14.0), 1400.0, CENTRE)
    rotation = cv2.Rodrigues(np.asarray(params[:3]))[0]
    focal = float(np.exp(params[3]))
    intrinsics = np.array([[focal, 0, SIZE[0] / 2], [0, focal, SIZE[1] / 2], [0, 0, 1.0]])
    projected = intrinsics @ rotation @ (np.array(RIMS_3D[0]) - CENTRE)
    truth = projected[:2] / projected[2]
    assert min(np.hypot(*(np.array(p) - truth)) for p in shown) < 2.0


def test_the_detector_contradicts_a_pose_that_puts_the_rim_elsewhere():
    camera = _camera()
    pose = _image_to_court((25.0, 14.0))
    shown = tracker.project_rims(camera, pose, SIZE)[0]
    def rim_box(x, y, width=40.0, conf=0.9):
        return [{"cls": "rim", "conf": conf,
                 "xyxy": [x - width / 2, y - 10, x + width / 2, y + 10]}]
    assert not tracker.disagrees_with_detector(camera, pose, SIZE, rim_box(*shown))
    assert tracker.disagrees_with_detector(camera, pose, SIZE,
                                           rim_box(shown[0] + 300, shown[1]))


def test_a_faint_detection_does_not_get_to_veto_a_pose():
    camera = _camera()
    pose = _image_to_court((25.0, 14.0))
    faint = [{"cls": "rim", "conf": 0.1, "xyxy": [900.0, 500.0, 940.0, 520.0]}]
    assert not tracker.disagrees_with_detector(camera, pose, SIZE, faint)


def test_a_hop_at_half_scale_comes_back_in_full_size_pixels():
    """The homography must describe the frame, not the shrunken copy of it."""
    import cv2
    rng = np.random.default_rng(0)
    grey = rng.integers(0, 255, (720, 1280), dtype=np.uint8)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    shift = np.array([[1.0, 0.0, 24.0], [0.0, 1.0, -12.0], [0.0, 0.0, 1.0]])
    moved = cv2.warpPerspective(grey, shift, (1280, 720))
    found = tracker.hop_at_scale(grey, moved, None, None, 0.5)
    assert found is not None
    assert np.allclose(found / found[2, 2], shift, atol=1.5)
