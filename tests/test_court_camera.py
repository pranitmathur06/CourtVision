"""The fixed-camera model, on synthetic cameras whose centre is known."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from courtvision.court_camera import (FixedCamera, decompose,  # noqa: E402
                                      estimate_centre, ptz_matrix, ptz_params)
from courtvision.court_refine import refine  # noqa: E402
from tests.test_court_refine import _court_error, _perturb  # noqa: E402

SIZE = (1000, 700)
CENTRE = np.array([130.0, 47.0, 35.0])        # beside the court, 35 ft up


def _look_at(target, roll_deg=0.0):
    forward = np.asarray(target, float) - CENTRE
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, -1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    roll = np.deg2rad(roll_deg)
    spin = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    return spin @ rotation


def _camera_frame(target, f=1400.0, roll_deg=0.5):
    """image -> court for a camera at CENTRE looking at `target`."""
    rvec = cv2.Rodrigues(_look_at(target, roll_deg))[0].ravel()
    return np.linalg.inv(ptz_matrix(np.r_[rvec, np.log(f)], CENTRE, SIZE))


def _grid():
    return np.array([[x, y] for x in np.linspace(5, 45, 5) for y in np.linspace(10, 84, 6)])


def test_decomposition_recovers_the_centre_of_an_exact_camera():
    matrix = _camera_frame([25, 20, 0])
    f, _, centre = decompose(np.linalg.inv(matrix), SIZE)
    assert abs(f - 1400) < 1
    assert np.allclose(centre, CENTRE, atol=0.05)


def test_the_centre_is_recovered_jointly_and_other_cameras_are_dropped():
    rng = np.random.default_rng(0)
    fits = []
    for k in range(12):
        target = [rng.uniform(15, 35), rng.uniform(10, 84), 0]
        fits.append((_camera_frame(target, f=rng.uniform(1100, 2000)), _grid()))
    # Two frames from a baseline camera -- a different centre entirely.
    other = np.array([25.0, -30.0, 20.0])
    for target in ([25, 30, 0], [20, 20, 0]):
        forward = np.asarray(target, float) - other
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, -1.0]); right /= np.linalg.norm(right)
        rotation = np.vstack([right, np.cross(forward, right), forward])
        rvec = cv2.Rodrigues(rotation)[0].ravel()
        fits.append((np.linalg.inv(ptz_matrix(np.r_[rvec, np.log(1300)], other, SIZE)), _grid()))
    camera, report = estimate_centre(fits, SIZE)
    assert camera is not None, report
    assert np.linalg.norm(camera.centre - CENTRE) < 0.5, report
    assert report["inliers"] == 12, report


def test_a_squeezed_court_is_not_a_pan_tilt_zoom_of_the_camera():
    """The failure seen at Toyota Center: both sidelines pulled inward."""
    truth = _camera_frame([25, 47, 0])
    camera = FixedCamera(CENTRE, SIZE)
    ok, _ = camera.explains(truth, _grid())
    assert ok
    squeeze = np.array([[1.25, 0, -6.25], [0, 1, 0], [0, 0, 1.0]])   # 50 ft court read as 40
    ok, cost = camera.explains(squeeze @ truth, _grid())
    assert not ok, cost


def test_refinement_with_the_camera_recovers_a_landmark_sized_error():
    from tests.test_court_refine import _render
    truth = _camera_frame([25, 20, 0], f=1100.0)
    image = _render(truth, clutter=False)
    start = _perturb(truth, 1.5, -1.0, 1.0)
    refined, info = refine(image, start, camera=FixedCamera(CENTRE, SIZE))
    assert info["refined"], info
    assert _court_error(refined, truth) < 0.1


def test_ptz_params_reproduce_an_exact_camera():
    truth = _camera_frame([30, 60, 0])
    params, cost = ptz_params(np.linalg.inv(truth), CENTRE, SIZE, _grid())
    assert cost < 0.01
