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
    right = np.cross(forward, [0, 0, 1.0])
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
        right = np.cross(forward, [0, 0, 1.0]); right /= np.linalg.norm(right)
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


def _other_camera_frame(target):
    other = np.array([25.0, -30.0, 20.0])
    forward = np.asarray(target, float) - other
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0]); right /= np.linalg.norm(right)
    rotation = np.vstack([right, np.cross(forward, right), forward])
    rvec = cv2.Rodrigues(rotation)[0].ravel()
    return np.linalg.inv(ptz_matrix(np.r_[rvec, np.log(1300)], other, SIZE))


def test_too_few_agreeing_frames_give_no_centre_rather_than_a_bad_one():
    """Six frames, two from another camera: dropping them leaves four, under
    MIN_FRAMES. The old loop kept all six and called them six inliers."""
    rng = np.random.default_rng(1)
    fits = [(_camera_frame([rng.uniform(15, 35), rng.uniform(10, 84), 0]), _grid()) for _ in range(4)]
    fits += [(_other_camera_frame(t), _grid()) for t in ([25, 30, 0], [20, 20, 0])]
    camera, report = estimate_centre(fits, SIZE)
    assert camera is None, report
    assert report.get("inliers", 0) < 6, report


def test_reported_inliers_are_the_frames_actually_kept():
    rng = np.random.default_rng(2)
    fits = [(_camera_frame([rng.uniform(15, 35), rng.uniform(10, 84), 0]), _grid()) for _ in range(8)]
    fits += [(_other_camera_frame(t), _grid()) for t in ([25, 30, 0], [20, 20, 0])]
    camera, report = estimate_centre(fits, SIZE)
    assert camera is not None, report
    assert report["inliers"] == 8 and report["worst_px"] <= 3.0, report


def test_a_frame_with_no_landmark_start_is_found_from_paint_alone():
    """The landmark model gave no start on ~30% of a whole game's views. With
    the camera fixed, the four remaining numbers are searched from paint."""
    from courtvision.court_register import register_frame
    from tests.test_court_refine import _render
    truth = _camera_frame([25, 20, 0], f=1100.0)
    image = _render(truth, clutter=False)
    # A broadcast shows stands around the floor; the search reads the floor's
    # silhouette, so the fixture must have one (an all-wood frame has none).
    court = np.array([[-3, -3], [53, -3], [53, 97], [-3, 97]], np.float64)
    h = np.c_[court, np.ones(4)] @ np.linalg.inv(truth).T
    outline = np.round(h[:, :2] / h[:, 2:3]).astype(np.int32)
    floor = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(floor, [outline], 1)
    image[floor == 0] = (40, 30, 35)
    matrix, info = register_frame(image, None, camera=FixedCamera(CENTRE, SIZE),
                                  polarities=("bright",))
    assert info["refined"], info
    assert _court_error(matrix, truth) < 0.1


def test_a_frame_from_another_camera_is_refused():
    """A view from a different centre is not a pan/tilt/zoom of this game's
    camera; forcing it into the model must not be accepted as a registration."""
    from courtvision.court_register import register_frame
    from tests.test_court_refine import _render
    other = _other_camera_frame([25, 20, 0])
    image = _render(other, clutter=False)
    start = _perturb(other, 0.5, -0.5, 0.3)
    matrix, info = register_frame(image, start, camera=FixedCamera(CENTRE, SIZE),
                                  polarities=("bright",), search=False, verify=True)
    assert not info["refined"] or _court_error(matrix, other) < 0.3, info


def test_another_floor_is_refused_by_its_key_colour():
    """Halftime highlights from other arenas fit this camera's geometry nearly
    as well as its own frames; their paint does not."""
    from courtvision.court_camera import floor_signature, same_floor
    from tests.test_court_refine import _render
    truth = _camera_frame([25, 15, 0], f=1100.0)

    def painted(key_bgr):
        image = _render(truth, clutter=False)
        lane = np.array([[17, 0], [33, 0], [33, 19], [17, 19]], np.float64)
        h = np.c_[lane, np.ones(4)] @ np.linalg.inv(truth).T
        cv2.fillPoly(image, [np.round(h[:, :2] / h[:, 2:3]).astype(np.int32)], key_bgr)
        return image

    home = painted((40, 40, 200))                     # a red key
    camera = FixedCamera(CENTRE, SIZE, floor=floor_signature(home, truth))
    assert same_floor(camera, painted((45, 38, 205)), truth)[0]
    ok, distance = same_floor(camera, painted((90, 40, 20)), truth)   # a navy key
    assert not ok and distance > 60, distance


def test_no_signature_or_no_visible_key_never_refuses():
    from courtvision.court_camera import same_floor
    truth = _camera_frame([25, 15, 0], f=1100.0)
    blank = np.zeros((700, 1000, 3), np.uint8)
    assert same_floor(FixedCamera(CENTRE, SIZE), blank, truth) == (True, None)
