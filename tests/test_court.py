"""Court registration: exact synthetic checks, then the properties that matter."""

import numpy as np
import pytest

from courtvision.court import (
    BASKET,
    COURT_WIDTH,
    HALF_COURT_LENGTH,
    LANDMARKS,
    distance_to_basket_ft,
    is_three_point_attempt,
    register,
)


def synthetic_camera() -> np.ndarray:
    """A plausible court->image homography to generate exact test data."""
    # Perspective: far baseline compressed, near sideline stretched.
    return np.array([
        [12.0, 2.0, 300.0],
        [0.0, -9.0, 640.0],
        [0.0, -0.006, 1.0],
    ])


def project(matrix: np.ndarray, court_xy) -> tuple[float, float]:
    v = matrix @ np.array([court_xy[0], court_xy[1], 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def test_register_recovers_exact_positions_from_a_known_camera():
    camera = synthetic_camera()
    names = ["baseline_left", "baseline_right", "half_court_left",
             "half_court_right", "free_throw_centre", "lane_left_ft"]
    correspondences = {n: project(camera, LANDMARKS[n]) for n in names}

    reg = register(correspondences)

    assert reg.max_error_ft < 0.01
    assert reg.is_usable()
    # A landmark that was NOT used must still land in the right place.
    held_out = project(camera, LANDMARKS["basket"])
    got = reg.to_court([held_out])[0]
    assert np.allclose(got, BASKET, atol=0.05)


def test_four_points_report_zero_error_however_wrong_they_are():
    """With exactly 4 correspondences the reported error is arithmetic, not evidence.

    A homography has 8 degrees of freedom and 4 point pairs give 8 equations, so
    the fit passes through every input exactly. Corrupting one correspondence by
    300 px still reports 0.000 ft while the true error on a held-out landmark
    grows to well over 2 ft. This is why register() asks for more than four.
    """
    camera = synthetic_camera()
    names = ["baseline_left", "baseline_right", "half_court_left", "half_court_right"]
    held_out_image = project(camera, LANDMARKS["free_throw_centre"])
    truth = np.array(LANDMARKS["free_throw_centre"])

    true_errors = []
    for shift in (40, 120, 300):
        correspondences = {n: project(camera, LANDMARKS[n]) for n in names}
        x, y = correspondences["baseline_right"]
        correspondences["baseline_right"] = (x + shift, y + shift * 0.6)

        reg = register(correspondences)

        assert reg.max_error_ft < 1e-4, "4 points always fit perfectly"
        assert reg.is_usable(), "and so always look trustworthy"
        true_errors.append(
            float(np.linalg.norm(reg.to_court([held_out_image])[0] - truth))
        )

    # The reported error stayed at zero throughout while the real one grew.
    assert true_errors[0] < true_errors[1] < true_errors[2]
    assert true_errors[-1] > 2.0


def test_bad_correspondence_is_reported_not_hidden():
    camera = synthetic_camera()
    names = ["baseline_left", "baseline_right", "half_court_left",
             "half_court_right", "free_throw_centre", "lane_left_ft"]
    correspondences = {n: project(camera, LANDMARKS[n]) for n in names}
    correspondences["free_throw_centre"] = (
        correspondences["free_throw_centre"][0] + 60,
        correspondences["free_throw_centre"][1] - 45,
    )

    reg = register(correspondences)

    assert reg.max_error_ft > 3.0
    assert not reg.is_usable()


def test_too_few_or_unknown_landmarks_are_rejected():
    with pytest.raises(ValueError, match="at least 4"):
        register({"baseline_left": (0, 0), "baseline_right": (1, 1),
                  "basket": (2, 2)})
    with pytest.raises(ValueError, match="unknown landmarks"):
        register({"baseline_left": (0, 0), "baseline_right": (1, 1),
                  "basket": (2, 2), "the_moon": (3, 3)})


def test_points_above_the_horizon_are_nan_not_a_plausible_position():
    """The horizon maps to infinity; say so rather than return a big number.

    A huge finite value would read downstream as a real position far up court.
    """
    camera = synthetic_camera()
    names = ["baseline_left", "baseline_right", "half_court_left",
             "half_court_right", "free_throw_centre", "lane_left_ft"]
    reg = register({n: project(camera, LANDMARKS[n]) for n in names})

    # Solve w = 0 from the fitted matrix rather than assuming the camera's.
    a, b, c = reg.matrix[2]
    x = 500.0
    horizon_y = -(a * x + c) / b
    assert np.isnan(reg.to_court([[x, horizon_y]])[0]).all()


def test_on_court_rejects_the_crowd():
    camera = synthetic_camera()
    names = ["baseline_left", "baseline_right", "half_court_left",
             "half_court_right", "free_throw_centre", "lane_left_ft"]
    reg = register({n: project(camera, LANDMARKS[n]) for n in names})
    on = project(camera, (25.0, 20.0))
    off = project(camera, (25.0, -40.0))       # deep behind the baseline
    mask = reg.on_court([on, off])
    assert bool(mask[0]) and not bool(mask[1])


@pytest.mark.parametrize(
    "point, expected",
    [
        ((25.0, 5.25), False),      # at the rim
        ((25.0, 19.0), False),      # free-throw line
        ((25.0, 32.0), True),       # top of the arc
        ((1.5, 8.0), True),         # corner three
        ((1.5, 20.0), False),       # same x, above the corner section
        ((5.0, 10.0), False),       # inside the arc, off the corner
    ],
)
def test_three_point_geometry_handles_the_corner(point, expected):
    assert is_three_point_attempt(point) is expected


def test_distance_to_basket():
    assert distance_to_basket_ft([BASKET])[0] == pytest.approx(0.0)
    got = distance_to_basket_ft([[25.0, 5.25 + 23.75]])[0]
    assert got == pytest.approx(23.75, abs=1e-6)


def test_court_constants_match_the_rulebook():
    assert COURT_WIDTH == 50.0
    assert HALF_COURT_LENGTH == 47.0
    assert LANDMARKS["lane_right_baseline"][0] - LANDMARKS["lane_left_baseline"][0] == 16.0
