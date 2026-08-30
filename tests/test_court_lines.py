"""Court-line detection and homography scoring, on synthetic courts."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from courtvision.court_lines import (
    alignment_score,
    canonical_court_points,
    court_line_mask,
)


def render_court(camera: np.ndarray, size=(720, 1280)) -> np.ndarray:
    """Draw the canonical court through a known camera, on wood-coloured floor."""
    image = np.zeros((size[0], size[1], 3), np.uint8)
    image[:] = (150, 190, 225)                     # BGR wood
    noise = np.random.default_rng(0).normal(0, 6, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    pts = canonical_court_points(step_ft=0.25)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    projected = homogeneous @ camera.T
    xy = projected[:, :2] / projected[:, 2:3]
    for x, y in xy:
        if 0 <= x < size[1] and 0 <= y < size[0]:
            cv2.circle(image, (int(x), int(y)), 2, (90, 70, 60), -1)   # dark line
    return image


def camera_matrix() -> np.ndarray:
    """court -> image."""
    return np.array([
        [22.0, 3.0, 120.0],
        [0.0, -13.0, 660.0],
        [0.0, -0.004, 1.0],
    ])


def test_mask_finds_the_painted_lines():
    image = render_court(camera_matrix())
    mask = court_line_mask(image)
    assert mask.any(), "no line pixels found on a synthetic court"
    # Lines are thin: they should be a small fraction of the frame.
    assert 0.001 < (mask > 0).mean() < 0.15


def test_correct_homography_scores_far_higher_than_a_wrong_one():
    camera = camera_matrix()
    image = render_court(camera)
    correct = np.linalg.inv(camera)                 # image -> court

    wrong = np.linalg.inv(camera @ np.array([
        [1.0, 0.0, 9.0], [0.0, 1.0, 11.0], [0.0, 0.0, 1.0],
    ]))

    good = alignment_score(correct, image)
    bad = alignment_score(wrong, image)
    assert good > 0.5, f"correct homography scored only {good:.2f}"
    assert good > bad + 0.2, f"correct {good:.2f} vs wrong {bad:.2f}"


def test_a_frame_with_no_court_scores_zero():
    blank = np.full((720, 1280, 3), 40, np.uint8)
    assert alignment_score(np.eye(3), blank) == 0.0


def test_projection_off_screen_is_not_counted_as_a_hit():
    """Most of a half court is off-camera; that must not read as success."""
    image = render_court(camera_matrix())
    # A homography that throws the whole court out of frame.
    far_away = np.linalg.inv(np.array([
        [22.0, 3.0, 90000.0], [0.0, -13.0, 90000.0], [0.0, -0.004, 1.0],
    ]))
    assert alignment_score(far_away, image) == 0.0


def test_canonical_points_cover_the_court_not_just_one_line():
    pts = canonical_court_points()
    assert len(pts) > 300
    assert pts[:, 0].min() < 1.0 and pts[:, 0].max() > 49.0
    assert pts[:, 1].min() < 1.0 and pts[:, 1].max() > 25.0


def test_two_sided_score_rejects_a_degenerate_zoomed_in_fit():
    """A homography showing a sliver of court must not score like a good one.

    A one-directional score (do the model's lines land on detected lines?)
    returned 0.998 for a camera whose court coordinates were hundreds of feet
    wrong: it had zoomed onto a patch where a couple of arcs coincided. Coverage
    — the share of detected line pixels the model explains — is what separates
    them.
    """
    from courtvision.court_lines import (
        canonical_court_points,
        homography_from_camera,
        line_distance_map,
        sample_line_pixels,
        score_homography,
    )

    camera = camera_matrix()
    image = render_court(camera)
    distance_map = line_distance_map(image)
    points = canonical_court_points()
    line_pixels = sample_line_pixels(image)

    honest = score_homography(camera, distance_map, points, line_pixels)
    # A camera zoomed far into one corner: whatever it shows will sit on lines,
    # but it explains almost none of what the image contains.
    zoomed = homography_from_camera(
        np.array([25.0, 14.0, 13.0, 25.0, 18.0, 12000.0]), image.shape[:2])
    degenerate = score_homography(zoomed, distance_map, points, line_pixels)

    assert honest > 0.5
    assert degenerate < honest / 2


def test_search_recovers_a_known_camera():
    """End to end: synthetic court, unknown camera, recovered by search."""
    from courtvision.court_lines import homography_from_camera, search_registration

    truth = np.array([25.0, -35.0, 40.0, 25.0, 20.0, 1400.0])
    camera = homography_from_camera(truth, (720, 1280))
    image = render_court(camera)

    # Bounds around the true placement. The unconstrained global sweep needs
    # ~250 iterations and ~40 s; narrowing it is the realistic way to use this
    # when the camera's rough position is known, and keeps the test quick.
    bounds = [(15.0, 35.0), (-45.0, -25.0), (30.0, 50.0),
              (20.0, 30.0), (15.0, 25.0), (1100.0, 1700.0)]
    found, score = search_registration(image, seed=0, max_iterations=60,
                                       bounds=bounds)

    assert found is not None and score > 0.5, f"search failed at {score:.3f}"
    truth_image_to_court = np.linalg.inv(camera)
    grid = np.array([[300.0, 400.0], [640.0, 300.0], [900.0, 500.0]])
    homogeneous = np.hstack([grid, np.ones((3, 1))])
    got = homogeneous @ found.T
    want = homogeneous @ truth_image_to_court.T
    got = got[:, :2] / got[:, 2:3]
    want = want[:, :2] / want[:, 2:3]
    assert np.abs(got - want).max() < 3.0, f"recovered court coords off by "\
        f"{np.abs(got - want).max():.1f} ft"
