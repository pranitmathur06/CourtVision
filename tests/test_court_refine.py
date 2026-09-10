"""Dense line refinement, on synthetic courts where the answer is known exactly."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from courtvision.court_lines import court_lines  # noqa: E402
from courtvision.court_refine import (HOLD_OUT, held_out_offsets,  # noqa: E402
                                      refine)

SIZE = (1000, 700)


def _truth():
    """image -> court for a synthetic broadcast-like camera."""
    court = np.array([[0, 0], [50, 0], [50, 94], [0, 94]], dtype=np.float32)
    image = np.array([[100, 640], [900, 640], [760, 110], [240, 110]],
                     dtype=np.float32)
    return cv2.getPerspectiveTransform(image, court)


def _render(image_to_court, clutter=True, seed=0):
    """A wood floor with painted lines, drawn at sub-pixel precision."""
    rng = np.random.default_rng(seed)
    width, height = SIZE
    image = np.full((height, width, 3), (150, 190, 215), np.uint8)
    court_to_image = np.linalg.inv(image_to_court)
    for polyline in court_lines():
        p = np.c_[np.asarray(polyline), np.ones(len(polyline))] @ court_to_image.T
        p = p[:, :2] / p[:, 2:3]
        cv2.polylines(image, [np.int32(np.round(p * 16))], False,
                      (245, 245, 245), 3, cv2.LINE_AA, shift=4)
    if clutter:
        for _ in range(10):                      # players
            x, y = rng.integers(150, 850), rng.integers(200, 600)
            cv2.rectangle(image, (x, y - 90), (x + 30, y), (40, 40, 60), -1)
        cv2.line(image, (300, 400), (380, 300), (250, 250, 250), 3)  # logo stroke
    noise = rng.normal(0, 5, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def _perturb(matrix, dx, dy, degrees):
    """Move a registration in court space: shift, then rotate about centre."""
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    rotate = np.array([[c, -s, 25 - 25 * c + 47 * s],
                       [s, c, 47 - 25 * s - 47 * c], [0, 0, 1]])
    shift = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    return shift @ rotate @ matrix


def _court_error(estimate, truth):
    """Median disagreement, in feet, over the visible floor."""
    grid = np.array([[x, y] for x in np.linspace(200, 800, 7)
                     for y in np.linspace(200, 620, 6)], dtype=np.float32)
    a = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), estimate).reshape(-1, 2)
    b = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), truth).reshape(-1, 2)
    return float(np.median(np.hypot(*(a - b).T)))


def test_refinement_recovers_a_landmark_sized_error():
    """Start where the landmark model starts -- ~2.5 ft off -- and converge."""
    truth = _truth()
    image = _render(truth)
    start = _perturb(truth, 2.0, -1.5, 1.5)
    before = _court_error(start, truth)
    refined, info = refine(image, start)
    after = _court_error(refined, truth)
    assert info["refined"], info
    assert before > 1.5
    assert after < 0.1, f"{before:.2f} ft -> {after:.2f} ft ({info})"


def test_a_known_error_is_reported_as_that_error():
    """The control two earlier estimators here failed.

    Shift the registration 0.5 ft across the court and the corner-three lines,
    whose normals run across the court, must read 0.5 ft -- not a damped
    fraction of it, which is what an estimator with a too-narrow window or a
    nearest-ridge search reports.
    """
    truth = _truth()
    image = _render(truth, clutter=False)
    lines = HOLD_OUT["near corners"]
    exact = held_out_offsets(image, truth, lines)
    moved = held_out_offsets(image, _perturb(truth, 0.5, 0.0, 0.0), lines)
    assert len(exact) > 10 and len(moved) > 10
    assert abs(np.median(exact)) < 0.05
    assert abs(abs(np.median(moved)) - 0.5) < 0.05, np.median(moved)


def test_held_out_lines_measure_accuracy_the_fit_never_saw():
    truth = _truth()
    image = _render(truth)
    start = _perturb(truth, 1.5, 1.0, 1.0)
    for name in ("near lane", "near arc", "half-court"):
        lines = HOLD_OUT[name]
        refined, info = refine(image, start, exclude_lines=lines)
        assert info["refined"], (name, info)
        offsets = held_out_offsets(image, refined, lines)
        assert len(offsets) > 10, name
        assert np.median(np.abs(offsets)) < 0.1, (name, np.median(np.abs(offsets)))


def test_a_diagonal_gross_start_is_refused_or_corrected():
    """The ICP failure: locking onto the wrong paint and reporting success.
    Refusing is acceptable; being wrong and marked refined is not."""
    truth = _truth()
    image = _render(truth)
    start = _perturb(truth, 12.0, 12.0, 0.0)
    refined, info = refine(image, start)
    if info["refined"]:
        assert _court_error(refined, truth) < 1.0, info
    else:
        assert np.allclose(refined, start), "a refusal must return the start"


@pytest.mark.xfail(strict=True, reason=(
    "Known blind spot, a deliberate trade. Segment consensus -- each straight "
    "segment choosing its paint as a unit -- was adopted because on the OKC "
    "calibration game it accepts 74% of registered frames against 60% for a "
    "single start, loses 9% of held-out lines against 14%, and on the 73 "
    "measurements both scored has identical accuracy (median change +0.000 ft). "
    "The cost: from a start translated 25 ft across the court it locks onto a "
    "sharp but partial alignment (23% of paint explained, against 87% for the "
    "truth on the same frame) and reports success, where the earlier design "
    "refused. A pure 25 ft translation is not what the landmark model produces: "
    "1 of 74 held-out measurements had a landmark start more than 3 ft off. "
    "strict=True: if this starts passing, the trade and its docs are stale."))
def test_a_start_translated_25_ft_across_the_court_is_caught():
    truth = _truth()
    image = _render(truth)
    start = _perturb(truth, 25.0, 0.0, 0.0)
    refined, info = refine(image, start)
    assert not info["refined"] or _court_error(refined, truth) < 1.0


@pytest.mark.xfail(strict=True, reason=(
    "Known blind spot, kept visible. From a start translated 30 ft along the "
    "court the fit keeps the across-court mapping exact and matches some "
    "along-court lines under a projective compression: sharp (every 2 ft "
    "neighbour finds 0% paint) but partial (29% of along-court samples, "
    "against 92% for the truth). Relative sharpness cannot see 'partial', and "
    "absolute coverage cannot be thresholded because occlusion puts true "
    "broadcast fits at 16-35%. A pure translation this large is not a failure "
    "the landmark model produces -- a consistent landmark mix-up gives a "
    "reflection, which orientation refuses, or a 180 degree rotation, which no "
    "geometry can see. strict=True: if this starts passing, the docs are stale."))
def test_a_start_translated_30_ft_along_the_court_is_caught():
    truth = _truth()
    image = _render(truth)
    start = _perturb(truth, 0.0, 30.0, 0.0)
    refined, info = refine(image, start)
    assert not info["refined"] or _court_error(refined, truth) < 1.0


def test_near_miss_starts_converge_or_refuse_never_lock_elsewhere():
    """The realistic case: the landmark fit is off by a few feet, not thirty.

    p90 landmark error on held-out games is about 3 ft; these starts sit at
    and beyond that. Every one must either reach the truth or come back
    refused -- and most should reach it, or the capture range is too small to
    be useful on the frames where the landmark model is weakest.
    """
    truth = _truth()
    image = _render(truth)
    converged = 0
    starts = ((6.0, 0.0, 1.0), (0.0, 6.0, -1.0), (-4.0, 4.0, 2.0),
              (5.0, -3.0, -2.0), (-6.0, -2.0, 1.5))
    for dx, dy, degrees in starts:
        start = _perturb(truth, dx, dy, degrees)
        refined, info = refine(image, start)
        if info["refined"]:
            error = _court_error(refined, truth)
            assert error < 0.1, (dx, dy, degrees, error, info)
            converged += 1
        else:
            assert np.allclose(refined, start)
    assert converged >= 4, f"only {converged} of {len(starts)} near-miss starts converged"


def test_the_180_degree_rotation_is_accepted_and_that_is_the_documented_limit():
    """End AND side swapped: the court maps onto itself, so every line agrees.

    This is the reverse-angle camera. No geometric check can see it -- the
    keypoint tests pin the same limit -- and it needs a temporal or feed-side
    cue. Pinned so the limit stays stated rather than silently assumed away.
    """
    truth = _truth()
    image = _render(truth)
    rotate = np.array([[-1.0, 0, 50.0], [0, -1.0, 94.0], [0, 0, 1.0]])
    refined, info = refine(image, rotate @ truth)
    assert info["refined"]
    assert _court_error(refined, truth) > 10.0


def test_player_boxes_keep_a_white_edge_from_pulling_the_fit():
    """A white jersey edge parallel to a lane line is the clutter that misled
    ICP. Masked by its player box, it must not move the answer."""
    truth = _truth()
    image = _render(truth, clutter=False)
    court_to_image = np.linalg.inv(truth)
    p = np.c_[[[18.0, 2.0], [18.0, 17.0]], [1, 1]] @ court_to_image.T
    p = (p[:, :2] / p[:, 2:3]).astype(int)
    cv2.line(image, tuple(p[0]), tuple(p[1]), (250, 250, 250), 4)
    x1, y1 = p.min(axis=0) - 10
    x2, y2 = p.max(axis=0) + 10
    start = _perturb(truth, 1.0, 0.5, 0.5)
    refined, info = refine(image, start, boxes=np.array([[x1, y1, x2, y2]]))
    assert info["refined"], info
    assert _court_error(refined, truth) < 0.1


def test_a_start_one_line_spacing_off_does_not_lock_onto_the_neighbour():
    """The broadcast failure: parallel lines 3 ft apart alias each other.

    A single start -- no multi-start to rescue it -- exactly one sideline to
    corner-three spacing off. Segment consensus must pull it to the truth
    rather than let part of the sideline settle on the corner-three line.
    """
    truth = _truth()
    image = _render(truth)
    for dx, dy in ((3.0, 0.0), (-3.0, 0.0), (0.0, 3.0)):
        refined, info = refine(image, _perturb(truth, dx, dy, 0.0),
                               starts=((0.0, 0.0),))
        assert info["refined"], (dx, dy, info)
        assert _court_error(refined, truth) < 0.1, (dx, dy, _court_error(refined, truth))
