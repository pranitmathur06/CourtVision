"""Court registration — image pixels to real court coordinates.

v3's remaining hard problem is "what play are they running", and that cannot
start from pixel positions. Two players 100 px apart are adjacent at the far
baseline and a passing lane apart in the near corner; formation only means
something in court coordinates. So registration comes first.

Coordinates are FEET on a half court, origin at the left end of the baseline
under the basket, x along the baseline (0-50), y into the court (0-47). Every
landmark below is from the NBA rulebook, not measured off a frame.

This module deliberately does NOT try to find the court automatically. Robust
line detection on arbitrary broadcast footage — with crowds, glare, logos
painted on the floor and a camera that pans — is a research problem, and a
homography that is silently wrong is worse than none: it puts players in
plausible but incorrect places and every downstream formation claim inherits
the error. What it provides is the calibration maths and an honest error
measurement, so a caller who supplies correspondences learns whether they were
good enough to use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# NBA half court, in feet.
COURT_WIDTH = 50.0
HALF_COURT_LENGTH = 47.0
BASKET = (25.0, 5.25)          # centre of the rim, 63 inches from the baseline
LANE_WIDTH = 16.0
FREE_THROW_LINE_Y = 19.0
THREE_POINT_RADIUS = 23.75
CORNER_THREE_X = 3.0           # corner three sits 22 ft from the basket, 3 ft in

LANDMARKS: dict[str, tuple[float, float]] = {
    "baseline_left":        (0.0, 0.0),
    "baseline_right":       (COURT_WIDTH, 0.0),
    "corner_three_left":    (CORNER_THREE_X, 0.0),
    "corner_three_right":   (COURT_WIDTH - CORNER_THREE_X, 0.0),
    "lane_left_baseline":   ((COURT_WIDTH - LANE_WIDTH) / 2, 0.0),
    "lane_right_baseline":  ((COURT_WIDTH + LANE_WIDTH) / 2, 0.0),
    "lane_left_ft":         ((COURT_WIDTH - LANE_WIDTH) / 2, FREE_THROW_LINE_Y),
    "lane_right_ft":        ((COURT_WIDTH + LANE_WIDTH) / 2, FREE_THROW_LINE_Y),
    "free_throw_centre":    (COURT_WIDTH / 2, FREE_THROW_LINE_Y),
    "basket":               BASKET,
    "half_court_left":      (0.0, HALF_COURT_LENGTH),
    "half_court_right":     (COURT_WIDTH, HALF_COURT_LENGTH),
}


@dataclass(frozen=True)
class Registration:
    """A fitted image-to-court homography and how much to trust it."""

    matrix: np.ndarray            # 3x3, image pixels -> court feet
    rms_error_ft: float           # reprojection error over the correspondences
    max_error_ft: float
    landmarks_used: tuple[str, ...]

    def is_usable(self, tolerance_ft: float = 3.0) -> bool:
        """Whether the fit is tight enough to reason about formation.

        Three feet is roughly a player's shoulder width. Beyond that, "who is
        setting a screen for whom" stops being answerable, so a caller should
        decline to guess rather than emit a confident wrong play name.
        """
        return self.max_error_ft <= tolerance_ft

    def to_court(self, points: np.ndarray) -> np.ndarray:
        """Map image points (N, 2) to court feet (N, 2)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
        projected = homogeneous @ self.matrix.T
        w = projected[:, 2:3]
        # A point on the horizon maps to infinity; report NaN rather than a huge
        # finite number that would look like a real position on the court.
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(np.abs(w) < 1e-9, np.nan, projected[:, :2] / w)
        return out

    def on_court(self, points: np.ndarray, margin_ft: float = 5.0) -> np.ndarray:
        """Boolean mask of mapped points that land on (or just off) the court."""
        court = self.to_court(points)
        inside = (
            (court[:, 0] >= -margin_ft)
            & (court[:, 0] <= COURT_WIDTH + margin_ft)
            & (court[:, 1] >= -margin_ft)
            & (court[:, 1] <= HALF_COURT_LENGTH + margin_ft)
        )
        return inside & ~np.isnan(court).any(axis=1)


def register(correspondences: dict[str, tuple[float, float]]) -> Registration:
    """Fit image -> court from named landmarks seen in the image.

    `correspondences` maps a LANDMARKS key to its pixel position. Four are the
    minimum for a homography; more are strongly preferred, because with exactly
    four the fit passes through every point and the reported error is zero
    whether or not the correspondences were right.
    """
    import cv2

    unknown = set(correspondences) - set(LANDMARKS)
    if unknown:
        raise ValueError(f"unknown landmarks: {sorted(unknown)}")
    if len(correspondences) < 4:
        raise ValueError(
            f"need at least 4 correspondences, got {len(correspondences)}"
        )

    names = tuple(sorted(correspondences))
    image_pts = np.array([correspondences[n] for n in names], dtype=np.float64)
    court_pts = np.array([LANDMARKS[n] for n in names], dtype=np.float64)

    matrix, _ = cv2.findHomography(image_pts, court_pts, method=0)
    if matrix is None:
        raise ValueError("homography fit failed; are the points collinear?")

    fitted = Registration(matrix, 0.0, 0.0, names)
    projected = fitted.to_court(image_pts)
    errors = np.linalg.norm(projected - court_pts, axis=1)
    return Registration(
        matrix=matrix,
        rms_error_ft=float(math.sqrt(float((errors ** 2).mean()))),
        max_error_ft=float(errors.max()),
        landmarks_used=names,
    )


def distance_to_basket_ft(court_points: np.ndarray) -> np.ndarray:
    """Straight-line distance from court positions to the rim, in feet."""
    pts = np.asarray(court_points, dtype=np.float64).reshape(-1, 2)
    return np.linalg.norm(pts - np.array(BASKET), axis=1)


def is_three_point_attempt(court_point: tuple[float, float]) -> bool:
    """Whether a shot from this court position is worth three.

    The line is not a plain circle: it is cut off by two straight corner
    sections, so a shot 23 ft from the rim is a three at the top of the arc and
    a two from the corner.
    """
    x, y = float(court_point[0]), float(court_point[1])
    if x <= CORNER_THREE_X or x >= COURT_WIDTH - CORNER_THREE_X:
        return y <= 14.0        # corner section runs to where the arc meets it
    return bool(distance_to_basket_ft([[x, y]])[0] >= THREE_POINT_RADIUS)
