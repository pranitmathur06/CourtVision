"""Find court lines in a frame, and score a homography against them.

`court.register()` takes correspondences on trust. Its own error figure only
measures how well the fit reproduces the points it was given, which — with four
of them — is zero however wrong they were. This module supplies the independent
check: project the canonical court into the image and ask whether its lines land
where the image actually has lines.

The detector is built from measurement, not assumption. On a Pistons broadcast
the lines look navy, but thresholding blue (hue 100-135) finds the bench area,
the floor advertising and the crowd while missing every arc. The lines sample at
hue 155-175 and, far more reliably, at grey 52-110 against a local median of
198-225. So they are found as DARK pixels relative to a local median, which also
survives the floor being lit brightly at the far sideline and shadowed near the
camera.

Known contamination: players are dark too, and their edges and jersey numbers
survive the blob filter. That inflates `alignment_score` slightly, since a
projected line can land on a player instead of a line. It biases toward
accepting a homography, so treat the score as an upper bound and keep the
threshold strict.
"""

from __future__ import annotations

import numpy as np

from courtvision.court import (
    COURT_WIDTH,
    CORNER_THREE_X,
    FREE_THROW_LINE_Y,
    HALF_COURT_LENGTH,
    LANDMARKS,
    THREE_POINT_RADIUS,
    BASKET,
)

DARKNESS_THRESHOLD = 25        # how much darker than local median a line pixel is
LOCAL_MEDIAN_KERNEL = 51
PLAYER_BLOB_KERNEL = 11


def court_line_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask of pixels that plausibly belong to a painted court line."""
    import cv2

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    local = cv2.medianBlur(grey, LOCAL_MEDIAN_KERNEL)
    darker = cv2.subtract(local, grey)
    candidates = cv2.inRange(darker, DARKNESS_THRESHOLD, 255)

    wood = cv2.inRange(hsv, (0, 35, 110), (30, 255, 255))
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(wood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(grey.shape, np.uint8)
    floor = np.zeros(grey.shape, np.uint8)
    cv2.drawContours(floor, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    floor = cv2.erode(floor, np.ones((9, 9), np.uint8))

    lines = cv2.bitwise_and(candidates, floor)
    # Players are dark but solid; lines are thin. Opening keeps only the solid
    # regions, which are then removed with a margin for their edges.
    blobs = cv2.morphologyEx(lines, cv2.MORPH_OPEN,
                             np.ones((PLAYER_BLOB_KERNEL, PLAYER_BLOB_KERNEL), np.uint8))
    return cv2.subtract(lines, cv2.dilate(blobs, np.ones((9, 9), np.uint8)))


def canonical_court_points(step_ft: float = 1.0) -> np.ndarray:
    """Points along the painted lines of a half court, in feet."""
    pieces: list[np.ndarray] = []

    def segment(a, b):
        n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) / step_ft), 2)
        return np.stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)], axis=1)

    pieces.append(segment((0, 0), (COURT_WIDTH, 0)))                    # baseline
    pieces.append(segment((0, 0), (0, HALF_COURT_LENGTH)))              # sidelines
    pieces.append(segment((COURT_WIDTH, 0), (COURT_WIDTH, HALF_COURT_LENGTH)))
    lane_l, lane_r = LANDMARKS["lane_left_baseline"], LANDMARKS["lane_right_baseline"]
    pieces.append(segment(lane_l, LANDMARKS["lane_left_ft"]))           # lane
    pieces.append(segment(lane_r, LANDMARKS["lane_right_ft"]))
    pieces.append(segment(LANDMARKS["lane_left_ft"], LANDMARKS["lane_right_ft"]))

    # Three-point line: two straight corner sections and the arc between them.
    corner_y = float(np.sqrt(max(THREE_POINT_RADIUS ** 2
                                 - (COURT_WIDTH / 2 - CORNER_THREE_X) ** 2, 0.0)))
    corner_y += BASKET[1]
    pieces.append(segment((CORNER_THREE_X, 0), (CORNER_THREE_X, corner_y)))
    pieces.append(segment((COURT_WIDTH - CORNER_THREE_X, 0),
                          (COURT_WIDTH - CORNER_THREE_X, corner_y)))
    start = np.arctan2(corner_y - BASKET[1], CORNER_THREE_X - BASKET[0])
    end = np.arctan2(corner_y - BASKET[1], COURT_WIDTH - CORNER_THREE_X - BASKET[0])
    angles = np.linspace(start, end, 120)
    pieces.append(np.stack([BASKET[0] + THREE_POINT_RADIUS * np.cos(angles),
                            BASKET[1] + THREE_POINT_RADIUS * np.sin(angles)], axis=1))

    # Free-throw circle.
    circle = np.linspace(0, 2 * np.pi, 90)
    pieces.append(np.stack([COURT_WIDTH / 2 + 6.0 * np.cos(circle),
                            FREE_THROW_LINE_Y + 6.0 * np.sin(circle)], axis=1))
    return np.vstack(pieces)


def alignment_score(
    image_to_court: np.ndarray, image: np.ndarray, tolerance_px: float = 6.0
) -> float:
    """Fraction of the projected court outline that lands on detected lines.

    1.0 means every visible piece of the canonical court fell on a line pixel;
    a wrong homography scores near zero because its lines land on bare floor.
    Points that project outside the frame are ignored rather than counted as
    misses — most of a half court is off-screen in a broadcast shot.
    """
    import cv2

    mask = court_line_mask(image)
    if not mask.any():
        return 0.0
    # Distance to the nearest detected line pixel, so "close enough" is cheap.
    distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)

    court_to_image = np.linalg.inv(image_to_court)
    pts = canonical_court_points()
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    projected = homogeneous @ court_to_image.T
    w = projected[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = np.where(np.abs(w) < 1e-9, np.nan, projected[:, :2] / w)

    h, wid = mask.shape
    inside = (
        ~np.isnan(xy).any(axis=1)
        & (xy[:, 0] >= 0) & (xy[:, 0] < wid)
        & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    )
    if inside.sum() < 20:
        return 0.0
    visible = xy[inside].astype(int)
    hits = distance[visible[:, 1], visible[:, 0]] <= tolerance_px
    return float(hits.mean())
