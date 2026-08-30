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


def line_distance_map(image: np.ndarray) -> np.ndarray:
    """Distance to the nearest detected line pixel, computed once.

    `alignment_score` re-detects lines on every call, which costs 46 ms. A
    search evaluates thousands of candidates, so it wants this precomputed and
    each evaluation reduced to a projection and a lookup.
    """
    import cv2

    mask = court_line_mask(image)
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)


def homography_from_camera(
    params: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray | None:
    """Court-to-image homography from 6 physical camera parameters.

    Parameters are (cx, cy, cz, tx, ty, focal): where the camera is in court
    feet, the point on the floor it is aimed at, and its focal length in pixels.

    Six physical parameters rather than a homography's eight free ones. Every
    point in this space is a camera that could exist; most of the 8-dimensional
    space is not, and a search there spends its time on homographies that fold
    the court through itself.
    """
    cx, cy, cz, tx, ty, focal = params
    if cz <= 1.0 or focal <= 1.0:
        return None
    centre = np.array([cx, cy, cz], dtype=np.float64)
    target = np.array([tx, ty, 0.0], dtype=np.float64)

    forward = target - centre
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return None
    forward /= norm
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    norm = np.linalg.norm(right)
    if norm < 1e-6:                      # looking straight down: right is undefined
        return None
    right /= norm
    down = np.cross(forward, right)

    rotation = np.stack([right, down, forward])          # world -> camera
    height, width = image_shape[:2]
    intrinsics = np.array([[focal, 0.0, width / 2.0],
                           [0.0, focal, height / 2.0],
                           [0.0, 0.0, 1.0]])
    # A court point is (X, Y, 0), so only the first two rotation columns matter.
    extrinsic = np.stack([rotation[:, 0], rotation[:, 1], -rotation @ centre], axis=1)
    return intrinsics @ extrinsic


def score_homography(
    court_to_image: np.ndarray,
    distance_map: np.ndarray,
    points: np.ndarray | None = None,
    line_pixels: np.ndarray | None = None,
    tolerance_px: float = 6.0,
) -> float:
    """How well a homography explains the image's lines, in BOTH directions.

    Asking only "do the model's lines land on detected lines" is not enough, and
    the failure is not subtle. A search using that alone returned a camera
    scoring 0.998 whose court coordinates were several hundred feet wrong: it
    had zoomed onto a small patch where a couple of arcs happened to coincide,
    and every projected point landed on a line because only a sliver of court
    was in frame. Measured against the true camera:

        true camera     recall 1.000   coverage 0.801
        search winner   recall 0.998   coverage 0.023

    So coverage — the share of DETECTED line pixels the model accounts for — is
    what separates them, and the score is the harmonic mean of the two. A
    homography must both land on lines and explain the lines that are there.
    """
    import cv2

    if points is None:
        points = canonical_court_points()
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    projected = homogeneous @ court_to_image.T
    w = projected[:, 2]
    valid = np.abs(w) > 1e-9
    if valid.sum() < 20:
        return 0.0
    xy = projected[valid, :2] / w[valid, None]

    height, width = distance_map.shape
    inside = ((xy[:, 0] >= 0) & (xy[:, 0] < width)
              & (xy[:, 1] >= 0) & (xy[:, 1] < height))
    if inside.sum() < 0.15 * len(points):
        return 0.0
    visible = xy[inside].astype(int)
    recall = float((distance_map[visible[:, 1], visible[:, 0]] <= tolerance_px).mean())
    if line_pixels is None or len(line_pixels) == 0 or recall == 0.0:
        return recall

    canvas = np.zeros((height, width), np.uint8)
    canvas[visible[:, 1], visible[:, 0]] = 255
    reach = int(tolerance_px) * 2 + 1
    canvas = cv2.dilate(canvas, np.ones((reach, reach), np.uint8))
    coverage = float((canvas[line_pixels[:, 1], line_pixels[:, 0]] > 0).mean())
    if coverage <= 0.0:
        return 0.0
    return 2.0 * recall * coverage / (recall + coverage)


def sample_line_pixels(image: np.ndarray, limit: int = 4000,
                       seed: int = 0) -> np.ndarray:
    """A random subset of detected line pixels, as (N, 2) x/y — for coverage."""
    mask = court_line_mask(image)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=int)
    if len(xs) > limit:
        pick = np.random.default_rng(seed).choice(len(xs), limit, replace=False)
        ys, xs = ys[pick], xs[pick]
    return np.stack([xs, ys], axis=1)


DEFAULT_CAMERA_BOUNDS = [
    (-60.0, 110.0),     # camera x, feet
    (-50.0, 90.0),      # camera y
    (12.0, 65.0),       # camera height
    (5.0, 45.0),        # aim point x
    (0.0, 40.0),        # aim point y
    (700.0, 3200.0),    # focal length, pixels
]


def search_registration(
    image: np.ndarray,
    seed: int = 0,
    max_iterations: int = 250,
    bounds: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray | None, float]:
    """Search camera parameters for the homography that best explains the lines.

    Returns (image_to_court, score), or (None, 0.0) if nothing plausible was
    found. Differential evolution over the six physical parameters.

    Random restarts plus Nelder-Mead was tried first and does not work: the
    objective's good basin is narrow, 600 restarts across the space never
    landed in it, and the refinement stalled at 0.233 against the true camera's
    0.888. Differential evolution finds it — on a synthetic court with a known
    camera it recovers court coordinates to 0.36 ft, roughly four inches.

    The full search needs its budget. At maxiter 25 and 60 it returns 0.198 and
    0.229 against the true camera's 0.888, so a cheap run is not a fast answer,
    it is a wrong one — check the score. Narrow `bounds` when the camera's rough
    placement is known; it converges far faster and more reliably than the
    default global sweep.

    The score is two-sided but still an upper bound, because players contaminate
    the line mask. Check it before trusting the result: a homography that
    explains the lines poorly is worse than none, since it puts players in
    plausible-looking but wrong places.
    """
    from scipy.optimize import differential_evolution

    distance_map = line_distance_map(image)
    if not np.isfinite(distance_map).any():
        return None, 0.0
    points = canonical_court_points()
    line_pixels = sample_line_pixels(image)
    if len(line_pixels) == 0:
        return None, 0.0
    shape = image.shape[:2]

    def negative_score(params: np.ndarray) -> float:
        matrix = homography_from_camera(params, shape)
        if matrix is None:
            return 0.0
        return -score_homography(matrix, distance_map, points, line_pixels)

    result = differential_evolution(
        negative_score, bounds or DEFAULT_CAMERA_BOUNDS,
        seed=seed, maxiter=max_iterations,
        popsize=25, tol=1e-6, polish=True, init="sobol",
    )
    score = float(-result.fun)
    matrix = homography_from_camera(result.x, shape)
    if matrix is None or score <= 0.0:
        return None, 0.0
    return np.linalg.inv(matrix), score


def search_camera(
    image: np.ndarray,
    seed: int = 0,
    max_iterations: int = 250,
    bounds: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray | None, float]:
    """As `search_registration`, but returns the 6 CAMERA PARAMETERS.

    A caller tracking a panning camera needs them: the next frame's search can
    start from tight bounds around this frame's solution, which is far faster
    and more reliable than searching the whole space again. The homography alone
    cannot be decomposed back into them unambiguously.
    """
    from scipy.optimize import differential_evolution

    distance_map = line_distance_map(image)
    if not np.isfinite(distance_map).any():
        return None, 0.0
    points = canonical_court_points()
    line_pixels = sample_line_pixels(image)
    if len(line_pixels) == 0:
        return None, 0.0
    shape = image.shape[:2]

    def negative_score(params: np.ndarray) -> float:
        matrix = homography_from_camera(params, shape)
        if matrix is None:
            return 0.0
        return -score_homography(matrix, distance_map, points, line_pixels)

    result = differential_evolution(
        negative_score, bounds or DEFAULT_CAMERA_BOUNDS,
        seed=seed, maxiter=max_iterations,
        popsize=25, tol=1e-6, polish=True, init="sobol",
    )
    score = float(-result.fun)
    if score <= 0.0 or homography_from_camera(result.x, shape) is None:
        return None, 0.0
    return np.asarray(result.x, dtype=float), score
