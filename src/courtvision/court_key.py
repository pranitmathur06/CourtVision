"""Register the court from the painted key, not from a parameter search.

`court_lines.search_camera` optimises six camera parameters against a mask of
dark thin structure. On real broadcast that objective has deep wrong minima and
its score cannot tell them from the right answer. Drawing the result showed
what it was actually producing: the court collapsed into a blob about a fifth
of its true size, sitting on the wrong basket -- and scoring 0.5 while doing it.
Measured across three runs, 100% of fits put the basket beyond 400 px from an
independently detected rim.

The key removes the search. An NBA lane is a painted rectangle whose corners
are exact court coordinates:

    (17, 0) (33, 0)     baseline
    (17, 19) (33, 19)   free-throw line

Four correspondences determine a homography outright -- no objective, no local
minimum, and nothing for a symmetric court to confuse, because the rim says
which basket the key belongs to.

Measured on 223 court frames of an uncut broadcast, against the search it
replaces:

                              search      key
    frames registered          28.3%     85.7%
    players on the court       75.0%     94.6%

Two details are load-bearing, and both came from looking at frames rather than
reasoning about the objective:

  * The paint is blue and so is the crowd -- this arena's seats and supporters
    are the same hue -- so a colour mask alone leaks into the stands and the
    key's contour comes back with eight or nine corners. The key is on WOOD and
    the crowd is not, which separates them.
  * The corner ORDER decides whether the homography is right or mirrored. The
    rim fixes it: the baseline edge is the pair nearer the basket.
"""

from __future__ import annotations

import numpy as np

from courtvision.court import COURT_WIDTH, FREE_THROW_LINE_Y, LANE_WIDTH

LANE_LEFT_X = (COURT_WIDTH - LANE_WIDTH) / 2.0      # 17 ft
LANE_RIGHT_X = (COURT_WIDTH + LANE_WIDTH) / 2.0     # 33 ft

# Corners in the order `key_homography` expects: baseline pair first, then the
# free-throw pair, wound consistently so the mapping never mirrors.
KEY_COURT_CORNERS = np.array([
    [LANE_LEFT_X, 0.0],
    [LANE_RIGHT_X, 0.0],
    [LANE_RIGHT_X, FREE_THROW_LINE_Y],
    [LANE_LEFT_X, FREE_THROW_LINE_Y],
], dtype=np.float32)

# The painted key's hue. This DEFAULT is one arena's blue and must not be
# trusted anywhere else -- measured across four broadcasts, three paint the key
# at hue 107-113 and one paints it 174, where this range finds the key on 10%
# of court frames instead of 97%. `detect_paint_hue` calibrates it from the
# video; the constant is only a fallback for when calibration fails.
PAINT_HUE = (95, 125)
# Half-width of the calibrated hue window.
PAINT_HUE_TOLERANCE = 15
# Hues within this of 0/180 wrap, and red paint sits exactly there.
HUE_WRAP = 180
PAINT_MIN_SATURATION = 90
PAINT_MIN_VALUE = 90
MIN_PAINT_AREA_PX = 3000
# Hardwood, used only to bound where paint may be found.
WOOD_HUE = (5, 30)
WOOD_MIN_SATURATION = 40
WOOD_MIN_VALUE = 90
MIN_WOOD_AREA_PX = 20000


def court_region(image: np.ndarray) -> np.ndarray | None:
    """Filled hull of the hardwood: the playing surface and its painted areas.

    Returned as a mask so paint can be searched inside it. The hull matters --
    the key is paint, not wood, so it is a hole in the wooden field and only a
    filled hull contains it.
    """
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    wood = ((hue >= WOOD_HUE[0]) & (hue <= WOOD_HUE[1])
            & (saturation >= WOOD_MIN_SATURATION)
            & (value >= WOOD_MIN_VALUE)).astype(np.uint8) * 255
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(wood, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_WOOD_AREA_PX:
        return None
    region = np.zeros(wood.shape, np.uint8)
    # The CONVEX HULL, and it is load-bearing. The key reaches the baseline, so
    # it is a NOTCH in the wooden region rather than an enclosed hole, and
    # filling the outer contour therefore leaves it out. Switching to the plain
    # contour looked like a clean fix -- skin tone falls in the hardwood hue
    # range, so spectators register as wood and the hull stretches over them --
    # and measured much worse: registrations fell 61 -> 27 and the gap to the
    # endpoint's shot spot rose 6.8 -> 13.1 ft at the rim. The hull bridges the
    # notch; the crowd it also swallows is harmless, because the key is still
    # the largest paint-coloured blob inside it.
    cv2.drawContours(region, [cv2.convexHull(biggest)], -1, 255, -1)
    return region


def key_quad(image: np.ndarray,
             exclude_boxes: "np.ndarray | None" = None,
             paint_hue: "tuple[int, int] | None" = None) -> np.ndarray | None:
    """The painted key's four image corners, unordered, or None.

    `exclude_boxes` are detected people. One team here wears the same blue as
    the paint, so a player standing in the key merges with it and drags the
    contour -- and the corners are what the whole registration rests on.
    """
    import cv2

    region = court_region(image)
    if region is None:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    low, high = paint_hue if paint_hue is not None else PAINT_HUE
    paint = ((hue >= low) & (hue <= high)
             & (saturation >= PAINT_MIN_SATURATION)
             & (value >= PAINT_MIN_VALUE)).astype(np.uint8) * 255
    paint = cv2.bitwise_and(paint, region)
    # exclude_boxes is accepted and deliberately NOT applied to the paint mask.
    # One team wears the paint's colour, so masking players out seems obviously
    # right -- and measured much worse: cutting their boxes leaves bites in the
    # key that morphological closing cannot restore, distorting the very corners
    # the registration rests on. Gap to the endpoint's shot spot went 6.8 -> 16.4
    # ft at the rim and 11.7 -> 25.7 ft on threes, and registrations fell from
    # 61 to 22. The contour is robust to a player standing in the key; it is not
    # robust to a hole punched in it.
    paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    paint = cv2.morphologyEx(paint, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(paint, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_PAINT_AREA_PX:
        return None
    perimeter = cv2.arcLength(biggest, True)
    for epsilon in (0.02, 0.03, 0.04, 0.05, 0.06):
        approximated = cv2.approxPolyDP(biggest, epsilon * perimeter, True)
        if len(approximated) == 4:
            return approximated.reshape(4, 2).astype(np.float32)
    # Players standing on a lane line break the contour into more corners than
    # four; the minimum-area rectangle is the honest fallback.
    return cv2.boxPoints(cv2.minAreaRect(biggest)).astype(np.float32)


def order_key_corners(quad: np.ndarray,
                      rim_px: tuple[float, float] | None) -> np.ndarray | None:
    """Order corners baseline-first, matching KEY_COURT_CORNERS.

    Without a rim the baseline edge cannot be told from the free-throw edge,
    and guessing produces a homography flipped end for end -- so this returns
    None rather than choose.
    """
    if rim_px is None or quad is None or len(quad) != 4:
        return None
    distance = np.hypot(quad[:, 0] - rim_px[0], quad[:, 1] - rim_px[1])
    order = np.argsort(distance)
    baseline = quad[order[:2]]
    free_throw = quad[order[2:]]
    axis = free_throw.mean(axis=0) - baseline.mean(axis=0)
    perpendicular = np.array([-axis[1], axis[0]])
    baseline = baseline[np.argsort([point @ perpendicular for point in baseline])]
    free_throw = free_throw[np.argsort([point @ perpendicular
                                        for point in free_throw])]
    return np.array([baseline[0], baseline[1], free_throw[1], free_throw[0]],
                    dtype=np.float32)


def key_homography(image: np.ndarray,
                   rim_px: tuple[float, float] | None,
                   exclude_boxes: "np.ndarray | None" = None,
                   paint_hue: "tuple[int, int] | None" = None
                   ) -> np.ndarray | None:
    """Image-to-court homography from the painted key, or None.

    None means the key was not found or the rim did not identify its baseline.
    A caller must treat that as "no registration", never as "close enough":
    every court coordinate downstream inherits this matrix.
    """
    import cv2

    quad = key_quad(image, exclude_boxes, paint_hue)
    ordered = order_key_corners(quad, rim_px)
    if ordered is None:
        return None
    try:
        court_to_image = cv2.getPerspectiveTransform(KEY_COURT_CORNERS, ordered)
        return np.linalg.inv(court_to_image)
    except (cv2.error, np.linalg.LinAlgError):
        return None


# ---------------------------------------------------------------------------
# Refining the key homography against the rest of the court.
#
# Four corners from one compact quad determine a homography exactly, but they
# constrain it only where they are. Extrapolated to the far arc and the
# division line, small corner errors amplify, and the measured gap between the
# feed's shot location and the nearest detected player was ~10 ft.
#
# The whole court is visible though, and now that the key has put the fit in
# the RIGHT basin, iterative closest point can use all of it. This is exactly
# the local refinement that could not work before: `search_camera` started
# blind and converged into wrong minima the line score could not distinguish
# from the answer. Started from the key, the same evidence becomes usable.

REFINE_ITERATIONS = 6
# Correspondences further than this from a line are outliers -- a projected
# point over a player, a scoreboard edge, a crowd gap -- and re-fitting to them
# drags the whole homography.
REFINE_MAX_SNAP_PX = 40.0
MIN_REFINE_POINTS = 30


def refine_homography(image: np.ndarray,
                      court_from_image: np.ndarray,
                      exclude_boxes: "np.ndarray | None" = None,
                      iterations: int = REFINE_ITERATIONS,
                      max_snap_px: float = REFINE_MAX_SNAP_PX
                      ) -> np.ndarray:
    """Tighten a court registration using every visible line, not just the key.

    Returns the refined image-to-court matrix, or the input unchanged when the
    evidence is too thin to improve on it -- refusing to move is correct when
    there is nothing to move toward.

    MEASURED AND HARMFUL. Do not enable this without reading the numbers.
    Against the endpoint's own shot locations, which know nothing about line
    pixels:

                       p50 gap    within 6 ft
        key only        10.2 ft      35.7%
        + ICP           16.5 ft      12.5%

    while the line-distance metric it optimises improved from 7.60 ft to
    1.30 ft. The metric moved one way and the truth the other, for the third
    time in this project.

    The cause is correspondence, not convergence: snapping a projected point to
    the NEAREST line pixel is wrong on a court full of parallel lines. A point
    on the three-point arc snaps to the free-throw circle, a far lane line to
    the near one, and re-fitting to those pairs drags the homography. Kept only
    so the measurement stays attached to the idea.
    """
    import cv2

    from courtvision.court_lines import canonical_court_points, court_line_mask

    mask = court_line_mask(image, exclude_boxes)
    if not mask.any():
        return court_from_image
    distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    # Nearest line pixel for every image position, so a projected point can be
    # snapped without searching.
    _, labels = cv2.distanceTransformWithLabels(
        255 - mask, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_REFINE_POINTS:
        return court_from_image
    # cv2 labels are 1-based indices into the sorted list of zero pixels.
    order = np.lexsort((xs, ys))
    line_xy = np.stack([xs[order], ys[order]], axis=1)

    court = canonical_court_points(1.0)
    height, width = mask.shape
    current = court_from_image
    for _ in range(iterations):
        try:
            court_to_image = np.linalg.inv(current)
        except np.linalg.LinAlgError:
            return court_from_image
        projected = np.hstack([court, np.ones((len(court), 1))]) @ court_to_image.T
        valid = np.abs(projected[:, 2]) > 1e-9
        image_xy = projected[valid, :2] / projected[valid, 2:3]
        source = court[valid]
        inside = ((image_xy[:, 0] >= 0) & (image_xy[:, 0] < width)
                  & (image_xy[:, 1] >= 0) & (image_xy[:, 1] < height))
        image_xy, source = image_xy[inside], source[inside]
        if len(image_xy) < MIN_REFINE_POINTS:
            return current
        columns = image_xy[:, 0].astype(int)
        rows = image_xy[:, 1].astype(int)
        snap_distance = distance[rows, columns]
        near = snap_distance <= max_snap_px
        if near.sum() < MIN_REFINE_POINTS:
            return current
        index = labels[rows[near], columns[near]] - 1
        index = np.clip(index, 0, len(line_xy) - 1)
        target = line_xy[index].astype(np.float32)
        updated, _ = cv2.findHomography(
            source[near].astype(np.float32), target, cv2.RANSAC, 5.0)
        if updated is None or not np.isfinite(updated).all():
            return current
        try:
            current = np.linalg.inv(updated)
        except np.linalg.LinAlgError:
            return current
    return current


# ---------------------------------------------------------------------------
# Corner precision is what limits this, not the search.
#
# approxPolyDP returns coarse polygon vertices, and the key is a small quad
# whose homography is extrapolated across a 94 ft court -- so a couple of
# pixels at a corner becomes feet at the far arc. Fitting a line to each EDGE
# of the contour and intersecting adjacent lines uses every pixel along that
# edge instead of the one vertex the approximation happened to pick.

EDGE_INLIER_PX = 3.0
MIN_EDGE_POINTS = 8


def _fit_line(points: np.ndarray) -> tuple[float, float, float] | None:
    """Total-least-squares line through points, as (a, b, c) with ax+by=c."""
    if len(points) < MIN_EDGE_POINTS:
        return None
    centre = points.mean(axis=0)
    centred = points - centre
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]])
    return float(normal[0]), float(normal[1]), float(normal @ centre)


def _intersect(first, second) -> tuple[float, float] | None:
    a1, b1, c1 = first
    a2, b2, c2 = second
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-9:
        return None
    return ((c1 * b2 - c2 * b1) / determinant,
            (a1 * c2 - a2 * c1) / determinant)


def precise_key_corners(image: np.ndarray,
                        quad: np.ndarray,
                        paint_hue: "tuple[int, int] | None" = None
                        ) -> np.ndarray | None:
    """Sub-pixel key corners, from the contour's four edges.

    Takes the coarse quad as a starting guess, assigns every contour pixel to
    the nearest of its four edges, fits a line per edge, and intersects
    neighbours. Returns None if any edge is too sparse to fit, because a corner
    from a two-pixel edge is worse than the approximation it replaces.

    `paint_hue` is the arena's own key colour from `detect_paint_hue`. It was
    referenced in the body without being a parameter: any call that reached
    the body raised NameError, and a call passing the keyword raised TypeError.
    A frame with no court returns before that line, which is why the fault
    stayed hidden -- along with there being no callers and no tests.
    """
    import cv2

    region = court_region(image)
    if region is None or quad is None or len(quad) != 4:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    low, high = paint_hue if paint_hue is not None else PAINT_HUE
    paint = ((hue >= low) & (hue <= high)
             & (saturation >= PAINT_MIN_SATURATION)
             & (value >= PAINT_MIN_VALUE)).astype(np.uint8) * 255
    paint = cv2.bitwise_and(paint, region)
    paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(paint, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)

    lines = []
    for index in range(4):
        start, end = quad[index], quad[(index + 1) % 4]
        edge = end - start
        length = float(np.hypot(*edge))
        if length < 1e-6:
            return None
        unit = edge / length
        normal = np.array([-unit[1], unit[0]])
        relative = outline - start
        along = relative @ unit
        across = np.abs(relative @ normal)
        on_edge = (along >= 0) & (along <= length) & (across <= EDGE_INLIER_PX)
        fitted = _fit_line(outline[on_edge])
        if fitted is None:
            return None
        lines.append(fitted)

    corners = []
    for index in range(4):
        point = _intersect(lines[index - 1], lines[index])
        if point is None:
            return None
        corners.append(point)
    corners = np.array(corners, dtype=np.float32)
    # Guard against a degenerate fit flinging a corner off the image.
    if np.abs(corners - quad).max() > 40.0:
        return None
    return corners


# ---------------------------------------------------------------------------
# Rejecting a key that cannot support a registration.
#
# Geometry at the basket measures p50 2.34 ft but p75 5.56 and p90 9.82: good
# on most frames, badly wrong on a minority. For a tool that tells a player
# where he should have been, a wrong court is worse than no court, so the tail
# must be refused rather than averaged in.
#
# Four hand-built guesses at what marks a bad frame all failed -- ICP
# refinement, masking players out of the paint, filling the contour instead of
# its hull, and a quad gate on border/convexity/angles (coverage 89.8% -> 50.4%
# for no accuracy gain at all). So the predictor was measured instead, by
# correlating candidate signals against the basket-to-rim error in feet:
#
#     distance from key centre to rim   |r| 0.45   <- strongest
#     quad area                         |r| 0.35
#     hardwood fraction                 |r| 0.34
#     pixels per foot                   |r| 0.12
#     rim confidence                    |r| 0.11
#
# By tercile of that distance: 1.80 ft, 2.65 ft, 6.98 ft, with the share within
# 3 ft falling 68% -> 63% -> 21%. It is physically meaningful rather than
# fitted: when the key's centre sits far from the rim in the image, the key and
# the rim belong to DIFFERENT baskets, and pairing them registers the court
# against a correspondence that was never true.

# 220 px, not the looser 280 or 340. Measured on basket-to-rim error in feet:
#
#     gate            coverage   p50     p75     within 3 ft
#     none              92.3%   2.84    6.65      50.7%
#     <= 280 px         76.4%   2.61    4.61      57.9%
#     <= 220 px         18.5%   1.72    2.43      79.1%
#
# Per-frame coverage collapses, and that is the wrong denominator: analysis is
# anchored to EVENTS, and a caller searching a few frames either side of one
# only needs a single frame to pass. Accuracy per accepted frame is what cannot
# be recovered later.
MAX_KEY_TO_RIM_PX = 220.0


def key_rim_distance(quad: np.ndarray,
                     rim_px: tuple[float, float] | None) -> float | None:
    """Pixels from the key's centre to the detected rim."""
    if quad is None or len(quad) != 4 or rim_px is None:
        return None
    centre = np.asarray(quad, dtype=float).mean(axis=0)
    return float(np.hypot(centre[0] - rim_px[0], centre[1] - rim_px[1]))


def key_matches_rim(quad: np.ndarray,
                    rim_px: tuple[float, float] | None,
                    max_px: float = MAX_KEY_TO_RIM_PX) -> bool:
    """Whether this key and this rim plausibly belong to the same basket."""
    distance = key_rim_distance(quad, rim_px)
    return distance is not None and distance <= max_px


def detect_paint_hue(images: "list[np.ndarray]",
                     tolerance: int = PAINT_HUE_TOLERANCE
                     ) -> "tuple[int, int] | None":
    """Calibrate the key's colour from the video itself.

    Every arena paints its own key. Measured across four broadcasts, three sit
    at hue 107-113 and one at 174, and a range fixed on the first finds the
    fourth's key on 10% of court frames instead of 97%.

    Takes the dominant saturated hue inside the court region -- the key is the
    largest strongly-coloured area on a wooden floor. Returns None when no such
    colour is found, so a caller can fall back rather than register against
    whatever happened to be reddest.
    """
    import cv2

    from courtvision.court_tracking import has_court

    counts = np.zeros(HUE_WRAP, dtype=np.int64)
    for image in images:
        if not has_court(image):
            continue
        region = court_region(image)
        if region is None:
            continue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        strong = ((region > 0)
                  & (hsv[:, :, 1] >= PAINT_MIN_SATURATION)
                  & (hsv[:, :, 2] >= PAINT_MIN_VALUE))
        if strong.sum() < 500:
            continue
        counts += np.bincount(hsv[:, :, 0][strong].ravel(),
                              minlength=HUE_WRAP)[:HUE_WRAP]
    if counts.sum() < 5000:
        return None
    # Wooden floors are hue 5-30 and dominate any court; exclude them so the
    # paint is what is left.
    searchable = counts.copy()
    searchable[:35] = 0
    if searchable.sum() < 1000:
        return None
    peak = int(np.argmax(searchable))
    return (max(0, peak - tolerance), min(HUE_WRAP - 1, peak + tolerance))
