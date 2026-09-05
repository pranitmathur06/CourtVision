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

# The painted key's hue. Wide enough for broadcast colour grading, and paired
# with a saturation floor that ordinary shadowed hardwood never reaches.
PAINT_HUE = (95, 125)
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
    cv2.drawContours(region, [cv2.convexHull(biggest)], -1, 255, -1)
    return region


def key_quad(image: np.ndarray) -> np.ndarray | None:
    """The painted key's four image corners, unordered, or None."""
    import cv2

    region = court_region(image)
    if region is None:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    paint = ((hue >= PAINT_HUE[0]) & (hue <= PAINT_HUE[1])
             & (saturation >= PAINT_MIN_SATURATION)
             & (value >= PAINT_MIN_VALUE)).astype(np.uint8) * 255
    paint = cv2.bitwise_and(paint, region)
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
                   rim_px: tuple[float, float] | None
                   ) -> np.ndarray | None:
    """Image-to-court homography from the painted key, or None.

    None means the key was not found or the rim did not identify its baseline.
    A caller must treat that as "no registration", never as "close enough":
    every court coordinate downstream inherits this matrix.
    """
    import cv2

    quad = key_quad(image)
    ordered = order_key_corners(quad, rim_px)
    if ordered is None:
        return None
    try:
        court_to_image = cv2.getPerspectiveTransform(KEY_COURT_CORNERS, ordered)
        return np.linalg.inv(court_to_image)
    except (cv2.error, np.linalg.LinAlgError):
        return None
