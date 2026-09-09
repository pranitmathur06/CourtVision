"""Court registration from learned landmarks, instead of one painted quad.

The painted-key method needs a single unoccluded quadrilateral. Mid-possession
five players stand in the paint, so it comes back a fragment, a triangle or a
self-intersecting bowtie: it registers 17% of frames there, and the frames it
does pass disagree with each other by 5.8 ft. It also does not travel -- gated,
on 745 frames from other arenas, it registered ONE.

Forty-eight landmarks spread across the whole floor fail differently. Any
handful visible anywhere determines the homography, RANSAC discards the bad
ones, and occlusion in the paint costs a few points rather than the fit.

## Where this schema came from

The dataset (850 images, CC BY 4.0) ships 48 keypoints per frame and no
statement of what any of them IS. Guessing would put a hand-made assumption
under every registration built on it, so the mapping was recovered instead:

1. **Canonical layout.** The court is a rigid plane, so any two annotated
   frames are related by a homography. Mapping all 849 annotations into a
   common frame through their shared keypoints, iteratively, placed 846 of them
   and located 32 indices agreeing to within 0.02 of a frame width.
2. **Rectification by the dataset's own symmetry.** `flip_idx` declares which
   landmarks mirror about the half-court line. Fifteen pairs give thirty
   equations -- `y_a + y_b = 94`, `x_a = x_b` -- for a homography's eight
   parameters; the court's real dimensions close the remaining scale freedom.
   No registration and no eyeballing were involved.
3. **Identification, as an independent check.** The fit was never told where a
   lane corner or an arc is. It placed the baseline corners at the corners,
   the lane at 17 and 33, the arc top at 25, the basket at (25, 5.25) and
   centre court at (25, 47) -- 18 of 32 within 2.5 ft of a named landmark, and
   the rest on landmarks the reference list had simply omitted.

Because every one of the 32 is a standard NBA marking, the constants below are
the EXACT court positions rather than the fit's estimates of them; the fit's
job was to identify them, not to measure a court whose dimensions are known.

An earlier attempt bootstrapped these coordinates from the painted-key
registration instead. It scattered by 15 ft and failed the symmetry check, and
gating it left one usable frame in 745. That check is why this file does not
contain those numbers.
"""

from __future__ import annotations

import numpy as np

# Court frame: x across 0-50, y along 0-94, matching `courtvision.court`.
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
_LANE_L, _LANE_R = 17.0, 33.0
_FT_Y = 19.0
_BASKET_Y = 5.25
_ARC_TOP_Y = _BASKET_Y + 23.75          # arc apex, 23.75 ft from the rim
_CORNER3_X = 3.0                        # corner three, 3 ft inside the sideline
_CORNER3_Y = 14.0                       # where the straight section meets the arc
_HASH_Y = 28.0                          # sideline hash, 28 ft from the baseline

#: Keypoint index -> (x, y) in court feet. Only the 32 indices the annotators
#: actually mark are present; the other 16 are never labelled in 850 frames.
KEYPOINTS: dict[int, tuple[float, float]] = {
    # Near baseline (y = 0) and its mirror at y = 94.
    0:  (0.0, 0.0),                       35: (0.0, COURT_LENGTH_FT),
    7:  (COURT_WIDTH_FT, 0.0),            42: (COURT_WIDTH_FT, COURT_LENGTH_FT),
    1:  (_CORNER3_X, 0.0),                36: (_CORNER3_X, COURT_LENGTH_FT),
    6:  (COURT_WIDTH_FT - _CORNER3_X, 0.0),
    41: (COURT_WIDTH_FT - _CORNER3_X, COURT_LENGTH_FT),
    3:  (_LANE_L, 0.0),                   38: (_LANE_L, COURT_LENGTH_FT),
    4:  (_LANE_R, 0.0),                   39: (_LANE_R, COURT_LENGTH_FT),
    # The baskets.
    8:  (25.0, _BASKET_Y),                34: (25.0, COURT_LENGTH_FT - _BASKET_Y),
    # Free-throw line: both lane corners and its centre.
    12: (_LANE_L, _FT_Y),                 29: (_LANE_L, COURT_LENGTH_FT - _FT_Y),
    13: (25.0, _FT_Y),                    30: (25.0, COURT_LENGTH_FT - _FT_Y),
    14: (_LANE_R, _FT_Y),                 31: (_LANE_R, COURT_LENGTH_FT - _FT_Y),
    # Where the corner three's straight section meets the arc.
    9:  (_CORNER3_X, _CORNER3_Y),
    32: (_CORNER3_X, COURT_LENGTH_FT - _CORNER3_Y),
    11: (COURT_WIDTH_FT - _CORNER3_X, _CORNER3_Y),
    33: (COURT_WIDTH_FT - _CORNER3_X, COURT_LENGTH_FT - _CORNER3_Y),
    # Arc apex.
    17: (25.0, _ARC_TOP_Y),               27: (25.0, COURT_LENGTH_FT - _ARC_TOP_Y),
    # Sideline hash marks, 28 ft from each baseline.
    16: (0.0, _HASH_Y),                   26: (0.0, COURT_LENGTH_FT - _HASH_Y),
    18: (COURT_WIDTH_FT, _HASH_Y),        28: (COURT_WIDTH_FT, COURT_LENGTH_FT - _HASH_Y),
    # Half-court line.
    20: (0.0, COURT_LENGTH_FT / 2),       22: (25.0, COURT_LENGTH_FT / 2),
}

#: The dataset's own left-right mirror permutation, kept because it is the one
#: external fact that can check a schema without trusting any registration.
FLIP_INDEX = [35, 36, 2, 38, 39, 5, 41, 42, 34, 32, 10, 33, 29, 30, 31, 15,
              26, 27, 28, 19, 20, 21, 22, 23, 24, 25, 16, 17, 18, 12, 13, 14,
              9, 11, 8, 0, 1, 37, 3, 4, 40, 6, 7, 43, 44, 45, 46, 47]

# Four points determine a homography exactly, and an exact fit reports zero
# error whether or not the correspondences were right -- the trap `court.py`
# already documents. Six leaves something for RANSAC to disagree with.
MIN_KEYPOINTS = 6
#: Reprojection tolerance for RANSAC, in pixels.
RANSAC_PX = 6.0
#: Tolerance when refitting a fused registration. This fit runs pixels ->
#: court, so cv2 measures its residual in the destination units, FEET, not
#: pixels. A landmark is worth about 0.03 ft per pixel on a broadcast frame,
#: so half a foot is a loose ~17 px -- deliberately, since the medians being
#: fitted have already had their outliers removed.
FUSE_RANSAC_FT = 0.5


def homography_from_keypoints(points: dict[int, tuple[float, float]],
                              min_points: int = MIN_KEYPOINTS,
                              ransac_px: float = RANSAC_PX):
    """Image-to-court homography from detected landmarks, or None.

    `points` maps keypoint index to its pixel position. Indices absent from
    `KEYPOINTS` are ignored rather than guessed at.

    Returns `(matrix, inliers)`. None means too few landmarks, or too few that
    agree -- which a caller must treat as "no registration", never as
    "close enough", because every court coordinate downstream inherits it.
    """
    import cv2

    usable = [(i, xy) for i, xy in points.items() if i in KEYPOINTS]
    if len(usable) < min_points:
        return None, 0
    image_pts = np.array([xy for _, xy in usable], dtype=np.float32)
    court_pts = np.array([KEYPOINTS[i] for i, _ in usable], dtype=np.float32)
    matrix, mask = cv2.findHomography(image_pts, court_pts, cv2.RANSAC,
                                      ransac_px)
    if matrix is None or mask is None:
        return None, 0
    inliers = int(mask.sum())
    if inliers < min_points:
        return None, inliers
    return matrix, inliers


def symmetry_error(schema: dict[int, tuple[float, float]]) -> float:
    """Median disagreement, in feet, with the dataset's own mirror pairs.

    An independent test of any candidate schema: `flip_idx` is annotation
    metadata, not something a fit can see, so a schema that satisfies it is
    right for a reason its own optimisation never had access to. The first
    attempt at these coordinates failed this at 65 ft.
    """
    errors = []
    for a, b in ((i, FLIP_INDEX[i]) for i in schema):
        if b not in schema or a >= b:
            continue
        ax, ay = schema[a]
        bx, by = schema[b]
        errors.append(abs(ay + by - COURT_LENGTH_FT) + abs(ax - bx))
    return float(np.median(errors)) if errors else float("inf")


def fuse_registrations(matrices, carries, probe, centre=None):
    """One registration for an instant, from a whole window of frames.

    A single frame's landmarks are noisy, but the camera is not: ORB aligns
    consecutive broadcast frames to about 0.5 px, and the court is rigid. So
    every frame in a window is an independent measurement of the SAME
    registration, and they can be combined.

    Each neighbour's registration is carried onto the centre frame's pixels and
    asked where a probe point lands in court feet; the median across the window
    is taken per probe, and a homography is refitted to those medians. The
    median rather than the mean because a wrong registration is not a small
    error -- it puts the play at the other end of the floor, and one of those
    would drag an average with it.

    `matrices[i]` maps frame i's pixels to court feet, or is None where that
    frame did not register. `carries[i]` maps frame i's pixels into frame i+1's;
    None breaks the chain, and frames beyond the break are dropped rather than
    guessed through. Returns `(matrix, used)` where `used` counts the frames
    that contributed -- 1 means no fusion happened and the caller has a plain
    single-frame registration, not a fused one.
    """
    import cv2

    if centre is None:
        centre = len(matrices) // 2
    probe = np.asarray(probe, dtype=np.float32).reshape(-1, 1, 2)

    # Pixel transform from the centre frame to frame i, walking outwards and
    # stopping at the first refused hop in each direction.
    to_frame = {centre: np.eye(3, dtype=np.float64)}
    for i in range(centre, len(matrices) - 1):
        if carries[i] is None or i not in to_frame:
            break
        to_frame[i + 1] = carries[i] @ to_frame[i]
    for i in range(centre, 0, -1):
        if carries[i - 1] is None or i not in to_frame:
            break
        to_frame[i - 1] = np.linalg.inv(carries[i - 1]) @ to_frame[i]

    estimates = []
    for i, matrix in enumerate(matrices):
        if matrix is None or i not in to_frame:
            continue
        moved = cv2.perspectiveTransform(probe, to_frame[i])
        estimates.append(cv2.perspectiveTransform(moved, matrix).reshape(-1, 2))
    if not estimates:
        return None, 0
    if len(estimates) == 1:
        return matrices[centre], 1

    court = np.median(np.stack(estimates), axis=0)
    fused, _ = cv2.findHomography(probe.reshape(-1, 2), court, cv2.RANSAC,
                                  FUSE_RANSAC_FT)
    if fused is None:
        return None, 0
    return fused, len(estimates)
