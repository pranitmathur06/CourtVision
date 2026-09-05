"""Carry a court registration across frames the line search cannot solve alone.

`court_lines.search_camera` solves each frame from scratch against its own line
mask, so a frame showing little court -- a drive into the paint, a zoom, bodies
over the stripes -- simply fails, and the registration is lost until the lines
come back.

That throws away the one thing a broadcast guarantees. The main camera is a
fixed-position pan-tilt-zoom rig, and for a camera that only rotates and zooms
the image-to-image homography is EXACT for every scene point, whatever its
depth. Independently, the court is a plane, so the court's own mapping between
two views is a homography even if the rig does translate. Both arguments land
in the same place: consecutive frames of one continuous shot are related by a
homography that can be recovered from background texture alone, with no court
line visible anywhere.

So a frame does not need to solve itself. It needs one solved frame somewhere
in its shot, and a chain of pairwise homographies back to it.

Two things this must not do. It must never chain across a camera cut, where the
relationship is not a homography at all -- `shot_boundaries.cut_frames` supplies
those. And it must not chain indefinitely: each hop multiplies its error into
every later hop, so drift is bounded by re-anchoring whenever the line search
succeeds and by refusing chains longer than `MAX_CHAIN`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Pairwise fits below this many RANSAC inliers are refused. A homography needs
# four correspondences; accepting anything near that floor lets a fit driven by
# noise pass, and a bad hop corrupts every frame after it.
MIN_INLIERS = 25
# Hops allowed away from a registered frame. Error compounds multiplicatively,
# so this is the difference between filling a gap and inventing coordinates.
MAX_CHAIN = 45
# ORB features per frame. The background here is crowd and hardwood texture,
# which is plentiful; the cost of more features is matching time, not accuracy.
ORB_FEATURES = 3000
# Lowe ratio for the two-nearest-neighbour test.
RATIO = 0.75
# Players are the only large moving objects in a shot. Their texture matches
# between frames just as well as the court's does, but it moves INDEPENDENTLY
# of the camera, so it argues for a homography the camera never underwent.
BOX_PAD_PX = 12


def _mask_without(shape: tuple[int, int],
                  boxes: np.ndarray | None) -> np.ndarray | None:
    """A feature mask that excludes player boxes, padded outward."""
    if boxes is None or len(boxes) == 0:
        return None
    mask = np.full(shape[:2], 255, dtype=np.uint8)
    height, width = shape[:2]
    for x1, y1, x2, y2 in np.asarray(boxes)[:, :4]:
        a = max(0, int(x1) - BOX_PAD_PX)
        b = max(0, int(y1) - BOX_PAD_PX)
        c = min(width, int(x2) + BOX_PAD_PX)
        d = min(height, int(y2) + BOX_PAD_PX)
        if c > a and d > b:
            mask[b:d, a:c] = 0
    return mask


def pairwise_homography(source: np.ndarray, target: np.ndarray,
                        source_boxes: np.ndarray | None = None,
                        target_boxes: np.ndarray | None = None,
                        min_inliers: int = MIN_INLIERS) -> np.ndarray | None:
    """Homography mapping SOURCE image coordinates into TARGET image coordinates.

    Returns None when the evidence is too weak to trust, which is the point: a
    refused hop leaves a frame unregistered, while a bad hop silently misplaces
    every player in it and every frame downstream.
    """
    import cv2

    grey_source = (source if source.ndim == 2
                   else cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))
    grey_target = (target if target.ndim == 2
                   else cv2.cvtColor(target, cv2.COLOR_BGR2GRAY))
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    keys_source, desc_source = orb.detectAndCompute(
        grey_source, _mask_without(grey_source.shape, source_boxes))
    keys_target, desc_target = orb.detectAndCompute(
        grey_target, _mask_without(grey_target.shape, target_boxes))
    if desc_source is None or desc_target is None:
        return None
    if len(keys_source) < min_inliers or len(keys_target) < min_inliers:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_source, desc_target, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < RATIO * n.distance]
    if len(good) < min_inliers:
        return None

    src = np.float32([keys_source[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([keys_target[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or inliers is None or int(inliers.sum()) < min_inliers:
        return None
    return matrix


def segment_of(index: int, cuts: Sequence[int], count: int) -> tuple[int, int]:
    """The [start, end) run of frames sharing a continuous camera with `index`."""
    start = 0
    for cut in sorted(cuts):
        if cut > index:
            return start, cut
        start = cut
    return start, count


def propagate(images: Sequence[np.ndarray],
              solved: dict[int, np.ndarray],
              cuts: Sequence[int] = (),
              boxes: dict[int, np.ndarray] | None = None,
              max_chain: int = MAX_CHAIN,
              verify=None) -> dict[int, np.ndarray]:
    """Extend `solved` court-from-image matrices to neighbouring frames.

    `solved` maps frame index -> the 3x3 taking IMAGE coordinates to COURT
    coordinates. Returns a new dict containing the originals plus every frame
    reached by a chain of pairwise homographies that stayed inside one camera
    segment and inside `max_chain` hops.

    `verify(index, matrix) -> bool` is checked at every hop and the chain stops
    the moment it fails. Pass one. `cuts` is a weak guard on real footage: on a
    120-second broadcast window the thumbnail-difference detector fired on
    nothing at all, and the difference measured AT true camera changes (median
    0.028) was actually LOWER than during ordinary play (0.031) -- at a third of
    a second between samples a cut is indistinguishable from a fast pan. So the
    cut list cannot be trusted to bound a chain, and evidence at each frame has
    to do that job instead.
    """
    if not solved:
        return {}
    boxes = boxes or {}
    out = dict(solved)
    count = len(images)

    for anchor in sorted(solved):
        start, end = segment_of(anchor, cuts, count)
        for step in (1, -1):
            # court_from_image for the anchor, then updated to hold the matrix
            # that carries the CURRENT frame's pixels into court coordinates.
            carried = solved[anchor]
            index = anchor
            for _ in range(max_chain):
                nxt = index + step
                if nxt < start or nxt >= end or nxt < 0 or nxt >= count:
                    break
                if nxt in solved:
                    break               # a real solution beats a propagated one
                hop = pairwise_homography(images[nxt], images[index],
                                          boxes.get(nxt), boxes.get(index))
                if hop is None:
                    break
                carried = carried @ hop
                if not np.isfinite(carried).all():
                    break
                if verify is not None and not verify(nxt, carried):
                    break
                # A later anchor may already have reached this frame in fewer
                # hops; keep whichever chain is shorter by visiting in order.
                out.setdefault(nxt, carried)
                index = nxt
    return out


# ---------------------------------------------------------------------------
# The rig does not move.
#
# `search_camera` optimises six parameters per frame: where the camera IS
# (cx, cy, cz), where it is aimed (tx, ty) and its focal length. But a
# broadcast main camera sits bolted to one position all night. Its position is
# a property of the ARENA, not of the frame, and re-deriving it 90,000 times is
# not merely wasted work -- it is three extra dimensions of freedom in which a
# frame with few visible lines can find a wrong answer that scores well.
#
# Solve the rig once from the frames that register confidently, then give every
# remaining frame a three-parameter problem: pan, tilt, zoom.

# Half-width of the box left around each solved rig coordinate. Not zero:
# differential evolution needs a non-degenerate interval, and a foot of slack
# absorbs the spread between individually-solved frames without reopening the
# search.
RIG_SLACK_FT = 1.0
# Focal length is the one intrinsic that genuinely varies -- the operator zooms
# constantly -- so it keeps its full range.
RIG_MIN_FRAMES = 5


def estimate_rig(params: Sequence[np.ndarray],
                 scores: Sequence[float],
                 min_score: float = 0.45) -> np.ndarray | None:
    """Camera position (cx, cy, cz) shared by every frame of one camera.

    Takes the MEDIAN over confidently-solved frames rather than the best single
    one. The best-scoring frame is still one sample of a noisy objective, and a
    rig fixed to its error would push that error into every frame afterwards.
    """
    pairs = [(float(s), np.asarray(p, dtype=float)[:3])
             for p, s in zip(params, scores) if p is not None]
    if len(pairs) < RIG_MIN_FRAMES:
        return None
    good = [p for s, p in pairs if s >= min_score]
    if len(good) < RIG_MIN_FRAMES:
        # `min_score` is a guess about a noisy objective, not a law. On real
        # broadcast the whole distribution can sit below it while the solutions
        # are still consistent -- an early run scored 20 of 23 frames between
        # 0.30 and 0.45 and refused to fix the rig at all. Fall back to the
        # best available frames; the median still rejects the outliers, and the
        # caller can see the spread to judge whether the rig is real.
        pairs.sort(key=lambda x: -x[0])
        good = [p for _, p in pairs[:max(RIG_MIN_FRAMES, len(pairs) // 3)]]
    return np.median(np.vstack(good), axis=0)


def rig_bounds(rig: np.ndarray,
               base: Sequence[tuple[float, float]] | None = None,
               slack: float = RIG_SLACK_FT) -> list[tuple[float, float]]:
    """Search bounds that pin the camera position and free only pan/tilt/zoom."""
    from courtvision.court_lines import DEFAULT_CAMERA_BOUNDS

    base = list(base or DEFAULT_CAMERA_BOUNDS)
    out: list[tuple[float, float]] = []
    for axis in range(3):
        lo, hi = base[axis]
        centre = float(np.clip(rig[axis], lo, hi))
        out.append((max(lo, centre - slack), min(hi, centre + slack)))
    out.extend(base[3:])
    return out


# ---------------------------------------------------------------------------
# Refusing frames that contain no court.
#
# Roughly 40% of a broadcast is not the floor at all: crowd reactions, bench
# close-ups, replays, graphics. No homography exists for those frames, so
# "coverage over all frames" was never the right denominator.
#
# Worse, the line-alignment score cannot reject them on its own. A close-up of
# a spectator's face scored 0.302 against a 0.30 accept threshold -- dark
# clothing and seat edges make line-like structure anywhere, and with almost no
# court visible the model only has to explain a handful of pixels. Accepting
# that frame does not merely waste a search; it places players on a court that
# is not in the picture, which is worse than reporting nothing.
#
# Hardwood is the signal the score lacks. Measured on frames classified by eye:
#
#     court wide shots   0.224 - 0.343
#     tight close-up     0.179
#     under-basket       0.161
#     crowd              0.012
#     a fan's face       0.036

WOOD_HUE = (5, 30)          # warm, in OpenCV's 0-180 hue scale
WOOD_MIN_SATURATION = 40
WOOD_MIN_VALUE = 90
MIN_WOOD_FRACTION = 0.20


def wood_fraction(image: np.ndarray) -> float:
    """Share of the frame that looks like a hardwood floor."""
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    wood = ((hue >= WOOD_HUE[0]) & (hue <= WOOD_HUE[1])
            & (saturation >= WOOD_MIN_SATURATION) & (value >= WOOD_MIN_VALUE))
    return float(wood.mean())


def has_court(image: np.ndarray,
              min_fraction: float = MIN_WOOD_FRACTION) -> bool:
    """Whether this frame shows enough floor for a registration to mean anything.

    Deliberately a gate BEFORE the search rather than a check after it: the
    search costs ~15 s a frame and this costs a millisecond, and a frame with no
    court must be refused even when the search reports a confident score.
    """
    return wood_fraction(image) >= min_fraction


# ---------------------------------------------------------------------------
# Absolute position, for frames that only have a plane homography.
#
# The rim is the strongest absolute anchor -- its 3D position is known exactly
# -- but projecting it needs the full camera, because at 10 ft it is off the
# court plane. A propagated frame carries only the 2D court homography, so the
# rim cannot gate it.
#
# What CAN gate it is the players. A court that has slid sideways still explains
# the lines, and still scores well; it cannot put ten basketball players inside
# a 94 x 50 ft rectangle. Measured on registrations that all passed the line
# gate at 0.346-0.373 median: 17% of players landed off the court entirely, with
# x reaching -27 and 76 ft on a floor that is 0 to 50.

COURT_MARGIN_FT = 15.0      # benches, photographers and inbounders stand off it
MAX_OFF_COURT_SHARE = 0.25


def players_on_court(court_to_image_inverse: np.ndarray,
                     feet_px: np.ndarray,
                     margin_ft: float = COURT_MARGIN_FT) -> float:
    """Share of detected feet that land on the floor under this homography.

    `feet_px` is (N, 2) image coordinates of the bottom-centre of each player
    box. Returns 0.0 when nothing projects, which callers must treat as a
    failure rather than as "no evidence against".
    """
    if feet_px is None or len(feet_px) == 0:
        return 0.0
    homogeneous = np.hstack([np.asarray(feet_px, dtype=float),
                             np.ones((len(feet_px), 1))])
    projected = homogeneous @ court_to_image_inverse.T
    w = projected[:, 2]
    valid = np.abs(w) > 1e-9
    if not valid.any():
        return 0.0
    court = projected[valid, :2] / w[valid, None]
    from courtvision.court_lines import COURT_LENGTH_FT
    from courtvision.court import COURT_WIDTH

    inside = ((court[:, 0] >= -margin_ft)
              & (court[:, 0] <= COURT_WIDTH + margin_ft)
              & (court[:, 1] >= -margin_ft)
              & (court[:, 1] <= COURT_LENGTH_FT + margin_ft))
    return float(inside.mean())


def plausible_positions(court_to_image_inverse: np.ndarray,
                        feet_px: np.ndarray,
                        max_off_share: float = MAX_OFF_COURT_SHARE) -> bool:
    """Whether a homography places players somewhere a player can stand."""
    return players_on_court(court_to_image_inverse, feet_px) >= 1.0 - max_off_share


# ---------------------------------------------------------------------------
# Which basket is this?
#
# A basketball court is symmetric. The far lane is a 16 ft lane with a
# free-throw circle and so is the near one, so a half-court model can explain
# the FAR half's lines perfectly while sitting 47 ft from where it belongs.
# Every line lands and the score is high; the court is simply in the wrong
# place. The line score cannot see this, by construction.
#
# Measured on every registration this pipeline produced for one broadcast --
# three separate runs, 44+ frames each carrying an independent rim detection:
#
#     projected basket-floor vs detected rim, horizontal
#         within 120 px    0.0%
#         beyond 400 px  100.0%
#         p10 -887 px, p50 +892 px, p90 +1038 px
#
# Bimodal at roughly plus and minus 900 px on a 1280 px frame: the two-basket
# signature. The vertical offset was a consistent ~+195 px, about right for a
# rim 10 ft above its floor point, so scale and tilt were fine. The fits were
# translated onto the wrong basket, not wrong in general.
#
# The rim resolves it. Its position is known exactly and a detector finds it
# independently, so it is the one piece of evidence that says WHICH basket --
# used as an accept/reject test, never blended into the objective, where it
# merely trades against line score (measured: anchors 47.8% -> 8.7%).

MAX_BASKET_OFFSET_PX = 150.0


def basket_offset_px(court_to_image: np.ndarray,
                     rim_px: tuple[float, float]) -> float | None:
    """Horizontal pixels between the model's basket and the detected rim.

    Compares the FLOOR point under the basket with the rim itself, so a
    vertical difference is expected -- the rim is 10 ft up. Horizontal
    agreement is what identifies the basket.
    """
    from courtvision.court import BASKET

    projected = court_to_image @ np.array([BASKET[0], BASKET[1], 1.0])
    if abs(projected[2]) < 1e-9 or not np.isfinite(projected).all():
        return None
    return float(projected[0] / projected[2] - rim_px[0])


def right_basket(court_to_image: np.ndarray,
                 rim_px: tuple[float, float] | None,
                 tolerance_px: float = MAX_BASKET_OFFSET_PX) -> bool:
    """Whether this fit is on the basket the detector actually sees.

    With no rim detected the question cannot be answered, and the honest answer
    is False: a frame whose basket is unverifiable should not be trusted with
    player coordinates.
    """
    if rim_px is None:
        return False
    offset = basket_offset_px(court_to_image, rim_px)
    return offset is not None and abs(offset) <= tolerance_px


# A court has 180-degree rotational symmetry about centre court. That symmetry
# is exactly why the fitter cannot tell the two halves apart -- and it is also
# the repair, because a fit that landed on the far half already describes the
# camera correctly. Only the LABELLING of court coordinates is rotated.
#
# Measured on fits that were 100% on the wrong basket:
#
#     run             as fitted                 rotated 180
#     reg6      941 px,   0.0% within 150   297 px,  29.5%
#     reg7      857 px,   0.0%               97 px, 100.0%
#     first     825 px,   0.0%               79 px,  63.6%
#
# No new search: the rotation is a 3x3, and the rim says which of the two
# orientations is the real one.


def court_rotation() -> np.ndarray:
    """(x, y) -> (COURT_WIDTH - x, COURT_LENGTH - y), as a 3x3 on court feet."""
    from courtvision.court import COURT_WIDTH
    from courtvision.court_lines import COURT_LENGTH_FT

    return np.array([[-1.0, 0.0, COURT_WIDTH],
                     [0.0, -1.0, COURT_LENGTH_FT],
                     [0.0, 0.0, 1.0]])


def orient_to_rim(court_from_image: np.ndarray,
                  rim_px: tuple[float, float] | None,
                  tolerance_px: float = MAX_BASKET_OFFSET_PX
                  ) -> np.ndarray | None:
    """Return `court_from_image` in the orientation the detected rim agrees with.

    Tries the fit as-is and rotated 180 degrees, and returns whichever puts the
    model's basket nearer the rim -- or None when neither is within tolerance,
    because a fit whose basket cannot be identified must not be trusted with
    player coordinates.
    """
    if rim_px is None:
        return None
    rotation = court_rotation()
    candidates = []
    for matrix in (court_from_image, rotation @ court_from_image):
        try:
            court_to_image = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            continue
        offset = basket_offset_px(court_to_image, rim_px)
        if offset is not None:
            candidates.append((abs(offset), matrix))
    if not candidates:
        return None
    offset, matrix = min(candidates, key=lambda pair: pair[0])
    return matrix if offset <= tolerance_px else None
