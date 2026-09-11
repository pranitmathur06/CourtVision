"""Sub-foot court registration: align every painted line, not twelve points.

The landmark model places about twelve points per frame, each to roughly
35 px, and a homography through them lands 1.62 ft from the annotators' court
on held-out games. That error is set by how precisely a landmark can be
localised -- the corner of a lane is a blob a network has to point at -- and
more epochs and longer fusion windows were both measured not to move it.

A painted line is a far better measurement than a point. It is a thin,
high-contrast ridge whose centre can be found to a fraction of a pixel
wherever it is visible, and an NBA floor carries hundreds of feet of them.
Every sample along every visible line is an independent constraint, so the fit
is driven by hundreds of sub-pixel measurements instead of a dozen coarse ones.

It is also arena-agnostic by construction. Every NBA floor carries identical
lines in identical places; only paint colour, logos and wood vary. The landmark
model has to generalise across appearance, but this stage only needs a start
inside its capture range, and then the geometry takes over.

## How this can fail, and what guards against it

This is the family of method that failed here before: an ICP refinement whose
line score improved 7.60 -> 1.30 ft while true error rose 10.2 -> 16.5. It
matched lines to whatever edges were nearest -- logos, limbs, the crowd -- and
optimised the match rather than the truth. The differences are specific:

- **Point-to-line, not point-to-point.** A sample may slide along its line;
  only its perpendicular offset is a residual, so nothing is asked of a sample
  that it cannot measure.
- **Orientation and continuity.** A candidate must run the way the projected
  line runs (structure tensor, within MAX_ANGLE_DEG) and stay bright a few
  pixels along it. A limb crossing the line, a logo stroke at another angle, or
  a blob fails.
- **Coarse to fine.** The search window halves every pass, so by the last one a
  neighbouring line is out of reach.
- **Robust loss and a weak prior.** Residuals go through soft-L1, and the
  landmark fit enters as a prior at its measured noise, so directions the
  visible lines cannot observe -- sliding along parallel lines -- stay where the
  landmarks put them rather than drifting.
- **A drift guard.** A refinement that moves the visible floor further than the
  landmark fit's own error could explain is refused, not returned.
- **Measured on lines it did not use.** `held_out_offsets` asks where a line's
  paint lies, in court feet, after a refinement that never saw that line.
  Nothing in that number comes from annotations or was optimised.
"""

from __future__ import annotations

import numpy as np

from .court_lines import court_lines

#: Search half-widths in pixels, one per pass. The first must exceed the
#: landmark fit's error (1.62 ft is ~50 px on the near side of a broadcast
#: frame); the last is small enough that no neighbouring line can be reached.
SEARCH_PX = (48, 24, 12, 6, 3)
#: In wide passes every straight segment chooses its paint as a UNIT.
#:
#: Parallel lines alias: the sideline and the corner-three line are 3 ft apart,
#: and with each sample free to take the brightest ridge in its own window, part
#: of a sideline settled on the corner line -- on broadcast, 6 ft starts landed
#: 3.4 ft from the landmark-started fit in 17 of 36 cases. A whole segment is
#: harder to fool: shifted onto its neighbour, a 94 ft sideline lines up only
#: along the 14 ft the corner line exists, so the profile averaged along the
#: segment still peaks on the true line.
#:
#: Aligning on curves first was tried and failed. Curves are not isolated: the
#: lane lines run 2 ft outside the free-throw circle and the free-throw line is
#: its diameter, with orientations matching the circle's at exactly those
#: points, so a wide window on a curve reaches straight lines anyway.
CONSENSUS_PX = 12
CONSENSUS_MIN_SAMPLES = 5
#: Spacing of samples along each court line.
SAMPLE_SPACING_FT = 0.5
#: A candidate ridge must run within this angle of the projected line.
MAX_ANGLE_DEG = 12.0
#: How line-like the neighbourhood must be (structure-tensor coherence).
MIN_COHERENCE = 0.3
#: Ridge peak above the profile's median, in top-hat grey levels.
MIN_CONTRAST = 6.0
#: The ridge must stay this bright, as a fraction of its peak, along the line.
MIN_CONTINUITY = 0.4
#: Fewer usable samples than this and the fit is not attempted.
MIN_SAMPLES = 30
#: Noise of a line-centre measurement, and of the landmark fit used as prior.
LINE_SIGMA_PX = 0.7
PRIOR_PX = 40.0
#: A refinement moving the visible floor further than this is refused. The
#: landmark fit's per-frame error on held-out games is p50 1.62 ft, p90 3.07;
#: allowing about two and a half times the p90 covers its tail while still
#: refusing a relock onto paint half the court away.
MAX_DRIFT_FT = 8.0
#: The final pass is iterated until the fit stops moving. Each pass observes
#: under the previous fit and then fits once, so a single final pass stops
#: short of convergence: one more plain 3 px pass took a synthetic white-lined
#: floor from 0.107 to 0.032 ft and a painted one from 0.033 to 0.026.
FINAL_ITERATIONS = 12
#: Converged when the visible line samples move less than this, median, in px.
FINAL_TOL_PX = 0.05
#: Support expansion. After the coarse-to-fine passes the fit is dominated by
#: whatever it locked onto first -- usually the key -- and paint further away
#: can sit a few pixels off its prediction. The narrow final windows cannot
#: reach it, so the evidence that would correct the far side is dropped and the
#: error persists because it is ignored. On the held-out test arenas, 83-91% of
#: annotated points more than 6 ft from used paint had visible paint 3-12 px
#: from the fit that the 3 px pass never matched (local error 0.23-0.62 ft).
#: So the fit is re-observed at EXPANSION_PX and newly found samples join it,
#: round after round, until too few new ones do.
#:
#: Only STRAIGHT segments may join: nothing protects a curve at 12 px -- a
#: free-throw circle's diameter is the key's top edge and its sides run 2 ft
#: from the lane. On a synthetic painted floor, 39% of the samples an
#: unrestricted expansion admitted sat on the wrong paint, almost all on
#: circles and restricted arcs.
#:
#: The joining samples are refitted at the same narrow robust scale as the
#: final pass. A looser one was the real cause of an expansion regression on
#: the painted floor (0.033 to 0.055 ft), established by ablation: with NO new
#: samples at all, the looser refit alone went 0.033 to 0.048, because it
#: changes which ridges are matched at 3 px. Three earlier fixes aimed at WHICH
#: samples join -- a paint-count acceptance check, straight segments only, and
#: excluding segments the fit already explained -- rested on a selection-effect
#: premise the ablation refuted; the last had no measurable effect and was
#: removed.
EXPANSION_PX = 12
EXPANSION_ROUNDS = 3
EXPANSION_MIN_NEW = 10
EXPANSION_ROBUST_PX = 0.75
#: Extra starts, in court feet, tried around the landmark registration.
#:
#: Parallel lines on a court sit as little as 3 ft apart -- the sideline and the
#: corner-three line -- and a start 6 ft off locked onto the wrong one with a
#: 0.106 px residual, 3.0 ft from the truth. Sharpness cannot see that; the
#: lock is sharp. But hypotheses on the SAME frame share the same occlusion, so
#: their absolute coverage is comparable where coverage across frames is not,
#: and the right alignment explains the most paint. Starts are spaced at that
#: 3 ft confusion distance.
#: Reduced from a 6 ft grid: on broadcast the wider starts found aliases
#: rather than the answer, and with curves anchoring the coarse stage every
#: start inside the capture range should converge to the same fit anyway.
MULTI_START_FT = ((0.0, 0.0), (3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0))
#: Is the solution a sharp fit to the paint, or merely a fit?
#:
#: A low residual proves nothing. Started 25 ft off on a synthetic court the
#: refinement locked onto the wrong paint with a 0.11 px residual; started 30 ft
#: off along the court it kept the across-court mapping exact, compressed the
#: along-court one, and scored 0.079 px over 405 samples, 28 ft from the truth.
#:
#: Raw coverage -- the share of visible line samples that find paint -- caught
#: those on synthetic frames and failed on broadcast: fans, graphics and
#: players hide much of the floor, so correct fits scored 16-35% and nearly
#: every frame was refused. Hidden samples are uninformative, not evidence of
#: misalignment. So coverage at the solution is compared with coverage after
#: moving the registration PEAK_SHIFT_FT each way, separately for each direction
#: the lines constrain. A correct fit is a sharp peak; occlusion scales the
#: solution and its neighbours alike and cancels in the ratio. A lock onto the
#: wrong lines leaves the lines pinning the wrong direction unexplained, and
#: moving along it costs nothing.
MIN_COVERAGE = 0.10
PEAK_SHIFT_FT = 2.0
#: Selected by scripts/select_refinement_threshold.py from
#: outputs/calib_okc_bright_v2.json (OKC calibration game, commit d715788,
#: made at threshold 0 with the common-sample ratio). No candidate reached a
#: 0.30 ft held-out median, so the rule's fallback took the lowest: 0.54 ft,
#: tied between 2.0 and 5.0, ties to the lower. The previous 3.0 came from the
#: same rule under the old ratio definition.
MIN_PEAK_RATIO = 2.0
#: A direction needs this many visible samples before it is judged; fewer means
#: no visible lines constrain it, and the landmark prior holds it.
MIN_GROUP_SAMPLES = 40
#: Margin around a player box inside which samples are ignored.
BOX_MARGIN_PX = 6
#: How far, in court feet, a point may lie from the paint the fit rests on and
#: still be trusted.
#:
#: A fit is only as good as its support. On the test arenas refined error was
#: 0.21 ft within 3 ft of used paint and 1.16 ft beyond 12 ft, and a fit resting
#: only on lines near one basket drew the far sideline diagonally across the
#: floor -- while its peak ratio, which judges sharpness on the paint the fit
#: used, looked as good as any correct fit's. Sharpness at the paint says
#: nothing about paint the fit never saw, so no frame-level score can certify
#: the far side. Trust is therefore per point: coordinates within this radius
#: of used line samples are asserted, the rest are flagged as extrapolated.
#: Selected by scripts/select_trust_radius.py on the OKC calibration game.
TRUST_RADIUS_FT = 6.0


def _line_samples(spacing_ft: float = SAMPLE_SPACING_FT):
    """Points along every painted line: unit tangents, line and segment ids."""
    points, tangents, ids, segments = [], [], [], []
    segment_id = 0
    for index, polyline in enumerate(court_lines()):
        p = np.asarray(polyline, dtype=np.float64)
        for a, b in zip(p[:-1], p[1:]):
            segment = b - a
            length = float(np.hypot(*segment))
            if length < 1e-9:
                continue
            count = max(1, int(length / spacing_ft))
            tangent = segment / length
            for s in (np.arange(count) + 0.5) / count:
                points.append(a + segment * s)
                tangents.append(tangent)
                ids.append(index)
                segments.append(segment_id)
            segment_id += 1
    return (np.array(points), np.array(tangents), np.array(ids, dtype=int),
            np.array(segments, dtype=int))


_POINTS, _TANGENTS, _LINE_IDS, _SEGMENTS = _line_samples()
_NORMALS = np.stack([-_TANGENTS[:, 1], _TANGENTS[:, 0]], axis=1)
_segment_ids, _segment_sizes = np.unique(_SEGMENTS, return_counts=True)
#: Samples on straight segments -- the ones segment consensus protects. Arcs
#: and circles are drawn as runs of one-sample segments and fall outside it.
_STRAIGHT = np.isin(_SEGMENTS, _segment_ids[_segment_sizes >= 5])


def _segment_consensus(profile, segments, offsets, slack_px):
    """Restrict each straight segment's samples to one shared painted line.

    Arcs are drawn as many short segments of one sample each, so this only ever
    binds genuinely straight runs.
    """
    masked = profile.copy()
    for segment in np.unique(segments):
        rows = np.flatnonzero(segments == segment)
        if len(rows) < CONSENSUS_MIN_SAMPLES:
            continue
        centre = offsets[int(profile[rows].mean(axis=0).argmax())]
        far = np.flatnonzero(np.abs(offsets - centre) > slack_px)
        masked[np.ix_(rows, far)] = -np.inf
    return masked


def _project(matrix: np.ndarray, points: np.ndarray):
    """Apply a homography; also report which points land in front of it."""
    h = np.c_[points, np.ones(len(points))] @ matrix.T
    w = h[:, 2]
    ok = w > 1e-9
    out = np.full((len(points), 2), np.nan)
    out[ok] = h[ok, :2] / w[ok, None]
    return out, ok


#: What counts as evidence of a painted line.
#:
#: "bright" finds thin ridges lighter than the floor -- white paint on wood,
#: as at OKC, where refinement was developed. "both" adds ridges darker than
#: the floor. "all" also treats a step edge between two painted regions as a
#: line.
#:
#: The unseen arena showed why all three exist. At Toyota Center the arc and
#: the circles are BLACK lines, and the lane is a solid red key with no line on
#: its boundary at all -- the lane line is only the edge between red paint and
#: wood, as are the sidelines against red out-of-bounds paint. Bright-only
#: refinement accepted 0 of 37 frames there; adding dark ridges recovered the
#: arcs but not the key. An edge is located where the paint changes, which for
#: NBA dimensions is the outer edge of the lane line, within about an inch of
#: the modelled position. At a white line the edge evidence also fires on both
#: flanks, so it is weighted below ridge evidence and the line centre still wins.
PAINT_POLARITY = "bright"
EDGE_WEIGHT = 0.5


def paint_response(image: np.ndarray, polarity: str | None = None) -> np.ndarray:
    """Evidence of painted lines, with the floor's slow shading removed."""
    import cv2

    polarity = polarity or PAINT_POLARITY
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), 1.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    bright = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kernel)
    if polarity == "bright":
        return bright
    ridges = bright + cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, kernel)
    if polarity == "both":
        return ridges
    if polarity == "all":
        # Per-pixel gradient in grey levels: a 3x3 Sobel on a unit ramp reads 8.
        gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        return ridges + EDGE_WEIGHT * np.sqrt(gx * gx + gy * gy)
    raise ValueError(f"unknown polarity {polarity!r}")


def _structure(response: np.ndarray):
    """Line-normal angle and coherence at every pixel."""
    import cv2

    gx = cv2.Sobel(response, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(response, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
    # For a ridge the gradient energy sits on its two flanks, pointing across
    # it, so the dominant gradient orientation is the line's NORMAL.
    normal = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    return normal, coherence


def _observe(response, structure, inverse, half_width, keep, boxes,
             consensus=True):
    """Find the painted ridge nearest each projected line sample.

    `inverse` maps court -> image. Returns image positions of the ridges found
    and the indices of the samples they belong to.
    """
    from scipy.ndimage import map_coordinates

    height, width = response.shape
    points = _POINTS[keep]
    tangents = _TANGENTS[keep]
    index = np.flatnonzero(keep)

    projected, front = _project(inverse, points)
    ahead, front2 = _project(inverse, points + tangents)
    valid = front & front2
    t_img = ahead - projected
    length = np.hypot(t_img[:, 0], t_img[:, 1])
    valid &= length > 1e-6
    t_img = t_img / np.where(length > 1e-6, length, 1.0)[:, None]
    n_img = np.stack([-t_img[:, 1], t_img[:, 0]], axis=1)

    margin = half_width + 8
    x, y = projected[:, 0], projected[:, 1]
    valid &= (x > margin) & (x < width - margin) & (y > margin) & (y < height - margin)
    if boxes is not None and len(boxes):
        for x1, y1, x2, y2 in np.asarray(boxes, dtype=float):
            inside = ((x >= x1 - BOX_MARGIN_PX) & (x <= x2 + BOX_MARGIN_PX)
                      & (y >= y1 - BOX_MARGIN_PX) & (y <= y2 + BOX_MARGIN_PX))
            valid &= ~inside
    if valid.sum() == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    candidates = index[valid]

    projected, t_img, n_img, index = (projected[valid], t_img[valid],
                                      n_img[valid], index[valid])
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    xs = projected[:, 0:1] + n_img[:, 0:1] * offsets
    ys = projected[:, 1:2] + n_img[:, 1:2] * offsets
    profile = map_coordinates(response, [ys.ravel(), xs.ravel()], order=1,
                              mode="constant").reshape(len(projected), -1)
    choose = profile
    if consensus and half_width >= CONSENSUS_PX:
        # Slack grows with the window: a slightly rotated start makes a long
        # segment's offset drift along its length.
        choose = _segment_consensus(profile, _SEGMENTS[index], offsets,
                                    max(4.0, half_width / 4.0))
    k = choose.argmax(axis=1)
    rows = np.arange(len(profile))
    peak = profile[rows, k]
    good = np.isfinite(choose[rows, k])
    good &= (k > 0) & (k < len(offsets) - 1)
    good &= peak - np.median(profile, axis=1) >= MIN_CONTRAST

    # Sub-pixel centre from a parabola through the peak and its neighbours.
    kk = np.clip(k, 1, len(offsets) - 2)
    y0, y1, y2 = profile[rows, kk - 1], profile[rows, kk], profile[rows, kk + 1]
    denominator = y0 - 2 * y1 + y2
    delta = np.where(denominator < 0, 0.5 * (y0 - y2) / np.where(
        denominator < 0, denominator, -1.0), 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    found = projected + n_img * (offsets[kk] + delta)[:, None]

    # Continuity: a painted line stays bright along its own direction.
    along = []
    for step in (-6.0, -3.0, 3.0, 6.0):
        p = found + t_img * step
        along.append(map_coordinates(response, [p[:, 1], p[:, 0]], order=1,
                                     mode="constant"))
    good &= np.mean(along, axis=0) >= MIN_CONTINUITY * np.maximum(peak, 1e-6)

    # Orientation: the ridge must run the way the projected line runs.
    normal, coherence = structure
    ix = np.clip(np.round(found[:, 0]).astype(int), 0, width - 1)
    iy = np.clip(np.round(found[:, 1]).astype(int), 0, height - 1)
    expected = np.arctan2(n_img[:, 1], n_img[:, 0])
    difference = np.abs(((normal[iy, ix] - expected) + np.pi / 2) % np.pi - np.pi / 2)
    good &= difference <= np.deg2rad(MAX_ANGLE_DEG)
    good &= coherence[iy, ix] >= MIN_COHERENCE

    return found[good], index[good], candidates


_NORMALISE = np.array([[1 / 50.0, 0, -25 / 50.0], [0, 1 / 50.0, -47 / 50.0],
                       [0, 0, 1.0]])
_DENORMALISE = np.linalg.inv(_NORMALISE)


def _compose(inverse0, params):
    """court -> image, as a perturbation of the starting one in normalised feet."""
    d = np.append(params, 0.0).reshape(3, 3)
    return inverse0 @ _DENORMALISE @ (np.eye(3) + d) @ _NORMALISE


def _fit(inverse0, observed, index, prior_points, robust_px, prior_inverse=None):
    """Least-squares court -> image homography, point-to-line plus a weak prior.

    The fit is parameterised around `inverse0`, where the search started, but
    the prior pulls toward `prior_inverse` -- the landmark registration, which
    is the actual evidence. A displaced start is a place to look from, not a
    belief about where the court is.
    """
    from scipy.optimize import least_squares

    points = _POINTS[index]
    ahead = points + _TANGENTS[index]
    prior_image, _ = _project(inverse0 if prior_inverse is None else prior_inverse,
                              prior_points)

    def residuals(params):
        inverse = _compose(inverse0, params)
        a, _ = _project(inverse, points)
        b, _ = _project(inverse, ahead)
        u = b - a
        u /= np.maximum(np.hypot(u[:, 0], u[:, 1]), 1e-9)[:, None]
        # Signed perpendicular distance, in pixels, from the paint to the line.
        line = (u[:, 0] * (observed[:, 1] - a[:, 1])
                - u[:, 1] * (observed[:, 0] - a[:, 0])) / LINE_SIGMA_PX
        moved, _ = _project(inverse, prior_points)
        prior = ((moved - prior_image) / PRIOR_PX).ravel()
        return np.nan_to_num(np.concatenate([line, prior]), nan=1e3,
                             posinf=1e3, neginf=-1e3)

    solution = least_squares(residuals, np.zeros(8), loss="soft_l1",
                             f_scale=max(robust_px / LINE_SIGMA_PX, 1.0),
                             x_scale="jac", max_nfev=200)
    inverse = _compose(inverse0, solution.x)
    a, _ = _project(inverse, points)
    b, _ = _project(inverse, ahead)
    u = b - a
    u /= np.maximum(np.hypot(u[:, 0], u[:, 1]), 1e-9)[:, None]
    final = np.abs(u[:, 0] * (observed[:, 1] - a[:, 1])
                   - u[:, 1] * (observed[:, 0] - a[:, 0]))
    return inverse, final


def prepare(image):
    """Paint response and structure tensor, computed once per frame.

    Measuring accuracy means refining a frame once per held-out line family,
    and these two images do not depend on which lines are held out.
    """
    response = paint_response(image)
    return response, _structure(response)


def sample_normals(index):
    """Court-space unit normals of the given line samples."""
    tangents = _TANGENTS[np.asarray(index, dtype=int)]
    return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)


def _hits(response, structure, matrix, keep, boxes, window):
    """Indices of the line samples that find paint, and of those that could."""
    _, found, visible = _observe(response, structure, np.linalg.inv(matrix),
                                 window, keep, boxes)
    return set(found.tolist()), set(visible.tolist())


def _peak_sharpness(response, structure, matrix, keep, boxes, window):
    """Worst coverage, and worst solution-to-neighbour ratio, over directions.

    The solution and each 2 ft neighbour are compared on the SAME samples --
    those usable under both. Recomputing visibility for the neighbour alone let
    a shift that pushed a group's samples off the usable frame, or into player
    boxes, collapse the neighbour's denominator: with nothing visible its
    smoothed rate reached 1.0, and a correct, fully covered fit read as not
    sharp and was refused. A shift that leaves too few common samples is
    uninformative, so it is skipped rather than counted against the fit.
    """
    found, visible = _hits(response, structure, matrix, keep, boxes, window)
    worst_coverage, coverage_judged = 1.0, False
    worst_ratio, ratio_judged = np.inf, False
    for axis in (0, 1):
        group = {i for i in visible if abs(_NORMALS[i, axis]) >= 0.7}
        if len(group) < MIN_GROUP_SAMPLES:
            continue
        worst_coverage = min(worst_coverage, len(found & group) / len(group))
        coverage_judged = True
        for sign in (-1.0, 1.0):
            move = np.eye(3)
            move[axis, 2] = sign * PEAK_SHIFT_FT
            n_found, n_visible = _hits(response, structure, move @ matrix, keep,
                                       boxes, window)
            common = group & n_visible
            if len(common) < MIN_GROUP_SAMPLES:
                continue
            # Add-one smoothing on counts over the same samples: bounded by what
            # the sample count supports, unlike the 1e-3 floor it replaced.
            ratio = (len(found & common) + 1) / (len(n_found & common) + 1)
            worst_ratio = min(worst_ratio, ratio)
            ratio_judged = True
    return {"coverage": float(worst_coverage) if coverage_judged else 0.0,
            "ratio": float(worst_ratio) if ratio_judged else 0.0}


def _refine_from(response, structure, start, prior, keep, boxes, passes):
    """One coarse-to-fine refinement from `start`; None if it runs out of paint."""
    base = np.linalg.inv(start)
    prior_inverse = np.linalg.inv(prior)
    inverse = base.copy()
    observed = np.zeros((0, 2))
    index = np.zeros(0, dtype=int)
    residual = np.zeros(0)
    for half_width in passes:
        observed, index, _ = _observe(response, structure, inverse, half_width,
                                      keep, boxes)
        if len(index) < MIN_SAMPLES:
            return None, f"only {len(index)} line samples at {half_width} px"
        visible = _POINTS[index]
        lo, hi = visible.min(axis=0), visible.max(axis=0)
        prior_points = np.array([[x, y] for x in np.linspace(lo[0], hi[0], 3)
                                 for y in np.linspace(lo[1], hi[1], 3)])
        inverse, residual = _fit(base, observed, index, prior_points,
                                 robust_px=max(half_width / 4.0, LINE_SIGMA_PX),
                                 prior_inverse=prior_inverse)
    for _ in range(FINAL_ITERATIONS - 1):
        more_obs, more_idx, _ = _observe(response, structure, inverse, passes[-1],
                                         keep, boxes)
        if len(more_idx) < MIN_SAMPLES:
            break
        visible = _POINTS[more_idx]
        lo, hi = visible.min(axis=0), visible.max(axis=0)
        prior_points = np.array([[x, y] for x in np.linspace(lo[0], hi[0], 3)
                                 for y in np.linspace(lo[1], hi[1], 3)])
        moved_inverse, moved_residual = _fit(
            base, more_obs, more_idx, prior_points,
            robust_px=max(passes[-1] / 4.0, LINE_SIGMA_PX), prior_inverse=prior_inverse)
        before_px, _ = _project(inverse, _POINTS[more_idx])
        after_px, _ = _project(moved_inverse, _POINTS[more_idx])
        observed, index = more_obs, more_idx
        inverse, residual = moved_inverse, moved_residual
        if np.median(np.hypot(*(after_px - before_px).T)) < FINAL_TOL_PX:
            break
    for _ in range(EXPANSION_ROUNDS):
        # A round is kept only if the refit then sits on MORE paint at the
        # narrowest window; otherwise it is reverted and expansion stops. This
        # is a cheap self-consistency guard. It did not catch the painted-floor
        # regression -- that round found more paint at 3 px and was still
        # worse; the cause was the refit's robust scale, above.
        before = (inverse, observed, index, residual)
        near_obs, near_idx, _ = _observe(response, structure, inverse, passes[-1],
                                         keep, boxes)
        wide_obs, wide_idx, _ = _observe(response, structure, inverse,
                                         EXPANSION_PX, keep & _STRAIGHT, boxes)
        new = ~np.isin(wide_idx, near_idx)
        if new.sum() < EXPANSION_MIN_NEW or len(near_idx) < MIN_SAMPLES:
            break
        union_obs = np.vstack([near_obs, wide_obs[new]])
        union_idx = np.concatenate([near_idx, wide_idx[new]])
        visible = _POINTS[union_idx]
        lo, hi = visible.min(axis=0), visible.max(axis=0)
        prior_points = np.array([[x, y] for x in np.linspace(lo[0], hi[0], 3)
                                 for y in np.linspace(lo[1], hi[1], 3)])
        inverse, residual = _fit(base, union_obs, union_idx, prior_points,
                                 robust_px=EXPANSION_ROBUST_PX,
                                 prior_inverse=prior_inverse)
        # Close on a narrow pass, so what is reported -- samples, residual,
        # drift -- describes paint the final fit actually sits on.
        final_obs, final_idx, _ = _observe(response, structure, inverse,
                                           passes[-1], keep, boxes)
        if len(final_idx) < MIN_SAMPLES:
            inverse, observed, index, residual = before
            break
        observed, index = final_obs, final_idx
        inverse, residual = _fit(base, observed, index, prior_points,
                                 robust_px=max(passes[-1] / 4.0, LINE_SIGMA_PX),
                                 prior_inverse=prior_inverse)
        _, settled, _ = _observe(response, structure, inverse, passes[-1], keep, boxes)
        if len(settled) <= len(near_idx):
            inverse, observed, index, residual = before
            break
    refined = np.linalg.inv(inverse)
    refined /= refined[2, 2]
    return (refined, observed, index, residual), ""


def _choose(candidates):
    """The hypothesis explaining the most paint, judged on common ground.

    `candidates` are (found, visible, outcome) with index sets. Shares are
    compared over the samples EVERY hypothesis can see. Comparing each one's
    own share, with a floor at 70% of the best-seeing one's visibility, could
    disqualify the right hypothesis purely because a wrong basin pulled more
    court into frame. The floor survives only as a fallback, when the
    hypotheses share too little court to compare on.
    """
    if not candidates:
        return None, -1.0
    common = set.intersection(*(visible for _, visible, _ in candidates))
    if len(common) >= MIN_SAMPLES:
        scored = [(len(found & common) / len(common), outcome)
                  for found, _, outcome in candidates]
    else:
        most = max(len(visible) for _, visible, _ in candidates)
        scored = [(len(found) / max(len(visible), 1), outcome)
                  for found, visible, outcome in candidates
                  if len(visible) >= MIN_RELATIVE_VISIBLE * most]
    best, best_score = None, -1.0
    for score, outcome in scored:          # ties keep the earlier start
        if score > best_score:
            best, best_score = outcome, score
    return best, best_score


#: A hypothesis must see at least this share of the court the best-seeing one
#: does before its explained share is compared. Otherwise one that pushes most
#: of the court out of frame competes on a smaller denominator.
MIN_RELATIVE_VISIBLE = 0.7


def refine(image, matrix, boxes=None, exclude_lines=(), passes=SEARCH_PX,
           prepared=None, starts=MULTI_START_FT):
    """Refine an image -> court homography by aligning every visible line.

    `matrix` is the starting registration (from the landmark model). `boxes`
    are player boxes in pixels; samples inside them are ignored, because a
    white jersey edge beside a line is exactly the clutter that misled ICP.
    `exclude_lines` holds court-line ids out of the fit, which is how the
    result is measured without circularity.

    Returns `(matrix, info)`. When `info["refined"]` is False the starting
    matrix comes back unchanged -- a refusal is never dressed as a result.
    """
    response, structure = prepared if prepared is not None else prepare(image)
    keep = ~np.isin(_LINE_IDS, list(exclude_lines))
    info = {"refined": False, "samples": 0, "residual_px": None,
            "drift_ft": None, "coverage": None, "peak_ratio": None,
            "explained": None, "reason": "", "support": None}

    candidates, last_reason = [], ""
    for dx, dy in starts:
        move = np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])
        outcome, reason = _refine_from(response, structure, move @ matrix,
                                       matrix, keep, boxes, passes)
        if outcome is None:
            last_reason = reason
            continue
        found, visible = _hits(response, structure, outcome[0], keep, boxes,
                               passes[-1])
        candidates.append((found, visible, outcome))
    best, best_score = _choose(candidates)
    if best is None:
        info["reason"] = last_reason
        return matrix, info
    refined, observed, index, residual = best

    # How far did the visible floor move from the LANDMARK registration?
    # Beyond what that fit's own error could explain means the lines were
    # matched to the wrong paint.
    before, _ = _project(matrix, observed)
    after, _ = _project(refined, observed)
    drift = float(np.median(np.hypot(*(after - before).T)))
    peak = _peak_sharpness(response, structure, refined, keep, boxes,
                           passes[-1])
    info.update(samples=int(len(index)), support=np.asarray(index, dtype=int),
                residual_px=float(np.median(residual)), drift_ft=drift,
                coverage=peak["coverage"], peak_ratio=peak["ratio"],
                explained=float(best_score))
    if drift > MAX_DRIFT_FT:
        info["reason"] = f"moved the floor {drift:.1f} ft; refused"
        return matrix, info
    if peak["coverage"] < MIN_COVERAGE:
        info["reason"] = f"paint under only {peak['coverage']:.0%} of the lines"
        return matrix, info
    if peak["ratio"] < MIN_PEAK_RATIO:
        info["reason"] = (f"not a sharp fit: {PEAK_SHIFT_FT:.0f} ft away "
                          f"explains nearly as much (ratio {peak['ratio']:.1f})")
        return matrix, info
    info["refined"] = True
    return refined, info


def support_distance(support, court_points):
    """Court feet from each point to the nearest line sample a fit rests on.

    `support` is `info["support"]` from `refine`: the samples found at the
    final, narrowest window under the returned fit. Infinite when there is none.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(court_points, dtype=np.float64).reshape(-1, 2)
    if support is None or len(support) == 0:
        return np.full(len(points), np.inf)
    distance, _ = cKDTree(_POINTS[np.asarray(support, dtype=int)]).query(points)
    return distance


def trusted(info, court_points, radius=None):
    """Which court points a registration may assert, by `TRUST_RADIUS_FT`.

    Nothing is trusted from a refused refinement: its matrix is the landmark
    fit, whose error (~1.6 ft) is not what this module certifies.
    """
    points = np.asarray(court_points, dtype=np.float64).reshape(-1, 2)
    if not info.get("refined") or info.get("support") is None:
        return np.zeros(len(points), dtype=bool)
    limit = TRUST_RADIUS_FT if radius is None else radius
    return support_distance(info["support"], points) <= limit


def family_samples(lines):
    """Line-sample indices of the given court lines, and their court points."""
    index = np.flatnonzero(np.isin(_LINE_IDS, list(lines)))
    return index, _POINTS[index]


def held_out_offsets(image, matrix, lines, boxes=None, half_width: int = 24,
                     prepared=None, details=False):
    """Where the paint of `lines` actually lies, in court feet, off the model.

    Pair with `refine(..., exclude_lines=lines)`: the registration then never
    saw these lines, so their measured offset is an accuracy figure rather than
    a residual. Returns signed perpendicular offsets, one per sample found.

    The window is deliberately wider than the errors being measured. A window
    that only just covered the answer would reject the large offsets at its
    edge and report the small ones -- the saturation that sank two earlier
    estimators here.
    """
    response, structure = prepared if prepared is not None else prepare(image)
    keep = np.isin(_LINE_IDS, list(lines))
    # No segment consensus here. It restricts each segment to a band around
    # its mean offset; at the ends of a slightly rotated long line that band
    # misses the paint, and those samples vanish -- reading 0.204 ft for a line
    # truly 0.308 ft off at 1 degree. A measurement must see every sample.
    observed, index, _ = _observe(response, structure, np.linalg.inv(matrix),
                                  half_width, keep, boxes, consensus=False)
    if not len(index):
        return (np.zeros(0), index) if details else np.zeros(0)
    court, _ = _project(matrix, observed)
    offsets = np.sum(sample_normals(index) * (court - _POINTS[index]), axis=1)
    return (offsets, index) if details else offsets


#: Line families worth holding out, by court_lines() index.
HOLD_OUT = {
    "boundary": (0,), "half-court": (1,), "centre circle": (2,),
    "near lane": (3,), "near FT circle": (4,), "near arc": (8,),
    "near corners": (6, 7), "far lane": (9,), "far FT circle": (10,),
    "far arc": (14,), "far corners": (12, 13),
}
