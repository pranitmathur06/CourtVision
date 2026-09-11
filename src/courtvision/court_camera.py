"""One broadcast camera per game: a fixed centre, pan/tilt/roll/zoom per frame.

A free homography has eight degrees of freedom, and a frame's paint rarely pins
all of them. With lines only near one basket the far side is extrapolation: a
refit that never saw the boundary drew the far sideline diagonally across the
floor, and a fit whose boundary samples reached an LED ad board and the score
graphic squeezed the whole court between them -- both with sharp peak ratios,
because sharpness is judged on the paint the fit used.

The main broadcast camera does not move during a game; it turns and zooms. So
every frame's homography is K(f) R [r1 r2 -R C] with ONE centre C for the game,
leaving four parameters per frame. Neither failure above is a rotation and a
zoom of a camera at C, so neither is representable.

Measured on human annotations before this was built (18 games, split-half: the
centre estimated jointly on half of each game's annotated frames, the other
half then fitted with it fixed): 0.20 ft median error on the annotated points,
p90 0.32, against 0.17 ft for a free homography on the same points. The
constraint costs ~0.03 ft where the paint already pins the fit and removes four
directions the paint often does not.

Assumed: square pixels, no skew, principal point at the image centre, no lens
distortion -- the split-half figures above already include what these cost.
Frames from another camera (baseline, replay, picture-in-picture) do not fit
the centre; `FixedCamera.explains` lets callers detect that rather than force
them into the model.
"""

from __future__ import annotations

import numpy as np

#: Frames whose paint the joint fit leaves further off than this (median, px)
#: are treated as another camera, or a wrong fit, and dropped from the centre.
OUTLIER_PX = 3.0
MIN_FRAMES = 6


def _rodrigues(rvec):
    import cv2
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]


def _intrinsics(f, size):
    width, height = size
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def ptz_matrix(params, centre, size):
    """court (x, y, 1) -> image, for rotation vector params[:3] and log-focal params[3]."""
    rotation = _rodrigues(params[:3])
    projection = _intrinsics(np.exp(params[3]), size) @ np.c_[rotation, -rotation @ centre]
    return projection[:, [0, 1, 3]]


def decompose(court_to_image, size):
    """Focal length, rotation and centre of a court -> image homography, or None."""
    width, height = size
    shift = np.array([[1, 0, -width / 2.0], [0, 1, -height / 2.0], [0, 0, 1.0]])
    g = shift @ np.asarray(court_to_image, dtype=np.float64)
    h1, h2 = g[:, 0], g[:, 1]
    if abs(h1[2] * h2[2]) < 1e-15:
        return None
    f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2])
    if not np.isfinite(f2) or f2 <= 0:
        return None
    f = float(np.sqrt(f2))
    a = np.diag([1 / f, 1 / f, 1.0]) @ g
    scale = 2.0 / (np.linalg.norm(a[:, 0]) + np.linalg.norm(a[:, 1]))
    r1, r2, t = a[:, 0] * scale, a[:, 1] * scale, a[:, 2] * scale
    if t[2] < 0:                                     # the court must be in front
        r1, r2, t = -r1, -r2, -t
    u, _, vt = np.linalg.svd(np.c_[r1, r2, np.cross(r1, r2)])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1
    return f, rotation, -rotation.T @ t


def _project(court_to_image, points):
    h = np.c_[points, np.ones(len(points))] @ np.asarray(court_to_image).T
    return h[:, :2] / h[:, 2:3]


def ptz_params(court_to_image, centre, size, points):
    """Pan/tilt/roll/zoom at `centre` best reproducing a homography on `points`."""
    import cv2
    from scipy.optimize import least_squares

    target = _project(court_to_image, points)
    found = decompose(court_to_image, size)
    f0 = found[0] if found else float(max(size))
    rotations = []
    if found:
        rotations += [found[1], found[1] @ np.diag([-1.0, -1.0, 1.0])]
    # Look from the centre at the points' middle, as a fallback start.
    forward = np.append(points.mean(axis=0), 0.0) - centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= max(np.linalg.norm(right), 1e-9)
    rotations.append(np.vstack([right, np.cross(forward, right), forward]))
    best = None
    for rotation in rotations:
        x0 = np.r_[cv2.Rodrigues(rotation)[0].ravel(), np.log(f0)]

        def residuals(p):
            return np.nan_to_num((_project(ptz_matrix(p, centre, size), points)
                                  - target).ravel(), nan=1e4, posinf=1e4, neginf=-1e4)
        solution = least_squares(residuals, x0, loss="soft_l1", f_scale=2.0)
        cost = float(np.median(np.hypot(*residuals(solution.x).reshape(-1, 2).T)))
        if best is None or cost < best[0]:
            best = (cost, solution.x)
    return best[1], best[0]


class FixedCamera:
    """A game's camera centre, and the four-parameter model it implies."""

    def __init__(self, centre, size):
        self.centre = np.asarray(centre, dtype=np.float64)
        self.size = tuple(int(v) for v in size)

    def parameterise(self, court_to_image, points):
        """(compose, x0): the model as a function of 4 parameters, started at the fit."""
        x0, _ = ptz_params(court_to_image, self.centre, self.size, points)
        return (lambda p: ptz_matrix(p, self.centre, self.size)), x0

    def explains(self, image_to_court, points, tolerance_px=OUTLIER_PX):
        """Whether a frame's registration is a pan/tilt/zoom of this camera."""
        _, cost = ptz_params(np.linalg.inv(image_to_court), self.centre, self.size, points)
        return cost <= tolerance_px, cost


def estimate_centre(fits, size, outlier_px=OUTLIER_PX):
    """One centre from many frames' registrations, by joint robust fitting.

    `fits` is a list of (image_to_court, court_points) -- each frame's
    registration and the court points it rests on (its support). Each frame's
    own decomposition is a poor estimate of the centre: focal length and depth
    trade off along the viewing ray, and on human annotations the median of
    per-frame centres fitted held-out frames at 0.57 ft. The centre is instead
    solved jointly with every frame's pan/tilt/roll/zoom; frames the joint
    model leaves more than `outlier_px` off are dropped (another camera, or a
    wrong fit) and the rest re-solved.

    Returns (FixedCamera or None, report).
    """
    from scipy.optimize import least_squares

    frames = []
    for image_to_court, points in fits:
        points = np.asarray(points, dtype=np.float64)
        if len(points) > 60:
            points = points[np.linspace(0, len(points) - 1, 60).astype(int)]
        court_to_image = np.linalg.inv(image_to_court)
        found = decompose(court_to_image, size)
        if found is None or len(points) < 8:
            continue
        frames.append((court_to_image, points, _project(court_to_image, points), found[2]))
    if len(frames) < MIN_FRAMES:
        return None, {"frames": len(frames), "reason": "too few frames"}
    centre = np.median(np.array([c for *_, c in frames]), axis=0)
    keep = list(range(len(frames)))
    # The outlier rule is always allowed to fire. An earlier version kept every
    # frame whenever dropping the outliers would leave fewer than MIN_FRAMES,
    # and still reported them all as inliers: a centre solved from six
    # near-duplicate frames, four of them over the limit, was reported as 6/6
    # and carried a test arena's pass. Too few frames agreeing on one centre
    # now means no centre.
    for _ in range(6):
        params = [ptz_params(frames[i][0], centre, size, frames[i][1])[0] for i in keep]

        def residuals(p):
            c = p[:3]
            out = []
            for k, i in enumerate(keep):
                model = ptz_matrix(p[3 + 4 * k: 7 + 4 * k], c, size)
                out.append((_project(model, frames[i][1]) - frames[i][2]).ravel())
            return np.nan_to_num(np.concatenate(out), nan=1e4, posinf=1e4, neginf=-1e4)
        solution = least_squares(residuals, np.r_[centre, np.concatenate(params)],
                                 loss="soft_l1", f_scale=2.0, max_nfev=200)
        centre = solution.x[:3]
        per_frame = []
        for k, i in enumerate(keep):
            model = ptz_matrix(solution.x[3 + 4 * k: 7 + 4 * k], centre, size)
            per_frame.append(float(np.median(np.hypot(*(_project(model, frames[i][1])
                                                        - frames[i][2]).T))))
        inliers = [i for i, e in zip(keep, per_frame) if e <= outlier_px]
        report = {"frames": len(frames), "inliers": len(inliers),
                  "centre": centre.tolist(), "residual_px": float(np.median(per_frame)),
                  "worst_px": float(max(per_frame))}
        if len(inliers) == len(keep):
            return FixedCamera(centre, size), report
        if len(inliers) < MIN_FRAMES:
            return None, dict(report, reason="too few frames agree on one centre")
        keep = inliers
    return None, dict(report, reason="outlier removal did not settle")


def look_at(target_xy, f, centre, roll_deg=0.0):
    """PTZ parameters for a camera at `centre` looking at court point `target_xy`."""
    import cv2
    forward = np.array([target_xy[0], target_xy[1], 0.0]) - np.asarray(centre, np.float64)
    forward /= np.linalg.norm(forward)
    # Image right is forward x world-up; image down is forward x right, which
    # points at the floor. Crossing with world-DOWN instead (as a first version
    # did) builds an upside-down camera: every hypothesis rolled 180 degrees.
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= max(np.linalg.norm(right), 1e-9)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    roll = np.deg2rad(roll_deg)
    spin = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1.0]])
    rotation = spin @ rotation
    return np.r_[cv2.Rodrigues(rotation)[0].ravel(), np.log(f)]


#: The landmark-free search: where the camera looks (court feet), how zoomed it
#: is (focal length as a multiple of image width), and a small roll.
SEARCH_X_FT = tuple(float(x) for x in range(0, 51, 5))
SEARCH_Y_FT = tuple(float(y) for y in range(-6, 101, 3))
SEARCH_F = (0.8, 1.0, 1.25, 1.55, 1.9, 2.35, 2.9, 3.6)
SEARCH_KEEP = 3
SEARCH_MIN_VISIBLE = 60
#: Scale of the chamfer score, in pixels: how far a projected line may sit from
#: painted line and still count for something.
CHAMFER_PX = 25.0


def _paint_distance(response, boxes):
    """Distance, in pixels, from every pixel to the nearest painted ridge."""
    import cv2
    level = max(6.0, float(np.percentile(response, 97)))
    ridge = (response >= level).astype(np.uint8)
    if boxes is not None and len(boxes):
        for x1, y1, x2, y2 in np.asarray(boxes, int):
            ridge[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
    return cv2.distanceTransform(1 - ridge, cv2.DIST_L2, 5)


def _chamfer(distance, court_to_image, points, boxes):
    """Smooth score of a hypothesis: how close its projected lines lie to paint."""
    height, width = distance.shape
    h = np.c_[points, np.ones(len(points))] @ court_to_image.T
    front = h[:, 2] > 1e-6
    xy = h[front, :2] / h[front, 2:3]
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < width - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < height - 1)
    xy = xy[inside]
    if boxes is not None and len(boxes):
        for x1, y1, x2, y2 in np.asarray(boxes, float):
            xy = xy[~((xy[:, 0] >= x1) & (xy[:, 0] <= x2) & (xy[:, 1] >= y1) & (xy[:, 1] <= y2))]
    if len(xy) < SEARCH_MIN_VISIBLE:
        return -1.0
    d = distance[xy[:, 1].astype(int), xy[:, 0].astype(int)]
    return float(np.sum(np.exp(-(d / CHAMFER_PX) ** 2)))


#: Downscale for the floor-overlap score; the silhouette needs no detail.
SEARCH_SCALE = 8
#: Court outline, and the two lanes, in court feet.
_COURT = np.array([[0, 0], [50, 0], [50, 94], [0, 94]], np.float64)
_LANES = (np.array([[17, 0], [33, 0], [33, 19], [17, 19]], np.float64),
          np.array([[17, 75], [33, 75], [33, 94], [17, 94]], np.float64))


def _floor_masks(image, boxes):
    """Floor pixels (wood or court paint, players filled in), downscaled."""
    import cv2
    from .candidates import court_region
    region = court_region(image, erode_px=0)
    if region is None:
        return None
    region = region.astype(np.uint8)
    if boxes is not None and len(boxes):
        for x1, y1, x2, y2 in np.asarray(boxes, int):      # players stand on floor
            region[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 1
    h, w = region.shape
    return cv2.resize(region, (w // SEARCH_SCALE, h // SEARCH_SCALE),
                      interpolation=cv2.INTER_AREA) > 0.5


def _floor_overlap(floor, court_to_image):
    """IoU of the projected court with the floor -- lanes in or out, whichever
    fits: some floors paint the key in the floor's colours, some do not."""
    import cv2
    scale = np.diag([1.0 / SEARCH_SCALE, 1.0 / SEARCH_SCALE, 1.0])
    h = np.c_[_COURT, np.ones(4)] @ (scale @ court_to_image).T
    if np.any(h[:, 2] <= 1e-6):
        return -1.0
    outline = np.round(h[:, :2] / h[:, 2:3]).astype(np.int32)
    if np.any(np.abs(outline) > 100000):
        return -1.0
    court = np.zeros(floor.shape, np.uint8)
    cv2.fillPoly(court, [outline], 1)
    best = -1.0
    for with_lanes in (True, False):
        mask = court.copy()
        if not with_lanes:
            for lane in _LANES:
                hl = np.c_[lane, np.ones(4)] @ (scale @ court_to_image).T
                cv2.fillPoly(mask, [np.round(hl[:, :2] / hl[:, 2:3]).astype(np.int32)], 0)
        m = mask.astype(bool)
        union = (m | floor).sum()
        if union:
            best = max(best, float((m & floor).sum()) / float(union))
    return best


def search_starts(camera, response, structure, boxes=None, keep=SEARCH_KEEP, image=None):
    """Image->court starts for a frame, from the floor's silhouette and its paint.

    With the centre fixed a frame is four numbers, so they can be searched.
    Scoring by paint alone failed on broadcast: the strongest ridges in a frame
    are the score graphic, ad boards and crowd, and a hypothesis that threw the
    court into the stands out-scored the true pose three to one. The floor is
    the better witness -- the projected court must cover the wood the camera
    sees -- so hypotheses are ranked by their overlap with the floor, polished
    on it, and handed to the refinement, which aligns the paint.
    """
    from scipy.optimize import minimize

    if image is None:
        return []
    floor = _floor_masks(image, boxes)
    if floor is None or floor.mean() < 0.05:
        return []
    width, _ = camera.size

    def model_of(x, y, logf, roll=0.0):
        return ptz_matrix(look_at((x, y), np.exp(logf) * width, camera.centre, roll),
                          camera.centre, camera.size)

    scored = []
    for fx in SEARCH_F:
        for x in SEARCH_X_FT:
            for y in SEARCH_Y_FT:
                score = _floor_overlap(floor, model_of(x, y, np.log(fx)))
                if score > 0:
                    scored.append((score, x, y, fx))
    scored.sort(key=lambda s: -s[0])
    seeds = []
    for score, x, y, fx in scored:
        if all(np.hypot(x - sx, y - sy) > 6.0 or abs(np.log(fx / sf)) > 0.3
               for _, sx, sy, sf in seeds):
            seeds.append((score, x, y, fx))
        if len(seeds) == 2 * keep:
            break
    polished = []
    for _, x, y, fx in seeds:
        best = minimize(lambda v: -_floor_overlap(floor, model_of(*v)),
                        [x, y, np.log(fx), 0.0], method="Powell",
                        options={"xtol": 0.02, "ftol": 1e-3, "maxfev": 300})
        polished.append((-best.fun, best.x))
    polished.sort(key=lambda s: -s[0])
    return [np.linalg.inv(model_of(*v)) for _, v in polished[:keep]]
