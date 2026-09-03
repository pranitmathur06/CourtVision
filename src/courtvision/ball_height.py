"""Ball height from a single broadcast camera.

One camera gives no depth, so a lone image point of the ball is a ray, not a
position. What breaks the ambiguity is physics: a ball in flight is a parabola,
and gravity is known. Six unknowns describe a free flight —

    x(t) = x0 + vx t
    y(t) = y0 + vy t
    z(t) = z0 + vz t - g t^2 / 2

— and every observed frame contributes two equations, so five or six frames
over-determine it. The court registration supplies the camera, and the fit is
whichever trajectory reprojects onto the pixels actually seen.

This exists because `shot_detection` needs height and nothing in the pipeline
could supply it: `ball_z` came from SportVU tracking, which pairs with no video.
The accuracy required is known and lenient — a swept ±4 ft error still leaves
emitted-commentary precision at 0.852 — so the bar is to be roughly right about
an arc, not to survey it.
"""

from __future__ import annotations

import numpy as np

GRAVITY_FT_S2 = 32.174


def _project(params: np.ndarray, points_xyz: np.ndarray,
             shape: tuple[int, int]) -> np.ndarray:
    """Project many world points at once. Mirrors `court_lines.project_world_point`."""
    cx, cy, cz, tx, ty, focal = params
    centre = np.array([cx, cy, cz], dtype=np.float64)
    forward = np.array([tx, ty, 0.0]) - centre
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rel = points_xyz - centre
    depth = rel @ forward
    height, width = shape
    out = np.full((len(points_xyz), 2), np.nan)
    ok = depth > 1e-6
    out[ok, 0] = width / 2 + focal * (rel[ok] @ right) / depth[ok]
    out[ok, 1] = height / 2 - focal * (rel[ok] @ up) / depth[ok]
    return out


def fit_flight(times: np.ndarray, pixels: np.ndarray, params: np.ndarray,
               shape: tuple[int, int],
               seed_xy: tuple[float, float]) -> np.ndarray | None:
    """Fit one ballistic arc to a run of observed ball pixels.

    Returns (x0, y0, z0, vx, vy, vz) in court feet, or None if it will not fit.
    """
    from scipy.optimize import least_squares

    good = ~np.isnan(pixels).any(axis=1)
    if good.sum() < 5:
        return None
    times = times[good] - times[good][0]
    pixels = pixels[good]

    def residual(state):
        x0, y0, z0, vx, vy, vz = state
        xyz = np.stack([
            x0 + vx * times,
            y0 + vy * times,
            z0 + vz * times - 0.5 * GRAVITY_FT_S2 * times ** 2,
        ], axis=1)
        got = _project(params, xyz, shape)
        diff = got - pixels
        diff[np.isnan(diff)] = 1e3
        return diff.ravel()

    guess = np.array([seed_xy[0], seed_xy[1], 7.0, 0.0, 0.0, 10.0])
    try:
        answer = least_squares(residual, guess, method="lm", max_nfev=400)
    except Exception:
        return None
    return answer.x


def heights(times: np.ndarray, pixels: np.ndarray, params: np.ndarray,
            shape: tuple[int, int], seed_xy: tuple[float, float],
            window: int = 9) -> np.ndarray:
    """Ball height per frame, from overlapping ballistic fits.

    A window rather than one global fit because a game is not one flight: the
    ball is dribbled, passed, held. Each window is fitted independently and a
    frame's height is the median over the windows covering it, so a window that
    straddles a bounce is outvoted rather than dominant.
    """
    n = len(times)
    votes: list[list[float]] = [[] for _ in range(n)]
    for start in range(0, max(n - window + 1, 1)):
        stop = min(start + window, n)
        state = fit_flight(times[start:stop], pixels[start:stop], params,
                           shape, seed_xy)
        if state is None:
            continue
        x0, y0, z0, vx, vy, vz = state
        local = times[start:stop] - times[start]
        z = z0 + vz * local - 0.5 * GRAVITY_FT_S2 * local ** 2
        for index, value in zip(range(start, stop), z):
            if -5.0 < value < 40.0:
                votes[index].append(float(value))
    return np.array([np.median(v) if v else np.nan for v in votes])
