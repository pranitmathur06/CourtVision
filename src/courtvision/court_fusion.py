"""One frame's registration, fused from the frames around it.

On a whole game at Toyota Center, a third of usable hand-labelled frames were
registered 1.6-3.5 ft off and ACCEPTED: the fit slid the key along the floor
onto other paint, which the fixed camera cannot forbid (a slide along the court
is a pan). Such a lock is a property of one frame's paint and one start; the
frames a fraction of a second either side see nearly the same view with
different players, blur and start, and mostly do not share it.

So each frame's registration is re-estimated from its neighbours: every
accepted registration within WINDOW_S is carried to the target frame by
frame-to-frame image tracking (ORB, ~0.5 px per hop), and the court position of
each point of the target frame is the MEDIAN over those candidates. A lock held
by fewer than half the window is outvoted.

The settings below were fixed before the method was run on any labelled frame.
"""

from __future__ import annotations

import numpy as np

#: Neighbours within this many seconds either side contribute.
WINDOW_S = 2.0
#: Sampling interval inside the window.
STEP_S = 0.2
#: Fewer accepted candidates than this and nothing is fused.
MIN_CANDIDATES = 3


def _grid(size):
    width, height = size
    xs = np.linspace(0.05 * width, 0.95 * width, 12)
    ys = np.linspace(0.3 * height, 0.95 * height, 8)
    return np.array([[x, y] for x in xs for y in ys], np.float64)


def _apply(matrix, points):
    h = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    return h[:, :2] / h[:, 2:3]


def fuse(candidates, size, min_candidates: int = MIN_CANDIDATES):
    """Median registration of the target frame from candidate image->court matrices.

    `candidates` are image->court homographies already expressed in the TARGET
    frame's pixels (neighbour registration composed with the tracked hop).
    Returns (matrix, spread_ft) -- spread is the median absolute deviation of
    the candidates about the fused court position -- or (None, None).
    """
    import cv2

    if len(candidates) < min_candidates:
        return None, None
    grid = _grid(size)
    courts = np.stack([_apply(m, grid) for m in candidates])        # (k, n, 2)
    finite = np.all(np.isfinite(courts), axis=(1, 2))
    courts = courts[finite]
    if len(courts) < min_candidates:
        return None, None
    median = np.median(courts, axis=0)
    spread = float(np.median(np.hypot(*(courts - median).transpose(2, 0, 1))))
    matrix, _ = cv2.findHomography(grid, median, 0)
    return matrix, spread


def chain(hops):
    """Compose frame-to-frame hops [H(t->t+1), H(t+1->t+2), ...] into H(t->t+k)."""
    total = np.eye(3)
    out = []
    for hop in hops:
        if hop is None:
            break
        total = hop @ total
        out.append(total.copy())
    return out
