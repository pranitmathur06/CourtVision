"""Solve the canonical->feet transform from symmetry, not from guesswork.

Two dead ends preceded this. Bootstrapping court coordinates from the
painted-key registration scattered by 15 ft and failed its own symmetry check;
gating that registration dropped it to 1 usable frame in 745, because these are
other arenas and the painted key does not travel. And the perspective in a
broadcast frame is too severe to identify 48 landmarks by eye without guessing.

But the dataset declares its own symmetry. `flip_idx` pairs each landmark with
its mirror about the half-court line, so for every pair (a, b):

    y_a + y_b = 94        (mirror about mid-length)
    x_a       = x_b       (same distance across)

Fifteen pairs give thirty equations for a homography's eight parameters. What
symmetry alone cannot fix is scale along each axis -- so the court's own
dimensions, 50 ft by 94 ft, close it.
"""
import json
import numpy as np
from scipy.optimize import least_squares

SP = "outputs"
FLIP = [35,36,2,38,39,5,41,42,34,32,10,33,29,30,31,15,26,27,28,19,20,21,22,23,
        24,25,16,17,18,12,13,14,9,11,8,0,1,37,3,4,40,6,7,43,44,45,46,47]
LENGTH, WIDTH = 94.0, 50.0

sch = {int(k): np.array(v) for k, v in
       json.load(open(f"{SP}/kp_schema.json"))["schema"].items()}
idx = sorted(sch)
P = np.array([sch[i] for i in idx])
pairs = [(a, FLIP[a]) for a in idx if FLIP[a] in sch and a < FLIP[a]]
centre = [a for a in idx if FLIP[a] == a]
print(f"  {len(idx)} canonical points, {len(pairs)} mirror pairs, "
      f"{len(centre)} on the centre line")


def apply(h, pts):
    G = np.append(h, 1.0).reshape(3, 3)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ G.T
    z = np.where(np.abs(hom[:, 2]) < 1e-9, 1e-9, hom[:, 2])
    return hom[:, :2] / z[:, None]


where = {i: n for n, i in enumerate(idx)}


def residuals(h):
    Q = apply(h, P)
    out = []
    for a, b in pairs:
        qa, qb = Q[where[a]], Q[where[b]]
        out.append(qa[1] + qb[1] - LENGTH)      # mirrored about mid-length
        out.append(qa[0] - qb[0])               # same distance across
    for c in centre:
        out.append(Q[where[c]][1] - LENGTH / 2)  # centre line sits at mid-court
    # Scale: the annotated landmarks should fill the court, not a corner of it.
    out.append((Q[:, 0].max() - Q[:, 0].min() - WIDTH) * 3.0)
    out.append((Q[:, 1].max() - Q[:, 1].min() - LENGTH) * 3.0)
    out.append((Q[:, 0].min() - 0.0) * 3.0)
    out.append((Q[:, 1].min() - 0.0) * 3.0)
    return np.array(out)


# Start from the affine map that puts the canonical bounding box on the court.
lo, hi = P.min(0), P.max(0)
start = np.array([WIDTH / (hi[0] - lo[0]), 0, -lo[0] * WIDTH / (hi[0] - lo[0]),
                  0, LENGTH / (hi[1] - lo[1]), -lo[1] * LENGTH / (hi[1] - lo[1]),
                  0, 0], dtype=float)
best = None
for jitter in range(24):
    guess = start.copy()
    if jitter:
        rng = np.random.default_rng(jitter)
        guess[:6] *= 1 + rng.normal(0, 0.15, 6)
        guess[6:] = rng.normal(0, 2e-3, 2)
    try:
        sol = least_squares(residuals, guess, method="lm", max_nfev=20000)
    except Exception:
        continue
    if best is None or sol.cost < best.cost:
        best = sol

Q = apply(best.x, P)
sym_y = [abs(Q[where[a]][1] + Q[where[b]][1] - LENGTH) for a, b in pairs]
sym_x = [abs(Q[where[a]][0] - Q[where[b]][0]) for a, b in pairs]
print(f"\n  after fitting:")
print(f"    |y_a + y_b - 94| : p50 {np.median(sym_y):.2f} ft  p90 {np.percentile(sym_y,90):.2f}")
print(f"    |x_a - x_b|      : p50 {np.median(sym_x):.2f} ft  p90 {np.percentile(sym_x,90):.2f}")
print(f"    extent: x {Q[:,0].min():.1f}..{Q[:,0].max():.1f}   "
      f"y {Q[:,1].min():.1f}..{Q[:,1].max():.1f}   (court is 50 x 94)")
json.dump({str(i): [float(Q[where[i]][0]), float(Q[where[i]][1])] for i in idx},
          open(f"{SP}/kp_court_feet.json", "w"), indent=1)
