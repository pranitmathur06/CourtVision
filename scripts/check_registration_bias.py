"""Absolute registration accuracy on broadcast, with no annotations.

The consistency test measures STABILITY: two registrations of one instant
agreeing. Any error that is a function of the court rather than the frame -- an
offset, a scale, a rotation -- is identical in both and passes it perfectly. So
0.45 ft of agreement is not 0.45 ft of accuracy, and nothing else measured on
broadcast footage was absolute.

This slides the registration by a known amount and samples brightness ON the
projected court lines. Real paint is bright against wood, so the offset where
brightness peaks is the registration's bias, in feet by construction. It
aggregates thousands of samples per frame rather than picking a nearest ridge
per point -- an earlier attempt did that and could not tell a correct
registration from one slipped 2 ft (1.84 ft against 1.99 ft), because with a
wide search window there is always some bright thing nearby.

Validated on 60 human-annotated frames, where the registration is known good:
peak +0.00 ft along the court and -0.02 ft across it, contrast 2.1x. A method
that reported a bias there would be measuring itself.
"""

from __future__ import annotations

import argparse

import numpy as np

OFFSETS = np.arange(-5.0, 5.01, 0.5)
MIN_SAMPLES = 200


def _dense_lines():
    import cv2  # noqa: F401

    from courtvision.court_lines import court_lines

    points = []
    for polyline in court_lines():
        p = np.array(polyline, dtype=np.float32)
        for a, b in zip(p[:-1], p[1:]):
            n = max(2, int(np.hypot(*(b - a)) * 3))
            points.extend(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.array(points, dtype=np.float32)[None]


def brightness_on_lines(frame, matrix, dense, dx: float, dy: float):
    """Mean paint-brightness sampled along the projected court lines."""
    import cv2

    shifted = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]]) @ matrix
    inverse = np.linalg.inv(shifted)
    height, width = frame.shape[:2]
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # A top-hat keeps thin bright ridges -- painted lines -- and drops the
    # slow brightness of the floor itself.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    paint = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kernel)
    image = cv2.perspectiveTransform(dense, inverse)[0]
    x, y = image[:, 0], image[:, 1]
    inside = (x > 5) & (x < width - 5) & (y > 5) & (y < height - 5)
    if inside.sum() < MIN_SAMPLES:
        return None
    return float(paint[y[inside].astype(int), x[inside].astype(int)].mean())


def peak_of(curve):
    """Offset maximising the curve, refined against its two neighbours."""
    values = np.array(curve)
    i = int(values.argmax())
    if 0 < i < len(values) - 1:
        a, b, c = values[i - 1], values[i], values[i + 1]
        denominator = a - 2 * b + c
        if denominator != 0:
            return OFFSETS[i] + 0.25 * (a - c) / denominator
    return float(OFFSETS[i])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/raw_clips/fullgame.mp4")
    parser.add_argument("--weights",
                        default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--calibrated", action="store_true",
                        help="apply CALIBRATION_FT first. The offset was "
                             "derived from the feed's shot chart, so measuring "
                             "it here -- with paint brightness, which never saw "
                             "the feed -- is an independent confirmation.")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.court_keypoints import (KEYPOINTS, calibrated,
                                             homography_from_keypoints)
    from courtvision.court_tracking import has_court
    from courtvision.device import resolve_device

    dense = _dense_lines()
    model = YOLO(args.weights)
    device = resolve_device()
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"FAIL - cannot open {args.video}")
        return 1

    along = {d: [] for d in OFFSETS}
    across = {d: [] for d in OFFSETS}
    frames = 0
    for t in np.linspace(700, 7100, args.samples):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok or not has_court(frame):
            continue
        result = model.predict(frame, device=device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        xy = result.keypoints.xy[0].cpu().numpy()
        conf = result.keypoints.conf[0].cpu().numpy()
        seen = {i: tuple(xy[i]) for i in range(len(xy))
                if i in KEYPOINTS and conf[i] >= args.conf and (xy[i] > 0).all()}
        matrix, _ = homography_from_keypoints(seen)
        if matrix is None:
            continue
        if args.calibrated:
            matrix = calibrated(matrix)
        frames += 1
        for d in OFFSETS:
            v = brightness_on_lines(frame, matrix, dense, 0.0, d)
            if v is not None:
                along[d].append(v)
            v = brightness_on_lines(frame, matrix, dense, d, 0.0)
            if v is not None:
                across[d].append(v)

    print(f"{args.video}: {frames} registered frames")
    for name, curve in (("along the court ", along), ("across the court", across)):
        means = [float(np.mean(curve[d])) if curve[d] else 0.0 for d in OFFSETS]
        print(f"  {name}  bias {peak_of(means):+.2f} ft   "
              f"peak/edge contrast {max(means)/means[0]:.2f}x")
    print("  [human-annotated reference: +0.00 / -0.02 ft at 2.1x contrast]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
