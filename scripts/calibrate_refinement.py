"""Set the refinement's acceptance threshold on the calibration game only.

The threshold decides which fits are trusted, so choosing it on the footage the
result is claimed for would be selection after seeing the answer. It is chosen
here on the OKC game -- an arena that appears in training anyway -- and the
unseen arena is scored afterwards with the value fixed.

Two populations, on the same frames:

- fits started from the landmark registration, which is where real use starts;
- fits started deliberately 20-30 ft wrong, which is the failure the guard
  exists for -- a lock onto the wrong paint that reports a tiny residual.

The peak ratio must separate them. A threshold that passes the wrong starts is
useless; one that refuses most true fits is useless in a different way.

It also measures the capture range without ground truth: a fit started 6 ft
off should land where the landmark-started fit landed. If it lands somewhere
else and still looks sharp, that is a false lock the ratio cannot see, and it
would be reported here rather than discovered downstream.
"""

from __future__ import annotations

import argparse

import numpy as np

WRONG = ((25.0, 0.0), (-20.0, 0.0), (0.0, 30.0), (0.0, -25.0), (12.0, 12.0))
NEAR = ((6.0, 0.0), (0.0, 6.0), (-4.0, 4.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/raw_clips/fullgame.mp4")
    parser.add_argument("--weights",
                        default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--start", type=float, default=900.0)
    parser.add_argument("--end", type=float, default=6900.0)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    from courtvision.court_refine import prepare, refine
    from courtvision.court_tracking import has_court
    from courtvision.device import resolve_device

    model = YOLO(args.weights)
    detector = YOLO(args.detector)
    device = resolve_device()
    capture = cv2.VideoCapture(args.video)

    def shifted(matrix, dx, dy):
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]]) @ matrix

    def error(a, b, height, width):
        grid = np.array([[x, y] for x in np.linspace(width * .15, width * .85, 6)
                         for y in np.linspace(height * .45, height * .9, 5)],
                        dtype=np.float32).reshape(-1, 1, 2)
        pa = cv2.perspectiveTransform(grid, a).reshape(-1, 2)
        pb = cv2.perspectiveTransform(grid, b).reshape(-1, 2)
        return float(np.median(np.hypot(*(pa - pb).T)))

    true_ratio, wrong_ratio, wrong_accepted = [], [], 0
    near_same, near_elsewhere, near_refused = 0, [], 0
    frames = 0
    for t in np.linspace(args.start, args.end, args.samples):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok or not has_court(frame):
            continue
        result = model.predict(frame, device=device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        xy = result.keypoints.xy[0].cpu().numpy()
        conf = result.keypoints.conf[0].cpu().numpy()
        start, _ = homography_from_keypoints(
            {i: tuple(xy[i]) for i in range(len(xy))
             if i in KEYPOINTS and conf[i] >= args.conf and (xy[i] > 0).all()})
        if start is None:
            continue
        found = detector.predict(frame, device=device, verbose=False)[0].boxes
        boxes = (found.xyxy.cpu().numpy()[found.cls.cpu().numpy() == 0]
                 if found is not None and len(found) else None)
        prepared = prepare(frame)
        height, width = frame.shape[:2]
        frames += 1

        reference, info = refine(frame, start, boxes=boxes, prepared=prepared)
        if info["peak_ratio"] is not None:
            true_ratio.append(info["peak_ratio"])
        for dx, dy in WRONG:
            _, winfo = refine(frame, shifted(start, dx, dy), boxes=boxes,
                              prepared=prepared)
            if winfo["peak_ratio"] is not None:
                wrong_ratio.append(winfo["peak_ratio"])
            wrong_accepted += bool(winfo["refined"])
        if not info["refined"]:
            continue
        for dx, dy in NEAR:
            landed, ninfo = refine(frame, shifted(start, dx, dy), boxes=boxes,
                                   prepared=prepared)
            if not ninfo["refined"]:
                near_refused += 1
            elif error(landed, reference, height, width) < 0.2:
                near_same += 1
            else:
                near_elsewhere.append(error(landed, reference, height, width))

    t, w = np.array(true_ratio), np.array(wrong_ratio)
    print(f"{args.video}: {frames} registered frames")
    if len(t):
        print(f"  landmark-started fits  peak ratio p10 {np.percentile(t,10):6.2f}"
              f"  p50 {np.median(t):6.2f}   (n={len(t)})")
    if len(w):
        print(f"  20-30 ft wrong starts  peak ratio p50 {np.median(w):6.2f}"
              f"  p90 {np.percentile(w,90):6.2f}  max {w.max():6.2f}   (n={len(w)})")
    for threshold in (1.5, 2.0, 3.0, 4.0, 6.0):
        keep_true = np.mean(t >= threshold) if len(t) else float("nan")
        keep_wrong = np.mean(w >= threshold) if len(w) else float("nan")
        print(f"    threshold {threshold:3.1f}: accepts {keep_true:5.0%} of true fits,"
              f" {keep_wrong:5.0%} of wrong starts")
    print(f"  wrong starts accepted at the shipped threshold: {wrong_accepted}")
    total = near_same + len(near_elsewhere) + near_refused
    if total:
        print(f"  6 ft starts: {near_same}/{total} converge to the same fit, "
              f"{near_refused} refused, {len(near_elsewhere)} land ELSEWHERE"
              + (f" (by {np.median(near_elsewhere):.1f} ft)" if near_elsewhere else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
