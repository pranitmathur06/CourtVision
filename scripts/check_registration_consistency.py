"""Two independent registrations of one instant must agree. No labels needed.

This is the test that exposed the painted key. A gate can only reject keys that
look wrong; it cannot tell you whether the ones it accepts are right, and the
accepted ones turned out to disagree with each other by 5.8 ft.

The trick is that ORB alignment between two frames 0.2 s apart is accurate to
about 0.5 px, and the court is rigid. So a point on the floor can reach court
coordinates two ways -- registered directly in frame A, or carried into frame B
by ORB and registered there -- and the two answers must match. Nothing here is
annotated, nothing is scored against a model of mine, and there is no threshold
chosen after seeing the answer: the two paths either agree or they do not.

A player moves under 3 ft in 0.2 s, so any disagreement much above that is the
registration moving, not the sport.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

GAP_S = 0.2


def _register(model, frame, device, conf, min_conf_points):
    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints

    result = model.predict(frame, device=device, verbose=False)[0]
    if result.keypoints is None or len(result.keypoints) == 0:
        return None
    xy = result.keypoints.xy[0].cpu().numpy()
    scores = (result.keypoints.conf[0].cpu().numpy()
              if result.keypoints.conf is not None else np.ones(len(xy)))
    seen = {i: tuple(xy[i]) for i in range(len(xy))
            if i in KEYPOINTS and scores[i] >= conf and (xy[i] > 0).all()}
    if len(seen) < min_conf_points:
        return None
    matrix, _ = homography_from_keypoints(seen)
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/raw_clips/fullgame.mp4")
    parser.add_argument("--weights",
                        default="runs/pose/checkpoints/court_keypoints/weights/best.pt")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--start", type=float, default=600.0,
                        help="skip the pre-game show; the anthem and the "
                             "coach interview are not court frames, and "
                             "sampling them once produced a calibration that "
                             "was right by luck")
    parser.add_argument("--end", type=float, default=7200.0)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.court_keypoints import registration_disagreement
    from courtvision.court_tracking import has_court, pairwise_homography
    from courtvision.device import resolve_device

    if not Path(args.weights).exists():
        print(f"FAIL - no weights at {args.weights}")
        return 1
    model = YOLO(args.weights)
    device = resolve_device()

    def register(frame):
        return _register(model, frame, device, args.conf, 6)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"FAIL - cannot open {args.video}")
        return 1
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    court_frames, both_registered, errors = 0, 0, []
    times = np.linspace(args.start, args.end, args.samples)
    for t in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok_a, frame_a = capture.read()
        capture.set(cv2.CAP_PROP_POS_MSEC, (t + GAP_S) * 1000)
        ok_b, frame_b = capture.read()
        if not (ok_a and ok_b) or not has_court(frame_a):
            continue
        court_frames += 1

        matrix_a = register(frame_a)
        matrix_b = register(frame_b)
        if matrix_a is None or matrix_b is None:
            continue
        carry = pairwise_homography(frame_a, frame_b)
        if carry is None:
            continue                      # ORB refused; not a registration failure
        both_registered += 1

        # Probe the lower half of the frame, which is the floor in a broadcast
        # camera; the upper half is crowd.
        height, width = frame_a.shape[:2]
        grid = np.array([[[x, y]]
                         for x in np.linspace(width * 0.15, width * 0.85, 6)
                         for y in np.linspace(height * 0.55, height * 0.92, 4)],
                        dtype=np.float32)
        errors.extend(registration_disagreement(matrix_a, matrix_b, carry, grid))

    errors = np.array(errors)
    print(f"{args.video}  fps {fps:.1f}")
    print(f"  court frames sampled     {court_frames}")
    print(f"  both frames registered   {both_registered}/{max(court_frames,1)}"
          f" = {both_registered/max(court_frames,1):.1%}")
    if len(errors):
        print(f"  the two paths disagree   p50 {np.median(errors):.2f} ft   "
              f"p90 {np.percentile(errors,90):.2f} ft   [gate p50 <= 2 ft; "
              f"painted key 5.8]")
    else:
        print("  no instant registered twice - nothing to compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
