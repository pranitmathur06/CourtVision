"""Does the paint land where the paint should be? An absolute check, no labels.

The ORB consistency test proves a registration is stable, not that it is right:
a constant offset agrees with itself perfectly across 0.2 s. This one pins the
answer to the court, by warping the frame into court coordinates and asking
what fraction of the edges it finds lie on lines that are actually painted.

Validated against the dataset's human annotations, by breaking a known-good
homography in the ways that matter:

    human annotation     0.380
    slipped 5 ft         0.274
    twisted 4 degrees    0.272
    wrong end            0.379   <- NOT DETECTED

**It cannot see an end swap, and no method of this kind can.** An NBA court's
markings are symmetric about half-court, so flipping ends maps every painted
line onto a painted line. The flip is composed with an affine reflection, which
preserves all projective structure -- the horizon and the foreshortening are
identical -- so it is invisible to perspective cues too. Which end of the floor
a frame shows is simply not recoverable from the floor's geometry.

That is not an excuse; it says where the end check has to live instead. It is
measured directly against human annotations in `eval_court_keypoints.py`, which
counts registrations landing on the wrong half, and it is the model's learned
appearance cues -- a far basket looks smaller and sits higher in frame -- that
have to supply it. This script is silent on the question by construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCALE = 8.0                 # pixels per foot in the top-down view
PAINT_SLACK_FT = 0.5
MIN_EDGES = 200


def line_agreement(frame, matrix, near_paint, size):
    """Fraction of warped edges lying on paint, or None if too few edges."""
    import cv2

    to_px = np.diag([SCALE, SCALE, 1.0]) @ matrix
    warped = cv2.warpPerspective(frame, to_px, size)
    seen = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8),
                               to_px, size) > 0
    # Erode so the warp's own boundary is not counted as a court line.
    seen = cv2.erode(seen.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    edges = (cv2.Canny(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), 60, 160) > 0)
    edges &= seen
    if edges.sum() < MIN_EDGES:
        return None
    return float((edges & near_paint).sum() / edges.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/raw_clips/fullgame.mp4")
    parser.add_argument("--weights",
                        default="runs/pose/checkpoints/court_keypoints/weights/best.pt")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--start", type=float, default=600.0)
    parser.add_argument("--end", type=float, default=7200.0)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.court_keypoints import (COURT_LENGTH_FT, COURT_WIDTH_FT,
                                             KEYPOINTS,
                                             homography_from_keypoints)
    from courtvision.court_lines import line_mask
    from courtvision.court_tracking import has_court
    from courtvision.device import resolve_device

    if not Path(args.weights).exists():
        print(f"FAIL - no weights at {args.weights}")
        return 1
    model = YOLO(args.weights)
    device = resolve_device()

    mask = line_mask(SCALE, thickness_ft=0.35)
    slack = max(3, int(round(PAINT_SLACK_FT * SCALE)) | 1)
    near_paint = cv2.dilate(mask, np.ones((slack, slack), np.uint8)) > 0
    size = (int(COURT_WIDTH_FT * SCALE), int(COURT_LENGTH_FT * SCALE))

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"FAIL - cannot open {args.video}")
        return 1

    court_frames, registered = 0, 0
    real, slipped = [], []
    for t in np.linspace(args.start, args.end, args.samples):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok or not has_court(frame):
            continue
        court_frames += 1
        result = model.predict(frame, device=device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        xy = result.keypoints.xy[0].cpu().numpy()
        conf = (result.keypoints.conf[0].cpu().numpy()
                if result.keypoints.conf is not None else np.ones(len(xy)))
        seen = {i: tuple(xy[i]) for i in range(len(xy))
                if i in KEYPOINTS and conf[i] >= args.conf and (xy[i] > 0).all()}
        matrix, _ = homography_from_keypoints(seen)
        if matrix is None:
            continue
        registered += 1
        score = line_agreement(frame, matrix, near_paint, size)
        # The same frame, registered 5 ft wrong. Carried alongside so the
        # comparison is on THIS broadcast rather than the dataset's, and a
        # score that fails to separate them is visible immediately.
        slip = np.array([[1, 0, 0], [0, 1, 5.0], [0, 0, 1]]) @ matrix
        control = line_agreement(frame, slip, near_paint, size)
        if score is not None:
            real.append(score)
        if control is not None:
            slipped.append(control)

    print(f"{args.video}")
    print(f"  court frames sampled   {court_frames}")
    print(f"  registered             {registered}/{max(court_frames,1)} = "
          f"{registered/max(court_frames,1):.1%}")
    if real:
        print(f"  line agreement         p50 {np.median(real):.3f}"
              f"   p10 {np.percentile(real,10):.3f}")
        print(f"  same frames, 5 ft off  p50 {np.median(slipped):.3f}"
              f"   (the gap is the evidence; on human annotations "
              f"0.380 against 0.274)")
    print("  end of floor           not checked here -- the court is symmetric "
          "about half-court, see the module docstring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
