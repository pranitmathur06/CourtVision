"""Registration during live play, on the denominator the painted key was scored on.

Round 26 measured the painted key at 17% by sampling the seconds before each
aligned shot -- mid-possession by construction, and main-camera by convention,
because that is when a broadcast is showing the play.

Scoring the keypoint model against `has_court` instead gave 78%, and the two
numbers were never comparable. `has_court` is a wood-fraction heuristic: it
fires on a player lying on the floor filling the frame, on a courtside
close-up where skin and jersey pass the colour test, and on the under-basket
stanchion camera. Inspecting the frames where the detector found no court at
all, three of six were not court views, two were alternate cameras absent from
any training set, and one was a genuine miss. Registering those is neither
possible nor wanted.

The denominator here is chosen by the FEED, which knows nothing about whether
registration succeeded, so it cannot be tuned to flatter the result -- unlike
excluding the frames that happened to fail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EVENTS = Path("outputs/aligned_events.json")
BEFORE_S = (1.0, 6.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/raw_clips/fullgame.mp4")
    parser.add_argument("--weights",
                        default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    from courtvision.device import resolve_device

    if not EVENTS.exists() or not Path(args.weights).exists():
        print("FAIL - need aligned events and weights")
        return 1
    shots = [e for e in json.load(EVENTS.open())["events"]
             if "Shot" in (e.get("action") or "")]
    model = YOLO(args.weights)
    device = resolve_device()
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"FAIL - cannot open {args.video}")
        return 1

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(shots), size=min(args.samples, len(shots)),
                        replace=False)
    registered, total, landmarks = 0, 0, []
    for k in chosen:
        at = shots[k]["video_s"] - rng.uniform(*BEFORE_S)
        if at < 60:
            continue
        capture.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        total += 1
        result = model.predict(frame, device=device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        xy = result.keypoints.xy[0].cpu().numpy()
        conf = (result.keypoints.conf[0].cpu().numpy()
                if result.keypoints.conf is not None else np.ones(len(xy)))
        seen = {i: tuple(xy[i]) for i in range(len(xy))
                if i in KEYPOINTS and conf[i] >= args.conf and (xy[i] > 0).all()}
        landmarks.append(len(seen))
        if homography_from_keypoints(seen)[0] is not None:
            registered += 1

    print(f"{args.video}: {len(shots)} aligned shots, {total} sampled")
    print(f"  registered   {registered}/{total} = {registered/max(total,1):.1%}"
          f"   [gate 90%; painted key on this same sampling: 17%]")
    if landmarks:
        print(f"  landmarks    p50 {np.median(landmarks):.0f} of {len(KEYPOINTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
