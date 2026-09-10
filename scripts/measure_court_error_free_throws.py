"""Absolute registration error, measured where a player's position is known exactly.

A free-throw shooter stands on the line, at (25, 19) in court feet. It is the
one moment in a game when a player's true position is known without tracking,
without identification, and without the feed supplying coordinates at all --
the rules supply them.

Identifying him is robust rather than circular: the other nine players line the
lane and the perimeter, the nearest of them more than six feet away, so a
registration wrong by several feet still picks the right player. A selection
that only worked when the answer was already right would be worthless here, and
this one does not depend on that.

Reported as a signed offset, because the direction matters: a consistent shift
is a systematic error that could in principle be corrected, while a scatter
around zero is per-frame noise that cannot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EVENTS = Path("outputs/aligned_events.json")
LINE = (25.0, 19.0)
#: The shooter is alone at the line; the nearest team-mate stands in the first
#: lane space, about 8 ft away. Anything past this is not the shooter.
CLAIM_FT = 6.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/raw_clips/fullgame.mp4")
    parser.add_argument("--weights",
                        default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--lead", type=float, default=1.2,
                        help="the event is logged at the outcome; the shooter "
                             "is at the line shortly before")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    from courtvision.device import resolve_device

    events = [e for e in json.load(EVENTS.open())["events"]
              if "Free Throw" in (e.get("action") or "")]
    model = YOLO(args.weights)
    detector = YOLO(args.detector)
    device = resolve_device()
    capture = cv2.VideoCapture(args.video)

    offsets, spare = [], []
    for event in events:
        at = event["video_s"] - args.lead
        if at < 60:
            continue
        capture.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
        ok, frame = capture.read()
        if not ok:
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
        boxes = detector.predict(frame, device=device, verbose=False)[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()[boxes.cls.cpu().numpy() == 0]
        if len(xyxy) < 4:
            continue
        feet = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)
        court = cv2.perspectiveTransform(
            feet.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(-1, 2)
        far = float(np.median(court[:, 1])) > 47.0
        line = np.array([50.0 - LINE[0], 94.0 - LINE[1]] if far else LINE)

        distance = np.hypot(*(court - line).T)
        nearest = int(distance.argmin())
        if distance[nearest] > CLAIM_FT:
            continue                      # nobody plausibly at the line
        offsets.append(court[nearest] - line)
        rest = np.delete(distance, nearest)
        if len(rest):
            spare.append(float(rest.min()))

    print(f"{len(events)} free throws, {len(offsets)} with a player at the line")
    if len(offsets) < 10:
        print("FAIL - too few")
        return 1
    off = np.array(offsets)
    magnitude = np.hypot(*off.T)
    print(f"  offset across the court   {np.median(off[:,0]):+.2f} ft "
          f"+/- {off[:,0].std():.2f}")
    print(f"  offset along the court    {np.median(off[:,1]):+.2f} ft "
          f"+/- {off[:,1].std():.2f}")
    print(f"  distance from the line    p50 {np.median(magnitude):.2f} ft   "
          f"p90 {np.percentile(magnitude,90):.2f} ft")
    if spare:
        print(f"  next-nearest player       p50 {np.median(spare):.2f} ft "
              f"(the margin the identification had to work with)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
