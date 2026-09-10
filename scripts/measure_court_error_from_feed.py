"""Absolute registration error on broadcast, with the shooter identified.

The earlier feed check matched each shot to the NEAREST detected player, so its
5.48 ft conflated registration error with not knowing who shot. The feed names
the shooter and the roster gives his number, so jersey OCR can pick him out --
turning a nearest-neighbour guess into a correspondence.

Remaining noise is the foot position of a detector box, the gap between the
recorded event and the release, and the feed's own precision. So this is still
an upper bound on registration error, but a much tighter one.

A control reports the same distance for a DIFFERENT player in the same frame.
If identifying the shooter did not help, the two are alike and the jersey step
is doing nothing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

SHOTS = Path("data/pbp/shotchart_0042400407.json")
JERSEYS = Path("data/pbp/jerseys_0042400407.json")
EVENTS = Path("outputs/aligned_events.json")
RELEASE_LEAD_S = 0.6


def _court(loc_x, loc_y, far):
    x, y = 25.0 + loc_x / 10.0, 5.25 + loc_y / 10.0
    return (50.0 - x, 94.0 - y) if far else (x, y)


def _surname(name):
    return re.sub(r"[^a-z]", "", name.split()[-1].lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/raw_clips/fullgame.mp4")
    parser.add_argument("--weights",
                        default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--limit", type=int, default=140)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--digit-floor", type=float, default=0.75,
                        help="jersey confidence; higher trades matches for "
                             "certainty, and a wrong identification adds a "
                             "large error rather than a small one")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    from courtvision.device import resolve_device
    from courtvision.digit_net import DigitReader

    shots = json.load(SHOTS.open())
    numbers = json.load(JERSEYS.open())
    events = [e for e in json.load(EVENTS.open())["events"]
              if "Shot" in (e.get("action") or "")]

    used, pairs = set(), []
    for row in shots:
        want = _surname(row["PLAYER_NAME"])
        for n, event in enumerate(events):
            if n in used:
                continue
            if want and want in re.sub(r"[^a-z]", "", event["description"].lower()):
                pairs.append((row, event))
                used.add(n)
                break

    model = YOLO(args.weights)
    detector = YOLO(args.detector)
    reader = DigitReader()
    device = resolve_device()
    capture = cv2.VideoCapture(args.video)

    matched, control, read_ok = [], [], 0
    for row, event in pairs[:args.limit]:
        want = numbers.get(row["PLAYER_NAME"])
        if not want:
            continue
        at = event["video_s"] - RELEASE_LEAD_S
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
        feed = np.array(_court(row["LOC_X"], row["LOC_Y"], far))

        shooter = None
        for n, (x1, y1, x2, y2) in enumerate(xyxy.astype(int)):
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            if crop.size == 0:
                continue
            got = reader.read(crop)
            if got and got[1] >= args.digit_floor and got[0] == want:
                shooter = n
                break
        if shooter is None:
            continue
        read_ok += 1
        matched.append(float(np.hypot(*(court[shooter] - feed))))
        others = [n for n in range(len(court)) if n != shooter]
        if others:
            control.append(float(np.hypot(*(court[others[0]] - feed))))

    print(f"{len(pairs)} paired shots; shooter identified by jersey in {read_ok}")
    if len(matched) < 10:
        print("FAIL - too few identifications to conclude")
        return 1
    m = np.array(matched)
    c = np.array(control)
    print(f"  shooter to feed location   p50 {np.median(m):.2f} ft   "
          f"p90 {np.percentile(m, 90):.2f} ft   n={len(m)}")
    print(f"  a DIFFERENT player         p50 {np.median(c):.2f} ft   "
          f"(if these are alike, identifying the shooter achieved nothing)")
    print(f"  within 3 ft                {np.mean(m < 3):.0%}"
          f"   within 6 ft {np.mean(m < 6):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
