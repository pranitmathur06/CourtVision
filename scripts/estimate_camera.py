"""Pass 1 over a game: register sampled frames freely, then solve the camera centre.

The centre is estimated from this game's own registrations -- no annotations --
by court_camera.estimate_centre, and written for every later pass (production
registration, evaluation) to load. Frames from other cameras and wrong fits are
dropped by the joint fit's outlier test, and the report says how many survived.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--detector", default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--offset-s", type=float, default=7.3,
                        help="shift the sample grid off the evaluators' grids, so "
                             "the frames that set the centre are not the ones scored")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    import courtvision.court_refine as court_refine
    from courtvision.court_camera import estimate_centre, floor_signature, game_floor
    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    from courtvision.court_register import register_frame
    from courtvision.court_tracking import has_court
    from courtvision.device import resolve_device
    from courtvision.provenance import code_provenance

    model, detector, device = YOLO(args.weights), YOLO(args.detector), resolve_device()
    capture = cv2.VideoCapture(args.video)
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    fits, times, size, signatures = [], [], None, []
    for t in np.linspace(600 + args.offset_s, duration - 300, args.samples):
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
        matrix, info = register_frame(frame, start, boxes=boxes)
        if info["refined"]:
            fits.append((matrix, court_refine._POINTS[info["support"]]))
            signatures.append(floor_signature(frame, matrix, boxes))
            times.append(float(t))
            size = (frame.shape[1], frame.shape[0])
    camera, report = estimate_centre(fits, size) if size else (None, {"reason": "no fits"})
    out = Path(args.out or f"outputs/camera/{Path(args.video).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "centre": camera.centre.tolist() if camera else None,
               "floor": game_floor(signatures) if signatures else None,
               "size": list(size) if size else None, "report": report, "times": times,
               **code_provenance(court_refine.__file__)}, open(out, "w"), indent=1)
    print(f"{args.video}: {len(fits)} refined frames -> {report}")
    return 0 if camera else 1


if __name__ == "__main__":
    raise SystemExit(main())
