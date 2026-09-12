"""Does the scale-trained rim detector fire where the old one finds nothing?

The honest test is not the synthetic validation split -- that is made of the
same crops as the training set. It is the real frames the pipeline misses:
alternate cameras and close-ups, where the four-class detector returns zero
rim boxes at confidence 0.01 and at every inference size, and where the
landmark model returns zero keypoints so nothing can register them either.

None of them can be in the training set: that set was built from frames where
the OLD detector was already confident, and on these it is not.

The other half is false alarms. A rim finder that fires on every orange thing
is worse than none, so the same model is run on frames known to contain NO rim
(close-ups of faces, crowd shots) and on main-camera frames where the true rim
position is known from the old detector.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

#: Frames hand-checked in Rounds 67-68 as showing a rim that nothing locates.
MISSED_WITH_RIM = (887.5, 1062.5, 1387.5, 2012.5, 2112.5, 3412.5, 3794.0)
#: Frames hand-checked as showing NO rim at all.
NO_RIM = (712.5, 812.5, 912.5, 2312.5, 2712.5, 5612.5, 5812.5, 6712.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", default="checkpoints/rim_scale/best.pt")
    parser.add_argument("--detections", default="outputs/detections/fullgame.json")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--main-frames", type=int, default=40)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device

    model, device = YOLO(args.weights), resolve_device()
    capture = cv2.VideoCapture(args.video)

    def rims(t):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok:
            return None, None
        found = model.predict(frame, device=device, verbose=False,
                              conf=args.conf, imgsz=args.imgsz)[0].boxes
        out = []
        if found is not None and len(found):
            for conf, box in zip(found.conf.cpu().numpy(), found.xyxy.cpu().numpy()):
                out.append((round(float(conf), 2), [int(v) for v in box]))
        out.sort(key=lambda b: -b[0])
        return frame, out

    print("FRAMES WITH A RIM THAT NOTHING CURRENTLY LOCATES")
    hits = 0
    for t in MISSED_WITH_RIM:
        _, found = rims(t)
        if found is None:
            continue
        hits += bool(found)
        print(f"  t={t:8.1f}  {len(found)} boxes  {found[:2]}")
    print(f"  fired on {hits}/{len(MISSED_WITH_RIM)} (positions still need an eye)\n")

    print("FRAMES WITH NO RIM AT ALL -- every box here is a false alarm")
    alarms = 0
    for t in NO_RIM:
        _, found = rims(t)
        if found is None:
            continue
        alarms += len(found)
        print(f"  t={t:8.1f}  {len(found)} boxes  {found[:2]}")
    print(f"  {alarms} false alarms over {len(NO_RIM)} frames\n")

    print("MAIN-CAMERA FRAMES -- does it agree with the old detector where that works?")
    cache = json.load(open(args.detections))["frames"]
    known = [(r["t"], max((b for b in r["boxes"] if b["cls"] == "rim"),
                          key=lambda b: b["conf"]))
             for r in cache
             if any(b["cls"] == "rim" and b["conf"] >= 0.6 for b in r["boxes"])]
    step = max(1, len(known) // args.main_frames)
    errs, missed = [], 0
    for t, box in known[::step][:args.main_frames]:
        _, found = rims(t)
        if not found:
            missed += 1
            continue
        centre = np.array([(box["xyxy"][0] + box["xyxy"][2]) / 2,
                           (box["xyxy"][1] + box["xyxy"][3]) / 2])
        width = max(box["xyxy"][2] - box["xyxy"][0], 1.0)
        best = min(np.hypot((b[1][0] + b[1][2]) / 2 - centre[0],
                            (b[1][1] + b[1][3]) / 2 - centre[1]) for b in found)
        errs.append(best / width)
    if errs:
        errs = np.array(errs)
        print(f"  {len(errs)} frames scored, {missed} with no box at all; "
              f"p50 {np.median(errs):.2f} rim widths, within 1.0 {(errs <= 1).mean():.0%}")
    capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
