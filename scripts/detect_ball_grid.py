"""Re-detect the ball on the evaluation grid at a chosen inference size.

The cached whole-game detections were made at the detector's default size, and
on that cache a ball box at 0.25 confidence appears on 53-62% of frames. No
selection rule can beat the candidates it is given, so before any more work on
choosing, the CEILING has to be measured: on the frames where a ball is really
there, does ANY candidate land on it?

Inference size is the lever this project has already seen move it -- 1280 to
2560 took ball coverage 0.733 to 0.892 in an earlier round -- and it is cheap
to test here because the grid is a few hundred frames, not a whole game. If a
larger size lifts the ceiling, the whole-game pass is worth its hours; if it
does not, those hours would have bought nothing and the work belongs in
selection instead.

Nothing is chosen here. Every candidate is written out with its confidence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=2560)
    parser.add_argument("--conf", type=float, default=0.05,
                        help="floor; deliberately low, so the ceiling is not "
                             "capped by a threshold chosen before it was known")
    parser.add_argument("--times", default=None,
                        help="JSON with a 'frames' list of {t}; default is the grid")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device
    from courtvision.provenance import code_provenance

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_rim_and_ball import SAMPLE_EVERY_S, sample_times

    capture = cv2.VideoCapture(args.video)
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    if args.times:
        times = [row["t"] for row in json.load(open(args.times))["frames"]]
    else:
        times = sample_times(duration, SAMPLE_EVERY_S)

    model, device = YOLO(args.detector), resolve_device()
    names = model.names
    rows, started = [], time.time()
    for n, t in enumerate(times):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        found = model.predict(frame, device=device, verbose=False,
                              imgsz=args.imgsz, conf=args.conf)[0].boxes
        boxes = []
        if found is not None and len(found):
            for cls, conf, box in zip(found.cls.cpu().numpy(),
                                      found.conf.cpu().numpy(),
                                      found.xyxy.cpu().numpy()):
                boxes.append({"cls": names[int(cls)], "conf": round(float(conf), 3),
                              "xyxy": [round(float(v), 1) for v in box]})
        rows.append({"t": round(float(t), 3), "boxes": boxes})
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(times)} frames, "
                  f"{(time.time() - started) / 60:.1f} min", flush=True)
    capture.release()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "detector": args.detector, "imgsz": args.imgsz,
               "conf": args.conf, "fps": None, "frames": rows,
               **code_provenance(__file__)}, open(out, "w"))
    balls = [sum(1 for b in r["boxes"] if b["cls"] == "ball") for r in rows]
    for floor in (0.10, 0.25, 0.50):
        have = sum(1 for r in rows
                   if any(b["cls"] == "ball" and b["conf"] >= floor for b in r["boxes"]))
        print(f"imgsz {args.imgsz}: ball box at conf>={floor:.2f} on "
              f"{have}/{len(rows)} frames ({have / max(len(rows), 1):.1%})")
    print(f"{np.mean(balls):.2f} ball candidates a frame at conf>={args.conf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
