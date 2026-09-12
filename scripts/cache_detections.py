"""Cache ball, rim and player boxes for a whole game, once.

Shot detection needs the ball's path relative to the rim over the whole
broadcast. Detection is the expensive part and does not change while the
detector does not, so it is done once and written to disk; everything after
it -- rim projection, shot rules, scoring -- reads the cache and is cheap to
iterate on.

Frames are read sequentially (a seek per frame costs more than decoding) and
sampled at --fps. Boxes come from the four-class detector, so the same pass
records where the rim WAS seen, which is the availability the projected rim
has to beat: 0.364 of frames, measured on this broadcast.

Each ball box also carries `over_court`: whether any floor lies beneath it in
the frame. A ball in flight is above the floor but still has court below it;
one detected on a spectator has none. Phase 2's gate asks for no ball
detections in the stands, and on a 300-frame sample of Finals G7 8.3% of ball
boxes above 0.25 confidence failed this test -- 13% of those between 0.15 and
0.30 confidence, 0% above 0.75 -- while 0 of the 16 near the rim did, which is
why the shot detector was unaffected by them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detector", default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.10,
                        help="low: a missed ball cannot be recovered later, a false one can be filtered")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO

    from courtvision.candidates import court_region
    from courtvision.device import resolve_device

    model, device = YOLO(args.detector), resolve_device()
    capture = cv2.VideoCapture(args.video)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    step = max(1, round(source_fps / args.fps))
    names = model.names
    rows, index, kept = [], 0, 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        if index % step:
            index += 1
            continue
        ok, frame = capture.retrieve()
        index += 1
        if not ok:
            break
        t = (index - 1) / source_fps
        found = model.predict(frame, device=device, verbose=False, conf=args.conf)[0].boxes
        row = {"t": round(float(t), 3), "boxes": []}
        region = None
        if found is not None and len(found):
            for cls, conf, box in zip(found.cls.cpu().numpy(), found.conf.cpu().numpy(),
                                      found.xyxy.cpu().numpy()):
                entry = {"cls": names[int(cls)], "conf": round(float(conf), 3),
                         "xyxy": [round(float(v), 1) for v in box]}
                if entry["cls"] == "ball":
                    if region is None:
                        region = court_region(frame, erode_px=0)
                    # Does the box actually touch the floor silhouette? The
                    # first version of this asked whether any court pixel lay
                    # BELOW the ball in its column, which is true of almost
                    # everything in a broadcast frame -- the crowd sits above
                    # the floor -- and duly flagged 0 of 78k ball boxes as off
                    # court, so it certified nothing.
                    x1, y1, x2, y2 = (int(np.clip(v, 0, lim - 1)) for v, lim in
                                      zip(box, (frame.shape[1], frame.shape[0],
                                                frame.shape[1], frame.shape[0])))
                    entry["over_court"] = bool(
                        region is not None and region[y1:y2 + 1, x1:x2 + 1].any())
                row["boxes"].append(entry)
        rows.append(row)
        kept += 1
        if kept % 2000 == 0:
            print(f"  {t / 60:.0f} min, {kept} frames cached", flush=True)
    out = Path(args.out or f"outputs/detections/{Path(args.video).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "fps": args.fps, "conf": args.conf,
               "detector": args.detector, "frames": rows}, open(out, "w"))
    counts = {name: sum(1 for r in rows if any(b["cls"] == name for b in r["boxes"])) for name in names.values()}
    print(f"{kept} frames at {args.fps} fps from {args.video}")
    for name, n in counts.items():
        print(f"  {name:8s} present on {n / max(kept, 1):.3f} of frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
