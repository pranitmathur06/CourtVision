"""Score the ball-handler against every human label, on frames never trained on.

Two rounds of labelling, split the same way both times: the frames sampled
UNIFORMLY are the evaluation and are never trained on, and the frames sampled
because the model was struggling there are the training set. That keeps the
number an estimate of in-game accuracy rather than of performance on hard cases.

    round 1  ball and handler together, 299 frames    150 eval / 149 train
    round 2  handler only, 366 frames                 108 eval / 258 train

A frame counts when the labeller said somebody had the ball -- verdict 'box'
(this box) or 'missing' (somebody has it and the detector drew no box for him).
'missing' counts as a MISS for every method, because it is one: you cannot name
the right box when there is no box. 'nobody' and 'unknown' are excluded.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HANDLER_IOU = 0.5
ROUNDS = (("data/labels/possession_labels.json", "data/labeling/possession/images"),
          ("data/labels/handler_labels.json", "data/labeling/handler/images"))


def wilson(hits, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def near(box, point):
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def load(split):
    """Rows from both rounds, normalised, for 'eval' or 'train'."""
    out = []
    for path, images in ROUNDS:
        if not Path(path).exists():
            continue
        for row in json.load(open(path))["frames"]:
            verdict = row.get("handler_verdict") or row.get("verdict")
            if verdict not in ("box", "missing"):
                continue
            wanted = row["pick"] == "random"
            if (split == "eval") != wanted:
                continue
            out.append({"image": str(Path(images) / row["file"]),
                        "verdict": verdict,
                        "box": row.get("handler_box"),
                        "at": row.get("handler_at"),
                        "game": row["game"], "t": row["t"]})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--ball", default="checkpoints/ball_v2/best.pt")
    parser.add_argument("--split", default="eval", choices=["eval", "train"])
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device

    rows = load(args.split)
    device = resolve_device()
    detector = YOLO(args.detector)
    ball_model = YOLO(args.ball)

    hits = {"handler_class": 0, "nearest_ball": 0}
    nobox = nohandler = scored = 0
    for row in rows:
        image = cv2.imread(row["image"])
        if image is None:
            continue
        scored += 1
        if row["verdict"] == "missing":
            nobox += 1
            continue                       # a miss for every method, by definition
        truth = row["box"]
        found = detector.predict(image, device=device, verbose=False,
                                 imgsz=args.imgsz, conf=0.25)[0].boxes
        players, handlers = [], []
        if found is not None and len(found):
            for cls, conf, box in zip(found.cls.cpu().numpy(),
                                      found.conf.cpu().numpy(),
                                      found.xyxy.cpu().numpy()):
                name = detector.names[int(cls)]
                value = [float(v) for v in box]
                if name in ("player", "handler"):
                    players.append(value)
                if name == "handler":
                    handlers.append((float(conf), value))
        if handlers:
            if iou(max(handlers)[1], truth) >= HANDLER_IOU:
                hits["handler_class"] += 1
        else:
            nohandler += 1
        seen = ball_model.predict(image, device=device, verbose=False,
                                  imgsz=args.imgsz, conf=0.05)[0].boxes
        point = None
        if seen is not None and len(seen):
            top = max(zip(seen.conf.cpu().numpy(), seen.xyxy.cpu().numpy()),
                      key=lambda e: e[0])[1]
            point = ((top[0] + top[2]) / 2, (top[1] + top[3]) / 2)
        if point and players:
            if iou(min(players, key=lambda p: near(p, point)), truth) >= HANDLER_IOU:
                hits["nearest_ball"] += 1

    print(f"  {scored} frames in the '{args.split}' split where somebody had the ball")
    print(f"    {nobox} of them the detector drew no box for him at all")
    print(f"    {nohandler} it drew no handler box anywhere\n")
    for key, label in (("handler_class", "the detector's handler class"),
                       ("nearest_ball", "the player nearest the predicted ball")):
        low, high = wilson(hits[key], scored)
        print(f"    {label:<38} {hits[key]:>3}/{scored} = {hits[key] / max(scored, 1):5.1%}"
              f"   (95% CI {low:.0%}-{high:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
