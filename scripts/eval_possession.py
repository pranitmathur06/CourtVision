"""Score the ball and the ball-handler against 299 hand-labelled frames.

THE SPLIT IS THE SAMPLING, and it is not arbitrary. Half the labelled frames
were drawn uniformly at random from the three broadcasts and half from the
frames where the ball model's best candidate was under 0.35 -- where it was
failing. So:

    the RANDOM half measures     -- an unbiased estimate of in-game accuracy,
                                    and it is never trained on
    the HARD half trains         -- the failures, which is where a label buys
                                    the most

Nothing here is measured on a frame that trained anything.

WHAT IS SCORED

  ball      the model's most confident candidate, against the hand-located
            centre, within TOLERANCE_PX. Frames the labeller marked "in here
            but I can't find it" are excluded from BOTH sides: counting them
            as misses would punish the model for frames where the truth is
            unavailable, and counting them as passes would flatter it.
  handler   whether the box the pipeline would call the ball handler is the
            one the labeller pointed at, by overlap. Frames where nobody has
            the ball are excluded; frames where the labeller said the detector
            drew no box for the handler are counted as a MISS, because they
            are one.

Wilson intervals throughout, because 130 frames is enough to separate 0.95
from 0.80 and not enough to pretend at a third digit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from courtvision.stats import wilson  # noqa: E402

TOLERANCE_PX = 28.0
#: A predicted handler box overlapping the labelled one by this much is right.
HANDLER_IOU = 0.5
LABELS = "data/labels/possession_labels.json"
FRAMES = "data/labeling/possession/images"




def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ball", default="checkpoints/ball_v2/best.pt")
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--split", default="random",
                        choices=["random", "hard", "all"])
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device

    rows = json.load(open(LABELS))["frames"]
    if args.split != "all":
        rows = [r for r in rows if r["pick"] == args.split]
    device = resolve_device()
    ball_model = YOLO(args.ball)
    detector = YOLO(args.detector)

    located = unfindable = 0
    ball_right = ball_drawn = 0
    ball_available = 0           # the truth is known AND the model proposed anything
    handler_asked = handler_right = handler_nobox = 0
    for row in rows:
        image = cv2.imread(str(Path(FRAMES) / row["file"]))
        if image is None:
            continue

        # ---- ball -----------------------------------------------------------
        if row["ball_verdict"] == "ball":
            located += 1
            found = ball_model.predict(image, device=device, verbose=False,
                                       imgsz=args.imgsz, conf=args.conf)[0].boxes
            best, top = None, -1.0
            if found is not None and len(found):
                for conf, box in zip(found.conf.cpu().numpy(),
                                     found.xyxy.cpu().numpy()):
                    if float(conf) > top:
                        top = float(conf)
                        best = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            if best is not None:
                ball_drawn += 1
                ball_available += 1
                if math.dist(best, row["ball"]) <= TOLERANCE_PX:
                    ball_right += 1
        elif row["ball_verdict"] == "unknown":
            unfindable += 1

        # ---- handler --------------------------------------------------------
        if row["handler_verdict"] == "missing":
            handler_asked += 1
            handler_nobox += 1
        elif row["handler_verdict"] == "box" and row.get("handler_box"):
            handler_asked += 1
            found = detector.predict(image, device=device, verbose=False,
                                     imgsz=args.imgsz, conf=0.25)[0].boxes
            claimed, top = None, -1.0
            if found is not None and len(found):
                for cls, conf, box in zip(found.cls.cpu().numpy(),
                                          found.conf.cpu().numpy(),
                                          found.xyxy.cpu().numpy()):
                    if detector.names[int(cls)] == "handler" and float(conf) > top:
                        top = float(conf)
                        claimed = [float(v) for v in box]
            if claimed and iou(claimed, row["handler_box"]) >= HANDLER_IOU:
                handler_right += 1

    print(f"  {len(rows)} frames in the '{args.split}' half\n")
    low, high = wilson(ball_right, located)
    print(f"  BALL, hand-located on {located} of them ({unfindable} marked unfindable "
          f"and excluded)")
    print(f"    most confident candidate within {TOLERANCE_PX:.0f} px:  "
          f"{ball_right}/{located} = {ball_right / max(located, 1):.1%}"
          f"   (95% CI {low:.0%}-{high:.0%})")
    print(f"    it proposed nothing at all on {located - ball_drawn} of them")
    if ball_drawn:
        low, high = wilson(ball_right, ball_drawn)
        print(f"    of the {ball_drawn} where it did propose something: "
              f"{ball_right / ball_drawn:.1%}  (95% CI {low:.0%}-{high:.0%})")

    low, high = wilson(handler_right, handler_asked)
    print(f"\n  HANDLER, on {handler_asked} frames where somebody had the ball")
    print(f"    the detector's handler is the right player: "
          f"{handler_right}/{handler_asked} = {handler_right / max(handler_asked, 1):.1%}"
          f"   (95% CI {low:.0%}-{high:.0%})")
    print(f"    of those, {handler_nobox} are frames where it drew no box for him at all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
