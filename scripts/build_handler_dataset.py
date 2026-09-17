"""Add human handler labels to the detector's set, and take the rule out.

The handler class names the wrong player 35% of the time and no rule downstream
rescues it, because it was TAUGHT a rule: `harvest_handler_labels.py` calls
whoever is nearest the most confident ball the handler, and 4,032 of the
training set's 4,032 handler boxes come from that. Proximity's own ceiling,
measured with the ball position supplied by hand, is 59%. A model cannot beat
its teacher.

So two changes, and the first matters as much as the second:

  THE RULE'S LABELS STOP BEING HANDLER LABELS. Every harvested handler box
  becomes a plain player box. That is not throwing data away -- the box is a
  real player and stays one -- it is refusing to call it a handler on the word
  of a rule that is wrong two times in five.

  HUMAN LABELS BECOME THE HANDLER CLASS. 239 frames across two rounds where a
  person said, cold and without seeing any model's guess, who had the ball.
  They are repeated REPEATS times, because 239 instances among 6,000 images is
  a class the trainer would barely see; the augmentation pipeline makes each
  copy a different crop, scale and flip.

A frame the labeller marked 'missing' -- somebody has it and the detector drew
no box for him -- carries only a click, not a box. A box is estimated from the
players standing at the same depth in that frame, and only when at least two of
them are there to estimate from. Those frames are the ones where the player
class fails, so they are worth the approximation; MISSING_SCALE keeps it modest.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

PLAYER, BALL, RIM, HANDLER = 0, 1, 2, 3
#: How many times each human-labelled frame appears in the training set.
REPEATS = 8
#: A 'missing' handler's box is estimated from players within this many pixels
#: of his feet in y -- the ones standing at the same depth, so the same size.
SAME_DEPTH_PX = 70.0
MISSING_SCALE = 1.0
ROUNDS = (("data/labels/possession_labels.json", "data/labeling/possession/images"),
          ("data/labels/handler_labels.json", "data/labeling/handler/images"))


def read_labels(path):
    boxes = defaultdict(list)
    for line in path.read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            boxes[int(parts[0])].append([float(v) for v in parts[1:5]])
    return boxes


def write_labels(path, boxes):
    lines = [f"{cls} {' '.join(f'{v:.6f}' for v in box)}"
             for cls in sorted(boxes) for box in boxes[cls]]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def to_yolo(box, width, height):
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
            (x2 - x1) / width, (y2 - y1) / height]


def training_rows():
    """Human-labelled frames NOT in the evaluation half."""
    out = []
    for path, images in ROUNDS:
        if not Path(path).exists():
            continue
        for row in json.load(open(path))["frames"]:
            verdict = row.get("handler_verdict") or row.get("verdict")
            if verdict not in ("box", "missing") or row["pick"] == "random":
                continue
            out.append({"image": Path(images) / row["file"], "verdict": verdict,
                        "box": row.get("handler_box"), "at": row.get("handler_at"),
                        "boxes": row.get("boxes")})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/labeled/detector_v2")
    parser.add_argument("--out", default="data/labeled/detector_v3")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device

    source, out = Path(args.source), Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ---- carry the set over, demoting every harvested handler to a player ---
    demoted = carried = 0
    for split in ("train", "val"):
        for image in sorted((source / "images" / split).glob("*.jpg")):
            label = source / "labels" / split / (image.stem + ".txt")
            boxes = read_labels(label) if label.exists() else defaultdict(list)
            if boxes.get(HANDLER):
                boxes[PLAYER] = boxes.get(PLAYER, []) + boxes[HANDLER]
                demoted += len(boxes[HANDLER])
                boxes[HANDLER] = []
            shutil.copy(image, out / "images" / split / image.name)
            write_labels(out / "labels" / split / (image.stem + ".txt"), boxes)
            carried += 1
    print(f"  {carried} frames carried over; {demoted} handler boxes from the "
          f"proximity rule demoted to plain players")

    # ---- the human frames, repeated -----------------------------------------
    detector = YOLO("runs/detect/outputs/train/detector/weights/best.pt")
    ball_model = YOLO("checkpoints/ball_v2/best.pt")
    device = resolve_device()

    rows = training_rows()
    added = estimated = skipped = 0
    for row in rows:
        frame = cv2.imread(str(row["image"]))
        if frame is None:
            skipped += 1
            continue
        height, width = frame.shape[:2]
        people = [b for b in (row["boxes"] or [])]
        handler = row["box"]
        if row["verdict"] == "missing":
            point = row["at"]
            if not point or not people:
                skipped += 1
                continue
            # players standing at the same depth are the same size on screen
            near = [b for b in people if abs(b[3] - point[1]) <= SAME_DEPTH_PX]
            if len(near) < 2:
                skipped += 1
                continue
            widths = sorted(b[2] - b[0] for b in near)
            heights = sorted(b[3] - b[1] for b in near)
            w = widths[len(widths) // 2] * MISSING_SCALE
            h = heights[len(heights) // 2] * MISSING_SCALE
            handler = [point[0] - w / 2, point[1] - h / 2,
                       point[0] + w / 2, point[1] + h / 2]
            estimated += 1
        if not handler:
            skipped += 1
            continue

        boxes = defaultdict(list)
        boxes[HANDLER].append(to_yolo(handler, width, height))
        for b in people:
            # the handler's own box is not also a player box
            if row["verdict"] == "box" and row["box"] and abs(b[0] - row["box"][0]) < 1 \
                    and abs(b[1] - row["box"][1]) < 1:
                continue
            boxes[PLAYER].append(to_yolo(b, width, height))
        # the ball and the rim are not in these labels, and an unlabelled ball
        # teaches "ball: background", so they are filled in confidently
        found = detector.predict(frame, device=device, verbose=False,
                                 imgsz=1280, conf=0.5)[0].boxes
        if found is not None and len(found):
            for cls, box in zip(found.cls.cpu().numpy(), found.xyxy.cpu().numpy()):
                if detector.names[int(cls)] == "rim":
                    boxes[RIM].append(to_yolo([float(v) for v in box], width, height))
        seen = ball_model.predict(frame, device=device, verbose=False,
                                  imgsz=1280, conf=0.5)[0].boxes
        if seen is not None and len(seen):
            for box in seen.xyxy.cpu().numpy():
                boxes[BALL].append(to_yolo([float(v) for v in box], width, height))

        for copy in range(args.repeats):
            name = f"hand_{row['image'].stem}_{copy}"
            shutil.copy(row["image"], out / "images" / "train" / f"{name}.jpg")
            write_labels(out / "labels" / "train" / f"{name}.txt", boxes)
            added += 1

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        "nc: 4\nnames: ['player', 'ball', 'rim', 'handler']\n")
    counts = defaultdict(int)
    for label in (out / "labels" / "train").glob("*.txt"):
        for cls, items in read_labels(label).items():
            counts[cls] += len(items)
    names = {PLAYER: "player", BALL: "ball", RIM: "rim", HANDLER: "handler"}
    print(f"  {len(rows) - skipped} human frames added {args.repeats}x = {added} images "
          f"({estimated} with a box estimated for a handler the detector missed, "
          f"{skipped} unusable)")
    print("  train set now: "
          + ", ".join(f"{names[c]} {counts[c]}" for c in sorted(counts)))
    print(f"  -> {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
