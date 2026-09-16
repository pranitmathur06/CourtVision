"""Rebuild the detector's training set without the fault that taught it the crowd.

THE FAULT. `harvest_handler_labels.py` mines frames where the ball is clearly
nearest one player, uses the single most confident ball detection to decide who
the handler is -- and then writes EVERY ball detection in the frame into the
label file as ground truth (its lines 122-125). So the detector's own false
positives came back as positives. Measured in the shipped training set: 12,075
ball labels across 3,823 images, one image carrying 24 of them, 11,610 of the
12,075 from harvested frames. A basketball game has one ball.

That is the whole explanation for a detector whose most confident ball is right
2 times in 13 on held-out hand-located frames, and which offers 17 candidates a
frame to choose from. It was trained to do that.

THIS REBUILDS THE SET FROM THE SAME SOURCES:

  harvested frames   one ball each -- the one nearest the handler, which is the
                     one the harvester's separation test actually validated.
                     Every other ball box in that frame becomes background,
                     which is the part that teaches suppression.
  public frames      untouched. Human labels from the Roboflow set.
  in-domain balls    data/ball_track: 765 crops from this project's own
                     broadcast, one ball each, already carrying 168 explicit
                     negatives, and never used by the 4-class detector -- it
                     only ever trained a separate ball-only model. Players and
                     rims on those crops are filled in by the current detector
                     at high confidence, because an unlabelled player in a
                     training image teaches "player: background".
  hard negatives     crops of the stands, centred on a ball candidate the old
                     detector fired at up there. Nothing in them is a ball and
                     nothing in them is a player on the floor. This is the
                     counterweight that existed in this project once, as 375
                     images, and was never wired in.

Held out: any frame within HOLDOUT_S of one of the thirteen hand-located ball
instants, so the evaluation in `eval_ball_choice.py` keeps measuring something
the model has not seen.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

PLAYER, BALL, RIM, HANDLER = 0, 1, 2, 3
SOURCE = Path("data/labeled/detector")
BALL_TRACK = Path("data/ball_track")
#: A harvested ball this far from the handler, in handler heights, is not the
#: ball the harvester validated.
BALL_NEAR_HANDLER = 2.5
#: Seconds around a hand-located truth instant that may not be trained on.
HOLDOUT_S = 20.0
#: Hard negatives: a candidate this far above the floor's top edge, and this far
#: from any player, is in the stands.
STANDS_ABOVE_PX = 120.0
STANDS_FROM_PLAYER_PX = 220.0
NEGATIVE_CROP = 640


def read_labels(path: Path):
    boxes = defaultdict(list)
    for line in path.read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            boxes[int(parts[0])].append([float(v) for v in parts[1:5]])
    return boxes


def write_labels(path: Path, boxes):
    lines = [f"{cls} {' '.join(f'{v:.6f}' for v in box)}"
             for cls in sorted(boxes) for box in boxes[cls]]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def one_ball(boxes):
    """Keep the ball the harvester's own test validated, and only that one."""
    balls = boxes.get(BALL, [])
    handlers = boxes.get(HANDLER, [])
    if not balls:
        return []
    if not handlers:
        return balls[:1]
    handler = handlers[0]
    scale = max(handler[3], 1e-6)
    ranked = sorted(balls, key=lambda b: math.dist((b[0], b[1]),
                                                   (handler[0], handler[1])) / scale)
    best = ranked[0]
    near = math.dist((best[0], best[1]), (handler[0], handler[1])) / scale
    return [best] if near <= BALL_NEAR_HANDLER else []


def truth_instants():
    out = []
    for name in ("ball_truth_handlocated", "ball_truth_hard"):
        path = Path("data/labeling/rim_ball") / f"{name}.json"
        if path.exists():
            out += [float(r["t"]) for r in json.load(open(path))["frames"]
                    if r.get("ball")]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/labeled/detector_v2")
    parser.add_argument("--negatives", type=int, default=1200)
    parser.add_argument("--detections", action="append",
                        default=["outputs/clip_detections_g7.json",
                                 "outputs/clip_detections_g1.json",
                                 "outputs/clip_detections_ecf.json"])
    parser.add_argument("--fill-in-domain", action="store_true", default=True,
                        help="pseudo-label players and rims on the ball crops")
    args = parser.parse_args()

    import cv2
    import numpy as np

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ---- the harvested and public frames, with the ball labels corrected ----
    kept_balls = dropped_balls = harvested = public = 0
    for split in ("train", "val"):
        for image in sorted((SOURCE / "images" / split).glob("*.jpg")):
            label = SOURCE / "labels" / split / (image.stem + ".txt")
            if not label.exists():
                continue
            boxes = read_labels(label)
            if image.name.startswith("harvest_"):
                before = len(boxes.get(BALL, []))
                boxes[BALL] = one_ball(boxes)
                kept_balls += len(boxes[BALL])
                dropped_balls += before - len(boxes[BALL])
                harvested += 1
            else:
                public += 1
            shutil.copy(image, out / "images" / split / image.name)
            write_labels(out / "labels" / split / (image.stem + ".txt"), boxes)
    print(f"  {harvested} harvested frames: {kept_balls} ball labels kept, "
          f"{dropped_balls} dropped as the detector's own false positives")
    print(f"  {public} public frames carried over untouched")

    # ---- the in-domain ball crops, with players and rims filled in ----------
    detector = None
    if args.fill_in_domain:
        from ultralytics import YOLO

        from courtvision.device import resolve_device
        detector = YOLO("runs/detect/outputs/train/detector/weights/best.pt")
        device = resolve_device()

    held_out = truth_instants()
    added, skipped_leak, negatives_in_set = 0, 0, 0
    for split in ("train", "val"):
        images = sorted((BALL_TRACK / "images" / split).glob("*.jpg"))
        for image in images:
            # fullgame_1149200.jpg -> 1149200 ms into the broadcast
            stamp = image.stem.split("_")[1].rstrip("neg_") if "_" in image.stem else ""
            seconds = float(stamp) / 1000.0 if stamp.isdigit() else None
            if (split == "train" and seconds is not None
                    and any(abs(seconds - t) <= HOLDOUT_S for t in held_out)):
                skipped_leak += 1
                continue
            frame = cv2.imread(str(image))
            if frame is None:
                continue
            boxes = defaultdict(list)
            source_label = BALL_TRACK / "labels" / split / (image.stem + ".txt")
            if source_label.exists():
                for box in read_labels(source_label).get(0, []):
                    boxes[BALL].append(box)
            if not boxes[BALL]:
                negatives_in_set += 1
            if detector is not None:
                found = detector.predict(frame, device=device, verbose=False,
                                         imgsz=640, conf=0.5)[0].boxes
                height, width = frame.shape[:2]
                if found is not None and len(found):
                    for cls, box in zip(found.cls.cpu().numpy(),
                                        found.xyxy.cpu().numpy()):
                        name = detector.names[int(cls)]
                        if name == "ball":
                            continue          # the human label is the truth here
                        target = {"player": PLAYER, "rim": RIM,
                                  "handler": HANDLER}.get(name)
                        if target is None:
                            continue
                        x1, y1, x2, y2 = [float(v) for v in box]
                        boxes[target].append([(x1 + x2) / 2 / width,
                                              (y1 + y2) / 2 / height,
                                              (x2 - x1) / width,
                                              (y2 - y1) / height])
            name = f"indomain_{split}_{image.stem}.jpg"
            shutil.copy(image, out / "images" / split / name)
            write_labels(out / "labels" / split / (Path(name).stem + ".txt"), boxes)
            added += 1
    print(f"  {added} in-domain ball crops added ({negatives_in_set} of them "
          f"already negatives), {skipped_leak} held out for being within "
          f"{HOLDOUT_S:.0f}s of a hand-located truth frame")

    # ---- hard negatives: the stands, where it kept finding basketballs ------
    wanted = []
    for path in args.detections:
        if not Path(path).exists():
            continue
        cached = json.load(open(path))
        for clip, rows in cached["clips"].items():
            for row in rows:
                court = row.get("court")
                if not court:
                    continue
                people = [b[2:] for b in row["d"] if b[0] in ("p", "h")]
                for b in row["d"]:
                    if b[0] != "b" or b[1] < 0.25:
                        continue
                    cx, cy = (b[2] + b[4]) / 2, (b[3] + b[5]) / 2
                    if cy > court[1] - STANDS_ABOVE_PX:
                        continue
                    if people and min(math.dist((cx, cy),
                                                ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2))
                                      for p in people) < STANDS_FROM_PLAYER_PX:
                        continue
                    wanted.append((path, clip, row["f"], cx, cy, b[1]))
    random.seed(11)
    random.shuffle(wanted)
    print(f"  {len(wanted)} ball candidates found in the stands; "
          f"taking {min(args.negatives, len(wanted))} as hard negatives")

    made = 0
    by_clip = defaultdict(list)
    for path, clip, frame_no, cx, cy, conf in wanted[:args.negatives]:
        by_clip[clip].append((frame_no, cx, cy))
    for clip, wants in by_clip.items():
        source = Path("docs/clips") / clip
        if not source.exists():
            continue
        capture = cv2.VideoCapture(str(source))
        frames = {}
        index = 0
        want_by_frame = defaultdict(list)
        for frame_no, cx, cy in wants:
            want_by_frame[frame_no].append((cx, cy))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in want_by_frame:
                frames[index] = frame.copy()
            index += 1
        capture.release()
        for frame_no, spots in want_by_frame.items():
            frame = frames.get(frame_no)
            if frame is None:
                continue
            height, width = frame.shape[:2]
            # the cache is in 1280x720 coordinates; the clip is 854 wide
            sx, sy = width / 1280.0, height / 720.0
            for cx, cy in spots:
                x = int(max(0, min(width - NEGATIVE_CROP // 2 * 2, cx * sx - NEGATIVE_CROP / 2)))
                y = int(max(0, min(height - 1, cy * sy - NEGATIVE_CROP / 2)))
                crop = frame[y:y + NEGATIVE_CROP, x:x + NEGATIVE_CROP]
                if crop.shape[0] < 64 or crop.shape[1] < 64:
                    continue
                name = f"stands_{clip.replace('.mp4','')}_{frame_no}_{int(cx)}.jpg"
                cv2.imwrite(str(out / "images" / "train" / name), crop)
                (out / "labels" / "train" / (Path(name).stem + ".txt")).write_text("")
                made += 1
    print(f"  {made} hard negatives written from the stands")

    data = out / "data.yaml"
    data.write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 4\n"
        "names: ['player', 'ball', 'rim', 'handler']\n")

    counts = defaultdict(int)
    per_image = defaultdict(int)
    for label in (out / "labels" / "train").glob("*.txt"):
        boxes = read_labels(label)
        for cls, items in boxes.items():
            counts[cls] += len(items)
        per_image[len(boxes.get(BALL, []))] += 1
    names = {PLAYER: "player", BALL: "ball", RIM: "rim", HANDLER: "handler"}
    print("\n  train set now: "
          + ", ".join(f"{names[c]} {counts[c]}" for c in sorted(counts)))
    print(f"  ball labels per image: {dict(sorted(per_image.items()))}")
    print(f"  -> {data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
