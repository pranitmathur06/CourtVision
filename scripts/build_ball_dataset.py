"""Ball training data labelled by the shot chart and the rim, not by hand.

The measured problem is not that the detector misses the ball. At a 0.03 floor
it puts a candidate 3-16 px from the truth on the frames checked. The problem
is RANK: that candidate carries a confidence of 0.05-0.11 and sits 20th to 50th
among 60-95 candidates, and four selection rules have now failed to dig it out
(motion, raw large-inference confidence, two-scale agreement, and the ray test
before them). The fix has to be at the source: the detector must score a real
ball above a head.

Which needs labels, and hand-labelling hundreds of balls is the thing this plan
was right to avoid where an external signal exists. One does:

    a made basket means the ball was AT THE RIM.

The official shot chart sits on the video's timeline from Round 65, and the rim
is the half of this pipeline that works (0.840, and 0.04 rim widths from the
detector's own box on main-camera frames). So within a short window of a made
shot, a candidate close to the rim is the ball -- and every OTHER candidate in
that frame is a hard negative, which is exactly what a ranking problem needs.

Neither the shot chart nor the rim knows anything about which ball candidates
this pipeline finds hard, so neither can flatter the result.

Two things this CANNOT do, stated rather than discovered later:

- It only labels the ball NEAR THE RIM. A detector trained on this alone will
  be better at the one place Phase 2 needs the ball and no better in the
  backcourt, and the evaluation must keep measuring the whole grid, not the
  rim.
- A frame where no candidate is really the ball would have its nearest
  candidate mislabelled. Hence NEAR_RIM_WIDTHS is tight, a candidate must beat
  the runner-up by MARGIN_WIDTHS to be taken at all, and a sample of what it
  produces is rendered for eye-checking before anything is trained on it.

THE RIM MUST COME FROM THE FRAME ITSELF. A first version took it from the
nearest evaluation-grid frame, up to 30 seconds away, and the camera pans: the
sample sheet came back with red boxes on shirts and in the crowd while the real
ball sat visible elsewhere in the same picture. Every label was worthless. The
rim now comes from the same detector call that proposes the balls, so it is
the rim in THIS frame or the frame is skipped.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

#: How far either side of a made shot to look.
WINDOW_S = 0.8
#: A candidate this close to the rim, at such a moment, is the ball.
NEAR_RIM_WIDTHS = 1.6
#: ...and it must be this much closer than the next candidate, so a frame with
#: two things by the rim labels neither.
MARGIN_WIDTHS = 0.8
#: Crops are taken at this size around the rim, then resized.
CROP_PX = 420
OUT_SIZE = 640


def pick_ball(candidates, rim, rim_width, near=NEAR_RIM_WIDTHS, margin=MARGIN_WIDTHS):
    """Index of the candidate that is the ball, or None if it is not clear.

    `candidates` are (x, y) centres. Returns None when nothing is close enough,
    or when two things are close enough to be confused.
    """
    if not len(candidates) or rim is None:
        return None
    distances = np.hypot(np.asarray(candidates)[:, 0] - rim[0],
                         np.asarray(candidates)[:, 1] - rim[1]) / max(rim_width, 1.0)
    order = np.argsort(distances)
    if distances[order[0]] > near:
        return None
    if len(order) > 1 and distances[order[1]] - distances[order[0]] < margin:
        return None
    return int(order[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--shots", required=True, help="align_shots_to_video.py output")

    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=2560)
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--step-s", type=float, default=0.2)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-dir", default=None,
                        help="write a contact sheet of what was labelled, for eyes")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device

    rng = random.Random(args.seed)
    shots = [s for s in json.load(open(args.shots))
             if s.get("made") and s.get("gap_s", 9) <= 2.0]
    print(f"{len(shots)} made baskets placed on the timeline")

    model, device = YOLO(args.detector), resolve_device()
    capture = cv2.VideoCapture(args.video)

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    samples = []
    made = {"train": 0, "val": 0}
    looked = taken = 0

    for shot in shots:
        for offset in np.arange(-WINDOW_S, WINDOW_S + 1e-9, args.step_s):
            t = float(shot["t"] + offset)
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            looked += 1
            found = model.predict(frame, device=device, verbose=False,
                                  imgsz=args.imgsz, conf=args.conf)[0].boxes
            boxes, rims = [], []
            if found is not None and len(found):
                names = model.names
                for cls, conf, box in zip(found.cls.cpu().numpy(),
                                          found.conf.cpu().numpy(),
                                          found.xyxy.cpu().numpy()):
                    if names[int(cls)] == "ball":
                        boxes.append([float(v) for v in box])
                    elif names[int(cls)] == "rim" and float(conf) >= 0.40:
                        rims.append(([float(v) for v in box], float(conf)))
            if not boxes or not rims:
                continue
            best_rim = max(rims, key=lambda r: r[1])[0]
            rim = [(best_rim[0] + best_rim[2]) / 2, (best_rim[1] + best_rim[3]) / 2]
            rim_width = max(best_rim[2] - best_rim[0], 1.0)
            centres = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes]
            index = pick_ball(centres, rim, rim_width)
            if index is None:
                continue
            taken += 1

            # Crop around the rim so the ball sits in a frame the detector will
            # meet at inference, with the hard negatives that share it.
            cx, cy = int(rim[0]), int(rim[1])
            half = CROP_PX // 2
            x0 = int(np.clip(cx - half, 0, frame.shape[1] - CROP_PX))
            y0 = int(np.clip(cy - half, 0, frame.shape[0] - CROP_PX))
            patch = frame[y0:y0 + CROP_PX, x0:x0 + CROP_PX]
            if patch.shape[0] < CROP_PX or patch.shape[1] < CROP_PX:
                continue
            box = boxes[index]
            if not (x0 <= box[0] and box[2] <= x0 + CROP_PX
                    and y0 <= box[1] and box[3] <= y0 + CROP_PX):
                continue
            scale = OUT_SIZE / CROP_PX
            label = (((box[0] + box[2]) / 2 - x0) * scale / OUT_SIZE,
                     ((box[1] + box[3]) / 2 - y0) * scale / OUT_SIZE,
                     (box[2] - box[0]) * scale / OUT_SIZE,
                     (box[3] - box[1]) * scale / OUT_SIZE)
            image = cv2.resize(patch, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_CUBIC)
            split = "val" if rng.random() < args.valid_fraction else "train"
            stem = f"{int(t * 1000)}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 93])
            with open(out / "labels" / split / f"{stem}.txt", "w") as handle:
                handle.write("0 %.6f %.6f %.6f %.6f\n" % label)
            made[split] += 1
            if args.sample_dir and len(samples) < 24:
                shown = image.copy()
                x1, y1 = int((label[0] - label[2] / 2) * OUT_SIZE), \
                    int((label[1] - label[3] / 2) * OUT_SIZE)
                x2, y2 = int((label[0] + label[2] / 2) * OUT_SIZE), \
                    int((label[1] + label[3] / 2) * OUT_SIZE)
                cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(shown, f"{t:.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)
                samples.append(cv2.resize(shown, (320, 320)))
            if taken % 50 == 0:
                print(f"  {taken} labelled from {looked} frames looked at", flush=True)
    capture.release()

    (out / "ball.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: ball\n")
    if args.sample_dir and samples:
        sample_dir = Path(args.sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        grid = [np.hstack(samples[i:i + 4]) for i in range(0, len(samples) // 4 * 4, 4)]
        cv2.imwrite(str(sample_dir / "ball_labels.jpg"), np.vstack(grid))
        print(f"  sample sheet: {sample_dir / 'ball_labels.jpg'}")
    print(f"{looked} frames near made baskets; {taken} carried an unambiguous ball; "
          f"train {made['train']}, val {made['val']}")
    print(f"  {out / 'ball.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
