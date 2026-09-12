"""A rim dataset for large rims, labelled by the detector it is meant to fix.

The gap measured in Round 67 is that the detector finds no rim at all on
close-ups and replay cameras, at any threshold or inference size. Part of that
is plainly SCALE, and that part is reproducible on the main camera: crop a
frame around its own rim and enlarge it, and the detector holds at 3x and
collapses to nothing at 6x. The same rim, the same pixels, only bigger.

So the training data can be made rather than labelled. On a main-camera frame
the detector already puts a confident, accurate box on the rim. Crop around it
and enlarge, and that box comes along -- a correctly labelled picture of a rim
at a size the detector has never seen. No hand labelling, and no risk of a
labeller's judgement entering a measurement.

What this CANNOT synthesise is viewpoint: an under-basket camera sees the ring
from below, through the net, and no crop of a sideline view produces that.
Whether scale alone closes enough of the gap is the question this dataset
exists to answer, and it is cheap enough to be worth asking before paying for
hand labels.

NEGATIVES matter as much: crops of the same frames with no rim in them, so a
single-class model does not simply learn to fire on anything orange. This
project has been burnt by exactly that.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

#: Only boxes this confident become labels. A wrong label is worse than none.
LABEL_CONF = 0.60
#: Zoom factors to synthesise. 1 keeps the original scale in the mix so the
#: model does not forget the size it already handles.
ZOOMS = (1.0, 1.6, 2.5, 4.0, 6.0, 8.0)
#: Every positive crop is matched by this many negatives.
NEGATIVE_RATIO = 1.0
OUT_SIZE = 640


def crop_around(frame, box, zoom, out_size=OUT_SIZE, jitter=0.25, rng=None):
    """A zoomed crop containing `box`, and the box's place in it (YOLO form)."""
    import cv2
    rng = rng or random.Random(0)
    height, width = frame.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half_w, half_h = width / (2 * zoom), height / (2 * zoom)
    cx += rng.uniform(-jitter, jitter) * half_w
    cy += rng.uniform(-jitter, jitter) * half_h
    x0 = int(np.clip(cx - half_w, 0, width - 2 * half_w))
    y0 = int(np.clip(cy - half_h, 0, height - 2 * half_h))
    x1, y1 = int(x0 + 2 * half_w), int(y0 + 2 * half_h)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None, None
    if not (x0 <= box[0] and box[2] <= x1 and y0 <= box[1] and box[3] <= y1):
        return None, None            # the rim fell outside; that is a negative
    patch = cv2.resize(frame[y0:y1, x0:x1], (out_size, out_size),
                       interpolation=cv2.INTER_CUBIC)
    sx, sy = out_size / (x1 - x0), out_size / (y1 - y0)
    bx0, by0 = (box[0] - x0) * sx, (box[1] - y0) * sy
    bx1, by1 = (box[2] - x0) * sx, (box[3] - y0) * sy
    label = ((bx0 + bx1) / 2 / out_size, (by0 + by1) / 2 / out_size,
             (bx1 - bx0) / out_size, (by1 - by0) / out_size)
    return patch, label


def crop_without(frame, boxes, zoom, out_size=OUT_SIZE, rng=None, tries=12):
    """A crop of the same frame containing none of `boxes`."""
    import cv2
    rng = rng or random.Random(0)
    height, width = frame.shape[:2]
    half_w, half_h = width / (2 * zoom), height / (2 * zoom)
    for _ in range(tries):
        x0 = rng.uniform(0, max(width - 2 * half_w, 1))
        y0 = rng.uniform(0, max(height - 2 * half_h, 1))
        x1, y1 = x0 + 2 * half_w, y0 + 2 * half_h
        if any(not (b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1) for b in boxes):
            continue
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        return cv2.resize(frame[int(y0):int(y1), int(x0):int(x1)],
                          (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--label-conf", type=float, default=LABEL_CONF)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2

    rng = random.Random(args.seed)
    cache = json.load(open(args.detections))
    usable = []
    for row in cache["frames"]:
        rims = [b for b in row["boxes"]
                if b["cls"] == "rim" and b["conf"] >= args.label_conf]
        if rims:
            usable.append((row["t"], [b["xyxy"] for b in rims]))
    rng.shuffle(usable)
    usable = usable[:args.frames]
    print(f"{len(usable)} frames carry a rim box at conf >= {args.label_conf}")

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.video)
    made = {"train": 0, "val": 0}
    negatives = 0
    for n, (t, boxes) in enumerate(usable):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        split = "val" if rng.random() < args.valid_fraction else "train"
        for zoom in ZOOMS:
            box = boxes[rng.randrange(len(boxes))]
            patch, label = crop_around(frame, box, zoom, rng=rng)
            if patch is None:
                continue
            stem = f"{int(t * 1000)}_z{zoom:g}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), patch,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(out / "labels" / split / f"{stem}.txt", "w") as handle:
                handle.write("0 %.6f %.6f %.6f %.6f\n" % label)
            made[split] += 1

            if rng.random() < NEGATIVE_RATIO:
                empty = crop_without(frame, boxes, zoom, rng=rng)
                if empty is not None:
                    stem = f"{int(t * 1000)}_z{zoom:g}_neg"
                    cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), empty,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    (out / "labels" / split / f"{stem}.txt").write_text("")
                    made[split] += 1
                    negatives += 1
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(usable)} frames, {sum(made.values())} crops",
                  flush=True)
    capture.release()

    (out / "rim.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: rim\n")
    print(f"train {made['train']}, val {made['val']}, of which {negatives} negatives")
    print(f"  {out / 'rim.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
