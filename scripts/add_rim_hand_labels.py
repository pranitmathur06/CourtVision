"""Fold hand-located alternate-camera rims into the scale dataset, with weight.

The scale dataset is 8,256 crops the detector labelled itself, and it taught a
model that reads main-camera rims at 0.04 rim widths and finds 1 of 15
alternate-camera rims. The alternate cameras are the whole of the remaining
gap, and no automatic source can supply them: the landmark model gives no
keypoints there and the detector gives no boxes, which is the definition of
the failure.

So 23 of them were located by hand. Against 8,256 synthetic crops they would
vanish, so each is written out REPEATED with independent random crops and
flips. That is a deliberate over-weighting of 23 examples and it can overfit
to them; the check on that is the held-out evaluation frames, which are
excluded from these labels by 30 seconds and were never looked at while
labelling.

Crops are cut WITHOUT rescaling the rim, as in build_rim_dataset.py: these
rings are 24 to 512 px across and their size IS the domain gap.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

CROP = 640
REPEATS = 26


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True, help="existing dataset to add to")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import cv2

    rng = random.Random(args.seed)
    rows = [r for r in json.load(open(args.labels))["frames"] if r.get("train", True)]
    out = Path(args.out)
    capture = cv2.VideoCapture(args.video)
    made = {"train": 0, "val": 0}
    for row in rows:
        capture.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        cx, cy = row["rim"]
        half = row["width"] / 2.0
        box = [cx - half, cy - half * 0.45, cx + half, cy + half * 0.45]
        split = "val" if rng.random() < args.valid_fraction else "train"
        for k in range(args.repeats):
            size = min(CROP, frame.shape[0], frame.shape[1])
            low_x = max(0, int(cx) - size + 12)
            high_x = min(frame.shape[1] - size, int(cx) - 12)
            low_y = max(0, int(cy) - size + 12)
            high_y = min(frame.shape[0] - size, int(cy) - 12)
            if low_x > high_x or low_y > high_y:
                continue
            x0, y0 = rng.randint(low_x, high_x), rng.randint(low_y, high_y)
            if not (x0 <= box[0] and box[2] <= x0 + size
                    and y0 <= box[1] and box[3] <= y0 + size):
                continue
            patch = frame[y0:y0 + size, x0:x0 + size]
            if patch.shape[0] != size or patch.shape[1] != size:
                continue
            bx = ((box[0] + box[2]) / 2 - x0) / size
            by = ((box[1] + box[3]) / 2 - y0) / size
            bw, bh = (box[2] - box[0]) / size, (box[3] - box[1]) / size
            if rng.random() < 0.5:
                patch = cv2.flip(patch, 1)
                bx = 1.0 - bx
            if not (0 < bx < 1 and 0 < by < 1):
                continue
            name = f"hand_{int(row['t'] * 1000)}_{k}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), patch,
                        [cv2.IMWRITE_JPEG_QUALITY, 94])
            with open(out / "labels" / split / f"{name}.txt", "w") as handle:
                handle.write("0 %.6f %.6f %.6f %.6f\n" % (bx, by, bw, bh))
            made[split] += 1
    capture.release()
    print(f"{len(rows)} hand labels -> train {made['train']}, val {made['val']} crops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
