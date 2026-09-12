"""Add verified rim proposals as positives and their rejections as hard negatives.

The scale-trained detector's low-confidence proposals on hard frames are
dominated by ONE mistake: the shooter's square inside the backboard, a bright
rectangle just above the rim, which it reads as a rim at confidences up to 0.80.
Verifying proposals by eye is quick -- twelve judgements a sheet -- and it
produces both halves of what a detector needs: the few that ARE rims, and a
labelled pile of the exact thing it keeps confusing for one.

The negatives matter more than the positives here. A crop centred on a rejected
proposal, written with an EMPTY label file, is the model being told that this
particular rectangle is not a basket.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

CROP = 640
POSITIVE_REPEATS = 10
NEGATIVE_REPEATS = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--verified", required=True)
    parser.add_argument("--held-out", required=True, help="rim truth json, for its held-out times")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import cv2

    rng = random.Random(args.seed)
    data = json.load(open(args.verified))
    truth = json.load(open(args.held_out))
    held = [f["t"] for f in truth["frames"] if not f.get("train", True)]
    # the frames the gate is scored on, recovered from the truth file's own rule
    def allowed(t):
        return all(abs(t - h) > 30.0 for h in held) if held else True

    out = Path(args.out)
    capture = cv2.VideoCapture(args.video)
    made = {"pos": 0, "neg": 0}

    def crops(entry, repeats, positive):
        capture.set(cv2.CAP_PROP_POS_MSEC, entry["t"] * 1000)
        ok, frame = capture.read()
        if not ok:
            return
        cx, cy = entry["rim"]
        w = max(entry["width"], 16.0)
        box = [cx - w / 2, cy - w * 0.22, cx + w / 2, cy + w * 0.22]
        for k in range(repeats):
            size = min(CROP, frame.shape[0], frame.shape[1])
            lo_x, hi_x = max(0, int(cx) - size + 12), min(frame.shape[1] - size, int(cx) - 12)
            lo_y, hi_y = max(0, int(cy) - size + 12), min(frame.shape[0] - size, int(cy) - 12)
            if lo_x > hi_x or lo_y > hi_y:
                return
            x0, y0 = rng.randint(lo_x, hi_x), rng.randint(lo_y, hi_y)
            patch = frame[y0:y0 + size, x0:x0 + size]
            if patch.shape[0] != size or patch.shape[1] != size:
                return
            split = "val" if rng.random() < 0.15 else "train"
            tag = "pos" if positive else "neg"
            name = f"prop_{tag}_{int(entry['t'] * 1000)}_{k}"
            if positive:
                if not (x0 <= box[0] and box[2] <= x0 + size
                        and y0 <= box[1] and box[3] <= y0 + size):
                    continue
                bx = ((box[0] + box[2]) / 2 - x0) / size
                by = ((box[1] + box[3]) / 2 - y0) / size
                bw, bh = (box[2] - box[0]) / size, (box[3] - box[1]) / size
                if not (0 < bx < 1 and 0 < by < 1):
                    continue
                with open(out / "labels" / split / f"{name}.txt", "w") as handle:
                    handle.write("0 %.6f %.6f %.6f %.6f\n" % (bx, by, bw, bh))
            else:
                (out / "labels" / split / f"{name}.txt").write_text("")
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), patch,
                        [cv2.IMWRITE_JPEG_QUALITY, 94])
            made["pos" if positive else "neg"] += 1

    for entry in data["accepted"]:
        if allowed(entry["t"]):
            crops(entry, POSITIVE_REPEATS, True)
    for entry in data["rejected"]:
        if allowed(entry["t"]):
            crops(entry, NEGATIVE_REPEATS, False)
    capture.release()
    print(f"{made['pos']} positive crops, {made['neg']} hard-negative crops added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
