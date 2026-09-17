"""A ball dataset at the ball's own scale, positives and negatives together.

The ball is about 26 px across on a 1280-wide broadcast. Training on whole
frames resized to 640 halves that to 13 px, which is most of why the detector
cannot find it: it is being shown something smaller than it will ever be asked
about. Everything here is a 640x640 crop at NATIVE resolution, so a ball is the
size in training that it is at inference, and inference tiles the frame the
same way.

What goes in:

  positives   one crop per ball label, from the corrected 4-class set -- the
              harvested frames now carrying one ball each rather than up to 24
              -- plus this project's own broadcast crops in data/ball_track.
              The ball is placed at a random offset inside the crop rather than
              centred, or the model learns "the ball is in the middle".
  negatives   crops of the stands where the old detector fired at a basketball,
              and crops of the floor and the benches taken from frames whose
              ball is somewhere else entirely. Roughly one negative for every
              two positives, because the failure being fixed is a false
              positive, not a miss.

Held out: crops from within HOLDOUT_S of EVERY hand-located ball instant this
project scores against, so the ball evaluators keep measuring unseen frames.

That used to mean thirteen instants, from `ball_truth_handlocated.json` and
`ball_truth_hard.json`. It is now several hundred: a uniform 25 s grid across
this broadcast, plus the uniform half of the possession round. Holding out only
the original thirteen while scoring against the rest would have trained on the
evaluation set the moment anyone retrained the ball -- the same family of
mistake as a by-file split whose val frames sit half a second from a training
frame. Nothing had retrained yet, so nothing was contaminated; this closes it
before something does.

The holdout is keyed to THIS broadcast. A timestamp in one game is not the same
moment as the same timestamp in another, so pooling every game's instants would
hold out a great deal of usable footage for no reason.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TILE = 640
#: Keep the ball at least this far inside the crop's edge.
MARGIN = 80
#: Seconds around a hand-located truth instant that may not be trained on.
HOLDOUT_S = 20.0
#: Negatives as a share of positives.
NEGATIVE_SHARE = 0.5
SOURCE = Path("data/labeled/detector_v2")
BALL_TRACK = Path("data/ball_track")
BALL_CLASS_IN_SOURCE = 1


#: The broadcast the tile sources are cut from. Holdout instants are taken from
#: this game only, because a timestamp means a different moment in another.
SOURCE_GAME = "2025 Finals G7"


def truth_instants(game: str = SOURCE_GAME):
    """Every instant the ball is scored against, in `game`.

    Missing one of these files is not a warning -- it is a silent leak the next
    retrain would inherit -- so every ball truth file in the directory is read
    by glob rather than by name.
    """
    out = []
    for path in sorted(Path("data/labeling/rim_ball").glob("ball_truth*.json")):
        blob = json.load(open(path))
        # A file with no game named is one of the originals, which are this
        # broadcast. A file naming a different game holds different moments and
        # blocking this broadcast's footage on them would cost a lot for nothing.
        if (blob.get("game") or SOURCE_GAME) != game:
            continue
        out += [float(r["t"]) for r in blob.get("frames", []) if r.get("ball")]
        # 'absent' and 'unknown' frames are scored too -- as the false-alarm
        # denominator and as exclusions -- so they must not be trained on either.
        out += [float(r["t"]) for r in blob.get("absent", [])]
        out += [float(r["t"]) for r in blob.get("unknown", [])]
    for labels in ("data/labels/possession_labels.json",
                   "data/labels/handler_labels.json"):
        path = Path(labels)
        if not path.exists():
            continue
        for row in json.load(open(path))["frames"]:
            if row.get("game") == game and row.get("pick") == "random":
                out.append(float(row["t"]))
    return sorted(set(out))


def blocked_share(instants, holdout=None, span=None):
    """Fraction of the broadcast the holdout puts out of reach, 0 when empty.

    Printed because it is the quantity that decides whether a broadcast can
    still supply training data at all, and it moves every time a labelling
    round lands.
    """
    holdout = HOLDOUT_S if holdout is None else holdout
    if not instants:
        return 0.0
    merged = []
    for point in sorted(instants):
        low, high = point - holdout, point + holdout
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    covered = sum(high - low for low, high in merged)
    reach = span if span else (max(instants) - min(instants)) + 2 * holdout
    return min(1.0, covered / reach) if reach > 0 else 0.0


def crop_around(image, cx, cy, tile=TILE):
    """A tile containing (cx, cy), placed randomly, clipped to the image."""
    height, width = image.shape[:2]
    if width <= tile or height <= tile:
        return None, None
    x = int(random.uniform(cx - tile + MARGIN, cx - MARGIN))
    y = int(random.uniform(cy - tile + MARGIN, cy - MARGIN))
    x = max(0, min(width - tile, x))
    y = max(0, min(height - tile, y))
    return image[y:y + tile, x:x + tile], (x, y)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/ball_tiles")
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    import cv2

    random.seed(args.seed)
    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    held = truth_instants()
    positives = negatives = held_out = 0

    def write(split, name, crop, boxes):
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        lines = [f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in boxes]
        (out / "labels" / split / f"{name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))

    # ---- positives from the corrected 4-class set --------------------------
    for split in ("train", "val"):
        for image_path in sorted((SOURCE / "images" / split).glob("*.jpg")):
            label = SOURCE / "labels" / split / (image_path.stem + ".txt")
            if not label.exists():
                continue
            balls = []
            for line in label.read_text().split("\n"):
                parts = line.split()
                if len(parts) >= 5 and int(parts[0]) == BALL_CLASS_IN_SOURCE:
                    balls.append([float(v) for v in parts[1:5]])
            if not balls:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            for n, ball in enumerate(balls):
                cx, cy = ball[0] * width, ball[1] * height
                crop, origin = crop_around(image, cx, cy)
                if crop is None:
                    continue
                x, y = origin
                boxes = []
                for other in balls:
                    ox, oy = other[0] * width - x, other[1] * height - y
                    ow, oh = other[2] * width, other[3] * height
                    if 0 <= ox <= TILE and 0 <= oy <= TILE:
                        boxes.append([ox / TILE, oy / TILE, ow / TILE, oh / TILE])
                if not boxes:
                    continue
                write(split, f"pos_{image_path.stem}_{n}", crop, boxes)
                positives += 1

    # ---- positives and negatives already cut at 640 in this broadcast ------
    for split in ("train", "val"):
        for image_path in sorted((BALL_TRACK / "images" / split).glob("*.jpg")):
            stamp = image_path.stem.split("_")[1] if "_" in image_path.stem else ""
            stamp = "".join(c for c in stamp if c.isdigit())
            seconds = float(stamp) / 1000.0 if stamp else None
            if (split == "train" and seconds is not None
                    and any(abs(seconds - t) <= HOLDOUT_S for t in held)):
                held_out += 1
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            label = BALL_TRACK / "labels" / split / (image_path.stem + ".txt")
            boxes = []
            if label.exists():
                for line in label.read_text().split("\n"):
                    parts = line.split()
                    if len(parts) >= 5:
                        boxes.append([float(v) for v in parts[1:5]])
            write(split, f"track_{image_path.stem}", image, boxes)
            if boxes:
                positives += 1
            else:
                negatives += 1

    # ---- negatives: the stands, and floor with no ball in the crop ---------
    stands = sorted((SOURCE / "images" / "train").glob("stands_*.jpg"))
    for image_path in stands:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        write("train", f"neg_{image_path.stem}", image, [])
        negatives += 1

    wanted = int(positives * NEGATIVE_SHARE) - negatives
    if wanted > 0:
        frames = sorted((SOURCE / "images" / "train").glob("harvest_*.jpg"))
        random.shuffle(frames)
        for image_path in frames:
            if wanted <= 0:
                break
            label = SOURCE / "labels" / "train" / (image_path.stem + ".txt")
            balls = []
            for line in label.read_text().split("\n") if label.exists() else []:
                parts = line.split()
                if len(parts) >= 5 and int(parts[0]) == BALL_CLASS_IN_SOURCE:
                    balls.append([float(v) for v in parts[1:5]])
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            for _ in range(2):
                x = random.randint(0, max(width - TILE, 0))
                y = random.randint(0, max(height - TILE, 0))
                if any(x <= b[0] * width <= x + TILE and y <= b[1] * height <= y + TILE
                       for b in balls):
                    continue                   # the ball is in this crop
                write("train", f"negfloor_{image_path.stem}_{x}_{y}",
                      image[y:y + TILE, x:x + TILE], [])
                negatives += 1
                wanted -= 1
                break

    (out / "ball.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        "nc: 1\nnames: ['ball']\n")
    train_n = len(list((out / "images" / "train").glob("*.jpg")))
    val_n = len(list((out / "images" / "val").glob("*.jpg")))
    print(f"  {positives} positives, {negatives} negatives")
    print(f"  {train_n} train tiles, {val_n} val tiles")
    print(f"  {held_out} tiles held out as too close to a hand-located frame")
    blocked = blocked_share(held)
    print(f"  the holdout blocks {blocked:.0%} of {SOURCE_GAME}'s span "
          f"({len(held)} scored instants at +/-{HOLDOUT_S:.0f}s)")
    if blocked > 0.6:
        print("  NOTE: this broadcast is now mostly EVALUATION footage. Its "
              "uniform grid\n        is dense enough that holding it out leaves "
              "little to train on --\n        which is correct, and means new "
              "ball training data has to come\n        from a broadcast that is "
              "not scored as heavily.")
    print(f"  -> {out / 'ball.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
