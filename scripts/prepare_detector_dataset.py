"""Build the V3 YOLO dataset (player / ball / rim) from the Roboflow export.

Source: `roboflow-jvuqo/basketball-player-detection-3-ycjdo` v18 on Roboflow
Universe — CC BY 4.0, 654 broadcast basketball frames with ten classes. It is the
first source found that labels the **ball**, which is what stage 5 (possession)
needs and what COCO `sports ball` could not deliver: yolo11n found the ball in
0/104 frames, and yolo11x only reached ~67% coverage with false positives below
conf 0.05.

Its ten classes collapse onto the spec's three. Note `referee` is DROPPED rather
than mapped to `player` — refs are on court but are not players, and teaching the
detector to ignore them is exactly the improvement over stock COCO that showed up
in the V2 overlays (which boxed refs and courtside fans).

An earlier version of this script used SportsMOT, which labels players only. That
is superseded: mixing it in would teach "no ball here" on every SportsMOT frame,
actively harming the ball class.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SOURCE = Path("data/labeled/roboflow/bpd18")
ROOT = Path("data/labeled/detector")

# Our dense class order; must match courtvision.detection.FINETUNED_CLASS_MAP.
PLAYER, BALL, RIM, HANDLER = 0, 1, 2, 3
OUT_NAMES = ["player", "ball", "rim", "handler"]

# Roboflow class index -> ours. Anything absent is dropped.
REMAP = {
    0: BALL,     # ball
    1: BALL,     # ball-in-basket — still a ball
    # 2: number — jersey digits, that is v2 OCR territory
    3: PLAYER,   # player
    # The three classes below all mark the player CONTROLLING the ball, in
    # different states. Merged into one `handler` class they give 191 training
    # instances instead of 86, and they answer stage 5's question directly:
    # possession by proximity is under-determined on broadcast footage (a
    # defender sits within 0.21 body-heights of the handler in the median frame),
    # so a learned handler beats any distance rule.
    4: HANDLER,  # player-in-possession
    5: HANDLER,  # player-jump-shot
    6: HANDLER,  # player-layup-dunk
    # 7: player-shot-block is the DEFENDER contesting a shot, not the handler,
    # so it stays a plain player.
    7: PLAYER,
    # 8: referee — deliberately dropped, see module docstring
    9: RIM,      # rim
}


def convert(split_in: str, split_out: str) -> tuple[int, dict[int, int]]:
    src_images = SOURCE / split_in / "images"
    src_labels = SOURCE / split_in / "labels"
    dst_images = ROOT / "images" / split_out
    dst_labels = ROOT / "labels" / split_out
    for d in (dst_images, dst_labels):
        d.mkdir(parents=True, exist_ok=True)

    counts: dict[int, int] = {PLAYER: 0, BALL: 0, RIM: 0, HANDLER: 0}
    n = 0
    for image_path in sorted(src_images.glob("*.jpg")):
        label_path = src_labels / (image_path.stem + ".txt")
        lines = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                mapped = REMAP.get(int(parts[0]))
                if mapped is None:
                    continue
                counts[mapped] += 1
                lines.append(" ".join([str(mapped), *parts[1:5]]))
        shutil.copy(image_path, dst_images / image_path.name)
        (dst_labels / (image_path.stem + ".txt")).write_text("\n".join(lines) + "\n")
        n += 1
    return n, counts


HARVEST = Path("data/labeled/handler_harvest")


def merge_harvest() -> int:
    """Append weakly-supervised handler frames to TRAIN only.

    Validation stays purely human-annotated so mAP keeps measuring against real
    labels rather than against the geometric rule that produced these.
    """
    if not (HARVEST / "images").is_dir():
        return 0
    dst_images = ROOT / "images" / "train"
    dst_labels = ROOT / "labels" / "train"
    added = 0
    for image_path in sorted((HARVEST / "images").glob("*.jpg")):
        label_path = HARVEST / "labels" / (image_path.stem + ".txt")
        if not label_path.exists():
            continue
        shutil.copy(image_path, dst_images / f"harvest_{image_path.name}")
        shutil.copy(label_path, dst_labels / f"harvest_{label_path.name}")
        added += 1
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V3 dataset")
    parser.add_argument("--no-harvest", action="store_true",
                        help="skip the weakly-supervised handler frames")
    args = parser.parse_args()

    if not SOURCE.is_dir():
        print(f"FAIL — no Roboflow export at {SOURCE}")
        return 1

    if ROOT.exists():
        shutil.rmtree(ROOT)

    totals: dict[int, int] = {PLAYER: 0, BALL: 0, RIM: 0, HANDLER: 0}
    images = 0
    for split_in, split_out in (("train", "train"), ("valid", "val")):
        n, counts = convert(split_in, split_out)
        images += n
        for k, v in counts.items():
            totals[k] += v
        print(f"  {split_out}: {n} images, "
              + ", ".join(f"{OUT_NAMES[k]}={v}" for k, v in sorted(counts.items())))

    if not args.no_harvest:
        added = merge_harvest()
        if added:
            print(f"  + {added} weakly-supervised handler frames added to train")

    (ROOT / "data.yaml").write_text(
        f"path: {ROOT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(OUT_NAMES)}\n"
        f"names: {OUT_NAMES}\n"
    )
    print(f"\nwrote {ROOT}/data.yaml (nc={len(OUT_NAMES)}, names={OUT_NAMES})")
    print(f"{images} images; boxes: "
          + ", ".join(f"{OUT_NAMES[k]}={v}" for k, v in sorted(totals.items())))
    print("\nSource: Roboflow Universe roboflow-jvuqo/"
          "basketball-player-detection-3-ycjdo v18 (CC BY 4.0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
