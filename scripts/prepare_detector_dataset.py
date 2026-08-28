"""Convert SportsMOT basketball detection into a YOLO-format dataset for V3.

Source: `sumeetn/sportsmot-basketball-detection` on Hugging Face — 572 train +
291 val frames of basketball with player bounding boxes. Verified by eye: boxes
are tight on the ten on-court players and exclude referees and the crowd, which
is exactly the failure mode stock COCO `person` shows (see outputs/v2_detections).

Two deliberate scope decisions, both recorded in the plan:

* **player only.** SportsMOT has no `ball` or `rim`. The ball is supplied at
  inference by stock COCO `sports ball` via `CompositeDetector`, so the pipeline
  still gets everything it consumes.
* **no rim.** Nothing downstream reads it — possession, team assignment, tracking
  and render all use `.players()` and `.ball()` only. Training a rim class with
  no data and no consumer would be pure ceremony.

Domain note: SportsMOT is FIBA footage; our clips are NBA broadcast. Camera
framing is comparable (wide court) but this is a real domain gap, and V3's mAP is
measured on SportsMOT's own held-out split, not on NBA frames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/labeled/detector")
REPO = "sumeetn/sportsmot-basketball-detection"
PLAYER_CLASS = 0


def write_split(rows: list[dict], split: str) -> int:
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for row in rows:
        image = cv2.imdecode(
            np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            continue
        height, width = image.shape[:2]
        stem = f"{split}_{row['image_id']:06d}_{written:04d}"
        cv2.imwrite(str(image_dir / f"{stem}.jpg"), image)

        lines = []
        for x, y, w, h in row["objects"]["bbox"]:
            # COCO [x,y,w,h] absolute -> YOLO [cx,cy,w,h] normalised 0-1.
            cx, cy = (x + w / 2.0) / width, (y + h / 2.0) / height
            nw, nh = w / width, h / height
            if not (0 < nw <= 1 and 0 < nh <= 1):
                continue
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            lines.append(f"{PLAYER_CLASS} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1
    return written


def main() -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description="Build the V3 YOLO dataset")
    parser.add_argument("--train", type=int, default=200,
                        help="training frames (spec section 5 wants a small subset first)")
    parser.add_argument("--val", type=int, default=80)
    args = parser.parse_args()

    counts = {}
    for split, source, limit in (
        ("train", "data/train-00000-of-00001.parquet", args.train),
        ("val", "data/validation-00000-of-00001.parquet", args.val),
    ):
        path = hf_hub_download(REPO, source, repo_type="dataset")
        rows = next(pq.ParquetFile(path).iter_batches(batch_size=limit)).to_pylist()
        counts[split] = write_split(rows, split)
        print(f"  {split}: {counts[split]} images -> {ROOT}/images/{split}")

    yaml_path = ROOT / "data.yaml"
    yaml_path.write_text(
        f"path: {ROOT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['player']\n"
    )
    print(f"\nwrote {yaml_path} (nc=1, names=['player'])")
    boxes = 0
    for s in ("train", "val"):
        for f in (ROOT / "labels" / s).glob("*.txt"):
            boxes += len([ln for ln in f.read_text().splitlines() if ln.strip()])
    print(f"{sum(counts.values())} images, {boxes} player boxes total")
    print("\nSource: sumeetn/sportsmot-basketball-detection (Hugging Face).")
    print("Ball comes from stock COCO at inference; rim is not modelled "
          "(no downstream consumer).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
