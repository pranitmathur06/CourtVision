"""V3 — Fine-tuning loop sanity check (spec §6).

Fine-tunes YOLO on the small labeled subset and compares mAP50-95 on the held-out
val split against the stock COCO baseline. Passing proves the training loop and
data pipeline work; it does not prove the detector is good enough for production.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

from courtvision.device import resolve_device

DATA_YAML = Path("data/labeled/detector/data.yaml")
CHECKPOINT = Path("checkpoints/detector.pt")
# Weak supervision grew the training set from 464 to 3,823 images, so the epoch
# count comes down proportionally: 12 epochs here is still ~2x the gradient steps
# of the original 25 epochs on 464 images, at a fraction of the wall clock.
EPOCHS = 12
IMG_SIZE = 640
BATCH = 8  # 16 GB unified memory is shared with the OS, but nano at 640px fits


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 detector fine-tuning gate")
    parser.add_argument(
        "--reuse-checkpoint", action="store_true",
        help="evaluate the existing checkpoint instead of retraining, so the gate "
             "can be re-verified without repeating a long fine-tune",
    )
    args = parser.parse_args()

    if not DATA_YAML.exists():
        print(f"V3 FAIL — no dataset at {DATA_YAML}; "
              "run scripts/prepare_detector_dataset.py")
        return 1

    device = resolve_device()

    # Baseline: stock COCO weights evaluated on our val split. It knows nothing
    # about "rim", so this number is expected to be low - that is the point.
    baseline = YOLO("yolo11n.pt").val(
        data=str(DATA_YAML), device=device, imgsz=IMG_SIZE, verbose=False
    )
    baseline_map = float(baseline.box.map)

    if args.reuse_checkpoint:
        if not CHECKPOINT.exists():
            print(f"V3 FAIL — --reuse-checkpoint but no checkpoint at {CHECKPOINT}")
            return 1
        model = YOLO(str(CHECKPOINT))
    else:
        model = YOLO("yolo11n.pt")
        model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=device,
            project="outputs/train",
            name="detector",
            exist_ok=True,
            verbose=False,
        )
        # Ultralytics resolves `project` under its own runs_dir, so the weights
        # are NOT at outputs/train/... . Ask the trainer where it actually saved
        # them rather than reconstructing the path and guessing wrong.
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        best = Path(getattr(model.trainer, "best", ""))
        if best.exists():
            shutil.copy(best, CHECKPOINT)
        else:
            print(f"V3 FAIL — trainer reported no best.pt (looked at {best})")
            return 1

    tuned = model.val(data=str(DATA_YAML), device=device, imgsz=IMG_SIZE, verbose=False)
    tuned_map = float(tuned.box.map)

    ok = tuned_map > baseline_map and CHECKPOINT.exists()
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V3 {verdict} — mAP50-95 baseline {baseline_map:.4f} -> fine-tuned "
        f"{tuned_map:.4f} on {EPOCHS} epochs; checkpoint {CHECKPOINT}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
