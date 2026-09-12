"""Train a single-class rim detector that has seen rims larger than broadcast size.

The four-class detector finds no rim at all on close-ups and replay cameras --
zero boxes at confidence 0.01 and at every inference size from 160 to 2560 --
and part of that is scale, reproducibly: the same rim on the same main-camera
frame detects at 3x zoom and vanishes at 6x.

`build_rim_dataset.py` makes the training data out of the detector's own
confident boxes, crop-zoomed, so nothing here rests on a labeller's judgement.
Single class deliberately: mixing rim-only crops into the four-class detector
would teach it "no ball and no player here" on every crop, which is the fault
`prepare_detector_dataset.py` already recorded for SportsMOT.

The honest test is NOT this dataset's own validation split, which is made of
the same synthetic crops. It is whether the model fires on the real
alternate-camera frames the pipeline misses -- frames that cannot be in the
training set, because the training set was built from frames where the old
detector was already confident, and on these it finds nothing at all.
"""

from __future__ import annotations

import argparse
from pathlib import Path

OUT = Path("checkpoints/rim_scale")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/rim_scale/rim.yaml")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--fraction", type=float, default=0.45,
                        help="share of the crops to train on; the full set at 640 px "
                             "took 30 min an epoch on this machine, which is more "
                             "than the question is worth until it is known whether "
                             "scale alone closes the gap")
    parser.add_argument("--name", default="rim_scale")
    args = parser.parse_args()

    from ultralytics import YOLO

    from courtvision.device import resolve_device

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=resolve_device(), name=args.name,
                project="runs/detect/outputs/train", exist_ok=True,
                # The crops are already a scale augmentation; letting the
                # trainer scale again mostly undoes the point of them.
                scale=0.2, mosaic=0.3, fraction=args.fraction, verbose=True)
    OUT.mkdir(parents=True, exist_ok=True)
    # Ultralytics nests `project` under its own runs dir, so the weights land
    # one level deeper than `project` says. Look for both.
    candidates = [Path("runs/detect/outputs/train") / args.name / "weights" / "best.pt",
                  Path("runs/detect/runs/detect/outputs/train") / args.name
                  / "weights" / "best.pt"]
    best = next((c for c in candidates if c.exists()), candidates[0])
    if best.exists():
        import shutil
        shutil.copy(best, OUT / "best.pt")
        print(f"copied {best} -> {OUT / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
