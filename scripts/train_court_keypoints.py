"""Fine-tune a pose model to find court landmarks on broadcast frames.

## The keypoint loss has to be retuned, or nothing is learned at all

Ultralytics scores keypoints with an OKS loss, `1 - exp(-e)`, where
`e = d^2 / ((2*sigma)^2 * area * 2)`. For any keypoint count other than COCO's
17 it invents `sigma = 1/nkpt`, here 1/48 = 0.021. That is a tolerance of
`2*sigma*sqrt(area)`, and `area` for a court is the whole frame -- so the
tolerance is about 24 px while a COCO-pretrained head, which puts body
keypoints near the middle of the box, starts roughly 280 px away.

That puts `e` near 71 at initialization, and `exp(-71)` is zero in float32.
The loss returns a flat 1.0 with **no gradient**, so the keypoint head never
moves. Measured: box mAP50 reached 0.995 while pose mAP stayed at exactly 0.0
and the predicted landmarks sat a median 281 px from truth.

`sigma` is therefore set so that the initial error lands near `e = 1`, where
the gradient is largest: an error of half the object's linear size should not
saturate. Convergence then happens in the quadratic region, which is the regime
COCO pose already trains in.

The painted key registers 17% of frames mid-possession because it needs one
unoccluded quadrilateral, and players stand in the paint. Landmarks spread
across the whole floor degrade instead of failing: occlusion costs a few points
and RANSAC absorbs them.

`yolo11*-pose` is the natural fit -- it already predicts a fixed set of ordered
keypoints per detected instance, which is exactly the court schema's shape, and
the COCO-pretrained backbone transfers because it has learned to localise
structure rather than people specifically.

850 images is small. That is survivable here for a reason it would not be for
an action classifier: a court is a rigid planar object photographed from a
narrow band of broadcast angles, so the model has to generalise over camera
pose and arena colour rather than over the shape of the thing itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATA = Path("data/labeled/court_keypoints/data.yaml")
#: `valid` is the Roboflow split name; the eval script selects its confidence
#: floor there and reports on `test`.
OUT = Path("checkpoints/court_keypoints")


def _resolved_data() -> Path:
    """Point the dataset config at absolute paths, idempotently.

    Roboflow ships `train: ../train/images`, which resolves relative to
    wherever ultralytics happens to think the dataset root is and silently
    fails to find images from anywhere else. The file is gitignored, so a fresh
    clone would otherwise need this fixed by hand before it could train.
    """
    root = DATA.parent.resolve()
    text = DATA.read_text()
    if "path:" not in text:
        for split in ("train", "valid", "test"):
            text = text.replace(f"{split}: ../{split}/images",
                                f"{split}: {split}/images")
        text = f"path: {root}\n" + text
        DATA.write_text(text)
    return DATA.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11s-pose.pt",
                        help="yolo11n/s/m-pose; s is the same trade the player "
                             "detector settled on, 6.3x faster than x for more "
                             "found objects")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960,
                        help="court lines are thin; 640 loses the far ones")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.18,
                        help="OKS tolerance per landmark, as a fraction of the "
                             "court's linear size; the 1/48 ultralytics picks "
                             "gives a dead gradient (see the module docstring). "
                             "Set to 0.0208 to reproduce that failure.")
    parser.add_argument("--device", default=None,
                        help="ultralytics defaults to CPU on this Mac -- 20 s "
                             "per iteration -- so the device is passed "
                             "explicitly, as the rest of the repo does")
    args = parser.parse_args()

    from ultralytics import YOLO

    from courtvision.device import resolve_device

    if not DATA.exists():
        print(f"FAIL - no dataset at {DATA}")
        return 1
    data = _resolved_data()
    model = YOLO(args.model)
    # Read by v8PoseLoss when it builds its criterion, which happens lazily on
    # the first forward pass -- so setting it on the trainer's model at
    # train start is early enough, and setting it on `model` here would not
    # survive the trainer rebuilding from weights.
    def _set_sigmas(trainer):
        import torch
        trainer.model.kpt_oks_sigmas = torch.full((48,), args.sigma)

    model.add_callback("on_train_start", _set_sigmas)
    device = args.device or resolve_device()
    print(f"training on {device}")
    model.train(data=str(data), epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, device=device,
                project=str(OUT.parent), name=OUT.name, exist_ok=True,
                # A court fills the frame, so the aggressive scale/translate
                # augmentation meant for objects moves landmarks out of view.
                scale=0.25, translate=0.05, degrees=0.0, shear=0.0,
                mosaic=0.0, fliplr=0.5, patience=30)
    print(f"  weights in {OUT}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
