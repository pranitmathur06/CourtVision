"""Fine-tune a pose model to find court landmarks on broadcast frames.

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
OUT = Path("checkpoints/court_keypoints")


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
    model = YOLO(args.model)
    device = args.device or resolve_device()
    print(f"training on {device}")
    model.train(data=str(DATA.resolve()), epochs=args.epochs,
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
