"""Retrain BALL DETECTION on motion-found labels, because ranking cannot help.

On five balls located by eye, the cached detector has no candidate within 28 px
on three of them and already ranks the ball first on the other two. Seven
selection rules have been built and rejected, the last of them a learned patch
ranker -- and no ranker can ever help with the three, because a ranker re-scores
candidates and those balls were never proposed at all. 0.400 is the detector's
ceiling, so the detector is what has to change.

The labels come from `find_ball_tracks.py` (83% correct by eye) and the crops
from `build_ball_detector_dataset.py`, cut without resizing so the ball stays
the 15-25 px it will actually be at inference.

Single class, and used ALONGSIDE the four-class detector rather than replacing
it: the crops carry no player or rim labels, so folding them into the main
detector would teach it "no player here" on every crop -- the fault
prepare_detector_dataset.py recorded for SportsMOT and Round 66 repeated for
the rim.

The honest test is not this dataset's validation split, which is made of the
same tracks. It is the hand-located balls on the evaluation grid, which no
track label touches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

OUT = Path("checkpoints/ball_detector")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/ball_track/ball.yaml")
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="ball_track")
    args = parser.parse_args()

    from ultralytics import YOLO

    from courtvision.device import resolve_device

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=resolve_device(), name=args.name,
                project="runs/detect/outputs/train", exist_ok=True,
                # The crop already fixes the scale; letting the trainer rescale
                # would undo the one thing the dataset is careful about.
                scale=0.15, mosaic=0.5, verbose=True)
    candidates = [Path("runs/detect/outputs/train") / args.name / "weights" / "best.pt",
                  Path("runs/detect/runs/detect/outputs/train") / args.name
                  / "weights" / "best.pt"]
    best = next((c for c in candidates if c.exists()), candidates[0])
    if best.exists():
        import shutil
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, OUT / "best.pt")
        print(f"copied {best} -> {OUT / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
