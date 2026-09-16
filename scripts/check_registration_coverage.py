"""Where court registration works, and what it needs in shot to work at all.

`check_registration_consistency.py` answers whether a registration is RIGHT --
two independent paths to the same floor point, ORB-carried, must agree -- and on
these three broadcasts the answer is 0.30 to 0.49 ft, against 5.8 ft for the
painted-key method it replaces. That part is settled and it is good.

This answers the other half: how OFTEN. The consistency check only reports on
instants where both frames registered, so it cannot see the frames that never
got a homography, and those are the gate. Measured here across a whole game,
binned by how much of the picture is floor:

    a lot of floor (>30%)          70% give the 4 landmarks a homography needs
    a normal wide play shot        0%
    a little floor                 0%
    almost none (close-up, replay) 0%

So the failure is not a threshold -- dropping the confidence floor from 0.6 to
0.3 moves coverage 75.0% to 76.6% and nothing else. It is that the landmark
model was trained on frames showing most of the court, and on a tighter shot it
finds nothing rather than finding the few landmarks that are there. The fix is
in its training data, not in this pipeline: scale and crop augmentation, so a
frame containing a quarter of the floor is something it has seen.

Reported per band because the aggregate hides it: a game is mostly wide shots,
so an average over all frames reads as a coverage problem when it is a
generalisation problem with a clean boundary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: A homography needs four points; below this the frame cannot register at all.
NEEDED = 4
BANDS = ((0.00, 0.05, "almost no floor in shot (close-up, crowd, replay)"),
         (0.05, 0.15, "a little floor"),
         (0.15, 0.30, "a normal wide play shot"),
         (0.30, 1.01, "a lot of floor"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/raw_clips/fullgame.mp4")
    parser.add_argument("--weights",
                        default="runs/pose/checkpoints/court_kp_960_ft/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--start", type=float, default=700.0)
    parser.add_argument("--end", type=float, default=7000.0)
    parser.add_argument("--every", type=float, default=60.0)
    args = parser.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO

    from courtvision.candidates import court_region
    from courtvision.device import resolve_device

    if not Path(args.weights).exists():
        print(f"FAIL - no weights at {args.weights}")
        return 1
    model, device = YOLO(args.weights), resolve_device()
    capture = cv2.VideoCapture(args.video)
    rows = []
    when = args.start
    while when < args.end:
        capture.set(cv2.CAP_PROP_POS_MSEC, when * 1000)
        ok, frame = capture.read()
        when += args.every
        if not ok:
            continue
        region = court_region(frame, erode_px=0)
        share = float(region.mean()) if region is not None else 0.0
        result = model.predict(frame, device=device, verbose=False,
                               imgsz=args.imgsz, conf=0.25)[0]
        found = 0
        if (result.keypoints is not None and result.keypoints.conf is not None
                and len(result.keypoints.conf)):
            found = int((result.keypoints.conf[0].cpu().numpy() >= args.conf).sum())
        rows.append((share, found))
    capture.release()

    if not rows:
        print("FAIL - no frames read")
        return 1
    usable = [r for r in rows if r[1] >= NEEDED]
    print(f"{args.video}  {len(rows)} frames sampled every {args.every:.0f}s")
    print(f"  a homography is possible on {len(usable)}/{len(rows)} "
          f"= {len(usable) / len(rows):.0%} of them\n")
    for low, high, name in BANDS:
        band = [r for r in rows if low <= r[0] < high]
        if not band:
            continue
        ok = [r for r in band if r[1] >= NEEDED]
        print(f"  {name:<46} {len(band):>3} frames, "
              f"{len(ok) / len(band):>4.0%} give {NEEDED}+ landmarks "
              f"(median found {int(np.median([r[1] for r in band]))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
