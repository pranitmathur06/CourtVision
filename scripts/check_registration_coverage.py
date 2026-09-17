"""Where court registration works, and what it needs in shot to work at all.

`check_registration_consistency.py` answers whether a registration is RIGHT --
two independent paths to the same floor point, ORB-carried, must agree -- and on
these three broadcasts the answer is 0.30 to 0.49 ft, against 5.8 ft for the
painted-key method it replaces. That part is settled and it is good.

This answers the other half: how OFTEN. The consistency check only reports on
instants where both frames registered, so it cannot see the frames that never
got a homography, and those are the gate.

AN EARLIER VERSION OF THIS FILE REPORTED 0% ON EVERY BAND BUT THE WIDEST, AND
THAT WAS TWO MISTAKES AT ONCE.

The first was sample size. The default `--every 60` gives about twenty frames a
band, and a 0/20 has a Wilson upper bound near 16%; it was a small-sample zero
being read as an absolute one. At `--every 10` -- 630 frames, thirty times the
sample -- the normal-wide-play band is 42%, not 0%.

The second was the knob. This file concluded "the failure is not a threshold"
after dropping the per-keypoint floor from 0.6 to 0.3 and watching coverage move
75.0% to 76.6%. But a pose model emits keypoints only for a DETECTED COURT, and
nobody had touched the DETECTION floor. At ultralytics' default of 0.25 the
model reports no court at all on 97% of tight shots; at 0.001 that falls to 55%,
and coverage over the whole game goes 54% to 79%. Measured on the same 630
frames:

    band                          n     0.25     0.001
    almost no floor              25       4%       52%
    a little floor               46       2%       50%
    a normal wide play shot      76       1%       42%
    a lot of floor              483      70%       89%

The frames were never invisible to the model. They were being thrown away by a
default nobody had examined, and `court_keypoints.COURT_DETECTION_CONF` is now
that number with the evidence beside it.

What remains after the threshold is still a training-data problem, and it is
still the same one: the model finds NO COURT on half the tight shots even at
0.001, against 0-4% that find a court and too few landmarks. That is
out-of-distribution rejection rather than a localisation failure, which is what
scale and crop augmentation is for -- the model was trained on frames showing
most of the court.

Reported per band because the aggregate hides it: a game is mostly wide shots,
so an average over all frames reads as a coverage problem when it is a
generalisation problem with a clean boundary.

THREE COUNTS PER BAND, NOT ONE, because "no homography" has two very different
causes and the fix differs:

    no court instance at all   the pose model did not detect a court, so there
                               are no keypoints to threshold. Out-of-distribution
                               rejection, and crop augmentation is the fix.
    instance, under 4 landmarks  it saw a court and could not place enough
                               points. A localisation problem, and augmentation
                               helps less.
    instance, 4+ landmarks     registers.

`--instance-conf` drops the detection threshold as far as you like. If a tight
shot yields no instance even at 0.001, the model is rejecting the frame outright
rather than struggling with it, which is the case augmentation is for.

The default sampling (`--every 60`) gives about twenty frames a band, and a 0/20
has a Wilson upper bound near 16%. Any before/after comparison must run at
`--every 10` or the "after" will not be resolvable either.
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
    parser.add_argument("--conf", type=float, default=0.6,
                        help="keypoint confidence floor")
    parser.add_argument("--instance-conf", type=float, default=None,
                        help="court DETECTION floor; drop it to 0.001 to ask "
                             "whether the model sees a court at all")
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
    from courtvision.court_keypoints import COURT_DETECTION_CONF
    if args.instance_conf is None:
        args.instance_conf = COURT_DETECTION_CONF
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
                               imgsz=args.imgsz, conf=args.instance_conf)[0]
        found, instance = 0, False
        if (result.keypoints is not None and result.keypoints.conf is not None
                and len(result.keypoints.conf)):
            instance = True
            found = int((result.keypoints.conf[0].cpu().numpy() >= args.conf).sum())
        rows.append((share, found, instance))
    capture.release()

    if not rows:
        print("FAIL - no frames read")
        return 1
    usable = [r for r in rows if r[1] >= NEEDED]
    print(f"{args.video}  {len(rows)} frames sampled every {args.every:.0f}s")
    print(f"  keypoint floor {args.conf}, court detection floor {args.instance_conf}")
    print(f"  a homography is possible on {len(usable)}/{len(rows)} "
          f"= {len(usable) / len(rows):.0%} of them\n")
    print(f"  {'band':<46}{'n':>5}{'no court':>10}{'<4 kp':>8}{'4+ kp':>8}"
          f"{'median':>8}")
    for low, high, name in BANDS:
        band = [r for r in rows if low <= r[0] < high]
        if not band:
            continue
        blind = [r for r in band if not r[2]]
        short = [r for r in band if r[2] and r[1] < NEEDED]
        ok = [r for r in band if r[1] >= NEEDED]
        print(f"  {name:<46}{len(band):>5}{len(blind) / len(band):>9.0%}"
              f"{len(short) / len(band):>8.0%}{len(ok) / len(band):>8.0%}"
              f"{int(np.median([r[1] for r in band])):>8}")
    blind_all = [r for r in rows if not r[2]]
    print(f"\n  the model saw no court at all on {len(blind_all)}/{len(rows)} "
          f"= {len(blind_all) / len(rows):.0%} of frames, at detection floor "
          f"{args.instance_conf}")
    if blind_all and args.instance_conf > 0.01:
        print("  -> re-run with --instance-conf 0.001. If these stay blind, the "
              "model is\n     rejecting the frame outright, which is what crop "
              "augmentation fixes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
