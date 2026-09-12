"""Tell the ball from a stationary head by how it MOVES, not by where it is.

Round 69 established that no single frame can do this. A ball 15 ft up over the
far side and a spectator behind it lie on the SAME RAY from a fixed camera
centre: worked by hand for a real candidate, its ray sits between 10.2 and 17.3
ft above the floor the whole time it is over the court, which is an ordinary
high arc. One frame carries no depth to separate them.

Motion does. And it does NOT need the camera's pose: ORB between a frame and
its neighbours removes the camera's own movement directly, which is the whole
job. Warp each neighbour's candidates into this frame and a spectator's head
lands where it already was, while the ball has gone somewhere else.

So each candidate gets a MOTION in pixels per second, measured in a frame where
the camera is standing still:

- near zero: furniture, a head, a shoulder, a logo. The game ball is never
  still for a fifth of a second, even in someone's hands -- players move.
- absurdly large: a mismatch, not a trajectory.
- in between: a thing that moves like a ball.

This is the same insight as the ray-space fixture rule, which catches objects
that hold ONE direction all game, applied at the other end of the time scale:
the fixture rule needs the whole game to see the scorer's-table ball, and this
needs a fifth of a second to see a spectator.

Nothing is chosen here. Every candidate keeps its confidence and gains a
motion, and `build_rim_ball.py` decides.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

#: Neighbours either side, in seconds. Two is enough to see movement and few
#: enough that the view has not cut.
OFFSETS = (-0.4, -0.2, 0.2, 0.4)
#: A candidate matched this close in a motion-compensated neighbour is the
#: same object, and its displacement is its motion.
MATCH_PX = 90.0
#: Below this a candidate has not moved: it is furniture or a person.
STILL_PX_PER_S = 12.0


def motion_of(point, warped_candidates, match_px=MATCH_PX):
    """Pixels per second, from the nearest match in each warped neighbour.

    Returns None when no neighbour offers a match -- the object appeared or the
    view cut, and silence is the honest answer rather than a made-up number.
    """
    speeds = []
    for dt, points in warped_candidates:
        if not len(points):
            continue
        distance = np.min(np.hypot(points[:, 0] - point[0], points[:, 1] - point[1]))
        if distance <= match_px:
            speeds.append(distance / abs(dt))
    if not speeds:
        return None
    return float(np.median(speeds))


def warp(points, homography):
    """Neighbour image points carried into the target frame."""
    if homography is None or not len(points):
        return np.zeros((0, 2))
    stacked = np.c_[np.asarray(points, np.float64), np.ones(len(points))]
    out = stacked @ np.asarray(homography, np.float64).T
    good = np.abs(out[:, 2]) > 1e-9
    return (out[good, :2] / out[good, 2:3])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--system", required=True,
                        help="build_rim_ball.py output; its frames set the times")
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=2560)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--orb-scale", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device
    from courtvision.provenance import code_provenance

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from track_camera import hop_at_scale

    rows = json.load(open(args.system))["frames"]
    if args.limit:
        rows = rows[:args.limit]
    model, device = YOLO(args.detector), resolve_device()
    capture = cv2.VideoCapture(args.video)

    def read(t):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        return frame if ok else None

    def balls(frame):
        found = model.predict(frame, device=device, verbose=False,
                              imgsz=args.imgsz, conf=args.conf)[0].boxes
        out = []
        if found is not None and len(found):
            names = model.names
            for cls, conf, box in zip(found.cls.cpu().numpy(),
                                      found.conf.cpu().numpy(),
                                      found.xyxy.cpu().numpy()):
                if names[int(cls)] != "ball":
                    continue
                out.append({"centre": [float((box[0] + box[2]) / 2),
                                       float((box[1] + box[3]) / 2)],
                            "conf": round(float(conf), 3)})
        return out

    out_rows, started, still = [], time.time(), 0
    for n, row in enumerate(rows):
        t = row["t"]
        frame = read(t)
        if frame is None:
            continue
        here = balls(frame)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        warped = []
        for dt in OFFSETS:
            other = read(t + dt)
            if other is None:
                continue
            hop = hop_at_scale(cv2.cvtColor(other, cv2.COLOR_BGR2GRAY), grey,
                               None, None, args.orb_scale)
            if hop is None:
                continue
            points = [c["centre"] for c in balls(other)]
            warped.append((dt, warp(points, hop)))

        for candidate in here:
            speed = motion_of(candidate["centre"], warped)
            candidate["motion"] = None if speed is None else round(speed, 1)
            candidate["still"] = bool(speed is not None and speed < STILL_PX_PER_S)
            still += int(candidate["still"])
        out_rows.append({"t": t, "candidates": here, "neighbours": len(warped)})
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(rows)} frames, {(time.time() - started) / 60:.1f} min",
                  flush=True)
    capture.release()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "imgsz": args.imgsz, "conf": args.conf,
               "offsets": list(OFFSETS), "still_px_per_s": STILL_PX_PER_S,
               "frames": out_rows, **code_provenance(__file__)}, open(out, "w"))
    total = sum(len(r["candidates"]) for r in out_rows)
    moved = sum(1 for r in out_rows for c in r["candidates"]
                if c["motion"] is not None and not c["still"])
    print(f"{len(out_rows)} frames, {total} candidates: {still} stand still, "
          f"{moved} move, {total - still - moved} could not be matched at all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
