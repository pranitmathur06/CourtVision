"""Answer 'where is the ball' from neighbouring frames when this one cannot.

Measured against ten hand-located balls, the detector's best candidate on the
scored frame is 189 px away at 4362 s and 36 px at 2862 s -- but on a frame a
fifth of a second later it is 19 px and 14 px away. The detector HAS those
balls. It just does not have them on the frame being asked about.

`rim_track` has filled the rim's gaps from its neighbours since Round 63 and
nobody thought it controversial; this is the same move for the ball, with the
one extra step the ball needs. The rim is filled by interpolating positions
directly, which works because the camera barely moves in a fifth of a second
and a rim does not move at all. A ball does. So the neighbours' candidates are
carried into THIS frame's pixels by ORB first, exactly as the track finder
does, and only then pooled.

This widens the candidate pool rather than choosing within it, so it cannot fix
a selection failure -- eight rules have failed at that -- but it can fix a
frame where nothing was proposed at all, which is what three of the ten misses
are.

Reported honestly: a candidate borrowed from t+0.2 is evidence about where the
ball was at t+0.2. Over that interval a ball in flight travels a long way, so
each pooled candidate keeps the offset it came from and the age is written out
with it, for a later pass to weigh or refuse.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

OFFSETS = (-0.4, -0.2, 0.2, 0.4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--system", required=True, help="its frames set the times")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--orb-scale", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from track_camera import hop_at_scale

    cache = json.load(open(args.detections))
    by_time = {round(r["t"], 3): r for r in cache["frames"]}
    times = np.array(sorted(by_time))
    rows = json.load(open(args.system))["frames"]
    if args.limit:
        rows = rows[:args.limit]

    def balls(t):
        if not len(times):
            return []
        j = int(np.argmin(np.abs(times - t)))
        if abs(times[j] - t) > 0.15:
            return []
        return [b for b in by_time[times[j]]["boxes"]
                if b["cls"] == "ball" and b["conf"] >= args.conf]

    capture = cv2.VideoCapture(args.video)

    def grey(t):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ok else None

    out_rows, started = [], time.time()
    for n, row in enumerate(rows):
        t = row["t"]
        here = [{"centre": [(b["xyxy"][0] + b["xyxy"][2]) / 2,
                            (b["xyxy"][1] + b["xyxy"][3]) / 2],
                 "conf": b["conf"], "age_s": 0.0} for b in balls(t)]
        target = grey(t)
        if target is not None:
            for dt in OFFSETS:
                other = balls(t + dt)
                if not other:
                    continue
                source = grey(t + dt)
                if source is None:
                    continue
                hop = hop_at_scale(source, target, None, None, args.orb_scale)
                if hop is None:
                    continue
                points = np.array([[(b["xyxy"][0] + b["xyxy"][2]) / 2,
                                    (b["xyxy"][1] + b["xyxy"][3]) / 2] for b in other])
                moved = np.c_[points, np.ones(len(points))] @ np.asarray(hop).T
                keep = np.abs(moved[:, 2]) > 1e-9
                moved = moved[keep, :2] / moved[keep, 2:3]
                for b, p in zip([o for o, k in zip(other, keep) if k], moved):
                    here.append({"centre": [float(p[0]), float(p[1])],
                                 "conf": b["conf"], "age_s": round(abs(dt), 2)})
        out_rows.append({"t": t, "candidates": here})
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(rows)} frames, {(time.time() - started) / 60:.1f} min",
                  flush=True)
    capture.release()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "detections": args.detections,
               "offsets": list(OFFSETS), "frames": out_rows}, open(out, "w"))
    borrowed = sum(1 for r in out_rows for c in r["candidates"] if c["age_s"] > 0)
    own = sum(1 for r in out_rows for c in r["candidates"] if c["age_s"] == 0)
    print(f"{len(out_rows)} frames: {own} candidates of their own, {borrowed} borrowed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
