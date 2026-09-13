"""Drop training labels that share a broadcast shot with an evaluation frame.

Labels mined from the SAME video the gate is measured on leak, and the leak is
easy to miss because the times look far apart in a list. Measured on the ball
track labels against the ten hand-located truth balls: the nearest training
label sits 1.7 s from one truth frame and 3.1 s from another, and 1262.5 s --
one of the detector's rank-1 successes -- has a label 1.7 s away. At 1.7 s it
is the same possession, the same camera and very nearly the same picture.

So the ball detector's 6 of 10 could not be trusted as measured, and this is
the filter that makes the re-measurement honest. The criterion is the one
`propagate_rim_labels.py` argues for: not clock distance, which a 25 s
evaluation grid makes unusable, and not ORB registration alone, which means
"the same fixed camera" and holds across a whole night in these arenas -- but
BOTH, the frames registering AND being within SAME_TAKE_S of each other.

Works on any label file shaped {"frames": [{"t": ..., ...}, ...]}, which is
every mined-label file in this repository.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--held-out", action="append", required=True,
                        help="repeatable; files whose frames the gate is scored on")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from propagate_rim_labels import HOLD_OUT_REACH_S, nearby, shares_a_shot
    from propagate_rim_labels import held_out_times

    data = json.load(open(args.labels))
    rows = data["frames"]
    held_out = np.asarray(held_out_times(args.held_out), np.float64)
    capture = cv2.VideoCapture(args.video)

    def grey_at(t):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ok else None

    # Evaluation frames are loaded once each and reused; the label file revisits
    # the same neighbourhoods many times.
    loaded = {}
    kept, dropped, started = [], 0, time.time()
    for n, row in enumerate(rows):
        t = float(row["t"])
        watch = []
        for v in nearby(t, held_out, HOLD_OUT_REACH_S):
            if v not in loaded:
                loaded[v] = grey_at(v)
            if loaded[v] is not None:
                # (frame, seconds apart): shares_a_shot needs BOTH the picture
                # and the gap, because registration alone means "the same fixed
                # camera", which these arenas satisfy all night. Passing a bare
                # frame silently reverts to that stricter, wrong rule -- it
                # dropped 156 of 196 hand labels before this was fixed.
                watch.append((loaded[v], v - t))
        if not watch:
            kept.append(row)
            continue
        grey = grey_at(t)
        if grey is None:
            continue
        if shares_a_shot(grey, watch):
            dropped += 1
            continue
        kept.append(row)
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(rows)}, {dropped} dropped, "
                  f"{(time.time() - started) / 60:.1f} min", flush=True)
    capture.release()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({**data, "frames": kept,
               "filtered_by": "same-shot ORB registration to an evaluation frame",
               "held_out_files": args.held_out,
               "dropped": dropped}, open(out, "w"))
    print(f"{len(rows)} labels in, {len(kept)} kept, {dropped} dropped for "
          f"sharing a shot with the evaluation ({dropped / max(len(rows), 1):.1%})")
    print(f"  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
