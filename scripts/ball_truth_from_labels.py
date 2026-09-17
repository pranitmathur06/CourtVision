"""Turn hand ball verdicts into a truth file, keeping the exclusions honest.

`make_ball_label_page.py` records three verdicts and only one of them is a
located ball. The other two are not noise to be dropped quietly:

    none      no ball in the picture. A frame the system must NOT claim a ball
              on, so it belongs in the false-alarm denominator, not the bin.
    unknown   the ball is in there and a person could not find it at 8x. It is
              excluded from BOTH sides, because scoring a system against truth
              nobody can establish measures the labeller, not the system.

The unknown COUNT is itself a headline measurement. Of 41 frames sampled this
way before, 28 came back unknown -- if that rate holds, it says more about what
this 1280x720 broadcast can support than any accuracy figure does, and it is
printed rather than buried.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def split(rows):
    """(located, absent, unknown) from verdict rows."""
    # Two label rounds, two field names for the same judgement.
    def verdict(row):
        return row.get("verdict") or row.get("ball_verdict")

    located = [r for r in rows if verdict(r) == "ball" and r.get("ball")]
    absent = [r for r in rows if verdict(r) == "none"]
    unknown = [r for r in rows if verdict(r) == "unknown"]
    return located, absent, unknown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--game", default=None,
                        help="keep only this game's rows. Two label rounds cover "
                             "three broadcasts and the ball is not equally hard "
                             "in each, so a pooled number hides a per-game "
                             "spread that the per-game harness exists to show.")
    parser.add_argument("--pick", default=None, choices=["random", "hard"],
                        help="'random' keeps the uniformly sampled half, which is "
                             "the only half that estimates in-game accuracy; "
                             "anything else in the file was sampled because the "
                             "model was already struggling there.")
    parser.add_argument("--frames", default=None,
                        help="directory holding the JPEGs the labeller actually "
                             "looked at. Recorded in the truth file so scorers "
                             "read THAT frame rather than seeking the video: the "
                             "labelled times were written at 30.0 fps and rounded "
                             "to 0.1 s, so a seek lands a frame or more away, and "
                             "a ball crosses several of its own widths in 33 ms.")
    parser.add_argument("--tolerance-px", type=float, default=28.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for path in args.labels:
        rows.extend(json.load(open(path))["frames"])
    if args.game:
        rows = [r for r in rows if r.get("game") == args.game]
    if args.pick:
        keep = (lambda r: r.get("pick") == "random") if args.pick == "random" \
            else (lambda r: r.get("pick") not in (None, "random"))
        rows = [r for r in rows if keep(r)]
    located, absent, unknown = split(rows)
    total = len(located) + len(absent) + len(unknown)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"note": "hand-located balls; 'absent' frames carry no ball and belong in "
                       "the false-alarm denominator; 'unknown' frames are excluded from "
                       "both sides of the accuracy",
               "tolerance_px": args.tolerance_px,
               "frames_dir": args.frames,
               "game": args.game,
               "unknown_count": len(unknown), "absent_count": len(absent),
               "frames": [{"t": r["t"], "ball": r["ball"],
                           "radius": r.get("radius", 0),
                           "file": r.get("file")} for r in located],
               "absent": [{"t": r["t"]} for r in absent],
               "unknown": [{"t": r["t"]} for r in unknown]}, open(out, "w"), indent=1)

    print(f"{total} frames judged")
    print(f"  {len(located):4d} ball located   <- the truth set the gate is scored on")
    print(f"  {len(absent):4d} no ball in frame")
    print(f"  {len(unknown):4d} present but not findable by eye ({len(unknown)/max(total,1):.0%})")
    if located:
        import numpy as np
        r = [x.get("radius", 0) for x in located]
        print(f"  ball radius px: min {min(r)} p50 {int(np.median(r))} max {max(r)}")
    if len(located) < 100:
        print(f"\n  NOTE: {len(located)} located balls is not yet enough to certify 95%. "
              f"At n=100 a true 0.95 measures 0.888-0.978; at n=13 it measures 0.667-0.986.")
    print(f"  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
