"""Score the drawn ball against hand-located truth, on the frames that have it.

The overlay's ball had never been measured. It was tuned by watching clips,
which is how it ended up sitting on a spectator for whole possessions: a
stationary orange thing in the crowd is free under a smoothness prior, and
eyes skim past it.

Truth is a ball position located by eye on a 1280 px panel with a 50 px grid,
BEFORE any system's claim was looked at, with a stated tolerance of 28 px. Truth
files are named on the command line; pass --truth more than once to pool them.

    ball_truth_uniform.json       135 balls on a UNIFORM 25 s grid across the
                                  whole game. This is the default and it is the
                                  one to quote: uniform sampling makes it an
                                  estimate of in-game accuracy.
    ball_truth_handlocated.json   the original 13, and
    ball_truth_hard.json          frames where the detector's best candidate is
                                  under 0.35 -- the ball is buried or absent.
                                  Together they are a HARD-CASE sample and the
                                  number they produce is a worst case, not an
                                  average. Scoring 5/13 and reading it as the
                                  in-game rate is the mistake this default fixes.

The window around each instant is detected once and cached, so a change to the
choosing costs seconds. `--cache` writes it; later runs read it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from courtvision.stats import wilson  # noqa: E402

import clip_boxes  # noqa: E402

#: Uniformly sampled, so the number it produces estimates in-game accuracy.
UNIFORM_TRUTH = ["data/labeling/rim_ball/ball_truth_uniform.json"]
#: The original hard-case sample, kept for the worst-case number.
HARD_TRUTH = ["data/labeling/rim_ball/ball_truth_handlocated.json",
              "data/labeling/rim_ball/ball_truth_hard.json"]
TOLERANCE_PX = 28.0
WINDOW_S = 3.0


def truth_points(paths):
    out = []
    for path in paths:
        if not Path(path).exists():
            continue
        for row in json.load(open(path))["frames"]:
            if row.get("ball"):
                out.append((float(row["t"]), [float(v) for v in row["ball"]],
                            Path(path).stem))
    return sorted(out)


def build_cache(video, points, out_path, imgsz=1280, step=2):
    """Detect the window around each truth instant, once."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    from courtvision.candidates import court_region, stands_on_court
    from courtvision.device import resolve_device

    model = YOLO("runs/detect/outputs/train/detector/weights/best.pt")
    device = resolve_device()
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(round(2 * WINDOW_S * fps))

    windows = {}
    for t, ball, source in points:
        capture.set(cv2.CAP_PROP_POS_MSEC, (t - WINDOW_S) * 1000)
        rows, region, court = [], None, None
        for f in range(count):
            ok, frame = capture.read()
            if not ok:
                break
            if f % step:
                continue
            found = model.predict(frame, device=device, verbose=False,
                                  imgsz=imgsz, conf=0.08)[0].boxes
            here = []
            if found is not None and len(found):
                for cls, conf, box in zip(found.cls.cpu().numpy(),
                                          found.conf.cpu().numpy(),
                                          found.xyxy.cpu().numpy()):
                    kind = {"player": "p", "handler": "h",
                            "ball": "b", "rim": "r"}.get(model.names[int(cls)])
                    if kind:
                        here.append([kind, round(float(conf), 3)]
                                    + [round(float(v), 1) for v in box])
            if f % (step * 3) == 0 or region is None:
                region = court_region(frame)
                if region is not None:
                    ys, xs = np.nonzero(region)
                    court = ([int(xs.min()), int(ys.min()),
                              int(xs.max()), int(ys.max())] if len(xs) else None)
            people = [b[2:] for b in here if b[0] in ("p", "h")]
            on = (stands_on_court(region, np.array(people, float)).tolist()
                  if people and region is not None else [])
            rows.append({"f": f, "d": here, "court": court, "on": on})
        windows[f"{t:.1f}"] = {"t": t, "ball": ball, "source": source,
                               "rows": rows}
        print(f"  detected {t:.1f}s ({len(rows)} frames)", flush=True)
    capture.release()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": video, "fps": fps, "frames_per_window": count,
               "step": step, "windows": windows}, open(out_path, "w"))
    return out_path


def score(cache_path, anchored=True):
    """Where the chooser puts the ball at each truth instant."""
    cached = json.load(open(cache_path))
    fps, count = cached["fps"], cached["frames_per_window"]
    middle = count // 2

    if not anchored:                       # the pass this replaces
        saved = (clip_boxes.BALL_FREE_PX, clip_boxes.BALL_FAR_PX,
                 clip_boxes.BALL_REJECT_PX, clip_boxes.BALL_LOFT_SHARE)
        clip_boxes.BALL_FREE_PX = 1e9
        clip_boxes.BALL_FAR_PX = 1e9
        clip_boxes.BALL_REJECT_PX = 1e9
        clip_boxes.BALL_LOFT_SHARE = 1e9

    found = elsewhere = missing = 0
    rows = []
    for key, window in sorted(cached["windows"].items(), key=lambda kv: kv[1]["t"]):
        detected = []
        for row in window["rows"]:
            people = [b for b in row["d"] if b[0] in ("p", "h")]
            on = row.get("on") or []
            keep = [b for i, b in enumerate(people)
                    if b[1] >= clip_boxes.PLAYER_CONF and (i >= len(on) or on[i])]
            detected.append({
                "f": row["f"],
                "p": [b[2:] for b in keep],
                "h": [b[2:] for b in keep if b[0] == "h"],
                "r": [b[2:] for b in row["d"]
                      if b[0] == "r" and b[1] >= clip_boxes.RIM_CONF],
                "b": [(b[2:], b[1]) for b in row["d"] if b[0] == "b"],
                "court": row.get("court"),
            })
        drawn = clip_boxes.assemble(detected, count, fps)
        near = min(range(count), key=lambda f: abs(f - middle))
        box = next((b for b in drawn[near] if b[0] == "b"), None)
        truth = window["ball"]
        if box is None:
            missing += 1
            rows.append((window["t"], window["source"], "no ball drawn", None))
            continue
        centre = ((box[1] + box[3]) / 2, (box[2] + box[4]) / 2)
        off = math.dist(centre, truth)
        if off <= TOLERANCE_PX:
            found += 1
        else:
            elsewhere += 1
        rows.append((window["t"], window["source"],
                     "located" if off <= TOLERANCE_PX else "wrong place",
                     round(off, 1)))

    if not anchored:
        (clip_boxes.BALL_FREE_PX, clip_boxes.BALL_FAR_PX,
         clip_boxes.BALL_REJECT_PX, clip_boxes.BALL_LOFT_SHARE) = saved
    return found, elsewhere, missing, rows




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/raw_clips/fullgame.mp4")
    parser.add_argument("--cache", default="outputs/ball_choice_windows_uniform.json")
    parser.add_argument("--truth", action="append", default=None,
                        help="a truth file; repeat to pool. Defaults to the "
                             "uniform 135; pass the two hard-case files instead "
                             "for the worst case.")
    parser.add_argument("--build", action="store_true",
                        help="detect the windows again (a few minutes)")
    args = parser.parse_args()

    points = truth_points(args.truth or UNIFORM_TRUTH)
    print(f"{len(points)} hand-located ball positions, tolerance {TOLERANCE_PX:.0f} px")
    if args.build or not Path(args.cache).exists():
        build_cache(args.video, points, args.cache)

    for label, anchored in (("as it shipped (no anchor)", False),
                            ("anchored to the play", True)):
        found, elsewhere, missing, rows = score(args.cache, anchored)
        total = found + elsewhere + missing
        low, high = wilson(found, total)
        print(f"\n  {label}")
        print(f"    located within {TOLERANCE_PX:.0f} px  {found}/{total} "
              f"= {found / max(total, 1):.1%}  (95% CI {low:.0%}-{high:.0%})")
        print(f"    drawn in the wrong place  {elsewhere}")
        print(f"    no ball drawn             {missing}")
        for t, source, verdict, off in rows:
            print(f"      {t:>8.1f}s  {verdict:<14}"
                  f"{'' if off is None else f'{off:6.1f} px off'}  {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
