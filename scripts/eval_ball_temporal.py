"""Does choosing the ball over TIME beat choosing it frame by frame?

`eval_ball_selection.py` measured the gap this exists to close. On 130 uniformly
sampled frames the detector PROPOSES a candidate within 28 px on 91.5% of them
and the pipeline REPORTS the right one on 78.5%. Thirteen points sit between
what is seen and what is said, and of the 119 frames where the ball is proposed
it is in the top two by confidence on 114. That is a selection problem.

`src/courtvision/ball_track.py` has held an exact Viterbi selector, with a
passing unit test, since before any of that was measured, and nothing has ever
called it. This calls it.

THREE NUMBERS, the same shape `eval_ball_selection.py` reports:

    oracle     a candidate somewhere in the window is within tolerance. The
               ceiling of any selector, and it is not 100%.
    argmax     the most confident candidate in the CENTRE frame. What ships.
    viterbi    the candidate the path picks for the centre frame.

Scored with exact McNemar on the frames where the two disagree, because the
argmax and the path answer the same frames and unpaired intervals cannot resolve
the effects this project produces -- the argument `eval_possession_temporal.py`
already makes and this one inherits.

THE CENTRE FRAME IS THE LABELLER'S OWN JPEG, never a seek. Truth times were
written at 30.0 fps and rounded to 0.1 s, so seeking lands a frame or more away
and a ball crosses several of its own widths in 33 ms; scored by seeking, the
ball detector reads eleven points lower than it is. Neighbours come from the
video, where being a frame out does not matter because they are a quarter of a
second away by construction.

THE TWO CONSTANTS ARE FITTED ON THE HARD HALF AND REPORTED ON THE UNIFORM HALF.
`MOVE_WEIGHT` and `MISSING_COST` were hand-set and never fitted. Two parameters
chosen on 97 frames and reported on 130 others is a defensible fit; choosing
them on the frames then quoted is how this project has previously talked itself
into numbers that did not survive.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from courtvision.ball_track import (MISSING_COST,  # noqa: E402
                                    MOVE_WEIGHT, choose)

#: Frames in the window and the gap between them. A ball is ballistic over a
#: quarter second and a possession is longer than the window either way.
WINDOW = 7
STEP_S = 0.25


def wilson(hits, total, z=1.96):
    if not total:
        return 0.0, 0.0
    p = hits / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(a: np.ndarray, b: np.ndarray):
    """Exact two-sided McNemar on paired right/wrong vectors."""
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(b & ~a))
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    smaller = min(only_a, only_b)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2.0 ** n)
    return only_a, only_b, min(1.0, 2.0 * tail)


def load_truth(paths):
    """(t, (x, y), frames_dir, file, game) per hand-located ball."""
    out = []
    for path in paths:
        blob = json.load(open(path))
        folder = blob.get("frames_dir")
        game = blob.get("game")
        for row in blob["frames"]:
            out.append((float(row["t"]), tuple(float(v) for v in row["ball"]),
                        folder, row.get("file"), game))
    return sorted(out)


def build_cache(video, truth, detector, out_path, window, step_s, imgsz, conf):
    """Candidates for every frame of every window, detected once.

    The expensive part by a wide margin, and it does not change when the two
    path constants do -- which is the whole point of fitting them.
    """
    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device

    model, device = YOLO(detector), resolve_device()
    capture = cv2.VideoCapture(video)
    half = window // 2
    rows = []
    for n, (when, ball, folder, name, game) in enumerate(truth):
        frames = []
        for slot in range(window):
            offset = (slot - half) * step_s
            image = None
            if slot == half and folder and name:
                saved = Path(folder) / name
                if saved.exists():
                    image = cv2.imread(str(saved))
            if image is None:
                capture.set(cv2.CAP_PROP_POS_MSEC, max(when + offset, 0.0) * 1000)
                ok, image = capture.read()
                if not ok:
                    frames.append([])
                    continue
            found = model.predict(image, device=device, verbose=False,
                                  imgsz=imgsz, conf=conf)[0].boxes
            here = []
            if found is not None and len(found):
                for cls, score, box in zip(found.cls.cpu().numpy(),
                                           found.conf.cpu().numpy(),
                                           found.xyxy.cpu().numpy()):
                    if model.names[int(cls)] != "ball":
                        continue
                    here.append([(float(box[0]) + float(box[2])) / 2,
                                 (float(box[1]) + float(box[3])) / 2,
                                 float(score)])
            frames.append(here)
        rows.append({"t": when, "ball": list(ball), "game": game,
                     "centre": half, "frames": frames})
        if (n + 1) % 25 == 0:
            print(f"    {n + 1}/{len(truth)} windows", flush=True)
    capture.release()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": video, "detector": detector, "window": window,
               "step_s": step_s, "imgsz": imgsz, "conf": conf, "rows": rows},
              open(out_path, "w"))
    return rows


def score(rows, tolerance, move_weight, missing_cost):
    """Right/wrong vectors for oracle, argmax and the path, per window."""
    oracle, argmax, path = [], [], []
    for row in rows:
        frames = row["frames"]
        centre = row["centre"]
        truth = row["ball"]
        here = frames[centre]

        def near(point):
            return (point is not None
                    and math.dist(point, truth) <= tolerance)

        oracle.append(any(near((c[0], c[1])) for c in here))
        best = max(here, key=lambda c: c[2], default=None)
        argmax.append(near((best[0], best[1])) if best else False)
        picked = choose([[tuple(c) for c in f] for f in frames],
                        move_weight=move_weight, missing_cost=missing_cost)
        path.append(near(picked[centre]) if picked else False)
    return (np.array(oracle, dtype=bool), np.array(argmax, dtype=bool),
            np.array(path, dtype=bool))


def fit(rows, tolerance, weights, costs):
    """The two constants, chosen to maximise accuracy on THESE rows."""
    best, chosen = -1.0, (MOVE_WEIGHT, MISSING_COST)
    for weight in weights:
        for cost in costs:
            _, _, path = score(rows, tolerance, weight, cost)
            if path.mean() > best:
                best, chosen = float(path.mean()), (weight, cost)
    return chosen, best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", action="append", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detector", default="checkpoints/ball_v2/best.pt")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--step-s", type=float, default=STEP_S)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--tolerance-px", type=float, default=28.0)
    parser.add_argument("--fit-cache", default=None,
                        help="a cache built from the HARD half; the two path "
                             "constants are fitted on it and reported on --cache")
    args = parser.parse_args()

    truth = load_truth(args.truth)
    if args.build or not Path(args.cache).exists():
        print(f"  detecting {len(truth)} windows of {args.window} frames")
        rows = build_cache(args.video, truth, args.detector, args.cache,
                           args.window, args.step_s, args.imgsz, args.conf)
    else:
        rows = json.load(open(args.cache))["rows"]

    weight, cost = MOVE_WEIGHT, MISSING_COST
    if args.fit_cache and Path(args.fit_cache).exists():
        fit_rows = json.load(open(args.fit_cache))["rows"]
        (weight, cost), got = fit(fit_rows, args.tolerance_px,
                                  [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04],
                                  [2.0, 3.0, 4.5, 6.0, 8.0, 12.0])
        print(f"  fitted on {len(fit_rows)} HARD windows: move_weight {weight}, "
              f"missing_cost {cost}  ({got:.1%} there)")
    else:
        print(f"  constants as written: move_weight {weight}, missing_cost {cost}")

    oracle, argmax, path = score(rows, args.tolerance_px, weight, cost)
    total = len(rows)
    print(f"\n  {total} hand-located balls, tolerance {args.tolerance_px:.0f} px, "
          f"window {args.window} x {args.step_s}s")
    for name, vector in (("oracle, a candidate is there", oracle),
                         ("argmax in the centre frame", argmax),
                         ("viterbi over the window", path)):
        hits = int(vector.sum())
        low, high = wilson(hits, total)
        print(f"    {name:<32} {hits:>3}/{total} = {hits / total:5.1%}"
              f"   (95% CI {low:.0%}-{high:.0%})")

    only_a, only_b, p = mcnemar(path, argmax)
    verdict = "significant" if p < 0.05 else "not significant"
    print(f"\n  paired, on the frames where they disagree (exact McNemar):")
    print(f"    viterbi vs argmax   {only_a} frames only viterbi gets right, "
          f"{only_b} only argmax   p = {p:.4f}  ({verdict})")

    games = sorted({row.get("game") for row in rows if row.get("game")})
    if len(games) > 1:
        print("\n  per game:")
        for game in games:
            keep = np.array([row.get("game") == game for row in rows], dtype=bool)
            n = int(keep.sum())
            print(f"    {game:<18} oracle {oracle[keep].mean():5.1%}   "
                  f"argmax {argmax[keep].mean():5.1%}   "
                  f"viterbi {path[keep].mean():5.1%}   (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
