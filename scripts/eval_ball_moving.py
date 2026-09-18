#!/usr/bin/env python3
"""Does the ball show itself by MOVING on the court? Fitted and reported apart.

ROUND 93 MEASURED THE OPPOSITE OF WHAT `ball_track.choose` ASSUMES. Its prior
rewards a candidate for holding still, and the decoys this detector emits are
things that hold still -- a logo, a head, a shoe -- while the frames where the
ball is hard to pick are the frames where it is flying. The fit agreed, driving
the smoothness weight to the bottom of the grid on all three broadcasts, which
is an optimiser saying switch the prior off.

WHAT IS NEW HERE IS THE FRAME OF REFERENCE. A logo moves with the camera; the
ball moves with the game. The rim is bolted to the building and the detector
draws it on three frames in four, so the rim's displacement between two frames
IS the camera's, for free. In that frame of reference the contrast between the
ball and its decoys goes from 2.2x to 6.1x.

THE TWO CONSTANTS ARE FITTED ON ONE HALF OF THE WINDOWS AND REPORTED ON THE
OTHER. `eval_ball_temporal.py` fits on the hard half and reports on the uniform
half; only the uniform windows are cached on disk, so the split here is by
window index and is stated rather than borrowed.

THE THREE ARMS ANSWER THE SAME FRAMES, so they are compared with exact McNemar
on the frames where they disagree and not with two intervals.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.ball_track import choose, choose_moving  # noqa: E402
from courtvision.stats import mcnemar, wilson  # noqa: E402

#: A choice this close to the hand-clicked ball is the ball. The tolerance
#: `eval_ball_selection.py` already uses, unchanged so the numbers compose.
TOLERANCE_PX = 28.0
#: Rows either side of the labelled frame the path may look at. The rows are
#: 1/15 s apart, so this is a little over half a second of context.
HALF_WINDOW = 4


def centre(box):
    return ((box[2] + box[4]) / 2.0, (box[3] + box[5]) / 2.0)


def rim_centre(row):
    rims = [b for b in row["d"] if b[0] == "r"]
    if not rims:
        return None
    return centre(max(rims, key=lambda b: b[1]))


def top_k(frames, k: int | None):
    """Keep only the k most confident candidates per frame.

    The detector emits a mean of fourteen ball boxes a frame and the ball is in
    the top TWO by confidence on 80% of the windows where it is proposed at
    all. Handing a path twelve hopeless candidates per frame gives it twelve
    more ways to build a cheap wrong route, so the candidate set is a knob in
    its own right and is swept rather than assumed.
    """
    if not k:
        return frames
    return [sorted(frame, key=lambda c: -c[2])[:k] for frame in frames]


def window_of(blob_window):
    """(candidates per row, camera shift per row, index of the labelled row)."""
    rows = blob_window["rows"]
    middle = len(rows) // 2
    lo = max(0, middle - HALF_WINDOW)
    hi = min(len(rows), middle + HALF_WINDOW + 1)
    slice_rows = rows[lo:hi]
    candidates = [[(*centre(b), b[1]) for b in row["d"] if b[0] == "b"]
                  for row in slice_rows]
    shifts: list[tuple[float, float] | None] = [None]
    for before, after in zip(slice_rows, slice_rows[1:]):
        one, two = rim_centre(before), rim_centre(after)
        shifts.append(None if one is None or two is None
                      else (two[0] - one[0], two[1] - one[1]))
    return candidates, shifts, middle - lo


def score(windows, still_px, still_weight, k=None):
    """(argmax hits, smooth hits, moving hits, oracle hits) as bool lists."""
    argmax, smooth, moving, oracle = [], [], [], []
    for window in windows:
        truth = window["ball"]
        candidates, shifts, at = window_of(window)
        candidates = top_k(candidates, k)
        here = candidates[at]
        if not here:
            argmax.append(False); smooth.append(False)
            moving.append(False); oracle.append(False)
            continue

        def near(point):
            return point is not None and math.dist(point, truth) <= TOLERANCE_PX

        best = max(here, key=lambda c: c[2])
        argmax.append(near((best[0], best[1])))
        oracle.append(any(near((x, y)) for x, y, _ in here))
        picked = choose([list(frame) for frame in candidates])
        smooth.append(near(picked[at]) if picked else False)
        picked = choose_moving([list(frame) for frame in candidates], shifts,
                               still_px=still_px, still_weight=still_weight)
        moving.append(near(picked[at]) if picked else False)
    return argmax, smooth, moving, oracle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", default="outputs/ball_choice_windows_uniform.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    blob = json.loads((ROOT / args.windows).read_text())
    raw = blob["windows"]
    windows = [w for w in (raw.values() if isinstance(raw, dict) else raw)
               if w.get("ball")]
    fit_half = windows[1::2]
    report_half = windows[0::2]

    grid_px = [2.0, 4.0, 6.0, 8.0, 12.0]
    grid_weight = [0.05, 0.1, 0.2, 0.4, 0.8]
    grid_k = [2, 3, 5, None]
    best = None
    for still_px, still_weight, k in itertools.product(grid_px, grid_weight, grid_k):
        _, _, moving, _ = score(fit_half, still_px, still_weight, k)
        rate = sum(moving) / max(1, len(moving))
        if best is None or rate > best[0]:
            best = (rate, still_px, still_weight, k)
    _, still_px, still_weight, k = best

    argmax, smooth, moving, oracle = score(report_half, still_px, still_weight, k)
    n = len(report_half)
    print()
    print("  eval_ball_moving.py -- the ball is the candidate that moves on the court")
    print()
    print(f"  fitted on {len(fit_half)} windows, reported on {n} it never saw")
    print(f"  chosen constants: still_px = {still_px}, still_weight = {still_weight}, "
          f"candidates kept = {k or 'all'}   (fit-half rate {best[0]:.3f})")
    print()
    print(f"  {'arm':<26} {'rate':>6} {'95% CI':>12}")
    print("  " + "-" * 48)
    for tag, arm in (("oracle (any candidate)", oracle),
                     ("argmax (what ships)", argmax),
                     ("viterbi, smoothness", smooth),
                     ("viterbi, court motion", moving)):
        hits = sum(arm)
        low, high = wilson(hits, n)
        print(f"  {tag:<26} {hits / max(1, n):>6.3f} {low:>5.2f}-{high:<5.2f}")
    print()
    for tag, arm in (("court motion vs argmax", moving),
                     ("court motion vs smoothness", moving)):
        other = argmax if "argmax" in tag else smooth
        wins, losses, p = mcnemar(arm, other)
        print(f"  {tag:<28} {wins} won, {losses} lost, p = {p:.4f}")
    print()
    print("  The arms answer the same frames, so the paired test is the result")
    print("  and the intervals above are context. A selector that wins 3 and")
    print("  loses 3 has changed nothing whatever its rate reads.")
    print()
    if args.out:
        Path(args.out).write_text(json.dumps({
            "windows": args.windows, "fit_n": len(fit_half), "report_n": n,
            "still_px": still_px, "still_weight": still_weight, "top_k": k,
            "oracle": sum(oracle) / max(1, n), "argmax": sum(argmax) / max(1, n),
            "smooth": sum(smooth) / max(1, n), "moving": sum(moving) / max(1, n),
        }, indent=2))
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
