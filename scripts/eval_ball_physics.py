#!/usr/bin/env python3
"""Is the tracked ball moving at a speed a basketball can move? No labels.

WHY THIS EXISTS. Every ball number in this repository needs hand-clicked
truth: 135 uniform frames on one broadcast, 13 on another, none on the fourth.
A metric that needs labels cannot follow a new broadcast, and "does this work
on an arbitrary game" is the whole question.

Physics does not need labels. A rim is 18 inches across, so the rim box in the
frame is a ruler, and it is the one object on the court whose real size is
known. The ball's displacement between two frames, divided by the rim's width
in those frames, is a distance in feet however far away the camera is and
however much it has zoomed. The camera's own motion comes out because the rim
moves with it -- the same trick `ball_track.choose_moving` uses.

THE BOUND. The hardest NBA pass measured is around 60 mph and a jump shot
leaves the hand near 25; 40 mph is 58.7 ft/s, and a rim is 1.5 ft, so 39 rim
widths a second is already generous by a wide margin. Anything faster is not a
ball, it is the selector jumping to a different object.

WHAT THE NUMBER MEANS, PRECISELY. It is not ball ACCURACY: a selector that
locks onto one stationary logo for a whole clip scores a perfect 1.000 here,
which is why this prints beside the labelled top-1 rate rather than instead of
it. It is a NECESSARY condition. A track that breaks it on half its steps
cannot be a trajectory, whatever its per-frame accuracy, and nothing built on
it -- a handler, a rebound, a possession -- can inherit anything but noise.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.ball_track import choose_moving  # noqa: E402
from courtvision.games import get, registry  # noqa: E402
from courtvision.stats import wilson  # noqa: E402

#: An NBA rim is 18 inches across.
RIM_FT = 1.5
#: Rim widths per second a basketball cannot exceed. 40 mph = 58.7 ft/s.
MAX_RIM_WIDTHS_PER_S = 58.7 / RIM_FT
#: A step across a shot boundary is not motion, it is a different camera, and
#: the rim jumps with it. Steps where the rim moves further than this in one
#: frame are dropped as cuts rather than counted as impossible ball speeds.
CUT_RIM_STEP_PX = 120.0


def rim_of(row):
    rims = [b for b in row["d"] if b[0] == "r"]
    if not rims:
        return None, None
    best = max(rims, key=lambda b: b[1])
    return (((best[2] + best[4]) / 2.0, (best[3] + best[5]) / 2.0),
            max(1.0, best[4] - best[2]))


def ball_series(rows, picker: str):
    """[(frame, centre)] for one clip, by the named selection rule."""
    rows = sorted(rows, key=lambda r: r["f"])
    if picker == "argmax":
        out = []
        for row in rows:
            balls = [b for b in row["d"] if b[0] == "b"]
            if not balls:
                continue
            best = max(balls, key=lambda b: b[1])
            out.append((int(row["f"]),
                        ((best[2] + best[4]) / 2.0, (best[3] + best[5]) / 2.0)))
        return out
    candidates = [[((b[2] + b[4]) / 2.0, (b[3] + b[5]) / 2.0, b[1])
                   for b in row["d"] if b[0] == "b"] for row in rows]
    shifts = [None]
    for before, after in zip(rows, rows[1:]):
        one, _ = rim_of(before)
        two, _ = rim_of(after)
        shifts.append(None if one is None or two is None
                      else (two[0] - one[0], two[1] - one[1]))
    picked = choose_moving(candidates, shifts)
    return [(int(row["f"]), point)
            for row, point in zip(rows, picked) if point is not None]


def speeds(rows, picker: str, fps: float, dropped: dict | None = None):
    """Court-relative speeds in rim widths per second, one per usable step.

    `dropped` collects why steps were not scored. A metric that silently
    discards half its denominator is not a measurement, so the shares are
    printed beside the rate.
    """
    series = dict(ball_series(rows, picker))
    ordered = sorted(series)
    rims = {int(row["f"]): rim_of(row) for row in rows}
    out = []
    for before, after in zip(ordered, ordered[1:]):
        gap = (after - before) / fps
        if gap <= 0:
            continue
        (one, width_one), (two, width_two) = rims.get(before, (None, None)), \
            rims.get(after, (None, None))
        if one is None or two is None:
            if dropped is not None:
                dropped["no rim in frame"] = dropped.get("no rim in frame", 0) + 1
            continue           # no ruler and no camera estimate: not scored
        camera = math.hypot(two[0] - one[0], two[1] - one[1])
        if camera > CUT_RIM_STEP_PX:
            if dropped is not None:
                dropped["a cut"] = dropped.get("a cut", 0) + 1
            continue           # a cut, not a step
        width = (width_one + width_two) / 2.0
        a, b = series[before], series[after]
        moved = math.hypot(b[0] - a[0] - (two[0] - one[0]),
                           b[1] - a[1] - (two[1] - one[1]))
        out.append(moved / width / gap)
    return out


def evaluate(key: str, *, limit: int | None = None) -> dict:
    broadcast = get(key)
    cache = json.loads((ROOT / broadcast.clip_detections).read_text())
    clips = cache["clips"]
    # `f` in a detection row is a VIDEO frame index, not a row index, so the
    # gap between two rows is converted with the VIDEO's fps. Using the cache's
    # `rate` (the sampling rate, 15 Hz) instead divided Houston's gaps by four
    # and made its ball look four times slower than it is -- and therefore
    # physically possible when it was not. Frames are not seconds; this file is
    # the fourth place in this repository that has had to learn it.
    fps = float(cache.get("fps") or broadcast.fps)
    names = sorted(clips)
    if limit:
        names = names[:limit]
    row = {"game": key, "label": broadcast.label, "video_fps": fps,
           "row_rate_hz": fps / float(cache.get("step") or 1)}
    for picker in ("argmax", "court motion"):
        values = []
        dropped: dict[str, int] = {}
        for name in names:
            values += speeds(clips[name], picker, fps, dropped)
        possible = sum(1 for v in values if v <= MAX_RIM_WIDTHS_PER_S)
        low, high = wilson(possible, len(values))
        row[picker] = {
            "steps": len(values),
            "possible": possible / max(1, len(values)),
            "ci": [low, high],
            "median_rw_per_s": statistics.median(values) if values else math.nan,
            "p90_rw_per_s": (statistics.quantiles(values, n=10)[8]
                             if len(values) > 10 else math.nan),
            "dropped": dropped,
            "scored_share": len(values) / max(1, len(values) + sum(dropped.values())),
        }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bar", type=float, default=0.85)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    keys = args.game or list(registry())
    results = []
    print()
    print("  eval_ball_physics.py -- can a basketball move the way this track says?")
    print(f"  the bound is {MAX_RIM_WIDTHS_PER_S:.0f} rim widths a second "
          f"(40 mph), measured in a rim's own width so zoom and distance drop out")
    print()
    print(f"  {'game':<6} {'selector':<14} {'possible':>9} {'95% CI':>12} "
          f"{'median':>8} {'p90':>8} {'steps':>7}  verdict")
    print("  " + "-" * 82)
    for key in keys:
        row = evaluate(key, limit=args.limit)
        results.append(row)
        for picker in ("argmax", "court motion"):
            got = row[picker]
            low, high = got["ci"]
            verdict = ("PASS" if low >= args.bar else
                       "PASS (point)" if got["possible"] >= args.bar else "FAIL")
            print(f"  {key:<6} {picker:<14} {got['possible']:>9.3f} "
                  f"{low:>5.2f}-{high:<5.2f} {got['median_rw_per_s']:>8.1f} "
                  f"{got['p90_rw_per_s']:>8.1f} {got['steps']:>7}  {verdict}")
            why = ", ".join(f"{k}: {v}" for k, v in sorted(got["dropped"].items()))
            print(f"  {'':<6} {'':<14} scored {got['scored_share']:.3f} of steps"
                  + (f" -- dropped {why}" if why else ""))
    print()
    print("  possible  share of steps a basketball could actually have made.")
    print("  median    typical speed in rim widths a second. A ball in play")
    print("            spends most of its time dribbled or held, so a healthy")
    print("            median is a few rim widths a second, not tens.")
    print()
    print("  THIS IS NOT ACCURACY. A selector that locks onto one stationary")
    print("  logo for a whole clip scores a perfect 1.000 here. It is a")
    print("  NECESSARY condition: a track that breaks physics on half its steps")
    print("  cannot be a trajectory, and nothing built on it inherits anything")
    print("  but noise. Quote it beside the labelled top-1 rate, never instead.")
    print()
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
