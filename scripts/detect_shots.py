"""Shots from the ball's path past the rim, scored against the official chart.

Declared before tuning: the rule and the split are fixed here, the thresholds
are chosen on the FIRST half of the game and reported on the SECOND, which the
sweep never sees. The earlier measurement on this same broadcast reached
F1 0.396 held out; the gate for Phase 2 is 0.60.

The rule, each piece from basketball rather than from tuning:

- The ball is the NEAREST raw candidate to the rim, not a tracked one: a shot
  is the fastest the ball moves, and a smoothness prior prefers a head
  (measured here before: tracked ball 181 px from the rim at the shot, nearest
  candidate 69 px).
- Distance is measured in RIM WIDTHS, so it means the same at any zoom: a rim
  is 1.5 ft across.
- A shot APPROACHES and RECEDES: far, then near, then far. A ball parked by
  the basket -- a rebound scrum, a held ball -- never approaches.
- One attempt per possession: two minima close together are one shot being
  scored twice, so they are merged.

Rim position comes from the four-class detector, filled across gaps no longer
than GAP_S (the camera moves slowly); the rim is seen on 0.372 of frames on
this broadcast, which is the availability the projected rim exists to beat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TOLERANCE_S = 3.0
GAP_S = 2.0
#: Defaults; --tune sweeps these on the first half only.
APPROACH = 1.6          # rim widths: how near the ball must come
FAR = 3.5               # rim widths: how far it must be before and after
MERGE_S = 6.0


def rim_track(frames, gap_s):
    """(t, x, y, width) of the rim per frame, filled across short gaps."""
    seen = []
    for row in frames:
        rims = [b for b in row["boxes"] if b["cls"] == "rim"]
        if not rims:
            continue
        b = max(rims, key=lambda b: b["conf"])["xyxy"]
        seen.append((row["t"], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], 1.0)))
    if not seen:
        return {}
    times = np.array([s[0] for s in seen])
    out = {}
    for row in frames:
        t = row["t"]
        j = int(np.searchsorted(times, t))
        left = seen[j - 1] if j > 0 else None
        right = seen[j] if j < len(seen) else None
        if left and abs(t - left[0]) <= 1e-6:
            out[t] = left[1:]
            continue
        if left and right and (right[0] - left[0]) <= gap_s:
            f = (t - left[0]) / max(right[0] - left[0], 1e-6)
            out[t] = tuple(l + f * (r - l) for l, r in zip(left[1:], right[1:]))
        elif left and t - left[0] <= gap_s / 2:
            out[t] = left[1:]
        elif right and right[0] - t <= gap_s / 2:
            out[t] = right[1:]
    return out


def distances(frames, rims):
    """(t, nearest ball-to-rim distance in rim widths) where both are known."""
    out = []
    for row in frames:
        rim = rims.get(row["t"])
        if rim is None:
            continue
        balls = [b["xyxy"] for b in row["boxes"] if b["cls"] == "ball"]
        if not balls:
            continue
        x, y, width = rim
        near = min(np.hypot((b[0] + b[2]) / 2 - x, (b[1] + b[3]) / 2 - y) for b in balls)
        out.append((row["t"], near / width))
    return out


def shots(series, approach, far, merge_s):
    """Times where the ball came near the rim, having been far before and after."""
    times = np.array([s[0] for s in series])
    values = np.array([s[1] for s in series])
    near = values <= approach
    events = []
    i = 0
    while i < len(near):
        if not near[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(near) and near[j + 1] and times[j + 1] - times[j] <= 1.0:
            j += 1
        before = values[max(0, i - 25):i]
        after = values[j + 1:j + 26]
        if before.size and after.size and before.max() >= far and after.max() >= far:
            events.append(float(times[i + int(np.argmin(values[i:j + 1]))]))
        i = j + 1
    merged = []
    for t in events:
        if merged and t - merged[-1] <= merge_s:
            continue
        merged.append(t)
    return merged


def score(predicted, truth, tolerance=TOLERANCE_S):
    used, hits = set(), 0
    for t in predicted:
        near = [(abs(t - x), i) for i, x in enumerate(truth) if abs(t - x) <= tolerance and i not in used]
        if near:
            used.add(min(near)[1])
            hits += 1
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", default="outputs/detections/fullgame.json")
    parser.add_argument("--shots", default="outputs/shots_on_video_0042400407.json")
    parser.add_argument("--max-gap-s", type=float, default=2.0,
                        help="how far an official shot may sit from a clock reading to be used")
    parser.add_argument("--tune", action="store_true", help="sweep thresholds on the first half only")
    args = parser.parse_args()

    detections = json.load(open(args.detections))
    frames = detections["frames"]
    official = [s for s in json.load(open(args.shots)) if s["gap_s"] <= args.max_gap_s]
    official.sort(key=lambda s: s["t"])
    middle = official[len(official) // 2]["t"]
    first = [s["t"] for s in official if s["t"] < middle]
    second = [s["t"] for s in official if s["t"] >= middle]

    rims = rim_track(frames, GAP_S)
    series = distances(frames, rims)
    print(f"{len(frames)} cached frames; rim known on {len(rims)} ({len(rims)/len(frames):.1%}); "
          f"ball and rim together on {len(series)} ({len(series)/len(frames):.1%})")
    print(f"official shots used {len(official)} (within {args.max_gap_s:g} s of a clock reading); "
          f"first half {len(first)}, second half {len(second)}, split at video {middle:.0f} s")

    best = (APPROACH, FAR, MERGE_S)
    if args.tune:
        grid = [(a, f, m) for a in (0.8, 1.2, 1.6, 2.0, 2.6) for f in (2.5, 3.5, 5.0) for m in (4.0, 6.0, 10.0)]
        scored = []
        for a, f, m in grid:
            p, r, f1 = score(shots(series, a, f, m), first)
            scored.append((f1, p, r, a, f, m))
        scored.sort(reverse=True)
        f1, p, r, a, f, m = scored[0]
        best = (a, f, m)
        print(f"tuned on the FIRST half: approach {a} rim widths, far {f}, merge {m:g} s -> P {p:.3f} R {r:.3f} F1 {f1:.3f}")
        for f1, p, r, a, f, m in scored[1:4]:
            print(f"   runner-up: approach {a} far {f} merge {m:g} -> F1 {f1:.3f}")

    a, f, m = best
    predicted = shots(series, a, f, m)
    for name, truth in (("FIRST half (tuned on)", first), ("SECOND half (held out)", second)):
        window = [t for t in predicted if (t < middle) == (truth is first)]
        p, r, f1 = score(window, truth)
        print(f"  {name:22s} predicted {len(window):3d}  P {p:.3f}  R {r:.3f}  F1 {f1:.3f}"
              + ("   -> PASS" if truth is second and f1 >= 0.60 else ("   -> FAIL" if truth is second else "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
