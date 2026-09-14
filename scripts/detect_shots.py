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
  the basket -- a rebound scrum, a held ball -- never approaches. MEASURED,
  and it earns nothing: against a control that fires on proximity alone this
  test rejects 0 of 286 candidate events on Game 1 and 2 of 319 on Game 7, and
  moves F1 on neither. The ball is simply never parked at the rim for the five
  seconds the windows span. Keep it if you like, but the detector should be
  described as what it is -- a ball-like box came within APPROACH rim widths
  of the rim while the clock was running -- not as an arc-shaped rule.
- One attempt per possession: two minima close together are one shot being
  scored twice, so they are merged.

Rim position comes from the four-class detector, filled across gaps no longer
than GAP_S (the camera moves slowly); the rim is seen on 0.372 of frames on
this broadcast, which is the availability the projected rim exists to beat.

Ball boxes are kept from BALL_CONF up. The cache is written at 0.10 so that no
real ball is lost, but at that level the detector puts ~2.9 "ball" boxes on
every frame, many of them on the rim and net themselves, and the nearest
candidate to the rim is then junk: over official shot windows the closest
approach read 0.33 rim widths against 0.40 at random moments -- no signal at
all. At 0.25 it reads 1.41 against 3.91, and the ball comes within one rim
width in 42% of shot windows against 17% of random ones. Filtering by box SIZE
instead throws away the real detections: the detector's genuine ball boxes are
loose, ~32 px wide at 720p against the rim's 39, and a 22 px cap drops coverage
from 53% of frames to 10%.

LIVE PLAY (added after the fact, and declared as such): a call is kept only
where the game clock is actually running. Of 181 second-half calls, 76 sat
more than 30 s from any official shot -- replays of a basket look exactly like
the basket, free throws are shots the field-goal chart does not contain, and
warm-ups and timeouts put balls through rims too. All three happen with the
clock stopped or off screen, while 98% of official attempts happen with it
running.

That idea came from looking at where the SECOND half's false alarms fell, so
the second-half figure it produces is optimistic: the thresholds were tuned on
the first half, but the rule was informed by the test half. A clean number
needs a different game, scored with everything frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TOLERANCE_S = 3.0
GAP_S = 2.0
#: Frozen at the values --tune chose on the FIRST half of Finals G7
#: (P 0.598 R 0.778 F1 0.676 there; 0.619 on that game's held-out half). They
#: are the defaults so another game can be scored without touching anything.
APPROACH = 2.6          # rim widths: how near the ball must come
FAR = 5.0               # rim widths: how far it must be before and after
MERGE_S = 6.0
BALL_CONF = 0.25
#: The clock must fall by this much within +/- LIVE_WINDOW_S of a call.
LIVE_WINDOW_S = 3.0
LIVE_DROP_S = 0.5


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
        balls = [b["xyxy"] for b in row["boxes"] if b["cls"] == "ball" and b["conf"] >= BALL_CONF]
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


def live_play(readings):
    """A test for 'the game clock was running here', from the clock readings."""
    times = np.array([r["t"] for r in readings])
    seconds = np.array([r["seconds"] for r in readings])
    periods = np.array([r["period"] for r in readings])

    def running(t):
        near = np.abs(times - t) <= LIVE_WINDOW_S
        if near.sum() < 2 or len(set(periods[near])) > 1:
            return False
        return (seconds[near].max() - seconds[near].min()) >= LIVE_DROP_S
    return running


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
    parser.add_argument("--clock", default="outputs/clock/fullgame.json")
    parser.add_argument("--all-play", action="store_true", help="keep calls made with the clock stopped")
    parser.add_argument("--tune", action="store_true", help="sweep thresholds on the first half only")
    parser.add_argument("--out", default=None,
                        help="write every call with its video time and whether the "
                             "official record agrees. The misses are the point: a "
                             "false alarm you can watch is the only honest way to "
                             "show what a 0.52 precision actually looks like.")
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
    running = (lambda t: True) if args.all_play else live_play(json.load(open(args.clock))["readings"])
    print(f"{len(frames)} cached frames; rim known on {len(rims)} ({len(rims)/len(frames):.1%}); "
          f"ball and rim together on {len(series)} ({len(series)/len(frames):.1%})")
    print(f"official shots used {len(official)} (within {args.max_gap_s:g} s of a clock reading); "
          f"first half {len(first)}, second half {len(second)}, split at video {middle:.0f} s")

    best = (APPROACH, FAR, MERGE_S)
    if args.tune:
        grid = [(a, f, m) for a in (0.8, 1.2, 1.6, 2.0, 2.6) for f in (2.5, 3.5, 5.0) for m in (4.0, 6.0, 10.0)]
        scored = []
        for a, f, m in grid:
            # Only the first half's predictions: scoring the whole game's
            # against the first half's shots counted every second-half call as
            # a false alarm and tuned for recall at any price.
            predicted = [t for t in shots(series, a, f, m) if t < middle and running(t)]
            p, r, f1 = score(predicted, first)
            scored.append((f1, p, r, a, f, m))
        scored.sort(reverse=True)
        f1, p, r, a, f, m = scored[0]
        best = (a, f, m)
        print(f"tuned on the FIRST half: approach {a} rim widths, far {f}, merge {m:g} s -> P {p:.3f} R {r:.3f} F1 {f1:.3f}")
        for f1, p, r, a, f, m in scored[1:4]:
            print(f"   runner-up: approach {a} far {f} merge {m:g} -> F1 {f1:.3f}")

    a, f, m = best
    predicted = [t for t in shots(series, a, f, m) if running(t)]
    for name, truth in (("FIRST half (tuned on)", first), ("SECOND half (held out)", second)):
        window = [t for t in predicted if (t < middle) == (truth is first)]
        p, r, f1 = score(window, truth)
        print(f"  {name:22s} predicted {len(window):3d}  P {p:.3f}  R {r:.3f}  F1 {f1:.3f}"
              + ("   -> PASS" if truth is second and f1 >= 0.60 else ("   -> FAIL" if truth is second else "")))

    if args.out:
        # Label each call against the official record, using the same greedy
        # matching `score` uses so the file and the printed F1 cannot disagree.
        truth = [s["t"] for s in official]
        used, calls = set(), []
        for t in sorted(predicted):
            near = [(abs(t - x), i) for i, x in enumerate(truth)
                    if abs(t - x) <= TOLERANCE_S and i not in used]
            match = min(near) if near else None
            if match:
                used.add(match[1])
            calls.append({"video_s": round(float(t), 1),
                          "half": "first" if t < middle else "second",
                          "agrees_with_official": bool(match),
                          "official_s": round(truth[match[1]], 1) if match else None,
                          "off_by_s": round(match[0], 1) if match else None})
        missed = [{"video_s": round(x, 1)} for i, x in enumerate(truth) if i not in used]
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"note": "shots the VISION pipeline called from pixels -- ball and rim "
                           "detected per frame, no play-by-play involved. Each call is "
                           "marked with whether the official record agrees.",
                   "detections": args.detections,
                   "approach_rim_widths": a, "far_rim_widths": f, "merge_s": m,
                   "tolerance_s": TOLERANCE_S,
                   "calls": calls,
                   "official_shots_missed": missed}, open(out, "w"), indent=1)
        hits = sum(1 for c in calls if c["agrees_with_official"])
        print(f"\n  wrote {len(calls)} calls ({hits} agree with the official record, "
              f"{len(calls)-hits} do not) and {len(missed)} official shots it never called")
        print(f"  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
