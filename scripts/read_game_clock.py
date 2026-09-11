"""Read the game clock off every second of a broadcast, with no hand labelling.

Phase 2 scores shots against the official shot chart, which gives each attempt
by period and game clock -- not by video time. So the video's timeline has to
be read off the scoreboard itself.

- Locate: autoscoreboard.locate_clock over a minute of frames one second
  apart -- the region that reads as 3-4 digits and whose last digit ticks.
- Learn: autoscoreboard.bootstrap_templates -- the ones digit counts down,
  and the tens digit changing marks the 0 -> 9 wrap that pins the alphabet.
- Read: scoreboard.read_clock on every sampled frame. The last minute of a
  period shows tenths ("35.9"), which comes back as three digits and reads
  equally as 3:59 -- and 3:59 looks like the clock jumping back up, which
  earlier turned one quarter into four. Both readings are kept and settled by
  the one thing a clock always does within a period: it falls. Take the
  LARGEST reading that is not above the previous one; when none is -- a new
  quarter, a replay, a misread -- keep the largest and leave it to the period
  split below. That reads 9:55 mid-quarter and 35.9 in the last minute alike.
  Preferring tenths whenever the clock was under a minute turned "1:05" into
  10.5 and split every quarter in two; dropping the sample instead threw away
  every quarter after the first, since each one starts above the last.
- Period: the clock only falls within a period, so the readings are cut at
  every upward jump over RESET_JUMP_S and the runs are numbered in order.
  Runs shorter than MIN_RUN readings are replays or stray misreads and are
  dropped. Rules keyed to exact values all failed on this broadcast: any
  upward jump as a reset gave 20 periods; requiring the last reading near zero
  merged three quarters; requiring the new one at 12:00 missed a quarter whose
  first readable frame was 11:24.
- Raw readings are saved beside the resolved ones, so a change of rule costs
  no video pass.
- Self-check: within a period the clock never goes up. Readings that do, by
  more than a second, are misreads, counted and dropped -- the same test that
  gave the score reader its 2% error rate with no labels.

Output: JSON rows {t, period, seconds, elapsed} for every frame read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PERIOD_S = 720.0
OT_S = 300.0
#: An upward jump this large ends a run of readings.
RESET_JUMP_S = 30.0
#: A run this short is a replay or a misread, not a period.
MIN_RUN = 20


def readings_from(text: str | None):
    """The seconds a clock text could mean: [minutes:seconds] and [seconds.tenths]."""
    if not text or ":" not in text:
        return []
    left, right = text.split(":")
    if not left.isdigit() or not right.isdigit():
        return []
    out = []
    if len(right) == 2 and int(right) <= 59:
        value = int(left) * 60 + int(right)
        if value <= PERIOD_S:
            out.append(float(value))
    digits = left + right
    if len(digits) == 3:                        # "359" is 3:59 or 35.9
        out.append(int(digits[:2]) + int(digits[2]) / 10.0)
    return out


def resolve(candidates):
    """[(t, [possible seconds])] -> [(t, seconds)], settling ties by continuity."""
    out, running = [], None
    for t, options in candidates:
        if not options:
            continue
        if running is None:
            seconds = max(options)
        else:
            below = [v for v in options if v <= running + 1.0]
            seconds = max(below) if below else max(options)
        out.append((t, seconds))
        running = seconds
    return out


def parse(text: str | None):
    """'M:SS', 'MM:SS' or a last-minute 'SS.T' read as 'S:ST' -> seconds, or None."""
    if not text or ":" not in text:
        return None
    left, right = text.split(":")
    if not left.isdigit() or not right.isdigit():
        return None
    if int(right) <= 59 and len(right) == 2:
        value = int(left) * 60 + int(right)
        return float(value) if value <= PERIOD_S else None
    digits = left + right                       # e.g. "1:82" was "18.2"
    if len(digits) == 3:
        return int(digits[:2]) + int(digits[2]) / 10.0
    return None


def assign_periods(readings):
    """[(t, seconds)] in video order -> [(t, period, seconds)], dropping short runs."""
    runs, current = [], []
    for t, seconds in readings:
        if current and seconds > current[-1][1] + RESET_JUMP_S:
            runs.append(current)
            current = []
        elif current and seconds > current[-1][1] + 1.0:
            continue                             # a small step back: a misread
        current.append((t, seconds))
    if current:
        runs.append(current)
    kept = [run for run in runs if len(run) >= MIN_RUN]
    dropped = sum(len(run) for run in runs if len(run) < MIN_RUN)
    out = [(t, period, seconds)
           for period, run in enumerate(kept, start=1) for t, seconds in run]
    return out, dropped


def elapsed(period: int, seconds: float) -> float:
    if period <= 4:
        return (period - 1) * PERIOD_S + (PERIOD_S - seconds)
    return 4 * PERIOD_S + (period - 5) * OT_S + (OT_S - seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--learn-start", type=float, default=900.0)
    parser.add_argument("--learn-seconds", type=int, default=120)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import cv2

    from courtvision.autoscoreboard import bootstrap_templates, locate_clock
    from courtvision.scoreboard import read_clock

    capture = cv2.VideoCapture(args.video)
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)

    def frame_at(t):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        return frame if ok else None

    location, templates = None, {}
    for start in (args.learn_start, args.learn_start + 600, args.learn_start + 1800, 2 * args.learn_start + 1800):
        frames = [f for f in (frame_at(start + s) for s in range(args.learn_seconds)) if f is not None]
        location = locate_clock(frames)
        if location is None:
            continue
        templates = bootstrap_templates(frames, location.roi)
        if len(templates) == 10:
            break
    if location is None or len(templates) < 10:
        print(f"FAIL - clock not located or digits not learned (roi {location}, digits {sorted(templates)})")
        return 1
    top, bottom, left, right = location.roi
    print(f"clock at rows {top}-{bottom}, cols {left}-{right}; ticks {location.ticks}/{location.samples}; digits {sorted(templates)}")

    candidates = []
    for t in np.arange(0.0, duration, args.step):
        frame = frame_at(float(t))
        if frame is None:
            continue
        text, _ = read_clock(frame[top:bottom, left:right], templates)
        options = readings_from(text)
        if options:
            candidates.append((float(t), options))
    raw = resolve(candidates)
    rows, misreads = assign_periods(raw)
    out = Path(args.out or f"outputs/clock/{Path(args.video).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "roi": list(location.roi), "step": args.step,
               "raw": [{"t": t, "options": options} for t, options in candidates],
               "readings": [{"t": t, "period": p, "seconds": s, "elapsed": elapsed(p, s)}
                            for t, p, s in rows]}, open(out, "w"), indent=0)
    periods = sorted({p for _, p, _ in rows})
    print(f"{len(np.arange(0.0, duration, args.step))} frames sampled; {len(raw)} read; "
          f"{misreads} dropped in short runs ({misreads / max(len(raw), 1):.1%}); periods seen {periods}")
    for p in periods:
        run = [(t, s) for t, q, s in rows if q == p]
        print(f"  period {p}: {len(run):4d} readings, clock {max(s for _, s in run):.0f} -> "
              f"{min(s for _, s in run):.1f} s, video {run[0][0]:.0f}..{run[-1][0]:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
