"""Read the game clock off every second of a broadcast, with no hand labelling.

Phase 2 scores shots against the official shot chart, which gives each attempt
by period and game clock -- not by video time. So the video's timeline has to
be read off the scoreboard itself.

- Locate: autoscoreboard.locate_clock over a minute of frames one second
  apart -- the region that reads as 3-4 digits and whose last digit ticks.
- Learn: autoscoreboard.bootstrap_templates -- the ones digit counts down,
  and the tens digit changing marks the 0 -> 9 wrap that pins the alphabet.
- Read: scoreboard.read_clock on every sampled frame. The last minute of a
  period shows "SS.T"; a three-digit read whose seconds exceed 59 is that.
- Period: the clock counts down within a period, so a reading more than
  PERIOD_JUMP_S above the running clock is a new period (12:00 after 0:00;
  overtime restarts at 5:00).
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
PERIOD_JUMP_S = 120.0


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
    """[(t, seconds)] in video order -> [(t, period, seconds)], dropping misreads."""
    out, period, running, misreads = [], 1, None, 0
    for t, seconds in readings:
        if running is None:
            running = seconds
        elif seconds > running + PERIOD_JUMP_S:
            period += 1                          # a reset: the next period began
        elif seconds > running + 1.0:
            misreads += 1                        # a clock that runs backwards
            continue
        running = seconds
        out.append((t, period, seconds))
    return out, misreads


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

    raw = []
    for t in np.arange(0.0, duration, args.step):
        frame = frame_at(float(t))
        if frame is None:
            continue
        text, _ = read_clock(frame[top:bottom, left:right], templates)
        seconds = parse(text)
        if seconds is not None:
            raw.append((float(t), seconds))
    rows, misreads = assign_periods(raw)
    out = Path(args.out or f"outputs/clock/{Path(args.video).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "roi": list(location.roi), "step": args.step,
               "readings": [{"t": t, "period": p, "seconds": s, "elapsed": elapsed(p, s)}
                            for t, p, s in rows]}, open(out, "w"), indent=0)
    periods = sorted({p for _, p, _ in rows})
    print(f"{len(np.arange(0.0, duration, args.step))} frames sampled; {len(raw)} read; "
          f"{misreads} dropped as running backwards ({misreads / max(len(raw), 1):.1%}); periods seen {periods}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
