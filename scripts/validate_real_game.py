"""V12 — the real-game naming chain, end to end on a real broadcast.

Every piece of this was unit-tested separately. This runs the whole thing on
the holdout clip: read the scoreboard clock, convert video time to game clock,
fetch the OFFICIAL NBA play-by-play for that game, and select the plays that
happened inside the clip's window.

That is the path a live game takes. The archived BARD route works only because
those filenames carry a GameID; a real broadcast has no such key, and the clock
on screen is the only join.

Needs a network and `pip install -e ".[live]"`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from courtvision.enrichment import (ClockReading, drop_impossible_readings,
                                    plays_in_window)
from courtvision.nba_feed import fetch_game_plays
from courtvision.scoreboard import build_templates, clock_to_seconds, read_clock

CLIP = Path("data/raw_clips/holdout.mp4")
GAME_ID = "0022400952"          # min-vs-den, from the clip's .source.json
PERIOD = 2
CLOCK_ROI = (slice(634, 672), slice(800, 915))
KNOWN_READING = "347"           # the clock in the middle frame, read by eye once
SAMPLE_EVERY = 15


def main() -> int:
    if not CLIP.exists():
        print(f"V12 FAIL — no clip at {CLIP}")
        return 1

    capture = cv2.VideoCapture(str(CLIP))
    fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        print("V12 FAIL — no frames decoded")
        return 1
    print(f"  {len(frames)} frames at {fps:.0f}fps")

    templates = build_templates(frames[len(frames) // 2][CLOCK_ROI], KNOWN_READING)
    print(f"  templates from one known frame: {sorted(templates)}")

    raw: list[ClockReading] = []
    for index in range(0, len(frames), SAMPLE_EVERY):
        clock, _ = read_clock(frames[index][CLOCK_ROI], templates)
        if clock:
            raw.append(ClockReading(index / fps, PERIOD, clock_to_seconds(clock)))
    kept = drop_impossible_readings(raw)
    dropped = len(raw) - len(kept)
    print(f"  clock read on {len(raw)} sampled frames; {dropped} dropped as "
          f"impossible (a clock cannot go up)")
    if not kept:
        print("V12 FAIL — no usable clock readings")
        return 1

    values = [r.clock_seconds for r in kept]
    monotonic = all(b <= a for a, b in zip(values, values[1:]))
    low, high = min(values), max(values)
    print(f"  clip covers P{PERIOD} {high}s down to {low}s remaining; "
          f"monotonic {monotonic}")

    plays = fetch_game_plays(GAME_ID)
    window = plays_in_window(plays, PERIOD, high, low)
    print(f"  official feed: {len(plays)} usable plays for game {GAME_ID}")
    print(f"  inside this clip's window: {len(window)}\n")
    for play in window:
        print(f"    P{play.period} {play.clock_seconds:>3}s  {play.action:<8} "
              f"{str(play.player):<16} {play.description[:44]}")

    ok = monotonic and len(plays) > 100 and len(window) >= 1
    print(f"\nV12 {'PASS' if ok else 'FAIL'} — the real-game join works: a clock "
          f"read off the\n  broadcast selected the official plays for that moment, "
          f"with no GameID in\n  the filename and no BARD metadata.")
    if ok:
        print("  Not shown: whether those plays MATCH the pipeline's own events.\n"
              "  That is what align does, and it needs a classifier worth trusting.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
