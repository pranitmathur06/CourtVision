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

import json
import sys
from pathlib import Path

import cv2

from courtvision.enrichment import (ClockReading, drop_impossible_readings,
                                    plays_in_window)
from courtvision.nba_feed import fetch_game_plays
from courtvision.scoreboard import (build_templates_from_many,
                                    clock_to_seconds, read_clock)

CLIP = Path("data/raw_clips/holdout.mp4")
GAME_ID = "0022400952"          # min-vs-den, from the clip's .source.json
PERIOD = 2
CLOCK_ROI = (slice(634, 672), slice(800, 915))
# Frames whose clock was read by eye once, to build templates. One frame is not
# enough: built from 3:47 alone the reader can only read values made of 3, 4 and
# 7, and it does not fail quietly on the rest — it matched a 7 against the 3
# template and returned a confident 3:43 on six frames.
KNOWN_FRAMES = ((0, "352"), (60, "351"), (105, "350"), (150, "349"),
                (225, "348"), (300, "347"))
# Nine of ten digits. A 6 never appears between 3:52 and 3:47, so it cannot
# be learned from this clip at all — the reader will decline any clock
# containing one rather than guess, which is the right failure.
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

    templates = build_templates_from_many(
        [(frames[index][CLOCK_ROI], reading) for index, reading in KNOWN_FRAMES])
    print(f"  templates from {len(KNOWN_FRAMES)} known frames: "
          f"{sorted(templates)} ({len(templates)}/10 digits)")

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

    # Now the part V12 previously skipped: do the PIPELINE's own events land at
    # the same game-clock moments as the official plays?
    print("\n  running the pipeline on the same clip...")
    from courtvision.config import Config
    from courtvision.enrichment import game_clock_at
    from scripts.run_pipeline import run_pipeline

    # narrate=False: this checks the JOIN, not the commentary, and narration
    # costs an API call per run.
    out_dir = Path("outputs/v12")
    run_pipeline(str(CLIP), str(out_dir), Config(), narrate=False)
    payload = json.loads((out_dir / "commentary.json").read_text())
    events = payload.get("events", [])
    print(f"  pipeline produced {len(events)} events")

    # Temporal proximity alone is a misleading score here, and the first version
    # of this reported it as one. The clock STOPS on a dead ball, so six events
    # spanning four seconds of video all mapped to 227s and all "matched" the
    # single official play there. Agreement means the ACTION agrees too.
    near_in_time = 0
    agreeing = 0
    collapsed: dict[int, int] = {}
    for event in events:
        at = game_clock_at(kept, event["time_s"])
        if at is None:
            print(f"    t={event['time_s']:>5.2f}s  {event['action']:<8} "
                  f"clock unknown here — nothing claimed")
            continue
        period, clock = at
        collapsed[clock] = collapsed.get(clock, 0) + 1
        near = [p for p in window if abs(p.clock_seconds - clock) <= 2]
        note = ""
        if near:
            near_in_time += 1
            official = near[0]
            same = official.action == event["action"]
            agreeing += same
            note = (f"  official: {official.player} {official.action} "
                    f"{'AGREES' if same else 'differs'}")
        print(f"    t={event['time_s']:>5.2f}s  {event['action']:<8} "
              f"P{period} {clock:>3}s{note}")

    stalled = max(collapsed.values()) if collapsed else 0
    print(f"\n  {near_in_time}/{len(events)} events fell within 2s of an official "
          f"play; {agreeing} agreed on the action")
    if stalled > 1:
        print(f"  {stalled} of them mapped to the SAME game second: the clock stops "
              f"on a dead\n  ball, so clock-based joining cannot separate events "
              f"inside a stoppage.")

    ok = monotonic and len(plays) > 100 and len(window) >= 1
    print(f"\nV12 {'PASS' if ok else 'FAIL'} — the real-game join works: a clock "
          f"read off the\n  broadcast selected the official plays for that moment, "
          f"with no GameID in\n  the filename and no BARD metadata.")
    if ok:
        print(f"  The pipeline's own actions agreed with the official play "
              f"{agreeing}/{near_in_time} times.\n"
              f"  That is the classifier's accuracy showing through, not the "
              f"join's: rebound\n  scored 0.29 on the probe and this clip is "
              f"mostly called rebound. V7 is what\n  moves that number.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
