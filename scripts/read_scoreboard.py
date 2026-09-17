"""Read the SCORE and the shot clock off the broadcast, not just the game clock.

THE GAP THIS CLOSES. `src/courtvision/scoreboard_events.py` turns per-frame
score and shot-clock readings into typed events, and its docstring carries the
best numbers in this project, measured on an uncut broadcast against the official
play-by-play with no ball detection anywhere in the path:

    3pt make   P 1.000  R 0.955  F1 0.977
    2pt make   P 0.951  R 0.929  F1 0.940
    any make   P 0.960  R 0.880  F1 0.918
    free throw P 0.946  R 0.795  F1 0.864

Deriving field goals from a true 25 Hz ball trajectory tops out at F1 0.859 --
the scoreboard beats the ceiling of the derived approach, because it is observed
rather than inferred. `docs/continuous-game-accuracy.md` concludes that the only
architecture that reaches 85% is "vision for shot timing, the scoreboard for
outcomes".

**And nothing in this repository produced those readings.** `run_broadcast.py`
requires `--readings` and no script writes it; `outputs/broadcast/timeline.json`
holds 449 events from a run whose input is gone. So the project's best result was
unreproducible, and `score_game_end_to_end.py` reports that rung as BLOCKED. This
is the missing driver.

HOW IT WORKS, and why it is not a second bootstrap problem.

`read_game_clock.py` learns digit templates unsupervised by exploiting the one
thing that makes a game clock special: it counts DOWN one per second, so the
ones digit's ordering is known and a 0 -> 9 wrap anchors it to absolute values.
A score has no such anchor -- it changes rarely and by one, two or three.

It does not need one. **The score is the same font on the same panel**, a couple
of hundred pixels from the clock, so the templates the clock already learned read
it directly. That turns an unsolved bootstrap into a lookup.

WHAT KEEPS A MISREAD FROM BECOMING A BASKET. Three constraints, none of them
tuned:

  MONOTONIC. A team's score never falls. A reading below the running maximum is
  a misread and is dropped, not smoothed.
  BOUNDED. It cannot rise by more than three in one possession
  (`scoreboard_events.MAX_POINTS_PER_SCORE`), so a jump of nine is a misread of
  the tens digit.
  CORROBORATED. A change has to hold for `HOLD_SAMPLES` consecutive readings
  before it is believed. A single frame with a graphic over the bar is the
  common failure and it never holds.

THE SHOT CLOCK IS FOUND BY BEHAVIOUR, NOT BY POSITION. `locate_clock` is already
known to lock onto the shot clock rather than the game clock -- both tick, both
read as 3-4 glyphs -- which is a recorded defect of that function and exactly the
property needed here. The two are told apart by what they DO: a game clock falls
monotonically across a whole period, a shot clock resets to 24 over and over. So
the region that resets most often, and is not the clock roi already known, is the
shot clock.

OUTPUT is the three streams `run_broadcast.py` wants, with the clock carried
through from the clock run so the three share one timebase.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from read_game_clock import stream  # noqa: E402

#: A score change must hold this many consecutive readings to be believed. One
#: frame with a graphic over the bar is the common failure and never holds.
HOLD_SAMPLES = 3
#: A shot clock resets to this. Used to tell it from the game clock, which falls
#: monotonically across a period and never jumps back up.
SHOT_CLOCK_RESET = 24
#: A reset counts when the reading rises by at least this much.
RESET_RISE = 5.0
#: Seconds between sampled frames. One is enough: a score changes at 0.022-0.035
#: per second against the clock's 0.35, which is the whole basis for telling
#: them apart.
STEP_S = 1.0


def digits_of(image, templates, min_score: float = 0.5):
    """An integer from a region of digits, or None when any glyph is unsure.

    Unlike the clock there is no colon and no fixed width: a score is one, two
    or three glyphs. A single unsure glyph makes the whole reading None rather
    than a guess, because a wrong tens digit is a nine-point basket.
    """
    from courtvision.scoreboard import (_match, clock_glyphs,
                                        normalise_polarity)

    glyphs = clock_glyphs(normalise_polarity(image))
    # A region that clips an adjacent digit leaves a sliver at one edge -- a
    # 17x8 fragment where the real digits are 17x19. It is not a digit and it
    # matched nothing above 0.23, which was rejecting the WHOLE reading and
    # made every one of 121 candidate regions read as illegible.
    if glyphs:
        widths = sorted(g.image.shape[1] for g in glyphs)
        typical = widths[len(widths) // 2]
        glyphs = [g for g in glyphs if g.image.shape[1] >= 0.55 * typical]
    if not 1 <= len(glyphs) <= 3:
        return None
    text = []
    for glyph in glyphs:
        digit, score = _match(glyph.image, templates)
        if digit == "?" or score < min_score:
            return None
        text.append(digit)
    try:
        return int("".join(text))
    except ValueError:
        return None


def steady(values, hold: int = HOLD_SAMPLES):
    """Only believe a value once it has repeated `hold` times in a row.

    Returns a list the same length, carrying the last CONFIRMED value at every
    position. This is what stops one frame of a graphic over the scoreboard from
    being read as a basket and then as a correction.
    """
    out, confirmed, run, candidate = [], None, 0, None
    for value in values:
        if value is None:
            out.append(confirmed)
            continue
        if value == candidate:
            run += 1
        else:
            candidate, run = value, 1
        if run >= hold:
            confirmed = candidate
        out.append(confirmed)
    return out


def monotonic(values, max_jump: int = 3):
    """A score never falls and never rises by more than one possession.

    Both are properties of basketball rather than thresholds, which is why
    neither is tuned. A fall is a misread; a jump of nine is a misread of the
    tens digit.
    """
    out, best = [], None
    for value in values:
        if value is None:
            out.append(best)
            continue
        if best is None:
            best = value
        elif value < best or value - best > max_jump:
            pass                      # keep the running maximum
        else:
            best = value
        out.append(best)
    return out


def pick_scores(candidates, clock_roi, readings_by_index, frames, templates):
    """The two score regions, chosen by how well they behave like scores.

    A region scores well when its readings are legible often, never fall, and
    rise a plausible number of times. Text that happens to read as digits fails
    the second test immediately.
    """
    scored = []
    for roi in candidates:
        if tuple(roi) == tuple(clock_roi):
            continue
        top, bottom, left, right = roi
        raw = [digits_of(frame[top:bottom, left:right], templates)
               for frame in frames]
        legible = sum(1 for v in raw if v is not None)
        if legible < len(frames) * 0.3:
            continue
        held = steady(raw)
        seen = [v for v in held if v is not None]
        if len(seen) < 10:
            continue
        falls = sum(1 for a, b in zip(seen, seen[1:]) if b < a)
        rises = sum(1 for a, b in zip(seen, seen[1:]) if b > a)
        if not rises:
            continue
        scored.append(({"roi": list(roi), "legible": legible / len(frames),
                        "falls": falls, "rises": rises,
                        "final": seen[-1]},
                       (falls / max(rises, 1), -legible)))
    scored.sort(key=lambda e: e[1])
    # The candidate generator emits many overlapping variants of one region, so
    # the top two by score are usually the SAME score twice. Take the best, then
    # the best that does not overlap it.
    chosen = []
    for row, _ in scored:
        top, bottom, left, right = row["roi"]
        if any(overlaps(row["roi"], other["roi"]) for other in chosen):
            continue
        chosen.append(row)
    return chosen


def overlaps(a, b, threshold: float = 0.2) -> bool:
    """Do two (top, bottom, left, right) regions share meaningful area?"""
    top = max(a[0], b[0])
    bottom = min(a[1], b[1])
    left = max(a[2], b[2])
    right = min(a[3], b[3])
    if bottom <= top or right <= left:
        return False
    inter = (bottom - top) * (right - left)
    area_a = (a[1] - a[0]) * (a[3] - a[2])
    area_b = (b[1] - b[0]) * (b[3] - b[2])
    return inter / min(area_a, area_b) > threshold


def find_shot_clock(candidates, clock_roi, frames, templates):
    """The ticking region that RESETS, which the game clock never does."""
    best, best_resets = None, 0
    for roi in candidates:
        if tuple(roi) == tuple(clock_roi):
            continue
        top, bottom, left, right = roi
        raw = [digits_of(frame[top:bottom, left:right], templates)
               for frame in frames]
        seen = [v for v in raw if v is not None]
        if len(seen) < len(frames) * 0.2:
            continue
        resets = sum(1 for a, b in zip(seen, seen[1:]) if b - a >= RESET_RISE)
        near_24 = sum(1 for v in seen if v == SHOT_CLOCK_RESET)
        if resets > best_resets and near_24:
            best, best_resets = (list(roi), raw), resets
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--clock", required=True,
                        help="read_game_clock.py output; supplies the roi the "
                             "templates are learned from, the timebase, and the "
                             "clock stream that is carried through")
    parser.add_argument("--learn-start", type=float, default=None,
                        help="seconds into the video to learn digits from; "
                             "defaults to the first minute the clock is running")
    parser.add_argument("--learn-seconds", type=float, default=90.0)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--step", type=float, default=STEP_S)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import numpy as np

    from courtvision.autoscoreboard import (bootstrap_templates,
                                            candidate_rois, locate_scores)

    clock_blob = json.load(open(args.clock))
    clock_roi = tuple(clock_blob["roi"])
    readings = clock_blob["readings"]
    if not readings:
        print("FAIL - the clock was never read; nothing can be anchored to it")
        return 1
    by_time = {float(r["t"]): r for r in readings}
    start = args.start if args.start is not None else min(by_time)
    end = args.end if args.end is not None else max(by_time)

    # ---- learn the digits, from the CLOCK, because it is the only region
    # whose value is known a priori --------------------------------------------
    learn_start = args.learn_start
    if learn_start is None:
        running = [r for r in readings if r.get("seconds") is not None]
        learn_start = float(running[len(running) // 4]["t"]) if running else start
    learn = list(stream(args.video, learn_start, args.learn_seconds, 1.0))
    if not learn:
        print(f"FAIL - could not read frames at {learn_start:.0f}s")
        return 1
    top, bottom, left, right = clock_roi
    templates = bootstrap_templates([f for f in learn], clock_roi)
    if len(templates) < 8:
        print(f"FAIL - learned only {len(templates)} digits from the clock at "
              f"{learn_start:.0f}s; try --learn-start elsewhere")
        return 1
    print(f"  learned {len(templates)} digit templates from the clock at "
          f"{learn_start:.0f}s")

    # ---- locate the score regions on the same panel --------------------------
    found = locate_scores(learn, near=clock_roi)
    rois = [tuple(c.roi) if hasattr(c, "roi") else tuple(c) for c in found]
    if not rois:
        rois = [tuple(r) for r in candidate_rois(learn[0].shape[0],
                                                 learn[0].shape[1])]
        print(f"  locate_scores found nothing; falling back to "
              f"{len(rois)} candidate regions")
    ranked = pick_scores(rois, clock_roi, by_time, learn, templates)
    if len(ranked) < 2:
        print(f"FAIL - found {len(ranked)} score-like regions, need two. "
              f"The scoreboard graphic may be laid out differently on this "
              f"broadcast; --learn-start elsewhere is the first thing to try.")
        return 1
    home, away = ranked[0], ranked[1]
    print(f"  home score at {home['roi']} (legible {home['legible']:.0%}, "
          f"{home['rises']} rises)")
    print(f"  away score at {away['roi']} (legible {away['legible']:.0%}, "
          f"{away['rises']} rises)")
    shot = find_shot_clock(rois, clock_roi, learn, templates)
    print(f"  shot clock at {shot[0]}" if shot else
          "  no shot clock found; possession changes will be unavailable")

    # ---- sweep -------------------------------------------------------------
    regions = {"home": tuple(home["roi"]), "away": tuple(away["roi"])}
    if shot:
        regions["shot"] = tuple(shot[0])
    raw = {name: [] for name in regions}
    times = []
    for n, frame in enumerate(stream(args.video, start, end - start, args.step)):
        when = start + n * args.step
        times.append(when)
        for name, (rtop, rbottom, rleft, rright) in regions.items():
            raw[name].append(digits_of(frame[rtop:rbottom, rleft:rright],
                                       templates))
        if n and n % 500 == 0:
            print(f"    {when:.0f}s of {end:.0f}s", flush=True)

    home_values = monotonic(steady(raw["home"]))
    away_values = monotonic(steady(raw["away"]))
    shot_values = steady(raw.get("shot", []), hold=1) if shot else []

    score_rows, shot_rows = [], []
    for i, when in enumerate(times):
        reading = by_time.get(round(when, 1)) or by_time.get(float(int(when)))
        if reading is None or reading.get("elapsed") is None:
            continue
        score_rows.append({"video_s": when, "elapsed": float(reading["elapsed"]),
                           "home": home_values[i], "away": away_values[i]})
        if shot_values:
            shot_rows.append({"video_s": when,
                              "elapsed": float(reading["elapsed"]),
                              "shot_clock": shot_values[i]})

    clock_rows = [{"video_s": float(r["t"]), "elapsed": r.get("elapsed"),
                   "period": r.get("period"), "seconds": r.get("seconds")}
                  for r in readings]
    out = Path(args.out or f"outputs/scoreboard/{Path(args.video).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "clock_source": args.clock,
               "regions": {k: list(v) for k, v in regions.items()},
               "step": args.step, "learn_start": learn_start,
               "clock": clock_rows, "score": score_rows,
               "shot_clock": shot_rows}, open(out, "w"))

    legible_home = sum(1 for r in score_rows if r["home"] is not None)
    final = (score_rows[-1]["home"], score_rows[-1]["away"]) if score_rows else (None, None)
    changes = sum(1 for a, b in zip(score_rows, score_rows[1:])
                  if a["home"] != b["home"] or a["away"] != b["away"])
    print(f"\n  {len(score_rows)} score readings over {end - start:.0f}s; "
          f"legible on {legible_home / max(len(score_rows), 1):.0%}")
    print(f"  {changes} score changes; final {final[0]}-{final[1]}")
    print(f"  {len(shot_rows)} shot-clock readings")
    print(f"  -> {out}")
    print("\n  Check the final score against the box score before trusting any "
          "event\n  derived from this. It is one number and it validates the "
          "whole sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
