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

TWO THINGS ABOUT THE PANEL THAT COST A WHOLE RUN. On this broadcast the band
reads `IND 23 | OKC 31 | 2ND 10:39 | 18`, and:

  THE FAR TEAM'S SCORE IS 324 px FROM THE CLOCK. `locate_scores` defaults to
  searching `within_px=260` of the clock, so the near team's score is found and
  the far team's never is. The first full-game run selected the shot clock and a
  clock fragment instead and reported a final score of 4-17 against a true
  110-111.

  THE SCORE DIGITS ARE 1.6x THE CLOCK'S. That turns out not to matter -- the
  matcher resizes each glyph to the template before comparing -- but the two
  scores sit on DIFFERENT backgrounds (one gold, one white) and only
  `normalise_polarity` makes that survivable.

REGIONS ARE CHOSEN ON THE WHOLE GAME, not on the learning window. Ninety seconds
of one quarter cannot tell a score from any other number on the panel: a score
rises once or twice in that time and so does the shot clock's tens digit. Across
a whole game a score is the only region that is NON-DECREASING throughout and
ends somewhere a basketball score can end.

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
#: How far from the clock to look for the scores. `None` means "everything to
#: the left of the clock, to the frame edge".
#:
#: IT USED TO BE A NUMBER AND THE NUMBER WAS WRONG TWICE. `locate_scores` ships
#: 260 and misses the far team by 64 px on a 720p Finals broadcast; 460 was
#: measured to fix that, and then misses the far team by 67 px on a 1080p
#: regular-season broadcast whose scorebug is laid out differently. A constant
#: fitted to the layouts already in the repository is not a constant, it is a
#: memory of them -- and the whole claim being tested is that a broadcast
#: nobody has seen fits right in.
#:
#: The reach was never doing the work anyway. What actually picks the score
#: regions is behaviour over a whole game -- non-decreasing, a rank correlation
#: with time of at least 0.85, and a final value inside FINAL_RANGE -- and the
#: reach is only a compute budget on how many boxes get that test. Spending
#: more of it is cheap and is the only setting that is not a guess about a
#: layout.
SEARCH_PX = None
#: A basketball game ends somewhere in here. Used only to reject a region that
#: is plainly not a score, never to correct one.
FINAL_RANGE = (55, 190)
#: Frames spread across the game to choose regions on.
PROBE_FRAMES = 24
#: Confidence floor for one frame's read. Low on purpose -- see `digits_of`.
READ_FLOOR = 0.15
#: How strongly a region's readings must rise with time to be a score.
MIN_RANK_CORRELATION = 0.85
#: Reported beside `rho`, never used as a gate. See `monotone_share`: it was
#: briefly a second gate at 0.70, and measured on Finals G1 that admitted 79
#: regions where rank correlation admits 8, because a region reading a CONSTANT
#: is trivially non-decreasing and scores 1.000.
REPORT_MONOTONE_SHARE = True


def digits_of(image, reader, min_score: float = READ_FLOOR):
    """An integer from a region of digits, or None when nothing segments.

    THE PER-FRAME READ IS DELIBERATELY WEAK AND THE SEQUENCE IS STRONG. The
    clock's own templates cannot read this font -- the score digits are 26x21
    against the clock's 19x17 with a heavier stroke, and "23" comes back as
    "11". The SVHN reader does read it, but its confidence is calibrated on
    jersey crops and does not transfer: correct reads of 59, 79 and 92 come back
    at 0.27 to 0.33, well under the 0.5 floor that makes sense for a jersey.

    So the floor here is low on purpose. What makes the result trustworthy is
    not any single frame but `steady` and `monotonic` over a whole game at one
    reading a second: a value has to repeat before it is believed, it can never
    fall, and it can never rise by more than one possession. A reader that is
    right most of the time and wrong differently each time is recovered
    completely by those three constraints; a confident reader that is wrong the
    same way twice would not be.
    """
    got = reader.read_scoreboard(image)
    if got is None:
        return None
    value, confidence = got
    return value if confidence >= min_score else None


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


def monotonic(values, times=None, max_points_per_minute: float = 12.0):
    """A score never falls, and cannot outrun the game's own scoring rate.

    THE JUMP BOUND IS PER UNIT OF TIME, NOT PER READING, and getting that wrong
    cost a whole run. "It cannot rise by more than three in one possession" is
    true of consecutive POSSESSIONS and false of consecutive READINGS: a replay,
    a graphic or a camera behind the basket hides the panel for half a minute,
    and the next legible frame is legitimately eight points later. Rejecting
    that as a misread pinned one team's score at 8 for an entire game while its
    own region had been seen reading 110.

    So a rise is accepted when it is plausible for the time that passed --
    twelve points a minute is roughly four times a real NBA pace, which makes
    this a guard against a misread tens digit rather than a model of scoring.
    Falls are still rejected outright: a score never goes down.
    """
    out, best, best_at = [], None, None
    for index, value in enumerate(values):
        when = times[index] if times is not None else index
        if value is None:
            out.append(best)
            continue
        if best is None:
            best, best_at = value, when
        elif value >= best:
            gap = max(when - best_at, 1e-6) if times is not None else 1.0
            allowed = max(3.0, max_points_per_minute * gap / 60.0)
            if value - best <= allowed:
                best, best_at = value, when
        out.append(best)
    return out


def rank_correlation(xs, ys) -> float:
    """Spearman's rho, written out so this needs no scipy."""
    n = len(xs)
    if n < 3:
        return 0.0

    def ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            share = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = share
            i = j + 1
        return out

    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    top = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    left = sum((a - mx) ** 2 for a in rx) ** 0.5
    right = sum((b - my) ** 2 for b in ry) ** 0.5
    return top / (left * right) if left and right else 0.0


def band_candidates(clock_roi, width, reach_px=SEARCH_PX):
    """Boxes along the clock's own row, to the left of it.

    `locate_scores` is built to find regions that READ AS DIGITS AND CHANGE
    RARELY anywhere on the lower third, and on this layout its candidates did
    not line up with the team scores at all. The scoreboard is one horizontal
    strip: the scores sit on the clock's row, to its left, at roughly the
    clock's height. Sliding boxes along that row and judging them by behaviour
    is both simpler and less broadcast-specific than a general locator.

    EVERY SIZE HERE IS A MULTIPLE OF THE CLOCK'S OWN HEIGHT, which is measured
    on the broadcast being read. The previous version used 70, 90 and 110 pixel
    boxes grown by 0, 6 and 12 -- numbers that describe a 44-pixel-tall clock on
    a 720p encode and describe nothing at all about a 28-pixel-tall one on a
    1080p encode, where the score digits are taller than the clock's and every
    candidate box would have cut them off. Expressed as ratios they reproduce
    the old boxes exactly on the footage they were fitted to, and they follow
    the graphic on footage they were not.
    """
    top, bottom, left, _ = clock_roi
    height = max(bottom - top, 1)
    out = []
    # A SUPERSET OF THE OLD BOXES, AND MUCH LARGER ONES. 0, 0.14, 0.27 and 0.5
    # of the clock's height reproduce the old (0, 6, 12) on a 44-pixel clock and
    # were fitted on two Finals broadcasts where the score and the clock are
    # rendered at similar sizes. On the Houston broadcast they are not: the
    # clock region is 28 px tall and each team's score sits in a coloured panel
    # whose digits need a 60 x 120 window -- larger than anything this could
    # build -- so the search returned ZERO score-like regions on a frame where
    # a hand-cut crop of the same pixels reads 18 and 15 correctly.
    #
    # The old sizes are kept rather than replaced, so the Finals result the
    # reader already produces cannot move; the new ones are added above them.
    for grow in sorted({0, round(height * 0.14), round(height * 0.27),
                        round(height * 0.5), round(height * 0.75),
                        round(height * 1.0)}):
        y0, y1 = max(top - grow, 0), bottom + grow
        # 1.6 to 4.5 clock-heights wide. The first three are the old 70, 90 and
        # 110 on a 44-pixel clock; the last two are what a score panel needs.
        for box_width in sorted({max(round(height * r), 12)
                                 for r in (1.6, 2.0, 2.5, 3.5, 4.5)}):
            x = 0 if reach_px is None else max(left - reach_px, 0)
            step = max(round(height * 0.23), 4)          # the old 10 at h=44
            while x + box_width < left - 10:
                out.append((y0, y1, x, x + box_width))
                x += step
    return out


def digits_never_shrink(values):
    """Readings kept, once a reading with FEWER digits than one already seen is
    discarded. A score goes from one digit to two to three and never back.

    THIS IS WHAT WAS ACTUALLY WRONG. On the Houston broadcast both team panels
    read the correct final scores -- 111 and 91, against an official 111-91 --
    and both were thrown away, at rank correlations of 0.752 and 0.457 against a
    0.85 gate. The sequences rise cleanly apart from a handful of frames that
    segment as a single "1": the score digits fail to separate and some other
    glyph on the panel wins the same-height vote.

    A single digit arriving after a two-digit reading is not a low score, it is
    a failed segmentation, and saying so costs no threshold and no fitting:

        OKC panel   rho 0.752 -> 0.999    22 readings -> 15
        HOU panel   rho 0.457 -> 0.996    22 readings -> 16

    The first attempt at this was a second gate on `monotone_share` below, and
    it was wrong for a reason worth keeping: a region reading a CONSTANT is
    trivially non-decreasing and scores 1.000, so on Finals G1 a 0.70 gate
    admitted 79 regions where rank correlation admits 8.
    """
    out, widest = [], 0
    for value in values:
        if value is None:
            out.append(None)
            continue
        width = len(str(int(value)))
        if width < widest:
            out.append(None)
            continue
        widest = max(widest, width)
        out.append(value)
    return out


def monotone_share(values) -> float:
    """Fraction of readings that lie on one non-decreasing path through time.

    Reported beside the rank correlation because it says something different --
    how much of what was read is consistent with a score, rather than how well
    all of it correlates with time. NOT a gate: a constant region scores 1.000.
    Measured on the Houston panels before the digit filter: 0.882 and 0.682,
    against 0.444 for a shot clock -- which resets to 24 and so cannot hold a
    long non-decreasing run -- and 0.333 for random digits.
    """
    if not values:
        return 0.0
    best = [1] * len(values)
    for i in range(len(values)):
        for j in range(i):
            if values[j] <= values[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
    return max(best) / len(values)


def pick_scores(candidates, clock_roi, probe, reader):
    """The two score regions, chosen on frames spread across the WHOLE game.

    `probe` is [(seconds, frame)] sampled over the broadcast. A score is the
    only region on the panel that is non-decreasing from tip to final buzzer and
    ends where a basketball score ends -- over ninety seconds nothing
    distinguishes it from the shot clock's tens digit, which is how the first
    run of this script chose the shot clock and called the game 4-17.
    """
    frames = [f for _, f in probe]
    times = [t for t, _ in probe]
    scored = []
    for roi in candidates:
        if tuple(roi) == tuple(clock_roi):
            continue
        top, bottom, left, right = roi
        raw = [digits_of(frame[top:bottom, left:right], reader)
               for frame in frames]
        # A score never loses a digit. See `digits_never_shrink`.
        raw = digits_never_shrink(raw)
        legible = sum(1 for v in raw if v is not None)
        if legible < len(frames) * 0.5:
            continue
        seen = [v for v in raw if v is not None]
        if len(seen) < 6:
            continue
        falls = sum(1 for a, b in zip(seen, seen[1:]) if b < a)
        rises = sum(1 for a, b in zip(seen, seen[1:]) if b > a)
        if not rises:
            continue
        # A score RISES WITH TIME. Demanding zero falls looked right and
        # rejected 308 of 324 regions including the real ones: the per-frame
        # read is right about two thirds of the time by design, so the true
        # score region shows falls too. What separates it is that its readings
        # are ordered by time and a misreading region's are not. Rank
        # correlation is the test that survives a noisy reader.
        order = [t for t, v in zip(times, raw) if v is not None]
        rho = rank_correlation(order, seen)
        share = monotone_share(seen)
        if rho < MIN_RANK_CORRELATION:
            continue
        if not FINAL_RANGE[0] <= seen[-1] <= FINAL_RANGE[1]:
            continue
        # A box offset by thirty pixels clips a digit and still rises with
        # time, so rank correlation alone chose two clipped boxes and the game
        # finished 77-8 against a true 111-110. A clipped box UNDER-READS, so
        # the widest span of values is the box that sees the whole number.
        span = seen[-1] - min(seen)
        scored.append(({"roi": list(roi), "legible": legible / len(frames),
                        "falls": falls, "rises": rises, "rho": round(rho, 3),
                        "span": span, "final": seen[-1]},
                       (-span, -rho, -legible)))
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


def find_shot_clock(candidates, clock_roi, frames, reader):
    """The ticking region that RESETS, which the game clock never does."""
    best, best_resets = None, 0
    for roi in candidates:
        if tuple(roi) == tuple(clock_roi):
            continue
        top, bottom, left, right = roi
        raw = [digits_of(frame[top:bottom, left:right], reader)
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
    parser.add_argument("--search-px", type=int, default=SEARCH_PX,
                        help="cap on how far left of the clock to look. The "
                             "default is no cap: two different fitted values "
                             "have each missed the far team on a layout they "
                             "were not fitted to. See SEARCH_PX.")
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
    from courtvision.digit_net import DigitReader
    reader = DigitReader(floor=0.0)
    print(f"  reading digits with the SVHN net; per-frame floor {READ_FLOOR}")

    # ---- probe the whole game, so a score can be told from a shot clock ----
    probe = []
    reach = [float(r["t"]) for r in readings]
    for when in np.linspace(min(reach), max(reach), PROBE_FRAMES):
        got = list(stream(args.video, float(when), 1.0, 1.0))
        if got:
            probe.append((float(when), got[0]))
    print(f"  probing {len(probe)} frames across the game to choose regions")

    # ---- locate the score regions on the same panel --------------------------
    found = band_candidates(clock_roi, learn[0].shape[1], args.search_px)
    rois = [tuple(c) for c in found]
    print(f"  {len(rois)} candidate regions along the clock's row")
    ranked = pick_scores(rois, clock_roi, probe, reader)
    if len(ranked) < 2:
        print(f"FAIL - found {len(ranked)} score-like regions, need two. "
              f"The scoreboard graphic may be laid out differently on this "
              f"broadcast; --learn-start elsewhere is the first thing to try.")
        return 1
    home, away = ranked[0], ranked[1]
    print(f"  home score at {home['roi']} (legible {home['legible']:.0%}, "
          f"rho {home['rho']}, span {home['span']}, final {home['final']})")
    print(f"  away score at {away['roi']} (legible {away['legible']:.0%}, "
          f"rho {away['rho']}, span {away['span']}, final {away['final']})")
    shot = find_shot_clock(rois, clock_roi, learn, reader)
    # the near team is the one whose score sits left of the other
    if home["roi"][2] > away["roi"][2]:
        home, away = away, home
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
                                       reader))
        if n and n % 500 == 0:
            print(f"    {when:.0f}s of {end:.0f}s", flush=True)

    home_values = monotonic(steady(raw["home"]), times)
    away_values = monotonic(steady(raw["away"]), times)
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
