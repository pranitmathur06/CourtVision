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
  dropped, and a run that does not START near the top of the clock is a replay
  too -- a quarter begins at 12:00, a replay of its last minute begins at 1:00
  -- so it is merged into the period it interrupts (NEW_PERIOD_SHARE). Rules keyed to exact values all failed on this broadcast: any
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
#: A reading may miss the running clock by this much and still continue it.
CONTINUE_TOL_S = 3.0
#: An upward jump this large ends a run of readings.
RESET_JUMP_S = 30.0
#: A run this short is a replay or a misread, not a period.
MIN_RUN = 20
#: A new period's first reading is near the top of the clock; a replay's is not.
NEW_PERIOD_SHARE = 0.8


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
    if len(digits) == 2:                        # "72" can only be 7.2
        # Under ten seconds the clock shows one digit and a tenth, which is two
        # glyphs and nothing else. There is no MM:SS reading of two digits, so
        # this offers exactly one candidate, in [0.0, 9.9], and `resolve` still
        # makes it prove itself against the next frame before believing it.
        out.append(int(digits[0]) + int(digits[1]) / 10.0)
    return out


def _continues(previous, previous_t, value, t):
    """Could `value` at time `t` be the same clock last seen as `previous`?

    A game clock either holds (a stoppage) or falls at real time, so a reading
    may sit anywhere between the previous one and the previous one minus the
    video time since, give or take CONTINUE_TOL_S for a dropped frame.
    """
    elapsed = max(0.0, t - previous_t)
    return previous - elapsed - CONTINUE_TOL_S <= value <= previous + CONTINUE_TOL_S


def resolve(candidates):
    """[(t, [possible seconds])] -> [(t, seconds)], settling ties by continuity.

    Ties go to the reading that continues the running clock, and a reading
    that continues nothing has to be confirmed by the next one before it is
    believed. Without that confirmation a single misread captures the clock:
    on Finals Game 1 one bad frame read 6:07 as 367 s, and every later "8:06 /
    80.6" pair then had to fall below it, so the rest of the quarter -- 738 s
    of it -- resolved to tenths and four official attempts landed in the wrong
    place. A misread survives one frame; a real jump (a new period) is still
    there on the next.
    """
    rows = [(t, options) for t, options in candidates if options]
    out, running, running_t = [], None, None
    for i, (t, options) in enumerate(rows):
        if running is None:
            seconds = max(options)
        else:
            fits = [v for v in options if _continues(running, running_t, v, t)]
            if fits:
                seconds = max(fits)
            else:
                # A SUB-TEN-SECOND READING MAY NEVER START A RUN. Reading the
                # last ten seconds of a period means a two-glyph text is now a
                # valid time, and a graphic that briefly leaves two glyphs
                # showing is a plausible source of one. A single stray is
                # already dropped by the confirmation below, but two
                # consecutive strays that happen to agree would confirm each
                # other -- and because `assign_periods` treats the jump back up
                # to the real clock as a new run starting near the top of its
                # period, that would turn one quarter into two. A genuine
                # countdown always arrives by continuing from a reading above
                # ten, so this costs nothing real.
                if all(v < 10.0 for v in options):
                    continue
                following = rows[i + 1] if i + 1 < len(rows) else None
                confirmed = [
                    v for v in options
                    if following
                    and any(_continues(v, t, w, following[0]) for w in following[1])
                ]
                if not confirmed:
                    continue                    # unconfirmed misread: drop it
                seconds = max(confirmed)
        out.append((t, seconds))
        running, running_t = seconds, t
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
    if len(digits) == 2:                        # "7:2" was "7.2"
        return int(digits[0]) + int(digits[1]) / 10.0
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
    out, period, top = [], 0, None
    for run in kept:
        start = run[0][1]
        # A period begins near the top of its own clock -- 12:00 for a quarter,
        # 5:00 for an overtime. Testing only against the previous period's top
        # merged overtime into the fourth quarter, because 5:00 is well under
        # four fifths of 12:00; the ECF game that ends in OT then had a fourth
        # quarter 3,387 seconds of video long and its overtime unplaceable.
        fresh = top is None or start >= NEW_PERIOD_SHARE * top
        overtime = (period >= 4 and not fresh
                    and start >= NEW_PERIOD_SHARE * OT_S
                    and start <= OT_S + RESET_JUMP_S)
        if fresh or overtime:
            period += 1
            top = start
        out += [(t, period, seconds) for t, seconds in run]
    return out, dropped


def elapsed(period: int, seconds: float) -> float:
    if period <= 4:
        return (period - 1) * PERIOD_S + (PERIOD_S - seconds)
    return 4 * PERIOD_S + (period - 5) * OT_S + (OT_S - seconds)


def stream(video, start, seconds, step, crop=None):
    """Frames every `step` seconds from `start`, decoded in one pass.

    ffmpeg reads the file once and drops the frames in between, which is what
    makes this minutes rather than hours; `crop` keeps only the clock, so most
    of the pixels are never copied out at all.
    """
    import subprocess

    import numpy as np

    filters = [f"fps=1/{step}"]
    if crop:
        top, bottom, left, right = crop
        filters.append(f"crop={right - left}:{bottom - top}:{left}:{top}")
    width, height = ((right - left, bottom - top) if crop else (None, None))
    command = ["ffmpeg", "-nostdin", "-loglevel", "error",
               "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(video),
               "-vf", ",".join(filters), "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-"]
    if not crop:
        # The learning pass needs whole frames, so ask ffmpeg for the size.
        import json as _json
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", str(video)],
            capture_output=True, text=True)
        stream_info = _json.loads(probe.stdout)["streams"][0]
        width, height = int(stream_info["width"]), int(stream_info["height"])

    size = width * height * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, bufsize=size * 4)
    try:
        while True:
            buffer = process.stdout.read(size)
            if not buffer or len(buffer) < size:
                return
            yield np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
    finally:
        process.stdout.close()
        process.terminate()
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--from-raw", default=None,
                        help="re-resolve the candidates saved by an earlier run, no video pass")
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--learn-start", type=float, default=900.0)
    parser.add_argument("--learn-seconds", type=int, default=120)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.from_raw:
        saved = json.load(open(args.from_raw))
        texts = [(r["t"], r["s"]) for r in saved.get("text", [])]
        # Re-parse from the TEXT when it is there, so a parser change can be
        # measured without touching the video; fall back to the saved options
        # for files written before the text was kept.
        if texts:
            candidates = [(t, readings_from(s)) for t, s in texts]
            candidates = [(t, o) for t, o in candidates if o]
        else:
            candidates = [(r["t"], r["options"]) for r in saved["raw"]]
        sampled = saved.get("sampled", len(candidates))
        location_roi, step = saved.get("roi"), saved.get("step", args.step)
        return _write(args, candidates, sampled, location_roi, step, texts)

    import cv2

    from courtvision.autoscoreboard import bootstrap_templates, locate_clock
    from courtvision.scoreboard import read_clock

    capture = cv2.VideoCapture(args.video)
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    location, templates = None, {}
    for start in (args.learn_start, args.learn_start + 600,
                  args.learn_start + 1800, 2 * args.learn_start + 1800):
        frames = list(stream(args.video, start, args.learn_seconds, 1.0))
        if not frames:
            continue
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
    print(f"clock at rows {top}-{bottom}, cols {left}-{right}; "
          f"ticks {location.ticks}/{location.samples}; digits {sorted(templates)}",
          flush=True)

    # Only the clock is read, so only the clock is decoded and moved: the crop
    # happens inside ffmpeg. Seeking to each second separately made this a
    # five-hour pass on a two-hour broadcast, because every seek decodes from
    # the keyframe before it; one sequential pass over the same video is
    # minutes. Same frames, same readings.
    # THE TEXT IS SAVED ALONGSIDE THE PARSED OPTIONS. It was not, and that made
    # every change to the parser cost a full sequential decode of the video --
    # forty minutes on a 1080p60 broadcast -- to find out whether it helped. A
    # frame whose text parsed to nothing was not even recorded, so `--from-raw`
    # could never see the readings a better parser would recover. With the text
    # on disk a parser change is re-resolved for free, and the frames the old
    # parser threw away are still there to be re-read.
    candidates, sampled, texts = [], 0, []
    for i, crop in enumerate(stream(args.video, 0.0, duration, args.step,
                                    crop=(top, bottom, left, right))):
        sampled += 1
        text, _ = read_clock(crop, templates, allow_tenths=True)
        if text:
            texts.append((i * args.step, text))
        options = readings_from(text)
        if options:
            candidates.append((i * args.step, options))
    return _write(args, candidates, sampled, list(location.roi), args.step, texts)


def _write(args, candidates, sampled, roi, step, texts=()):
    """Resolve the candidates, split them into periods, save and report."""
    raw = resolve(candidates)
    rows, misreads = assign_periods(raw)
    out = Path(args.out or f"outputs/clock/{Path(args.video).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "roi": roi, "step": step,
               "sampled": sampled,
               "text": [{"t": t, "s": text} for t, text in texts],
               "raw": [{"t": t, "options": options} for t, options in candidates],
               "readings": [{"t": t, "period": p, "seconds": s, "elapsed": elapsed(p, s)}
                            for t, p, s in rows]}, open(out, "w"), indent=0)
    periods = sorted({p for _, p, _ in rows})
    print(f"{sampled} frames sampled; {len(raw)} read; "
          f"{misreads} dropped in short runs ({misreads / max(len(raw), 1):.1%}); periods seen {periods}")
    for p in periods:
        run = [(t, s) for t, q, s in rows if q == p]
        print(f"  period {p}: {len(run):4d} readings, clock {max(s for _, s in run):.0f} -> "
              f"{min(s for _, s in run):.1f} s, video {run[0][0]:.0f}..{run[-1][0]:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
