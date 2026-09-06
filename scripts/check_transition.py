"""Transition, scored against the shot clock rather than against anybody's eye.

Every other play type in this project ran aground on the same thing: no feed
says whether a possession contained a screen, so the labels had to come from a
person, and a person looking at dots is not a reliable oracle.

Transition is different. The shot clock states exactly how long a possession
has been running -- it resets to 24 and counts down -- so "the offense attacked
early" is a fact the feed already holds. The detector never sees the clock; it
only watches the ball cover ground. That makes this a real labelled test of an
offensive play type, with thousands of examples and no hand labels anywhere.

Possessions in the ambiguous middle are excluded rather than forced to a side.
A possession that arrives with 15 seconds left is neither a break nor a walk-up,
and scoring it either way measures the threshold, not the detector.
"""

from __future__ import annotations

import argparse
import glob
import math
import sys

sys.path.insert(0, "src")

from courtvision.court import BASKET
from courtvision.plays import detect_transition
from courtvision.tracking_data import load_game

SAMPLE_HZ = 10.0
# Shot clock when the possession ENDS -- which is when the shot went up, or
# the ball was lost. Reading it on arrival in the frontcourt instead labelled
# two thirds of all possessions "early", because crossing half court takes a
# few seconds in any offense; transition is about finishing early, not
# arriving early. Above the first is a break, below the second a half-court
# set, and between them nobody agrees, so those are not scored.
EARLY_S = 17.0
LATE_S = 13.0
MIN_POSSESSION_FRAMES = 20
# A team must hold the ball for this long before it counts as a change of
# possession. Without it every deflection, every pass the tracker briefly
# assigns to the wrong man, and every rebound in traffic splits a possession in
# two -- which produced 390 possessions a game against a real 200, and an
# even split of early and late that basketball does not have.
MIN_CONTROL_S = 1.5


def stretches(game, gap_s: float = 2.0, floor: int = 40):
    out, current = [], []
    for frame in game.frames:
        if current and abs(frame.time_s - current[-1].time_s) > gap_s:
            if len(current) > floor:
                out.append(current)
            current = []
        current.append(frame)
    if len(current) > floor:
        out.append(current)
    return out


def possessions(chunk, teams, clocks):
    """Split a stretch where the controlling TEAM changes and keeps the ball.

    Splitting on the first frame the other team touches it turns every
    deflection and every contested rebound into a possession, and a shot clock
    read against those fragments means nothing.
    """
    runs, current, side = [], [], None
    pending, since = None, None
    for frame in chunk:
        held = frame.handler()
        now = teams.get(held.track_id) if held else None
        current.append(frame)
        if now is None:
            continue
        if side is None:
            side = now
            continue
        if now == side:
            pending, since = None, None
            continue
        if pending != now:
            pending, since = now, frame.time_s
            continue
        if frame.time_s - since < MIN_CONTROL_S:
            continue
        # The other team has held it long enough; the possession ended when
        # they first got it, not now.
        cut = len(current) - int(MIN_CONTROL_S * SAMPLE_HZ) - 1
        if cut >= MIN_POSSESSION_FRAMES:
            runs.append((side, current[:cut]))
            current = current[cut:]
        side, pending, since = now, None, None
    if side is not None and len(current) >= MIN_POSSESSION_FRAMES:
        runs.append((side, current))
    return runs


def truth_of(frames, teams, side, clocks):
    """'early', 'late' or None, from the shot clock when the possession ends.

    A possession ends with a shot or a loss of the ball, so the clock at its
    last frame says how long the offense took. That is what transition means
    and it is a fact the feed already holds -- the detector never sees it.
    """
    # The LOWEST reading, not the last one. A made basket resets the clock to
    # 24 the instant it drops, so the final frames of a possession often read
    # like a fresh one; the minimum is how far this possession actually ran
    # the clock down, and it survives the reset.
    readings = [clocks[f.index] for f in frames
                if clocks.get(f.index) is not None
                and not math.isnan(clocks[f.index])]
    if not readings:
        return None
    clock = min(readings)
    if clock >= EARLY_S:
        return "early"
    if clock <= LATE_S:
        return "late"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", nargs="*")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    paths = (args.games or sorted(glob.glob("data/tracking/*.json")))[:args.limit]
    if not paths:
        print("no tracking games found")
        return 1

    tp = fp = fn = tn = 0
    scored = skipped = 0
    for path in paths:
        game = load_game(path, target_hz=SAMPLE_HZ)
        clocks = {f.index: c for f, c in zip(game.frames, game.shot_clocks)}
        for chunk in stretches(game):
            for side, frames in possessions(chunk, game.teams, clocks):
                truth = truth_of(frames, game.teams, side, clocks)
                if truth is None:
                    skipped += 1
                    continue
                positions, handlers, times = [], [], []
                for frame in frames:
                    positions.append({t.track_id: ((t.box.x1 + t.box.x2) / 2,
                                                   t.box.y2)
                                      for t in frame.tracks if t.label != "ball"})
                    held = frame.handler()
                    handlers.append(held.track_id if held else None)
                    times.append(frame.time_s)
                called = bool(detect_transition(positions, handlers, times))
                scored += 1
                if truth == "early":
                    tp += called; fn += not called
                else:
                    fp += called; tn += not called
        print(f"    {path.split('/')[-1]}: {scored} scored, {skipped} ambiguous",
              flush=True)

    print(f"\n  {scored} possessions with an unambiguous shot clock "
          f"({skipped} in the middle, not scored)")
    print(f"    early and called transition      {tp}")
    print(f"    late  and called transition      {fp}")
    print(f"    early and not called             {fn}")
    print(f"    late  and not called             {tn}")
    total = tp + fp + fn + tn
    if not total:
        return 1
    print(f"\n    accuracy   {(tp + tn) / total:.0%}")
    if tp + fp:
        print(f"    precision  {tp / (tp + fp):.0%} ({tp}/{tp + fp})")
    if tp + fn:
        print(f"    recall     {tp / (tp + fn):.0%} ({tp}/{tp + fn})")
    base = max(tp + fn, tn + fp) / total
    print(f"    always guessing the commoner answer would give {base:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
