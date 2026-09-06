"""How many screens a game contains, against how many it should.

The only check on play detection that needs no labels, and the only one that
has held up. A real NBA game holds roughly 60-80 ball screens and 80-100
off-ball screens; a detector reporting 600 is not measuring screens whatever
its precision looks like on a hand-labelled sample.

Run over several games, because one game's spread is wide and the point is the
median. Coordinates come from tracking data, so this measures the play
DEFINITIONS and says nothing about the vision stack -- see
docs/continuous-game-accuracy.md for why the two cannot yet be joined.
"""

from __future__ import annotations

import argparse
import collections
import glob
import statistics
import sys

sys.path.insert(0, "src")

from courtvision.plays import detect_off_ball_screens, detect_screens
from courtvision.tracking_data import load_game

SAMPLE_HZ = 10.0
# What a real game contains, from published play-type counts.
REAL_ON_BALL = (60, 80)
REAL_OFF_BALL = (80, 100)
ON_BALL_NAMES = {"ball_screen", "pick_and_roll", "pick_and_pop",
                 "dribble_handoff"}


def continuous_stretches(game, gap_s: float = 2.0, floor: int = 40):
    """Split frames wherever the clock jumps; SportVU events are windows."""
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


def screens_in(game) -> collections.Counter:
    found: collections.Counter = collections.Counter()
    for chunk in continuous_stretches(game):
        positions, handlers, times, offense = [], [], [], []
        for frame in chunk:
            spot = {t.track_id: ((t.box.x1 + t.box.x2) / 2, t.box.y2)
                    for t in frame.tracks if t.label != "ball"}
            positions.append(spot)
            held = frame.handler()
            handlers.append(held.track_id if held else None)
            times.append(frame.time_s)
            side = game.teams.get(held.track_id) if held else None
            offense.append({t for t in spot if game.teams.get(t) == side}
                           if side else set())
        for play in (detect_screens(positions, handlers, times, offense)
                     + detect_off_ball_screens(positions, handlers, times,
                                               offense)):
            found[play.name] += 1
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", nargs="*", default=None,
                        help="tracking JSON files (default: data/tracking/*)")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    paths = args.games or sorted(glob.glob("data/tracking/*.json"))
    paths = paths[:args.limit]
    if not paths:
        print("no tracking games found; see scripts/fetch_tracking_game.py")
        return 1

    on_ball, off_ball, totals = [], [], collections.Counter()
    print(f"  {'game':<14}{'on-ball':>9}{'off-ball':>10}{'total':>8}")
    for path in paths:
        game = load_game(path, target_hz=SAMPLE_HZ)
        found = screens_in(game)
        totals.update(found)
        on = sum(n for name, n in found.items() if name in ON_BALL_NAMES)
        off = sum(found.values()) - on
        on_ball.append(on); off_ball.append(off)
        print(f"  {game.game_id:<14}{on:>9}{off:>10}{on + off:>8}", flush=True)

    print(f"\n  {'':<14}{'median':>9}{'real':>12}")
    for name, seen, real in (("on-ball", on_ball, REAL_ON_BALL),
                             ("off-ball", off_ball, REAL_OFF_BALL)):
        middle = statistics.median(seen)
        verdict = "ok" if real[0] <= middle <= real[1] else "OUT OF RANGE"
        print(f"  {name:<14}{middle:>9.0f}{f'{real[0]}-{real[1]}':>12}   {verdict}")

    print("\n  by type, across all games:")
    everything = sum(totals.values())
    for name, count in totals.most_common():
        print(f"    {name:<22}{count:>6}{count / everything:>7.0%}")
    print("\n  This counts plays; it does not check that any individual one is"
          "\n  right. Per-play accuracy needs labels -- see"
          "\n  docs/continuous-game-accuracy.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
