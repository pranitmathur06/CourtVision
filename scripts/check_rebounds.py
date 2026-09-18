"""Who got the rebound, scored against the feed.

The same shape as the assist check and for the same reason: the play-by-play
credits a rebounder by name on every rebound, tracking data carries the same
game clock, and the detector never sees the credited name. About 97 rebounds a
game makes it the largest per-event label set available.

A rebound is logged when the ball is SECURED, so the question is who is holding
it at that moment. The confidence gate is the same idea as the assist reader's:
a player who "held" the ball for a tenth of a second did not secure a rebound,
that is the tracker resolving a scramble badly, and declining beats guessing.

THIS DOES NOT READ A BROADCAST. `load_game` below opens SportVU tracking: every
player located to the inch at 25 Hz, the ball's height known, identities stable
for the whole game. The number this prints is the event LOGIC's accuracy on
perfect inputs, and it is not evidence that the same question can be answered
from pixels. `scripts/eval_play_events.py` asks that question and gets an
answer at or below the majority class on all four broadcasts. Quote the two
together or neither.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import sys

sys.path.insert(0, "src")

from courtvision.tracking_data import load_game

SAMPLE_HZ = 10.0
FULL_RATE_HZ = 25.0
# The feed logs a rebound AFTER the player has secured it: measured on one
# game, the credited player first has the ball a second or more BEFORE the
# logged moment far more often than after. So the window straddles it.
BEFORE_S = 1.5
SETTLE_S = 2.0
CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")
MIN_HOLD_S = 0.4
_min_hold = MIN_HOLD_S


def rebounds_of(game_id: str):
    """[(period, seconds left, credited player id)]."""
    from nba_api.stats.endpoints import playbyplayv3

    actions = playbyplayv3.PlayByPlayV3(game_id=game_id,
                                        timeout=60).get_dict()["game"]["actions"]
    out = []
    for action in actions:
        if (action.get("actionType") or "") != "Rebound":
            continue
        matched = CLOCK.fullmatch((action.get("clock") or "").strip())
        person = action.get("personId")
        if not matched or action.get("period") is None or not person:
            continue
        # Team rebounds carry a team id rather than a player and cannot be
        # attributed to anybody, so they are not scored.
        if person > 1_000_000:
            continue
        out.append((int(action["period"]),
                    int(matched.group(1)) * 60 + float(matched.group(2)),
                    person))
    return out


# Above this the ball is off the rim or in flight, not in anybody's hands.
OFF_THE_RIM_Z_FT = 9.0


def secured_by(game, index: int, ball_z):
    """Who FIRST comes down with the ball after it leaves the rim.

    Not the longest holder: after a defensive rebound the ball is outletted at
    once to a guard who then dribbles for several seconds, so the longest hold
    in the window is the guard rather than the rebounder. And not simply the
    first holder from the logged moment either, because the feed logs a rebound
    after it is secured. The ball's height gives the true start: the miss comes
    off the rim, and whoever first holds it afterwards took the rebound.
    """
    first = max(0, index - int(BEFORE_S * SAMPLE_HZ))
    last = min(len(game.frames) - 1, index + int(SETTLE_S * SAMPLE_HZ))
    start = first
    for step in range(min(last, len(ball_z) - 1), first - 1, -1):
        height = ball_z[step]
        if not math.isnan(height) and height >= OFF_THE_RIM_Z_FT:
            start = step
            break
    step = start
    while step <= last:
        held = game.frames[step].handler()
        if held is None:
            step += 1
            continue
        candidate = held.track_id
        end = step
        while end + 1 <= last:
            nxt = game.frames[end + 1].handler()
            if nxt is None or nxt.track_id != candidate:
                break
            end += 1
        if (end - step + 1) / SAMPLE_HZ >= _min_hold:
            return candidate
        step = end + 1
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", nargs="*")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--min-hold", type=float, default=MIN_HOLD_S)
    parser.add_argument("--hz", type=float, default=SAMPLE_HZ,
                        help="sampling rate; a scramble may resolve faster "
                             "than the 10 Hz the rest of the pipeline uses")
    args = parser.parse_args()

    globals()["_min_hold"] = args.min_hold
    globals()["SAMPLE_HZ"] = args.hz
    paths = (args.games or sorted(glob.glob("data/tracking/*.json")))
    paths = paths[args.skip:args.skip + args.limit]

    total = resolved = right = 0
    for path in paths:
        game = load_game(path, target_hz=SAMPLE_HZ)
        moment = {}
        for i, (period, clock) in enumerate(zip(game.periods, game.game_clocks)):
            moment.setdefault((period, round(clock)), i)
        try:
            events = rebounds_of(game.game_id)
        except Exception as error:                       # noqa: BLE001
            print(f"  {game.game_id}: play-by-play unavailable ({error})")
            continue
        total += len(events)
        found = hit = 0
        for period, clock, credited in events:
            index = None
            for offset in (0, -1, 1):
                index = moment.get((period, round(clock) + offset))
                if index is not None:
                    break
            if index is None:
                continue
            who = secured_by(game, index, game.ball_z)
            if who is None:
                continue
            found += 1; hit += who == credited
        resolved += found; right += hit
        print(f"  {game.game_id}: {found}/{len(events)} resolved, {hit} correct",
              flush=True)

    print(f"\n  {resolved} of {total} rebounds resolved "
          f"({resolved / max(total, 1):.0%} coverage) at min-hold {_min_hold}s")
    if resolved:
        rate = right / resolved
        se = math.sqrt(rate * (1 - rate) / resolved)
        print(f"  rebounder matches the credited player: {right}/{resolved} = "
              f"{rate:.0%}   (95% {max(0, rate - 1.96 * se):.0%} to "
              f"{min(1, rate + 1.96 * se):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
