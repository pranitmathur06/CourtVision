"""Who made the pass that created the basket, scored against the feed.

Every other play-side measurement in this project lacked per-event ground
truth: no feed says whether a possession held a screen, and the shot-clock
labels for transition describe a possession rather than an action. Assists are
different. The play-by-play credits an assister by name on every assisted
basket, tracking data carries the same game clock, and the two align exactly.

That makes this a real per-event test of an offensive action: find the pass
before the shot, and see whether it came from the player the league credits.

The detector sees only positions and who holds the ball. It never sees the
credited name.
"""

from __future__ import annotations

import argparse
import collections
import glob
import math
import re
import sys

sys.path.insert(0, "src")

from courtvision.tracking_data import load_game

# Above this the ball is in the air, not in anybody's hands. Walking back from
# a logged basket without stepping over the shot arc lands on the OPPONENT who
# has already collected the ball out of the net -- which is what made the first
# version of this pick a passer from the other team.
IN_FLIGHT_Z_FT = 9.0

SAMPLE_HZ = 10.0
# How far back from the logged shot time to look for the shooter. The feed's
# clock is whole seconds, and the ball is in flight at the moment it stops.
SHOT_SEARCH_S = 3.0
# A pass older than this did not create the basket.
PASS_WINDOW_S = 6.0
ASSIST = re.compile(r"\(([^()]+?)\s+\d+\s+AST\)")
CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")


def assisted_baskets(game_id: str):
    """[(period, seconds left, scorer id, assister family name)]."""
    from nba_api.stats.endpoints import playbyplayv3

    actions = playbyplayv3.PlayByPlayV3(game_id=game_id,
                                        timeout=60).get_dict()["game"]["actions"]
    out = []
    for action in actions:
        if action.get("shotResult") != "Made":
            continue
        named = ASSIST.search(action.get("description") or "")
        matched = CLOCK.fullmatch((action.get("clock") or "").strip())
        if not named or not matched or action.get("period") is None:
            continue
        out.append((int(action["period"]),
                    int(matched.group(1)) * 60 + float(matched.group(2)),
                    action.get("personId"),
                    named.group(1).strip()))
    return out


def family(name: str) -> str:
    """The part of a name the play-by-play uses, lowercased."""
    cleaned = name.replace(" Jr.", "").replace(" Sr.", "").replace(" III", "")
    return cleaned.split()[-1].lower()


def passer_given_scorer(game, index: int, scorer, teams, ball_z):
    """The teammate who last had the ball before the scorer shot it.

    Conditioning on "my shooter guess happened to be right" measures only the
    possessions where the tracking was already unambiguous, which is a
    selection of the easy ones. The feed names the scorer on every basket, so
    the honest version anchors on that name and is scored on every event -- and
    it is also how the product works, the feed supplying the event and vision
    supplying who did what.
    """
    back = int((SHOT_SEARCH_S + PASS_WINDOW_S) * SAMPLE_HZ)
    floor = max(0, index - back)
    # Walk back to the last moment the SCORER held it: that is the release.
    release = None
    for step in range(index, floor, -1):
        held = game.frames[step].handler()
        if held is not None and held.track_id == scorer:
            release = step
            break
    if release is None:
        return None
    for step in range(release, floor, -1):
        held = game.frames[step].handler()
        if held is None or held.track_id == scorer:
            continue
        if teams.get(held.track_id) != teams.get(scorer):
            return None               # the ball came from the other team
        return held.track_id
    return None


def passer_before(game, index: int, teams, ball_z):
    """(passer id, shooter id) from the handler sequence before a shot.

    The shot arc is the landmark. Walking back from the logged basket, the ball
    is first found low in the hands of whoever collected it afterwards; the
    frames where it was above head height are the shot, and the shooter is the
    last player to hold it BEFORE that.
    """
    back = int((SHOT_SEARCH_S + PASS_WINDOW_S) * SAMPLE_HZ)
    floor = max(0, index - back)

    # Step back over the flight of the shot.
    release = None
    airborne = False
    for step in range(index, floor, -1):
        height = ball_z[step] if step < len(ball_z) else float("nan")
        if not math.isnan(height) and height >= IN_FLIGHT_Z_FT:
            airborne = True
        elif airborne:
            release = step
            break
    if release is None:
        release = index

    holders = []
    for step in range(release, floor, -1):
        held = game.frames[step].handler()
        if held is None:
            continue
        if not holders or holders[-1] != held.track_id:
            holders.append(held.track_id)
        if len(holders) >= 2:
            break
    if len(holders) < 2:
        return None, holders[0] if holders else None
    shooter, passer = holders[0], holders[1]
    if teams.get(shooter) != teams.get(passer):
        return None, shooter          # a steal, not a pass
    return passer, shooter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", nargs="*")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    paths = (args.games or sorted(glob.glob("data/tracking/*.json")))[:args.limit]

    total_baskets = 0
    scored = right = missing = 0
    by_scorer = by_scorer_right = 0
    shooter_hits = right_given_shooter = 0
    reasons: collections.Counter = collections.Counter()
    wrong_examples = []
    for path in paths:
        game = load_game(path, target_hz=SAMPLE_HZ)
        by_moment = {}
        for i, (period, clock) in enumerate(zip(game.periods, game.game_clocks)):
            by_moment.setdefault((period, round(clock)), i)
        try:
            baskets = assisted_baskets(game.game_id)
        except Exception as error:                       # noqa: BLE001
            print(f"  {game.game_id}: play-by-play unavailable ({error})")
            continue

        total_baskets += len(baskets)
        found = 0
        shooter_right = 0
        for period, clock, scorer, credited in baskets:
            index = None
            for offset in (0, -1, 1, -2, 2):
                index = by_moment.get((period, round(clock) + offset))
                if index is not None:
                    break
            if index is None:
                missing += 1
                continue
            anchored = passer_given_scorer(game, index, scorer, game.teams,
                                           game.ball_z)
            if anchored is not None:
                by_scorer += 1
                named = game.names.get(anchored, "")
                by_scorer_right += bool(named) and family(named) == family(credited)
            else:
                reasons["the scorer never held the ball in the window"] += 1

            passer, shooter = passer_before(game, index, game.teams,
                                            game.ball_z)
            if passer is None:
                missing += 1
                reasons["no teammate held it before the shot"] += 1
                continue
            found += 1
            scored += 1
            shooter_right += shooter == scorer
            guess = game.names.get(passer, "")
            correct = bool(guess) and family(guess) == family(credited)
            if shooter == scorer:
                shooter_hits += 1
                right_given_shooter += correct
            if correct:
                right += 1
            elif len(wrong_examples) < 6:
                wrong_examples.append(
                    f"credited {credited}, found {guess or passer}")
        print(f"  {game.game_id}: {found}/{len(baskets)} located; the player "
              f"it calls the shooter is the scorer {shooter_right}/{found}",
              flush=True)

    print(f"\n  {scored} assisted baskets with a pass found "
          f"({missing} not locatable in the tracking data)")
    if scored:
        print(f"  passer matches the credited assister: {right}/{scored} = "
              f"{right / scored:.0%}")
    print(f"\n  ANCHORED on the feed's scorer, which is how the product works:")
    print(f"    resolved on {by_scorer} of {total_baskets} assisted baskets "
          f"({by_scorer / max(total_baskets, 1):.0%})")
    if by_scorer:
        rate = by_scorer_right / by_scorer
        se = (rate * (1 - rate) / by_scorer) ** 0.5
        print(f"    passer matches the credited assister: {by_scorer_right}"
              f"/{by_scorer} = {rate:.0%}   "
              f"(95% {max(0, rate - 1.96 * se):.0%} to {min(1, rate + 1.96 * se):.0%})")
    print(f"\n  deriving the shooter too, with no feed help at all:")
    if shooter_hits:
        print(f"    when the shooter is identified correctly ({shooter_hits} of "
              f"{scored}), the passer is right "
              f"{right_given_shooter}/{shooter_hits} = "
              f"{right_given_shooter / shooter_hits:.0%} -- but that is a "
              f"selected subset")
    for reason, count in reasons.most_common():
        print(f"    {reason}: {count}")
    if wrong_examples:
        print("\n  where it disagreed:")
        for line in wrong_examples:
            print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
