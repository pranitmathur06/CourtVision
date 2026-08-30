"""Plays as geometry over time — screens, rolls, pops and hand-offs.

`formation.py` names arrangements at one instant. A play is a sequence, and this
is that layer: it watches court positions across frames and reports the screen
actions whose definition is purely geometric.

Why this needs no labelled data. A pick-and-roll IS a definition, not a
category someone assigned: two offensive players converge, one of them holding
the ball, and afterwards the screener moves toward the basket while the handler
uses the space. Every term there is measurable in court feet. The same is true
of a pick-and-pop (screener retreats beyond the arc instead) and a dribble
hand-off (the ball changes hands at the point of contact).

What this deliberately does NOT do is name plays that are conventions rather
than geometry. "Spain pick-and-roll" is a back-screen on the roller by a third
player, and while the geometry is arguably reachable, the naming is a coaching
convention that varies between teams; "Horns Flare" and its cousins are set
calls, invisible to a camera. Those need labelled play types, which neither
BARD nor SpaceJam carries — BARD's captions are event-level, listing jersey
number, colour and one of nine action types, with no play names anywhere.

All positions are COURT FEET from `courtvision.court`. Pixels would make every
threshold here meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from courtvision.court import BASKET, THREE_POINT_RADIUS, distance_to_basket_ft

SCREEN_CONTACT_FT = 5.0        # how close two players get for a screen to count
SCREEN_SEPARATION_FT = 9.0     # how far apart they must have been before it
ROLL_GAIN_FT = 6.0             # ground the screener makes toward the rim
POP_GAIN_FT = 4.0              # ground the screener gives up, moving out
WINDOW_FRAMES = 12             # how long after contact the outcome is judged


@dataclass(frozen=True)
class Play:
    """A screen action, with the tracks and moment that justify it."""

    name: str
    time_s: float
    screener_id: int
    handler_id: int
    evidence: str

    def __str__(self) -> str:
        return (f"{self.name} at {self.time_s:.1f}s: "
                f"{self.screener_id} screens for {self.handler_id} ({self.evidence})")


def _distance(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def detect_screens(
    positions: list[dict[int, tuple[float, float]]],
    handlers: list[int | None],
    times: list[float],
) -> list[Play]:
    """Find screen actions across a sequence of frames.

    `positions[i]` maps track id to court feet in frame i; `handlers[i]` is the
    track holding the ball, or None. Tracks may appear and vanish — they do,
    constantly — so every lookup tolerates absence.
    """
    if not (len(positions) == len(handlers) == len(times)):
        raise ValueError("positions, handlers and times must be the same length")

    plays: list[Play] = []
    # Keyed on the UNORDERED pair. After a hand-off the ball is with the other
    # player, so the same two tracks in contact would otherwise be reported a
    # second time with the roles swapped. A genuine re-screen by the same pair
    # later in the possession is therefore reported once, which is the right
    # trade against emitting a phantom screen after every hand-off.
    claimed: set[frozenset[int]] = set()

    for index in range(1, len(positions)):
        handler = handlers[index]
        if handler is None or handler not in positions[index]:
            continue
        handler_now = positions[index][handler]

        for other, screener_now in positions[index].items():
            if other == handler or frozenset((handler, other)) in claimed:
                continue
            if _distance(handler_now, screener_now) > SCREEN_CONTACT_FT:
                continue

            # They must have come together: two players standing side by side
            # for a whole possession are not setting a screen.
            # Scan the WHOLE window: they may have closed gradually, so the
            # single most recent frame is not enough to tell whether they came
            # together or were always side by side.
            approached = any(
                handler in positions[index - back] and other in positions[index - back]
                and _distance(positions[index - back][handler],
                              positions[index - back][other]) >= SCREEN_SEPARATION_FT
                for back in range(1, min(WINDOW_FRAMES, index) + 1)
            )
            if not approached:
                continue

            outcome = _classify_outcome(positions, handlers, index, handler, other)
            if outcome is None:
                continue
            name, evidence = outcome
            claimed.add(frozenset((handler, other)))
            plays.append(Play(name, times[index], other, handler, evidence))

    return plays


def _classify_outcome(
    positions: list[dict[int, tuple[float, float]]],
    handlers: list[int | None],
    index: int,
    handler: int,
    screener: int,
) -> tuple[str, str] | None:
    """What the screener did afterwards decides which play this was."""
    start = positions[index][screener]
    start_to_rim = distance_to_basket_ft([start])[0]

    best_roll, best_pop, handoff_at = 0.0, 0.0, None
    for ahead in range(1, WINDOW_FRAMES + 1):
        step = index + ahead
        if step >= len(positions):
            break
        # The ball moving to the screener at contact is a hand-off, not a screen.
        if handoff_at is None and handlers[step] == screener:
            handoff_at = step
        if screener not in positions[step]:
            continue
        to_rim = distance_to_basket_ft([positions[step][screener]])[0]
        best_roll = max(best_roll, start_to_rim - to_rim)
        best_pop = max(best_pop, to_rim - start_to_rim)

    if handoff_at is not None:
        return "dribble_handoff", "ball changed hands at the point of contact"
    if best_roll >= ROLL_GAIN_FT:
        return "pick_and_roll", f"screener cut {best_roll:.0f} ft toward the rim"
    if best_pop >= POP_GAIN_FT and start_to_rim + best_pop >= THREE_POINT_RADIUS:
        return "pick_and_pop", f"screener stepped out {best_pop:.0f} ft beyond the arc"
    return "ball_screen", "screen set, outcome not resolved"


def detect_off_ball_screens(
    positions: list[dict[int, tuple[float, float]]],
    handlers: list[int | None],
    times: list[float],
) -> list[Play]:
    """Screens between two players where NEITHER has the ball."""
    plays: list[Play] = []
    claimed: set[frozenset[int]] = set()

    for index in range(1, len(positions)):
        handler = handlers[index]
        ids = [t for t in positions[index] if t != handler]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pair = frozenset((a, b))
                if pair in claimed:
                    continue
                if _distance(positions[index][a], positions[index][b]) > SCREEN_CONTACT_FT:
                    continue
                if any(
                    a in positions[index - back] and b in positions[index - back]
                    and _distance(positions[index - back][a],
                                  positions[index - back][b]) >= SCREEN_SEPARATION_FT
                    for back in range(1, min(WINDOW_FRAMES, index) + 1)
                ):
                    claimed.add(pair)
                    plays.append(Play(
                        "off_ball_screen", times[index], a, b,
                        "two players converged away from the ball"))
    return plays


SET_WINDOW_S = 2.0             # how long a set's actions may span
RESCREEN_WINDOW_S = 2.5


def detect_sets(
    positions: list[dict[int, tuple[float, float]]],
    handlers: list[int | None],
    times: list[float],
) -> list[Play]:
    """Named sets that are COMPOSITIONS of screen actions, not new perception.

    I previously called these a data gap. That was too quick. Some named sets
    are conventions with no geometric content — "Horns Flare" is a call, and
    teams differ on it — but others are definitions built from actions already
    detected here:

    * **Spain pick-and-roll**: a pick-and-roll, then a third offensive player
      back-screens the roller while the roller is still going to the rim.
    * **Double drag**: two ball screens for the same handler, by different
      screeners, in quick succession.
    * **Re-screen**: the same screener screens the same handler twice.

    Each is the co-occurrence of primitives with a timing and identity
    constraint, so no labelled play types are needed. What still does need
    labels is any set whose name is a call rather than a shape.
    """
    ball_screens = detect_screens(positions, handlers, times)
    off_ball = detect_off_ball_screens(positions, handlers, times)
    sets: list[Play] = []

    rolls = [p for p in ball_screens if p.name == "pick_and_roll"]
    for roll in rolls:
        # A back-screen on the ROLLER, set by someone else, just after the roll.
        for screen in off_ball:
            if not (0.0 <= screen.time_s - roll.time_s <= SET_WINDOW_S):
                continue
            participants = {screen.screener_id, screen.handler_id}
            if roll.screener_id not in participants:
                continue
            third = (participants - {roll.screener_id}).pop() \
                if len(participants - {roll.screener_id}) == 1 else None
            if third is None or third == roll.handler_id:
                continue
            sets.append(Play(
                "spain_pick_and_roll", roll.time_s, roll.screener_id,
                roll.handler_id,
                f"{third} back-screens the roller {roll.screener_id} "
                f"{screen.time_s - roll.time_s:.1f}s after the ball screen"))
            break

    by_handler: dict[int, list[Play]] = {}
    for play in ball_screens:
        by_handler.setdefault(play.handler_id, []).append(play)
    for handler, plays in by_handler.items():
        ordered = sorted(plays, key=lambda p: p.time_s)
        for first, second in zip(ordered, ordered[1:]):
            gap = second.time_s - first.time_s
            if gap > RESCREEN_WINDOW_S:
                continue
            if first.screener_id == second.screener_id:
                sets.append(Play(
                    "re_screen", second.time_s, second.screener_id, handler,
                    f"{second.screener_id} screens again {gap:.1f}s later"))
            else:
                sets.append(Play(
                    "double_drag", second.time_s, second.screener_id, handler,
                    f"second screener {second.screener_id} {gap:.1f}s after "
                    f"{first.screener_id}"))
    return sorted(sets, key=lambda p: p.time_s)
