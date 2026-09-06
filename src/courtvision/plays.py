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
# A screen is SET: the screener plants and takes the contact. Two players
# running past each other in transition satisfy every other test here -- they
# were far apart, they came together, one of them carried on -- and both of the
# false positives in the first hand-labelled sample were exactly that. A
# planted screener covers almost no ground while the contact happens.
# In feet per second, so the rule does not change meaning with the sample rate.
# A player jogs at 8 and sprints past 15; a screener taking contact is under 4.
SCREENER_MAX_SPEED_FTS = 4.0
SCREENER_SETTLE_FRAMES = 2


def _is_planted(
    positions: list[dict[int, tuple[float, float]]],
    times: list[float],
    index: int,
    screener: int,
) -> bool:
    """Whether the screener was holding still as the contact happened."""
    here = positions[index].get(screener)
    if here is None:
        return False
    for back in range(1, SCREENER_SETTLE_FRAMES + 1):
        step = index - back
        if step < 0:
            break
        there = positions[step].get(screener)
        if there is None:
            continue
        elapsed = abs(times[index] - times[step])
        if elapsed <= 0:
            continue
        if _distance(here, there) / elapsed > SCREENER_MAX_SPEED_FTS:
            return False
    return True


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
    offense: "list[set[int]] | None" = None,
) -> list[Play]:
    """Find screen actions across a sequence of frames.

    `positions[i]` maps track id to court feet in frame i; `handlers[i]` is the
    track holding the ball, or None. Tracks may appear and vanish — they do,
    constantly — so every lookup tolerates absence.

    `offense[i]` is the set of track ids attacking in frame i. Pass it whenever
    teams are known: a screen is set BY a teammate, and a defender closing on
    the handler looks identical to a screener without it. Left None the pairing
    is unrestricted, which is what a caller with no team split has to accept.
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

        attacking = offense[index] if offense is not None else None
        if attacking is not None and handler not in attacking:
            continue
        for other, screener_now in positions[index].items():
            if other == handler or frozenset((handler, other)) in claimed:
                continue
            if attacking is not None and other not in attacking:
                continue    # a defender arriving is pressure, not a screen
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

            if not _is_planted(positions, times, index, other):
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
    offense: "list[set[int]] | None" = None,
) -> list[Play]:
    """Screens between two players where NEITHER has the ball.

    `offense` matters more here than anywhere else. Without it every pair of
    non-handlers is a candidate, and of the thirty-six pairs among nine
    non-handlers only six are teammates on offense — the rest are defenders
    crossing, or an attacker and the man guarding him, who are near each other
    by definition. Measured on a full game of exact coordinates, dropping the
    filter is the difference between roughly eighty off-ball screens and six
    hundred.
    """
    plays: list[Play] = []
    claimed: set[frozenset[int]] = set()

    for index in range(1, len(positions)):
        handler = handlers[index]
        attacking = offense[index] if offense is not None else None
        ids = [t for t in positions[index]
               if t != handler and (attacking is None or t in attacking)]
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
                    screener, cutter = _screener_and_cutter(positions, index, a, b)
                    if not _is_planted(positions, times, index, screener):
                        continue
                    claimed.add(pair)
                    name, evidence = classify_off_ball_screen(
                        positions, handlers, index, cutter=cutter,
                        screener=screener)
                    plays.append(Play(name, times[index], screener, cutter,
                                      evidence))
    return plays



CUTTER_MOVE_FT = 4.0           # how far a cutter must go for a direction to mean anything
# How far ahead of contact the roles are judged. A screener plants and stays;
# the cutter is gone.
ROLE_WINDOW_FRAMES = 8


def _screener_and_cutter(
    positions: list[dict[int, tuple[float, float]]],
    index: int,
    first: int,
    second: int,
) -> tuple[int, int]:
    """Which of the two set the screen, and which used it.

    Taking the lower track id as the screener — which this did — gets the roles
    backwards half the time, and the roles are not cosmetic: every off-ball
    screen is named by the direction the CUTTER travels, so reversing them
    turns a pin down into a back screen. Detection was unaffected and the
    naming was scrambled, which is why a game came back with seven pin downs.

    A screen is set and then left. The screener is whichever of the two has
    gone LESS far by the end of the window.
    """
    def travelled(track: int) -> float:
        start = positions[index].get(track)
        if start is None:
            return 0.0
        far = 0.0
        for ahead in range(1, ROLE_WINDOW_FRAMES + 1):
            step = index + ahead
            if step < len(positions) and track in positions[step]:
                far = max(far, _distance(start, positions[step][track]))
        return far

    return ((first, second) if travelled(first) <= travelled(second)
            else (second, first))


def classify_off_ball_screen(
    positions: list[dict[int, tuple[float, float]]],
    handlers: list[int | None],
    index: int,
    cutter: int,
    screener: int,
) -> tuple[str, str]:
    """Name an off-ball screen by where the cutter goes afterwards.

    I had filed these under "needs labelled play types". That was wrong in the
    same way as before: a flare screen is not a name someone assigned, it is a
    direction. The screen types differ by where the cutter ends up relative to
    two fixed things — the rim, and the ball:

    * **back screen** — cutter goes toward the rim
    * **flare** — cutter goes away from the ball, staying out on the perimeter
    * **pin down** — cutter comes up toward the ball, away from the rim
    * **cross screen** — cutter crosses the lane without much change in either

    Both distances are measurable in court feet, so this needs no labels. What
    still does is anything named after a CALL rather than a shape.
    """
    start = positions[index].get(cutter)
    handler = handlers[index]
    ball = positions[index].get(handler) if handler is not None else None
    if start is None:
        return "off_ball_screen", "two players converged away from the ball"

    best = None
    for ahead in range(1, WINDOW_FRAMES + 1):
        step = index + ahead
        if step >= len(positions) or cutter not in positions[step]:
            continue
        here = positions[step][cutter]
        travelled = _distance(start, here)
        if best is None or travelled > best[0]:
            best = (travelled, here)
    if best is None or best[0] < CUTTER_MOVE_FT:
        return "off_ball_screen", "screen set, cutter did not commit anywhere"

    travelled, end = best
    to_rim = distance_to_basket_ft([start])[0] - distance_to_basket_ft([end])[0]
    to_ball = (_distance(start, ball) - _distance(end, ball)) if ball else 0.0

    if to_rim >= CUTTER_MOVE_FT:
        return "back_screen", f"cutter went {to_rim:.0f} ft toward the rim"
    if ball is not None and to_ball <= -CUTTER_MOVE_FT and to_rim <= 0:
        return "flare_screen", f"cutter flared {-to_ball:.0f} ft away from the ball"
    if ball is not None and to_ball >= CUTTER_MOVE_FT and to_rim <= -CUTTER_MOVE_FT / 2:
        return "pin_down", f"cutter came {to_ball:.0f} ft up toward the ball"
    return "cross_screen", f"cutter crossed {travelled:.0f} ft without changing depth"


SET_WINDOW_S = 2.0             # how long a set's actions may span
RESCREEN_WINDOW_S = 2.5


TRANSITION_GAIN_FT = 25.0      # ground the ball must make toward the rim
TRANSITION_SPEED_FT_S = 12.0   # and how fast, to be a break rather than a walk-up
STAGGER_WINDOW_S = 2.0


def detect_transition(
    positions: list[dict[int, tuple[float, float]]],
    handlers: list[int | None],
    times: list[float],
) -> list[Play]:
    """The ball advancing at speed — a break, not a walk-up.

    Transition is one of the largest play-type categories in basketball
    analytics and is purely geometric: the ball covers ground toward the rim
    quickly. Both terms are measurable, so no labels are needed.

    A half-court set walks the ball up at a few feet per second and covers
    little ground once it arrives. A break covers most of the visible court in
    a couple of seconds. The thresholds separate those, and anything between is
    left unnamed rather than forced into one.
    """
    plays: list[Play] = []
    claimed: set[int] = set()

    for start in range(len(positions)):
        handler = handlers[start]
        if handler is None or handler in claimed or handler not in positions[start]:
            continue
        began = distance_to_basket_ft([positions[start][handler]])[0]
        for end in range(start + 1, len(positions)):
            if handlers[end] != handler or handler not in positions[end]:
                break
            elapsed = times[end] - times[start]
            if elapsed <= 0:
                continue
            gained = began - distance_to_basket_ft([positions[end][handler]])[0]
            if gained >= TRANSITION_GAIN_FT and gained / elapsed >= TRANSITION_SPEED_FT_S:
                claimed.add(handler)
                plays.append(Play(
                    "transition", times[start], handler, handler,
                    f"ball advanced {gained:.0f} ft toward the rim in "
                    f"{elapsed:.1f}s"))
                break
            if elapsed > 4.0:
                break
    return plays


def detect_stagger(
    positions: list[dict[int, tuple[float, float]]],
    handlers: list[int | None],
    times: list[float],
) -> list[Play]:
    """Two different screeners for the SAME cutter in quick succession.

    A stagger is the off-ball twin of a double drag, and composes the same way:
    take the off-ball screens, group them by cutter, and look for two with
    different screeners close together in time. No new perception, and no
    labels.
    """
    screens = detect_off_ball_screens(positions, handlers, times)
    by_cutter: dict[int, list[Play]] = {}
    for play in screens:
        by_cutter.setdefault(play.handler_id, []).append(play)

    plays: list[Play] = []
    for cutter, group in by_cutter.items():
        ordered = sorted(group, key=lambda p: p.time_s)
        for first, second in zip(ordered, ordered[1:]):
            gap = second.time_s - first.time_s
            if gap <= STAGGER_WINDOW_S and first.screener_id != second.screener_id:
                plays.append(Play(
                    "stagger_screen", second.time_s, second.screener_id, cutter,
                    f"{first.screener_id} then {second.screener_id} screen for "
                    f"{cutter}, {gap:.1f}s apart"))
    return plays



def detect_sets(
    positions: list[dict[int, tuple[float, float]]],
    handlers: list[int | None],
    times: list[float],
    formations: list[str | None] | None = None,
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

    Given `formations` — the formation name per frame, from
    `formation.classify_formation` — it also names sets that are a SHAPE plus an
    ACTION:

    * **Horns flare**: the horns alignment, then a flare screen out of it.
    * **Horns set**: the horns alignment, then a ball screen.

    "Horns Flare" was the example I kept giving for something that needs
    labelled play types. It does not: horns is an arrangement that can be seen
    in one frame, a flare is a direction a cutter takes, and the set is their
    conjunction inside a window. What genuinely still needs labels is a name
    that is a CALL rather than a shape — a coach's word for a set, which teams
    disagree on and no camera can see.
    """
    ball_screens = detect_screens(positions, handlers, times)
    off_ball = detect_off_ball_screens(positions, handlers, times)
    sets: list[Play] = []

    if formations:
        for index, shape in enumerate(formations):
            if shape != "horns" or index >= len(times):
                continue
            began = times[index]
            for play in off_ball + ball_screens:
                if not (0.0 <= play.time_s - began <= SET_WINDOW_S):
                    continue
                if play.name == "flare_screen":
                    sets.append(Play(
                        "horns_flare", play.time_s, play.screener_id,
                        play.handler_id,
                        f"flare out of horns {play.time_s - began:.1f}s after "
                        f"the alignment"))
                    break
                if play.name in {"pick_and_roll", "pick_and_pop", "ball_screen"}:
                    sets.append(Play(
                        "horns_set", play.time_s, play.screener_id,
                        play.handler_id,
                        f"{play.name} out of horns "
                        f"{play.time_s - began:.1f}s after the alignment"))
                    break

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
