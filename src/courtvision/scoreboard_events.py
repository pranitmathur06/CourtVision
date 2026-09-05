"""Events the scoreboard STATES, rather than events vision infers.

Every number here was measured on an uncut broadcast (game 0042400407) against
the official play-by-play at +-5s, with no ball detection anywhere in the path:

    3pt make        P 1.000  R 0.955  F1 0.977
    2pt make        P 0.951  R 0.929  F1 0.940
    any make        P 0.960  R 0.880  F1 0.918
    free throw      P 0.946  R 0.795  F1 0.864
    foul            P 0.568  R 0.894  F1 0.694
    missed FG       P 0.682  R 0.645  F1 0.663

For comparison, deriving field goals from BALL TRAJECTORY tops out at F1 0.859
even on 25 Hz tracking data with stable player ids and a true ball height --
that is the ceiling of the derived approach, and the scoreboard beats it. This
module exists because observed beats derived every time it is available.

The inputs are per-frame readings, not video: a clock reading immune to
condensing and cuts (`clock_reader`), and score/shot-clock digits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# A single possession cannot legally change a team's score by more than three
# points, so a larger jump is a misread, not a basket. Bounding it is what lets
# monotonicity do the rest of the work.
MAX_POINTS_PER_SCORE = 3
# Digits flicker. Requiring the score never to decrease discards the majority of
# OCR noise for free, because a misread is as likely to fall as to rise.
# One foul produces several clock stalls -- the whistle, then a pause between
# each free throw. Merging them into one stoppage took foul precision from
# 0.289 to 0.568; without it, gating on free throws made results WORSE
# (F1 0.439 -> 0.417) because half of all raw stalls sit near a free throw.
MERGE_STOPPAGE_S = 4.0
# Below this a "stall" is just two reads landing on the same second.
MIN_STALL_S = 2.5
# Reads further apart than this are a coverage gap, not a stopped clock.
MAX_READ_GAP_S = 1.5


@dataclass(frozen=True)
class ScoreEvent:
    """A scoring play, typed by how much the score moved."""

    elapsed_s: float
    points: int
    team: str

    @property
    def action(self) -> str:
        return {1: "free_throw", 2: "two_point_make",
                3: "three_point_make"}[self.points]


def score_events(readings: Iterable[tuple[float, int | None, int | None]],
                 teams: tuple[str, str] = ("home", "away")) -> list[ScoreEvent]:
    """Typed scoring events from (elapsed_s, home, away) score readings.

    The magnitude of the jump carries the event type and nothing else needed to
    read it: +1 is a free throw, +2 and +3 field goals. The score reader was
    already emitting changes; only the size was being thrown away.

    Readings must be time-ordered. A reading with either side missing is
    skipped rather than guessed.
    """
    events: list[ScoreEvent] = []
    best_home = best_away = 0
    started = False
    for elapsed, home, away in readings:
        if home is None or away is None:
            continue
        if not started:
            best_home, best_away, started = home, away, True
            continue
        if home < best_home or away < best_away:
            continue                      # a fall is a misread, not a basket
        gained_home = home - best_home
        gained_away = away - best_away
        if gained_home > MAX_POINTS_PER_SCORE or gained_away > MAX_POINTS_PER_SCORE:
            continue                      # too big to be one possession
        if gained_home and gained_away:
            continue                      # both cannot score at once
        if gained_home:
            events.append(ScoreEvent(elapsed, gained_home, teams[0]))
        elif gained_away:
            events.append(ScoreEvent(elapsed, gained_away, teams[1]))
        best_home, best_away = home, away
    return events


def stoppages(readings: Sequence[tuple[float, float | None]],
              merge_s: float = MERGE_STOPPAGE_S,
              min_stall_s: float = MIN_STALL_S,
              max_gap_s: float = MAX_READ_GAP_S) -> list[tuple[float, float]]:
    """Dead balls, as (elapsed_s, stall_duration_s).

    Game time only stops for a foul, a timeout, a violation, the ball out of
    bounds, or the end of a period. `readings` is (video_time_s, elapsed_s),
    because detecting a stall needs BOTH clocks: video time advancing while
    game time stands still.

    A stall requires readable clocks on both sides, so a gap in coverage cannot
    masquerade as a stopped clock.
    """
    usable = [(v, e) for v, e in readings if e is not None]
    usable.sort()
    raw: list[tuple[float, float]] = []
    index = 0
    while index < len(usable) - 1:
        end = index
        while (end + 1 < len(usable)
               and usable[end + 1][0] - usable[end][0] <= max_gap_s
               and abs(usable[end + 1][1] - usable[index][1]) <= 0.6):
            end += 1
        span = usable[end][0] - usable[index][0]
        if end > index and span >= min_stall_s:
            raw.append((usable[index][1], span))
            index = end
        else:
            index += 1

    merged: list[list[float]] = []
    for elapsed, span in raw:
        if merged and elapsed - merged[-1][0] <= merge_s:
            merged[-1][1] += span
        else:
            merged.append([elapsed, span])
    return [(a, b) for a, b in merged]


def possession_changes(readings: Sequence[tuple[float, float | None]],
                       min_jump_s: float = 3.0) -> list[float]:
    """Elapsed times where the shot clock reset, i.e. the ball changed hands.

    `readings` is (elapsed_s, shot_clock_s). A reset is the clock jumping UP;
    it counts down otherwise, so the direction alone identifies the event.
    """
    out: list[float] = []
    previous: float | None = None
    for elapsed, value in readings:
        if value is None:
            previous = None
            continue
        if previous is not None and value > previous + min_jump_s:
            out.append(elapsed)
        previous = value
    return out


def missed_shots(changes: Sequence[float], makes: Sequence[float],
                 separation_s: float = 4.0) -> list[float]:
    """Possessions that ended without points.

    Measured F1 0.663 against missed field goals, and 0.726 against misses and
    turnovers together -- the gap is exactly the turnovers, which end a
    possession identically as far as a scoreboard can see. Separating them
    needs a sensor this module does not have.
    """
    return [c for c in changes
            if not any(abs(c - m) <= separation_s for m in makes)]


def fouls(stops: Sequence[tuple[float, float]],
          free_throws: Sequence[float],
          window_s: float = 8.0) -> list[float]:
    """Stoppages followed by a free throw, which are fouls at P 0.960.

    Recall is only 0.511 because non-shooting fouls produce no free throw at
    all; `stops` alone reaches R 0.894 at P 0.568. Which to use depends on
    whether a caller would rather miss fouls or invent them.
    """
    return [elapsed for elapsed, _ in stops
            if any(0 <= ft - elapsed <= window_s for ft in free_throws)]
