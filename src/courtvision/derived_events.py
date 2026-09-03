"""Derive possession events from the possession timeline, not from pixels.

Measured separability of each action against ordinary play, 200-240 windows,
VideoMAE features, both representations:

    class     player crop   full frame
    rebound        +0.042      +0.113
    steal          +0.054      +0.037
    block          -0.170      -0.125

Shot is learnable and lands at 0.94x of the official count over a full game.
Rebound is learnable only from the whole frame. Steal is barely separable in
either view, and block is BELOW chance — the classifier does worse than always
guessing. No amount of extra data fixes a signal that is not there, and that is
why steal ran at 33.8x the official count however it was trained.

The reason is that these are not visual categories. A steal is not a look; it
is possession changing team without a shot. A rebound is possession resolving
after a shot goes up. A block is a shot the defence retains. They are defined by
the possession structure, which the pipeline already computes to 9/9 on the
human-annotated answer key, and by shots, which the classifier already gets
right.

So derive them instead of recognising them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from courtvision.types import Event


@dataclass(frozen=True)
class Possession:
    """A stretch of one team holding the ball."""
    team: str
    track_id: int | None
    start_s: float
    end_s: float


# A shot's outcome resolves within a couple of seconds of the attempt.
SHOT_WINDOW_S = 3.0
# Ignore possession flickers shorter than this. Track ids restart at every
# broadcast cut — a full game yields 14,640 of them — so a bare team change is
# mostly tracker noise. Swept against BARD's own labels for one game, counting
# derived steals against the 18 real ones:
#
#     min_seconds   derived   steal ratio
#             0.6       377        20.4x
#             3.0        98         5.3x
#             5.0        65         3.5x
#             8.0        26         1.4x
#
# An NBA possession averages about fourteen seconds, so a six-second floor
# discards flickers while keeping real possessions. It is deliberately below
# the 8.0 that scored best on that footage: those were concatenated clips
# averaging 18.8 s, which truncates possessions and flatters a high threshold.
MIN_POSSESSION_S = 6.0


def possessions(times: Sequence[float], holders: Sequence[int | None],
                teams: dict[int, str],
                min_seconds: float = MIN_POSSESSION_S) -> list[Possession]:
    """Collapse a per-frame holder timeline into team possessions."""
    spans: list[Possession] = []
    for time_s, holder in zip(times, holders):
        team = teams.get(holder) if holder is not None else None
        if team is None:
            continue
        if spans and spans[-1].team == team:
            spans[-1] = Possession(team, spans[-1].track_id, spans[-1].start_s, time_s)
        else:
            spans.append(Possession(team, holder, time_s, time_s))
    return [s for s in spans if s.end_s - s.start_s >= min_seconds]


# A steal hands the ball over a short distance — a defender takes it off the
# handler. A bad pass travels far before an opponent collects it, and a loose
# ball further still. Measured against the 13 official steals in 0021500492,
# filtering non-shot possession changes by how far the ball moved between the
# old holder and the new one:
#
#     no limit   47 emitted  3.62x  precision 0.19
#     <= 15 ft   25          1.92x  precision 0.24
#     <=  6 ft   10          0.77x  precision 0.40
#
# 15 ft brings the count inside tolerance; 6 ft buys precision at the cost of
# recall. Neither makes steal a solved class — see the ceiling note in
# docs/continuous-game-accuracy.md.
MAX_HANDOFF_FT = 15.0


# A rebound is the first player to establish possession after a miss — and
# `possessions` above cannot see it, because it collapses the timeline by TEAM.
# An offensive rebound keeps the ball with the same team, so the span simply
# continues and no change is ever emitted. Offensive boards are about a quarter
# of all rebounds, which put a hard ceiling near 0.75 on recall no matter how
# good perception was.
#
# So rebounds are found on the PLAYER timeline instead: after a missed shot,
# the first player to hold the ball for `min_hold_s` got the board, whichever
# side he is on.
REBOUND_WINDOW_S = 4.0
MIN_HOLD_S = 0.5
HOLD_GAP_S = 1.0


def _player_spans(times: Sequence[float], holders: Sequence[int | None],
                  min_hold_s: float, gap_s: float) -> list[tuple[int, float, float]]:
    """Contiguous stretches of one player holding the ball, bridging short gaps."""
    spans: list[list] = []
    for time_s, holder in zip(times, holders):
        if holder is None:
            continue
        if spans and spans[-1][0] == holder and time_s - spans[-1][2] <= gap_s:
            spans[-1][2] = time_s
        else:
            spans.append([holder, time_s, time_s])
    return [(h, a, b) for h, a, b in spans if b - a >= min_hold_s]


def rebounds(times: Sequence[float], holders: Sequence[int | None],
             teams: dict[int, str], missed_times: Sequence[float],
             window_s: float = REBOUND_WINDOW_S,
             min_hold_s: float = MIN_HOLD_S,
             gap_s: float = HOLD_GAP_S) -> list[Event]:
    """One rebound per missed shot: the first player to establish possession."""
    spans = _player_spans(times, holders, min_hold_s, gap_s)
    out: list[Event] = []
    claimed: set[int] = set()
    for shot_time in sorted(missed_times):
        for index, (holder, start, _end) in enumerate(spans):
            if index in claimed or start <= shot_time:
                continue
            if start - shot_time > window_s:
                break
            claimed.add(index)
            out.append(Event(time_s=start, track_id=holder,
                             team=teams.get(holder), action="rebound",
                             possession_change=True))
            break
    return sorted(out, key=lambda e: e.time_s)


def derive(times: Sequence[float], holders: Sequence[int | None],
           teams: dict[int, str], shots: Sequence[Event],
           min_seconds: float = MIN_POSSESSION_S,
           positions: dict[int, dict[float, tuple[float, float]]] | None = None,
           max_handoff_ft: float = MAX_HANDOFF_FT,
           made_times: Sequence[float] = ()) -> list[Event]:
    """Steals and rebounds from possession changes, anchored on shots.

    A change of team within `SHOT_WINDOW_S` after a MISSED shot is a rebound:
    the attempt went up and the other side collected it. A change with no shot
    behind it is a steal or a turnover — possession lost without an attempt.

    A change after a MADE shot is neither. It is the inbound that follows a
    basket, and emitting it as a rebound was roughly half of all rebounds this
    produced, since about half of field goals go in. `made_times` comes from
    `shot_detection.makes`; leaving it empty restores the old behaviour.
    """
    shot_times = sorted(s.time_s for s in shots)
    made = sorted(made_times)
    out: list[Event] = []
    spans = possessions(times, holders, teams, min_seconds)
    for previous, current in zip(spans, spans[1:]):
        if previous.team == current.team:
            continue
        changed_at = current.start_s
        recent_shot = any(0.0 <= changed_at - t <= SHOT_WINDOW_S for t in shot_times)
        if recent_shot:
            continue          # rebounds come from `rebounds` on the player timeline
        if any(0.0 <= changed_at - t <= SHOT_WINDOW_S for t in made):
            continue                     # the inbound after a basket, not a rebound
        if not recent_shot and positions is not None:
            # No shot behind it, so this is a steal or a turnover. A steal is
            # the short handover; drop the long ones.
            start = positions.get(previous.track_id, {}).get(round(previous.end_s, 1))
            end = positions.get(current.track_id, {}).get(round(current.start_s, 1))
            if start is not None and end is not None:
                moved = math.hypot(start[0] - end[0], start[1] - end[1])
                if moved > max_handoff_ft:
                    continue
        out.append(Event(
            time_s=changed_at,
            track_id=current.track_id,
            team=current.team,
            action="rebound" if recent_shot else "steal",
            possession_change=True,
        ))
    return out
