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


def derive(times: Sequence[float], holders: Sequence[int | None],
           teams: dict[int, str], shots: Sequence[Event],
           min_seconds: float = MIN_POSSESSION_S) -> list[Event]:
    """Steals and rebounds from possession changes, anchored on shots.

    A change of team within `SHOT_WINDOW_S` after a shot is a rebound: the
    attempt went up and the other side collected it. A change with no shot
    behind it is a steal or a turnover — possession lost without an attempt.
    """
    shot_times = sorted(s.time_s for s in shots)
    out: list[Event] = []
    spans = possessions(times, holders, teams, min_seconds)
    for previous, current in zip(spans, spans[1:]):
        if previous.team == current.team:
            continue
        changed_at = current.start_s
        recent_shot = any(0.0 <= changed_at - t <= SHOT_WINDOW_S for t in shot_times)
        out.append(Event(
            time_s=changed_at,
            track_id=current.track_id,
            team=current.team,
            action="rebound" if recent_shot else "steal",
            possession_change=True,
        ))
    return out
