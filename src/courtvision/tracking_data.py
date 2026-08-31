"""SportVU tracking JSON to the pipeline's own types.

The event-derivation half of this pipeline never needed pixels. `possession.py`
reads boxes, `derived_events.py` reads a possession timeline, and `plays.py`
and `formation.py` already demand court FEET and warn that pixels make their
thresholds meaningless. What all of them lacked was a good timeline: inferred
from broadcast video, one game produced 14,640 track ids and possession
resolved on 56% of frames.

Tracking data supplies it exactly. Stable player ids for a whole game, the ball
in three dimensions, real team ids, and the game clock — so the homography in
`court.py` and the scoreboard OCR in `scoreboard.py` are both bypassed.

What this CANNOT do is measure the vision stack. Tracking covers 2015-10-27 to
2016-01-23 and no release pairs it with broadcast video, so the two never meet
on the same game. This measures the ceiling: what the event logic achieves when
perception is perfect.

Schema, verified against 0021500492 (CHA at TOR, 2016-01-01):

    moment[0]  quarter
    moment[1]  unix ms
    moment[2]  game clock seconds, descending from 720
    moment[3]  shot clock, NaN near a period end
    moment[4]  unused, always None
    moment[5]  11 positions: [0] ball [-1, -1, x, y, z]
                             [1:] players [teamid, playerid, x, y, z]

Player z is present and always 0.0 — it is not a measurement, and the reference
visualisers ignore it. Ball z is real.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from courtvision.types import BALL, PLAYER, Box, Frame, Track

# Court is 94 x 50 feet; positions are in those units.
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
# `possession.normalized_distance` divides by the player's box height, so the
# box must carry a height in the SAME units as x and y or every threshold
# becomes meaningless. A player is about six and a half feet.
PLAYER_HEIGHT_FT = 6.5
PLAYER_WIDTH_FT = 2.2
BALL_SIZE_FT = 0.8
PERIOD_LENGTH_S = 720.0
BALL_TRACK_ID = -1


@dataclass(frozen=True)
class TrackingGame:
    game_id: str
    game_date: str
    frames: list[Frame]
    teams: dict[int, str]
    names: dict[int, str]
    periods: list[int]
    game_clocks: list[float]
    ball_z: list[float]

    def __len__(self) -> int:
        return len(self.frames)


def _box(x: float, y: float, height: float, width: float) -> Box:
    """A box centred on (x, y) with the given extent, in court feet."""
    return Box(x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0)


def elapsed_seconds(period: int, game_clock: float) -> float:
    """Seconds since tip-off, so frames from different periods order correctly.

    Overtime periods run five minutes rather than twelve, which a naive
    `(period - 1) * 720` gets wrong for anything after the first overtime.
    """
    if period <= 4:
        before = (period - 1) * PERIOD_LENGTH_S
        return before + (PERIOD_LENGTH_S - game_clock)
    before = 4 * PERIOD_LENGTH_S + (period - 5) * 300.0
    return before + (300.0 - game_clock)


def load_game(path: str | Path, target_hz: float | None = 10.0) -> TrackingGame:
    """Read one game's JSON into Frames, teams and names.

    SportVU events OVERLAP — each is a window around a play-by-play event, so
    the same instant appears in several of them. Moments are de-duplicated by
    their unix timestamp; without that a game yields several times more frames
    than it has instants, and every rate computed from it is wrong.

    `target_hz` downsamples the 25 Hz feed. 10 Hz matches what the video path
    uses, which keeps the possession constants comparable; None keeps all 25.
    """
    payload = json.loads(Path(path).read_text())
    events = payload.get("events", [])
    if not events:
        raise ValueError(f"{path} contains no events")

    teams: dict[int, str] = {}
    names: dict[int, str] = {}
    team_labels: dict[int, str] = {}
    for event in events:
        for side in ("home", "visitor"):
            block = event.get(side) or {}
            team_id = block.get("teamid")
            if team_id is not None and team_id not in team_labels:
                # Which physical team is called "A" is arbitrary; nothing
                # downstream may depend on it (see team_assignment.py).
                team_labels[team_id] = "A" if not team_labels else "B"
            for player in block.get("players", []):
                pid = player.get("playerid")
                if pid is None:
                    continue
                names[pid] = f"{player.get('firstname','')} {player.get('lastname','')}".strip()
                if team_id is not None:
                    teams[pid] = team_labels[team_id]

    seen: dict[int, list] = {}
    for event in events:
        for moment in event.get("moments") or []:
            if not moment or len(moment) < 6 or not moment[5]:
                continue
            stamp = moment[1]
            if stamp is None or stamp in seen:
                continue
            seen[stamp] = moment

    ordered = sorted(seen.values(), key=lambda m: elapsed_seconds(m[0], m[2]))

    # Resample on GAME TIME, not on index. The feed is 25 Hz, so a stride of
    # round(25/10)=2 gives 12.5 Hz rather than 10 — and the clock stops, so
    # index spacing is not time spacing anyway. `min_hold_frames` and
    # `max_gap_frames` in possession smoothing are counted in frames and assume
    # uniform spacing, so this has to be right.
    if target_hz:
        # Take the moment NEAREST each target time, not the first one past it.
        # On a 40 ms grid "first past 100 ms" always lands on 120, giving 8.3 Hz
        # instead of 10; nearest alternates 80/120 and averages the target.
        interval = 1.0 / target_hz
        stamps = [elapsed_seconds(m[0], m[2]) for m in ordered]
        kept, cursor, target = [], 0, stamps[0] if stamps else 0.0
        while cursor < len(ordered):
            while cursor + 1 < len(ordered) and \
                    abs(stamps[cursor + 1] - target) <= abs(stamps[cursor] - target):
                cursor += 1
            kept.append(ordered[cursor])
            target += interval
            while cursor < len(ordered) and stamps[cursor] < target:
                cursor += 1
        ordered = kept

    frames: list[Frame] = []
    periods: list[int] = []
    clocks: list[float] = []
    ball_z: list[float] = []
    for index, moment in enumerate(ordered):
        period, _stamp, game_clock = moment[0], moment[1], moment[2]
        positions = moment[5]
        tracks: list[Track] = []
        for entry in positions:
            if len(entry) < 5:
                continue
            team_id, player_id, x, y, z = entry[0], entry[1], entry[2], entry[3], entry[4]
            if team_id == -1:
                tracks.append(Track(BALL_TRACK_ID,
                                    _box(x, y, BALL_SIZE_FT, BALL_SIZE_FT),
                                    BALL, 1.0))
            else:
                tracks.append(Track(int(player_id),
                                    _box(x, y, PLAYER_HEIGHT_FT, PLAYER_WIDTH_FT),
                                    PLAYER, 1.0))
        if not tracks:
            continue
        frames.append(Frame(index, elapsed_seconds(period, game_clock), tuple(tracks)))
        periods.append(int(period))
        clocks.append(float(game_clock))
        ball = next((e for e in positions if len(e) >= 5 and e[0] == -1), None)
        ball_z.append(float(ball[4]) if ball else math.nan)

    return TrackingGame(
        game_id=str(payload.get("gameid", "")),
        game_date=str(payload.get("gamedate", "")),
        frames=frames,
        teams=teams,
        names=names,
        periods=periods,
        game_clocks=clocks,
        ball_z=ball_z,
    )
