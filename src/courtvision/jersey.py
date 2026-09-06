"""Turn jersey numbers into names by voting over a track.

The READER lives in `courtvision.digit_net`; this module is the layer above it.
A general scene-text reader (`read_number` below) was the first attempt and is
kept for comparison, but it answers on 17% of crops and is wrong often enough
to be unusable alone. The trained digit classifier replaced it.

A single frame is a bad witness either way. Measured on 130 hand-labelled
broadcast crops, the good reader answers on a quarter of them and is right 88%
of the time when it does -- players turn, arms cross the chest, and the number
blurs. No amount of preprocessing fixes a player facing away.

But a player stays on the floor for a whole possession, so the same track
offers dozens of chances. Voting over a track converts a weak per-frame reader
into a strong per-player one, and the roster makes it stronger still: only
about fifteen numbers per team are possible, so a read that is not one of them
is discarded rather than believed.

Two guards matter more than the reader:

  * A number must be on the roster of the team whose COLOUR that player wears.
    Reading "7" on a player in the other kit is evidence of a misread, not of
    a seventh.
  * A verdict needs both a minimum number of votes and a margin over the
    runner-up. Attributing a play to the wrong player is the one error a
    coaching tool cannot make, so an uncertain track stays anonymous.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

# Crop geometry, as fractions of the player box. Swept on a real broadcast: the
# torso band from 15% to 52% of box height read best.
CROP_X = (0.15, 0.85)
CROP_Y = (0.15, 0.52)
# Upscale before reading. The digits are ~50 px tall in a 720p frame, below
# what a general scene-text reader expects.
UPSCALE = 6
# A player smaller than this is too far away for the number to survive.
MIN_BOX_HEIGHT_PX = 100
# Votes needed before a track is named, and how far ahead of second place.
MIN_VOTES = 3
MIN_MARGIN = 2


@dataclass(frozen=True)
class Verdict:
    """A named track, with the evidence that named it."""

    track_id: int
    number: str
    name: str
    team: str
    votes: int
    runner_up: int

    @property
    def confident(self) -> bool:
        return (self.votes >= MIN_VOTES
                and self.votes - self.runner_up >= MIN_MARGIN)


def jersey_crop(image: np.ndarray, box) -> np.ndarray | None:
    """The torso region of a player box, upscaled for reading."""
    import cv2

    x1, y1, x2, y2 = (float(v) for v in box[:4])
    width, height = x2 - x1, y2 - y1
    if height < MIN_BOX_HEIGHT_PX:
        return None
    frame_h, frame_w = image.shape[:2]
    a = int(max(0, x1 + CROP_X[0] * width))
    c = int(min(frame_w, x1 + CROP_X[1] * width))
    b = int(max(0, y1 + CROP_Y[0] * height))
    d = int(min(frame_h, y1 + CROP_Y[1] * height))
    if c - a < 4 or d - b < 6:
        return None
    crop = image[b:d, a:c]
    if crop.size == 0:
        return None
    return cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE,
                      interpolation=cv2.INTER_CUBIC)


def read_number(reader, crop: np.ndarray,
                min_confidence: float = 0.4) -> str | None:
    """One jersey number from a crop, or None.

    Only one- and two-digit results are accepted: anything longer is the reader
    finding structure in a jersey's piping or a sponsor logo.
    """
    if crop is None or crop.size == 0:
        return None
    try:
        found = reader.readtext(crop, allowlist="0123456789", detail=1,
                                text_threshold=min_confidence)
    except Exception:
        return None
    for _, text, confidence in found:
        cleaned = text.strip().lstrip("0") or "0"
        if cleaned.isdigit() and len(cleaned) <= 2 and confidence >= min_confidence:
            return cleaned
    return None


class JerseyVoter:
    """Accumulates per-frame reads and names tracks once the evidence is in."""

    def __init__(self, roster: dict[str, tuple[str, str]]):
        """`roster` maps jersey number -> (team, player name)."""
        self.roster = roster
        self._votes: dict[int, Counter] = defaultdict(Counter)
        self._teams: dict[int, Counter] = defaultdict(Counter)

    def observe(self, track_id: int, number: str | None,
                team: str | None = None) -> None:
        """Record one read. A number off the roster is discarded, not stored."""
        if team is not None:
            self._teams[track_id][team] += 1
        if number is None or number not in self.roster:
            return
        if team is not None and self.roster[number][0] != team:
            # The number belongs to the other team's roster, so this is a
            # misread rather than evidence.
            return
        self._votes[track_id][number] += 1

    def verdict(self, track_id: int) -> Verdict | None:
        """The best-supported name for a track, or None while it is unclear."""
        counter = self._votes.get(track_id)
        if not counter:
            return None
        ranked = counter.most_common(2)
        number, votes = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        team, name = self.roster[number]
        found = Verdict(track_id, number, name, team, votes, runner_up)
        return found if found.confident else None

    def all_verdicts(self) -> dict[int, Verdict]:
        """Every track confident enough to name."""
        out = {}
        for track_id in self._votes:
            found = self.verdict(track_id)
            if found is not None:
                out[track_id] = found
        return out


def roster_from_endpoint(team_ids_and_codes, season: str) -> dict:
    """Jersey number -> (team code, player name), from the league's own roster.

    The map is what makes a weak reader usable: about fifteen numbers per team
    are possible, so most misreads can simply be dropped.
    """
    from nba_api.stats.endpoints import commonteamroster

    roster: dict[str, tuple[str, str]] = {}
    for team_id, code in team_ids_and_codes:
        payload = commonteamroster.CommonTeamRoster(
            team_id=team_id, season=season, timeout=60).get_dict()
        table = payload["resultSets"][0]
        headers = table["headers"]
        number_at = headers.index("NUM")
        name_at = headers.index("PLAYER")
        for row in table["rowSet"]:
            raw = row[number_at]
            if raw is None:
                continue
            number = str(raw).lstrip("0") or "0"
            roster[number] = (code, row[name_at])
    return roster


# ---------------------------------------------------------------------------
# Who is actually on the floor.
#
# Reading a number from 720p broadcast is weak -- 16% of crops, and voting over
# a track is limited by how long a track survives. But naming a player does not
# require reading 28 possible numbers. The feed says exactly who is out there:
# the box score gives five starters a side, and every substitution is logged as
# "SUB: <in> FOR <out>". Applying them in order maintains the on-court five.
#
# That turns identification into a FIVE-way assignment per team, which a weak
# reader can break ties in even when it cannot solve the problem alone.
#
# Live caveat, which is the whole reason this is kept separate from the reader:
# substitutions arrive on the feed like everything else, so a live system is
# behind by however long the feed lags. Starters are known before tip-off; the
# five drifts from truth until the first substitution arrives.

SUB_PATTERN = None


def parse_substitution(description: str) -> tuple[str, str] | None:
    """('player coming in', 'player going out') from a substitution line."""
    import re

    global SUB_PATTERN
    if SUB_PATTERN is None:
        SUB_PATTERN = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$",
                                 re.IGNORECASE)
    matched = SUB_PATTERN.match((description or "").strip())
    if not matched:
        return None
    return matched.group(1).strip(), matched.group(2).strip()


class OnCourt:
    """The five players per team currently on the floor.

    Names are matched on the family name the play-by-play uses, which is what
    substitution lines carry. A substitution naming somebody not currently on
    the floor is applied anyway -- the feed is the authority, and refusing it
    would leave the five permanently wrong after one missed event.
    """

    def __init__(self, starters: dict[str, list[str]]):
        """`starters` maps team code -> five player names."""
        self.teams = {team: list(names) for team, names in starters.items()}

    def substitute(self, coming_in: str, going_out: str) -> bool:
        """Apply one substitution. True when a team was actually changed."""
        for team, names in self.teams.items():
            for index, name in enumerate(names):
                if name.split()[-1].lower() == going_out.split()[-1].lower():
                    names[index] = coming_in
                    return True
        return False

    def candidates(self, team: str) -> list[str]:
        """The five names a detected player on this team could be."""
        return list(self.teams.get(team, []))

    def everyone(self) -> list[str]:
        out = []
        for names in self.teams.values():
            out.extend(names)
        return out


def on_court_timeline(actions, starters: dict[str, list[str]]):
    """[(elapsed_s, {team: [five names]})] across a game.

    Built by replaying substitutions over the starting fives, so the five is
    exact wherever the feed is -- the uncertainty is in WHEN, not who, and that
    is the alignment problem solved elsewhere.
    """
    import re

    from courtvision.event_alignment import elapsed_seconds

    clock = re.compile(r"PT(\d+)M([\d.]+)S")
    state = OnCourt(starters)
    timeline = [(0.0, {team: list(names)
                       for team, names in state.teams.items()})]
    for action in actions:
        if (action.get("actionType") or "").strip() != "Substitution":
            continue
        matched = clock.fullmatch((action.get("clock") or "").strip())
        period = action.get("period")
        if not matched or period is None:
            continue
        moment = elapsed_seconds(
            period, int(matched.group(1)) * 60 + float(matched.group(2)))
        parsed = parse_substitution(action.get("description") or "")
        if parsed is None:
            continue
        if state.substitute(*parsed):
            timeline.append((moment, {team: list(names)
                                      for team, names in state.teams.items()}))
    return timeline
