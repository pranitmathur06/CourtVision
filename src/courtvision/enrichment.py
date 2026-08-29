"""v3 — replace `Player 7` with a real identity, using play-by-play data.

The insight that makes this tractable: **you do not have to recognise a player to
name them.** Jersey-number OCR is a genuinely hard vision problem — small text,
motion blur, occlusion, perspective (spec §10 says as much). But basketball games
already produce an authoritative, timestamped event stream: the official
play-by-play. If you know which game a clip is from and roughly where in the game
clock it sits, naming becomes a JOIN, not a recognition task.

Two regimes:

* **Archived** — you know the GameID because you chose the game. Fetch official
  play-by-play (the `nba_api` package wraps stats.nba.com), then align detected
  events to it by game clock. The clock is readable from the broadcast scoreboard
  overlay: fixed position, large high-contrast digits, far easier than jersey OCR.
* **Live** — identical, with a live feed. A few seconds of latency is fine for
  commentary. Official real-time feeds are commercial.

BARD clips need neither, and that is what makes this testable today: each clip's
metadata already carries `GameID`, `GameEventID` and the official play
description, e.g. `"Cunningham REBOUND (Off:0 Def:1)"`.

**The danger this module is built around:** a misalignment does not produce a
vague answer, it produces a confidently WRONG name. So `align` refuses to guess:
an event with no confident match keeps its anonymous `Player N` label. That is
the same discipline `commentary.validate_commentary` applies — it is better to
say less than to say something false about a real person.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass, replace

from courtvision.types import Event

# BARD stores the official description in the `title` query parameter.
_TITLE = re.compile(r"title=([^&]+)")
_GAME_ID = re.compile(r"GameID=(\d+)")
_EVENT_ID = re.compile(r"GameEventID=(\d+)")

# Leading player token in an NBA play description: "Cunningham REBOUND ...",
# "K. Johnson REBOUND ...", "Williams 2' Running Dunk ...".
#
# A name token must contain a lowercase letter; the action verbs that follow are
# ALL CAPS (REBOUND, STEAL, BLOCK). Without that distinction the pattern happily
# swallows the verb and reports the player as "Cunningham REBOUND".
# Matches a leading name token in any script, then Python's Unicode-aware
# str methods decide whether it is really a name. Enumerating uppercase ranges in
# a character class does not work: `[A-Z]` misses "Šengün" entirely (returning no
# name at all) and `[A-Za-z]` truncates "Jokić" to "Joki". Both misname a real
# person in published commentary, which is the one thing this must not do.
_LEADING_NAME = re.compile(
    r"^((?:[^\W\d_]\.\s*)*[^\W\d_][^\W\d_'\-]*(?:\s+(?:Jr\.|Sr\.|II|III|IV))?)"
)


def _is_name_token(token: str) -> bool:
    """A name starts with a capital and is not an ALL-CAPS action verb."""
    core = token.replace(".", " ").split()
    if not core:
        return False
    last = core[-1]
    return last[:1].isupper() and any(c.islower() for c in last)

# Play-by-play verbs mapped onto our action vocabulary.
_ACTION_WORDS: dict[str, str] = {
    "REBOUND": "rebound",
    "STEAL": "steal",
    "BLOCK": "block",
    "Shot": "shot",
    "Dunk": "shot",
    "Layup": "shot",
    "Free Throw": "shot",
    "AST": "pass",
}


@dataclass(frozen=True)
class PlayByPlayEvent:
    """One official play-by-play entry."""

    game_id: str
    event_id: int
    description: str
    player: str | None
    action: str | None


def parse_nba_url(url: str) -> PlayByPlayEvent | None:
    """Parse one BARD `urls` field into a play-by-play event."""
    game = _GAME_ID.search(url)
    event = _EVENT_ID.search(url)
    title = _TITLE.search(url)
    if not (game and event and title):
        return None
    description = urllib.parse.unquote(title.group(1))
    return PlayByPlayEvent(
        game_id=game.group(1),
        event_id=int(event.group(1)),
        description=description,
        player=extract_player(description),
        action=extract_action(description),
    )


def extract_player(description: str) -> str | None:
    """Leading player name from an official play description."""
    match = _LEADING_NAME.match(description.strip())
    if not match:
        return None
    name = match.group(1).strip()
    # A description that opens with a Title-Case verb names nobody.
    if name in _ACTION_WORDS or not _is_name_token(name):
        return None
    return name or None


def extract_action(description: str) -> str | None:
    """Map an official description onto our action vocabulary."""
    for word, action in _ACTION_WORDS.items():
        if word in description:
            return action
    return None


def align(
    events: Sequence[Event],
    plays: Sequence[PlayByPlayEvent],
    require_action_match: bool = True,
) -> list[Event]:
    """Attach real player names to events, refusing to guess.

    Events are matched to plays in order. A play is only consumed when its action
    agrees with the event's (unless `require_action_match` is False), because a
    name attached to the wrong play is worse than no name: the commentary would
    state something false about a real person with full confidence.

    Events with no confident match are returned unchanged, keeping their
    anonymous track id.
    """
    named: list[Event] = []
    remaining = list(plays)

    for event in events:
        match_index = None
        for index, play in enumerate(remaining):
            if play.player is None:
                continue
            if require_action_match and play.action != event.action:
                continue
            match_index = index
            break

        if match_index is None:
            named.append(event)
            continue

        play = remaining.pop(match_index)
        named.append(replace(event, player_name=play.player))

    return named


def plays_for_clip(clip_path: str, metadata_csv: str) -> list[PlayByPlayEvent]:
    """Official plays for the game a clip came from.

    BARD names its clips `<away>-vs-<home>-<GameID>/<n>.mp4`, so the game is
    recoverable from the path alone — no scoreboard OCR needed. That is what
    makes BARD the right harness for building this layer before wiring a live
    feed: the join key is already there.

    Returns [] when the clip is not from a known game, which is the honest
    answer for arbitrary footage; the pipeline then narrates anonymously.
    """
    import csv
    import re as _re
    from pathlib import Path as _Path

    # A sidecar written by select_clip.py survives the copy to a generic name.
    sidecar = _Path(clip_path).with_suffix(".source.json")
    origin = ""
    if sidecar.exists():
        import json

        origin = json.loads(sidecar.read_text()).get("bard_path", "")

    stem = _Path(clip_path).stem
    parent = _Path(clip_path).parent.name
    game = None
    for candidate in (origin, stem, parent):
        match = _re.search(r"(\d{10})", candidate)
        if match:
            game = match.group(1)
            break
    if game is None or not _Path(metadata_csv).exists():
        return []

    plays: list[PlayByPlayEvent] = []
    with open(metadata_csv) as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            play = parse_nba_url(row["urls"])
            if play and play.game_id == game and play.player and play.action:
                plays.append(play)
    return sorted(plays, key=lambda p: p.event_id)
