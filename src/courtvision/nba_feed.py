"""Official NBA play-by-play, mapped onto this project's action vocabulary.

Steps 1 and 2 of wiring a real game (docs/v3-live-data.md). Archived BARD clips
carry the GameID in their filenames, so their join key is free; a real broadcast
needs the official feed fetched and aligned by clock.

`nba_api` is imported lazily and is an optional dependency: everything else here
runs without a network, and the core pipeline should not require one.

The mapping is not a simple lookup, and the reason matters. In this feed a
STEAL and a BLOCK have an EMPTY `actionType` — they are identifiable only from
the description text. On one checked game (0022400861) the 40 entries with no
action type were exactly the 17 steals and 23 blocks. Keying on `actionType`
alone silently loses both classes, which happen to be two of the seven this
project classifies.
"""

from __future__ import annotations

import re

from courtvision.enrichment import PlayByPlayEvent

# actionType -> this project's ACTIONS. Steals and blocks are absent on purpose;
# they carry no actionType and are matched from the description below.
ACTION_TYPES: dict[str, str] = {
    "Made Shot": "shot",
    "Missed Shot": "shot",
    "Free Throw": "shot",
    "Rebound": "rebound",
    "Turnover": "other",
    "Foul": "other",
    "Violation": "other",
}
# Not actions: clock and roster bookkeeping.
IGNORED_TYPES = {"period", "Substitution", "Timeout", "Jump Ball",
                 "Instant Replay", "Ejection"}

_CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")


def parse_iso_clock(text: str) -> int | None:
    """`PT11M44.00S` to seconds remaining. Returns None if unparseable."""
    match = _CLOCK.fullmatch(text.strip())
    if not match:
        return None
    return int(match.group(1)) * 60 + int(float(match.group(2)))


def action_for(entry: dict) -> str | None:
    """This project's action for one feed entry, or None to skip it."""
    action_type = (entry.get("actionType") or "").strip()
    if action_type in IGNORED_TYPES:
        return None
    if action_type in ACTION_TYPES:
        return ACTION_TYPES[action_type]
    if not action_type:
        # Steals and blocks live here, named only in the description.
        description = (entry.get("description") or "").upper()
        if "STEAL" in description:
            return "steal"
        if "BLOCK" in description:
            return "block"
    return None


def plays_from_actions(game_id: str, actions: list[dict]) -> list[PlayByPlayEvent]:
    """Convert raw feed entries into PlayByPlayEvents. No network."""
    plays: list[PlayByPlayEvent] = []
    for entry in actions:
        action = action_for(entry)
        if action is None:
            continue
        clock = parse_iso_clock(entry.get("clock") or "")
        player = (entry.get("playerNameI") or "").strip() or None
        plays.append(PlayByPlayEvent(
            game_id=game_id,
            event_id=int(entry.get("actionNumber") or 0),
            description=entry.get("description") or "",
            player=player,
            action=action,
            period=entry.get("period"),
            clock_seconds=clock,
        ))
    return plays


def fetch_game_plays(game_id: str, timeout: int = 30) -> list[PlayByPlayEvent]:
    """Fetch and convert one game's play-by-play. Needs `nba_api` and a network."""
    from nba_api.stats.endpoints import playbyplayv3

    payload = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=timeout).get_dict()
    actions = payload.get("game", {}).get("actions", [])
    return plays_from_actions(game_id, actions)
