"""Mapping the official NBA feed onto this project's action vocabulary.

Fixtures are shaped exactly like real entries from game 0022400861, so these
run without a network.
"""

import pytest

from courtvision.nba_feed import (
    action_for,
    fetch_game_plays,
    parse_iso_clock,
    plays_from_actions,
)


def entry(**kwargs):
    base = {"actionNumber": 1, "clock": "PT11M44.00S", "period": 1,
            "playerNameI": "K. Johnson", "actionType": "Rebound",
            "subType": "", "description": "K. Johnson REBOUND (Off:0 Def:1)"}
    base.update(kwargs)
    return base


@pytest.mark.parametrize("text, seconds", [
    ("PT11M44.00S", 704), ("PT12M00.00S", 720), ("PT00M00.30S", 0),
    ("PT00M07.90S", 7),
])
def test_parse_iso_clock(text, seconds):
    assert parse_iso_clock(text) == seconds


@pytest.mark.parametrize("text", ["", "11:44", "PTXM00S", "garbage"])
def test_unparseable_clock_is_none_not_zero(text):
    """Zero would read as 'end of period' and mis-join every nearby play."""
    assert parse_iso_clock(text) is None


def test_steals_and_blocks_have_no_action_type_and_come_from_the_description():
    """The quirk this module exists for.

    In this feed a STEAL and a BLOCK carry an EMPTY actionType. On game
    0022400861 the 40 entries with no action type were exactly the 17 steals
    and 23 blocks — two of the seven classes this project handles. Keying on
    actionType alone loses both silently.
    """
    steal = entry(actionType="", description="Williams STEAL (1 STL)")
    block = entry(actionType="", description="Claxton BLOCK (1 BLK)")
    assert action_for(steal) == "steal"
    assert action_for(block) == "block"


@pytest.mark.parametrize("action_type, expected", [
    ("Made Shot", "shot"), ("Missed Shot", "shot"), ("Free Throw", "shot"),
    ("Rebound", "rebound"), ("Turnover", "other"), ("Foul", "other"),
])
def test_action_type_mapping(action_type, expected):
    assert action_for(entry(actionType=action_type)) == expected


@pytest.mark.parametrize("action_type", [
    "period", "Substitution", "Timeout", "Jump Ball", "Instant Replay", "Ejection",
])
def test_bookkeeping_entries_are_skipped(action_type):
    assert action_for(entry(actionType=action_type)) is None


def test_an_empty_action_type_that_is_neither_is_skipped():
    assert action_for(entry(actionType="", description="something else")) is None


def test_plays_carry_period_and_clock_for_alignment():
    plays = plays_from_actions("0022400861", [entry()])
    assert len(plays) == 1
    play = plays[0]
    assert play.period == 1 and play.clock_seconds == 704
    assert play.player == "K. Johnson" and play.action == "rebound"
    assert play.game_id == "0022400861"


def test_a_missing_player_becomes_none_not_empty_string():
    plays = plays_from_actions("g", [entry(playerNameI="")])
    assert plays[0].player is None


def test_the_converted_plays_feed_the_clock_window_selector():
    """End to end: feed entries -> plays -> the window an aligner would ask for."""
    from courtvision.enrichment import plays_in_window

    actions = [
        entry(actionNumber=7, clock="PT11M44.00S", actionType="Missed Shot"),
        entry(actionNumber=8, clock="PT11M42.00S", actionType="Rebound"),
        entry(actionNumber=9, clock="PT10M00.00S", actionType="Rebound"),
    ]
    plays = plays_from_actions("g", actions)
    got = plays_in_window(plays, period=1, from_clock_s=704, to_clock_s=700)
    assert [p.event_id for p in got] == [7, 8]


def test_fetch_is_optional_and_lazy():
    """nba_api is an optional dependency; importing this module must not need it."""
    import courtvision.nba_feed as feed
    assert callable(feed.fetch_game_plays)
