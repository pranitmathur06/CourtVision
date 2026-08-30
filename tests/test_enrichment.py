"""v3 — naming real players from official play-by-play."""

from courtvision.enrichment import (
    PlayByPlayEvent,
    align,
    extract_action,
    extract_player,
    parse_nba_url,
)
from courtvision.types import Event

BARD_URL = (
    "https://www.nba.com/stats/events/?CFID=&CFPARAMS=&GameEventID=8"
    "&GameID=0022400861&Season=2024-25&flag=1"
    "&title=K.%20Johnson%20REBOUND%20(Off:0%20Def:1)"
)


def test_parses_a_bard_url_into_a_play():
    play = parse_nba_url(BARD_URL)
    assert play is not None
    assert play.game_id == "0022400861"
    assert play.event_id == 8
    assert play.description == "K. Johnson REBOUND (Off:0 Def:1)"
    assert play.player == "K. Johnson"
    assert play.action == "rebound"


def test_returns_none_for_a_url_without_the_needed_fields():
    assert parse_nba_url("https://www.nba.com/stats/events/?flag=1") is None


def test_extracts_players_from_real_descriptions():
    assert extract_player("Cunningham REBOUND (Off:0 Def:1)") == "Cunningham"
    assert extract_player("Duren 1' Running Layup (2 PTS)") == "Duren"
    assert extract_player("K. Johnson REBOUND (Off:0 Def:1)") == "K. Johnson"


def test_extracts_actions_from_real_descriptions():
    assert extract_action("Cunningham REBOUND (Off:0 Def:1)") == "rebound"
    assert extract_action("Williams 2' Running Dunk (2 PTS)") == "shot"
    assert extract_action("Duren Free Throw 1 of 2 (3 PTS)") == "shot"
    assert extract_action("Someone STEAL") == "steal"
    assert extract_action("no recognisable verb here") is None


def play(player, action, event_id=1):
    return PlayByPlayEvent("0022400861", event_id, f"{player} {action}", player, action)


def event(action, track_id=7, time_s=0.0):
    return Event(time_s, track_id, "A", action, False)


def test_aligns_a_matching_action_and_attaches_the_name():
    out = align([event("rebound")], [play("Cunningham", "rebound")])
    assert out[0].player_name == "Cunningham"
    assert out[0].track_id == 7  # the track id is kept, not replaced


def test_refuses_to_name_when_the_action_disagrees():
    """A name on the wrong play is a false statement about a real person."""
    out = align([event("dribble")], [play("Cunningham", "rebound")])
    assert out[0].player_name is None


def test_consumes_each_play_at_most_once():
    events = [event("rebound"), event("rebound")]
    plays = [play("Cunningham", "rebound", 1)]
    out = align(events, plays)
    assert out[0].player_name == "Cunningham"
    assert out[1].player_name is None


def test_matches_in_order_across_several_plays():
    events = [event("rebound"), event("shot"), event("steal")]
    plays = [play("A. One", "rebound", 1), play("B. Two", "shot", 2),
             play("C. Three", "steal", 3)]
    out = align(events, plays)
    assert [e.player_name for e in out] == ["A. One", "B. Two", "C. Three"]


def test_skips_plays_that_name_nobody():
    anonymous = PlayByPlayEvent("g", 1, "REBOUND", None, "rebound")
    out = align([event("rebound")], [anonymous, play("Cunningham", "rebound", 2)])
    assert out[0].player_name == "Cunningham"


def test_events_survive_with_no_plays_at_all():
    out = align([event("rebound"), event("shot")], [])
    assert all(e.player_name is None for e in out)
    assert len(out) == 2


def test_recovers_the_game_id_from_a_bard_clip_path(tmp_path):
    """BARD encodes the GameID in the path, so no scoreboard OCR is needed."""
    from courtvision.enrichment import plays_for_clip

    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "urls;actions;numerosity\n"
        + BARD_URL.replace(";", "%3B") + ";[];1\n"
    )
    plays = plays_for_clip("data/raw/bkn-vs-det-0022400861/12.mp4", str(csv_path))
    assert [p.player for p in plays] == ["K. Johnson"]


def test_returns_no_plays_for_an_unknown_clip(tmp_path):
    from courtvision.enrichment import plays_for_clip

    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("urls;actions;numerosity\n" + BARD_URL + ";[];1\n")
    # A clip with no recoverable game id must narrate anonymously, not guess.
    assert plays_for_clip("some/random/clip.mp4", str(csv_path)) == []


def test_returns_no_plays_when_metadata_is_missing():
    from courtvision.enrichment import plays_for_clip

    assert plays_for_clip("bkn-vs-det-0022400861/1.mp4", "does/not/exist.csv") == []


def test_accented_names_survive_intact():
    """NBA rosters are full of them; truncating one misnames a real person."""
    assert extract_player("Jokić REBOUND (Off:1 Def:3)") == "Jokić"
    assert extract_player("Dončić 3PT Shot (12 PTS)") == "Dončić"
    assert extract_player("Šengün REBOUND (Off:0 Def:5)") == "Šengün"
    assert extract_player("Porziņģis BLOCK (1 BLK)") == "Porziņģis"


def _ev(time_s, track_id, action, name=None, team="A"):
    from courtvision.types import Event
    return Event(time_s, track_id, team, action, False, player_name=name)


def test_one_track_cannot_be_two_people():
    """Track 17 came out as Randle at 4.08s and McDaniels at 4.88s on V9."""
    from courtvision.enrichment import enforce_identity_consistency

    events = [
        _ev(4.08, 17, "rebound", "Randle"),
        _ev(4.48, 17, "rebound", "Randle"),
        _ev(4.88, 17, "steal", "McDaniels"),
    ]
    out = enforce_identity_consistency(events)
    assert {e.player_name for e in out} == {"Randle"}, [e.player_name for e in out]


def test_a_tie_drops_the_name_rather_than_guessing():
    """Anonymous beats confidently wrong about a real person."""
    from courtvision.enrichment import enforce_identity_consistency

    events = [_ev(1.0, 5, "rebound", "Randle"), _ev(2.0, 5, "steal", "McDaniels")]
    out = enforce_identity_consistency(events)
    assert all(e.player_name is None for e in out)


def test_one_name_cannot_be_on_two_teams():
    """McDaniels was team A as track 17 and team B as track 18 simultaneously."""
    from courtvision.enrichment import enforce_identity_consistency

    events = [
        _ev(4.88, 17, "steal", "McDaniels", team="A"),
        _ev(5.28, 17, "steal", "McDaniels", team="A"),
        _ev(7.28, 18, "rebound", "McDaniels", team="B"),
    ]
    out = enforce_identity_consistency(events)
    named = {(e.track_id, e.player_name) for e in out if e.player_name}
    assert named == {(17, "McDaniels")}, named


def test_consistent_names_are_left_alone():
    from courtvision.enrichment import enforce_identity_consistency

    events = [
        _ev(0.1, 12, "rebound", "Randle"),
        _ev(1.1, 19, "rebound", "Murray"),
        _ev(2.1, 12, "shot", "Randle"),
    ]
    out = enforce_identity_consistency(events)
    assert [(e.track_id, e.player_name) for e in out] == [
        (12, "Randle"), (19, "Murray"), (12, "Randle")]


def test_events_without_a_track_are_untouched():
    from courtvision.enrichment import enforce_identity_consistency

    events = [_ev(3.28, None, "other", None)]
    assert enforce_identity_consistency(events)[0].player_name is None
