from courtvision.commentary import (
    Commentary,
    CommentaryLine,
    format_timestamp,
    generate_commentary,
    validate_commentary,
)
from courtvision.config import Config
from courtvision.types import Event


def event(time_s=0.0, track_id=7, team="A", action="shot", change=False) -> Event:
    return Event(time_s, track_id, team, action, change)


class FakeNarrator:
    """Returns canned responses in order, so the graph is testable without an API key."""

    def __init__(self, *responses: Commentary) -> None:
        self._responses = list(responses)
        self.calls = 0

    def narrate(self, events) -> Commentary:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def test_format_timestamp():
    assert format_timestamp(0.0) == "0:00"
    assert format_timestamp(14.2) == "0:14"
    assert format_timestamp(75.0) == "1:15"
    assert format_timestamp(605.0) == "10:05"


def test_validate_accepts_a_faithful_line():
    events = [event(time_s=14.0, track_id=7, team="A")]
    lines = [CommentaryLine(time_s=14.0, text="Player 7 (Team A) rises for a jump shot.")]
    assert validate_commentary(events, lines) == []


def test_validate_rejects_a_fabricated_player():
    events = [event(track_id=7)]
    lines = [CommentaryLine(time_s=0.0, text="Player 3 drives to the basket.")]
    errors = validate_commentary(events, lines)
    assert len(errors) == 1
    assert "Player 3" in errors[0]


def test_validate_rejects_a_fabricated_team():
    events = [event(team="A")]
    lines = [CommentaryLine(time_s=0.0, text="Player 7 of Team B pulls up.")]
    assert validate_commentary(events, lines) != []


def test_validate_rejects_a_line_count_mismatch():
    events = [event(), event(time_s=1.0)]
    lines = [CommentaryLine(time_s=0.0, text="Player 7 shoots.")]
    assert any("count" in e.lower() for e in validate_commentary(events, lines))


def test_validate_rejects_a_drifting_timestamp():
    events = [event(time_s=14.0)]
    lines = [CommentaryLine(time_s=40.0, text="Player 7 shoots.")]
    assert any("timestamp" in e.lower() for e in validate_commentary(events, lines))


def test_validate_allows_a_line_naming_no_player():
    events = [Event(0.0, None, None, "other", False)]
    lines = [CommentaryLine(time_s=0.0, text="The ball is loose under the basket.")]
    assert validate_commentary(events, lines) == []


def test_generate_returns_lines_when_the_first_attempt_is_clean():
    events = [event()]
    good = Commentary(lines=[CommentaryLine(time_s=0.0, text="Player 7 (Team A) shoots.")])
    narrator = FakeNarrator(good)
    lines, errors = generate_commentary(events, narrator, Config())
    assert errors == []
    assert len(lines) == 1
    assert narrator.calls == 1


def test_generate_retries_once_on_a_fabrication_then_succeeds():
    events = [event()]
    bad = Commentary(lines=[CommentaryLine(time_s=0.0, text="Player 3 shoots.")])
    good = Commentary(lines=[CommentaryLine(time_s=0.0, text="Player 7 (Team A) shoots.")])
    narrator = FakeNarrator(bad, good)
    lines, errors = generate_commentary(events, narrator, Config())
    assert narrator.calls == 2
    assert errors == []
    assert "Player 7" in lines[0].text


def test_generate_gives_up_after_max_attempts_and_reports_errors():
    events = [event()]
    bad = Commentary(lines=[CommentaryLine(time_s=0.0, text="Player 3 shoots.")])
    narrator = FakeNarrator(bad)
    config = Config()
    lines, errors = generate_commentary(events, narrator, config)
    assert narrator.calls == config.llm_max_attempts
    assert errors != []


def test_generate_on_empty_events_makes_no_call():
    narrator = FakeNarrator(Commentary(lines=[]))
    lines, errors = generate_commentary([], narrator, Config())
    assert lines == []
    assert errors == []
    assert narrator.calls == 0


def test_validate_rejects_a_real_name_when_the_event_has_none():
    """Naming a real person on a guess is the worst failure mode here."""
    events = [event(track_id=7, team="A")]
    lines = [CommentaryLine(time_s=0.0, text="Cunningham rises for the jumper.")]
    assert validate_commentary(events, lines) != []


def test_validate_accepts_a_real_name_the_event_carries():
    named = Event(0.0, 7, "A", "shot", False, player_name="Cunningham")
    lines = [CommentaryLine(time_s=0.0, text="Cunningham rises for the jumper.")]
    assert validate_commentary([named], lines) == []


def test_validate_still_accepts_anonymous_commentary():
    events = [event(track_id=7, team="A")]
    lines = [CommentaryLine(time_s=0.0, text="Player 7 of Team A rises for the shot.")]
    assert validate_commentary(events, lines) == []


def test_validate_rejects_substituting_a_different_real_player():
    """Naming the wrong real person is as bad as inventing one."""
    named = Event(0.0, 7, "A", "shot", False, player_name="Cunningham")
    lines = [CommentaryLine(time_s=0.0, text="Jokic rises for the jumper.")]
    errors = validate_commentary([named], lines)
    assert errors and "Jokic" in errors[0]


def test_validate_accepts_a_partial_match_of_a_two_part_name():
    named = Event(0.0, 7, "A", "rebound", False, player_name="K. Johnson")
    lines = [CommentaryLine(time_s=0.0, text="Johnson hauls in the rebound.")]
    assert validate_commentary([named], lines) == []


def test_validate_accepts_an_accented_name_the_event_carries():
    named = Event(0.0, 7, "B", "pass", False, player_name="Jokić")
    lines = [CommentaryLine(time_s=0.0, text="Jokić delivers a pass for Team B.")]
    assert validate_commentary([named], lines) == []


def test_validate_rejects_an_accented_name_the_event_does_not_carry():
    named = Event(0.0, 7, "B", "pass", False, player_name="Randle")
    lines = [CommentaryLine(time_s=0.0, text="Jokić delivers a pass for Team B.")]
    assert validate_commentary([named], lines) != []
