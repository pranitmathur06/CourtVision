from courtvision.events import build_events, dominant_holder
from courtvision.types import ActionWindow


def window(start: int, end: int, label: str = "dribble") -> ActionWindow:
    return ActionWindow(start, end, start / 10.0, end / 10.0, label, 0.9)


def test_dominant_holder_picks_the_majority():
    assert dominant_holder([1, 1, 2, 1], 0, 3) == 1


def test_dominant_holder_ignores_none():
    assert dominant_holder([None, 3, 3, None], 0, 3) == 3


def test_dominant_holder_returns_none_when_all_none():
    assert dominant_holder([None, None], 0, 1) is None


def test_dominant_holder_respects_the_window_bounds():
    # Player 9 dominates overall but is outside the requested window.
    assert dominant_holder([1, 1, 9, 9, 9], 0, 1) == 1


def test_build_events_attaches_holder_and_team():
    events = build_events([window(0, 3, "shot")], [5, 5, 5, 5], {5: "A"})
    assert len(events) == 1
    assert events[0].action == "shot"
    assert events[0].track_id == 5
    assert events[0].team == "A"
    assert events[0].time_s == 0.0


def test_build_events_marks_the_first_event_as_no_possession_change():
    events = build_events([window(0, 3)], [5, 5, 5, 5], {5: "A"})
    assert events[0].possession_change is False


def test_build_events_flags_a_change_of_holder():
    windows = [window(0, 1), window(2, 3)]
    events = build_events(windows, [5, 5, 6, 6], {5: "A", 6: "B"})
    assert events[0].possession_change is False
    assert events[1].possession_change is True
    assert events[1].track_id == 6
    assert events[1].team == "B"


def test_build_events_does_not_flag_a_repeated_holder():
    # Different actions so both survive collapsing; the point is that the same
    # player keeping the ball is not a change of possession.
    windows = [window(0, 1, "dribble"), window(2, 3, "shot")]
    events = build_events(windows, [5, 5, 5, 5], {5: "A"})
    assert len(events) == 2
    assert events[1].possession_change is False


def test_build_events_handles_a_window_with_no_holder():
    events = build_events([window(0, 1)], [None, None], {})
    assert events[0].track_id is None
    assert events[0].team is None
    assert events[0].possession_change is False


def test_build_events_handles_a_holder_with_no_team():
    events = build_events([window(0, 1)], [5, 5], {})
    assert events[0].track_id == 5
    assert events[0].team is None


def test_build_events_returns_events_in_time_order():
    windows = [window(4, 5), window(0, 1), window(2, 3)]
    events = build_events(windows, [1] * 6, {1: "A"})
    assert [e.time_s for e in events] == sorted(e.time_s for e in events)


def test_build_events_on_empty_input():
    assert build_events([], [], {}) == []


def test_build_events_collapses_consecutive_duplicate_windows():
    """Overlapping windows describing one action produce ONE event."""
    windows = [window(0, 1, "rebound"), window(2, 3, "rebound"), window(4, 5, "rebound")]
    events = build_events(windows, [5] * 6, {5: "A"})
    assert len(events) == 1
    assert events[0].action == "rebound"
    assert events[0].time_s == 0.0  # reported at the first window's start


def test_build_events_keeps_a_repeated_action_by_a_different_player():
    windows = [window(0, 1, "rebound"), window(2, 3, "rebound")]
    events = build_events(windows, [5, 5, 6, 6], {5: "A", 6: "B"})
    assert len(events) == 2
    assert [e.track_id for e in events] == [5, 6]


def test_build_events_keeps_a_different_action_by_the_same_player():
    windows = [window(0, 1, "dribble"), window(2, 3, "shot")]
    events = build_events(windows, [5] * 4, {5: "A"})
    assert [e.action for e in events] == ["dribble", "shot"]


def test_build_events_reports_an_action_again_after_something_else_intervenes():
    windows = [window(0, 1, "dribble"), window(2, 3, "pass"), window(4, 5, "dribble")]
    events = build_events(windows, [5] * 6, {5: "A"})
    assert [e.action for e in events] == ["dribble", "pass", "dribble"]
