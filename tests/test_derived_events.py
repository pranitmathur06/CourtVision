"""Steals and rebounds come from possession structure, not from pixels.

Measured lift over chance, separating each action from ordinary play:

    rebound  +0.042 (crop)  +0.113 (full frame)
    steal    +0.054         +0.037
    block    -0.170         -0.125

Block is below chance — the classifier does worse than always guessing. Steal is
barely separable either way, which is why it emitted 33.8x the official count
however it was trained. These are possession events, so derive them.
"""

from courtvision.derived_events import MIN_POSSESSION_S, derive, possessions
from courtvision.types import Event


def _shot(time_s: float) -> Event:
    return Event(time_s=time_s, track_id=1, team="A", action="shot",
                 possession_change=False)


def _timeline(pattern):
    """pattern: list of (holder, frames). Returns (times, holders)."""
    times, holders, t = [], [], 0.0
    for holder, count in pattern:
        for _ in range(count):
            times.append(t)
            holders.append(holder)
            t += 0.1
    return times, holders


def test_possession_change_after_a_shot_is_a_rebound():
    times, holders = _timeline([(1, 20), (2, 20)])
    events = derive(times, holders, {1: "A", 2: "B"}, [_shot(1.5)])
    assert [e.action for e in events] == ["rebound"]
    assert events[0].team == "B"


def test_possession_change_with_no_shot_behind_it_is_a_steal():
    times, holders = _timeline([(1, 20), (2, 20)])
    events = derive(times, holders, {1: "A", 2: "B"}, [])
    assert [e.action for e in events] == ["steal"]


def test_a_shot_long_before_the_change_does_not_make_it_a_rebound():
    """Three seconds is the outcome window; a shot ten seconds earlier is over."""
    times, holders = _timeline([(1, 20), (2, 20)])
    events = derive(times, holders, {1: "A", 2: "B"}, [_shot(-10.0)])
    assert [e.action for e in events] == ["steal"]


def test_the_ball_staying_with_one_team_is_not_an_event():
    """Track ids churn constantly; a new id on the same team is not a turnover."""
    times, holders = _timeline([(1, 20), (7, 20), (9, 20)])
    events = derive(times, holders, {1: "A", 7: "A", 9: "A"}, [])
    assert events == []


def test_a_flicker_too_short_to_be_a_possession_is_ignored():
    times, holders = _timeline([(1, 20), (2, 2), (1, 20)])
    spans = possessions(times, holders, {1: "A", 2: "B"})
    assert [s.team for s in spans] == ["A", "A"], \
        f"a {MIN_POSSESSION_S}s floor must drop the two-frame flicker"


def test_frames_with_no_holder_do_not_break_a_possession():
    times, holders = _timeline([(1, 10), (None, 10), (1, 10)])
    spans = possessions(times, holders, {1: "A"})
    assert len(spans) == 1, "the ball in flight is not a change of possession"
