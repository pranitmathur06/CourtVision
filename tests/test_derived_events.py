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
    times, holders = _timeline([(1, 100), (2, 100)])
    events = derive(times, holders, {1: "A", 2: "B"}, [_shot(9.0)])
    assert [e.action for e in events] == ["rebound"]
    assert events[0].team == "B"


def test_possession_change_with_no_shot_behind_it_is_a_steal():
    times, holders = _timeline([(1, 100), (2, 100)])
    events = derive(times, holders, {1: "A", 2: "B"}, [])
    assert [e.action for e in events] == ["steal"]


def test_a_shot_long_before_the_change_does_not_make_it_a_rebound():
    """Three seconds is the outcome window; a shot ten seconds earlier is over."""
    times, holders = _timeline([(1, 100), (2, 100)])
    events = derive(times, holders, {1: "A", 2: "B"}, [_shot(-10.0)])
    assert [e.action for e in events] == ["steal"]


def test_the_ball_staying_with_one_team_is_not_an_event():
    """Track ids churn constantly; a new id on the same team is not a turnover."""
    times, holders = _timeline([(1, 100), (7, 100), (9, 100)])
    events = derive(times, holders, {1: "A", 7: "A", 9: "A"}, [])
    assert events == []


def test_a_flicker_too_short_to_be_a_possession_is_ignored():
    times, holders = _timeline([(1, 100), (2, 3), (1, 100)])
    spans = possessions(times, holders, {1: "A", 2: "B"})
    assert [s.team for s in spans] == ["A", "A"], \
        f"a {MIN_POSSESSION_S}s floor must drop the two-frame flicker"


def test_frames_with_no_holder_do_not_break_a_possession():
    times, holders = _timeline([(1, 50), (None, 20), (1, 50)])
    spans = possessions(times, holders, {1: "A"})
    assert len(spans) == 1, "the ball in flight is not a change of possession"


def test_the_default_floor_is_long_enough_to_survive_track_churn():
    """Track ids restart at every broadcast cut — 14,640 in one game.

    A sub-second floor let churn read as possession changes and produced 377
    derived events where 101 were real. Swept against BARD's labels, the steal
    ratio falls 20.4x -> 5.3x -> 3.5x -> 1.4x as the floor goes 0.6 -> 3 -> 5
    -> 8 seconds.
    """
    from courtvision.derived_events import MIN_POSSESSION_S

    assert MIN_POSSESSION_S >= 5.0, "a short floor turns tracker noise into events"
    assert MIN_POSSESSION_S <= 10.0, "an NBA possession averages about 14 s"
