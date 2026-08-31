"""The SportVU adapter, pinned against hand-built moments.

Every trap here cost real time when it was discovered on live data, so each has
a test: overlapping events, index-vs-time resampling, and the fact that player z
is not a measurement.
"""

import json
import math

import pytest

from courtvision.tracking_data import (BALL_TRACK_ID, PLAYER_HEIGHT_FT,
                                       elapsed_seconds, load_game)
from courtvision.types import BALL, PLAYER


def _moment(quarter, stamp, clock, ball_xy=(47.0, 25.0), ball_z=4.0,
            players=((1610612761, 100, 20.0, 20.0), (1610612766, 200, 60.0, 30.0))):
    positions = [[-1, -1, ball_xy[0], ball_xy[1], ball_z]]
    for team, pid, x, y in players:
        positions.append([team, pid, x, y, 0.0])
    return [quarter, stamp, clock, 12.0, None, positions]


def _game(events, tmp_path, name="0021500001.json"):
    payload = {"gameid": "0021500001", "gamedate": "2016-01-01", "events": events}
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _event(moments):
    roster_home = [{"firstname": "Home", "lastname": "One", "playerid": 100,
                    "jersey": "1", "position": "G"}]
    roster_away = [{"firstname": "Away", "lastname": "Two", "playerid": 200,
                    "jersey": "2", "position": "F"}]
    return {"eventId": "1",
            "home": {"teamid": 1610612761, "abbreviation": "TOR",
                     "players": roster_home},
            "visitor": {"teamid": 1610612766, "abbreviation": "CHA",
                        "players": roster_away},
            "moments": moments}


def test_overlapping_events_are_deduplicated_by_timestamp(tmp_path):
    """SportVU events are windows around plays, so the same instant repeats.

    On a real game this removed 62% of raw moments — 211,445 down to 79,963.
    Without it every rate computed from the feed is inflated.
    """
    shared = [_moment(1, 1000 + i * 40, 720 - i * 0.04) for i in range(10)]
    path = _game([_event(shared), _event(shared)], tmp_path)
    game = load_game(path, target_hz=None)
    assert len(game.frames) == 10, "the same timestamps must collapse to one frame each"


def test_resampling_is_by_game_time_not_index(tmp_path):
    """The clock stops, so index spacing is not time spacing.

    Possession smoothing counts FRAMES and assumes uniform spacing, so a stride
    over indices silently changes what min_hold_frames means.
    """
    moments = [_moment(1, 1000 + i * 40, 720 - i * 0.04) for i in range(250)]
    path = _game([_event(moments)], tmp_path)
    game = load_game(path, target_hz=10.0)
    times = [f.time_s for f in game.frames]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert gaps, "expected several frames"
    mean_gap = sum(gaps) / len(gaps)
    assert 0.09 <= mean_gap <= 0.11, f"expected ~10 Hz, got {1 / mean_gap:.1f} Hz"


def test_ball_is_its_own_track_id_and_players_keep_theirs(tmp_path):
    path = _game([_event([_moment(1, 1000, 720.0)])], tmp_path)
    game = load_game(path, target_hz=None)
    frame = game.frames[0]
    ball = frame.ball()
    assert ball is not None and ball.track_id == BALL_TRACK_ID
    assert ball.label == BALL
    assert sorted(t.track_id for t in frame.players()) == [100, 200]
    assert all(t.label == PLAYER for t in frame.players())


def test_player_boxes_carry_height_in_the_same_units_as_position(tmp_path):
    """possession.normalized_distance divides by box height.

    A zero or pixel-scaled height makes every possession threshold meaningless
    against court feet.
    """
    path = _game([_event([_moment(1, 1000, 720.0)])], tmp_path)
    game = load_game(path, target_hz=None)
    player = game.frames[0].players()[0]
    assert math.isclose(player.box.height, PLAYER_HEIGHT_FT)
    assert player.box.width > 0


def test_teams_collapse_to_exactly_two_labels(tmp_path):
    path = _game([_event([_moment(1, 1000, 720.0)])], tmp_path)
    game = load_game(path, target_hz=None)
    assert set(game.teams.values()) == {"A", "B"}
    assert game.teams[100] != game.teams[200]


def test_names_come_from_the_roster(tmp_path):
    path = _game([_event([_moment(1, 1000, 720.0)])], tmp_path)
    game = load_game(path, target_hz=None)
    assert game.names[100] == "Home One"
    assert game.names[200] == "Away Two"


def test_a_malformed_moment_is_skipped_not_fatal(tmp_path):
    good = _moment(1, 1000, 720.0)
    path = _game([_event([[1, 1040, 719.0], None, good])], tmp_path)
    game = load_game(path, target_hz=None)
    assert len(game.frames) == 1


def test_elapsed_seconds_orders_periods_and_handles_overtime():
    assert elapsed_seconds(1, 720.0) == 0.0
    assert elapsed_seconds(1, 0.0) == 720.0
    assert elapsed_seconds(2, 720.0) == 720.0
    # Overtime is five minutes, not twelve; (period-1)*720 would be wrong.
    assert elapsed_seconds(5, 300.0) == 4 * 720.0
    assert elapsed_seconds(6, 300.0) == 4 * 720.0 + 300.0


def test_a_game_with_no_events_is_rejected(tmp_path):
    path = _game([], tmp_path)
    with pytest.raises(ValueError):
        load_game(path)
