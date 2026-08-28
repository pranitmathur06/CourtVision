from courtvision.types import ACTIONS, BALL, PLAYER, Box, Frame, Track


def test_box_geometry():
    box = Box(10.0, 20.0, 30.0, 60.0)
    assert box.width == 20.0
    assert box.height == 40.0
    assert box.center == (20.0, 40.0)


def test_frame_players_excludes_ball():
    player = Track(1, Box(0, 0, 10, 20), PLAYER, 0.9)
    ball = Track(-1, Box(5, 5, 7, 7), BALL, 0.8)
    frame = Frame(0, 0.0, (player, ball))
    assert frame.players() == (player,)


def test_frame_ball_returns_highest_confidence_ball():
    low = Track(-1, Box(0, 0, 2, 2), BALL, 0.3)
    high = Track(-1, Box(9, 9, 11, 11), BALL, 0.7)
    frame = Frame(0, 0.0, (low, high))
    assert frame.ball() is high


def test_frame_ball_returns_none_when_absent():
    frame = Frame(0, 0.0, (Track(1, Box(0, 0, 10, 20), PLAYER, 0.9),))
    assert frame.ball() is None


def test_actions_are_the_five_spec_labels():
    assert ACTIONS == ("dribble", "pass", "shot", "rebound", "other")
