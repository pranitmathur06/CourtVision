import pytest

from courtvision.config import Config
from courtvision.possession import (
    normalized_distance,
    possession_timeline,
    raw_holder,
    smooth_holders,
)
from courtvision.types import BALL, PLAYER, Box, Frame, Track


def player(track_id: int, x: float, y: float, height: float = 80.0) -> Track:
    return Track(track_id, Box(x, y, x + 30.0, y + height), PLAYER, 0.9)


def ball(x: float, y: float) -> Track:
    return Track(-1, Box(x - 5, y - 5, x + 5, y + 5), BALL, 0.9)


def test_normalized_distance_is_scale_invariant():
    """A player twice as far away has a half-size box; the ratio must not change."""
    near = normalized_distance(player(1, 0, 0, height=80.0), ball(15.0, 120.0))
    far = normalized_distance(player(1, 0, 0, height=40.0), ball(15.0, 60.0))
    assert near == pytest.approx(far)


def test_normalized_distance_is_zero_at_the_player_centre():
    subject = player(1, 0, 0)
    center_x, center_y = subject.box.center
    assert normalized_distance(subject, ball(center_x, center_y)) == pytest.approx(0.0)


def test_raw_holder_picks_the_nearest_player():
    near, far = player(1, 100, 100), player(2, 300, 100)
    center_x, center_y = near.box.center
    frame = Frame(0, 0.0, (near, far, ball(center_x, center_y)))
    assert raw_holder(frame, max_norm_dist=0.8) == 1


def test_raw_holder_returns_none_when_ball_is_far_from_everyone():
    frame = Frame(0, 0.0, (player(1, 100, 100), ball(600.0, 20.0)))
    assert raw_holder(frame, max_norm_dist=0.8) is None


def test_raw_holder_returns_none_without_a_ball():
    frame = Frame(0, 0.0, (player(1, 100, 100),))
    assert raw_holder(frame, max_norm_dist=0.8) is None


def test_raw_holder_returns_none_without_players():
    frame = Frame(0, 0.0, (ball(100.0, 100.0),))
    assert raw_holder(frame, max_norm_dist=0.8) is None


def test_smooth_requires_sustained_possession_before_switching():
    # Player 2 appears for a single frame — noise, not a real change of possession.
    raw = [1, 1, 1, 2, 1, 1, 1]
    assert smooth_holders(raw, min_hold_frames=3, max_gap_frames=5) == [1] * 7


def test_smooth_switches_after_sustained_possession():
    raw = [1, 1, 1, 2, 2, 2, 2]
    assert smooth_holders(raw, min_hold_frames=3, max_gap_frames=5) == [1, 1, 1, 2, 2, 2, 2]


def test_smooth_bridges_a_short_gap():
    # Ball briefly occluded; the holder should carry through.
    # Player 1 needs min_hold_frames of support first to become the holder at all.
    raw = [1, 1, 1, None, None, 1]
    assert smooth_holders(raw, min_hold_frames=3, max_gap_frames=5) == [1, 1, 1, 1, 1, 1]


def test_smooth_drops_the_holder_after_a_long_gap():
    raw = [1, 1, 1, None, None, None, None]
    result = smooth_holders(raw, min_hold_frames=3, max_gap_frames=2)
    assert result == [1, 1, 1, 1, 1, None, None]


def test_smooth_needs_support_to_establish_the_first_holder():
    raw = [1, None, None, None, None, None]
    result = smooth_holders(raw, min_hold_frames=3, max_gap_frames=0)
    assert result[0] is None


def test_smooth_handles_empty_input():
    assert smooth_holders([], min_hold_frames=3, max_gap_frames=5) == []


def test_possession_timeline_matches_synthetic_truth(synthetic):
    """Ground truth with in-flight gaps bridged by the previous holder."""
    from tests.fixtures.synthetic import StubDetector

    stub = StubDetector(synthetic)
    frames = []
    for index in range(synthetic.n_frames):
        detections = stub.detect_at(index)
        tracks = []
        player_index = 0
        for det in detections:
            if det.label == PLAYER:
                tracks.append(Track(player_index, det.box, det.label, det.conf))
                player_index += 1
            else:
                tracks.append(Track(-1, det.box, det.label, det.conf))
        frames.append(Frame(index, index / synthetic.fps, tuple(tracks)))

    timeline = possession_timeline(frames, Config())

    # Gaps shorter than max_gap_frames are bridged, so compare against truth with
    # each None replaced by the preceding holder.
    expected: list[int | None] = []
    previous: int | None = None
    for holder in synthetic.holder_by_frame:
        if holder is None:
            expected.append(previous)
        else:
            expected.append(holder)
            previous = holder

    matches = sum(1 for got, want in zip(timeline, expected) if got == want)
    accuracy = matches / len(expected)
    assert accuracy >= 0.9, f"possession accuracy {accuracy:.2f} on synthetic truth"


def handler(track_id: int, x: float, y: float, height: float = 80.0) -> Track:
    from courtvision.types import HANDLER
    return Track(track_id, Box(x, y, x + 30.0, y + height), HANDLER, 0.9)


def test_raw_holder_prefers_the_learned_handler_over_proximity():
    """The handler wins even when another player is nearer the ball.

    Proximity cannot resolve a crowd (a defender is within 0.21 body-heights of
    the handler in the median frame); the learned handler uses appearance cues
    geometry cannot see. See raw_holder's docstring for the data progression.
    """
    near_player = player(1, 100, 100)
    marked = handler(2, 400, 100)
    cx, cy = near_player.box.center
    frame = Frame(0, 0.0, (near_player, marked, ball(cx, cy)))
    assert raw_holder(frame, max_norm_dist=0.8) == 2


def test_raw_holder_falls_back_to_proximity_without_a_handler():
    near, far = player(1, 100, 100), player(2, 400, 100)
    cx, cy = near.box.center
    frame = Frame(0, 0.0, (near, far, ball(cx, cy)))
    assert raw_holder(frame, max_norm_dist=0.8) == 1


def test_handler_counts_as_a_player_for_geometry():
    marked = handler(5, 100, 100)
    frame = Frame(0, 0.0, (marked,))
    assert frame.players() == (marked,)
    assert frame.handler() is marked


def test_visible_unclaimed_ball_releases_possession():
    """A ball seen far from everyone is evidence, not a gap to bridge."""
    # Track 1 holds, then the ball goes loose and nobody reclaims it.
    raw = [1, 1, 1, None, None, None]
    seen = [True] * 6
    result = smooth_holders(raw, min_hold_frames=3, max_gap_frames=3, ball_seen=seen)
    assert result == [1, 1, 1, None, None, None]


def test_isolated_unclaimed_frame_is_treated_as_noise():
    """One bad ball box must not end a possession the holder plainly keeps.

    In V6 the ball was measured 1.41 body-heights from track 6 in a single frame,
    between two frames that put it at 0.13. Releasing on that frame lost a
    possession the tracker had right.
    """
    raw = [6, 6, 6, None, 6, 6]
    seen = [True] * 6
    result = smooth_holders(raw, min_hold_frames=3, max_gap_frames=3, ball_seen=seen)
    assert result == [6, 6, 6, 6, 6, 6]


def test_unseen_ball_still_bridges_the_gap():
    """Absent information is not evidence; the old bridging behaviour stands."""
    raw = [1, 1, 1, None, None, 1]
    seen = [True, True, True, False, False, True]
    result = smooth_holders(raw, min_hold_frames=3, max_gap_frames=3, ball_seen=seen)
    assert result == [1, 1, 1, 1, 1, 1]


def test_ball_seen_flag_is_optional():
    """Callers that cannot say whether the ball was seen keep the old behaviour."""
    raw = [1, 1, 1, None, None, None]
    assert smooth_holders(raw, min_hold_frames=3, max_gap_frames=3) == [1] * 6
