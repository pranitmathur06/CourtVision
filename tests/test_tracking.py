from courtvision.tracking import PlayerTracker, count_id_switches
from courtvision.types import BALL, PLAYER, Box, Detection, Frame, Track
from tests.fixtures.synthetic import StubDetector


def test_ball_passes_through_untracked():
    tracker = PlayerTracker()
    tracks = tracker.update([Detection(Box(0, 0, 5, 5), BALL, 0.9)])
    assert len(tracks) == 1
    assert tracks[0].label == BALL
    assert tracks[0].track_id == -1


def test_players_receive_non_negative_ids():
    tracker = PlayerTracker()
    detections = [
        Detection(Box(10, 10, 40, 80), PLAYER, 0.9),
        Detection(Box(200, 10, 230, 80), PLAYER, 0.9),
    ]
    # ByteTrack needs a couple of frames before it promotes tentative tracks.
    for _ in range(5):
        tracks = tracker.update(detections)
    players = [t for t in tracks if t.label == PLAYER]
    assert len(players) == 2
    assert all(t.track_id >= 0 for t in players)
    assert len({t.track_id for t in players}) == 2


def test_ids_are_stable_on_the_synthetic_clip(synthetic):
    """Players move on non-crossing lanes, so a correct tracker never switches IDs."""
    stub = StubDetector(synthetic)
    tracker = PlayerTracker()
    frames = []
    for index in range(synthetic.n_frames):
        tracks = tracker.update(stub.detect_at(index))
        frames.append(Frame(index, index / synthetic.fps, tuple(tracks)))

    switches = count_id_switches(frames, synthetic.player_boxes)
    assert switches == 0, f"expected no ID switches on non-crossing lanes, got {switches}"


def test_count_id_switches_detects_a_swap():
    box_a, box_b = Box(0, 0, 10, 20), Box(100, 0, 110, 20)
    truth = [{0: box_a, 1: box_b}, {0: box_a, 1: box_b}]
    frames = [
        Frame(0, 0.0, (Track(7, box_a, PLAYER, 0.9), Track(8, box_b, PLAYER, 0.9))),
        # Player 0's box is now labelled with the other id — one switch.
        Frame(1, 0.1, (Track(8, box_a, PLAYER, 0.9), Track(7, box_b, PLAYER, 0.9))),
    ]
    assert count_id_switches(frames, truth) == 2


def test_count_id_switches_is_zero_when_stable():
    box_a = Box(0, 0, 10, 20)
    truth = [{0: box_a}, {0: box_a}]
    frames = [
        Frame(0, 0.0, (Track(7, box_a, PLAYER, 0.9),)),
        Frame(1, 0.1, (Track(7, box_a, PLAYER, 0.9),)),
    ]
    assert count_id_switches(frames, truth) == 0
