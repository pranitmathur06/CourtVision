import json

import cv2
import numpy as np

from courtvision.commentary import CommentaryLine
from courtvision.render import build_log, draw_frame, render_video, write_log
from courtvision.types import BALL, PLAYER, Box, Event, Frame, Track


def make_frame(index: int = 0) -> Frame:
    return Frame(
        index,
        index / 10.0,
        (
            Track(1, Box(10, 10, 40, 90), PLAYER, 0.9),
            Track(2, Box(100, 10, 130, 90), PLAYER, 0.9),
            Track(-1, Box(20, 40, 30, 50), BALL, 0.8),
        ),
    )


def test_draw_frame_returns_a_same_shaped_image():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    drawn = draw_frame(image, make_frame(), {1: "A", 2: "B"}, holder_id=1)
    assert drawn.shape == image.shape
    assert drawn.dtype == np.uint8


def test_draw_frame_does_not_mutate_the_input():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    draw_frame(image, make_frame(), {1: "A", 2: "B"}, holder_id=1)
    assert image.sum() == 0


def test_draw_frame_actually_draws_something():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    drawn = draw_frame(image, make_frame(), {1: "A", 2: "B"}, holder_id=1)
    assert drawn.sum() > 0


def test_draw_frame_tolerates_missing_team_labels():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    drawn = draw_frame(image, make_frame(), {}, holder_id=None)
    assert drawn.shape == image.shape


def test_render_video_writes_every_frame(tmp_path):
    images = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(12)]
    frames = [make_frame(i) for i in range(12)]
    out = tmp_path / "out.mp4"

    written = render_video(images, frames, {1: "A", 2: "B"}, [1] * 12, str(out), fps=10)

    assert written == 12
    assert out.exists() and out.stat().st_size > 0
    capture = cv2.VideoCapture(str(out))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    capture.release()


def test_build_log_pairs_events_with_commentary():
    events = [Event(14.0, 7, "A", "shot", True)]
    lines = [CommentaryLine(time_s=14.0, text="Player 7 (Team A) shoots.")]
    log = build_log(events, lines)

    assert len(log["events"]) == 1
    entry = log["events"][0]
    assert entry["timestamp"] == "0:14"
    assert entry["time_s"] == 14.0
    assert entry["track_id"] == 7
    assert entry["team"] == "A"
    assert entry["action"] == "shot"
    assert entry["possession_change"] is True
    assert entry["commentary"] == "Player 7 (Team A) shoots."


def test_build_log_handles_missing_commentary():
    log = build_log([Event(1.0, 7, "A", "shot", False)], [])
    assert log["events"][0]["commentary"] is None


def test_write_log_produces_valid_json(tmp_path):
    out = tmp_path / "commentary.json"
    write_log(
        [Event(14.0, 7, "A", "shot", True)],
        [CommentaryLine(time_s=14.0, text="Player 7 (Team A) shoots.")],
        str(out),
    )
    parsed = json.loads(out.read_text())
    assert parsed["events"][0]["commentary"] == "Player 7 (Team A) shoots."
