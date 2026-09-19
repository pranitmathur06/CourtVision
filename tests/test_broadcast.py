"""Reading the source broadcast at the moments a clip's detections came from."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.broadcast import SourceReader, clip_starts  # noqa: E402


def test_a_row_with_no_footage_is_not_a_clip(tmp_path):
    """The cutter records the attempt either way, and treating a row with no
    clip as coverage is its own bug."""
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"clips": [
        {"clip": None, "start_s": 10.0},
        {"clip": "a.mp4", "start_s": 20.0},
    ]}))
    assert clip_starts(index) == {"a.mp4": 20.0}


def test_a_bare_list_index_works_too(tmp_path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps([{"clip": "a.mp4", "start_s": 1.5}]))
    assert clip_starts(index) == {"a.mp4": 1.5}


def test_a_video_that_will_not_open_says_so_rather_than_raising(tmp_path):
    reader = SourceReader(tmp_path / "nothing.mp4")
    assert reader.ok is False
    assert list(reader.frames(0.0, {0: "a"}, 2)) == []
    reader.close()


def _a_broadcast():
    from courtvision.games import get

    for key in ("hou", "g7", "g1", "ecf"):
        broadcast = get(key)
        if (ROOT / broadcast.video).exists() and (ROOT / broadcast.clip_index).exists():
            return broadcast
    return None


def test_the_seek_lands_on_the_clip_s_own_first_frame():
    """`round(start_s * fps) + f` is wrong by up to 24 frames. A POS_MSEC seek
    lands where ffmpeg landed when it cut the clip, so the two agree with each
    other and not with the arithmetic -- and getting it wrong scored 0.400
    against a shipped 0.872."""
    import cv2
    import numpy as np

    broadcast = _a_broadcast()
    if broadcast is None:
        pytest.skip("no broadcast video on this machine")
    starts = clip_starts(ROOT / broadcast.clip_index)
    name, start_s = sorted(starts.items())[0]
    clip_path = ROOT / broadcast.clip_dir / name
    if not clip_path.exists():
        pytest.skip("that clip is not on disk")

    clip = cv2.VideoCapture(str(clip_path))
    ok, first = clip.read()
    clip.release()
    assert ok

    reader = SourceReader(ROOT / broadcast.video)
    if not reader.ok:
        pytest.skip("the broadcast would not open")
    try:
        got = dict(reader.frames(start_s, {0: "first"}, 2))
    finally:
        reader.close()
    assert "first" in got

    def thumbnail(image):
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(grey, (80, 45)).astype(np.float32)

    # The clip is a re-encode at a third the width, so they never match
    # exactly; a different MOMENT is many times further away than this.
    distance = float(np.abs(thumbnail(got["first"]) - thumbnail(first)).mean())
    assert distance < 10.0


def test_the_frames_come_back_at_broadcast_resolution():
    broadcast = _a_broadcast()
    if broadcast is None:
        pytest.skip("no broadcast video on this machine")
    starts = clip_starts(ROOT / broadcast.clip_index)
    reader = SourceReader(ROOT / broadcast.video)
    if not reader.ok:
        pytest.skip("the broadcast would not open")
    try:
        got = dict(reader.frames(sorted(starts.values())[0], {0: "a"}, 2))
    finally:
        reader.close()
    assert got["a"].shape[0] >= 720
