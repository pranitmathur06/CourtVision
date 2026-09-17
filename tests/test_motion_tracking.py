"""The tracker that replaced ByteTrack, and the proof the move changed nothing.

`tests/fixtures/tracker_golden.json` is what `clip_boxes.track()` produced on six
real clips of cached detections BEFORE it moved into the library. Every path --
the batch function, the streaming class underneath it, and PlayerTracker on top
-- has to reproduce it box for box.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from courtvision.motion_tracking import (MotionTracker, TrackerConfig,
                                         deduplicate, interpolate, iou, smooth,
                                         track)

FIXTURE = Path("tests/fixtures/tracker_golden.json")
DETECTIONS = Path("outputs/clip_detections_ecf.json")


def boxes_of(rows, player_conf):
    out = []
    for row in rows:
        on = row.get("on") or []
        people = [b for b in row["d"] if b[0] in ("p", "h")]
        out.append([[float(v) for v in b[2:6]] for n, b in enumerate(people)
                    if b[1] >= player_conf and (n >= len(on) or on[n])])
    return out


@pytest.mark.skipif(not (FIXTURE.exists() and DETECTIONS.exists()),
                    reason="needs the cached detections the fixture was cut from")
def test_the_library_reproduces_what_the_script_produced_before_the_move():
    golden = json.load(open(FIXTURE))
    cache = json.load(open(DETECTIONS))
    for name in golden["clips"]:
        want = golden["tracks"][name]
        got = track(boxes_of(cache["clips"][name], golden["player_conf"]), fps=15.0)
        assert sorted(str(t) for t in got) == sorted(want), name
        for tid, by_frame in got.items():
            assert sorted(str(f) for f in by_frame) == sorted(want[str(tid)])
            for frame, box in by_frame.items():
                for i in range(4):
                    assert box[i] == pytest.approx(want[str(tid)][str(frame)][i],
                                                   abs=1e-4)


def test_the_thresholds_are_seconds_so_a_re_rate_is_not_a_new_algorithm():
    """TRACK_MAX_AGE was 15 FRAMES: one second at 15 Hz, three at 5 Hz.

    Nothing announced that, and a game-scale run sampling at 5 Hz was silently
    running a tracker that held a lost player three times as long.
    """
    config = TrackerConfig(max_age_s=1.0)
    assert config.frames(config.max_age_s, 15.0) == 15
    assert config.frames(config.max_age_s, 5.0) == 5
    assert config.frames(0.01, 5.0) == 1, "never round a threshold down to zero"


def test_a_player_keeps_his_identity_while_he_is_detected():
    per_frame = [[[10 + i, 10, 30 + i, 60]] for i in range(8)]
    tracks = track(per_frame, fps=15.0)
    assert len(tracks) == 1
    assert len(next(iter(tracks.values()))) == 8


def test_a_player_missed_for_a_moment_is_the_same_player_when_he_returns():
    """The gap that used to end a track and start a new person."""
    per_frame = [[[10, 10, 30, 60]], [[12, 10, 32, 60]], [],
                 [[16, 10, 36, 60]], [[18, 10, 38, 60]]]
    assert len(track(per_frame, fps=15.0)) == 1


def test_a_gap_longer_than_the_tracker_will_hold_starts_somebody_new():
    config = TrackerConfig(max_age_s=0.1)          # ~2 frames at 15 Hz
    per_frame = ([[[10, 10, 30, 60]]] + [[]] * 6 + [[[10, 10, 30, 60]]])
    assert len(track(per_frame, fps=15.0, config=config)) == 2


def test_a_camera_cut_ends_every_track():
    """No re-identification, so carrying a track across a cut would be a claim.

    This makes the identity COUNT worse and the identities right, which is why
    counts were never an accuracy measure.
    """
    per_frame = [[[10, 10, 30, 60]]] * 6
    assert len(track(per_frame, fps=15.0)) == 1
    assert len(track(per_frame, fps=15.0, cuts=[3])) == 2


def test_the_streaming_and_batch_paths_are_the_same_code():
    per_frame = [[[10 + i, 10, 30 + i, 60], [100 - i, 20, 120 - i, 70]]
                 for i in range(6)]
    batch = track(per_frame, fps=15.0)
    tracker = MotionTracker(fps=15.0)
    streamed: dict[int, dict[int, list[float]]] = {}
    for frame, boxes in enumerate(per_frame):
        for tid, box in tracker.update(boxes, frame):
            streamed.setdefault(tid, {})[frame] = box
    assert streamed == batch


def test_update_answers_in_the_order_it_was_asked():
    """The caller zips its own detections against the reply."""
    tracker = MotionTracker(fps=15.0)
    boxes = [[10, 10, 30, 60], [100, 10, 120, 60], [200, 10, 220, 60]]
    got = tracker.update(boxes, 0)
    assert len(got) == len(boxes)
    for (_, box), original in zip(got, boxes):
        assert list(box) == original


def test_overlap_smoothing_gap_filling_and_dedup():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    flat = smooth({0: [0, 0, 10, 10], 1: [10, 0, 20, 10], 2: [20, 0, 30, 10]},
                  window=1)
    assert flat[1][0] == pytest.approx(10.0)
    filled = interpolate({0: [0, 0, 10, 10], 4: [40, 0, 50, 10]}, max_gap=10)
    assert sorted(filled) == [0, 1, 2, 3, 4]
    assert filled[2][0] == pytest.approx(20.0)
    assert interpolate({0: [0, 0, 10, 10], 40: [400, 0, 410, 10]}, max_gap=10) \
        .keys() == {0, 40}
    assert deduplicate([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]) == [0, 2]
