"""The physics bound on a ball track, and the unit errors it is prone to."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "eval_ball_physics", ROOT / "scripts" / "eval_ball_physics.py")
physics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(physics)


def _row(frame, ball=None, rim=None):
    detections = []
    if ball is not None:
        detections.append(["b", 0.9, ball[0] - 5, ball[1] - 5,
                           ball[0] + 5, ball[1] + 5])
    if rim is not None:
        x, y, width = rim
        detections.append(["r", 0.9, x - width / 2, y - 10,
                           x + width / 2, y + 10])
    return {"f": frame, "d": detections}


def test_the_bound_is_forty_miles_an_hour_in_rim_widths():
    # 58.7 ft/s over an 18-inch rim.
    assert physics.MAX_RIM_WIDTHS_PER_S == pytest.approx(39.1, abs=0.1)


def test_a_dribbled_ball_is_physically_possible():
    """Two rim widths in a fifteenth of a second is 45 ft/s -- fast, legal."""
    rows = [_row(0, (500.0, 500.0), (800.0, 300.0, 40.0)),
            _row(2, (540.0, 500.0), (800.0, 300.0, 40.0))]
    got = physics.speeds(rows, "argmax", 30.0)
    assert len(got) == 1
    assert got[0] < physics.MAX_RIM_WIDTHS_PER_S


def test_a_teleport_across_the_frame_is_not():
    rows = [_row(0, (100.0, 500.0), (800.0, 300.0, 40.0)),
            _row(2, (1100.0, 500.0), (800.0, 300.0, 40.0))]
    got = physics.speeds(rows, "argmax", 30.0)
    assert got[0] > physics.MAX_RIM_WIDTHS_PER_S


def test_the_camera_is_subtracted_so_a_pan_is_not_a_fast_ball():
    """The ball is still on the court; the whole frame panned 60 px."""
    rows = [_row(0, (500.0, 500.0), (800.0, 300.0, 40.0)),
            _row(2, (560.0, 500.0), (860.0, 300.0, 40.0))]
    got = physics.speeds(rows, "argmax", 30.0)
    assert got[0] == pytest.approx(0.0, abs=1e-6)


def test_zoom_drops_out_because_the_rim_is_the_ruler():
    """The same real motion filmed at two zoom levels reads the same speed."""
    wide = [_row(0, (500.0, 500.0), (800.0, 300.0, 20.0)),
            _row(2, (520.0, 500.0), (800.0, 300.0, 20.0))]
    tight = [_row(0, (500.0, 500.0), (800.0, 300.0, 40.0)),
             _row(2, (540.0, 500.0), (800.0, 300.0, 40.0))]
    assert physics.speeds(wide, "argmax", 30.0)[0] == pytest.approx(
        physics.speeds(tight, "argmax", 30.0)[0])


def test_a_pan_bigger_than_any_camera_move_is_treated_as_a_cut():
    """The rim cannot cross 800 px between two frames a fifteenth of a second
    apart. That is a different camera, and its apparent ball motion is not a
    ball speed."""
    rows = [_row(0, (500.0, 500.0), (200.0, 300.0, 40.0)),
            _row(2, (500.0, 500.0), (1000.0, 300.0, 40.0))]
    assert physics.speeds(rows, "argmax", 30.0) == []


def test_a_cut_is_dropped_rather_than_counted_as_an_impossible_ball():
    rows = [_row(0, (500.0, 500.0), (200.0, 300.0, 40.0)),
            _row(2, (500.0, 500.0), (1000.0, 300.0, 40.0))]
    assert physics.speeds(rows, "argmax", 30.0) == []


def test_a_step_with_no_rim_is_not_scored_rather_than_guessed():
    rows = [_row(0, (500.0, 500.0), (800.0, 300.0, 40.0)),
            _row(2, (900.0, 500.0), None)]
    assert physics.speeds(rows, "argmax", 30.0) == []


def test_the_gap_is_converted_with_the_VIDEO_fps_not_the_row_rate():
    """`f` is a video frame index. Dividing by the 15 Hz sampling rate instead
    of the 60 fps video made Houston's ball look four times slower than it is,
    and therefore physically possible when it was not."""
    rows = [_row(0, (500.0, 500.0), (800.0, 300.0, 40.0)),
            _row(4, (700.0, 500.0), (800.0, 300.0, 40.0))]
    fast = physics.speeds(rows, "argmax", 59.94)[0]
    slow = physics.speeds(rows, "argmax", 14.985)[0]
    assert fast == pytest.approx(slow * 4.0, rel=1e-6)


def test_a_track_frozen_on_one_spot_passes_and_that_is_why_it_is_not_accuracy():
    """Stated as a test because the docstring's warning is easy to skip: a
    selector locked onto a logo scores a perfect 1.000 here."""
    rows = [_row(f, (500.0, 500.0), (800.0, 300.0, 40.0)) for f in (0, 2, 4, 6)]
    got = physics.speeds(rows, "argmax", 30.0)
    assert got and all(v == pytest.approx(0.0, abs=1e-9) for v in got)


def test_rim_of_takes_the_most_confident_rim():
    row = {"f": 0, "d": [["r", 0.2, 0.0, 0.0, 10.0, 10.0],
                         ["r", 0.9, 100.0, 100.0, 140.0, 120.0]]}
    centre, width = physics.rim_of(row)
    assert centre == (120.0, 110.0)
    assert width == pytest.approx(40.0)


def test_no_rim_at_all_answers_none_rather_than_raising():
    assert physics.rim_of({"f": 0, "d": []}) == (None, None)
