"""The rim/ball metric, on hand-built cases where the answer is known."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_rim_and_ball as metric  # noqa: E402


def _rim(x, y, width=40.0):
    return {"centre": [x, y], "width": width}


def test_a_point_within_one_object_width_is_located():
    truth = [{"t": 0.0, "rim": [_rim(100, 100)], "ball": None}]
    assert metric.score(truth, {0.0: {"rim": [[130, 100]]}})["rim"]["located"] == 1
    assert metric.score(truth, {0.0: {"rim": [[145, 100]]}})["rim"]["located"] == 0


def test_a_frame_the_system_never_saw_is_a_miss_not_an_exemption():
    """Registration failing is a failure to find the rim, not a smaller test."""
    truth = [{"t": 0.0, "rim": [_rim(100, 100)], "ball": None}]
    report = metric.score(truth, {})
    assert report["rim"] == {"visible": 1, "located": 0, "missed": 1, "accuracy": 0.0,
                             "frames_without": 0, "false_alarms": 0}


def test_both_baskets_must_each_be_matched():
    truth = [{"t": 0.0, "rim": [_rim(100, 100), _rim(900, 100)], "ball": None}]
    one = metric.score(truth, {0.0: {"rim": [[100, 100]]}})["rim"]
    assert (one["visible"], one["located"], one["false_alarms"]) == (2, 1, 0)
    both = metric.score(truth, {0.0: {"rim": [[100, 100], [900, 100]]}})["rim"]
    assert (both["visible"], both["located"], both["false_alarms"]) == (2, 2, 0)


def test_one_report_cannot_be_matched_to_two_truths():
    truth = [{"t": 0.0, "rim": [_rim(100, 100), _rim(110, 100)], "ball": None}]
    report = metric.score(truth, {0.0: {"rim": [[105, 100]]}})["rim"]
    assert (report["located"], report["missed"]) == (1, 1)


def test_a_report_where_nothing_is_visible_is_a_false_alarm():
    truth = [{"t": 0.0, "rim": [], "ball": None}]
    report = metric.score(truth, {0.0: {"rim": [[7, 7]], "ball": [9, 9]}})
    assert report["rim"]["false_alarms"] == 1 and report["ball"]["false_alarms"] == 1
    assert report["rim"]["visible"] == 0


def test_a_ball_on_a_spectator_is_both_a_miss_and_a_false_alarm():
    truth = [{"t": 0.0, "rim": [], "ball": {"centre": [500, 300], "width": 20}}]
    report = metric.score(truth, {0.0: {"ball": [900, 80]}})["ball"]
    assert (report["located"], report["missed"], report["false_alarms"]) == (0, 1, 1)


def test_the_grid_covers_the_video_and_avoids_zero():
    times = metric.sample_times(100.0, every_s=10.0)
    assert times[0] == 5.0 and times[-1] < 100.0 and len(times) == 10


def test_wilson_brackets_the_estimate():
    lo, hi = metric.wilson(95, 100)
    assert lo < 0.95 < hi and lo > 0.88
