"""The judgement format, on cases where the intended truth is obvious."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_truth  # noqa: E402


def test_ok_confirms_the_system_s_own_report_as_the_truth_point():
    out = make_truth.parse_spec("ok", [[100.0, 50.0]], 2.0, 34.0, [])
    assert out == [{"centre": [100.0, 50.0], "width": 34.0}]


def test_a_hand_point_is_scaled_from_panel_to_frame():
    out = make_truth.parse_spec("x:120,88", [], 2.0, 34.0, [])
    assert out == [{"centre": [240.0, 176.0], "width": 34.0}]


def test_a_dash_means_the_object_is_not_in_the_picture():
    assert make_truth.parse_spec("-", [[1.0, 2.0]], 2.0, 34.0, []) == []


def test_two_baskets_confirm_two_reports_in_order():
    out = make_truth.parse_spec("ok;ok", [[10.0, 10.0], [900.0, 12.0]], 2.0, 34.0, [])
    assert [o["centre"] for o in out] == [[10.0, 10.0], [900.0, 12.0]]


def test_a_confirmed_report_can_sit_beside_an_unclaimed_one():
    out = make_truth.parse_spec("ok;+x:520,90", [[10.0, 10.0]], 2.0, 34.0, [])
    assert [o["centre"] for o in out] == [[10.0, 10.0], [1040.0, 180.0]]


def test_confirming_a_report_that_was_never_made_is_an_error():
    """Otherwise a typo silently invents truth that flatters the system."""
    with pytest.raises(ValueError):
        make_truth.parse_spec("ok", [], 2.0, 34.0, [])


def test_an_unreadable_spec_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        make_truth.parse_spec("probably-fine", [], 2.0, 34.0, [])


def test_the_chain_from_judgements_to_a_score_holds_together(tmp_path):
    """system -> judgements -> truth -> score, on a case worked out by hand."""
    import json
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_rim_and_ball as metric

    system = {"frames": [
        {"t": 12.5, "rim": [[100.0, 50.0]], "ball": [300.0, 200.0]},   # both right
        {"t": 37.5, "rim": [[700.0, 60.0]], "ball": None},             # rim right, ball missed
        {"t": 62.5, "rim": [], "ball": [640.0, 360.0]},                # nothing there at all
    ]}
    system_path = tmp_path / "system.json"
    system_path.write_text(json.dumps(system))

    judgements = tmp_path / "judgements.txt"
    judgements.write_text(
        "0 rim=ok ball=ok\n"
        "1 rim=ok ball=x:200,150\n"       # a ball is there; panel coords double
        "2 rim=- ball=-\n")

    out = tmp_path / "truth.json"
    _sys.argv = ["make_truth", "--judgements", str(judgements),
                 "--system", str(system_path), "--out", str(out)]
    assert make_truth.main() == 0

    truth = json.loads(out.read_text())["frames"]
    assert truth[1]["ball"]["centre"] == [400.0, 300.0]

    report = metric.score(truth, {r["t"]: r for r in system["frames"]})
    assert report["rim"] == {"visible": 2, "located": 2, "missed": 0, "accuracy": 1.0,
                             "frames_without": 1, "false_alarms": 0}
    assert report["ball"]["visible"] == 2 and report["ball"]["located"] == 1
    assert report["ball"]["false_alarms"] == 1      # the claim on the empty frame


def test_unknown_is_neither_present_nor_absent():
    """Calling a hard-to-see ball absent would delete a miss and flatter the system."""
    assert make_truth.parse_spec("?", [[1.0, 2.0]], 2.0, 18.0, []) is make_truth.UNKNOWN


def test_an_unknown_object_is_left_out_of_both_sides_of_the_score():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_rim_and_ball as metric

    truth = [{"t": 0.0, "rim": [], "ball": None, "ball_unknown": True},
             {"t": 1.0, "rim": [], "ball": {"centre": [10.0, 10.0], "width": 18.0}}]
    report = metric.score(truth, {0.0: {"ball": [999.0, 999.0]}, 1.0: {"ball": [10.0, 10.0]}})
    assert report["ball"]["visible"] == 1 and report["ball"]["located"] == 1
    assert report["ball"]["false_alarms"] == 0     # the unknown frame votes on nothing
