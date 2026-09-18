"""What the per-game report must never do.

The failures this file guards against all happened in this repository: a rate
printed without the denominator it was taken over, an empty measurement reported
as a confident zero, and an architectural ceiling presented as a model failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_by_game import (Arm, _inside, _iou, clips_arm,  # noqa: E402
                          alignment_arm)
from courtvision.games import Broadcast  # noqa: E402


def test_nothing_measured_is_not_a_claim_of_zero():
    """An arm with no data is 0.00-1.00 and NO DATA, never 0.000 and FAIL.

    A new broadcast has no labels, so half this report's arms have n=0 on
    arrival. Printing those as zeroes would say the stack scored nothing on a
    game nobody has measured."""
    arm = Arm("ball: top-1", 0, 0, labelled=True)
    assert (arm.low, arm.high) == (0.0, 1.0)
    assert arm.verdict(0.90) == "NO DATA"
    assert arm.rate is None
    assert "0.00-1.00" in arm.row(0.90)


def test_a_capped_arm_says_capped_and_not_failed():
    """Vision-only capture is limited to the share of plays that are shots.

    Printing FAIL there suggests a bug where there is a design limit; the report
    has to distinguish "the model is not good enough" from "this architecture
    cannot answer the question at all"."""
    arm = Arm("e2e vision: captured", 430, 1000, cap=0.43)
    assert arm.verdict(0.90) == "CAPPED 100% of 0.43"
    assert Arm("x", 430, 1000, cap=0.95).verdict(0.90) == "FAIL"


def test_capped_says_how_much_of_its_own_ceiling_the_arm_reaches():
    """"CAPPED at 0.43" read identically for an arm at 0.43 and one at 0.01, so
    a mode performing at 2% of what its architecture allows looked like one
    performing at its limit."""
    assert Arm("x", 10, 1000, cap=0.43).verdict(0.90) == "CAPPED 2% of 0.43"
    assert Arm("x", 244, 1000, cap=0.43).verdict(0.90) == "CAPPED 57% of 0.43"


def test_pass_needs_the_lower_end_of_the_interval():
    """13 for 13 is not evidence of 0.90; its interval reaches 0.72."""
    tight = Arm("a", 950, 1000)
    assert tight.verdict(0.90) == "PASS"
    loose = Arm("b", 13, 13)
    assert loose.rate == 1.0
    assert loose.verdict(0.90) == "PASS (point)", "a tiny n must not read as PASS"
    assert loose.low < 0.90


def test_capped_is_decided_before_a_verdict_is_reached_but_after_no_data():
    """An arm with a cap and no data is NO DATA: there is nothing to cap."""
    assert Arm("x", 0, 0, cap=0.4).verdict(0.90) == "NO DATA"


def test_a_row_always_carries_its_denominator():
    """`docs/continuous-game-accuracy.md` keeps "0.941 at coverage 0.437" as a
    warning. No printed rate may appear without the n it was taken over."""
    row = Arm("some arm", 7, 9).row(0.90)
    assert "n=9" in row and "0.778" in row


def test_iou_and_inside_are_the_shapes_they_claim():
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert _inside([0, 0, 10, 10], (5, 5), 1.0) is True
    assert _inside([0, 0, 10, 10], (100, 100), 1.0) is False


def _game(tmp_path, **published):
    return Broadcast(key="t", game_id="1", label="T",
                     video=str(tmp_path / "v.mp4"), prefix="t",
                     published={k: str(v) for k, v in published.items()})


def test_alignment_reports_per_class_and_skips_classes_too_small_to_read(tmp_path):
    aligned = tmp_path / "aligned.json"
    aligned.write_text(json.dumps({
        "per_action": {"Made Shot (2PT)": {"located": 48, "total": 48},
                       "Ejection": {"located": 1, "total": 2}},
        "events": [{}] * 49}))
    arms, vectors = alignment_arm(_game(tmp_path, aligned=aligned))
    assert arms[0].hits == 49 and arms[0].total == 50
    names = [a.name.strip() for a in arms]
    assert "align Made Shot (2PT)" in names
    assert "align Ejection" not in names, "n=2 cannot carry a per-class rate"
    assert "Ejection" not in vectors


def test_a_missing_artefact_is_no_data_rather_than_an_exception(tmp_path):
    """A broadcast mid-pipeline must still produce a report for what exists."""
    arms, vectors = alignment_arm(_game(tmp_path, aligned=tmp_path / "gone.json"))
    assert arms[0].total == 0 and vectors == {}
    assert clips_arm(_game(tmp_path, clip_index=tmp_path / "gone.json"))[0].total == 0


def test_clips_arm_counts_rows_without_footage_separately(tmp_path):
    """Rebounds are indexed and not cut. A row with no clip is still a row the
    timeline answers; conflating the two would make a size decision look like a
    coverage loss."""
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"clips": [
        {"clip": "t000100.mp4", "error_s": 0.0},
        {"clip": None, "error_s": 0.5},
        {"clip": "t000300.mp4", "error_s": 4.0}]}))
    arm = clips_arm(_game(tmp_path, clip_index=index))[0]
    assert (arm.hits, arm.total) == (2, 3)
    assert "2 of 3 rows carry footage" in arm.note
