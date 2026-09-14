"""The three verdicts, kept apart."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ball_truth_from_labels import split  # noqa: E402


def test_a_located_ball_is_truth():
    located, absent, unknown = split([{"verdict": "ball", "ball": [10, 20], "t": 1.0}])
    assert len(located) == 1 and not absent and not unknown


def test_no_ball_in_frame_is_not_truth_and_not_unknown():
    located, absent, unknown = split([{"verdict": "none", "ball": None, "t": 1.0}])
    assert not located and len(absent) == 1 and not unknown


def test_unfindable_is_its_own_bin():
    located, absent, unknown = split([{"verdict": "unknown", "ball": None, "t": 1.0}])
    assert not located and not absent and len(unknown) == 1


def test_a_ball_verdict_with_no_coordinates_is_not_counted_as_located():
    # Guards against a half-finished row inflating the truth set.
    located, _, _ = split([{"verdict": "ball", "ball": None, "t": 1.0}])
    assert not located


def test_unjudged_frames_fall_into_nothing():
    located, absent, unknown = split([{"verdict": None, "ball": None, "t": 1.0}])
    assert not located and not absent and not unknown
