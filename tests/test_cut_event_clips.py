"""Which logged rows are worth footage, and stable clip naming."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cut_event_clips import clip_name, wanted  # noqa: E402


def test_a_made_shot_is_worth_watching():
    assert wanted({"action": "Made Shot (2PT)", "video_s": 540.0, "error_s": 0.0})


def test_a_substitution_is_not():
    assert not wanted({"action": "Substitution", "video_s": 540.0, "error_s": 0.0})


def test_a_timeout_is_not():
    assert not wanted({"action": "Timeout", "video_s": 540.0, "error_s": 0.0})


def test_an_event_the_aligner_placed_badly_is_skipped():
    # Six seconds cut five seconds off target shows a different possession.
    assert not wanted({"action": "Rebound", "video_s": 540.0, "error_s": 9.0})


def test_a_small_alignment_error_is_still_worth_cutting():
    assert wanted({"action": "Rebound", "video_s": 540.0, "error_s": 2.0})


def test_an_event_with_no_video_time_cannot_be_cut():
    assert not wanted({"action": "Rebound", "video_s": None, "error_s": 0.0})


def test_the_name_is_stable_for_the_same_instant():
    a = {"video_s": 1637.5}
    assert clip_name(a) == clip_name(dict(a))


def test_different_instants_get_different_names():
    assert clip_name({"video_s": 1637.5}) != clip_name({"video_s": 1637.6})
