"""Judging two masks against each other on the same frames."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "compare_masks", ROOT / "scripts" / "compare_masks.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


def _cache(on_flags, ball=(110.0, 100.0)):
    """One clip, one frame: two people and a ball in the first one's hands."""
    detections = [["p", 0.9, 0.0, 0.0, 100.0, 300.0],
                  ["p", 0.9, 400.0, 0.0, 500.0, 300.0]]
    if ball is not None:
        detections.append(["b", 0.9, ball[0] - 5, ball[1] - 5,
                           ball[0] + 5, ball[1] + 5])
    return {"clips": {"a.mp4": [{"f": 0, "d": detections, "on": on_flags}]}}


def _write(tmp_path, name, blob):
    path = tmp_path / name
    path.write_text(json.dumps(blob))
    return path


def test_a_mask_that_keeps_the_carrier_scores_it(tmp_path):
    carrier, counts = compare.judge(_write(tmp_path, "a.json",
                                           _cache([True, True])))
    assert carrier[("a.mp4", 0)] is True
    assert counts[("a.mp4", 0)] is True


def test_a_mask_that_drops_the_carrier_scores_that(tmp_path):
    carrier, _ = compare.judge(_write(tmp_path, "b.json", _cache([False, True])))
    assert carrier[("a.mp4", 0)] is False


def test_a_frame_with_no_ball_has_no_carrier_but_still_has_a_count(tmp_path):
    carrier, counts = compare.judge(_write(tmp_path, "c.json",
                                           _cache([True, True], ball=None)))
    assert ("a.mp4", 0) not in carrier
    assert ("a.mp4", 0) in counts


def test_a_ball_nobody_is_near_has_no_carrier(tmp_path):
    carrier, _ = compare.judge(_write(tmp_path, "d.json",
                                      _cache([True, True], ball=(2000.0, 2000.0))))
    assert ("a.mp4", 0) not in carrier


def test_fourteen_people_kept_breaks_the_count(tmp_path):
    people = [["p", 0.9, i * 200.0, 0.0, i * 200.0 + 100.0, 300.0]
              for i in range(14)]
    blob = {"clips": {"a.mp4": [{"f": 0, "d": people, "on": [True] * 14}]}}
    _, counts = compare.judge(_write(tmp_path, "e.json", blob))
    assert counts[("a.mp4", 0)] is False


def test_the_same_player_drawn_by_both_detectors_counts_once(tmp_path):
    people = [["p", 0.9, i * 200.0, 0.0, i * 200.0 + 100.0, 300.0]
              for i in range(13)]
    people.append(["h", 0.9, 4.0, 2.0, 98.0, 302.0])
    blob = {"clips": {"a.mp4": [{"f": 0, "d": people, "on": [True] * 14}]}}
    _, counts = compare.judge(_write(tmp_path, "f.json", blob))
    assert counts[("a.mp4", 0)] is True


def test_a_missing_mask_entry_is_treated_as_kept(tmp_path):
    """A row written before the mask existed must not read as "dropped
    everybody", which would make any new mask look like an improvement."""
    blob = _cache([])
    carrier, _ = compare.judge(_write(tmp_path, "g.json", blob))
    assert carrier[("a.mp4", 0)] is True
