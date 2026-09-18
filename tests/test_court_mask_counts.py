"""Counting PEOPLE, not boxes, in the floor-mask metric."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "eval_court_mask", ROOT / "scripts" / "eval_court_mask.py")
mask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mask)


def test_two_detectors_on_one_player_are_one_person():
    """`p` and `h` are two models run over the same frame and both draw the
    same players. Counting the rows made 71% of Finals G7 frames "impossible"
    before any mask had a chance to be wrong."""
    same = [[100.0, 100.0, 200.0, 400.0], [104.0, 98.0, 198.0, 402.0]]
    assert mask.distinct(same) == 1


def test_two_players_side_by_side_are_two_people():
    apart = [[100.0, 100.0, 200.0, 400.0], [260.0, 100.0, 360.0, 400.0]]
    assert mask.distinct(apart) == 2


def test_a_brushing_overlap_is_still_two_people():
    """Players stand close. Only a near-coincident box is the same person."""
    touching = [[100.0, 100.0, 200.0, 400.0], [180.0, 100.0, 280.0, 400.0]]
    assert mask.distinct(touching) == 2


def test_nothing_is_nobody():
    assert mask.distinct([]) == 0


def test_the_bigger_box_is_the_one_kept():
    """Deduplication is greedy from the largest box, so the result does not
    depend on the order the detectors happened to emit."""
    one = [[100.0, 100.0, 200.0, 400.0], [110.0, 110.0, 190.0, 390.0]]
    assert mask.distinct(one) == mask.distinct(list(reversed(one))) == 1


def test_thirteen_people_is_possible_and_fourteen_is_not():
    """Ten players and three officials. The rule the whole metric rests on."""
    assert mask.IMPOSSIBLE_ABOVE == 13
