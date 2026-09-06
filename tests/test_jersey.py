"""Naming a track from repeated, unreliable jersey reads."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.jersey import JerseyVoter, Verdict, jersey_crop, read_number

ROSTER = {
    "0": ("IND", "Tyrese Haliburton"),
    "2": ("IND", "Andrew Nembhard"),
    "43": ("IND", "Pascal Siakam"),
    "7": ("OKC", "Chet Holmgren"),
    "2v": ("OKC", "Shai Gilgeous-Alexander"),
}


def test_jersey_crop_rejects_a_distant_player():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    # 60 px tall: too far away for a number to survive.
    assert jersey_crop(image, [10, 10, 50, 70]) is None


def test_jersey_crop_upscales_a_close_player():
    pytest.importorskip("cv2")
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    crop = jersey_crop(image, [100, 100, 180, 260])
    assert crop is not None
    assert crop.shape[0] > 160, "should be upscaled well beyond the source band"


def test_read_number_rejects_long_strings():
    class Reader:
        def readtext(self, *_args, **_kwargs):
            return [(None, "12345", 0.9)]
    assert read_number(Reader(), np.zeros((10, 10, 3), dtype=np.uint8)) is None


def test_read_number_strips_leading_zeros_but_keeps_zero():
    class Reader:
        def __init__(self, text):
            self.text = text

        def readtext(self, *_args, **_kwargs):
            return [(None, self.text, 0.9)]
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    assert read_number(Reader("07"), crop) == "7"
    assert read_number(Reader("0"), crop) == "0"


def test_read_number_survives_a_reader_that_throws():
    class Reader:
        def readtext(self, *_args, **_kwargs):
            raise RuntimeError("model exploded")
    assert read_number(Reader(), np.zeros((10, 10, 3), dtype=np.uint8)) is None


def test_voter_names_a_track_with_enough_agreement():
    voter = JerseyVoter(ROSTER)
    for _ in range(4):
        voter.observe(1, "43", team="IND")
    verdict = voter.verdict(1)
    assert verdict is not None
    assert verdict.name == "Pascal Siakam" and verdict.votes == 4


def test_voter_stays_silent_below_the_vote_floor():
    voter = JerseyVoter(ROSTER)
    voter.observe(1, "43", team="IND")
    voter.observe(1, "43", team="IND")
    # Two votes is not enough to attribute a play to a named player.
    assert voter.verdict(1) is None


def test_voter_stays_silent_without_a_margin():
    voter = JerseyVoter(ROSTER)
    for _ in range(3):
        voter.observe(1, "43", team="IND")
    for _ in range(3):
        voter.observe(1, "2", team="IND")
    assert voter.verdict(1) is None, "a tie must not be resolved arbitrarily"


def test_voter_discards_numbers_off_the_roster():
    voter = JerseyVoter(ROSTER)
    for _ in range(9):
        voter.observe(1, "58", team="IND")     # nobody wears 58
    assert voter.verdict(1) is None


def test_voter_discards_a_number_from_the_other_team():
    voter = JerseyVoter(ROSTER)
    for _ in range(9):
        # 7 is an OKC number; this player wears IND colours.
        voter.observe(1, "7", team="IND")
    assert voter.verdict(1) is None


def test_voter_handles_reads_with_no_team_known():
    voter = JerseyVoter(ROSTER)
    for _ in range(4):
        voter.observe(1, "43")
    assert voter.verdict(1).name == "Pascal Siakam"


def test_all_verdicts_returns_only_confident_tracks():
    voter = JerseyVoter(ROSTER)
    for _ in range(4):
        voter.observe(1, "43", team="IND")
    voter.observe(2, "0", team="IND")
    found = voter.all_verdicts()
    assert set(found) == {1}


def test_verdict_confidence_rule():
    assert Verdict(1, "43", "P", "IND", votes=3, runner_up=1).confident
    assert not Verdict(1, "43", "P", "IND", votes=3, runner_up=2).confident
    assert not Verdict(1, "43", "P", "IND", votes=2, runner_up=0).confident
