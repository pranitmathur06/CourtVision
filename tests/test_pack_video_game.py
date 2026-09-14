"""Names read off the play-by-play, the clock, and what confidence means here."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pack_video_game import clock_of, confidence, player_from  # noqa: E402


def test_a_scorer_is_read_from_the_front():
    assert player_from("Nembhard 14' Pullup Jump Shot (2 PTS)") == "Nembhard"


def test_a_miss_prefix_is_stripped():
    assert player_from("MISS Jal. Williams 10' Step Back Jump Shot") == "Jal. Williams"


def test_an_all_caps_verb_does_not_become_part_of_the_name():
    assert player_from("Nesmith REBOUND (Off:0 Def:1)") == "Nesmith"
    assert player_from("Nesmith BLOCK (1 BLK)") == "Nesmith"


def test_a_jump_ball_yields_no_player_rather_than_a_guess():
    assert player_from("Jump Ball Hartenstein vs. Turner (Nesmith gains possession)") is None


def test_nothing_in_nothing_out():
    assert player_from("") is None and player_from(None) is None


def test_the_clock_counts_down_within_a_period():
    assert clock_of(16.0) == (1, "11:44")
    assert clock_of(0.0) == (1, "12:00")


def test_the_second_period_starts_again_at_twelve():
    assert clock_of(720.0) == (2, "12:00")
    assert clock_of(740.0) == (2, "11:40")


def test_overtime_is_five_minutes():
    assert clock_of(2880.0) == (5, "5:00")


def test_confidence_is_alignment_not_detection():
    # Exactly placed is 1.0; the floor is where the clip would show another play.
    assert confidence(0.0) == 1.0
    assert confidence(2.0) == 0.5
    assert confidence(9.0) == 0.0
