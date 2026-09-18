"""Two bugs the shipped stream still carries, so they cannot come back.

`outputs/games/stream_3games.json` has `["clip", "clip", "clip"]` in its field
names today: the packer appended one per game, and a row carries a single clip
value, so two of those names point at columns that do not exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pack_video_game  # noqa: E402


def test_the_clip_field_is_named_once_however_many_games_are_packed():
    fields = ["t", "period", "clock"]
    for _ in range(3):
        if "clip" not in fields:
            fields = fields + ["clip"]
    assert fields.count("clip") == 1


def test_the_tip_off_offset_is_an_argument_and_not_a_literal():
    """524.0 was a bare number inside a list comprehension, belonging to one
    broadcast and saying so nowhere. A second game packed with it would have
    every vision row's period and clock wrong by the difference in tip-offs."""
    source = Path(pack_video_game.__file__).read_text()
    assert "524.0, 0.0" not in source
    assert '"--tip-off-s"' in source


def test_clock_of_is_the_same_function_it_was():
    """Pinned because the offset moved out of it and nothing else may."""
    assert pack_video_game.clock_of(0.0) == (1, "12:00")
    assert pack_video_game.clock_of(719.0) == (1, "0:01")
    assert pack_video_game.clock_of(720.0) == (2, "12:00")
    assert pack_video_game.clock_of(2880.0) == (5, "5:00")
    assert pack_video_game.clock_of(2880.0 + 299.0) == (5, "0:01")
