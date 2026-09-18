"""The play-by-play cache, and the one way it could poison a pipeline.

`data/pbp_cache/` held 1,230 games since the tracking work and nothing read it.
Now the aligner does, which means a bad cache entry is a bad alignment for every
later run -- silently, because a cache hit prints nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cache_pbp import load  # noqa: E402


def test_a_hit_does_not_touch_the_network(tmp_path):
    path = tmp_path / "0000000001.json"
    path.write_text(json.dumps([{"period": 1, "clock": "PT12M00.00S"}]))
    assert load("0000000001", path, allow_fetch=False) == [
        {"period": 1, "clock": "PT12M00.00S"}]


def test_an_empty_cache_file_is_not_a_game_with_no_plays(tmp_path):
    """An empty list on disk is indistinguishable from a failed fetch. It must
    not be served as an answer, or one bad run silently empties every later
    alignment."""
    path = tmp_path / "0000000002.json"
    path.write_text("[]")
    with pytest.raises(FileNotFoundError):
        load("0000000002", path, allow_fetch=False)


def test_a_missing_cache_with_fetching_off_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        load("0000000003", tmp_path / "nope.json", allow_fetch=False)


def test_the_four_registered_games_are_cached_and_have_periods():
    """Every registered broadcast can be aligned with the network unplugged."""
    from courtvision.games import registry
    for key, game in registry().items():
        assert game.pbp.exists(), f"{key} has no cached play-by-play"
        actions = json.loads(game.pbp.read_text())
        assert actions, f"{key}'s cache is empty"
        periods = sorted({a.get("period") for a in actions if a.get("period")})
        assert periods[:4] == [1, 2, 3, 4], f"{key} is missing a period: {periods}"


def test_the_cache_shape_is_what_the_aligner_reads():
    """`align_game_events.classify` reads actionType, description, subType and
    the aligner reads clock and period. A cache in another shape would align to
    nothing and report it as a game with no events."""
    from courtvision.games import get
    import align_game_events
    actions = json.loads(get("hou").pbp.read_text())
    labelled = [a for a in actions if align_game_events.classify(a)]
    assert len(labelled) > 300, "the unseen game classifies almost nothing"
    timed = [a for a in labelled
             if align_game_events.clock_seconds(a.get("clock") or "") is not None]
    assert len(timed) == len(labelled), "an action carries no readable clock"
