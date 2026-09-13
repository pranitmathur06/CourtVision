"""Choosing which frames still need a verdict."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from render_rim_sheets import unlabelled  # noqa: E402


def _grid(n):
    return [{"t": 12.5 + 25 * i, "rim": []} for i in range(n)]


def test_frames_already_judged_are_skipped():
    verdicts = {0: {"rim": "ok"}, 1: {"rim": "miss"}, 2: {"ball": "ok"}}
    picked = unlabelled(_grid(10), verdicts, want=10)
    assert 0 not in picked and 1 not in picked
    # 2 has a ball verdict but no rim verdict, so it still needs one.
    assert 2 in picked


def test_it_asks_for_no_more_than_it_was_asked_for():
    assert len(unlabelled(_grid(100), {}, want=12)) == 12


def test_it_returns_everything_left_when_that_is_fewer():
    assert unlabelled(_grid(3), {}, want=12) == [0, 1, 2]


def test_the_picks_are_spread_through_the_game_not_bunched():
    picked = unlabelled(_grid(300), {}, want=10)
    assert max(picked) - min(picked) > 200


def test_nothing_is_returned_when_everything_is_judged():
    verdicts = {i: {"rim": "ok"} for i in range(5)}
    assert unlabelled(_grid(5), verdicts, want=10) == []


def test_the_picks_are_distinct():
    picked = unlabelled(_grid(300), {}, want=25)
    assert len(set(picked)) == len(picked)


def test_frames_after_the_whistle_can_be_excluded():
    grid = _grid(100)                      # t = 12.5 + 25i, so #100 is 2512.5 s
    picked = unlabelled(grid, {}, want=100, end_s=1000.0)
    assert picked
    assert all(grid[i]["t"] <= 1000.0 for i in picked)


def test_without_an_end_the_whole_video_is_in_scope():
    grid = _grid(100)
    assert len(unlabelled(grid, {}, want=100)) == 100
