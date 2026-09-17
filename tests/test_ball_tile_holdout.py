"""The ball tile builder must not train on a frame the ball is scored against.

Nothing had retrained the ball since the uniform labels landed, so nothing was
contaminated. These tests exist so nothing does: the holdout used to name two
files by hand while the evaluators had moved on to several hundred instants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_ball_tiles import (HOLDOUT_S, SOURCE_GAME,  # noqa: E402
                              blocked_share, truth_instants)


def test_every_scored_instant_is_held_out_not_just_the_original_thirteen():
    held = set(truth_instants(SOURCE_GAME))
    legacy = json.load(open("data/labeling/rim_ball/ball_truth_handlocated.json"))
    for row in legacy["frames"]:
        if row.get("ball"):
            assert float(row["t"]) in held
    uniform = json.load(open("data/labeling/rim_ball/ball_truth_uniform.json"))
    for row in uniform["frames"]:
        assert float(row["t"]) in held
    assert len(held) > 100, "the holdout is back to naming a handful of files"


def test_frames_with_no_ball_are_held_out_too():
    """'absent' is the false-alarm denominator and 'unknown' is an exclusion.

    Both are scored against, so training on either leaks just as surely as
    training on a located ball does.
    """
    held = set(truth_instants(SOURCE_GAME))
    uniform = json.load(open("data/labeling/rim_ball/ball_truth_uniform.json"))
    for key in ("absent", "unknown"):
        for row in uniform.get(key, []):
            assert float(row["t"]) in held


def test_another_games_instants_do_not_block_this_broadcast():
    """A timestamp means a different moment in a different game.

    Pooling them blocked 119% of one broadcast's span -- more than the game is
    long -- and would have thrown away usable footage for nothing.
    """
    mine = set(truth_instants(SOURCE_GAME))
    other = set(truth_instants("2025 Finals G1"))
    assert other and mine
    assert len(other) < len(mine)
    assert other != mine


def test_the_uniform_half_is_held_out_and_the_hard_half_is_not():
    """The hard half is training data by design; the uniform half is the eval."""
    held = set(truth_instants(SOURCE_GAME))
    rows = json.load(open("data/labels/possession_labels.json"))["frames"]
    mine = [r for r in rows if r["game"] == SOURCE_GAME]
    uniform = [r for r in mine if r["pick"] == "random"]
    assert uniform, "no uniform rows for the source game"
    assert all(float(r["t"]) in held for r in uniform)


def test_blocked_share_merges_overlapping_windows():
    # three instants 1 s apart, +/-20 s: one block of 42 s, not three of 40
    assert abs(blocked_share([100.0, 101.0, 102.0], holdout=20.0, span=420.0)
               - 42.0 / 420.0) < 1e-9
    assert blocked_share([], holdout=20.0, span=100.0) == 0.0


def test_the_holdout_cost_on_the_tile_source_is_reported_and_large():
    """It blocks most of this broadcast, and that is the correct answer.

    A uniform 25 s evaluation grid over one game is dense enough that holding it
    out leaves little of that game to train on. The builder prints the share so
    the next person does not read a small tile count as a bug.
    """
    share = blocked_share(truth_instants(SOURCE_GAME), holdout=HOLDOUT_S)
    assert 0.6 < share <= 1.0
