"""Grouping frames by their source game -- the fix for the frame-level leak."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")
from split_court_keypoints import game_of  # noqa: E402


def test_both_naming_conventions_group_to_the_same_game():
    """The dataset uses two. An earlier version handled one, so every frame of
    the other became its own 'game' and adjacent frames went back on both
    sides of the split -- the leak this script exists to remove."""
    assert game_of(
        "boston-celtics-new-york-knicks-game-1-q1-01_54-01_48_mp4-0005_jpg.rf.ab.jpg"
    ) == "boston-celtics-new-york-knicks-game-1"
    a = game_of("golden-state-warriors-houston-rockets-game-1-09_49-09_44_mp4-0004_jpg.rf.cd.jpg")
    b = game_of("golden-state-warriors-houston-rockets-game-1-11_16-11_09_mp4-0002_jpg.rf.ef.jpg")
    assert a == b == "golden-state-warriors-houston-rockets-game-1"


def test_every_real_filename_groups_to_a_plausible_game():
    root = Path("data/labeled/court_keypoints")
    names = [p.name for split in ("train", "valid", "test")
             for p in (root / split / "images").glob("*.jpg")]
    if not names:
        pytest.skip("dataset not present")
    games = {game_of(n) for n in names}
    assert len(games) < len(names) / 10, f"{len(games)} games from {len(names)} frames"
    for game in games:
        assert "_mp4" not in game and ".rf." not in game and "-q" not in game, game


def test_the_shipped_splits_share_no_game():
    root = Path("data/labeled/court_keypoints_by_game")
    if not root.exists():
        pytest.skip("by-game split not built")
    seen: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        for path in (root / split / "images").glob("*.jpg"):
            game = game_of(path.name)
            assert seen.setdefault(game, split) == split, (
                f"{game} appears in both {seen[game]} and {split}")
