"""The registry has to be the only place a broadcast's paths are written down.

These pin the two properties that make it worth having: a path is DERIVED from
the key unless it was published under another name, and two broadcasts cannot
quietly share one.
"""

from __future__ import annotations

import json

import pytest

from courtvision.games import Broadcast, get, registry


def _write(tmp_path, entries):
    path = tmp_path / "games.json"
    path.write_text(json.dumps(entries))
    return path


def test_every_path_is_derived_from_the_key(tmp_path):
    path = _write(tmp_path, {"zz": {"game_id": "0000000001", "label": "Z",
                                    "video": "data/games/vid.mp4", "prefix": "z"}})
    game = get("zz", path)
    assert str(game.clip_detections) == "outputs/clip_detections_zz.json"
    assert str(game.aligned) == "outputs/games/aligned_zz.json"
    assert str(game.clip_dir) == "data/clips/zz"
    assert str(game.clip_index) == "data/clips/zz/index.json"
    # Video-stem paths follow the FILE, not the key, because the clock reader
    # and the detection cache are keyed on the encode they read.
    assert str(game.clock) == "outputs/clock/vid.json"
    assert str(game.detections) == "outputs/detections/vid.json"


def test_a_published_name_survives_and_nothing_else_does(tmp_path):
    path = _write(tmp_path, {
        "zz": {"game_id": "0000000001", "label": "Z", "video": "v.mp4",
               "prefix": "z", "published": {"clip_index": "docs/clips/index.json"}}})
    game = get("zz", path)
    assert str(game.clip_index) == "docs/clips/index.json"
    # ...and the override is scoped to the one key it names.
    assert str(game.overlays) == "data/clips/zz/overlays.json"
    assert str(game.clip_dir) == "data/clips/zz"


def test_two_games_cannot_share_a_clip_prefix(tmp_path):
    """Clip files are `f001110.mp4`; a shared letter overwrites another game."""
    path = _write(tmp_path, {
        "a": {"game_id": "1", "label": "A", "video": "a.mp4", "prefix": "x"},
        "b": {"game_id": "2", "label": "B", "video": "b.mp4", "prefix": "x"}})
    with pytest.raises(ValueError, match="share clip prefix"):
        get("a", path)


def test_a_prefix_must_be_one_letter(tmp_path):
    path = _write(tmp_path, {"a": {"game_id": "1", "label": "A",
                                   "video": "a.mp4", "prefix": "ab"}})
    with pytest.raises(ValueError, match="one letter"):
        get("a", path)


def test_a_game_answers_to_its_key_its_official_id_and_its_video_stem(tmp_path):
    path = _write(tmp_path, {"zz": {"game_id": "0042400401", "label": "Z",
                                    "video": "data/games/abc.mp4", "prefix": "z"}})
    assert get("zz", path).key == "zz"
    assert get("0042400401", path).key == "zz"
    assert get("abc", path).key == "zz"


def test_an_unknown_game_names_the_ones_it_knows(tmp_path):
    path = _write(tmp_path, {"zz": {"game_id": "1", "label": "Z",
                                    "video": "a.mp4", "prefix": "z"}})
    with pytest.raises(KeyError, match="zz"):
        get("nope", path)


def test_the_shipped_registry_loads_and_its_videos_are_distinct():
    games = registry()
    assert len(games) >= 4, "the fourth broadcast is the acceptance test"
    videos = [g.video for g in games.values()]
    assert len(set(videos)) == len(videos), "two entries point at one file"
    ids = [g.game_id for g in games.values()]
    assert len(set(ids)) == len(ids), "two entries claim one official game"


def test_the_held_out_broadcast_is_held_out_per_arm_and_not_wholesale():
    """A boolean here published a contaminated number as an acceptance test.

    The Houston footage has never been near the event pipeline and is
    thoroughly inside the court-registration work -- the accuracy log says so
    in as many words. So it is held out for one set of arms and not the other,
    and the registry has to be able to say that."""
    held = [g for g in registry().values() if g.held_out]
    assert held, "no broadcast is held out for anything; there is no acceptance test"
    for game in held:
        assert game.held_out_for("alignment"), (
            f"{game.key} is held out for no event arm")
        assert not game.held_out_for("registration"), (
            "docs/continuous-game-accuracy.md records that Toyota Center was "
            "diagnosed on; claiming it as a registration hold-out is false")
        assert not game.unseen, "no broadcast here is held out for every arm"
        assert game.held_out_note, "a hold-out claim with no reason is unreadable"


def test_an_unknown_arm_is_not_held_out():
    """Guilty until the registry says otherwise: the failure being guarded
    against is publishing a contaminated number, so the quiet default has to be
    the one that cannot do it."""
    from courtvision.games import get
    assert get("hou").held_out_for("some_arm_invented_next_month") is False


def test_two_games_cannot_share_a_video_stem(tmp_path):
    """Different files, same stem, same clock and detection artefacts."""
    path = _write(tmp_path, {
        "a": {"game_id": "1", "label": "A", "video": "x/game.mp4", "prefix": "a"},
        "b": {"game_id": "2", "label": "B", "video": "y/game.mp4", "prefix": "b"}})
    with pytest.raises(ValueError, match="video stem"):
        get("a", path)


def test_two_games_cannot_claim_one_official_id(tmp_path):
    path = _write(tmp_path, {
        "a": {"game_id": "1", "label": "A", "video": "a.mp4", "prefix": "a"},
        "b": {"game_id": "1", "label": "B", "video": "b.mp4", "prefix": "b"}})
    with pytest.raises(ValueError, match="both claim official"):
        get("a", path)


def test_a_mistyped_published_name_is_refused(tmp_path):
    """It used to fall through to the derived path with no complaint, which is
    the quietest way to score a game against the wrong file."""
    path = _write(tmp_path, {
        "a": {"game_id": "1", "label": "A", "video": "a.mp4", "prefix": "a",
              "published": {"clip_indx": "docs/clips/index.json"}}})
    with pytest.raises(ValueError, match="not paths"):
        get("a", path)


def test_derived_paths_cannot_collide_across_the_shipped_registry():
    """Every artefact of every game, in one bag, with no duplicates.

    A collision here means one broadcast's run would overwrite another's
    evidence, which is the failure this registry exists to prevent.
    """
    seen: dict[str, str] = {}
    fields = ("clock", "scoreboard", "detections", "clip_detections", "aligned",
              "clip_index", "overlays", "vision_shots", "report", "end_to_end",
              "roster", "pbp")
    for key, game in registry().items():
        for field in fields:
            path = str(getattr(game, field))
            assert path not in seen, (
                f"{key}.{field} and {seen[path]} are both {path}")
            seen[path] = f"{key}.{field}"


def test_a_broadcast_with_no_published_names_has_none():
    """A NEW game must be fully derived; a `published` map on one would mean
    somebody hand-placed a path again, which is the thing being removed."""
    for key, game in registry().items():
        if game.unseen:
            assert not game.published, (
                f"{key} is the acceptance test and must not carry hand-written "
                f"paths")


def test_frozen_so_a_caller_cannot_repoint_a_game():
    game = Broadcast(key="k", game_id="1", label="L", video="v.mp4", prefix="v")
    with pytest.raises(Exception):
        game.video = "other.mp4"                      # type: ignore[misc]


def test_the_scripts_that_had_hardcoded_game_dicts_reproduce_them():
    """Five copies of the same five facts, now derived. Pinned to the originals.

    `make_possession_label_page.py`, `make_handler_label_page.py`,
    `build_possession_windows.py` and `rebuild_detector_dataset.py` each carried
    their own map of game to video and clip index. Adding a broadcast meant
    editing all of them and getting every one to agree, which is why there were
    three games in this repository and not four. These are the values they held.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import build_possession_windows
    import make_handler_label_page
    import make_possession_label_page
    import rebuild_detector_dataset

    was = {
        "clip_detections_g7": ("data/raw_clips/fullgame.mp4",
                               "docs/clips/index.json", "2025 Finals G7"),
        "clip_detections_g1": ("data/games/iVhcru3Gli0.mp4",
                               "docs/clips/index_finals_g1.json", "2025 Finals G1"),
        "clip_detections_ecf": ("data/games/T1d3VxVnDUo.mp4",
                                "docs/clips/index_ecf_g1.json", "2025 ECF G1"),
    }
    for page in (make_handler_label_page, make_possession_label_page):
        for stem, value in was.items():
            assert page.GAMES[stem] == value, f"{page.__name__} moved {stem}"
    for stem, (video, index, label) in was.items():
        assert build_possession_windows.GAMES[label] == (
            video, index, f"outputs/{stem}.json")
        assert rebuild_detector_dataset.VIDEO_FOR[stem] == video
        assert rebuild_detector_dataset.INDEX_FOR[stem] == index
    # ...and the fourth broadcast is now in all of them, for free.
    assert "clip_detections_hou" in make_handler_label_page.GAMES
    assert "clip_detections_hou" in rebuild_detector_dataset.VIDEO_FOR
