"""The driver's claim is "no hand-edited constants". These check it.

The acceptance test for this pipeline is a broadcast nothing was tuned on. A
driver that quietly passed a per-game threshold, or pointed one game's stage at
another game's cache, would make that test pass while proving nothing -- so the
things pinned here are the ones that would let it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import add_broadcast  # noqa: E402
from courtvision.games import get, registry  # noqa: E402


def _plan(key):
    return add_broadcast.stages_for(
        get(key), "weights.pt", None, 5.0, [])


def test_every_stage_names_a_script_that_exists():
    for stage in _plan("hou"):
        assert stage.argv, f"{stage.name} has no command"
        script = Path(stage.argv[0])
        assert script.exists(), f"{stage.name} invokes a missing {script}"


def test_every_flag_a_stage_passes_is_a_flag_that_script_accepts():
    """A wrong flag surfaces hours into a run, after the expensive stages."""
    import re
    for stage in _plan("hou"):
        source = Path(stage.argv[0]).read_text()
        accepted = set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', source))
        passed = {a for a in stage.argv[1:] if a.startswith("--")}
        unknown = passed - accepted
        assert not unknown, f"{stage.name} passes {unknown}, which it does not accept"


def test_no_stage_is_pointed_at_another_broadcasts_artefact():
    """Every path in every command belongs to the game being run.

    `detect_shots.py --shots` defaults to one specific game's official shot
    times, filename and all, and the driver did not override it -- so the fourth
    broadcast's shot detector was scored against the 2025 Finals Game 7.
    """
    others = [g for k, g in registry().items() if k != "hou"]
    text = " ".join(" ".join(s.argv or []) for s in _plan("hou"))
    for game in others:
        assert game.video not in text
        assert str(game.clip_detections) not in text
        assert str(game.aligned) not in text
        assert game.game_id not in text, (
            f"a stage names game {game.game_id} while running hou")


def test_every_default_that_names_another_game_is_overridden():
    """A stage that relies on a default carrying a game id in its filename is a
    stage that scores a new broadcast against an old one."""
    import re
    for stage in _plan("hou"):
        source = Path(stage.argv[0]).read_text()
        risky = {flag for flag, value in
                 re.findall(r'add_argument\("(--[a-z0-9-]+)"[^)]*?default="([^"]*)"',
                            source, re.S)
                 if re.search(r"\d{10}", value)}
        assert risky <= set(stage.argv), (
            f"{stage.name} leaves {risky - set(stage.argv)} at a default that "
            f"names a specific game")


def test_the_vision_arm_is_fed_the_detection_cache_and_not_a_report():
    """`score_game_end_to_end.vision_shot_stream` reads `blob["frames"]` and
    re-runs detect_shots' rim tracking over them. Handed detect_shots' own
    OUTPUT instead it finds no frames, emits zero calls, and prints a vision
    arm of 0.000 with no error anywhere."""
    game = get("hou")
    score = [s for s in _plan("hou") if s.name == "score"][0]
    detections = score.argv[score.argv.index("--detections") + 1]
    assert detections.endswith(str(game.detections)), detections
    assert str(game.vision_shots) not in detections


def test_no_stage_carries_a_tuning_flag():
    """`--tune` sweeps thresholds on the game being scored. A driver that
    tuned per game would make the acceptance test pass while proving nothing."""
    for stage in _plan("hou"):
        assert "--tune" not in (stage.argv or [])


def test_a_stage_with_its_outputs_on_disk_is_skipped(tmp_path, capsys):
    made = tmp_path / "made.json"
    made.write_text("{}")
    stage = add_broadcast.Stage("x", [made], ["nothing.py"])
    row = add_broadcast.run(stage, force=False, dry=False)
    assert row["status"].startswith("skipped")
    assert "[skip]" in capsys.readouterr().out


def test_force_reruns_a_stage_whose_outputs_exist(tmp_path):
    made = tmp_path / "made.json"
    made.write_text("{}")
    stage = add_broadcast.Stage("x", [made], ["-c", "pass"])
    assert add_broadcast.run(stage, force=True, dry=True)["status"] == "would run"


def test_a_stage_that_exits_zero_without_producing_its_output_is_a_failure(tmp_path):
    """The loudest silent failure available: a script that prints a diagnostic,
    returns 0 and writes nothing, leaving the next stage to read a missing file."""
    stage = add_broadcast.Stage("x", [tmp_path / "never.json"], ["-c", "pass"])
    row = add_broadcast.run(stage, force=False, dry=False)
    assert row["status"] == "FAILED"
    assert row["missing"] == [str(tmp_path / "never.json")]


def test_the_acceptance_broadcast_is_held_out_for_the_arms_this_driver_scores():
    """A run on the acceptance game has to be able to say which of its numbers
    are hold-outs. Every arm this driver produces is an EVENT arm; the same
    broadcast is not a hold-out for court registration and must never be
    reported as one."""
    game = get("hou")
    for arm in ("clock", "alignment", "clips", "handler", "ball", "end_to_end"):
        assert game.held_out_for(arm), f"{arm} is not held out on the test game"
    assert not game.held_out_for("registration")


@pytest.mark.parametrize("key", sorted(registry()))
def test_every_registered_game_produces_a_runnable_plan(key):
    plan = _plan(key)
    assert {s.name for s in plan} >= {"clock", "align", "clips", "score", "report"}
    names = {s.name for s in plan}
    for stage in plan:
        for need in stage.needs:
            assert need in names, f"{stage.name} needs {need}, which is not a stage"
