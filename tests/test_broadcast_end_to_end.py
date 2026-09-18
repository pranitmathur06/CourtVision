"""The seam: what the driver writes is what the report reads.

Every other test here checks one script. This builds a whole small broadcast --
a registry entry, a clock read, an alignment, a clip index, a cached
play-by-play -- and drives the report over it, because the failure this guards
against is not inside either script. It is a stage writing one shape and the
next reading another, which is exactly what `--detections` was doing between
`detect_shots.py` and `score_game_end_to_end.py`: a valid file, a valid flag, a
silent zero.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def broadcast(tmp_path):
    """A registry with one game, and the artefacts a real run would leave."""
    video = tmp_path / "tiny.mp4"
    video.write_bytes(b"not really a video")
    registry = tmp_path / "games.json"
    registry.write_text(json.dumps({"tiny": {
        "game_id": "0099900001", "label": "A Tiny Broadcast",
        "video": str(video), "prefix": "z",
        "held_out": {"alignment": True, "registration": False},
        "held_out_note": "made up by a test",
        "published": {
            "clock": str(tmp_path / "clock.json"),
            "aligned": str(tmp_path / "aligned.json"),
            "clip_dir": str(tmp_path / "clips"),
            "clip_index": str(tmp_path / "clips" / "index.json"),
        }}}))

    # A clock read: one reading a second across two minutes of the first period.
    (tmp_path / "clock.json").write_text(json.dumps({
        "video": str(video), "roi": [0, 20, 0, 40], "step": 1.0, "raw": [],
        "readings": [{"t": float(t), "period": 1, "seconds": 720.0 - t,
                      "elapsed": float(t)} for t in range(120)]}))

    # An alignment: nine of ten rows located.
    (tmp_path / "aligned.json").write_text(json.dumps({
        "game_id": "0099900001", "overall_rate": 0.9,
        "per_action": {"Made Shot (2PT)": {"located": 9, "total": 10}},
        "events": [{"elapsed_s": float(t), "video_s": float(t),
                    "action": "Made Shot (2PT)", "error_s": 0.0,
                    "description": "Somebody 14' Jump Shot"}
                   for t in range(9)]}))

    # A clip index cut from that alignment, plus one row that was not.
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "index.json").write_text(json.dumps({"clips": [
        {"clip": f"z{t:06d}.mp4", "video_s": float(t), "start_s": float(t) - 3,
         "action": "Made Shot (2PT)", "error_s": 0.0} for t in range(9)]
        + [{"clip": "z000099.mp4", "video_s": 99.0, "start_s": 96.0,
            "action": "Made Shot (2PT)", "error_s": 0.5}]}))

    Path("data/pbp_cache").mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "pbp_cache" / "0099900001.json"
    cache.write_text(json.dumps([
        {"period": 1, "clock": "PT12M00.00S", "actionType": "Made Shot",
         "description": "Somebody 14' Jump Shot", "subType": "Jump Shot"}]))
    yield registry
    cache.unlink(missing_ok=True)


def _report(registry, *extra):
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_by_game.py"),
         "--game", "tiny", "--registry", str(registry), "--no-registration",
         *extra],
        capture_output=True, text=True, cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_a_broadcast_assembled_from_stage_outputs_produces_a_report(broadcast):
    text = _report(broadcast)
    assert "A Tiny Broadcast" in text
    assert "alignment: overall" in text
    # 9 of 10 located.
    assert "0.900" in text


def test_the_report_says_which_arms_are_a_hold_out_and_which_are_not(broadcast):
    text = _report(broadcast)
    assert "HELD OUT for: alignment" in text
    assert "NOT held out for: registration" in text
    assert "must not be quoted as any" in text


def test_a_clip_cut_from_an_alignment_that_has_since_changed_is_caught(broadcast):
    """The row at 99 s is in the index and not in the alignment. Before this arm
    existed, Finals G7's clips and its alignment were 33% apart and two rows of
    one report were computed against two different truths with nothing saying
    so."""
    text = _report(broadcast)
    line = [ln for ln in text.splitlines()
            if "still match the alignment" in ln][0]
    assert "0.900" in line, line


def test_the_labelled_arms_say_no_data_rather_than_zero(broadcast):
    """A broadcast nobody has labelled must not read as one the stack failed."""
    text = _report(broadcast)
    for arm in ("handler: detector drew him", "ball: top-1"):
        line = [ln for ln in text.splitlines() if ln.strip().startswith(arm)][0]
        assert "NO DATA" in line and "0.00-1.00" in line, line


def test_the_report_writes_a_json_whose_arms_carry_their_denominators(broadcast, tmp_path):
    out = tmp_path / "report.json"
    _report(broadcast, "--out", str(out))
    payload = json.loads(out.read_text())
    assert payload["game"] == "tiny"
    assert payload["held_out"] == ["alignment"]
    by_name = {a["name"]: a for a in payload["arms"]}
    overall = by_name["alignment: overall"]
    assert (overall["hits"], overall["total"]) == (9, 10)
    assert overall["low"] < overall["rate"] < overall["high"]
    for arm in payload["arms"]:
        if arm["total"]:
            assert arm["hits"] <= arm["total"], arm["name"]


def test_a_sixty_hertz_broadcast_is_sampled_at_the_same_rate_as_a_thirty():
    """`clip_detect_raw.STEP = 2` is in FRAMES: 15 Hz at 30 fps and 30 Hz at 60.

    A fourth broadcast would have got twice the temporal resolution of the
    other three, at twice the cost, and a cache that is not comparable with
    theirs -- the same mistake `TrackerConfig` exists to stop making with
    `TRACK_MAX_AGE = 15`.
    """
    import clip_detect_raw
    for fps, expected in ((29.97, 2), (30.0, 2), (59.94, 4), (60.0, 4)):
        step = max(1, int(round(fps / clip_detect_raw.RATE)))
        assert step == expected, f"{fps} fps -> step {step}"
        assert abs(fps / step - 15.0) < 0.1
    # And this is why it has to be 15: `clip_boxes.assemble` hands the detected
    # sequence straight to `motion_tracking.track()`, which defaults to
    # TUNED_FPS and converts every threshold from seconds at that rate. A 30 Hz
    # detected sequence would make `max_age_s = 1.0` mean half a second and the
    # tracker a different algorithm, with nothing announcing it.
    from courtvision.motion_tracking import TUNED_FPS
    assert clip_detect_raw.RATE == TUNED_FPS


def test_the_labelling_pages_place_a_frame_at_its_real_time():
    """Both pages computed `start_s + f / 30.0` with the literal 30.

    On a 60 fps broadcast that puts every frame at twice its real time, so a
    labelled instant would name a moment in the middle of the next play and
    every label placed on a fourth broadcast would be attached to the wrong
    picture."""
    for name in ("make_handler_label_page.py", "make_possession_label_page.py"):
        source = (ROOT / "scripts" / name).read_text()
        assert 'row["f"] / 30.0' not in source, f"{name} still divides by 30"
        assert 'row["f"] / rate' in source
        assert 'cached.get("fps")' in source
