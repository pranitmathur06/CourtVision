"""Recomputing a cached mask without rerunning the detector."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "remask_detections", ROOT / "scripts" / "remask_detections.py")
remask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(remask)


def test_it_refuses_to_overwrite_the_cache_it_reads(monkeypatch, capsys):
    """Writing over the input would make the before-and-after unmeasurable,
    which is the only reason to run this at all."""
    from courtvision.games import get

    broadcast = get("hou")
    target = str(ROOT / broadcast.clip_detections)
    monkeypatch.setattr(sys, "argv",
                        ["remask_detections.py", "--game", "hou",
                         "--out", target, "--erode-share", "0.0"])
    assert remask.main() == 2
    assert "refusing" in capsys.readouterr().out


def test_the_erosion_it_used_is_recorded_in_the_file_it_writes(tmp_path):
    """A cache that does not say which mask built it cannot be compared with
    another one."""
    source = inspect_source()
    assert 'cache["mask_erode_share"] = erode_share' in source
    assert 'cache["mask_rebuilt_from"]' in source


def inspect_source() -> str:
    return (ROOT / "scripts" / "remask_detections.py").read_text()


def test_it_rewrites_only_the_mask(tmp_path):
    """The detections cost a GPU pass and must survive untouched; only `on`
    is recomputed."""
    source = inspect_source()
    body = source[source.index("def remask"):source.index("def main")]
    assert 'row["on"] = keep' in body
    assert 'row["d"]' in body
    assert 'row["d"] =' not in body


def test_one_clip_remasks_and_records_what_built_it(tmp_path):
    """An integration check on real footage: the mask it writes lines up with
    the boxes it is indexed against, and the file says which mask made it."""
    import json as _json

    from courtvision.games import get

    broadcast = get("hou")
    if not (ROOT / broadcast.clip_detections).exists():
        import pytest

        pytest.skip("no cached detections for hou")
    out = tmp_path / "one.json"
    got = remask.remask("hou", 0.0, out, limit=1, kit_max_lab=None)
    assert got["frames"] > 0
    blob = _json.loads(out.read_text())
    assert blob["mask_erode_share"] == 0.0
    assert blob["mask_kit_max_lab"] is None
    assert blob["mask_rebuilt_from"].endswith("clip_detections_hou.json")
    name = sorted(blob["clips"])[0]
    for row in blob["clips"][name]:
        people = [b for b in row["d"] if b[0] in ("p", "h")]
        assert len(row["on"]) == len(people)
        assert all(isinstance(flag, bool) for flag in row["on"])


def test_the_detections_themselves_survive_untouched(tmp_path):
    """The boxes cost a pass over the broadcast; only `on` may change."""
    import json as _json

    from courtvision.games import get

    broadcast = get("hou")
    if not (ROOT / broadcast.clip_detections).exists():
        import pytest

        pytest.skip("no cached detections for hou")
    before = _json.loads((ROOT / broadcast.clip_detections).read_text())
    out = tmp_path / "two.json"
    remask.remask("hou", 0.0, out, limit=1, kit_max_lab=None)
    after = _json.loads(out.read_text())
    name = sorted(after["clips"])[0]
    for old, new in zip(before["clips"][name], after["clips"][name]):
        assert old["d"] == new["d"]
        assert old["f"] == new["f"]
