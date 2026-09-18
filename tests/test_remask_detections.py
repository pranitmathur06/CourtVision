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
