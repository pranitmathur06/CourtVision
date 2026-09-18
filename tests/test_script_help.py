"""Every script's --help must render.

A literal `%` in an argparse help string is a format specifier, and argparse
only discovers that when it renders the help -- so `--help` raised TypeError
while every other invocation worked. The help text of this project's scripts
carries measured percentages routinely, so this is a trap it will keep walking
into.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Scripts whose --help is cheap to render: no model is loaded at import.
SCRIPTS = [
    "eval_kits.py",
    "eval_play_events.py",
    "eval_ball_physics.py",
    "eval_ball_moving.py",
    "fit_court_mask.py",
    "remask_detections.py",
    "compare_masks.py",
    "clip_detect_raw.py",
    "eval_court_mask.py",
    "eval_by_game.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_renders(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        capture_output=True, text=True, timeout=180,
        cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage" in result.stdout.lower()
