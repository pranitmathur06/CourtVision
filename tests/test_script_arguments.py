"""Every `args.X` a script reads must be an argument it declares.

`clip_detect_raw.py` read `args.game` and had no `--game` flag. The fitted
court erosion it was wired to look up in Round 116 therefore crashed on its
first real run, and nothing noticed: the `--help` test passes because argparse
renders usage and exits BEFORE the body runs, and no test ran the body.

This reads the argparse calls and the attribute accesses out of the source and
compares them, which needs no model, no video and no GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(p for p in (ROOT / "scripts").glob("*.py")
                 if not p.name.startswith("_"))


def declared(tree: ast.AST) -> set[str]:
    """Every destination argparse will put on the namespace."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                out.add(keyword.value.value)
                break
        else:
            flags = [a.value for a in node.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            long = [f for f in flags if f.startswith("--")]
            name = (long[0] if long else flags[0] if flags else None)
            if name:
                out.add(name.lstrip("-").replace("-", "_"))
    return out


def read(tree: ast.AST) -> set[str]:
    """Every `args.X` the source reads, excluding method calls.

    Only `args`, which is this repository's convention. Watching speculative
    aliases as well produced the one false positive this found across 137
    scripts: a local dict named `options` whose `.get` was passed as a sort
    key, which is a bare method reference and not a call.
    """
    called = {node.func for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    out: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "args"
                and node not in called):
            out.add(node.attr)
    return out


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_every_argument_read_is_declared(path):
    tree = ast.parse(path.read_text())
    if "add_argument" not in path.read_text():
        pytest.skip("no argument parser")
    missing = read(tree) - declared(tree)
    assert not missing, f"{path.name} reads {sorted(missing)} but declares no such flag"
