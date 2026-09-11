"""Every `from courtvision... import name` in src/ and scripts/ resolves.

Most imports in this project are lazy -- inside the function that needs them --
so a removed name raises ImportError only when that function runs. ee94b99
rewrote court_lines and left eight import statements naming things it no
longer defined, and no test reached any of them. This reads every import statically,
so a caller nobody exercises is still checked.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Broken by ee94b99 and owned by the work reworking registration. Strict xfail
# fails the moment a script's imports resolve, so delete its entry then.
KNOWN_BROKEN = {
    "scripts/register_broadcast.py":
        "imports court_lines.search_camera and friends, removed in ee94b99",
    "scripts/validate_registration.py":
        "imports court_lines.search_camera and friends, removed in ee94b99",
}


def _sources() -> list[str]:
    paths = [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]
    return sorted(str(p.relative_to(ROOT)) for p in paths)


def _unresolved(relative: str) -> list[str]:
    path = ROOT / relative
    parts = Path(relative).with_suffix("").parts
    package = ".".join(parts[1:-1]) if parts[0] == "src" else None
    missing = []
    for node in ast.walk(ast.parse(path.read_text(), filename=relative)):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            module = importlib.util.resolve_name(
                "." * node.level + (node.module or ""), package)
        elif node.module and node.module.split(".")[0] == "courtvision":
            module = node.module
        else:
            continue
        loaded = importlib.import_module(module)
        for alias in node.names:
            if hasattr(loaded, alias.name):
                continue
            try:    # `from courtvision import submodule`
                importlib.import_module(f"{module}.{alias.name}")
            except ModuleNotFoundError:
                missing.append(f"line {node.lineno}: {module}.{alias.name}")
    return missing


@pytest.mark.parametrize("relative", [
    pytest.param(path, marks=pytest.mark.xfail(strict=True,
                                               reason=KNOWN_BROKEN[path]))
    if path in KNOWN_BROKEN else path
    for path in _sources()
])
def test_courtvision_imports_resolve(relative):
    assert _unresolved(relative) == []


def test_known_broken_entries_still_exist():
    assert set(KNOWN_BROKEN) <= set(_sources())
