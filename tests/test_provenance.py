"""A result must name the commit of the code that produced it."""

import subprocess
from pathlib import Path

import pytest

from courtvision.provenance import code_provenance


def test_the_main_checkout_reports_its_own_head():
    import courtvision.court_refine as court_refine
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert code_provenance(court_refine.__file__)["commit"] == head


def test_a_module_in_a_worktree_reports_the_worktree_commit_not_the_cwd():
    """The defect: an old-code run from a worktree recorded the main HEAD."""
    worktrees = subprocess.run(["git", "worktree", "list", "--porcelain"],
                               capture_output=True, text=True).stdout
    paths = [line.split(" ", 1)[1] for line in worktrees.splitlines()
             if line.startswith("worktree ")][1:]
    candidates = [Path(p) / "src/courtvision/court_refine.py" for p in paths]
    candidates = [c for c in candidates if c.exists()]
    if not candidates:
        pytest.skip("no linked worktree with the package")
    module = candidates[0]
    expected = subprocess.run(["git", "-C", str(module.parent), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    main = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    got = code_provenance(str(module))
    assert got["commit"] == expected
    if expected != main:
        assert got["commit"] != main, "must not report the cwd repository's HEAD"
