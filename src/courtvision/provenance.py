"""Which code produced a result -- asked of the code, not of the working directory.

An evaluation run from the main repository but importing an older copy of the
package from a git worktree recorded the main repository's HEAD (efae531) in
its dump, although every line of refinement it ran came from 6d302e4. Anyone
reading that dump later would have credited the old code's result to the new
code. The commit has to come from the repository the imported module lives in.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def code_provenance(module_file: str) -> dict:
    """Path, commit and dirty flag of the checkout containing `module_file`."""
    folder = str(Path(module_file).resolve().parent)

    def git(*args):
        return subprocess.run(["git", "-C", folder, *args],
                              capture_output=True, text=True).stdout.strip()

    return {"module_path": str(Path(module_file).resolve()),
            "commit": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain"))}
