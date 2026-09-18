"""The page's totals must come from the WHOLE log, never from the retrieved rows.

This is the property the retrieval plan calls the highest-value regression guard
in the project, and it is one line of the page today:

    ALL.filter(r => r.gid === game && r.player === who).forEach(...)

`localAnswer` is handed `rows` -- what the search matched -- and for a question
about points it deliberately ignores them. The reason is in the page's own
comment: the play-by-play writes each scorer's RUNNING TOTAL in the description,
"(29 PTS)", and a filter that narrowed to made shots drops the free throws the
running total also lives on. Answer from the hits and the number is quietly low.

The moment an embedding retriever goes in, `rows` becomes a top-k list and that
is precisely when this breaks -- silently, with a plausible number. So the test
replaces the retriever with one that returns NOTHING and asserts the total is
unchanged, which is the strongest form of the claim: the answer cannot depend on
the hit list because it does not read it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent.parent / "docs" / "index.html"
#: Extracted by name and evaluated on their own, so this needs no DOM.
WANTED = ("aggregate", "namedAlready", "ordinal", "topOf", "localAnswer")


def _function(source: str, name: str) -> str:
    """The text of one top-level `function name(...) {...}`, by brace matching."""
    start = source.index(f"function {name}(")
    depth, i = 0, source.index("{", start)
    opened = False
    while i < len(source):
        if source[i] == "{":
            depth, opened = depth + 1, True
        elif source[i] == "}":
            depth -= 1
            if opened and depth == 0:
                return source[start:i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces in {name}")


@pytest.fixture(scope="module")
def page() -> str:
    assert PAGE.exists(), f"{PAGE} is the shipped page and must be here"
    return PAGE.read_text()


def _rows():
    """One scorer, whose points arrive as field goals AND free throws.

    HIS LAST POINTS ARE FREE THROWS, and that is the whole fixture. A first
    version of this put a three-pointer last, so narrowing the rows to shots
    still left the row carrying "(7 PTS)" in the hit list -- and the test passed
    with the guard deliberately broken. A test that cannot fail is not a test.
    Free throws closing out a possession is also the common case in a real game,
    which is why the page's own comment names them.
    """
    out = []
    running = 0
    for n, (action, points, detail) in enumerate([
            ("shot", 2, "Jones 14' Jump Shot"),
            ("made shot", 3, "Jones 26' 3PT Jump Shot"),
            ("free_throw", 1, "Jones Free Throw 1 of 2"),
            ("free_throw", 1, "Jones Free Throw 2 of 2"),
    ]):
        running += points
        out.append({"gid": "g", "player": "Jones", "action": action,
                    "period": 1, "clock": f"10:{59 - n:02d}",
                    "detail": f"{detail} ({running} PTS)", "clip": None})
    # ...and another player's rows, which must never leak into his total.
    out.append({"gid": "g", "player": "Smith", "action": "made shot",
                "period": 1, "clock": "9:00", "detail": "Smith Dunk (40 PTS)",
                "clip": None})
    return out


def _run(page_source: str, rows, retrieved) -> str:
    node = shutil.which("node")
    if node is None:                                   # pragma: no cover
        pytest.skip("node is not installed")
    body = "\n".join(_function(page_source, name) for name in WANTED)
    script = f"""
let ALL = {json.dumps(rows)};
let game = "g";
{body}
const rows = {json.dumps(retrieved)};
process.stdout.write(localAnswer("how many points did Jones score",
                                 rows, rows, ["Jones"]));
"""
    done = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_the_total_is_right_when_every_row_is_retrieved(page):
    out = _run(page, _rows(), _rows())
    assert "**7**" in out, out


def test_the_total_is_unchanged_when_the_retriever_returns_a_narrow_slice(page):
    """A search for "shots" drops the two free throws. The running total lives
    on those rows too, so an answer computed from the hits reads 5, not 7."""
    rows = _rows()
    narrowed = [r for r in rows if "Free Throw" not in r["detail"]]
    assert len(narrowed) < len(rows)
    out = _run(page, rows, narrowed)
    assert "**7**" in out, out
    assert "**5**" not in out, "the total was computed from the hit list"


def test_this_file_actually_catches_the_regression_it_describes(page):
    """The guard, guarded. Break the one line and the narrow-slice test must
    fail -- otherwise this whole file is decoration."""
    broken = page.replace("ALL.filter(r => r.gid === game && r.player === who)",
                          "rows.filter(r => r.player === who)")
    assert broken != page, "the line this file is about has moved"
    rows = _rows()
    narrowed = [r for r in rows if "Free Throw" not in r["detail"]]
    out = _run(broken, rows, narrowed)
    assert "**5**" in out and "**7**" not in out, (
        "answering from the hit list did not change the number, so the fixture "
        "cannot detect the failure it is written for")


def test_the_total_is_unchanged_when_the_retriever_returns_one_row(page):
    """The shape an embedding retriever produces: a top-k list, not a filter.

    This is the case the guard exists for -- it is the one that will arrive,
    and it fails silently with a plausible number."""
    rows = _rows()
    out = _run(page, rows, rows[:1])
    assert "**7**" in out, out


def test_another_players_rows_never_reach_this_total(page):
    """Reading the whole log is only safe because it is filtered by player and
    game. Dropping either would make every scorer's total the same."""
    out = _run(page, _rows(), _rows())
    assert "**40**" not in out, out


def test_the_page_still_reads_the_whole_log_for_points(page):
    """The structural half of the guard, so a refactor that keeps the tests
    passing by changing the data cannot also quietly change the source."""
    points = page[page.index("if (/\\bpoints?\\b"):]
    points = points[:points.index("if (/\\bhow many\\b")]
    assert "ALL.filter(" in points, (
        "the points branch no longer reads the whole log; it now answers from "
        "the retrieved rows, which is the failure this file exists to catch")
    assert "r.gid === game" in points and "r.player === who" in points
