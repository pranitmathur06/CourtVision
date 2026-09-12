"""The verdict tally, on cases worked out by hand."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import score_verdicts as verdicts  # noqa: E402


def test_ok_and_miss_both_count_as_visible():
    got = {0: {"rim": "ok"}, 1: {"rim": "miss"}}
    visible, located, unknown, absent, _ = verdicts.tally(got, "rim")
    assert (visible, located, unknown, absent) == (2, 1, 0, 0)


def test_two_baskets_one_found_counts_as_two_visible_and_one_located():
    got = {0: {"rim": "ok+miss"}}
    visible, located, *_ = verdicts.tally(got, "rim")
    assert (visible, located) == (2, 1)


def test_a_dash_is_absent_and_a_question_is_neither():
    got = {0: {"rim": "-"}, 1: {"rim": "?"}}
    visible, located, unknown, absent, _ = verdicts.tally(got, "rim")
    assert (visible, located, unknown, absent) == (0, 0, 1, 1)


def test_the_ceiling_suffix_marks_a_miss_that_selection_could_have_caught():
    got = {0: {"ball": "miss+c"}, 1: {"ball": "miss"}}
    visible, located, _, _, recoverable = verdicts.tally(got, "ball")
    assert (visible, located, recoverable) == (2, 0, 1)


def test_the_ceiling_suffix_does_not_inflate_the_visible_count():
    got = {0: {"ball": "miss+c"}}
    visible, located, *_ = verdicts.tally(got, "ball")
    assert (visible, located) == (1, 0)


def test_a_missing_object_key_means_not_in_shot():
    assert verdicts.tally({0: {"ball": "ok"}}, "rim")[3] == 1


def test_parsing_keeps_the_frame_index(tmp_path):
    path = tmp_path / "v.txt"
    path.write_text("3 rim=ok ball=miss+c   # a note\n\n7 rim=- ball=-\n")
    got = verdicts.parse(path)
    assert got == {3: {"rim": "ok", "ball": "miss+c"}, 7: {"rim": "-", "ball": "-"}}
