"""Choosing the court erosion against two bounds the sport supplies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "fit_court_mask", ROOT / "scripts" / "fit_court_mask.py")
fit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit)


def _rows(pairs, gate=None, fill=True):
    """{(erode, gate, fill): {carrier_kept, over_ok}}, denominators fine."""
    return {(share, gate, fill): {"carrier_kept": carrier, "carrier_n": 100,
                                  "over_ok": over, "over_n": 100}
            for share, (carrier, over) in pairs.items()}


def test_it_takes_the_most_carriers_among_masks_that_obey_the_rule():
    rows = _rows({0.0: (0.98, 0.80), 0.03: (0.90, 0.96), 0.0625: (0.85, 0.99)})
    assert fit.choose(rows) == (0.03, None, True)


def test_a_mask_that_admits_the_crowd_is_refused_however_many_carriers_it_keeps():
    """Keeping everybody never drops the carrier. One bound alone is
    degenerate, which is why there are two."""
    rows = _rows({0.0: (1.00, 0.10), 0.0625: (0.60, 0.99)})
    assert fit.choose(rows) == (0.0625, None, True)


def test_nothing_clearing_the_floor_answers_none_rather_than_guessing():
    rows = _rows({0.0: (0.9, 0.5), 0.03: (0.8, 0.6)})
    assert fit.choose(rows) is None


def test_a_tie_on_carriers_goes_to_the_larger_erosion():
    """Admitting the front row costs more downstream than it costs here."""
    rows = _rows({0.0: (0.90, 0.99), 0.03: (0.90, 0.99)})
    assert fit.choose(rows) == (0.03, None, True)


def test_an_empty_denominator_is_not_treated_as_a_pass():
    rows = {(0.0, None, True): {"carrier_kept": 1.0, "carrier_n": 0,
                                "over_ok": 1.0, "over_n": 0}}
    assert fit.choose(rows) is None


def test_the_kit_gate_is_the_second_way_to_exclude_the_front_row():
    """Erosion removes the front row and the baseline corner together. The kit
    gate removes only people wearing neither kit, so where both clear the
    over-keeping floor the tighter gate is preferred to the blunt one."""
    rows = {}
    rows.update(_rows({0.0: (0.90, 0.96)}, gate=None))
    rows.update(_rows({0.0: (0.90, 0.99)}, gate=26.0))
    assert fit.choose(rows) == (0.0, 26.0, True)


def test_a_kit_gate_that_deletes_players_is_refused_like_any_other_mask():
    rows = {}
    rows.update(_rows({0.0: (0.95, 0.96)}, gate=None))
    rows.update(_rows({0.0: (0.40, 1.00)}, gate=18.0))
    assert fit.choose(rows) == (0.0, None, True)


def test_not_filling_the_floor_is_preferred_when_it_ties():
    """A filled mask is a larger mask and admits more of the front row, so on
    a tie the smaller claim wins -- and whether a broadcast needs filling is
    decided by these two bounds, not by a belief about courts."""
    rows = {}
    rows.update(_rows({0.0: (0.90, 0.99)}, fill=True))
    rows.update(_rows({0.0: (0.90, 0.99)}, fill=False))
    assert fit.choose(rows) == (0.0, None, False)


def test_filling_wins_when_it_actually_keeps_more_carriers():
    rows = {}
    rows.update(_rows({0.0: (0.95, 0.99)}, fill=True))
    rows.update(_rows({0.0: (0.80, 0.99)}, fill=False))
    assert fit.choose(rows) == (0.0, None, True)


def test_the_floor_and_the_impossible_count_are_the_declared_ones():
    assert fit.OVER_FLOOR == 0.95
    assert fit.IMPOSSIBLE_ABOVE == 13


def test_two_detectors_on_one_player_count_once_here_too():
    same = [[100.0, 100.0, 200.0, 400.0], [104.0, 98.0, 198.0, 402.0]]
    assert fit.distinct(same) == 1
    assert fit.distinct(same + [[600.0, 100.0, 700.0, 400.0]]) == 2


def test_the_floor_is_scored_against_the_rows_it_is_reused_for():
    """The pipeline finds the floor once and applies it while the camera pans.
    Scoring only the frame it was found on measures a mask nothing ever
    applies -- and does so unevenly, since a tighter mask has less margin and
    the same pan pushes more feet outside it."""
    import inspect

    source = inspect.getsource(fit.measure)
    assert "rows[start:start + COURT_EVERY]" in source
    assert fit.COURT_EVERY == 3


def test_the_fit_reads_the_broadcast_and_not_the_clips():
    """A floor taken from the 854x480 clip scores 0.836 kept ball carrier where
    the same setting on the source scores 0.866, so a fit run on clips chooses
    between settings by a number three points below the one that ships."""
    import inspect

    source = inspect.getsource(fit.measure)
    assert "broadcast.video" in source
    assert "CAP_PROP_POS_MSEC" in source
    # Not even for the torso colour: the source frame is already decoded here.
    assert "clip_dir" not in inspect.getsource(fit)


def test_the_seek_is_copied_from_the_pipeline_not_computed():
    """round(start_s * fps) + f is wrong by up to 24 frames: a POS_MSEC seek
    lands where ffmpeg landed when it cut the clip, and the arithmetic does
    not."""
    import inspect

    source = inspect.getsource(fit.measure)
    assert "round(start" not in source
    assert "position * step" in inspect.getdoc(fit.measure)


def test_the_boxes_stay_in_source_pixels():
    """Scaling them to the clip puts every player's feet in the corner of a
    1280x720 mask, which reads as the mask having got tighter."""
    import inspect

    source = inspect.getsource(fit._score_row)
    assert "people = [[b[2], b[3], b[4], b[5]]" in source


def test_the_carrier_is_found_by_the_one_shared_definition():
    """Five copies of this loop is how `wilson` ended up with four."""
    import inspect

    assert "carrier_of" in inspect.getsource(fit._score_row)


def test_choose_reads_the_fit_half_only():
    """The rule must not see the half it is reported on. Without this the fit
    proposed a setting for Finals G7 that cost twelve points of ball carrier
    over the whole broadcast, and the sample it was chosen on could not see
    it."""
    import inspect

    source = inspect.getsource(fit.main)
    assert 'choose(got["shares"]["fit"])' in source
    assert 'choose(got["shares"]["report"])' not in source


def test_the_split_is_by_clip_and_not_by_frame():
    """Frames within a clip are the same camera on the same possession, so a
    frame split puts near-duplicates on both sides and the report half
    confirms whatever the fit half chose."""
    import inspect

    source = inspect.getsource(fit.measure)
    assert 'half = "fit" if index % 2 == 0 else "report"' in source


def test_both_halves_are_measured_for_every_setting():
    import inspect

    source = inspect.getsource(fit.measure)
    assert 'for half in ("fit", "report")' in source


def test_a_choice_near_the_floor_is_flagged_as_unsettleable():
    """The fit is wrong by 0.8 points of over-keeping where the mask is loose
    and 6.6 points of carrier where it is tight, so its estimate is least
    trustworthy exactly where the constraint binds."""
    import inspect

    source = inspect.getsource(fit.main)
    assert "TOO_CLOSE_TO_CALL" in source
    assert fit.TOO_CLOSE_TO_CALL == 0.02
    # 0.951 against a floor of 0.95 is inside the margin; 0.978 is not.
    assert 0.951 - fit.OVER_FLOOR < fit.TOO_CLOSE_TO_CALL
    assert 0.978 - fit.OVER_FLOOR > fit.TOO_CLOSE_TO_CALL
