"""The scoreboard sensors, including the misreads they must survive."""

from __future__ import annotations

from courtvision.scoreboard_events import (fouls, missed_shots,
                                           possession_changes, score_events,
                                           stoppages)


def test_score_events_type_by_magnitude():
    readings = [(0.0, 0, 0), (10.0, 2, 0), (20.0, 5, 0), (30.0, 5, 1)]
    events = score_events(readings)
    assert [e.points for e in events] == [2, 3, 1]
    assert [e.action for e in events] == [
        "two_point_make", "three_point_make", "free_throw"]
    assert [e.team for e in events] == ["home", "home", "away"]


def test_score_events_ignore_a_falling_score():
    # A digit misread is as likely to fall as to rise; monotonicity is free.
    readings = [(0.0, 10, 8), (1.0, 4, 8), (2.0, 12, 8)]
    events = score_events(readings)
    assert [e.points for e in events] == [2]


def test_score_events_reject_an_impossible_jump():
    readings = [(0.0, 0, 0), (1.0, 9, 0)]
    assert score_events(readings) == []


def test_score_events_skip_missing_readings():
    readings = [(0.0, 0, 0), (1.0, None, 0), (2.0, 2, 0)]
    assert [e.points for e in score_events(readings)] == [2]


def test_score_events_ignore_both_sides_moving():
    readings = [(0.0, 0, 0), (1.0, 2, 2)]
    assert score_events(readings) == []


def test_stoppages_need_both_clocks():
    # video time advances 4s while game time holds at 100.0
    readings = [(0.0, 100.0), (1.0, 100.0), (2.0, 100.0),
                (3.0, 100.0), (4.0, 100.0), (5.0, 99.0)]
    found = stoppages(readings)
    assert len(found) == 1
    assert found[0][0] == 100.0


def test_stoppages_ignore_a_coverage_gap():
    # game time frozen but the reads are 9s apart: nothing was observed between
    readings = [(0.0, 100.0), (9.0, 100.0)]
    assert stoppages(readings) == []


def test_stoppages_merge_a_foul_and_its_free_throws():
    readings = []
    # three separate stalls, each 3s, two seconds apart
    time = 0.0
    for _ in range(3):
        for _ in range(4):
            readings.append((time, 200.0))
            time += 1.0
        time += 2.0
    merged = stoppages(readings, merge_s=30.0)
    assert len(merged) == 1, "one foul must not read as three stoppages"


def test_possession_changes_only_count_upward_jumps():
    # A real reset jumps from a low value back to 24. A 22 -> 24 step is a
    # misread flicker, not a possession, which is why min_jump_s exists.
    readings = [(0.0, 8.0), (1.0, 6.0), (2.0, 5.0), (3.0, 24.0), (4.0, 23.0)]
    assert possession_changes(readings) == [3.0]


def test_possession_changes_reset_across_a_missing_read():
    readings = [(0.0, 5.0), (1.0, None), (2.0, 24.0)]
    assert possession_changes(readings) == []


def test_missed_shots_are_possessions_without_points():
    changes = [10.0, 40.0, 70.0]
    makes = [41.0]
    assert missed_shots(changes, makes) == [10.0, 70.0]


def test_fouls_need_a_following_free_throw():
    stops = [(10.0, 9.0), (50.0, 9.0)]
    assert fouls(stops, free_throws=[14.0]) == [10.0]
    # a free throw BEFORE the stoppage does not explain it
    assert fouls(stops, free_throws=[9.0]) == []


def test_a_gap_moves_the_baseline_instead_of_killing_the_rest_of_the_game():
    """A jump too big for one possession is SEVERAL, seen across a hidden panel.

    Nothing can be attributed for it, so nothing is emitted -- but the baseline
    has to move, or every later reading is measured against a score the game
    left behind. The first time this module was given readings from a real
    sweep it emitted ZERO events from 4,330 of them: one early gap pinned the
    baseline and 4,254 readings were discarded as implausible jumps.
    """
    from courtvision.scoreboard_events import score_events

    readings = [(0.0, 0, 0), (10.0, 2, 0),       # a normal basket
                (20.0, 12, 0),                    # a hidden stretch, +10
                (30.0, 14, 0), (40.0, 17, 0)]     # and play resumes
    events = score_events(readings)
    assert [e.points for e in events] == [2, 2, 3]
    assert events[-1].elapsed_s == 40.0


def test_both_sides_rising_at_once_also_moves_the_baseline():
    from courtvision.scoreboard_events import score_events

    readings = [(0.0, 0, 0), (10.0, 3, 2), (20.0, 5, 2)]
    events = score_events(readings)
    assert [e.points for e in events] == [2]


# -- the score-region search, which was a hand-fitted constant twice ----------

def _band_candidates(*args, **kwargs):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from read_scoreboard import band_candidates
    return band_candidates(*args, **kwargs)


#: The clock region the reader measured on the two 720p Finals broadcasts, and
#: on the 1080p regular-season one. Both are real, from `outputs/clock/*.json`.
FINALS_720P = (627, 671, 806, 916)          # 110 wide, 44 tall
HOUSTON_1080P = (996, 1024, 777, 887)       # 110 wide, 28 tall


def test_the_box_sizes_land_within_two_pixels_of_the_fitted_ones():
    """1.6, 2.0 and 2.5 clock-heights are 70, 88 and 110 on a 44-pixel clock.

    The constants they replace were 70, 90 and 110, so this is NOT exact -- the
    middle box is 2 px narrower -- and a first draft of the round that made this
    change claimed it was. What it has to be is close enough that the broadcast
    the constants were measured on still gets boxes around the same digits.

    And the constants were never right for all three 720p broadcasts anyway: the
    ECF clock region is 36 px tall, not 44, so a fixed 70-90-110 was already
    describing a graphic that broadcast does not have."""
    boxes = _band_candidates(FINALS_720P, 1280)
    # A SUPERSET: the three fitted widths are all here, with larger ones above
    # them, because the Houston score panels need a window bigger than anything
    # the fitted sizes could build.
    assert {70, 88, 110} <= set(r[3] - r[2] for r in boxes)
    assert {44, 56, 68} <= set(r[1] - r[0] for r in boxes)
    # The 36-px ECF clock, which the old constants did not fit.
    ecf = _band_candidates((595, 631, 1046, 1186), 1280)
    assert sorted({r[3] - r[2] for r in ecf}) == [58, 72, 90, 126, 162]


def test_the_search_follows_the_clock_onto_a_broadcast_it_was_not_fitted_to():
    """On the 1080p encode the locator's clock region is 28 px and the score
    digits are 38, measured off the frame at t=2000.

    From a 28-pixel clock the old grows of 0, 6 and 12 build boxes at most 52 px
    tall but only 45 to 110 wide at fixed widths; the ratios follow the clock
    instead, giving heights up to 56 and widths from 45."""
    boxes = _band_candidates(HOUSTON_1080P, 1920)
    assert max(r[1] - r[0] for r in boxes) >= 38
    assert sorted({r[3] - r[2] for r in boxes}) == [45, 56, 70, 98, 126]
    # And a window that actually contains the OKC score panel exists: the
    # digits sit at roughly x 215-335, y 982-1038 on the frame at t=2000 s, and
    # a hand-cut crop of exactly those pixels reads 18. With the fitted sizes
    # alone the search returned ZERO score-like regions on this broadcast.
    assert any(r[0] <= 982 and r[1] >= 1038 and r[2] <= 215 and r[3] >= 335
               for r in boxes), "no candidate can contain the score panel"


def test_the_reach_is_not_a_number_fitted_to_the_layouts_already_here():
    """`locate_scores` ships 260 px and misses the far team by 64 px on a 720p
    Finals broadcast. 460 was measured to fix that and misses the far team by
    67 px on Houston, whose far score sits 527 px from the clock. The reach was
    never doing the work -- behaviour over a whole game picks the regions -- so
    it searches to the frame edge."""
    for roi, width in ((FINALS_720P, 1280), (HOUSTON_1080P, 1920)):
        boxes = _band_candidates(roi, width)
        assert min(r[2] for r in boxes) == 0, "the search stops short of the edge"
        # ...and in particular it reaches past both fitted constants.
        assert min(r[2] for r in boxes) < roi[2] - 460


def test_every_candidate_sits_left_of_the_clock_and_inside_the_frame():
    for roi, width in ((FINALS_720P, 1280), (HOUSTON_1080P, 1920)):
        for top, bottom, left, right in _band_candidates(roi, width):
            assert 0 <= left < right <= roi[2], "a candidate overlaps the clock"
            assert 0 <= top < bottom


# -- a score never loses a digit ---------------------------------------------

def _sb():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import read_scoreboard
    return read_scoreboard


def test_a_single_digit_after_a_double_is_a_failed_segmentation():
    """Both Houston panels read the correct final scores -- 111 and 91 against
    an official 111-91 -- and both were thrown away, at rank correlations of
    0.752 and 0.457 against a 0.85 gate. The sequences rise cleanly apart from
    frames that segment as a single "1", where the score digits fail to separate
    and another glyph on the panel wins the same-height vote."""
    sb = _sb()
    okc = [0, 10, 17, 18, 26, 36, 36, 50, 1, 59, 66, 73, 77, 4, 92, 101, 111]
    hou = [0, 9, 1, 13, 15, 22, 30, 34, 1, 43, 1, 2, 58, 62, 67, 72, 1, 1, 83,
           1, 83, 91]
    for values, before, after in ((okc, 0.752, 0.999), (hou, 0.457, 0.996)):
        kept = sb.digits_never_shrink(values)
        seen = [v for v in kept if v is not None]
        order = [i for i, v in enumerate(kept) if v is not None]
        assert abs(sb.rank_correlation(list(range(len(values))), values)
                   - before) < 0.01
        assert abs(sb.rank_correlation(order, seen) - after) < 0.01
        assert sb.rank_correlation(order, seen) >= sb.MIN_RANK_CORRELATION


def test_a_one_digit_score_early_in_the_game_is_kept():
    """A score really is one digit until somebody reaches ten, so the rule is
    "never FEWER than already seen", not "always two"."""
    sb = _sb()
    assert sb.digits_never_shrink([0, 2, 5, 9, 11, 13]) == [0, 2, 5, 9, 11, 13]


def test_the_filter_does_not_rescue_a_shot_clock():
    """It resets to 24, so no amount of dropping readings makes it rise."""
    sb = _sb()
    clock = [24, 18, 7, 24, 14, 24, 9, 21, 24, 3, 24, 16, 24, 11, 24, 24, 8, 19]
    kept = sb.digits_never_shrink(clock)
    seen = [v for v in kept if v is not None]
    order = [i for i, v in enumerate(kept) if v is not None]
    assert sb.rank_correlation(order, seen) < sb.MIN_RANK_CORRELATION


def test_monotone_share_is_reported_and_never_gates():
    """A region reading a CONSTANT is trivially non-decreasing and scores 1.000.
    Measured on Finals G1, a 0.70 gate on it admitted 79 regions with a
    plausible final score where rank correlation admits 8."""
    sb = _sb()
    assert sb.monotone_share([10] * 20) == 1.0
    assert not hasattr(sb, "MIN_MONOTONE_SHARE"), (
        "monotone_share is a diagnostic, not a threshold")
    source = (Path(__file__).resolve().parent.parent
              / "scripts" / "read_scoreboard.py").read_text()
    assert "if rho < MIN_RANK_CORRELATION:" in source


from pathlib import Path  # noqa: E402


def test_a_stray_wide_reading_does_not_ratchet_the_digit_rule():
    """One spurious three-digit read must not discard every later two-digit one.

    That is the same failure as the clock's tenths capture and the score's
    monotonic ratchet, arrived at from a third direction: a constraint adopted
    from a single observation and never released. On Finals G1 it turned a
    111-110 game into 445-235."""
    sb = _sb()
    assert sb.digits_never_shrink([12, 14, 445, 16, 18, 20]) == [12, 14, 445, 16, 18, 20]
    # ...and a real crossing into three digits, seen twice, does stick.
    kept = sb.digits_never_shrink([88, 95, 99, 101, 104, 9, 107])
    assert kept[:5] == [88, 95, 99, 101, 104]
    assert kept[5] is None, "a one-digit read after three is a segmentation miss"


def test_candidates_are_judged_on_their_snapped_region():
    """Widening the sizes so a coloured team panel could fit let a 154x110 box
    win on Finals G1 -- one spanning several numbers, which read 445 and beat
    the correct region on the span tie-break precisely because it OVER-reads.
    Judging a candidate on the digits inside it removes the incentive."""
    source = (Path(__file__).resolve().parent.parent
              / "scripts" / "read_scoreboard.py").read_text()
    block = source[source.index("def pick_scores("):source.index("def find_shot_clock(")]
    assert "read_region(frame, roi, reader)" in block
    assert "digits_of(frame[top:bottom" not in block
