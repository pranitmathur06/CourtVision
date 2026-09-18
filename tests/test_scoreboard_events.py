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


def test_the_box_sizes_reproduce_the_fitted_ones_on_the_footage_they_were_fitted_to():
    """1.6, 2.0 and 2.5 clock-heights are 70, 88 and 110 on a 44-pixel clock.

    The constants they replace were 70, 90 and 110. Expressed as ratios they
    have to land back on the same boxes for the broadcast they were measured on,
    or the change is not a generalisation but a different search."""
    boxes = _band_candidates(FINALS_720P, 1280)
    assert sorted({r[3] - r[2] for r in boxes}) == [70, 88, 110]
    assert sorted({r[1] - r[0] for r in boxes}) == [44, 56, 68, 88]


def test_the_search_follows_the_clock_onto_a_broadcast_it_was_not_fitted_to():
    """On the 1080p encode the clock is 28 px tall and the score digits are 52.

    Every fixed box size here was too short for them. The ratios give boxes up
    to 56 px tall, which is the only reason the digits fit inside one."""
    boxes = _band_candidates(HOUSTON_1080P, 1920)
    assert max(r[1] - r[0] for r in boxes) >= 52
    assert sorted({r[3] - r[2] for r in boxes}) == [45, 56, 70]


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
