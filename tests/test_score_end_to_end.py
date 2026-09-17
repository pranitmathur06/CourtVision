"""The composed end-to-end scorer: the measurement this project deferred.

Every guard here exists because the number it produces is small, and a small
number is exactly when it is tempting to widen a tolerance, drop a class, or
lean on the live-play gate until it looks better. These tests make the only way
to raise it be improving the system.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from score_game_end_to_end import (CLASSES, NOT_A_PLAY,  # noqa: E402
                                   Call, Play, describes, free_throw_trips,
                                   match, score_mode, Stream, weighted_f1)


def play(t, kind="field_goal", made=None, points=None, label="Missed Shot"):
    return Play(video_s=t, kind=kind, label=label, made=made, points=points,
                align_error_s=0.0, description="")


def call(t, kind="field_goal", asserts=("kind",), made=None, points=None):
    return Call(video_s=t, kind=kind, asserts=frozenset(asserts), made=made,
                points=points)


def stream(calls, vision=("ball boxes",), feed=()):
    return Stream(source="test", vision_derived=vision, feed_derived=feed,
                  calls=tuple(calls))


# ---- the vocabulary and the denominator ------------------------------------

def test_a_made_three_and_a_missed_shot_are_both_field_goal_attempts():
    assert CLASSES["Made Shot (3PT)"] == CLASSES["Missed Shot"] == "field_goal"


def test_an_assist_is_not_a_play_because_it_rides_on_the_basket_it_assisted():
    """classify() appends Assist to the SAME row as the made shot.

    Counting it would score one basket twice.
    """
    assert "Assist" in NOT_A_PLAY


def test_a_substitution_is_not_a_play():
    assert "Substitution" in NOT_A_PLAY and "Timeout" in NOT_A_PLAY


def test_two_free_throws_at_one_stoppage_are_one_trip_and_two_attempts():
    """The clock is frozen for a trip, so both attempts share a video second.

    Asking a system to emit two calls a tenth of a second apart caps free-throw
    recall by construction, which Round 4 measured at 0.53.
    """
    plays = [play(100.0, "free_throw"), play(100.4, "free_throw"),
             play(400.0, "free_throw")]
    merged, attempts, trips = free_throw_trips(plays, merge_s=12.0)
    assert attempts == 3 and trips == 2 and len(merged) == 2


# ---- matching ---------------------------------------------------------------

def test_a_call_within_tolerance_of_the_same_class_matches():
    assert match([call(102.0)], [play(100.0)], 3.0) == {0: 0}


def test_a_call_four_seconds_away_does_not_match():
    assert match([call(104.0)], [play(100.0)], 3.0) == {}


def test_a_rebound_call_does_not_match_the_missed_shot_it_sits_on():
    """Coarse class equality, not proximity.

    Counting by proximity alone is what let 'shot 0.94x' mean nothing.
    """
    assert match([call(100.5, "rebound")], [play(100.0, "field_goal")], 3.0) == {}


def test_one_official_play_cannot_be_claimed_by_two_calls():
    got = match([call(100.1), call(100.2)], [play(100.0)], 3.0)
    assert len(got) == 1


def test_matching_does_not_depend_on_the_order_the_calls_arrive_in():
    """detect_shots.score and run_broadcast.match both walk in list order."""
    plays = [play(100.0), play(103.5)]
    calls = [call(100.2), call(103.4)]
    forward = match(calls, plays, 3.0)
    backward = match(list(reversed(calls)), plays, 3.0)
    assert {calls[i].video_s for i in forward} == \
           {list(reversed(calls))[i].video_s for i in backward}


def test_identity_is_never_a_matching_key():
    """Requiring a name would multiply every number by about a half.

    Rebound attribution is 45-47% against a 72% ceiling and jersey identity is
    45%, so a scorer that required names would be a jersey-OCR benchmark in an
    end-to-end costume.
    """
    named = Call(video_s=100.0, kind="field_goal", player="Nembhard")
    anonymous = Call(video_s=100.0, kind="field_goal", player=None)
    assert match([named], [play(100.0)], 3.0) == match([anonymous], [play(100.0)], 3.0)


# ---- describing -------------------------------------------------------------

def test_a_call_that_asserts_only_its_class_cannot_lose_described_precision():
    assert describes(call(100.0, asserts=("kind",)), play(100.0, made=True))


def test_a_call_that_claims_a_make_on_a_miss_is_timestamped_but_not_described():
    wrong = call(100.0, asserts=("kind", "made"), made=True)
    assert not describes(wrong, play(100.0, made=False))


# ---- the two headline numbers ----------------------------------------------

def test_a_class_the_mode_never_emits_scores_zero_and_keeps_its_full_weight():
    """The whole point. A shots-only system cannot be 100% by staying quiet."""
    plays = [play(10.0 * i, "field_goal") for i in range(4)] \
        + [play(500.0 + 10.0 * i, "rebound") for i in range(4)]
    got = score_mode(stream([call(10.0 * i) for i in range(4)]), plays,
                     label="shots only", span_s=600.0)
    assert got.per_class["rebound"]["weight"] == pytest.approx(0.5)
    assert got.per_class["rebound"]["f1"] == 0.0
    assert got.captured == pytest.approx(0.5)
    assert got.architectural_coverage == pytest.approx(0.5)


def test_precision_counts_every_emitted_call_including_the_dead_ball_ones():
    plays = [play(100.0)]
    got = score_mode(stream([call(100.0), call(300.0), call(500.0)]), plays,
                     label="noisy", span_s=600.0)
    assert got.emitted == 3 and got.timestamped == 1


def test_a_perfect_stream_on_one_class_reports_high_precision_and_low_coverage():
    """'field goals only, 0.941 at coverage 0.437' is in the record as a warning."""
    plays = [play(10.0 * i, "field_goal") for i in range(3)] \
        + [play(500.0 + 10.0 * i, "rebound") for i in range(7)]
    got = score_mode(stream([call(10.0 * i) for i in range(3)]), plays,
                     label="quiet", span_s=600.0)
    assert got.timestamped == got.emitted           # perfect precision
    assert got.captured_coverage < 0.35             # and it says almost nothing


def test_the_assertion_budget_records_how_little_a_quiet_system_claims():
    got = score_mode(stream([call(100.0, asserts=("kind",))]), [play(100.0)],
                     label="quiet", span_s=600.0)
    assert got.assertion_budget["kind"] == 1.0
    assert got.assertion_budget["made"] == 0.0


# ---- circularity ------------------------------------------------------------

def test_the_live_play_filter_removes_truth_from_the_denominator_as_well_as_calls():
    """Otherwise the filter is credited for suppressing calls in regions the
    truth could never occupy, which is the circular cell this scorer exists to
    make unprintable."""
    plays = [play(100.0), play(900.0)]
    calls = [call(100.0), call(900.0)]
    live = lambda t: t < 500.0            # noqa: E731
    got = score_mode(stream(calls), plays, label="gated", span_s=1000.0,
                     live_mask=live)
    assert got.emitted == 1, "the call outside live play is dropped"
    assert sum(v["official"] for v in got.per_class.values()) == 1, \
        "and so is the play outside it"
    assert got.per_class["field_goal"]["recall"] == 1.0


def test_a_blocked_mode_is_reported_as_blocked_and_not_as_zero():
    """vision+scoreboard has no reader in this repo. Reporting it as 0 would
    hide that the one architecture measured to reach 85% is unmeasured."""
    blocked = Stream(source="scoreboard", vision_derived=(), feed_derived=(),
                     calls=(), runnable=False, blocked_because="no readings")
    got = score_mode(blocked, [play(100.0)], label="blocked", span_s=600.0)
    assert got.captured == 0.0 and got.per_class == {}
    assert not got.stream.runnable and got.stream.blocked_because


# ---- the weighted statistic -------------------------------------------------

def test_the_weights_come_from_the_whole_game_not_from_a_resample():
    """A bootstrap draw containing no rebounds must not reweight the statistic
    it is putting an interval around."""
    weights = {"field_goal": 0.5, "rebound": 0.5}
    only_shots = weighted_f1([call(100.0)], [play(100.0)], 3.0, weights)
    assert only_shots == pytest.approx(0.5)
