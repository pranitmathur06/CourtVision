"""Placing official events on the video timeline."""

from __future__ import annotations

from courtvision.event_alignment import (align, coverage, elapsed_seconds)


def _readings(count: int = 100, offset: float = 500.0):
    """Video running at 1 fps, game clock counting down from 12:00 in period 1."""
    return [{"t": float(i), "elapsed": float(i), "period": 1, "sec": 720.0 - i}
            for i in range(count)] + [
        # a later period, reached after a gap the clock cannot read
        {"t": float(offset + i), "elapsed": 720.0 + float(i), "period": 2,
         "sec": 720.0 - i} for i in range(20)]


def test_elapsed_seconds_counts_from_the_start_of_the_game():
    assert elapsed_seconds(1, 720.0) == 0.0
    assert elapsed_seconds(1, 0.0) == 720.0
    assert elapsed_seconds(3, 660.0) == 1500.0


def test_align_places_an_event_at_its_video_time():
    aligned = align([(40.0, "Steal")], _readings())
    assert len(aligned) == 1
    assert aligned[0].video_s == 40.0
    assert aligned[0].error_s == 0.0
    assert aligned[0].action == "Steal"


def test_align_drops_an_event_the_clock_never_saw():
    # 300s of game time falls in the gap between the two reading runs.
    assert align([(300.0, "Block")], _readings()) == []


def test_align_uses_the_period_digit_for_a_period_boundary():
    # The start of period 2 is 720.0s elapsed; the clock cannot read the break,
    # but the period digit is on screen from video time 500.
    aligned = align([(720.0, "period")], _readings())
    assert len(aligned) == 1
    assert aligned[0].video_s == 500.0


def test_align_respects_the_tolerance():
    # Readings every 4s, so an event at 22.0 is 2s from the nearest at 20.0.
    sparse = [{"t": float(i), "elapsed": float(i), "period": 1, "sec": 720.0 - i}
              for i in range(0, 100, 4)]
    assert align([(22.0, "Foul")], sparse, tolerance_s=3.0)
    assert align([(22.0, "Foul")], sparse, tolerance_s=1.0) == []


def test_align_carries_descriptions_through():
    aligned = align([(10.0, "Steal")], _readings(),
                    descriptions=["Caruso STEAL (1 STL)"])
    assert aligned[0].description == "Caruso STEAL (1 STL)"


def test_align_returns_nothing_without_readable_clocks():
    blind = [{"t": 1.0, "elapsed": None, "period": None, "sec": None}]
    assert align([(10.0, "Steal")], blind) == []


def test_coverage_reports_per_action_totals():
    events = [(10.0, "Steal"), (20.0, "Steal"), (300.0, "Block")]
    aligned = align(events, _readings())
    report = coverage(events, aligned)
    assert report["Steal"] == (2, 2)
    assert report["Block"] == (0, 1), "the unlocatable event still counts"
