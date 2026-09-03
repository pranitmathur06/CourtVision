"""Synthetic scoreboards, so the logic is pinned without needing a broadcast."""

import numpy as np
import pytest

from courtvision.autoscoreboard import (MAX_SCORE_RATE, bootstrap_templates,
                                        candidate_rois, locate_clock)

def _digit(value: int) -> np.ndarray:
    """A 20x12 glyph that is ONE connected component and clearly distinct.

    Two fixture bugs preceded this. Seven-segment bars split each digit into two
    8px halves, below the 12px floor segment_glyphs applies. Then holes of 2x4
    pixels made neighbouring digits indistinguishable once downscaled — real
    broadcast digits differ far more than that, so the fixture was
    understating the signal rather than the code missing it.
    """
    cell = np.full((20, 12), 255, np.uint8)
    bits = value % 10
    # Large, well-separated cut-outs, kept inside a 2px border so the glyph
    # cannot be split into pieces.
    if bits & 1:
        cell[3:9, 3:9] = 0
    if bits & 2:
        cell[11:17, 3:9] = 0
    if bits & 4:
        cell[3:17, 3:5] = 0
    if bits & 8:
        cell[3:17, 7:9] = 0
    return cell


def _frame(minutes: int, seconds: int, height=720, width=1280,
           origin=(600, 900)) -> np.ndarray:
    """White digits on a dark bar, like a real broadcast."""
    frame = np.full((height, width, 3), 30, np.uint8)
    top, left = origin
    frame[top - 4:top + 26, left - 6:left + 60] = 0
    digits = f"{minutes}{seconds:02d}"
    for i, ch in enumerate(digits):
        glyph = _digit(int(ch))
        x = left + i * 18
        frame[top:top + 20, x:x + 12] = np.dstack([glyph] * 3)
    return frame


def _run(start_minutes: int, start_seconds: int, count: int) -> list[np.ndarray]:
    frames, m, s = [], start_minutes, start_seconds
    for _ in range(count):
        frames.append(_frame(m, s))
        s -= 1
        if s < 0:
            s, m = 59, m - 1
    return frames


def test_candidate_rois_cover_the_lower_third_only():
    boxes = candidate_rois(720, 1280)
    assert boxes
    assert all(top >= 720 * 0.7 for top, _, _, _ in boxes), \
        "scoreboards live low; searching the whole frame wastes time and invites"
    assert all(right <= 1280 for _, _, _, right in boxes)


def test_locates_the_ticking_region_and_not_a_static_one():
    """A score does not change every second; a clock does. That is the signal."""
    frames = _run(8, 35, 12)
    found = locate_clock(frames)
    assert found is not None, "should find a clock that ticks every frame"
    top, bottom, left, right = found.roi
    # The clock was drawn at (600, 900); the box must contain it.
    assert top <= 600 <= bottom and left <= 900 <= right


def test_returns_nothing_when_the_screen_is_frozen():
    """No tick means no clock — a still frame must not yield a false location."""
    frames = [_frame(8, 35) for _ in range(10)]
    assert locate_clock(frames) is None


def test_bootstrap_labels_digits_absolutely_using_the_wrap():
    """8:02 -> 8:01 -> 8:00 -> 7:59 pins 0 and 9 without anyone reading them."""
    frames = _run(8, 12, 16)
    found = locate_clock(frames)
    assert found is not None
    templates = bootstrap_templates(frames, found.roi)
    assert templates, "a run crossing a minute boundary must anchor"
    assert "9" in templates and "0" in templates


def test_bootstrap_gives_up_without_a_wrap_rather_than_guessing():
    """Without the tens digit changing there is no absolute anchor.

    Returning a relative-only alphabet would silently mislabel every clock read
    that followed, which is worse than reading nothing.
    """
    frames = _run(8, 55, 6)          # 8:55 -> 8:50, no wrap
    found = locate_clock(frames)
    if found is not None:
        assert bootstrap_templates(frames, found.roi) == {}


def _score_frame(score: int, height=720, width=1280, origin=(600, 900)):
    """A two-digit score in the same style, so it reads as glyphs."""
    frame = np.full((height, width, 3), 30, np.uint8)
    top, left = origin
    frame[top - 4:top + 26, left - 6:left + 46] = 0
    for i, ch in enumerate(f"{score:02d}"):
        glyph = _digit(int(ch))
        x = left + i * 18
        frame[top:top + 20, x:x + 12] = np.dstack([glyph] * 3)
    return frame


def test_the_ticking_part_of_a_clock_is_never_read_as_a_score():
    """Rate is the discriminator, so the seconds digit -- which changes on
    almost every sample -- must fall outside the score band.

    It does NOT follow that a clock is safe from being read as a score. The
    MINUTES digit changes about once a minute, which is a score's rate, and
    this returns it. Rate alone cannot separate them; something that knows a
    score only ever increases would be needed, and that is not built.
    """
    from courtvision.autoscoreboard import locate_scores
    frames = _run(5, 30, 40)
    for found in locate_scores(frames):
        assert found.ticks / found.samples <= MAX_SCORE_RATE
        assert found.ticks < len(frames) // 2, \
            "a region ticking every second is the clock, not a score"


def test_a_score_that_ticks_up_occasionally_is_found():
    from courtvision.autoscoreboard import locate_scores
    frames, value = [], 40
    for i in range(40):
        frames.append(_score_frame(value))
        if i % 10 == 9:                      # scores every ten samples
            value += 2
    found = locate_scores(frames)
    assert found, "a rarely-changing digit region should read as a score"
    assert any(0.01 <= c.ticks / c.samples <= 0.20 for c in found)


def test_static_text_is_not_a_score():
    """A team abbreviation reads as glyphs but never changes, and a region that
    never changes carries no information about scoring."""
    from courtvision.autoscoreboard import locate_scores
    assert locate_scores([_score_frame(77) for _ in range(40)]) == []


def test_score_change_times_fires_only_when_the_score_moves():
    """What matters for the make/miss lever is that a reported change is real:
    a false one marks a miss as a basket. Whether every change is caught
    depends on how cleanly the region is framed, so this pins the direction
    that costs precision, not perfect recall."""
    from courtvision.autoscoreboard import locate_scores, score_change_times
    frames, times, value = [], [], 40
    for i in range(40):
        frames.append(_score_frame(value))
        times.append(float(i))
        if i % 10 == 9:
            value += 2
    truth = {10.0, 20.0, 30.0}
    roi = locate_scores(frames)[0].roi
    changes = score_change_times(frames, times, roi)
    assert changes, "no change detected at all"
    assert set(changes) <= truth, f"reported a change when the score held: {changes}"
