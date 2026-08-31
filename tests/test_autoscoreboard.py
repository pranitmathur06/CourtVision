"""Synthetic scoreboards, so the logic is pinned without needing a broadcast."""

import numpy as np
import pytest

from courtvision.autoscoreboard import (bootstrap_templates, candidate_rois,
                                        locate_clock)

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
