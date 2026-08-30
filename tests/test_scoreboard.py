"""Reading the game clock off a broadcast scoreboard."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from courtvision.scoreboard import (
    build_templates,
    clock_glyphs,
    clock_to_seconds,
    read_clock,
    segment_glyphs,
)


def render_bar(clock_text: str, period: str = "1ST") -> np.ndarray:
    """A scoreboard bar: small period label, large clock, dark on bright."""
    bar = np.full((60, 220, 3), 240, np.uint8)
    cv2.putText(bar, period, (8, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(bar, clock_text, (70, 45), cv2.FONT_HERSHEY_DUPLEX, 1.3, (15, 15, 15), 3)
    return bar


def test_clock_digits_are_separated_from_the_period_label_by_height():
    """A scoreboard puts the period in smaller type beside the clock."""
    bar = render_bar("10:59")
    everything = segment_glyphs(bar)
    clock = clock_glyphs(bar)
    assert len(clock) == 4, [g.height for g in everything]
    assert len(everything) > len(clock), "the period label should also segment"
    # Compare by position: a Glyph holds a numpy image, so `in` on the
    # dataclass raises "truth value of an array is ambiguous".
    clock_xs = {g.x for g in clock}
    label = [g for g in everything if g.x not in clock_xs]
    assert label, "expected the period label to segment separately"
    assert min(g.height for g in clock) > max(g.height for g in label)


def test_round_trip_a_known_clock():
    bar = render_bar("10:59")
    templates = build_templates(bar, "1059")
    assert sorted(templates) == ["0", "1", "5", "9"]
    text, score = read_clock(bar, templates)
    assert text == "10:59" and score > 0.9


def test_a_digit_with_no_template_is_refused_not_guessed():
    """An unreadable clock is normal; a wrong one mis-joins every later play."""
    templates = build_templates(render_bar("10:59"), "1059")
    text, _ = read_clock(render_bar("10:34"), templates)   # 3 and 4 unseen
    assert text is None


def test_build_templates_rejects_a_mismatched_reading():
    """Wrong labels would poison every later read, so refuse loudly."""
    bar = render_bar("10:59")
    with pytest.raises(ValueError, match="segmented 4 clock digits"):
        build_templates(bar, "105")


def test_a_blank_bar_reads_nothing():
    blank = np.full((60, 220, 3), 240, np.uint8)
    templates = build_templates(render_bar("10:59"), "1059")
    assert read_clock(blank, templates)[0] is None


@pytest.mark.parametrize("text, seconds", [("10:59", 659), ("0:07", 7), ("12:00", 720)])
def test_clock_to_seconds(text, seconds):
    assert clock_to_seconds(text) == seconds


def test_single_digit_minute_clock_is_read():
    """Under ten minutes the clock loses a digit; three glyphs must still work."""
    templates = build_templates(render_bar("10:59"), "1059")
    text, _ = read_clock(render_bar("9:15"), templates)
    assert text == "9:15"
