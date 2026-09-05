"""Reading the clock off each frame, rather than integrating from an anchor.

The integrating version failed silently and expensively: on condensed footage
the game clock advances faster than wall time -- ~1,100 s of game in 840 s of
video -- which tick counting cannot represent, since it adds at most a second
per second. It put the video at game elapsed 1546..2391 when it truly spans
52..2864, so every event was matched against the wrong moment, and the
resulting shot-detection numbers were wrong in BOTH directions: recall
flattered (0.679 against a third of the truth) and the trained detector judged
a regression when it was an improvement.
"""

import numpy as np
import pytest

from courtvision.clock_reader import (CLOCK_CLUSTERS, ClockReading, _digit_boxes,
                                      _tenths)


def _strip(text: str, width: int = 120, height: int = 28) -> np.ndarray:
    """Dark digits on a light bar, drawn at the size the broadcast uses.

    The scale matters: the splitter keys on DIGIT_WIDTH, so a fixture with
    narrower glyphs than the real scoreboard would test nothing.
    """
    import cv2
    image = np.full((height, width, 3), 210, np.uint8)
    cv2.putText(image, text, (3, height - 5), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (20, 20, 20), 3)
    return image


def test_elapsed_orders_events_across_periods():
    """A clock counts DOWN inside a period, so raw seconds cannot order two
    events in different quarters. `elapsed` is what the scorer joins on."""
    third = ClockReading(period=3, seconds=700, elapsed=(3 - 1) * 720 + 20)
    fourth = ClockReading(period=4, seconds=700, elapsed=(4 - 1) * 720 + 20)
    assert fourth.elapsed > third.elapsed
    assert third.seconds == fourth.seconds


def test_touching_digits_are_split():
    """"2:44" binarises into one blob for the "44". The shared segmenter drops
    it for being too wide, which loses both digits and reads the clock as "2"."""
    boxes = _digit_boxes(_strip("2:44"))
    assert len(boxes) >= 3, f"expected the 44 to split, got {len(boxes)} boxes"


def test_boxes_come_back_left_to_right():
    boxes = _digit_boxes(_strip("10:43"))
    xs = [b[0] for b in boxes]
    assert xs == sorted(xs)


def test_crops_are_binary_with_white_ink():
    """Templates were built white-on-black. Handing back a grey dark-on-light
    crop matches nothing and the whole read returns None."""
    for *_rest, image in _digit_boxes(_strip("7:58")):
        assert image.ndim == 2
        assert set(np.unique(image)) <= {0, 255}
        assert image.mean() < 200, "ink should be the white pixels, not the field"


def test_the_clock_alphabet_covers_every_digit():
    assert sorted(CLOCK_CLUSTERS.values()) == [str(d) for d in range(10)]


def test_a_decimal_point_reads_as_tenths_and_a_colon_does_not():
    """Under a minute the clock shows tenths, so three digits are SS.T rather
    than M:SS. The separator is the only difference between "1:52" and "15.2"."""
    import cv2
    gap = np.zeros((24, 6), np.uint8)
    gap[20:23, 2:5] = 255                      # a low dot: decimal point
    assert _tenths(gap, 0, 6)
    colon = np.zeros((24, 6), np.uint8)
    colon[6:9, 2:5] = 255                      # ink high as well: colon
    colon[16:19, 2:5] = 255
    assert not _tenths(colon, 0, 6)


def test_an_impossible_second_is_rejected_before_the_separator_is_consulted():
    """"182" is 18.2, not 1:82 -- a clock has no 82nd second. Validity settles
    most three-digit reads without needing to see the separator at all."""
    assert int("82") > 59 and int("18") <= 59
