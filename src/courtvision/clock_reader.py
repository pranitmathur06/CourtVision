"""Read the game clock off the scoreboard, per frame, with no assumptions.

The earlier approach integrated clock TICKS from an anchor: start from one known
value and add a second of game time for every second of video where the clock
was seen to advance. That is wrong for anything but a continuous live broadcast,
and it failed silently. On a condensed game the clock advances faster than wall
time -- between video 60 s and 900 s the game clock moved ~1,100 s in 840
seconds -- which tick counting cannot represent, since it can add at most one
second per second. The resulting map was off by ~800 s, and every event scored
against it was matched to the wrong moment.

Reading each frame independently is immune to all of it: condensed footage,
camera cuts, replays, and a video whose segments are not in game order.

The strip carries the period and the clock together -- "2nd  11:11" -- so one
region gives both: the leading digit is the period, the letters after it are the
ordinal suffix, and the digits that follow are the clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Glyph clusters recovered from the broadcast and labelled by inspection.
#
# TWO alphabets, not one. The period text is set in a smaller face than the
# clock, so a template fitted to "11:11" matches the "2" of "2nd" at a distance
# of 0.30 -- which is no match at all -- while the same digit inside the clock
# matches at 0.03. Sharing one alphabet silently returned the wrong period.
CLOCK_CLUSTERS = {0: "1", 1: "5", 2: "3", 3: "2", 4: "0",
                  5: "4", 6: "8", 7: "6", 8: "9", 9: "7"}
PERIOD_CLUSTERS = {4: "1", 5: "2", 2: "3", 1: "4"}
MATCH_TOLERANCE = 0.16


@dataclass(frozen=True)
class ClockReading:
    period: int
    seconds: float        # remaining in the period
    elapsed: float        # since tip-off, so events across periods are ordered

    @property
    def text(self) -> str:
        return f"{self.period} {int(self.seconds)//60}:{int(self.seconds)%60:02d}"


def _normalise(glyph: np.ndarray) -> np.ndarray:
    import cv2
    if glyph.ndim == 3:
        glyph = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(glyph, (12, 18)).astype(np.float32)
    spread = small.max() - small.min()
    return (small - small.min()) / (spread if spread > 1e-6 else 1.0)


def _symbol(glyph: np.ndarray, centroids: np.ndarray,
            alphabet: dict[int, str]) -> str | None:
    """The digit this glyph shows, or None if it is a letter or unreadable."""
    sample = _normalise(glyph)
    distances = np.abs(centroids - sample).mean(axis=(1, 2))
    best = int(np.argmin(distances))
    if distances[best] > MATCH_TOLERANCE:
        return None
    return alphabet.get(best)


# Digits on this scoreboard are ~15 px wide and often TOUCH: "2:44" binarises
# into one blob for the "44", which the shared segmenter drops for being too
# wide, so those digits vanish entirely and the clock reads as a single digit.
# Segmenting here instead, and splitting any blob whose width says it holds more
# than one digit, keeps them.
DIGIT_WIDTH = 15
MIN_DIGIT_HEIGHT = 10


def _digit_boxes(strip: np.ndarray):
    """Digit crops as (x, y, w, h, image), splitting blobs that merged two.

    The image returned is BINARY with the ink white, because that is how the
    templates were built; handing back a grey dark-on-light crop matches
    nothing and the whole read silently returns None.
    """
    import cv2

    grey = strip if strip.ndim == 2 else cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if binary.mean() > 127:                 # ink must be the white pixels
        binary = 255 - binary
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if h < MIN_DIGIT_HEIGHT or area < 12:
            continue
        parts = max(1, round(w / DIGIT_WIDTH))
        for part in range(parts):
            px = x + part * w // parts
            pw = w // parts
            boxes.append((px, y, pw, h, binary[y:y + h, px:px + pw]))
    return sorted(boxes, key=lambda b: b[0])


def _tenths(strip: np.ndarray, left: float, right: float) -> bool:
    """True if the separator is a decimal point rather than a colon.

    Under a minute the NBA clock shows tenths -- 18.2 -- and three digits are
    then SS.T rather than M:SS. "1:52" and "15.2" are the same three digits, so
    the separator is the only thing that tells them apart: a colon puts ink in
    the upper half of the gap, a decimal point does not.
    """
    import cv2
    if right <= left + 1:
        return False
    gap = strip[:, int(left):int(right)]
    if gap.size == 0:
        return False
    grey = gap if gap.ndim == 2 else cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)
    ink = grey > (int(grey.min()) + int(grey.max())) / 2
    if not ink.any():
        return False
    upper = ink[: ink.shape[0] // 2].sum()
    return upper == 0


def read(image: np.ndarray, profile: dict) -> ClockReading | None:
    """One frame to a clock reading, or None when the scoreboard is not shown.

    `profile` carries the two regions and their template sets, because every
    broadcast lays its scoreboard out differently.
    """
    from courtvision.scoreboard import clock_glyphs, normalise_polarity

    def glyphs_in(roi):
        top, bottom, left, right = roi
        strip = normalise_polarity(image[top:bottom, left:right])
        return strip, _digit_boxes(strip)

    _, period_glyphs = glyphs_in(profile["period_roi"])
    period = None
    for _, _, _, _, crop in period_glyphs:
        found = _symbol(crop, profile["period_centroids"], PERIOD_CLUSTERS)
        if found is not None:
            period = int(found)
            break
    if period is None or not 1 <= period <= 4:
        return None

    strip, clock = glyphs_in(profile["clock_roi"])
    digits = [(sym, box) for sym, box in
              ((_symbol(box[4], profile["clock_centroids"], CLOCK_CLUSTERS), box)
               for box in clock) if sym is not None]
    if not 3 <= len(digits) <= 4:
        return None

    values = "".join(sym for sym, _ in digits)
    if len(digits) == 4:
        if int(values[2:]) > 59:
            return None
        seconds = int(values[:2]) * 60 + int(values[2:])
    else:
        # Three digits are either M:SS or, in the last minute, SS.T. "182" is
        # 18.2 and not 1:82, because a clock has no 82nd second -- so validity
        # decides it whenever it can, and the separator only breaks genuine
        # ties like "1:52" against "15.2".
        as_clock = int(values[1:]) <= 59
        as_tenths = int(values[:2]) <= 59
        if as_clock and as_tenths:
            gap = _tenths(strip,
                          digits[1][1][0] + digits[1][1][2], digits[2][1][0])
            as_clock = not gap
        if as_clock:
            seconds = int(values[0]) * 60 + int(values[1:])
        elif as_tenths:
            seconds = int(values[:2]) + int(values[2]) / 10.0
        else:
            return None
    if not 0 <= seconds <= 720:
        return None
    return ClockReading(period=period, seconds=seconds,
                        elapsed=(period - 1) * 720 + (720 - seconds))
