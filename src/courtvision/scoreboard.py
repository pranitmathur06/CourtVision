"""Read the game clock off the broadcast scoreboard.

This is step 3 of wiring a real game (docs/v3-live-data.md). The naming layer
already works on archived BARD clips because their filenames carry the GameID,
so the play-by-play join key is free. Arbitrary broadcast footage has no such
key: the only way to line video up with official play-by-play is the clock on
screen.

No OCR dependency. The digits are large, bold, and near-black on a bright bar,
which template matching handles: threshold, take connected components, keep the
tallest cluster (the clock digits are noticeably taller than the period label
beside them), and match each against a template.

Templates are per-broadcast, because every network draws its own scoreboard.
`build_templates` makes them from one frame whose value you know.

The reader validates itself without any labelling: a game clock only ever
counts DOWN. Read a run of frames and a correct reader produces a
non-increasing sequence, while a broken one produces noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_DIGIT_HEIGHT = 12
MAX_DIGIT_HEIGHT = 45
TALL_CLUSTER_TOLERANCE = 4      # px; separates clock digits from the period label
# Widest a glyph may be relative to its height. `w > h` looked reasonable and is
# wrong: it was calibrated on one broadcast's narrow font and rejected every
# digit on a second, where a bolder face renders 19 px wide against 17 tall.
# It still rejects the scoreboard's own outline, which is three times wider
# than tall.
MAX_ASPECT = 1.3


@dataclass(frozen=True)
class Glyph:
    """One segmented character, with where it sat."""

    x: int
    y: int
    width: int
    height: int
    image: np.ndarray           # binary, digit white on black


def normalise_polarity(roi: np.ndarray) -> np.ndarray:
    """Return the crop with dark digits on a bright ground, whichever it began as.

    This module was written against a scoreboard drawn dark-on-bright. TNT draws
    the opposite — white digits on a black bar — and against that the segmenter
    returned zero glyphs from a crop where the clock is plainly legible, which
    reads as "no scoreboard here" rather than "wrong polarity".

    Deciding by the median rather than a fixed threshold: a bar that is mostly
    dark holds bright digits and needs inverting, and vice versa.
    """
    import cv2

    grey = roi if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_not(roi) if float(np.median(grey)) < 128 else roi


def segment_glyphs(roi: np.ndarray) -> list[Glyph]:
    """Every character-shaped blob in the scoreboard crop, left to right."""
    import cv2

    grey = roi if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    glyphs: list[Glyph] = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not (MIN_DIGIT_HEIGHT <= h <= MAX_DIGIT_HEIGHT):
            continue
        if w < 3 or w > MAX_ASPECT * h or area < 30:
            continue
        glyphs.append(Glyph(int(x), int(y), int(w), int(h),
                            binary[y:y + h, x:x + w]))
    return sorted(glyphs, key=lambda g: g.x)


def clock_glyphs(roi: np.ndarray) -> list[Glyph]:
    """Just the clock digits: the tallest cluster.

    A broadcast scoreboard puts the period ("1ST") next to the clock in smaller
    type — 16 px against 22 in the sample — so height separates them without
    needing to know the layout.
    """
    glyphs = segment_glyphs(roi)
    if not glyphs:
        return []
    tallest = max(g.height for g in glyphs)
    return [g for g in glyphs if g.height >= tallest - TALL_CLUSTER_TOLERANCE]


def build_templates(roi: np.ndarray, reading: str) -> dict[str, np.ndarray]:
    """Templates from one frame whose clock value you already know.

    `reading` is the digits in order, e.g. "1059" for 10:59. Every network draws
    its own scoreboard, so templates do not transfer between broadcasts.

    Polarity is normalised here for the same reason `read_clock` does it: ESPN
    draws white on dark, and without this the segmenter found zero digits in a
    crop that plainly shows 6:36 — reported as "check the crop", which sent me
    looking at coordinates that were already correct.
    """
    glyphs = clock_glyphs(normalise_polarity(roi))
    if len(glyphs) != len(reading):
        raise ValueError(
            f"segmented {len(glyphs)} clock digits but was told {len(reading)}: "
            f"{reading!r}. Check the crop before trusting any template from it.")
    return {digit: glyph.image for digit, glyph in zip(reading, glyphs)}


def build_templates_from_many(
    samples: "list[tuple[np.ndarray, str]]",
) -> dict[str, np.ndarray]:
    """Merge templates from several frames, each with its known clock value.

    One frame cannot cover ten digits, and the gap is not cosmetic. Built from a
    single frame reading 3:47, the reader could only ever read clock values made
    of 3, 4 and 7 — and worse, it did not fail silently on the rest: it matched a
    7 against the 3 template and returned a confident 3:43. Six frames of the
    holdout clip came back wrong that way.

    Later samples do not overwrite earlier ones, so the first clear rendering of
    a digit wins.
    """
    templates: dict[str, np.ndarray] = {}
    for roi, reading in samples:
        for digit, image in build_templates(roi, reading).items():
            templates.setdefault(digit, image)
    return templates


def _match(image: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, float]:
    import cv2

    best, best_score = "?", -1.0
    for digit, template in templates.items():
        resized = cv2.resize(image, (template.shape[1], template.shape[0]))
        score = float(cv2.matchTemplate(resized, template,
                                        cv2.TM_CCOEFF_NORMED)[0][0])
        if score > best_score:
            best, best_score = digit, score
    return best, best_score


def read_clock(
    roi: np.ndarray, templates: dict[str, np.ndarray], min_score: float = 0.5,
    allow_tenths: bool = False,
) -> tuple[str | None, float]:
    """Read the clock as MM:SS (or M:SS), with the weakest digit's score.

    Returns (None, score) when a digit cannot be matched confidently. An
    unreadable clock is normal — replays, timeouts, graphics over the bar — and
    a wrong time silently mis-joins every play that follows it.

    `allow_tenths` ALSO READS THE LAST TEN SECONDS OF A PERIOD, which this
    function has never been able to see. Under a minute the NBA clock shows
    tenths, and at "18.2" that is three glyphs — which this reads as "1:82" and
    the caller's parser already turns back into 18.2 seconds. At "7.2" it is
    TWO, and two glyphs failed the length gate here and returned None. Measured
    across the four broadcasts in `data/games.json`: readings with the clock
    between 10 and 20 seconds, 79 / 34 / 342 / 25; readings under 10 seconds,
    0 / 0 / 0 / 9. The reader is blind to the end of every period, and 39% of
    every unaligned official event in all four games sits in that blind spot.

    Off by default because three other callers assume MM:SS, and a two-glyph
    read of a clock half-covered by a graphic is a plausible misread rather than
    a plausible time — it is safe here only because `read_game_clock.resolve`
    makes an unconfirmed reading prove itself against the next frame.
    """
    glyphs = clock_glyphs(normalise_polarity(roi))
    if allow_tenths and len(glyphs) == 2:
        digits, scores = [], []
        for glyph in glyphs:
            digit, score = _match(glyph.image, templates)
            digits.append(digit)
            scores.append(score)
        weakest = min(scores)
        if weakest < min_score or "?" in digits:
            return None, weakest
        return f"{digits[0]}:{digits[1]}", weakest
    if not 3 <= len(glyphs) <= 4:
        return None, 0.0
    digits, scores = [], []
    for glyph in glyphs:
        digit, score = _match(glyph.image, templates)
        digits.append(digit)
        scores.append(score)
    weakest = min(scores)
    if weakest < min_score or "?" in digits:
        return None, weakest
    text = "".join(digits)
    return f"{text[:-2]}:{text[-2:]}", weakest


def clock_to_seconds(clock: str) -> int:
    """MM:SS to seconds remaining in the period."""
    minutes, seconds = clock.split(":")
    return int(minutes) * 60 + int(seconds)
