"""Find the clock and learn its digits automatically, with no human reading it.

Per-broadcast profiles were the thing stopping this from scaling. Every network
draws its own scoreboard, so each one needed a crop and a handful of frames
whose value someone had read by eye. Two profiles existed; a third broadcast
read 0 clock samples and its games were unusable — which capped the training
set at three labelled games when steal alone needs about twenty.

Both halves can be derived from the clock's own behaviour instead.

LOCATING it: the seconds digit changes every second and nothing else on a
scoreboard does. Score, period and team name are static across a few seconds of
play. So the clock is the region that segments into digit-shaped glyphs AND
whose rightmost glyph keeps changing.

LABELLING the digits: the ones digit counts down 9,8,...,1,0,9,... so sampling
one frame per second gives a sequence that must decrease by one. That fixes the
digits RELATIVE to each other. To pin them absolutely, watch the digit to its
left: it only changes when the ones digit wraps from 0 to 9. That single
observation anchors the whole alphabet, and twenty seconds of footage shows
every digit at least once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_GLYPHS = 3
MAX_GLYPHS = 4


@dataclass(frozen=True)
class ClockLocation:
    roi: tuple[int, int, int, int]      # top, bottom, left, right
    ticks: int                          # how often the last glyph changed
    samples: int


def _glyph_signature(image: np.ndarray) -> np.ndarray:
    """Small normalised bitmap, so the same digit matches itself across frames."""
    import cv2

    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (10, 14), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32)
    spread = small.max() - small.min()
    return (small - small.min()) / spread if spread > 1e-6 else small * 0


def _same(a: np.ndarray, b: np.ndarray, tolerance: float = 0.12) -> bool:
    return bool(np.mean(np.abs(a - b)) < tolerance)


def candidate_rois(height: int, width: int) -> list[tuple[int, int, int, int]]:
    """Boxes in the lower third, where every broadcast puts its scoreboard."""
    boxes = []
    for top in range(int(height * 0.78), int(height * 0.95), 12):
        for left in range(int(width * 0.05), int(width * 0.92), 40):
            boxes.append((top, min(top + 40, height), left, min(left + 110, width)))
    return boxes


def locate_clock(frames: list[np.ndarray]) -> ClockLocation | None:
    """Pick the region that reads as digits and ticks once per frame.

    `frames` must be about one second apart — the tick is the whole signal.
    """
    from courtvision.scoreboard import clock_glyphs, normalise_polarity

    if len(frames) < 4:
        return None
    height, width = frames[0].shape[:2]
    best: ClockLocation | None = None
    for top, bottom, left, right in candidate_rois(height, width):
        last: np.ndarray | None = None
        ticks, seen = 0, 0
        for frame in frames:
            glyphs = clock_glyphs(normalise_polarity(frame[top:bottom, left:right]))
            if not MIN_GLYPHS <= len(glyphs) <= MAX_GLYPHS:
                last = None
                continue
            seen += 1
            signature = _glyph_signature(glyphs[-1].image)
            if last is not None and not _same(signature, last):
                ticks += 1
            last = signature
        # Needs to read as digits most of the time AND actually be counting.
        if seen >= max(4, len(frames) // 2) and ticks >= seen - 2:
            if best is None or ticks > best.ticks:
                best = ClockLocation((top, bottom, left, right), ticks, seen)
    return best


def bootstrap_templates(frames: list[np.ndarray],
                        roi: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    """Learn digit templates from a one-second-per-frame run, unsupervised.

    The ones digit decreases by one each second; the digit to its left changes
    only when it wraps 0 -> 9. That wrap is the anchor that turns a relative
    ordering into absolute digits.
    """
    from courtvision.scoreboard import clock_glyphs, normalise_polarity

    top, bottom, left, right = roi
    ones: list[np.ndarray | None] = []
    tens: list[np.ndarray | None] = []
    for frame in frames:
        glyphs = clock_glyphs(normalise_polarity(frame[top:bottom, left:right]))
        if not MIN_GLYPHS <= len(glyphs) <= MAX_GLYPHS:
            ones.append(None)
            tens.append(None)
            continue
        ones.append(glyphs[-1].image)
        tens.append(_glyph_signature(glyphs[-2].image))

    # Find a wrap: the tens glyph changes between consecutive readable frames.
    anchor = None
    for i in range(1, len(ones)):
        if ones[i] is None or ones[i - 1] is None:
            continue
        if tens[i] is not None and tens[i - 1] is not None and \
                not _same(tens[i], tens[i - 1]):
            anchor = i          # ones[i] is 9, ones[i-1] is 0
            break
    if anchor is None:
        return {}

    templates: dict[str, np.ndarray] = {}
    # Walk outward from the anchor; each step back adds one to the digit.
    for offset in range(0, len(ones) - anchor):
        index = anchor + offset
        if ones[index] is None:
            break
        digit = (9 - offset) % 10
        templates.setdefault(str(digit), ones[index])
    for offset in range(0, anchor + 1):
        index = anchor - 1 - offset
        if index < 0 or ones[index] is None:
            break
        digit = offset % 10
        templates.setdefault(str(digit), ones[index])
    return templates
