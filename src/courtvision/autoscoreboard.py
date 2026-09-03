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

# A clock ticks only while play is live -- about a third of wall time.
MIN_TICK_RATE = 0.25
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


# One box shape cannot frame both a MM:SS clock and a two-digit score, and a
# box that is too wide swallows whatever sits beside it. On a real broadcast a
# 110 px box starting 40 px to the left of the clock took in the shot-clock
# panel and segmented to a single glyph instead of three; the same coarseness
# left the score crop containing a team logo. Several widths and a finer step
# make a clean frame something the search finds rather than something it has to
# be lucky about.
ROI_HEIGHTS = (28, 36, 44)
ROI_WIDTHS = (60, 85, 110, 140)


def candidate_rois(height: int, width: int) -> list[tuple[int, int, int, int]]:
    """Boxes in the lower third, where every broadcast puts its scoreboard."""
    boxes = []
    for box_height in ROI_HEIGHTS:
        for top in range(int(height * 0.76), int(height * 0.95), 8):
            bottom = top + box_height
            if bottom > height:
                continue
            for box_width in ROI_WIDTHS:
                for left in range(int(width * 0.03), int(width * 0.92), 16):
                    right = left + box_width
                    if right > width:
                        continue
                    boxes.append((top, bottom, left, right))
    return boxes


def _reads_as_digits(frames, roi, low, high, probe=6) -> bool:
    """Cheap rejection: does this box segment into the right glyph count?

    The finer grid is many thousands of boxes, and scoring each one over every
    frame is not affordable. Nearly all of them fail on the first few frames.
    """
    from courtvision.scoreboard import clock_glyphs, normalise_polarity

    top, bottom, left, right = roi
    ok = 0
    for frame in frames[:probe]:
        glyphs = clock_glyphs(normalise_polarity(frame[top:bottom, left:right]))
        if low <= len(glyphs) <= high:
            ok += 1
    return ok >= probe - 1


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
        if not _reads_as_digits(frames, (top, bottom, left, right),
                                MIN_GLYPHS, MAX_GLYPHS):
            continue
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
        #
        # It must NOT require a tick every sampled second. A game clock advances
        # only during live play, which is roughly a third of broadcast wall time
        # -- dead balls, fouls, free throws and timeouts all hold it. Measured on
        # a real TSN broadcast, the correctly aligned clock ticks 13 times in 37
        # legible samples (0.35); demanding `ticks >= seen - 2` rejected it and
        # every other region, which is why this module found nothing on real
        # footage while passing on synthetic scoreboards that always tick.
        #
        # Rate still discriminates, which is the part that matters: a box offset
        # left reads the TENS-of-seconds digit and ticks at 0.23, one in ten, so
        # ranking by rate prefers the correctly aligned box.
        if seen >= max(4, len(frames) // 2) and ticks >= MIN_TICK_RATE * seen:
            rate = ticks / seen
            if best is None or rate > best.ticks / max(best.samples, 1):
                best = ClockLocation((top, bottom, left, right), ticks, seen)
    return best


# The make/miss lever does not need the score's VALUE, only the fact that it
# changed: a shot that scores is followed within a second or two by the score
# ticking up. That is change detection on a small region, not OCR, and it is
# far more robust than reading digits.
#
# Score and clock are told apart by how often they change. Sampled at 1 Hz on a
# real broadcast the game clock's last glyph changes about 0.35 of the time (it
# only advances during live play) and the shot clock similarly; a team's score
# changes roughly every thirtieth sample. Static text -- team abbreviations,
# the network bug -- reads as glyphs but never changes at all, so a rate floor
# excludes it.
MIN_SCORE_RATE = 0.01
MAX_SCORE_RATE = 0.20
# Two or three digits, never one. A lone digit changing about once a minute is
# the clock's MINUTES, which sits in the same rate band as a score and was
# genuinely returned as one; a team's score is in double figures for all but
# the opening minutes, so requiring two digits separates them without needing
# to read either.
MIN_SCORE_GLYPHS = 2
MAX_SCORE_GLYPHS = 3


def _never_returns(frames, roi, tolerance: float = 0.12) -> bool:
    """True if the region never goes back to an appearance it already had.

    A score only ever increases, so its rendered digits never repeat: 42 comes
    once and never again. Static text that merely flickers -- a team crest
    beside an abbreviation, an anti-aliased edge -- alternates between two
    appearances and returns constantly. That is the difference the rate band
    alone could not see, and on a real broadcast "CHI" was outranking every
    genuine score without it.

    The signature is the whole crop rather than the last glyph, because a
    score's LAST digit does repeat: 40, 42 ... 50 shows a ones digit of 0 twice.

    NOT USED BY `locate_scores`, and the reason is worth keeping. Applied to a
    real broadcast it rejected all 167 candidates rather than narrowing them:
    video compression makes a crop wobble slightly between frames, so at a 12x8
    signature almost every region "returns" to an earlier appearance sooner or
    later. The idea is sound and the implementation is too brittle for
    compressed footage; it needs a tolerance fitted to the codec's noise, or a
    comparison over the digits alone rather than the whole crop.
    """
    import cv2

    from courtvision.scoreboard import normalise_polarity

    seen: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for frame in frames:
        top, bottom, left, right = roi
        crop = normalise_polarity(frame[top:bottom, left:right])
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(crop, (12, 8)).astype(np.float32)
        spread = small.max() - small.min()
        small = (small - small.min()) / (spread if spread > 1e-6 else 1.0)
        if previous is not None and _same(small, previous, tolerance):
            continue                      # unchanged, nothing to check
        if any(_same(small, old, tolerance) for old in seen):
            return False                  # came back to an earlier appearance
        seen.append(small)
        previous = small
    return True


def locate_scores(frames: list[np.ndarray],
                  min_rate: float = MIN_SCORE_RATE,
                  max_rate: float = MAX_SCORE_RATE) -> list[ClockLocation]:
    """Regions that read as digits and change rarely -- the two team scores.

    `frames` should be about one second apart. Returns the candidates by
    increasing change rate; a scoreboard shows two, one per team.
    """
    from courtvision.scoreboard import clock_glyphs, normalise_polarity

    if len(frames) < 8:
        return []
    height, width = frames[0].shape[:2]
    found: list[ClockLocation] = []
    for top, bottom, left, right in candidate_rois(height, width):
        if not _reads_as_digits(frames, (top, bottom, left, right),
                                MIN_SCORE_GLYPHS, MAX_SCORE_GLYPHS):
            continue
        last: np.ndarray | None = None
        changes, seen = 0, 0
        for frame in frames:
            glyphs = clock_glyphs(normalise_polarity(frame[top:bottom, left:right]))
            if not MIN_SCORE_GLYPHS <= len(glyphs) <= MAX_SCORE_GLYPHS:
                last = None
                continue
            seen += 1
            signature = _glyph_signature(glyphs[-1].image)
            if last is not None and not _same(signature, last):
                changes += 1
            last = signature
        if seen < max(6, len(frames) * 3 // 4):
            continue
        rate = changes / seen
        if min_rate <= rate <= max_rate:
            found.append(ClockLocation((top, bottom, left, right), changes, seen))
    # Most changes first. Sorting the other way ranks the most STATIC regions
    # top, which is how a team abbreviation whose segmentation flickers came
    # first on a real broadcast ahead of any score.
    return sorted(found, key=lambda c: -c.ticks / max(c.samples, 1))


def score_change_times(frames: list[np.ndarray], times: list[float],
                       roi: tuple[int, int, int, int]) -> list[float]:
    """When the digits in `roi` changed. One entry per change, at its time."""
    from courtvision.scoreboard import clock_glyphs, normalise_polarity

    top, bottom, left, right = roi
    out: list[float] = []
    last: np.ndarray | None = None
    for frame, time_s in zip(frames, times):
        glyphs = clock_glyphs(normalise_polarity(frame[top:bottom, left:right]))
        if not MIN_SCORE_GLYPHS <= len(glyphs) <= MAX_SCORE_GLYPHS:
            continue
        signature = _glyph_signature(glyphs[-1].image)
        if last is not None and not _same(signature, last):
            out.append(time_s)
        last = signature
    return out


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
