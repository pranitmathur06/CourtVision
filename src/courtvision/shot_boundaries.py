"""Find camera cuts, so windows never straddle two unrelated scenes.

Broadcast video is cut constantly — wide play, a replay, a bench reaction, back
to play. The pipeline slides a 16-frame window with an 8-frame stride, and a
window spanning a cut contains two different scenes, which is not an action at
all. Tracking restarts across a cut too, so possession fragments.

Measured on 123 clips joined end to end, one cut every nine seconds: shot
detection collapsed to 11 of 97, where the same checkpoint finds 246 of 262 on
uncut footage. That is the cut, not the model.

Detection is a mean-absolute-difference between consecutive frames on a small
greyscale thumbnail. A cut changes almost every pixel at once; play, however
fast, does not.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Fraction of full-scale intensity that counts as a scene change.
CUT_THRESHOLD = 0.28
# A real shot lasts longer than this; anything shorter is a flash or a fade.
MIN_SEGMENT_FRAMES = 12


def _thumb(frame: np.ndarray) -> np.ndarray:
    import cv2

    grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def cut_frames(frames: Sequence[np.ndarray]) -> list[int]:
    """Indices where a new camera shot begins (never 0)."""
    if len(frames) < 2:
        return []
    cuts: list[int] = []
    previous = _thumb(frames[0])
    for index in range(1, len(frames)):
        current = _thumb(frames[index])
        if float(np.mean(np.abs(current - previous))) > CUT_THRESHOLD:
            cuts.append(index)
        previous = current
    return cuts


def segments(frame_count: int, cuts: Sequence[int]) -> list[tuple[int, int]]:
    """Half-open [start, end) spans between cuts, dropping ones too short to use.

    A segment shorter than a single classification window cannot produce one, so
    keeping it only invites windows that straddle its edges.
    """
    bounds = [0] + [c for c in cuts if 0 < c < frame_count] + [frame_count]
    spans = []
    for start, end in zip(bounds, bounds[1:]):
        if end - start >= MIN_SEGMENT_FRAMES:
            spans.append((start, end))
    return spans
