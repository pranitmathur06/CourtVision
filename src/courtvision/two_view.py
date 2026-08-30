"""Two views of a window, because one crop cannot answer both questions.

The action classifier reads a crop centred on the ball-handler. That is the
right view for ball-proximate actions — dribble 0.90, block 0.86, shot 0.87,
because the ball is inside the crop. It is the wrong view for a rebound, which
is the ball coming off the rim with players converging on it: the event happens
outside the crop, so `rebound` became that model's label for anything generic
and claimed 82% of a game while scoring 0.78 on held-out clips.

Switching everything to full frames was not available — SpaceJam ships
pre-cropped clips with no recoverable source, so dribble and pass can only ever
be crops. So rebound is decided by a second, narrow model that sees the whole
frame (0.860 on balanced held-out windows), and the crop model's own rebound
output is discarded rather than trusted.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

REBOUND = "rebound"


class TwoViewClassifier:
    """Crop model for the action; full-frame model for whether it is a rebound.

    `rim_threshold` is the probability the rim model must assign before a window
    is called a rebound. It exists because the two errors are not symmetric: a
    false rebound is a fabricated event in the commentary, while a missed one is
    a silence. The default leans against fabricating.
    """

    # classify_windows checks this before handing over full frames.
    wants_full_frames = True

    def __init__(self, crop_classifier, rim_classifier,
                 rim_threshold: float = 0.5) -> None:
        self._crop = crop_classifier
        self._rim = rim_classifier
        self._threshold = rim_threshold

    def classify_batch(
        self,
        crops: Sequence[np.ndarray],
        frames: Sequence[np.ndarray] | None = None,
    ) -> list[tuple[str, float]]:
        crop_results = self._crop.classify_batch(crops)
        if frames is None or self._rim is None:
            return crop_results

        rim_results = self._rim.classify_batch(frames)
        merged: list[tuple[str, float]] = []
        for (label, conf), (rim_label, rim_conf) in zip(crop_results, rim_results):
            is_rebound = rim_label == REBOUND and rim_conf >= self._threshold
            if is_rebound:
                merged.append((REBOUND, rim_conf))
            elif label == REBOUND:
                # The crop model says rebound and the view that can actually see
                # one disagrees. Do not fall through to its second choice: that
                # is the same unreliable signal one rank down.
                merged.append(("background", rim_conf))
            else:
                merged.append((label, conf))
        return merged
