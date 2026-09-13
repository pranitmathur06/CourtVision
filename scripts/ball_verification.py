"""Propose widely, then ask the ball-specific model about each proposal, centred.

Thirteen selection approaches have failed on the same wall: the true ball sits
somewhere down a list of candidates and nothing in the candidate's own picture
tells it from a head. This is the fourteenth, and it is the first to move the
delivered number, because it stops trying to rank the candidates and asks a
different question about each one.

The two detectors available have opposite faults, measured on 13 hand-located
balls:

    clean whole @1280, conf 0.03    ceiling  7/13   4.6 candidates   4 top-1
    four-class tile 320 @640        ceiling 10/13   112 candidates   0 top-1
    everything pooled               ceiling 12/13   230 candidates

One is precise and half-blind; the other sees nearly everything and cannot
order it. So: take the proposals from the wide sources, and score each with the
PRECISE model run on a 640 px crop CENTRED on that proposal.

Centring is the point. `ball_clean` was trained on 640 px windows cut without
resizing, so a centred 640 px crop is exactly its training distribution, with
the candidate in the middle and most of the clutter gone -- a different and
easier question than "find the ball in this 1280x720 frame", asked once per
proposal. A proposal the ball model will not endorse at its own scale is not a
ball.

    delivered by pooled confidence   4/13   0.308
    delivered by verification        5/13   0.385

WHAT IT CANNOT DO, from the same measurement. On five frames the true ball's
verification score is 0.00: `ball_clean` does not see it even centred, at conf
0.01. Those are the loose-ball scramble, the ball held by a dribbler, and the
low wide camera. Verification can only re-order proposals the precise model is
capable of recognising, so the remaining gap is the detector's, not the
selector's, and no further selection work will touch it.
"""

from __future__ import annotations

import numpy as np

#: Proposals closer than this are the same object seen by two sources.
MERGE_PX = 14.0
#: The crop handed to the verifier, and how close its detection must land.
WINDOW_PX = 640
AGREE_PX = 12.0
VERIFY_CONF = 0.01


def merge(candidates, radius=MERGE_PX):
    """Collapse near-duplicate proposals, keeping the most confident.

    `candidates` are (x, y, confidence). Pooling four detector configurations
    proposes the same ball up to four times, and a duplicate is not evidence.
    """
    kept = []
    for candidate in sorted(candidates, key=lambda c: -c[2]):
        if all(np.hypot(candidate[0] - k[0], candidate[1] - k[1]) > radius
               for k in kept):
            kept.append(candidate)
    return kept


def crop_window(shape, point, window=WINDOW_PX):
    """Top-left of a `window` crop centred on `point`, clipped to the frame."""
    height, width = shape[:2]
    window_w, window_h = min(window, width), min(window, height)
    x0 = int(np.clip(point[0] - window_w // 2, 0, max(width - window_w, 0)))
    y0 = int(np.clip(point[1] - window_h // 2, 0, max(height - window_h, 0)))
    return x0, y0, window_w, window_h


def best_endorsement(detections, point, agree=AGREE_PX):
    """Highest confidence among `detections` landing on `point`, else 0.0.

    `detections` are (x, y, confidence) in full-frame coordinates. Silence is
    0.0 rather than None: a proposal the ball model declines to endorse has
    been scored, not skipped.
    """
    best = 0.0
    for x, y, confidence in detections:
        if np.hypot(x - point[0], y - point[1]) <= agree:
            best = max(best, float(confidence))
    return best


def ranked(candidates, scores):
    """Indices best first, ties broken by the earlier candidate."""
    return sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
