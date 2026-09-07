"""Choosing which player, at which moment, to show an action classifier.

The ball-screen classifier reaches 86-94% on curated film and scored 0 of 5 on
this project's broadcast. The model was not the problem -- it was being handed
spectators, empty boxes, and moments where nothing was happening. This module
is the fix, and each guard here exists because its absence was measured.

**Only people who belong to a kit.** A person detector finds spectators,
officials and coaches, and no size or brightness rule excludes them: a
courtside spectator is as large as a player, and officials stand on the same
lit floor. What does exclude them is that the ten players wear two colours and
nobody else wears either. Clustering torso colour into two balanced groups and
keeping only cluster members drops the rest -- the same split measured at 96.3%
elsewhere in this project.

**Action-centred windows.** The training tubes are centred ON the action; a
clip sampled at an arbitrary instant catches the set-up or the aftermath of a
screen rather than the screen. Scoring several offsets and keeping the peak
restores that, which is ordinary dense action detection rather than a
concession.
"""

from __future__ import annotations

import numpy as np

# Torso colours further than this from both cluster centres belong to nobody:
# an official, a coach, somebody in the front row.
MAX_KIT_DISTANCE_LAB = 26.0
# A person smaller than this cannot be read.
MIN_BOX_HEIGHT_PX = 90


def kit_members(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Boolean mask of the boxes whose torso matches one of the two kits.

    Returns all-False when the colours do not separate into two groups, which
    is the honest answer on a frame showing a huddle, a replay or a bench.
    """
    from courtvision.tactics import torso_colours

    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) < 6:
        return np.zeros(len(boxes), dtype=bool)
    colours = torso_colours(image, boxes)
    usable = np.isfinite(colours).all(axis=1)
    if usable.sum() < 6:
        return np.zeros(len(boxes), dtype=bool)

    values = colours[usable]
    axis = int(np.argmax(values.std(axis=0)))
    order = np.argsort(values[:, axis])
    ordered = values[order]
    best = None
    for cut in range(2, len(ordered) - 1):
        left, right = ordered[:cut], ordered[cut:]
        score = (left.var(axis=0).sum() * len(left)
                 + right.var(axis=0).sum() * len(right))
        if best is None or (score, abs(len(left) - len(right))) < best[0]:
            best = ((score, abs(len(left) - len(right))), cut)
    if best is None:
        return np.zeros(len(boxes), dtype=bool)
    cut = best[1]
    centres = (ordered[:cut].mean(axis=0), ordered[cut:].mean(axis=0))
    # Two kits are further apart than either is wide, or there is only one.
    spread = (np.sqrt(ordered[:cut].var(axis=0).sum())
              + np.sqrt(ordered[cut:].var(axis=0).sum()))
    if float(np.linalg.norm(centres[0] - centres[1])) < max(12.0, spread):
        return np.zeros(len(boxes), dtype=bool)

    keep = np.zeros(len(boxes), dtype=bool)
    for index in np.flatnonzero(usable):
        gap = min(float(np.linalg.norm(colours[index] - centre))
                  for centre in centres)
        keep[index] = gap <= MAX_KIT_DISTANCE_LAB
    return keep


def near_ball(boxes: np.ndarray, ball, limit_px: float = 320.0) -> np.ndarray:
    """Mask of boxes within `limit_px` of the ball, feet to ball."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if ball is None or not len(boxes):
        return np.zeros(len(boxes), dtype=bool)
    centre_x = (boxes[:, 0] + boxes[:, 2]) / 2
    return np.hypot(centre_x - ball[0], boxes[:, 3] - ball[1]) <= limit_px


def peak_score(scores) -> float:
    """The strongest response across a set of offsets around one moment.

    A screen lasts about a second inside a possession that lasts fifteen, so a
    clip cut at an arbitrary instant usually shows the approach or the
    aftermath. The classifier was trained on windows centred on the action, and
    taking the maximum over nearby offsets is how that distribution is matched
    at inference.
    """
    values = [float(s) for s in scores if s is not None]
    return max(values) if values else 0.0
