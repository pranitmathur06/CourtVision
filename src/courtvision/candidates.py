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
# How far inside the court's visible edge a player's feet must land. The court
# mask closes over gaps and so spills slightly past the real boundary; eroding
# it pulls the edge back in. Swept on live frames: at 45 px the survivors
# average 6.6 a frame, which is what a broadcast camera shows of ten players,
# while 0 px keeps 8.1 and admits the front row.
COURT_ERODE_PX = 45
FEET_ON_COURT_SHARE = 0.45


def court_region(image: np.ndarray, erode_px: int = COURT_ERODE_PX):
    """The largest connected run of floor: wood and painted court together.

    Colour cannot separate players from spectators in this arena -- the crowd
    wears the home kit's colour, so a fan in a blue shirt clusters with a
    player in a blue jersey. Position can: a player stands on the floor and a
    spectator stands beyond its edge.
    """
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    wood = (hue >= 5) & (hue <= 30) & (sat >= 40) & (val >= 110)
    paint = (hue >= 95) & (hue <= 130) & (sat >= 90) & (val >= 90)
    mask = ((wood | paint).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == biggest).astype(np.uint8)
    if erode_px:
        region = cv2.erode(region, np.ones((erode_px, erode_px), np.uint8))
    return region


def stands_on_court(region: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Mask of the boxes whose feet land inside the court region."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    keep = np.zeros(len(boxes), dtype=bool)
    if region is None:
        return keep
    height, width = region.shape[:2]
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        span = x2 - x1
        a, c = int(x1 + 0.25 * span), int(x1 + 0.75 * span)
        b = int(y2 - 4)
        d = int(min(height, y2 + 0.10 * (y2 - y1)))
        a, c = max(0, a), min(width, c)
        if c - a < 3 or d - b < 2:
            continue
        keep[i] = float(region[b:d, a:c].mean()) >= FEET_ON_COURT_SHARE
    return keep


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
