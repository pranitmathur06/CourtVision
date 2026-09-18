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
#: The same erosion as a share of FRAME HEIGHT. 45 px was swept on a 720p
#: broadcast, where it is 6.25% of the height; on a 1080p one the same 45 px is
#: 4.2%, so the constant silently means two different things on two files. A
#: share means the same thing on both, and it is what `fit_court_mask.py`
#: chooses per broadcast.
COURT_ERODE_SHARE = 45 / 720
#: The mask is computed at this frame height whatever the input's, then scaled
#: back. Without it the 25x25 closing and the erosion are different physical
#: distances on a 720p and a 1080p broadcast, so a constant swept on one is
#: meaningless on the other and nothing measured on a downscaled clip transfers
#: to the pipeline's full-resolution frame. Everything here is a shape
#: operation, so the mask loses nothing by being found small and scaled up.
CANONICAL_MASK_HEIGHT = 720
#: Where `fit_court_mask.py` leaves its per-broadcast choice. Under `data/`
#: rather than `outputs/` because it is a DECISION the pipeline depends on, not
#: a result: `outputs/` is ignored by git, so a fitted mask living there would
#: silently not exist on any other machine and every broadcast would quietly
#: fall back to the shipped constant.
COURT_ERODE_FILE = "data/court_erode.json"
#: A ball this far from a player's box EDGE, in units of that player's own box
#: height, is in his hands. Scale-free on purpose: a player at the far sideline
#: is a third the pixels of one under the basket, and a gate in pixels would
#: hold the near player to a stricter standard than the far one.
#:
#: Box EDGE rather than box centre because that is what was measured: wrists
#: from a pose model did not beat it and box centre lost to it outright
#: (Round 110). One definition because three scripts had grown their own, which
#: is how `wilson` ended up with four copies.
HOLD_GATE = 0.45
FEET_ON_COURT_SHARE = 0.45


def to_box(point, box) -> float:
    """Distance from a point to the nearest edge of a box; 0 inside it."""
    import math

    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def carrier_of(row, people, *, hold_gate: float = HOLD_GATE) -> int | None:
    """Which of `people` is holding the most confident ball, or None.

    `people` are boxes as [x1, y1, x2, y2] in the same coordinates as the row's
    detections. Returns an index so the caller can look the box up in whatever
    parallel list it keeps -- a mask, a colour, a track id.
    """
    balls = [b for b in row["d"] if b[0] == "b"]
    if not balls or not people:
        return None
    ball = max(balls, key=lambda b: b[1])
    centre = ((ball[2] + ball[4]) / 2.0, (ball[3] + ball[5]) / 2.0)
    nearest = min(range(len(people)), key=lambda i: to_box(centre, people[i]))
    height = people[nearest][3] - people[nearest][1]
    if height <= 0 or to_box(centre, people[nearest]) > hold_gate * height:
        return None
    return nearest


def court_region(image: np.ndarray, erode_px: int | None = COURT_ERODE_PX,
                 erode_share: float | None = None):
    """The largest connected run of floor: wood and painted court together.

    Colour cannot separate players from spectators in this arena -- the crowd
    wears the home kit's colour, so a fan in a blue shirt clusters with a
    player in a blue jersey. Position can: a player stands on the floor and a
    spectator stands beyond its edge.

    `erode_share` overrides `erode_px` and is measured against the frame's own
    height, so one number means the same thing on a 720p and a 1080p
    broadcast. How much to erode is a trade between admitting the front row and
    deleting a player, and both sides of it are measurable without labels --
    see `scripts/fit_court_mask.py`.

    THE PAINTED KEY IS FOUND BY SHAPE, NOT BY COLOUR. The `paint` rule below
    accepts hue 95-130, which is blue, and accepts 0.4% of Houston's red key.
    Anything the wood encloses -- a key, a centre logo, a sponsor decal -- is a
    HOLE in the wood mask, and filling holes captures it whatever colour it is.
    Worth 3.6 points of kept ball-carrier on the red-key broadcast on its own,
    and it cannot be wrong about a colour it never looks at.
    """
    import cv2

    full_height, full_width = image.shape[:2]
    if full_height != CANONICAL_MASK_HEIGHT and full_height > 0:
        scale = CANONICAL_MASK_HEIGHT / full_height
        image = cv2.resize(image, (max(1, int(round(full_width * scale))),
                                   CANONICAL_MASK_HEIGHT),
                           interpolation=cv2.INTER_AREA)

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

    # Fill what the floor encloses. Flooding the complement in from a corner
    # reaches everything OUTSIDE the region; whatever it cannot reach is a hole.
    outside = (1 - region).astype(np.uint8)
    cv2.floodFill(outside, np.zeros((region.shape[0] + 2, region.shape[1] + 2),
                                    np.uint8), (0, 0), 2)
    region = ((region == 1) | (outside == 1)).astype(np.uint8)

    if erode_share is not None:
        erode_px = int(round(erode_share * CANONICAL_MASK_HEIGHT))
    if erode_px:
        region = cv2.erode(region, np.ones((erode_px, erode_px), np.uint8))
    if region.shape[0] != full_height or region.shape[1] != full_width:
        region = cv2.resize(region, (full_width, full_height),
                            interpolation=cv2.INTER_NEAREST)
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
