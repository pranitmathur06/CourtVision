"""What colour is THIS arena's floor? Learned from the broadcast, not listed.

THE RULE IT REPLACES IS A LIST OF COLOURS SOMEBODY HAD SEEN. `candidates.
court_region` accepts hue 5-30 as wood and hue 95-130 as paint, which is tan
and blue. Toyota Center's court is largely RED -- hue 174, saturation 230,
measured under the feet of the players standing on it -- and the rule accepts
none of it. On that broadcast 76% of the frames where the floor mask drops the
man holding the ball have ZERO floor under his feet, and the mask keeps him on
0.572 of frames against 0.860 and 0.906 on the two blue-and-tan arenas.

Filling what the wood encloses recovers a painted KEY, because a key is
surrounded by wood. It does not recover a court whose paint runs to the
sideline, and no list of hues will cover the next arena either.

HOW IT IS LEARNED WITHOUT LABELS. A player stands on the floor. So the strip of
pixels just below a player's box is floor, by the definition of standing, and a
person detector that already runs on every frame of this pipeline hands over
hundreds of such samples a minute. Collect their colours, keep the ones that
recur, and that is the arena's floor -- tan at Paycom Center, red at Toyota
Center, whatever it turns out to be at the next one.

WHAT IT CANNOT DO. The strip below a box is floor only when the box is a player
standing on the court. A box drawn around somebody in the front row samples the
seats; a box on a player who is airborne samples whatever is behind him. Those
are wrong but they are not systematic -- the crowd is many colours and the
floor is one or two -- so keeping only colours that RECUR is what makes the
method work, and a bin cut-off is what "recur" means.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

#: How far below a box's bottom edge the floor strip starts and ends, in
#: pixels of the frame the box was drawn on. Starting AT the edge samples the
#: player's shoe.
STRIP_FROM, STRIP_TO = 4, 14
#: The strip spans this fraction of the box's width, centred -- between the
#: feet rather than outside them.
STRIP_WIDTH = (0.25, 0.75)
#: A box shorter than this is too far away for the strip below it to be more
#: than a few pixels of anything.
MIN_BOX_H = 60.0
#: Hue and saturation bins. Hue is 0-179 in OpenCV, saturation 0-255.
HUE_BINS, SAT_BINS = 36, 16
#: Keep the bins holding this share of the collected mass. The floor is one or
#: two colours and everything sampled by mistake is many, so the tail is where
#: the mistakes are.
KEEP_MASS = 0.90
#: A pixel darker than this is a shadow or the stands, whatever its hue.
MIN_VALUE = 80
#: Below this many samples the histogram is noise and the caller should fall
#: back to the listed colours.
MIN_SAMPLES = 200


@dataclass(frozen=True)
class FloorColour:
    """The (hue, saturation) bins this arena's floor occupies."""

    bins: np.ndarray                     # bool, HUE_BINS x SAT_BINS
    samples: int

    def mask(self, image: np.ndarray) -> np.ndarray:
        """Pixels of `image` whose colour is this floor."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = np.minimum(hsv[:, :, 0].astype(np.int32) * HUE_BINS // 180,
                         HUE_BINS - 1)
        sat = np.minimum(hsv[:, :, 1].astype(np.int32) * SAT_BINS // 256,
                         SAT_BINS - 1)
        return (self.bins[hue, sat] & (hsv[:, :, 2] >= MIN_VALUE))

    def share(self) -> float:
        """How much of the colour space this accepts. A floor model that takes
        half of it is not a floor model, and the caller should say so rather
        than mask a whole frame with it."""
        return float(self.bins.mean())


def strip_below(image: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
    """The floor a player is standing on, or None if the box cannot give one."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if y2 - y1 < MIN_BOX_H:
        return None
    span = x2 - x1
    left = int(round(x1 + span * STRIP_WIDTH[0]))
    right = int(round(x1 + span * STRIP_WIDTH[1]))
    top, bottom = int(round(y2)) + STRIP_FROM, int(round(y2)) + STRIP_TO
    left, right = max(0, left), min(width, right)
    top, bottom = max(0, top), min(height, bottom)
    if right - left < 3 or bottom - top < 2:
        return None
    return image[top:bottom, left:right]


def collect(image: np.ndarray, boxes: Sequence[Sequence[float]]) -> list:
    """One median (hue, saturation, value) per player box on this frame."""
    out = []
    for box in boxes:
        strip = strip_below(image, box)
        if strip is None or strip.size == 0:
            continue
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        out.append(np.median(hsv, axis=0))
    return out


def learn(samples: Sequence[Sequence[float]]) -> FloorColour | None:
    """The floor's colours, from samples taken under players' feet."""
    usable = [s for s in samples if s is not None and s[2] >= MIN_VALUE]
    if len(usable) < MIN_SAMPLES:
        return None
    data = np.asarray(usable, dtype=np.float64)
    hue = np.minimum((data[:, 0] * HUE_BINS // 180).astype(int), HUE_BINS - 1)
    sat = np.minimum((data[:, 1] * SAT_BINS // 256).astype(int), SAT_BINS - 1)
    counts = np.zeros((HUE_BINS, SAT_BINS), dtype=np.int64)
    np.add.at(counts, (hue, sat), 1)

    flat = counts.ravel()
    order = np.argsort(flat)[::-1]
    running = np.cumsum(flat[order])
    needed = int(np.searchsorted(running, KEEP_MASS * running[-1]) + 1)
    bins = np.zeros(flat.shape, dtype=bool)
    bins[order[:needed]] = True
    return FloorColour(bins=bins.reshape(counts.shape), samples=len(usable))


#: Where `fit_floor_colour.py` leaves the learned floors. Under `data/`
#: because it is an input the pipeline depends on, not a result.
FLOOR_FILE = "data/floor_colour.json"


def load(path, key: str) -> "FloorColour | None":
    """This broadcast's learned floor, or None if it has not been fitted.

    None is the honest answer and the caller must fall back to the listed
    colours rather than mask nothing: an arena whose floor has not been learned
    is not an arena with no floor.
    """
    import json
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return None
    blob = json.loads(path.read_text()).get(key)
    if not blob:
        return None
    return FloorColour(bins=np.asarray(blob["bins"], dtype=bool),
                       samples=int(blob.get("samples", 0)))
