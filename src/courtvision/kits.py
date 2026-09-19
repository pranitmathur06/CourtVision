"""Which of the two kits is this player wearing, for one whole broadcast.

WHY THIS IS NOT `team_assignment`. That module clusters torso colour inside one
clip and names the clusters "A" and "B" with the explicit warning that which
physical team gets which letter is arbitrary. That is enough to colour an
overlay and not enough for anything else: two clips of the same game can
disagree, so no fact spanning clips -- who shot, who rebounded, whether a pass
crossed teams -- can be asked of it.

This fits ONE model for the whole broadcast, so kit 0 in the first quarter is
kit 0 in the fourth. Which kit is Houston is still unknown and still does not
matter: every question asked of it here is a SAME-KIT question, which is
invariant to the naming. "The rebound went to the shooting team" and "the pass
came from a teammate" are both of that shape, and both are real film facts.

HOW IT IS SCORED WITHOUT LABELS. When the ball is live there are exactly ten
players on the court, five a side. So on any frame where the detector draws
exactly ten player boxes, a correct kit model splits them 5-5. A model that
calls everyone the same kit splits 10-0 and scores zero. Nobody labels
anything; the rule comes from the sport. This is the same argument the floor
mask uses, and it is the reason both metrics exist at all.

THE FIVE-A-SIDE TEST AS FIRST WRITTEN WAS WRONG AND IS KEPT ONLY AS A
DIAGNOSTIC. Ten boxes is almost never the ten players: there are THIRTEEN
people on a live court -- ten players and up to three officials -- so a
ten-box frame is usually ten of thirteen. Measured that way a good model
scored 0.44, and no colour feature moved it, because the thing being measured
was mostly the detector's choice of who to draw.

THE TEST THAT SURVIVES IS ONE-SIDED. Neither kit can ever have SIX players on
the court. That holds on a frame with any number of boxes, needs no assumption
about who the detector found, and a model that calls everybody one kit breaks
it on every frame with six boxes. It is a violation rate, and the model is
scored by how rarely it claims something the sport forbids.

A ONE-SIDED TEST CAN BE GAMED BY SAYING NOTHING, so the officials are a class
rather than a rejection threshold: k=3, the two largest clusters are the kits,
the third is the officials. That has an arithmetic prediction attached -- three
officials among thirteen people is 0.231 -- and the measured rejected share
lands at 0.23-0.31 on four broadcasts, which is the evidence that the third
cluster really is the stripes and not an abstention hiding the hard boxes.
Both numbers are printed side by side; neither means anything alone.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from courtvision.team_assignment import TORSO_X, TORSO_Y

#: A person box below this confidence is not worth a colour sample.
MIN_BOX_CONF = 0.50
#: A box smaller than this is too few pixels for a torso mean to mean anything.
#: 40 px of height is about a player at the far sideline in a 1080p broadcast.
MIN_BOX_H = 40.0
#: Below this margin the model is guessing between the two kits, and says so
#: rather than answering. Margin is (d_far - d_near) / (d_far + d_near), so it
#: is 0 on the decision boundary and 1 when a sample sits on a centre.
MIN_MARGIN = 0.06


def torso_lab(image: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
    """Mean CIELAB colour of the jersey region of one box, or None if degenerate."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh < MIN_BOX_H:
        return None
    cx1 = int(round(x1 + bw * TORSO_X[0]))
    cx2 = int(round(x1 + bw * TORSO_X[1]))
    cy1 = int(round(y1 + bh * TORSO_Y[0]))
    cy2 = int(round(y1 + bh * TORSO_Y[1]))
    cx1, cx2 = max(0, cx1), min(width, cx2)
    cy1, cy2 = max(0, cy1), min(height, cy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float64)


@dataclass(frozen=True)
class KitModel:
    """Two kit centres and an officials centre in CIELAB, fitted per broadcast.

    `centres[0]` and `centres[1]` are the kits, ordered deterministically by
    their coordinates so two fits of the same samples agree. Which is which is
    arbitrary and nothing may depend on it -- every question asked of this
    model is a SAME-KIT question. `officials` is the third, smallest cluster.
    """

    centres: np.ndarray
    officials: np.ndarray

    @staticmethod
    def fit(samples: Iterable[Sequence[float]], *, seed: int = 0) -> "KitModel | None":
        """Three-means on torso colour: two kits and the officials."""
        from collections import Counter

        data = np.asarray([list(s) for s in samples], dtype=np.float64)
        if data.ndim != 2 or len(data) < 60:
            return None
        from sklearn.cluster import KMeans

        fitted = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(data)
        sizes = Counter(int(label) for label in fitted.labels_)
        ranked = [index for index, _ in sizes.most_common()]
        kits = sorted(ranked[:2], key=lambda i: tuple(fitted.cluster_centers_[i]))
        return KitModel(
            centres=fitted.cluster_centers_[kits].copy(),
            officials=fitted.cluster_centers_[ranked[2]].copy(),
        )

    def _distances(self, point: np.ndarray) -> tuple[np.ndarray, float]:
        return (np.linalg.norm(self.centres - point, axis=1),
                float(np.linalg.norm(self.officials - point)))

    def kit(self, colour: Sequence[float] | None) -> tuple[int | None, float]:
        """(kit index, margin), or (None, margin) for an official, a missing
        colour, or a colour inside the margin between the two kits.

        Declining beats guessing here for the same reason the assist reader
        declines on a short hold: a wrong attribution is worse than none.
        """
        if colour is None:
            return None, 0.0
        point = np.asarray(colour, dtype=np.float64)
        kit_distances, official_distance = self._distances(point)
        near = int(np.argmin(kit_distances))
        far = 1 - near
        total = kit_distances[near] + kit_distances[far]
        margin = 0.0 if total <= 0 else float(
            (kit_distances[far] - kit_distances[near]) / total)
        if official_distance < kit_distances[near]:
            return None, margin
        if margin < MIN_MARGIN:
            return None, margin
        return near, margin

    def belongs_on_court(self, colour: Sequence[float] | None,
                         max_lab: float) -> bool:
        """Is this torso one of the two kits or the officials' stripes?

        A spectator in the front row wears neither, and the front row is the
        thing eroding the court's edge exists to remove -- bluntly, since it
        removes the baseline corner with it. This is the same exclusion by
        colour instead of by position, and it can be measured against the same
        two label-free bounds the erosion is measured against.

        A missing colour answers True: a box too small or too clipped to read
        is not evidence of a spectator, and declining to exclude is the safe
        direction for a filter whose failure deletes players.
        """
        if colour is None:
            return True
        point = np.asarray(colour, dtype=np.float64)
        kit_distances, official_distance = self._distances(point)
        return min(float(kit_distances.min()), official_distance) <= max_lab

    def separation(self) -> float:
        """Distance between the two kit centres, in CIELAB units.

        A broadcast where both teams wear near-identical colours has a small
        one and no kit model can work on it. Printing it is how that case
        announces itself instead of arriving as a bad accuracy."""
        return float(np.linalg.norm(self.centres[0] - self.centres[1]))


def over_five(counts: Sequence[tuple[int, int]]) -> dict[str, float]:
    """How often a frame claimed six or more players in one kit.

    `counts` is one (kit0, kit1) pair per frame. The sport forbids six, so
    every such frame is a model error -- an over-count can never be the
    detector missing somebody, only the model mis-colouring somebody or the
    detector drawing a person twice.
    """
    total = len(counts)
    if total == 0:
        return {"n": 0, "violations": 0, "rate": math.nan}
    bad = sum(1 for a, b in counts if max(a, b) > 5)
    return {"n": total, "violations": bad, "rate": bad / total}


def split_report(splits: Sequence[int]) -> dict[str, float]:
    """Five-a-side, kept as a DIAGNOSTIC only -- see the module docstring.

    `splits[i]` is min(kit0, kit1) on a frame where exactly ten boxes were
    given a kit. It is noisy and its denominator is small, and it must not be
    quoted as the kit model's accuracy.
    """
    total = len(splits)
    if total == 0:
        return {"n": 0, "exact": 0.0, "within_one": 0.0, "mean_minority": math.nan}
    exact = sum(1 for s in splits if s == 5)
    within = sum(1 for s in splits if s >= 4)
    return {
        "n": total,
        "exact": exact / total,
        "within_one": within / total,
        "mean_minority": sum(splits) / total,
    }


def rows_spread(rows: Sequence[dict], want: int) -> list[dict]:
    """`want` detection rows spread evenly across a clip."""
    if len(rows) <= want:
        return list(rows)
    step = len(rows) / want
    return [rows[int(i * step)] for i in range(want)]


def people(row: dict, conf: float = MIN_BOX_CONF) -> list[list[float]]:
    """Player boxes of one detection row, in SOURCE pixels.

    `p` only. The `h` boxes are the handler model's view of the same people,
    and counting both was the bug that made tracking look twice as good as it
    is (Round 110)."""
    return [list(b[2:]) for b in row["d"] if b[0] == "p" and b[1] >= conf]


def sample_clip(path, rows: Sequence[dict], source_size: Sequence[float],
                *, want: int) -> list[tuple[dict, list[list[float]], list]]:
    """[(row, boxes in CLIP pixels, torso colours)] for `want` frames of a clip.

    Clips are written at 854x480 for the page while the detections are in the
    source video's own pixels, so every box is scaled here. Reading a torso at
    source coordinates out of a 480p clip lands on another player or off the
    frame entirely -- which is how this first ran, reporting every box as too
    small to sample.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    scale_x = capture.get(cv2.CAP_PROP_FRAME_WIDTH) / float(source_size[0])
    scale_y = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / float(source_size[1])
    out = []
    try:
        for row in rows_spread(rows, want):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["f"]))
            ok, image = capture.read()
            if not ok:
                continue
            boxes = [[b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                     for b in people(row)]
            out.append((row, boxes, [torso_lab(image, b) for b in boxes]))
    finally:
        capture.release()
    return out


def sample_frame(path, row: dict, source_size: Sequence[float], frame_index: int):
    """One named frame of a clip: (boxes in clip pixels, torso colours, scale)."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    try:
        scale_x = capture.get(cv2.CAP_PROP_FRAME_WIDTH) / float(source_size[0])
        scale_y = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / float(source_size[1])
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, image = capture.read()
        if not ok:
            return None
        boxes = [[b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                 for b in people(row)]
        return boxes, [torso_lab(image, b) for b in boxes], (scale_x, scale_y)
    finally:
        capture.release()


def sample_broadcast(reader, start_s: float, rows: Sequence[dict], step: int,
                     *, want: int) -> list[tuple[dict, list[list[float]], list]]:
    """[(row, boxes, torso colours)] for `want` rows of one clip, FROM SOURCE.

    The clip version of this reads an 854x480 file and scales the boxes down to
    it. This reads the broadcast, so the boxes need no scaling at all -- they
    are already in its pixels -- and the torso crop has the resolution the
    detector saw. Every accuracy number in this project is supposed to be about
    the broadcast; see `courtvision.broadcast` for what the cheap path costs.
    """
    picked = {position: rows[position]
              for position in _positions(len(rows), want)}
    out: list[tuple[dict, list[list[float]], list]] = []
    for row, image in reader.frames(start_s,
                                    {position: row
                                     for position, row in picked.items()}, step):
        boxes = people(row)
        out.append((row, boxes, [torso_lab(image, box) for box in boxes]))
    return out


def _positions(total: int, want: int) -> list[int]:
    """`want` row positions spread evenly over `total` rows."""
    if total <= 0:
        return []
    if total <= want:
        return list(range(total))
    stride = total / want
    return [int(i * stride) for i in range(want)]
