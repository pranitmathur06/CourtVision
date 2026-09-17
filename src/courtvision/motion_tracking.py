"""Player identities across a broadcast: Hungarian assignment over motion.

This replaces ByteTrack, which `tracking.PlayerTracker` wrapped and which
supervision deprecated in 0.28 and removes in 0.31 with nothing offered in its
place. It is not a like-for-like swap: it was written in `clip_boxes.py` because
ByteTrack was measurably not good enough on this footage, and it is promoted
here so the whole pipeline gets what the clip renderer already had.

Measured on six-second clips of a broadcast, ten players on the floor:

                                  ByteTrack-era    this
    identities per clip                127          34
    median track life                 0.13 s      1.00 s
    identities living under 0.5 s        77%     under half
    players drawn per frame (p50)          7        9-10

Two things caused the 127, and neither was the detector.

  THE CAMERA MOVES. On a pan a stationary player's box slides several of its own
  widths between frames, so its overlap with its own previous box is zero and it
  becomes a new person. The global shift is estimated from the matches
  themselves -- the median displacement of everything that did match -- and
  applied before the rest are matched.

  A PLAYER DISAPPEARS FOR A MOMENT. Behind another player, at the edge of frame,
  or simply missed. Matching only against the previous frame ends the track; a
  track here survives `max_age_s` of absence, moving at its last known velocity,
  and is picked up again when it reappears.

Assignment is Hungarian over 1 - IoU rather than greedy, so one obvious match no
longer steals the box a better one needed.

## Every threshold is in SECONDS, and that is not cosmetic

The original constants were in FRAMES of the sampled sequence. `TRACK_MAX_AGE =
15` reads "half a second" in its own comment and is one second at the 15 Hz the
clip renderer samples -- and three seconds at the 5 Hz the full-game cache uses.
Nothing announced that; the tracker simply became a different algorithm when the
sampling rate changed, which is exactly the kind of thing that makes a game-scale
number disagree with a clip-scale one for no visible reason. `TrackerConfig`
holds seconds and converts at construction, so a re-rate changes the sampling
and not the model.

## What is NOT here

No re-identification. A track ends at a camera cut and the same player comes
back as somebody new, which is why a whole game still fragments into hundreds of
identities. That is a separate piece of work with its own measurement, and
calling this "tracking" without saying so would overstate it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: The rate the tracker's thresholds were tuned at. Only used to document the
#: frame-count equivalents; nothing here assumes it.
TUNED_FPS = 15.0


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Overlap of two xyxy boxes, 0 when they do not touch."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class TrackerConfig:
    """Thresholds in seconds and in units that do not depend on frame rate.

    The frame-count equivalents at 15 Hz, which is what the shipped clip
    renderer uses and what these were tuned on, are in the comments.
    """
    #: A track survives this long without a detection before it is retired.
    #: Long enough to cross behind another player. (15 frames at 15 Hz.)
    max_age_s: float = 1.0
    #: Overlap that matches outright, before the distance gate is considered.
    min_iou: float = 0.18
    #: ...and a box whose centre is within this share of the predicted box's own
    #: size matches too, which is what carries a track through a fast pan.
    gate_share: float = 1.4
    #: Two boxes overlapping this much in one frame are one player twice.
    duplicate_iou: float = 0.55
    #: How much of the new velocity to believe, against the old.
    velocity_blend: float = 0.4
    #: A gap shorter than this is filled so a box does not blink.
    #: (10 frames at 15 Hz.)
    max_gap_s: float = 0.667
    #: Half-width of the centred moving average applied to a track.
    smooth_window: int = 2

    def frames(self, seconds: float, fps: float) -> int:
        """Seconds to frames of the SAMPLED sequence, at least one."""
        return max(1, int(round(seconds * fps)))


class MotionTracker:
    """The tracker as a stream: one frame in, that frame's identities out.

    `track()` below is a loop over this, so the batch path the clip renderer
    uses and the streaming path `tracking.PlayerTracker` uses are the same code
    and cannot drift. The golden fixture pins that they produce what the
    original batch function produced, box for box.
    """

    def __init__(self, fps: float = TUNED_FPS, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.fps = fps
        self.max_age = self.config.frames(self.config.max_age_s, fps)
        self._live: dict[int, dict] = {}
        self._next = 0
        self._shift = (0.0, 0.0)

    def end_segment(self) -> None:
        """Retire every live track, because the camera cut.

        A track continuing through a cut asserts that the player either side of
        it is the same person, and nothing here can know that -- there is no
        re-identification. Ending them is the honest answer and it makes the
        identity COUNT worse while making the identities themselves right.
        """
        self._live, self._shift = {}, (0.0, 0.0)

    def update(self, boxes: Sequence[Sequence[float]], frame: int
               ) -> list[tuple[int, list[float]]]:
        """(track id, box) for each box given, in the order they were given."""
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        config = self.config
        predicted = {}
        for tid, state in self._live.items():
            age = frame - state["frame"]
            vx, vy = state["vel"]
            dx = vx * age + self._shift[0] * age
            dy = vy * age + self._shift[1] * age
            b = state["box"]
            predicted[tid] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]

        matches = {}
        if predicted and boxes:
            ids = sorted(predicted)
            cost = np.ones((len(ids), len(boxes)), dtype=float)
            for i, tid in enumerate(ids):
                pb = predicted[tid]
                width = max(pb[2] - pb[0], 1.0)
                height = max(pb[3] - pb[1], 1.0)
                for j, box in enumerate(boxes):
                    overlap = iou(pb, box)
                    near = (abs((box[0] + box[2]) / 2 - (pb[0] + pb[2]) / 2)
                            < config.gate_share * width
                            and abs((box[1] + box[3]) / 2 - (pb[1] + pb[3]) / 2)
                            < config.gate_share * height)
                    sized = 0.5 <= (box[2] - box[0]) / width <= 2.0
                    if overlap >= config.min_iou or (near and sized):
                        cost[i, j] = 1.0 - max(overlap, 0.05)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1.0:
                    matches[ids[i]] = j

        moved = [((boxes[j][0] + boxes[j][2]) / 2
                  - (predicted[t][0] + predicted[t][2]) / 2,
                  (boxes[j][1] + boxes[j][3]) / 2
                  - (predicted[t][1] + predicted[t][3]) / 2)
                 for t, j in matches.items()]
        if len(moved) >= 3:
            self._shift = (float(np.median([m[0] for m in moved])),
                           float(np.median([m[1] for m in moved])))
        else:
            self._shift = (0.0, 0.0)

        keep = 1.0 - config.velocity_blend
        out: dict[int, tuple[int, list[float]]] = {}
        taken = set(matches.values())
        for tid, j in matches.items():
            box = boxes[j]
            state = self._live[tid]
            gap = max(frame - state["frame"], 1)
            centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            was = ((state["box"][0] + state["box"][2]) / 2,
                   (state["box"][1] + state["box"][3]) / 2)
            vel = ((centre[0] - was[0]) / gap, (centre[1] - was[1]) / gap)
            state["vel"] = (keep * state["vel"][0] + config.velocity_blend * vel[0],
                            keep * state["vel"][1] + config.velocity_blend * vel[1])
            state["box"], state["frame"] = box, frame
            out[j] = (tid, list(box))
        for j, box in enumerate(boxes):
            if j in taken:
                continue
            self._live[self._next] = {"box": box, "frame": frame,
                                      "vel": (0.0, 0.0)}
            out[j] = (self._next, list(box))
            self._next += 1
        self._live = {tid: s for tid, s in self._live.items()
                      if frame - s["frame"] <= self.max_age}
        return [out[j] for j in range(len(boxes))]


def track(per_frame: Sequence[Sequence[Sequence[float]]], fps: float = TUNED_FPS,
          config: TrackerConfig | None = None,
          cuts: Sequence[int] = ()) -> dict[int, dict[int, list[float]]]:
    """{track id: {frame index: box}} across gaps and across camera motion.

    `cuts` are frame indices where the camera cut; every live track is retired
    at one. See `MotionTracker.end_segment`.
    """
    tracker = MotionTracker(fps=fps, config=config)
    breaks = set(int(c) for c in cuts)
    tracks: dict[int, dict[int, list[float]]] = {}
    for frame, boxes in enumerate(per_frame):
        if frame in breaks:
            tracker.end_segment()
        for tid, box in tracker.update(boxes, frame):
            tracks.setdefault(tid, {})[frame] = box
    return tracks


def smooth(by_frame: dict[int, Sequence[float]], window: int = 2
           ) -> dict[int, list[float]]:
    """Centred moving average over a track, keyed by frame."""
    order = sorted(by_frame)
    out = {}
    for i, f in enumerate(order):
        low, high = max(0, i - window), min(len(order), i + window + 1)
        chunk = [by_frame[order[k]] for k in range(low, high)]
        out[f] = [sum(b[k] for b in chunk) / len(chunk) for k in range(4)]
    return out


def interpolate(by_frame: dict[int, Sequence[float]], max_gap: int = 10
                ) -> dict[int, list[float]]:
    """Fill short gaps in a track so a box does not blink."""
    order = sorted(by_frame)
    out = dict(by_frame)
    for a, b in zip(order, order[1:]):
        gap = b - a
        if gap <= 1 or gap > max_gap:
            continue
        for k in range(1, gap):
            share = k / gap
            out[a + k] = [by_frame[a][i]
                          + (by_frame[b][i] - by_frame[a][i]) * share
                          for i in range(4)]
    return out


def deduplicate(boxes: Sequence[Sequence[float]], threshold: float = 0.55
                ) -> list[int]:
    """Indices to keep when two boxes in one frame are one player twice.

    Two tracks settle on the same player often enough to matter -- usually one
    of them carried through a gap by prediction -- and then he is drawn twice.
    The caller decides the order, so the first of a colliding pair survives.
    """
    keep, placed = [], []
    for i, box in enumerate(boxes):
        if any(iou(box, other) > threshold for other in placed):
            continue
        placed.append(box)
        keep.append(i)
    return keep
