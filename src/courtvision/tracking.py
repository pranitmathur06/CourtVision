"""Stage 3 — persistent player identities.

Only players are tracked. The ball is small, fast and often occluded; running it
through a tracker yields flickering IDs that help nothing, since stage 5 needs
only its position. Ball and rim therefore pass through with track_id = -1.

THIS NO LONGER WRAPS BYTETRACK. `sv.ByteTrack` is deprecated in supervision 0.28
and removed in 0.31 with nothing offered in its place, and this file was written
as the seam that would keep that swap to one file. The swap has now happened,
and not because of the deprecation: `motion_tracking` was built in the clip
renderer because ByteTrack was measurably not good enough on this footage --
127 identities per six-second clip for ten players, 77% of them living under
half a second, against 34 and under half. The whole pipeline now gets what the
clip renderer already had.

What is NOT fixed by it: there is still no re-identification, so a track ends at
a camera cut and the same player returns as somebody new.
"""

from __future__ import annotations

from collections.abc import Sequence

from courtvision.motion_tracking import MotionTracker, TrackerConfig
from courtvision.types import HANDLER, PLAYER, Box, Detection, Frame, Track

UNTRACKED = -1


class PlayerTracker:
    """`motion_tracking` behind a `Detection` list in / `Track` list out seam.

    `fps` is the rate of the SAMPLED sequence, not of the video: the tracker's
    thresholds are in seconds and a caller detecting every second frame of 30 Hz
    footage is running at 15. Getting this wrong does not raise -- it silently
    changes how long a track survives an occlusion, which is why it is an
    argument rather than a constant.
    """

    def __init__(self, fps: float = 15.0, config: TrackerConfig | None = None) -> None:
        self._tracker = MotionTracker(fps=fps, config=config)
        self._frame = 0

    def end_segment(self) -> None:
        """The camera cut; nothing carries across it. See MotionTracker."""
        self._tracker.end_segment()

    def update(self, detections: list[Detection]) -> list[Track]:
        # Handlers are people too: they must be tracked so their identity
        # persists, but their label has to survive tracking.
        players = [d for d in detections if d.label in (PLAYER, HANDLER)]
        others = [d for d in detections if d.label not in (PLAYER, HANDLER)]

        tracks: list[Track] = [
            Track(UNTRACKED, d.box, d.label, d.conf) for d in others
        ]
        boxes = [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in players]
        for detection, (track_id, box) in zip(
                players, self._tracker.update(boxes, self._frame)):
            tracks.append(Track(int(track_id),
                                Box(box[0], box[1], box[2], box[3]),
                                detection.label, detection.conf))
        self._frame += 1
        return tracks


def _iou(a: Box, b: Box) -> float:
    inter_x1, inter_y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    inter_x2, inter_y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def count_id_switches(
    frames: Sequence[Frame], truth: Sequence[dict[int, Box]]
) -> int:
    """Count how often a ground-truth player's assigned track_id changes.

    Each truth box is matched to the tracked box with the highest IoU. A switch is
    counted whenever a truth player's matched track_id differs from the previous
    frame in which it was matched at all.
    """
    last_id: dict[int, int] = {}
    switches = 0

    for frame, truth_boxes in zip(frames, truth):
        players = frame.players()
        for truth_index, truth_box in truth_boxes.items():
            if not players:
                continue
            best = max(players, key=lambda t: _iou(t.box, truth_box))
            if _iou(best.box, truth_box) <= 0.0:
                continue
            previous = last_id.get(truth_index)
            if previous is not None and previous != best.track_id:
                switches += 1
            last_id[truth_index] = best.track_id

    return switches
