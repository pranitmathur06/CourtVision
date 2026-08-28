"""Stage 3 — persistent player identities via ByteTrack.

Only players are tracked. The ball is small, fast and often occluded; running it
through a tracker yields flickering IDs that help nothing, since stage 5 needs
only its position. Ball and rim therefore pass through with track_id = -1.

`sv.ByteTrack` is deprecated in supervision 0.28 and removed in 0.31, and the
library currently offers no in-package replacement — hence the `<0.31` pin in
pyproject.toml. This wrapper is the seam that keeps that swap to one file.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import supervision as sv

from courtvision.types import PLAYER, Box, Detection, Frame, Track

UNTRACKED = -1


class PlayerTracker:
    """ByteTrack behind a `Detection` list in / `Track` list out interface."""

    def __init__(self) -> None:
        self._tracker = sv.ByteTrack()

    def update(self, detections: list[Detection]) -> list[Track]:
        players = [d for d in detections if d.label == PLAYER]
        others = [d for d in detections if d.label != PLAYER]

        tracks: list[Track] = [
            Track(UNTRACKED, d.box, d.label, d.conf) for d in others
        ]

        if players:
            sv_detections = sv.Detections(
                xyxy=np.array(
                    [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in players],
                    dtype=np.float32,
                ),
                confidence=np.array([d.conf for d in players], dtype=np.float32),
                class_id=np.zeros(len(players), dtype=int),
            )
            tracked = self._tracker.update_with_detections(sv_detections)
            for xyxy, conf, track_id in zip(
                tracked.xyxy, tracked.confidence, tracked.tracker_id
            ):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                tracks.append(
                    Track(int(track_id), Box(x1, y1, x2, y2), PLAYER, float(conf))
                )

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
