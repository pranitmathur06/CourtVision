"""Deterministic synthetic basketball clip with exact ground truth.

Four players move on straight, non-crossing horizontal lanes so a correct tracker
produces zero ID switches. The ball is drawn at the current holder's centre, with
two scripted "in flight" gaps where no player holds it — those exercise the
possession smoother's gap-bridging.

Everything here is test-only and must never be imported by src/courtvision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from courtvision.types import BALL, PLAYER, Box, Detection

WIDTH, HEIGHT = 640, 360
COURT_COLOR = (60, 90, 130)          # BGR, a dull brown-ish court
TEAM_COLORS = {"A": (40, 40, 200), "B": (200, 60, 40)}   # BGR: A red, B blue
BALL_COLOR = (30, 150, 240)          # BGR orange
PLAYER_W, PLAYER_H = 28, 72
BALL_R = 7

# player index -> team. Players 0,1 are team A; 2,3 are team B.
TEAMS: dict[int, str] = {0: "A", 1: "A", 2: "B", 3: "B"}

# (first_frame, last_frame_inclusive, holder). holder None means ball in flight.
HOLDER_SCHEDULE: list[tuple[int, int, int | None]] = [
    (0, 19, 0),
    (20, 21, None),
    (22, 34, 2),
    (35, 36, None),
    (37, 49, 1),
]


@dataclass(frozen=True)
class SyntheticTruth:
    path: str
    fps: int
    n_frames: int
    teams: dict[int, str]
    holder_by_frame: list[int | None]
    player_boxes: list[dict[int, Box]]
    ball_boxes: list[Box | None]


def _holder_at(frame_index: int) -> int | None:
    for start, end, holder in HOLDER_SCHEDULE:
        if start <= frame_index <= end:
            return holder
    return None


def _player_box(player_index: int, frame_index: int, n_frames: int) -> Box:
    """Each player owns a horizontal lane and oscillates within it. Lanes never overlap."""
    lane_x = 60 + player_index * 150
    phase = 2.0 * math.pi * frame_index / max(n_frames, 1)
    x = lane_x + 25.0 * math.sin(phase + player_index)
    y = 120.0 + 40.0 * math.sin(phase * 2.0 + player_index)
    return Box(x, y, x + PLAYER_W, y + PLAYER_H)


def make_clip(path: str, n_frames: int = 50, fps: int = 10) -> SyntheticTruth:
    """Render the clip to `path` and return its exact ground truth."""
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")

    holder_by_frame: list[int | None] = []
    player_boxes: list[dict[int, Box]] = []
    ball_boxes: list[Box | None] = []

    for frame_index in range(n_frames):
        image = np.full((HEIGHT, WIDTH, 3), COURT_COLOR, dtype=np.uint8)

        boxes = {i: _player_box(i, frame_index, n_frames) for i in TEAMS}
        for player_index, box in boxes.items():
            cv2.rectangle(
                image,
                (int(box.x1), int(box.y1)),
                (int(box.x2), int(box.y2)),
                TEAM_COLORS[TEAMS[player_index]],
                thickness=-1,
            )

        holder = _holder_at(frame_index)
        if holder is None:
            # Ball in flight: park it high above the court, far from every player.
            bx, by = WIDTH / 2.0, 30.0
        else:
            bx, by = boxes[holder].center
        cv2.circle(image, (int(bx), int(by)), BALL_R, BALL_COLOR, thickness=-1)

        writer.write(image)
        holder_by_frame.append(holder)
        player_boxes.append(boxes)
        ball_boxes.append(Box(bx - BALL_R, by - BALL_R, bx + BALL_R, by + BALL_R))

    writer.release()
    return SyntheticTruth(
        path=path,
        fps=fps,
        n_frames=n_frames,
        teams=dict(TEAMS),
        holder_by_frame=holder_by_frame,
        player_boxes=player_boxes,
        ball_boxes=ball_boxes,
    )


class StubDetector:
    """Replays ground-truth boxes as if a perfect detector produced them.

    Lets every downstream stage be tested without model weights. Note it is
    indexed by frame number, not by image content — it is a fixture, not a model.
    """

    def __init__(self, truth: SyntheticTruth) -> None:
        self._truth = truth

    def detect_at(self, frame_index: int) -> list[Detection]:
        detections = [
            Detection(box, PLAYER, 0.99)
            for box in self._truth.player_boxes[frame_index].values()
        ]
        ball = self._truth.ball_boxes[frame_index]
        if ball is not None:
            detections.append(Detection(ball, BALL, 0.95))
        return detections


class StubActionClassifier:
    """Always returns the same label. Lets stages 7 and 9 be tested without weights."""

    def __init__(self, label: str = "dribble", conf: float = 0.99) -> None:
        self._label = label
        self._conf = conf

    def classify(self, clip: np.ndarray) -> tuple[str, float]:
        return self._label, self._conf
