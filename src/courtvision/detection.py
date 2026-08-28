"""Stages 2-3 — object detection.

Every downstream stage codes against the `Detector` protocol, never against
ultralytics directly, so tests can substitute a stub with no model weights.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from courtvision.types import BALL, PLAYER, Box, Detection

# Stock COCO ids we care about. There is no COCO class for a basketball rim,
# which is exactly why V3 fine-tuning exists.
COCO_CLASS_MAP: dict[int, str] = {0: PLAYER, 32: BALL}

# After fine-tuning we own the class order, so it is dense and starts at zero.
# Player only: the fine-tuning source (SportsMOT) labels players and nothing else.
# `ball` is supplied at inference by stock COCO via CompositeDetector, and `rim`
# is not modelled at all because no downstream stage reads it — possession,
# team assignment, tracking and render all use .players() and .ball() only.
FINETUNED_CLASS_MAP: dict[int, str] = {0: PLAYER}


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


def boxes_from_result(
    result, class_map: dict[int, str], conf: float
) -> list[Detection]:
    """Convert one ultralytics result into our types, filtering by class and confidence."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    detections: list[Detection] = []
    for xyxy, class_id, score in zip(boxes.xyxy, boxes.cls, boxes.conf):
        label = class_map.get(int(class_id))
        if label is None or float(score) < conf:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        detections.append(Detection(Box(x1, y1, x2, y2), label, float(score)))
    return detections


class YoloDetector:
    """Ultralytics YOLO behind the `Detector` protocol."""

    def __init__(
        self,
        weights: str,
        device: str,
        conf: float,
        class_map: dict[int, str],
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._device = device
        self._conf = conf
        self._class_map = class_map

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            image, device=self._device, conf=self._conf, verbose=False
        )
        if not results:
            return []
        return boxes_from_result(results[0], self._class_map, self._conf)


class CompositeDetector:
    """Fine-tuned player detector + stock COCO for the ball.

    Our fine-tuning data labels players only, but stage 5 (possession) needs the
    ball. COCO's `sports ball` class detects basketballs adequately, so the two
    are combined behind the one `Detector` interface every other stage codes to.

    Cost is a second forward pass per frame. Detection is ~15% of pipeline time
    (see docs/profile-v1.md), so this is affordable; it is also the obvious thing
    to collapse once a single detector is trained on both classes.
    """

    def __init__(self, player_detector: Detector, ball_detector: Detector) -> None:
        self._player = player_detector
        self._ball = ball_detector

    def detect(self, image: np.ndarray) -> list[Detection]:
        players = [d for d in self._player.detect(image) if d.label == PLAYER]
        balls = [d for d in self._ball.detect(image) if d.label == BALL]
        return players + balls


def load_finetuned(weights_path: str, device: str, conf: float) -> YoloDetector:
    """Load the fine-tuned player detector on its own (players only, no ball)."""
    return YoloDetector(weights_path, device, conf, FINETUNED_CLASS_MAP)


# The ball detector is deliberately the LARGE COCO model, not the nano one used
# elsewhere. Measured on our sample clip: yolo11n found the ball in 0/104 frames
# at conf 0.25 (peak confidence 0.116), while yolo11x reached 0.652 and covered
# most frames. A basketball at broadcast distance is small, fast and blurred —
# exactly where a nano backbone gives up.
BALL_WEIGHTS = "yolo11x.pt"


def load_pipeline_detector(
    weights_path: str, device: str, conf: float, ball_conf: float
) -> CompositeDetector:
    """The detector the pipeline actually runs: fine-tuned players + COCO ball."""
    return CompositeDetector(
        YoloDetector(weights_path, device, conf, FINETUNED_CLASS_MAP),
        YoloDetector(BALL_WEIGHTS, device, ball_conf, {32: BALL}),
    )
