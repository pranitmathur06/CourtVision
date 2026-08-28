"""Stages 2-3 — object detection.

Every downstream stage codes against the `Detector` protocol, never against
ultralytics directly, so tests can substitute a stub with no model weights.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from courtvision.types import BALL, HANDLER, PLAYER, RIM, Box, Detection

# Stock COCO ids we care about. There is no COCO class for a basketball rim,
# which is exactly why V3 fine-tuning exists.
COCO_CLASS_MAP: dict[int, str] = {0: PLAYER, 32: BALL}

# After fine-tuning we own the class order, so it is dense and starts at zero.
# Must match scripts/prepare_detector_dataset.py.
FINETUNED_CLASS_MAP: dict[int, str] = {0: PLAYER, 1: BALL, 2: RIM, 3: HANDLER}


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
        conf_by_label: dict[str, float] | None = None,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._device = device
        self._conf = conf
        self._class_map = class_map
        # The ball is small, fast and motion-blurred, so it scores lower than a
        # player even when correctly found. One global threshold either loses the
        # ball or floods the frame with weak player boxes; per-label thresholds
        # avoid that trade.
        self._conf_by_label = dict(conf_by_label or {})
        self._predict_conf = min([conf, *self._conf_by_label.values()])

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            image, device=self._device, conf=self._predict_conf, verbose=False
        )
        if not results:
            return []
        found = boxes_from_result(results[0], self._class_map, self._predict_conf)
        return [
            d for d in found
            if d.conf >= self._conf_by_label.get(d.label, self._conf)
        ]


def load_finetuned(weights_path: str, device: str, conf: float) -> YoloDetector:
    """Load the fine-tuned player detector on its own (players only, no ball)."""
    return YoloDetector(weights_path, device, conf, FINETUNED_CLASS_MAP)


def load_pipeline_detector(
    weights_path: str, device: str, conf: float, ball_conf: float
) -> YoloDetector:
    """The detector the pipeline runs: one fine-tuned model for player/ball/rim.

    This replaces an earlier two-model composite (fine-tuned players + stock COCO
    `sports ball`). That workaround existed only because the first detector
    dataset had no ball labels; COCO's ball never worked well enough for
    possession anyway.
    """
    return YoloDetector(
        weights_path, device, conf, FINETUNED_CLASS_MAP, {BALL: ball_conf}
    )
