"""Stage 6 — classify short windows of play into the five spec actions.

Windowing is pure and tested. The model is a fine-tuned VideoMAE, which expects
exactly 16 frames at 224x224 — hence `Config.action_window_frames = 16`.
Per spec §4, nothing here is trained from scratch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from courtvision.config import Config
from courtvision.types import ACTIONS, ActionWindow, Frame

FRAME_SIZE = 224


def plan_windows(n_frames: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs. Partial trailing windows are dropped."""
    if n_frames < size:
        return []
    return [
        (start, start + size - 1)
        for start in range(0, n_frames - size + 1, stride)
    ]


class ActionClassifier(Protocol):
    def classify(self, clip: np.ndarray) -> tuple[str, float]: ...


class VideoMaeClassifier:
    """Fine-tuned VideoMAE behind the `ActionClassifier` protocol."""

    def __init__(self, weights_dir: str, device: str) -> None:
        import torch
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        self._torch = torch
        self._processor = VideoMAEImageProcessor.from_pretrained(weights_dir)
        self._model = VideoMAEForVideoClassification.from_pretrained(weights_dir)
        self._model.to(device).eval()
        self._device = device

    def classify(self, clip: np.ndarray) -> tuple[str, float]:
        """clip: (n_frames, height, width, 3) uint8 RGB."""
        inputs = self._processor(list(clip), return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        probabilities = logits.softmax(dim=-1)[0]
        index = int(probabilities.argmax())
        return self._model.config.id2label[index], float(probabilities[index])


def classify_windows(
    images: Sequence[np.ndarray],
    frames: Sequence[Frame],
    classifier: ActionClassifier,
    config: Config,
) -> list[ActionWindow]:
    """Slide a window over the clip and classify each one."""
    import cv2

    windows: list[ActionWindow] = []
    for start, end in plan_windows(
        len(images), config.action_window_frames, config.action_stride_frames
    ):
        clip = np.stack(
            [
                cv2.cvtColor(
                    cv2.resize(images[i], (FRAME_SIZE, FRAME_SIZE)), cv2.COLOR_BGR2RGB
                )
                for i in range(start, end + 1)
            ]
        )
        label, conf = classifier.classify(clip)
        if label not in ACTIONS:
            label = "other"
        windows.append(
            ActionWindow(
                start_index=start,
                end_index=end,
                start_time_s=frames[start].time_s,
                end_time_s=frames[end].time_s,
                label=label,
                conf=conf,
            )
        )
    return windows
