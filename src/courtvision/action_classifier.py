"""Stage 6 — classify short windows of play into the spec's actions.

Windowing is pure and tested. The model is a fine-tuned VideoMAE, which expects
exactly 16 frames at 224x224 — hence `Config.action_window_frames = 16`.
Per spec §4, nothing here is trained from scratch.

**Each window is cropped to the ball-handler, not passed whole.** Two reasons,
and they agree:

1. The training data (SpaceJam) is clips cropped to a single player. Feeding
   whole 1280x720 broadcast frames at inference would be a domain mismatch
   severe enough to make the classifier useless.
2. It is the right question anyway. "Dribble or pass?" is a property of one
   player; ten players are on court doing different things, so asking it of a
   whole frame is ill-posed.

A window with no possession holder is labelled `other` without invoking the
model — no ball-handler means no ball-handler action to recognise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from courtvision.config import Config
from courtvision.types import ACTIONS, ActionWindow, Frame

FRAME_SIZE = 224
# Fraction of the player box height added as padding around the crop, so the
# ball and the player's arms stay in frame the way SpaceJam's crops do.
CROP_MARGIN = 0.25
# SpaceJam clips are 128x176 (portrait). Crops taken at inference are forced to
# the same aspect so the processor's resize to 224x224 stretches training and
# inference frames identically. A square crop would distort differently from the
# training data — a source artifact the model could latch onto instead of the
# action.
CROP_ASPECT = 128 / 176


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

    def classify_batch(
        self, clips: Sequence[np.ndarray]
    ) -> list[tuple[str, float]]: ...


def _classify_batch_fallback(
    classifier: "ActionClassifier", clips: Sequence[np.ndarray]
) -> list[tuple[str, float]]:
    """One-at-a-time fallback for classifiers that implement only `classify`."""
    batch = getattr(classifier, "classify_batch", None)
    if callable(batch):
        return list(batch(clips))
    return [classifier.classify(clip) for clip in clips]


class VideoMaeClassifier:
    """Fine-tuned VideoMAE behind the `ActionClassifier` protocol."""

    def __init__(self, weights_dir: str, device: str, batch_size: int = 8) -> None:
        import torch
        from transformers import VideoMAEImageProcessor

        from courtvision.videomae import load_videomae_classifier

        self._torch = torch
        self._processor = VideoMAEImageProcessor.from_pretrained(weights_dir)
        # Our own fine-tuned checkpoints already use the current parameter names,
        # so this restores nothing for them — but it keeps the loading path
        # identical whether we point at a local checkpoint or a hub model.
        self._model, _ = load_videomae_classifier(weights_dir)
        self._model.to(device).eval()
        self._device = device
        self._batch_size = batch_size

    def classify(self, clip: np.ndarray) -> tuple[str, float]:
        """clip: (n_frames, height, width, 3) uint8 RGB."""
        return self.classify_batch([clip])[0]

    def classify_batch(
        self, clips: Sequence[np.ndarray]
    ) -> list[tuple[str, float]]:
        """Classify several windows in one forward pass.

        Stage 6 is 76.8% of pipeline runtime (docs/profile-v1.md) and the windows
        are independent, so running them one at a time wastes most of the device.
        Batching is the single largest speed win available and costs no accuracy.
        """
        if not clips:
            return []
        results: list[tuple[str, float]] = []
        for start in range(0, len(clips), self._batch_size):
            chunk = clips[start : start + self._batch_size]
            inputs = self._processor([list(c) for c in chunk], return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with self._torch.no_grad():
                logits = self._model(**inputs).logits
            for row in logits.softmax(dim=-1):
                index = int(row.argmax())
                results.append(
                    (self._model.config.id2label[index], float(row[index]))
                )
        return results


def crop_player(image: np.ndarray, box, margin: float = CROP_MARGIN) -> np.ndarray:
    """Crop around a player box with padding, at SpaceJam's aspect ratio.

    The crop is widened or heightened to CROP_ASPECT before resizing, so a player
    fills the frame the same way here as in the training clips.
    """
    import cv2

    height, width = image.shape[:2]
    pad = box.height * margin
    x1, y1 = box.x1 - pad, box.y1 - pad
    x2, y2 = box.x2 + pad, box.y2 + pad

    # Force the target aspect around the same centre before clamping.
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    if w / h > CROP_ASPECT:
        h = w / CROP_ASPECT
    else:
        w = h * CROP_ASPECT
    x1, x2 = cx - w / 2.0, cx + w / 2.0
    y1, y2 = cy - h / 2.0, cy + h / 2.0

    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(width, x2)), int(min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return cv2.resize(image, (FRAME_SIZE, FRAME_SIZE))
    return cv2.resize(image[y1:y2, x1:x2], (FRAME_SIZE, FRAME_SIZE))


def _holder_box(frame: Frame, track_id: int):
    for track in frame.players():
        if track.track_id == track_id:
            return track.box
    return None


def classify_windows(
    images: Sequence[np.ndarray],
    frames: Sequence[Frame],
    classifier: ActionClassifier,
    config: Config,
    holders: Sequence[int | None] | None = None,
) -> list[ActionWindow]:
    """Slide a window over the clip and classify the ball-handler in each one.

    `holders` is the per-frame possession timeline from stage 5. Without it every
    window falls back to the whole frame, which is only appropriate for a
    classifier trained on whole frames.
    """
    import cv2

    windows: list[ActionWindow] = []
    pending: list[tuple[int, np.ndarray]] = []
    for start, end in plan_windows(
        len(images), config.action_window_frames, config.action_stride_frames
    ):
        holder = None
        if holders is not None:
            seen = [h for h in holders[start : end + 1] if h is not None]
            if seen:
                holder = Counter(seen).most_common(1)[0][0]

        if holders is not None and holder is None:
            # Nobody has the ball across this window; there is no ball-handler
            # action to classify, so do not spend a forward pass guessing.
            label, conf = "other", 0.0
        else:
            crops = []
            last_box = None
            for i in range(start, end + 1):
                box = _holder_box(frames[i], holder) if holder is not None else None
                box = box or last_box
                last_box = box or last_box
                crop = (
                    crop_player(images[i], box)
                    if box is not None
                    else cv2.resize(images[i], (FRAME_SIZE, FRAME_SIZE))
                )
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            label, conf = None, None  # filled in by the batched pass below
            pending.append((len(windows), np.stack(crops)))

        windows.append(
            ActionWindow(
                start_index=start,
                end_index=end,
                start_time_s=frames[start].time_s,
                end_time_s=frames[end].time_s,
                label=label if label is not None else "other",
                conf=conf if conf is not None else 0.0,
            )
        )

    # One batched forward for every window that needs the model.
    if pending:
        outputs = _classify_batch_fallback(classifier, [clip for _, clip in pending])
        for (index, _), (label, conf) in zip(pending, outputs):
            if label not in ACTIONS:
                label = "other"
            windows[index] = replace(windows[index], label=label, conf=conf)
    return windows
