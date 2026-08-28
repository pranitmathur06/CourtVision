"""V7 — Action classifier sanity check (spec §6).

Fine-tunes VideoMAE on a small labeled clip subset and reports held-out accuracy.
With 5 classes, random chance is ~20%; the spec asks for well above that, not
barely above it, so the bar here is 40%.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

from courtvision.device import resolve_device
from courtvision.types import ACTIONS

DATA_DIR = Path("data/labeled/actions")
OUT_DIR = Path("checkpoints/action_classifier")
BASE_MODEL = "MCG-NJU/videomae-base"
N_FRAMES = 16
FRAME_SIZE = 224
EPOCHS = 8
VAL_FRACTION = 0.2
CHANCE = 1.0 / len(ACTIONS)
REQUIRED_ACCURACY = 0.40


def load_clip(path: Path) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(
            cv2.cvtColor(cv2.resize(image, (FRAME_SIZE, FRAME_SIZE)), cv2.COLOR_BGR2RGB)
        )
    capture.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    # Sample N_FRAMES evenly, repeating the last frame if the clip is short.
    indices = np.linspace(0, len(frames) - 1, N_FRAMES).round().astype(int)
    return np.stack([frames[i] for i in indices])


def main() -> int:
    import torch
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    if not DATA_DIR.is_dir():
        print(f"V7 FAIL — no labeled clips at {DATA_DIR}/<action>/*.mp4")
        return 1

    samples: list[tuple[Path, int]] = []
    for label_index, action in enumerate(ACTIONS):
        for clip_path in sorted((DATA_DIR / action).glob("*.mp4")):
            samples.append((clip_path, label_index))

    if len(samples) < 20:
        print(f"V7 FAIL — only {len(samples)} labeled clips; need at least 20")
        return 1

    random.Random(0).shuffle(samples)
    split = int(len(samples) * (1 - VAL_FRACTION))
    train, val = samples[:split], samples[split:]

    device = resolve_device()
    processor = VideoMAEImageProcessor.from_pretrained(BASE_MODEL)
    model = VideoMAEForVideoClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(ACTIONS),
        id2label={i: a for i, a in enumerate(ACTIONS)},
        label2id={a: i for i, a in enumerate(ACTIONS)},
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(EPOCHS):
        random.Random(epoch).shuffle(train)
        total_loss = 0.0
        for clip_path, label_index in train:
            inputs = processor(list(load_clip(clip_path)), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor([label_index], device=device)
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss)
        print(f"  epoch {epoch + 1}/{EPOCHS} train loss {total_loss / len(train):.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for clip_path, label_index in val:
            inputs = processor(list(load_clip(clip_path)), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            predicted = int(model(**inputs).logits.argmax(dim=-1))
            correct += int(predicted == label_index)

    accuracy = correct / len(val)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)

    ok = accuracy >= REQUIRED_ACCURACY
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V7 {verdict} — held-out accuracy {accuracy:.3f} on {len(val)} clips "
        f"(chance {CHANCE:.3f}, required {REQUIRED_ACCURACY:.2f}); saved to {OUT_DIR}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
