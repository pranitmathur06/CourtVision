"""V7 — Action classifier sanity check (spec §6).

Fine-tunes VideoMAE on a small labeled clip subset and reports held-out accuracy.
With 5 classes, random chance is ~20%; the spec asks for well above that, not
barely above it, so the bar here is 40%.
"""

from __future__ import annotations

import collections
import random
import sys
from pathlib import Path

import numpy as np

from courtvision.device import resolve_device
from courtvision.types import ACTIONS

DATA_DIR = Path("data/labeled/actions")
OUT_DIR = Path("checkpoints/action_classifier")
# The supervised Kinetics-400 checkpoint, NOT the plain `videomae-base`.
# `videomae-base` is the self-supervised MAE checkpoint: masked-autoencoder
# features are strong under full fine-tuning but weak under linear probing, and
# freezing it produced BELOW-chance accuracy (0.208 vs 0.333) while train loss
# fell steadily — the signature of a head learning non-transferable features.
BASE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
N_FRAMES = 16
FRAME_SIZE = 224
EPOCHS = 8
VAL_FRACTION = 0.2
# The bar is set from the classes that actually have data, not from len(ACTIONS).
# BARD labels no `dribble` or `pass`, so training on the spec's five classes with
# three populated would make a "20% chance" baseline a fiction — real chance
# would be 1/3. Margin is how far above chance we require, so the result means
# "the model learned something", not "the model guessed the majority class".
REQUIRED_MARGIN_OVER_CHANCE = 0.15


def load_clip(path: Path) -> np.ndarray:
    """Decode exactly N_FRAMES evenly-spaced frames.

    grab() advances the decoder without producing an image; only the frames we
    actually want are retrieve()d and resized. Decoding every frame of a 720p
    clip to keep 16 of them made this script ~10x slower than the model itself.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise ValueError(f"no frames in {path}")

    wanted = set(np.linspace(0, total - 1, N_FRAMES).round().astype(int).tolist())
    frames, index = [], 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        if index in wanted:
            ok, image = capture.retrieve()
            if ok:
                frames.append(cv2.cvtColor(
                    cv2.resize(image, (FRAME_SIZE, FRAME_SIZE)), cv2.COLOR_BGR2RGB))
        index += 1
    capture.release()

    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    while len(frames) < N_FRAMES:      # short clip: repeat the last frame
        frames.append(frames[-1])
    return np.stack(frames[:N_FRAMES])


def main() -> int:
    import torch
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    if not DATA_DIR.is_dir():
        print(f"V7 FAIL — no labeled clips at {DATA_DIR}/<action>/*.mp4")
        return 1

    # Only classes with actual clips take part; see REQUIRED_MARGIN_OVER_CHANCE.
    populated = [a for a in ACTIONS if list((DATA_DIR / a).glob("*.mp4"))]
    missing = [a for a in ACTIONS if a not in populated]
    if len(populated) < 2:
        print(f"V7 FAIL — only {len(populated)} populated class(es) in {DATA_DIR}")
        return 1
    if missing:
        print(f"V7 note — no training clips for {missing}; training on {populated}. "
              "The classifier cannot predict a class it never saw.")

    samples: list[tuple[Path, int]] = []
    counts = {}
    for label_index, action in enumerate(populated):
        clips = sorted((DATA_DIR / action).glob("*.mp4"))
        counts[action] = len(clips)
        samples.extend((clip_path, label_index) for clip_path in clips)
    print(f"V7 clips per class: {counts}")

    if len(samples) < 20:
        print(f"V7 FAIL — only {len(samples)} labeled clips; need at least 20")
        return 1

    chance = 1.0 / len(populated)

    random.Random(0).shuffle(samples)
    split = int(len(samples) * (1 - VAL_FRACTION))
    train, val = samples[:split], samples[split:]

    device = resolve_device()
    processor = VideoMAEImageProcessor.from_pretrained(BASE_MODEL)
    model = VideoMAEForVideoClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(populated),
        id2label={i: a for i, a in enumerate(populated)},
        label2id={a: i for i, a in enumerate(populated)},
        ignore_mismatched_sizes=True,
    ).to(device)

    # Decode and preprocess every clip once, not once per epoch.
    print(f"  decoding {len(train)} train + {len(val)} val clips (once)...")
    def prepare(items):
        out = []
        for clip_path, label_index in items:
            tensors = processor(list(load_clip(clip_path)), return_tensors="pt")
            out.append(({k: v for k, v in tensors.items()}, label_index))
        return out

    train_cache, val_cache = prepare(train), prepare(val)

    # Train the head plus the last two encoder blocks. Full fine-tuning of 86M
    # parameters overfits a few hundred clips; a frozen backbone alone was not
    # adaptable enough. Unfreezing the top blocks is the middle ground, with a
    # much lower learning rate there than on the freshly-initialised head.
    ADAPT = ("encoder.layer.10.", "encoder.layer.11.", "fc_norm")
    head_params, block_params = [], []
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
            head_params.append(param)
        elif any(tag in name for tag in ADAPT):
            param.requires_grad = True
            block_params.append(param)
        else:
            param.requires_grad = False
    print(f"  training {sum(p.numel() for p in head_params + block_params):,} of "
          f"{sum(p.numel() for p in model.parameters()):,} parameters "
          f"(head + last 2 blocks)")

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": 1e-3},
         {"params": block_params, "lr": 1e-5}])
    model.train()
    for epoch in range(EPOCHS):
        order = list(range(len(train_cache)))
        random.Random(epoch).shuffle(order)
        total_loss = 0.0
        for i in order:
            inputs, label_index = train_cache[i]
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor([label_index], device=device)
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach())
        print(f"  epoch {epoch + 1}/{EPOCHS} train loss {total_loss / len(train_cache):.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, label_index in val_cache:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            predicted = int(model(**inputs).logits.argmax(dim=-1))
            correct += int(predicted == label_index)

    accuracy = correct / len(val)
    # A model that always guesses the commonest class scores its share, which
    # exceeds uniform chance whenever the split is imbalanced. Beat the harder one.
    val_counts = collections.Counter(label for _, label in val_cache)
    majority = max(val_counts.values()) / len(val_cache)
    baseline = max(chance, majority)
    required = baseline + REQUIRED_MARGIN_OVER_CHANCE
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)

    ok = accuracy >= required
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V7 {verdict} — held-out accuracy {accuracy:.3f} on {len(val)} clips "
        f"across {len(populated)} classes {populated} "
        f"(uniform chance {chance:.3f}, majority-class {majority:.3f}, "
        f"required {required:.3f}); saved to {OUT_DIR}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
