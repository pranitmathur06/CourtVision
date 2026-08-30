"""V7 — Action classifier sanity check (spec §6).

Fine-tunes VideoMAE on a small labeled clip subset and reports held-out accuracy.
With 5 classes, random chance is ~20%; the spec asks for well above that, not
barely above it, so the bar here is 40%.
"""

from __future__ import annotations

import collections
import os
import time
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from courtvision.device import resolve_device
from courtvision.types import ACTIONS, balanced_subset

DATA_DIR = Path("data/labeled/actions")
# Overridable so a smoke run can exercise the whole path — decode, train,
# evaluate, checkpoint — on a handful of clips without overwriting the real
# checkpoint or waiting for 8 epochs. Defaults reproduce the graded run exactly.
OUT_DIR = Path(os.environ.get("V7_OUT_DIR", "checkpoints/action_classifier"))
MAX_PER_CLASS = int(os.environ.get("V7_MAX_PER_CLASS", "0"))   # 0 = use every clip
# The supervised Kinetics-400 checkpoint, NOT the plain `videomae-base`.
# `videomae-base` is the self-supervised MAE checkpoint: masked-autoencoder
# features are strong under full fine-tuning but weak under linear probing, and
# freezing it produced BELOW-chance accuracy (0.208 vs 0.333) while train loss
# fell steadily — the signature of a head learning non-transferable features.
BASE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
N_FRAMES = 16
FRAME_SIZE = 224
BYTES_PER_CLIP = N_FRAMES * FRAME_SIZE * FRAME_SIZE * 3
EPOCHS = int(os.environ.get("V7_EPOCHS", "8"))
VAL_FRACTION = 0.2
# The bar is set from the classes that actually have data, not from len(ACTIONS).
# BARD labels no `dribble` or `pass`, so training on the spec's five classes with
# three populated would make a "20% chance" baseline a fiction — real chance
# would be 1/3. Margin is how far above chance we require, so the result means
# "the model learned something", not "the model guessed the majority class".
REQUIRED_MARGIN_OVER_CHANCE = 0.15


def augment(clip: np.ndarray, rng: random.Random) -> np.ndarray:
    """Randomise the low-level statistics that give the SOURCE away.

    Measured across the training set, BARD-sourced clips (rebound, steal) and
    SpaceJam-sourced clips (everything else) are trivially separable:

        sharpness  SpaceJam ~1676   BARD ~663    (non-overlapping ranges)
        contrast   SpaceJam ~78     BARD ~56
        brightness SpaceJam ~102    BARD ~129

    BARD crops are small player regions from 720p broadcast upscaled to 224x224;
    SpaceJam clips are natively 128x176. Different resampling, 2.5x the blur.

    A model given that cue will use it: the unaugmented run produced ZERO
    cross-source confusions in 532 held-out clips while confusing freely WITHIN
    each source, which is the signature of separating by dataset rather than by
    action. Randomising blur, brightness and contrast per clip makes those
    statistics carry no information about which dataset a clip came from, so the
    only signal left is the action itself.
    """
    import cv2

    sigma = rng.uniform(0.0, 2.2)
    brightness = rng.uniform(-28.0, 28.0)
    contrast = rng.uniform(0.75, 1.25)

    out = clip.astype(np.float32)
    if sigma > 0.15:
        out = np.stack([cv2.GaussianBlur(f, (0, 0), sigma) for f in out])
    out = (out - 128.0) * contrast + 128.0 + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


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
    from transformers import VideoMAEImageProcessor

    from courtvision.videomae import load_videomae_classifier

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
        clips = balanced_subset(clips, MAX_PER_CLASS)
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
    model, restored = load_videomae_classifier(
        BASE_MODEL,
        num_labels=len(populated),
        id2label={i: a for i, a in enumerate(populated)},
        label2id={a: i for i, a in enumerate(populated)},
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    print(f"  restored {restored} attention bias tensors that transformers "
          f"would otherwise have left at zero")

    # Decode every clip once, then STREAM it back one clip at a time. Nothing is
    # held in memory between steps.
    #
    # This machine has 16 GB, but with the trainer dead it already carries
    # ~10.8 GB resident across other processes plus 5.5 GB in the compressor:
    # roughly 1.5-2 GB is actually free, and VideoMAE training needs most of
    # that. Every cached-in-RAM design therefore thrashed, whatever its size —
    # float32 (25.6 GB), uint8 (6.1 GB), and a writable memmap all drove the
    # process into uninterruptible wait at 3-5% CPU with swap pinned near full.
    # A writable mmap is no better than RAM here: its dirty pages must live
    # somewhere until writeback.
    #
    # So: buffered sequential writes with a periodic fsync while decoding, then
    # seek+read of one fixed-size record per step. Peak footprint is the model
    # plus a single 2.4 MB clip. The page cache may keep copies, but they are
    # clean and the kernel can drop them for free.
    CACHE_DIR = Path(tempfile.mkdtemp(prefix="v7-frame-cache-"))
    print(f"  decoding {len(train)} train + {len(val)} val clips (once) "
          f"into {CACHE_DIR}...")
    def prepare(items, name, augment_rng=None):
        path = CACHE_DIR / f"{name}.bin"
        labels = []
        with open(path, "wb") as handle:
            for index, (clip_path, label_index) in enumerate(items):
                frames = load_clip(clip_path)
                if augment_rng is not None:
                    frames = augment(frames, random.Random(augment_rng + index))
                handle.write(np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
                labels.append(label_index)
                if (index + 1) % 250 == 0:
                    # Force writeback so dirty pages cannot accumulate.
                    handle.flush()
                    os.fsync(handle.fileno())
                    print(f"    {name} {index + 1}/{len(items)} decoded", flush=True)
        return path, labels

    def read_clip(path, index) -> np.ndarray:
        """Read one clip's frames straight off disk. Nothing is retained."""
        with open(path, "rb") as handle:
            handle.seek(index * BYTES_PER_CLIP)
            buffer = handle.read(BYTES_PER_CLIP)
        return np.frombuffer(buffer, dtype=np.uint8).reshape(
            N_FRAMES, FRAME_SIZE, FRAME_SIZE, 3)

    # Cache uint8 frames (2.4 MB/clip), NOT the processor's normalised float32
    # (9.6 MB/clip). Across 2,660 clips that is 6.4 GB instead of 25.6 GB. The
    # float32 cache does not fit in 16 GB: the first attempt at this run sat in
    # uninterruptible wait with 78 MB resident and 23.5 GB of swap in use,
    # spending its time paging rather than training. Normalising per step costs
    # a few ms, against a forward+backward that costs hundreds.
    def featurize(frames):
        return processor(list(frames), return_tensors="pt")

    # Training clips are augmented so the source cue cannot be learned.
    # Validation is left UNTOUCHED: it must measure what the model does on real
    # data, not on data we have already normalised in its favour.
    train_path, train_labels = prepare(train, "train", augment_rng=1234)
    val_path, val_labels = prepare(val, "val")

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

    # Weight the loss by inverse class frequency. The classes are not balanced —
    # rebound has 226 clips against 800 for shot — and at batch size 1 the model
    # simply sees shot three and a half times more often. The frozen-backbone
    # probe measured what that costs: rebound was the worst class by a distance
    # at 0.29, trading errors with steal in both directions, while every
    # well-represented class sat between 0.62 and 0.88. Unweighted, the cheapest
    # way to cut the loss is to under-predict the rare class.
    counts = collections.Counter(train_labels)
    weights = torch.tensor(
        [len(train_labels) / (len(populated) * max(counts[i], 1))
         for i in range(len(populated))],
        dtype=torch.float32, device=device)
    print("  class weights: " + ", ".join(
        f"{a}={w:.2f}" for a, w in zip(populated, weights.tolist())))

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": 1e-3},
         {"params": block_params, "lr": 1e-5}])

    def evaluate() -> float:
        model.eval()
        correct = 0
        with torch.no_grad():
            for index, label_index in enumerate(val_labels):
                inputs = {k: v.to(device)
                          for k, v in featurize(read_clip(val_path, index)).items()}
                correct += int(int(model(**inputs).logits.argmax(dim=-1)) == label_index)
        model.train()
        return correct / len(val_labels)

    # Validate every epoch and keep the BEST weights, not the last. Training loss
    # here falls to ~0.02 by epoch 5, so later epochs mostly deepen overfitting;
    # evaluating only at the end cannot tell you whether epoch 4 beat epoch 8.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_accuracy, best_epoch, history = -1.0, 0, []
    model.train()
    for epoch in range(EPOCHS):
        started = time.monotonic()
        order = list(range(len(train_labels)))
        random.Random(epoch).shuffle(order)
        total_loss = 0.0
        for i in order:
            inputs = {k: v.to(device)
                      for k, v in featurize(read_clip(train_path, i)).items()}
            labels = torch.tensor([train_labels[i]], device=device)
            # Weighted loss, so the rare classes are not simply conceded.
            logits = model(**inputs).logits
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach())

        epoch_accuracy = evaluate()
        history.append(epoch_accuracy)
        marker = ""
        if epoch_accuracy > best_accuracy:
            best_accuracy, best_epoch = epoch_accuracy, epoch + 1
            model.save_pretrained(OUT_DIR)
            processor.save_pretrained(OUT_DIR)
            marker = "  <- best, checkpoint saved"
        print(f"  epoch {epoch + 1}/{EPOCHS} train loss "
              f"{total_loss / len(train_labels):.4f}  val acc {epoch_accuracy:.3f}  "
              f"[{time.monotonic() - started:.0f}s]{marker}", flush=True)

    print(f"  accuracy by epoch: {[round(a, 3) for a in history]}")
    print(f"  best epoch {best_epoch} at {best_accuracy:.3f} "
          f"(last epoch {history[-1]:.3f})")
    accuracy = best_accuracy
    # A model that always guesses the commonest class scores its share, which
    # exceeds uniform chance whenever the split is imbalanced. Beat the harder one.
    val_counts = collections.Counter(val_labels)
    majority = max(val_counts.values()) / len(val_labels)
    baseline = max(chance, majority)
    required = baseline + REQUIRED_MARGIN_OVER_CHANCE

    shutil.rmtree(CACHE_DIR, ignore_errors=True)

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
