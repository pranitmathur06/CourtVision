"""Can a trivial classifier still tell BARD from SpaceJam after augmentation?

V7 scored 0.810 but with 0/532 cross-source confusions: the model could tell the
two corpora apart, so part of that score was reading dataset identity rather than
basketball. The fix was blur/brightness/contrast augmentation on training clips.

This checks whether that fix actually works, WITHOUT training anything. If cheap
image statistics still separate the sources after augmentation, the leak is still
open and a GPU run would just reproduce it more expensively.

Verdict is the cross-validated accuracy of a logistic regression on those
statistics. Chance is 0.5. Near 1.0 means the sources remain trivially separable.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/labeled/actions")
BARD = {"rebound", "steal"}
PER_SOURCE = 80          # clips sampled from each corpus
FRAMES_PER_CLIP = 4      # frames averaged per clip


def features(frames: np.ndarray) -> np.ndarray:
    """Cheap per-clip statistics, averaged over sampled frames."""
    import cv2

    rows = []
    for frame in frames:
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        edges = cv2.Canny(grey, 100, 200)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(grey.astype(np.float32))))
        h, w = spectrum.shape
        centre = spectrum[h // 4:3 * h // 4, w // 4:3 * w // 4].sum()
        rows.append([
            cv2.Laplacian(grey, cv2.CV_64F).var(),   # sharpness
            float(grey.mean()), float(grey.std()),   # brightness, contrast
            float(hsv[..., 1].mean()),               # saturation
            float(edges.mean()),                     # edge density
            float(centre / max(spectrum.sum(), 1)),  # low-frequency share
            *[float(frame[..., c].mean()) for c in range(3)],
        ])
    return np.asarray(rows).mean(axis=0)


NAMES = ["sharpness", "brightness", "contrast", "saturation",
         "edge_density", "low_freq_share", "mean_R", "mean_G", "mean_B"]


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from scripts.validate_v7 import augment, load_clip

    rng = random.Random(0)
    clips: list[tuple[Path, int]] = []
    for action_dir in sorted(DATA_DIR.iterdir()):
        if not action_dir.is_dir():
            continue
        found = sorted(action_dir.glob("*.mp4"))
        clips.extend((p, int(action_dir.name in BARD)) for p in found)
    if not clips:
        print(f"FAIL — no clips under {DATA_DIR}")
        return 1

    by_source: dict[int, list[Path]] = {0: [], 1: []}
    for path, source in clips:
        by_source[source].append(path)
    sample: list[tuple[Path, int]] = []
    for source, paths in by_source.items():
        rng.shuffle(paths)
        sample.extend((p, source) for p in paths[:PER_SOURCE])
    print(f"sampling {sum(1 for _, s in sample if s == 0)} SpaceJam and "
          f"{sum(1 for _, s in sample if s == 1)} BARD clips\n")

    raw_rows, aug_rows, labels = [], [], []
    for index, (path, source) in enumerate(sample):
        frames = load_clip(path)
        picks = np.linspace(0, len(frames) - 1, FRAMES_PER_CLIP).round().astype(int)
        raw_rows.append(features(frames[picks]))
        aug_rows.append(features(augment(frames, random.Random(1234 + index))[picks]))
        labels.append(source)

    X_raw, X_aug, y = np.array(raw_rows), np.array(aug_rows), np.array(labels)

    def separability(X):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        return cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()

    raw_acc, aug_acc = separability(X_raw), separability(X_aug)

    print(f"  {'feature':<16}{'SpaceJam':>11}{'BARD':>11}   overlap?")
    for i, name in enumerate(NAMES):
        a, b = X_aug[y == 0, i], X_aug[y == 1, i]
        overlap = not (a.max() < b.min() or b.max() < a.min())
        print(f"  {name:<16}{a.mean():>11.1f}{b.mean():>11.1f}   "
              f"{'yes' if overlap else 'NO — disjoint ranges'}")

    print(f"\n  source separability (5-fold CV, chance = 0.500)")
    print(f"    raw clips:       {raw_acc:.3f}")
    print(f"    after augment:   {aug_acc:.3f}")

    # Augmentation is meant to destroy the source cue, not merely dent it.
    if aug_acc > 0.75:
        print(f"\n  LEAK OPEN — statistics alone still identify the corpus at "
              f"{aug_acc:.1%}.\n  Training on this data will reproduce the "
              f"dataset-bias result. Fix the\n  augmentation before spending GPU "
              f"time.")
        return 1
    print(f"\n  LEAK CLOSED — {aug_acc:.1%} is near chance, so the cheap cues are "
          f"gone.\n  A model that still separates sources would have to be using "
          f"something\n  subtler, which training can legitimately be asked about.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
