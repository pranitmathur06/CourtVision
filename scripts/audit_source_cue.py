"""Does corpus membership predict the LABEL? That is the confound that matters.

V7 scored 0.810 with 0/532 cross-source confusions, so part of that score was
reading dataset identity rather than basketball. The first attempted fix was
blur/brightness/contrast augmentation, aimed at making the corpora look alike.
That was the wrong target, and this script is what showed it:

  raw clips                        0.988 separable
  after augmentation               0.950
  greyscale + standardisation      0.981

The corpora stay separable because they ARE different video. That is harmless
on its own. The damage came from composition — every rebound/steal clip was
BARD and every other class SpaceJam, so corpus membership PREDICTED the label
for 660 of 2,660 clips and the model could score on a quarter of the label mass
without learning an action.

So the headline number here is not "can statistics tell the corpora apart"
(they always will) but "how much of the label does knowing the corpus buy you".
That is computed exactly from the class composition, no sampling and no model.
"""

from __future__ import annotations

import collections
import random
import sys
from pathlib import Path

import numpy as np

from courtvision.types import clip_source

DATA_DIR = Path("data/labeled/actions")
PER_SOURCE = 80
FRAMES_PER_CLIP = 4


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
            cv2.Laplacian(grey, cv2.CV_64F).var(),
            float(grey.mean()), float(grey.std()),
            float(hsv[..., 1].mean()),
            float(edges.mean()),
            float(centre / max(spectrum.sum(), 1)),
            *[float(frame[..., c].mean()) for c in range(3)],
        ])
    return np.asarray(rows).mean(axis=0)


NAMES = ["sharpness", "brightness", "contrast", "saturation",
         "edge_density", "low_freq_share", "mean_R", "mean_G", "mean_B"]


def composition() -> dict[str, collections.Counter]:
    """clips per (class, corpus), attributed per clip rather than per class."""
    table: dict[str, collections.Counter] = {}
    for action_dir in sorted(DATA_DIR.iterdir()):
        if not action_dir.is_dir():
            continue
        counts: collections.Counter = collections.Counter()
        for clip in action_dir.glob("*.mp4"):
            counts[clip_source(clip)] += 1
        if counts:
            table[action_dir.name] = counts
    return table


def confound_gain(table: dict[str, collections.Counter]) -> tuple[float, float, float]:
    """Best label accuracy from corpus alone, vs from the majority class alone.

    Guessing the commonest class overall needs no corpus knowledge. Guessing the
    commonest class WITHIN each corpus is the best a pure source-detector can
    do. The gap is exactly what the confound is worth to a model.
    """
    per_source: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    totals: collections.Counter = collections.Counter()
    for action, counts in table.items():
        for source, n in counts.items():
            per_source[source][action] += n
            totals[action] += n
    grand = sum(totals.values())
    majority = max(totals.values()) / grand
    source_only = sum(max(c.values()) for c in per_source.values()) / grand
    return source_only, majority, source_only - majority


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from scripts.validate_v7 import augment, load_clip

    table = composition()
    if not table:
        print(f"FAIL — no clips under {DATA_DIR}")
        return 1

    print(f"  {'class':<9}{'SpaceJam':>10}{'BARD':>7}   drawn from")
    for action, counts in sorted(table.items()):
        both = len(counts) > 1
        print(f"  {action:<9}{counts.get('SpaceJam', 0):>10}{counts.get('BARD', 0):>7}   "
              f"{'BOTH corpora' if both else next(iter(counts))+' only'}")

    source_only, majority, gain = confound_gain(table)
    print(f"\n  label accuracy from CORPUS alone:   {source_only:.3f}")
    print(f"  label accuracy from majority class: {majority:.3f}")
    print(f"  what knowing the corpus buys:       {gain:+.3f}")

    rng = random.Random(0)
    by_source: dict[str, list[Path]] = collections.defaultdict(list)
    for action_dir in sorted(DATA_DIR.iterdir()):
        if action_dir.is_dir():
            for clip in sorted(action_dir.glob("*.mp4")):
                by_source[clip_source(clip)].append(clip)
    sample: list[tuple[Path, int]] = []
    for source, paths in by_source.items():
        rng.shuffle(paths)
        sample.extend((p, int(source == "BARD")) for p in paths[:PER_SOURCE])

    picks = np.linspace(0, 15, FRAMES_PER_CLIP).round().astype(int)
    raw_rows, aug_rows, labels = [], [], []
    for index, (path, source) in enumerate(sample):
        frames = load_clip(path)
        raw_rows.append(features(frames[picks]))
        aug_rows.append(features(augment(frames, random.Random(1234 + index))[picks]))
        labels.append(source)
    X_raw, X_aug = np.nan_to_num(np.array(raw_rows)), np.nan_to_num(np.array(aug_rows))
    y = np.array(labels)

    def separability(X):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        return cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()

    print(f"\n  corpus separability from image statistics "
          f"({int((y == 0).sum())} SpaceJam / {int((y == 1).sum())} BARD, chance 0.500)")
    print(f"    raw:           {separability(X_raw):.3f}")
    print(f"    after augment: {separability(X_aug):.3f}")
    print("    Expected to stay high — they are genuinely different video. This is")
    print("    only harmful when corpus also predicts the label.")

    # A pure source-detector should do no better than ignoring the source.
    if gain > 0.05:
        print(f"\n  CONFOUND OPEN — corpus membership is worth {gain:+.3f} of label")
        print( "  accuracy. Populate more classes from both corpora before training.")
        return 1
    print(f"\n  CONFOUND CLOSED — corpus membership is worth only {gain:+.3f}, so a")
    print( "  model cannot score by recognising the dataset. Cross-source")
    print( "  generalisation on shared classes is now a meaningful test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
