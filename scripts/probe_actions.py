"""Linear probe on frozen VideoMAE features — real 7-class numbers without a GPU.

This is NOT V7. V7 fine-tunes the last two encoder blocks and needs ~2 GB,
which this machine does not have: a reduced attempt measured 3.9 s of CPU per
60 s of wall clock, 6% of one core, because it was paging. A forward pass under
no_grad peaks at 0.48 GB, so features can be extracted here and a classifier
trained on top of them.

A frozen backbone is a weaker model than fine-tuning, and an earlier attempt at
one scored 0.208 — below chance. That was on `videomae-base`, the
self-supervised MAE checkpoint, whose features are known to probe badly. This
uses the supervised Kinetics checkpoint, whose features are meant to transfer.
Whether they transfer to cropped basketball clips is exactly what this measures.

Its purpose is to answer the question V7 exists to answer, a run earlier: does
the classifier separate the seven actions, and does it generalise across corpora
on the classes drawn from both? Treat the accuracy as a floor for V7, not as a
substitute for it.
"""

from __future__ import annotations

import collections
import random
import sys
from pathlib import Path

import numpy as np

from courtvision.types import ACTIONS, balanced_subset, clip_source

DATA_DIR = Path("data/labeled/actions")
BASE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
CACHE = Path("outputs/action_features.npz")
VAL_FRACTION = 0.2


def build_split() -> tuple[list[tuple[Path, int]], list[str]]:
    """Exactly validate_v7's split: same seed, ordering and fraction."""
    populated = [a for a in ACTIONS if list((DATA_DIR / a).glob("*.mp4"))]
    samples: list[tuple[Path, int]] = []
    for label_index, action in enumerate(populated):
        for clip in balanced_subset(list((DATA_DIR / action).glob("*.mp4")), 0):
            samples.append((clip, label_index))
    random.Random(0).shuffle(samples)
    return samples, populated


def extract(samples, populated) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from transformers import VideoMAEImageProcessor

    from courtvision.device import resolve_device
    from courtvision.videomae import load_videomae_classifier
    from scripts.validate_v7 import load_clip

    if CACHE.exists():
        blob = np.load(CACHE, allow_pickle=True)
        if len(blob["paths"]) == len(samples) and list(blob["paths"]) == [
            str(p) for p, _ in samples
        ]:
            print(f"  reusing cached features from {CACHE}")
            return blob["features"], blob["labels"]
        print("  cache does not match the current clip set; re-extracting")

    device = resolve_device()
    processor = VideoMAEImageProcessor.from_pretrained(BASE_MODEL)
    model, restored = load_videomae_classifier(BASE_MODEL)
    model = model.to(device).eval()
    print(f"  device={device}, restored {restored} attention bias tensors")

    features = np.zeros((len(samples), 768), dtype=np.float32)
    labels = np.zeros(len(samples), dtype=np.int64)
    with torch.no_grad():
        for index, (clip_path, label_index) in enumerate(samples):
            inputs = processor(list(load_clip(clip_path)), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pooled = model.videomae(**inputs).last_hidden_state.mean(dim=1)
            features[index] = pooled.squeeze(0).float().cpu().numpy()
            labels[index] = label_index
            if (index + 1) % 250 == 0:
                print(f"    {index + 1}/{len(samples)} clips", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, features=features, labels=labels,
                        paths=np.array([str(p) for p, _ in samples]))
    return features, labels


def main() -> int:
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    samples, populated = build_split()
    counts = collections.Counter(populated[i] for _, i in samples)
    print(f"probe over {len(samples)} clips, {len(populated)} classes: {dict(counts)}\n")

    features, labels = extract(samples, populated)
    split = int(len(samples) * (1 - VAL_FRACTION))
    X_train, y_train = features[:split], labels[:split]
    X_val, y_val = features[split:], labels[split:]
    val_paths = [p for p, _ in samples[split:]]
    print(f"\n  train {len(X_train)}, val {len(X_val)}")

    # An MLP head, not logistic regression. Measured on the same cached
    # features: 0.750 against 0.708, so the frozen features carry more signal
    # than a linear head extracts. It does NOT help rebound (0.316 against
    # 0.289), because rebound's problem is the features, not the head — rebound
    # against steal as a two-class problem scores 0.682 against a 0.658
    # majority baseline.
    model = make_pipeline(StandardScaler(),
                          MLPClassifier((256,), max_iter=600, random_state=0))
    model.fit(X_train, y_train)
    predicted = model.predict(X_val)
    accuracy = float((predicted == y_val).mean())

    counts_val = collections.Counter(y_val.tolist())
    majority = max(counts_val.values()) / len(y_val)
    chance = 1.0 / len(populated)
    print(f"  accuracy {accuracy:.3f}  (uniform chance {chance:.3f}, "
          f"majority class {majority:.3f})\n")

    print(f"  {'class':<9}{'corpus':<10}{'n':>5}{'acc':>7}   most confused with")
    shared = []
    for index, action in enumerate(populated):
        per_corpus = {}
        for corpus in ("SpaceJam", "BARD"):
            mask = [i for i in range(len(y_val))
                    if y_val[i] == index and clip_source(val_paths[i]) == corpus]
            if not mask:
                continue
            hit = sum(predicted[i] == index for i in mask)
            wrong = collections.Counter(
                populated[predicted[i]] for i in mask if predicted[i] != index)
            note = f"{wrong.most_common(1)[0][0]} ({wrong.most_common(1)[0][1]})" \
                if wrong else "-"
            per_corpus[corpus] = hit / len(mask)
            print(f"  {action:<9}{corpus:<10}{len(mask):>5}{hit/len(mask):>7.2f}   {note}")
        if len(per_corpus) == 2:
            shared.append((action, per_corpus))

    if shared:
        print("\n  cross-corpus generalisation (classes drawn from both)")
        worst = max(abs(v["SpaceJam"] - v["BARD"]) for _, v in shared)
        for action, v in shared:
            print(f"    {action:<9} SpaceJam {v['SpaceJam']:.2f}  BARD {v['BARD']:.2f}"
                  f"  gap {abs(v['SpaceJam'] - v['BARD']):.2f}")
        print(f"    worst gap {worst:.2f}"
              f"{'  — still corpus-dependent' if worst > 0.25 else '  — acceptable'}")

    # Which corpus each class draws from, measured rather than assumed.
    corpus_of = []
    for action in populated:
        found = {clip_source(c) for c in (DATA_DIR / action).glob("*.mp4")}
        corpus_of.append("both" if len(found) > 1 else next(iter(found), "?"))

    # The old gate asked for cross-source confusions to be non-zero. It only
    # means anything between classes that live in ONE corpus each: mistaking a
    # BARD-only action for a SpaceJam-only one is a mistake the model could not
    # make if it were reading the dataset instead of the action.
    single = [i for i, c in enumerate(corpus_of) if c != "both"]
    cross = sum(1 for i in range(len(y_val))
                if y_val[i] in single and predicted[i] in single
                and y_val[i] != predicted[i]
                and corpus_of[y_val[i]] != corpus_of[predicted[i]])
    eligible = sum(1 for i in range(len(y_val)) if y_val[i] in single)
    print(f"\n  cross-corpus confusions between single-corpus classes: "
          f"{cross}/{eligible}")
    print("  (was 0/532 before the rebalance — the model could not confuse them")
    print("   because corpus membership told it which group a clip belonged to)")

    print("\n  confusion matrix (rows = truth)")
    header = "".join(f"{a[:5]:>7}" for a in populated)
    print(f"    {'':<9}{header}")
    for i, action in enumerate(populated):
        row = "".join(
            f"{sum(1 for k in range(len(y_val)) if y_val[k] == i and predicted[k] == j):>7}"
            for j in range(len(populated)))
        print(f"    {action:<9}{row}")

    print(f"\n  This is a frozen-backbone probe, not V7. Read it as a floor.")
    return 0 if accuracy > majority else 1


if __name__ == "__main__":
    sys.exit(main())
