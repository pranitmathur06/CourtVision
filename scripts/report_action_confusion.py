"""Per-class accuracy and confusion for the trained action classifier.

Rebound clips come from BARD while every other class comes from SpaceJam. If the
model were separating by SOURCE rather than by action, rebound would score
near-perfectly while the SpaceJam classes confused among themselves. This checks
for exactly that.

Reuses validate_v7's split (same seed and ordering) so the held-out clips here are
the ones the model was scored on.
"""

from __future__ import annotations

import collections
import os
import random
import sys
from pathlib import Path

from courtvision.device import resolve_device
from courtvision.types import ACTIONS, balanced_subset, clip_source

DATA_DIR = Path("data/labeled/actions")
# Same overrides as validate_v7, so a smoke run can exercise this report
# against a throwaway checkpoint on the same split the model was scored on.
MODEL_DIR = Path(os.environ.get("V7_OUT_DIR", "checkpoints/action_classifier"))
MAX_PER_CLASS = int(os.environ.get("V7_MAX_PER_CLASS", "0"))
VAL_FRACTION = 0.2


def main() -> int:
    import torch
    from transformers import VideoMAEImageProcessor

    from courtvision.videomae import load_videomae_classifier
    from scripts.validate_v7 import load_clip

    if not MODEL_DIR.is_dir():
        print(f"FAIL — no trained classifier at {MODEL_DIR}")
        return 1

    populated = [a for a in ACTIONS if list((DATA_DIR / a).glob("*.mp4"))]
    samples: list[tuple[Path, int]] = []
    for index, action in enumerate(populated):
        clips = sorted((DATA_DIR / action).glob("*.mp4"))
        clips = balanced_subset(clips, MAX_PER_CLASS)
        samples.extend((p, index) for p in clips)
    random.Random(0).shuffle(samples)
    val = samples[int(len(samples) * (1 - VAL_FRACTION)):]

    # Which corpus each class actually draws from, measured, not assumed.
    source_of = []
    for action in populated:
        found = {clip_source(p) for p in (DATA_DIR / action).glob("*.mp4")}
        source_of.append("both" if len(found) > 1 else next(iter(found), "?"))
    print("class -> corpus: " + ", ".join(
        f"{a}={s}" for a, s in zip(populated, source_of)) + "\n")

    device = resolve_device()
    processor = VideoMAEImageProcessor.from_pretrained(str(MODEL_DIR))
    model, _ = load_videomae_classifier(str(MODEL_DIR))
    model = model.to(device).eval()

    confusion: collections.Counter = collections.Counter()
    with torch.no_grad():
        for clip_path, label_index in val:
            inputs = processor(list(load_clip(clip_path)), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            predicted = int(model(**inputs).logits.argmax(dim=-1))
            confusion[(label_index, predicted, clip_source(clip_path))] += 1

    print(f"per-class accuracy on {len(val)} held-out clips\n")
    print(f"  {'class':<9}{'source':<10}{'n':>5}{'acc':>7}   most confused with")
    shared = []
    for true_index, action in enumerate(populated):
        per_source = {}
        for source in ("SpaceJam", "BARD"):
            total = sum(v for (t, _, src), v in confusion.items()
                        if t == true_index and src == source)
            if not total:
                continue
            hit = confusion.get((true_index, true_index, source), 0)
            wrong = sorted(((v, p) for (t, p, src), v in confusion.items()
                            if t == true_index and src == source and p != true_index),
                           reverse=True)
            note = f"{populated[wrong[0][1]]} ({wrong[0][0]})" if wrong else "-"
            per_source[source] = hit / total
            print(f"  {action:<9}{source:<10}{total:>5}{hit/total:>7.2f}   {note}")
        if len(per_source) == 2:
            shared.append((action, per_source))

    # A class drawn from BOTH corpora is the honest test. If the model learned
    # the action, it scores similarly on that class whichever corpus the clip
    # came from. A large gap means it is still leaning on corpus identity.
    if shared:
        print("\n  cross-source generalisation on classes present in both corpora")
        worst = 0.0
        for action, per_source in shared:
            gap = abs(per_source["SpaceJam"] - per_source["BARD"])
            worst = max(worst, gap)
            print(f"    {action:<9} SpaceJam {per_source['SpaceJam']:.2f}  "
                  f"BARD {per_source['BARD']:.2f}  gap {gap:.2f}")
        print(f"    worst gap: {worst:.2f}"
              f"{'  — LARGE, still corpus-dependent' if worst > 0.25 else '  — acceptable'}")
    else:
        print("\n  NO class is populated from both corpora, so corpus membership")
        print("  still predicts the label and no honest cross-source test exists.")
        print("  Run: scripts/add_bard_action.py --action shot --count 400")

    cross = sum(v for (t, p, _), v in confusion.items()
                if t != p and source_of[t] != source_of[p] and
                source_of[t] != "both" and source_of[p] != "both")
    total = sum(confusion.values())
    print(f"\n  cross-corpus confusions (single-corpus classes only): "
          f"{cross}/{total} ({cross/max(total,1):.1%})")
    print("  Zero here means the model never mistakes a BARD-only action for a")
    print("  SpaceJam-only one, which is what reading the corpus would look like.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
