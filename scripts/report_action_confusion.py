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
import random
import sys
from pathlib import Path

from courtvision.device import resolve_device
from courtvision.types import ACTIONS

DATA_DIR = Path("data/labeled/actions")
MODEL_DIR = Path("checkpoints/action_classifier")
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
        samples.extend((p, index) for p in sorted((DATA_DIR / action).glob("*.mp4")))
    random.Random(0).shuffle(samples)
    val = samples[int(len(samples) * (1 - VAL_FRACTION)):]

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
            confusion[(label_index, predicted)] += 1

    print(f"per-class accuracy on {len(val)} held-out clips\n")
    print(f"  {'class':<9}{'n':>5}{'acc':>7}   most confused with")
    for true_index, action in enumerate(populated):
        total = sum(v for (t, _), v in confusion.items() if t == true_index)
        hit = confusion.get((true_index, true_index), 0)
        if not total:
            continue
        wrong = sorted(((v, p) for (t, p), v in confusion.items()
                        if t == true_index and p != true_index), reverse=True)
        note = f"{populated[wrong[0][1]]} ({wrong[0][0]})" if wrong else "-"
        source = "BARD" if action in ("rebound", "steal") else "SpaceJam"
        print(f"  {action:<9}{total:>5}{hit/total:>7.2f}   {note:<18} [{source}]")

    BARD = {"rebound", "steal"}
    def group_acc(names):
        pairs = [(confusion.get((i, i), 0),
                  sum(v for (t, _), v in confusion.items() if t == i))
                 for i, a in enumerate(populated) if a in names]
        total = sum(n for _, n in pairs)
        return (sum(h for h, _ in pairs) / total) if total else 0.0

    bard_names = {a for a in populated if a in BARD}
    sj_names = {a for a in populated if a not in BARD}
    bard_acc, sj_acc = group_acc(bard_names), group_acc(sj_names)

    # Cross-source confusion is the real tell: if the model reads SOURCE rather
    # than action, BARD classes and SpaceJam classes would never be mistaken for
    # one another.
    cross = sum(v for (t, p), v in confusion.items()
                if (populated[t] in BARD) != (populated[p] in BARD))
    total = sum(confusion.values())

    print(f"\n  BARD classes    {sorted(bard_names)}: {bard_acc:.2f}")
    print(f"  SpaceJam classes {sorted(sj_names)}: {sj_acc:.2f}")
    print(f"  cross-source confusions: {cross}/{total} ({cross/max(total,1):.1%})")
    print("\n  If the model were separating by SOURCE, the BARD group would be")
    print("  near-perfect AND cross-source confusions would be ~0. Mistakes that")
    print("  cross the source boundary mean it is reading the action instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
