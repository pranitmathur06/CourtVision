"""Binary rebound detector on FULL FRAMES, as a second view beside the crop model.

The action classifier reads a crop centred on the ball-handler. That works for
ball-proximate actions — dribble 0.90, block 0.86, shot 0.87 — because the ball
is inside the crop. It cannot work for a rebound, which is the ball coming off
the rim with players converging: the event is outside the frame, so `rebound`
became the crop model's label for anything generic and claimed 82% of a game.

Switching everything to full frames is not available: SpaceJam ships
pre-cropped clips with no recoverable source, so dribble and pass can only ever
be crops. Hence a second, narrow model that answers one question the first one
cannot — is this window a rebound — from the view that can actually see it.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

BASE = "MCG-NJU/videomae-base-finetuned-kinetics"
DATA = Path("data/rim")
OUT = Path("checkpoints/rim_detector")
EPOCHS = 6
VAL_FRACTION = 0.2


def main() -> int:
    import torch
    from transformers import VideoMAEImageProcessor

    from courtvision.device import resolve_device
    from courtvision.types import stratified_split
    from courtvision.videomae import load_videomae_classifier
    from scripts.validate_v7 import load_clip

    per_class = {name: sorted((DATA / name).glob("*.mp4"))
                 for name in ("background", "rebound")
                 if (DATA / name).is_dir()}
    if len(per_class) < 2:
        print(f"FAIL — need both classes under {DATA}, found {list(per_class)}")
        return 1
    counts = {k: len(v) for k, v in per_class.items()}
    print(f"rim detector clips: {counts}")
    # Same per-class split as V7, so one class's size cannot move the other's.
    train, val = stratified_split(per_class, VAL_FRACTION)
    order = {name: i for i, name in enumerate(sorted(per_class))}
    train = [(p, order[p.parent.name]) for p, _ in train]
    val = [(p, order[p.parent.name]) for p, _ in val]
    print(f"  train {len(train)}, val {len(val)}  labels {order}")

    device = resolve_device()
    processor = VideoMAEImageProcessor.from_pretrained(BASE)
    model, restored = load_videomae_classifier(
        BASE, num_labels=2, ignore_mismatched_sizes=True,
        id2label={i: n for n, i in order.items()},
        label2id=dict(order))
    model = model.to(device)
    print(f"  restored {restored} attention bias tensors")

    def decode(items):
        return [(load_clip(p), y) for p, y in items]

    train_data, val_data = decode(train), decode(val)

    ADAPT = ("encoder.layer.10.", "encoder.layer.11.", "fc_norm")
    head, blocks = [], []
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
            head.append(param)
        elif any(t in name for t in ADAPT):
            param.requires_grad = True
            blocks.append(param)
        else:
            param.requires_grad = False
    optimizer = torch.optim.AdamW([{"params": head, "lr": 1e-3},
                                   {"params": blocks, "lr": 1e-5}])

    def evaluate():
        model.eval()
        correct = 0
        per_class_hits: dict[int, list[int]] = {0: [0, 0], 1: [0, 0]}
        with torch.no_grad():
            for frames, y in val_data:
                x = {k: v.to(device) for k, v in
                     processor(list(frames), return_tensors="pt").items()}
                pred = int(model(**x).logits.argmax(-1))
                correct += pred == y
                per_class_hits[y][1] += 1
                per_class_hits[y][0] += pred == y
        model.train()
        return correct / len(val_data), per_class_hits

    OUT.mkdir(parents=True, exist_ok=True)
    best = -1.0
    model.train()
    for epoch in range(EPOCHS):
        idx = list(range(len(train_data)))
        random.Random(epoch).shuffle(idx)
        total = 0.0
        for i in idx:
            frames, y = train_data[i]
            x = {k: v.to(device) for k, v in
                 processor(list(frames), return_tensors="pt").items()}
            loss = model(**x, labels=torch.tensor([y], device=device)).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(loss.detach())
        accuracy, hits = evaluate()
        mark = ""
        if accuracy > best:
            best = accuracy
            model.save_pretrained(OUT)
            processor.save_pretrained(OUT)
            mark = "  <- best, saved"
        detail = {n: f"{hits[i][0]}/{hits[i][1]}" for n, i in order.items()}
        print(f"  epoch {epoch + 1}/{EPOCHS} loss {total / len(train_data):.4f} "
              f"acc {accuracy:.3f} {detail}{mark}", flush=True)

    majority = max(sum(1 for _, y in val_data if y == v) for v in (0, 1)) / len(val_data)
    print(f"\n  best {best:.3f} against a {majority:.3f} majority baseline")
    if best < majority + 0.10:
        print("RIM FAIL — not enough separation to be worth consulting")
        return 1
    print(f"RIM PASS — {best:.3f} on {len(val_data)} held-out full-frame windows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
