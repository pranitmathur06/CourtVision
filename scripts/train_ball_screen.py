"""A ball-screen detector trained on two independent label sources.

Two things the seven-way run established. Screen errors go almost entirely to
`pick_and_roll_defensive` -- the same ball screen labelled from the two sides
of the ball -- so those belong in one class. And the tactical classes overfit
at 152-223 training clips, so the shortfall is data.

Both public sources label screens: SpaceJam has 712 `pick` clips, MultiSports
223 `screen` plus 203 `pick_and_roll_defensive`. Pooling them roughly triples
the positives.

The test is deliberately split BY SOURCE. The two datasets crop differently --
SpaceJam tight to one player, MultiSports at 1.6x so the screened man is in
frame -- and are drawn from different games and cameras. A model that holds up
on both has learned the action; one that holds up on the source it saw most of
has learned the dataset. A pooled test number would hide the difference.
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

SPACEJAM = Path("data/labeled/picks")
MULTISPORTS = Path("data/labeled/multisports/clips")
MS_POSITIVE = ("screen", "pick_and_roll_defensive")
MS_NEGATIVE = ("pass", "dribble", "sag", "drive", "interfere_shot")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE, FRAMES = 224, 16


def read_mp4(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(img, (SIZE, SIZE))[:, :, ::-1])
    cap.release()
    if not frames:
        return None
    while len(frames) < FRAMES:
        frames.append(frames[-1])
    return np.stack([frames[int(i * (len(frames) - 1) / (FRAMES - 1))]
                     for i in range(FRAMES)]).astype(np.uint8)


def load_source(split):
    """(clips, labels, source) for one split, from both datasets."""
    xs, ys, src = [], [], []
    for label, name in ((1, "pick"), (0, "not_pick")):
        for path in sorted(glob.glob(str(SPACEJAM / split / name / "*.mp4"))):
            clip = read_mp4(path)
            if clip is not None:
                xs.append(clip); ys.append(label); src.append("spacejam")
    for name in MS_POSITIVE + MS_NEGATIVE:
        label = 1 if name in MS_POSITIVE else 0
        for path in sorted(glob.glob(str(MULTISPORTS / split / name / "*.npy"))):
            xs.append(np.load(path)); ys.append(label); src.append("multisports")
    return np.stack(xs), np.array(ys), np.array(src)


def as_tensor(raw):
    import torch
    arr = raw.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr.transpose(0, 1, 4, 2, 3).copy())


def predict(model, x, device, batch=8):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits = model(pixel_values=as_tensor(x[i:i + batch]).to(device)).logits
            out.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(out)


def score(name, truth, guess):
    tp = int(((guess == 1) & (truth == 1)).sum())
    fp = int(((guess == 1) & (truth == 0)).sum())
    fn = int(((guess == 0) & (truth == 1)).sum())
    tn = int(((guess == 0) & (truth == 0)).sum())
    n = len(truth)
    if not n:
        return
    acc = (tp + tn) / n
    se = math.sqrt(acc * (1 - acc) / n)
    recall = tp / max(tp + fn, 1)
    rse = math.sqrt(recall * (1 - recall) / max(tp + fn, 1))
    print(f"    {name:<14}{n:>5} clips   accuracy {acc:.0%} "
          f"({max(0,acc-1.96*se):.0%}-{min(1,acc+1.96*se):.0%})   "
          f"recall {recall:.0%} ({max(0,recall-1.96*rse):.0%}-"
          f"{min(1,recall+1.96*rse):.0%})   precision {tp/max(tp+fp,1):.0%}")


def main() -> int:
    import torch
    from courtvision.videomae import load_videomae_classifier

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-from", type=int, default=10)
    parser.add_argument("--batch", type=int, default=6)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    xtr, ytr, str_ = load_source("train")
    xv, yv, sv = load_source("val")
    xte, yte, ste = load_source("test")
    print(f"  train {len(ytr)} ({int(ytr.sum())} screens)  "
          f"val {len(yv)}  test {len(yte)}")
    for source in ("spacejam", "multisports"):
        m = str_ == source
        print(f"    {source}: {int(m.sum())} train clips, "
              f"{int(ytr[m].sum())} screens", flush=True)

    model, _ = load_videomae_classifier(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        num_labels=2, ignore_mismatched_sizes=True)
    model = model.to(device)
    trainable = []
    for name, param in model.named_parameters():
        keep = name.startswith("classifier") or name.startswith("fc_norm")
        if ".layer." in name:
            keep = keep or int(name.split(".layer.")[1].split(".")[0]) >= args.train_from
        param.requires_grad = keep
        if keep:
            trainable.append(param)
    counts = np.bincount(ytr, minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1),
                           dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)

    best = None
    for epoch in range(args.epochs):
        model.train()
        order = np.random.RandomState(epoch).permutation(len(ytr))
        started, total = time.time(), 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            logits = model(pixel_values=as_tensor(xtr[idx]).to(device)).logits
            loss = loss_fn(logits, torch.from_numpy(ytr[idx]).long().to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(idx)
        guess = predict(model, xv, device)
        acc = float((guess == yv).mean())
        print(f"  epoch {epoch}  loss {total/len(ytr):.3f}  val {acc:.0%}  "
              f"({time.time()-started:.0f}s)", flush=True)
        if best is None or acc > best:
            best = acc
            torch.save(model.state_dict(), "checkpoints/ball_screen.pt")

    model.load_state_dict(torch.load("checkpoints/ball_screen.pt",
                                     map_location=device))
    guess = predict(model, xte, device)
    print("\n  TEST — read once, reported per source")
    score("pooled", yte, guess)
    for source in ("spacejam", "multisports"):
        m = ste == source
        score(source, yte[m], guess[m])
    print("\n  For reference: SpaceJam alone as a binary task was 87%;")
    print("  MultiSports `screen` inside the seven-way task was 71% recall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
