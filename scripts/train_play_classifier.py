"""Recognise basketball actions from film, on MultiSports' human labels.

Every other play measurement in this project ran on SportVU tracking
coordinates, which exist for 636 historical games and never for the film a
coach actually studies. This runs on broadcast video, and its labels are
human-annotated action tubes -- a person marked which player, in which frames,
performing which action -- rather than rules this project wrote itself.

Seven classes, chosen so the task is the real one:

  screen, pick_and_roll_defensive, sag, drive, interfere_shot
      the plays, offensive and defensive
  pass, dribble
      the background they have to be told apart FROM

A model that separates a screen from `walk` has learned nothing. These classes
overlap in space and time and often involve the same two players, which is what
makes the task worth measuring.

The split is by VIDEO: two tubes from one possession share players, court and
camera, so splitting by clip would leak. The test set is read once.
"""

from __future__ import annotations

import argparse
import collections
import glob
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

ROOT = Path("data/labeled/multisports/clips")
CLASSES = ["screen", "pick_and_roll_defensive", "sag", "drive",
           "interfere_shot", "pass", "dribble"]
PLAYS = CLASSES[:5]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_split(split):
    xs, ys = [], []
    for index, name in enumerate(CLASSES):
        for path in sorted(glob.glob(str(ROOT / split / name / "*.npy"))):
            xs.append(np.load(path)); ys.append(index)
    return np.stack(xs), np.array(ys)


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


def report(name, truth, guess):
    print(f"\n  {name} ({len(truth)} clips)")
    overall = float((guess == truth).mean())
    se = math.sqrt(overall * (1 - overall) / len(truth))
    print(f"    overall accuracy {overall:.0%} "
          f"(95% {max(0, overall-1.96*se):.0%} to {min(1, overall+1.96*se):.0%})")
    print(f"    {'class':<26}{'n':>5}{'recall':>9}{'precision':>11}")
    for index, label in enumerate(CLASSES):
        real = truth == index
        said = guess == index
        if not real.sum():
            continue
        recall = float((guess[real] == index).mean())
        precision = float((truth[said] == index).mean()) if said.sum() else 0.0
        print(f"    {label:<26}{int(real.sum()):>5}{recall:>8.0%}{precision:>10.0%}")
    play_ids = [CLASSES.index(p) for p in PLAYS]
    mask = np.isin(truth, play_ids)
    if mask.sum():
        among = float((guess[mask] == truth[mask]).mean())
        print(f"    the five PLAY classes alone: {among:.0%} "
              f"on {int(mask.sum())} clips (chance {1/len(CLASSES):.0%})")


def main() -> int:
    import torch
    from courtvision.videomae import load_videomae_classifier

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-from", type=int, default=10,
                        help="first transformer block to unfreeze")
    parser.add_argument("--batch", type=int, default=6)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    xtr, ytr = load_split("train")
    xv, yv = load_split("val")
    xte, yte = load_split("test")
    print(f"  train {len(ytr)}  val {len(yv)}  test {len(yte)}")
    print("  train mix:", dict(collections.Counter(
        CLASSES[i] for i in ytr).most_common()), flush=True)

    model, restored = load_videomae_classifier(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        num_labels=len(CLASSES), ignore_mismatched_sizes=True)
    model = model.to(device)
    trainable = []
    for name, param in model.named_parameters():
        keep = name.startswith("classifier") or name.startswith("fc_norm")
        if ".layer." in name:
            keep = keep or int(name.split(".layer.")[1].split(".")[0]) >= args.train_from
        param.requires_grad = keep
        if keep:
            trainable.append(param)
    print(f"  restored {restored} attention biases; training "
          f"{sum(p.numel() for p in trainable)/1e6:.1f}M parameters", flush=True)

    # The play classes are outnumbered by pass and dribble roughly four to one,
    # so the loss is weighted; without it the model answers "pass" and scores
    # well on an average that hides every class we care about.
    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(float)
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5,
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
        if best is None or acc > best[0]:
            best = (acc, epoch)
            torch.save(model.state_dict(), "checkpoints/play_classifier.pt")

    model.load_state_dict(torch.load("checkpoints/play_classifier.pt",
                                     map_location=device))
    report("VAL", yv, predict(model, xv, device))
    report("TEST — read once", yte, predict(model, xte, device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
