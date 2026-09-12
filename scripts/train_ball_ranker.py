"""Re-score the detector's ball candidates, because the problem is rank, not recall.

Measured on hand-located balls: at a 0.03 floor the detector puts a candidate
3-16 px from the truth, and that candidate carries a confidence of 0.05-0.11
while sitting 20th to 50th among 60-95 proposals. Four selection rules have
failed to dig it out -- the court-volume ray test, motion, raw large-inference
confidence, and two-scale agreement. The detector's own score is the thing that
is wrong, so the thing to fix is the score.

A DETECTOR is the wrong tool for that and a CLASSIFIER is the right one. The
boxes are already proposed; what is missing is a better opinion about each one.
So: a small convolutional net over the patch inside each candidate box, trained
to say ball or not-ball, used at inference to re-rank the candidates the
detector already gives. It trains in minutes rather than hours and it attacks
exactly the measured failure.

Positives are the track-labelled balls (`find_ball_tracks.py`), whose labels
were checked by eye before use. Negatives are the OTHER candidates in those
same frames -- heads, shoulders, headbands, shoes, the scorer's-table ball --
which is precisely the population the ranker has to beat, and they come free
with the positives rather than being invented.

The split is by TIME, not at random: patches from the same second are nearly
the same picture, and splitting them across train and validation would score
the model on what it memorised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PATCH = 48
#: Patches within this many seconds of each other go to the same side of the
#: split; a track supplies several near-identical frames.
SPLIT_BLOCK_S = 20.0


def patch_of(frame, box, size=PATCH, pad=1.6):
    """The square patch a candidate box sits in, normalised to `size`."""
    import cv2
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max((x2 - x1), (y2 - y1)) * pad / 2
    half = max(half, 8.0)
    h, w = frame.shape[:2]
    a, b = int(np.clip(cx - half, 0, w - 1)), int(np.clip(cx + half, 1, w))
    c, d = int(np.clip(cy - half, 0, h - 1)), int(np.clip(cy + half, 1, h))
    if b - a < 4 or d - c < 4:
        return None
    return cv2.resize(frame[c:d, a:b], (size, size), interpolation=cv2.INTER_CUBIC)


def build(video, labels, size=PATCH):
    """(patches, targets, times) from the track labels and their frame-mates."""
    import cv2
    capture = cv2.VideoCapture(video)
    patches, targets, times = [], [], []
    for row in labels:
        capture.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        good = patch_of(frame, row["ball"], size)
        if good is None:
            continue
        patches.append(good)
        targets.append(1)
        times.append(row["t"])
        for other in row.get("others", []):
            bad = patch_of(frame, other, size)
            if bad is None:
                continue
            patches.append(bad)
            targets.append(0)
            times.append(row["t"])
    capture.release()
    return np.array(patches), np.array(targets), np.array(times)


def split_by_time(times, valid_fraction=0.25, block_s=SPLIT_BLOCK_S):
    """Train/validation masks that never share a moment of the game."""
    blocks = np.floor(np.asarray(times) / block_s).astype(int)
    unique = np.unique(blocks)
    cut = int(len(unique) * (1 - valid_fraction))
    train_blocks = set(unique[:cut].tolist())
    is_train = np.array([b in train_blocks for b in blocks])
    return is_train, ~is_train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--labels", required=True, help="find_ball_tracks.py output")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out", default="checkpoints/ball_ranker.pt")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    from courtvision.device import resolve_device

    labels = json.load(open(args.labels))["frames"]
    patches, targets, times = build(args.video, labels)
    if not len(patches):
        print("no patches built")
        return 1
    train_mask, valid_mask = split_by_time(times)
    print(f"{len(patches)} patches, {int(targets.sum())} of them balls; "
          f"train {int(train_mask.sum())}, val {int(valid_mask.sum())}")
    if not valid_mask.any() or not train_mask.any():
        print("not enough distinct moments to split honestly")
        return 1

    device = resolve_device()
    x = torch.tensor(patches, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y = torch.tensor(targets, dtype=torch.float32)
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(64, 1)).to(device)
    # The negatives outnumber the positives many times over, which is the real
    # ratio at inference; the loss is weighted so the model still cares.
    weight = float((targets == 0).sum()) / max(float((targets == 1).sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)

    xt, yt = x[train_mask].to(device), y[train_mask].to(device)
    xv, yv = x[valid_mask].to(device), y[valid_mask].to(device)
    best = None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), 64):
            batch = order[i:i + 64]
            optimiser.zero_grad()
            loss = loss_fn(model(xt[batch]).squeeze(1), yt[batch])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(xv).squeeze(1))
        # The metric that matters is RANK: on a frame, does the ball outscore
        # its frame-mates? Average precision stands in for that here.
        order = torch.argsort(scores, descending=True)
        hits = yv[order]
        ranks = torch.cumsum(hits, 0) / torch.arange(1, len(hits) + 1, device=device)
        ap = float((ranks * hits).sum() / max(float(hits.sum()), 1.0))
        if best is None or ap > best:
            best = ap
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "patch": PATCH}, args.out)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1:3d}  val average precision {ap:.3f}", flush=True)
    baseline = float(yv.mean())
    print(f"best validation average precision {best:.3f} "
          f"(a coin weighted by the class balance would score {baseline:.3f})")
    print(f"  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
