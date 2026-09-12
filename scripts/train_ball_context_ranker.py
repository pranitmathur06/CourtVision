"""Rank ball candidates on their NEIGHBOURHOOD, because the ball itself is 15 px.

Rendered at 4x, the candidates that outscore the true ball are human heads. At
2162.5 s a bald head carries 0.82 against the ball's 0.34; at 4362.5 s a
spectator's head carries 0.52 against 0.05. A head at 20 px is round,
skin-toned and textureless, which is a basketball, and that is also why the
orange prior failed in Round 84 -- heads are orange too.

What makes it obvious to a person is not the blob. It is the crowd around it:
a head sits in dense rows of spectators, a ball sits over the floor. So the
discriminating information is in the NEIGHBOURHOOD, and the patch ranker
rejected in Round 72 could not have found it -- it cropped 48 px with a pad of
1.6, about 30 px of context, which is the ball and nothing else.

This ranker takes CONTEXT_PX around the candidate instead: wide enough to hold
the floor, the crowd barrier, a body under a head. The patch is CENTRED on the
candidate, which is also what tells the model which object in a crowded
neighbourhood it is being asked about -- two candidates a few pixels apart get
two patches that differ by that shift. A first version added a fourth channel
marking the candidate; a test showed the mark was constant, because centring
had already put every candidate in the middle, so it was dropped rather than
kept as decoration.

Positives are the track-labelled balls, negatives their frame-mates -- the
population the ranker must actually beat, arriving free with the positives.
Both are passed through `filter_labels_by_shot.py` first: the unfiltered track
labels sit 1.7 s from an evaluation frame, which is the same picture.

The split is by TIME in blocks, because patches from the same second are nearly
the same picture and splitting them at random scores memorisation.

Whether this works is an open question and the honest failure mode is stated
now: the crowd/floor distinction is partly a COLOUR statistic of the
background, and this arena's crowd is uniformly blue. A model that learns "blue
surround means not-ball" will not survive a different arena, so the validation
split must be read as in-arena only, and any claim beyond this game needs the
second game's labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: The neighbourhood, in full-resolution pixels, resized to INPUT_PX.
CONTEXT_PX = 192
INPUT_PX = 64
#: Patches within this many seconds share a side of the split.
SPLIT_BLOCK_S = 20.0


def context_patch(frame, centre, context=CONTEXT_PX, size=INPUT_PX):
    """The neighbourhood around `centre`, centred on it and zero-padded at edges.

    Returns (size, size, 3) scaled to the unit range, or None when the centre
    lies outside the frame. Padding rather than sliding the window is what
    keeps the candidate in the middle, which is the model's only statement of
    which object it is scoring.
    """
    import cv2

    height, width = frame.shape[:2]
    half = context // 2
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x0, y0 = cx - half, cy - half
    patch = np.zeros((context, context, 3), np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + context, width), min(y0 + context, height)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return patch.astype(np.float32) / 255.0


def split_by_time(times, valid_fraction=0.25, block_s=SPLIT_BLOCK_S):
    """Train/validation masks that never share a moment of the game."""
    blocks = np.floor(np.asarray(times, np.float64) / block_s).astype(int)
    unique = np.unique(blocks)
    cut = int(len(unique) * (1 - valid_fraction))
    train_blocks = set(unique[:cut].tolist())
    is_train = np.array([b in train_blocks for b in blocks])
    return is_train, ~is_train


def build(video, labels):
    """(patches, targets, times) from track labels and their frame-mates."""
    import cv2

    capture = cv2.VideoCapture(video)
    patches, targets, times = [], [], []
    for row in labels:
        capture.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        ball = row["ball"]
        good = context_patch(frame, ((ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2))
        if good is None:
            continue
        patches.append(good)
        targets.append(1)
        times.append(row["t"])
        for other in row.get("others", []):
            bad = context_patch(frame, ((other[0] + other[2]) / 2,
                                        (other[1] + other[3]) / 2))
            if bad is None:
                continue
            patches.append(bad)
            targets.append(0)
            times.append(row["t"])
    capture.release()
    return np.array(patches, np.float32), np.array(targets), np.array(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True,
                        metavar="VIDEO:LABELS", help="repeatable")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", default="checkpoints/ball_context_ranker.pt")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    from courtvision.device import resolve_device

    patches, targets, times = [], [], []
    for spec in args.source:
        video, _, labels_path = spec.rpartition(":")
        rows = json.load(open(labels_path))["frames"]
        p, t, s = build(video, rows)
        if not len(p):
            continue
        patches.append(p)
        targets.append(t)
        times.append(s)
        print(f"  {Path(video).stem}: {len(p)} patches, {int(t.sum())} balls", flush=True)
    if not patches:
        print("no patches built")
        return 1
    patches = np.concatenate(patches)
    targets = np.concatenate(targets)
    times = np.concatenate(times)

    train_mask, valid_mask = split_by_time(times)
    print(f"{len(patches)} patches, {int(targets.sum())} balls; "
          f"train {int(train_mask.sum())}, val {int(valid_mask.sum())}")
    if not train_mask.any() or not valid_mask.any():
        print("not enough distinct moments to split honestly")
        return 1

    device = resolve_device()
    x = torch.tensor(patches).permute(0, 3, 1, 2)
    y = torch.tensor(targets, dtype=torch.float32)
    model = nn.Sequential(
        nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(96, 1)).to(device)
    weight = float((targets == 0).sum()) / max(float((targets == 1).sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    optimiser = torch.optim.Adam(model.parameters(), lr=1.5e-3)

    xt, yt = x[train_mask].to(device), y[train_mask].to(device)
    xv, yv = x[valid_mask].to(device), y[valid_mask].to(device)
    best = None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), 64):
            batch = order[i:i + 64]
            optimiser.zero_grad()
            loss_fn(model(xt[batch]).squeeze(1), yt[batch]).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(xv).squeeze(1))
        order = torch.argsort(scores, descending=True)
        hits = yv[order]
        precision = torch.cumsum(hits, 0) / torch.arange(1, len(hits) + 1, device=device)
        average = float((precision * hits).sum() / max(float(hits.sum()), 1.0))
        if best is None or average > best:
            best = average
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "context": CONTEXT_PX,
                        "input": INPUT_PX}, args.out)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1:3d}  val average precision {average:.3f}",
                  flush=True)
    print(f"best validation average precision {best:.3f} "
          f"(the class balance alone would score {float(yv.mean()):.3f})")
    print(f"  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
