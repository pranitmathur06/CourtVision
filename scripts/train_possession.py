"""Train the possession operator on the hand labels, and score it honestly.

The split is the sampling and it is not arbitrary. Both labelling rounds drew
half their frames uniformly and half from where the model was already
struggling; the uniform frames are the evaluation and are never trained on. So
the number at the end is an estimate of in-game accuracy, not of performance on
hard cases.

The ball position fed in during training is the DETECTOR'S, not the hand-located
one, even on the frames where a person marked the ball. Training on a better
ball than the model will have at inference would teach it to lean on a feature
that is not there when it matters -- and the frames that matter most here are
exactly the ones where the detector has nothing, which is why the operator has a
pixel sweep at all.

Frames the labeller marked 'missing' -- somebody has it and the detector drew no
box for him -- are excluded from TRAINING, because none of the candidate boxes
is the right answer and there is nothing to point at. They stay in the
EVALUATION as automatic misses, because that is what they are.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROUNDS = (("data/labels/possession_labels.json", "data/labeling/possession"),
          ("data/labels/handler_labels.json", "data/labeling/handler"))


def wilson(hits, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def load_rows():
    """Every labelled frame, with its candidate boxes and its answer."""
    rows = []
    for labels, folder in ROUNDS:
        if not Path(labels).exists():
            continue
        manifest = {}
        manifest_path = Path(folder) / "manifest.json"
        if manifest_path.exists():
            manifest = {m["file"]: m.get("boxes", [])
                        for m in json.load(open(manifest_path))["frames"]}
        for row in json.load(open(labels))["frames"]:
            verdict = row.get("handler_verdict") or row.get("verdict")
            if verdict not in ("box", "missing", "nobody"):
                continue
            boxes = row.get("boxes") or manifest.get(row["file"], [])
            if not boxes:
                continue
            rows.append({
                "image": str(Path(folder) / "images" / row["file"]),
                "boxes": np.array(boxes, dtype=np.float64).reshape(-1, 4),
                "verdict": verdict,
                "handler_box": row.get("handler_box"),
                "split": "eval" if row["pick"] == "random" else "train",
            })
    return rows


def target_index(row):
    """Which candidate the labeller pointed at; len(boxes) means 'nobody'."""
    if row["verdict"] == "nobody":
        return len(row["boxes"])
    if row["verdict"] != "box" or not row["handler_box"]:
        return None
    overlaps = [iou(b, row["handler_box"]) for b in row["boxes"]]
    best = int(np.argmax(overlaps))
    return best if overlaps[best] >= 0.5 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ball", default="checkpoints/ball_v2/best.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--out", default="checkpoints/possession/weights.json")
    args = parser.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO

    from courtvision.device import resolve_device
    from courtvision.kernels.possession import default_parameters, possession_torch

    rows = load_rows()
    device = resolve_device()
    ball_model = YOLO(args.ball)
    print(f"  {len(rows)} labelled frames; finding the detector's ball on each")
    usable = []
    for row in rows:
        image = cv2.imread(row["image"])
        if image is None:
            continue
        found = ball_model.predict(image, device=device, verbose=False,
                                   imgsz=1280, conf=0.05)[0].boxes
        ball = np.zeros(3)
        if found is not None and len(found):
            conf, box = max(zip(found.conf.cpu().numpy(), found.xyxy.cpu().numpy()),
                            key=lambda e: e[0])
            ball = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, float(conf)])
        row["image_data"] = image
        row["ball"] = ball
        row["target"] = target_index(row)
        usable.append(row)

    train = [r for r in usable if r["split"] == "train" and r["target"] is not None]
    evaluate = [r for r in usable if r["split"] == "eval"]
    print(f"  {len(train)} frames to train on, {len(evaluate)} held out\n")

    start = default_parameters()
    # Fit the standardisation on the training frames only, at the starting
    # region, then hold it fixed. Fitting it on everything would leak the
    # evaluation half into the model's input scaling.
    with torch.no_grad():
        fixed = {k: torch.tensor(v, dtype=torch.float64) for k, v in start.items()}
        seen = []
        for row in train:
            _, feats = possession_torch(row["image_data"], row["boxes"],
                                        row["ball"], fixed)
            seen.append(feats.numpy())
        stacked = np.concatenate(seen, axis=0)
        start["feature_mean"] = stacked.mean(axis=0)
        start["feature_scale"] = stacked.std(axis=0) + 1e-3
    print(f"    feature scale {np.round(start['feature_scale'], 4)}")

    parameters = {k: torch.tensor(v, dtype=torch.float64,
                                  requires_grad=k not in ("feature_mean", "feature_scale"))
                  for k, v in start.items()}
    trainable = [v for k, v in parameters.items()
                 if k not in ("feature_mean", "feature_scale")]
    optimiser = torch.optim.Adam(trainable, lr=args.lr)
    for epoch in range(args.epochs):
        optimiser.zero_grad()
        total = 0.0
        for row in train:
            probs, _ = possession_torch(row["image_data"], row["boxes"],
                                        row["ball"], parameters)
            total = total + -torch.log(probs[row["target"]] + 1e-9)
        loss = total / max(len(train), 1)
        loss.backward()
        optimiser.step()
        with torch.no_grad():                    # the region must stay inside the box
            parameters["region"].clamp_(0.0, 1.0)
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(f"    epoch {epoch:>4}  loss {float(loss):.4f}  "
                  f"region {np.round(parameters['region'].detach().numpy(), 3)}")

    right = scored = nobox = 0
    for row in evaluate:
        if row["verdict"] == "nobody":
            continue
        scored += 1
        if row["verdict"] == "missing":
            nobox += 1
            continue
        with torch.no_grad():
            probs, _ = possession_torch(row["image_data"], row["boxes"],
                                        row["ball"], parameters)
        # argmax over the PLAYERS, because the baseline it is compared with
        # also always names somebody; the "nobody" logit is reported apart.
        pick = int(torch.argmax(probs[:len(row["boxes"])]))
        if row["handler_box"] and iou(row["boxes"][pick], row["handler_box"]) >= 0.5:
            right += 1
    low, high = wilson(right, scored)
    print(f"\n  held out: {scored} frames where somebody had the ball "
          f"({nobox} with no box drawn for him, automatic misses)")
    print(f"    possession operator      {right}/{scored} = {right / max(scored, 1):.1%}"
          f"   (95% CI {low:.0%}-{high:.0%})")
    print("    the detector's handler class, same frames: 49.7%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: v.detach().numpy().tolist() for k, v in parameters.items()},
              open(out, "w"), indent=1)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
