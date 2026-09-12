"""Score a ball detector AND its selector on the unbiased hand-located truth.

Declared before the numbers exist, and deliberately reporting three things
that Round 74 through Round 84 kept conflating:

    proposed   a candidate lies within tolerance SOMEWHERE in the list.
               This is the ceiling: no selector can beat it.
    delivered  the candidate the system would actually report -- the
               highest-scoring one -- is within tolerance. This is the number
               the Phase 2 gate is about.
    rank       where the true ball sits in the scored order, when proposed.

A single figure hid a real result once already: the ball detector was recorded
as "3 of 10, unchanged" while being run at imgsz 2560 on a 1280x720 frame,
twice the size its crops were cut at. At its own scale the same weights propose
three candidates instead of fifty-nine and put the ball at rank 1 or 2. The
ceiling and the delivery had moved in opposite directions and one number could
not say so.

`--ranker` re-scores the detector's candidates with the context ranker instead
of using the detector's confidence. Both orderings are reported from the same
candidate list, so the comparison is of SELECTORS and not of two pipelines.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def wilson(hits, total, z=1.96):
    """95% interval for a proportion; small samples need it stated."""
    if total == 0:
        return 0.0, 1.0
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def scored_order(candidates, scores):
    """Indices of `candidates` best first, ties broken by the earlier one."""
    return sorted(range(len(candidates)), key=lambda i: (-scores[i], i))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--truth", action="append", required=True,
                        help="repeatable; ball_truth_*.json")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="1280 on a 1280x720 frame is the NATIVE scale the "
                             "crops were cut at; anything larger shows the model "
                             "a ball bigger than it was trained on")
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--ranker", default=None,
                        help="train_ball_context_ranker.py output")
    args = parser.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO

    from courtvision.device import resolve_device

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_ball_context_ranker import context_patch

    rows, tolerance = [], None
    for path in args.truth:
        data = json.load(open(path))
        tolerance = float(data.get("tolerance_px", tolerance or 28.0))
        rows.extend(data["frames"])
    device = resolve_device()
    model = YOLO(args.detector)

    ranker = None
    if args.ranker:
        import torch.nn as nn
        blob = torch.load(args.ranker, map_location="cpu", weights_only=False)
        ranker = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(96, 1))
        ranker.load_state_dict(blob["state"])
        ranker.eval().to(device)

    capture = cv2.VideoCapture(args.video)
    proposed = delivered_conf = delivered_rank = 0
    ranks_conf, ranks_rank, counts = [], [], []
    for row in rows:
        capture.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        found = model.predict(frame, device=device, verbose=False,
                              imgsz=args.imgsz, conf=args.conf)[0].boxes
        candidates, confidences = [], []
        if found is not None and len(found):
            names = model.names
            for cls, conf, box in zip(found.cls.cpu().numpy(),
                                      found.conf.cpu().numpy(),
                                      found.xyxy.cpu().numpy()):
                if names[int(cls)] != "ball":
                    continue
                candidates.append(((float(box[0]) + float(box[2])) / 2,
                                   (float(box[1]) + float(box[3])) / 2))
                confidences.append(float(conf))
        counts.append(len(candidates))
        if not candidates:
            print(f"{row['t']:8.1f}  no candidate at all")
            continue
        tx, ty = row["ball"]
        gaps = [np.hypot(cx - tx, cy - ty) for cx, cy in candidates]
        nearest = int(np.argmin(gaps))
        if gaps[nearest] > tolerance:
            print(f"{row['t']:8.1f}  {len(candidates):3d} candidates, nearest "
                  f"{gaps[nearest]:5.0f} px -- not proposed")
            continue
        proposed += 1

        order = scored_order(candidates, confidences)
        rank = order.index(nearest) + 1
        ranks_conf.append(rank)
        delivered_conf += int(rank == 1)

        line = (f"{row['t']:8.1f}  {len(candidates):3d} candidates, "
                f"ball at rank {rank} by confidence")
        if ranker is not None:
            patches = [context_patch(frame, c) for c in candidates]
            usable = [i for i, p in enumerate(patches) if p is not None]
            if usable:
                batch = torch.tensor(np.stack([patches[i] for i in usable])).permute(
                    0, 3, 1, 2).to(device)
                with torch.no_grad():
                    got = torch.sigmoid(ranker(batch).squeeze(1)).cpu().numpy()
                scores = np.full(len(candidates), -1.0)
                scores[usable] = got
                order = scored_order(candidates, scores)
                rank = order.index(nearest) + 1
                ranks_rank.append(rank)
                delivered_rank += int(rank == 1)
                line += f", rank {rank} by the ranker"
        print(line)
    capture.release()

    total = len(rows)
    print(f"\n{total} hand-located balls, tolerance {tolerance:.0f} px, "
          f"{np.mean(counts):.1f} candidates a frame")
    for name, hits, ranks in (("proposed (the ceiling)", proposed, None),
                              ("delivered, by confidence", delivered_conf, ranks_conf),
                              ("delivered, by the ranker", delivered_rank, ranks_rank)):
        if name.endswith("ranker") and ranker is None:
            continue
        low, high = wilson(hits, total)
        extra = f"   ranks {sorted(ranks)}" if ranks else ""
        print(f"  {name:26s} {hits:2d}/{total:<3d} {hits / total:.3f}   "
              f"95% CI {low:.3f}-{high:.3f}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
