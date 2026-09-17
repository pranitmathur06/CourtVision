"""Score the trained possession kernels against the shipped handler class.

PAIRED, BECAUSE THE FRAMES ARE THE SAME FRAMES. Two Wilson intervals on one
157-frame set overlap at almost any effect this project can produce -- the
interval is about half-width 8 points and the honest effects are 5. But the two
methods answer the SAME frames, so the question is not "are these two rates
different" but "when they disagree, who is right more often", and that is
McNemar's test on the discordant pairs. It is the right test here and it is a
far more powerful one.

WHAT IS COMPARED
  the detector's handler class     what ships today, 49.7% on these frames
  kernel 1 alone                   the centre frame's softmax
  kernels 1 + 2                    the posterior after the scan over time

A frame counts when the labeller said somebody had the ball. 'missing' -- he had
it and the detector drew no box for him -- is a miss for every method, because
none of the candidates is the right answer.

SCORED BY IoU 0.5 AGAINST THE LABELLER'S BOX, WHICH IS NOT THE SAME AS NAMING
THE RIGHT INDEX. `eval_handler.py` -- where the 50% the handler class scores
comes from -- asks whether the box a method points at overlaps the box a person
drew. Players overlap on a basketball court, so two candidates can both clear
0.5 on the same man, and scoring one of them right and the other wrong measures
the box list rather than the method. Scoring these kernels by index while the
baseline was scored by overlap understated them by seven frames.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def wilson(hits, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(a: np.ndarray, b: np.ndarray):
    """Exact two-sided McNemar on paired right/wrong vectors.

    Only the frames where the two disagree carry information. Under the null
    each disagreement is a fair coin, so the p-value is a binomial tail -- and
    exact rather than the chi-square approximation, because with 20-odd
    discordant pairs the approximation is not trustworthy.
    """
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(b & ~a))
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    smaller = min(only_a, only_b)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2.0 ** n)
    return only_a, only_b, min(1.0, 2.0 * tail)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--weights", default="checkpoints/possession/temporal.json")
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--images", default="data/labeling")
    args = parser.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO

    from courtvision.device import resolve_device
    from courtvision.kernels.possession import (BETA, EPS, GRID, SURROUND,
                                                temporal_torch)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "trainer", Path(__file__).with_name("train_possession_temporal.py"))
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    weights = json.load(open(args.weights))
    parameters = {k: torch.tensor(np.asarray(v), dtype=torch.float64)
                  for k, v in weights.items()}
    rows = [r for r in trainer.load_windows(args.windows) if r["split"] == "eval"]
    detector = YOLO(args.detector)
    device = resolve_device()

    names, truth_boxes = [], {}
    for folder in ("possession", "handler"):
        path = Path(args.images) / folder / "manifest.json"
        if path.exists():
            for m in json.load(open(path))["frames"]:
                truth_boxes[m["file"]] = (Path(args.images) / folder / "images"
                                          / m["file"], m.get("boxes", []))
    # the box a person actually drew, which is what everything is scored against
    drawn = {}
    for labels in ("data/labels/possession_labels.json",
                   "data/labels/handler_labels.json"):
        if Path(labels).exists():
            for r in json.load(open(labels))["frames"]:
                if r.get("handler_box"):
                    drawn[r["file"]] = r["handler_box"]

    right = {"handler_class": [], "kernel1": [], "kernel1+2": []}
    for row in rows:
        if row["verdict"] == "nobody":
            continue
        names.append(row["name"])
        if row["verdict"] == "missing" or row["target"] < 0:
            for key in right:
                right[key].append(False)
            continue
        window = trainer.open_window(row, torch, torch.float64, with_patches=True)
        with torch.no_grad():
            scores = trainer.window_scores(window, parameters, torch, GRID, BETA,
                                           SURROUND, EPS)
            tracks = scores.shape[1] - 1
            alone = int(torch.argmax(scores[row["centre"], :tracks]))
            posterior = temporal_torch(scores, parameters["stay_raw"], row["centre"])
            together = int(torch.argmax(posterior[:tracks]))
        image_path, boxes = truth_boxes[row["name"]]
        candidates = np.asarray(boxes, dtype=np.float64)
        truth = drawn.get(row["name"])
        right["kernel1"].append(bool(truth) and iou(candidates[alone], truth) >= 0.5)
        right["kernel1+2"].append(bool(truth) and iou(candidates[together], truth) >= 0.5)

        # the shipped method, on the same frame: its most confident handler box
        image = cv2.imread(str(image_path))
        found = detector.predict(image, device=device, verbose=False,
                                 imgsz=1280, conf=0.25)[0].boxes
        handlers = []
        if found is not None and len(found):
            for cls, conf, box in zip(found.cls.cpu().numpy(),
                                      found.conf.cpu().numpy(),
                                      found.xyxy.cpu().numpy()):
                if detector.names[int(cls)] == "handler":
                    handlers.append((float(conf), [float(v) for v in box]))
        right["handler_class"].append(
            bool(handlers) and bool(truth) and iou(max(handlers)[1], truth) >= 0.5)

    total = len(names)
    print(f"  {total} held-out frames where somebody had the ball\n")
    table = {k: np.array(v) for k, v in right.items()}
    for key, label in (("handler_class", "the detector's handler class"),
                       ("kernel1", "kernel 1 alone, the centre frame"),
                       ("kernel1+2", "kernels 1+2, the scan over time")):
        hits = int(table[key].sum())
        low, high = wilson(hits, total)
        print(f"    {label:<36} {hits:>3}/{total} = {hits / total:5.1%}"
              f"   (95% CI {low:.0%}-{high:.0%})")

    print("\n  paired, on the frames where they disagree (exact McNemar):")
    for a, b in (("kernel1+2", "handler_class"), ("kernel1+2", "kernel1"),
                 ("kernel1", "handler_class")):
        only_a, only_b, p = mcnemar(table[a], table[b])
        verdict = "significant" if p < 0.05 else "not significant"
        print(f"    {a:<10} vs {b:<14} {only_a:>3} frames only {a} gets right, "
              f"{only_b:>3} only {b}   p = {p:.4f}  ({verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
