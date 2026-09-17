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

TWO CORRECTIONS, made when this was first pointed at a uniform sample and its
answer disagreed with `eval_possession.py` on the same 130 frames:

  IT WAS SCORING THE WRONG PICTURE. Truth times were written at 30.0 fps and
  rounded to 0.1 s, so seeking the video by timestamp lands a frame or more
  away, and a ball crosses several of its own widths in 33 ms. It now reads the
  JPEG the labeller looked at, named by the truth file's `frames_dir`, and only
  seeks when there is none. Worth 11 points.

  IT WAS ASKING THE WRONG QUESTION. `delivered` asked whether the candidate
  NEAREST to truth ranked first. Two detections often land on the same ball, and
  when the other one outranks it that reads as a miss although the system --
  which reports its top-scoring candidate -- was right. It now asks whether the
  reported candidate is within tolerance, which is what the product does. Worth
  another 10 points.

With both, this file and `eval_possession.py` agree to the frame: 102/130.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from courtvision.stats import wilson  # noqa: E402




def first_within(order, gaps, tolerance):
    """Rank of the best-scoring candidate that is ACTUALLY ON the ball.

    Not the rank of the single nearest one. Two detections often land on the
    same ball, and asking whether the NEAREST of them ranks first calls it a
    miss whenever the other one outranks it -- while the system, which reports
    its top-scoring candidate, was right. Scored that way this file read 92/130
    where eval_possession.py, asking whether the reported candidate is within
    tolerance, read 102/130 on the same frames with the same weights. The
    difference was ten frames of definition, not of detector.
    """
    for place, index in enumerate(order, start=1):
        if gaps[index] <= tolerance:
            return place
    return len(order) + 1


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
    parser.add_argument("--frames", default=None,
                        help="override the truth file's frames_dir")
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

    rows, tolerance, frames_dir_hint = [], None, None
    for path in args.truth:
        data = json.load(open(path))
        tolerance = float(data.get("tolerance_px", tolerance or 28.0))
        frames_dir_hint = data.get("frames_dir") or frames_dir_hint
        rows.extend(data["frames"])
    if args.frames:
        frames_dir_hint = args.frames
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

    # Read the frame the LABELLER looked at when the truth file names one. The
    # labelled times were written at 30.0 fps and rounded to 0.1 s, so seeking
    # the video by timestamp lands a frame or more away -- and at 33 ms a ball
    # crosses several of its own widths. Scored by seeking, this detector reads
    # 57% on the same 130 frames that eval_possession.py, which reads the saved
    # JPEG, scores at 78.5%. The frame, not the detector, was the difference.
    frames_dir = Path(frames_dir_hint) if frames_dir_hint else None
    capture = cv2.VideoCapture(args.video)
    proposed = delivered_conf = delivered_rank = 0
    ranks_conf, ranks_rank, counts = [], [], []
    read_from_disk = 0
    for row in rows:
        saved = frames_dir / row["file"] if frames_dir and row.get("file") else None
        if saved is not None and saved.exists():
            frame = cv2.imread(str(saved))
            read_from_disk += 1
        else:
            frame = None
        if frame is None:
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
        rank = first_within(order, gaps, tolerance)
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
                rank = first_within(order, gaps, tolerance)
                ranks_rank.append(rank)
                delivered_rank += int(rank == 1)
                line += f", rank {rank} by the ranker"
        print(line)
    capture.release()

    total = len(rows)
    print(f"\n  {read_from_disk}/{total} frames read from the labeller's own "
          f"JPEG; the rest were seeked in the video")
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
