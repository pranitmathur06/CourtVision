"""Where the ball actually fails: recall, rank, or appearance. Measured, not guessed.

Nine approaches to the ball have been built and rejected. This asks the prior
question -- which HALF of the problem is left -- on the unbiased hand-located
truth set, and it answers two things that the ledger had never separated.

RECALL is nearly solved by slicing the frame. Whole-frame inference at 2560 px
puts a candidate within tolerance on 6 of 10 truth balls; slicing into 320 px
tiles at 30% overlap and running each at 640 px gets 7, and pooling the five
configurations below covers 9. Upscaling the whole frame is NOT the same
operation as slicing it, which is why "large inference" was rejected in Round
71 while tiling helps: a 15 px ball in a 320 px tile is a 30 px ball in the
network's input, with its own detection context, rather than a 15 px ball in a
crowd of 1,280.

    config                  within tolerance   the ball's rank by confidence
    whole 2560                   6/10          9, 15, 44, 52, 54, 62
    whole 3840                   4/10          6, 17, 40, 60
    tile 320 / 0.3 @ 640         7/10          19, 36, 58, 59, 66, 75, 97
    tile 256 / 0.4 @ 640         5/10          16, 59, 95, 191, 213
    tile 192 / 0.4 @ 640         6/10          23, 49, 51, 112, 124, 170

RANK is the whole of the remaining problem, and it is worse than it looked. In
EVERY configuration the true ball is the top-confidence candidate ZERO times
out of ten. More proposals make recall better and ranking harder: tiling at 192
px proposes 189 candidates a frame to find a ball at rank 170.

And COLOUR, which is the obvious thing to rank with and was missing from the
ledger, does not work. A basketball is orange, so the share of a candidate box
that is orange in HSV should say ball. Measured against the candidates the
detector already ranks ABOVE the true ball:

    true balls   n=7    orange share  min 0.000   p50 0.075   max 0.650
    false above  n=403  orange share              p50 0.000   p90 0.511

    threshold   true kept   false kept
      0.05        4/7        154/403
      0.20        2/7        100/403
      0.40        2/7         60/403

The false candidates are MORE orange than the balls. Three of seven true balls
contain no orange pixel at all. A basketball at 15 px, motion-blurred over a
bright maple floor, is a grey smudge, while the floor itself, players' skin and
one team's gold kit all sit in the orange band. There is no threshold here that
is not worse than useless.

Taken together these say something the ledger had been circling: AT THIS SIZE A
SINGLE PATCH CARRIES ALMOST NO INFORMATION. That is why the learned patch
ranker moved the true ball's mean rank from 1.0 to 2.0 rather than upward, and
it means any future selector has to use something other than the picture inside
the box -- time, or the geometry of the flight, or the players around it.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

#: (name, tile, overlap, inference size). A tile of None is the whole frame.
CONFIGS = (("whole 2560", None, 0.0, 2560),
           ("whole 3840", None, 0.0, 3840),
           ("tile 320/0.3 @640", 320, 0.3, 640),
           ("tile 256/0.4 @640", 256, 0.4, 640),
           ("tile 192/0.4 @640", 192, 0.4, 640))
CONF = 0.03
#: OpenCV hue is 0-179; a basketball sits near 5-20 when it is not blurred.
ORANGE_HUE = (3, 22)
ORANGE_MIN_SATURATION = 110
ORANGE_MIN_VALUE = 60


def tile_origins(width, height, tile, overlap):
    """Top-left corners covering the frame, with the right and bottom edges hit."""
    step = max(int(tile * (1 - overlap)), 1)
    xs = sorted(set(list(range(0, max(width - tile, 0) + 1, step)) + [max(width - tile, 0)]))
    ys = sorted(set(list(range(0, max(height - tile, 0) + 1, step)) + [max(height - tile, 0)]))
    return [(x, y) for y in ys for x in xs]


def orange_share(patch_hsv, hue=ORANGE_HUE, min_s=ORANGE_MIN_SATURATION,
                 min_v=ORANGE_MIN_VALUE):
    """Share of pixels that are basketball-orange, or None for an empty patch."""
    if patch_hsv.size == 0:
        return None
    h, s, v = (patch_hsv[..., 0].astype(float), patch_hsv[..., 1].astype(float),
               patch_hsv[..., 2].astype(float))
    return float(((h >= hue[0]) & (h <= hue[1]) & (s >= min_s) & (v >= min_v)).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--truth", required=True, help="ball_truth_handlocated.json")
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--conf", type=float, default=CONF)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device

    truth = json.load(open(args.truth))
    tolerance = float(truth.get("tolerance_px", 28.0))
    rows = truth["frames"]
    model, device = YOLO(args.detector), resolve_device()
    capture = cv2.VideoCapture(args.video)

    def balls(frame, tile, overlap, size):
        windows = ([(0, 0)] if tile is None
                   else tile_origins(frame.shape[1], frame.shape[0], tile, overlap))
        out = []
        for x0, y0 in windows:
            piece = (frame if tile is None
                     else frame[y0:y0 + tile, x0:x0 + tile])
            found = model.predict(piece, device=device, verbose=False,
                                  imgsz=size, conf=args.conf)[0].boxes
            if found is None or not len(found):
                continue
            for cls, conf, box in zip(found.cls.cpu().numpy(),
                                      found.conf.cpu().numpy(),
                                      found.xyxy.cpu().numpy()):
                if model.names[int(cls)] != "ball":
                    continue
                out.append([float(box[0]) + x0, float(box[1]) + y0,
                            float(box[2]) + x0, float(box[3]) + y0, float(conf)])
        return out

    report = {name: {"hit": 0, "top1": 0, "ranks": [], "counts": []}
              for name, *_ in CONFIGS}
    true_orange, false_orange = [], []
    pooled_hit = 0
    for row in rows:
        capture.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        tx, ty = row["ball"]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        anywhere = False
        for name, tile, overlap, size in CONFIGS:
            candidates = balls(frame, tile, overlap, size)
            report[name]["counts"].append(len(candidates))
            if not candidates:
                continue
            gaps = [np.hypot((c[0] + c[2]) / 2 - tx, (c[1] + c[3]) / 2 - ty)
                    for c in candidates]
            nearest = int(np.argmin(gaps))
            if gaps[nearest] > tolerance:
                continue
            anywhere = True
            report[name]["hit"] += 1
            order = sorted(range(len(candidates)), key=lambda i: -candidates[i][4])
            rank = order.index(nearest) + 1
            report[name]["ranks"].append(rank)
            report[name]["top1"] += int(rank == 1)
            if name.startswith("tile 320"):
                for position, index in enumerate(order):
                    box = candidates[index]
                    x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
                    x2, y2 = min(int(box[2]), frame.shape[1]), min(int(box[3]), frame.shape[0])
                    share = orange_share(hsv[y1:y2, x1:x2])
                    if share is None:
                        continue
                    if index == nearest:
                        true_orange.append(share)
                    elif position < rank - 1:
                        false_orange.append(share)
        pooled_hit += int(anywhere)
    capture.release()

    print(f"{len(rows)} hand-located balls, tolerance {tolerance:.0f} px\n")
    print(f"{'config':22s} {'within tol':>11s} {'top-1':>6s} {'candidates':>11s}  ranks")
    for name, *_ in CONFIGS:
        r = report[name]
        print(f"{name:22s} {r['hit']:>6d}/{len(rows):<4d} {r['top1']:>6d} "
              f"{np.mean(r['counts']):>11.0f}  {sorted(r['ranks'])}")
    print(f"\npooled over every configuration: {pooled_hit}/{len(rows)} have a "
          f"candidate within tolerance SOMEWHERE")
    if true_orange and false_orange:
        t, f = np.array(true_orange), np.array(false_orange)
        print(f"\ntrue balls   n={len(t):<4d} orange share  min {t.min():.3f} "
              f"p50 {np.median(t):.3f} max {t.max():.3f}")
        print(f"false above  n={len(f):<4d} orange share  "
              f"p50 {np.median(f):.3f} p90 {np.percentile(f, 90):.3f}")
        for threshold in (0.05, 0.10, 0.20, 0.30, 0.40):
            print(f"  threshold {threshold:.2f}: keeps {int((t >= threshold).sum())}/"
                  f"{len(t)} true, {int((f >= threshold).sum())}/{len(f)} false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
