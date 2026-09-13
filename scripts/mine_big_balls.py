"""Label CLOSE-UP balls, which are the whole of the ball's remaining gap.

Measured in Round 89: on five of the thirteen truth frames `ball_clean` scores
the true ball 0.00 even on a 640 px crop centred on it at conf 0.01. Rendered,
those frames are not typical broadcast balls at all -- at 1062 s the ball is
over 100 px across behind a player's hands, at 1638 s it fills the frame as
leather texture.

`build_ball_detector_dataset.py` says "CROPS ARE TAKEN WITHOUT RESIZING, which
is the point... the ball is 15-25 px and its whole difficulty is its size".
That was right for the problem it was solving and it is exactly why the model
is blind here: it has never seen a ball bigger than about 30 px. This is the
same pathology as the rim's remaining misses, where a 500 px ring is as far
outside the rim model's distribution as a 500 px rim is outside any of it.

The tempting fix is crop-and-zoom -- take the 15 px balls and scale them up.
That is what `build_rim_dataset.py` does, and on the rim it did NOT fix the
close-ups, because an upscaled small ball is blurry while a real close-up is
sharp with visible leather seams. The domain gap is in the texture, not only
the size. So these labels are mined SHARP, from frames that really are
close-ups.

The mine is a fact from the same round: the FOUR-CLASS detector finds 1062 s at
imgsz 416 and 2862 s at imgsz 800, sizes BELOW native. Downscaling shrinks a
110 px ball into the 15-25 px band that detector knows. So running it small
over the game finds exactly the balls the ball model cannot see, and their
boxes, scaled back up, are labels at the size that matters.

WHAT STOPS IT LABELLING RUBBISH, declared before it was run:

- MIN_BALL_PX in ORIGINAL pixels. Small balls are already covered by 1,050
  native crops; this run is only for the ones outside that distribution, and
  restricting it also removes most of the population a false positive could
  come from.
- AGREE_SIZES: the box must be found at two or more downscale sizes, within
  AGREE_FRACTION of a ball width. One low-confidence hit at one size is a
  guess; the same box at 320 and 416 is an object.
- MIN_CONF kept low on purpose (the true positives here run 0.16-0.28) with
  the agreement requirement doing the filtering instead of the threshold.
- Same-TAKE exclusion against every evaluation instant, via the rule argued
  for in `propagate_rim_labels.py`: ORB registers AND within SAME_TAKE_S.
- A contact sheet, looked at before anything trains on it. The shot-chart ball
  labeller produced pure garbage that only a rendered sample caught.

A HEAD IS A BALL AT LOW RESOLUTION, and the first run proved it. Every one of
24 sampled labels was a face: Bill Russell in a documentary still, the singer
during the national anthem, a hand holding a trophy. Downscaling is exactly the
operation that removes the features telling a head from a ball, so agreement
across sizes corroborates nothing -- a head is a stable object and every size
finds it.

Two things follow, both applied here:

- The scan must cover the GAME, not the broadcast. The first run started at
  t=0 and its whole sample came from the pre-game montage and the anthem,
  where there is no basketball at all. `--start-s` is now required to be
  inside play.
- A candidate sitting where a HEAD sits is refused. The four-class detector
  gives player boxes in the same pass; a ball candidate in the top
  HEAD_FRACTION of one, near its horizontal middle, is a head. This is the
  rule measured in Round 84 -- it removed 37 of 403 false candidates and 0 of
  7 true ones -- and it is far stronger here, because in a close-up the player
  box is large and well defined rather than 30 px of distant torso.

What this cannot do: it inherits the four-class detector's blind spots, so a
close-up ball that detector misses at every downscale stays unlabelled. It is a
way to reach a NEW part of the size distribution cheaply, not a complete
labelling of it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

#: Inference sizes below the frame's own 1280, which shrink a close-up ball
#: into the band the four-class detector was trained on.
DOWNSCALES = (256, 320, 416, 512, 640)
#: Only balls bigger than this, in original pixels: the native crops already
#: cover everything smaller and that is where false positives would come from.
MIN_BALL_PX = 42.0
#: Two sizes must find it, at centres this close as a share of the box width.
AGREE_SIZES = 2
AGREE_FRACTION = 0.6
MIN_CONF = 0.10
#: A take is contiguous in appearance AND in time; see propagate_rim_labels.
SAME_TAKE_S = 20.0
HOLD_OUT_REACH_S = 45.0
#: A candidate in the top this-much of a player's box, within HEAD_WIDTH of its
#: horizontal middle, is that player's head and not a ball.
HEAD_FRACTION = 0.30
HEAD_WIDTH = 0.55
PLAYER_CONF = 0.25


def box_size(box):
    """The longer side of a box, which is the ball's diameter under occlusion."""
    return max(box[2] - box[0], box[3] - box[1])


def centre_of(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def looks_like_a_head(box, players, fraction=HEAD_FRACTION, width=HEAD_WIDTH):
    """Is this candidate sitting where a head sits on a detected player?

    Downscaling removes exactly the detail that tells a head from a ball, so
    the position has to do the work instead. A head is at the top of its
    player's box and near its middle; a ball a player is holding is not.
    """
    cx, cy = centre_of(box)
    for player in players:
        x0, y0, x1, y1 = player[:4]
        w, h = x1 - x0, y1 - y0
        if w < 4 or h < 4:
            continue
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        if (cy - y0) / h <= fraction and abs((cx - (x0 + x1) / 2) / w) <= width / 2:
            return True
    return False


def agreeing_groups(found, agree_fraction=AGREE_FRACTION, need=AGREE_SIZES):
    """Group boxes seen at different sizes into objects, keeping the agreed ones.

    `found` is a list of (size, box, conf). Returns one (box, conf, sizes) per
    object that at least `need` DISTINCT inference sizes located, the box being
    the highest-confidence member. A single low-confidence hit at one size is a
    guess; the same thing at two sizes is an object.
    """
    groups = []
    for size, box, conf in sorted(found, key=lambda e: -e[2]):
        cx, cy = centre_of(box)
        placed = False
        for group in groups:
            gx, gy = centre_of(group["box"])
            if np.hypot(cx - gx, cy - gy) <= agree_fraction * box_size(group["box"]):
                group["sizes"].add(size)
                placed = True
                break
        if not placed:
            groups.append({"box": box, "conf": conf, "sizes": {size}})
    return [(g["box"], g["conf"], sorted(g["sizes"]))
            for g in groups if len(g["sizes"]) >= need]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--held-out", action="append", default=[])
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument("--step-s", type=float, default=2.0)
    parser.add_argument("--min-ball-px", type=float, default=MIN_BALL_PX)
    parser.add_argument("--conf", type=float, default=MIN_CONF)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from propagate_rim_labels import held_out_times, nearby, shares_a_shot

    model, device = YOLO(args.detector), resolve_device()
    held_out = np.asarray(held_out_times(args.held_out), np.float64)
    capture = cv2.VideoCapture(args.video)
    end = args.end_s
    if end is None:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        end = capture.get(cv2.CAP_PROP_FRAME_COUNT) / max(fps, 1.0)

    def frame_at(t):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        return frame if ok else None

    watch = {}
    found_rows, samples, started = [], [], time.time()
    looked = excluded = heads = 0
    times = np.arange(args.start_s, end, args.step_s)
    for n, t in enumerate(times):
        frame = frame_at(float(t))
        if frame is None:
            continue
        looked += 1
        seen, players = [], []
        native = model.predict(frame, device=device, verbose=False,
                               imgsz=1280, conf=PLAYER_CONF)[0].boxes
        if native is not None and len(native):
            for cls, _, box in zip(native.cls.cpu().numpy(),
                                   native.conf.cpu().numpy(),
                                   native.xyxy.cpu().numpy()):
                if model.names[int(cls)] in ("player", "handler"):
                    players.append([float(v) for v in box])
        for size in DOWNSCALES:
            boxes = model.predict(frame, device=device, verbose=False,
                                  imgsz=size, conf=args.conf)[0].boxes
            if boxes is None or not len(boxes):
                continue
            for cls, conf, box in zip(boxes.cls.cpu().numpy(),
                                      boxes.conf.cpu().numpy(),
                                      boxes.xyxy.cpu().numpy()):
                if model.names[int(cls)] != "ball":
                    continue
                box = [float(v) for v in box]
                if box_size(box) < args.min_ball_px:
                    continue
                if looks_like_a_head(box, players):
                    heads += 1
                    continue
                seen.append((size, box, float(conf)))
        if not seen:
            continue
        agreed = agreeing_groups(seen)
        if not agreed:
            continue

        # Only now pay for the hold-out, which costs a seek and an ORB fit.
        greys = []
        for v in nearby(float(t), held_out, HOLD_OUT_REACH_S):
            if abs(v - float(t)) > SAME_TAKE_S:
                continue
            if v not in watch:
                other = frame_at(v)
                watch[v] = (None if other is None
                            else cv2.cvtColor(other, cv2.COLOR_BGR2GRAY))
            if watch[v] is not None:
                greys.append(watch[v])
        if greys and shares_a_shot(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), greys):
            excluded += 1
            continue

        best = max(agreed, key=lambda e: e[1])
        found_rows.append({"t": round(float(t), 3), "ball": [round(v, 1) for v in best[0]],
                           "conf": round(best[1], 3), "sizes": best[2],
                           "px": round(box_size(best[0]), 1),
                           "others": [[round(v, 1) for v in b] for b, _, _ in agreed
                                      if b is not best[0]]})
        if args.sample_dir and len(samples) < 24:
            shown = frame.copy()
            b = best[0]
            cv2.rectangle(shown, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (0, 0, 255), 3)
            cv2.putText(shown, f"{t:.0f}s {box_size(b):.0f}px {best[1]:.2f} @{best[2]}",
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            samples.append(cv2.resize(shown, (420, 236)))
        if len(found_rows) >= args.limit:
            break
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(times)} frames, {len(found_rows)} big balls, "
                  f"{(time.time() - started) / 60:.1f} min", flush=True)
    capture.release()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "detector": args.detector,
               "downscales": list(DOWNSCALES), "min_ball_px": args.min_ball_px,
               "agree_sizes": AGREE_SIZES, "frames": found_rows}, open(out, "w"))
    if args.sample_dir and samples:
        sample_dir = Path(args.sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        grid = [np.hstack(samples[i:i + 4]) for i in range(0, len(samples) // 4 * 4, 4)]
        if grid:
            cv2.imwrite(str(sample_dir / "big_balls.jpg"), np.vstack(grid))
            print(f"  sample sheet: {sample_dir / 'big_balls.jpg'}")
    sizes = [r["px"] for r in found_rows]
    print(f"{looked} frames looked at, {len(found_rows)} big balls, "
          f"{excluded} dropped for sharing a take with the evaluation, "
          f"{heads} candidates refused for sitting where a head sits")
    if sizes:
        print(f"  ball size px: min {min(sizes):.0f} p50 {np.median(sizes):.0f} "
              f"max {max(sizes):.0f}")
    print(f"  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
