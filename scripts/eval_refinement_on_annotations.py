"""Refinement accuracy on UNSEEN ARENAS, scored against human annotations.

Declared before the refinement was run on any of these images -- see the commit
that adds this file, and docs Round 51.

- Images: the keypoint dataset's by-game TEST split, four games the landmark
  model never trained on. Inferred from the 2025 playoff schedule (not stated
  in the data), three are at arenas in no split at all: TD Garden (BOS-NYK g1),
  Kaseya Center (CLE-MIA g3), Fiserv Forum (IND-MIL g4). The fourth, LAL-MIN g3,
  is at Target Center, which hosts training games; it is reported apart as a
  seen arena.
- Nothing about the refinement has ever been tuned on these images.
- Settings: paint polarity "bright", MIN_PEAK_RATIO 3.0 -- chosen on the OKC
  calibration game by the fallback rule: no candidate reached a held-out
  median of 0.30 ft there, so the threshold minimising OKC's conservative
  median. Other thresholds and the "all" polarity are printed for context and
  select nothing.
- Metric: per image, the median distance in court feet between a registration
  and the one fitted to the human annotations, over the annotated visible
  landmarks. A refused frame falls back to its landmark registration, as the
  pipeline would; "accepted only" and "with fallback" are both reported, beside
  landmark-only on the same images.
- The reference is imperfect: annotators agree with the recovered schema to
  about 0.35 ft per landmark (leave-one-out), so errors near that level cannot
  be resolved by it.

Unlike the held-out-line protocol, this scores the PRODUCTION fit -- every line
used -- against something no part of the refinement produced.

Added after the declared run, reporting only (settings and metric unchanged):
a per-arena breakdown, which the declared output lacked and whose absence let
a pooled 0.23 ft hide that TD Garden carried it; and commit provenance in the
dump.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path("data/labeled/court_keypoints_by_game/test")
PRIMARY_POLARITY = "bright"
PRIMARY_THRESHOLD = 3.0
THRESHOLDS = (0.0, 2.0, 3.0, 5.0, 7.0, 10.0)
UNSEEN_GAMES = ("boston-celtics-new-york-knicks-game-1",
                "cleveland-cavaliers-miami-heat-game-3",
                "indiana-pacers-milwaukee-bucks-game-4")
TARGET_FT = 0.30


def _summary(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return "n=  0"
    return (f"n={len(v):3d}  p50 {np.median(v):5.2f} ft  p90 {np.percentile(v, 90):5.2f} ft"
            f"  within {TARGET_FT} {np.mean(v <= TARGET_FT):4.0%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--detector", default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--dump", default="outputs/refine_on_annotations.json")
    args = parser.parse_args()

    import cv2
    from PIL import Image
    from ultralytics import YOLO

    import courtvision.court_refine as court_refine
    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    from courtvision.device import resolve_device
    sys.path.insert(0, "scripts")
    from eval_court_keypoints import _truth
    from split_court_keypoints import game_of

    court_refine.MIN_PEAK_RATIO = 0.0     # refine once; thresholds applied after, exactly
    model, detector, device = YOLO(args.weights), YOLO(args.detector), resolve_device()
    rows = []
    for path in sorted((DATA / "images").glob("*.jpg")):
        label = DATA / "labels" / (path.stem + ".txt")
        if not label.exists():
            continue
        width, height = Image.open(path).size
        truth = _truth(label, width, height)
        if len(truth) < 8:
            continue
        reference, _ = homography_from_keypoints(truth)
        if reference is None:
            continue
        frame = cv2.imread(str(path))
        game = game_of(path.name)
        row = {"image": path.name, "game": game, "unseen": game in UNSEEN_GAMES,
               "size": [width, height], "registered": False}
        rows.append(row)
        result = model.predict(frame, device=device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        xy = result.keypoints.xy[0].cpu().numpy()
        conf = result.keypoints.conf[0].cpu().numpy()
        start, _ = homography_from_keypoints(
            {i: tuple(xy[i]) for i in range(len(xy))
             if i in KEYPOINTS and conf[i] >= args.conf and (xy[i] > 0).all()})
        if start is None:
            continue
        found = detector.predict(frame, device=device, verbose=False)[0].boxes
        boxes = (found.xyxy.cpu().numpy()[found.cls.cpu().numpy() == 0]
                 if found is not None and len(found) else None)
        probe = np.array([[p] for p in truth.values()], dtype=np.float32)

        def error(matrix):
            a = cv2.perspectiveTransform(probe, matrix).reshape(-1, 2)
            b = cv2.perspectiveTransform(probe, reference).reshape(-1, 2)
            return float(np.median(np.hypot(*(a - b).T)))

        row.update(registered=True, landmark_err=error(start))
        for polarity in ("bright", "all"):
            response = court_refine.paint_response(frame, polarity)
            refined, info = court_refine.refine(
                frame, start, boxes=boxes,
                prepared=(response, court_refine._structure(response)))
            row[polarity] = {"refined": bool(info["refined"]), "ratio": info["peak_ratio"],
                             "err": error(refined) if info["refined"] else None,
                             "reason": info["reason"]}

    from courtvision.provenance import code_provenance
    json.dump({"meta": {"settings": [PRIMARY_POLARITY, PRIMARY_THRESHOLD],
                        "weights": args.weights, "conf": args.conf,
                        **code_provenance(court_refine.__file__)},
               "rows": rows}, open(args.dump, "w"), indent=1)
    sizes = sorted({tuple(r["size"]) for r in rows})
    print(f"{len(rows)} test images with a usable reference; sizes {sizes[:4]}")

    def block(name, group, polarity, threshold):
        reg = [r for r in group if r["registered"]]
        acc = [r for r in reg if r[polarity]["refined"] and (r[polarity]["ratio"] or 0) >= threshold]
        ids = {r["image"] for r in acc}
        fallback = [r[polarity]["err"] if r["image"] in ids else r["landmark_err"] for r in reg]
        print(f"  {name}: {len(group)} images, {len(reg)} registered, {len(acc)} refined "
              f"({len(acc)/max(len(reg),1):.0%})")
        print(f"      landmark only       {_summary([r['landmark_err'] for r in reg])}")
        print(f"      refined, accepted   {_summary([r[polarity]['err'] for r in acc])}")
        print(f"      with fallback       {_summary(fallback)}")

    for name, group in (("UNSEEN ARENAS", [r for r in rows if r["unseen"]]),
                        ("seen arena", [r for r in rows if not r["unseen"]])):
        print(f"\n== PRIMARY ({PRIMARY_POLARITY}, threshold {PRIMARY_THRESHOLD}) -- {name}")
        block(name, group, PRIMARY_POLARITY, PRIMARY_THRESHOLD)
    print("\n== per arena, PRIMARY settings")
    for game in sorted({r["game"] for r in rows}):
        group = [r for r in rows if r["game"] == game]
        block(f"{game} ({'unseen' if group[0]['unseen'] else 'seen'})", group,
              PRIMARY_POLARITY, PRIMARY_THRESHOLD)
    print("\n-- context only, selects nothing: unseen arenas, accepted-only p50 (acceptance)")
    unseen = [r for r in rows if r["unseen"] and r["registered"]]
    for polarity in ("bright", "all"):
        cells = []
        for threshold in THRESHOLDS:
            acc = [r for r in unseen if r[polarity]["refined"] and (r[polarity]["ratio"] or 0) >= threshold]
            e = [r[polarity]["err"] for r in acc]
            cells.append(f"{threshold:>4}: {np.median(e) if e else float('nan'):4.2f} ({len(acc)/max(len(unseen),1):3.0%})")
        print(f"    {polarity:6s} " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
