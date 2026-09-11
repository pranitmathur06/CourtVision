"""Trusted registration accuracy, per arena, scored against human annotations.

Declared before it was run; see the commit that adds this file.

What is claimed: every court point a registration ASSERTS -- one within
court_refine.TRUST_RADIUS_FT of the paint its fit rests on -- is within the
0.30 ft goal at the median, on EVERY arena, not pooled. Points beyond the
radius are flagged as extrapolated and are reported separately, not hidden.

- Production path: court_register.register_frame (bright and all evidence,
  sharper accepted fit), MIN_PEAK_RATIO and TRUST_RADIUS_FT as in the code at
  the commit that runs this. The radius was selected on the OKC calibration
  game by scripts/select_trust_radius.py; nothing here selects anything.
- Images: the keypoint dataset's by-game test split. TD Garden, Kaseya Center
  and Fiserv Forum are at arenas in no training split; Target Center hosts
  training games and is reported as seen. These images have informed design
  since Round 51, so this is a re-run on known images, not a first look.
- Per annotated floor point (hoop keypoints excluded): error is the court
  distance between the registration and the homography fitted to the human
  annotations. A refused frame asserts nothing: its points are untrusted.
- Reported per arena: trusted error p50 / p75 / within 0.3 over points; the
  share of annotated points and of detected players' feet that are trusted
  (coverage); and untrusted points' error, beside landmark-only.
- PASS for an arena: trusted p50 <= 0.30 ft. The reference itself carries
  ~0.13 ft of noise, which is not subtracted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path("data/labeled/court_keypoints_by_game/test")
ARENAS = {"boston-celtics-new-york-knicks-game-1": ("TD Garden", True),
          "cleveland-cavaliers-miami-heat-game-3": ("Kaseya Center", True),
          "indiana-pacers-milwaukee-bucks-game-4": ("Fiserv Forum", True),
          "los-angeles-lakers-minnesota-timberwolve-game-3": ("Target Center", False)}
HOOPS = (8, 34)
TARGET_FT = 0.30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--detector", default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--dump", default="outputs/trusted_on_annotations.json")
    args = parser.parse_args()

    import cv2
    from PIL import Image
    from ultralytics import YOLO

    import courtvision.court_refine as court_refine
    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    from courtvision.court_refine import support_distance
    from courtvision.court_register import register_frame
    from courtvision.device import resolve_device
    from courtvision.provenance import code_provenance
    sys.path.insert(0, "scripts")
    from eval_court_keypoints import _truth
    from split_court_keypoints import game_of

    radius = court_refine.TRUST_RADIUS_FT
    model, detector, device = YOLO(args.weights), YOLO(args.detector), resolve_device()
    rows = []
    for path in sorted((DATA / "images").glob("*.jpg")):
        label = DATA / "labels" / (path.stem + ".txt")
        if not label.exists():
            continue
        width, height = Image.open(path).size
        floor = {k: v for k, v in _truth(label, width, height).items() if k not in HOOPS}
        if len(floor) < 8:
            continue
        reference, _ = homography_from_keypoints(floor)
        if reference is None:
            continue
        frame = cv2.imread(str(path))
        row = {"image": path.name, "game": game_of(path.name), "registered": False}
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
        matrix, info = register_frame(frame, start, boxes=boxes)
        probe = np.array([[p] for p in floor.values()], dtype=np.float32)
        truth = cv2.perspectiveTransform(probe, reference).reshape(-1, 2)
        err = np.hypot(*(cv2.perspectiveTransform(probe, matrix).reshape(-1, 2) - truth).T)
        lm = np.hypot(*(cv2.perspectiveTransform(probe, start).reshape(-1, 2) - truth).T)
        # Trust is decided as production decides it: at the court position
        # the registration itself gives the point, not the annotators' one.
        estimate = cv2.perspectiveTransform(probe, matrix).reshape(-1, 2)
        dist = (support_distance(info["support"], estimate) if info["refined"]
                else np.full(len(truth), np.inf))
        feet_dist = []
        if boxes is not None and len(boxes):
            feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]].astype(np.float32)
            court = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), matrix).reshape(-1, 2)
            inside = (court[:, 0] > -3) & (court[:, 0] < 53) & (court[:, 1] > -3) & (court[:, 1] < 97)
            feet_dist = (support_distance(info["support"], court[inside]) if info["refined"]
                         else np.full(int(inside.sum()), np.inf)).tolist()
        row.update(registered=True, refined=bool(info["refined"]),
                   polarity=info.get("polarity"), err=err.tolist(), landmark_err=lm.tolist(),
                   dist=dist.tolist(), feet_dist=feet_dist)

    json.dump({"meta": {"trust_radius_ft": radius,
                        "min_peak_ratio": court_refine.MIN_PEAK_RATIO,
                        **code_provenance(court_refine.__file__)}, "rows": rows},
              open(args.dump, "w"), indent=1)
    print(f"TRUST_RADIUS_FT {radius}   MIN_PEAK_RATIO {court_refine.MIN_PEAK_RATIO}")
    print("  arena            images  refined  points trusted  feet trusted   "
          "trusted p50   p75    within 0.3   untrusted p50   landmark p50   verdict")
    verdicts = []
    for game, (name, unseen) in ARENAS.items():
        group = [r for r in rows if r["game"] == game and r["registered"]]
        if not group:
            continue
        err = np.concatenate([r["err"] for r in group])
        dist = np.concatenate([r["dist"] for r in group])
        lm = np.concatenate([r["landmark_err"] for r in group])
        feet = np.concatenate([r["feet_dist"] for r in group]) if any(r["feet_dist"] for r in group) else np.zeros(0)
        t = dist <= radius
        p50 = float(np.median(err[t])) if t.any() else float("inf")
        verdict = "PASS" if p50 <= TARGET_FT else "FAIL"
        if unseen:
            verdicts.append(verdict)
        print(f"  {name:15s}{'' if unseen else ' (seen)':7s} {len(group):3d}   "
              f"{np.mean([r['refined'] for r in group]):5.0%}    {t.mean():5.0%}          "
              f"{np.mean(feet <= radius) if len(feet) else float('nan'):5.0%}          "
              f"{p50:5.2f}    {np.percentile(err[t], 75) if t.any() else float('nan'):5.2f}   "
              f"{np.mean(err[t] <= TARGET_FT) if t.any() else 0:5.0%}         "
              f"{np.median(err[~t]) if (~t).any() else float('nan'):5.2f}          "
              f"{np.median(lm):5.2f}        {verdict}")
    print(f"  unseen arenas passing: {verdicts.count('PASS')} of {len(verdicts)}")
    return 0 if verdicts and all(v == "PASS" for v in verdicts) else 2


if __name__ == "__main__":
    raise SystemExit(main())
