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

`--camera loo` (added before this script was first run, with the fixed-camera
model): each image is registered as a pan/tilt/roll/zoom of its game's camera,
whose centre is solved from the FREE production registrations of the game's
OTHER test images -- leave-one-out, no annotations. An image whose game yields
no centre keeps the free registration and is counted as such. A camera fit
that is refused asserts nothing, exactly as a refused free fit.

Revised after an independent review, before any re-run:
- Leave-one-CLIP-out. The images come in short clips of consecutive frames;
  Kaseya's seven are one 5-second clip, so "the other test images" were six
  near-duplicates of the frame being scored -- nothing like production, where
  scripts/estimate_camera.py samples a whole game. A centre now comes only from
  images of OTHER clips of the same game; a game with one clip gets no camera
  and keeps its free registration, counted as such.
- `--split valid` scores the valid split's two arenas that appear in no
  training game: Crypto.com Arena (LAL-MIN g1) and Toyota Center (GSW-HOU g1).
  The landmark model used the valid split for model selection, so its start
  there is slightly optimistic; the refinement and camera never saw it.
- Error is also reported per clip, because points within an image and frames
  within a clip are not independent: the clip is the unit of evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("data/labeled/court_keypoints_by_game")
ARENAS = {"test": {"boston-celtics-new-york-knicks-game-1": ("TD Garden", True),
                   "cleveland-cavaliers-miami-heat-game-3": ("Kaseya Center", True),
                   "indiana-pacers-milwaukee-bucks-game-4": ("Fiserv Forum", True),
                   "los-angeles-lakers-minnesota-timberwolve-game-3": ("Target Center", False)},
          "valid": {"los-angeles-lakers-minnesota-timberwolve-game-1": ("Crypto.com Arena", True),
                    "golden-state-warriors-houston-rockets-game-1": ("Toyota Center", True),
                    "boston-celtics-new-york-knicks-game-4": ("Madison Square Garden", False)}}


def clip_of(name: str) -> str:
    """The clip an image was cut from: everything before its frame number."""
    return name.split("_mp4-")[0]


def arena_of(game: str, split: str):
    return next(((name, unseen) for prefix, (name, unseen) in ARENAS[split].items()
                 if game.startswith(prefix)), (game, False))
HOOPS = (8, 34)
TARGET_FT = 0.30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--detector", default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--dump", default="outputs/trusted_on_annotations.json")
    parser.add_argument("--camera", choices=("none", "loo"), default="none")
    parser.add_argument("--split", choices=("test", "valid"), default="test")
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

    from courtvision.court_camera import estimate_centre

    def _score(row, frame, start, boxes, floor, reference, matrix, info):
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

    radius = court_refine.TRUST_RADIUS_FT
    model, detector, device = YOLO(args.weights), YOLO(args.detector), resolve_device()
    rows, pending = [], []
    data = ROOT / args.split
    for path in sorted((data / "images").glob("*.jpg")):
        label = data / "labels" / (path.stem + ".txt")
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
        row = {"image": path.name, "game": arena_of(game_of(path.name), args.split)[0],
               "clip": clip_of(path.name), "n_points": len(floor), "registered": False}
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
        if args.camera == "loo":
            # Scored after every free fit of the game exists; see below.
            pending.append((row, frame, start, boxes, floor, reference, matrix, info))
            continue
        _score(row, frame, start, boxes, floor, reference, matrix, info)

    if args.camera == "loo":
        free = {}
        for row, frame, start, boxes, floor, reference, matrix, info in pending:
            if info["refined"]:
                free.setdefault(row["game"], []).append(
                    (row["clip"], matrix, court_refine._POINTS[info["support"]]))
        for row, frame, start, boxes, floor, reference, matrix, info in pending:
            others = [(m, pts) for clip, m, pts in free.get(row["game"], [])
                      if clip != row["clip"]]
            camera, report = estimate_centre(others, (frame.shape[1], frame.shape[0]))
            row["camera"] = report
            if camera is not None:
                matrix, info = register_frame(frame, start, boxes=boxes, camera=camera)
            _score(row, frame, start, boxes, floor, reference, matrix, info)

    json.dump({"meta": {"trust_radius_ft": radius, "camera": args.camera,
                        "min_peak_ratio": court_refine.MIN_PEAK_RATIO,
                        **code_provenance(court_refine.__file__)}, "rows": rows},
              open(args.dump, "w"), indent=1)
    print(f"{args.split} split   camera {args.camera}   TRUST_RADIUS_FT {radius}   "
          f"MIN_PEAK_RATIO {court_refine.MIN_PEAK_RATIO}")
    print("  arena                    images clips  camera  refined  points trusted  feet trusted  "
          "trusted p50   p75   within 0.3   clips<=0.3   untrusted p50  landmark p50  verdict")
    verdicts = []
    for name, unseen in ARENAS[args.split].values():
        everything = [r for r in rows if r["game"] == name]
        group = [r for r in everything if r["registered"]]
        if not group:
            continue
        err = np.concatenate([r["err"] for r in group])
        dist = np.concatenate([r["dist"] for r in group])
        lm = np.concatenate([r["landmark_err"] for r in group])
        feet = (np.concatenate([r["feet_dist"] for r in group])
                if any(r["feet_dist"] for r in group) else np.zeros(0))
        t = dist <= radius
        # Coverage over EVERY annotated point, including images the landmark
        # model could not start on -- they assert nothing.
        total_points = sum(r["n_points"] for r in everything)
        p50 = float(np.median(err[t])) if t.any() else float("inf")
        clips = sorted({r["clip"] for r in everything})
        clip_p50 = []
        for clip in clips:
            members = [r for r in group if r["clip"] == clip]
            if members:
                e = np.concatenate([r["err"] for r in members])
                d = np.concatenate([r["dist"] for r in members])
                if (d <= radius).any():
                    clip_p50.append(float(np.median(e[d <= radius])))
        cameras = sum(1 for r in everything if (r.get("camera") or {}).get("centre"))
        verdict = "PASS" if p50 <= TARGET_FT else "FAIL"
        if unseen:
            verdicts.append(verdict)
        print(f"  {name:22s}{'' if unseen else ' (seen)':7s}{len(everything):4d}  {len(clips):4d}  "
              f"{cameras:5d}   {np.mean([r['refined'] for r in group]):5.0%}   "
              f"{t.sum() / max(total_points, 1):6.0%}        "
              f"{np.mean(feet <= radius) if len(feet) else float('nan'):6.0%}       "
              f"{p50:5.2f}     {np.percentile(err[t], 75) if t.any() else float('nan'):5.2f}  "
              f"{np.mean(err[t] <= TARGET_FT) if t.any() else 0:5.0%}      "
              f"{sum(c <= TARGET_FT for c in clip_p50):3d}/{len(clip_p50):<3d}      "
              f"{np.median(err[~t]) if (~t).any() else float('nan'):5.2f}         "
              f"{np.median(lm):5.2f}      {verdict}")
    print(f"  unseen arenas passing: {verdicts.count('PASS')} of {len(verdicts)}")
    return 0 if verdicts and all(v == "PASS" for v in verdicts) else 2


if __name__ == "__main__":
    raise SystemExit(main())
