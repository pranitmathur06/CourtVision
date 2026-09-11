"""Whole-game accuracy against the AVERAGE of two labellers' clicks.

Declared before it was run; see the commit that adds this file.

One hand click at 1080p carries ~0.25 ft of its own noise: two labellers who
clicked the same 176 spots on the Toyota Center game disagree by 5.6 px,
0.35 ft median. Against one labeller a perfect registration would still score
~0.25 ft, too close to the 0.30 ft goal to tell a pass from a fail. Averaging
two independent clicks of the same spot divides that noise by sqrt(2), ~0.18 ft.

- Spots: every click in the REFERENCE set (the reoriented labeller, whose
  landmark names agree with their frame's consensus 80% of the time as given)
  that the OTHER labeller also clicked, within 25 px. Its landmark is the
  reference set's; its position the mean of the two clicks. Nothing from any
  registration chooses the spots.
- Stray landmark names: a spot is kept only if it belongs to its frame's label
  consensus (RANSAC over the averaged spots alone, 1 ft) and the frame keeps
  at least MIN_SPOTS such spots.
- Registration: production -- landmark start, register_frame with the game's
  fixed camera, floor signature and lens (outputs/camera/<video>.json), and
  the landmark-free search. Pixels map to the court through to_court.
- Error: court distance from each averaged spot, through the registration, to
  its landmark, under the court symmetry nearest the fit (conventions differ
  by tens of feet and cannot hide a sub-foot error).
- PASS: median over all spots <= 0.30 ft. Reported beside it: p75, share
  within 0.3, per-frame medians, and how many frames were registered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MATCH_PX = 25.0
MIN_SPOTS = 6
TARGET_FT = 0.30
SYMMETRIES = ((False, False), (True, False), (False, True), (True, True))


def symmetric(court, flip_x, flip_y):
    out = np.array(court, np.float64).copy()
    if flip_x:
        out[:, 0] = 50.0 - out[:, 0]
    if flip_y:
        out[:, 1] = 94.0 - out[:, 1]
    return out


def averaged_spots(reference, other, court_of):
    """[(court_xy, mean_px)] for reference clicks the other labeller also made."""
    spots = []
    others = np.array(list(other.values()), np.float64) if other else np.zeros((0, 2))
    for landmark, click in reference.items():
        if int(landmark) not in court_of or not len(others):
            continue
        click = np.asarray(click, np.float64)
        d = np.hypot(*(others - click).T)
        j = int(np.argmin(d))
        if d[j] < MATCH_PX:
            spots.append((court_of[int(landmark)], (click + others[j]) / 2.0))
    return spots


def consensus(spots):
    """Mask of spots a RANSAC fit over the spots alone places within 1 ft."""
    import cv2
    if len(spots) < 4:
        return np.zeros(len(spots), bool)
    px = np.array([s[1] for s in spots], np.float64)
    court = np.array([s[0] for s in spots], np.float64)
    _, mask = cv2.findHomography(px, court, cv2.RANSAC, 1.0, maxIters=5000, confidence=0.999)
    return mask.ravel().astype(bool) if mask is not None else np.zeros(len(spots), bool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="data/labeling/court_toyota_1080_v3")
    parser.add_argument("--other", default="data/labeling/court_toyota_1080_precise/labels.json")
    parser.add_argument("--weights", default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--detector", default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--dump", default="outputs/whole_game_two_labellers.json")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    import courtvision.court_refine as court_refine
    from courtvision.court_camera import FixedCamera
    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    from courtvision.court_register import register_frame, to_court
    from courtvision.device import resolve_device
    from courtvision.provenance import code_provenance

    root = Path(args.reference)
    manifest = json.load(open(root / "manifest.json"))
    reference = json.load(open(root / "labels.json"))
    other = json.load(open(args.other))
    court_of = {p["id"]: (p["x"], p["y"]) for p in reference["points"]}
    entry = json.load(open(f"outputs/camera/{Path(manifest['video']).stem}.json"))
    camera = FixedCamera(entry["centre"], entry["size"], entry.get("floor"), entry.get("k1"))
    model, detector, device = YOLO(args.weights), YOLO(args.detector), resolve_device()
    rows = []
    for item in manifest["frames"]:
        ref = reference["frames"].get(item["file"], {}).get("points", {})
        oth = other["frames"].get(item["file"], {}).get("points", {})
        spots = averaged_spots(ref, oth, court_of)
        keep = consensus(spots)
        if keep.sum() < MIN_SPOTS:
            continue
        spots = [s for s, k in zip(spots, keep) if k]
        frame = cv2.imread(str(root / "images" / item["file"]))
        result = model.predict(frame, device=device, verbose=False)[0]
        start = None
        if result.keypoints is not None and len(result.keypoints):
            xy = result.keypoints.xy[0].cpu().numpy()
            conf = result.keypoints.conf[0].cpu().numpy()
            start, _ = homography_from_keypoints(
                {i: tuple(xy[i]) for i in range(len(xy))
                 if i in KEYPOINTS and conf[i] >= args.conf and (xy[i] > 0).all()})
        found = detector.predict(frame, device=device, verbose=False)[0].boxes
        boxes = (found.xyxy.cpu().numpy()[found.cls.cpu().numpy() == 0]
                 if found is not None and len(found) else None)
        matrix, info = register_frame(frame, start, boxes=boxes, camera=camera)
        row = {"file": item["file"], "spots": len(spots), "refined": bool(info["refined"])}
        if info["refined"]:
            px = np.array([s[1] for s in spots], np.float64)
            truth = np.array([s[0] for s in spots], np.float64)
            estimate = to_court(matrix, info, px, (frame.shape[1], frame.shape[0]))
            options = [np.hypot(*(estimate - symmetric(truth, fx, fy)).T) for fx, fy in SYMMETRIES]
            row["err"] = min(options, key=np.median).tolist()
        rows.append(row)
    json.dump({"meta": {"reference": args.reference, "other": args.other, "camera": entry,
                        **code_provenance(court_refine.__file__)}, "rows": rows},
              open(args.dump, "w"), indent=1)
    scored = [r for r in rows if r["refined"]]
    print(f"{len(rows)} frames with {MIN_SPOTS}+ averaged consensus spots; {len(scored)} registered")
    if not scored:
        return 2
    err = np.concatenate([r["err"] for r in scored])
    per_frame = sorted(round(float(np.median(r["err"])), 2) for r in scored)
    p50 = float(np.median(err))
    print(f"  spots {len(err)}  p50 {p50:.2f} ft  p75 {np.percentile(err, 75):.2f}  "
          f"within 0.3 {np.mean(err <= TARGET_FT):.0%}  -> {'PASS' if p50 <= TARGET_FT else 'FAIL'}")
    print(f"  per-frame medians: {per_frame}")
    return 0 if p50 <= TARGET_FT else 2


if __name__ == "__main__":
    raise SystemExit(main())
