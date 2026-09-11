"""Registration accuracy across a WHOLE GAME, scored against hand-placed landmarks.

Declared before any labels exist; see the commit that adds this file.

Every earlier annotation test scored short clips of a few consecutive frames.
Production registers every frame of a 2.5-hour broadcast: wide and tight
views, both ends, both halves, lighting and graphics that change. This scores
exactly that.

- Frames: data/labeling/<set>/manifest.json -- every 45 s across the whole
  broadcast (offset from every other grid in the project), kept if
  `has_court`. Labelled blind in data/labeling/<set>/label.html: nothing from
  any registration is shown to the labeller.
- Reference: per frame, a least-squares homography through the labelled
  landmarks (>= MIN_POINTS). Its own leave-one-out residual is reported, so the
  reference's noise is known rather than assumed.
- Registration: production. Landmark start, then `register_frame` with the
  game's fixed camera from scripts/estimate_camera.py, whose sampled frames
  are a different grid from these. Also reported without the camera.
- Metric: per labelled point, court distance between the registration and the
  reference. Frames the labeller marked unusable are excluded and counted;
  frames the landmark model cannot start on, or production refuses, assert
  nothing and count against coverage, never as errors.
- Reported: trusted (within TRUST_RADIUS_FT of used paint) p50 / p75 / within
  0.3; the same over ALL points registered; per-frame medians; coverage of
  labelled points and of frames. PASS: trusted p50 <= 0.30 ft AND all-point
  p50 <= 0.30 ft -- a whole game, not only the ground near paint.

Revised after the first run on real labels, and before looking at any
registration against the revised reference (the first run's registration
errors, ~36 ft, were meaningless -- the reference itself was 6 ft
inconsistent):
- Misassigned clicks. Many labelled frames held a few points attached to the
  wrong landmark (landmark names were only shown on hover; the arc apex was
  placed on a sideline). The reference is now fitted with RANSAC over the
  LABELS ALONE (1 ft in court space), then refitted on the inliers; points
  it rejects are not scored, and a frame needs MIN_POINTS inliers and a
  leave-one-out residual under MAX_REFERENCE_FT. Nothing from any
  registration enters that choice. The strict, as-declared numbers are still
  printed.
- The court's symmetry. A court looks the same rotated 180 degrees, so which
  end a labeller calls y = 0 is a convention, not a measurement; the same
  holds for left and right. Each frame's labels are compared under the four
  symmetries of the court, and the one nearest the registration is used.
  The alternatives differ by tens of feet, so this aligns conventions and
  cannot hide a sub-foot error.

`--landmarks intersections` (added after the 1080p run, and justified from the
labels alone): a click on a tangent point or an unmarked spot -- where the
corner three meets the arc, the arc apex, a circle's crown, the centre of the
free-throw line or of the court -- is placed by eye along a smooth curve or a
featureless line, and the labels themselves show it: against the other
clicks of their frame such points sit 0.73 ft off at the median, line
intersections 0.52 ft. This option builds the reference and scores only line
intersections. It was chosen after the per-landmark errors against the
registration had been seen, which is recorded here; the label-only evidence is
what justifies it.

Label repair (added when a second labeller's file arrived, before any
registration was scored against it): that labeller placed sidelines and
corners consistently but swapped the two sides of the lane (x = 17 and 33) --
the free-throw corner nearer the far sideline marked 33, and so on -- so a
frame mixed two conventions and its clicks disagreed by ~20 ft. Each frame's
labels are tried as given and with the lane pairs swapped, the sides swapped,
or both, and the variant whose clicks agree best (most RANSAC inliers, then the
smallest leave-one-out residual) is used. The choice reads only the labels.

`--fuse` (added after the diagnosis that a third of usable frames were
accepted 1.6-3.5 ft off, before fusion was run on any labelled frame): a third
arm registers every frame within court_fusion.WINDOW_S of the labelled one,
carries each accepted registration to it by chained frame-to-frame tracking,
and takes the per-point median (court_fusion.fuse). The window is read from
the game video, as production would; the labelled frame is the one read at the
manifest time. Trust for the fused arm uses the union of the candidates'
support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MIN_POINTS = 8
TARGET_FT = 0.30
MAX_REFERENCE_FT = 0.5
#: Tangent junctions, arc and circle crowns, and unmarked centres.
AMBIGUOUS = {9, 11, 32, 33, 13, 30, 17, 27, 103, 104, 22}
SYMMETRIES = ((False, False), (True, False), (False, True), (True, True))
#: Landmark pairs a labeller can swap by reading the diagram mirrored.
LANE_PAIRS = ((3, 4), (12, 14), (38, 39), (29, 31))
SIDE_PAIRS = ((0, 7), (1, 6), (16, 18), (35, 42), (36, 41), (26, 28), (20, 102), (100, 101))


def repair(points_by_id, court_of):
    """The labels' most self-consistent reading under lane/side swaps.

    `points_by_id` maps landmark id -> click. Returns (points, variant).
    """
    import cv2
    best = None
    for lane in (False, True):
        for side in (False, True):
            swap = {}
            for a, b in (LANE_PAIRS if lane else ()) + (SIDE_PAIRS if side else ()):
                swap[a], swap[b] = b, a
            pts = [(court_of[swap.get(int(k), int(k))], v) for k, v in points_by_id.items()
                   if swap.get(int(k), int(k)) in court_of]
            if len(pts) < MIN_POINTS:
                continue
            px = np.array([q[1] for q in pts], np.float64)
            court = np.array([q[0] for q in pts], np.float64)
            _, mask = cv2.findHomography(px, court, cv2.RANSAC, 1.0, maxIters=5000, confidence=0.999)
            inliers = int(mask.sum()) if mask is not None else 0
            _, loo = reference(pts)
            key = (inliers, -loo)
            if best is None or key > best[0]:
                best = (key, pts, ("lane" if lane else "") + ("+side" if side else ""))
    return (best[1], best[2] or "as given") if best else ([], "none")


def robust_reference(points):
    """RANSAC over the labels alone; (inlier mask, leave-one-out residual) or None."""
    import cv2
    px = np.array([p[1] for p in points], np.float64)
    court = np.array([p[0] for p in points], np.float64)
    if len(points) < MIN_POINTS:
        return None
    _, mask = cv2.findHomography(px, court, cv2.RANSAC, 1.0, maxIters=5000, confidence=0.999)
    if mask is None:
        return None
    keep = mask.ravel().astype(bool)
    if keep.sum() < MIN_POINTS:
        return None
    _, loo = reference([p for p, k in zip(points, keep) if k])
    return keep, loo


def symmetric(court, flip_x, flip_y):
    out = np.array(court, np.float64).copy()
    if flip_x:
        out[:, 0] = 50.0 - out[:, 0]
    if flip_y:
        out[:, 1] = 94.0 - out[:, 1]
    return out


def reference(points):
    """Least-squares image -> court homography and its leave-one-out residual (ft)."""
    import cv2
    px = np.array([p[1] for p in points], np.float64)
    court = np.array([p[0] for p in points], np.float64)
    matrix, _ = cv2.findHomography(px, court, 0)
    loo = []
    for k in range(len(points)):
        keep = np.arange(len(points)) != k
        m, _ = cv2.findHomography(px[keep], court[keep], 0)
        if m is not None:
            q = cv2.perspectiveTransform(px[k:k + 1].reshape(-1, 1, 2), m).reshape(2)
            loo.append(float(np.hypot(*(q - court[k]))))
    return matrix, (float(np.median(loo)) if loo else float("nan"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="court_toyota")
    parser.add_argument("--weights", default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--detector", default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--camera", default=None,
                        help="outputs/camera/<video>.json; default from the manifest's video")
    parser.add_argument("--dump", default=None)
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--landmarks", choices=("all", "intersections"), default="all")
    parser.add_argument("--repair", action="store_true",
                        help="read each frame's labels under lane/side swaps, keep the most consistent")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    import courtvision.court_refine as court_refine
    from courtvision.court_camera import FixedCamera
    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    from courtvision.court_refine import support_distance
    from courtvision.court_fusion import STEP_S, WINDOW_S, chain, fuse
    from courtvision.court_register import register_frame
    from courtvision.court_tracking import pairwise_homography
    from courtvision.device import resolve_device
    from courtvision.provenance import code_provenance

    root = Path("data/labeling") / args.set
    manifest = json.load(open(root / "manifest.json"))
    labels = json.load(open(root / "labels.json"))
    camera_file = Path(args.camera or f"outputs/camera/{Path(manifest['video']).stem}.json")
    entry = json.load(open(camera_file))
    camera = (FixedCamera(entry["centre"], entry["size"], entry.get("floor"))
              if entry.get("centre") else None)
    if camera is None:
        print(f"no camera centre in {camera_file}; the camera arm is skipped")
    court_of = {p["id"]: (p["x"], p["y"]) for p in labels["points"]}
    model, detector, device = YOLO(args.weights), YOLO(args.detector), resolve_device()
    capture = cv2.VideoCapture(manifest["video"]) if args.fuse else None

    def start_and_boxes(image):
        result = model.predict(image, device=device, verbose=False)[0]
        begin = None
        if result.keypoints is not None and len(result.keypoints):
            xy = result.keypoints.xy[0].cpu().numpy()
            conf = result.keypoints.conf[0].cpu().numpy()
            begin, _ = homography_from_keypoints(
                {i: tuple(xy[i]) for i in range(len(xy))
                 if i in KEYPOINTS and conf[i] >= args.conf and (xy[i] > 0).all()})
        found = detector.predict(image, device=device, verbose=False)[0].boxes
        found_boxes = (found.xyxy.cpu().numpy()[found.cls.cpu().numpy() == 0]
                       if found is not None and len(found) else None)
        return begin, found_boxes

    def fused_registration(t, target):
        """Median of the window's accepted registrations, carried to `target`."""
        offsets = np.arange(STEP_S, WINDOW_S + 1e-9, STEP_S)
        candidates, supports = [], []
        begin, bx = start_and_boxes(target)
        if begin is not None:
            m, inf = register_frame(target, begin, boxes=bx, camera=camera)
            if inf["refined"]:
                candidates.append(m)
                supports.append(court_refine._POINTS[inf["support"]])
        for sign in (1.0, -1.0):
            images = [target]
            for dt in offsets:
                capture.set(cv2.CAP_PROP_POS_MSEC, (t + sign * dt) * 1000)
                ok, img = capture.read()
                if not ok:
                    break
                images.append(img)
            hops = [pairwise_homography(images[k], images[k + 1]) for k in range(len(images) - 1)]
            for img, carried in zip(images[1:], chain(hops)):
                begin, bx = start_and_boxes(img)
                if begin is None:
                    continue
                m, inf = register_frame(img, begin, boxes=bx, camera=camera)
                if inf["refined"]:
                    candidates.append(m @ carried)
                    supports.append(court_refine._POINTS[inf["support"]])
        matrix, spread = fuse(candidates, (target.shape[1], target.shape[0]))
        support = np.vstack(supports) if supports else np.zeros((0, 2))
        return matrix, spread, len(candidates), support

    rows, unusable, unlabelled = [], 0, 0
    for item in manifest["frames"]:
        rec = labels["frames"].get(item["file"])
        if rec is None:
            unlabelled += 1
            continue
        if rec.get("skip"):
            unusable += 1
            continue
        kept = {k: v for k, v in rec["points"].items()
                if args.landmarks == "all" or int(k) not in AMBIGUOUS}
        if args.repair:
            pts, variant = repair(kept, court_of)
        else:
            pts, variant = [(court_of[int(k)], v) for k, v in kept.items()], "as given"
        if len(pts) < MIN_POINTS:
            unlabelled += 1
            continue
        ref, ref_noise = reference(pts)
        robust = robust_reference(pts)
        frame = cv2.imread(str(root / "images" / item["file"]))
        row = {"file": item["file"], "t": item["t"], "n_points": len(pts), "label_variant": variant,
               "reference_loo_ft": ref_noise, "arms": {},
               "robust_loo_ft": robust[1] if robust else None,
               "inliers": robust[0].tolist() if robust else None}
        rows.append(row)
        result = model.predict(frame, device=device, verbose=False)[0]
        start = None
        if result.keypoints is not None and len(result.keypoints):
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
        px = np.array([p[1] for p in pts], np.float32).reshape(-1, 1, 2)
        truth = np.array([p[0] for p in pts], np.float64)
        for arm, cam in (("free", None), ("camera", camera)):
            if arm == "camera" and cam is None:
                continue
            matrix, info = register_frame(frame, start, boxes=boxes, camera=cam)
            estimate = cv2.perspectiveTransform(px, matrix).reshape(-1, 2)
            err = np.hypot(*(estimate - truth).T)
            # Robust: label inliers only, under the court symmetry nearest the fit.
            robust_err = None
            if robust and robust[1] <= MAX_REFERENCE_FT:
                keep = robust[0]
                options = [np.hypot(*(estimate[keep] - symmetric(truth[keep], fx, fy)).T)
                           for fx, fy in SYMMETRIES]
                robust_err = min(options, key=np.median).tolist()
            dist = (support_distance(info["support"], estimate) if info["refined"]
                    else np.full(len(truth), np.inf))
            row["arms"][arm] = {"refined": bool(info["refined"]), "err": err.tolist(),
                                "dist": dist.tolist(), "robust_err": robust_err,
                                "robust_dist": (dist[robust[0]].tolist()
                                                if robust_err is not None else None)}
        if args.fuse:
            matrix, spread, n_candidates, support_points = fused_registration(item["t"], frame)
            refined = matrix is not None
            if refined:
                estimate = cv2.perspectiveTransform(px, matrix).reshape(-1, 2)
                err = np.hypot(*(estimate - truth).T)
                from scipy.spatial import cKDTree
                dist = cKDTree(support_points).query(estimate)[0]
            else:
                err = np.full(len(truth), np.inf)
                dist = np.full(len(truth), np.inf)
            robust_err = None
            if refined and robust and robust[1] <= MAX_REFERENCE_FT:
                keep = robust[0]
                options = [np.hypot(*(estimate[keep] - symmetric(truth[keep], fx, fy)).T)
                           for fx, fy in SYMMETRIES]
                robust_err = min(options, key=np.median).tolist()
            row["arms"]["fused"] = {"refined": refined, "err": err.tolist(), "dist": dist.tolist(),
                                    "robust_err": robust_err, "candidates": n_candidates,
                                    "spread_ft": spread,
                                    "robust_dist": (dist[robust[0]].tolist()
                                                    if robust_err is not None else None)}

    if args.dump:
        json.dump({"meta": {"set": args.set, "camera": entry, **code_provenance(court_refine.__file__),
                            "trust_radius_ft": court_refine.TRUST_RADIUS_FT},
                   "rows": rows}, open(args.dump, "w"), indent=1)
    radius = court_refine.TRUST_RADIUS_FT
    frames = len(manifest["frames"])
    print(f"{manifest['video']}: {frames} sampled court frames, {len(rows)} labelled and "
          f"scored, {unusable} marked unusable, {unlabelled} unlabelled or under {MIN_POINTS} points")
    noise = [r["reference_loo_ft"] for r in rows]
    print(f"  reference leave-one-out noise p50 {np.nanmedian(noise):.2f} ft")
    good = [r for r in rows if r["robust_loo_ft"] is not None and r["robust_loo_ft"] <= MAX_REFERENCE_FT]
    print(f"  robust reference: {len(good)}/{len(rows)} frames usable, "
          f"{sum(sum(r['inliers']) for r in good)} of {sum(r['n_points'] for r in good)} "
          f"points kept, leave-one-out p50 "
          f"{np.median([r['robust_loo_ft'] for r in good]) if good else float('nan'):.2f} ft")
    for arm in ("free", "camera", "fused"):
        scored = [r for r in good if arm in r["arms"]]
        refined = [r for r in scored if r["arms"][arm]["refined"] and r["arms"][arm]["robust_err"]]
        if not refined:
            continue
        err = np.concatenate([r["arms"][arm]["robust_err"] for r in refined])
        dist = np.concatenate([r["arms"][arm]["robust_dist"] for r in refined])
        t = dist <= court_refine.TRUST_RADIUS_FT
        total = sum(sum(r["inliers"]) for r in good)
        per_frame = [float(np.median(r["arms"][arm]["robust_err"])) for r in refined]
        tp50 = float(np.median(err[t])) if t.any() else float("inf")
        ap50 = float(np.median(err))
        print(f"  ROBUST {arm:6s} frames refined {len(refined)}/{len(good)}  points trusted "
              f"{t.sum() / max(total, 1):.0%}   trusted p50 {tp50:.2f} p75 "
              f"{np.percentile(err[t], 75) if t.any() else float('nan'):.2f} within 0.3 "
              f"{np.mean(err[t] <= TARGET_FT) if t.any() else 0:.0%}  |  all points p50 {ap50:.2f} "
              f"p75 {np.percentile(err, 75):.2f} within 0.3 {np.mean(err <= TARGET_FT):.0%}  |  "
              f"frames <= 0.3 {np.mean(np.array(per_frame) <= TARGET_FT):.0%}  -> "
              f"{'PASS' if tp50 <= TARGET_FT and ap50 <= TARGET_FT else 'FAIL'}")
    print("  STRICT (as declared):")
    verdict = 1
    for arm in ("free", "camera"):
        scored = [r for r in rows if arm in r["arms"]]
        if not scored:
            continue
        total = sum(r["n_points"] for r in rows)
        refined = [r for r in scored if r["arms"][arm]["refined"]]
        err = np.concatenate([r["arms"][arm]["err"] for r in refined]) if refined else np.zeros(0)
        dist = np.concatenate([r["arms"][arm]["dist"] for r in refined]) if refined else np.zeros(0)
        t = dist <= radius
        tp50 = float(np.median(err[t])) if t.any() else float("inf")
        ap50 = float(np.median(err)) if len(err) else float("inf")
        per_frame = [float(np.median(r["arms"][arm]["err"])) for r in refined]
        passed = tp50 <= TARGET_FT and ap50 <= TARGET_FT
        print(f"  {arm:6s} frames refined {len(refined)}/{len(rows)}  points trusted "
              f"{t.sum() / max(total, 1):.0%} of labelled")
        print(f"         trusted p50 {tp50:.2f}  p75 {np.percentile(err[t], 75) if t.any() else float('nan'):.2f}"
              f"  within 0.3 {np.mean(err[t] <= TARGET_FT) if t.any() else 0:.0%}   |   all registered points "
              f"p50 {ap50:.2f}  p75 {np.percentile(err, 75) if len(err) else float('nan'):.2f}"
              f"  within 0.3 {np.mean(err <= TARGET_FT) if len(err) else 0:.0%}")
        print(f"         per-frame medians: p50 {np.median(per_frame) if per_frame else float('nan'):.2f}"
              f"  frames <= 0.3 {np.mean(np.array(per_frame) <= TARGET_FT) if per_frame else 0:.0%}"
              f"   -> {'PASS' if passed else 'FAIL'}")
        if arm == "camera" or camera is None:
            verdict = 0 if passed else 2
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
