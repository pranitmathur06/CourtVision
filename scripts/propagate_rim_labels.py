"""Carry 49 hand-located rims across the game by ORB, into thousands of labels.

Four rim training runs have now changed the measured accuracy by nothing at
all -- 23 hand labels, then 41, then 27 with 68 hard negatives, against a
scale-crop dataset of 8,256. The reason is arithmetic: 8,256 crops the
detector cut from frames where it ALREADY finds a rim cannot teach it a
viewpoint it has never seen, and 49 examples cannot outvote them.

And crop-and-zoom is structurally incapable of fixing that. It crops around
the detector's own confident boxes, so every crop it makes comes from a
camera the detector already handles. The failures are the other cameras:
measured at frame 887.5 s, an overhead-behind-backboard shot with a ring
filling a quarter of the picture, the detector proposes NOTHING at conf 0.02
at 1280, 2560 and 3200 px, and the scale model nothing over a 3x3 tiling.

The multiplier is that a rim is BOLTED TO THE FLOOR and these cameras are
bolted to the building. The overhead camera shoots the same backboard from
the same mount all night. So a rim located by hand in one frame can be
carried by ORB to every other frame that camera shot, and one hand label
becomes as many labels as that camera has frames.

This is `court_tracking.pairwise_homography` used for labelling instead of
for tracking, and it is the same trick `track_camera.py` uses to pose grid
frames from a distant anchor.

WHAT STOPS IT LABELLING RUBBISH, all declared here before it was run:

- MIN_INLIERS and MIN_INLIER_RATIO on the ORB fit. A frame from a DIFFERENT
  camera has no static structure in common with the anchor, so it fails here
  rather than being propagated onto. This is the main protection.
- MAX_RESIDUAL_PX on the inliers, so a fit that is merely arithmetically
  possible does not pass as a fit.
- The carried box must stay a plausible rim: its centre inside the frame by a
  margin, its width within [MIN_SCALE, MAX_SCALE] of the anchor's, and its
  four carried corners still convex and near-square. A homography that has
  gone wrong makes a bow-tie or a sliver, and those are cheap to catch.
- SAME-TAKE EXCLUSION from every evaluation frame. The first version of this
  held frames out by clock distance, and it produced ZERO labels: the
  evaluation grid samples every 25 s, so a 30 s radius blankets all 9,355 s
  of the game. Widening the radius to make room would have been a threshold
  moved after seeing the result, which is the error this project keeps
  repeating, so the rule was replaced rather than relaxed.

  What the hold-out is actually for is that the model must not be trained on
  a picture of an evaluation frame. Two frames 30 s apart share no camera, no
  possession and no player position; two frames in the same continuous SHOT
  are the same picture. So the criterion is the same shot, and ORB states it
  exactly: a candidate is excluded when a nearby evaluation frame registers
  to it. That is tighter than 30 s where it matters and admits the rest of
  the game, which is the point.
- A contact sheet is rendered and looked at BEFORE anything trains on this.
  The shot-chart ball labeller produced pure garbage that only a rendered
  sample caught, and no dataset in this repository is trusted unseen again.

What this CANNOT do, stated now rather than discovered later: it multiplies
COVERAGE of the cameras that were labelled, not VARIETY. Fifty anchors
propagated to five thousand frames is still fifty viewpoints, and a camera
nobody labelled stays unlabelled. The check on that is the same held-out
evaluation, which contains cameras these anchors do not.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

#: ORB fit quality. A frame from another camera fails these rather than
#: receiving a nonsense box.
MIN_INLIERS = 40
MIN_INLIER_RATIO = 0.35
MAX_RESIDUAL_PX = 3.0
#: The carried box must still look like a rim.
MIN_SCALE, MAX_SCALE = 0.45, 2.2
#: Carried corners: shortest side over longest, so a sliver or a bow-tie dies.
MIN_SQUARENESS = 0.45
EDGE_MARGIN_PX = 4
#: How far to look for an evaluation frame that might share a shot with a
#: candidate. Beyond this no broadcast take survives, so no ORB test is needed.
HOLD_OUT_REACH_S = 45.0


def fit_with_report(grey_source, grey_target, min_inliers=MIN_INLIERS,
                    min_ratio=MIN_INLIER_RATIO, max_residual=MAX_RESIDUAL_PX):
    """`pairwise_homography`'s fit, with the numbers it does not return.

    The shared routine gates on an inlier count and then hands back a matrix
    alone, which is right for tracking and not enough here: a labeller has to
    refuse a fit that is merely arithmetically possible, so the inlier RATIO
    and the residual on the inliers have to be visible. Same detector, same
    ratio test, same RANSAC threshold -- only the report is added.

    Returns (homography, inliers, ratio, residual_px) or (None, 0, 0.0, inf).
    """
    import cv2

    from courtvision.court_tracking import ORB_FEATURES, RATIO

    nothing = (None, 0, 0.0, float("inf"))
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    keys_source, desc_source = orb.detectAndCompute(grey_source, None)
    keys_target, desc_target = orb.detectAndCompute(grey_target, None)
    if desc_source is None or desc_target is None:
        return nothing
    if len(keys_source) < min_inliers or len(keys_target) < min_inliers:
        return nothing

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_source, desc_target, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < RATIO * n.distance]
    if len(good) < min_inliers:
        return nothing

    src = np.float32([keys_source[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([keys_target[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or mask is None:
        return nothing
    mask = mask.ravel().astype(bool)
    count = int(mask.sum())
    ratio = count / float(len(good))
    if count < min_inliers or ratio < min_ratio:
        return None, count, ratio, float("inf")

    carried = cv2.perspectiveTransform(src[mask], matrix).reshape(-1, 2)
    residual = float(np.median(np.hypot(*(carried - dst[mask].reshape(-1, 2)).T)))
    if residual > max_residual:
        return None, count, ratio, residual
    return matrix, count, ratio, residual


def carried_box(homography, centre, width):
    """Anchor rim (centre, width) carried by `homography`, or None if spoilt.

    Returns (cx, cy, w, h). The box is carried as FOUR CORNERS rather than a
    centre and a size, because a homography changes a square into a general
    quadrilateral and only the corners show when that has gone wrong.
    """
    if homography is None:
        return None
    half = width / 2.0
    corners = np.array([[centre[0] - half, centre[1] - half],
                        [centre[0] + half, centre[1] - half],
                        [centre[0] + half, centre[1] + half],
                        [centre[0] - half, centre[1] + half]], np.float64)
    stacked = np.c_[corners, np.ones(4)] @ np.asarray(homography, np.float64).T
    if np.any(np.abs(stacked[:, 2]) < 1e-9):
        return None
    out = stacked[:, :2] / stacked[:, 2:3]
    sides = np.hypot(*(np.roll(out, -1, axis=0) - out).T)
    if sides.min() <= 1e-6:
        return None
    if sides.min() / sides.max() < MIN_SQUARENESS:
        return None
    # Convex, and going round one way: a crossed quad has a sign change.
    edges = np.roll(out, -1, axis=0) - out
    following = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(crosses > 0) or np.all(crosses < 0)):
        return None
    cx, cy = out[:, 0].mean(), out[:, 1].mean()
    w = (out[:, 0].max() - out[:, 0].min())
    h = (out[:, 1].max() - out[:, 1].min())
    return float(cx), float(cy), float(w), float(h)


def acceptable(box, anchor_width, frame_shape,
               min_scale=MIN_SCALE, max_scale=MAX_SCALE, margin=EDGE_MARGIN_PX):
    """Is a carried box worth writing out as a label?"""
    if box is None:
        return False
    cx, cy, w, h = box
    height, width = frame_shape[:2]
    if not (margin <= cx <= width - margin and margin <= cy <= height - margin):
        return False
    if w < 8 or h < 8:
        return False
    scale = w / max(anchor_width, 1e-6)
    return min_scale <= scale <= max_scale


def held_out_times(paths, grid_key="frames"):
    """Every instant the gate is scored on, from each evaluation file given."""
    out = []
    for path in paths:
        data = json.load(open(path))
        rows = data.get(grid_key, data if isinstance(data, list) else [])
        for row in rows:
            if isinstance(row, dict) and "t" in row:
                out.append(float(row["t"]))
    return sorted(out)


def nearby(t, held_out, reach=HOLD_OUT_REACH_S):
    """Evaluation instants close enough to `t` to be worth an ORB test."""
    if not len(held_out):
        return []
    lo = int(np.searchsorted(held_out, t - reach))
    hi = int(np.searchsorted(held_out, t + reach))
    return [float(v) for v in held_out[lo:hi]]


def shares_a_shot(grey, others, min_inliers=MIN_INLIERS):
    """Does any of `others` register to `grey`, i.e. is it the same take?

    `others` are already-loaded greyscale evaluation frames. Registration is
    the operational definition of "the same picture from the same camera",
    which is the thing a training set must not contain a copy of.
    """
    for other in others:
        if other is None:
            continue
        if fit_with_report(other, grey)[0] is not None:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--anchors", required=True,
                        help="rim_truth_alternate_cameras.json")
    parser.add_argument("--held-out", action="append", default=[],
                        help="repeatable; every file whose frames the gate scores")
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--reach-s", type=float, default=25.0,
                        help="how far either side of an anchor to try. A camera "
                             "is used in bursts; beyond this the ORB gate says no "
                             "anyway and the frames cost time to decode.")
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2

    anchors = [f for f in json.load(open(args.anchors))["frames"]]
    held_out = np.asarray(held_out_times(args.held_out), np.float64)
    print(f"{len(anchors)} anchors, {len(held_out)} evaluation instants; a "
          f"candidate is dropped when one of them registers to it")

    capture = cv2.VideoCapture(args.video)

    def grey_at(t):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        return (frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if ok else (None, None)

    found, samples, started = [], [], time.time()
    tried = excluded_anchors = excluded_frames = 0
    for n, anchor in enumerate(anchors):
        at = float(anchor["t"])
        frame_a, grey_a = grey_at(at)
        if grey_a is None:
            continue
        # The evaluation frames this anchor's window could possibly touch,
        # loaded once so the shot test is a comparison and not a video seek.
        watch = [grey_at(v)[1] for v in nearby(at, held_out,
                                               HOLD_OUT_REACH_S + args.reach_s)]
        watch = [g for g in watch if g is not None]
        if shares_a_shot(grey_a, watch):
            excluded_anchors += 1
            continue
        centre, width = anchor["rim"], float(anchor["width"])
        offsets = np.arange(-args.reach_s, args.reach_s + 1e-9, args.step_s)
        for offset in offsets:
            t = at + float(offset)
            if t < 0 or abs(offset) < 1e-9:
                continue
            frame_b, grey_b = grey_at(t)
            if grey_b is None:
                continue
            tried += 1
            if shares_a_shot(grey_b, watch):
                excluded_frames += 1
                continue
            hop, inliers, ratio, residual = fit_with_report(grey_a, grey_b)
            if hop is None:
                continue
            box = carried_box(hop, centre, width)
            if not acceptable(box, width, frame_b.shape):
                continue
            cx, cy, w, h = box
            found.append({"t": round(t, 3), "rim": [round(cx, 1), round(cy, 1)],
                          "width": round(w, 1), "height": round(h, 1),
                          "from_anchor_s": at, "inliers": inliers,
                          "inlier_ratio": round(ratio, 3),
                          "residual_px": round(residual, 2),
                          "note": anchor.get("note", "")})
            if args.sample_dir and len(samples) < 24 and abs(offset) > 2.0:
                shown = frame_b.copy()
                cv2.rectangle(shown, (int(cx - w / 2), int(cy - h / 2)),
                              (int(cx + w / 2), int(cy + h / 2)), (0, 0, 255), 3)
                cv2.putText(shown, f"{t:.1f}s from {at:.0f}s", (8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                samples.append(cv2.resize(shown, (420, 236)))
        print(f"  anchor {n + 1}/{len(anchors)} at {at:.0f}s: {len(found)} labels so far, "
              f"{(time.time() - started) / 60:.1f} min", flush=True)
    capture.release()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "anchors": args.anchors,
               "min_inliers": MIN_INLIERS,
               "held_out_by": "same-shot ORB registration to an evaluation frame",
               "frames": found}, open(out, "w"))
    if args.sample_dir and samples:
        sample_dir = Path(args.sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        grid = [np.hstack(samples[i:i + 4]) for i in range(0, len(samples) // 4 * 4, 4)]
        if grid:
            cv2.imwrite(str(sample_dir / "propagated_rims.jpg"), np.vstack(grid))
            print(f"  sample sheet: {sample_dir / 'propagated_rims.jpg'}")
    print(f"{tried} frames tried, {len(found)} labels carried "
          f"({len(found) / max(tried, 1):.1%}); {excluded_anchors} anchors and "
          f"{excluded_frames} frames dropped for sharing a shot with the evaluation")
    print(f"  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
