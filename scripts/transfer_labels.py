"""Carry hand-placed landmarks from one encode of a game to another.

The whole-game labels were placed on a 480p copy. The same broadcast exists at
1080p, and re-running the test there separates what resolution costs from
what the method gets wrong -- without asking anyone to label again.

A frame read at the same time from two encodes is the same moment to within a
frame, but not necessarily the same pixels (a frame apart during a pan, and a
different scale). So each labelled frame is registered to its 1080p twin by
image features (ORB homography, RANSAC), and each click is mapped through it.
A frame is dropped when the match is weak (few inliers) or inconsistent (a
large median reprojection error): a click moved by a bad homography would
become a wrong label and an unfair error.

Nothing here reads any court registration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MIN_INLIERS = 150
MAX_REPROJECTION_PX = 1.5        # in the target (1080p) image


def match(source, target):
    """Homography source->target by ORB+RANSAC, with inlier count and residual."""
    import cv2
    orb = cv2.ORB_create(nfeatures=6000)
    gs = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    gt = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    ks, ds = orb.detectAndCompute(gs, None)
    kt, dt = orb.detectAndCompute(gt, None)
    if ds is None or dt is None:
        return None, 0, float("inf")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ds, dt, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None, len(good), float("inf")
    src = np.float32([ks[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kt[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if matrix is None:
        return None, 0, float("inf")
    inl = mask.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src[inl], matrix)
    residual = float(np.median(np.hypot(*(proj - dst[inl]).reshape(-1, 2).T)))
    return matrix, int(inl.sum()), residual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-set", default="court_toyota")
    parser.add_argument("--target-set", default="court_toyota_1080")
    parser.add_argument("--video", default="data/games/FZAUuuuREg0_1080p.mp4")
    args = parser.parse_args()

    import cv2

    src_root = Path("data/labeling") / args.source_set
    dst_root = Path("data/labeling") / args.target_set
    (dst_root / "images").mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(src_root / "manifest.json"))
    labels = json.load(open(src_root / "labels.json"))
    capture = cv2.VideoCapture(args.video)
    frames_out, labels_out, report = [], {}, []
    size = None
    for item in manifest["frames"]:
        capture.set(cv2.CAP_PROP_POS_MSEC, item["t"] * 1000)
        ok, target = capture.read()
        if not ok:
            continue
        size = [target.shape[1], target.shape[0]]
        cv2.imwrite(str(dst_root / "images" / item["file"]), target, [cv2.IMWRITE_JPEG_QUALITY, 97])
        frames_out.append(dict(item))
        rec = labels["frames"].get(item["file"])
        if rec is None:
            continue
        source = cv2.imread(str(src_root / "images" / item["file"]))
        matrix, inliers, residual = match(source, target)
        usable = matrix is not None and inliers >= MIN_INLIERS and residual <= MAX_REPROJECTION_PX
        report.append((item["file"], inliers, residual, usable))
        if not usable:
            continue
        pts = {}
        for k, (x, y) in rec["points"].items():
            q = cv2.perspectiveTransform(np.float32([[[x, y]]]), matrix).reshape(2)
            pts[k] = [round(float(q[0]), 2), round(float(q[1]), 2)]
        labels_out[item["file"]] = {"t": rec["t"], "skip": rec.get("skip", False),
                                    "points": pts, "transfer_inliers": inliers,
                                    "transfer_residual_px": residual}
    json.dump({"video": args.video, "fps": capture.get(cv2.CAP_PROP_FPS), "size": size,
               "frames": frames_out}, open(dst_root / "manifest.json", "w"), indent=1)
    json.dump({"video": args.video, "size": size, "points": labels["points"],
               "frames": labels_out, "source_set": args.source_set},
              open(dst_root / "labels.json", "w"), indent=1)
    kept = sum(1 for r in report if r[3])
    print(f"{len(report)} labelled frames: {kept} transferred, {len(report) - kept} dropped")
    for name, inl, res, ok in report:
        if not ok:
            print(f"  dropped {name}: inliers {inl}, residual {res:.2f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
