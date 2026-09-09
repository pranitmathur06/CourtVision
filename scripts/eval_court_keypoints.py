"""Score court registration the way the product needs it: in feet, per frame.

Ultralytics reports pose mAP over OKS, which is a pixel-similarity score with a
sigma the library invents for any non-COCO keypoint count. It is not the
quantity that matters here. What matters is: on what fraction of frames do we
get a homography at all, and when we do, how far off is a point on the floor.

The truth homography comes from the human annotations plus the recovered
schema, so this measures MODEL vs HUMAN -- if the schema itself were wrong,
both sides would be wrong together and this would still look good. The schema
is checked separately, by mirror symmetry and by leave-one-out landmark
prediction (0.35 ft, held out); the absolute check on real broadcast comes from
the ORB consistency test, which needs no annotations at all.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DATA = Path("data/labeled/court_keypoints")


def _truth(label: Path, width: int, height: int):
    """Visible (v=2) landmarks only.

    The v=0 points carry coordinates, but they are filler: fitted on the
    visible points, they land a median 48.7 ft from where the schema says they
    are, with 0.8% inside 3 ft. Training or scoring on them would be training
    on noise.
    """
    from courtvision.court_keypoints import KEYPOINTS

    parts = label.read_text().split()
    if len(parts) != 5 + 48 * 3:
        return {}
    out = {}
    for i in range(48):
        x, y, v = (float(t) for t in parts[5 + 3 * i: 8 + 3 * i])
        if i in KEYPOINTS and v == 2:
            out[i] = (x * width, y * height)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",
                        default="runs/pose/checkpoints/court_keypoints/weights/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="per-landmark confidence floor")
    args = parser.parse_args()

    import cv2
    from PIL import Image
    from ultralytics import YOLO

    from courtvision.court_keypoints import (KEYPOINTS, MIN_KEYPOINTS,
                                             homography_from_keypoints)
    from courtvision.device import resolve_device

    weights = Path(args.weights)
    if not weights.exists():
        print(f"FAIL - no weights at {weights}")
        return 1
    model = YOLO(str(weights))
    device = resolve_device()

    images = sorted((DATA / args.split / "images").glob("*.jpg"))
    registered, scored, wrong_half = 0, [], 0
    found: list[int] = []
    errs: list[float] = []
    for path in images:
        label = DATA / args.split / "labels" / (path.stem + ".txt")
        if not label.exists():
            continue
        width, height = Image.open(path).size
        truth = _truth(label, width, height)
        if len(truth) < MIN_KEYPOINTS + 2:
            continue                       # no trustworthy reference to score against
        scored.append(path)

        result = model.predict(str(path), device=device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        xy = result.keypoints.xy[0].cpu().numpy()
        conf = (result.keypoints.conf[0].cpu().numpy()
                if result.keypoints.conf is not None else np.ones(len(xy)))
        seen = {i: tuple(xy[i]) for i in range(len(xy))
                if i in KEYPOINTS and conf[i] >= args.conf
                and (xy[i] > 0).all()}
        found.append(len(seen))
        matrix, _ = homography_from_keypoints(seen)
        if matrix is None:
            continue
        reference, _ = homography_from_keypoints(truth)
        if reference is None:
            continue
        registered += 1

        # Disagreement in feet over the floor the camera actually shows,
        # sampled at the annotated landmarks rather than a blind pixel grid.
        probe = np.array([[truth[i]] for i in truth], dtype=np.float32)
        got = cv2.perspectiveTransform(probe, matrix).reshape(-1, 2)
        want = cv2.perspectiveTransform(probe, reference).reshape(-1, 2)
        err = np.hypot(*(got - want).T)
        errs.extend(err)
        # The failure that made the painted key useless: right court, wrong end.
        wrong_half += (np.median(got[:, 1]) > 47) != (np.median(want[:, 1]) > 47)

    errs = np.array(errs)
    n = len(scored)
    print(f"split {args.split}: {n} frames with a usable reference")
    print(f"  landmarks found per frame   p50 {np.median(found or [0]):.0f} "
          f"(of {len(KEYPOINTS)} in the schema)")
    print(f"  registered                  {registered}/{n} = {registered/max(n,1):.1%}"
          f"   [gate 90%]")
    if len(errs):
        print(f"  court error                 p50 {np.median(errs):.2f} ft   "
              f"p90 {np.percentile(errs, 90):.2f} ft   [gate p50 <= 2 ft]")
        print(f"  within 2 ft                 {np.mean(errs < 2):.1%}")
    print(f"  wrong end of the floor      {wrong_half}/{max(registered,1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
