"""Leave-one-out: does the schema predict a landmark it was not fitted to?

Checked in because the 0.35 ft figure was quoted from a script that no longer
existed, which makes a number unreproducible even when it is right.

What this DOES measure: whether each landmark's court position is consistent
with the others, per frame, across arenas. Hold one out, fit on the rest, ask
where the held-out one lands.

What it does NOT measure, and the eval docstring used to imply it did: whether
the schema is correct in absolute terms. It is invariant to any projective
transform of the whole schema -- `truth . G` scores identically, because the
per-frame fit simply absorbs `G`. It also measures agreement with where
ANNOTATORS clicked, which is not the same as where the paint is. The absolute
question is settled by `check_court_lines.py`, whose reference is true NBA
geometry rather than either schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path("data/labeled/court_keypoints")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default=None,
                        help="JSON of index -> [x, y] in feet; defaults to the "
                             "shipped KEYPOINTS")
    parser.add_argument("--min-visible", type=int, default=9)
    args = parser.parse_args()

    import cv2
    from PIL import Image

    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints

    schema = KEYPOINTS
    if args.schema:
        schema = {int(k): tuple(v) for k, v in json.load(open(args.schema)).items()}

    errors: list[float] = []
    per_index: dict[int, list[float]] = {}
    for split in ("train", "valid", "test"):
        for label in sorted((DATA / split / "labels").glob("*.txt")):
            image = next((DATA / split / "images").glob(label.stem + ".*"), None)
            if image is None:
                continue
            width, height = Image.open(image).size
            parts = label.read_text().split()
            if len(parts) != 5 + 48 * 3:
                continue
            visible = {}
            for i in range(48):
                x, y, v = (float(t) for t in parts[5 + 3 * i: 8 + 3 * i])
                if i in schema and v == 2:
                    visible[i] = (x * width, y * height)
            if len(visible) < args.min_visible:
                continue
            for held in visible:
                rest = {i: xy for i, xy in visible.items() if i != held}
                matrix, _ = homography_from_keypoints(
                    rest, require_orientation=False)
                if matrix is None:
                    continue
                got = cv2.perspectiveTransform(
                    np.array([[visible[held]]], dtype=np.float32), matrix).ravel()
                d = float(np.hypot(*(got - np.array(schema[held]))))
                errors.append(d)
                per_index.setdefault(held, []).append(d)

    err = np.array(errors)
    print(f"held-out landmark prediction  n={len(err)}  "
          f"p50 {np.median(err):.2f} ft  p90 {np.percentile(err, 90):.2f} ft  "
          f"<3ft {np.mean(err < 3):.1%}")
    worst = sorted(per_index.items(), key=lambda kv: -np.median(kv[1]))[:4]
    print("worst indices: " + ", ".join(
        f"{i} {np.median(v):.2f} ft (n={len(v)})" for i, v in worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
