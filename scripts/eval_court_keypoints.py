"""Score court registration the way the product needs it: in feet, per frame.

Ultralytics reports pose mAP over OKS, which is a pixel-similarity score
against a sigma the library invents for any non-COCO keypoint count. It is not
the quantity that matters here. What matters is: on what fraction of frames do
we get a homography at all, and when we do, how far off is a point on the floor.

**The confidence floor is chosen on the validation split, and the result is
reported on test.** Sweeping a threshold on the split you then quote is how
this project has previously talked itself into numbers that did not survive.
The selection rule is fixed here, before any number is seen: maximise the
fraction of frames that register with a court error under the gate. It is one
scalar and it is the gate itself, so there is no room to pick a threshold that
flatters a particular metric.

The truth homography comes from the human annotations plus the recovered
schema, so this measures MODEL vs HUMAN -- if the schema were wrong, both sides
would be wrong together and this would still look good. The schema is checked
separately, by mirror symmetry and by leave-one-out landmark prediction
(0.35 ft, held out); the absolute check on real broadcast is the ORB
consistency test, which uses no annotations at all.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DATA = Path("data/labeled/court_keypoints")
GATE_FT = 2.0
CONF_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def _truth(label: Path, width: int, height: int) -> dict[int, tuple[float, float]]:
    """Visible (v=2) landmarks only.

    The v=0 points carry coordinates, but they are filler: fitted on the
    visible points, they land a median 48.7 ft from where the schema says their
    landmark is, with 0.8% inside 3 ft, and none of them fall outside the image
    so "annotated but off-frame" does not explain it.
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


def _predict(model, path: Path, device: str):
    """Landmarks and their confidences, in pixels."""
    from courtvision.court_keypoints import KEYPOINTS

    result = model.predict(str(path), device=device, verbose=False)[0]
    if result.keypoints is None or len(result.keypoints) == 0:
        return {}
    xy = result.keypoints.xy[0].cpu().numpy()
    conf = (result.keypoints.conf[0].cpu().numpy()
            if result.keypoints.conf is not None else np.ones(len(xy)))
    return {i: (tuple(xy[i]), float(conf[i])) for i in range(len(xy))
            if i in KEYPOINTS and (xy[i] > 0).all()}


def _gather(model, split: str, device: str):
    """Predictions and reference homography per frame, computed once."""
    import cv2

    from courtvision.court_keypoints import MIN_KEYPOINTS, homography_from_keypoints

    frames = []
    for path in sorted((DATA / split / "images").glob("*.jpg")):
        label = DATA / split / "labels" / (path.stem + ".txt")
        if not label.exists():
            continue
        from PIL import Image
        width, height = Image.open(path).size
        truth = _truth(label, width, height)
        if len(truth) < MIN_KEYPOINTS + 2:
            continue          # no trustworthy reference to score against
        reference, _ = homography_from_keypoints(truth)
        if reference is None:
            continue
        probe = np.array([[xy] for xy in truth.values()], dtype=np.float32)
        frames.append((_predict(model, path, device), reference, probe))
    return frames


def _score(frames, conf: float):
    import cv2

    from courtvision.court_keypoints import homography_from_keypoints

    registered, wrong_half, usable = 0, 0, 0
    errors: list[float] = []
    found: list[int] = []
    for predicted, reference, probe in frames:
        seen = {i: xy for i, (xy, c) in predicted.items() if c >= conf}
        found.append(len(seen))
        matrix, _ = homography_from_keypoints(seen)
        if matrix is None:
            continue
        registered += 1
        got = cv2.perspectiveTransform(probe, matrix).reshape(-1, 2)
        want = cv2.perspectiveTransform(probe, reference).reshape(-1, 2)
        err = np.hypot(*(got - want).T)
        errors.extend(err)
        # The failure that made the painted key unusable: right court, wrong end.
        wrong_half += (np.median(got[:, 1]) > 47) != (np.median(want[:, 1]) > 47)
        usable += np.median(err) <= GATE_FT
    return {"n": len(frames), "registered": registered, "usable": usable,
            "errors": np.array(errors), "wrong_half": wrong_half,
            "found": float(np.median(found)) if found else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",
                        default="runs/pose/checkpoints/court_keypoints/weights/best.pt")
    parser.add_argument("--select-on", default="valid")
    parser.add_argument("--report-on", default="test")
    args = parser.parse_args()

    from ultralytics import YOLO

    from courtvision.device import resolve_device

    weights = Path(args.weights)
    if not weights.exists():
        print(f"FAIL - no weights at {weights}")
        return 1
    model = YOLO(str(weights))
    device = resolve_device()

    selection = _gather(model, args.select_on, device)
    scores = {c: _score(selection, c) for c in CONF_GRID}
    # Pre-registered rule: the fraction of frames that register AND land inside
    # the gate. Ties break to the higher floor, which is the more conservative.
    best = max(CONF_GRID, key=lambda c: (scores[c]["usable"] / max(scores[c]["n"], 1), c))
    print(f"confidence floor chosen on '{args.select_on}' "
          f"({scores[best]['n']} frames): {best}")
    for c in CONF_GRID:
        s = scores[c]
        marker = " <-" if c == best else ""
        print(f"    {c:.1f}  usable {s['usable']/max(s['n'],1):6.1%}"
              f"  registered {s['registered']/max(s['n'],1):6.1%}{marker}")

    s = _score(_gather(model, args.report_on, device), best)
    n = max(s["n"], 1)
    print(f"\nreported on '{args.report_on}': {s['n']} frames")
    print(f"  landmarks found per frame   p50 {s['found']:.0f}")
    print(f"  registered                  {s['registered']}/{s['n']} = "
          f"{s['registered']/n:.1%}   [gate 90%]")
    if len(s["errors"]):
        print(f"  court error                 p50 {np.median(s['errors']):.2f} ft"
              f"   p90 {np.percentile(s['errors'], 90):.2f} ft   "
              f"[gate p50 <= {GATE_FT:.0f} ft]")
    print(f"  frames inside the gate      {s['usable']}/{s['n']} = {s['usable']/n:.1%}")
    print(f"  wrong end of the floor      {s['wrong_half']}/{max(s['registered'],1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
