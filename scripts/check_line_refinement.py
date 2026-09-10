"""Registration accuracy on broadcast, measured by lines the fit never saw.

For each court frame: register with the landmark model, then for each family of
painted lines in turn, refine WITHOUT that family and ask where its paint
actually lies, in court feet. The refinement never saw those lines, so the
offset is an accuracy figure rather than a residual, and nothing in it comes
from annotations -- which matters, because the annotations are themselves only
consistent to about 0.35 ft and cannot certify anything finer.

The same lines are measured under the plain landmark registration as a
baseline, so the gain is on identical frames and identical paint.

A control runs on every measurement: the refined registration is shifted by a
known 0.5 ft and the offsets re-measured on the same samples. Their change,
divided by each sample's normal component, must read 0.5 ft. Two earlier
estimators in this project reported damped values (1.99 against 1.84 for a
2 ft slip; 1.0 for 2.0) and would have certified a wrong registration; this
one is not believed unless it passes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SHIFT_FT = 0.5
MIN_PER_FAMILY = 15


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/games/FZAUuuuREg0.mp4")
    parser.add_argument("--weights",
                        default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--start", type=float, default=600.0)
    parser.add_argument("--end", type=float, default=8400.0)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    from courtvision.court_refine import (HOLD_OUT, held_out_offsets, prepare,
                                          refine, sample_normals)
    from courtvision.court_tracking import has_court
    from courtvision.device import resolve_device

    for path in (args.video, args.weights, args.detector):
        if not Path(path).exists():
            print(f"FAIL - missing {path}")
            return 1
    model = YOLO(args.weights)
    detector = YOLO(args.detector)
    device = resolve_device()
    capture = cv2.VideoCapture(args.video)

    court_frames = registered = accepted = 0
    coverage, residual, reasons = [], [], {}
    base_err, refined_err = [], []            # per (frame, family) medians
    by_family: dict[str, list[float]] = {}
    control = []

    for t in np.linspace(args.start, args.end, args.samples):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok or not has_court(frame):
            continue
        court_frames += 1
        result = model.predict(frame, device=device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        xy = result.keypoints.xy[0].cpu().numpy()
        conf = result.keypoints.conf[0].cpu().numpy()
        seen = {i: tuple(xy[i]) for i in range(len(xy))
                if i in KEYPOINTS and conf[i] >= args.conf and (xy[i] > 0).all()}
        start, _ = homography_from_keypoints(seen)
        if start is None:
            continue
        registered += 1
        found = detector.predict(frame, device=device, verbose=False)[0].boxes
        boxes = (found.xyxy.cpu().numpy()[found.cls.cpu().numpy() == 0]
                 if found is not None and len(found) else None)
        prepared = prepare(frame)

        full, info = refine(frame, start, boxes=boxes, prepared=prepared)
        if info["coverage"] is not None:
            coverage.append(info["coverage"])
        if not info["refined"]:
            reasons[info["reason"].split(";")[0][:40]] = reasons.get(
                info["reason"].split(";")[0][:40], 0) + 1
            continue
        accepted += 1
        residual.append(info["residual_px"])

        for family, lines in HOLD_OUT.items():
            base = held_out_offsets(frame, start, lines, boxes, prepared=prepared)
            fitted, finfo = refine(frame, start, boxes=boxes, exclude_lines=lines,
                                   prepared=prepared)
            if not finfo["refined"]:
                continue
            offsets, index = held_out_offsets(frame, fitted, lines, boxes,
                                              prepared=prepared, details=True)
            if len(offsets) < MIN_PER_FAMILY or len(base) < MIN_PER_FAMILY:
                continue
            refined_err.append(float(np.median(np.abs(offsets))))
            base_err.append(float(np.median(np.abs(base))))
            by_family.setdefault(family, []).append(refined_err[-1])

            # Control: shift by a known amount in each court axis and require
            # the measured change to equal it, on the same samples.
            for axis in (0, 1):
                move = np.eye(3)
                move[axis, 2] = SHIFT_FT
                shifted, sindex = held_out_offsets(
                    frame, move @ fitted, lines, boxes, prepared=prepared,
                    details=True)
                common, a, b = np.intersect1d(index, sindex, return_indices=True)
                if not len(common):
                    continue
                component = sample_normals(common)[:, axis]
                strong = np.abs(component) > 0.8
                if strong.sum() >= 5:
                    change = (shifted[b] - offsets[a])[strong] / component[strong]
                    control.append(float(np.median(change)))

    n = max(court_frames, 1)
    print(f"{args.video}")
    print(f"  court frames {court_frames}   landmark-registered {registered} "
          f"({registered/n:.0%})   refinement accepted {accepted} "
          f"({accepted/max(registered,1):.0%} of registered)")
    if reasons:
        print("  refused: " + ", ".join(f"{k} x{v}" for k, v in reasons.items()))
    if coverage:
        print(f"  coverage p10/p50 {np.percentile(coverage,10):.2f}/"
              f"{np.median(coverage):.2f}   residual p50 "
              f"{np.median(residual) if residual else float('nan'):.2f} px")
    if control:
        c = np.array(control)
        verdict = "PASS" if abs(np.median(c) - SHIFT_FT) < 0.05 else "FAIL"
        print(f"  CONTROL  known {SHIFT_FT} ft shift reads {np.median(c):.3f} ft "
              f"(p10 {np.percentile(c,10):.3f}, p90 {np.percentile(c,90):.3f})"
              f"  -> {verdict}")
    if refined_err:
        b, r = np.array(base_err), np.array(refined_err)
        print(f"  HELD-OUT line error   landmark only  p50 {np.median(b):.2f} ft"
              f"   p90 {np.percentile(b,90):.2f} ft")
        print(f"                        refined        p50 {np.median(r):.2f} ft"
              f"   p90 {np.percentile(r,90):.2f} ft   [target 0.30]")
        print(f"                        within 0.3 ft  {np.mean(r <= 0.3):.0%} "
              f"of {len(r)} frame-family measurements")
        for family, values in sorted(by_family.items()):
            print(f"    {family:15s} n={len(values):3d}  p50 {np.median(values):.2f} ft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
