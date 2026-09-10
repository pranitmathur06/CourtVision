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
    # Required: a default here once pointed at the held-out arena, so an
    # omitted flag would have scored the footage meant to stay unseen.
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights",
                        default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--start", type=float, default=600.0)
    parser.add_argument("--end", type=float, default=8400.0)
    parser.add_argument("--single-start", action="store_true",
                        help="refine from the landmark registration only. On "
                             "broadcast, paint supports several sharp solutions "
                             "about one line spacing apart, and extra starts "
                             "jump between them; this isolates what the "
                             "landmark start alone achieves")
    parser.add_argument("--min-peak-ratio", type=float, default=None,
                        help="override the acceptance threshold. Calibration "
                             "dumps are made at 0, so the selection script can "
                             "apply each candidate to frames AND refits exactly; "
                             "a dump gated at 2 cannot simulate 5, because its "
                             "held-out refits were already gated at 2")
    parser.add_argument("--polarity", choices=("bright", "both", "all"), default=None,
                        help="which ridges count as paint; default is the "
                             "module's PAINT_POLARITY")
    parser.add_argument("--dump", default=None,
                        help="write every (frame, family) measurement to JSON, "
                             "so a tail can be traced to its cause -- a far-off "
                             "landmark start, a lock, one line family -- rather "
                             "than read off an aggregate over a handful of frames")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    import courtvision.court_refine as court_refine
    from courtvision.court_refine import (HOLD_OUT, held_out_offsets, prepare,
                                          refine, sample_normals)
    if args.polarity:
        court_refine.PAINT_POLARITY = args.polarity
    if args.min_peak_ratio is not None:
        court_refine.MIN_PEAK_RATIO = args.min_peak_ratio
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
    starts = {"starts": ((0.0, 0.0),)} if args.single_start else {}

    court_frames = registered = accepted = 0
    coverage, residual, reasons = [], [], {}
    base_err, refined_err = [], []            # per (frame, family) medians
    by_family: dict[str, list[float]] = {}
    control = []
    attempted = missing = refit_refused = 0
    records: list[dict] = []
    frame_list: list[dict] = []

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

        full, info = refine(frame, start, boxes=boxes, prepared=prepared,
                            **starts)
        frame_list.append({"t": float(t), "refined": bool(info["refined"]),
                           "peak_ratio": info["peak_ratio"],
                           "reason": info["reason"]})
        if info["coverage"] is not None:
            coverage.append(info["coverage"])
        if not info["refined"]:
            reasons[info["reason"].split(";")[0][:40]] = reasons.get(
                info["reason"].split(";")[0][:40], 0) + 1
            continue
        accepted += 1
        residual.append(info["residual_px"])
        frame_info = {"t": float(t), "drift_ft": info["drift_ft"],
                      "peak_ratio": info["peak_ratio"],
                      "explained": info.get("explained"),
                      "residual_px": info["residual_px"]}

        for family, lines in HOLD_OUT.items():
            base = held_out_offsets(frame, start, lines, boxes, prepared=prepared)
            if len(base) < MIN_PER_FAMILY:
                continue          # the family is not visible in this frame
            attempted += 1
            fitted, finfo = refine(frame, start, boxes=boxes, exclude_lines=lines,
                                   prepared=prepared, **starts)
            record = dict(frame_info, family=family,
                          landmark_err=float(np.median(np.abs(base))),
                          refit_ratio=finfo["peak_ratio"],
                          refit_reason=finfo["reason"],
                          held_out_drift=finfo["drift_ft"])
            records.append(record)
            if not finfo["refined"]:
                # Counted, not skipped: a visible family whose refit is refused
                # was previously in neither the failures nor the denominator.
                refit_refused += 1
                record["refined_err"] = None
                record["found"] = 0
                continue
            offsets, index = held_out_offsets(frame, fitted, lines, boxes,
                                              prepared=prepared, details=True)
            record["found"] = int(len(offsets))
            if len(offsets) < MIN_PER_FAMILY:
                record["refined_err"] = None
                # Visible, but its paint is not near where the refined fit
                # puts it. Dropping this silently would score only the fits
                # that were already close -- a fit locked one line spacing
                # (3 ft, ~100 px) off can never be measured by a 24 px window,
                # so it would vanish from the statistics instead of counting
                # against them.
                missing += 1
                continue
            refined_err.append(float(np.median(np.abs(offsets))))
            base_err.append(float(np.median(np.abs(base))))
            record["refined_err"] = refined_err[-1]
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

    if args.dump:
        import json
        import subprocess
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        meta = {"video": args.video, "args": vars(args), "commit": commit,
                "dirty": bool(subprocess.run(["git", "status", "--porcelain"],
                                             capture_output=True, text=True).stdout.strip()),
                "min_peak_ratio": court_refine.MIN_PEAK_RATIO,
                "polarity": court_refine.PAINT_POLARITY,
                "court_frames": court_frames, "registered": registered,
                "accepted": accepted, "control": control,
                "frames": frame_list}
        json.dump({"meta": meta, "records": records}, open(args.dump, "w"), indent=1)
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
    if len(control) < 10:
        # A control that did not run is a failure, not an absence: an estimator
        # that loses its samples after the shift produces no entry at all.
        print(f"  CONTROL  FAIL -- only {len(control)} control measurements ran")
    if control:
        c = np.array(control)
        verdict = "PASS" if abs(np.median(c) - SHIFT_FT) < 0.05 and len(c) >= 10 else "FAIL"
        print(f"  CONTROL  known {SHIFT_FT} ft shift reads {np.median(c):.3f} ft "
              f"(p10 {np.percentile(c,10):.3f}, p90 {np.percentile(c,90):.3f})"
              f"  -> {verdict}")
    if attempted:
        print(f"  visible held-out families {attempted}: refit refused "
              f"{refit_refused}, paint not found {missing}  -- both counted as "
              f"failures, not dropped")
    if refined_err:
        b = np.array(base_err)
        # Every statistic counts lost and refused as failures. The p90 used to
        # be taken over found lines only while the p50 counted them -- which
        # reported 0.67 ft for a tail that is really 0.89.
        r = np.array(refined_err + [np.inf] * (missing + refit_refused))
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
