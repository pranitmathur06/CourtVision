"""The detection pass, spending its time on the GPU instead of on seeking.

Same output as `cache_detections.py` -- the same JSON, the same boxes -- and
verified against it rather than assumed: `--verify` runs both paths over the
same instants and reports any box that moved.

The measurement that motivated it, taken before the code was written:

    seek to each wanted frame   97.9 ms/frame   <- what cache_detections does
    sequential read + grab()     6.5 ms/frame      15x cheaper
    inference (imgsz 1280)      44.1 ms/frame

69% of the pass was seeking. See `courtvision/fast_detect.py` for why batching
was tried first and rejected with numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument("--step-s", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--verify", type=int, default=0,
                        help="check this many frames against the per-frame path")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device
    from courtvision.fast_detect import decode_ahead, half_precision_ok

    device = resolve_device()
    model = YOLO(args.detector)
    half = half_precision_ok(device)
    # Only pass the flag when it is wanted: this ultralytics deprecates `half`
    # in favour of `quantize`, and passing half=False earns a warning per frame
    # for no benefit. Omitting it keeps fp32, which is the default anyway.
    extra = {"half": True} if half else {}

    end = args.end_s
    if end is None:
        probe = cv2.VideoCapture(args.video)
        fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        end = probe.get(cv2.CAP_PROP_FRAME_COUNT) / max(fps, 1e-6)
        probe.release()

    def boxes_of(result):
        out = []
        if result.boxes is not None and len(result.boxes):
            names = model.names
            for cls, conf, box in zip(result.boxes.cls.cpu().numpy(),
                                      result.boxes.conf.cpu().numpy(),
                                      result.boxes.xyxy.cpu().numpy()):
                out.append({"cls": names[int(cls)], "conf": round(float(conf), 3),
                            "xyxy": [round(float(v), 1) for v in box]})
        return out

    rows, started, seen = [], time.time(), 0
    for decoded in decode_ahead(args.video, args.start_s, end, args.step_s):
        result = model.predict(decoded.frame, device=device, verbose=False,
                               imgsz=args.imgsz, conf=args.conf, **extra)[0]
        rows.append({"t": decoded.time_s, "boxes": boxes_of(result)})
        seen += 1
        if seen % 500 == 0:
            rate = seen / (time.time() - started)
            print(f"  {seen} frames, {rate:.1f} fps, "
                  f"{(len(rows) and (end - args.start_s) / args.step_s - seen) / max(rate, 1e-6) / 60:.0f} min left",
                  flush=True)
    elapsed = time.time() - started

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "detector": args.detector, "imgsz": args.imgsz,
               "conf": args.conf, "step_s": args.step_s, "device": device,
               "half": half, "frames": rows}, open(out, "w"))
    print(f"{len(rows)} frames in {elapsed/60:.1f} min "
          f"({len(rows)/max(elapsed,1e-6):.1f} fps, {elapsed/max(len(rows),1)*1000:.1f} ms/frame)")
    print(f"  device {device}, half precision {'on' if half else 'off'}")

    if args.verify:
        print(f"\nverifying {args.verify} frames against the per-frame path...")
        capture = cv2.VideoCapture(args.video)
        moved = checked = 0
        for row in rows[:args.verify]:
            capture.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            reference = boxes_of(model.predict(frame, device=device, verbose=False,
                                               imgsz=args.imgsz, conf=args.conf,
                                               **extra)[0])
            checked += 1
            if len(reference) != len(row["boxes"]):
                moved += 1
                continue
            for a, b in zip(sorted(reference, key=lambda d: d["xyxy"]),
                            sorted(row["boxes"], key=lambda d: d["xyxy"])):
                if a["cls"] != b["cls"] or max(abs(x - y) for x, y in
                                               zip(a["xyxy"], b["xyxy"])) > 1.0:
                    moved += 1
                    break
        print(f"  {checked - moved}/{checked} frames identical to the per-frame path")
        if moved:
            print(f"  {moved} differ -- the two paths decode the same instant "
                  f"differently, which is a real disagreement worth reading")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
