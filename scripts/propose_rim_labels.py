"""Propose rim boxes on hard frames so labelling becomes verification.

Locating a rim by eye on a gridded panel yields about 1.6 labels per six-frame
sheet, and roughly half are then disqualified for sitting too close to a frame
the gate is scored on. A few hundred trainable labels at that rate is about a
thousand frames looked at.

Verification is much faster than location. The scale-trained detector already
fires on some alternate-camera rims -- 3 of 7 frames nothing else can do -- so
run it at a LOW confidence on mined frames, cut a magnified crop around every
proposal, and lay them out in a numbered grid. The labeller says which numbers
are really rims. Twelve judgements a sheet instead of two locations, and the
accepted box comes with pixel coordinates already attached.

The risk this carries is the one that matters: a proposal-driven label set can
only contain rims the model can already nearly see, so it will not teach it a
viewpoint it has never fired on at all. That is why it is a SUPPLEMENT to the
hand-located set and not a replacement, and why the proposals are cut at a very
low confidence -- the interesting ones are the faint ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PANE = 250
COLS = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", required=True, help="mine_rim_frames.py frames.json")
    parser.add_argument("--weights", default="checkpoints/rim_scale/best.pt")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--per-sheet", type=int, default=12)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device

    model, device = YOLO(args.weights), resolve_device()
    rows = json.load(open(args.frames))["frames"]
    capture = cv2.VideoCapture(args.video)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panes, listing, sheets = [], [], []
    for row in rows:
        if len(listing) >= args.limit:
            break
        t = row["t"]
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        found = model.predict(frame, device=device, verbose=False,
                              imgsz=args.imgsz, conf=args.conf)[0].boxes
        if found is None or not len(found):
            continue
        for conf, box in sorted(zip(found.conf.cpu().numpy(), found.xyxy.cpu().numpy()),
                                key=lambda p: -p[0])[:2]:
            x1, y1, x2, y2 = [float(v) for v in box]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half = max(x2 - x1, y2 - y1, 40.0) * 1.9
            a = int(np.clip(cx - half, 0, frame.shape[1] - 1))
            b = int(np.clip(cx + half, 1, frame.shape[1]))
            c = int(np.clip(cy - half, 0, frame.shape[0] - 1))
            d = int(np.clip(cy + half, 1, frame.shape[0]))
            if b - a < 8 or d - c < 8:
                continue
            pane = cv2.resize(frame[c:d, a:b], (PANE, PANE),
                              interpolation=cv2.INTER_CUBIC)
            sx, sy = PANE / (b - a), PANE / (d - c)
            cv2.rectangle(pane, (int((x1 - a) * sx), int((y1 - c) * sy)),
                          (int((x2 - a) * sx), int((y2 - c) * sy)), (0, 0, 255), 2)
            index = len(listing)
            cv2.putText(pane, f"{index}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            cv2.putText(pane, f"{t:.0f}s c{float(conf):.2f}", (6, PANE - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            panes.append(pane)
            listing.append({"index": index, "t": round(t, 2),
                            "rim": [round(cx, 1), round(cy, 1)],
                            "width": round(x2 - x1, 1), "conf": round(float(conf), 3)})
            if len(panes) == args.per_sheet:
                sheets.append(_write(out_dir, len(sheets), panes))
                panes = []
    if panes:
        sheets.append(_write(out_dir, len(sheets), panes))
    capture.release()
    json.dump({"video": args.video, "weights": args.weights, "conf": args.conf,
               "proposals": listing}, open(out_dir / "proposals.json", "w"), indent=0)
    print(f"{len(listing)} proposals over {len(sheets)} sheets in {out_dir}")
    return 0


def _write(out_dir, n, panes):
    import cv2
    rows = [np.hstack(panes[i:i + COLS]) for i in range(0, len(panes), COLS)]
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    path = out_dir / f"prop{n:02d}.jpg"
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 93])
    return path


if __name__ == "__main__":
    raise SystemExit(main())
