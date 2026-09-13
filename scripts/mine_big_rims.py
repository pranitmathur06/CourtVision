"""Find the CLOSE-UP rim frames by colour and shape, so hand labelling pays.

The rim's four remaining misses are close-ups and alternate cameras, and the
measured cause is that there is no training data for them. Hand labelling is
the answer, but `mine_rim_frames.py` finds frames "nothing can locate a rim
in", and triaged by eye that set yields ONE big close-up ring in twenty --
the rest are distant rims the model already handles, or no rim at all. At that
rate 198 frames of a person's time buys about fifteen useful labels.

This mines for the thing that is actually scarce. A close-up basketball ring is
the most distinctive object in the picture: a LARGE, STRONGLY ORANGE, FLATTENED
ELLIPSE. None of that needs learning, and a classical detector cannot inherit
the blind spots of the model it is meant to supply, which is the whole point --
`build_rim_dataset.py` can only crop where the detector already succeeds, and
that circularity is why four training runs changed nothing.

Checked against the one close-up ring located by hand so far: at 888.5 s the
colour mask gives a contour 461 px wide starting at x=591, against a hand label
of centre 805 width 450 -- the same ring to within a few pixels, found with no
detector at all.

WHAT IT IS NOT: a rim detector. It proposes frames for a person to label, and
it fires happily on the scoreboard's red bands, a State Farm hoarding and any
other big orange thing -- the exact objects that fooled the trained model in
Round 85. Its output is a labelling queue, never a label. Every proposal is
confirmed by eye in `label.html` before it becomes training data, and the
frames it misses are simply frames nobody labels.

WHAT STOPS IT WASTING THAT TIME, declared before it was run:

- MIN_WIDTH_PX, because small rings are the class that already works.
- The broadcast furniture bands top and bottom are excluded: the scoreboard
  and the lower-third are permanently red and permanently there.
- ASPECT, a ring seen from anywhere is wider than it is tall, but never a
  sliver: a long thin red bar is an advertising hoarding.
- SPACING_S, so one long replay cannot supply the whole queue and teach one
  camera angle -- the same trap `mine_rim_frames.py` names.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

#: Only rings this wide: below it is the class the model already handles.
MIN_WIDTH_PX = 150
#: Hue either side of red (OpenCV's 0-179 wraps), well saturated and lit.
HUE_LOW, HUE_HIGH = 12, 172
MIN_SATURATION, MIN_VALUE = 120, 70
#: Height over width. A ring is flattened by perspective but never a sliver;
#: a long thin red bar is an advertising hoarding.
ASPECT = (0.10, 0.95)
MIN_AREA = 400
#: The permanently-red broadcast furniture, as a share of frame height.
TOP_BAND, BOTTOM_BAND = 0.06, 0.82
#: Two queued frames must be this far apart, so one replay cannot fill the queue.
SPACING_S = 4.0


def ring_candidates(frame, min_width=MIN_WIDTH_PX, top=TOP_BAND, bottom=BOTTOM_BAND):
    """Large orange flattened blobs, best first. A labelling queue, not labels."""
    import cv2

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (((h <= HUE_LOW) | (h >= HUE_HIGH))
            & (s >= MIN_SATURATION) & (v >= MIN_VALUE)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        # Size is gated by area and width below. An earlier version also
        # required len(contour) >= 20, which with CHAIN_APPROX_SIMPLE measures
        # how COMPRESSIBLE the outline is rather than how big it is -- a real
        # ring is curved and keeps many points, so it passed by accident while
        # any straight-edged shape was refused whatever its size.
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        x, y, w, box_h = cv2.boundingRect(contour)
        if w < min_width:
            continue
        middle = y + box_h / 2.0
        if middle < top * height or middle > bottom * height:
            continue
        aspect = box_h / max(w, 1)
        if not (ASPECT[0] <= aspect <= ASPECT[1]):
            continue
        out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(box_h),
                    "aspect": round(float(aspect), 2)})
    return sorted(out, key=lambda r: -r["w"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-s", type=float, default=650.0)
    parser.add_argument("--end-s", type=float, default=7900.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--min-width", type=float, default=MIN_WIDTH_PX)
    parser.add_argument("--spacing-s", type=float, default=SPACING_S)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2

    capture = cv2.VideoCapture(args.video)
    found, started, last = [], time.time(), -1e9
    times = np.arange(args.start_s, args.end_s, args.step_s)
    for n, t in enumerate(times):
        if float(t) - last < args.spacing_s:
            continue
        capture.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        rings = ring_candidates(frame, args.min_width)
        if not rings:
            continue
        last = float(t)
        found.append({"index": len(found), "t": round(float(t), 2),
                      "widest": rings[0]["w"], "candidates": rings[:3]})
        if len(found) >= args.limit:
            break
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(times)}, {len(found)} queued, "
                  f"{(time.time() - started) / 60:.1f} min", flush=True)
    capture.release()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "min_width_px": args.min_width,
               "note": "a labelling QUEUE found by colour and shape, never labels",
               "frames": found}, open(out, "w"))
    widths = [r["widest"] for r in found]
    print(f"{len(found)} frames queued for labelling")
    if widths:
        print(f"  widest blob px: min {min(widths)} p50 {np.median(widths):.0f} "
              f"max {max(widths)}")
    print(f"  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
