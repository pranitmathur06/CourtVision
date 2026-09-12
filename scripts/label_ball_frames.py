"""Render frames for hand-locating the ball, with a grid to read positions off.

Every automatic label source tried finds the ball only where it is already
easy. The shot chart finds it at the rim; the track finder needs it in free
flight, unoccluded and moving fast. The frames the detectors MISS are the
opposite of those -- the ball held still, half behind a hand, 15 px across
among a crowd of same-sized objects -- so no signal in this repo points at
them and they have to be located by a person.

These frames are chosen by position in the video, not by whether anything found
a ball in them, so the set is representative of the grid the gate is scored on
rather than of what the detector happens to like.

The grid overlay is the whole point: a ball located by eye is worth nothing
without a coordinate, and a coordinate read off an unmarked panel carries tens
of pixels of error -- which is more than the tolerance. Lines every GRID_STEP
with their positions printed let a centre be read to a few pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GRID_STEP = 50
PANEL_W = 1280


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--times", required=True,
                        help="JSON with a 'frames' list of {t}, e.g. a system output")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--per-sheet", type=int, default=2)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import cv2

    rows = json.load(open(args.times))["frames"]
    picked = list(range(args.first, len(rows), max(args.stride, 1)))[:args.count]
    capture = cv2.VideoCapture(args.video)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    panel_h = int(round(PANEL_W * height / width))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles, sheets, listing = [], [], []
    for index in picked:
        t = rows[index]["t"]
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        tile = cv2.resize(frame, (PANEL_W, panel_h))
        for x in range(0, PANEL_W, GRID_STEP):
            heavy = x % (GRID_STEP * 4) == 0
            cv2.line(tile, (x, 0), (x, panel_h), (60, 60, 60), 2 if heavy else 1)
            if heavy:
                cv2.putText(tile, str(x), (x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (120, 255, 120), 1)
        for y in range(0, panel_h, GRID_STEP):
            heavy = y % (GRID_STEP * 4) == 0
            cv2.line(tile, (0, y), (PANEL_W, y), (60, 60, 60), 2 if heavy else 1)
            if heavy:
                cv2.putText(tile, str(y), (3, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (120, 255, 120), 1)
        cv2.putText(tile, f"#{index} {t:.1f}s", (8, panel_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(tile)
        listing.append({"index": index, "t": t})
        if len(tiles) == args.per_sheet:
            path = out_dir / f"ball{len(sheets):02d}.jpg"
            cv2.imwrite(str(path), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 94])
            sheets.append(path)
            tiles = []
    if tiles:
        path = out_dir / f"ball{len(sheets):02d}.jpg"
        cv2.imwrite(str(path), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 94])
        sheets.append(path)
    capture.release()
    json.dump({"video": args.video, "panel": [PANEL_W, panel_h],
               "grid_step": GRID_STEP, "frames": listing},
              open(out_dir / "frames.json", "w"), indent=0)
    print(f"{len(listing)} frames over {len(sheets)} sheets in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
