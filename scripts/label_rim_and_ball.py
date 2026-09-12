"""Render sampled frames for hand-labelling the rim and the ball.

One tile per frame: the whole picture (so "is it visible at all?" can be
answered honestly, including the frames where the system reports nothing), and
beside it a zoom on each thing the system claims (so "is that claim right?"
can be answered at the pixel).

Proposals are drawn but not assumed. Every one of them is either confirmed or
overruled by the labeller, and a frame where the system says nothing still
gets looked at -- that is where misses hide, and a labelling pass that only
checks the system's own output can only ever measure precision.

Colours: GREEN a projected rim, RED a detected rim, YELLOW the chosen ball,
CYAN every other candidate the detector offered. The cyan matters as much as
the yellow: it separates "the ball was never found" from "the wrong candidate
was taken", which are different problems with different fixes.

The index printed on each tile is the frame's position in the grid, which is
what the labels are keyed by.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TILE_W, TILE_H = 640, 360
ZOOM = 180
PER_SHEET = 8


def draw(frame, row, scale):
    """The whole picture at `scale`, with every proposal marked."""
    import cv2
    small = cv2.resize(frame, (TILE_W, TILE_H))
    for x, y in row.get("projected") or []:
        cv2.circle(small, (int(x * scale[0]), int(y * scale[1])), 13, (0, 255, 0), 2)
    for x, y in row.get("detected") or []:
        cv2.circle(small, (int(x * scale[0]), int(y * scale[1])), 9, (0, 0, 255), 2)
    for x, y in row.get("candidates") or []:
        if row.get("ball") and abs(x - row["ball"][0]) < 1 and abs(y - row["ball"][1]) < 1:
            continue
        cv2.rectangle(small, (int(x * scale[0]) - 6, int(y * scale[1]) - 6),
                      (int(x * scale[0]) + 6, int(y * scale[1]) + 6), (255, 255, 0), 1)
    if row.get("ball"):
        x, y = row["ball"]
        cv2.rectangle(small, (int(x * scale[0]) - 9, int(y * scale[1]) - 9),
                      (int(x * scale[0]) + 9, int(y * scale[1]) + 9), (0, 255, 255), 2)
    return small


def zoom_on(frame, centre, size=ZOOM, half=45):
    """A magnified square around `centre`, or grey if there is nothing to show."""
    import cv2
    if centre is None:
        return np.full((size, size, 3), 60, np.uint8)
    h, w = frame.shape[:2]
    cx, cy = int(centre[0]), int(centre[1])
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.full((size, size, 3), 60, np.uint8)
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    cv2.drawMarker(out, (int((cx - x0) * size / max(x1 - x0, 1)),
                         int((cy - y0) * size / max(y1 - y0, 1))),
                   (0, 255, 255), cv2.MARKER_CROSS, 22, 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--system", required=True, help="track_rim_and_ball.py output")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--sheet", type=int, default=PER_SHEET)
    args = parser.parse_args()

    import cv2

    rows = json.load(open(args.system))["frames"]
    chosen = rows[args.first:args.first + args.count] if args.count else rows[args.first:]
    capture = cv2.VideoCapture(args.video)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale = (TILE_W / width, TILE_H / height)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles, made = [], []
    for k, row in enumerate(chosen):
        index = args.first + k
        capture.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        panel = draw(frame, row, scale)
        rim_zoom = zoom_on(frame, (row.get("projected") or row.get("detected") or [None])[0])
        ball_zoom = zoom_on(frame, row.get("ball"))
        side = np.vstack([rim_zoom, ball_zoom])
        side = cv2.resize(side, (ZOOM, TILE_H))
        tile = np.hstack([panel, side])
        cv2.putText(tile, f"#{index} {row['t']:.0f}s "
                          f"{'REG' if row.get('registered') else 'no-reg'} "
                          f"rim:{len(row.get('rim') or [])} "
                          f"ball:{'y' if row.get('ball') else 'n'}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (255, 255, 255), 1)
        tiles.append(tile)
        if len(tiles) == args.sheet:
            made.append(_write(out_dir, len(made), tiles))
            tiles = []
    if tiles:
        made.append(_write(out_dir, len(made), tiles))
    capture.release()
    print(f"{len(made)} sheets in {out_dir}")
    for path in made:
        print(f"  {path}")
    return 0


def _write(out_dir, n, tiles):
    import cv2
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    path = out_dir / f"sheet{n:02d}.jpg"
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


if __name__ == "__main__":
    raise SystemExit(main())
