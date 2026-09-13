"""Draw the rim claim the system actually reports, for verdict labelling.

The verdicts are scored against what the system OUTPUTS, and that has already
gone wrong twice. The diagnostic sheets in earlier rounds drew two circles --
the rim projected from the camera pose and the rim the detector found -- and on
three frames the projected circle was judged instead of the claim, turning
three correct frames into misses. Then two more frames were marked "miss" from
downscaled panels when one holds no rim at all and one holds nothing anybody
can resolve.

So this draws ONE thing: every entry in the frame's reported `rim` list, which
is exactly what `eval_rim_and_ball.py` scores. Nothing else is overlaid,
because anything else on the picture is something to mistake for the answer.

Each frame is rendered at full width with its grid index, and a magnified pane
of each claim beside it, so "is the claim on the ring" can be answered without
squinting at a thumbnail -- which is the failure this exists to prevent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def unlabelled(grid, verdicts, want, start=0):
    """Indices of grid frames with no rim verdict yet, spread through the game."""
    have = {i for i, row in verdicts.items() if "rim" in row}
    free = [i for i in range(start, len(grid)) if i not in have]
    if len(free) <= want:
        return free
    step = len(free) / float(want)
    return [free[int(k * step)] for k in range(want)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--system", required=True, help="build_rim_ball.py output")
    parser.add_argument("--verdicts", action="append", default=[],
                        help="repeatable; frames already judged are skipped")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--per-sheet", type=int, default=4)
    parser.add_argument("--pane", type=int, default=150)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import cv2

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_verdicts import parse

    grid = json.load(open(args.system))["frames"]
    verdicts = {}
    for path in args.verdicts:
        verdicts.update(parse(path))
    wanted = unlabelled(grid, verdicts, args.count)
    print(f"{len(grid)} grid frames, {len(verdicts)} already judged; "
          f"rendering {len(wanted)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.video)
    rows, made = [], []
    for index in wanted:
        row = grid[index]
        capture.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        shown = frame.copy()
        claims = row.get("rim") or []
        for claim in claims:
            cv2.circle(shown, (int(claim[0]), int(claim[1])), 26, (0, 0, 255), 3)
        cv2.putText(shown, f"#{index}  {row['t']:.1f}s  {len(claims)} claim(s)",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        panel = cv2.resize(shown, (900, 506))

        panes = []
        for claim in claims[:2]:
            cx, cy = int(claim[0]), int(claim[1])
            half = args.pane // 2
            x0 = int(np.clip(cx - half, 0, frame.shape[1] - args.pane))
            y0 = int(np.clip(cy - half, 0, frame.shape[0] - args.pane))
            pane = frame[y0:y0 + args.pane, x0:x0 + args.pane]
            if pane.shape[0] != args.pane or pane.shape[1] != args.pane:
                continue
            pane = cv2.resize(pane, (253, 253), interpolation=cv2.INTER_NEAREST)
            cv2.circle(pane, (int((cx - x0) * 253 / args.pane),
                              int((cy - y0) * 253 / args.pane)),
                       int(26 * 253 / args.pane), (0, 0, 255), 2)
            panes.append(pane)
        while len(panes) < 2:
            panes.append(np.zeros((253, 253, 3), np.uint8))
        rows.append(np.hstack([panel, np.vstack(panes)]))
        made.append(index)

    capture.release()
    for n in range(0, len(rows), args.per_sheet):
        sheet = np.vstack(rows[n:n + args.per_sheet])
        path = out_dir / f"rim{n // args.per_sheet:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    json.dump({"system": args.system, "indices": made},
              open(out_dir / "frames.json", "w"))
    print(f"{len(made)} frames over {(len(rows) + args.per_sheet - 1) // args.per_sheet} "
          f"sheets in {out_dir}")
    print(f"  indices: {made}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
