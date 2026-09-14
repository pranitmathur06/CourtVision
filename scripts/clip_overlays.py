"""Boxes for every clip, so the footage shows what the pipeline actually saw.

A clip on its own shows the play. It does not show the vision stack, which is
the thing this project is. Drawing the detector's own boxes over the footage
puts the two side by side: you watch the play and watch what was detected while
it happened, including the frames where nothing was.

The boxes come from the cached detection pass (`outputs/detections/*.json`, one
row every 0.2 s), not from a fresh run, so what is drawn is exactly what the
rest of the pipeline consumed -- the same boxes that fed shot detection. A
re-run would be a different pass and could disagree with the numbers the page
reports.

SCALED AT DRAW TIME, NOT HERE. Boxes are stored in the source video's 1280x720
pixels and the page scales them to whatever size the clip is displayed at. The
clips are 426 px wide, but a reader who full-screens one should still get boxes
in the right place, and storing pre-scaled coordinates would pin them to one
size forever.

Only the classes worth seeing are kept. `rim` is included because a shot call
depends on it and its absence explains a miss; `handler` is separated from
`player` because who has the ball is the single most useful thing to see.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: The detection cache's own step. Clips are cut on whole tenths, so every clip
#: frame lands on one of these.
STEP_S = 0.2
#: Short codes, because this file is downloaded by a browser: p player,
#: h handler, b ball, r rim.
CODE = {"player": "p", "handler": "h", "ball": "b", "rim": "r"}
#: Below this a box is noise the pipeline itself would not have used.
MIN_CONF = 0.25


def window(boxes_by_time, start_s, duration_s, step_s=STEP_S, min_conf=MIN_CONF):
    """[(offset_s, [[code, x1, y1, x2, y2], ...]), ...] for one clip."""
    out = []
    steps = int(round(duration_s / step_s))
    for i in range(steps):
        t = round(start_s + i * step_s, 1)
        row = boxes_by_time.get(t)
        if not row:
            continue
        drawn = []
        for b in row:
            code = CODE.get(b["cls"])
            if code is None or b["conf"] < min_conf:
                continue
            drawn.append([code] + [int(round(v)) for v in b["xyxy"]])
        if drawn:
            out.append([round(i * step_s, 1), drawn])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--index", action="append", required=True,
                        help="repeatable; cut_event_clips index files")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--min-conf", type=float, default=MIN_CONF)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cache = json.load(open(args.detections))
    by_time = {round(r["t"], 1): r["boxes"] for r in cache["frames"]}

    clips = {}
    for path in args.index:
        for c in json.load(open(path))["clips"]:
            name = c["clip"]
            if name in clips:
                continue
            start = c.get("start_s")
            if start is None:
                start = round(float(c["video_s"]) - 3.0, 1)
            clips[name] = round(float(start), 1)

    overlays, empty = {}, 0
    for name, start in sorted(clips.items()):
        frames = window(by_time, start, args.duration, min_conf=args.min_conf)
        if not frames:
            empty += 1
            continue
        overlays[name] = frames

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"note": "detector boxes in the SOURCE video's 1280x720 pixels; the page "
                       "scales them to however large the clip is shown. Codes: p player, "
                       "h handler, b ball, r rim.",
               "source_size": cache.get("size") or [1280, 720],
               "step_s": STEP_S, "min_conf": args.min_conf,
               "clips": overlays}, open(out, "w"), separators=(",", ":"))
    boxes = sum(len(b) for f in overlays.values() for _, b in f)
    print(f"{len(overlays)} clips carry boxes, {empty} have none in the cache")
    print(f"  {boxes} boxes, {out.stat().st_size/1e6:.2f} MB")
    print(f"  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
