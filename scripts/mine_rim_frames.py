"""Find the frames whose rim nothing in this repo can locate, for hand labelling.

Round 67 measured the rim at 0.758 over a whole video and named the gap: every
in-game miss is a camera the main one is not -- under-basket, baseline, a rim
close-up, a replay inside a graphic. On those frames the landmark model returns
no keypoints and the detector returns no rim box at any threshold or inference
size, so there is no signal left to bootstrap from and the boxes have to come
from a person.

This picks the frames worth that person's time:

- no pose (the fixed camera could not register it), AND
- no rim box from the detector even at a floor of DETECTOR_FLOOR, AND
- spread through the video, so one long replay sequence cannot supply most of
  the training set and teach the detector that one camera angle.

Mined blind, that mostly yields mid-court views and close-ups with no rim in
them at all -- good negatives, but the positives are what is scarce. So
`--after-shots` aimed the search at the seconds after a made basket, on the
theory that a replay follows one. It does not: that window lands on the
opponent's inbound and the transition back, and 8 of 8 frames sampled from it
had no rim in them at all.

`--clock-stopped` is the prior that works. A replay airs when the game clock
is NOT running, and the clock was already read off the scoreboard in Round 65
by a pass that knows nothing about which frames this pipeline finds hard, so
it cannot flatter the result.

Frames are rendered large -- these are the close-ups, and the whole point is
that the rim is big in them -- with a grid overlay so a centre and a width can
be read off by eye. A rim is unambiguous to label, which is why hand labels are
defensible here and were not for screens.

The output feeds `build_rim_dataset.py`, which mixes them with rim boxes the
camera model gives FREE on main-camera frames, where the projection is already
accurate to 0.13 ft.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: A detector box at or above this counts as "the detector already has it".
DETECTOR_FLOOR = 0.10
#: Two mined frames must be at least this far apart in the video.
SPACING_S = 6.0
GRID_STEP = 100


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--step-s", type=float, default=2.0)
    parser.add_argument("--spacing-s", type=float, default=SPACING_S)
    parser.add_argument("--limit", type=int, default=180)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=None,
                        help="stop here. Without it --clock-stopped treats everything "
                             "after the final buzzer as stopped, and a whole batch of "
                             "sheets came back as trophy presentations.")
    parser.add_argument("--after-shots", default=None,
                        help="align_shots_to_video.py output; look just after makes")
    parser.add_argument("--after-window", default="3,14",
                        help="seconds after a made shot to search")
    parser.add_argument("--clock-stopped", default=None,
                        help="read_game_clock.py output; search only where the game "
                             "clock is NOT running, which is when replays air")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-sheet", type=int, default=4)
    args = parser.parse_args()

    import cv2

    cache = json.load(open(args.detections))
    by_time = {round(row["t"], 3): row for row in cache["frames"]}
    cache_times = np.array(sorted(by_time)) if by_time else np.array([])

    posed = set()
    for row in json.load(open(args.poses))["frames"]:
        if row.get("pose"):
            posed.add(round(row["t"], 1))

    def detector_has_rim(t):
        if not len(cache_times):
            return False
        j = int(np.argmin(np.abs(cache_times - t)))
        if abs(cache_times[j] - t) > 0.3:
            return False
        return any(b["cls"] == "rim" and b["conf"] >= DETECTOR_FLOOR
                   for b in by_time[cache_times[j]]["boxes"])

    capture = cv2.VideoCapture(args.video)
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)

    stopped = None
    if args.clock_stopped:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from detect_shots import live_play
        running = live_play(json.load(open(args.clock_stopped))["readings"])
        stopped = lambda when: not running(when)      # noqa: E731

    if args.after_shots:
        shots = [s["t"] for s in json.load(open(args.after_shots))
                 if s.get("made") and s.get("gap_s", 0) <= 2.0]
        low, high = (float(v) for v in args.after_window.split(","))
        times = sorted({round(s + d, 3) for s in shots
                        for d in np.arange(low, high, args.step_s)})
    else:
        times = [float(t) for t in np.arange(args.start_s, duration, args.step_s)]

    wanted, last = [], -1e9
    for t in times:
        t = float(t)
        if t < args.start_s or t >= duration:
            continue
        if args.end_s is not None and t > args.end_s:
            break
        if round(t, 1) in posed or detector_has_rim(t):
            continue
        if stopped is not None and not stopped(t):
            continue
        if t - last < args.spacing_s:
            continue
        wanted.append(t)
        last = t
    step = max(1, len(wanted) // args.limit)
    wanted = wanted[::step][:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles, sheets, index = [], [], 0
    listing = []
    for t in wanted:
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        tile = cv2.resize(frame, (800, 450))
        for x in range(0, 800, GRID_STEP):
            cv2.line(tile, (x, 0), (x, 450), (70, 70, 70), 1)
            cv2.putText(tile, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (120, 220, 120), 1)
        for y in range(0, 450, GRID_STEP):
            cv2.line(tile, (0, y), (800, y), (70, 70, 70), 1)
            cv2.putText(tile, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (120, 220, 120), 1)
        cv2.putText(tile, f"#{index} {t:.0f}s", (6, 444), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        tiles.append(tile)
        listing.append({"index": index, "t": round(t, 3)})
        index += 1
        if len(tiles) == args.per_sheet:
            sheets.append(_write(out_dir, len(sheets), tiles))
            tiles = []
    if tiles:
        sheets.append(_write(out_dir, len(sheets), tiles))
    capture.release()

    json.dump({"video": args.video, "grid_step": GRID_STEP, "tile": [800, 450],
               "frames": listing}, open(out_dir / "frames.json", "w"), indent=0)
    print(f"{len(listing)} frames nothing can locate a rim in, over {len(sheets)} sheets")
    print(f"  {out_dir}/frames.json")
    return 0


def _write(out_dir, n, tiles):
    import cv2
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    path = out_dir / f"mine{n:02d}.jpg"
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 93])
    return path


if __name__ == "__main__":
    raise SystemExit(main())
