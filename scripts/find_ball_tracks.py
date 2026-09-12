"""Label balls by finding smooth tracks, so a detector can learn to rank them.

The ball is measured to be BOTH a detection and a selection problem in roughly
equal parts: of five hand-located balls, two had no candidate within 90 px, two
were the detector's top candidate and were found, and one existed at rank 3 and
was lost. Fixing the ranking means training, and training means labels for the
LOW-CONFIDENCE true balls -- exactly the ones no existing signal points at.

A track is that signal. Over a short window, with the camera's own motion
removed by ORB, a real ball traces a smooth path at a ball-like speed. A
spectator's head does not move at all; a shoe or a shoulder moves with its
player and rarely in a straight, fast, consistent line. So a chain of
candidates, one per frame, whose motion-compensated positions step smoothly is
a ball -- and the OTHER candidates in those frames are hard negatives, which is
what a ranking problem needs.

This is the Viterbi tracker that failed in Round 62 with the one change that
matters: it ran in RAW IMAGE pixels, where a panning camera makes a stationary
head look fast and a shot look impossible, and so it preferred heads. Motion
compensation is what makes smoothness mean what it is supposed to mean.

Deliberately conservative: MIN_LENGTH frames, a speed band, and a straightness
test, because a wrong label is worse than a missing one. It will label only a
fraction of the balls in a game, and that is the right trade for training data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

#: Frames in a window, at STEP_S apart.
STEP_S = 0.2
WINDOW = 7
#: A track must span this many frames.
MIN_LENGTH = 5
#: Motion-compensated speed, pixels per frame, that a ball moves at. Below the
#: floor it is furniture; above the ceiling it is a mismatch, not a trajectory.
#:
#: The floor was 14 and the labels came back 58% correct -- a RUNNING PLAYER'S
#: shoulder traces just as smooth a path as a ball, so smoothness alone cannot
#: separate them and 42% of the labels were heads, shoulders and headbands.
#: Raised so that only a ball in FLIGHT qualifies: a pass or a shot moves
#: several times faster than a player can run, and that gap is the one thing
#: about the motion that a player cannot imitate. It labels far fewer balls,
#: which is the right trade for training data.
MIN_SPEED_PX = 40.0
MAX_SPEED_PX = 260.0
#: Step-to-step change in velocity, as a share of speed. A ball's path bends
#: gently; a jump between two different objects does not.
MAX_TURN = 0.55


def smooth_enough(points, min_speed=MIN_SPEED_PX, max_speed=MAX_SPEED_PX,
                  max_turn=MAX_TURN):
    """Is this chain of motion-compensated positions a ball's path?"""
    points = np.asarray(points, np.float64)
    if len(points) < 3:
        return False
    steps = np.diff(points, axis=0)
    speeds = np.hypot(steps[:, 0], steps[:, 1])
    if speeds.min() < min_speed or speeds.max() > max_speed:
        return False
    turns = np.hypot(*np.diff(steps, axis=0).T)
    return bool(np.all(turns <= max_turn * np.maximum(speeds[:-1], 1e-6) + 6.0))


def best_track(frames, min_length=MIN_LENGTH, min_speed=MIN_SPEED_PX):
    """Greedy chain over per-frame candidate positions, or None.

    `frames` is a list of arrays of motion-compensated (x, y) positions, one
    array per frame in time order.
    """
    best = None
    for start in range(len(frames) - min_length + 1):
        for seed in range(len(frames[start])):
            chain = [(start, seed)]
            points = [frames[start][seed]]
            for j in range(start + 1, len(frames)):
                if not len(frames[j]):
                    break
                last = points[-1]
                predicted = (2 * np.asarray(points[-1]) - np.asarray(points[-2])
                             if len(points) > 1 else np.asarray(last))
                distances = np.hypot(frames[j][:, 0] - predicted[0],
                                     frames[j][:, 1] - predicted[1])
                pick = int(np.argmin(distances))
                if distances[pick] > MAX_SPEED_PX:
                    break
                chain.append((j, pick))
                points.append(frames[j][pick])
            if len(chain) >= min_length and smooth_enough(points, min_speed):
                if best is None or len(chain) > len(best[0]):
                    best = (chain, points)
    return best


def _write(path, video, detections, found):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": video, "detections": detections,
               "min_length": MIN_LENGTH, "frames": found}, open(out, "w"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True,
                        help="a dense cache; its own times set the windows")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--min-speed", type=float, default=MIN_SPEED_PX)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument("--stride-windows", type=int, default=6,
                        help="skip windows so the labels are spread over the game")
    parser.add_argument("--orb-scale", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=40,
                        help="write the labels found so far this often. Without it a "
                             "long run can only be stopped by throwing its work away, "
                             "which is how 407 labels were lost once.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from track_camera import hop_at_scale

    cache = json.load(open(args.detections))
    rows = [r for r in cache["frames"]
            if r["t"] >= args.start_s and (args.end_s is None or r["t"] <= args.end_s)]
    capture = cv2.VideoCapture(args.video)

    found, saved, started = [], 0, time.time()
    for w in range(0, len(rows) - WINDOW, WINDOW * args.stride_windows):
        window = rows[w:w + WINDOW]
        greys, cands = [], []
        ok_all = True
        for row in window:
            capture.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
            ok, frame = capture.read()
            if not ok:
                ok_all = False
                break
            greys.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cands.append([b for b in row["boxes"]
                          if b["cls"] == "ball" and b["conf"] >= args.conf])
        if not ok_all or any(not c for c in cands):
            continue

        # Carry every frame's candidates into the middle frame's pixels.
        middle = len(greys) // 2
        compensated = []
        for i, (grey, boxes) in enumerate(zip(greys, cands)):
            points = np.array([[(b["xyxy"][0] + b["xyxy"][2]) / 2,
                                (b["xyxy"][1] + b["xyxy"][3]) / 2] for b in boxes])
            if i == middle:
                compensated.append(points)
                continue
            hop = hop_at_scale(grey, greys[middle], None, None, args.orb_scale)
            if hop is None:
                compensated.append(np.zeros((0, 2)))
                continue
            stacked = np.c_[points, np.ones(len(points))] @ np.asarray(hop).T
            good = np.abs(stacked[:, 2]) > 1e-9
            compensated.append(stacked[good, :2] / stacked[good, 2:3])

        track = best_track(compensated, min_speed=args.min_speed)
        if track is None:
            continue
        chain, _ = track
        for j, pick in chain:
            if pick >= len(cands[j]):
                continue
            found.append({"t": window[j]["t"],
                          "ball": cands[j][pick]["xyxy"],
                          "conf": cands[j][pick]["conf"],
                          "others": [b["xyxy"] for k, b in enumerate(cands[j])
                                     if k != pick]})
        if len(found) >= args.limit:
            break
        if len(found) - saved >= args.save_every:
            _write(args.out, args.video, args.detections, found)
            saved = len(found)
            print(f"  {len(found)} labelled, {(time.time() - started) / 60:.1f} min",
                  flush=True)

    capture.release()

    _write(args.out, args.video, args.detections, found)
    confs = [f["conf"] for f in found]
    print(f"{len(found)} ball labels from tracks; confidence p50 "
          f"{np.median(confs) if confs else float('nan'):.2f}, "
          f"{sum(1 for c in confs if c < 0.3)} of them below 0.30 "
          f"(the ones the detector is getting wrong)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
