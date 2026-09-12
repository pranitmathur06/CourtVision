"""Label the balls nobody has labelled: the held ones and the dribbled ones.

Every ball label this project owns is a ball IN FLIGHT.
`find_ball_tracks.py:51` sets a motion-compensated floor of 40 px/frame and
says why -- at 14 px/frame the labels came back 58% correct, because a running
player's shoulder traces just as smooth a path as a ball, and only flight moves
faster than a player can run.

So the detector has never seen a ball at rest in someone's hands, and the
evaluation samples the game uniformly, where most balls are held or dribbled.
The truth set says so in its own notes: "held by the dribbler", "loose-ball
scramble". That is the recall gap, and no amount of re-ranking closes it.

TWO THINGS MAKE THIS TRACTABLE NOW THAT IT WAS NOT IN ROUND 73.

First, the native-scale single-class detector proposes about FOUR candidates a
frame where the four-class detector at imgsz 2560 proposes fifty-nine. Chaining
four candidates is a different problem from chaining sixty; the 58%-correct
disaster was a chaining problem dressed as a speed problem.

Second, and this is the actual idea: A DRIBBLED BALL BOUNCES, AND NOTHING ELSE
IN THE PICTURE DOES. Once the camera's own motion is removed by ORB, a
dribble is a periodic vertical reversal of 20-40 px at 1-3 Hz. A head does not
do that. A shoulder does not do that. A headband on a running player rises and
falls by a few pixels, not by a ball's diameter twice a second. It is the one
signature that separates the population this dataset is missing from the
population that poisoned it last time, and it is available exactly where the
speed floor is useless.

A chain is accepted as a ball when it is a

    DRIBBLE   at least MIN_REVERSALS vertical direction changes, each with an
              amplitude of at least MIN_BOUNCE_PX, with bounded acceleration
              and a total path length over MIN_TRAVEL_PX so that a stationary
              object with detector jitter cannot qualify.

FLIGHT is available behind --accept-flight and is OFF, because the first run
showed why it should be. It is mis-calibrated here to begin with: 40 px/frame
at 0.2 s is 200 px/s, and the same constant at 0.1 s demands 400 px/s. Worse,
the one flight chain it accepted in a whole game was a COACH'S HEAD on a
close-up sideline shot -- where the depth is shallow and ORB cannot compensate
the camera, so every object appears to move fast at once and "fast and smooth"
describes the whole picture.

The dribble rule is immune to that failure, which is the argument for it
standing alone: a compensation failure makes everything drift together, and
drift is not a periodic reversal. Nothing in a basketball arena bounces except
the ball.

Sampled at STEP_S = 0.1 s because a dribble at 2-3 Hz needs several samples per
cycle; at the 0.2 s of the flight miner a bounce aliases away entirely.

Deliberately conservative, for the same reason as before: a wrong label is
worse than a missing one, and a contact sheet is rendered for eyes BEFORE
anything trains on this.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

STEP_S = 0.1
WINDOW = 13
MIN_LENGTH = 8
#: Flight, as in find_ball_tracks.py.
MIN_FLIGHT_PX = 40.0
MAX_SPEED_PX = 260.0
MAX_TURN = 0.55
#: Dribble. A bounce is a real vertical excursion, not detector jitter.
MIN_REVERSALS = 2
MIN_BOUNCE_PX = 18.0
MIN_TRAVEL_PX = 40.0
#: A dribble's step-to-step acceleration, in pixels, allowed to be generous
#: because the bounce itself is a sharp reversal.
MAX_ACCEL_PX = 70.0


def flight_like(points, min_speed=MIN_FLIGHT_PX, max_speed=MAX_SPEED_PX,
                max_turn=MAX_TURN):
    """The Round 73 rule: fast, and turning gently."""
    points = np.asarray(points, np.float64)
    if len(points) < 3:
        return False
    steps = np.diff(points, axis=0)
    speeds = np.hypot(steps[:, 0], steps[:, 1])
    if speeds.min() < min_speed or speeds.max() > max_speed:
        return False
    turns = np.hypot(*np.diff(steps, axis=0).T)
    return bool(np.all(turns <= max_turn * np.maximum(speeds[:-1], 1e-6) + 6.0))


def bounces(vertical, min_amplitude=MIN_BOUNCE_PX):
    """Count confirmed vertical turning points worth at least `min_amplitude`.

    A hysteresis zigzag: the series has to retrace by `min_amplitude` from its
    running extreme before a turn is counted, so jitter around a stationary
    point contributes nothing however noisy it is, and a single rise-and-fall
    counts once rather than twice.
    """
    vertical = np.asarray(vertical, np.float64)
    if len(vertical) < 3:
        return 0
    count, direction = 0, 0
    extreme = float(vertical[0])
    low = high = float(vertical[0])
    for value in vertical[1:]:
        value = float(value)
        if direction == 0:
            low, high = min(low, value), max(high, value)
            if value >= low + min_amplitude:
                direction, extreme = 1, value
            elif value <= high - min_amplitude:
                direction, extreme = -1, value
            continue
        if direction > 0:
            if value > extreme:
                extreme = value
            elif value <= extreme - min_amplitude:
                count += 1
                direction, extreme = -1, value
        else:
            if value < extreme:
                extreme = value
            elif value >= extreme + min_amplitude:
                count += 1
                direction, extreme = 1, value
    return count


def dribble_like(points, min_reversals=MIN_REVERSALS, min_travel=MIN_TRAVEL_PX,
                 max_accel=MAX_ACCEL_PX):
    """Does this motion-compensated chain bounce like a dribble?"""
    points = np.asarray(points, np.float64)
    if len(points) < 5:
        return False
    steps = np.diff(points, axis=0)
    if np.hypot(*steps.T).sum() < min_travel:
        return False
    if np.hypot(*np.diff(steps, axis=0).T).max() > max_accel:
        return False
    return bounces(points[:, 1]) >= min_reversals


def best_chain(frames, min_length=MIN_LENGTH):
    """Longest greedy chain over per-frame motion-compensated positions."""
    best = None
    for start in range(len(frames) - min_length + 1):
        for seed in range(len(frames[start])):
            chain = [(start, seed)]
            points = [frames[start][seed]]
            for j in range(start + 1, len(frames)):
                if not len(frames[j]):
                    break
                predicted = (2 * np.asarray(points[-1]) - np.asarray(points[-2])
                             if len(points) > 1 else np.asarray(points[-1]))
                gaps = np.hypot(frames[j][:, 0] - predicted[0],
                                frames[j][:, 1] - predicted[1])
                pick = int(np.argmin(gaps))
                if gaps[pick] > MAX_SPEED_PX:
                    break
                chain.append((j, pick))
                points.append(frames[j][pick])
            if len(chain) >= min_length:
                if best is None or len(chain) > len(best[0]):
                    best = (chain, points)
    return best


def classify(points, accept_flight=False):
    """'flight', 'dribble' or None for a motion-compensated chain.

    Flight is off by default: see the module docstring. It costs a real ball
    now and then and it bought a coach's head.
    """
    if dribble_like(points):
        return "dribble"
    if accept_flight and flight_like(points):
        return "flight"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="native scale; the crops were cut without resizing")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument("--stride-s", type=float, default=9.0,
                        help="between windows, so labels spread over the game")
    parser.add_argument("--held-out", action="append", default=[],
                        help="repeatable; a window sharing a shot with one of "
                             "these frames is skipped entirely")
    parser.add_argument("--accept-flight", action="store_true",
                        help="also take fast smooth chains. Off: the only one a "
                             "whole game produced was a coach's head.")
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--save-every", type=int, default=60)
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from propagate_rim_labels import (HOLD_OUT_REACH_S, fit_with_report,
                                      held_out_times, nearby, shares_a_shot)

    model, device = YOLO(args.detector), resolve_device()
    held_out = np.asarray(held_out_times(args.held_out), np.float64)
    capture = cv2.VideoCapture(args.video)
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / max(
        capture.get(cv2.CAP_PROP_FPS), 1.0)
    end = args.end_s if args.end_s is not None else duration

    guard = {}

    def grey_at(t):
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        return (frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if ok else (None, None)

    def balls(frame):
        found = model.predict(frame, device=device, verbose=False,
                              imgsz=args.imgsz, conf=args.conf)[0].boxes
        out = []
        if found is not None and len(found):
            names = model.names
            for cls, conf, box in zip(found.cls.cpu().numpy(),
                                      found.conf.cpu().numpy(),
                                      found.xyxy.cpu().numpy()):
                if names[int(cls)] != "ball":
                    continue
                out.append({"xyxy": [float(v) for v in box], "conf": float(conf)})
        return out

    def write():
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"video": args.video, "detector": args.detector,
                   "imgsz": args.imgsz, "step_s": STEP_S,
                   "accepted_as": "flight or dribble",
                   "frames": found}, open(out, "w"))

    found, samples, started, saved = [], [], time.time(), 0
    kinds = {"flight": 0, "dribble": 0}
    starts = np.arange(args.start_s, max(end - WINDOW * STEP_S, args.start_s),
                       args.stride_s)
    for n, begin in enumerate(starts):
        times = [float(begin + i * STEP_S) for i in range(WINDOW)]
        frames, greys, cands = [], [], []
        ok_all = True
        for t in times:
            frame, grey = grey_at(t)
            if grey is None:
                ok_all = False
                break
            frames.append(frame)
            greys.append(grey)
            cands.append(balls(frame))
        if not ok_all or any(not c for c in cands):
            continue

        middle = len(greys) // 2
        if len(held_out):
            watch = []
            for v in nearby(times[middle], held_out, HOLD_OUT_REACH_S):
                if v not in guard:
                    guard[v] = grey_at(v)[1]
                if guard[v] is not None:
                    watch.append(guard[v])
            if shares_a_shot(greys[middle], watch):
                continue

        compensated = []
        for i, (grey, boxes) in enumerate(zip(greys, cands)):
            points = np.array([[(b["xyxy"][0] + b["xyxy"][2]) / 2,
                                (b["xyxy"][1] + b["xyxy"][3]) / 2] for b in boxes])
            if i == middle:
                compensated.append(points)
                continue
            hop = fit_with_report(grey, greys[middle])[0]
            if hop is None:
                compensated.append(np.zeros((0, 2)))
                continue
            stacked = np.c_[points, np.ones(len(points))] @ np.asarray(hop).T
            good = np.abs(stacked[:, 2]) > 1e-9
            compensated.append(stacked[good, :2] / stacked[good, 2:3])

        chain = best_chain(compensated)
        if chain is None:
            continue
        links, points = chain
        kind = classify(points, args.accept_flight)
        if kind is None:
            continue
        kinds[kind] += 1
        for j, pick in links:
            if pick >= len(cands[j]):
                continue
            found.append({"t": times[j], "ball": cands[j][pick]["xyxy"],
                          "conf": cands[j][pick]["conf"], "kind": kind,
                          "others": [b["xyxy"] for k, b in enumerate(cands[j])
                                     if k != pick]})
        if args.sample_dir and len(samples) < 24:
            shown = frames[links[len(links) // 2][0]].copy()
            box = cands[links[len(links) // 2][0]][links[len(links) // 2][1]]["xyxy"]
            cv2.rectangle(shown, (int(box[0]) - 12, int(box[1]) - 12),
                          (int(box[2]) + 12, int(box[3]) + 12), (0, 0, 255), 2)
            cv2.putText(shown, f"{kind} {times[middle]:.1f}s", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            samples.append(cv2.resize(shown, (426, 240)))
        if len(found) >= args.limit:
            break
        if len(found) - saved >= args.save_every:
            write()
            saved = len(found)
            print(f"  {len(found)} labels ({kinds['flight']} flight, "
                  f"{kinds['dribble']} dribble), window {n + 1}/{len(starts)}, "
                  f"{(time.time() - started) / 60:.1f} min", flush=True)
    capture.release()

    write()
    if args.sample_dir and samples:
        sample_dir = Path(args.sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        grid = [np.hstack(samples[i:i + 4]) for i in range(0, len(samples) // 4 * 4, 4)]
        if grid:
            cv2.imwrite(str(sample_dir / "harvested_balls.jpg"), np.vstack(grid))
            print(f"  sample sheet: {sample_dir / 'harvested_balls.jpg'}")
    print(f"{len(found)} labels from {kinds['flight']} flight chains and "
          f"{kinds['dribble']} dribble chains")
    print(f"  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
