"""Propose screen candidates by motion, so the labelling pool is not 5% positive.

Sampling a random kit-member near the ball gives a base rate around 5-10%: at
any instant most players are not screening. Labelling to 50 positives that way
costs six hundred to a thousand clips.

A screener is geometrically distinctive and needs no court registration to
spot. In image space, with camera motion removed by the same ORB warp used
everywhere else, he is a player who comes to a STOP while a team-mate passes
within about a body-width. That is the definition, not a proxy, and it should
enrich the pool by several times.

The proposal is deliberately loose -- it decides which clips are worth LOOKING
at, and the label still comes from watching. A tight proposal rule would just
be the old geometric detector wearing a different hat, and its errors would
become the labels.
"""
import sys, json, math, random
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, "src")
sys.path.insert(0, "/private/tmp/claude-501/-Users-sanikakapoor-CourtVision/b07b2c60-60e4-4569-acba-6e12d2475831/scratchpad")
from ultralytics import YOLO
from courtvision.court_tracking import has_court
from courtvision.candidates import (court_region, kit_members, near_ball,
                                    stands_on_court, MIN_BOX_HEIGHT_PX)
from fast_candidates import read_span, camera_warp, detect_all, DETECTOR, IMGSZ

CACHE = "outputs/screen_proposals"
# A screener's apparent speed, in box-heights per second, once the camera is
# taken out. Generous: this selects what to look at, not what to believe.
MAX_SCREENER_SPEED = 0.9
# How close the passing team-mate comes, in box-heights.
PASS_WITHIN = 1.3
MIN_PASSER_SPEED = 1.4


def centres(boxes):
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)


def propose(frames, dets, mid, region_mask, ball):
    """Indices into dets[mid] that look like somebody setting a screen."""
    here = dets[mid]
    if len(here) < 4:
        return []
    tall = (here[:, 3] - here[:, 1]) >= MIN_BOX_HEIGHT_PX
    on_court = stands_on_court(region_mask, here)
    eligible = np.flatnonzero(tall & on_court)
    if len(eligible) < 3:
        return []

    # Follow every eligible box one step each way, camera removed, to get an
    # apparent velocity in box-heights per second.
    speeds, positions = {}, {}
    for step, weight in ((mid - 2, -1), (mid + 2, 1)):
        if step < 0 or step >= len(frames):
            continue
        warp = camera_warp(frames[mid], frames[step])
        if warp is None:
            continue
        pts = np.hstack([centres(here), np.ones((len(here), 1))]) @ warp.T
        ok = np.abs(pts[:, 2]) > 1e-9
        moved = np.full((len(here), 2), np.nan)
        moved[ok] = pts[ok, :2] / pts[ok, 2:3]
        other = dets[step]
        if not len(other):
            continue
        got = centres(other)
        for i in eligible:
            if not ok[i]:
                continue
            gaps = np.hypot(got[:, 0] - moved[i, 0], got[:, 1] - moved[i, 1])
            k = int(np.argmin(gaps))
            height = here[i, 3] - here[i, 1]
            if gaps[k] > 0.7 * height:
                continue
            travel = gaps[k] / max(height, 1.0)
            speeds.setdefault(i, []).append(travel / (2 / 10.0))
            positions.setdefault(i, []).append(got[k])

    spots = centres(here)
    out = []
    for i in eligible:
        if i not in speeds or not speeds[i]:
            continue
        if float(np.mean(speeds[i])) > MAX_SCREENER_SPEED:
            continue                     # this one is running, not screening
        height = here[i, 3] - here[i, 1]
        # Somebody else, moving, passes close by.
        for j in eligible:
            if j == i or j not in speeds or not speeds[j]:
                continue
            if float(np.mean(speeds[j])) < MIN_PASSER_SPEED:
                continue
            if np.hypot(*(spots[j] - spots[i])) <= PASS_WITHIN * height:
                out.append(int(i))
                break
    return out


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    out_dir = Path(CACHE)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    ball_at = {}
    for entry in json.load(open("outputs/ball_detections.json"))["frames"]:
        if entry["ball"]:
            best = max(entry["ball"], key=lambda b: b[2])
            if best[2] >= 0.25:
                ball_at[round(entry["t"], 1)] = (best[0], best[1])
    shots = [e["video_s"] for e in json.load(
        open("outputs/aligned_events.json"))["events"] if "Shot" in e["action"]]

    cap = cv2.VideoCapture("data/raw_clips/fullgame.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    det = YOLO(DETECTOR)
    rng = random.Random(7)
    found = tried = 0
    for _ in range(want * 20):
        if found >= want:
            break
        tried += 1
        moment = rng.choice(shots) - rng.uniform(1.0, 8.0)
        frames, times = read_span(cap, fps, moment - 0.6, moment + 0.6)
        if len(frames) < 8:
            continue
        mid = min(range(len(times)), key=lambda i: abs(times[i] - moment))
        if not has_court(frames[mid]):
            continue
        ball = ball_at.get(round(times[mid], 1))
        if ball is None:
            continue
        dets = detect_all(det, frames)
        region = court_region(frames[mid])
        if region is None:
            continue
        picks = propose(frames, dets, mid, region, ball)
        if not picks:
            continue
        # Keep the proposal that also belongs to a kit and sits near the ball,
        # so the clip is comparable with the random-sample candidates.
        boxes = dets[mid]
        ok = kit_members(frames[mid], boxes) & near_ball(boxes, ball, 420.0)
        picks = [i for i in picks if ok[i]] or picks
        chosen = boxes[picks[0]]
        name = f"{found:04d}_t{moment:.1f}"
        np.savez_compressed(out_dir / f"{name}.npz",
                            box=np.array(chosen, dtype=float),
                            t=np.array([moment]))
        index.append(dict(name=name, t=round(float(moment), 2),
                          box=[float(v) for v in chosen]))
        json.dump(index, open(f"{CACHE}/index.json", "w"))
        found += 1
    print(f"  {found} moments offered a screener-shaped candidate "
          f"out of {tried} tried ({found/max(tried,1):.0%})")
    print(f"  random near-ball sampling accepted about 40% of moments, so this")
    print(f"  is more selective -- the question is whether it is ENRICHED.")


if __name__ == "__main__":
    main()
