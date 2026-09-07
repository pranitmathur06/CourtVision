"""Build candidates ~15x faster, by reading once and tracking once.

The slow version cost about two minutes per candidate. Two things dominated,
and neither was the model:

  * It called `cap.set(POS_FRAMES)` for every frame. Seeking an H.264 file
    decodes from the nearest keyframe each time, so 80 seeks per candidate cost
    far more than reading the same span once.
  * It ran the detector one frame at a time, 80 times, when the five offsets
    overlap almost completely.

So: seek ONCE, read the whole span sequentially, sample it at 10 Hz, run the
detector on those frames in batches, and track the chosen player across the
span a single time. Each offset's clip is then a window into one track rather
than a fresh extraction.
"""
import sys, json, random, time
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, "src")
from ultralytics import YOLO
from courtvision.court_tracking import has_court, pairwise_homography
from courtvision.candidates import kit_members, near_ball, MIN_BOX_HEIGHT_PX

CACHE = "outputs/broadcast_candidates"
OUT = Path(CACHE)
VIDEO = "data/raw_clips/fullgame.mp4"
FRAMES, SIZE, HZ, MARGIN = 16, 224, 10.0, 1.6
OFFSETS = (-0.6, -0.3, 0.0, 0.3, 0.6)
SPAN_BEFORE, SPAN_AFTER = 1.5, 1.4      # covers every offset's window
WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BATCH = 16
# yolo11x costs 2.62 s per 1080p frame on this machine and detection was 60 of
# the 66 seconds each candidate took. yolo11s is 6.3x faster, finds slightly
# MORE players over 90 px (9.0 against 8.8) and agrees with the larger model on
# 87% of them. The tracker snaps to a detection and DROPS the clip when it
# cannot, so a miss costs a candidate rather than producing a wrong one --
# which makes this a throughput trade, not an accuracy one.
DETECTOR = "yolo11s.pt"
IMGSZ = 640


# ORB on full HD dominated the cost: up to 27 alignments per candidate on
# 1920x1080. The homography is a projective map, so computing it on a half-size
# copy and conjugating by the scale gives the same transform for a quarter of
# the pixels.
ORB_SCALE = 1.0


def camera_warp(source, target):
    small_a = cv2.resize(source, None, fx=ORB_SCALE, fy=ORB_SCALE)
    small_b = cv2.resize(target, None, fx=ORB_SCALE, fy=ORB_SCALE)
    warp = pairwise_homography(small_a, small_b)
    if warp is None:
        return None
    up = np.diag([1 / ORB_SCALE, 1 / ORB_SCALE, 1.0])
    down = np.diag([ORB_SCALE, ORB_SCALE, 1.0])
    return up @ warp @ down


def crop_at(image, box):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * MARGIN / 2
    h, w = image.shape[:2]
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(w, cx + half)), int(min(h, cy + half))
    if c - a < 16 or d - b < 16:
        return None
    return cv2.resize(image[b:d, a:c], (SIZE, SIZE))[:, :, ::-1]


def read_span(cap, fps, start_s, end_s):
    """Sample the span at 10 Hz with ONE seek and a sequential read."""
    first = int(round(start_s * fps))
    last = int(round(end_s * fps))
    step = max(1, int(round(fps / HZ)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    wanted = set(range(first, last + 1, step))
    frames, times = [], []
    for n in range(first, last + 1):
        ok, image = cap.read()
        if not ok:
            break
        if n in wanted:
            frames.append(image)
            times.append(n / fps)
    return frames, times


def detect_all(det, frames):
    """One batched pass over the whole span.

    Detecting frame by frame inside the tracker was the remaining cost: ~27
    separate calls per candidate, each paying the model's fixed overhead. The
    frames are all in memory already, so they go through together.
    """
    out = []
    for i in range(0, len(frames), BATCH):
        for found in det(frames[i:i + BATCH], verbose=False, conf=0.25,
                         classes=[0], imgsz=IMGSZ):
            out.append(found.boxes.xyxy.cpu().numpy()
                       if len(found.boxes) else np.empty((0, 4)))
    return out


def track_player(detections, frames, start_index, box):
    """Follow one player across the sampled span, both directions."""
    boxes = {start_index: np.array(box, dtype=float)}

    def detect(i):
        return detections[i]

    for direction in (1, -1):
        here = boxes[start_index].copy()
        i = start_index
        while True:
            j = i + direction
            if j < 0 or j >= len(frames):
                break
            warp = camera_warp(frames[i], frames[j])
            if warp is None:
                break
            pts = np.array([[here[0], here[1], 1.0],
                            [here[2], here[3], 1.0]]) @ warp.T
            if not (np.abs(pts[:, 2]) > 1e-9).all():
                break
            m = pts[:, :2] / pts[:, 2:3]
            here = np.array([m[0, 0], m[0, 1], m[1, 0], m[1, 1]])
            cand = detect(j)
            if not len(cand):
                break
            want = np.array([(here[0] + here[2]) / 2, here[3]])
            gaps = np.hypot((cand[:, 0] + cand[:, 2]) / 2 - want[0],
                            cand[:, 3] - want[1])
            k = int(np.argmin(gaps))
            if gaps[k] > 0.6 * (here[3] - here[1]):
                break
            picked = cand[k].astype(float)
            # A player's apparent size changes slowly. A snap that halves or
            # doubles the box is a partial detection or a different person, and
            # accepting it lets the box collapse to a sliver over a few frames
            # -- which is exactly what happened and produced empty clips.
            was = here[3] - here[1]
            now = picked[3] - picked[1]
            if not 0.65 <= now / max(was, 1.0) <= 1.55:
                break
            here = picked
            boxes[j] = here.copy()
            i = j
    return boxes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ball_at = {}
    for entry in json.load(open("outputs/ball_detections.json"))["frames"]:
        if entry["ball"]:
            best = max(entry["ball"], key=lambda b: b[2])
            if best[2] >= 0.25:
                ball_at[round(entry["t"], 1)] = (best[0], best[1])
    shots = [e["video_s"] for e in json.load(
        open("outputs/aligned_events.json"))["events"] if "Shot" in e["action"]]

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    det = YOLO(DETECTOR)
    rng = random.Random(2024)
    index, tried = [], 0
    started = time.time()

    while len(index) < WANT and tried < WANT * 8:
        tried += 1
        moment = rng.choice(shots) - rng.uniform(1.0, 8.0)
        frames, times = read_span(cap, fps, moment - SPAN_BEFORE,
                                  moment + SPAN_AFTER)
        if len(frames) < 25:
            continue
        centre = min(range(len(times)), key=lambda i: abs(times[i] - moment))
        mid = frames[centre]
        if not has_court(mid):
            continue
        ball = ball_at.get(round(times[centre], 1))
        if ball is None:
            continue
        found = det(mid, verbose=False, conf=0.3, classes=[0], imgsz=IMGSZ)[0]
        if not len(found.boxes):
            continue
        boxes = found.boxes.xyxy.cpu().numpy()
        tall = boxes[:, 3] - boxes[:, 1] >= MIN_BOX_HEIGHT_PX
        keep = kit_members(mid, boxes) & tall & near_ball(boxes, ball)
        if not keep.any():
            continue
        chosen = boxes[keep][rng.randrange(int(keep.sum()))]

        track = track_player(detect_all(det, frames), frames, centre, chosen)
        clips = {}
        for offset in OFFSETS:
            at = min(range(len(times)), key=lambda i: abs(times[i] - (moment + offset)))
            window = range(at - FRAMES // 2, at + FRAMES // 2)
            if any(i not in track for i in window):
                continue
            pieces = [crop_at(frames[i], track[i]) for i in window]
            if any(p is None for p in pieces):
                continue
            clips[f"o{offset:+.1f}"] = np.stack(pieces).astype(np.uint8)
        if not clips:
            continue
        name = f"{len(index):04d}_t{moment:.1f}"
        np.savez_compressed(OUT / f"{name}.npz", **clips)
        index.append(dict(name=name, t=round(moment, 2),
                          box=[float(v) for v in chosen]))
        json.dump(index, open(f"{CACHE}/index.json", "w"))
        if len(index) % 25 == 0:
            rate = (time.time() - started) / len(index)
            print(f"    {len(index)} candidates, {tried} moments, "
                  f"{rate:.1f}s each", flush=True)
    print(f"  {len(index)} candidates from {tried} moments in "
          f"{(time.time()-started)/60:.0f} min")


if __name__ == "__main__":
    main()
