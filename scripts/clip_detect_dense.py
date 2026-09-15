"""Detect every frame of each clip, then stabilise, so the boxes stop jumping.

The overlay was built from the cached pass, which samples every 0.2 s. Played
back at 30 fps that is one box update every six frames: the box sits still
while the player keeps moving, then snaps. It reads as the detector being
wildly unstable when most of what you see is the sampling rate.

So: detect EVERY frame of every clip, then smooth.

    sampling      5 Hz  ->  30 Hz      six times the updates
    linking       greedy overlap across adjacent frames
    smoothing     a short centred moving average over each tracklet
    the rim       one box for the whole clip, because it does not move

THE RIM IS THE CLEAREST WIN. A rim is bolted to the building and the camera
barely moves inside six seconds, so per-frame rim boxes disagreeing with each
other is pure detector noise. Taking the median box over the clip replaces a
flickering rectangle with a still one, and drops the frames where it was missed
entirely.

THE BALL IS THE HARDEST and is not smoothed the same way. It genuinely moves
fast, so averaging it would lag the throw; gaps shorter than MAX_BALL_GAP are
filled by interpolation and longer ones are left empty, because a ball nobody
detected for half a second is not a ball whose position is known.

Jersey reads happen on this denser sampling too, every JERSEY_EVERY frames.
The measured problem with naming players was too few legible crops per player
per clip; this is the cheap way to get more of them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

#: Frames either side in the smoothing window. 2 means a 5-frame average, about
#: a sixth of a second -- enough to kill jitter, short enough to follow a cut.
SMOOTH = 2
#: Two boxes in consecutive frames are the same object above this overlap.
LINK_IOU = 0.4
#: A tracklet shorter than this is noise, not a player.
MIN_TRACK_FRAMES = 6
#: Ball gaps up to this many frames are interpolated; longer ones are left out.
MAX_BALL_GAP = 6
#: Read a jersey this often. Every frame is wasted -- consecutive frames show
#: the same pose -- and every third is still six times the old sampling.
JERSEY_EVERY = 3


def smooth_track(boxes, window=SMOOTH):
    """Centred moving average over a tracklet's boxes."""
    out = []
    for i in range(len(boxes)):
        lo, hi = max(0, i - window), min(len(boxes), i + window + 1)
        chunk = boxes[lo:hi]
        out.append([sum(b[k] for b in chunk) / len(chunk) for k in range(4)])
    return out


def median_box(boxes):
    """The box a static object occupies, robust to the frames it was missed in."""
    if not boxes:
        return None
    n = len(boxes)
    mid = n // 2
    return [sorted(b[k] for b in boxes)[mid] for k in range(4)]


def fill_gaps(series, length, max_gap=MAX_BALL_GAP):
    """Interpolate short gaps in {frame: box}; leave long ones empty."""
    if not series:
        return {}
    known = sorted(series)
    out = dict(series)
    for a, b in zip(known, known[1:]):
        gap = b - a
        if gap <= 1 or gap > max_gap:
            continue
        for k in range(1, gap):
            f = k / gap
            out[a + k] = [series[a][i] + (series[b][i] - series[a][i]) * f
                          for i in range(4)]
    return out


def link(per_frame, min_iou=LINK_IOU):
    """{tracklet: {frame: box}} from [[box, ...], ...]."""
    from identify_players import iou

    tracks: dict[int, dict[int, list]] = defaultdict(dict)
    previous: list[tuple[int, list]] = []
    next_id = 0
    for f, boxes in enumerate(per_frame):
        assigned, taken = [], set()
        for box in boxes:
            best, score = None, min_iou
            for tid, pbox in previous:
                if tid in taken:
                    continue
                v = iou(box, pbox)
                if v >= score:
                    best, score = tid, v
            if best is None:
                best, next_id = next_id, next_id + 1
            taken.add(best)
            tracks[best][f] = box
            assigned.append((best, box))
        previous = assigned
    return tracks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--roster", default=None)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--identity-out", default=None)
    args = parser.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO

    from courtvision.device import resolve_device
    from identify_players import name_tracklets, team_for_clusters

    roster = json.load(open(args.roster)) if args.roster else None
    reader = crop_fn = None
    if roster:
        from courtvision.digit_net import DigitReader
        from courtvision.jersey import jersey_crop
        reader, crop_fn = DigitReader("models/jersey_digits.pt"), jersey_crop

    clips = {}
    for path in args.index:
        for c in json.load(open(path))["clips"]:
            clips.setdefault(c["clip"], round(float(c.get("start_s",
                             float(c["video_s"]) - 3.0)), 1))
    names = sorted(clips)
    if args.limit:
        names = names[:args.limit]

    model, device = YOLO(args.detector), resolve_device()
    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames_per_clip = int(round(args.duration * fps))

    overlays, identities = {}, {}
    crops = reads = 0
    for n, clip in enumerate(names):
        start = clips[clip]
        capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        players, balls, rims, pictures = [], {}, [], []
        for f in range(frames_per_clip):
            ok, frame = capture.read()
            if not ok:
                break
            found = model.predict(frame, device=device, verbose=False,
                                  imgsz=args.imgsz, conf=args.conf)[0].boxes
            here = []
            if found is not None and len(found):
                for cls, conf, box in zip(found.cls.cpu().numpy(),
                                          found.conf.cpu().numpy(),
                                          found.xyxy.cpu().numpy()):
                    kind = model.names[int(cls)]
                    b = [float(v) for v in box]
                    if kind in ("player", "handler"):
                        here.append(b)
                    elif kind == "ball":
                        if f not in balls or conf > balls[f][1]:
                            balls[f] = (b, float(conf))
                    elif kind == "rim":
                        rims.append(b)
            players.append(here)
            pictures.append(frame if (roster and f % JERSEY_EVERY == 0) else None)

        tracks = link(players)
        tracks = {t: b for t, b in tracks.items() if len(b) >= MIN_TRACK_FRAMES}

        named = {}
        if roster and reader is not None:
            votes: dict[int, Counter] = defaultdict(Counter)
            cluster_reads: dict[int, Counter] = defaultdict(Counter)
            kit_of: dict[int, Counter] = defaultdict(Counter)
            from courtvision.tactics import torso_colours
            for f, frame in enumerate(pictures):
                if frame is None:
                    continue
                present = [(t, b[f]) for t, b in tracks.items() if f in b]
                if len(present) < 2:
                    continue
                colours = torso_colours(frame, np.array([b for _, b in present], float))
                axis = int(np.argmax(np.nanstd(colours, axis=0))) if len(present) > 2 else 0
                finite = colours[np.isfinite(colours).all(axis=1)]
                mid = float(np.median(finite[:, axis])) if len(finite) else 0.0
                for k, (tid, box) in enumerate(present):
                    crop = crop_fn(frame, box)
                    if crop is None or crop.size == 0:
                        continue
                    crops += 1
                    got = reader.read(crop)
                    if not got:
                        continue
                    reads += 1
                    number = got[0]
                    votes[tid][number] += 1
                    side = 1 if (np.isfinite(colours[k]).all()
                                 and colours[k][axis] >= mid) else 0
                    kit_of[tid][side] += 1
                    cluster_reads[side][number] += 1
            cluster_team = team_for_clusters(cluster_reads, roster)
            by_track = {t: cluster_team.get(c.most_common(1)[0][0])
                        for t, c in kit_of.items() if c}
            named = name_tracklets(votes, by_track, roster)
        identities[clip] = {str(t): who for t, who in named.items()}

        # smooth each tracklet, stabilise the rim, fill short ball gaps
        smoothed: dict[int, dict[int, list]] = {}
        for tid, by_frame in tracks.items():
            order = sorted(by_frame)
            vals = smooth_track([by_frame[f] for f in order])
            smoothed[tid] = dict(zip(order, vals))
        rim = median_box(rims)
        ball = fill_gaps({f: b for f, (b, _) in balls.items()}, frames_per_clip)

        rows = []
        for f in range(frames_per_clip):
            drawn = []
            for tid, by_frame in smoothed.items():
                if f in by_frame:
                    drawn.append(["p", tid, named.get(tid, "")]
                                 + [int(round(v)) for v in by_frame[f]])
            if f in ball:
                drawn.append(["b", -1, ""] + [int(round(v)) for v in ball[f]])
            if rim:
                drawn.append(["r", -2, ""] + [int(round(v)) for v in rim])
            if drawn:
                rows.append([round(f / fps, 3), drawn])
        overlays[clip] = rows
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(names)} clips"
                  + (f", {reads}/{crops} crops read" if crops else ""), flush=True)
    capture.release()

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"note": "per-frame boxes at the video's own rate, tracklet-linked and "
                       "smoothed; the rim is one box for the clip and short ball gaps "
                       "are interpolated. Rows are [code, tracklet, name, x1,y1,x2,y2].",
               "source_size": [1280, 720], "fps": fps,
               "clips": overlays}, open(target, "w"), separators=(",", ":"))
    print(f"{len(overlays)} clips at {fps:.0f} fps -> {target.stat().st_size/1e6:.1f} MB")
    if args.identity_out:
        Path(args.identity_out).write_text(json.dumps(identities, indent=1))
        named_clips = sum(1 for v in identities.values() if v)
        print(f"  {reads}/{crops} crops read ({reads/max(crops,1):.0%}); "
              f"{named_clips}/{len(identities)} clips named someone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
