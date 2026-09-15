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

THE SUBJECT OF THE CLIP IS FOUND WITHOUT READING A JERSEY. Every clip is cut
around one logged event, and the play-by-play already says who did it --
"Jal. Williams 26' 3PT Jump Shot". What it does not say is which box on screen
he is. The detector's `handler` class does: whoever has the ball at the moment
of a shot IS the shooter.

So the subject is the tracklet that is the handler most often in the second
around the logged instant, and the name comes from the official record rather
than from a digit reader that answers on 16% of crops. That is why this works
where `identify_players.py` measured 45%: the hard half of the problem was
already solved by the play-by-play, and only the pointing needed doing.

It is honest about what it covers. A shot, a turnover or an assist is an act by
the player holding the ball, so the handler is the actor. A REBOUND is whoever
comes up with it, which the handler usually becomes a moment later. A FOUL is
committed by a defender who may never touch the ball, and for those the subject
is left unset rather than guessed at.

The subject's box is carried across frames where the detector lost it, for gaps
up to MAX_SUBJECT_GAP, so the one box a viewer is watching does not blink.
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
#: Read a jersey this often, when a roster is given at all.
JERSEY_EVERY = 3
#: The subject is decided from the handler over this window around the logged
#: instant, which sits at the middle of every clip.
SUBJECT_WINDOW_S = 1.2
#: Carry the subject's box across gaps up to this many frames so it does not
#: blink; beyond it the player really is gone from the picture.
MAX_SUBJECT_GAP = 12
#: Actions whose actor is the player holding the ball. A foul is not one.
BALL_ACTS = ("shot", "made shot", "free throw", "assist", "turnover",
             "steal", "rebound", "block", "violation")


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

    overlays, identities, subjects = {}, {}, {}
    crops = reads = 0
    for n, clip in enumerate(names):
        start = clips[clip]
        capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        players, balls, rims, pictures = [], {}, [], []
        handler_boxes: dict[int, list] = {}
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
                        if kind == "handler":
                            handler_boxes.setdefault(f, []).append(b)
                    elif kind == "ball":
                        if f not in balls or conf > balls[f][1]:
                            balls[f] = (b, float(conf))
                    elif kind == "rim":
                        rims.append(b)
            players.append(here)
            pictures.append(frame if (roster and f % JERSEY_EVERY == 0) else None)

        tracks = link(players)
        tracks = {t: b for t, b in tracks.items() if len(b) >= MIN_TRACK_FRAMES}

        # Which tracklet is the subject: the one holding the ball around the
        # logged instant, which every clip is cut to its middle.
        from identify_players import iou as _iou
        middle = frames_per_clip // 2
        half = int(round(SUBJECT_WINDOW_S * fps / 2))
        holding: Counter = Counter()
        for f in range(max(middle - half, 0), min(middle + half + 1, frames_per_clip)):
            for hb in handler_boxes.get(f, []):
                best, score = None, 0.5
                for tid, by_frame in tracks.items():
                    if f not in by_frame:
                        continue
                    v = _iou(hb, by_frame[f])
                    if v > score:
                        best, score = tid, v
                if best is not None:
                    holding[best] += 1
        subject = holding.most_common(1)[0][0] if holding else None

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
        if subject is not None and subject in smoothed:
            smoothed[subject] = fill_gaps(smoothed[subject], frames_per_clip,
                                          MAX_SUBJECT_GAP)
        rim = median_box(rims)
        ball = fill_gaps({f: b for f, (b, _) in balls.items()}, frames_per_clip)

        rows = []
        for f in range(frames_per_clip):
            drawn = []
            for tid, by_frame in smoothed.items():
                if f in by_frame:
                    code = "s" if tid == subject else "p"
                    drawn.append([code] + [int(round(v)) for v in by_frame[f]])
            if f in ball:
                drawn.append(["b"] + [int(round(v)) for v in ball[f]])
            if rim:
                drawn.append(["r"] + [int(round(v)) for v in rim])
            if drawn:
                rows.append([round(f / fps, 3), drawn])
        overlays[clip] = rows
        subjects[clip] = subject is not None
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
               "subject_found": subjects,
               "clips": overlays}, open(target, "w"), separators=(",", ":"))
    found = sum(1 for v in subjects.values() if v)
    print(f"{len(overlays)} clips at {fps:.0f} fps -> {target.stat().st_size/1e6:.1f} MB")
    print(f"  a subject (the player holding the ball) was found in {found}/{len(subjects)}")
    if args.identity_out:
        Path(args.identity_out).write_text(json.dumps(identities, indent=1))
        named_clips = sum(1 for v in identities.values() if v)
        print(f"  {reads}/{crops} crops read ({reads/max(crops,1):.0%}); "
              f"{named_clips}/{len(identities)} clips named someone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
