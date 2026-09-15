"""Boxes worth putting on screen: the rim tracked, one ball, players on the court.

Three faults, each measured on the previous pass before being fixed.

THE RIM WAS PINNED. It was drawn as one median box for the whole clip, on the
reasoning that a rim is bolted to the building. It is -- but the CAMERA PANS,
so a rim that is still in the world moves across the picture, and a box pinned
to one place drifts off it. Measured: the drawn rim's horizontal spread within
a clip was 0.0 px in every clip, while the real one crosses hundreds. The rim
is now tracked like anything else: linked across frames, smoothed, gaps
interpolated. Two rims are both kept when both are visible, because a wide
shot shows both baskets and picking one would blink between them.

THE BALL JUMPED. Exactly one ball box was drawn per frame -- the highest
confidence one -- so on a frame where a head outscored the ball the box
teleported to the head and back. That reads as several balls. The fix is to
stop choosing per frame: all candidates are kept, and a path through them is
chosen by dynamic programming that pays for confidence and is charged for
sudden jumps, with an explicit "no ball this frame" state so a genuinely
missing ball is a gap rather than a lurch to a spectator.

PLAYERS INCLUDED THE CROWD. Up to 18 boxes on a floor holding ten, the extras
being people in the stands and on the bench. `candidates.stands_on_court`
answers this directly: a player's feet land on the floor and a spectator's do
not.

The detector is unchanged. Every fix here is about which of its outputs are
believed and how they are joined up over time.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

#: Per-frame linking: two boxes this close are the same object.
LINK_IOU = 0.4
#: A track shorter than this is noise.
MIN_TRACK_FRAMES = 5
#: Centred moving average, in frames either side.
SMOOTH = 2
#: Gaps up to this many frames are interpolated, for anything tracked.
MAX_GAP = 10
#: Players must be this confident, and standing on the floor.
PLAYER_CONF = 0.40
#: Rims are small and often half-occluded, so they are believed lower than
#: players -- but not so low that the arena's red signage gets in. Measured:
#: at 0.25 the "rim" landed on the State Farm board above the basket.
RIM_CONF = 0.40
#: ...and a rim is visible for most of a clip while a sign the detector
#: mistook for one is not, so only the most persistent tracks are drawn, and
#: never more than the two baskets a wide shot can show.
RIM_MIN_SHARE = 0.35
RIM_MAX_TRACKS = 2
#: ...and the ball lower still, because the path chooser -- not the threshold --
#: is what rejects a bad candidate.
BALL_CONF = 0.10
#: Ball candidates kept per frame for the path search.
BALL_TOPK = 6
#: Pixels a ball may move between frames before the jump costs more than it is
#: worth. A pass crosses the screen in well under a second at 30 fps.
BALL_STEP_PX = 110.0
#: What it costs to say "no ball in this frame" rather than accept a candidate.
BALL_MISS_COST = 0.55


def best_path(per_frame, step_px=BALL_STEP_PX, miss_cost=BALL_MISS_COST):
    """One ball through the clip, by dynamic programming over the candidates.

    `per_frame` is [[(box, conf), ...], ...]. Returns {frame: box} for the
    frames the path actually claims a ball on.

    The score pays confidence for taking a candidate and is charged for the
    distance from the previous one, so a high-confidence head far from where
    the ball just was loses to a lower-confidence ball nearby. A null state
    carries `miss_cost`, which is what stops the path lurching across the
    picture to keep a box on screen every frame.
    """
    n = len(per_frame)
    if not n:
        return {}
    # state -1 is "no ball this frame"
    scores = [{} for _ in range(n)]
    back = [{} for _ in range(n)]
    for j, (_, conf) in enumerate(per_frame[0]):
        scores[0][j] = conf
    scores[0][-1] = -miss_cost
    for f in range(1, n):
        for j, (box, conf) in enumerate(per_frame[f]):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            best, best_from = None, None
            for i, prior in scores[f - 1].items():
                if i == -1:
                    move = 0.0                      # coming back from a gap is free
                else:
                    pbox = per_frame[f - 1][i][0]
                    px, py = (pbox[0] + pbox[2]) / 2, (pbox[1] + pbox[3]) / 2
                    move = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / step_px
                value = prior + conf - move
                if best is None or value > best:
                    best, best_from = value, i
            scores[f][j] = best
            back[f][j] = best_from
        best, best_from = None, None
        for i, prior in scores[f - 1].items():
            value = prior - miss_cost
            if best is None or value > best:
                best, best_from = value, i
        scores[f][-1] = best
        back[f][-1] = best_from

    state = max(scores[n - 1], key=lambda k: scores[n - 1][k])
    picked = {}
    for f in range(n - 1, -1, -1):
        if state != -1:
            picked[f] = per_frame[f][state][0]
        state = back[f].get(state, -1) if f else -1
    return picked


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    ub = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


def link(per_frame, min_iou=LINK_IOU):
    """{track: {frame: box}} by greedy overlap against the previous frame."""
    tracks: dict[int, dict[int, list]] = defaultdict(dict)
    previous: list[tuple[int, list]] = []
    nxt = 0
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
                best, nxt = nxt, nxt + 1
            taken.add(best)
            tracks[best][f] = box
            assigned.append((best, box))
        previous = assigned
    return tracks


def smooth(by_frame, window=SMOOTH):
    """Centred moving average over a track, keyed by frame."""
    order = sorted(by_frame)
    out = {}
    for i, f in enumerate(order):
        lo, hi = max(0, i - window), min(len(order), i + window + 1)
        chunk = [by_frame[order[k]] for k in range(lo, hi)]
        out[f] = [sum(b[k] for b in chunk) / len(chunk) for k in range(4)]
    return out


def interpolate(by_frame, max_gap=MAX_GAP):
    """Fill short gaps in a track so a box does not blink."""
    order = sorted(by_frame)
    out = dict(by_frame)
    for a, b in zip(order, order[1:]):
        gap = b - a
        if gap <= 1 or gap > max_gap:
            continue
        for k in range(1, gap):
            t = k / gap
            out[a + k] = [by_frame[a][i] + (by_frame[b][i] - by_frame[a][i]) * t
                          for i in range(4)]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--court-every", type=int, default=5,
                        help="recompute the court mask this often; it moves slowly")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO

    from courtvision.candidates import court_region, stands_on_court
    from courtvision.device import resolve_device

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
    count = int(round(args.duration * fps))

    out, dropped_crowd, kept_players = {}, 0, 0
    for n, clip in enumerate(names):
        capture.set(cv2.CAP_PROP_POS_MSEC, clips[clip] * 1000)
        players, rims, balls, handlers = [], [], [], []
        region = None
        for f in range(count):
            ok, frame = capture.read()
            if not ok:
                break
            if f % args.court_every == 0:
                region = court_region(frame)
            found = model.predict(frame, device=device, verbose=False,
                                  imgsz=args.imgsz, conf=BALL_CONF)[0].boxes
            here_p, here_r, here_b, here_h = [], [], [], []
            if found is not None and len(found):
                for cls, conf, box in zip(found.cls.cpu().numpy(),
                                          found.conf.cpu().numpy(),
                                          found.xyxy.cpu().numpy()):
                    kind = model.names[int(cls)]
                    b = [float(v) for v in box]
                    c = float(conf)
                    if kind in ("player", "handler") and c >= PLAYER_CONF:
                        here_p.append(b)
                        if kind == "handler":
                            here_h.append(b)
                    elif kind == "rim" and c >= RIM_CONF:
                        here_r.append(b)
                    elif kind == "ball":
                        here_b.append((b, c))
            # A spectator's feet are not on the floor.
            if here_p and region is not None:
                on = stands_on_court(region, np.array(here_p, float))
                dropped_crowd += int((~on).sum())
                kept_players += int(on.sum())
                here_p = [b for b, keep in zip(here_p, on) if keep]
                here_h = [b for b in here_h if any(iou(b, p) > 0.9 for p in here_p)]
            players.append(here_p)
            rims.append(here_r)
            here_b.sort(key=lambda e: -e[1])
            balls.append(here_b[:BALL_TOPK])
            handlers.append(here_h)

        ball = best_path(balls)
        player_tracks = {t: interpolate(smooth(b))
                         for t, b in link(players).items() if len(b) >= MIN_TRACK_FRAMES}
        rim_found = [b for b in link(rims).values() if len(b) >= MIN_TRACK_FRAMES]
        rim_found.sort(key=len, reverse=True)
        rim_tracks = {i: interpolate(smooth(b)) for i, b in enumerate(rim_found)
                      if len(b) >= RIM_MIN_SHARE * count}
        rim_tracks = dict(list(rim_tracks.items())[:RIM_MAX_TRACKS])

        # the subject: the track holding the ball around the logged instant
        middle, half = count // 2, int(round(0.6 * fps))
        holding = defaultdict(int)
        for f in range(max(middle - half, 0), min(middle + half + 1, count)):
            for hb in handlers[f] if f < len(handlers) else []:
                best, score = None, 0.5
                for tid, by_frame in player_tracks.items():
                    if f in by_frame and iou(hb, by_frame[f]) > score:
                        best, score = tid, iou(hb, by_frame[f])
                if best is not None:
                    holding[best] += 1
        subject = max(holding, key=holding.get) if holding else None

        rows = []
        for f in range(count):
            drawn = []
            for tid, by_frame in player_tracks.items():
                if f in by_frame:
                    drawn.append(["s" if tid == subject else "p"]
                                 + [int(round(v)) for v in by_frame[f]])
            for by_frame in rim_tracks.values():
                if f in by_frame:
                    drawn.append(["r"] + [int(round(v)) for v in by_frame[f]])
            if f in ball:
                drawn.append(["b"] + [int(round(v)) for v in ball[f]])
            rows.append(drawn)
        out[clip] = rows
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(names)} clips", flush=True)
    capture.release()

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"note": "per-frame boxes: players standing on the court, the rim tracked "
                       "rather than pinned, and one ball chosen as a path through the "
                       "candidates. Rows are [code, x1, y1, x2, y2].",
               "source_size": [1280, 720], "fps": fps, "clips": out},
              open(target, "w"), separators=(",", ":"))
    print(f"{len(out)} clips -> {target.stat().st_size/1e6:.1f} MB")
    print(f"  players kept {kept_players}, dropped as off-court {dropped_crowd} "
          f"({dropped_crowd/max(kept_players+dropped_crowd,1):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
