"""Boxes worth putting on screen: the rim tracked, one ball, players on the court.

Four faults, each measured on the pass before it was fixed.

THE RIM WAS PINNED. It was drawn as one median box for the whole clip, on the
reasoning that a rim is bolted to the building. It is -- but the CAMERA PANS,
so a rim that is still in the world moves across the picture, and a box pinned
to one place drifts off it. Measured: the drawn rim's horizontal spread within
a clip was 0.0 px in every clip, while the real one crosses hundreds. The rim
is now tracked like anything else: linked across frames, smoothed, gaps
interpolated.

THE BALL JUMPED. Exactly one ball box was drawn per frame -- the highest
confidence one -- so on a frame where a head outscored the ball the box
teleported to the head and back. The fix is to stop choosing per frame: all
candidates are kept and a path through them is chosen by dynamic programming
that pays for confidence and is charged for sudden jumps.

THE BALL THEN SAT IN THE CROWD, which is what a smoothness prior buys when
nothing says where a ball can be. A shirt in row eight does not move, so the
path chooser was charged nothing for staying on it, while the real ball costs
30 px a frame. Measured on the pass this replaces: 17.8% of drawn ball frames
were more than 250 px from every player and 18.9% were above all of them, up
in the stands. A ball is now ANCHORED -- scored on its distance to the nearest
player or rim, and rejected outright beyond REJECT_PX or outside the floor's
own bounding box extended upward for shots and lobs.

PLAYERS INCLUDED THE CROWD. Up to 18 boxes on a floor holding ten, the extras
being people in the stands and on the bench. `candidates.stands_on_court`
answers this directly: a player's feet land on the floor and a spectator's do
not.

The detector is unchanged. Every fix here is about which of its outputs are
believed and how they are joined up over time -- which is why the pass is now
split in two: `clip_detect_raw.py` writes down everything the detector saw,
once, and this reads that file and decides, in seconds rather than hours.
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
#: ...and a rim is visible for a good share of the frames it appears in while
#: a sign the detector mistook for one is not. The share used to be measured
#: against the WHOLE clip, which dropped the rim in every clip where the camera
#: only swings to the basket for the shot -- that is most of them.
RIM_MIN_SHARE = 0.12
RIM_MAX_TRACKS = 2
#: ...and the ball lower still, because the path chooser -- not the threshold --
#: is what rejects a bad candidate.
BALL_CONF = 0.35
#: Ball candidates kept per frame for the path search.
BALL_TOPK = 6
#: Pixels a ball may move between frames before the jump costs more than it is
#: worth. A pass crosses the screen in well under a second at 30 fps.
BALL_STEP_PX = 110.0
#: What it costs to say "no ball in this frame" rather than accept a candidate.
BALL_MISS_COST = 0.55
#: WHETHER A BALL IS DRAWN AT ALL. It is again, and this is the measurement
#: that decided it both times. Thirteen frames of this game carry a ball
#: position located by eye before any system's claim was looked at, with a
#: 28 px tolerance (`scripts/eval_ball_choice.py`).
#:
#:   what shipped, with the old detector      right 1 of 13
#:   ...its best candidate, any chooser       6 of 13 was the ceiling
#:   the specialist trained on corrected
#:   labels, feeding this chooser             right 6 of 12 it draws
#:   ...with the confidence floor below       right 5 of the 7 it draws
#:
#: The floor is the operating point: the ball is drawn on about half the
#: frames and is right about seven times in ten when it is, rather than drawn
#: always and right one time in eight. Thirteen frames is a small sample and
#: the confidence interval is wide; it is the truth that exists.
DRAW_BALL = True
#: The detector's confidence on a ball is anti-correlated with being right --
#: measured on the thirteen hand-located frames, its most confident candidate
#: was correct twice while a correct one existed six times, usually at 0.1-0.3
#: with a confident decoy in the crowd. So confidence above this counts for
#: nothing extra and position decides.
BALL_CONF_CAP = 0.35
#: What a ball is measured against: every player, or only the one the detector
#: says is holding it.
BALL_ANCHOR = "player"
#: A ball within this of a player or the rim is free. A held ball is inside a
#: player's box; a dribbled one is just below it; a shot leaves from the hands.
BALL_FREE_PX = 110.0
#: Beyond that it is charged one unit of confidence per this many pixels, so a
#: 0.9-confidence blob 300 px from anybody loses to a 0.3 one in the play.
BALL_FAR_PX = 150.0
#: And beyond this it is not a candidate at all. The far corner of the floor
#: is about 900 px from the near one at 1280 wide, so a ball this far from
#: every player AND the rim is in the seats.
BALL_REJECT_PX = 420.0
#: How far above the floor's own bounding box a ball may still be in play --
#: the arc of a shot and the top of a lob live here, the scoreboard does not.
BALL_LOFT_SHARE = 0.55
#: A rim is a flat ellipse, and it is flat at every distance -- which is the
#: test, not its size. The 50 hand-located rims in this game run from 18 px
#: wide to 586, so an upper bound of 95 would have thrown away every close-up,
#: which is exactly the frame where the rim matters.
RIM_W_PX = (16.0, 420.0)
RIM_ASPECT = (1.5, 6.5)
#: A track survives this many detected frames of absence before it is retired.
#: At 15 Hz that is half a second -- long enough to cross behind another player.
TRACK_MAX_AGE = 15
#: Overlap that matches outright, before the distance gate is considered.
TRACK_MIN_IOU = 0.18
#: ...and a box whose centre is within this share of the predicted box's own
#: size matches too, which is what carries a track through a fast pan.
TRACK_GATE_SHARE = 1.4
#: Two boxes overlapping this much in one frame are one player twice.
DUPLICATE_IOU = 0.55
#: A rim track this short is noise...
RIM_MIN_FRAMES = 6


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


def dist_to_box(point, box):
    """Distance from a point to a box, 0 inside it."""
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return (dx * dx + dy * dy) ** 0.5


def anchor_penalty(box, anchors, court):
    """What a ball candidate costs for being where it is, or None to reject.

    A ball is near somebody -- held, dribbled, thrown at a receiver, or at the
    rim. Nothing in this project said so before, and the path chooser happily
    settled on a stationary orange thing in the crowd because standing still
    is free under a smoothness prior. This is the term that says otherwise.
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if court:
        x1, y1, x2, y2 = court
        loft = BALL_LOFT_SHARE * max(y2 - y1, 1.0)
        if not (x1 - 60 <= cx <= x2 + 60 and y1 - loft <= cy <= y2 + 40):
            return None
    if not anchors:
        return 0.0
    near = min(dist_to_box((cx, cy), a) for a in anchors)
    if near > BALL_REJECT_PX:
        return None
    return max(0.0, (near - BALL_FREE_PX) / BALL_FAR_PX)


def rim_shaped(box):
    """A rim is a small flat ellipse; a hoarding above the basket is not."""
    w, h = box[2] - box[0], box[3] - box[1]
    if h <= 0 or w <= 0:
        return False
    return (RIM_W_PX[0] <= w <= RIM_W_PX[1]
            and RIM_ASPECT[0] <= w / h <= RIM_ASPECT[1])


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


def track(per_frame, max_age=TRACK_MAX_AGE, min_iou=TRACK_MIN_IOU,
          gate=TRACK_GATE_SHARE):
    """{track: {frame: box}}, joined across gaps and across camera motion.

    Greedy overlap against the previous frame alone gave 127 identities per
    six-second clip for the ten players on the floor, 77% of them living under
    half a second. Two things caused that, and neither is the detector:

      THE CAMERA MOVES. On a pan a stationary player's box slides several of
      its own widths between frames, so the overlap with its own previous box
      is zero and it becomes a new person. The global shift is estimated here
      from the matches themselves -- the median displacement of everything
      that did match -- and applied before matching the rest.

      A PLAYER DISAPPEARS FOR A MOMENT. Behind another player, at the edge of
      frame, or simply missed. Matching only against the previous frame ends
      the track; a track here survives `max_age` frames of absence, moving at
      its last known velocity, and is picked up again when it reappears.

    Assignment is Hungarian over 1 - IoU rather than greedy, so one obvious
    match no longer steals the box a better one needed.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    tracks, live, nxt = defaultdict(dict), {}, 0
    shift = (0.0, 0.0)
    for f, boxes in enumerate(per_frame):
        predicted = {}
        for tid, state in live.items():
            age = f - state["frame"]
            vx, vy = state["vel"]
            dx = vx * age + shift[0] * age
            dy = vy * age + shift[1] * age
            b = state["box"]
            predicted[tid] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]

        matches = {}
        if predicted and boxes:
            ids = sorted(predicted)
            cost = np.ones((len(ids), len(boxes)), dtype=float)
            for i, tid in enumerate(ids):
                pb = predicted[tid]
                width = max(pb[2] - pb[0], 1.0)
                height = max(pb[3] - pb[1], 1.0)
                for j, box in enumerate(boxes):
                    overlap = iou(pb, box)
                    near = (abs((box[0] + box[2]) / 2 - (pb[0] + pb[2]) / 2) < gate * width
                            and abs((box[1] + box[3]) / 2 - (pb[1] + pb[3]) / 2) < gate * height)
                    sized = 0.5 <= (box[2] - box[0]) / width <= 2.0
                    if overlap >= min_iou or (near and sized):
                        cost[i, j] = 1.0 - max(overlap, 0.05)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1.0:
                    matches[ids[i]] = j

        moved = [((boxes[j][0] + boxes[j][2]) / 2 - (predicted[t][0] + predicted[t][2]) / 2,
                  (boxes[j][1] + boxes[j][3]) / 2 - (predicted[t][1] + predicted[t][3]) / 2)
                 for t, j in matches.items()]
        if len(moved) >= 3:
            shift = (float(np.median([m[0] for m in moved])),
                     float(np.median([m[1] for m in moved])))
        else:
            shift = (0.0, 0.0)

        taken = set(matches.values())
        for tid, j in matches.items():
            box = boxes[j]
            state = live[tid]
            gap = max(f - state["frame"], 1)
            centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            was = ((state["box"][0] + state["box"][2]) / 2,
                   (state["box"][1] + state["box"][3]) / 2)
            vel = ((centre[0] - was[0]) / gap, (centre[1] - was[1]) / gap)
            state["vel"] = (0.6 * state["vel"][0] + 0.4 * vel[0],
                            0.6 * state["vel"][1] + 0.4 * vel[1])
            state["box"], state["frame"] = box, f
            tracks[tid][f] = box
        for j, box in enumerate(boxes):
            if j in taken:
                continue
            live[nxt] = {"box": box, "frame": f, "vel": (0.0, 0.0)}
            tracks[nxt][f] = box
            nxt += 1
        live = {tid: s for tid, s in live.items() if f - s["frame"] <= max_age}
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


#: Half-width, in seconds, of the window the possession kernels scan around the
#: logged instant. Matches the 0.75 s either side the windows were trained on.
POSSESSION_HALF_S = 0.75


def kernel_subject(detected, player_tracks, middle, half, parameters):
    """The subject, from the possession kernels rather than the handler class.

    Scored on 157 held-out frames a person labelled: 59.2% against the handler
    class's 49.7%, paired McNemar p = 0.031. The handler class remains the
    default because it needs no weights and no pixels; this path needs both.

    A frame contributes a column of log-scores over the tracks, and the scan
    over time combines them with the learned cost of changing hands. A track the
    detector lost in a frame keeps its last box, exactly as in training, and
    `stale` is the penalty the model learned for that.
    """
    import numpy as np

    from courtvision.kernels.possession import temporal_reference

    window = [d for d in detected
              if middle - half <= d["f"] <= middle + half and "feat" in d]
    # Only tracks that are ON SCREEN at the logged instant can be the subject.
    # Without this the scan is free to name a track the window saw and the
    # middle frame did not, and the clip is then drawn with no subject at all --
    # which it did, on one clip in six.
    ids = sorted(t for t, by_frame in player_tracks.items()
                 if by_frame.get(middle) is not None)
    if not window or not ids:
        return None
    rows = []
    for d in window:
        here = np.full(len(ids) + 1, float(parameters["nobody"]))
        for k, tid in enumerate(ids):
            box = player_tracks[tid].get(d["f"])
            if box is None:
                here[k] = float(parameters["nobody"])
                continue
            # which detection in this frame is that track, if any
            best, score = None, 0.9
            for j, candidate in enumerate(d["p"]):
                overlap = iou(box, candidate)
                if overlap > score:
                    best, score = j, overlap
            if best is None or best >= len(d["feat"]):
                here[k] = float(parameters["nobody"]) + float(parameters["stale"])
            else:
                here[k] = d["feat"][best]
        rows.append(here)
    posterior, _, _ = temporal_reference(np.array(rows), float(parameters["stay_raw"]),
                                         target=-1, centre=len(rows) // 2)
    best = int(np.argmax(posterior[:len(ids)]))
    return ids[best]


def assemble(detected, count, fps, parameters=None):
    """Rows of drawn boxes for one clip, from what the detector saw.

    `detected` is one entry per DETECTED frame, in order:
    {"f": video frame number, "p": player boxes already filtered to the floor,
     "h": handler boxes, "r": rim boxes, "b": [(box, conf)], "court": box}
    Frames between detected ones are recovered by interpolation, which is why
    detecting every second frame costs nothing visible.
    """
    order = [d["f"] for d in detected]

    # The ball, anchored: a candidate is scored on confidence less what its
    # distance from the play costs, and dropped outright when it is nowhere
    # near it. Then one path through what is left.
    balls = []
    for d in detected:
        anchors = list(d["p"]) + [b for b in d["r"] if rim_shaped(b)]
        if BALL_ANCHOR == "handler" and d["h"]:
            anchors = list(d["h"])
        here = []
        for box, conf in d["b"]:
            if conf < BALL_CONF:
                continue
            penalty = anchor_penalty(box, anchors, d.get("court"))
            if penalty is None:
                continue
            here.append((box, min(conf, BALL_CONF_CAP) - penalty))
        here.sort(key=lambda e: -e[1])
        balls.append(here[:BALL_TOPK])
    ball = {order[i]: box for i, box in best_path(balls).items()}
    ball = interpolate(ball)

    players = track([d["p"] for d in detected])
    player_tracks = {}
    for t, by_index in players.items():
        if len(by_index) < MIN_TRACK_FRAMES:
            continue
        smoothed = smooth(by_index)
        player_tracks[t] = interpolate({order[i]: b for i, b in smoothed.items()})

    rims = track([[b for b in d["r"] if rim_shaped(b)] for d in detected])
    keep = [by_index for by_index in rims.values()
            if len(by_index) >= max(RIM_MIN_FRAMES, RIM_MIN_SHARE * len(detected))]
    keep.sort(key=len, reverse=True)
    rim_tracks = [interpolate({order[i]: b for i, b in smooth(by_index).items()})
                  for by_index in keep[:RIM_MAX_TRACKS]]

    # The subject: the track holding the ball around the logged instant, which
    # is the middle of the clip. The play-by-play says who did it; the handler
    # class says which box on screen he is.
    middle, half = count // 2, int(round(0.6 * fps))
    holding = defaultdict(int)
    for i, d in enumerate(detected):
        if not (middle - half <= d["f"] <= middle + half):
            continue
        for hb in d["h"]:
            best, score = None, 0.5
            for tid, by_frame in player_tracks.items():
                box = by_frame.get(d["f"])
                if box is not None and iou(hb, box) > score:
                    best, score = tid, iou(hb, box)
            if best is not None:
                holding[best] += 1
    subject = max(holding, key=holding.get) if holding else None
    if parameters is not None:
        chosen = kernel_subject(detected, player_tracks, middle,
                                int(round(POSSESSION_HALF_S * fps)), parameters)
        if chosen is not None:
            subject = chosen

    # Two tracks can settle on the same player -- one of them usually held
    # through a gap by prediction -- and then he is drawn twice. The longer
    # track keeps him.
    order = sorted(player_tracks, key=lambda t: -len(player_tracks[t]))
    rows = []
    for f in range(count):
        drawn, placed = [], []
        for tid in order:
            box = player_tracks[tid].get(f)
            if box is None:
                continue
            if any(iou(box, other) > DUPLICATE_IOU for other in placed):
                continue
            placed.append(box)
            drawn.append(["s" if tid == subject else "p"]
                         + [int(round(v)) for v in box])
        for by_frame in rim_tracks:
            if f in by_frame:
                drawn.append(["r"] + [int(round(v)) for v in by_frame[f]])
        if DRAW_BALL and f in ball:
            drawn.append(["b"] + [int(round(v)) for v in ball[f]])
        rows.append(drawn)
    return rows


def from_cache(path, limit=None):
    """Read clip_detect_raw.py's file into what assemble() wants."""
    cached = json.load(open(path))
    names = sorted(cached["clips"])
    if limit:
        names = names[:limit]
    out = {}
    for clip in names:
        detected = []
        for row in cached["clips"][clip]:
            people = [b for b in row["d"] if b[0] in ("p", "h")]
            on = row.get("on") or []
            keep = [b for i, b in enumerate(people)
                    if b[1] >= PLAYER_CONF and (i >= len(on) or on[i])]
            detected.append({
                "f": row["f"],
                "p": [b[2:] for b in keep],
                "h": [b[2:] for b in keep if b[0] == "h"],
                "r": [b[2:] for b in row["d"] if b[0] == "r" and b[1] >= RIM_CONF],
                "b": [(b[2:], b[1]) for b in row["d"] if b[0] == "b"],
                "court": row.get("court"),
            })
        out[clip] = detected
    return cached, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-cache", default=None,
                        help="clip_detect_raw.py output; no video pass, seconds not hours")
    parser.add_argument("--video", default=None)
    parser.add_argument("--index", action="append", default=[])
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--court-every", type=int, default=5,
                        help="recompute the court mask this often; it moves slowly")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--possession", default=None,
                        help="weights from train_possession_temporal.py. With "
                             "them the subject comes from the fused possession "
                             "kernels (59.2%% on held-out frames) instead of the "
                             "detector's handler class (49.7%%); without them "
                             "nothing changes, so the old path stays measurable.")
    args = parser.parse_args()

    parameters = None
    if args.possession:
        from courtvision.kernels.possession import load_parameters
        parameters = load_parameters(args.possession)
        print(f"  subject from the possession kernels: {args.possession}")

    if args.from_cache:
        cached, clips = from_cache(args.from_cache, args.limit)
        fps = cached.get("fps", 30.0)
        count = cached.get("frames_per_clip", int(round(args.duration * fps)))
        size = cached.get("source_size", [1280, 720])
        if parameters is not None and not any(
                "feat" in d for rows in clips.values() for d in rows):
            parser.error("--possession needs the pixels and the cache has none; "
                         "run without --from-cache, or add features to the cache")
        out = {clip: assemble(detected, count, fps, parameters)
               for clip, detected in clips.items()}
        report(out, count)
        return write(args.out, out, size, fps)

    if not (args.video and args.index):
        parser.error("--video and --index are required without --from-cache")

    import cv2
    import numpy as np
    from ultralytics import YOLO

    from courtvision.candidates import court_region, stands_on_court
    from courtvision.device import resolve_device
    from courtvision.kernels.possession import frame_logits

    clips = {}
    for path in args.index:
        for c in json.load(open(path))["clips"]:
            if not c.get("clip"):
                continue        # indexed, deliberately not cut
            clips.setdefault(c["clip"], round(float(c.get("start_s",
                             float(c["video_s"]) - 3.0)), 1))
    names = sorted(clips)
    if args.limit:
        names = names[:args.limit]

    model, device = YOLO(args.detector), resolve_device()
    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(round(args.duration * fps))

    out = {}
    for n, clip in enumerate(names):
        capture.set(cv2.CAP_PROP_POS_MSEC, clips[clip] * 1000)
        detected, region, court = [], None, None
        for f in range(count):
            ok, frame = capture.read()
            if not ok:
                break
            if f % args.court_every == 0:
                region = court_region(frame)
                if region is not None:
                    ys, xs = np.nonzero(region)
                    court = ([int(xs.min()), int(ys.min()),
                              int(xs.max()), int(ys.max())] if len(xs) else None)
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
                here_p = [b for b, keep in zip(here_p, on) if keep]
                here_h = [b for b in here_h if any(iou(b, p) > 0.9 for p in here_p)]
            here = {"f": f, "p": here_p, "h": here_h,
                    "r": here_r, "b": here_b, "court": court}
            if parameters is not None and here_p and abs(f - count // 2) <= \
                    round(POSSESSION_HALF_S * fps):
                # Only near the logged instant: the sweep is 64 samples per
                # player and there is no reason to pay for it on frames the
                # scan will never look at.
                top = max(here_b, key=lambda e: e[1], default=None)
                ball = np.zeros(3)
                if top is not None:
                    box, conf = top
                    ball = np.array([(box[0] + box[2]) / 2,
                                     (box[1] + box[3]) / 2, conf])
                here["feat"] = frame_logits(frame, np.array(here_p, float),
                                            ball, parameters)
            detected.append(here)
        out[clip] = assemble(detected, count, fps, parameters)
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(names)} clips", flush=True)
    capture.release()
    report(out, count)
    return write(args.out, out, [1280, 720], fps)


def report(out, count):
    """What the boxes look like, in the terms the last pass failed on."""
    import math

    frames = ball_frames = rim_frames = 0
    orphan = lofted = 0
    people = []
    for rows in out.values():
        for f in rows:
            frames += 1
            ps = [b for b in f if b[0] in ("p", "s")]
            rs = [b for b in f if b[0] == "r"]
            bs = [b for b in f if b[0] == "b"]
            people.append(len(ps))
            rim_frames += bool(rs)
            if not bs:
                continue
            ball_frames += 1
            c = ((bs[0][1] + bs[0][3]) / 2, (bs[0][2] + bs[0][4]) / 2)
            anchors = [b[1:] for b in ps + rs]
            if anchors:
                near = min(dist_to_box(c, a) for a in anchors)
                orphan += near > 250
                top = min(a[1] for a in anchors)
                lofted += c[1] < top - 60
    people.sort()
    print(f"  {len(out)} clips, {frames} frames")
    print(f"  ball drawn on {ball_frames / max(frames, 1):.1%} of frames; "
          f"of those {orphan / max(ball_frames, 1):.1%} more than 250 px from "
          f"anything and {lofted / max(ball_frames, 1):.1%} above everything")
    print(f"  rim drawn on {rim_frames / max(frames, 1):.1%} of frames")
    print(f"  players per frame p50 {people[len(people) // 2] if people else 0}, "
          f"p90 {people[int(len(people) * 0.9)] if people else 0}")


def write(target, out, size, fps):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"note": "per-frame boxes: players standing on the court, the rim tracked "
                       "rather than pinned, and one ball chosen as a path through the "
                       "candidates that is anchored to the play. "
                       "Rows are [code, x1, y1, x2, y2].",
               "source_size": size, "fps": fps, "clips": out},
              open(target, "w"), separators=(",", ":"))
    print(f"{len(out)} clips -> {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
