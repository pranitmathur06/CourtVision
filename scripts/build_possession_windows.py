"""Cut a window of frames around every labelled frame, and link the players.

The per-frame operator in `kernels/possession.py` answers "who has the ball in
this picture" and tops out near the shipped baseline. Possession is not a
per-frame quantity: a player holds the ball for seconds at a time and hand-offs
are rare, so the frames either side of a labelled one are evidence about it.
This script assembles that evidence once, because the expensive parts -- video
decode, detection, ball detection -- must not run inside a training loop.

WHAT A WINDOW IS. WINDOW frames STEP_S apart, centred on the labelled frame.
The centre frame's boxes are the ones the labeller actually saw, in the order
they saw them, so the target index means what it meant before. The other frames'
boxes come from the detection cache the labelling pages were built from
(`outputs/clip_detections_*.json`, every second frame at 1280), which is why no
detector runs here.

LINKING, AND WHAT IT COSTS. Tracks are grown outward from the centre one step at
a time by best IoU, so a box only ever has to match a box 0.25 s old. A track
with no match in a frame keeps its last known box and is marked not-present; the
pixel sweep then reads the floor where the player used to be, which is weak
evidence rather than absent evidence, and the `present` mask is stored so the
model can be told to ignore it.

WHAT IS STORED. Per player per frame, the chromaticity patch of his box grown by
the surround radius -- not the whole frame, which would be 8x the bytes -- in
fp16, with the box in patch coordinates. Sampling inside that patch is
arithmetically identical to sampling the full frame, so the windows change the
amount of evidence and not the operator.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

#: Frames in a window, and the gap between them. 7 x 0.25 s reaches 0.75 s
#: either side -- long enough to cross the frames where the ball is hidden by a
#: body, short enough that a player moves less than his own width between them.
WINDOW = 7
STEP_S = 0.25
#: A box matches the track it grew from above this IoU.
LINK_IOU = 0.25
#: Margin around a stored box, as a fraction of its height, so the surround ring
#: of a sample on the box edge is still inside the patch.
PATCH_MARGIN = 0.20
PLAYER_CONF = 0.35
EPS = 1e-6

#: Which broadcast each labelled game is, from `data/games.json`. Was a
#: hardcoded dict here and in two labelling pages; see `courtvision.games`.
def _games():
    from courtvision.games import registry
    return {g.label: (str(g.video), str(g.clip_index), str(g.clip_detections))
            for g in registry().values()}


GAMES = _games()
ROUNDS = (("data/labels/possession_labels.json", "data/labeling/possession"),
          ("data/labels/handler_labels.json", "data/labeling/handler"))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def load_rows():
    """Every labelled frame with an answer, and the boxes the labeller saw."""
    rows = []
    for labels, folder in ROUNDS:
        if not Path(labels).exists():
            continue
        manifest = {}
        path = Path(folder) / "manifest.json"
        if path.exists():
            manifest = {m["file"]: m.get("boxes", [])
                        for m in json.load(open(path))["frames"]}
        for row in json.load(open(labels))["frames"]:
            verdict = row.get("handler_verdict") or row.get("verdict")
            if verdict not in ("box", "missing", "nobody"):
                continue
            boxes = row.get("boxes") or manifest.get(row["file"], [])
            if not boxes:
                continue
            rows.append({"file": row["file"], "game": row["game"], "t": float(row["t"]),
                         "boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                         "verdict": verdict, "handler_box": row.get("handler_box"),
                         "split": "eval" if row["pick"] == "random" else "train",
                         "image": str(Path(folder) / "images" / row["file"])})
    return rows


def detections_by_time(cache_path, index_path):
    """Video time -> the cached detections nearest it, for one game."""
    cache = json.load(open(cache_path))
    fps = float(cache["fps"])
    starts = {c["clip"]: float(c.get("start_s", float(c["video_s"]) - 3.0))
              for c in json.load(open(index_path))["clips"] if c.get("clip")}
    table = {}
    for clip, frames in cache["clips"].items():
        if clip not in starts:
            continue
        for row in frames:
            when = starts[clip] + row["f"] / fps
            table[round(when, 2)] = row["d"]
    times = np.array(sorted(table))
    return table, times


def nearest_detections(table, times, when, tolerance=0.12):
    if not len(times):
        return None
    i = int(np.searchsorted(times, when))
    best = min((abs(times[j] - when), j) for j in (i - 1, i, i + 1)
               if 0 <= j < len(times))
    return table[round(float(times[best[1]]), 2)] if best[0] <= tolerance else None


def target_index(row):
    """Which candidate the labeller pointed at; len(boxes) means 'nobody'."""
    if row["verdict"] == "nobody":
        return len(row["boxes"])
    if row["verdict"] != "box" or not row["handler_box"]:
        return -1
    overlaps = [iou(b, row["handler_box"]) for b in row["boxes"]]
    best = int(np.argmax(overlaps))
    return best if overlaps[best] >= 0.5 else -1


def chromaticity(patch):
    patch = patch.astype(np.float32)
    blue, green, red = patch[..., 0], patch[..., 1], patch[..., 2]
    return (red - 0.5 * (green + blue)) / (red + green + blue + EPS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory for the .npz windows")
    parser.add_argument("--ball", default="checkpoints/ball_v2/best.pt")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--step", type=float, default=STEP_S)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.device import resolve_device
    from courtvision.kernels.possession import SURROUND

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if args.limit:
        rows = rows[:args.limit]
    by_game = defaultdict(list)
    for row in rows:
        by_game[row["game"]].append(row)

    device = resolve_device()
    ball_model = YOLO(args.ball)
    half = args.window // 2
    written = skipped = 0

    for game, group in sorted(by_game.items()):
        if game not in GAMES:
            print(f"  {game}: no video mapped, skipping {len(group)} frames")
            skipped += len(group)
            continue
        video_path, index_path, cache_path = GAMES[game]
        if not all(Path(p).exists() for p in (video_path, index_path, cache_path)):
            print(f"  {game}: missing video or cache, skipping {len(group)} frames")
            skipped += len(group)
            continue
        table, times = detections_by_time(cache_path, index_path)
        capture = cv2.VideoCapture(video_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 29.97
        print(f"  {game}: {len(group)} frames, {len(times)} cached detections, "
              f"{fps:.2f} fps")

        for done, row in enumerate(sorted(group, key=lambda r: r["t"])):
            target = target_index(row)
            centre = row["boxes"]
            tracks = len(centre)
            offsets = [(k - half) * args.step for k in range(args.window)]

            frames, boxes_t, present_t, balls = [], [], [], []
            live = centre.copy()                       # the tracks, most recent boxes
            ok = True
            # Grow outward from the centre so a match is only ever one step old.
            order = [half] + [half + s * d for s in range(1, half + 1) for d in (-1, 1)]
            store = {}
            state = {half: centre.copy()}
            for slot in order:
                when = row["t"] + offsets[slot]
                if slot == half:
                    # The centre frame is the labelled JPEG itself, not a seek.
                    # Frame times were written as start + f/30.0 and rounded to
                    # 0.1 s, so seeking by time lands one frame out about half
                    # the time -- 33 ms in which a ball crosses several of its
                    # own widths, which moved the swept peak by 0.26. The
                    # neighbours are 0.25 s away and a frame either side of them
                    # changes nothing, so only the centre has to be exact.
                    frame = cv2.imread(row["image"])
                    grabbed = frame is not None
                else:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, max(int(round(when * fps)), 0))
                    grabbed, frame = capture.read()
                if not grabbed or frame is None:
                    ok = False
                    break
                if slot == half:
                    linked, present = centre.copy(), np.ones(tracks, dtype=bool)
                else:
                    step = -1 if slot < half else 1
                    previous = state[slot - step]
                    found = nearest_detections(table, times, when)
                    candidates = [b[2:6] for b in (found or [])
                                  if b[0] in ("p", "h") and b[1] >= PLAYER_CONF]
                    linked = previous.copy()
                    present = np.zeros(tracks, dtype=bool)
                    taken = set()
                    pairs = sorted(((iou(previous[i], c), i, j)
                                    for i in range(tracks)
                                    for j, c in enumerate(candidates)),
                                   reverse=True)
                    for score, i, j in pairs:
                        if score < LINK_IOU or present[i] or j in taken:
                            continue
                        linked[i] = candidates[j]
                        present[i] = True
                        taken.add(j)
                state[slot] = linked
                seen = ball_model.predict(frame, device=device, verbose=False,
                                          imgsz=1280, conf=0.05)[0].boxes
                ball = np.zeros(3, dtype=np.float32)
                if seen is not None and len(seen):
                    conf, box = max(zip(seen.conf.cpu().numpy(),
                                        seen.xyxy.cpu().numpy()), key=lambda e: e[0])
                    ball = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                                     float(conf)], dtype=np.float32)
                store[slot] = (frame, linked, present, ball)
            if not ok:
                skipped += 1
                continue

            # ---- cut the patches, one per player per frame ------------------
            height, width = store[half][0].shape[:2]
            patch_h = patch_w = 0
            geometry = {}
            for slot in range(args.window):
                _, linked, _, _ = store[slot]
                for i, box in enumerate(linked):
                    grow = PATCH_MARGIN * max(box[3] - box[1], 1.0) \
                        + SURROUND * max(box[3] - box[1], 1.0)
                    x0 = int(math.floor(box[0] - grow))
                    y0 = int(math.floor(box[1] - grow))
                    x1 = int(math.ceil(box[2] + grow))
                    y1 = int(math.ceil(box[3] + grow))
                    geometry[(slot, i)] = (x0, y0, x1, y1)
                    patch_h = max(patch_h, y1 - y0)
                    patch_w = max(patch_w, x1 - x0)

            patches = np.zeros((args.window, tracks, patch_h, patch_w), dtype=np.float16)
            local = np.zeros((args.window, tracks, 4), dtype=np.float32)
            present_all = np.zeros((args.window, tracks), dtype=bool)
            ball_all = np.zeros((args.window, 3), dtype=np.float32)
            for slot in range(args.window):
                frame, linked, present, ball = store[slot]
                chroma = chromaticity(frame)
                present_all[slot] = present
                ball_all[slot] = ball
                for i, box in enumerate(linked):
                    x0, y0, x1, y1 = geometry[(slot, i)]
                    # Take the patch by CLAMPED index rather than by slice, so
                    # the part of it outside the frame repeats the edge pixel.
                    # That is exactly what `_sample_bilinear` does to a sample
                    # off the edge of the full image; zero-filling instead put
                    # a 0.04 error on the swept peak of every player standing
                    # against the touchline, which is not a rounding error but
                    # a different picture.
                    rows_ = np.clip(np.arange(y0, y1), 0, height - 1)
                    cols_ = np.clip(np.arange(x0, x1), 0, width - 1)
                    cut = chroma[np.ix_(rows_, cols_)]
                    patches[slot, i, :y1 - y0, :x1 - x0] = cut.astype(np.float16)
                    local[slot, i] = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
            np.savez_compressed(
                out / f"{row['file'].replace('.jpg', '')}.npz",
                patches=patches, boxes=local, present=present_all, ball=ball_all,
                origin=np.array([[geometry[(s, i)][:2] for i in range(tracks)]
                                 for s in range(args.window)], dtype=np.float32),
                target=np.int32(target), centre=np.int32(half),
                split=row["split"], game=game, t=np.float32(row["t"]),
                verdict=row["verdict"], name=row["file"])
            written += 1
            if done % 25 == 0:
                print(f"    {done + 1}/{len(group)}  {row['file']}  "
                      f"{tracks} tracks  patch {patch_h}x{patch_w}")
        capture.release()

    print(f"\n  {written} windows written to {out} ({skipped} frames skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
