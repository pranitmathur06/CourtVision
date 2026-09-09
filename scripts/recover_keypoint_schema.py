"""Recover what each of the 48 keypoints IS, from the annotations themselves.

The dataset ships no mapping from keypoint index to court landmark, and
guessing one would poison every homography built on it. But the court is a
rigid plane: any two annotated frames of it are related by a homography, so
the 849 annotations jointly over-determine the canonical layout.

Anchor on the frame with the most visible points, map every other frame into
it through the keypoints they share, and take the median position of each
index. That yields the schema in an arbitrary projective frame; a handful of
identifiable landmarks then fix it to real court feet.
"""
import glob, json
import numpy as np, cv2

SP = "outputs"
MIN_SHARED = 6


def load_all():
    out = []
    for split in ("train", "valid", "test"):
        for lab in sorted(glob.glob(f"data/labeled/court_keypoints/{split}/labels/*.txt")):
            v = open(lab).read().split()
            if len(v) < 149:
                continue
            kp = np.array(v[5:5+144], dtype=float).reshape(48, 3)
            seen = kp[:, 2] == 2
            pts = kp[:, :2]
            span = (np.hypot(*(pts[seen].max(0) - pts[seen].min(0)))
                    if seen.sum() >= 4 else 0.0)
            if seen.sum() >= 6 and span >= 0.2:
                out.append((seen, pts))
    return out


frames = load_all()
print(f"  {len(frames)} usable annotations")

# Anchor: the frame whose visible points are most numerous AND well spread.
score = [int(s.sum()) * float(np.hypot(*(p[s].max(0) - p[s].min(0))))
         for s, p in frames]
anchor = int(np.argmax(score))
seen0, pts0 = frames[anchor]
print(f"  anchor has {int(seen0.sum())} visible points")

# Grow the canonical set iteratively.
#
# A single anchor only admits frames sharing >=6 points WITH IT, which left 435
# of 849 out. Each admitted frame contributes indices the anchor never saw, so
# re-running against the accumulated set lets those frames join on the second
# pass. Repeat until nothing new is placed.
acc = {i: [] for i in range(48)}
for i in np.flatnonzero(seen0):
    acc[int(i)].append(pts0[i])

def canonical():
    # One observation is enough to seed: the anchor contributes exactly one
    # per index, and requiring three left the reference empty on round one.
    return {i: np.median(np.array(v), axis=0)
            for i, v in acc.items() if len(v) >= 1}

placed = set()
for round_no in range(1, 7):
    ref = canonical()
    added = 0
    for n, (seen, pts) in enumerate(frames):
        if n in placed:
            continue
        shared = [i for i in np.flatnonzero(seen) if i in ref]
        if len(shared) < MIN_SHARED:
            continue
        src = pts[shared].astype(np.float32)
        dst = np.array([ref[i] for i in shared], dtype=np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.01)
        if H is None or mask is None or int(mask.sum()) < MIN_SHARED:
            continue
        mine = np.flatnonzero(seen)
        hom = np.hstack([pts[mine], np.ones((len(mine), 1))]) @ H.T
        good = np.abs(hom[:, 2]) > 1e-9
        mapped = hom[good, :2] / hom[good, 2:3]
        for idx, xy in zip(mine[good], mapped):
            if -2.0 < xy[0] < 3.0 and -2.0 < xy[1] < 3.0:
                acc[int(idx)].append(xy)
        placed.add(n); added += 1
    ref_after = canonical()
    print(f"    round {round_no}: +{added} frames, {len(placed)}/{len(frames)} "
          f"placed, {len(ref_after)} indices located", flush=True)
    if added == 0:
        break

schema, counts, spread = {}, {}, {}
for i in range(48):
    if len(acc[i]) >= 8:
        arr = np.array(acc[i])
        schema[i] = np.median(arr, axis=0).tolist()
        counts[i] = len(arr)
        # Disagreement among observations of the same index: a landmark that
        # is genuinely one point should be tight.
        spread[i] = float(np.median(np.hypot(*(arr - np.median(arr, axis=0)).T)))
print(f"\n  located {len(schema)} of 48 indices")
if spread:
    tight = [i for i in schema if spread[i] < 0.02]
    print(f"  {len(tight)} of them agree to within 0.02 of the anchor frame")
    print(f"  scatter per index: p50 {np.median(list(spread.values())):.4f} "
          f"p90 {np.percentile(list(spread.values()),90):.4f}")
else:
    print("  none reached the support floor -- nothing to report")
json.dump({"schema": schema, "counts": counts, "spread": spread},
          open(f"{SP}/kp_schema.json", "w"))
