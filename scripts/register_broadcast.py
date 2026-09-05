"""Register a whole broadcast: sparse solved anchors, dense cheap propagation.

Solving every frame is not merely slow, it is infeasible. The six-parameter
camera search measured ~41 s per frame on this machine; at 93,553 frames that
is over a month for one game. Even constrained to pan/tilt/zoom it stays far
too slow to run densely.

The structure of the problem removes the need to. A broadcast is one fixed rig,
so:

  1. the rig's POSITION is solved once for the whole game, not per frame;
  2. anchors are solved sparsely, on a stride, as a 3-parameter problem;
  3. every frame between anchors is reached by ORB propagation, which costs
     milliseconds because it matches two images rather than searching a space.

Cost becomes (frames / stride) searches plus one cheap hop per frame, and the
accuracy of the dense frames is checked against court lines they never used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from courtvision.court_lines import (homography_from_camera,  # noqa: E402
                                     line_distance_map, score_homography,
                                     search_camera)
from courtvision.court_tracking import (estimate_rig, has_court,  # noqa: E402
                                        propagate, rig_bounds, wood_fraction)
from courtvision.shot_boundaries import cut_frames  # noqa: E402

MIN_SCORE = 0.30
# A propagated frame is held to a lower bar than a solved one: it only has to
# remain consistent with the lines, not to have been found from them.
VERIFY_SCORE = 0.20


def read_frames(video: str, start_s: float, duration_s: float, stride: int):
    import cv2

    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    wanted = int(duration_s * fps / stride)
    images = []
    while len(images) < wanted:
        ok, image = capture.read()
        if not ok:
            break
        for _ in range(stride - 1):
            capture.grab()
        images.append(image)
    capture.release()
    return images, fps


def detect_people(images):
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    boxes = {}
    for index, image in enumerate(images):
        result = model(image, verbose=False, conf=0.25, classes=[0])[0]
        if len(result.boxes):
            boxes[index] = result.boxes.xyxy.cpu().numpy()
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/raw_clips/fullgame.mp4")
    parser.add_argument("--start", type=float, default=1500.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--stride", type=int, default=10,
                        help="video frames between samples")
    parser.add_argument("--anchor-every", type=int, default=12,
                        help="samples between SOLVED anchors")
    parser.add_argument("--rig", help="JSON with a cached rig, to skip stage 1")
    parser.add_argument("--out", default="outputs/registration.json")
    args = parser.parse_args()

    images, fps = read_frames(args.video, args.start, args.duration, args.stride)
    if not images:
        print("FAIL — no frames read")
        return 1
    # Refuse frames with no floor BEFORE spending a search on them. Roughly 40%
    # of a broadcast is crowd, bench or replay, and the line score cannot reject
    # those on its own -- a spectator's face scored 0.302 against a 0.30 gate.
    court = [i for i, image in enumerate(images) if has_court(image)]
    print(f"  {len(images)} samples @ stride {args.stride} from {args.start}s")
    print(f"  {len(court)}/{len(images)} = {len(court)/len(images):.1%} show court"
          f" (median wood {np.median([wood_fraction(im) for im in images]):.3f})")
    if not court:
        print("FAIL — no frame in this window shows the court")
        return 1
    boxes = detect_people(images)
    # cut_frames is not usable here. On this window it fired on nothing, and the
    # thumbnail difference measured AT true camera changes (median 0.028) was
    # LOWER than during ordinary play (0.031) -- at a third of a second between
    # samples a cut looks exactly like a fast pan. What IS exact is the court
    # gate: a court -> non-court transition is a camera change by definition.
    court_set = set(court)
    cuts = [i for i in range(1, len(images))
            if (i in court_set) != (i - 1 in court_set)]
    print(f"  {len(cuts)} camera changes from court transitions"
          f" ({len(cut_frames(images))} from thumbnail differences)")

    rig = None
    if args.rig and Path(args.rig).exists():
        rig = np.array(json.loads(Path(args.rig).read_text())["rig"])
        print(f"  rig from cache: x={rig[0]:.1f} y={rig[1]:.1f} z={rig[2]:.1f}")

    anchors = court[::args.anchor_every]
    if rig is None:
        start = time.time()
        params, scores = [], []
        for index in anchors:
            found, score = search_camera(
                images[index], seed=0, max_iterations=250, bounds=None,
                rim_px=None, exclude_boxes=boxes.get(index))
            if found is not None:
                params.append(found)
                scores.append(score)
        rig = estimate_rig(params, scores)
        if rig is None:
            print("FAIL — could not fix the rig")
            return 1
        print(f"  rig solved from {len(params)} frames in {time.time()-start:.0f}s:"
              f" x={rig[0]:.1f} y={rig[1]:.1f} z={rig[2]:.1f}")

    bounds = rig_bounds(rig)
    start = time.time()
    solved: dict[int, np.ndarray] = {}
    for index in anchors:
        found, score = search_camera(
            images[index], seed=0, max_iterations=120, bounds=bounds,
            rim_px=None, exclude_boxes=boxes.get(index))
        if found is None or score < MIN_SCORE:
            continue
        matrix = homography_from_camera(found, images[index].shape[:2])
        if matrix is not None:
            solved[index] = np.linalg.inv(matrix)
    anchor_rate = len(solved) / max(len(anchors), 1)
    print(f"  anchors solved: {len(solved)}/{len(anchors)} = {anchor_rate:.1%}"
          f"  ({(time.time()-start)/max(len(anchors),1):.1f}s each)")

    # Verify every hop against the frame's OWN line evidence, which propagation
    # never uses -- it matches background texture. A chain that has drifted onto
    # the wrong court stops here rather than filling the gap with fiction.
    maps: dict[int, np.ndarray] = {}

    def line_map(index: int) -> np.ndarray:
        if index not in maps:
            maps[index] = line_distance_map(images[index], boxes.get(index))
        return maps[index]

    def verify(index: int, matrix: np.ndarray) -> bool:
        if index not in court_set:
            return False
        try:
            court_to_image = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return False
        return score_homography(court_to_image, line_map(index)) >= VERIFY_SCORE

    start = time.time()
    full = propagate(images, solved, cuts=cuts, boxes=boxes, verify=verify)
    # A propagated frame is only meaningful if it shows court; chains can run a
    # few hops into a close-up before the matcher gives up.
    full = {i: m for i, m in full.items() if i in set(court)}
    coverage = len(full) / len(court)
    print(f"  after propagation: {len(full)}/{len(court)} court frames"
          f" = {coverage:.1%}"
          f"  ({(time.time()-start)/max(len(images),1)*1000:.0f} ms/frame)")

    groups = {"anchor": [], "propagated": []}
    for index, image in enumerate(images):
        matrix = full.get(index)
        if matrix is None or not np.isfinite(matrix).all():
            continue
        try:
            court_to_image = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            continue
        score = score_homography(court_to_image,
                                 line_distance_map(image, boxes.get(index)))
        groups["anchor" if index in solved else "propagated"].append(float(score))
    print(f"\n  line evidence, which propagation never used")
    print(f"  {'group':<12}{'n':>5}{'median':>9}{'>=0.30':>9}")
    for name, values in groups.items():
        if not values:
            print(f"  {name:<12}{0:>5}")
            continue
        array = np.array(values)
        print(f"  {name:<12}{len(array):>5}{np.median(array):>9.3f}"
              f"{(array >= MIN_SCORE).mean():>9.1%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rig": rig.tolist(), "stride": args.stride, "start_s": args.start,
        "fps": fps, "coverage": coverage, "anchor_rate": anchor_rate,
        "line_evidence": {k: v for k, v in groups.items()},
        "matrices": {str(i): full[i].tolist() for i in sorted(full)},
    }))
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
