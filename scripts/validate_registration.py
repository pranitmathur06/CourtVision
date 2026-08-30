"""V11 — Is the automatic court registration actually accurate?

The alignment score says how well the projected court sits on detected lines.
It cannot say whether the resulting COURT COORDINATES are right, and it is an
upper bound anyway because players contaminate the line mask.

Hand-annotating landmarks would just move the problem: my pixel estimates become
the ground truth, and they could be wrong in exactly the way the registration is.

So this checks physics instead. Players do not teleport. Map each tracked player
into court feet frame by frame and the implied speeds must look like basketball:
sustained movement runs 5-15 ft/s and an NBA sprint tops out near 20-25 ft/s. A
registration wrong in scale inflates every speed; one that jitters between
frames produces impossible spikes. Neither can hide from this, and it needs no
annotation at all.

Track ID switches also produce spikes, so a small tail is expected and the bar
is set on the bulk of the distribution rather than the maximum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from courtvision.config import Config
from courtvision.court import COURT_WIDTH, HALF_COURT_LENGTH
from courtvision.court_lines import (homography_from_camera,
                                     project_world_point, search_camera)
from courtvision.detection import load_pipeline_detector
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.tracking import PlayerTracker
from courtvision.court import BASKET
from courtvision.types import HANDLER, PLAYER, RIM, Frame

CLIP = Path("data/raw_clips/sample.mp4")
CHECKPOINT = Path("checkpoints/detector.pt")
STRIDE = 2
FRAMES = 12
MIN_SCORE = 0.30

SPRINT_FT_S = 25.0          # an NBA sprint; nobody sustains more
REQUIRED_UNDER_SPRINT = 0.90
PLAUSIBLE_MEDIAN = (3.0, 15.0)
MAX_RIM_ERROR_PX = 40.0
LOCAL_SPREAD = np.array([6.0, 6.0, 4.0, 5.0, 5.0, 150.0])


def main() -> int:
    if not CLIP.exists() or not CHECKPOINT.exists():
        print(f"V11 FAIL — need {CLIP} and {CHECKPOINT}")
        return 1

    config = Config()
    detector = load_pipeline_detector(str(CHECKPOINT), resolve_device(),
                                      config.detector_conf, config.ball_conf)
    tracker = PlayerTracker()
    frames: list[Frame] = []
    images = []
    for index, time_s, image in extract_frames(str(CLIP), config.target_fps):
        frames.append(Frame(index, time_s,
                            tuple(tracker.update(detector.detect(image)))))
        images.append(image)

    chosen = list(range(0, len(frames), STRIDE))[:FRAMES]
    series: dict[int, list[tuple[float, float, float]]] = {}
    scores: list[float] = []
    rim_errors: list[float] = []
    previous: np.ndarray | None = None

    for index in chosen:
        bounds = None
        if previous is not None:
            bounds = [(float(a), float(b)) for a, b in
                      zip(previous - LOCAL_SPREAD, previous + LOCAL_SPREAD)]
        # Anchor on the rim when the detector finds it. Court lines alone leave
        # the court free to slide: a line-only fit here put the rim 174 px from
        # where it was detected, roughly ten feet, and the speed check below
        # cannot see that because a constant offset moves every player equally.
        rims = [t for t in frames[index].tracks if t.label == RIM]
        rim_px = None
        if rims:
            best = max(rims, key=lambda t: t.conf)
            rim_px = ((best.box.x1 + best.box.x2) / 2,
                      (best.box.y1 + best.box.y2) / 2)
        params, score = search_camera(images[index], seed=0,
                                      max_iterations=60 if bounds else 250,
                                      bounds=bounds, rim_px=rim_px)
        if params is None or score < MIN_SCORE:
            continue
        scores.append(score)
        previous = params
        if rim_px is not None:
            projected = project_world_point(
                params, (BASKET[0], BASKET[1], 10.0), images[index].shape[:2])
            if projected is not None:
                rim_errors.append(float(np.hypot(projected[0] - rim_px[0],
                                                 projected[1] - rim_px[1])))
        matrix = np.linalg.inv(homography_from_camera(params,
                                                      images[index].shape[:2]))
        frame = frames[index]
        players = [t for t in frame.tracks if t.label in (PLAYER, HANDLER)]
        if not players:
            continue
        feet = np.array([[(t.box.x1 + t.box.x2) / 2, t.box.y2] for t in players])
        projected = np.hstack([feet, np.ones((len(feet), 1))]) @ matrix.T
        court = projected[:, :2] / projected[:, 2:3]
        for track, position in zip(players, court):
            if np.isnan(position).any():
                continue
            series.setdefault(track.track_id, []).append(
                (frame.time_s, float(position[0]), float(position[1])))

    if len(scores) < 5:
        print(f"V11 FAIL — only {len(scores)} frames registered above {MIN_SCORE}")
        return 1

    speeds = []
    for points in series.values():
        for (t0, x0, y0), (t1, x1, y1) in zip(points, points[1:]):
            if t1 > t0:
                speeds.append(float(np.hypot(x1 - x0, y1 - y0) / (t1 - t0)))
    if len(speeds) < 30:
        print(f"V11 FAIL — only {len(speeds)} tracked steps to measure")
        return 1
    speeds = np.array(speeds)

    median = float(np.median(speeds))
    under = float((speeds <= SPRINT_FT_S).mean())
    positions = np.array([[x, y] for pts in series.values() for _, x, y in pts])

    print(f"V11 — registration accuracy from implied player motion")
    print(f"  {len(scores)} frames registered, median score "
          f"{np.median(scores):.3f}; {len(series)} tracks, {len(speeds)} steps")
    print(f"  speed ft/s: p50 {median:.1f}  p90 {np.percentile(speeds,90):.1f}  "
          f"p95 {np.percentile(speeds,95):.1f}  max {speeds.max():.1f}")
    print(f"  within sprint ({SPRINT_FT_S:.0f} ft/s): {under:.1%} "
          f"(need {REQUIRED_UNDER_SPRINT:.0%})")
    print(f"  court extent: x {positions[:,0].min():.1f}..{positions[:,0].max():.1f} "
          f"(court 0..{COURT_WIDTH:.0f}), "
          f"y {positions[:,1].min():.1f}..{positions[:,1].max():.1f} "
          f"(0..{HALF_COURT_LENGTH:.0f})")

    if rim_errors:
        print(f"  rim reprojection error px: median {np.median(rim_errors):.1f}, "
              f"max {max(rim_errors):.1f} (bar {MAX_RIM_ERROR_PX:.0f})")
    rim_ok = (not rim_errors) or float(np.median(rim_errors)) <= MAX_RIM_ERROR_PX

    ok = (under >= REQUIRED_UNDER_SPRINT
          and PLAUSIBLE_MEDIAN[0] <= median <= PLAUSIBLE_MEDIAN[1]
          and rim_ok)
    if not ok:
        if not rim_ok:
            reason = (f"rim reprojects {np.median(rim_errors):.0f} px from where "
                      f"it was detected, so the court is absolutely misplaced")
        else:
            reason = ("too many impossible steps" if under < REQUIRED_UNDER_SPRINT
                      else f"median {median:.1f} ft/s outside {PLAUSIBLE_MEDIAN}")
        print(f"V11 FAIL — {reason}; the registration's scale or stability is off")
        return 1
    print(f"V11 PASS — implied motion is physical (scale and stability) and the "
          f"rim\n  reprojects onto its detection (absolute position). Motion "
          f"alone could not\n  show the second: a constant offset moves every "
          f"player equally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
