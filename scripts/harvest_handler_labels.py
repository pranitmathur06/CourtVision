"""Harvest ball-handler pseudo-labels from BARD clips by weak supervision.

The Roboflow dataset gives only 191 `handler` instances, and a detector trained
on that is confidently wrong (0.62 confidence on the wrong player in a checked
frame); preferring it scored 1/4 against the V6 answer key where plain proximity
scored 3/4. Proximity itself is capped: measured across the whole parameter
space it reaches 5/7 and no further, because a defender sits within 0.21
body-heights of the handler in the median frame.

The way out is more handler labels, and they can be harvested rather than hand
drawn. Proximity is not always ambiguous — in roughly 39% of frames the ball is
clearly nearest ONE player, well separated from the runner-up. Those frames give
reliable labels for free. Training on them teaches the detector the APPEARANCE of
a ball-handler (hands on the ball, body squared to it), which is exactly the cue
that generalises to the crowded frames proximity cannot resolve.

This is standard weak supervision: the teacher (a geometric rule) is only
consulted where it is confident, and the student learns a signal the teacher
never had access to.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from courtvision.config import Config
from courtvision.detection import load_pipeline_detector
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.possession import normalized_distance
from courtvision.types import BALL, HANDLER, PLAYER, RIM, Track

SOURCE_CLIPS = Path("data/labeled/bard_meta/clips")
OUT = Path("data/labeled/handler_harvest")

# A frame only yields a label when the nearest player is BOTH close to the ball
# and clearly clear of the runner-up. These are deliberately strict: a wrong
# pseudo-label is worse than a missing one.
MAX_NEAREST = 0.45
MIN_SEPARATION = 0.55

PLAYER_CLASS, BALL_CLASS, RIM_CLASS, HANDLER_CLASS = 0, 1, 2, 3


def to_yolo(box, width: int, height: int) -> str | None:
    cx = (box.x1 + box.x2) / 2.0 / width
    cy = (box.y1 + box.y2) / 2.0 / height
    w = (box.x2 - box.x1) / width
    h = (box.y2 - box.y1) / height
    if not (0 < w <= 1 and 0 < h <= 1):
        return None
    return f"{min(max(cx,0.0),1.0):.6f} {min(max(cy,0.0),1.0):.6f} {w:.6f} {h:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest handler pseudo-labels")
    parser.add_argument("--clips", type=int, default=120)
    parser.add_argument("--max-per-clip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clips = sorted(SOURCE_CLIPS.rglob("*.mp4"))
    if not clips:
        print(f"FAIL — no source clips under {SOURCE_CLIPS}")
        return 1
    random.Random(args.seed).shuffle(clips)
    clips = clips[: args.clips]
    print(f"scanning {len(clips)} BARD clips for unambiguous possession frames")

    config = Config()
    detector = load_pipeline_detector(
        "checkpoints/detector.pt", resolve_device(),
        config.detector_conf, config.ball_conf,
    )

    if OUT.exists():
        shutil.rmtree(OUT)
    images_dir, labels_dir = OUT / "images", OUT / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    kept = scanned = 0
    for clip_index, clip in enumerate(clips):
        per_clip = 0
        for _, _, image in extract_frames(str(clip), config.target_fps):
            if per_clip >= args.max_per_clip:
                break
            scanned += 1
            detections = detector.detect(image)
            balls = [d for d in detections if d.label == BALL]
            people = [d for d in detections if d.label in (PLAYER, HANDLER)]
            if not balls or len(people) < 2:
                continue

            ball = max(balls, key=lambda d: d.conf)
            as_tracks = [Track(-1, d.box, d.label, d.conf) for d in people]
            ranked = sorted(as_tracks, key=lambda t: normalized_distance(t, Track(-1, ball.box, BALL, ball.conf)))
            ball_track = Track(-1, ball.box, BALL, ball.conf)
            d0 = normalized_distance(ranked[0], ball_track)
            d1 = normalized_distance(ranked[1], ball_track)
            if d0 > MAX_NEAREST or (d1 - d0) < MIN_SEPARATION:
                continue

            height, width = image.shape[:2]
            lines = []
            handler_line = to_yolo(ranked[0].box, width, height)
            if handler_line is None:
                continue
            lines.append(f"{HANDLER_CLASS} {handler_line}")
            for other in ranked[1:]:
                line = to_yolo(other.box, width, height)
                if line:
                    lines.append(f"{PLAYER_CLASS} {line}")
            for d in detections:
                if d.label == BALL:
                    line = to_yolo(d.box, width, height)
                    if line:
                        lines.append(f"{BALL_CLASS} {line}")
                elif d.label == RIM:
                    line = to_yolo(d.box, width, height)
                    if line:
                        lines.append(f"{RIM_CLASS} {line}")

            stem = f"h{clip_index:04d}_{per_clip}"
            cv2.imwrite(str(images_dir / f"{stem}.jpg"), image)
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            kept += 1
            per_clip += 1

        if (clip_index + 1) % 20 == 0:
            print(f"  {clip_index+1}/{len(clips)} clips, {kept} labels harvested")

    rate = kept / scanned if scanned else 0.0
    print(f"\nharvested {kept} handler pseudo-labels from {scanned} frames "
          f"({rate:.1%} of frames were unambiguous enough)")
    print(f"thresholds: nearest <= {MAX_NEAREST}, separation >= {MIN_SEPARATION}")
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
