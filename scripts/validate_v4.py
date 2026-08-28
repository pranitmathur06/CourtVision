"""V4 — Tracking sanity check (spec §6).

Runs the fine-tuned detector + ByteTrack over a real 10-second clip and writes an
annotated video with track IDs drawn, for visual inspection.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2

from courtvision.config import Config
from courtvision.detection import load_finetuned
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.tracking import PlayerTracker
from courtvision.types import PLAYER

CLIP = Path("data/raw_clips/sample.mp4")
CHECKPOINT = Path("checkpoints/detector.pt")
OUT_PATH = Path("outputs/v4_tracking.mp4")
MAX_SECONDS = 10

# The spec's V4 criterion is that track IDs stay CONSISTENT, not that few IDs
# exist. A raw unique-ID cap conflates real ID switching with players legitimately
# walking out of a panning broadcast shot and back in — ByteTrack carries no
# re-ID model (spec §4 picked it for exactly that reason), so a re-entry is
# always a new ID and no amount of tuning changes that.
#
# So measure stability directly: what share of player detections belong to tracks
# that persist. Churn shows up as many detections on very short tracks.
STABLE_TRACK_FRAMES = 10
MIN_STABLE_SHARE = 0.80


def main() -> int:
    if not CLIP.exists() or not CHECKPOINT.exists():
        print(f"V4 FAIL — need clip at {CLIP} and checkpoint at {CHECKPOINT}")
        return 1

    config = Config()
    detector = load_finetuned(str(CHECKPOINT), resolve_device(), config.detector_conf)
    tracker = PlayerTracker()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    lifetimes: Counter[int] = Counter()
    n_frames = 0

    for index, time_s, image in extract_frames(str(CLIP), config.target_fps):
        if time_s > MAX_SECONDS:
            break
        tracks = tracker.update(detector.detect(image))
        for track in tracks:
            if track.label != PLAYER:
                continue
            lifetimes[track.track_id] += 1
            cv2.rectangle(
                image,
                (int(track.box.x1), int(track.box.y1)),
                (int(track.box.x2), int(track.box.y2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                f"#{track.track_id}",
                (int(track.box.x1), int(track.box.y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        if writer is None:
            height, width = image.shape[:2]
            writer = cv2.VideoWriter(
                str(OUT_PATH),
                cv2.VideoWriter_fourcc(*"mp4v"),
                config.target_fps,
                (width, height),
            )
        writer.write(image)
        n_frames += 1

    if writer is not None:
        writer.release()

    total = sum(lifetimes.values())
    stable = sum(v for v in lifetimes.values() if v >= STABLE_TRACK_FRAMES)
    share = stable / total if total else 0.0
    churn = sum(1 for v in lifetimes.values() if v <= 3)

    ok = n_frames > 0 and share >= MIN_STABLE_SHARE
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V4 {verdict} — {n_frames} frames, {len(lifetimes)} track IDs, "
        f"{share:.1%} of detections on tracks lasting >={STABLE_TRACK_FRAMES} "
        f"frames (need >={MIN_STABLE_SHARE:.0%}); {churn} tracks lasted <=3 frames"
    )
    print(f"  review {OUT_PATH} for ID stability when players cross")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
