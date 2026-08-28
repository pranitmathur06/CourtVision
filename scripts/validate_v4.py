"""V4 — Tracking sanity check (spec §6).

Runs the fine-tuned detector + ByteTrack over a real 10-second clip and writes an
annotated video with track IDs drawn, for visual inspection.
"""

from __future__ import annotations

import sys
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


def main() -> int:
    if not CLIP.exists() or not CHECKPOINT.exists():
        print(f"V4 FAIL — need clip at {CLIP} and checkpoint at {CHECKPOINT}")
        return 1

    config = Config()
    detector = load_finetuned(str(CHECKPOINT), resolve_device(), config.detector_conf)
    tracker = PlayerTracker()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    seen_ids: set[int] = set()
    n_frames = 0

    for index, time_s, image in extract_frames(str(CLIP), config.target_fps):
        if time_s > MAX_SECONDS:
            break
        tracks = tracker.update(detector.detect(image))
        for track in tracks:
            if track.label != PLAYER:
                continue
            seen_ids.add(track.track_id)
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

    # 10 players on court; many more unique IDs than that means heavy switching.
    ok = n_frames > 0 and len(seen_ids) <= 20
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V4 {verdict} — {n_frames} frames, {len(seen_ids)} unique track IDs "
        f"(want <=20); review {OUT_PATH} for ID stability when players cross"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
