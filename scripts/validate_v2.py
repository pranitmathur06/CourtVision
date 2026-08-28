"""V2 — Stock detector sanity check (spec §6).

Runs off-the-shelf YOLO11n with COCO weights on 5 frames of a real clip, before
any basketball-specific fine-tuning. This is the baseline that V3 must beat, and
it confirms ultralytics + MPS actually work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from courtvision.config import Config
from courtvision.detection import COCO_CLASS_MAP, YoloDetector
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.types import PLAYER

CLIP = Path("data/raw_clips/sample.mp4")
OUT_DIR = Path("outputs/v2_detections")
N_FRAMES = 5
MIN_PLAYERS_PER_FRAME = 2


def main() -> int:
    if not CLIP.exists():
        print(f"V2 FAIL — no clip at {CLIP}")
        return 1

    config = Config()
    device = resolve_device()
    # yolo11n.pt downloads on first use (~6 MB) into the ultralytics cache.
    detector = YoloDetector("yolo11n.pt", device, config.detector_conf, COCO_CLASS_MAP)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts: list[int] = []
    for index, _, image in extract_frames(str(CLIP), config.target_fps):
        if index >= N_FRAMES:
            break
        detections = [d for d in detector.detect(image) if d.label == PLAYER]
        counts.append(len(detections))
        for det in detections:
            cv2.rectangle(
                image,
                (int(det.box.x1), int(det.box.y1)),
                (int(det.box.x2), int(det.box.y2)),
                (0, 255, 0),
                2,
            )
        cv2.imwrite(str(OUT_DIR / f"det_{index:03d}.png"), image)

    ok = bool(counts) and all(c >= MIN_PLAYERS_PER_FRAME for c in counts)
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V2 {verdict} — device={device}; players per frame {counts} "
        f"(need >={MIN_PLAYERS_PER_FRAME} each); overlays in {OUT_DIR}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
