"""V1 — Frame extraction sanity check (spec §6).

Extracts frames from a real clip at the target FPS, confirms the count matches
duration x fps within rounding, and saves a few frames as images for eyeballing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from courtvision.config import Config
from courtvision.extraction import expected_frame_count, extract_frames, probe

CLIP = Path("data/raw_clips/sample.mp4")
OUT_DIR = Path("outputs/v1_frames")


def main() -> int:
    if not CLIP.exists():
        print(f"V1 FAIL — no clip at {CLIP}; place a 10-60s basketball clip there")
        return 1

    config = Config()
    source_fps, source_count = probe(str(CLIP))
    duration_s = source_count / source_fps
    expected = expected_frame_count(source_count, source_fps, config.target_fps)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actual = 0
    for index, time_s, image in extract_frames(str(CLIP), config.target_fps):
        actual += 1
        if index < 5:
            cv2.imwrite(str(OUT_DIR / f"frame_{index:03d}_t{time_s:.2f}.png"), image)

    # Allow one frame of slack for rounding at the tail.
    ok = abs(actual - expected) <= 1
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V1 {verdict} — source {source_fps:.2f}fps x {duration_s:.2f}s; "
        f"expected {expected} frames at {config.target_fps}fps, got {actual}; "
        f"sample images in {OUT_DIR}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
