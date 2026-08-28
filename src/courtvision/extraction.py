"""Stage 1 — decode a video and yield frames at a target sampling rate.

Sampling is done by keeping a source frame whenever its scaled index advances,
which spreads the kept frames evenly instead of taking a contiguous prefix.
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


def probe(video_path: str) -> tuple[float, int]:
    """Return (source_fps, source_frame_count) without decoding the whole file."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return fps, count


def expected_frame_count(
    source_frame_count: int, source_fps: float, target_fps: int
) -> int:
    """How many frames `extract_frames` will yield. Never upsamples."""
    if target_fps >= source_fps:
        return source_frame_count
    return int(source_frame_count * target_fps / source_fps)


def extract_frames(
    video_path: str, target_fps: int
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (output_index, time_s, bgr_image) at approximately `target_fps`.

    time_s is derived from the *source* frame index, so timestamps stay true to
    the original clip regardless of the sampling rate.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(target_fps)
    ratio = min(target_fps / source_fps, 1.0)

    try:
        source_index = 0
        output_index = 0
        kept = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            # Keep this frame if the running quota of kept frames has advanced.
            if int((source_index + 1) * ratio) > kept:
                kept += 1
                yield output_index, source_index / source_fps, image
                output_index += 1
            source_index += 1
    finally:
        capture.release()
