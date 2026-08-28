"""Every tunable threshold in the pipeline. No magic numbers elsewhere."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Stage 1 — extraction
    target_fps: int = 10

    # Stage 2 — detection
    detector_conf: float = 0.25

    # Stage 5 — possession.
    # Distance from ball centre to player centre, divided by that player's box
    # height, so the threshold is scale-invariant as players move up/down court.
    possession_max_norm_dist: float = 0.8
    # A challenger must hold the ball this many consecutive frames to take over.
    possession_min_hold_frames: int = 3
    # Ball may vanish (occlusion, in flight) this many frames before the holder is dropped.
    possession_max_gap_frames: int = 5

    # Stage 6 — action classification. VideoMAE expects exactly 16 frames.
    action_window_frames: int = 16
    action_stride_frames: int = 8

    # Stage 8 — commentary
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 16000
    llm_max_attempts: int = 2
