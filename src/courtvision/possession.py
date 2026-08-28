"""Stage 5 — who has the ball, by proximity plus temporal smoothing.

This is a heuristic, not ground truth (spec §10). It is wrong on contested
rebounds, blocked shots and balls in flight. The design goal is "mostly right and
never flickering", because stage 7 turns these into discrete events and a single
bad frame there becomes a fabricated possession change in the commentary.

Two decisions do the work:

* Distance is divided by the player's box height, so one threshold works for
  players near and far from the camera. A raw pixel threshold would track
  perspective, not possession.
* Switching possession requires `min_hold_frames` of consecutive support
  (hysteresis), and a missing ball carries the previous holder for up to
  `max_gap_frames` before possession is dropped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from courtvision.config import Config
from courtvision.types import Frame, Track


def normalized_distance(player: Track, ball: Track) -> float:
    """Centre-to-centre distance in units of the player's box height."""
    px, py = player.box.center
    bx, by = ball.box.center
    return math.hypot(bx - px, by - py) / max(player.box.height, 1e-6)


def raw_holder(frame: Frame, max_norm_dist: float) -> int | None:
    """Nearest player to the ball, if close enough. No temporal context."""
    ball = frame.ball()
    players = frame.players()
    if ball is None or not players:
        return None

    nearest = min(players, key=lambda p: normalized_distance(p, ball))
    if normalized_distance(nearest, ball) > max_norm_dist:
        return None
    return nearest.track_id


def smooth_holders(
    raw: Sequence[int | None], min_hold_frames: int, max_gap_frames: int
) -> list[int | None]:
    """Apply hysteresis and gap-bridging to a per-frame raw holder sequence."""
    smoothed: list[int | None] = []
    current: int | None = None
    gap = 0

    for index, candidate in enumerate(raw):
        if candidate is None:
            # Ball missing: carry the current holder for a bounded number of frames.
            gap += 1
            if gap > max_gap_frames:
                current = None
        elif candidate == current:
            gap = 0
        else:
            # A different player claims the ball. Require sustained support so a
            # single noisy frame cannot flip possession.
            window = list(raw[index : index + min_hold_frames])
            if len(window) == min_hold_frames and all(c == candidate for c in window):
                current = candidate
                gap = 0
            else:
                gap = 0
        smoothed.append(current)

    return smoothed


def possession_timeline(
    frames: Sequence[Frame], config: Config
) -> list[int | None]:
    """Per-frame holder track_id (or None) for a whole clip."""
    raw = [raw_holder(frame, config.possession_max_norm_dist) for frame in frames]
    return smooth_holders(
        raw,
        min_hold_frames=config.possession_min_hold_frames,
        max_gap_frames=config.possession_max_gap_frames,
    )
