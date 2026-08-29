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
    """Who holds the ball this frame, with no temporal context.

    Uses nearest-player proximity. The detector also predicts a `handler` class,
    and preferring it was tried and MEASURED TO BE WORSE: it fires in only 24% of
    frames and is confidently wrong when it does (0.62 confidence on the wrong
    player in a hand-checked frame), because it has just 191 training instances
    against 5,282 for `player`. Scored against the answer key, preferring the
    handler gave 1/4 where plain proximity gives 3/4.

    The class is still trained and still exposed as `Frame.handler()` — with more
    possession annotation it is the right long-term signal, since proximity is
    genuinely under-determined here (a defender sits within 0.21 body-heights of
    the handler in the median frame). It is simply not good enough yet to trust.

    Ball-motion correlation was also tried — matching each player's displacement
    against the ball's — and scored 0/2 on testable moments where instantaneous
    proximity scored 2/2. The ball is detected in only ~63% of frames, so motion
    windows are too gappy to be reliable.
    """
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
            #
            # Support is counted over the next `min_hold_frames` frames in which
            # the ball was actually SEEN, skipping frames where it was missed.
            # Requiring strictly consecutive frames looked reasonable but fails
            # badly in practice: the ball is only detected in about two thirds of
            # broadcast frames, so a run of three in a row is rare and possession
            # would stick on a stale holder. A gap is missing data, not evidence
            # against the candidate.
            observed = [c for c in raw[index:] if c is not None][:min_hold_frames]
            if len(observed) == min_hold_frames and all(
                c == candidate for c in observed
            ):
                current = candidate
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
