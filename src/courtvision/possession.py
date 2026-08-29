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
from courtvision.types import BALL, Frame, Track


def normalized_distance(player: Track, ball: Track) -> float:
    """Centre-to-centre distance in units of the player's box height."""
    px, py = player.box.center
    bx, by = ball.box.center
    return math.hypot(bx - px, by - py) / max(player.box.height, 1e-6)


def raw_holder(frame: Frame, max_norm_dist: float) -> int | None:
    """Who holds the ball this frame, with no temporal context.

    Prefers the detector's learned `handler` prediction and falls back to
    nearest-player when it does not fire.

    Proximity alone is genuinely under-determined on broadcast footage: a
    defender sits within 0.21 body-heights of the ball-handler in the median
    frame, and no distance metric or smoothing setting scores better than 5/7 on
    the answer key. The learned handler answers the question directly, using
    appearance cues — hands on the ball, body squared to it — that geometry
    cannot see.

    Trusting the handler was WRONG at first and is right now, purely because of
    training data. Weak supervision (scripts/harvest_handler_labels.py) took it
    from 191 instances to 3,550:

        191 instances -> fires 24% @0.55 conf -> V6 1/4  (worse than proximity)
        680           -> fires 49% @0.85      -> V6 5/7
        3,550         -> fires 60% @0.84      -> V6 6/7  (passes)

    Rejected alternatives, with numbers, are in docs/possession-investigation.md:
    box-edge distance (worse), box containment (resolves 25/67 frames), and
    ball-motion correlation (0/2 where proximity scored 2/2, because the ball is
    detected in only ~63% of frames so motion windows are too gappy).
    """
    handler = frame.handler()
    if handler is not None:
        return handler.track_id

    ball = frame.ball()
    players = frame.players()
    if ball is None or not players:
        return None

    nearest = min(players, key=lambda p: normalized_distance(p, ball))
    if normalized_distance(nearest, ball) > max_norm_dist:
        return None
    return nearest.track_id


def smooth_holders(
    raw: Sequence[int | None],
    min_hold_frames: int,
    max_gap_frames: int,
    ball_seen: Sequence[bool] | None = None,
) -> list[int | None]:
    """Apply hysteresis and gap-bridging to a per-frame raw holder sequence.

    `ball_seen` separates two situations that `raw` alone cannot distinguish,
    because both arrive as None:

    * the ball was not detected — absent information, so carrying the previous
      holder across the gap is right;
    * the ball WAS detected but sat further than the threshold from every
      player — positive evidence that nobody is holding it, so carrying the
      previous holder is wrong.

    Collapsing the second case into the first is what made the pipeline credit
    a player during a pass. In V6 the ball was visible in every frame from
    5.78s to 6.78s at up to 1.11 body-heights from the nearest player, and
    possession stayed pinned on track 20 throughout.

    One such observation is enough to release. Releasing only sets the holder to
    None, which is a claim of ignorance; the failure it replaces is crediting a
    specific player with a ball that is demonstrably in flight."""
    smoothed: list[int | None] = []
    current: int | None = None
    gap = 0

    for index, candidate in enumerate(raw):
        if candidate is None:
            unclaimed = (ball_seen is not None and index < len(ball_seen)
                         and ball_seen[index])
            if unclaimed and current is not None:
                # A visible, unclaimed ball usually means possession ended. But a
                # single frame of it can also be a bad ball box: at 4.08s the ball
                # was measured 1.41 body-heights away between two frames that put
                # it at 0.13, which no real ball does. Distance cannot separate
                # those; the holder's own behaviour can. Keep possession only if
                # this holder demonstrably has the ball again almost immediately.
                # At 6.08s track 20 never reclaimed it — the next raw holders were
                # 3, 23, then nothing — and that is a genuine loose ball.
                if current in raw[index + 1: index + 1 + max_gap_frames]:
                    gap = 0
                else:
                    current = None
                    gap = 0
            elif unclaimed:
                gap = 0
            else:
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
    seen = [any(t.label == BALL for t in frame.tracks) for frame in frames]
    return smooth_holders(
        raw,
        min_hold_frames=config.possession_min_hold_frames,
        max_gap_frames=config.possession_max_gap_frames,
        ball_seen=seen,
    )
