"""Stage 7 — turn per-frame signals into discrete events.

Spec §10: commentary quality depends entirely on this stage. If these events are
noisy, stage 8 will narrate the noise confidently. So the holder for a window is
the majority vote across its frames, not the value at some single instant.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from courtvision.types import ActionWindow, Event


def dominant_holder(
    holders: Sequence[int | None], start: int, end: int
) -> int | None:
    """Most frequent non-None holder across the inclusive frame range."""
    votes = Counter(h for h in holders[start : end + 1] if h is not None)
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def build_events(
    windows: Sequence[ActionWindow],
    holders: Sequence[int | None],
    teams: dict[int, str],
) -> list[Event]:
    """Merge action windows, possession and team labels into ordered events."""
    events: list[Event] = []
    previous_holder: int | None = None

    for window in sorted(windows, key=lambda w: w.start_time_s):
        holder = dominant_holder(holders, window.start_index, window.end_index)
        # A change is only meaningful between two known holders; going to or from
        # "nobody" is the ball being in flight, not a turnover.
        changed = (
            holder is not None
            and previous_holder is not None
            and holder != previous_holder
        )
        events.append(
            Event(
                time_s=window.start_time_s,
                track_id=holder,
                team=teams.get(holder) if holder is not None else None,
                action=window.label,
                possession_change=changed,
            )
        )
        if holder is not None:
            previous_holder = holder

    return events
