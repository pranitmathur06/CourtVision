"""Stage 7 — turn per-frame signals into discrete events.

Spec §10: commentary quality depends entirely on this stage. If these events are
noisy, stage 8 will narrate the noise confidently. So the holder for a window is
the majority vote across its frames, not the value at some single instant.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from courtvision.types import ActionWindow, Event, BACKGROUND


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
    """Merge action windows, possession and team labels into ordered events.

    Windows labelled `background` are dropped: ordinary play is not an event.
    Consecutive windows describing the SAME action by the SAME player collapse
    into one event. Windows overlap by design (size 16, stride 8), so a single
    real action spans several of them; emitting one event each produced
    commentary like "Another board credited to Player 12" three times for one
    rebound. An event is a thing that happened, not a window that was scored.
    """
    events: list[Event] = []
    previous_holder: int | None = None
    last_key: tuple[str, int | None] | None = None

    for window in sorted(windows, key=lambda w: w.start_time_s):
        if window.label == BACKGROUND:
            # Ordinary play is not something that happened. It exists so the
            # classifier can decline to name an action, and it breaks the
            # collapse chain so two real actions either side of a lull stay
            # separate events.
            last_key = None
            continue
        holder = dominant_holder(holders, window.start_index, window.end_index)
        # A change is only meaningful between two known holders; going to or from
        # "nobody" is the ball being in flight, not a turnover.
        changed = (
            holder is not None
            and previous_holder is not None
            and holder != previous_holder
        )
        key = (window.label, holder)
        if key == last_key:
            # Same action, same player, adjacent window: already reported.
            continue
        last_key = key

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
