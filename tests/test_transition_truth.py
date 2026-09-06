"""The shot-clock ground truth for transition, which is easy to get wrong.

Both rules here were bugs first. Splitting a possession at the first touch by
the other team invented twice as many possessions as a game has; reading the
clock at the last frame of a possession read a reset that had already
happened.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_transition import SAMPLE_HZ, possessions, truth_of


class Track:
    def __init__(self, track_id, label):
        self.track_id, self.label = track_id, label
        self.box = type("Box", (), dict(x1=0.0, y1=0.0, x2=1.0, y2=1.0))()


class Frame:
    def __init__(self, index, time_s, holder):
        self.index, self.time_s = index, time_s
        self._holder = holder

    def handler(self):
        return Track(self._holder, "handler") if self._holder else None

    @property
    def tracks(self):
        return ()


TEAMS = {1: "A", 2: "A", 8: "B", 9: "B"}


def _run(holders):
    return [Frame(i, i / SAMPLE_HZ, h) for i, h in enumerate(holders)]


def test_a_brief_touch_does_not_end_a_possession():
    """A deflection is not a change of possession."""
    holders = [1] * 40 + [8] * 5 + [1] * 40      # half a second the other way
    found = possessions(_run(holders), TEAMS, {})
    assert len(found) == 1
    assert found[0][0] == "A"


def test_a_sustained_change_does_end_it():
    holders = [1] * 40 + [8] * 40
    found = possessions(_run(holders), TEAMS, {})
    assert [side for side, _ in found] == ["A", "B"]


def test_the_lowest_clock_reading_is_the_one_that_counts():
    """A made basket resets the clock, so the last frame reads like a new one."""
    frames = _run([1] * 30)
    clocks = {f.index: c for f, c in
              zip(frames, [24.0 - i * 0.5 for i in range(29)] + [24.0])}
    assert truth_of(frames, TEAMS, "A", clocks) == "late"


def test_a_quick_possession_is_transition():
    frames = _run([1] * 30)
    clocks = {f.index: 24.0 - i * 0.2 for i, f in enumerate(frames)}
    assert truth_of(frames, TEAMS, "A", clocks) == "early"


def test_the_middle_band_is_refused_rather_than_forced():
    frames = _run([1] * 30)
    clocks = {f.index: 24.0 - i * 0.3 for i, f in enumerate(frames)}   # ends ~15
    assert truth_of(frames, TEAMS, "A", clocks) is None


def test_a_possession_with_no_clock_at_all_is_refused():
    frames = _run([1] * 30)
    assert truth_of(frames, TEAMS, "A", {f.index: math.nan for f in frames}) is None
