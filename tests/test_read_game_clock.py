"""The clock resolver, on sequences built so the right answer is known."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import read_game_clock as clock  # noqa: E402


def _ambiguous(seconds):
    """What the reader itself makes of a clock face showing `seconds`.

    Built through readings_from so the test cannot drift from the parser: the
    board shows M:SS above a minute and SS.T below it, and both arrive as the
    same three digits, which is the ambiguity the resolver exists to settle.
    """
    if seconds >= 60:
        whole = int(round(seconds))
        return clock.readings_from(f"{whole // 60}:{whole % 60:02d}")
    digits = f"{round(seconds * 10):03d}"
    return clock.readings_from(f"{digits[0]}:{digits[1:]}")


def test_a_falling_clock_reads_as_minutes_not_tenths():
    rows = [(float(i), _ambiguous(500 - i)) for i in range(30)]
    assert [s for _, s in clock.resolve(rows)] == [500.0 - i for i in range(30)]


def test_the_last_minute_reads_as_tenths():
    """Coming down through a minute, "359" is 35.9 s and not 3:59."""
    rows = [(float(i), _ambiguous(65 - i)) for i in range(30)]
    resolved = [s for _, s in clock.resolve(rows)]
    assert resolved[:6] == [65.0, 64.0, 63.0, 62.0, 61.0, 60.0]
    assert resolved[6] == 59.0 and resolved[-1] == 36.0   # tenths, not minutes


def test_one_misread_does_not_capture_the_rest_of_the_quarter():
    """Finals Game 1: 6:07 read as 367 s, and 738 s of clock followed it down."""
    rows = [(float(i), _ambiguous(500 - i)) for i in range(30)]
    rows[10] = (10.0, [367.0, 60.7])
    resolved = dict(clock.resolve(rows))
    assert 10.0 not in resolved                       # dropped, not believed
    assert resolved[11.0] == 489.0                    # the clock carries on


def test_a_new_period_is_believed_once_the_next_reading_confirms_it():
    rows = [(float(i), [10.0 - i / 10.0]) for i in range(5)]
    rows += [(float(5 + i), _ambiguous(719 - i)) for i in range(5)]
    resolved = [s for _, s in clock.resolve(rows)]
    assert resolved[5] == 719.0 and resolved[-1] == 715.0


def test_a_lone_jump_at_the_very_end_is_not_believed():
    rows = [(float(i), _ambiguous(500 - i)) for i in range(10)]
    rows.append((10.0, [720.0]))
    assert [t for t, _ in clock.resolve(rows)] == [float(i) for i in range(10)]
