"""Reading the game clock off a broadcast scoreboard."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from courtvision.scoreboard import (
    build_templates,
    build_templates_from_many,
    clock_glyphs,
    clock_to_seconds,
    read_clock,
    segment_glyphs,
)


def render_bar(clock_text: str, period: str = "1ST") -> np.ndarray:
    """A scoreboard bar: small period label, large clock, dark on bright."""
    bar = np.full((60, 220, 3), 240, np.uint8)
    cv2.putText(bar, period, (8, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(bar, clock_text, (70, 45), cv2.FONT_HERSHEY_DUPLEX, 1.3, (15, 15, 15), 3)
    return bar


def test_clock_digits_are_separated_from_the_period_label_by_height():
    """A scoreboard puts the period in smaller type beside the clock."""
    bar = render_bar("10:59")
    everything = segment_glyphs(bar)
    clock = clock_glyphs(bar)
    assert len(clock) == 4, [g.height for g in everything]
    assert len(everything) > len(clock), "the period label should also segment"
    # Compare by position: a Glyph holds a numpy image, so `in` on the
    # dataclass raises "truth value of an array is ambiguous".
    clock_xs = {g.x for g in clock}
    label = [g for g in everything if g.x not in clock_xs]
    assert label, "expected the period label to segment separately"
    assert min(g.height for g in clock) > max(g.height for g in label)


def test_round_trip_a_known_clock():
    bar = render_bar("10:59")
    templates = build_templates(bar, "1059")
    assert sorted(templates) == ["0", "1", "5", "9"]
    text, score = read_clock(bar, templates)
    assert text == "10:59" and score > 0.9


def test_a_digit_with_no_template_is_refused_not_guessed():
    """An unreadable clock is normal; a wrong one mis-joins every later play."""
    templates = build_templates(render_bar("10:59"), "1059")
    text, _ = read_clock(render_bar("10:34"), templates)   # 3 and 4 unseen
    assert text is None


def test_build_templates_rejects_a_mismatched_reading():
    """Wrong labels would poison every later read, so refuse loudly."""
    bar = render_bar("10:59")
    with pytest.raises(ValueError, match="segmented 4 clock digits"):
        build_templates(bar, "105")


def test_a_blank_bar_reads_nothing():
    blank = np.full((60, 220, 3), 240, np.uint8)
    templates = build_templates(render_bar("10:59"), "1059")
    assert read_clock(blank, templates)[0] is None


@pytest.mark.parametrize("text, seconds", [("10:59", 659), ("0:07", 7), ("12:00", 720)])
def test_clock_to_seconds(text, seconds):
    assert clock_to_seconds(text) == seconds


def test_single_digit_minute_clock_is_read():
    """Under ten minutes the clock loses a digit; three glyphs must still work."""
    templates = build_templates(render_bar("10:59"), "1059")
    text, _ = read_clock(render_bar("9:15"), templates)
    assert text == "9:15"


def test_templates_merge_across_frames():
    """One frame cannot cover ten digits, and the gap is not cosmetic.

    Built from 3:47 alone, the reader can only read values made of 3, 4 and 7 —
    and it does not fail quietly on the rest. On the holdout clip it matched a 7
    against the 3 template and returned a confident 3:43 for six frames.
    """
    from courtvision.scoreboard import build_templates_from_many

    samples = [(render_bar("3:52"), "352"), (render_bar("3:49"), "349"),
               (render_bar("10:47"), "1047")]
    templates = build_templates_from_many(samples)
    assert sorted(templates) == ["0", "1", "2", "3", "4", "5", "7", "9"]


def test_the_first_rendering_of_a_digit_wins():
    from courtvision.scoreboard import build_templates_from_many

    first = render_bar("3:47")
    merged = build_templates_from_many([(first, "347"), (render_bar("3:19"), "319")])
    from courtvision.scoreboard import build_templates
    assert np.array_equal(merged["3"], build_templates(first, "347")["3"])


def test_merged_templates_read_a_value_no_single_frame_contained():
    from courtvision.scoreboard import build_templates_from_many

    templates = build_templates_from_many(
        [(render_bar("3:52"), "352"), (render_bar("9:41"), "941")])
    text, score = read_clock(render_bar("2:19"), templates)
    assert text == "2:19" and score > 0.9


def test_a_wide_bold_font_is_not_rejected():
    """`w > h` looked reasonable and rejected every digit on a second broadcast.

    ESPN renders the clock 19 px wide against 17 tall; the earlier network's
    font was narrower. The bound must allow a digit slightly wider than it is
    tall while still rejecting the scoreboard outline, which is far wider.
    """
    from courtvision.scoreboard import segment_glyphs

    bar = np.full((60, 220, 3), 240, np.uint8)
    # A deliberately wide, bold glyph: wider than tall, but not a bar.
    cv2.rectangle(bar, (40, 20), (63, 40), (20, 20, 20), -1)
    found = [g for g in segment_glyphs(bar) if g.width > g.height]
    assert found, "a digit slightly wider than tall must survive segmentation"


def test_touching_digits_are_refused_not_mis_templated():
    """Connected components cannot split glyphs that touch.

    A rendering where two digits merge segments as one, and build_templates
    raises rather than pairing the wrong image with a digit — bad templates
    would poison every later read silently.
    """
    from courtvision.scoreboard import build_templates_from_many

    bar = np.full((60, 220, 3), 240, np.uint8)
    cv2.rectangle(bar, (70, 18), (130, 44), (20, 20, 20), -1)   # one solid blob
    with pytest.raises(ValueError, match="segmented"):
        build_templates_from_many([(bar, "347")])


def test_normalise_polarity_inverts_a_white_on_black_bar():
    """TNT draws white digits on black; this module was written for the reverse.

    Against a white-on-black bar the segmenter returned zero glyphs from a crop
    where the clock is plainly legible — a wrong-polarity failure that looks
    exactly like "no scoreboard here".
    """
    import numpy as np

    from courtvision.scoreboard import normalise_polarity, segment_glyphs

    dark = np.zeros((36, 72, 3), np.uint8)
    dark[8:28, 8:18] = 255          # a bright digit-shaped blob
    dark[8:28, 26:36] = 255
    assert segment_glyphs(dark) == [], "raw white-on-black finds nothing"
    assert len(segment_glyphs(normalise_polarity(dark))) >= 2


def test_normalise_polarity_leaves_a_dark_on_bright_bar_alone():
    import numpy as np

    from courtvision.scoreboard import normalise_polarity

    bright = np.full((36, 72, 3), 240, np.uint8)
    bright[8:28, 8:18] = 0
    assert np.array_equal(normalise_polarity(bright), bright)


def test_period_boundaries_come_from_clock_resets():
    """A clock only counts down, so a big jump upward is a new period.

    Reading "2ND"/"3RD" text would be a second OCR problem with its own
    failure modes; the monotonicity the reader already validates itself with is
    enough to segment periods.
    """
    from scripts.label_live_game import PERIOD_SECONDS

    # 12:00 -> 0:10, then reset to 12:00: one boundary, not two periods of noise.
    readings = [720, 500, 200, 10, 720, 600, 300]
    period, previous, periods = 1, None, []
    for seconds in readings:
        if previous is not None and seconds > previous + 60:
            period += 1
        previous = seconds
        periods.append(period)
    assert periods == [1, 1, 1, 1, 2, 2, 2]
    assert PERIOD_SECONDS == 720


def test_a_small_upward_jump_is_a_misread_not_a_period():
    readings = [500, 498, 505, 495]          # 505 is noise
    period, previous, kept = 1, None, []
    for seconds in readings:
        if previous is not None and seconds > previous + 60:
            period += 1
        elif previous is not None and seconds > previous + 2:
            continue
        previous = seconds
        kept.append(seconds)
    assert kept == [500, 498, 495], "a 7-second jump up must be dropped"
    assert period == 1


def test_every_mapped_game_has_a_broadcast_profile():
    """A missing profile silently falls back to TNT's crop and reads nothing.

    That is exactly what happened on the first ESPN game: 0 clock samples, and
    the failure looked like "this broadcast has no scoreboard" rather than
    "we cropped the wrong part of the screen".
    """
    from scripts.label_live_game import BROADCASTS, GAME_BROADCAST

    for game_id, name in GAME_BROADCAST.items():
        assert name in BROADCASTS, f"{game_id} maps to unknown broadcast {name}"


def test_broadcast_profiles_are_well_formed():
    from scripts.label_live_game import BROADCASTS

    for name, profile in BROADCASTS.items():
        top, bottom, left, right = profile["roi"]
        assert bottom > top and right > left, f"{name} has an empty crop"
        assert len(profile["anchors"]) >= 3, (
            f"{name}: templates from one or two frames cover too few digits; "
            "the first attempt read 1 frame in 60")
        for _, digits in profile["anchors"]:
            assert digits.isdigit(), f"{name}: anchors carry digits, no colon"
            assert 3 <= len(digits) <= 4


def test_build_templates_handles_white_on_dark():
    """The builder must normalise polarity like the reader does.

    read_clock was fixed for white-on-dark but build_templates was not, so an
    ESPN crop that plainly shows 6:36 raised "segmented 0 clock digits" — a
    message that blames the crop coordinates, which were correct.
    """
    import numpy as np

    from courtvision.scoreboard import build_templates

    dark = np.zeros((36, 96, 3), np.uint8)
    for i, x in enumerate((8, 30, 52)):
        dark[8:28, x:x + 12] = 255
    templates = build_templates(dark, "636")
    assert set(templates) == {"6", "3"}


def test_each_broadcast_names_the_video_its_anchors_were_read_from():
    """Anchors are (fraction, value) from ONE recording and do not transfer.

    Fraction 0.28 is 8:13 in one game and something else in the next, which
    surfaced as "segmented 0 clock digits but was told 3" while the crop was
    perfectly correct.
    """
    from scripts.label_live_game import BROADCASTS

    for name, profile in BROADCASTS.items():
        assert profile.get("anchor_video"), (
            f"{name}: anchors must name the video they were read from")


# -- the last ten seconds of a period, which the reader could not see ---------

def test_two_glyphs_are_not_read_unless_tenths_are_asked_for():
    """Three other callers assume MM:SS, so the default must not change.

    A clock half-covered by a graphic can present two glyphs, and reading those
    as a time would be a confident wrong answer where the old behaviour was an
    honest None."""
    templates = build_templates_from_many(
        [(render_bar("7:2"), "72"), (render_bar("10:59"), "1059")])
    assert read_clock(render_bar("7:2"), templates)[0] is None


def test_the_blind_spot_is_exactly_under_ten_seconds():
    """Under a minute the clock shows tenths. At 18.2 that is three glyphs,
    which this has always read as "1:82" and the caller turns back into 18.2.
    At 7.2 it is two, and two failed the length gate -- so across the four
    registered broadcasts there are 79/34/342/25 readings between 10 and 20
    seconds and 0/0/0/9 under 10."""
    templates = build_templates_from_many(
        [(render_bar("7:2"), "72"), (render_bar("10:59"), "1059")])
    text, _ = read_clock(render_bar("7:2"), templates, allow_tenths=True)
    assert text == "7:2"
    # ...and a four-glyph clock still reads exactly as before, with or without.
    assert read_clock(render_bar("10:59"), templates)[0] == "10:59"
    assert read_clock(render_bar("10:59"), templates,
                      allow_tenths=True)[0] == "10:59"


def test_two_digits_have_exactly_one_reading_and_it_is_under_ten_seconds():
    """There is no MM:SS reading of two digits, so this adds one candidate in
    [0.0, 9.9] and never competes with a minutes reading."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from read_game_clock import parse, readings_from
    assert readings_from("7:2") == [7.2]
    assert parse("7:2") == 7.2
    assert readings_from("0:9") == [0.9]
    # ...and it leaves every longer reading exactly as it was.
    assert readings_from("1:82") == [18.2]
    assert readings_from("3:59") == [239.0, 35.9]
    assert readings_from("10:34") == [634.0]


def test_a_two_digit_misread_cannot_capture_the_clock():
    """`resolve` makes a reading that continues nothing prove itself against the
    NEXT frame. A stray 7.2 in the middle of a period is not confirmed and is
    dropped, which is the same guard that stopped one bad 6:07 from turning 738
    seconds of Finals G1 into tenths."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from read_game_clock import resolve
    good = [(float(i), [600.0 - i]) for i in range(6)]
    stray = good[:3] + [(3.0, [7.2])] + [(float(i), [600.0 - i]) for i in range(4, 7)]
    assert [round(v, 1) for _, v in resolve(stray)] == [600.0, 599.0, 598.0,
                                                        596.0, 595.0, 594.0]


def test_a_real_run_of_tenths_at_a_period_end_is_believed():
    """The whole point: a genuine countdown through the blind spot survives."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from read_game_clock import resolve
    rows = [(0.0, [12.4]), (1.0, [11.4]), (2.0, [10.4]), (3.0, [9.4]),
            (4.0, [8.4]), (5.0, [7.4])]
    assert [round(v, 1) for _, v in resolve(rows)] == [12.4, 11.4, 10.4,
                                                       9.4, 8.4, 7.4]


def test_a_sub_ten_second_reading_may_never_start_a_run():
    """Two consecutive strays that agree would otherwise confirm each other.

    `assign_periods` treats the jump back up to the real clock as a new run
    starting near the top of its period, so a pair of stray two-glyph reads in
    the middle of a quarter would turn one quarter into two -- a worse failure
    than the blind spot the tenths reading fixes. A genuine countdown always
    arrives by continuing from a reading above ten seconds."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from read_game_clock import resolve
    rows = ([(float(i), [630.0 - i]) for i in range(4)]
            + [(4.0, [7.2]), (5.0, [6.7])]
            + [(float(i), [626.0 - (i - 6)]) for i in range(6, 10)])
    values = [round(v, 1) for _, v in resolve(rows)]
    assert 7.2 not in values and 6.7 not in values, values
    assert values == [630.0, 629.0, 628.0, 627.0, 626.0, 625.0, 624.0, 623.0]
