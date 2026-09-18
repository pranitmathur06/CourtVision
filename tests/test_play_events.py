"""Rebound and assist truth, and the geometry that answers them from pixels.

The truth derivations get the most attention here. A wrong predictor shows up
as a bad accuracy; a wrong TRUTH shows up as a good one, which is why the first
run of this arm reported 1.000 on both -- a hit counter that counted every row
whether it hit or not.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "eval_play_events", ROOT / "scripts" / "eval_play_events.py")
epe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(epe)


def _rebound(video_s, who, off, dfn):
    return {"video_s": video_s, "action": "Rebound",
            "description": f"{who} REBOUND (Off:{off} Def:{dfn})"}


def test_rebound_type_comes_from_which_counter_moved():
    events = [_rebound(10.0, "Holmgren", 0, 1),
              _rebound(20.0, "Holmgren", 1, 1),
              _rebound(30.0, "Holmgren", 1, 2)]
    assert [kind for _, kind in epe.rebound_truth(events)] == ["def", "off", "def"]


def test_two_players_keep_separate_counters():
    events = [_rebound(10.0, "Sengun", 1, 0), _rebound(11.0, "Dort", 0, 1),
              _rebound(12.0, "Sengun", 1, 1)]
    assert [k for _, k in epe.rebound_truth(events)] == ["off", "def", "def"]


def test_a_team_rebound_is_not_attributed_to_anybody():
    events = [{"video_s": 5.0, "action": "Rebound", "description": "HOU team REBOUND"}]
    assert epe.rebound_truth(events) == []


def test_the_truth_never_needs_a_roster():
    """Two of the four broadcasts have no roster file, and a truth that needed
    one would be a truth that cannot follow a new broadcast. The counters in
    the feed's own description are enough, so the signature takes the events
    and nothing else."""
    import inspect

    assert list(inspect.signature(epe.rebound_truth).parameters) == ["events"]
    assert list(inspect.signature(epe.assist_truth).parameters) == ["events"]


def test_assist_truth_pairs_the_assist_row_with_its_basket():
    events = [
        {"video_s": 703.0, "action": "Made Shot (2PT)", "description": "Dort 13'"},
        {"video_s": 703.0, "action": "Assist", "description": "Dort 13' (1 AST)"},
        {"video_s": 902.0, "action": "Made Shot (2PT)", "description": "Smith Jr. 13'"},
    ]
    assert [truth for _, truth in epe.assist_truth(events)] == [True, False]


def test_free_throws_are_not_counted_as_baskets():
    events = [{"video_s": 1.0, "action": "Free Throw (made)", "description": "x"}]
    assert epe.assist_truth(events) == []


def test_an_index_row_with_no_footage_is_not_coverage():
    rows = [{"clip": None, "start_s": 100.0}, {"clip": "h001030.mp4", "start_s": 100.0}]
    row, offset = epe.clip_for(rows, 103.0, 6.0)
    assert row["clip"] == "h001030.mp4"
    assert offset == pytest.approx(3.0)


def test_the_clip_with_the_instant_furthest_from_its_edges_wins():
    """A rebound one frame from a clip's end has no securing player in it."""
    rows = [{"clip": "a.mp4", "start_s": 97.5}, {"clip": "b.mp4", "start_s": 100.0}]
    row, _ = epe.clip_for(rows, 103.0, 6.0)
    assert row["clip"] == "b.mp4"


def test_no_clip_covers_an_instant_outside_them_all():
    rows = [{"clip": "a.mp4", "start_s": 100.0}]
    assert epe.clip_for(rows, 500.0, 6.0) == (None, None)


def test_distance_to_a_box_is_zero_inside_it():
    assert epe.to_box((50.0, 50.0), [0.0, 0.0, 100.0, 100.0]) == 0.0
    assert epe.to_box((110.0, 50.0), [0.0, 0.0, 100.0, 100.0]) == pytest.approx(10.0)


class _AlwaysKit:
    """A kit model that answers from the x coordinate handed to it."""

    def __init__(self, kits):
        self._kits = kits

    def kit(self, colour):
        return (self._kits.get(colour), 1.0) if colour in self._kits else (None, 0.0)


class _Frames:
    """Stands in for a decoded clip: the colour IS the box's left edge."""

    def colour(self, frame, box):
        return box[0]

    def close(self):
        pass


def _row(f, ball_xy, people):
    detections = [["b", 0.9, ball_xy[0] - 5, ball_xy[1] - 5,
                   ball_xy[0] + 5, ball_xy[1] + 5]]
    detections += [["p", 0.9, *box] for box in people]
    return {"f": f, "d": detections}


def test_a_ball_far_from_everybody_has_no_holder():
    rows = [_row(0, (900.0, 900.0), [[0.0, 0.0, 100.0, 300.0]])]
    assert epe.holders(rows, _Frames(), _AlwaysKit({0.0: 0}), 0, 10) == []


def test_the_holder_is_the_nearest_box_edge_and_carries_its_kit():
    rows = [_row(0, (110.0, 100.0), [[0.0, 0.0, 100.0, 300.0],
                                     [400.0, 0.0, 500.0, 300.0]])]
    got = epe.holders(rows, _Frames(), _AlwaysKit({0.0: 1, 400.0: 0}), 0, 10)
    assert len(got) == 1 and got[0][1] == 1


def test_an_official_holding_the_ball_is_declined_not_guessed():
    rows = [_row(0, (110.0, 100.0), [[0.0, 0.0, 100.0, 300.0]])]
    assert epe.holders(rows, _Frames(), _AlwaysKit({}), 0, 10) == []


def test_a_rebound_by_the_shooting_team_reads_offensive():
    rows = [_row(0, (110.0, 100.0), [[0.0, 0.0, 100.0, 300.0]]),
            _row(60, (110.0, 100.0), [[0.0, 0.0, 100.0, 300.0]])]
    said, _ = epe.judge_rebound(rows, _Frames(), _AlwaysKit({0.0: 1}),
                                at_frame=60, fps=30.0, shot_frame=0)
    assert said == "off"


def test_a_rebound_by_the_other_kit_reads_defensive():
    rows = [_row(0, (110.0, 100.0), [[0.0, 0.0, 100.0, 300.0]]),
            _row(60, (510.0, 100.0), [[400.0, 0.0, 500.0, 300.0]])]
    said, _ = epe.judge_rebound(rows, _Frames(), _AlwaysKit({0.0: 1, 400.0: 0}),
                                at_frame=60, fps=30.0, shot_frame=0)
    assert said == "def"


def test_one_man_holding_it_the_whole_way_is_not_an_assist():
    rows = [_row(f, (110.0, 100.0), [[0.0, 0.0, 100.0, 300.0]])
            for f in range(0, 90, 15)]
    said, _ = epe.judge_assist(rows, _Frames(), _AlwaysKit({0.0: 1}),
                               at_frame=90, fps=30.0)
    assert said is False


def test_a_pass_between_teammates_reads_as_an_assist():
    rows = [_row(0, (810.0, 100.0), [[800.0, 0.0, 900.0, 300.0]]),
            _row(60, (110.0, 100.0), [[0.0, 0.0, 100.0, 300.0]])]
    said, _ = epe.judge_assist(rows, _Frames(), _AlwaysKit({0.0: 1, 800.0: 1}),
                               at_frame=60, fps=30.0)
    assert said is True


def test_a_pass_from_the_other_kit_is_not_an_assist():
    rows = [_row(0, (810.0, 100.0), [[800.0, 0.0, 900.0, 300.0]]),
            _row(60, (110.0, 100.0), [[0.0, 0.0, 100.0, 300.0]])]
    said, _ = epe.judge_assist(rows, _Frames(), _AlwaysKit({0.0: 1, 800.0: 0}),
                               at_frame=60, fps=30.0)
    assert said is False


def test_a_vote_beats_the_first_frame_when_the_first_frame_is_wrong():
    """`holders` returns one attribution per frame, each right about two times
    in three. Taking the first inherits that rate; taking the majority does
    not, and this is the cheapest variance reduction in the file."""
    seen = [(0, 1, None, None), (4, 0, None, None), (8, 0, None, None),
            (12, 0, None, None)]
    assert seen[0][1] == 1
    assert epe.vote(seen) == (0, pytest.approx(0.75))


def test_a_vote_on_nothing_declines():
    assert epe.vote([]) == (None, 0.0)


def test_the_anchored_arm_declines_without_a_map():
    rows = [_row(60, (110.0, 100.0), [[0.0, 0.0, 100.0, 300.0]])]
    said, why = epe.judge_rebound_anchored(rows, _Frames(), _AlwaysKit({0.0: 1}),
                                           at_frame=60, fps=30.0, shot_frame=0,
                                           anchor={}, shooting_team="OKC")
    assert said is None and why == "no anchor"


def test_the_anchored_arm_needs_one_attribution_not_two():
    """The old arm asked vision who shot AND who rebounded, so a 0.66
    attribution entered the answer twice. Here the feed supplies the shooting
    team and vision supplies only the rebounder."""
    rows = [_row(60, (110.0, 100.0), [[0.0, 0.0, 100.0, 300.0]])]
    model = _AlwaysKit({0.0: 1})
    anchor = {1: "HOU", 0: "OKC"}
    assert epe.judge_rebound_anchored(rows, _Frames(), model, 60, 30.0, 0,
                                      anchor, "HOU")[0] == "off"
    assert epe.judge_rebound_anchored(rows, _Frames(), model, 60, 30.0, 0,
                                      anchor, "OKC")[0] == "def"


def test_a_kit_missing_from_the_anchor_declines_rather_than_guessing():
    rows = [_row(60, (110.0, 100.0), [[0.0, 0.0, 100.0, 300.0]])]
    said, why = epe.judge_rebound_anchored(rows, _Frames(), _AlwaysKit({0.0: 1}),
                                           60, 30.0, 0, {0: "OKC"}, "OKC")
    assert said is None and why == "kit not in the anchor"


def test_team_codes_joins_the_play_by_play_by_description():
    """563 of 563 on the held-out broadcast, which is why the join is on the
    description and not on a reconstructed clock."""
    from courtvision.games import get

    broadcast = get("hou")
    codes = epe.team_codes(broadcast)
    assert codes
    assert set(codes.values()) == {"OKC", "HOU"}
