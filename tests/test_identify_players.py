"""Linking, kit-to-team mapping, and when a tracklet is named at all."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from identify_players import iou, link, name_tracklets, team_for_clusters  # noqa: E402

ROSTER = {"2": {"Indiana Pacers": "Andrew Nembhard",
                "Oklahoma City Thunder": "Shai Gilgeous-Alexander"},
          "8": {"Oklahoma City Thunder": "Jalen Williams"},
          "43": {"Indiana Pacers": "Pascal Siakam"}}


def test_identical_boxes_overlap_completely():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_disjoint_boxes_do_not_overlap():
    assert iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0


def test_a_player_drifting_keeps_one_tracklet():
    frames = [(0.0, [[0, 0, 10, 20]]), (0.2, [[1, 1, 11, 21]]), (0.4, [[2, 2, 12, 22]])]
    ids = [assigned[0][0] for _, assigned in link(frames)]
    assert len(set(ids)) == 1


def test_a_player_who_jumps_across_the_frame_starts_a_new_tracklet():
    frames = [(0.0, [[0, 0, 10, 20]]), (0.2, [[400, 300, 410, 320]])]
    ids = [assigned[0][0] for _, assigned in link(frames)]
    assert len(set(ids)) == 2


def test_two_players_keep_separate_tracklets():
    frames = [(0.0, [[0, 0, 10, 20], [100, 0, 110, 20]]),
              (0.2, [[1, 0, 11, 20], [101, 0, 111, 20]])]
    first, second = link(frames)
    assert {i for i, _ in first[1]} == {i for i, _ in second[1]}
    assert len({i for i, _ in second[1]}) == 2


def test_a_cluster_is_given_the_team_its_reads_match():
    # Only OKC has an 8, so a cluster reading 8 is OKC however it is coloured.
    got = team_for_clusters({0: Counter({"8": 3, "2": 1})}, ROSTER)
    assert got[0] == "Oklahoma City Thunder"


def test_a_cluster_with_no_usable_reads_gets_no_team():
    assert team_for_clusters({0: Counter({"99": 5})}, ROSTER)[0] is None


def test_the_team_decides_which_player_a_repeated_number_is():
    # #2 is Nembhard on Indiana and Gilgeous-Alexander on OKC.
    votes = {1: Counter({"2": 4})}
    assert name_tracklets(votes, {1: "Indiana Pacers"}, ROSTER)[1] == "Andrew Nembhard"
    assert name_tracklets(votes, {1: "Oklahoma City Thunder"}, ROSTER)[1] \
        == "Shai Gilgeous-Alexander"


def test_one_read_is_not_enough_to_name_anyone():
    assert name_tracklets({1: Counter({"8": 1})}, {1: "Oklahoma City Thunder"}, ROSTER) == {}


def test_a_split_vote_between_two_real_numbers_names_nobody():
    # Both are OKC numbers, so neither is dropped and neither wins.
    votes = {1: Counter({"8": 2, "2": 2})}
    assert name_tracklets(votes, {1: "Oklahoma City Thunder"}, ROSTER) == {}


def test_an_off_roster_misread_is_dropped_rather_than_splitting_the_vote():
    # "7" is nobody's number here, so it is evidence of a misread, not a tie.
    votes = {1: Counter({"8": 2, "7": 2})}
    assert name_tracklets(votes, {1: "Oklahoma City Thunder"}, ROSTER)[1] == "Jalen Williams"


def test_a_tracklet_with_no_team_is_left_unnamed():
    assert name_tracklets({1: Counter({"8": 5})}, {1: None}, ROSTER) == {}
