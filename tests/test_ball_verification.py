"""Merging proposals, centring the crop, and scoring an endorsement."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ball_verification import (  # noqa: E402
    WINDOW_PX, best_endorsement, crop_window, merge, ranked,
)


def test_duplicate_proposals_collapse_to_the_most_confident():
    merged = merge([(100.0, 100.0, 0.2), (104.0, 103.0, 0.7)])
    assert merged == [(104.0, 103.0, 0.7)]


def test_distinct_proposals_both_survive():
    assert len(merge([(100.0, 100.0, 0.2), (400.0, 300.0, 0.1)])) == 2


def test_merging_keeps_the_best_first():
    merged = merge([(0.0, 0.0, 0.1), (500.0, 0.0, 0.9)])
    assert merged[0][2] == 0.9


def test_nothing_in_nothing_out():
    assert merge([]) == []


def test_the_crop_is_centred_on_the_candidate():
    x0, y0, w, h = crop_window((720, 1280, 3), (640, 360))
    assert x0 + w // 2 == 640 and y0 + h // 2 == 360


def test_the_crop_is_clipped_at_the_frame_edge_not_run_off_it():
    x0, y0, w, h = crop_window((720, 1280, 3), (10, 10))
    assert x0 == 0 and y0 == 0
    assert x0 + w <= 1280 and y0 + h <= 720


def test_a_frame_shorter_than_the_window_still_yields_a_crop():
    x0, y0, w, h = crop_window((480, 600, 3), (300, 240))
    assert w <= 600 and h <= 480 and x0 >= 0 and y0 >= 0


def test_the_window_is_the_size_the_crops_were_cut_at():
    # The whole argument: a centred 640 px crop is the training distribution.
    assert WINDOW_PX == 640


def test_a_detection_on_the_candidate_endorses_it():
    assert best_endorsement([(100.0, 100.0, 0.8)], (104.0, 100.0)) == 0.8


def test_a_detection_elsewhere_in_the_crop_endorses_nothing():
    assert best_endorsement([(300.0, 300.0, 0.9)], (100.0, 100.0)) == 0.0


def test_no_detection_scores_zero_rather_than_being_skipped():
    assert best_endorsement([], (100.0, 100.0)) == 0.0


def test_the_strongest_endorsement_on_the_candidate_wins():
    detections = [(101.0, 100.0, 0.3), (99.0, 101.0, 0.75)]
    assert best_endorsement(detections, (100.0, 100.0)) == 0.75


def test_ranking_puts_the_best_score_first():
    assert ranked([0, 1, 2], [0.1, 0.9, 0.4]) == [1, 2, 0]


def test_an_unendorsed_field_keeps_a_stable_order():
    assert ranked([0, 1, 2], [0.0, 0.0, 0.0]) == [0, 1, 2]
