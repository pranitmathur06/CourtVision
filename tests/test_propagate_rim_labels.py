"""The gates that stop ORB propagation writing rubbish labels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from propagate_rim_labels import (  # noqa: E402
    acceptable, carried_box, nearby, shares_a_shot, supported_at_the_rim,
)


def test_identity_carries_the_box_unchanged():
    box = carried_box(np.eye(3), (100.0, 80.0), 40.0)
    assert box == pytest.approx((100.0, 80.0, 40.0, 40.0), abs=1e-6)


def test_a_translation_moves_it_and_keeps_its_size():
    homography = np.array([[1, 0, 30.0], [0, 1, -12.0], [0, 0, 1]])
    cx, cy, w, h = carried_box(homography, (100.0, 80.0), 40.0)
    assert (cx, cy) == pytest.approx((130.0, 68.0))
    assert (w, h) == pytest.approx((40.0, 40.0))


def test_a_scaling_carries_the_width_too():
    homography = np.diag([2.0, 2.0, 1.0])
    _, _, w, h = carried_box(homography, (100.0, 80.0), 40.0)
    assert (w, h) == pytest.approx((80.0, 80.0))


def test_a_sliver_is_refused():
    # Squashed 20:1 in y: the ring has become a line and is not a label.
    assert carried_box(np.diag([1.0, 0.05, 1.0]), (100.0, 80.0), 40.0) is None


def test_a_crossed_quadrilateral_is_refused():
    # The projective denominator changes sign across the box, so the corners
    # come back turned inside out. Centred on x=0 with a half-width of 150,
    # 1 + 0.01x runs from -0.5 to 2.5 and passes through zero on the way.
    homography = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.01, 0.0, 1.0]])
    assert carried_box(homography, (0.0, 80.0), 300.0) is None


def test_a_projectively_squashed_box_is_refused_as_unacceptable():
    # Not crossed -- just shrunk to a few pixels by a strong projective term,
    # which is what a slipped fit usually produces rather than a bow-tie.
    homography = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.02, 0.0, 1.0]])
    box = carried_box(homography, (100.0, 80.0), 40.0)
    assert not acceptable(box, 40.0, (720, 1280, 3))


def test_a_degenerate_homography_is_refused():
    assert carried_box(np.zeros((3, 3)), (100.0, 80.0), 40.0) is None


def test_none_in_none_out():
    assert carried_box(None, (100.0, 80.0), 40.0) is None


def test_a_box_off_the_edge_is_not_acceptable():
    assert not acceptable((-5.0, 80.0, 40.0, 40.0), 40.0, (720, 1280, 3))
    assert not acceptable((100.0, 900.0, 40.0, 40.0), 40.0, (720, 1280, 3))


def test_a_box_that_changed_size_absurdly_is_not_acceptable():
    assert not acceptable((100.0, 80.0, 400.0, 400.0), 40.0, (720, 1280, 3))
    assert not acceptable((100.0, 80.0, 4.0, 4.0), 40.0, (720, 1280, 3))


def test_a_sane_carried_box_is_acceptable():
    assert acceptable((300.0, 200.0, 52.0, 48.0), 40.0, (720, 1280, 3))


def test_only_evaluation_frames_within_reach_are_worth_testing():
    held = np.array([10.0, 100.0, 500.0])
    assert nearby(100.0, held, reach=45.0) == [100.0]
    assert sorted(nearby(60.0, held, reach=45.0)) == [100.0]
    assert nearby(300.0, held, reach=45.0) == []


def test_nothing_is_nearby_when_no_evaluation_file_was_given():
    assert nearby(123.0, np.array([])) == []


def test_a_frame_registering_seconds_away_shares_its_take():
    rng = np.random.default_rng(0)
    picture = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    # The same picture shifted a little, two seconds apart: one take.
    shifted = np.roll(picture, 7, axis=1)
    assert shares_a_shot(shifted, [(picture, 2.0)])


def test_the_same_fixed_camera_much_later_is_a_different_take():
    # The heart of the correction. These cameras are bolted to the building,
    # so registration alone would exclude the whole game's worth of them.
    rng = np.random.default_rng(0)
    picture = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    shifted = np.roll(picture, 7, axis=1)
    assert not shares_a_shot(shifted, [(picture, 900.0)])


def test_an_unrelated_frame_does_not_share_a_take_even_seconds_away():
    rng = np.random.default_rng(0)
    assert not shares_a_shot(
        rng.integers(0, 255, (240, 320), dtype=np.uint8),
        [(rng.integers(0, 255, (240, 320), dtype=np.uint8), 1.0)])


def test_a_bare_frame_with_no_time_keeps_the_stricter_behaviour():
    rng = np.random.default_rng(0)
    picture = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    assert shares_a_shot(np.roll(picture, 7, axis=1), [picture])


def test_nothing_to_compare_against_shares_nothing():
    rng = np.random.default_rng(0)
    assert not shares_a_shot(rng.integers(0, 255, (240, 320), dtype=np.uint8), [])


def test_a_take_boundary_is_where_it_was_declared():
    from propagate_rim_labels import SAME_TAKE_S
    rng = np.random.default_rng(0)
    picture = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    shifted = np.roll(picture, 7, axis=1)
    assert shares_a_shot(shifted, [(picture, SAME_TAKE_S - 0.1)])
    assert not shares_a_shot(shifted, [(picture, SAME_TAKE_S + 0.1)])


def test_inliers_clustered_on_the_ring_are_support():
    points = np.array([[100.0 + dx, 80.0 + dy]
                       for dx in (-10, 0, 10) for dy in (-10, 0, 10)])
    assert supported_at_the_rim(points, (100.0, 80.0), 40.0)


def test_inliers_far_from_the_ring_are_not_support():
    # A fit held up entirely by the scoreboard and the crowd.
    points = np.array([[900.0, 40.0], [1000.0, 60.0], [1100.0, 30.0],
                       [950.0, 700.0], [1010.0, 690.0], [880.0, 660.0]])
    assert not supported_at_the_rim(points, (100.0, 80.0), 40.0)


def test_a_couple_of_nearby_inliers_are_not_enough():
    points = np.array([[100.0, 80.0], [104.0, 84.0]])
    assert not supported_at_the_rim(points, (100.0, 80.0), 40.0)


def test_no_inliers_at_all_is_not_support():
    assert not supported_at_the_rim(np.zeros((0, 2)), (100.0, 80.0), 40.0)
    assert not supported_at_the_rim(None, (100.0, 80.0), 40.0)


def test_a_bare_frame_is_treated_as_coincident():
    # Callers holding only pictures keep the stricter behaviour rather than
    # silently skipping the time test.
    rng = np.random.default_rng(0)
    picture = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    assert shares_a_shot(np.roll(picture, 7, axis=1), [picture])


def test_a_registering_frame_far_apart_in_time_is_a_different_take():
    # The same fixed camera, a quarter of an hour later. It registers, and it
    # is not the same take -- this is the case that wrongly dropped 156 of 196
    # hand labels when the caller passed bare frames.
    rng = np.random.default_rng(0)
    picture = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    assert not shares_a_shot(np.roll(picture, 7, axis=1), [(picture, 900.0)])


def test_a_registering_frame_seconds_away_is_the_same_take():
    rng = np.random.default_rng(0)
    picture = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    assert shares_a_shot(np.roll(picture, 7, axis=1), [(picture, 3.0)])
