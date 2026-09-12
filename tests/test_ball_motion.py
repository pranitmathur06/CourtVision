"""The motion measure, on warps where the answer is arithmetic."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_motion as motion  # noqa: E402


def test_a_still_object_has_no_motion():
    """A spectator's head lands where it already was, once the camera is undone."""
    point = [500.0, 300.0]
    neighbours = [(-0.2, np.array([[500.0, 300.0]])), (0.2, np.array([[500.5, 299.5]]))]
    speed = motion.motion_of(point, neighbours)
    assert speed is not None and speed < motion.STILL_PX_PER_S


def test_a_moving_object_has_motion():
    point = [500.0, 300.0]
    neighbours = [(-0.2, np.array([[460.0, 300.0]])), (0.2, np.array([[540.0, 300.0]]))]
    speed = motion.motion_of(point, neighbours)
    assert speed is not None and speed > 100.0


def test_an_unmatched_candidate_reports_nothing_rather_than_a_guess():
    point = [500.0, 300.0]
    neighbours = [(-0.2, np.array([[10.0, 10.0]])), (0.2, np.array([[20.0, 20.0]]))]
    assert motion.motion_of(point, neighbours) is None


def test_a_candidate_with_no_neighbours_at_all_reports_nothing():
    assert motion.motion_of([1.0, 1.0], []) is None


def test_the_warp_carries_points_through_the_homography():
    shift = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, -10.0], [0.0, 0.0, 1.0]])
    out = motion.warp([[100.0, 100.0]], shift)
    assert np.allclose(out, [[120.0, 90.0]])


def test_a_refused_hop_warps_nothing_rather_than_inventing_positions():
    assert len(motion.warp([[1.0, 2.0]], None)) == 0


def test_one_wild_neighbour_does_not_drag_the_speed():
    """The median is used because a single bad match is not a trajectory."""
    point = [500.0, 300.0]
    neighbours = [(-0.4, np.array([[500.0, 300.0]])),
                  (-0.2, np.array([[501.0, 300.0]])),
                  (0.2, np.array([[589.0, 300.0]]))]
    assert motion.motion_of(point, neighbours) < motion.STILL_PX_PER_S * 2
