"""The rule that turns 'a basket was made' into a ball label."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_ball_dataset as data  # noqa: E402


def test_the_candidate_at_the_rim_is_the_ball():
    rim, width = [500.0, 300.0], 40.0
    cands = [[900.0, 100.0], [512.0, 305.0], [200.0, 600.0]]
    assert data.pick_ball(cands, rim, width) == 1


def test_nothing_near_the_rim_labels_nothing():
    """A made basket does not guarantee the detector saw the ball."""
    rim, width = [500.0, 300.0], 40.0
    assert data.pick_ball([[900.0, 100.0]], rim, width) is None


def test_two_things_by_the_rim_label_neither():
    """A hand and a ball at the rim are not worth guessing between."""
    rim, width = [500.0, 300.0], 40.0
    cands = [[508.0, 300.0], [516.0, 302.0]]
    assert data.pick_ball(cands, rim, width) is None


def test_a_clear_winner_beside_a_far_runner_up_is_taken():
    rim, width = [500.0, 300.0], 40.0
    cands = [[505.0, 300.0], [600.0, 300.0]]
    assert data.pick_ball(cands, rim, width) == 0


def test_no_candidates_and_no_rim_are_both_refused():
    assert data.pick_ball([], [1.0, 1.0], 40.0) is None
    assert data.pick_ball([[1.0, 1.0]], None, 40.0) is None
