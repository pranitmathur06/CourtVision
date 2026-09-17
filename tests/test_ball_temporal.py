"""Choosing the ball over time, and the measured reason it does not help.

The selector in `ball_track.py` assumes "a ball moves smoothly and a false
positive teleports". On the frames where this detector's argmax is wrong that is
backwards: the decoy moves 6.5 px between adjacent frames and the real ball 90.5,
because the frames where the ball is hard to see are the frames where it is in
flight. These tests pin both the scoring and that finding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_ball_temporal import mcnemar, score, wilson  # noqa: E402


def window(frames, truth=(100.0, 100.0), centre=1, game="g"):
    return [{"frames": frames, "centre": centre, "ball": list(truth), "game": game}]


def test_a_confident_candidate_on_the_ball_is_delivered_by_both():
    frames = [[[100.0, 100.0, 0.9]], [[100.0, 100.0, 0.9]], [[100.0, 100.0, 0.9]]]
    oracle, argmax, path = score(window(frames), 28.0, 0.02, 6.0)
    assert oracle[0] and argmax[0] and path[0]


def test_a_ball_nobody_proposed_is_a_miss_for_every_method():
    frames = [[[500.0, 500.0, 0.9]]] * 3
    oracle, argmax, path = score(window(frames), 28.0, 0.02, 6.0)
    assert not oracle[0] and not argmax[0] and not path[0]


def test_the_oracle_only_counts_the_centre_frame():
    """A ball visible either side but not in the labelled frame is not a hit.

    Truth was established on one picture; crediting a method for finding the
    ball in a neighbour would score it against a frame nobody labelled.
    """
    frames = [[[100.0, 100.0, 0.9]], [[400.0, 400.0, 0.9]], [[100.0, 100.0, 0.9]]]
    oracle, _, _ = score(window(frames), 28.0, 0.02, 6.0)
    assert not oracle[0]


def test_a_stationary_decoy_beats_the_real_ball_under_a_smoothness_prior():
    """The measured failure, as a test.

    The decoy sits still and outranks the ball on confidence; the ball is in
    flight and moves 90 px a frame. A prior that rewards not moving prefers the
    decoy, which is why fitting the motion weight drives it to zero.
    """
    decoy = [0.0, 0.0, 0.6]
    frames = [[[10.0, 100.0, 0.5], decoy],
              [[100.0, 100.0, 0.5], decoy],
              [[190.0, 100.0, 0.5], decoy]]
    _, argmax, path = score(window(frames), 28.0, 0.05, 6.0)
    assert not argmax[0], "the decoy should win on confidence"
    assert not path[0], "and a smoothness prior should prefer it too"


def test_turning_the_motion_weight_off_makes_the_path_agree_with_argmax():
    """Which is what the fit chose on all three broadcasts.

    With no motion term the path optimises confidence alone, frame by frame,
    and the Viterbi degenerates into the per-frame argmax it was meant to beat.
    """
    frames = [[[10.0, 100.0, 0.5], [0.0, 0.0, 0.6]],
              [[100.0, 100.0, 0.9], [0.0, 0.0, 0.6]],
              [[190.0, 100.0, 0.5], [0.0, 0.0, 0.6]]]
    _, argmax, path = score(window(frames), 28.0, 0.0, 6.0)
    assert bool(argmax[0]) == bool(path[0])


def test_mcnemar_pairs_and_wilson_brackets():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    only_a, only_b, p = mcnemar(a, b)
    assert (only_a, only_b) == (1, 1) and p == 1.0
    low, high = wilson(78, 130)
    assert low < 78 / 130 < high
