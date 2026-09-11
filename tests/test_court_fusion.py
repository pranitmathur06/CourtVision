"""Temporal fusion, on registrations whose truth is known."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from courtvision.court_fusion import chain, fuse  # noqa: E402
from tests.test_court_refine import _court_error, _perturb, _truth  # noqa: E402

SIZE = (1000, 700)


def test_a_lock_held_by_a_minority_is_outvoted():
    truth = _truth()
    good = [_perturb(truth, dx, dy, 0.0) for dx, dy in ((0.05, 0), (-0.05, 0.03), (0, -0.04), (0.02, 0.02))]
    locked = [_perturb(truth, 3.0, 0.0, 0.0), _perturb(truth, 3.1, 0.2, 0.5)]
    fused, spread = fuse(good + locked, SIZE)
    assert _court_error(fused, truth) < 0.1
    assert _court_error(locked[0], truth) > 2.5


def test_too_few_candidates_fuse_nothing():
    truth = _truth()
    assert fuse([truth, truth], SIZE) == (None, None)


def test_hops_compose_in_order_and_stop_at_a_failed_hop():
    a = np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1.0]])
    b = np.array([[1, 0, 0], [0, 1, 7], [0, 0, 1.0]])
    out = chain([a, b, None, a])
    assert len(out) == 2
    assert np.allclose(out[1], b @ a)
