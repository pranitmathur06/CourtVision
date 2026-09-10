"""Per-frame choice of paint evidence, on floors painted two different ways."""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from courtvision.court_register import register_frame  # noqa: E402
from test_court_refine import (_court_error, _perturb, _render,  # noqa: E402
                               _render_painted, _truth)


def test_a_white_lined_floor_is_registered_accurately():
    truth = _truth()
    matrix, info = register_frame(_render(truth), _perturb(truth, 1.5, -1.0, 1.0))
    assert info["refined"], info
    assert _court_error(matrix, truth) < 0.1


def test_a_floor_painted_like_toyota_center_is_registered_with_all_evidence():
    """Black lines and a red key with no lane line: bright evidence cannot see
    it, so the choice must fall to the evidence that can."""
    truth = _truth()
    matrix, info = register_frame(_render_painted(truth), _perturb(truth, 1.5, -1.0, 1.0))
    assert info["refined"], info
    assert info["polarity"] == "all", info["polarity"]
    assert _court_error(matrix, truth) < 0.1


def test_a_floor_with_no_lines_falls_back_to_the_landmark_registration():
    truth = _truth()
    start = _perturb(truth, 1.5, -1.0, 1.0)
    blank = np.full((700, 1000, 3), (150, 190, 215), np.uint8)
    matrix, info = register_frame(blank, start)
    assert not info["refined"]
    assert np.allclose(matrix, start), "the fallback must be the landmark fit, unchanged"
    assert set(info["tried"]) == {"bright", "all"}
