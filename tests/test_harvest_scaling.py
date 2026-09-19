"""The dribble harvester's bounds, and the resolution they were tuned at."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "harvest_ball_tracks", ROOT / "scripts" / "harvest_ball_tracks.py")
harvest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harvest)


def _dribble(bounce_px, steps=12):
    """A chain that bounces vertically by `bounce_px`, camera already removed."""
    points = []
    for i in range(steps):
        points.append([i * bounce_px * 0.6,
                       (bounce_px if i % 2 else 0.0)])
    return np.array(points, dtype=float)


def test_a_dribble_tuned_at_720_is_found_at_720():
    assert harvest.dribble_like(_dribble(30.0), frame_height=720.0)


def test_the_same_dribble_filmed_at_1080_is_found_too():
    """A pixel is not a distance -- it is a distance divided by the frame's
    height. The same physical dribble on a 1080-line source is half again as
    many pixels in every bound, so it exceeded MAX_ACCEL_PX and its chain was
    thrown away. Thirty minutes of the 1080p broadcast yielded 2 chains where a
    720p one yields dozens."""
    at_1080 = _dribble(30.0 * 1.5)
    assert harvest.dribble_like(at_1080, frame_height=1080.0)


def test_and_would_have_been_rejected_by_the_unscaled_bounds():
    """The bug, kept as a test so the fix cannot be undone quietly."""
    at_1080 = _dribble(30.0 * 1.5)
    assert not harvest.dribble_like(at_1080, frame_height=720.0)


def test_a_stationary_object_with_jitter_is_not_a_dribble():
    """Detector jitter on a head must not qualify at any resolution."""
    rng = np.random.default_rng(0)
    jitter = rng.normal(0.0, 1.5, (12, 2))
    for height in (720.0, 1080.0):
        assert not harvest.dribble_like(jitter, frame_height=height)


def test_a_bigger_frame_demands_a_bigger_bounce():
    """Scaling must not make the test EASIER at higher resolution, or a 720p
    head would start qualifying on a 1080p broadcast."""
    small = _dribble(20.0)
    assert harvest.dribble_like(small, frame_height=720.0)
    assert not harvest.dribble_like(small, frame_height=1080.0)


def test_the_scale_factor_is_the_ratio_of_heights():
    assert harvest.scaled(18.0, 1080.0) == 27.0
    assert harvest.scaled(18.0, 720.0) == 18.0
