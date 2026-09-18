"""Counting PEOPLE, not boxes, in the floor-mask metric."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "eval_court_mask", ROOT / "scripts" / "eval_court_mask.py")
mask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mask)


def test_two_detectors_on_one_player_are_one_person():
    """`p` and `h` are two models run over the same frame and both draw the
    same players. Counting the rows made 71% of Finals G7 frames "impossible"
    before any mask had a chance to be wrong."""
    same = [[100.0, 100.0, 200.0, 400.0], [104.0, 98.0, 198.0, 402.0]]
    assert mask.distinct(same) == 1


def test_two_players_side_by_side_are_two_people():
    apart = [[100.0, 100.0, 200.0, 400.0], [260.0, 100.0, 360.0, 400.0]]
    assert mask.distinct(apart) == 2


def test_a_brushing_overlap_is_still_two_people():
    """Players stand close. Only a near-coincident box is the same person."""
    touching = [[100.0, 100.0, 200.0, 400.0], [180.0, 100.0, 280.0, 400.0]]
    assert mask.distinct(touching) == 2


def test_nothing_is_nobody():
    assert mask.distinct([]) == 0


def test_the_bigger_box_is_the_one_kept():
    """Deduplication is greedy from the largest box, so the result does not
    depend on the order the detectors happened to emit."""
    one = [[100.0, 100.0, 200.0, 400.0], [110.0, 110.0, 190.0, 390.0]]
    assert mask.distinct(one) == mask.distinct(list(reversed(one))) == 1


def test_thirteen_people_is_possible_and_fourteen_is_not():
    """Ten players and three officials. The rule the whole metric rests on."""
    assert mask.IMPOSSIBLE_ABOVE == 13


def _floor(height: int, key_colour=(40, 40, 190)):
    """A wooden floor with a painted key inside it, at a given resolution."""
    import cv2
    import numpy as np

    width = int(height * 16 / 9)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (30, 30, 30)
    cv2.rectangle(image, (int(width * 0.1), int(height * 0.4)),
                  (int(width * 0.9), int(height * 0.95)), (60, 140, 190), -1)
    cv2.rectangle(image, (int(width * 0.4), int(height * 0.55)),
                  (int(width * 0.6), int(height * 0.8)), key_colour, -1)
    return image


def test_the_mask_is_the_same_shape_at_every_resolution():
    """The closing and the erosion are different physical distances on a 720p
    and a 1080p frame unless the mask is found at a canonical height. Without
    this, a constant swept on one broadcast is meaningless on the other and
    nothing measured on a downscaled clip transfers to the pipeline."""
    from courtvision.candidates import court_region

    shares = []
    for height in (480, 720, 1080):
        region = court_region(_floor(height), erode_px=None, erode_share=0.03)
        assert region.shape[:2] == (height, int(height * 16 / 9))
        shares.append(float(region.mean()))
    assert max(shares) - min(shares) < 0.01


def test_a_red_key_is_kept_because_the_wood_encloses_it():
    """The paint rule accepts hue 95-130, which is blue, and 0.4% of Houston's
    red key. Holes in the wood are filled by shape, so the colour never
    matters."""
    from courtvision.candidates import court_region

    red = court_region(_floor(720, key_colour=(40, 40, 190)), erode_px=0)
    blue = court_region(_floor(720, key_colour=(190, 60, 40)), erode_px=0)
    assert abs(float(red.mean()) - float(blue.mean())) < 0.01


def test_eroding_more_keeps_less():
    from courtvision.candidates import court_region

    image = _floor(720)
    loose = court_region(image, erode_px=None, erode_share=0.0)
    tight = court_region(image, erode_px=None, erode_share=0.0625)
    assert float(tight.mean()) < float(loose.mean())
