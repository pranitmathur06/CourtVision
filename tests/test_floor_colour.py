"""Learning what colour an arena's floor is, from under the players' feet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.floor_colour import (  # noqa: E402
    MIN_BOX_H,
    MIN_SAMPLES,
    FloorColour,
    collect,
    learn,
    load,
    strip_below,
)


def _court(colour, height=400, width=600):
    """A floor of one colour with a player standing on it."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = colour
    cv2.rectangle(image, (200, 100), (260, 260), (200, 180, 160), -1)
    return image, [200.0, 100.0, 260.0, 260.0]


def test_the_strip_is_below_the_box_and_not_the_shoe():
    image, box = _court((40, 40, 200))
    strip = strip_below(image, box)
    assert strip is not None
    # Every pixel is floor, not the player's box, which ends at y=260.
    assert (strip.reshape(-1, 3) == np.array([40, 40, 200])).all()


def test_a_box_too_small_gives_no_strip():
    image, _ = _court((40, 40, 200))
    assert strip_below(image, [200.0, 100.0, 260.0, 100.0 + MIN_BOX_H - 1]) is None


def test_a_box_at_the_bottom_edge_gives_no_strip():
    image, _ = _court((40, 40, 200))
    assert strip_below(image, [200.0, 200.0, 260.0, 399.0]) is None


def test_a_red_floor_is_learned_and_a_red_floor_is_accepted():
    """The listed rule accepts hue 5-30 and 95-130 -- tan and blue. Toyota
    Center's court is red, and the rule accepts none of it."""
    image, box = _court((40, 40, 200))
    floor = learn(collect(image, [box]) * MIN_SAMPLES)
    assert floor is not None
    assert floor.mask(image)[350, 300]


def test_a_blue_floor_learns_a_different_model():
    red, box = _court((40, 40, 200))
    blue, _ = _court((200, 60, 40))
    red_floor = learn(collect(red, [box]) * MIN_SAMPLES)
    blue_floor = learn(collect(blue, [box]) * MIN_SAMPLES)
    assert not red_floor.mask(blue)[350, 300]
    assert not blue_floor.mask(red)[350, 300]


def test_too_few_samples_answers_none_rather_than_a_bad_model():
    image, box = _court((40, 40, 200))
    assert learn(collect(image, [box]) * 3) is None


def test_a_dark_pixel_is_not_floor_whatever_its_hue():
    image, box = _court((12, 12, 20))
    assert learn(collect(image, [box]) * MIN_SAMPLES) is None


def test_the_model_covers_only_a_little_of_the_colour_space():
    """A model accepting half of colour space would mask a whole frame, and
    `fit_floor_colour.py` refuses one."""
    image, box = _court((40, 40, 200))
    floor = learn(collect(image, [box]) * MIN_SAMPLES)
    assert floor.share() < 0.05


def test_the_tail_of_the_samples_is_dropped():
    """A box drawn round somebody in the front row samples the seats. Those
    are wrong but not systematic -- the crowd is many colours and the floor is
    one -- so keeping only what recurs is what makes this work."""
    floor_sample = np.array([10.0, 200.0, 200.0])
    crowd = [np.array([float(h), 200.0, 200.0]) for h in range(60, 170, 4)]
    floor = learn([floor_sample] * MIN_SAMPLES + crowd)
    assert floor is not None
    kept = floor.bins.any(axis=1)
    # The floor's own hue survives and almost none of the crowd's 28 do. The
    # cut-off keeps 90% of the mass, so a handful of crowd bins ride along --
    # what matters is that the great majority do not.
    assert kept[10 * 36 // 180]
    assert kept.sum() <= 8


def test_an_unfitted_broadcast_answers_none_and_not_an_empty_floor(tmp_path):
    """An arena whose floor has not been learned is not an arena with no
    floor; the caller must fall back to the listed colours."""
    path = tmp_path / "floors.json"
    assert load(path, "hou") is None
    path.write_text(json.dumps({"g7": {"bins": [[1]], "samples": 5}}))
    assert load(path, "hou") is None
    assert load(path, "g7") is not None


def test_a_loaded_floor_masks_what_the_fitted_one_did(tmp_path):
    image, box = _court((40, 40, 200))
    floor = learn(collect(image, [box]) * MIN_SAMPLES)
    path = tmp_path / "floors.json"
    path.write_text(json.dumps({"hou": {"bins": floor.bins.astype(int).tolist(),
                                        "samples": floor.samples}}))
    again = load(path, "hou")
    assert np.array_equal(again.bins, floor.bins)
    assert again.mask(image)[350, 300]


def test_court_region_uses_the_learned_floor_when_given_one():
    from courtvision.candidates import court_region

    image, box = _court((40, 40, 200))
    floor = learn(collect(image, [box]) * MIN_SAMPLES)
    listed = court_region(image, erode_px=0)
    learned = court_region(image, erode_px=0, floor=floor)
    assert learned is not None
    # The listed rule finds no red court at all; the learned one finds it.
    assert float(learned.mean()) > 0.5
    assert listed is None or float(listed.mean()) < 0.2
