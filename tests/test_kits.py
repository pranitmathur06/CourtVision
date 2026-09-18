"""The kit model, and the sport rule that scores it without labels."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.kits import (
    KitModel,
    MIN_BOX_H,
    over_five,
    rows_spread,
    split_report,
    torso_lab,
)


def _samples(seed: int = 0):
    """Two kits and a smaller grey cluster, the shape a real broadcast has."""
    rng = np.random.default_rng(seed)
    return np.vstack([
        rng.normal([185, 128, 128], 4.0, (200, 3)),   # a white kit
        rng.normal([70, 150, 110], 4.0, (200, 3)),    # a dark kit
        rng.normal([125, 128, 128], 3.0, (60, 3)),    # the officials
    ])


def test_fit_finds_the_two_kits_and_leaves_the_officials_out():
    model = KitModel.fit(_samples())
    assert model is not None
    assert model.kit([185, 128, 128])[0] is not None
    assert model.kit([70, 150, 110])[0] is not None
    assert model.kit([185, 128, 128])[0] != model.kit([70, 150, 110])[0]
    # The stripes are a class, not a low-confidence kit.
    assert model.kit([125, 128, 128])[0] is None


def test_which_kit_is_which_is_stable_across_fits():
    one = KitModel.fit(_samples(), seed=0)
    two = KitModel.fit(_samples(), seed=7)
    assert one.kit([185, 128, 128])[0] == two.kit([185, 128, 128])[0]


def test_separation_is_small_when_both_teams_wear_the_same_colour():
    rng = np.random.default_rng(1)
    same = np.vstack([rng.normal([150, 128, 128], 3.0, (300, 3)),
                      rng.normal([120, 128, 128], 3.0, (60, 3))])
    model = KitModel.fit(same)
    assert model.separation() < 40.0


def test_a_missing_colour_declines_rather_than_guessing():
    model = KitModel.fit(_samples())
    assert model.kit(None) == (None, 0.0)


def test_fit_refuses_too_few_samples():
    assert KitModel.fit(np.zeros((10, 3))) is None


def test_over_five_counts_only_what_the_sport_forbids():
    # Six of one kit is impossible; five and five, and five and four, are not.
    got = over_five([(5, 5), (5, 4), (6, 3), (2, 7)])
    assert got["n"] == 4
    assert got["violations"] == 2
    assert got["rate"] == pytest.approx(0.5)


def test_over_five_on_nothing_says_nothing():
    assert over_five([])["n"] == 0


def test_calling_everybody_one_kit_fails_the_rule_outright():
    assert over_five([(8, 0), (9, 0), (7, 0)])["rate"] == 1.0


def test_split_report_is_the_diagnostic_it_says_it_is():
    got = split_report([5, 5, 4, 3])
    assert got["exact"] == pytest.approx(0.5)
    assert got["within_one"] == pytest.approx(0.75)


def test_torso_of_a_box_too_small_to_read_is_none():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert torso_lab(image, [10, 10, 30, 10 + MIN_BOX_H - 1]) is None


def test_torso_reads_the_jersey_and_not_the_court():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    image[:, :] = (20, 200, 20)                 # the floor, bright green
    image[120:220, 140:260] = (200, 20, 20)     # the torso, blue in BGR
    colour = torso_lab(image, [100, 60, 300, 300])
    floor = torso_lab(np.full((400, 400, 3), (20, 200, 20), dtype=np.uint8),
                      [100, 60, 300, 300])
    assert np.linalg.norm(colour - floor) > 40.0


def test_rows_spread_covers_the_clip_and_never_over_runs_it():
    rows = [{"f": i} for i in range(90)]
    picked = rows_spread(rows, 5)
    assert len(picked) == 5
    assert picked[0]["f"] == 0
    assert all(row in rows for row in picked)
    assert rows_spread(rows[:3], 5) == rows[:3]


def test_a_spectator_wears_neither_kit_nor_stripes():
    """The front row is what eroding the court's edge exists to remove, and
    erosion removes the baseline corner with it. Colour is the same exclusion
    without the collateral."""
    model = KitModel.fit(_samples())
    assert model.belongs_on_court([185, 128, 128], 26.0)      # a white kit
    assert model.belongs_on_court([70, 150, 110], 26.0)       # a dark kit
    assert model.belongs_on_court([125, 128, 128], 26.0)      # an official
    assert not model.belongs_on_court([120, 60, 200], 26.0)   # somebody else


def test_a_torso_too_small_to_read_is_not_evidence_of_a_spectator():
    """Declining to exclude is the safe direction for a filter whose failure
    deletes players."""
    model = KitModel.fit(_samples())
    assert model.belongs_on_court(None, 26.0)


def test_a_looser_gate_never_excludes_more():
    model = KitModel.fit(_samples())
    colour = [120, 60, 200]
    assert model.belongs_on_court(colour, 200.0)
    assert not model.belongs_on_court(colour, 5.0)
