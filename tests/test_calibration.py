import math

from courtvision.calibration import (GAME_RATES, logit_shift, serving_prior,
                                     training_prior)


ACTIONS = ("dribble", "pass", "shot", "rebound", "block", "steal", "other",
           "background")


def test_priors_are_normalised():
    assert math.isclose(sum(serving_prior(ACTIONS)), 1.0, rel_tol=1e-9)
    counts = {a: 400 for a in ACTIONS}
    assert math.isclose(sum(training_prior(counts, ACTIONS)), 1.0, rel_tol=1e-9)


def test_rare_at_serve_but_common_in_training_is_pushed_down():
    """Rebound is 436 of ~4100 training clips but a rare event in a game.

    This is exactly the mismatch that produced 2,253 rebounds against 83 real
    ones, so its shift must be negative.
    """
    counts = {a: 436 for a in ACTIONS}
    shift = dict(zip(ACTIONS, logit_shift(counts, ACTIONS)))
    assert shift["rebound"] < 0
    assert shift["background"] > 0, "ordinary play is common at serve, rare in training"
    assert shift["rebound"] < shift["shot"], "shots outnumber rebounds in a game"


def test_strength_zero_is_a_no_op():
    counts = {a: 436 for a in ACTIONS}
    assert all(abs(v) < 1e-12 for v in logit_shift(counts, ACTIONS, strength=0.0))


def test_strength_scales_linearly():
    counts = {a: 436 for a in ACTIONS}
    full = logit_shift(counts, ACTIONS, strength=1.0)
    half = logit_shift(counts, ACTIONS, strength=0.5)
    assert all(math.isclose(h, f / 2, rel_tol=1e-9) for h, f in zip(half, full))


def test_a_class_absent_from_training_does_not_divide_by_zero():
    counts = {a: 436 for a in ACTIONS if a != "block"}
    shift = logit_shift(counts, ACTIONS)
    assert all(math.isfinite(v) for v in shift)


def test_every_action_has_a_declared_game_rate():
    """A missing rate silently defaults to 1.0 and would distort the prior."""
    for action in ACTIONS:
        assert action in GAME_RATES, f"{action} has no declared game rate"
