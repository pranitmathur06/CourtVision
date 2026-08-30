"""Correct for the gap between the training prior and the serving prior.

A classifier trained on class-balanced clips implicitly assumes every class is
equally likely. A game is nothing like that: over 84 minutes BARD labels 180
shots and 83 rebounds, and the overwhelming majority of windows contain no
box-score event at all. Served under that mismatch the model emitted 2,253
rebounds against 83 real ones.

This is the textbook fix and it needs no retraining. If a model learns
p_train(y|x) under prior p_train(y), then the same evidence under a different
prior p_serve(y) is

    p_serve(y|x)  ∝  p_train(y|x) · p_serve(y) / p_train(y)

which in log space is a per-class additive shift. It changes only the
comparison BETWEEN classes, so a window whose evidence is overwhelming stays
where it is while a marginal one moves to the class the prior favours.

The serving prior comes from BARD's own labels for a real game, not from taste.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Actions per 48-minute game, both teams, from BARD's labels for
# chi-vs-tor-0022401223 scaled to a full game, plus the continuous-play classes
# estimated from possession counts. `background` carries the rest: a game is
# mostly nobody doing anything a box score would record.
GAME_RATES: dict[str, float] = {
    "shot": 180.0,
    "rebound": 83.0,
    "steal": 18.0,
    "block": 14.0,
    "other": 130.0,     # free throws, fouls, turnovers
    "pass": 460.0,
    "dribble": 700.0,
    "background": 4000.0,
}


def serving_prior(actions: Sequence[str]) -> list[float]:
    """Normalised expected frequency of each class in continuous play."""
    rates = [max(GAME_RATES.get(a, 1.0), 1e-6) for a in actions]
    total = sum(rates)
    return [r / total for r in rates]


def training_prior(counts: dict[str, int], actions: Sequence[str]) -> list[float]:
    """Normalised frequency of each class in the training set."""
    values = [max(float(counts.get(a, 0)), 1e-6) for a in actions]
    total = sum(values)
    return [v / total for v in values]


def logit_shift(counts: dict[str, int], actions: Sequence[str],
                strength: float = 1.0) -> list[float]:
    """Per-class additive shift taking train-prior logits to serve-prior logits.

    `strength` scales the correction: 0 disables it, 1 applies it in full. It is
    a dial rather than a constant because the serving rates are estimates, and
    over-correcting trades a flood of false rebounds for a silence of missed
    ones. Tune it against scripts/evaluate_game.py, not by eye.
    """
    serve = serving_prior(actions)
    train = training_prior(counts, actions)
    return [strength * (math.log(s) - math.log(t)) for s, t in zip(serve, train)]
