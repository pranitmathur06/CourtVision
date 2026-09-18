"""The statistics this project scores with, in one place.

`wilson` had four copies -- `eval_possession.py:46`, `eval_handler.py:29`,
`eval_possession_temporal.py:39`, `eval_ball_selection.py:33` -- and `iou` had
more. They did NOT agree: three answered an empty denominator with (0, 0) and
the fourth with (0, 1), which is the difference between "measured, and zero" and
"nothing measured". Consolidating them is what made that show up, and the fourth
was right. A scorer whose interval drifts from another scorer's is a scorer
whose numbers cannot be compared, and comparing numbers is this project's whole
method.

WHY THESE THREE AND NOT A LIBRARY. Wilson because a normal approximation is
wrong at the sample sizes here -- at n=13 it puts an interval outside [0, 1].
Exact McNemar because the chi-square version is not trustworthy on twenty-odd
discordant pairs, which is what a 157-frame set produces. A cluster bootstrap
because boxes within a frame are strongly correlated and a Wilson interval on
box counts is too narrow by roughly the square root of the boxes per frame.

AND WHY PAIRED TESTS AT ALL. From `docs/v8-possession-kernels.md`: "on 157 frames
an interval is about 8 points wide either side and the honest effects in this
project are 5 to 10, so unpaired intervals cannot resolve them and the paired
test is not a nicety." Two methods answering the SAME frames are compared on the
frames where they disagree, and nowhere else.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion, correct at small n.

    AN EMPTY DENOMINATOR ANSWERS (0, 1), NOT (0, 0). Nothing measured is not a
    claim of zero -- it is a claim of no idea, and the interval that says so is
    the whole width. Three of the four copies this replaced returned (0, 0),
    which reports a class with no instances as a confident failure; the fourth
    returned (0, 1) and had a test named
    `test_nothing_measured_is_not_a_claim_of_zero` explaining why. Consolidating
    them is what made the four disagree out loud.

    It matters more now than it did: a per-game report on a NEW broadcast will
    routinely have classes with no instances yet, and every one of them would
    have printed 0%-0%.
    """
    if not total:
        return 0.0, 1.0
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total
                         + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired right/wrong vectors.

    Returns (only a right, only b right, p). Only the disagreements carry
    information: under the null each is a fair coin, so the p-value is a
    binomial tail. Exact rather than chi-square because with twenty-odd
    discordant pairs the approximation is not trustworthy.
    """
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    smaller = min(only_a, only_b)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2.0 ** n)
    return only_a, only_b, min(1.0, 2.0 * tail)


def homogeneity(vectors: Sequence[Sequence[bool]]) -> tuple[float, int, float]:
    """Do these groups share one rate? Returns (statistic, df, p).

    The per-game regression alarm. Three broadcasts of 150 frames each cannot
    each carry an interval tight enough to see a five-point move, but asking
    whether one game's rate DIFFERS from the others is a far more powerful
    question on the same data -- and it is the question "did this game regress"
    actually is.

    IT IS NOT COCHRAN'S Q AND IT WAS CALLED THAT. Cochran's Q compares k
    treatments measured on the SAME subjects -- matched binary outcomes, the
    k-sample generalisation of McNemar. What is being compared here is k
    independent groups: different games, different frames, no pairing. The
    right test is the chi-square test for homogeneity of proportions, which is
    what the arithmetic below has always been. The statistic was correct and the
    name was borrowed from the wrong test, which in a file whose whole argument
    is that paired and unpaired comparisons are different things was not a
    harmless label.

    Each vector is one group's per-item outcome. p is from a chi-square survival
    function computed without scipy, so this module stays importable anywhere.
    """
    groups = [list(v) for v in vectors if len(v)]
    if len(groups) < 2:
        return 0.0, 0, 1.0
    rates = [sum(g) / len(g) for g in groups]
    pooled = sum(sum(g) for g in groups) / sum(len(g) for g in groups)
    if pooled in (0.0, 1.0):
        return 0.0, len(groups) - 1, 1.0
    statistic = sum(len(g) * (r - pooled) ** 2 for g, r in zip(groups, rates))
    statistic /= pooled * (1 - pooled)
    df = len(groups) - 1
    return statistic, df, chi_square_tail(statistic, df)


#: The old name. It was wrong -- see `homogeneity` -- and it is kept only so an
#: older script does not break silently on import.
cochran_q = homogeneity


def chi_square_tail(statistic: float, df: int) -> float:
    """P(X > statistic) for a chi-square with `df` degrees of freedom.

    Written out rather than imported so this module has no scipy dependency and
    can be used by anything, including the parts that run on a bare pod.
    """
    if df <= 0 or statistic <= 0:
        return 1.0
    if df % 2 == 0:                      # even df has a closed form
        half = statistic / 2.0
        term = math.exp(-half)
        total = term
        for k in range(1, df // 2):
            term *= half / k
            total += term
        return min(1.0, total)
    # odd df: regularised upper incomplete gamma by continued fraction
    shape, x = df / 2.0, statistic / 2.0
    if x < shape + 1.0:
        term = 1.0 / shape
        total = term
        for n in range(1, 500):
            term *= x / (shape + n)
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        lower = total * math.exp(-x + shape * math.log(x) - math.lgamma(shape))
        return max(0.0, min(1.0, 1.0 - lower))
    tiny = 1e-300
    b = x + 1.0 - shape
    c, d = 1.0 / tiny, 1.0 / b
    result = d
    for i in range(1, 500):
        an = -i * (i - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        step = d * c
        result *= step
        if abs(step - 1.0) < 1e-14:
            break
    return max(0.0, min(1.0, result * math.exp(
        -x + shape * math.log(x) - math.lgamma(shape))))


def cluster_bootstrap(groups: Sequence[tuple[int, int]], draws: int = 10000,
                      seed: int = 0) -> tuple[float, float]:
    """Interval for a rate whose items are correlated inside each group.

    `groups` is (hits, total) per group -- per frame, for boxes. Resampling
    GROUPS with replacement preserves the correlation; resampling items does
    not, and a Wilson interval over boxes is too narrow by roughly the square
    root of the boxes per frame.
    """
    import numpy as np

    if not groups:
        return 0.0, 0.0
    hits = np.array([g[0] for g in groups], dtype=float)
    totals = np.array([g[1] for g in groups], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(groups)
    rates = []
    for _ in range(draws):
        pick = rng.integers(0, n, size=n)
        denominator = totals[pick].sum()
        if denominator > 0:
            rates.append(hits[pick].sum() / denominator)
    if not rates:
        return 0.0, 0.0
    return (float(np.percentile(rates, 2.5)),
            float(np.percentile(rates, 97.5)))


def block_bootstrap(items: Sequence[tuple[float, bool]], span_s: float,
                    block_s: float = 120.0, draws: int = 1000,
                    seed: int = 0) -> tuple[float, float]:
    """Interval for a rate over a timeline, resampling whole blocks.

    `items` is (time in seconds, right/wrong). Events near each other in a game
    are not independent -- a missed possession costs several of them -- so
    resampling individual events reports an interval far tighter than the truth.
    Round 65 measured that: resampling attempts gave a meaningless 0.375-0.456.
    """
    import numpy as np

    if not items or span_s <= 0:
        return 0.0, 0.0
    blocks: dict[int, list[bool]] = {}
    for when, right in items:
        blocks.setdefault(int(when // block_s), []).append(bool(right))
    keys = sorted(blocks)
    if len(keys) < 2:
        return wilson(sum(1 for _, r in items if r), len(items))
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(draws):
        pick = rng.integers(0, len(keys), size=len(keys))
        drawn = [v for i in pick for v in blocks[keys[i]]]
        if drawn:
            rates.append(sum(drawn) / len(drawn))
    if not rates:
        return 0.0, 0.0
    return (float(np.percentile(rates, 2.5)),
            float(np.percentile(rates, 97.5)))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Overlap of two xyxy boxes, 0 when they do not touch."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0
