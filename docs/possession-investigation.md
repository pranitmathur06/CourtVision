# V6 possession: what was tried, measured, and rejected

Spec §10 says possession is "a proximity heuristic, not ground truth... the goal
is mostly right, not perfect." This records how far that goes on real NBA
broadcast footage, so nobody repeats the dead ends.

## The core difficulty, measured

On the sample clip, over frames where the ball is detected and ≥2 players are present:

| quantity | value |
|---|---|
| median distance, nearest player to ball | 0.44 body-heights |
| median distance, runner-up | 0.87 |
| **median separation between them** | **0.21** |
| frames where runner-up is within 0.3 | **61%** |

A defender is about as close to the ball as the ball-handler in most frames.
Proximity is therefore *under-determined*, not merely noisy.

## Approaches tried

| approach | result | verdict |
|---|---|---|
| Centre-to-centre proximity | 61% of frames ambiguous | baseline |
| **Box-edge distance** | 69% ambiguous — *worse* | rejected |
| Box containment | resolves exactly one player in 25/67 frames | insufficient |
| **Ball-motion correlation** | 0/2 where proximity scored 2/2 | rejected |
| Learned `handler`, 191 instances | 1/4 on the answer key vs 3/4 for proximity | rejected |
| Learned `handler`, 680 instances | fires 49% of frames @0.85 conf (was 24% @0.55) | improved |
| Parameter sweep over distance/hysteresis/gap | **5/7, no configuration anywhere beats it** | ceiling |

**Why motion correlation failed:** the ball is detected in only ~63% of frames,
so a displacement window rarely has the ball at both endpoints. Widening the
window to find frames that do have it spans too much time to be discriminative.

**Why the parameter sweep plateaus:** the two remaining failures pull in opposite
directions. One needs a *looser* distance threshold to catch a real handler; the
other needs a *tighter* one to reject a ball in flight. No single threshold
satisfies both.

## What actually moves the needle: weak supervision

Proximity is not always ambiguous. In ~12% of frames the ball is clearly nearest
one player and well clear of the runner-up. Those frames label themselves, and
training on them teaches the detector the *appearance* of a ball-handler — hands
on the ball, body squared to it — which is the cue proximity can never access in
a crowd.

`scripts/harvest_handler_labels.py` collects these with deliberately strict
thresholds (nearest ≤ 0.45 body-heights, separation ≥ 0.55): a wrong pseudo-label
is worse than a missing one. Harvested labels go to **train only**; validation
stays human-annotated so mAP keeps measuring against real annotations.

Yield improved as the detector improved — a virtuous loop:

| round | source | handler instances | fires | confidence |
|---|---|---|---|---|
| 0 | Roboflow only | 191 | 24% | 0.55 |
| 1 | + 489 harvested | 680 | 49% | 0.85 |
| 2 | + 3,359 harvested | 3,550 | *pending* | *pending* |

## Evaluation design note

The V6 answer key stores **image positions**, not track IDs. An earlier version
stored IDs; retraining the detector silently invalidated every entry while the
gate went on reporting a confident number. A point on the ball-handler's body is
a fact about the footage and survives any model change.
