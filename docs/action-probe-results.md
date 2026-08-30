# Frozen-backbone probe — first real 7-class numbers

Run on the M2, no GPU. **This is not V7.** V7 fine-tunes the last two encoder
blocks and needs ~2 GB this machine does not have. A forward pass under
`no_grad` peaks at 0.48 GB, so the backbone runs here and a classifier trains on
its features. A frozen backbone is strictly weaker than fine-tuning, so read
every number below as a **floor for V7, not a substitute**.

Same split as V7: seed 0, 20% held out, 2,764 train / 691 val.

## Headline

| | |
|---|---|
| accuracy | **0.750** |
| uniform chance | 0.143 |
| majority class | 0.234 |

## Per class, per corpus

| class | corpus | n | acc | most confused with |
|---|---|---:|---:|---|
| dribble | SpaceJam | 88 | 0.88 | pass (9) |
| pass | SpaceJam | 84 | 0.69 | shot (9) |
| shot | SpaceJam | 69 | 0.81 | pass (6) |
| shot | BARD | 82 | 0.70 | other (18) |
| rebound | BARD | 38 | **0.29** | steal (14) |
| block | SpaceJam | 79 | 0.78 | shot (12) |
| steal | BARD | 89 | 0.67 | rebound (17) |
| other | SpaceJam | 77 | 0.71 | block (8) |
| other | BARD | 85 | 0.62 | shot (12) |

## The test that matters

`shot` and `other` are populated from both corpora, so the same action can be
scored on each. A model reading dataset identity would do well on one and badly
on the other.

| class | SpaceJam | BARD | gap |
|---|---:|---:|---:|
| shot | 0.81 | 0.70 | 0.12 |
| other | 0.71 | 0.62 | 0.09 |

Worst gap **0.12**, against a 0.25 threshold. The classifier handles the same
action from either corpus. That comparison did not exist before the rebalance —
no class was drawn from both.

## Caveats, stated plainly

**Cross-corpus confusions between single-corpus classes: 1/378.** Better than
the old 0/532, but still low. Some of that is legitimate — a model that learned
actions *should* rarely mistake dribbling for rebounding. It is weaker evidence
than the gap above, and it does not by itself rule out residual corpus reliance.

**Rebound is the weak class at 0.29**, with 14 of its 38 held-out clips called
`steal`, and `steal` returning the favour 17 times. Rebounds and steals are both
ball-recovery scrambles, so some confusion is real, but rebound also has the
least data (226 clips, against 400-800 elsewhere). It needs more clips before
its number means much.

**691 validation clips**, so per-class cells are 38-89 and the per-corpus splits
are smaller still. Treat single-class figures as indicative.

## A stronger head, and the tension it exposes

The head was switched from logistic regression to a 256-unit MLP on the same
frozen features:

| | logistic | MLP |
|---|---:|---:|
| accuracy | 0.708 | **0.750** |
| `shot` cross-corpus gap | **0.12** | 0.17 |
| `other` cross-corpus gap | 0.09 | **0.01** |
| cross-corpus confusions | **1/378** | 0/378 |

Accuracy is +0.042, and that is the better floor for V7. But the `shot` gap
widened and cross-corpus confusions fell to zero, and both move in the
direction of more corpus reliance, not less. A more capable head extracts more
signal from the same features — including whatever residual corpus signal
survives.

Both gaps remain inside the 0.25 bar, so the MLP is kept and the headline floor
is 0.750. The tension is recorded rather than smoothed over: if V7 comes back
with a widening `shot` gap, this is the reason to suspect first, and the
logistic numbers above are the comparison point.

## Why rebound fails, measured

Rebound at 0.29 was the standout weakness, so before spending GPU time on a fix
aimed at it, the fix was tested on the cached features. Three heads, seconds
each:

| head | overall | rebound | confused with |
|---|---:|---:|---|
| logistic (current) | 0.708 | 0.289 | steal 14, shot 7 |
| logistic + balanced class weights | 0.713 | **0.342** | steal 13, shot 6 |
| MLP (256) | **0.750** | 0.316 | steal 13, shot 7 |

Class weighting moves rebound 0.289 to 0.342 — real, and far short of a fix. A
stronger head gains 0.04 overall while doing nothing for rebound. Every head
loses rebound to steal, 13 or 14 of 38 every time.

So the confusion is in the FEATURES, not the head. The cleanest test is rebound
against steal as a two-class problem, the easiest possible version:

    rebound vs steal, two classes only: 0.682
    (226 rebound, 434 steal; majority-class baseline 0.658)

**+0.024 over always answering "steal".** The frozen Kinetics features barely
encode the difference between a rebound and a steal at all.

This matters for what V7 is expected to do. No head can fix a feature problem,
so the class weighting added to V7 should be expected to buy something like the
+0.05 measured here, not a repair. Fine-tuning the top blocks is the only
remaining lever that CHANGES the features, which is why it is the right
intervention — and also why it might not be enough. A rebound and a steal are
both scrambles for a loose ball, and they may be genuinely hard to separate
from a cropped clip of one player.

## What V7 still has to answer

Whether fine-tuning the top blocks lifts 0.708, and by how much — particularly
for rebound. This probe says the data is sound and the confound is closed. It
does not say the model is finished.
