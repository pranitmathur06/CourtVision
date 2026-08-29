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
| accuracy | **0.708** |
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

## What V7 still has to answer

Whether fine-tuning the top blocks lifts 0.708, and by how much — particularly
for rebound. This probe says the data is sound and the confound is closed. It
does not say the model is finished.
