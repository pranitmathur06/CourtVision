# V7 on a GPU — the run, and what it settled

RTX 4090, EU-RO-1, 8 epochs over 2,764 training clips at ~108 s each. Total
cost of the whole session, including v2 and a pod deleted for a missing SSH
key: **$0.43**.

## V7 PASS — 0.753 on 691 held-out clips

    accuracy by epoch: [0.667, 0.753, 0.708, 0.734, 0.742, 0.713, 0.735, 0.750]
    best epoch 2 at 0.753 (bar was 0.384)

Train loss fell 0.95 to 0.04 while validation stalled after epoch 2, so
everything past it was overfitting. Keeping the BEST checkpoint rather than the
last is what made that harmless.

**Fine-tuning bought +0.003 over the frozen probe's 0.750.** That is the
headline finding and it was predicted here before the run: no head, no pooling
and no reweighting moved the number, so the limit was never the method.

## The confound fix is confirmed

    shot   SpaceJam 0.83  BARD 0.78   gap 0.05
    other  SpaceJam 0.73  BARD 0.61   gap 0.12

Both inside the 0.25 bar, and `shot` improved on the probe's 0.17. The model
handles the same action from either corpus, which is what the rebalance was
for.

## Rebound collapsed, exactly as the crop analysis predicted

    rebound  BARD  38 clips  0.03  — 21 of 38 called steal
    steal    BARD  89 clips  0.88

One correct out of 38. The frozen probe managed 0.32, so fine-tuning made
rebound **worse**, despite class weighting at 2.10x.

That is not a surprise, it is the predicted failure. The rim appears in 0 of 40
sampled clips of every class: the crops are framed on the ball-handler, and a
rebound is defined by the ball coming off a rim that is never in shot. When two
classes are indistinguishable in the input, the loss-minimising move is to
concede the rarer one, and a stronger model does that more completely than a
weak one. Class weighting could not prevent it.

**The fix is not more training.** Everything else improved — dribble 0.92,
block 0.90, steal 0.88, shot 0.83/0.78 — so the model is learning; it simply
cannot learn this one.

### The crop is NOT the cause — tested properly, on two GPUs

A basket-relative crop was the natural remedy, and widening does bring the rim
back: at margin 1.5 it appears in 5 of 12 clips against 0 of 12 at 0.25.

The first attempt to test it was confounded — it cropped to the
highest-confidence player while the real clips crop to the BALL-HANDLER, so it
varied which person was centred as well as how much court was visible.
`scripts/crop_margin_experiment.py` holds that selection identical to
add_bard_action and changes only the margin. 240 clips, 5-fold CV:

    margin 0.25   acc 0.629   lift +0.129
    margin 1.0    acc 0.600   lift +0.100
    margin 2.0    acc 0.637   lift +0.137

**No difference.** An eightfold change in visible court moves nothing, so the
recommendation I wrote after the first GPU run — regenerate the clips with a
basket-relative crop — is wrong and is withdrawn.

### So rebound is not fixable by any lever tried

  * not the head — logistic, weighted logistic and an MLP all fail alike
  * not the pooling — six temporal variants, +0.017 at best
  * not the data volume — the learning curve plateaus at 0.735
  * not class weighting — 2.10x still let the model concede the class
  * not fine-tuning — it made rebound worse, 0.32 to 0.03
  * not the crop — 0.25, 1.0 and 2.0 are indistinguishable

Rebound and steal are not separable in this footage. Both are a player
collecting a loose ball, and from BARD's clips that is the same event; the
labels may also be genuinely ambiguous, since a steal and a defensive rebound
can look identical from one angle. The next thing worth trying is different
DATA — a source that labels them distinctly, or clips that show the ball's
trajectory before the collection — not another model.

## §7.2 on a two-GPU box: no benefit, and now measured

    single cuda:0   0.27s   windowing wait 0.214s   classification wait 0.0
    split  0/1      0.28s   windowing wait 0.227s   classification wait 0.0

Splitting detection and action across two 4090s is marginally SLOWER. The
reason is in the waits: classification never waits at all, so the classifier is
not the bottleneck — window construction is. Disaggregation helps when two
stages contend for one device, and here they do not.

The machinery is correct and verified (`[PASS] two devices for §7.2`, backpressure,
no deadlock). The premise does not hold for this pipeline as structured.

## The kernel's speedup does not generalise

    single-GPU box   fused 0.221 ms  vs reference 0.302 ms   1.4x, 3/3 runs
    two-GPU box      fused 0.310 ms  vs reference 0.302 ms   1.0x, PASS 1/5 runs

Same GPU model, same code, same 32-vCPU Ryzen 7950X. The 1.4x measured on the
first box is not reproducible on the second. The kernel is correct — it matches
the OpenCV oracle to 0.1456 against a 2.0 tolerance on both — but "1.4x faster"
was a property of one machine, not of the kernel, and should not be quoted.
