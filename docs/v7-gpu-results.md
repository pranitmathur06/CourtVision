# V7 on a GPU — four sessions, $2.29, and what actually mattered

The number moved three times. Every move came from fixing the data or the
measurement; none came from more training.

    run   overall  rebound  block   what changed
    1       0.753     0.03   0.90*  baseline
    2       0.784     0.49   0.67*  rebound/steal resampled to a 20% window
    3       0.793     0.40   0.88   per-class stratified split

    * measured on a split that drifted — see below.

## Final: V7 PASS at 0.793 on 692 held-out clips

    dribble  SpaceJam  0.88      block   SpaceJam  0.88
    pass     SpaceJam  0.80      steal   BARD      0.85
    shot     SpaceJam  0.87      other   SpaceJam  0.74
    shot     BARD      0.78      other   BARD      0.76
    rebound  BARD      0.40

    cross-source gaps: shot 0.09, other 0.02 — the best measured
    cross-corpus confusions between single-corpus classes: 0/692
    bar was 0.381 (uniform chance 0.143, majority class 0.231)

Best at epoch 5. Training loss falls to 0.04 while validation stalls, so
keeping the BEST checkpoint rather than the last is what keeps that harmless.

## Fix one: sample the action, not the whole clip

BARD source clips run 8-10 s and the sampler took 16 frames evenly across the
whole clip — half a second apart, against an action lasting about a second.
Fourteen of sixteen frames showed unrelated play, so a rebound clip and a steal
clip were largely the same footage.

Measured on 240 clips with the ball-handler selection held fixed:

    margin  window   accuracy   lift
      0.25     1.0      0.621  +0.121   <- what we were training on
      0.25     0.2      0.713  +0.213
      1.0      0.2      0.717  +0.217

Nearly double the lift, holding at both crop margins while margin itself
changes nothing. Regenerating rebound and steal at window 0.2 took rebound from
0.03 to 0.49.

This is why five spatial hypotheses all failed: the problem was never what was
in frame, it was when.

## Fix two: a validation split that could not drift

Block appeared to fall from 0.90 to 0.67, and that was mostly measurement. The
split concatenated every class in ACTIONS order and shuffled globally, so it
depended on each class's SIZE. Changing rebound from 226 clips to 223 shifted
everything ordered after it:

    dribble  run1  88  run2  88  in both  88
    pass     run1  84  run2  84  in both  84
    shot     run1 151  run2 151  in both 151
    block    run1  79  run2  78  in both   15
    other    run1 162  run2 163  in both   96

Only 15 of 79 block validation clips survived. With a per-class stratified
split, block is 0.88 and never regressed.

For two runs, every cross-run per-class comparison in this project was partly
comparing different clips. That is worth remembering before trusting any
before/after table.

## Still open: rebound at 0.40

The weakest class, losing 25 of 45 to steal. A real limit rather than a bug: a
defensive rebound and a steal are both a player collecting a loose ball, and
BARD labels them from the play-by-play rather than from what the footage shows.
There are only 223 rebound clips, so more data would help — but the ceiling is
set by how separable the two events are on video at all.

## v2 §7.1 — the kernel compiles and matches, but is not reliably faster

`torso_color.cu` compiles under nvcc 12.8 and matches the OpenCV oracle to
0.1456 against a 2.0 tolerance, on every box tried. Speed does not reproduce:

    single-GPU box   fused 0.221 ms  vs reference 0.302 ms   1.4x, 3/3 runs
    two-GPU box      fused 0.310 ms  vs reference 0.302 ms   1.0x, 1/5 runs

Same GPU model, same 32-vCPU Ryzen 7950X. "1.4x faster" was a property of one
machine, not of the kernel, and should not be quoted.

Two bugs only compiling could find. `load_inline` was called with
`cpp_sources=""`, so nvcc compiled the .cu perfectly and the build then died in
torch's generated glue with "'torso_mean_lab' was not declared in this scope" —
a CUDA-shaped error that was not a CUDA problem. And the upload carried 3,574
macOS AppleDouble files, one of which matches the `*.mp4` glob and crashed V7
after the clips had shipped.

## v2 §7.2 — verified, and it does not help this pipeline

On two 4090s, with the real detector and classifier:

    single cuda:0   0.27s   windowing wait 0.214s   classification wait 0.0
    split  0/1      0.28s   windowing wait 0.227s   classification wait 0.0

Splitting is marginally slower. The waits say why: classification never waits,
so the classifier is not the bottleneck — window construction is.
Disaggregation helps when two stages contend for one device, and here they do
not. The machinery is correct (`[PASS] two devices for §7.2`, backpressure, no
deadlock); the premise does not hold for this workload.
