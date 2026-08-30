# V7 on a GPU — six sessions, $3.65, and what actually moved the number

    run  overall  rebound  block  what changed
     1     0.753     0.03   0.90* baseline
     2     0.784     0.49   0.67* rebound/steal resampled to a 20% window
     3     0.793     0.40   0.88  per-class stratified split
     4     0.822     0.84   0.78  rebound windowed by its place in the sequence
     5     0.816     0.78   0.86  rebound capped at 436 to reclose the confound

    * measured on a split that drifted — see "Fix two".

Not one of those moves came from a better model. Every one came from fixing
the data or the measurement.

## Final: V7 PASS at 0.816 on 735 held-out clips

    dribble  SpaceJam  0.90      block   SpaceJam  0.86
    pass     SpaceJam  0.79      steal   BARD      0.82
    shot     SpaceJam  0.84      other   SpaceJam  0.81
    shot     BARD      0.78      other   BARD      0.76
    rebound  BARD      0.78

    cross-source gaps: shot 0.06, other 0.05 — the best measured
    cross-corpus confusions between single-corpus classes: 1/735
    corpus membership is worth +0.010 of label accuracy — CONFOUND CLOSED
    bar was 0.368 (uniform chance 0.143, majority class 0.218)

Every class is between 0.76 and 0.90. Run 4 scored higher overall (0.822) and
on rebound (0.84), and is NOT the one to keep: it had 799 rebound clips, which
made rebound BARD's dominant class and reopened the confound to +0.099. A model
could score by noticing the corpus and guessing. 0.816 with the confound closed
is worth more than 0.822 with it open.

## Fix one: sample the action, not the whole clip

BARD source clips run 8-10 s and the sampler took 16 frames evenly across the
whole clip — half a second apart, against an action lasting about a second.
Fourteen of sixteen frames showed unrelated play, so a rebound clip and a steal
clip were largely the same footage.

    margin  window   accuracy   lift     (rebound vs steal, 240 clips)
      0.25     1.0      0.621  +0.121    <- what we were training on
      0.25     0.2      0.713  +0.213
      1.0      0.2      0.717  +0.217

Nearly double the lift at both crop margins, while margin itself changed
nothing. That is why five spatial hypotheses had all failed: the problem was
never what was in frame, it was when.

## Fix two: a validation split that could not drift

Block appeared to fall from 0.90 to 0.67. Mostly measurement. The split
concatenated every class in ACTIONS order and shuffled globally, so it depended
on each class's SIZE — changing rebound from 226 clips to 223 reshuffled block,
steal and other while leaving dribble, pass and shot byte-identical:

    dribble  run1  88  run2  88  in both  88
    block    run1  79  run2  78  in both  15

Only 15 of 79 block validation clips survived. With a per-class stratified
split block is 0.86 and never regressed. For two runs, every cross-run
per-class comparison in this project was partly comparing different clips.

## Fix three: use the clips that were being thrown away

Rebound sat at 0.40 and I called it a data ceiling. It was not. BARD has no
timestamps, only an ordered list of actions per clip, and the selector demanded
the clip be unambiguous — 223 rebound clips out of the 4,709 that contain one.
The exclusion was never that those clips are wrong; it was that a midpoint
window looks at the shot rather than the rebound. In 3,127 of them the rebound
is the second of two actions, so it sits about three quarters through.

The ordering places it well enough: the nth of m actions falls at roughly
(n + 0.5) / m. Rebound went 0.40 to 0.78 at the same clip count as steal.

## The one that bit back

Generating 799 rebound clips reopened the confound at +0.099, because rebound
became BARD's largest class. The audit caught it. Capping rebound at 436, level
with steal, returns it to +0.010:

    rebound n   corpus-only  majority  confound
          436         0.228     0.218    +0.010
          600         0.261     0.209    +0.052  OPEN
          799         0.298     0.199    +0.099  OPEN

The 363 surplus clips are kept in data/labeled/actions_spare, since they are
good clips and would be usable if the SpaceJam side were grown to match.

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
