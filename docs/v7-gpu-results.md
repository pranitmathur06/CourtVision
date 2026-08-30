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

**The fix is not more training.** It is regenerating rebound and steal clips
with a basket-relative crop. Everything else improved: dribble 0.92, block
0.90, steal 0.88, shot 0.83/0.78.

## Two bugs that only compiling could find

`load_inline` was called with `cpp_sources=""`. nvcc compiled the .cu perfectly
and the build then died in torch's generated glue with "'torso_mean_lab' was
not declared in this scope" — which reads like a CUDA failure and is not one.
The generated main.cpp needs the declaration handed to it, and the .cu's own
PYBIND11_MODULE duplicates the one load_inline writes.

The bundle also carried 3,574 macOS AppleDouble files, and `._0010446_flipped
.mp4` matches the `*.mp4` glob, so V7 crashed on "no frames in ..." after the
clips had shipped. Both fixed.

## v2 §7.1 — the kernel is real now

    OpenCV reference   0.302 ms/frame
    fused kernel       0.221 ms/frame   (1.4x)

Compiles under nvcc 12.8, matches the OpenCV oracle to 0.1456 against a 2.0
tolerance, and is faster across three consecutive runs on an idle GPU.

The first measurement said 1.2x, then 0.4x. Both were taken while V7 was
training at 71% GPU. A benchmark on a contended GPU is not a benchmark, and
reporting the 1.2x would have been reporting noise.

§7.2 disaggregation runs, but this box has one GPU so the overlap cannot be
demonstrated — `StagePlan(detection="cuda:0", action="cuda:1")` still needs a
two-GPU box.
