# The two possession kernels on a real GPU

Measured on a rented RTX A6000, CUDA 12.8, torch 2.14.0+cu130, one session.
Reproduce with `scripts/bench_possession_gpu.py`.

`docs/v7-gpu-results.md` is the reason this file states the machine in its first
line. The one kernel this project shipped before measured 1.4x faster than the
unfused path on one box and 1.0x on another, and the honest conclusion was that
1.4x was a property of the box rather than of the kernel.

## Correctness on the GPU

The host harness (`scripts/verify_possession_numerics.py`) already compiles the
real `.cu` as host C++ and checks the arithmetic and both hand-derived backwards
against NumPy and autograd, on a machine with no GPU. What it cannot check is
that the thing builds and launches under nvcc. It does:

| | max abs difference |
|---|---|
| kernel 1 forward, probabilities vs the portable torch path | 7.5e-09 |
| kernel 1 forward, features | 2.1e-07 |
| kernel 1 backward vs autograd, worst parameter | 1.9e-06 |
| kernel 2 forward, posterior vs the NumPy oracle | 1.0e-07 |
| kernel 2 forward, loss | 1.7e-06 |
| kernel 2 backward, d/dscores vs the oracle | 1.9e-06 |
| kernel 2 backward, d/dstay | 1.3e-06 |

float32 kernels against float64 oracles, so these are at the noise floor.

**The GPU build caught a bug the host harness structurally could not.** The
entry-point declarations passed to `load_inline` as `cpp_sources` were kept by
hand and had gone stale: the feature standardisation added two tensors to both
of kernel 1's entry points and this copy did not follow. nvcc compiled the `.cu`
perfectly, the extension linked, and the *import* failed with

    undefined symbol: _Z19possession_backwardN2at6TensorES0_...

which is a mangled-name mismatch and reads like nothing at all. The declarations
are now read out of the `.cu` by `entry_point_declarations()`, so the two cannot
disagree again.

## Speed, on this machine, three runs each

10 players, 7 frames, 200 calls per timed run, milliseconds:

| | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|
| kernel 1 fused (forward) | 0.058 | 0.058 | 0.058 | **0.058** |
| kernel 1 portable torch (forward) | 2.290 | 2.322 | 2.346 | 2.322 |
| kernel 2 fused (forward + backward) | 0.037 | 0.037 | 0.037 | **0.037** |
| kernel 2 portable torch (forward + backward) | 5.298 | 4.619 | 2.980 | 4.619 |

That is 40x and 126x, and the honest reading of those numbers is **launch
overhead, not arithmetic**. The portable paths are eager torch on one frame:
kernel 1 is a few dozen small ops and kernel 2 is a Python loop over seven
frames issuing several launches each, so most of the milliseconds are the CPU
telling the GPU to do very little. That is exactly the cost fusion removes and
exactly why the plan argued for it -- seven passes over data that fits in
registers, each writing an intermediate to global memory for the next to read
back -- but it is not a claim that the fused arithmetic is 126x better. A
batched torch implementation would close most of the gap and is the comparison
that would settle it.

## Occupancy, and Nsight

**`ncu` does not run on a RunPod container.** It answers

    ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
    Performance Counters on the target device 0

which is gated by the *host* driver (`NVreg_RestrictProfilingToAdminUsers`) and
is not lifted by being root inside the container. `--section LaunchStats` alone
is refused the same way. So `docs/v2-gpu-runbook.md` §6 is still open, and this
is why.

What can be had without counters is occupancy, straight from the runtime via
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, which reads the compiled
kernel's registers and shared memory:

| at 64 threads/block | blocks/SM | threads/SM | of the 1536 an SM holds |
|---|---|---|---|
| kernel 1 | 16 | 1024 | 67% |
| kernel 2 | 16 | 1024 | 67% |

84 SMs, 48 KB of shared memory per block. Both sit at the 16-blocks-per-SM
hardware cap rather than at a register or shared-memory limit, so 67% is as
high as 64-thread blocks can reach; 96 threads would reach 100% and is worth
trying.

**The thing occupancy actually exposes is the launch shape, not the kernels.**
Both launch one block. A grid of one block uses one of 84 SMs, so the card is
99% idle no matter how good the block's occupancy is. The kernels are written
one-block-per-frame and are correct that way, but they are only worth their
speed on a *batch* of frames, and nothing calls them that way yet.

## Is kernel 1 memory-bound, as the plan claimed?

Not established. Reading a 720x1280 frame once is 2.8 MB, which over the
measured 0.058 ms is 47 GB/s against roughly 768 GB/s of peak bandwidth -- but
the sweep only touches the players' boxes, so that figure is an upper bound on
the traffic and not a measurement of it. Settling it needs counters, and
counters need `ncu`.

## What the kernels score, on 157 held-out frames

Never trained on, uniformly sampled, so this estimates in-game accuracy and not
performance on hard cases. `scripts/eval_possession_temporal.py`.

| | right | | |
|---|---|---|---|
| the detector's handler class, what ships | 78/157 | **49.7%** | CI 42-57% |
| kernel 1 alone, the centre frame | 76/157 | 48.4% | CI 41-56% |
| kernels 1 + 2, the scan over time | **93/157** | **59.2%** | CI 51-67% |

Paired, which is the test that fits: the frames are the same frames, so the
question is who is right when they disagree. Exact McNemar on the discordant
pairs:

| | only the first right | only the second | p |
|---|---|---|---|
| kernels 1+2 vs the handler class | 29 | 14 | **0.031** |
| kernels 1+2 vs kernel 1 alone | 34 | 8 | **0.0001** |
| kernel 1 alone vs the handler class | 19 | 21 | 0.87 |

So the per-frame operator ties what ships, and **the scan over time is where the
gain is** -- +10.8 points over the per-frame operator on the same weights, and
+9.5 over the shipped handler class.

Two Wilson intervals would have called this nothing: 51-67% against 42-57%
overlaps. On 157 frames an interval is about 8 points wide either side and the
honest effects in this project are 5 to 10, so unpaired intervals cannot resolve
them and the paired test is not a nicety.

### A measurement bug that cost seven frames, in my own favour to fix

These kernels were first scored by "did it name the right index", while the
49.7% baseline comes from `eval_handler.py` asking "does the box it points at
overlap the box a person drew, at IoU 0.5". Players overlap on a basketball
court, so two candidates can both clear 0.5 on the same man; counting one right
and the other wrong measures the box list rather than the method. Scoring the
kernels the baseline's way is what the table above does, and the check that it
is the right fix rather than a flattering one is that the handler class comes
out at exactly 78/157 -- the canonical number, to the frame.

### Choosing the configuration without spending the evaluation set

Five variants were trained. The one reported was chosen on **training loss**,
before its held-out number was looked at: region fixed at the upper body,
standardisation fitted on the training windows' centre frames, no end-to-end
phase. Notes on the ones not chosen, since the reasons are the interesting part:

- **Fitting the standardisation across the whole window** rather than the centre
  frames cost the per-frame operator points: a neighbour frame carries a stale
  box wherever the detector lost a player, and those drag the mean and scale
  away from the distribution the answer is read from.
- **Starting the region from the one kernel 1 learned end to end** -- 0.64 to
  0.91 down the body, where a dribbled ball is -- was *worse* on training loss
  (1.111 against 1.053). Sensible-sounding and wrong.
- **The end-to-end phase**, which unfreezes the sampling region, raised training
  loss every time. The cause is not the region: Adam does not carry its moments
  across a warm restart, and beginning the head again at full rate threw the
  loss from 1.07 to 1.22 in four epochs. Restarting the head at a tenth of the
  rate fixed that, and the phase still did not pay on 314 windows.
- **Cross-validating inside the training half** (`--cv 5`) gives 23.3% +/- 2.8
  with the scan against 21.6% +/- 5.7 without -- same ordering, far lower
  numbers, because the training half was sampled from frames where the model was
  already struggling and is a harder distribution by construction.

The spread across those variants is about 3 points of held-out accuracy, which
is the noise floor of a 157-frame set and the reason the configuration was
picked without looking at it.

### What the model learned

The stay bonus trains to **3.79**, a strong preference for the same player
keeping the ball from frame to frame -- which is what a basketball possession
is, learned from 314 labelled windows rather than assumed. The staleness penalty
settles near zero (-0.02), so carrying a box forward through a frame where the
detector lost a player costs the model almost nothing; 0.25 s is short enough
that the old box is still roughly right.

## Wired into the pipeline, behind a flag

`scripts/clip_boxes.py --possession checkpoints/possession/temporal.json` picks
the clip's subject from the kernels instead of the detector's handler class.
Without the flag nothing changes, so the old path stays measurable.

On 25 clips of the ECF game the two disagree on 16, which is the rate the
held-out set predicts (they disagree on 43 of 157 frames there, and the kernels
are right on 29 of those). The kernels also draw a subject on more clips -- 19
of 25 against 16 -- because the handler class simply fails to fire on some
frames and a scan over a window does not have to.

Writing it turned up one defect worth naming: the scan is free to name a track
the *window* saw and the *logged frame* did not, and the clip is then rendered
with no subject box at all. It did exactly that on one clip in six. Candidate
tracks are now restricted to those on screen at the logged instant.
