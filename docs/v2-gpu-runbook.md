# v2 runbook — what to run once you have NVIDIA GPUs

Everything here is implemented and unit-tested on CPU/MPS. What it has **not**
had is a CUDA device. This is the order to work through on a rented box.

## 0. Before renting: what is already banked

Spec §7.1 says profile before writing any CUDA. That is done
(`docs/profile-v1.md`), and it changed the plan: the bottleneck was **action
classification at 76.8%**, not detection at 14.6%. Profiling without stage 6
showed detection at 92%, which would have aimed all the kernel work at the wrong
stage.

Two wins landed with no CUDA at all:

| change | effect |
|---|---|
| Batched the action classifier's windows | stage 6 **76.8% → 35.8%** |
| Collapsed duplicate overlapping-window events | 10 events → 5, better commentary |

Do the free wins before paying for hardware.

## 1. Renting the box

### What to rent

**2x RTX 4090 (or 3090 / A5000) on ONE instance.** Not A100/H100: YOLO11n is
5 MB and VideoMAE-base 344 MB, so VRAM is nowhere near the constraint and you
would be paying 3-5x for headroom you cannot use. The two GPUs must be on the
same instance — §7.2 pipelines stage output between them, which two separate
single-GPU boxes cannot do.

V7 training only needs **one** GPU. If you want to train first and do v2 later,
rent 1x 4090 now and a 2-GPU box when you get to §7.2.

### Where

**RunPod** (runpod.io) is the recommendation: per-second billing, persistent
volumes, and a browser terminal, so there is no SSH key setup before you can
run anything. Vast.ai is cheaper per hour but is a marketplace — host quality
varies, and a machine disappearing mid-run costs more than it saves.

Order of operations on RunPod:

1. Add credit (~$10 covers everything in this document several times over).
2. **Deploy → Pods → GPU Cloud**, filter to RTX 4090, pick a host with 2x
   available if you are doing §7.2.
3. Template: **PyTorch 2.x / CUDA 12.x**. This matters — a bare Ubuntu image
   means installing the CUDA toolkit yourself, and `torso_color.cu` needs
   `nvcc`, which the PyTorch templates already carry.
4. Set container disk to **≥40 GB**. The frame cache alone is ~8 GB and the
   default 10-20 GB will fill.
5. Deploy, then **Connect → Start Web Terminal**.

> Prices move and mine may be stale — check the live figure before you commit.
> As a sanity band, a 4090 has been roughly $0.35-0.70/hr and a 2x box roughly
> $0.70-1.40/hr. Everything in this runbook is a few hours of work, so expect
> single-digit dollars total.

**The only real cost risk is forgetting to destroy the pod.** Stopping is not
destroying — a stopped pod still bills for its volume. Destroy it when done.

### If runpodctl says it has no credentials

`~/.runpod/config.toml` existing does NOT mean a key is set. On this machine it
existed with a **2-character `apikey`** — a placeholder, where a real key is
40-odd characters — so `runpodctl user` returned `no_credentials` while the
file's presence made it look configured.

Credential order is `RUNPOD_API_KEY` env → `.env` → `~/.runpod/config.toml`, so
either works:

```bash
export RUNPOD_API_KEY=...        # nothing written to disk
runpodctl config --apiKey ...    # persists it to config.toml
```

Tell them apart by the error: `no_credentials` means nothing was found at all,
while `unauthorized` / 401 means a key was found and rejected.

### Getting the code and data there

The repo has a remote but this branch has never been pushed, and `data/` is
gitignored, so the clips travel separately. From your Mac:

```bash
git push -u origin feat/v1-pipeline
```

Then on the pod, either run the bootstrap, which clones, installs, checks
that CUDA survived the install and runs the tests:

```bash
git clone -b feat/v1-pipeline https://github.com/pronton1234/CourtVision.git
cd CourtVision && bash scripts/bootstrap_pod.sh
```

...or do it by hand:

```bash
git clone -b feat/v1-pipeline https://github.com/pronton1234/CourtVision.git
cd CourtVision && python3 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev]"
```

And from your Mac, to move the clips (RunPod shows the host/port under
**Connect → SSH**):

```bash
rsync -avz -e "ssh -p <PORT>" data/labeled/actions root@<HOST>:/workspace/CourtVision/data/labeled/
rsync -avz -e "ssh -p <PORT>" data/raw_clips checkpoints root@<HOST>:/workspace/CourtVision/
# V9 names players from the official play list, which run_pipeline reads from
# this one CSV. Omit it and V9 still runs but never names anybody.
rsync -avz -e "ssh -p <PORT>" data/labeled/bard_meta/dataset.csv \
  root@<HOST>:/workspace/CourtVision/data/labeled/bard_meta/
```

rsync appends the source directory name, so `data/labeled/actions` sent to
`data/` lands at `data/actions` and every script then reports no clips.

That is ~496 MB compressed. Send only these; `data/labeled` holds 15 GB across
`bard_meta/clips` (11 GB), `detector` (1.3 GB), `handler_harvest` (1.1 GB),
`spacejam` (682 MB) and `roboflow` (166 MB), and no stage on the pod opens any
of them. `scripts/launch_gpu_run.sh` builds its bundle from an explicit include
list for that reason: an exclude list goes stale the moment a new directory
appears, and the failure mode is a silent 12 GB upload to a machine billing by
the second. Do **not** send `data/labeled/detector` —
it is 9.3 GB and nothing here needs it unless you retrain the detector.

## 2. Train V7 first — it is the thing that is blocked

This is why you are renting. V7 cannot run on the Mac at all: training needs
~2 GB and the machine has ~1.6 GB free with browsers holding 4.1 GB. Three
attempts thrashed rather than trained.

```bash
./.venv/bin/python -m pytest -q          # expect 132 passed
bash scripts/run_gpu_suite.sh 2>&1 | tee outputs/suite.log
```

That one command runs everything in dependency order — corpus audit, V7
training, the per-class and cross-source report, V9 end-to-end, and the v2
kernel/disaggregation checks — logging each to `outputs/` and printing a
PASS/FAIL summary. It deliberately does not abort on a failing stage, because
a later stage often explains an earlier one and the rental is metered.

It starts with the corpus audit for a reason: if the clips did not all rsync
across, every number after that is meaningless and you want to know in the
first thirty seconds, not after the training run.

To run the training step alone instead:

```bash
./.venv/bin/python -u scripts/validate_v7.py 2>&1 | tee outputs/v7_run.log
```

**Expected wall-clock:** decode is ~2.5 minutes (measured: 90 s for 2,660 clips
before the corpus rebalance, and it is CPU-bound so it transfers). Training is
8 epochs over ~2,764 clips at batch size 1; on a 4090 that should land in
**13-30 minutes**, so under 40 minutes end to end. Treat the training half as an
estimate — it has never run on CUDA. The epoch line prints its own elapsed
seconds; multiply the first one by 8.

For reference, the same run on the M2 measured 95 s for 67 training steps, which
extrapolates to ~6.7 hours. Local was never viable on time, independent of the
memory ceiling that actually blocked it.

Batch size is deliberately left at 1 so the number stays comparable with the
0.810 baseline. Raising it would be faster but would change the experiment.

The loss IS class-weighted now, by inverse frequency. The classes are not
balanced — 188 rebound clips against 649 shot in the training split — and the
frozen-backbone probe measured what that costs: rebound was the worst class by
a distance at 0.29, trading errors with steal both ways, while every
well-represented class sat between 0.62 and 0.88. Unweighted, the cheapest way
to cut the loss is to concede the rare class. Rebound is weighted 3.5x shot.
Watch rebound specifically in the per-class report; it is the class this run
most needs to move. But temper the expectation, because the fix was tested on
the frozen features first: class weighting moved rebound 0.289 to 0.342 there,
and no head helped more. Rebound against steal as a TWO-class problem scores
0.682 against a 0.658 majority baseline, so the frozen features barely separate
those two actions at all. Fine-tuning is the only lever that changes the
features — the right intervention, and not a guaranteed one.

Then the check that actually matters:

```bash
./.venv/bin/python -m scripts.report_action_confusion
```

**Accuracy is not the pass criterion.** The earlier 0.810 was partly measuring
dataset identity. The cause was composition, not appearance: every rebound and
steal clip came from BARD and every other class from SpaceJam, so corpus
membership predicted the label for 660 of 2,660 clips. Augmentation cannot fix
that, and measurably did not — image statistics still separated the corpora 95%
of the time after blur/brightness/contrast jitter, and 98% before.

The fix was to populate `shot` and `other` from BOTH corpora, using BARD's
2PT/3PT shot and foul/turnover events:

```
BEFORE  2,660 clips  corpus-only 0.314  majority 0.163  confound +0.150
AFTER   3,455 clips  corpus-only 0.241  majority 0.232  confound +0.010
```

`scripts/audit_source_cue.py` recomputes that on the box if you want to confirm
it travelled. The pass criterion is now the **cross-source generalisation** block
in the confusion report: `shot` and `other` are scored separately on their
SpaceJam and BARD clips. Similar accuracy on both means the model learned the
action. A gap above ~0.25 means it is still leaning on corpus identity, and the
headline accuracy should not be trusted.

Expect the accuracy to be lower than 0.810 — that number was inflated. A lower
honest figure is the point.

Finally, re-run the end-to-end gate on the retrained 7-class model:

```bash
./.venv/bin/python -m scripts.validate_v9
```

## 3. Verify everything, in one command

```bash
./.venv/bin/python -m scripts.verify_v2_gpu
```

Runs the whole sequence with a PASS/FAIL per step:

1. CUDA present, and whether there are two devices for §7.2
2. `torso_color.cu` compiles under nvcc. Its arithmetic is already checked:
   `scripts/verify_kernel_numerics.py` compiles the real source with host CUDA
   stubs, runs it under 256 real threads with a real barrier, and compares
   against the OpenCV oracle — clean compile, worst channel error 0.462 against
   a 1.5 tolerance, and the tree reduction matching the serial path to 0.0003.
   That runs in CI, so a build failure here is an nvcc or toolchain problem
   rather than a bug in the kernel body
3. The kernel matches the OpenCV oracle numerically
4. It is actually **faster** than the reference (a correct-but-slower kernel is
   not worth the risk of using)
5. The disaggregated pipeline runs, with per-stage wait times.
   Its LOGIC is already verified without CUDA by
   `scripts/verify_disaggregation.py`, which runs the real pipeline on real
   frames: 12 windows from 12 classifier calls, backpressure applied at
   queue_size=1 (producer waits 0.27 s rather than buffering the clip), and
   a raising classifier propagating instantly rather than deadlocking. What
   two GPUs add is the overlap SPEEDUP, not the correctness

Do not wire the kernel into `team_assignment` until step 3 passes. A subtly wrong
kernel shifts team assignments silently, which is worse than no kernel.

## 4. Re-profile on real hardware

```bash
./.venv/bin/python -m scripts.profile_pipeline
```

Relative stage costs transfer from MPS; absolute numbers do not. Re-measure
before optimising further, for the same reason §7.1 exists.

## 5. §7.2 — disaggregated serving

`courtvision.serving.StagePlan` carries device placement as data:

```python
StagePlan(detection="cuda:0", action="cuda:1")   # 2-GPU box
StagePlan.single("cuda:0")                        # 1 GPU, for comparison
```

Run both and compare `StageTiming.waiting`. That is the number that matters: a
stage starved at its input wants more upstream throughput; a stage blocked on its
output is ahead of its consumer. Expect the single-device plan to show large
waits and the split plan to show them shrink.

## 6. Nsight

`nsys profile` and `ncu` are NVIDIA-only and are the tools spec §7.1 names. Use
them on the fused kernel to confirm it is memory-bound (it should be — it does
almost no arithmetic per pixel) and to check occupancy at 256 threads/block.

## 7. Remaining §7 items, in value order

1. **Jersey OCR + pose** (§7.3) — replaces `Player 7` with a real identity.
   See `docs/v3-live-data.md`: for archived games the play-by-play API is a far
   cheaper route to names than OCR.
2. **Multi-camera** (§7.4).
3. **Live/streaming** (§7.5).
