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
4. Set container disk to **≥40 GB**. The frame cache alone is 6 GB and the
   default 10-20 GB will fill.
5. Deploy, then **Connect → Start Web Terminal**.

> Prices move and mine may be stale — check the live figure before you commit.
> As a sanity band, a 4090 has been roughly $0.35-0.70/hr and a 2x box roughly
> $0.70-1.40/hr. Everything in this runbook is a few hours of work, so expect
> single-digit dollars total.

**The only real cost risk is forgetting to destroy the pod.** Stopping is not
destroying — a stopped pod still bills for its volume. Destroy it when done.

### Getting the code and data there

The repo has a remote but this branch has never been pushed, and `data/` is
gitignored, so the clips travel separately. From your Mac:

```bash
git push -u origin feat/v1-pipeline
```

Then on the pod:

```bash
git clone -b feat/v1-pipeline https://github.com/pronton1234/CourtVision.git
cd CourtVision && python3 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev]"
```

And from your Mac, to move the clips (RunPod shows the host/port under
**Connect → SSH**):

```bash
rsync -avz -e "ssh -p <PORT>" data/labeled/actions data/raw_clips \
  root@<HOST>:/workspace/CourtVision/data/
```

That is ~160 MB, a minute or two. Do **not** send `data/labeled/detector` —
it is 9.3 GB and nothing here needs it unless you retrain the detector.

## 2. Train V7 first — it is the thing that is blocked

This is why you are renting. V7 cannot run on the Mac at all: training needs
~2 GB and the machine has ~1.6 GB free with browsers holding 4.1 GB. Three
attempts thrashed rather than trained.

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -c "from courtvision.device import resolve_device; print(resolve_device())"   # cuda
./.venv/bin/python -u scripts/validate_v7.py 2>&1 | tee outputs/v7_run.log
```

**Expected wall-clock:** decode is ~2 minutes (measured: 90 s for 2,660 clips,
and it is CPU-bound so it transfers). Training is 8 epochs over 2,128 clips at
batch size 1; on a 4090 that should land in **10-25 minutes**, so under half an
hour end to end. Treat the training half as an estimate — it has never run on
CUDA. The first epoch line tells you the real per-epoch cost; multiply by 8.

Batch size is deliberately left at 1 so the number stays comparable with the
0.810 baseline. Raising it would be faster but would change the experiment.

Then the check that actually matters:

```bash
./.venv/bin/python -m scripts.report_action_confusion
```

Accuracy is **not** the pass criterion here. The previous 0.810 was partly
measuring dataset identity: SpaceJam clips have sharpness ~1676 and BARD ~663,
non-overlapping ranges, and the model scored 0/532 cross-source confusions —
it could tell the two corpora apart. This run adds blur/brightness/contrast
augmentation to training only. **Success is cross-source confusions becoming
non-zero.** Expect accuracy to fall from 0.810; a lower honest number beats a
higher misleading one. If confusions are still 0, the leak is not blur and the
real cue has not been found yet — do not report a passing number in that case.

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
2. `torso_color.cu` compiles — it never has, so expect build errors first time
3. The kernel matches the OpenCV oracle numerically
4. It is actually **faster** than the reference (a correct-but-slower kernel is
   not worth the risk of using)
5. The disaggregated pipeline runs, with per-stage wait times

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
