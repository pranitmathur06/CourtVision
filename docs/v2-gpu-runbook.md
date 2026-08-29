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

## 1. Machine

**2× consumer GPUs (RTX 4090 or 3090) on ONE instance.** Not A100/H100 — the
models are small (YOLO11n is 5 MB, VideoMAE-base 344 MB) and VRAM is not the
constraint. Two GPUs must share an instance for §7.2; two separate single-GPU
boxes cannot pipeline.

RunPod or Vast.ai, per-hour. Destroy when done — idle instances are the only real
cost risk.

## 2. Setup

```bash
git clone <repo> && cd CourtVision
python3 -m venv .venv && ./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/python -m pytest -q          # expect 111 passed
./.venv/bin/python -c "from courtvision.device import resolve_device; print(resolve_device())"   # cuda
```

`resolve_device()` picks `cuda` with no code change — that was the point of
routing every device decision through it from Task 1.

## 3. Verify the kernel BEFORE trusting it

```bash
./.venv/bin/python -c "from courtvision.kernels.torso_color import verify_fused_kernel; verify_fused_kernel()"
```

This compiles `torso_color.cu` and checks it against the OpenCV oracle. It has
never been compiled — expect to fix build errors. Do not wire it into the
pipeline until this prints PASS.

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
