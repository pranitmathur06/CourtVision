# v1 Pipeline Profile — 2026-08-28

Spec §7.1 step 1: profile before writing any CUDA. Run with
`./.venv/bin/python -m scripts.profile_pipeline`.

**Setup:** Apple M2, `mps`, synthetic 640×360 clip, 50 frames @10fps, stock
`yolo11n` COCO weights, base `videomae-base`. Warmup pass excluded. Stages 3–5
fed ground-truth boxes so they carry realistic load (stock COCO YOLO finds no
"people" in synthetic rectangles).

| stage | seconds | % total | ms/frame |
|---|---:|---:|---:|
| 1 extraction | 0.032 | 0.5% | 0.6 |
| 2 detection | 0.874 | 14.6% | 17.5 |
| 3 tracking | 0.048 | 0.8% | 1.0 |
| 4 team assignment | 0.288 | 4.8% | 5.8 |
| 5 possession | 0.000 | 0.0% | 0.0 |
| **6 action classification** | **4.590** | **76.8%** | **91.8** |
| 7 events | 0.003 | 0.1% | 0.1 |
| 9 render | 0.143 | 2.4% | 2.9 |
| **TOTAL** | **5.979** | 100% | 119.6 |

Throughput: **0.84× realtime** — 5.0s of footage took 6.0s to process.

## The finding

**Action classification is the bottleneck at 76.8%**, ~918 ms per 16-frame
window. Detection — the stage everyone assumes dominates — is 14.6%. This is
precisely the "bottleneck is somewhere unexpected" case spec §7.1 warns about:
profiling *without* stage 6 showed detection at 92%, which would have sent kernel
work to the wrong place entirely.

Why stage 6 is so expensive: VideoMAE-base is a ViT over 16×224×224 frames
(~1568 patch tokens per window), and windows currently run **one at a time**,
unbatched, with stride 8 — so every frame is encoded roughly twice.

## What this means for v2 (spec §7.1)

Ranked by value, and honestly:

1. **Batch the action-classifier windows.** They are independent and currently
   sequential. This is a pure throughput win on any GPU and costs no accuracy.
   Do this first — it needs no CUDA at all.
2. **Revisit stride and model size.** Stride 8 double-encodes every frame;
   stride 16 halves stage 6 outright. A smaller video backbone is another lever.
   Also free, also no CUDA.
3. **Custom kernel targets — not the top two stages.** Detection and action
   classification are standard forward passes already running vendor-tuned
   kernels (cuDNN/Metal); hand-writing those is a losing fight. The genuine
   candidates are the un-optimised pre/post-processing:
   - **team assignment, 5.8 ms/frame** — third-largest, and pure NumPy/CPU:
     torso crop → BGR→Lab conversion → per-crop mean. Fuses well into one kernel.
   - **render, 2.9 ms/frame** — per-frame draw calls.
   - letterbox resize and NMS inside detection pre/post-processing.

So the honest ordering is: batching and stride give the big wins for free; the
custom kernel is a genuine learning exercise best aimed at stage 4, not stage 6.

## Caveats

- MPS, not CUDA. Relative stage ranking transfers; absolute numbers do not.
- Synthetic 640×360 input. On 1080p footage, extraction and the team-assignment
  crops grow; YOLO and VideoMAE letterbox to fixed sizes and would not change.
- Stock/base weights. Fine-tuning changes accuracy, not layer shapes, so per-call
  cost is representative.
- Single run, not averaged. The 77%/15% split is far too lopsided for run-to-run
  noise to alter the conclusion.
- Stage 8 (commentary) excluded: one network round-trip, latency-bound, no kernel
  applies.
