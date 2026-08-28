# NBA Vision-to-Text Pipeline — v1 Implementation Spec

**Status:** Ready for implementation
**Owner:** Pronton
**Purpose of this doc:** A build-ready spec for a scoped-down v1 of a system that watches a basketball clip and produces (a) player/ball detections, (b) who has the ball, (c) a play-by-play classification, and (d) natural-language commentary. This is the foundation for a larger distributed, multi-GPU version — v1 deliberately stays on one GPU and uses proven, pretrained building blocks so the *pipeline logic* gets validated before anything gets optimized for speed or scale.

---

## 1. Goal

Given a short basketball clip (10–60 seconds, single camera angle), produce:
1. Bounding boxes for players, the ball, and the rim, with persistent per-player IDs across frames
2. A team label (A/B) for each player
3. Which player has the ball at each moment (possession)
4. A classified action for short windows of play (dribble / pass / shot / rebound / other)
5. A natural-language commentary log synced to timestamps (e.g., *"0:14 — Player 7 (Team A) drives and passes to Player 3, who attempts a jump shot."*)
6. An annotated output video with overlays (boxes, team colors, possession highlight) + the commentary as a text/JSON log

## 2. Explicit non-goals for v1 (do not build these yet)

- ❌ Distributed / multi-GPU serving (that's v2 — see Section 7)
- ❌ Jersey number OCR (defer — use team color + tracker ID only in v1)
- ❌ Multi-camera or live/streaming input (pre-recorded single clip only)
- ❌ The custom CUDA kernel (v2 — profile v1 first to find the *real* bottleneck before writing one)
- ❌ Training any model fully from scratch (v1 fine-tunes pretrained models only)

Keeping these out is intentional. v1's only job is to prove the pipeline logic is correct, end to end, on one GPU, before anything else gets layered on.

## 3. v1 Architecture (single GPU, sequential stages)

Run each stage as a separate, inspectable step — do not fuse them into one script until each stage is individually validated (see Section 6).

```
clip.mp4
   │
   ▼
[1] Frame Extraction (OpenCV)
   │  → frames at fixed FPS (e.g. 10 fps to start; raise later)
   ▼
[2] Detection (YOLO, fine-tuned)
   │  → per-frame boxes: {player, ball, rim}
   ▼
[3] Tracking (ByteTrack)
   │  → persistent track_id per player across frames
   ▼
[4] Team Assignment (k-means on jersey pixel color, k=2)
   │  → team label per track_id
   ▼
[5] Possession Heuristic (rule-based, no model)
   │  → nearest player bbox to ball bbox, distance-thresholded,
   │    temporally smoothed to avoid frame-to-frame flicker
   ▼
[6] Action Classification (fine-tuned video model, 16-frame windows)
   │  → {dribble, pass, shot, rebound, other} per window
   ▼
[7] Event Structuring (plain Python)
   │  → discrete events: {timestamp, player, team, action, possession_change}
   ▼
[8] Commentary Generation (LLM via LangGraph)
   │  → structured events → natural-language play-by-play
   ▼
[9] Output Renderer (OpenCV overlay + JSON/text log)
   │  → annotated .mp4 + commentary.json
```

Stage 8 is the natural place to reuse your existing LangGraph/agent experience: the input is a clean structured event list, and the prompt's job is just narration — not decision-making — which keeps hallucination risk low (the LLM is describing events you already computed, not inventing them).

## 4. Tooling and model choices (and why)

| Stage | Tool | Why |
|---|---|---|
| Frame extraction | OpenCV | Faster decode than PIL/torchvision defaults; also reused later for the output renderer |
| Detection | YOLO11 (or RF-DETR if available) via `ultralytics` or Hugging Face | Fast, well-supported, easy to fine-tune on a small labeled set |
| Tracking | ByteTrack | Lightweight, no re-ID model required, good default for single-camera tracking |
| Team assignment | scikit-learn `KMeans`, k=2, on cropped jersey pixels | No training needed; simple and works well when teams have distinct colors |
| Action classification | A pretrained video model (e.g. Hugging Face `VideoMAE` or `TimeSformer`, or `torchvision`'s `r3d_18`) fine-tuned on a small labeled clip set | Don't train a video architecture from scratch — fine-tune a small pretrained one; save "from scratch" effort for v2's custom kernel work |
| Commentary | Any accessible LLM, orchestrated via LangGraph | Directly reuses your existing agent-building experience |

## 5. Datasets to pull labeled clips from

Use these for fine-tuning the detector and action classifier — check each project's current access/license terms before pulling data, since availability and terms can change:

- **BARD** (2026) — basketball action recognition with jersey numbers, team colors, and multi-label captions: `https://github.com/GabrieleGiudic/BARD`
- **NCAA basketball dataset** — 257 full games, ~14,500 labeled action clips + player bounding boxes (good for detector fine-tuning at scale)
- **DeepSportradar-v1** — multi-label dataset with 3D localization, calibration, and instance segmentation
- **SpaceJam** — smaller, good for a first smoke test; there's also a working reference implementation to study (not to copy wholesale): `https://github.com/hkair/Basketball-Action-Recognition`

Reference implementations worth reading (for ideas on possession detection and team assignment, not for direct reuse):
- `https://github.com/Wasim7x/BasketVision`
- `https://github.com/HanaFEKI/AI_BasketBall_Analysis_v1`
- `https://blog.roboflow.com/identify-basketball-players/`

**Suggested plan:** start with a *small* labeled subset (50–200 frames/clips) pulled from SpaceJam or a small BARD slice for the validation checkpoints in Section 6, then scale to the full dataset only once the pipeline is proven correct.

## 6. Validation checklist — build and verify in this exact order

This is the core of "validate before you build more." Each item is a small, standalone script with a clear pass/fail print statement — not folded into the main pipeline yet. Do not move to the next item until the current one passes. This catches a broken stage while it's still isolated, instead of buried under three later stages.

- [ ] **V1 — Frame extraction sanity check.** Load one sample clip, extract frames with OpenCV at the target FPS, print frame count and confirm it matches `duration × fps` within rounding. *Pass: frame count matches expectation, frames visually look correct when saved as images.*
- [ ] **V2 — Stock detector sanity check.** Run an off-the-shelf pretrained YOLO (COCO weights, `person` class only) on 5 sample frames, before any basketball-specific fine-tuning. *Pass: it draws reasonable boxes around people — this is your baseline, and confirms the detection library/environment works before you invest in fine-tuning.*
- [ ] **V3 — Fine-tuning loop sanity check.** Fine-tune the detector on a small labeled subset (50–200 frames) for player/ball/rim classes. *Pass: mAP on a held-out slice of that small subset improves over the stock baseline — this proves your training loop and data pipeline work end to end before scaling to the full dataset.*
- [ ] **V4 — Tracking sanity check.** Run ByteTrack on the fine-tuned detector's output for one 10-second clip. *Pass: visually inspect the annotated video — track IDs stay consistent, without excessive ID switching when players cross paths.*
- [ ] **V5 — Team assignment sanity check.** Run k-means color clustering on cropped player regions from the tracked clip. *Pass: the 2 clusters visually correspond to the two teams' actual jersey colors on a sample of frames.*
- [ ] **V6 — Possession heuristic sanity check.** Implement the distance-based possession rule and manually verify it against 10 known "who has the ball" moments you've identified by eye in the clip. *Pass: at least 8/10 correct, with the misses being genuinely ambiguous moments (contested rebounds, passes in flight), not obvious errors.*
- [ ] **V7 — Action classifier sanity check.** Fine-tune the video model on a small labeled clip subset. *Pass: accuracy on a held-out split is meaningfully above random-chance baseline (with ~5 classes, random ≈ 20% — aim well above that, not just barely).*
- [ ] **V8 — Commentary generation sanity check.** Feed 5–10 hand-constructed structured events into the LangGraph prompt. *Pass: manually review the output — commentary references the correct player/team/action and reads naturally, with no fabricated events not present in the input.*
- [ ] **V9 — End-to-end run.** Run the full pipeline on one held-out clip not used in any fine-tuning step. *Pass: output video + commentary log are produced without crashing, and a spot-check of 5 random moments in the output roughly matches what's actually happening in the clip.*

Once V1–V9 all pass, v1 is done. Everything past this point is v2 (Section 7) — don't start it early.

## 7. v2 extension path (design for it now, build it later)

Once v1 is validated end to end, the natural next steps — in a rough order of value:

1. **Profile v1 first.** Before writing any CUDA, profile the real pipeline (Nsight Systems or even simple wall-clock timing per stage) to find the actual bottleneck. It's very possible the bottleneck is somewhere unexpected (e.g., per-crop feature extraction, not detection itself) — write the custom kernel for what profiling actually shows, not for what seems obvious in advance.
2. **Split the pipeline across GPUs.** Detection on one GPU, action classification on another, commentary generation on a third — a real disaggregated serving setup instead of a single sequential process. This is the direct callback to distributed inference serving.
3. **Add jersey OCR + pose estimation** for real player identification, replacing the "Team A/Team B by track ID" placeholder from v1.
4. **Multi-camera support**, if you want the "true" broadcast-style experience.
5. **Live/streaming input**, once the batch version is solid.

## 8. Suggested repo structure

```
nba-vision-pipeline/
├── data/
│   ├── raw_clips/
│   └── labeled/              # small subset for validation checkpoints
├── src/
│   ├── extraction.py         # V1
│   ├── detection.py          # V2/V3
│   ├── tracking.py           # V4
│   ├── team_assignment.py    # V5
│   ├── possession.py         # V6
│   ├── action_classifier.py  # V7
│   ├── events.py             # structuring, feeds V8
│   ├── commentary.py         # V8, LangGraph agent
│   └── render.py             # V9, output video + log
├── checkpoints/               # fine-tuned model weights
├── scripts/
│   ├── validate_v1.py ... validate_v9.py   # one script per checklist item
│   └── run_pipeline.py       # full end-to-end run, only after V1-V9 pass
├── outputs/
└── NBA_VISION_PIPELINE_SPEC.md   # this file
```

## 9. Kickoff instructions (for Claude Code)

Work through this spec top to bottom. Concretely:

1. Scaffold the repo structure in Section 8.
2. Implement `scripts/validate_v1.py` first — nothing else. Get it passing before writing any other code.
3. Move through V2 → V9 in order, one script at a time, each in its own file under `scripts/`. Do not write `run_pipeline.py` until V1–V9 all pass individually.
4. For each validation step, print a clear `PASS`/`FAIL` plus the relevant metric (frame count, mAP, ID-switch count, accuracy, etc.) so pass/fail is unambiguous, not a judgment call buried in output.
5. Use the small labeled subset (50–200 clips/frames) for all validation steps — do not pull the full dataset until V1–V9 pass on the subset.
6. Stop and flag for review if any validation step fails twice in a row after a fix attempt, rather than working around it silently — some of these (possession accuracy, action classifier accuracy) have real, known difficulty ceilings, and a failure might mean the approach needs rethinking, not just debugging.
7. Do not start Section 7 (v2) work until this document is updated to mark all of Section 6 complete.

## 10. Known hard problems / honest risks

Worth knowing going in, so a rough patch doesn't read as "I did something wrong":

- **Possession detection is a proximity heuristic, not ground truth** — it will be wrong on contested rebounds, blocked shots, and balls in flight. This is expected; the goal is "mostly right," not perfect.
- **Jersey-number OCR (deferred to v2) is a genuinely hard, actively-researched problem** — small text, motion blur, partial occlusion. Don't be surprised if it needs its own tuning pass later.
- **Action classification accuracy will likely plateau below what you'd want** on a small fine-tuning set — this is a real signal about dataset size, not necessarily a modeling bug. If it plateaus, more labeled data usually helps more than a fancier model.
- **Commentary quality depends entirely on event-structuring quality** — if Stage 7's events are noisy, Stage 8 will confidently narrate the noise. When commentary looks wrong, check the structured events first, not the LLM prompt.