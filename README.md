# CourtVision

Basketball video in. Named play-by-play out.

![CourtVision annotating an NBA possession](docs/media/demo.gif)

Boxes are coloured by team, numbered by track. The **yellow box is whoever has the
ball** — inferred from the footage, not from a scoreboard feed.

```json
{ "time_s": 4.08, "action": "steal",   "team": "A", "player_name": "Conley" }
{ "time_s": 5.68, "action": "rebound", "team": "A", "player_name": "Randle" }
{ "time_s": 7.28, "action": "pass",    "team": "B", "player_name": "Jokić" }
```

Those names come from joining the event timeline against the official play-by-play
— no jersey OCR. If a match isn't confident, the event keeps its anonymous track id
rather than guessing.

```bash
python -m scripts.run_pipeline clip.mp4 --out outputs/run
```

Nine stages: extract → detect → track → team assignment → possession → action
classification → events → commentary → render.

## What it does well

Action classification, 735 held-out clips, seven classes:

| | | | |
|---|---|---|---|
| dribble | 0.90 | block | 0.86 |
| shot | 0.84 | steal | 0.82 |
| other | 0.81 | pass | 0.79 |
| rebound | 0.78 | **overall** | **0.816** |

Chance is 0.143. Possession resolution is 9/9 on the human-annotated answer key.
Commentary is guarded against fabrication — an unrecognised name fails the gate.

The numbers are trustworthy in a specific way: SpaceJam and BARD clips are
visually distinguishable, so an earlier 0.810 was partly reading *which dataset*
a clip came from. Classes were rebalanced until corpus membership was worth
+0.010 of accuracy, and the same action now scores within 0.06 whichever corpus
it came from.

## What it does not do yet

**It has been validated on clips, not on games.** Running 84 minutes of continuous
footage — 50,304 frames — surfaced the gap:

| action | emitted/min | realistic/min | |
|---|---|---|---|
| rebound | 26.9 | 1.8 | **15× too many** |
| steal | 3.4 | 0.3 | 11× too many |
| pass | 0.2 | 9.6 | **48× too few** |

The classifier was trained on a balanced mix of curated action clips. A real game
is mostly ordinary play. The model has never seen that prior, so it forces every
window into an action class and rebound absorbs the slack.

Two structural findings from the same run: 268 broadcast cuts fragment tracking
into **8,602 track ids** (19 on a short clip), which defeats event de-duplication
and inflates 3,383 events out of maybe 400 real ones; and the annotated output is
4.8 GB for 84 minutes.

The pipeline itself scales — **47.7 minutes to process 84 minutes of video**, 1.76×
real time, with memory flat in clip length.

## Where things stand

| | |
|---|---|
| v1 — pipeline | done, 11 gates, 284 tests |
| v2 — CUDA kernel | compiles, matches an OpenCV oracle; speedup does not reproduce across machines |
| v2 — dual-GPU split | verified, and measurably *not* worth it — the classifier was never the bottleneck |
| v3 — player naming | working end to end against the official feed |
| v3 — play recognition | geometric sets only (pick-and-roll, horns, DHO); real playbook recognition is open |

Detail: [full-game findings](docs/full-game-findings.md) · [model results](docs/v7-gpu-results.md) ·
[GPU runbook](docs/v2-gpu-runbook.md) · [v1 spec](docs/spec-v1.md)
