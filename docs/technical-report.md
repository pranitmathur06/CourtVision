# CourtVision — Technical Report

A nine-stage computer-vision pipeline that turns basketball broadcast video into
named, timestamped play-by-play. This document is the engineering account: what
was built, what was measured, what was tried and rejected, and where the system
currently fails.

The short version for skimmers is [the README](../README.md). This is the long
version, and it assumes you want the numbers.

---

## Contents

1. [Scope and framing](#1-scope-and-framing)
2. [Architecture](#2-architecture)
3. [Data model and streaming design](#3-data-model-and-streaming-design)
4. [Stage design decisions](#4-stage-design-decisions)
5. [Action classification — the long fight](#5-action-classification--the-long-fight)
6. [Possession — a measured ceiling](#6-possession--a-measured-ceiling)
7. [Naming players without OCR](#7-naming-players-without-ocr)
8. [Play recognition as geometry, not labels](#8-play-recognition-as-geometry-not-labels)
9. [Performance: profile, batch, split, fuse](#9-performance-profile-batch-split-fuse)
10. [What 84 minutes of real footage broke](#10-what-84-minutes-of-real-footage-broke)
11. [Verification strategy](#11-verification-strategy)
12. [Honest limitations and roadmap](#12-honest-limitations-and-roadmap)
13. [Reproducing this](#13-reproducing-this)

---

## 1. Scope and framing

The goal: given a basketball clip, emit `{time, action, team, player}` events and
natural-language commentary, with an annotated video alongside.

The constraint that shaped everything: **validate each stage in isolation before
composing them.** A nine-stage pipeline where stage 3 is subtly wrong produces
plausible-looking output that is wrong for reasons invisible at the end. So
every stage has a standalone gate script (`scripts/validate_v1.py` …
`validate_v9.py`) that prints `PASS`/`FAIL` with the metric that justifies it,
and no stage was composed into the pipeline until its gate passed.

Explicit non-goals for v1, held to deliberately:

| Deferred | Why |
|---|---|
| Multi-GPU serving | Prove the logic on one device first |
| Jersey-number OCR | Genuinely hard research problem; a cheaper route existed (§7) |
| Custom CUDA kernels | Profile first — write the kernel for the measured bottleneck (§9) |
| Training from scratch | Fine-tune pretrained backbones; save scratch effort for the kernel |
| Multi-camera / live streaming | Batch, single-angle, archived clips |

The full original spec, including the validation checklist, is
[docs/spec-v1.md](spec-v1.md).

---

## 2. Architecture

Nine sequential stages, each individually inspectable:

```
clip.mp4
  │
  ├─[1] extraction ─────── OpenCV decode at fixed FPS → FrameStore (disk-backed)
  ├─[2] detection ──────── YOLO11, fine-tuned: {player, ball, rim, handler}
  ├─[3] tracking ───────── ByteTrack → persistent track_id
  ├─[4] team assignment ── k-means (k=2) over mean torso colour in CIELAB
  ├─[5] possession ─────── learned handler + proximity fallback + hysteresis
  ├─[6] action classify ── VideoMAE fine-tuned, 16-frame windows, stride 8
  ├─[7] event structuring  windows + possession + teams → discrete events
  ├─[8] commentary ─────── LangGraph: narrate → validate → retry
  └─[9] render ─────────── OpenCV overlay video + commentary.json
```

Two cross-cutting layers sit beside the pipeline rather than inside it:

- **Enrichment** (`enrichment.py`, `nba_feed.py`, `scoreboard.py`) — joins events
  against the official NBA play-by-play to attach real player names (§7).
- **Court understanding** (`court.py`, `court_lines.py`, `formation.py`,
  `plays.py`) — image→court homography, formation classification, and screen /
  set detection from court coordinates (§8).

Run it:

```bash
python -m scripts.run_pipeline clip.mp4 --out outputs/run
```

**Module sizes** — ~4,000 lines of `src/`, ~4,200 of validation scripts, ~3,000
of tests across 287 test cases. The largest modules are the ones with the most
measured design behind them: `court_lines.py` (467), `plays.py` (424),
`enrichment.py` (384).

---

## 3. Data model and streaming design

`types.py` defines the contract every stage speaks: `Box`, `Track`, `Frame`,
`ActionWindow`, `Event`. Stages consume and return these, never raw tensors, so a
stage can be swapped or tested in isolation.

### FrameStore: memory flat in clip length

The original pipeline held every decoded frame in a Python list. At 1280×720
that is 2.8 MB per frame:

| clip length | frames | in-memory cost |
|---|---:|---:|
| 8.8 s (test clip) | 88 | 243 MB |
| 19 min | 11,400 | **31.9 GB** |

The pipeline was structurally limited to short clips, and the limit was invisible
because every test clip was under ten seconds. `framestore.FrameStore` is an
append-only `Sequence` backed by JPEGs on disk with a 48-frame LRU read cache.
Because every consumer (`collect_samples`, `classify_windows`, `render_video`)
already took a `Sequence[np.ndarray]`, nothing else changed.

JPEG is lossy, which matters here specifically because stage 4 clusters *colour*.
The encode settings were measured, not assumed:

| setting | torso-colour error (of 255) | size |
|---|---:|---|
| quality 95, 4:2:0 (default) | 4.6 | baseline |
| quality 100, 4:2:0 | 4.2 | larger |
| **quality 95, 4:4:4** | **1.0** | +16% |

Chroma subsampling, not quality, was the cost. `tests/test_framestore.py` pins
the resulting error so a future encode change cannot silently degrade team
assignment.

---

## 4. Stage design decisions

### Stage 4 — team assignment

k-means with k=2 over the mean torso colour of each player crop, converted to
CIELAB so distance in colour space approximates perceptual distance. A track is
assigned one team for its whole life by majority vote across frames — a person
does not change teams mid-possession, and letting the label flicker per-frame
produced events attributed to the wrong side.

### Stage 5 — possession

Two decisions carry this stage:

- **Distance is normalised by the player's box height**, so one threshold works
  for players near and far from the camera. A raw pixel threshold would be
  tracking perspective, not possession.
- **Switching possession requires hysteresis** — `min_hold_frames` of consecutive
  support — and a missing ball carries the previous holder for up to
  `max_gap_frames`. Stage 7 turns these into discrete events, so a single bad
  frame becomes a fabricated possession change in the commentary.

The full investigation, including everything that was rejected, is §6.

### Stage 7 — event structuring

Windows are size 16 with stride 8, so they overlap by design and a single real
action spans several of them. Emitting one event per window produced commentary
like *"Another board credited to Player 12"* three times for one rebound. So:

- The holder for a window is the **majority vote** across its frames, not the
  value at an arbitrary instant.
- Consecutive windows with the same action *and* the same holder **collapse into
  one event**. An event is a thing that happened, not a window that was scored.
- `background` windows are **dropped**, and they break the collapse chain — two
  real shots either side of a lull stay two events rather than merging.

### Stage 8 — commentary, and the anti-fabrication gate

The LLM's job is narration, not decision-making: every fact it needs is already
computed upstream. LangGraph owns the control flow (`narrate → validate → retry
or finish`); the Anthropic SDK is called inside the narrate node.

`validate_commentary` enforces the boundary mechanically, in both directions:

- A real player name in the text when the event carries none → **error**.
- A *different* real name than the event's → **error**.

That second check was missing at first, and a fabricated "Jokić" passed
validation. It now reports:

```
line 0: names 'Jokic', but the event's player is 'K. Johnson'
```

The capitalisation check is an **allowlist, not a blocklist**: an unrecognised
capitalised word is *assumed* to be a name and flagged. A false flag costs one
retry; a missed fabrication puts a false claim about a real, identifiable person
into the output. That asymmetry decides the design.

---

## 5. Action classification — the long fight

Seven action classes plus `background`, fine-tuned from VideoMAE-base on 16-frame
windows. Final held-out accuracy: **0.816 on 735 clips**.

| class | acc | | class | acc |
|---|---:|---|---|---:|
| dribble | 0.90 | | block | 0.86 |
| shot | 0.84 | | steal | 0.82 |
| other | 0.81 | | pass | 0.79 |
| rebound | 0.78 | | **overall** | **0.816** |

Against uniform chance 0.143 and majority class 0.218. Every class between 0.76
and 0.90.

**None of the improvement came from a better model.** Six GPU sessions costing
$3.65 total:

| run | overall | rebound | what changed |
|---|---|---|---|
| 1 | 0.753 | 0.03 | baseline |
| 2 | 0.784 | 0.49 | rebound/steal resampled to a 20% window |
| 3 | 0.793 | 0.40 | per-class stratified split |
| 4 | 0.822 | 0.84 | rebound windowed by its place in the action sequence |
| 5 | **0.816** | 0.78 | rebound capped at 436 to reclose the confound |

Every move was a data or measurement fix.

### Fix one: sample the action, not the clip

BARD source clips run 8–10 s and the sampler took 16 frames evenly across the
whole clip — half a second apart, against an action lasting about a second.
Fourteen of sixteen frames showed unrelated play, so a rebound clip and a steal
clip were largely *the same footage*.

| crop margin | window | accuracy | lift |
|---|---|---:|---:|
| 0.25 | 1.0 (whole clip) | 0.621 | +0.121 |
| 0.25 | 0.2 | 0.713 | +0.213 |
| 1.0 | 0.2 | 0.717 | +0.217 |

Nearly double the lift at both crop margins, while the margin itself changed
nothing. Five spatial hypotheses had already failed; the problem was never *what*
was in frame, it was *when*.

### Fix two: a validation split that cannot drift

Block appeared to fall from 0.90 to 0.67 — mostly measurement artifact. The split
concatenated every class in `ACTIONS` order and shuffled globally, so it depended
on each class's **size**. Changing rebound from 226 clips to 223 reshuffled
block, steal and other while leaving dribble, pass and shot byte-identical:

```
dribble  run1  88  run2  88  in both  88
block    run1  79  run2  78  in both  15
```

Only 15 of 79 block validation clips survived between runs. For two runs, *every*
cross-run per-class comparison in this project was partly comparing different
clips. With a per-class stratified split, block is 0.86 and never regressed.

### Fix three: use the clips that were being thrown away

Rebound sat at 0.40 and I had written it off as a data ceiling. It was not. BARD
has no timestamps, only an ordered list of actions per clip, and the selector
demanded the clip be unambiguous — 223 rebound clips out of the 4,709 that
contain one. But the exclusion was never that those clips are wrong; it was that
a midpoint window looks at the shot rather than the rebound. In 3,127 of them the
rebound is the second of two actions, so it sits about three quarters through.

Position the window at roughly `(n + 0.5) / m` for the nth of m actions. Rebound
went **0.40 → 0.78** at the same clip count as steal.

### The confound: measuring the dataset, not the action

The earlier headline of 0.810 was partly measuring *which dataset a clip came
from*. SpaceJam and BARD are visually distinguishable, and every rebound and
steal clip came from BARD while every other class came from SpaceJam — so corpus
membership predicted the label for 660 of 2,660 clips.

Augmentation cannot fix this, and measurably did not: image statistics still
separated the corpora 95% of the time after blur/brightness/contrast jitter, and
98% before. The fix was **composition** — populate `shot` and `other` from *both*
corpora using BARD's 2PT/3PT shot and foul/turnover events:

```
BEFORE  2,660 clips   corpus-only 0.314   majority 0.163   confound +0.150
AFTER   3,455 clips   corpus-only 0.241   majority 0.232   confound +0.010
```

`scripts/audit_source_cue.py` recomputes this on demand, and it bit back once:
generating 799 rebound clips made rebound BARD's largest class and reopened the
confound to +0.099.

| rebound n | corpus-only | majority | confound |
|---:|---:|---:|---:|
| 436 | 0.228 | 0.218 | **+0.010** |
| 600 | 0.261 | 0.209 | +0.052 OPEN |
| 799 | 0.298 | 0.199 | +0.099 OPEN |

Run 4 scored higher overall (0.822) and on rebound (0.84) and **is not the model
that was kept**. 0.816 with the confound closed is worth more than 0.822 with it
open.

The pass criterion is therefore not accuracy but **cross-source generalisation**:
`shot` and `other` are scored separately on their SpaceJam and BARD clips. Final
gaps are 0.06 and 0.05 against a 0.25 threshold — the model handles the same
action from either corpus.

### The `background` class

The full-game run (§10) emitted 2,253 rebounds against 83 real ones. A confidence
floor cannot fix this, because the model is *certain*: 60 random windows sampled
from continuous game footage came back 47 rebound at a mean confidence of 0.955.

The cause is that every training clip was **cut to contain an action**, so every
window the classifier ever saw was an action and it has no way to say "nothing
here." `background` is that missing class: per BARD clip, the window furthest
from every labelled action, taken only when it clears them by 0.18 of the clip. A
clip runs 8–10 s and its actions occupy about 1.6 s, so this is a player bringing
the ball up or resetting — real broadcast footage from exactly the distribution
the pipeline is served.

It is appended last in `ACTIONS` so no existing label index moves.

**What BARD cannot supply is dead time** — timeouts, free throws, inbounds. A
real game has more nothing-happening than this, which makes the training set
conservative rather than complete, and is why final validation still wants
continuous footage.

---

## 6. Possession — a measured ceiling

The spec called possession "a proximity heuristic, not ground truth." This is how
far that goes on real broadcast footage.

### The core difficulty, quantified

Over frames where the ball is detected and ≥2 players are present:

| quantity | value |
|---|---:|
| median distance, nearest player to ball | 0.44 body-heights |
| median distance, runner-up | 0.87 |
| **median separation** | **0.21** |
| frames where runner-up is within 0.3 | **61%** |

A defender is about as close to the ball as the ball-handler in most frames.
Proximity is genuinely under-determined here.

### What was tried and rejected

| approach | result | verdict |
|---|---|---|
| Centre-to-centre proximity | 61% of frames ambiguous | baseline |
| Box-edge distance | 69% ambiguous — *worse* | rejected |
| Box containment | resolves exactly one player in 25/67 frames | insufficient |
| Ball-motion correlation | 0/2 where proximity scored 2/2 | rejected |
| Learned `handler`, 191 instances | 1/4 vs proximity's 3/4 | rejected |
| Parameter sweep over distance/hysteresis/gap | **5/7, nothing beats it** | ceiling |

Motion correlation failed because the ball is detected in only ~63% of frames, so
a displacement window rarely has the ball at both endpoints; widening it spans too
much time to discriminate. The parameter sweep plateaued because the two remaining
failures pull in opposite directions — one needs a looser threshold to catch a
real handler, the other a tighter one to reject a ball in flight.

### What actually worked: weak supervision

Proximity is not *always* ambiguous. In ~12% of frames the ball is clearly nearest
one player and well clear of the runner-up. **Those frames label themselves**, and
training the detector on them teaches it the *appearance* of a ball-handler —
hands on the ball, body squared to it — which is the cue proximity can never
access in a crowd.

`scripts/harvest_handler_labels.py` collects them with deliberately strict
thresholds (nearest ≤ 0.45 body-heights, separation ≥ 0.55), because a wrong
pseudo-label is worse than a missing one. Harvested labels go to **train only**;
validation stays human-annotated so mAP keeps measuring against real annotations.

Yield improved as the detector improved — a virtuous loop:

| round | handler instances | fires | confidence | V6 |
|---|---:|---:|---:|---|
| 0 | 191 | 24% | 0.55 | 1/4 — worse than proximity |
| 1 | 680 | 49% | 0.85 | 5/7 |
| 2 | 3,550 | 60% | 0.84 | 6/7 → **9/9 on the expanded key** |

`raw_holder` now prefers the learned handler and falls back to nearest-player when
it does not fire.

### An evaluation-design lesson worth stealing

The V6 answer key originally stored **track IDs**. Retraining the detector
silently invalidated every entry while the gate went on reporting a confident
number. The key now stores **image positions** — a point on the ball-handler's
body is a fact about the footage and survives any model change.

---

## 7. Naming players without OCR

The spec deferred jersey-number OCR as "a genuinely hard, actively-researched
problem" — small text, motion blur, partial occlusion.

**But you do not have to recognise a player to name them.** Basketball already
produces an authoritative, timestamped event stream: the official play-by-play.
Given the game and roughly where in the game clock a moment sits, naming becomes a
**join**, not a recognition problem.

### The chain

```
scoreboard clock  →  game clock  →  play-by-play window  →  align()  →  name
   (template          (interpolate,     (nba_api             (only on a
    matching)          timidly)          playbyplayv3)        confident match)
```

**Reading the clock, without an OCR dependency.** The digits are large, bold and
near-black on a bright bar. `scoreboard.py` thresholds, takes connected
components, and keeps the tallest cluster — clock digits are 22 px against the
period label's 16, so height separates them without knowing the layout — then
template-matches each. Templates are per-broadcast, built from frames whose value
you know.

**It validates itself with no labelling at all**, because a game clock only counts
*down*. Built from one frame reading 10:59 — giving templates for 0, 1, 5 and 9
only — it read 14 of 21 sampled frames, every value monotonically non-increasing:

```
11:09 → 11:01 → 11:00 → 10:59 → 10:55
```

It **declined** the other seven rather than guessing. An unreadable clock is
normal (replays, timeouts, graphics over the bar); a wrong one silently mis-joins
every play that follows.

**The interpolation is deliberately timid.** A game clock is not a linear function
of video time — it stops for fouls, timeouts, free throws and reviews, and a
replay can run while it is stopped. So bracketing readings are used only when they
are close together and inside the same period, and extrapolation past the first or
last reading is refused outright. Returning `None` is not a failure.

**The feed.** `nba_feed.py` was verified against the live feed for game
0022400861: **476 usable plays out of 552 entries**, every one carrying period and
clock, 446 with a player. One quirk silently costs two of seven classes — in this
feed a **STEAL and a BLOCK carry an empty `actionType`** and are named only in the
description text. On that game the 40 entries with no action type were exactly the
17 steals and 23 blocks. Keying on `actionType` alone loses both with no error.

### The safety property

A misalignment does not produce a vague answer. It produces a **confidently wrong
name — a false statement about a real, identifiable person.** So `align` refuses
to guess: no action agreement, no name, and the event keeps its anonymous
`Player N`. `validate_commentary` (§4) enforces the same rule at the output.

### V12 — the whole chain on a real broadcast

`scripts/validate_real_game.py`, and it took two rounds to get honest:

```
531 frames at 60fps
templates from 6 known frames: 9/10 digits
clock read on 36 sampled frames; 0 dropped as impossible
clip covers P2 232s down to 227s remaining, monotonic
official feed: 413 plays; 1 inside the window
```

The first version **passed while reporting a clock that never moved** across 8.8
seconds. Built from one frame reading 3:47, it could only read values made of 3, 4
and 7 — and it did not decline the others, it matched a 7 against the 3 template
and returned a confident 3:43.

Comparing pipeline events against the official feed:

```
t=3.28s  other    P2 229s   official: A. Edwards other   AGREES
t=4.08s  rebound  P2 228s   official: A. Edwards other   differs
t=4.88s  steal    P2 227s   official: A. Edwards other   differs
t=5.68s  other    P2 227s   official: A. Edwards other   AGREES

2 of 6 events agreed on the action
```

Two things that number is **not**. It is not the join failing — the clock reads
correctly and selects the right official play every time. And "6 of 7 events fall
within 2 seconds" is not a success either, which is what the first version of this
gate reported: **the clock stops on a dead ball**, so four events spanning four
seconds of video all mapped to 227s and all matched the same play. Clock-based
joining cannot separate events inside a stoppage, and saying so is more useful
than the flattering number.

What 2-of-6 measures is the classifier.

### Why run vision at all, then?

Because play-by-play is *discrete events* and vision is *continuous*: where
players were, how the play developed, who was defending, off-ball movement. Fused,
they produce commentary neither could alone. That is the actual case for the
system.

---

## 8. Play recognition as geometry, not labels

### Court registration, and a trap

`court.register()` fits an image→court homography from named NBA landmarks and
reports its error in feet.

The trap it guards: a homography has 8 degrees of freedom and 4 point pairs give 8
equations, so **with exactly four correspondences the fit is exact and reports
zero error however wrong the inputs were.** Corrupting one by 300 px still reports
0.000 ft while the true held-out error passes 2 ft. Supply more than four, and
check `is_usable()` before trusting anything downstream.

### Finding the court automatically

`court_lines.court_line_mask()` finds painted lines, and measurement overturned
the obvious approach. The Pistons floor lines *look* navy, but thresholding blue
(hue 100–135) finds the bench area, floor advertising and the crowd while missing
every arc. The lines actually sample at hue 155–175, and far more reliably at
**grey 52–110 against a local median of 198–225**. Finding them as dark pixels
relative to a local median also survives the floor being brightly lit at the far
sideline and shadowed near the camera.

`search_registration()` then recovers the camera by differential evolution over
**six physical parameters** (camera position, aim point, focal length) rather than
a homography's eight free ones — because every point in the physical space is a
camera that could exist, and most of the 8-dimensional space is not.

Results: **0.36 ft** on a synthetic court with a known camera; on the real
broadcast it scores 0.42 and puts **10 of 11 detected players on the court** — an
independent check that never touches the alignment score.

Two instructive failures:

- **A one-directional score was not enough.** Asking only "do the model's lines
  land on detected lines" returned a camera scoring **0.998 whose court
  coordinates were hundreds of feet wrong** — it had zoomed onto a patch where two
  arcs coincided, so every projected point sat on a line. Measured against the true
  camera: recall 1.000 / coverage 0.801 versus recall 0.998 / **coverage 0.023**.
  The score is now the harmonic mean of both directions.
- **Random restarts do not find it.** 600 restarts plus Nelder-Mead stalled at
  0.233 against the true camera's 0.888. The good basin is narrow. Differential
  evolution finds it but needs its budget — at maxiter 25 and 60 it returns 0.198
  and 0.229, so a cheap run is a *wrong* answer rather than a fast one.

### V11 — validating registration by physics

The alignment score cannot say whether the resulting court *coordinates* are
right, and hand-annotating landmarks would only move the problem — my pixel
estimates become ground truth and could be wrong in the same way the registration
is. So `scripts/validate_registration.py` checks physics instead: **players do not
teleport**, so implied speeds must look like basketball.

```
12 frames registered, median score 0.397; 14 tracks, 102 steps
speed ft/s: p50 7.9   p90 21.7   p95 25.0   max 33.9
within sprint (25 ft/s): 95.1%
court extent: x -6.3..54.3 (court 0..50), y -1.9..39.0 (0..47)
```

A median of 7.9 ft/s is what sustained NBA movement looks like, and a registration
wrong in *scale* would inflate every one of these. **What V11 cannot see:** a
court offset by a constant passes unchanged — this measures relative motion, not
absolute position. The x extent running −6.3 to 54.3 against a 50 ft court is a
real hint of that.

### The labels-versus-definitions distinction

The recurring finding in this layer: **most named basketball actions are
definitions, not categories, and definitions need no labelled data.**

Built, needing no labels — every term measurable in court feet:

| detector | definition |
|---|---|
| pick-and-roll / pop / DHO | two offensive players converge, one with the ball; the screener then cuts to the rim, steps beyond the arc, or takes the ball |
| back / flare / pin-down / cross screen | *direction* the cutter takes relative to the rim and the ball |
| Spain PnR, double drag, re-screen | compositions of the above with timing and identity constraints |
| Horns, five-out, post-up, isolation | one-frame arrangements; returns `unknown` **with a reason** when unjustifiable |
| horns flare, horns set | a shape plus an action inside a two-second window |
| transition | 25 ft of ground toward the rim at ≥12 ft/s — separates a break from a walk-up |

"Horns Flare" was cited three times in my own notes as needing labelled play
types. It does not: horns is an arrangement visible in one frame, a flare is a
direction a cutter takes, and the set is their conjunction. Each time the claim
was re-examined it turned out to be a composition rather than a label.

**Not yet demonstrated on real footage.** The screen detectors are tested on
synthetic trajectories and have **never fired on the sample clip** — 0 on-ball and
0 off-ball across 26 registered frames. `scripts/diagnose_screens.py` says why:

```
f 4  pair(1,7)   6.8 ft now, 16.1 ft earlier  — separated, 1.8 ft short
f11  pair(3,5)   6.5 ft now, 11.3 ft earlier  — separated, 1.5 ft short
f12  pair(3,5)   5.5 ft now, 11.3 ft earlier  — separated, 0.5 ft short
```

The converging-after-separating pattern **is present**. Those pairs stop at 5.5–6.8
ft against a 5 ft contact threshold. Registration on real footage carries a couple
of feet of error, and a player's position is the bottom-centre of a box rather than
a point, so a genuine screen easily measures 6–7 ft. This is a resolution limit,
not a bug, and it is deliberately left alone — widening the threshold would
manufacture detections nobody can verify.

**The honest limit.** Set *calls* — coaching vocabulary — genuinely need labelled
play types, and BARD does not carry them: its captions are event-level (jersey
number, colour, one of nine action types). The distinction that survives: *flare*
and *horns* describe what bodies did; a *call* describes what a coach shouted.

---

## 9. Performance: profile, batch, split, fuse

### Profile first — and it changed the plan

The spec said profile before writing any CUDA. Doing so overturned the obvious
target:

| stage | % total | ms/frame |
|---|---:|---:|
| 1 extraction | 0.5% | 0.6 |
| 2 detection | 14.6% | 17.5 |
| 3 tracking | 0.8% | 1.0 |
| 4 team assignment | 4.8% | 5.8 |
| **6 action classification** | **76.8%** | **91.8** |
| 9 render | 2.4% | 2.9 |

Detection — the stage everyone assumes dominates — was 14.6%. And crucially,
**profiling *without* stage 6 showed detection at 92%**, which would have aimed
every hour of kernel work at the wrong stage.

### Two wins with no CUDA at all

| change | effect |
|---|---|
| Batch the action classifier's independent windows | stage 6 **76.8% → 35.8%** |
| Collapse duplicate overlapping-window events | 10 events → 5, better commentary |

Do the free wins before paying for hardware.

### The custom kernel (v2 §7.1)

Chosen by the profile, not intuition: detection and classification are standard
forward passes already running vendor-tuned cuDNN kernels, and hand-writing those
is a losing fight. **Team assignment is the largest stage with no optimised
implementation behind it** — plain NumPy and OpenCV, one Python call per player
per frame.

`kernels/torso_color.cu` fuses torso-crop → sRGB→CIELAB → mean colour. The win is
memory traffic and launch overhead, not arithmetic: the unfused version
materialises a crop, writes it, reads it back for colour conversion, writes Lab
pixels, reads them again to reduce. The kernel reads each pixel once, converts in
registers, and reduces in shared memory. One block per (frame, player) pair.

**Correctness.** `verify_kernel_numerics.py` compiles the real `.cu` source with
host CUDA stubs, runs it under 256 real threads with a real barrier, and compares
against an OpenCV oracle — worst channel error 0.462 against a 1.5 tolerance, and
the tree reduction matching the serial path to 0.0003. That runs in CI, so a
failure on a GPU box is a toolchain problem rather than a kernel bug. On real
CUDA it matched the oracle to 0.1456 against a 2.0 tolerance on every box tried.

**Speed, honestly.** It does not reproduce:

```
single-GPU box   fused 0.221 ms  vs reference 0.302 ms   1.4x, 3/3 runs
two-GPU box      fused 0.310 ms  vs reference 0.302 ms   1.0x, 1/5 runs
```

Same GPU model, same 32-vCPU Ryzen 7950X. **"1.4× faster" was a property of one
machine, not of the kernel, and should not be quoted.**

Two bugs only compiling could find. `load_inline` was called with
`cpp_sources=""`, so nvcc compiled the `.cu` perfectly and the build died in
torch's generated glue with `'torso_mean_lab' was not declared in this scope` — a
CUDA-shaped error that was not a CUDA problem. And the data upload carried 3,574
macOS AppleDouble files, one of which matched the `*.mp4` glob and crashed the
training run *after* the clips had shipped.

### Disaggregated serving (v2 §7.2) — verified, and not worth it

`serving.StagePlan` carries device placement as **data, not code**, so the same
pipeline is single-device locally and multi-GPU on a rented box:

```python
StagePlan(detection="cuda:0", action="cuda:1")   # 2-GPU box
StagePlan.single("cuda:0")                        # 1 GPU, for comparison
```

Threads rather than processes — torch releases the GIL inside device kernels, so
two threads driving two GPUs genuinely overlap, and processes would add IPC cost
for the frame tensors, which is the opposite of the point.

The logic is verified without CUDA (`verify_disaggregation.py`): 12 windows from
12 classifier calls, backpressure applied at `queue_size=1` (the producer waits
0.27 s rather than buffering the clip), and a raising classifier propagating
instantly rather than deadlocking.

On two 4090s with the real models:

```
single cuda:0   0.27s   windowing wait 0.214s   classification wait 0.0
split  0/1      0.28s   windowing wait 0.227s   classification wait 0.0
```

**Splitting is marginally slower.** The waits say why: classification never waits,
so the classifier is not the bottleneck — window construction is. Disaggregation
helps when two stages contend for one device, and here they do not. The machinery
is correct; the premise does not hold for this workload.

That is a real result. It cost a few dollars to learn and it prevented a whole
optimisation track built on a false premise.

---

## 10. What 84 minutes of real footage broke

Everything above was validated on clips of eight to ten seconds. This run
processed **50,304 frames** — 570× the reference clip. It is the most important
result in the project.

Footage: 268 broadcast clips from one game, concatenated. Real NBA video, but with
hard cuts between plays rather than one continuous feed (which matters for
Finding 2 and not for the others).

### The pipeline scales

```
1-3 extract/detect/track   1095.6s  38.3%
4   team assignment         419.1s  14.6%
5   possession               12.7s   0.4%
6   action classification   512.3s  17.9%
9   render                  822.5s  28.7%
TOTAL                      2862.2s   →  47.7 min for 84 min of video
```

**1.76× real time on one RTX 4090**, memory flat in clip length. That half passed
cleanly.

### Finding 1 — the classifier is calibrated for clips, not games

3,383 events over 84 minutes — about 40 a minute, where real basketball produces
four to eight.

| action | emitted/min | realistic/min | ratio |
|---|---:|---:|---|
| rebound | 26.9 | 1.8 | **15× too many** |
| steal | 3.4 | 0.3 | 11× too many |
| shot | 1.6 | 3.5 | half as many |
| pass | 0.2 | 9.6 | **48× too few** |

Rebound alone is 2,253 of 3,383 events — 67% of everything reported. Scored
against BARD's own labels for that game (`scripts/evaluate_game.py`):

```
shot     137 vs 180    0.76x
rebound 2253 vs  83   27.14x
steal    283 vs  18   15.72x
block     13 vs  14    0.93x
other    451 vs 130    3.47x
```

**Shot and block were already right. The failure is specific, not general.**

0.78 held-out accuracy on rebound clips and 27× over-prediction on a game are both
true at once. That is what distribution shift looks like — see §5 for the
`background` class built in response.

### Finding 2 — cuts fragment tracking, and events inherit the damage

**8,602 track IDs**, against 19 on a short clip. ByteTrack has no re-identification
by design, so every one of the 268 cuts restarts identity for all ten players.

That propagates: `build_events` collapses consecutive windows sharing an action
*and* a holder, but consecutive windows rarely share a holder ID when IDs churn,
so de-duplication does almost nothing. The median gap between reported events is
0.80 s — exactly the window stride, meaning nearly every window emitted an event.

A continuous feed would fragment far less than this concatenation does, so treat
8,602 as a worst case. The mechanism is real regardless: **possession-level
identity needs to survive a camera cut, and today it does not.**

### Finding 3 — the render is impractical at length

4.8 GB for 84 minutes, written frame by frame with `mp4v`. Fine for a ten-second
clip, unusable as a deliverable.

### What this changed

The honest claim is now narrower and better evidenced: **the pipeline runs on
game-length video, and the action model is not yet usable on it.** Everything
measured on clips remains true on clips.

---

## 11. Verification strategy

Three layers, deliberately distinct:

**1. Unit tests — 287 cases, ~3,000 lines.** One test module per source module.
These pin behaviour that is cheap to break silently: the JPEG chroma error
(§3), the possession hysteresis, the commentary validator's both-directions
check, the kernel's tree reduction against a serial path.

**2. Gate scripts — `validate_v1.py` … `validate_v9.py`, plus V11/V12.** One per
pipeline stage, each printing `PASS`/`FAIL` with the metric that justifies it.
Run in dependency order and not composed until each passed alone.

**3. Audits that check the *measurement*, not the model.** This is the layer that
found most of the real problems:

- `audit_source_cue.py` — is accuracy measuring the action or the dataset? (§5)
- `report_action_confusion.py` — cross-source generalisation as the pass criterion
- `validate_registration.py` — physics as ground truth where labels would beg the
  question (§8)
- `evaluate_game.py` — score against a game's own labels, not held-out clips

Two structural lessons from failures:

- **The V6 answer key stores image positions, not track IDs**, because retraining
  the detector silently invalidated an ID-based key while the gate reported a
  confident number.
- **The GPU bundle is built from an include list, not an exclude list**, because
  an exclude list goes stale the moment a new directory appears and the failure
  mode is a silent 12 GB upload to a machine billing by the second. The suite is
  run from an extracted bundle in an empty directory, so it cannot quietly lean on
  the development tree.

---

## 12. Honest limitations and roadmap

| area | state |
|---|---|
| v1 pipeline | Done — 11 gates, 287 tests, runs end to end |
| Action classifier on **clips** | 0.816, every class 0.76–0.90, confound closed |
| Action classifier on **games** | **Not usable.** 27× over-prediction on rebound; `background` class built, not yet retrained and re-measured |
| Tracking across cuts | Broken by design — ByteTrack has no re-ID. 8,602 IDs on 84 minutes |
| Render at length | 4.8 GB / 84 min with `mp4v`; needs segment extraction or a modern codec |
| Possession | 9/9 on the answer key; the underlying signal is genuinely ambiguous in 61% of frames |
| Player naming | Working end to end against the official feed; cannot separate events inside a dead-ball stoppage |
| Court registration | 0.36 ft synthetic, 0.42 score on real footage — works, but not unattended |
| Screen / set detection | Geometric sets implemented; **never fired on real footage** — a 1–2 ft registration resolution limit |
| Set *calls* | Genuine data gap. No dataset carries them |
| CUDA kernel | Compiles, matches the oracle; speedup does not reproduce across machines |
| Dual-GPU split | Verified, and measurably not worth it for this workload |

**Next, in value order:**

1. Retrain with `background` and re-run `evaluate_game.py` on the full game — the
   single change most likely to make game-length output usable.
2. Sample dead time (timeouts, free throws, inbounds), which BARD cannot supply.
3. Re-identification across cuts, so possession-level identity survives a camera
   change.
4. Segment-based rendering instead of a single full-length file.
5. Improve registration to the point where the screen detectors can fire.

---

## 13. Reproducing this

```bash
python -m venv .venv && ./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/python -m pytest -q
```

Run the pipeline on a clip:

```bash
python -m scripts.run_pipeline clip.mp4 --out outputs/run
```

Live play-by-play (optional, network) needs `pip install -e ".[live]"`. It is
optional on purpose: the pipeline runs on archived clips with no network at all.

GPU work — training V7, the CUDA kernel, and the disaggregation check — is a
rented-box workflow documented step by step in
[docs/v2-gpu-runbook.md](v2-gpu-runbook.md), including what to rent, what *not* to
rent (VRAM is nowhere near the constraint; YOLO11n is 5 MB and VideoMAE-base 344
MB), and what to upload. One command runs everything in dependency order:

```bash
bash scripts/run_gpu_suite.sh 2>&1 | tee outputs/suite.log
```

It starts with the corpus audit deliberately: if the clips did not all transfer,
every number after that is meaningless and you want to know in the first thirty
seconds, not after the training run.

---

## Data attribution

- **BARD** (Basketball Action Recognition Dataset), Gabriele Giudici, 2025 —
  <https://github.com/GabrieleGiudic/BARD> — CC BY 4.0. Clips are sourced from NBA
  broadcast footage; the CC BY licence covers the annotations.
- **basketball-player-detection-3** (v18), Roboflow Universe, workspace
  `roboflow-jvuqo` — CC BY 4.0. Used for detector training.
- **SpaceJam** — action clips, used for the non-BARD half of the action corpus.
- Official play-by-play via `stats.nba.com`, wrapped by `nba_api`.

## Further reading

[Full-game findings](full-game-findings.md) ·
[V7 GPU results](v7-gpu-results.md) ·
[v1 profile](profile-v1.md) ·
[Possession investigation](possession-investigation.md) ·
[Live data & play recognition](v3-live-data.md) ·
[GPU runbook](v2-gpu-runbook.md) ·
[Original v1 spec](spec-v1.md)
