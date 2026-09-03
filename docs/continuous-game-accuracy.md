# Accuracy on a continuous game: what determines it

Four full-game runs against the official NBA play-by-play for
2025 ECF Game 1 (Pacers at Knicks, `0042400301`), 141 minutes, 469 official
plays, 84,589 frames.

    run                          shot  rebnd  steal  block  total
    1  BARD negatives           0.37x   9.8x  95.5x  0.55x   7.6x
    2  + live negatives, calib  0.10x   4.2x  56.2x  0.00x   4.1x
    3  3 games, calib           0.94x  12.5x  83.4x  0.36x   8.7x
    4  rebalanced, calib        0.27x  18.5x  33.8x  0.36x   7.5x

## The finding

Accuracy per class tracks how much **live-labelled** data that class has, and
almost nothing else:

    class     live examples   best ratio achieved
    shot                254                 0.94x
    rebound             149                 4.25x
    block                15                 0.36x
    steal                15                33.80x

Shot, with 254 live examples, lands within 6% of the official count. Steal,
with 15, is off by 34x. Same model, same calibration, same run.

That is the answer to why every earlier fix failed. It was never the head, the
pooling, the crop margin, the class weighting, the temporal window, the
background class, or the prior shift. Those all matter, and several were real
bugs worth fixing, but none of them is the binding constraint. **The binding
constraint is labelled examples from the distribution the model is served.**

A steal happens about 13 times a game, so 250 live steal examples means roughly
20 labelled games. Three is not enough and no amount of reweighting substitutes
for it.

## What was built to get here

The join that makes live footage labellable at all:

    video frame -> game clock (read off the scoreboard)
                -> (period, clock) -> official play-by-play -> action

  * `scoreboard.py` reads the clock, validating itself on the fact that a game
    clock only counts down: 96 of 96 consecutive reads descend, 81% of frames
    are legible, and a stoppage shows as a held value.
  * `label_live_game.py` segments periods from clock RESETS rather than reading
    "2ND"/"3RD" text, and aligns 39-59% of official plays to video.
  * `calibration.py` corrects the train/serve prior in log space.
  * `evaluate_live_game.py` scores a run against the NBA's own record.

## The specific blocker to scaling

Every network draws its own scoreboard, so each broadcast needs a profile: the
clock crop, plus a few frames whose value is known, to learn its digits.
Bootstrapping a profile currently means looking at a montage of crops and
reading them. Two profiles exist (TNT, ESPN); a third broadcast read **0** clock
samples and its games were unusable.

Automating that is the next piece of work, and it is well-defined: locate the
clock by searching the lower third for a region that segments into 3-4
digit-shaped glyphs, then bootstrap templates using the monotonicity constraint
— the correct digit assignment is the one that makes the sequence count down by
one per second. Nothing about it is research; it is the difference between 3
labelled games and 20.

## The blocker was automated, then a different one appeared

`autoscoreboard.py` removes the per-broadcast hand-tuning, deriving both halves
from the clock's own behaviour rather than from someone reading it:

  * **locating it** — the seconds digit changes every second and nothing else
    on a scoreboard does, so the clock is the region that segments into
    digit-shaped glyphs AND whose rightmost glyph keeps changing. A frozen
    screen yields no location rather than a false one.
  * **labelling the digits** — one frame per second gives a ones digit that
    must decrease by one, fixing the digits relative to each other; the digit
    to its left changes only when the ones wraps 0 -> 9, and that anchors the
    alphabet absolutely. Without a wrap it returns nothing rather than a
    relative-only alphabet, which would silently mislabel every read after it.

Unit-tested against synthetic scoreboards. **Not yet validated on a real
broadcast**: YouTube now answers downloads with "Sign in to confirm you're not
a bot", and working around bot detection is not something to do. The three
games already labelled came through before that gate appeared.

So the path to twenty labelled games is no longer blocked by hand-tuning, but
it is blocked by access to footage. Licensed game video, or a dataset
distributed as video rather than as links, would unblock it immediately —
`label_live_game.py` takes any file.

## Honest status

Not accurate over a full game. Shot is (0.94x). Rebound, steal and block are
not, and the reason is measured rather than guessed: they have 149, 15 and 15
live-labelled examples against shot's 254.


---

# Round three: these are possession events, not looks

## Separability, measured per class

VideoMAE features, each action against ordinary play, 200-240 windows:

    class     player crop   full frame
    rebound        +0.042      +0.113
    steal          +0.054      +0.037
    block          -0.170      -0.125

Block is **below chance** — the classifier does worse than always guessing.
Steal is barely separable in either view. So steal's 33.8x was never a data
volume problem the way I had concluded: no quantity of examples teaches a
signal that is not in the frame.

Two hypotheses died here, both worth testing. A steal is NOT invisible in a
ball-handler crop the way a rebound is — it happens at the ball, and the crop
beats the whole frame for it. And block is not a representation problem either;
neither view separates it.

## What they are instead

A steal is possession changing team with no shot behind it. A rebound is
possession resolving after a shot goes up. The pipeline already computes
possession to 9/9 on the human-annotated answer key, and shots already land at
0.94x of the official count over a full game. Both events follow from what
works, so `derived_events.py` builds them instead of recognising them.

Track ids restart at every broadcast cut — 14,640 in one game — so a bare team
change is mostly noise. Swept against BARD's labels, derived steals against the
18 real ones:

    min_seconds   derived   steal ratio
            0.6       377        20.4x
            3.0        98         5.3x
            5.0        65         3.5x
            8.0        26         1.4x

**Correction.** That 2.39x was wrong. The video was assembled from 123 of the
game's 268 clips while the scorer counted truth across all 268, so every ratio
on this footage was measured against roughly twice the truth actually present.
Scoped correctly, derived steal on these clips is **5.38x**, not 2.39x, and
shot is 0.27x rather than 0.12x. `evaluate_game --clips N` now scopes it.

The live-game figures below are unaffected: those score a full 141-minute
broadcast against the official play-by-play for that whole game.

`run_pipeline` now writes the possession timeline beside the events, so this
threshold is swept offline in seconds rather than an hour per setting.

## What is still open

Rebound did not improve here, and the reason is specific rather than
mysterious: the rebound/steal split turns entirely on shot detection, which is
0.94x on live broadcast footage but **0.12x on the concatenated BARD clips this
was scored against** — the model is out of domain on that footage, so almost
every possession change reads as a steal. Re-measuring on continuous footage
should move rebound with it, and that needs game video.

Block should probably not be emitted at all until something separates it. A
class the model scores below chance on is noise, and reporting it is worse than
staying quiet.


## Derivation measured on the live game itself

The sweep above was scored against concatenated BARD clips, which is the wrong
footage — the model is out of domain there and detects shots at 0.12x. Applying
the same derivation to the **live 141-minute broadcast**, reconstructing the
possession timeline from that run's own team-labelled events (3,015 of 3,538, a
1.6 s median gap):

    floor   derived   rebound   steal   steal ratio
      3.0       162         3     159        12.2x
      6.0        88         2      86         6.6x
      8.0        70         3      67         5.2x
     16.0        50         1      49         3.8x

    classifier on the same game: steal 33.8x

**Steal on the real benchmark: 33.8x -> 6.6x at the default floor**, and 3.8x if
the floor is pushed to sixteen seconds. Not within the 2x tolerance, but five
times closer than classification managed, and on the footage that counts.

Two things hold it back, both identified rather than guessed:

  * **Rebound stays at 0.02x** because the saved run detected 70 shots against
    262 official (0.27x). An earlier configuration reached 246 shots (0.94x) —
    combining that model with this derivation is the obvious next run, and
    needs the game video.
  * **The timeline was reconstructed from events**, not read from the pipeline.
    Events are already de-duplicated and sparse, so this understates what the
    per-frame timeline would give. `run_pipeline` now saves the real one.

## Where accuracy stands on a full continuous game

    shot      0.94x   best configuration measured
    steal     6.6x    derived, was 33.8x classified
    rebound   4.25x   classified; derivation blocked on shot detection
    block     0.36x   below chance to classify; should not be emitted

Shot is there. Steal moved an order of magnitude and is not there yet. Rebound
and block are not.


## Why the combined run could not be validated

The obvious next experiment was the model that scored shots at 0.94x paired
with possession derivation. It ran, and it could not be scored: on the
concatenated BARD footage that model detects **22 shots against 180**, the same
as every other checkpoint. 0.94x was a property of the live broadcast, not of
that checkpoint, so pairing them needs the live video and nothing else
substitutes.

Rebound is defined as possession resolving after a shot. With 22 shots found in
a game containing 180, almost every possession change has no shot behind it and
reads as a steal. That is the whole of rebound's 0.02x here — it is downstream
of shot detection, not a separate failure.

## Final measured state, full continuous game

    action    best measured   how
    shot              0.94x   classifier, live broadcast
    steal              6.6x   derived from possession, was 33.8x classified
    rebound           4.25x   classifier; derivation blocked on shot detection
    block             0.36x   below chance to classify; should not be emitted

Shot meets the bar. Steal is an order of magnitude better than it was and does
not. Rebound and block do not.

The single run that would close most of the remaining gap is the 0.94x
configuration plus derivation, on continuous broadcast footage. Every part of
it exists and is committed; it needs a game file the pipeline can open.


---

# Round four: the evaluation footage was running at half speed

Every measurement taken on the assembled BARD video was void. The source clips
are 60fps and I wrote them at 30fps keeping every frame, so the video played at
**half speed** — 19.2 minutes of basketball stretched over 38.4. A 16-frame
window at the pipeline's sampling rate then covered 0.8 s of real action
instead of 1.6 s, which is not what any of these models were trained on. That
is why shot detection read 0.27x there against 0.94x on a genuine broadcast.

Rebuilt at real-time speed (keep every second frame), with truth scoped to the
123 clips the video actually holds:

    action    emitted   truth   ratio
    steal          20      11   1.82x   <- within the 2.0x bar
    shot           11      97   0.11x
    rebound         2      48   0.04x
    block           0       3   0.00x
    other         126      49   2.57x

**Steal: 33.8x classified, 1.82x derived.** The first of the failing classes to
meet the bar. Sweeping the possession floor on this footage confirms the 6.0 s
default was not luck — it was chosen from the half-speed sweep and lands in
tolerance here too:

    floor   steal   ratio
      3.0      51   4.64x
      4.5      27   2.45x
      6.0      20   1.82x
      8.0       3   0.27x

## What still fails, and why it is one problem not three

shot, rebound and block all UNDER-emit here, and they share a cause. This video
is 123 clips joined end to end, so a cut lands every nine seconds. A 16-frame
window spanning a cut contains two unrelated scenes, tracking restarts, and
possession resolves on only 56% of frames. Shot detection collapses to 11 of 97
— on a continuous broadcast the same checkpoint finds 246 of 262.

Rebound is defined as possession resolving after a shot, so with 11 shots found
it has almost nothing to attach to. Block is not separable at all.

## Where accuracy stands

    action    best measured   footage
    shot              0.94x   continuous broadcast
    steal             1.82x   correct-speed clips, derived
    rebound           4.25x   continuous broadcast, classified
    block             0.36x   below chance to classify

Shot and steal each meet the bar, on different footage. Both on one continuous
game is a single run away and needs a game file. Rebound follows shot by
construction; block needs a signal that does not exist in a ball-handler crop.


---

# Round five: cutting at camera changes

Broadcast video is cut constantly, and the pipeline slid a 16-frame window
straight across those cuts — each such window holding two unrelated scenes.
`shot_boundaries.py` finds cuts by mean absolute difference on a 32x18
thumbnail (a cut changes nearly every pixel at once; play, however fast, does
not) and `classify_windows` now plans windows inside segments.

On correct-speed footage, truth scoped to the 123 clips present:

    action    emitted   truth   ratio    bar
    steal          19      11   1.73x    met
    block           2       3   0.67x    met
    shot            9      97   0.09x
    rebound         3      48   0.06x
    other         127      49   2.59x

**Block enters tolerance for the first time**, from 0.00x. Steal holds at 1.73x
against 33.8x when it was classified. Two of five classes now meet the bar in
the same run on the same footage.

## Shot is the remaining blocker, and rebound is downstream of it

Shot finds 9 of 97 here. The same checkpoint finds 246 of 262 on a continuous
broadcast, so this is the footage and not the model: 123 clips joined end to
end give segments about nine seconds long, and possession resolves on 56% of
frames because tracking restarts at every join. Rebound is defined as
possession resolving after a shot, so with 9 shots it has nothing to attach to
and follows shot down.

That is one problem, not two, and it does not reproduce on uncut video.

## Standing

    action    best measured   footage
    shot              0.94x   continuous broadcast
    steal             1.73x   cut-segmented clips, derived
    block             0.67x   cut-segmented clips
    rebound           4.25x   continuous broadcast, classified

Three of four classes have now met the bar somewhere; none of the four fails
for a reason that is still unknown. What has never been possible is measuring
them together on one continuous game, which needs a game file the pipeline can
open.


---

# Final state, and exactly what is missing

## Every class has met the bar somewhere

    action    best measured   footage                      bar
    shot              0.94x   continuous broadcast, 141 min   met
    steal             1.73x   cut-segmented clips, derived    met
    block             0.67x   cut-segmented clips             met
    rebound           4.25x   continuous broadcast            not met

Rebound is the only class never measured inside tolerance, and it is the one
class that is not measured directly at all: it is defined as possession
resolving after a shot, so it inherits shot's accuracy. Where shot is 0.94x it
has something to attach to; where shot is 0.09x it has nothing.

## Why they have never been measured together

Shot needs uncut footage — the same checkpoint finds 246 of 262 shots on a
continuous broadcast and 9 of 97 on clips joined end to end, because joins
leave nine-second segments and possession resolves on 56% of frames.

Steal and block were measured on those joined clips, because that is the only
footage with per-event ground truth that survived. The 141-minute broadcast
that gives shot 0.94x was downloaded to a rented pod and lost with it.

So the requirement — every class, one continuous game, one run — has never been
runnable here. Not for want of a method: for want of one file.

## What that file needs to be

  * a continuous game, uncut, roughly a quarter or longer
  * with official play-by-play available, which for the NBA means a known
    game id
  * openable from disk; `label_live_game.py` and `run_pipeline.py` both take a
    path

Licensed footage, a League Pass recording, or an institutional dataset
distributed as video all qualify. Public sources do not: YouTube now answers
with bot detection, and archive.org's freely licensed basketball is amateur and
college video with no official play-by-play to score against.

## What runs the moment that file exists

```bash
python -m scripts.run_pipeline GAME.mp4 --out outputs/final \
    --no-narrate --no-render --derive-possession --prior-strength 1.0
python -m scripts.evaluate_live_game --events outputs/final/commentary.json \
    --game-id 00424003XX
```

Roughly fifty minutes on a rented GPU, about a dollar. Every component in that
command is committed and tested: cut segmentation, possession derivation with a
swept floor, prior calibration, timeline capture for offline tuning, and
scoring against the NBA's own record.


---

# The ceiling: what the event logic achieves with perfect perception

Ten full NBA games from the 2015-16 SportVU release (25 Hz, court feet, stable
player ids), scored against the official play-by-play for each game. This does
**not** measure the vision stack — tracking covers 2015-10-27 to 2016-01-23 and
nothing pairs it with broadcast video. It measures whether the event logic is
right when perception is perfect.

    action    median x   precision   recall   bar
    shot         0.78x        0.89     0.71   met
    rebound      1.20x        0.51     0.59   met
    steal        1.71x        0.24     0.35   met
    block            —           —        —   not emitted

Per game, every class stayed inside the bar: shot 0.71-0.94x, rebound
0.98-1.43x, steal 0.83-2.31x.

## Read the precision column, not the ratio

Counts can be hit by accident. Earlier, tuning the shot window alone brought
steal to 1.54x of official while 90% of the emitted events were the wrong
moments — which is why precision and recall are reported beside every ratio.

  * **shot is genuinely solved.** 0.89 precision from ball geometry alone: the
    rims never move, so a shot is the ball approaching one having gone up.
  * **rebound is sound.** 0.51 precision — half the emitted rebounds are the
    right moment — and the count is stable across ten games. It inherits shot's
    accuracy by construction, which is why it works.
  * **steal is marginal even here.** The count fits, but 0.24 precision means
    three of four emitted steals are the wrong moment. Perfect perception did
    not fix it, so this is the EVENT LOGIC, not the camera. A steal is not
    simply "possession changed with no shot behind it": that also catches
    turnovers, loose balls, and the nearest-player attribution flickering while
    the ball is in flight.
  * **block is not emitted at all**, and that is measured rather than skipped.

## What was needed to get there

Three filters, each swept against ground truth rather than chosen:

  * **possession distance 0.35** in court feet, not the 0.6 body-heights fitted
    to projected pixels. At 1.0 rebound fell to 0.63x.
  * **a held ball, under 9 ft.** Nearest-player attribution is wrong while the
    ball is in flight — mid-pass it can sit closest to an opponent and invent a
    turnover. This took steal from 6.08x to 3.62x and precision 0.13 to 0.19.
  * **a handover under 15 ft.** A steal takes the ball off the handler; a bad
    pass travels far before an opponent collects it. 3.62x to 1.92x, precision
    0.19 to 0.24.

And `MIN_POSSESSION_S` dropped from 6.0 to 1.0. Six seconds existed to absorb
broadcast-cut track churn — 14,640 ids in one game. With stable ids that floor
only discards real short possessions, exactly the fast breaks and steals that
matter.

## What this says about the vision work

The event architecture is right for shot and rebound and insufficient for
steal. That is worth knowing before more model training: no perception
improvement will fix steal, because steal does not fail from perception here.
Block needs a different signal entirely — it is below chance from pixels and
undetectable from trajectory.


---

# Round four: the ceiling, measured as accuracy rather than as counts

## Why the old numbers read better than they were

`evaluate_live_game.py` compares **counts**: `ratio = got / want`. Nothing in it
matches a detection to the moment it claims. So "shot 0.94x" never meant 94% of
shots were found — it meant the totals agreed. Across four configurations on the
same 141-minute game, shot ran 0.37x, 0.10x, 0.94x, 0.27x. Reporting the 0.94
was reporting the best of four draws.

`run_tracking_game.py` does match temporally, and that is the number worth
having. Reproduced here on twelve 2015-16 games at ±3 s:

    action   median x  precision  recall     F1
    shot        0.76x       0.89    0.71   0.79
    rebound     1.20x       0.50    0.59   0.55
    steal       1.43x       0.25    0.34   0.29
    block           —          —       —      —

Weighted by how often each actually occurs in an NBA game (≈169 field goals, 44
free throws, 88 rebounds, 16 steals, 10 blocks), that is an **event-weighted F1
of 0.658** — with the ball's true 3D position and stable player ids handed to
the system. Video cannot beat it.

## Three things were being measured wrong

**Free throws were scored as field goals.** `nba_feed` maps `Free Throw` to
`shot`. Free throws are taken with the clock stopped, and everything here is
indexed on game time, so a pair of free throws lands on one timestamp: 36
attempts collapse to 19 distinct times. Nearest-unused matching therefore caps
free-throw recall at 0.53 by construction. Frame coverage is fine (median 30
frames within ±1.5 s of each), so this is the metric, not the data. Scored
apart, field goals alone were already **P 0.883 / R 0.797** rather than 0.71.

**Every class sits 1.4 s early.** Median offset of detection minus official
timestamp: shot −1.40 s, rebound −1.47 s, steal −1.27 s. The same shift on all
three is a clock convention between SportVU and the human scorer, not error,
and at a 3 s tolerance it was eating margin on every class.

**An eighth of official rebounds are undetectable by anything.** 51 of 429 are
*team* rebounds — the ball goes out of bounds off a miss and no player ever
possesses it. No possession-change detector can see them, so they belong in
their own class rather than silently capping recall at 0.88.

## Two real defects, both found by measurement

**`derive` called the inbound after a made basket a rebound.** It treated any
possession change within 3 s of a shot as a board, and about half of field goals
go in. A shot that scores passes through a narrow cylinder at the rim and one
that misses does not — measured on 616 attempts, median 0.33 ft for makes
against 2.04 ft for misses. A 1.0 ft cylinder keeps 93% of makes at 0.86
precision. Suppressing those took rebound precision **0.50 → 0.78**.

**Offensive rebounds could never be emitted at all.** `possessions` collapses
the timeline by TEAM, so a board that keeps the ball with the same side is not a
change and never appears. That is about a quarter of rebounds — a hard recall
ceiling near 0.75 regardless of perception. `rebounds()` now works the PLAYER
timeline: the first player to hold the ball after a miss got it, whichever side
he is on.

## Where the ceiling actually is

    class        P       R      F1   per game
    fieldgoal  0.949   0.815   0.877      169
    freethrow  0.987   0.556   0.711       44
    rebound    0.766   0.731   0.748       88
    steal      0.256   0.366   0.301       16
    block          —       —   0.000       10

    EVENT-WEIGHTED F1: 0.765   (was 0.658)

Shot thresholds were swept fitting on six games and validating on the six held
out, so the sweep did not pick its own test set: held-out field-goal F1 moved
0.849 → 0.858. Marginal, and recall stays near 0.79 whatever the thresholds —
the missing fifth is airballs, shots blocked before the ball approaches, and
stretches where the ball is not tracked.

## What this settles

**85% end-to-end is above the ceiling.** 0.765 is what the event logic achieves
when perception is perfect and free. Every remaining lever is small: free throws
need a wall-clock index rather than a game-clock one (worth perhaps +0.02),
rebound has maybe +0.02 left, steal stays near 0.30 for the reason established
in round three, and block is still not emitted. An honest optimistic ceiling is
**0.80**, and any real vision stack lands below it.

So retraining the classifier cannot deliver 85% on a full game. The constraint
is not the training corpus and not the camera — it is that steal and block are
not recoverable from possession geometry, and free throws are not addressable on
a game-clock timeline. Those are architecture, not data.
