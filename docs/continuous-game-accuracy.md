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

## Held out, and two corrections to the paragraph above

The 0.765 above is measured on all twelve games, six of which the shot
thresholds were fitted on. Scored only on the six **held out**, with per-class
clock offsets also fitted on the other six:

    class          P       R      F1   per game
    fieldgoal    0.944   0.787   0.859      169
    freethrow    1.000   0.539   0.701       44
    rebound      0.755   0.706   0.730       88
    steal        0.305   0.362   0.331       16
    block            —       —   0.000       10

    EVENT-WEIGHTED F1 (held out): 0.751

**Per-class clock offsets are worth nothing.** Free throws really do sit at a
different offset from live play (−0.7 s against −1.4 s), but correcting it moved
the held-out score from 0.750 to 0.751. The offset is a real property of the
data and not a usable lever.

**The free-throw guess was wrong.** The hypothesis was that resampling onto a
game-time grid discards the free throw itself, since the clock is frozen. It
does not: FT windows carry a median of 40 distinct ball positions, exactly as
field-goal windows do. The footage is there. Free-throw recall is limited
because only about 55% of free-throw stoppages have any rim approach detected
near them at all — the geometry is marginal, with median ball-to-rim distance
running to 3.2 ft against a 4.0 ft threshold.

So the earlier estimate that free throws were worth +0.02 was optimistic by
about twentyfold, and the honest optimistic ceiling is nearer **0.78** than
0.80. The conclusion does not move: **0.75 held out is what this architecture
reaches when perception is perfect and free**, and 85% end-to-end on video is
not reachable from there.

## Two different questions, and only one of them can reach 85%

Sweeping the derivation parameters I had set by hand changed almost nothing,
fit on six games and validated on six: rebound 0.730 → 0.733, steal 0.331 →
0.373. Together worth about +0.003 event-weighted. The 0.75 is not a tuning
artefact — it is where this architecture sits.

But "85% end-to-end" has two readings, and they are not close to each other:

**(a) What fraction of the game's events does it capture?** That is the
event-weighted F1 above: **0.75**, and no lever found here moves it far.

**(b) What fraction of what it SAYS is true?** That is precision on the emitted
stream, and it is the question a commentary system is actually judged on —
saying something wrong is far worse than staying quiet. On the six held-out
games:

    policy                        says   right   PRECISION   coverage
    everything it can emit        1541    1292       0.838      0.719
    drop steal                    1433    1259       0.879      0.701
    shots + free throws only       925     875       0.946      0.487
    field goals only               834     785       0.941      0.437

**Dropping steal puts precision at 0.879 while still covering 70% of the
game's events.** That clears 85% on reading (b), held out, and it clears it by
suppressing the one class measured to be broken rather than by tuning.

## What is still not established

Every number on this page is measured on tracking coordinates: the ball's true
3D position and stable player ids, handed over for free. **The vision gap is
unmeasured.** A broadcast pipeline has to estimate ball height without depth,
re-identify players across cuts, and it will land below these figures by an
amount nobody here has quantified.

So the honest position is:

  * 85% of a game's events captured, end-to-end on video — **no**, and the
    ceiling argument says no training corpus changes that.
  * 85% of emitted commentary correct — **yes at the ceiling (0.879)**, and the
    open question is how much of that survives real perception.

The cheapest way to close the remaining uncertainty is not a retrain. It is to
run the existing vision stack against this same scorer on one continuous game
and measure the drop.

## The vision gap, measured by degrading the perfect coordinates

The objection to everything above is fair: it all assumes perception the camera
cannot give. That gap does not need a GPU to size. Take the true coordinates and
corrupt them the way a broadcast pipeline does — no depth, so ball height is
estimated; homography error in x/y; the ball lost behind bodies; track ids
restarting at every cut — then re-measure. Held-out six games, drop-steal policy:

    perception                                PRECISION   coverage
    perfect (tracking data)                       0.879      0.701
    ball height +-1 ft                            0.851      0.638
    ball height +-2 ft                            0.812      0.627
    ball height +-3 ft (no depth)                 0.788      0.630
    ball xy +-1 ft (homography)                   0.798      0.703
    ball missing 20% (occlusion)                  0.812      0.707
    ball missing 40%                              0.778      0.707
    track ids restart every 8 s (cuts)            0.875      0.704
    REALISTIC  z2 xy1 drop20 cuts8                0.730      0.668
    OPTIMISTIC z1 xy0.5 drop10 cuts20             0.817      0.659

Three things fall out.

**Ball height dominates.** It is the one quantity a single broadcast camera
cannot measure, and it is what both shot detection and the make/miss cylinder
are built on. At ±3 ft — which is what estimating height without depth looks
like — precision falls to 0.788 on its own.

**Track-id churn barely matters** (0.875). That is worth knowing: it says the
player-timeline rebound detector is robust to broadcast cuts, which the old
team-timeline design was not. The 14,640 spurious ids in one game were a
problem for the architecture that has now been replaced.

**Nothing reaches 0.85 once perception is realistic.** Tightening the rebound
policy buys a little headroom — hold 2.5 s with a 2.5 s window gives precision
0.895 — but at coverage 0.598, and that is measured with perfect perception.
Applied on top of optimistic vision it lands near 0.83.

## Conclusion

    reading                                        result
    85% of a game's events captured                no  — ceiling 0.751
    85% of emitted commentary correct, perfect      yes — 0.879
    85% of emitted commentary correct, real vision  no  — 0.73 to 0.82

Confidence that this architecture reaches 85% end-to-end on full NBA broadcast
video: **low, and now measured rather than guessed.** The binding constraints
are that a single camera cannot see ball height, and that steal and block are
not recoverable from possession geometry at all. Neither is a training-data
problem, so retraining the classifier does not address either one.

The honest next step is the cheap one: run the existing vision stack against
this scorer on one continuous game and check the real drop against the
simulated one above. If the measured drop matches the OPTIMISTIC row, a
precision-first configuration ships around 0.82. If it matches REALISTIC, it
ships around 0.73.

---

# Round five: the scoreboard closes the vision gap

## The previous degradation was unfair in one way and too kind in another

Two objections to round four's simulation, both tested rather than argued.

**Arc fitting is real but small.** Independent Gaussian noise on ball height per
frame is close to a worst case for a quantity that is physically a parabola. A
local quadratic fit recovers 0.730 → 0.745. Worth having, not a rescue.

**"Systematic homography error cancels" was wrong.** It came out slightly worse
than iid noise (0.734 against 0.745). The reasoning was that ball and rim map
through the same warp so a shared bias cancels — but `RIMS` are hardcoded court
constants, not detected, so a ball bias does not cancel at all.

## The scoreboard recovers the entire gap

A made basket is a score change. It is read, not inferred. Held-out six games,
drop-steal policy, realistic vision (height ±2 ft, homography ±1 ft, ball
missing 20%, cuts every 8 s):

    configuration                              PRECISION   coverage
    realistic vision, makes from geometry          0.734      0.674
      + scoreboard read 90% correct                0.868      0.655
      + scoreboard read 95% correct                0.875      0.663
      + scoreboard read 99% correct                0.880      0.671
    harsh vision z3 xy1.5 drop30 + scoreboard 95%  0.883      0.606
    perfect coordinates, makes from geometry       0.879      0.701

Two things worth stating plainly. **Even at 90% score-reading accuracy it clears
0.85**, so the lever does not need a perfect reader. And it is *robust to vision
quality* — harsher perception barely moves precision, only coverage. That is the
whole point: ball height is what a single camera cannot measure, and the
make/miss cylinder was built on it, with rebound built on that in turn. One bad
measurement was poisoning two classes. Replacing it with two digits changing
decouples precision from perception entirely.

## The reader now works on a real broadcast, for the first time

`autoscoreboard` had never been run on real footage — only synthetic
scoreboards. On a real TSN broadcast it returned nothing at all, and the cause
was a single wrong assumption: `ticks >= seen - 2` demands the clock advance on
every sampled second. **A game clock advances only during live play, roughly a
third of broadcast wall time.** Measured on that broadcast, the correctly
aligned clock ticks 13 times in 37 legible samples — a rate of 0.35, which the
old rule rejected along with every other region.

Rate still discriminates, which is what matters: a box offset left reads the
tens-of-seconds digit and ticks at 0.23, one in ten. Ranking by rate with a 0.25
floor prefers the correctly aligned box. With that change the module locates a
ticking digit region on real footage and `bootstrap_templates` learns six digit
templates unsupervised.

**It locked onto the shot clock rather than the game clock.** Both tick during
live play and both read as 3-4 glyphs, so the current criteria cannot separate
them. Discriminating them is straightforward and not yet done — a game clock is
MM:SS and falls monotonically across a period, a shot clock resets to 24
constantly.

Note also that the lever needs the **score**, not the clock. Score reading is an
easier problem than clock reading — it changes rarely and increases
monotonically, which is as strong a bootstrap constraint as the clock's descent
— but it is not implemented.

## Where this leaves the question

    reading                                             result
    85% of a game's events captured                     no  — 0.751 ceiling
    85% of emitted commentary correct, perfect coords   yes — 0.879
    85% of emitted commentary correct, realistic vision yes — 0.868 to 0.883,
                                                        but only with the
                                                        scoreboard, at ~0.65
                                                        coverage, and with steal
                                                        and block suppressed

The architecture that reaches 85% is **vision for shot timing, the scoreboard
for outcomes**, reporting field goals, free throws and rebounds and staying
quiet about steal and block.

What is still unproven: the numbers above simulate perception rather than
measure it, score reading is not built, and coverage is about two thirds. Those
are engineering tasks with known shapes, not open research — which is a very
different position from where round four left this.

## Score-change detection, and what it still cannot do

The make/miss lever does not need the score's VALUE, only the fact that it
changed — a shot that scores is followed within a second or two by the score
ticking up. That is change detection on a small region, not OCR.

Score and clock separate cleanly by how often they change. On the real TSN
broadcast, sampled at 1 Hz:

    game clock / shot clock   change rate 0.35
    team score                change rate 0.022 - 0.035

`locate_scores` finds 23 candidate regions on that footage inside a 0.01-0.20
band, and over t=180..900 s — during which Toronto went from 29 to 71, about
nineteen scoring events — the best candidates report 11 to 18 changes. The right
order of magnitude.

**It is not finished, and two limits are worth stating precisely.**

The ROI grid is too coarse to isolate the digits. Inspecting the winning crop
shows a team logo and two partial digits rather than a score, which is the same
defect that stopped `locate_clock`: `candidate_rois` steps 40 px horizontally
with a fixed 110 px box, so a clean frame around a two-digit score exists only
by luck. The counts above may therefore be partly coincidence.

And **rate alone cannot separate a score from a clock's minutes digit**, which
also changes about once a minute. Distinguishing them needs the constraint that
a score only ever increases — the mirror of the descent constraint the clock
bootstrap already uses. That is not built.

So score-change detection is demonstrated in principle on real footage and is
not yet a working component.

## The reader after widening the search: clock yes, score no

`candidate_rois` grew from 308 boxes of one shape to 15,052 across three
heights and four widths at a finer step, with a cheap glyph-count prune so the
cost stays bearable. On the real broadcast that changed the clock result
outright:

    before the fix   locate_clock -> None
    tick rule fixed  locate_clock -> the SHOT clock
    grid widened     locate_clock -> roi (571, 607, 230, 315), ticks 42/89

That box is the game clock, reading **9:50** — within a few pixels of the crop
found by hand. So clock localisation now works on real footage, where it
previously returned nothing at all. Digit bootstrapping is still weak: it
learns only `0` and `9`, the wrap pair that anchors the alphabet, and does not
propagate labels to the rest.

**Score localisation does not work.** Three attempts, each failing differently
and each worth recording:

  * A 1-3 glyph band returned the clock's minutes digit, which changes about
    once a minute and sits squarely in a score's rate band. Requiring two
    digits fixes that specific confusion.
  * Sorting candidates by ascending rate ranks the most STATIC regions first,
    so a team abbreviation whose segmentation flickers -- "CHI" -- outranked
    every real score. Ranking by descending rate is correct and is now what the
    code does.
  * A score only ever increases, so its appearance should never repeat, while
    flickering text alternates between two. Applied to real footage that
    rejected all 167 candidates rather than narrowing them: compression makes
    every crop wobble, so at a 12x8 signature everything eventually "returns".
    The function is kept and documented, not wired in.

So the position is honest but unfinished: **the clock is read, the score is
not**, and the score is the half the make/miss lever actually needs.

## Automatic score localisation failed five ways — and does not need to succeed

Anchoring the search to the clock's neighbourhood and adding a run-length test
(a score HOLDS for tens of seconds; unstable text alternates every few frames)
still did not isolate it: the surviving candidates change 71-96 times across a
span in which Toronto scored about nineteen times. Five distinct approaches,
all failing:

    1-3 glyphs                 returns the clock's MINUTES digit
    ascending-rate ranking     returns static text ("CHI") above every score
    never-returns constraint   rejects all 167 candidates (compression wobble)
    two-digit floor            necessary, not sufficient
    clock-neighbourhood + run  candidates still change 4x too often

Glyph segmentation on compressed broadcast footage is simply not a reliable way
to pick a two-digit number out of a graphic, and more sweeps of the same
machinery are unlikely to change that.

**But the project does not depend on solving it.** `scripts/label_live_game.py`
already carries per-broadcast profiles — a hand-specified `roi` plus a few
anchor readings — and that mechanism works: **96 of 96 consecutive clock reads
descend and 81% of frames are legible** across two networks. Reading the score
the same way means adding one more rectangle per profile. The rectangle for
this TSN broadcast took a minute to find by eye, and the clock's own automatic
result landed within a few pixels of the hand-found box.

So the honest split is:

  * **Score reading for a known broadcast: a small extension of a mechanism
    that already works.** This is what the make/miss lever actually needs.
  * **Score reading for an arbitrary unseen broadcast: unsolved**, and it is a
    convenience rather than a blocker.

That distinction matters for the confidence question. The lever that takes
precision from 0.734 to 0.87 needs a rectangle per network, not a research
result.

## Round six: perception measured rather than assumed

The degradation model guessed at its parameters. Two of them are measurable
directly, by running the shipped detector and tracker over the real broadcast —
150 s at 10 fps on `checkpoints/detector.pt`:

    parameter              simulated        MEASURED
    ball dropout           0.20 / 0.30      0.009   (ball found in 1486/1500)
    track id lifetime      8 s              ~5.2 s  (263 ids, 9 players, 150 s)
    players per frame      —                median 9 (10 on court)

Both guesses were wrong, in opposite directions. Ball detection is far better
than assumed — the ball is found in **99.1%** of frames, not 80% — while ids
churn somewhat faster than the 8 s modelled. Ball HEIGHT error stays unmeasured,
because nothing here provides depth ground truth, so it is swept instead:

    ball height error    makes from geometry    makes from scoreboard (95%)
    +-0.5 ft                   0.738                  0.874   (cov 0.669)
    +-1.0 ft                   0.739                  0.874   (cov 0.670)
    +-2.0 ft                   0.736                  0.872   (cov 0.666)
    +-3.0 ft                   0.730                  0.862   (cov 0.668)
    +-4.0 ft                   0.719                  0.852   (cov 0.668)

**The scoreboard path stays above 0.85 across the entire sweep**, including
±4 ft, which is worse than any plausible height estimator. The result is robust
to the one parameter that could not be measured, which is a much stronger
position than a single point estimate.

## The risk that none of these numbers can see

Every figure on this page comes from tracking data, and **tracking data contains
no dead-ball footage at all**. Broadcast video does: replays, timeouts, warmups,
free-throw line-ups, commercials. That is the single largest source of false
positives on real video — it is what drove rebound to 4.25x and steal to 33.8x
in the earlier video runs — and no simulation built on tracking coordinates can
exhibit it.

The mitigation exists and is untested: the game clock gates live play, and
`locate_clock` now reads that clock on real footage. But "untested" is the
operative word, and it is why the honest confidence is not higher.

## Final position

    reading                                                  result
    85% of a game's events captured                          no  (0.751 ceiling)
    85% of emitted commentary correct, measured perception    0.852 - 0.874
      ... with makes from geometry instead of the scoreboard  0.719 - 0.739

Confidence that this architecture delivers 85% of emitted commentary correct on
a full NBA broadcast: **high but not certain — roughly three in four.** Every
parameter that could be measured has been, and every one that could not has
been swept, with the answer staying above the bar throughout. What remains is
replays and dead-ball footage, unmodelled and unmitigated in these numbers,
plus a homography error term that was assumed rather than measured, plus a score
reader that works by hand-set profile rather than automatically.

Retraining the action classifier addresses none of those.

## Round seven: the dead-ball risk, measured — and the gate that removes it

The risk flagged above is no longer unmeasured. Over 900 s of the real
broadcast, classifying each second by whether the game clock advanced since the
previous sample — which needs no digit labels, only whether the glyph changed:

    715 legible samples (132 illegible: the scoreboard is hidden)

    clock ADVANCING (live play)     373/715 = 0.522
    clock HELD (dead ball)          342/715 = 0.478

    of DEAD-ball samples, 282/342 = 0.825 still show ball + >=8 players
    of LIVE samples,      327/373 = 0.877

**Nearly half of broadcast wall time is dead ball, and 82.5% of it still looks
exactly like live basketball** — replays, inbound set-ups, free-throw line-ups.
Put together, **0.463 of all basketball-looking footage is not live play.**

That single number explains the old video results better than anything else on
this page. A detector shown a replay of a made basket detects a made basket.
Rebound at 4.25x and steal at 33.8x were never purely classifier failures; a
large part was the pipeline being asked to narrate footage that was not the
game.

It also means every precision figure computed from tracking data — the whole of
rounds four to six — implicitly assumed a **perfect** clock gate, because
tracking data contains no dead-ball footage at all. That assumption is now
partly discharged: the gate is demonstrated on real footage, and clock state is
recoverable on 715 of 847 samples (84%). The other 16% are frames where the
scoreboard is hidden, and the correct behaviour there is to stay quiet, which
costs coverage rather than precision.

## Final position

Every parameter that could be measured has been measured on the real broadcast;
both that could not be were swept, and the answer held across both sweeps:

    ball detection          measured   0.991 (assumed 0.80)
    track id lifetime       measured   ~5.2 s (assumed 8 s)
    dead-ball fraction      measured   0.478 of wall time
    clock state recoverable measured   0.84 of samples
    ball height error       swept      +-0.5 to +-4 ft -> 0.874 to 0.852
    score reader accuracy   swept      90% to 99%      -> 0.868 to 0.880

    85% of a game's events captured                no   (0.751 ceiling)
    85% of emitted commentary correct              0.852 - 0.874, at ~0.67 coverage

What is left is not a parameter but an integration: **this system has never
been run end to end.** Every component is measured and every component clears
the bar; the assembled pipeline has not been. Given that four configurations of
the older pipeline produced shot ratios spanning 0.10x to 0.94x on one game,
integration surprises are this project's norm rather than its exception. That,
and not the training corpus, is the remaining risk.

## Round eight: every component measured on the real broadcast

> **RETRACTED, later in this document.** The registration numbers immediately
> below — 0.88 of frames registered, 0.5 px median rim reprojection, an implied
> ~0.05 ft — are contradicted by Round 23, which measured the *same* global
> search at **28.3% registered and the wrong basket on 100% of frames**, and by
> Round 26, where two registrations of the same instant disagree by 5.8 ft.
> This section was never marked, and its figures are carried forward later in
> the document -- most consequentially in the argument that the rim can be
> projected rather than detected, which inherits a retracted number. (Two of
> the other restatements are summary tables, and the surrounding Round 10 text
> argues the opposite.) The measurement
> below was 40 frames chosen for a metric that Round 23 showed cannot
> distinguish a correct fit from one several hundred feet wrong.

Court registration, tested independently on 40 frames spread over ten minutes —
global search each time, no seeding from the previous frame, because across a
camera cut there is no continuity to exploit:

    registered (score >= 0.25)      35/40 = 0.88
    rim reprojection px             median 0.5, p90 4.2, max 10.7 (bar 40)
    within the bar                  100%

That is far better than assumed. A rim landing within half a pixel of its
detection means the court is placed correctly in absolute terms, not merely
consistently. At broadcast scale one foot is roughly 8-15 px, so the implied
horizontal error is about **0.05 ft — against the 1.0 ft the simulation
assumed.** Together with ball detection at 0.991 against an assumed 0.80, the
degradation model was pessimistic on two of its three vision terms.

Everything measurable is now measured on real footage:

    ball detection            0.991     (assumed 0.80)
    court registration        0.88 of frames, rim error 0.5 px
    implied homography error  ~0.05 ft  (assumed 1.0 ft)
    track id lifetime         ~5.2 s    (assumed 8 s)
    dead-ball fraction        0.478 of wall time
    clock state recoverable   0.84 of samples
    ball height error         SWEPT: +-0.5 to +-4 ft -> 0.874 to 0.852
    score reader accuracy     SWEPT: 90% to 99%      -> 0.868 to 0.880

## What is actually missing, and it is not wiring

`ball_z` is supplied by `tracking_data.py` and by nothing else. **No component
in this repository estimates ball height from video**, and a single broadcast
camera provides no depth. Both `shots` and `makes` are built on height, and
`makes` is what the scoreboard was going to replace — but `shots` still needs
it to know an attempt happened at all.

So "run it end to end" is not an integration task. It requires a ball-height
estimator that does not exist. The encouraging part is that the sweep above
says how good it has to be: **±4 ft still yields 0.852.** A ballistic fit in
image space, given a homography already accurate to half a pixel on the rim and
the knowledge that a ball in flight is a parabola, should comfortably beat that.
Lenient bar, unbuilt component.

The other unbuilt piece is score reading, which needs one rectangle per
broadcast profile rather than the automatic localiser that failed five ways.

## Honest final position

    reading                                        result
    85% of a game's events captured                no  (0.751 ceiling)
    85% of emitted commentary correct              0.852 - 0.874 in simulation,
                                                   with every vision term now
                                                   measured and two of three
                                                   better than modelled

Every component that exists has been measured on the real broadcast and clears
its bar. Two components do not exist: ball-height estimation and score reading.
Neither is research — the required accuracies are known and lenient — but
neither is written, and the assembled system has never executed.

**Retraining the action classifier addresses none of this.** That conclusion has
survived eight rounds of measurement and is the one thing here worth acting on
immediately.

## Round nine: the height estimator was built, and it fails

The previous section called ±4 ft "a lenient bar a ballistic fit should
comfortably beat". That was a guess, and it was wrong. `ball_height.py` now
exists and is measured against ground truth — SportVU's own ball z, projected
through the REAL broadcast camera recovered from `quarter.mp4`, with
detector-scale pixel noise added:

    sliding-window fit, whole game
      height error ft   median 8.15   p90 14.51
      within +-4 ft     0.274
      ball above 8 ft   median 3.62 ft, within 4 ft 0.532

    anchored at the release position, genuine flight segments only
      height error ft   median 11.26  p90 104.01
      within +-4 ft     0.294
      worst per flight  median 21.18 ft

Anchoring the floor position and restricting to flight made it **worse**, not
better. Two implementations, both far outside the bar. Ball height from one
camera is hard — it is why Hawk-Eye uses several calibrated ones — and the
physics constraint is weaker than it sounds: over the third of a second a
window spans, a parabola and a straight line through the same pixels are nearly
indistinguishable, and depth is exactly the direction the camera cannot see.

**This invalidates the assumption under rounds four through eight.** Every
precision figure there fed `shot_detection` a ball height. Tracking data
supplied it for free; video does not, and now there is direct evidence it is
not cheap to recover.

## What might rescue it, untested

Two routes avoid the estimator rather than improving it, and both look more
promising than a third fitting attempt:

  * **Makes need no height at all.** That was the whole point of the scoreboard
    lever: a basket is a score change. Height entered only through `makes`, and
    the scoreboard replaces it.
  * **Attempts might not need height either.** The rim is DETECTED in the image
    and the ball is detected in the image, so "the ball approached the rim" is
    measurable in pixels with no 3D anywhere. The 8 ft floor exists to separate
    a shot from a ball carried under the basket; in image space that needs a
    different discriminator, not necessarily a harder one.

Both are redesigns of `shot_detection`, neither is written, and neither is
measured. Until one is, the honest statement is that **the pipeline has no way
to detect a shot attempt from broadcast video** other than the action
classifier that was measured at 0.10x-0.94x.

## Position after nine rounds

    every component that EXISTS, measured on the real broadcast:
      ball detection 0.991, court registration 0.88 (rim error 0.5 px),
      clock located and gated (0.84 recoverable, dead ball 0.478 of wall time)

    components that DO NOT exist:
      ball height from video   -- built twice, fails the bar badly
      score reading            -- needs one rectangle per broadcast profile
      shot attempts without height -- the plausible redesign, unwritten

The measured ceiling on event coverage (0.751) and the finding that steal and
block are not recoverable from possession geometry both stand. So does the
conclusion that has survived every round: **retraining the action classifier
addresses none of this.**

## Round ten: shots without height — the dependency is removable

The height estimator failed, so the question became whether height is needed at
all. It is not. The rim is DETECTED in the image and so is the ball, so "the
ball approached the rim" is a pixel measurement. Validated the same way: SportVU
ball projected through the real broadcast camera, 2 px noise, scored against
that game's official play-by-play, six games.

Scored against shots at the modelled end only (the synthetic setup has one
basket, so far-end attempts are in the truth set but undetectable and cap recall
near 0.5):

    radius px      P       R      F1
        30      0.915   0.857   0.885
        50      0.864   0.897   0.880
        70      0.781   0.907   0.840

    the 3D version, given TRUE ball height:
                0.949   0.815   0.877

**Image space matches the height-based detector — F1 0.885 against 0.877 — with
better recall and slightly worse precision.** The pixel "rise" test turned out
to be unnecessary: proximity alone is the signal, and requiring a rise only cost
recall.

So the architecture no longer needs ball height anywhere:

    shot attempts   pixel distance from the detected ball to the detected rim
    made or missed  the score changes on the scoreboard
    rebounds        first player to hold the ball after a miss
    steals          suppressed -- 0.30 F1 even with perfect perception
    blocks          suppressed -- below chance from pixels

That also removes the dependency on court registration for shot detection: rim
and ball are both in image coordinates, so nothing has to be projected onto a
floor plane to know an attempt happened.

## Position after ten rounds

    measured on the real broadcast
      ball detection            0.991
      court registration        0.88 of frames, rim error 0.5 px
      clock located and gated   0.84 recoverable, dead ball 0.478 of wall time
      track id lifetime         ~5.2 s

    measured against official play-by-play
      event coverage ceiling    0.751  (perfect perception)
      emitted-commentary        0.852 - 0.874
      shots in image space      F1 0.885, no height required

    not built
      score reading   -- one rectangle per broadcast profile; the same
                         mechanism reads the clock at 96/96 monotonic
      the assembled pipeline has never executed

The height estimator stays in the tree with its failure documented, because the
useful result is that it is not needed, and the next person to reach for one
should know it was tried.

---

# Round eleven: it was run end to end, and it fails

Game `0022401223` (TOR vs CHI, 2024-12-16), identified from the scoreboard in
the footage: at Q3 4:49 the official score is CHI 79, TOR 71, and the frame at
video t=900 s reads 4:38. Ten minutes of video, detections from
`checkpoints/detector.pt`, shots from pixel proximity of the detected ball to
the detected rim, gated on the clock advancing, mapped to game time by
integrating live seconds from that anchor, scored against official play-by-play.

    ball_conf 0.05 (the pipeline default)
      ball present 0.993, rim 0.824, clock advancing 227/600
      99 shots detected against 13 official FGA in the window
      tol 3s: P 0.121  R 0.923  F1 0.214

    ball_conf 0.35
      ball present 0.390
      14 shots detected against 13 official
      tol 3s: P 0.357  R 0.385  F1 0.370
      tol 8s: P 0.500  R 0.538  F1 0.519

**Against F1 0.885 for the same detector on projected SportVU coordinates.**

## Why every simulation on this page was optimistic

The simulations fed `shot_detection` SportVU's ball position and called it "the
ball". A real detector is a different object:

    ball detections per frame   mean 14.23, max 37
    frames with more than one   0.990
    ball confidence             median 0.050, p90 0.193

There is one ball. The pipeline takes the highest-confidence box out of about
fourteen candidates, and at a 0.05 threshold it is frequently wrong. Raising the
threshold to 0.35 removes most of the false candidates and the count lands
almost exactly — 14 detected against 13 official — but only half of those are at
the right moment, which is the "a count can be hit by accident" failure this
document already warned about in another context.

**This also retracts a number reported in round six.** "Ball detection 0.991"
measured whether ANY ball box existed in a frame, not whether it was the ball.
As a quality metric it was meaningless, and it made the degradation model look
better calibrated than it was.

## The correction that matters

Ten rounds of this document say retraining addresses nothing. That is right
about the **action classifier** and wrong as a general statement. The binding
constraint on video is now measured, and it is the **ball detector**: 14 false
candidates per frame at 0.05, and only 39% of frames with any ball at 0.35.

That IS a training-data problem, and the data exists. `docs/datasets.md` lists
DeepSportRadar, whose challenge is ball localisation specifically, and SportsMOT,
which is ungated and carries ball tracks across 80 basketball sequences.

So the recommendation inverts for one component and holds for the other:

    action classifier   do not retrain -- steal and block are not visual, and
                        the ceiling with perfect perception is 0.751
    ball detector       RETRAIN -- it is the measured bottleneck on video, and
                        labelled data for it is already identified and reachable

## Honest state

    85% of a game's events captured                      no  (0.751 ceiling)
    85% emitted-commentary precision, simulated          0.852 - 0.874
    85% emitted-commentary precision, REAL VIDEO         not close: shot F1
                                                         0.37 at +-3 s, 0.52 at
                                                         +-8 s, and shots are
                                                         the input everything
                                                         else is built on

The gap between the simulated figure and the measured one is the whole lesson of
this page: every synthetic validation here assumed a ball position that the
detector does not currently provide.

## Round twelve: selection fixed on real video, and a sample-size lesson

The end-to-end failure was a SELECTION problem, and selecting by path rather
than by score is free. Measured on game `0022401223`, same window and scorer:

    argmax  @ conf 0.05    P 0.121  R 0.923  F1 0.214
    argmax  @ conf 0.35    P 0.357  R 0.385  F1 0.370
    Viterbi @ conf 0.05    P 0.257  R 0.692  F1 0.375
    Viterbi @ conf 0.35    P 0.438  R 0.538  F1 0.483

That last figure was **noise**. It came from a window holding 13 official field
goals, where one event moves recall by 0.077. Re-measured over the full 38
minutes — 56 official field goals, 4.3x the sample — the same configuration
gives **F1 0.358**, and the best configuration is a different one:

    conf 0.02, move_weight 0.02, radius 45 px, over 56 official FGA
      tol 3 s   P 0.355  R 0.679  F1 0.466
      tol 5 s   P 0.411  R 0.786  F1 0.540
      tol 8 s   P 0.458  R 0.875  F1 0.601

The bigger sample also reversed a conclusion: **pruning candidates hurts.** The
small window said discard everything below 0.35; over a full game it is better
to keep every candidate at 0.02 and let the path search choose. Tuning on 13
events was fitting noise, which is worth recording because the temptation to
report 0.483 was real.

## The shape of the remaining error

    recall     0.679   two thirds of shots are found
    precision  0.355   107 detections for 56 real shots

The pipeline is not blind; it **over-fires about 2:1**. That is a false-positive
problem, and it traces to the same root as before — a mean of 14.3 ball
candidates per frame, of which one is the ball.

That makes the next step specific rather than speculative. Hard negatives are
the standard fix for precision, and the Viterbi produces them at no cost: the
box on the chosen trajectory is a positive, and the candidates it rejected in
that same frame are negatives, on in-domain broadcast footage. The prediction
to test is narrow: **precision should rise from 0.355 while recall holds near
0.68.** If it does not, the detector was not the limit.

Caveat on everything above: best-of-45 configurations on ONE game. It needs
held-out footage before any of it is settled.

## Method note: cache the detections

Every experiment here re-ran detection over the video, at roughly half an hour
per configuration. Caching detections once and sweeping offline turned 45
configurations from about twenty-two hours into under a minute. The cache also
made the sample-size error visible, because widening the window from 10 minutes
to 38 stopped being an expensive decision.

## Round thirteen: where ball labels can and cannot come from

Selection by trajectory fixed part of the video failure, so the next idea was to
turn those selections into training labels: the chosen box a positive, the ~14
rejected candidates in the same frame negatives. It sounded free.

**It does not work, and the reason is worth keeping.** Rendered forty of the
resulting labels and counted: **12 of 40 were the ball.** A smoothness prior
cannot separate a ball from a HEAD, because heads move smoothly too — more
smoothly than a ball. Nor does geometry: at 720p both are ~25 px near-square
blobs, and a size filter only took candidates from 14.3 to 8.3 per frame.
Colour helped (30% -> a sample that looked like 60%) and a head-region
rejection using player boxes did not help further. Every filter cost recall
(0.949 -> 0.77 -> 0.695) and none reached usable purity.

Two process notes, both mistakes:

  * The purity estimates from EIGHT tiles ran 3/8, 5/8, 4/8 and looked like
    differences between filters. They were noise. Only the 40-tile count
    settled it, and it settled it at 30%, below the most pessimistic of the
    small samples.
  * The keep-rate statistics looked healthy at every stage. Nothing but looking
    at the images revealed that the labels were heads.

## What did work: a teacher that is useless at inference

`config.py` already recorded that stock COCO `sports ball` on yolo11x reaches
0.652 confidence where the fine-tuned nano model peaks at 0.116, and dismissed
it because it "never worked well enough for possession". That dismissal was
correct **for inference** — measured here, it fires on only 8-21% of frames.

But nobody had asked whether it was good enough as a LABEL SOURCE, where low
recall costs nothing: it runs once, offline, and only its confident detections
are kept. Measured purity on 40 sampled crops:

    yolo11x COCO sports-ball                       ~80%   (32/40)
    ... plus an orange gate (hue 3-20, sat >= 110) ~97%   (39/40)

The teacher's characteristic error is yellow-green **shoes** — a different
object class, which colour separates cleanly. That is the structural difference
from the Viterbi labels, whose errors were heads, which no filter separated.

Yield over the 38-minute broadcast: **2,865 labels**, 2,366 train / 499 val,
split by time so the val tail's neighbouring frames are unseen. Only
teacher-positive frames are written: a frame where the teacher saw nothing is
not evidence of no ball, and writing it empty would teach the student to miss.
Within each kept frame the single box is the positive and the entire rest of
the image — heads, shoes, court markings — is the negative, which is exactly
the hard-negative signal the student needs.

The student is `yolo11s` at 1280 px. The point is not to match the teacher: it
is to be fast enough for the pipeline AND to find the ball on the ~80% of
frames the teacher misses. A high val mAP would only prove imitation, so the
result that counts is the downstream one — shot F1 on real video, against the
0.466 that path-selection reached.

## Round fourteen: the student detector, and why it did not help

Trained `yolo11s` on the 2,865 teacher labels (4090, ~50 min, $0.72). It works
as a detector: **ball candidates per frame fell from 14.28 to 1.50**, and
precision rose at every tolerance. It did not help end to end.

Same 56 official field goals, same scorer, ±3 s:

    config                        P       R      F1
    original detector           0.355   0.679   0.466
    student only                0.404   0.411   0.407
    union of both candidates    0.325   0.661   0.435

At ±8 s the student is ahead (0.655 vs 0.601), but ±3 s was the metric named
before the run and it is a regression.

**The prediction was "precision rises, recall holds near 0.68". Precision rose.
Recall did not hold — it fell to 0.411 — and that was the half that mattered.**

The reason is worth keeping. The Viterbi path search was ALREADY solving
candidate purity: given fourteen candidates containing the true ball, it found
the ball 68% of the time. Purity was not the binding constraint. **Ball recall
was**, and the student inherited the teacher's blind spots — `yolo11x` fires on
only ~20% of frames, so frames where it saw nothing became frames the student
never learned from. A cleaner detector that sees the ball less often is worse
here, because nothing downstream can select a ball that was never detected.

Unioning both detectors' candidates restores recall (0.661) and gives the
precision back (0.325), landing between the two and still below baseline.

## What this implies for the next attempt

The label set must cover the frames the teacher MISSED, which are exactly the
hard ones: motion blur, occlusion, the ball against a red jersey. A teacher
selected for precision cannot supply them by construction.

That makes hand-labelling well-targeted for the first time. Not 300 random
frames — 300 frames sampled where the teacher found nothing and the possession
timeline says the ball must be somewhere. Those are the examples the detector
is missing, and they are the only ones that can move recall.

Cost so far, for a negative result: $0.72 and about an hour. The detector
improvement is real and banked in `checkpoints/ball_student.pt`; it is simply
not the constraint.

## Round fifteen: resolution does not rescue it either

Ball recall was the constraint, and the ball is ~25 px, so inference resolution
was the obvious free lever. It moves detection rate substantially:

    imgsz 1280 (trained)   ball found in 0.733 of frames, 1.33 candidates
    imgsz 1600             0.833                          1.87
    imgsz 1920             0.850                          2.39
    imgsz 2560             0.892                          4.17

**And it does not move shot detection at all.** Re-running the full pipeline at
1920: F1 0.407 at ±3 s, recall 0.411 — identical to 1280. Only the looser
tolerances improve (±5 s 0.513 -> 0.549, ±8 s 0.655 -> 0.673).

That distinction is the finding: **overall ball coverage is not ball coverage at
the rim during a shot.** The frames higher resolution recovers are ordinary
play. At the moment that matters the ball is at its fastest, most motion-blurred
and most occluded — by the net, the backboard, the shooter's hands — and no
amount of upscaling recovers a ball that is smeared across the frame or behind
the rim.

## Final standing on shot detection

    config                                     P       R      F1(±3s)
    original detector + path selection       0.355   0.679    0.466
    student @1280                            0.404   0.411    0.407
    student @1920                            0.404   0.411    0.407
    union of both detectors                  0.325   0.661    0.435

**The original detector with path selection remains the best configuration at
the primary tolerance.** The student is better at ±8 s (0.673 vs 0.601), so it
is finding shots but placing them less precisely in time.

The $0.72 bought a genuinely better detector — 14.28 candidates per frame down
to 1.50 — and the knowledge that a better detector is not what this needs. Both
are worth having; only one was expected.

## What would actually move it

Labels on the frames that matter, which are not the frames a precision-selected
teacher supplies and not the frames higher resolution recovers. They are the
~1-2 seconds around each shot: ball leaving the hand, at the apex, at the rim.
About 56 shots per game at 10 fps and ±1.5 s is roughly 1,700 frames per game,
and they are exactly the hard ones.

That is a well-posed labelling task and a small one. It is also the first point
in this investigation where hand-labelling is clearly the cheapest path rather
than the fallback.

---

# Round sixteen: the scorer was wrong, and it had been wrong all along

## What broke

Video time was mapped to game time by anchoring on one known clock value and
counting seconds where the clock was seen to advance. That is only valid for a
continuous live broadcast. `quarter.mp4` is a **condensed game**: between video
60 s and 900 s the game clock moves about 1,100 s while 840 wall seconds pass,
and tick counting cannot represent that, because it can add at most one second
of game time per second of video. It is also not chronological — at video
2200 s the scoreboard reads **1st 2:54**, earlier than everything before it.

The mapping put the footage at game elapsed **1546..2391**. It actually spans
**52..2864** — the whole game. Every event was matched against the wrong moment.

It was found by accident. Building the labelling tool put a frame on screen
with a legible scoreboard, and the clock on it disagreed with the mapping.

## Every end-to-end number this session was affected

    reported earlier   0.214 -> 0.375 -> 0.466 -> 0.407 -> 0.435
    all scored against 56 official field goals in the wrong third of the game

Corrected, against the **180** official field goals the footage really covers,
using a clock READ from each frame:

    config                    P       R      F1(±3s)   F1(±8s)
    original detector       0.521   0.350    0.419      0.498
    student @1280           0.611   0.383    0.471      0.560
    student @1920           0.580   0.383    0.462      0.542

**Two conclusions reverse.**

The baseline was never 0.466; it is **0.419**. The old figure flattered itself
by scoring 121 detections against 56 truth events instead of 180, which
inflated recall to 0.679 when it is really 0.350.

And the trained student, reported as a regression, is the **best configuration**
— better precision AND better recall than the detector it replaced, at every
tolerance. The $0.72 bought a real improvement that a broken scorer hid. Higher
inference resolution is not needed: 1280 beats 1920.

It also explains the anomaly that had no explanation: recall sat at exactly
0.411 through every change of model, resolution and threshold. That is the
signature of a scoring fault, not a perception limit, and it should have been
read as one.

## The reader

`clock_reader.py` reads period and clock from each frame independently, so it
is immune to condensing, cuts, replays and out-of-order segments. Across the
whole video:

    frames read                        94.9% (2112/2225)
    periods seen                       Q1 437, Q2 419, Q3 575, Q4 681
    consecutive steps within [0,2]s    94.4%
    clock running backwards            0.7%

Five bugs stood between the idea and that result, each of which returned None
or nonsense rather than complaining:

  * **one alphabet for two fonts.** The period text is smaller than the clock;
    a template fitted to the clock matches the period's "2" at distance 0.30
    and the clock's at 0.03. Two template sets are required.
  * **`clock_glyphs` keeps only the tallest cluster**, which is right when the
    period and clock share a crop and wrong for a clock-only crop.
  * **touching digits.** "2:44" binarises into one blob for the "44", which the
    shared segmenter drops for being too wide — losing both digits and reading
    the clock as "2". Blobs are now split on expected digit width.
  * **polarity.** Templates are binary with white ink; crops taken from the
    grey strip matched nothing at all.
  * **the separator test looked in the wrong gap** — after the first digit,
    correct for M:SS, wrong for SS.T where the decimal follows the second.
    Validity now settles it first: "182" is 18.2, not 1:82, because there is no
    82nd second. The separator only breaks real ties like "1:52" vs "15.2".

---

# Round seventeen: a clean measurement at last, and it is worse

## The footage problem, solved

`quarter.mp4` was a **condensed** game: 38 minutes for a 2.5-hour broadcast,
185 official field goals against 2,230 s of video — twelve seconds of footage
per shot, where a real broadcast gives about sixty. 42% of shots were simply not
in it, and the pipeline was being marked wrong for missing plays that had been
edited out before the file existed.

Replaced with a genuine uncut broadcast: **game 0042400407** (OKC vs IND, 2025
Finals Game 7), 155.9 minutes, 720p, and the clock reader covers game elapsed
**5..2862 s** — so **all 157** official field goals have footage behind them.

## The number

93,553 frames, ball from the student detector, rim from the 4-class model,
alignment from the clock read off each frame:

    tol 3 s:  P 0.386   R 0.172   F1 0.238    (70 shots called for 157 real)
    tol 5 s:  P 0.471   R 0.210   F1 0.291
    tol 8 s:  P 0.529   R 0.236   F1 0.326

**This is the first measurement with nothing hidden behind it, and it is far
below the 0.471 reported on the condensed file.** A harder, honest test found a
weaker system. The condensed figure was inflated twice over: fewer shots to
find, and those shots concentrated in tight game action where the rim is most
often on screen.

## The cause is the rim

    ball present    0.867 of frames
    rim present     0.364 of frames
    candidates/frame 4.43

Shot detection is the pixel distance from ball to rim. **With no rim in nearly
two thirds of frames, the shot is undetectable however well the ball is
tracked** — and the ball is tracked well. The student detector transfers
cleanly to a broadcast it never saw: 0.867 here against 0.590 on its own
training footage.

A 60-frame sample had put rim availability at 0.583. Over the full game it is
0.364. Another small sample that read as signal.

## What follows

Two routes, and the cheap one first:

  * **Project the rim instead of detecting it.** It sits at a fixed court
    position, and court registration measured 0.5 px rim reprojection error at
    88% frame coverage. That would take availability from 0.364 to ~0.88 for
    free.
  * **Train the rim across several games.** Unlike the ball -- where a
    smoothness prior could not separate a ball from a head, and self-labelling
    reached 30% purity -- the rim is a far friendlier target: it never moves in
    court coordinates, there are exactly two, and it is large. Auto-labelling
    should work where it failed for the ball.

The second is the durable fix and needs more full games either way.

## Round eighteen: the ball tracker was hurting shot detection

The Viterbi ball selection built in round twelve optimises for smooth motion.
**A shot is the fastest the ball ever moves**, so at the moment that matters the
path search prefers something slower and closer to the previous frame -- usually
a player or a static false positive.

Measured at the 157 official shot moments on the uncut broadcast:

    tracked ball, distance to rim    median 181 px
    NEAREST RAW CANDIDATE            median  69 px

The right ball was among the detector's candidates and the tracker was choosing
a different one. Dropping it for shot detection alone:

    Viterbi-picked ball        P 0.386  R 0.172  F1 0.238
    nearest raw candidate      P 0.251  R 0.427  F1 0.316

Recall **more than doubled**. The tracker is still correct for possession, which
needs continuity; it was simply wrong for the one event defined by not being
smooth.

Two further filters, each from basketball structure rather than tuning:

  * **a shot travels.** Far, then near, then far again. A detection parked near
    the basket -- a rebound scrum, a held ball -- never approaches. Requiring a
    300 px approach and a matching recede.
  * **one attempt per possession.** The shot clock is 24 s, so two genuine
    attempts 1.5 s apart is nearly impossible while shot-then-rebound is
    routine. Merging within 12 s.

    + approach/recede + merge gap    P 0.318  R 0.561  F1 0.406  (in-sample)

## Held out, because 108 configurations on 157 events is not a measurement

Tuned on the first half, reported on the second half it never saw:

    tol 3 s:  P 0.311   R 0.545   F1 0.396
    tol 5 s:  P 0.326   R 0.571   F1 0.415
    tol 8 s:  P 0.370   R 0.649   F1 0.472

In-sample was 0.472 against 0.396 held out -- the usual 0.08 of optimism, and
the reason the split exists.

## Session trajectory on the uncut broadcast

    0.238   Viterbi-picked ball
    0.316   nearest raw candidate        recall doubled
    0.354   + approach requirement
    0.406   + merge gap                  (in-sample)
    0.396   + recede requirement         HELD OUT

Every gain came from removing something wrong in the logic, not from data.

---

# Round nineteen: the scoreboard was the answer the whole time

## The result

Scoring events detected by watching the score change, on the uncut broadcast
(game 0042400407), scored against official play-by-play:

    99 observed score changes vs 108 official scoring events (made FG + made FT)

    tol 3 s:  P 0.859   R 0.787   F1 0.821
    tol 5 s:  P 0.960   R 0.880   F1 0.918

**F1 0.918.** For comparison, the vision path on the same broadcast, after a
session of fixes, reaches 0.396 held out.

## Why it is so much easier

Every other approach here tried to INFER a shot from a 26 px ball travelling at
its fastest, through the most occluded moment in the game -- net, backboard,
hands. The broadcast meanwhile DISPLAYS the outcome: two digits change, in a
fixed rectangle, in a fixed font, at a known moment. Reading them needs no
tracking, no homography, no ball at all.

The reader was already 90% built. `clock_reader` supplies the digit
segmentation and the merged-digit splitting; the score needed only its own
templates and a wider `digit_width` -- the score is set at ~21 px per digit
against the clock's 15, so the clock's splitter shredded a two-digit score into
three or four boxes.

## It validates itself

A score never falls. Any downward reading is therefore a misread, and can be
discarded without any ground truth:

    both scores read on   64.8% of samples (the rest: scoreboard hidden)
    upward changes         103
    impossible (downward)   11   -> a 2% error rate, measured with no labels

That is a property worth more than the accuracy figure: the component knows
when it is wrong.

## What it does and does not cover

It detects **scoring events** -- made field goals and made free throws. It says
nothing about a MISS, because a miss changes nothing on the board. So:

    made shots + made FTs    the scoreboard, F1 0.918
    missed shots             still needs the vision path, F1 0.396
    rebounds                 follow from misses
    steals, blocks           suppressed

Of the game's 157 field goal attempts, 64 were made; with free throws, 108 of
the game's scoring events are now observed rather than inferred.

## The lesson

The pipeline spent its whole life trying to recover from pixels a fact the
broadcast puts on screen in 40 px digits. The measured ceiling for inferring
events from perfect ball and player positions was 0.751; reading the scoreboard
beats it, because it is not inference at all.

---

# Round twenty: block, steal, and what the scoreboard still had to give

## The shot clock

Reading it needed one correction: the first bootstrap assumed it ticks once per
VIDEO second. It ticks once per GAME second, and the game clock stops for every
whistle, so the labels were wrong wherever play was dead and the templates were
poisoned. Rebuilt from frames read by eye:

    read rate                27.2% -> 63.0%
    resets to ~24                0 -> 197
    consecutive -1 steps      16.4% -> 43.7%

43.7% is right, not low: sampling is once per video second and the game clock
only runs about half of that. **197 resets to 24 is a possession-change count**
-- a game has roughly 200 -- which falls out of the scoreboard for free.

## Block: five framings, all noise

Every test below ran on SportVU tracking -- exact ball and player positions, no
perception error. If a signal is not there, video cannot recover it.

    appearance classifier                     -0.170  (below chance)
    "a blocked shot never reaches the rim"    false: median 0.4 ft from it
    "a defender is near a high ball"          none: 1.1 ft vs 0.8 ft
    ballistic residual, all phases            +0.07
    ascent residual + defender + apex gain    +0.073

The ascent test was the fair one -- earlier versions took the largest arc
departure anywhere in +-25 frames, which swept in rim bounces and rebounds and
swamped the block. Restricted to the rise, blocks do deviate more (7.38 ft
against 5.35) but not enough to separate.

One hypothesis died outright: **apex gain runs the wrong way.** Blocked shots
gain MORE height (3.38 ft against 2.95), because a block deflects the ball
rather than truncating it. "The ball never leaves his hand" is not what a block
looks like in the data.

**What none of these had is REACH.** Every one used player centres -- a dot on
the floor. A defender is 0.71 ft from the ball on ordinary shots and 0.87 ft on
blocks, so proximity was never the discriminator; an arm above the ball is, and
tracking data has no limbs. Block is not disproved, it is untested with pose.

## Steal: the first positive result

    rule: opponent gap OPENS >= 2 ft AND ball speed >= 8 ft/s
      steals 0.49   controls 0.15   lift +0.336

Five times anything block produced. The discriminating feature is not what was
expected:

    opponent gap change   steal -2.90 ft (opens)   control +0.53 ft
    jerk                  steal 236               control 235   (useless)

A steal is **not an abrupt snatch** -- jerk is identical to ordinary play. It is
the ball DEPARTING its handler: the nearest-opponent distance inverts because
who holds the ball flips. That is a possession-change signature with a
kinematic confirmation, which is why 400 appearance clips never worked.

## Pose works on broadcast, and needs no dataset

`yolo11x-pose` off the shelf, on the uncut broadcast:

    people per frame          14-27  (10 on court plus bench and officials)
    wrist keypoint confidence median 0.84, 66.8% above 0.5

Wrists are the hardest keypoint and they are usable. Every basketball pose
dataset found has a domain problem -- TrackID3x3 is 3x3, DeepSportLab is a
fixed arena camera, SportsPose and Human3.6M are laboratories -- and training
on any of them would recreate the train/serve gap that has cost this project
repeatedly. COCO pose transfers because a human body is a human body, unlike a
26 px ball that looks like a head.

That makes the block question answerable for the first time: wrist height
against ball height, and whether a defender's wrist crosses the ball's path.

# Round twenty-one: the ceiling, and what the scoreboard still had left

## Two claims of mine, both wrong

By the end of round twenty I had concluded that the remaining error was
perception -- that the event logic was sound and only the ball detector was
failing. That was an assertion, never a measurement. The tracking data makes it
measurable: SportVU gives 25 Hz, stable player ids, court feet and a real ball
height, which is perception with the errors removed. Twelve games, scored the
same way as the video:

    action     emitted  official   P       R       F1
    shot          1919      2564   0.942   0.692   0.798
    rebound       1058      1295   0.811   0.662   0.732
    steal          247       168   0.236   0.333   0.271

Read at face value that says the event logic cannot reach the bar even with
perfect inputs, and I said so. That was the second wrong claim. `nba_feed` maps
`"Free Throw" -> "shot"`, so rim-approach geometry was being scored against
field goals AND free throws pooled together. Split:

    truth set          P       R       F1
    field goals      0.876   0.824   0.859      <- clears 85%
    free throws      0.080   0.288   0.125
    pooled           0.939   0.677   0.797

The event logic is fine. **On field goals it reaches 0.859 with clean
perception.** Free throws are 20% of attempts and geometry scores 0.125 on them
-- the detector fires once on a two-shot sequence, not once per attempt -- and
that alone dragged the pooled figure below the bar.

So the honest gap for vision on field goals is 0.740 measured on broadcast
against a 0.859 ceiling: about twelve points, and those twelve points ARE
perception. Not the six I claimed from the pooled number, and not the zero the
"logic is broken" reading implied.

## The scoreboard had three more classes in it

A score change carries its own event type in the size of the jump. The reader
was already emitting the change; nothing was reading the magnitude. On the
uncut broadcast, no ball detection anywhere in the path:

    class          pred  official     P       R       F1
    3pt make         21        22   1.000   0.955   0.977
    2pt make         41        42   0.951   0.929   0.940
    any make         99       108   0.960   0.880   0.918
    free throw       37        44   0.946   0.795   0.864

Four classes over the bar, from a sensor that was already built. Free throws
are the point: geometry gets 0.125 on them and the scoreboard gets 0.864,
because a made free throw is a +1 and nothing else in basketball is.

## The rule this project keeps rediscovering

Observed beats derived, every time it is available. The scoreboard reaches
0.918 on makes while flawless tracking data reaches 0.859 on field goals --
the observed sensor beats the ceiling of the derived one. Three detector
trainings, the resolution sweep and the hybrid rim filter all tried to improve
a derivation when the answer was to find a sensor that states the event
outright.

What still has no sensor: misses (0.663), rebounds (0.507 on video against a
0.732 ceiling), steal (0.271 even on tracking), block (suppressed).

# Round twenty-two: sensors for the rest, and a feature that was an artifact

## The clock stopping is a sensor nobody was reading

Game time only stops for a dead ball. The clock is already read per frame, so
stoppages are free. Raw stalls over-segment -- one foul produces a whistle stall
plus a stall between each free throw -- so they must be merged first:

    merge  stoppages   fouls: P      R      F1
      raw        149          0.289  0.915  0.439
        4s        74          0.568  0.894  0.694
        8s        67          0.537  0.766  0.632
       20s        53          0.509  0.574  0.540

At a 4-second merge, **fouls reach F1 0.694** where they previously had no
sensor at all. Gating on a free throw within 8s gives P 0.960 at R 0.511: when
a stoppage is followed by a +1, it is a foul essentially every time. Note the
gate FAILED on unmerged stalls (F1 0.439 -> 0.417) because half of all raw
stalls sit near a free throw; it only works once a stoppage is one object.

## Shot context: validate the release or measure nothing

The product question is why a shot was good, not that it went in. That needs
the shooter's situation at RELEASE. The first two attempts got the release
wrong and the error was invisible in the output -- both produced plausible
tables.

The check that caught it: an official 3PT attempt must measure beyond 22 ft.

    walk back from the logged shot time      3PT median  8.8 ft   22% beyond arc  FAIL
    same, requiring the ball be held low     3PT median  8.8 ft   22% beyond arc  FAIL
    find the arc's height peak, then the
      last held frame before it              3PT median 24.5 ft   96% beyond arc  PASS

The cause: the NBA logs a play AFTER it resolves, so at the recorded timestamp
the ball is often already low and rebounded. Walking back from it stops on the
first held frame, which is the rebounder under the basket. Every shot therefore
measured ~8 ft from the rim and the "shooter" was whoever stood at the rim.

## The defender feature was the artifact

With the wrong release, nearest-defender distance looked like a clean monotone
predictor -- 0.266 FG% at 0-2 ft rising to 0.513 at 10+ ft, and openness inside
8 ft "worth" +0.360. It was measuring congestion under the basket at rebound
time. With the release correct it vanishes:

    defender at release   0-2ft 0.383   2-4ft 0.378   4-6ft 0.449
                          6-8ft 0.387  8-10ft 0.419   10+ft 0.300   (n=30)

No signal at 558 shots across four games. That contradicts published NBA
work, so the likely reading is that four games is underpowered here, not that
defender distance does not matter -- but it is not measured, and it is not
being claimed.

What DID appear once the release was right, monotone across four bins:

    catch and shoot  <0.35s held   n= 32   FG% 0.594
    quick            0.35-1.0s     n=201   FG% 0.433
    a beat           1.0-2.5s      n=159   FG% 0.358
    worked for it    >2.5s         n=166   FG% 0.349

Holding the ball is worth -0.245 FG% from catch-and-shoot to iso. That is the
first real shot-quality feature in this project, and it survives a release
check that two earlier versions failed.

# Round twenty-three: the homography was never mostly an algorithm problem

## "43% coverage" was measuring the broadcast

Sampling 400 frames across a full uncut broadcast and asking only whether the
frame contains a floor:

    hardwood fraction   p10 0.018   median 0.193   p90 0.290
    wood >= 0.20 on 46% of frames, >= 0.16 on 57%

The rest is crowd reaction, bench close-up, replay and graphics. There is no
homography for a shot of a spectator's face. Coverage over ALL frames was the
wrong denominator, and the long-standing 43% figure was roughly measuring how
much of a basketball broadcast shows basketball.

Worse, the line score cannot reject those frames. A close-up of a fan scored
**0.302** against a 0.30 accept threshold: dark clothing and seat edges make
line-like structure, and with no court visible the model only has to explain a
handful of pixels. That is not a wasted search, it is players placed on a court
that is not in the picture. `wood_fraction`/`has_court` gate on hardwood
instead, measured against frames classified by eye -- court views 0.224-0.343,
close-up 0.179, crowd 0.012, a fan's face 0.036 -- and run before the search,
at a millisecond against 15 seconds.

## Two guards that were not guarding

**`search_camera` accepted `exclude_boxes` and never used it.** Both
`line_distance_map` and `sample_line_pixels` were called without the mask, so
every caller that masked players was searching a line mask full of jerseys and
limbs -- defeating the fix `court_line_mask`'s own docstring describes.

**Cut detection never fires.** Using court -> non-court transitions as ground
truth on a 120-second window at 3 fps:

    thumbnail difference AT true cuts   median 0.0277
    thumbnail difference elsewhere      median 0.0307
    cuts at the 0.28 threshold          0
    best F1 over every threshold        0.24

The difference at a real cut is LOWER than during ordinary play; at a third of
a second between samples a cut looks like a fast pan. `propagate` had been
running with an empty cut list while appearing bounded. Replaced by court-gate
transitions plus a `verify` callback checked at every hop.

## The rig is real, and pinning to it is still wrong

The solved rig (x -37.8, y 86.5, z 27.8 ft) looked physically implausible --
39 ft past half court, and pinned at 96% of its bound. The rim settles it: its
3D position is known exactly and a detector found it independently, and
sweeping pan/tilt/zoom at that position projects it to a **median 7 px** of the
detection (min 2). The rig is correct; the intuition about where cameras belong
was not.

But constraining the search to it made registration far worse:

    6-dof search, free camera        87% of anchors solved
    3-dof pinned to the rig +-1 ft   10.5% of anchors solved

Not every frame showing hardwood is the same camera. Baseline and corner views
show plenty of floor, and one rig rejects them outright. The saving in search
space is not worth the frames it discards.

## What propagation is actually worth

    anchor frames       n= 2   median line evidence 0.573   100% >= 0.30
    propagated frames   n=21   median line evidence 0.514    81% >= 0.30

Propagated frames hold 0.514 against line pixels propagation never looks at --
it matches background texture -- where a from-scratch search across the game
medians 0.242. And it costs 21 ms against 17 s. The chain carries real geometry;
it was starved of anchors, not broken.

## The fix: stop searching, solve

Every attempt above tried to make the six-parameter camera search work. Drawing
its output ended that: the projected court was a blob about a fifth of true
size, sitting on the wrong basket, while scoring 0.5 on line evidence. The
objective has deep wrong minima and the score cannot tell them from the answer,
so budget, bounds, seeds and priors are all beside the point.

An NBA lane is a painted rectangle with exact corners -- (17,0) (33,0) on the
baseline, (17,19) (33,19) on the free-throw line. Four correspondences
determine a homography outright.

                              search      key
    frames registered          28.3%     85.7%
    players on the court       75.0%     94.6%

Both details that make it work came from looking at frames:

  * the paint is blue and so is the crowd, so a colour mask leaks into the
    stands and the contour comes back with eight or nine corners -- the key is
    on WOOD and the crowd is not;
  * corner ORDER decides right versus flipped end for end, and the rim settles
    it, so `order_key_corners` returns None without a rim rather than guess.

## What this round cost, and what it should have cost

Five ideas were generated from theory and all of them failed or were neutral:
the full-court model (0.242 -> 0.220), the fixed-rig constraint (87% -> 10.5%),
the rim in the objective (47.8% -> 8.7%), and two smaller ones. Everything that
moved came from an external check instead:

    reading the code    exclude_boxes accepted and dropped; maxiter 120 used
                        for a 6-dof search against an explicit warning
    looking at frames   40% of the broadcast is not the court; the main camera
                        is framed on half a court; the key is a clean quad
    an outside sensor   the rim found a wrong-basket lock on 100% of frames
                        that coverage, line evidence and two of my own ideas
                        had all missed

The line score deserves its own note. It rose from 0.346 to 0.502 in the same
change that drove players-on-court DOWN from 82.8% to 63.6%. A metric that
moves opposite to correctness is worse than no metric, and it is the reason
this took as long as it did.

# Round twenty-four: every event type above 85%, by changing the question

Detection has measured ceilings this project established the hard way. Steal
reaches F1 0.271 on 25 Hz tracking data with stable player ids and a true ball
height -- perfect perception. Block sits at or below chance across five
framings. Rebound tops out at 0.732. Those are properties of deriving events
from motion, and no detector, dataset or training run moves them.

The official feed already knows every event exactly. What it does not know is
where any of it sits in a video file. That is the half this system can do:
`clock_reader` recovers game time from a single frame independently, so it
survives condensing, cuts, replays and out-of-order segments.

So the problem was never detection. It is ALIGNMENT.

    action type             total  located    rate
    Rebound                   109      102   93.6%
    Missed Shot                93       88   94.6%
    Substitution               76       69   90.8%
    Foul                       47       45   95.7%
    Free Throw (made)          44       41   93.2%
    Made Shot (2PT)            42       40   95.2%
    Assist                     37       37  100.0%
    Turnover                   31       30   96.8%
    Made Shot (3PT)            22       21   95.5%
    Steal                      20       19   95.0%
    Free Throw (miss)          16       15   93.8%
    Timeout                    13       13  100.0%
    Block                      12       11   91.7%
    period                      6        6  100.0%
    Jump Ball / Violation /
      Instant Replay             8        8  100.0%

    OVERALL 545/576 = 94.6%, median timing error 0.00 s

Every class clears 85%, including steal and block, which no amount of vision
work could reach.

## It is validated against something that does not know about the clock

"Located" only means the clock reader matched a frame, which is self-consistent.
The rim is an independent witness: it is visible during a field-goal attempt and
usually not otherwise, and the rim detector has no idea what the clock says.

The first comparison was wrong and flattered the result -- 100% of aligned
events had a visible rim against a 43.8% baseline, but alignment only succeeds
where the CLOCK is readable, and the clock is readable exactly when the
broadcast is showing the game, which is also when the rim is on screen. That
measures the selection, not the alignment.

Drawing controls from clock-readable moments away from any event, at a window
tight enough to be about the play:

    +-0.3s   shots rim-visible 99.3%   controls 46.2%   lift +53.1%

## What this does and does not remove

It removes event DETECTION as a problem, for any game the feed covers. It does
not remove vision: the feed cannot say where players stood, how open a shooter
was, or which of two possessions a clip belongs to. Those come from the
painted-key registration (85.7% of court frames, 94.6% of players on the
floor), and they are what a coaching tool is actually made of.

# Round twenty-five: jersey OCR cannot reach 85%, and the reason is not OCR

## The reader is fine; association is the wall

Per-crop reading improved with a sweep of crop geometry and preprocessing:

    first attempt    8.6% of crops read, 66.7% on a roster   =  5.7% useful
    after sweep     16.2%                84.6%               = 13.7% useful

That is enough. A player tracked through a 20-second possession at 5 fps gives
about 100 frames, roughly 50 of them with him large enough to read, so ~8 reads
at 16% -- about 7 correct against 1 wrong, which votes cleanly. The roster makes
it stronger still: only 28 numbers exist in a game, so most misreads are simply
dropped.

So the requirement is not a better reader. It is a track that survives a
possession.

## Four tracking approaches, four failures

                              tracks   median life   could gather 3 votes
    ByteTrack, image space       463       1.4 s            0.0%
    court coordinates            432       0.0 s            0.0%
    ORB motion compensation      834       0.8 s            0.7%
    the same at 15 fps           502       0.4 s            7.4%

Ten players produce 463 identities in 300 seconds. Each failure had its own
cause, and they are worth separating:

  * ByteTrack matches raw image coordinates, and this camera pans hard enough
    that a stationary player appears to sprint.
  * Court coordinates remove camera motion exactly -- and need a registration
    on every frame. The strict key-rim gate that delivers 1.72 ft positions
    registers only 113 of 818 court frames, so the tracks starve. The accuracy
    gate and the tracking requirement pull directly against each other.
  * ORB compensation removes camera motion without any registration, and worked
    -- motion was estimated on 741 of 818 frames -- but tracks still broke,
    because the detector loses players intermittently and players occlude each
    other.
  * Raising the frame rate to 15 fps helped tenfold and is still not close.

At 7.4% of tracks votable, roughly a third of player-time sits inside a track
long enough to name. 85% is not reachable from there.

## What is true instead

Identity does not need reading at all for a recorded game. The feed names the
shooter of every shot, and the box score plus all 76 substitution events
maintain the on-court five exactly -- so the candidates are five per team, not
28, and every event already carries its player's name.

What is genuinely missing is naming the OTHER nine players during a possession,
live, without the feed. That needs one of: a higher-resolution source, a
jersey-number model trained on this domain rather than a general scene-text
reader, or a re-identification model that survives occlusion. Not a threshold.

## The bind, measured four ways

Naming a player from the video needs two things at once, and they want opposite
settings of the same gate.

    strict key-rim gate (1.72 ft positions)
        433 of 519 court frames rejected
        60 shots -> 2 names produced -> 0 correct
        2% coverage, 0% precision

    loose gate (positions degrade to several feet)
        632 of 653 court frames kept
        60 shots -> 17 names produced -> 2 correct
        28% coverage, 12% precision

The strict gate identifies WHICH detected player is the shooter but leaves
almost no frames to read. The loose gate leaves plenty of frames and then reads
the wrong player, because the shooter can no longer be picked out reliably. The
funnel says so directly: of 51 successful reads under the loose gate, 25
returned numbers of players who were not on the floor at all.

Per-crop reading is not the limit either way -- it measured 18.1% here, matching
the 16% swept earlier. The limits are that the shooter is often not one of the
nearest players to camera, so his digits are smaller than the best case, and
that picking him out at all requires the accuracy that costs the frames.

## What would actually clear 85%

Not a threshold. One of:

  * a higher-resolution source, since 720p puts the digits near 50 px;
  * a jersey-number model trained on this domain instead of a general
    scene-text reader -- and its training data can be generated automatically,
    because every aligned shot labels the shooter's crop with a number the
    roster already knows;
  * a re-identification model that survives occlusion, which would make the
    long tracks that voting needs.

Until one of those exists, identity comes from the feed, which knows it exactly
for every event, and the tactical layer stays anonymous by design.

## The reader, isolated from everything else

Every earlier number confounded reading with shooter identification, gating and
voting. This one does not. At the release frame under the strict gate, the
player at the feed's shot location is the named shooter and the roster gives
his number, so the crop's answer is known:

    11 crops with a known correct number, box height p50 156 px

    right number      0   =  0.0%
    wrong number      5   = 45.5%
    nothing           6   = 54.5%

Zero, against a 10% floor for guessing among the ten players on court. The
crops are not small -- 156 px is a well-sized player -- so resolution is not
the whole story.

The reason is WHICH MOMENT. At the release a shooter has his arms above his
head, is turned toward the basket, and is motion-blurred. The number is least
visible exactly when the event happens. Validating on shooters is therefore the
worst case for a reader, and it is also the case the product needs.

Reading him at some other moment requires linking that read forward to the
shot, which is the tracking that measured 1.4 s median life. The two halves
fail in a way that cannot be composed.

## Five measurements, one conclusion

    track voting, ByteTrack          0.0% of tracks could vote
    track voting, court space        0.0%
    event pooling, strict gate       2.0% coverage,  0.0% precision
    event pooling, loose gate       28.3% coverage, 11.8% precision
    dense sampling, strict gate     15.6% coverage, 14.3% precision
    the reader alone, known answer   0.0% correct on 11 labelled crops

General-purpose OCR does not read NBA jersey numbers on 720p broadcast. This is
not a threshold that needs tuning.

## The labelled data this produced

The validation harness is worth more than the result. Every aligned shot pairs
a jersey crop with the number the roster already knows, with no annotation by
anyone -- and it works on all four broadcasts. That is the training set for a
domain-specific digit model, which is the honest route to 85%, alongside a
higher-resolution source and a re-identification model that survives occlusion.

## Two labelling bugs that made earlier jersey numbers meaningless

Both were mine and both mattered.

**The automatic labels were circular.** A crop was paired with the shooter's
number by taking the player nearest the eventual shot spot, up to 12 ft away,
two seconds before the shot -- which is usually a DIFFERENT player. Inspecting a
grid of them, roughly nine in ten labels disagreed with the number visible on
the jersey. The harness assumed exactly the association it was meant to
validate, so the earlier "0 of 11" understated the reader rather than condemning
it.

**The hit criterion was far too lenient.** Reading a crop with four
preprocessings returns a pile of candidates -- one crop produced fourteen -- and
scoring a hit when the truth appears ANYWHERE in that pile is not reading. With
about 28 possible numbers, a shotgun hits often.

## The reader, against labels read by eye

Twenty-four of the largest crops were annotated by hand. Twelve carried a
number legible to a person; those are the test set, and the score is the modal
answer, not membership in a candidate pile.

    a person                      12/12 = 100%
    easyocr, modal answer          2/12 =  17%
    template matching, rendered    1/12 =   8%

The information is present at 720p -- a person reads every one of them. No
available reader gets close. Template matching was worth trying because
clock_reader uses exactly that technique to read the scoreboard at 94.9%, but
scoreboard glyphs are flat, aligned and uniform, while jersey digits sit on
curved fabric in a team-specific typeface under stadium light.

## Where this leaves identity

Jersey OCR does not reach 85% and will not without a model trained on this
domain. Training one needs a labelled set, and the only trustworthy labelling
route measured here is a person looking at crops -- the automatic route is
circular. That is a real piece of work, not a threshold.

Identity for recorded games remains exact from the feed, which names every
event's player, and the on-court five is exact from the box score plus
substitutions. The tactical layer is anonymous by design and does not need any
of this.

# Round twenty-six: play detection needs a registration that survives a play

## What the shot-frame number was hiding

Registration is quoted at 85.7% of court frames and 1.72 ft. Both are true of
frames chosen AT a shot, where the camera is pointed at a basket and the lane is
comparatively clear. A play is not a moment, it is six seconds, and sampled
continuously through the six seconds before each shot the same gate passes 17%.

The rejections are almost all one cause:

    key far from rim      307    79%
    OK                     63    16%
    no court               16     4%
    no rim detection        4     1%

Looking at the frames explains it in one glance. Mid-possession there are five
players standing in and around the paint, so the painted quad comes back as a
fragment, a triangle, or a self-intersecting bowtie. The gate is doing its job
by refusing them.

## Scarce is the smaller problem; the survivors disagree

Two registrations of the same instant can be compared exactly, because camera
motion between two broadcast frames is recoverable. Measured on the rim, ORB
alignment is accurate to **0.5 px median** (p90 1.9 px) -- camera motion is not
a source of error at all. So carrying frame A's registration to frame B and
comparing with B's own registration measures the REGISTRATIONS.

    gap      n   p50 ft  p75 ft  p90 ft
    0.2s   122      5.8    17.4    23.6
    1.0s    70     12.3    22.0    26.7
    3.0s    20     11.8    16.8    26.7

Two 1.72 ft registrations would disagree by about 2.4 ft. These disagree by 5.8
ft at a fifth of a second. The gate-passing frames mid-possession are not
merely rare, they are frequently wrong, which the shot-frame measurement never
showed.

No property of the quad separates the good from the bad well enough to use:

    side ratio of the worse quad   n     p50 gap   within 3 ft
      0.00 to 0.20                18      16.0 ft         11%
      0.20 to 0.35               112      16.3 ft         26%
      0.35 to 0.50                96       9.7 ft         16%
      0.50 to 1.00                23       2.7 ft         57%

The last row is the only encouraging one and it holds 9% of the pairs.

## Three ways to get one good registration per possession, all measured

Since alignment is exact, a window does not need 31 registrations. It needs
ONE, and camera motion carries it everywhere else. Three ways to find that one:

**Consensus.** Warp every candidate into a common frame and keep the largest
mutually-agreeing cluster -- RANSAC, with camera motion supplying the
correspondence. 5 of 25 windows produced a consensus at all, and it was no
better than picking a candidate arbitrarily: nearest player to the feed's shot
spot p50 10.7 ft consensus against 9.0 ft lone. Too few candidates are right
for a vote among them to mean anything.

**Anchor at the release and propagate back.** Trust the one frame the feed
corroborates -- where a detected player really is at the reported shot location
-- and chain outward. 3 of 18 shots produced an anchor the feed corroborated
within 5 ft, and those chains reached back a median of 1.4 s before alignment
failed. Not enough for a six-second play.

**A rim-placement gate.** Proposed, then withdrawn: the rim is ten feet above
the floor, so a floor homography maps its pixel to where the sight-line meets
the ground, well beyond the basket. That displacement is geometry, not error,
and the measurement built on it was meaningless. Recorded because it is exactly
the kind of plausible test that has to be checked before it is believed.

## Where that leaves offense and defense

Play detection from this broadcast is blocked on registration, not on the play
definitions. Nothing about screens, rolls or pin-downs is hard to state in court
feet; there is currently no reliable way to get the court feet for six
consecutive seconds of a possession.

So the play logic is being developed and measured on SportVU tracking data,
where coordinates are exact, identities are stable and the ball is known. That
measures the ceiling -- the definitions -- and says nothing about the vision
stack, exactly as the tracking-data plan requires it to be reported.

## The first thing that measurement found

`tracking_data` labelled every person PLAYER and nobody HANDLER, so
`Frame.handler` returned None on every frame and the on-ball half of play
detection silently did nothing: **zero ball screens across an entire game**. The
handler is not a guess on exact coordinates -- it is the nearest player to a
ball that is not in flight -- and with it set on 53% of frames the detectors
report:

    cross_screen      244    ball_screen       143    back_screen    136
    off_ball_screen   147    dribble_handoff   108    flare_screen    68
    pick_and_pop       45    pick_and_roll      44    pin_down         9

944 screen actions in one game against a real figure near 150. The definitions
are roughly five times too loose, and pin-downs -- one of the most common
actions in the sport -- come back nine times, so the off-ball naming is wrong
as well. Both are now measurable, which they were not an hour ago.

## Two definition bugs, found by running on exact coordinates

**Screens were being detected between opponents.** `detect_off_ball_screens`
considered every pair of non-handlers. Among nine non-handlers that is
thirty-six pairs, of which six are teammates on offense; the rest are two
defenders crossing, or an attacker and the man guarding him, who are close to
each other by definition. Passing the attacking five in:

    944 screen actions  ->  260

On-ball is now 131 against a real 90-110, and off-ball 129 against 80-100 --
about 1.3x too loose rather than 6x.

**Roles were assigned by track id.** The lower id was taken as the screener and
the higher as the cutter, which is right half the time by luck. The roles are
not cosmetic: every off-ball screen is named by the direction the CUTTER
travels, so reversing them turns a pin down into a back screen. A screen is set
and then left, so the screener is whichever of the two travels less afterwards.
That moved 27 previously-unnamed screens into real names.

What is NOT yet known is whether the names are right. There is no per-possession
play label in any feed, so the naming accuracy is unmeasured, and the remaining
1.3x looseness cannot be attributed without one. Hand labels rendered from
tracking coordinates are the next step -- unlike broadcast crops, a 2D plot of
a possession is unambiguous to label.

## Play detection, finally measured

No feed carries a per-possession play label, so ground truth had to be made.
On broadcast that is hopeless; on tracking coordinates a possession drawn as a
2D plot -- ten dots, exact positions, the paths they took -- is legible enough
to label by eye.

Two rules kept the labels honest. The detector's answer was never drawn on the
panel, because seeing "pick_and_roll" printed before deciding would make the
labels agree with the detector by construction -- the same circularity that
made the auto-generated jersey labels worthless. And half the panels were
moments the detector fired on, half moments it did not, because labelling only
its own firings measures precision and calls it accuracy.

A first sample of 16 panels gave precision 75% and recall 86%, and both false
positives were the same thing: two players running past each other in
transition, which satisfies every other test -- they were far apart, they came
together, one carried on. What was missing is that a screen is SET. The
screener plants and takes the contact.

Adding that (screener under 4 ft/s, expressed as a speed so it does not change
meaning with the sample rate) took a game from 260 screen actions to 156:
on-ball 63 against a real 60-80, off-ball 93 against 80-100.

The synthetic fixtures in `test_plays.py` had to be corrected to accept it.
They moved screeners 2 ft per 0.1 s -- 20 ft/s, about double a human sprint --
so no plausible "the screener is stationary" rule could have passed them. A
detector tuned to accept those trajectories is tuned to accept transition
run-bys.

### The result, and a lesson about sample size

Sixteen fresh panels, labelled blind after the rule was added, gave 88%
precision and 88% recall. Twelve more panels took it to:

    screen present or absent, 28 panels
      true positives 10   false positives 4   misses 2   agreed empty 12
      precision 71%   (10/14, 95% 48-95%)
      recall    83%   (10/12, 95% 62-100%)
      none / on-ball / off-ball, exact   21/28 = 75%

The 88/88 was noise. Twelve extra panels added three false positives and one
miss, and the interval at n=16 was wide enough to contain all of it. Reporting
the 88 would have been the same mistake as every inflated number earlier in
this file, arrived at honestly rather than by threshold-shopping.

**Play detection does not meet the 85% bar. It is at 71% precision and 83%
recall**, on 28 hand-labelled panels, and the bound is wide.

All four false positives are off-ball, and three are two attackers passing
close without either of them screening. The on-ball side did not produce a
single false positive in the sample. That is the next thing to fix, and it
needs more labels before any change to it can be believed -- at n=28 the
interval is 48-95%, which cannot distinguish a fix from noise.

## The play measurement does not survive a stricter protocol

The 71% precision above credited the detector whenever a screen was visible
somewhere in the panel. But the detector does not claim "a screen happened
here" -- it names two specific players. Scoring it on the looser question
counts a firing on the wrong pair as correct.

Re-run with the pair as the unit -- every sampled moment shows the two
offensive players who come closest, drawn across four instants, chosen by
proximity and never by whether the detector fired -- on 18 close pairs:

    tp 1   fp 5   fn 1   tn 11
    precision 17% (1/6)      recall 50% (1/2)

The same detector, the same game, the same labeller. The difference is entirely
in what counts as a hit.

### Why this is reported rather than fixed

Three things have to be true before any of these numbers means something, and
only the first is:

  * **The detector is measurable.** It is, on tracking coordinates.
  * **The labels are reliable.** They are not. Adjudicating the panel
    disagreements found one labelled "no screen" that plainly held one -- the
    pair had simply been missed among ten players and their paths. Errors in
    the other direction are equally likely and were not looked for, since
    hunting only for errors that favour the detector biases the result upward.
  * **The labelling criterion is the real definition.** It is not. The rule
    used here -- converge from about nine feet, one player planted, then part
    -- is a reasonable reading, but a screen set from seven feet is still a
    screen, and the threshold was chosen by eye. Most of the five pair-level
    false positives are rejections on exactly that margin.

So the honest statement is not that play detection is 17% or 71%, but that
**it cannot be certified against ground truth this project can currently
produce.** The bottleneck is the labels, not the detector.

### What does hold up

One check needs no labels at all, and it passes. Counts per game against what a
real NBA game contains:

    on-ball screens      63    real 60-80
    off-ball screens     93    real 80-100
    total               156    real ~150

Three separate corrections moved it there -- the missing ball handler, the
opponent pairing, and the planted-screener rule -- and each moved it toward the
real figure from a different direction. That is weak evidence, being a
distribution rather than a per-play score, but it is evidence that did not come
from anybody's eye.

### What would settle it

Synergy play-type data gives per-player season totals by category
(PRBallHandler, PRRollMan, Handoff, OffScreen, Cut). It is not per-possession,
so it cannot score a single detection, but it can score a PLAYER: a big whose
season is 30% roll-man should be detected rolling far more often than a guard
who is 2%. That is a real external check on the naming, it needs no hand
labels, and it is the next thing worth building.

## Synergy as an external check, and the control that deflated it

Synergy reports, per player per 2015-16 season, what share of his offensive
possessions are PRRollMan, PRBallHandler and OffScreen. That is external to
everything here and needs no hand labels, so it looked like the way out of the
labelling problem: given the two players in a detected ball screen, the one who
rolls for a living should be the screener.

Across twelve tracking games, 899 on-ball and 1137 off-ball detections:

    the detector's screener is the roll man   513/700 = 73%
    a RANDOM other teammate would be          422/702 = 60%

    off-ball, cutter is the screen user       507/804 = 63%

The control is the whole result. 73% sounds like a finding until you pick a
different teammate at random and get 60%, because the comparison is against the
BALL HANDLER, and a ball handler is by definition someone with a high
PRBallHandler share. Almost anyone else on the floor looks more like a roll man
than he does. The detector is worth 13 points over a random teammate, not 23
over a coin flip.

So Synergy does not certify play detection either, and the reason is worth
keeping: an external metric with a large sample is still worthless without a
null to measure it against. Every promising number in this project that later
dissolved had that shape.

## Where play detection actually stands

Four measurements, none supporting 85%:

    counts per game            on-ball 76 (real 60-80), off-ball 99 (80-100)
    panel-level hand labels    71% precision -- but credits firings on the
                               wrong pair, so it overstates
    pair-level hand labels     17% precision (1/6) -- correct unit, tiny sample,
                               and labels of demonstrated unreliability
    Synergy role check         73% against a 60% null

The counts are the only check that passes, and it is a distribution rather than
a per-play score: a detector could produce the right total by making
compensating errors. That it reaches the right range on six different games,
after three independent corrections each pushing from a different direction, is
worth something, but it is not 85% and should not be reported as though it were.

What would settle it is a per-possession play label from someone who knows the
game -- an expert pass over a few hundred possessions -- or a published labelled
dataset. Neither exists in this project. Every remaining route measured here
either lacks a null, lacks a sample, or scores a looser question than the one
the detector answers.

# Round twenty-seven: transition, the one play type with real labels

## Why this one is different

Every other play type ran aground on the same thing: no feed says whether a
possession contained a screen, so the labels had to come from a person looking
at dots, and that person proved unreliable. Transition is different. The shot
clock states exactly how long a possession ran, the detector never sees it, and
there are about 1,900 labelled possessions across twelve games. Thresholds are
fitted on six games and reported on six the fit never saw.

## Getting the labels right took three corrections

None of them was a modelling choice; each was found by checking the output
against what basketball looks like.

  * Splitting a possession at the first frame the other team touched the ball
    produced **390 possessions a game against a real 200** -- every deflection
    and contested rebound became one. Requiring a second and a half of
    sustained control gives 203 and 202 in two games, median length 13.8 s.
  * Reading the clock at a possession's last frame read a reset that had
    already happened: a made basket resets to 24 the instant it drops, so half
    of all possessions came back unlabelled. The MINIMUM reading across the
    possession is how far the clock actually ran down.
  * Reading it on arrival in the frontcourt labelled two thirds of possessions
    "early", because crossing half court takes a few seconds in any offense.
    Transition is about finishing early, not arriving early.

The labels then read 8-12% transition, which is the shape the sport has. That
imbalance also means accuracy is worthless as a score: **always answering
half-court gives 88% and detects nothing**, so everything below is F1 on the
transition class.

## Two faults that made half of all breaks invisible

`distance_to_basket_ft` measures to `BASKET`, which names ONE end of the court.
On full-court coordinates every possession attacking the far rim was measured
as retreating, so half of all breaks could not be detected at any threshold.

And the detector stopped the moment the handler changed -- but a break begins
with an outlet pass and often ends with a different player finishing. A test
asserted this behaviour, `test_transition_needs_the_same_handler_throughout`,
and its fixture put both players at identical coordinates so it could not tell
a pass from a steal in the first place.

## The results, and a leak caught before it was reported

    threshold rule, before either fault was fixed      F1 0.15
    threshold rule, after both                         F1 0.32
    logistic model over possession geometry            F1 0.57   <- leaked
    the same model, first four seconds only            F1 0.37

The 0.57 was the model reading the label back to itself. Its dominant feature
was POSSESSION LENGTH, weighted more than twice as hard as anything else -- and
the label is "the possession ended within seven seconds". Length is a near-copy
of the answer.

The constraint the task should have carried from the start is that only the
OPENING of a possession may be looked at. Predicting whether a possession will
finish inside seven seconds from its first four is a prediction; measuring it
over the whole possession is a restatement. With that imposed the model scores
**F1 0.37 held out** -- precision 37%, recall 38% -- barely ahead of the single
threshold rule.

## What that says

Transition does not reach 85% either, and the reason looks intrinsic rather
than fixable. The label is "the offense finished early"; the evidence is "the
ball was pushed". A team can push the length of the floor in four seconds and
then run twelve more seconds of offense, and no feature of the opening
distinguishes that from a break that ends in a layup. The two events are
correlated, not the same, and F1 0.37 is roughly what that correlation supports.

So the honest position on plays, across every route tried:

    screens, counts per game        on-ball 76 (real 60-80), off-ball 99 (80-100)
    screens, panel hand labels      71% -- but scores a looser question
    screens, pair hand labels       17% on six firings
    screens, Synergy roles          73% against a 60% null
    transition, shot clock          F1 0.37 held out

The counts are the only check that passes, and they are a distribution rather
than a per-play score. Nothing here supports 85%.

# Round twenty-eight: the passer who created the basket — 87%

## Why this one could be measured when the others could not

Every earlier play-side attempt lacked per-event ground truth. No feed says
whether a possession held a screen, so the labels had to come from a person
looking at dots, and that person was demonstrably unreliable. The shot-clock
labels for transition describe a possession rather than an action.

Assists are different. The play-by-play credits an assister by name on every
assisted basket — "Vucevic 19' Jump Shot (2 PTS) (Payton 1 AST)" — and tracking
data carries the same game clock. All 54 assisted baskets in the first game
aligned at **zero** clock offset. The detector never sees the credited name.

## Two faults, and what they cost

**The handler had no memory.** Marking the nearest player to the ball frame by
frame is right most of the time and catastrophically wrong the rest: a defender
an inch nearer for a tenth of a second takes the ball off the man dribbling it.
On assisted baskets the player holding it before the shooter came back as an
**opponent 31 times in 54**. A handover margin — you keep the ball until
somebody is clearly nearer — fixed it.

**Walking back from a basket lands on the wrong possession.** At the logged
moment the ball is already through the net and in the hands of whoever
collected it. The ball's HEIGHT marks the shot: step back over the frames where
it was above head height, and the shooter is the last man to hold it before
that.

    29%   as first written
    66%   after stepping over the shot arc
    74%   after giving the handler memory

## Anchoring, not conditioning

At that point the measurement said "93% when the shooter is identified
correctly". That number is worthless: it is scored only on the possessions
where the tracking was already unambiguous, which is a selection of the easy
ones. It is the same shape as every inflated number in this file.

The feed names the scorer on every basket, so the honest version anchors on
that name and is scored on **every** event. It is also how the product works:
the endpoint supplies the event, vision supplies who did what.

    ANCHORED on the feed's scorer, twelve games
      resolved on 227 of 515 assisted baskets (44%)
      passer matches the credited assister  198/227 = 87%   (95% 83-92%)

**87% on 227 events with exact per-event labels.** The point estimate clears the
bar and the interval reaches a little below it, which is the same standing as
the jersey reader's 88%.

Coverage is 44% because SportVU events are windows around plays, so the scorer
is often not in the tracking window at all. That is a property of this dataset,
not of the method.

## What this does and does not settle

It settles one offensive action: given that a basket was scored, who created it.
That is a real film-study primitive and the first play-side capability here to
reach the bar on a measurement with nothing wrong in it.

It says nothing about screens, which still have no trustworthy labels, or about
transition, which sits at F1 0.37. And it inherits the broadcast limitation
recorded in round twenty-six: measured on tracking coordinates, it does not
measure the vision stack.

## Tightening it: a confidence gate, cross-validated

87% on 227 events had a 95% interval of 83-92%, whose lower bound sits under
the bar. Two things could close that, and the error analysis said which: wrong
attributions have a passer who held the ball for a median of **0.2 s against
0.7 s** for correct ones. A tenth of a second is not a pass; it is the tracker
resolving a contested moment badly.

The first use of that signal was backwards. Stepping OVER such a blip to take
the man before him raised coverage from 43% to 58% and dropped accuracy from
86% to 81%, because the events it newly resolves are exactly the contested
ones. Declining to answer is the right trade, and the same one the jersey
reader makes: a wrong attribution is worse than none.

Accuracy then rises monotonically with the requirement, which is what a real
confidence signal looks like rather than noise:

    hold required   coverage   accuracy     (fitting games 0-5)
      0.0 s            43%        86%
      0.2 s            37%        87%
      0.3 s            30%        92%
      0.5 s            25%        94%

The threshold is chosen by a rule fixed before any held-out game was scored --
the lowest requirement reaching 90% on the fitting half -- and each fold is
reported on the half that did not choose it:

    chose 0.3s on games 0-5     reported on games 6-11    76/85 = 89%
    chose 0.5s on games 6-11    reported on games 0-5     61/65 = 94%

    pooled held-out   137/150 = 91%   (95% 87% to 96%)

**91%, with the interval's lower bound at 87%.** This is the first play-side
number in this project to clear 85% with confidence rather than on a point
estimate, and every event in it was scored by a threshold chosen without it.
The shipped default is 0.4 s, the mean of the two the folds selected.

Coverage is 29% of assisted baskets. That is the deliberate half of the design;
the involuntary half is that SportVU events are windows around plays, so the
scorer is often not in the tracking window at all.

# Round twenty-nine: rebounds do not attribute, and why assists do

Rebounds have the same exact per-event ground truth as assists -- the feed
credits a rebounder by name, about 97 a game, the largest label set available --
so the same anchored method was applied. It does not work.

    first sustained holder after the logged moment      43%
    longest holder in a window straddling the moment    45%
    first holder after the ball comes off the rim       47%
    the same with ball-velocity matching                46%
    the same at 25 Hz instead of 10                     45%

The ceiling is not much higher than the result. The credited rebounder holds
the ball **anywhere in a seven-second window only 72% of the time**, so more
than a quarter of rebounds cannot be attributed by this route at all.

The reason is specific and it explains why assists behave differently. Ball
possession is derived from proximity, and a rebound is a scramble: several
players are within arm's reach of the ball at once, which is exactly the
condition proximity cannot resolve. A pass has a clear giver and a clear
receiver separated in space and time, and that is why the same machinery
reaches 91% there.

Two ideas that sounded right and were not:

  * **Longest hold in the window.** After a defensive rebound the ball is
    outletted at once to a guard who dribbles for several seconds, so the
    longest hold is the guard, not the rebounder.
  * **Velocity matching** -- a held ball travels with its holder, a ball merely
    passing near somebody does not. Physically true, and it changed nothing:
    46% against 47%, with coverage falling from 85% to 74%.

## The per-event scorecard

    assists, who made the pass       91%   (cross-validated, 137/150)
    rebounds, who secured it         47%   (ceiling 72%)
    screens                          not labellable by this project
    transition                       F1 0.37

What separates the first line from the rest is not the method, which is the
same in each case. It is whether the moment being attributed has one obvious
owner. A pass does. A scramble does not, a screen is a judgement, and a
transition is a property of a possession rather than an act.

# Round thirty: screens from film, on labels nobody here wrote — 87%

## The labels existed all along

This file spent several rounds concluding that no trustworthy screen labels
were available and that screen detection therefore could not be certified.
That was wrong. SpaceJam labels **712 clips as `pick`** -- a person watched each
one -- and the archive has been sitting in `data/labeled/` throughout.
`fetch_spacejam_subset.py` mapped class 7 into "other" alongside walk and run,
and that mapping was taken at face value instead of being checked against the
source annotations.

Every screen number reported before this one -- 71%, 17%, 50% -- was scored
against labels this project made up. This one is not.

## Construction

**Split by SOURCE clip.** SpaceJam ships every clip twice, as `<id>` and
`<id>_flipped`, mirror images of the same footage. Splitting by clip would put
a near-duplicate of a training example in the test set. Verified: zero source
clips appear in both train and test.

**Hard negatives.** The negative pool is drawn from defence, no_action and
ball-in-hand -- the things a set screen actually resembles -- with `walk` and
`run` capped at a tenth each. A classifier that separates screens from walking
has learned nothing.

## Result

    frozen Kinetics features, linear probe      77%
    fine-tuned, one epoch                       87%

    TEST (216 clips, no mirror of any training clip)
      accuracy  87%   (95% 82% to 91%)
      precision 88%   recall 86%

The frozen probe is the useful control: 77% against a 52% chance rate says the
signal is partly generic motion, and the jump to 87% says most of it is not.

One epoch. Training was stopped early to free the machine, so this is a floor
rather than a ceiling.

## What it does and does not establish

It runs on **video**, which every other play measurement here does not.
Assists at 91%, the screen geometry, transition -- all of those need SportVU
coordinates that exist for 636 historical games and never for the film a coach
studies. This runs on the film.

It does not establish transfer. SpaceJam clips are tightly cropped to one
player from a fixed pool of games; a screen is a relationship, and a crop that
excludes the man being screened is a good reason to expect the 87% to be
conservative AND a good reason not to assume it survives on other footage. The
MultiSports run that follows uses a wider crop for exactly that reason.

# Round thirty-one: seven classes on film — the tactical ones fall short

MultiSports annotates the specific player performing an action, frame by frame,
on 720p broadcast footage, and covers the defensive side that nothing else
labels. 3,035 clips were cut at a 1.6x crop -- wider than SpaceJam's, because a
screen is a RELATIONSHIP and a crop that excludes the man being screened cannot
show one -- and split by video so no two tubes from a possession straddle it.

    class                      n   recall  precision
    screen                    45     71%      68%
    pick_and_roll_defensive   42     50%      64%
    sag                       40     57%      53%
    drive                     52     40%      88%
    interfere_shot            74     91%      91%
    pass                     103     95%      91%
    dribble                  103     87%      69%

    overall 77%;  the five PLAY classes alone 65%  (chance 14%)

The easy classes are at 87-95%, so the pipeline works. The tactical ones are
50-71%, and that is the finding.

## More training will not fix it

Validation peaked at 81% after one epoch and fell to 78% after the next while
training loss halved. That is overfitting, on 152-223 training clips per
tactical class: a data-size limit, not a training-length one. Training longer
here would only have moved the training loss.

## Where the errors go, which is what they mean

    truth \ predicted        screen  pnr_def  sag  drive  interfere  pass  dribble
    screen                      32      7      3     0       0        2     1
    pick_and_roll_defensive     10     21      4     1       1        1     4
    drive                        0      1      3    21       0        0    27

Screen errors land almost entirely on `pick_and_roll_defensive` -- 7 one way,
10 the other -- which is the same ball screen labelled from the two sides of
the ball. That is a boundary between labels, not a failure to see the action.
Merged into a single "ball screen" class:

    recall 80% (95% 72-89%)   precision 88%   on 87 clips

`drive` behaves the same way against `dribble`: 27 of 52 drives are called
dribbles, and a drive IS a dribble toward the basket. Its 40% recall against
88% precision says the model rarely says "drive" and is usually right when it
does -- a threshold artifact rather than blindness.

## What this says about the plan

Screens scored 87% as a binary task on SpaceJam and 71% here. The difference is
the seven-way discrimination, not screen recognition. Merging the two sides of
the ball screen recovers most of it, and the remaining gap is training data:
223 screens is a tenth of what the easy classes have.

Both label sources hold screens -- 712 in SpaceJam, 223 plus 203 defensive-side
in MultiSports -- so the next step is to pool them and test on each source
separately. Training on both and holding up on both would be evidence of
recognising the action rather than the dataset.

# Round thirty-two: a ball-screen detector that holds on both datasets — 92%

Two facts from the seven-way run pointed here. Screen errors went almost
entirely to `pick_and_roll_defensive` -- one ball screen labelled from the two
sides of the ball -- so those belong in one class. And the tactical classes
overfit at 152-223 training clips, so the shortfall was data, not method.

Both public sources label screens. Pooled: 916 screen clips in training, 490
from SpaceJam and 426 from MultiSports.

## Tested per source, on purpose

The two datasets differ in ways a model could learn INSTEAD of the action:
SpaceJam crops tight to one player, MultiSports at 1.6x with the screened man
in frame; different games, cameras and resolutions. A pooled score averages
over that and hides it, so the test is reported per source and the honest
figure is the WEAKER of the two.

    TEST — read once
      pooled        675 clips   accuracy 92% (89-94)   recall 88%   precision 84%
      spacejam      216 clips   accuracy 86% (81-91)   recall 88%   precision 85%
      multisports   459 clips   accuracy 94% (92-96)   recall 87%   precision 83%

Both sources clear 85% on accuracy and both sit at 87-88% recall. Precision is
the weakest column at 83-85%, and MultiSports' 83% is below the bar.

A model that held up only on the source it saw most of would have learned the
dataset. This one holds on both, across different crops, cameras and games,
which is the strongest evidence available here that it recognises the action.

## The progression, and what each step bought

    frozen Kinetics features, linear probe          77%   (chance 52%)
    fine-tuned on SpaceJam alone, binary            87%
    MultiSports `screen` inside a seven-way task    71% recall
    both sources pooled, the two sides merged       92% pooled, 86%/94% per source

The 71% is not a regression -- it is the same model asked a harder question,
and the confusion matrix showed the loss went to the defensive-side label
rather than to pass or dribble.

## What is still not established

Transfer to THIS project's broadcast. Both datasets are other people's footage
at other camera angles. Cross-source generalisation is the closest available
proxy and it is encouraging, but it is a proxy.

And this is one play type. Pin-downs, flares, Horns and the defensive coverages
-- drop, hedge, ICE -- are not labelled in either source. Formations remain the
one part of that reachable by rule, for the reason recorded in round thirty:
a formation is a configuration two people would agree on, and an action is not.

# Round thirty-three: the screen model does not transfer to our broadcast

The ball-screen detector reaches 92% pooled and 86%/94% per source on SpaceJam
and MultiSports. Run on this project's own footage it does not work.

    model positives labelled            5
    of which were really screens        0
    95% upper bound on precision       45%   -- excludes the 85% bar

One of those five positives (score 0.51) was a SPECTATOR standing behind the
baseline. That is the clearest single statement of the problem.

## Getting to a fair test took six pipeline fixes, and one remains

None of these were model problems, and every one failed SILENTLY -- producing
clips that looked reasonable in aggregate and firing rates I described as
"plausible" while the subject was often not in frame.

  1. **Candidates were near-camera bystanders.** Picking a random detection
     with a tall box selects whoever is closest to the camera, usually a
     weak-side player thirty feet from the play. Both training sets are action
     tubes -- every clip centres on a player DOING something -- so this asked
     the model about a category it has never seen. Fixed: sample the players
     nearest the ball.
  2. **The crop box did not follow the player.** It was carried by ORB, which
     removes the CAMERA's motion, not the player's, so it slid off anybody who
     moved -- four frames in six came back empty. A screen is exactly where
     players move. Fixed: snap to the nearest detection each frame.
  3. **A third of clips were spectators.** The person detector finds people in
     the stands, and a box-height filter cannot exclude them because a
     courtside spectator is as large as a player. Partly fixed by requiring the
     patch under the feet to be lit floor: court 179-200 in value, crowd 42-126.
  4. **Officials and staff still pass that filter**, because they stand on the
     lit floor. Three colour filters were tried and none separated them:
     saturation puts a grey referee (101) inside the players' range (90-175),
     and kit-hue fails because this arena's court is painted blue and the crowd
     wears blue, so the background supplies the kit colour.
  5. **Identity switches mid-clip.** The nearest-detection snap jumps to the
     wrong player when two cross, despite a body-height gate.
  6. **The ball detector marks referees and spectators**, so "sample near the
     ball" was partly sampling near a bald head. This also makes the ball
     unusable as labelling context.

After all of that, 30% of extracted clips are still unusable -- non-player or
broken -- and roughly one moment in four survives every gate.

## What this does and does not mean

It does NOT mean screen recognition failed. That is demonstrated at 86-94%
across two independent datasets with different cameras, crops and games, and
the cross-source result rules out learning one dataset.

It means the model is being fed rubbish. A clip whose subject is a spectator,
or who walks out of frame, or who is swapped for another player halfway
through, carries no screen for the model to recognise -- and the model, having
only ever seen well-formed action tubes, answers anyway.

The gap is the extraction pipeline: which player to crop, at which moment,
keeping the box on him, and knowing he is a player at all. That is ordinary
engineering with a clear target, not a modelling dead end, and it is now
enumerated rather than guessed at.

## The honest per-source picture

    SpaceJam test        86%   (216 clips, human labels)
    MultiSports test     94%   (459 clips, human labels)
    our broadcast       0/5    (95% upper bound 45%)

Two curated datasets agreeing and the live pipeline failing is the signature of
a data-delivery problem, not a perception one.

# Round thirty-four: fixing the delivery, not the model

The screen classifier scored 0 of 5 on this broadcast while reaching 86% and
94% on two public datasets. Everything below is a fault in how clips reached
it. None required touching the model or the bar.

  1. **Sampling the whole file.** Candidates were drawn uniformly across two
     hours. The game runs 540-7209 s; the rest is pre-game packages, replay
     inserts and the trophy ceremony -- one top-ranked "screen" was a Nuggets
     trophy presentation. No court/ball/kit gate excludes those, because a
     celebration has a lit floor and people in two colours. Candidates now come
     from 1-8 s before each of the 149 aligned shots, which is guaranteed live
     play and where screens concentrate. Acceptance went from 25% to 43%.
  2. **Bystanders instead of players near the ball.** Both training sets are
     action tubes centred on a player DOING something; a random tall detection
     is whoever is nearest the camera, usually a weak-side player thirty feet
     away.
  3. **Spectators and officials.** No size or brightness rule excludes them --
     a courtside spectator is as large as a player and officials stand on the
     same lit floor. Three colour heuristics failed, one of them because this
     arena's court is painted blue and the crowd wears blue, so the background
     supplies the kit colour. What works is cluster membership: the ten players
     wear two colours and nobody else wears either, so `candidates.kit_members`
     keeps only members of the two balanced torso-colour clusters and refuses
     entirely when they do not separate.
  4. **A crop that did not follow the player.** The box was carried by ORB,
     which removes the CAMERA's motion and not the player's, so four frames in
     six came back empty. It now snaps to the nearest detection each frame, and
     a snap further than 0.6 body-heights drops the clip rather than switching
     to the wrong player.
  5. **Clips cut at arbitrary instants.** The training tubes are centred ON the
     action. A screen lasts about a second inside a fifteen-second possession,
     so an arbitrary instant shows the approach or the aftermath. Scoring five
     offsets across +-0.6 s and keeping the peak is ordinary dense action
     detection; it raised the firing rate from 15% to 30% on identical clips,
     so this fault alone was costing half the detections.

## What the ranking does now

Eight clips labelled blind, top of the ranking against low-scoring controls:

    score  label        why
     0.94  engagement   defender engaged with a possible screener
     0.66  engagement   defender engaged, contact unclear
     0.47  broken       box empty for half the clip
     0.03  no           transition run
     0.03  no           transition run
     0.00  no           transition run, alone
     0.00  broken       box drifts off the player
     0.00  no           shot/rebound, camera flash

The ordering is real: player ENGAGEMENT at the top, open-floor running at the
bottom, cleanly separated. That is the opposite of the previous run, where the
top of the ranking held a trophy ceremony and a spectator.

## What is still not established, and why

Two clips at the top is not a measurement. And these stills cannot settle
whether a congested defensive engagement is a SCREEN -- which is the same
limit that made this project's own screen labels unusable three times over.

The blocker is now throughput, not method: extraction runs at about two minutes
per candidate, dominated by a YOLO pass on every frame of every offset. A few
hundred candidates would give a real precision figure at the top of the
ranking, and that is a compute problem with an obvious fix -- batch the
detector, or track between sparse detections rather than detecting every frame.

## The extractor, made fast enough to measure with

Building candidates cost two minutes each, which put a few hundred out of
reach. Two optimisations were guessed at and neither helped; profiling settled
it in one run:

    read the span      0.32 s
    court check        0.01 s
    detection         60.11 s      <-- everything
    camera alignment   5.50 s

Detection was 60 of 66 seconds, at 2.6 s per 1080p frame for yolo11x. `person`
is the easiest class in COCO and these players are large, so the biggest model
was buying nothing:

    model     s/frame  speedup  players >=90px  agreement with 11x
    yolo11x      2.62     1.0x            8.8          --
    yolo11s      0.41     6.3x            9.0          87%
    yolo11n      0.16    16.5x            5.9          62%

yolo11s finds slightly MORE large players than yolo11x while running 6.3x
faster. A missed detection makes the tracker drop the clip rather than take the
wrong player, so this is a throughput trade rather than an accuracy one.
Candidates now cost about 18 s: an eightfold speedup, and 220 in an hour.

Half-resolution ORB was tried too. It saved three seconds of sixty-six and
destabilised the warp -- boxes collapsed to slivers over a few frames and the
clips came back empty. Reverted, with a size guard added: a snap that changes
the box height by more than about half is a partial detection or a different
person, and the clip is dropped.

## Why colour cannot find the players in this arena

The candidate filter clustered torso colour into two kits and kept only
members, which worked on synthetic frames and failed on the broadcast: a
labelling sheet came back with three of six clips showing people at the
sideline. The reason is specific and worth recording -- this is a "blue out"
crowd, so the spectators wear the home kit's colour, and a fan in a blue shirt
clusters with a player in a blue jersey. No threshold on that axis separates
them.

Position does. The court is one large connected region of wood and paint; a
player stands inside it and a spectator stands beyond its edge. The mask closes
over gaps and so spills slightly past the true boundary, so it is eroded before
testing, and the amount was swept rather than guessed:

    erosion   people kept per frame   dropped as off-court
        0 px                   8.1                     12
       25 px                   7.1                     29
       45 px                   6.6                     38
       65 px                   5.7                     55

A broadcast camera shows six to nine of the ten players, so 45 px is the
setting that matches the sport.

**Referees remain.** They stand on the court, so position cannot exclude them,
and colour cannot either for the reason above. Contamination is roughly one
clip in six and is treated as a known quantity: clips are marked player or
non-player during labelling, only players train or score, and the rate is
reported separately rather than folded into the model's error.

# Round thirty-five: labelling a training set is a base-rate problem

Fine-tuning the screen classifier on this broadcast needs positives, and the
obstacle is not the labelling effort per clip -- it is how few clips contain a
screen at all.

## Two faults in the labelling instrument, found before trusting it

**Four frames straddle the screen.** Sheets at 0.4 s spacing produced ZERO
screens across eighteen clips sampled from shot lead-ins, which is not credible
about the sport. Contact-and-run-off lasts well under a second. At eight frames
0.18 s apart the same footage shows it, and those eighteen labels were
discarded rather than used.

**Colour cannot find players here, and position can only half.** Recorded in the
previous round; the residue is officials, who stand on the court and wear
neither kit.

## The base rate, measured

Nine clips sampled as a random kit-member near the ball, labelled at eight
frames against a criterion written before any sheet was seen:

    clear screens 0    unclear 2    clear no 7

At a 5-10% base rate, fifty positives costs six hundred to a thousand labelled
clips -- five to eight hours of reading, by a labeller whose screen judgement
has failed four distinct ways in this project.

## Proposing by motion instead

A screener is geometrically distinctive and needs no court registration: with
camera motion removed by the same ORB warp used elsewhere, he is a player who
comes to a STOP while a team-mate passes within about a body-width. The rule is
deliberately loose, because it selects what to LOOK at and never what to
believe -- a tight rule would be the old geometric detector wearing a different
hat, and its errors would silently become the training labels.

Nine of its proposals, same view and same criterion:

    source              clear screens   unclear   clear no
    random near-ball          0 (0%)      2        7 (78%)
    motion proposals          2 (22%)     5 (56%)  2 (22%)

The proposer accepts 25% of moments against 40% for random sampling, so it is
more selective; more to the point it is ENRICHED. Random sampling returns
mostly open-floor running; every proposal is at least a contact situation.

## What that buys and what it does not

At 22% clear positives, fifty of them costs about 230 labelled clips rather
than six hundred to a thousand -- roughly a quarter of the work, which makes a
fine-tuning set reachable.

The 56% unclear rate is the honest caveat. Even at eight frames, most contact
situations cannot be resolved into screener-and-cutter from stills, which is
the same wall NETS described when it said experts disagree on edge cases and
its own annotators watched video with replay and freeze.

# Round thirty-six: rebuilding from the foundation, and three defects on the way

## Why the rebuild

Two halves of this project sit side by side and explain each other. On exact
court coordinates it works: assist attribution 91% cross-validated, screen
counts in the real range, formations at plausible rates, transition measurable
-- none of it needing a hand label, because geometry defines the answer and the
feed scores it. On broadcast video the foundation is not there: registration
17% mid-possession with survivors disagreeing by 5.8 ft, 463 track identities
for ten players, a ball detector that fires on referees.

So every action and tactic model has been asked to infer from a crop of one
player what geometry would state outright. That is why a screen classifier
scores 86-94% on curated clips and 0 of 5 here, why congestion is
indistinguishable from a screen, and why four attempts at hand-labelling
screens drifted four different ways: the label is a *relationship* and the
evidence was a picture of one participant.

## Phase 0: three things already written, wired to nothing

**The strict gate was not connected.** `build_game_model.court_positions`
called `key_homography` bare. The gate is not a free improvement, it is a
trade, and the numbers are on record:

    gate            coverage   p50 error   within 3 ft
    none              92.3%      2.84 ft       50.7%
    <= 220 px         18.5%      1.72 ft       79.1%

Every position this project has produced came from the top row while the
bottom row's 1.72 ft was quoted downstream.

The gate is now applied, and **it costs more than the first version of this
section claimed.** Measured on the script's own 30 events rather than argued
from independence: **30/30 events carry context ungated, 14/30 gated.** The
eight offsets do not rescue it -- they span 6.5 s of a single shot, and the
gate rejects on camera framing, so when the framing is wrong it is wrong for
all eight. An earlier commit had already measured 51.4% of shots by this route
and that number was not carried forward.

And 1.72 ft is a proxy rather than established truth. The gate variable is the
pixel distance from the key centre to the rim; the metric that scored it is the
projected rim against `BASKET` in feet. Those are mechanically coupled -- the
key centre maps a fixed 4.25 ft from the basket -- so the gate largely selects
on its own evaluation, and says nothing about error at the far arc where a
small quad extrapolates worst. Round 26's independent check, carrying
gate-passing registrations across 0.2 s with 0.5 px alignment, found them
disagreeing by **5.8 ft**, where 1.72 ft registrations would disagree by about
2.4. The honest description of the gate is "it rejects obviously-wrong keys",
not "it delivers 1.72 ft".

**`detect_paint_hue` was wired to nothing.** Every arena paints its own key;
three broadcasts of four sit at hue 107-113 and one at 174, and on that fourth
arena the hardcoded blue finds the key on 10% of court frames against 87% with
calibration. (97% is the figure for an arena the default already suits, and
quoting it here was a misreading of that table.)

The first version of the calibration was wrong in a way worth recording. It
swept 60-600 s, but the game runs 540-7209 s, so ten of twelve probes were
pre-game; of the five frames that passed `has_court`, three were the anthem
line-up, a player introduction and a coach close-up, because skin and warm-ups
fall inside the "wood" hue range. It reached the right answer by luck. It now
samples a few seconds before aligned shots -- live play by construction -- and
**only adopts the calibrated range if it finds more keys than the default on
those same frames**, because a wrong calibration is silent and strictly worse
than the default.

**`precise_key_corners` raised `NameError`** on any call that reached its body,
and `TypeError` on any call passing the keyword. A frame with no court returns
before that line, which -- with no callers and no tests -- is why a broken
function sat in the module unnoticed.

## Retraction

Round eight's registration figures (0.88 of frames, 0.5 px rim, ~0.05 ft) are
retracted in place. Round 23 measured the same search at 28.3% registered with
the wrong basket on 100% of frames. The Round 8 numbers were still being cited
forward as the basis for projecting the rim rather than detecting it.

## Round 37 - the schema is not the bottleneck, and two-thirds of the labels are filler

Phase 1 replaces the painted key with 48 learned landmarks. Before training
anything, two facts about the dataset had to be established, because both would
have been invisible failures.

**The `v=0` keypoints carry coordinates, and the coordinates are junk.** Of
40,848 annotated slots, 10,678 are flagged visible (`v=2`), 12,765 sit at the
origin, and **17,405 carry a real-looking position with `v=0`**. Those 17,405
look like free supervision -- they would have tripled the training signal. They
are not real. Fitting a homography on the `v=2` points alone and reprojecting
the `v=0` points puts them a median **48.7 ft** from where the schema says
their landmark is, with **0.8%** inside 3 ft, and none of them fall outside the
image, so "annotated but off-frame" does not explain it. Training on them would
have taught the model to place two-thirds of the court at random.

**The recovered schema predicts landmarks it was never fitted to, to 0.35 ft.**
Holding out each visible landmark in turn, fitting on the rest, and asking
where the held-out one lands: **p50 0.35 ft, p90 1.04 ft, 99.4% within 3 ft**,
over 10,501 predictions on 840 frames from many arenas. This is the
leave-one-out number, not the fit residual (0.23 ft), so it is a prediction
rather than a measure of its own fit.

That matters for what remains. The Phase 1 gate is 2 ft of court error, and the
schema plus human landmark positions deliver 0.35 ft -- roughly six times the
headroom. **Every remaining foot of error is the detector's**, not the
geometry's. The worst two indices are 8 and 34 at 1.17 ft, which are the
baskets: the one "landmark" in the schema that is not painted on the floor, so
annotators are clicking a rim 10 ft up and parallax is expected.

**Training was running on the CPU.** Ultralytics logged `torch-2.13.0 CPU
(Apple M2)` and took 20 s per iteration -- 120 epochs would have been over two
days. The rest of the repo routes through `courtvision.device.resolve_device`;
the trainer did not, and ultralytics does not pick MPS on its own. Passing the
device explicitly: 5.8 s/it at batch 16, about seven times faster.

## Round 38 - the keypoint loss had no gradient

The first training run looked healthy: box mAP50 0.995, box loss 4.8 -> 0.51.
Pose mAP was **exactly 0.0** at every validation, which reads as "still early",
and pose loss sat at 10.67 of a ceiling near 12.

It was not early. Ultralytics scores keypoints with `1 - exp(-e)` where
`e = d^2 / ((2*sigma)^2 * area * 2)`, and for any keypoint count other than
COCO's 17 it invents `sigma = 1/nkpt` -- here 1/48 = 0.021. The tolerance that
implies is `2*sigma*sqrt(area)`, and `area` for a court is the whole frame, so
about 24 px. A COCO-pretrained pose head places body keypoints near the middle
of the box and starts roughly 280 px from the court's landmarks. That is
`e = 71`, and **`exp(-71)` is zero in float32**: the loss returns a flat 1.0
with no gradient, and the keypoint head cannot move at all.

What settled it was measuring the landmarks themselves rather than the metric:
**281 px median error** after two epochs. A dead gradient and a slow start are
indistinguishable in the loss curve; they are not indistinguishable in feet.

`sigma` is now chosen so the initial error lands near `e = 1`, where the
gradient is largest. Matched at six epochs, 640 px, same seed and schedule:

    sigma        landmark error    pose mAP50
    0.021        346 px            0.0 (exactly)
    0.18          77 px            nonzero
    0.35         263 px            near zero

The curve has an optimum rather than a direction, which is the useful part: the
obvious reading of the bug -- "the tolerance is too tight, so loosen it" --
picks 0.35 and lands 3.4x worse than 0.18. Too small saturates `exp(-e)` to no
gradient; too large weakens it, since the gradient scales as `1/sigma^2`. Only
the sweep distinguishes those, and the loss curve cannot, because its magnitude
depends on sigma too.

The default arm is **worse at six epochs than it was at two** (346 px against
281). It is not learning slowly; it is not learning, and the drift is noise --
which is what a zero gradient predicts and what the loss curve alone could
never have shown.

**A correction.** The first write-up of this cited "pose loss 4.6 against 11.7"
as evidence that the gradient had been restored. That comparison is not valid:
`sigma` appears in the loss itself, so raising it shrinks the reported number
whether or not any prediction improved. Loss magnitudes are incomparable across
sigma by construction. The evidence is the landmark error in pixels and the
pose mAP, both of which are computed independently of the training loss.

**How much accuracy is actually needed.** Differentiating the reference
homographies at the annotated landmarks gives 0.0292 ft per pixel (p90 0.0361),
so the 2 ft gate corresponds to about **68 px** of median landmark error --
which six epochs almost reaches. This is a conservative conversion: a
homography fitted over roughly twelve landmarks averages down independent
per-landmark noise, so the court error should come in below what this scaling
predicts.

**Registration can also be fused across frames.** ORB aligns consecutive
broadcast frames to 0.5 px and the court is rigid, so every frame in a window
is an independent measurement of the same registration. `fuse_registrations`
carries the neighbours onto the centre frame and takes the median court
position per probe -- median, because a wrong registration is not a small error
but the other end of the floor, and a single one would drag an average. A
refused ORB hop truncates the window instead of being chained through.

## Round 39 - an absolute registration check, and the one thing it cannot see

The ORB consistency test proves a registration is **stable**. It does not prove
it is **right**: a constant offset, or the wrong basket, agrees with itself
perfectly across 0.2 s and passes. Something has to pin the answer to the court
itself, and the court is painted, so warping a frame into court coordinates and
asking what fraction of its edges land on lines that are actually painted is an
absolute check that needs no annotations, no feed, and no model of ours.

Validated by breaking a known-good homography -- the dataset's human
annotations -- in the ways that matter, over 58 frames:

    human annotation     0.380
    slipped 5 ft         0.274
    twisted 4 degrees    0.272
    wrong end            0.379   <- NOT DETECTED

**The end swap is invisible, and no check of this kind can see it.** An NBA
court's markings are symmetric about half-court, so flipping ends maps every
painted line onto a painted line. The flip composes with an affine reflection,
which preserves all projective structure, so the horizon and the foreshortening
are identical as well and perspective cues do not help either. Which end of the
floor a frame shows is not recoverable from the floor's geometry.

This is worth stating plainly because Round 23 measured the painted key putting
the **wrong basket on 100% of frames**, and a line-agreement check would have
certified every one of them. The end has to be established elsewhere: from the
model's learned appearance -- a far basket is smaller and sits higher in frame
-- and it is measured against human annotations in `eval_court_keypoints.py`,
which counts registrations landing on the wrong half. The line check is silent
on it by construction, and a test pins the mask's symmetry so that if a future
change made it asymmetric, this blind spot would not quietly stop being true.

The check also carries its own control: every frame is scored a second time
with a registration slipped 5 ft, on the same broadcast. The gap between the
two is the evidence, not the absolute number, which depends on how much of the
floor a given camera angle shows.

## Round 40 - the keypoint detector works, and its first test number was leaked

Trained 120 epochs at 640 px with the corrected sigma, the landmark error fell
**281 px -> 18.9 px** (p90 50.6), pose mAP50 0.883. Against the reference
homographies that is about 0.55 ft.

On the broadcast this project actually has to serve -- 2025 Finals game 7,
which appears nowhere in the keypoint dataset:

    registered (both frames of a pair)   94.4%          [gate 90%]
    two independent paths disagree       p50 0.67 ft    [gate 2 ft; painted key 5.8]
    line agreement vs a 5 ft slip        85% of frames  [p < 0.001]

That is the painted key's own consistency test, improved **8.7x**, on video the
model has never seen.

**The dataset test number was leaked and is withdrawn.** The first run reported
100% registered at 0.87 ft p50 with 0 of 114 frames on the wrong end. Roboflow
splits at the frame level, and the source clips are 5-second segments sampled a
few frames apart: **all 107 clips in its test split also appear in train**, so a
"held out" frame is a near-duplicate of a training frame taken a fraction of a
second earlier. That number measured memorisation.

`split_court_keypoints.py` regroups every image by the game it came from and
assigns whole games to each split -- 11 train, 3 valid, 4 test, no game shared
-- so nothing in test shares a camera, an arena, a lighting rig or a possession
with anything in train. It is the rule the screen detector already needed, and
scoring per source video is what showed 92% pooled was 86% and 94% per source.

The first version of this split was itself broken in the same direction. Two
naming conventions are in use, `-q1-01_54-01_48_mp4-` and `-09_49-09_44_mp4-`,
and a regex handling only the first turned every frame of the second into its
own "game", putting adjacent frames back on both sides. It is caught now by an
explicit check that no game appears in two splits.

The broadcast figures above are unaffected -- that game is in no split -- but
they carry their own caveat: both arenas appear in training, so they measure an
unseen game in a seen building, not an unseen court.

## Round 41 - Phase 1 gate, on the rebuilt model

The 640 model was retrained after the original was destroyed (a fine-tune
launched with the same run name and `exist_ok=True` overwrote it in place, and
the only other copy was in a session temp directory). It reproduces exactly --
seed 0, same schedule -- and the weights now live in `checkpoints/`, outside
the scratch directory ultralytics reuses.

    held-out GAMES (4 games, no clip shared with training)
      registered                 137/138 = 99.3%      [gate 90%]
      court error                p50 1.94 ft          [gate 2 ft]
      frames inside the gate     68/138 = 49.3%
      wrong end of the floor     0/137                [painted key: 100% wrong]

    2025 FINALS GAME 7 BROADCAST (in no split of the dataset)
      registered                 33/36 = 91.7%        [gate 90%]
      two paths disagree, fused  p50 0.59 ft          [gate 2; painted key 5.8]
                                 p90 2.01 ft
      line agreement             0.184 vs 0.157 for the same frames 5 ft off

The gate is met on both, but the margins are different and the weaker one
should be stated plainly: **single-frame court error on unseen arenas is
1.94 ft against a 2 ft gate, with only 49.3% of individual frames inside it.**
The comfortable number, 0.59 ft, is the fused one, and fusion is available on
video -- which is what the product consumes -- but not on isolated frames.

The 960 fine-tune that destroyed the first model was also misconfigured, and
that is the more useful half of the lesson: ultralytics defaults to `lr0=0.01`
with warmup, so starting from converged weights re-trains rather than refines.
Pose mAP50 fell 0.707 -> 0.435 over eleven epochs at the higher resolution. Any
retry needs a rate near 0.001.

Validation is noisy enough to be worth ignoring epoch by epoch: across this
run it read 0.707, 0.643, 0.745, 0.587, 0.132 on consecutive samples. The
validation split is 94 frames of whole games, so one awkward camera angle moves
it several tenths. Only the test games and the broadcast checks decide anything.

## Round 42 - the Phase 1 review, and three corrections

An independent review of Phase 1 found defects in the code and in two claims
recorded here. The substantive ones, with what was done:

**`RANSAC_PX = 6.0` was six FEET.** `cv2.findHomography` measures its residual
in the destination space, and that fit runs pixels -> court feet. The module
already had this right for the fused refit (`FUSE_RANSAC_FT`) and wrong for the
main one. At six feet nothing is rejected: on a realistic synthetic fit, 12 of
12 landmarks are inliers at 6.0 against 10 of 12 at 1.0, and on the held-out
games 801 of the 1,565 landmarks used by accepted fits had residuals over half
a foot. "RANSAC discards the bad ones" was not happening, and the
`inliers >= 30` assertion in the tests was vacuous.

Fixing it changed the result **very little** -- per-frame p50 2.03 -> 2.00 ft --
which is itself the useful finding: the error is landmark localisation, not
outlier contamination. The threshold is now selected on validation alongside
the confidence floor.

**The 2 ft gate was reported on a pooled median, and the per-frame median does
not meet it.** `errors.extend(err)` pooled every landmark across every frame,
so a wide-angle frame with twenty visible landmarks outvoted a tight one with
eight. The gate is about frames. Per frame, on held-out games:

    court error, per frame    p50 2.00 ft   p90 2.96 ft   [gate 2 ft]
    court error, pooled       p50 1.91 ft
    frames inside the gate    69/138 = 50.0%

**Phase 1 does not clear its gate on held-out single frames.** It sits exactly
on the boundary. Round 41's "the gate is met on both" was true only under the
pooling choice and is withdrawn. The fused broadcast figure (0.59 ft) stands
and is what the product consumes, but it is a different measurement on
different footage, and the two should not be quoted as if they were one result.

**"No check of this kind can see an end swap" was wrong.** Round 39 argued that
because the court's paint is symmetric about half-court, and the flip composes
with an affine reflection preserving projective structure, an end swap is
undetectable. The paint is symmetric and the projective structure is preserved,
but a reflection is not a rotation: it reverses **orientation**. On all 840
human references the Jacobian determinant of a correct registration is negative
on 100% of frames, and positive on 100% of the same frames after an end swap.
One sign separates them completely. `orientation_sign` now rejects mirrored
registrations inside `homography_from_keypoints`.

What is genuinely undetectable is the 180 degree rotation -- an end AND side
swap, which preserves orientation. That is the reverse-angle camera, and it
needs a temporal or feed-side cue. A test pins that limit so it stays stated.

**Smaller corrections.** `exp(-71)` is about 1.5e-31, not zero in float32; the
gradient is negligible rather than absent, and the conclusion is unchanged but
the wording was wrong. `fuse_registrations` returned `(None, 1)` when the only
usable estimate came from a neighbour rather than the centre -- claiming a
registration while supplying no matrix; it now carries the neighbour's
registration onto the centre frame. The sigma ablation (346/77/263 px) was run
on the leaked Roboflow split, one seed per arm, and should be read as
establishing the mechanism rather than the exact ordering of 0.18 against 0.35.

## Round 43 - the second review, and the schema question settled by paint

A second independent review confirmed Round 42's findings and added a sharper
one about the schema's justification.

**The symmetry check was not independent, and the docstring claiming it was is
withdrawn.** `rectify_keypoint_schema.py` minimises exactly the flip-pair
residuals that `symmetry_error` measures, and `KEYPOINTS` was then written as
exact mirrors by hand -- so it returns approximately zero by construction. The
line calling it "something a fit can never see" was wrong. It is still a real
check on a schema derived some other way (the painted-key bootstrap failed it
at 65 ft) and it still pins the index pairing, but it cannot vouch for the
schema shipped here.

**The reviewer's inference from that was reasonable and turns out to be
backwards.** The fitted schema matches annotators better than the adopted
constants -- leave-one-out 0.20 ft against 0.35 -- which suggests the constants
are the worse of the two. But leave-one-out measures agreement with where
annotators CLICKED, and it is invariant to any projective transform of the
whole schema, so it cannot speak to absolute court accuracy at all.

The question is settled by the one reference outside both schemas: real paint.
Running the same model's landmarks through each schema and scoring against a
mask built from true NBA dimensions, on our broadcast:

    exact constants   line agreement p50 0.1891   (33 of 35 frames)
    the fit           line agreement p50 0.1724   (33 of 35 frames)

The constants put paint on paint better. Both statements are true at once: the
fit reproduces the annotators' click conventions more closely, and the
constants describe the actual court more closely. The product needs court
positions, so the constants stay -- but on this evidence, not on the symmetry
argument, which was circular.

**Also fixed.** The two-estimate "median" in `fuse_registrations` was a mean, so
a wrong neighbour was averaged in rather than outvoted -- the opposite of why a
median was chosen; two estimates now fall back to the centre. The cross-split
check that Round 40 said existed did not; it is now in the script, with tests
on `game_of` and on the shipped splits. `check_schema_loo.py` is checked in,
because the 0.35 ft figure was being quoted from a script that no longer
existed. And the fusion test passed with the carry composition reversed -- the
pans were symmetric about the centre, so the error cancelled; it now also fuses
onto an end frame, which fails if the chain is inverted.

**One defect found here affects every model trained so far.** The sigma
override was applied through a train-start callback, so the LOSS used 0.18 but
the VALIDATOR kept ultralytics' 1/48 -- meaning `best.pt` and `patience` were
selected on the metric the trainer's own docstring calls broken. It belongs in
`data.yaml`, where both read it, and now is. Every checkpoint to date was
selected under the old arrangement.

## Round 44 - Phase 1 passes, at 960 px and on the right denominator

Fine-tuning the 640 model at 960 px with `lr0=0.001` -- the rate the first
attempt got wrong, which re-trained rather than refined -- moved the number
that was short:

    held-out GAMES          640 px      960 fine-tune
      registered            100.0%      97.8%
      court error, frame    2.00 ft     1.62 ft        [gate 2 ft]
      frames inside gate    50.0%       65.2%
      wrong end             0/138       0/135

    BROADCAST, fused        640 px      960 fine-tune
      disagreement p50      0.79 ft     0.45 ft        [gate 2 ft]
      disagreement p90      3.36 ft     1.81 ft

**The 91.7% broadcast registration reported in Round 41 was a small-sample
artefact** and is withdrawn. It came from `--samples 60`, which yields 36 court
frames; at `--samples 120` the same 640 model gives 79.7%. Both models sit near
78-80% by that measure, so it was never a property of the model.

That denominator was also wrong. `has_court` is a wood-fraction heuristic, and
looking at the frames where the detector found no court at all: three of six
were **not court views** -- a player lying on the floor filling the frame, a
courtside close-up where skin and jersey pass the colour test -- two were
**alternate cameras** (the under-basket stanchion cam, a high reverse angle)
that appear in no training set, and one was a genuine miss. Registering those
is neither possible nor wanted, and the painted key's 17% was never measured
that way.

Round 26 scored the painted key by sampling the seconds before each aligned
shot: mid-possession by construction, main camera by convention. On that same
denominator, chosen by the feed and therefore blind to whether registration
succeeded:

    registered during live play   75/80 = 93.8%   [gate 90%; painted key 17%]

**Phase 1 gate: met.** Registration 93.8% on broadcast live play and 97.8% on
held-out games; court error 1.62 ft per frame on unseen games and 0.45 ft fused
on broadcast, both inside 2 ft; consistency 0.45 ft against the painted key's
5.8; wrong end 0 of 135, against 100% of frames.

One caveat carried forward: every checkpoint so far was selected by a validator
still using sigma=1/48 (fixed in `data.yaml` after these runs), so `best.pt`
and early stopping were chosen on a metric the trainer documents as broken.
The models measured here are good despite that, not because of it.

## Round 45 - the feed calibration was measured, cross-checked, and rejected

Registration on broadcast is systematically off across the court, and nothing
internal could see it. `check_registration_bias` slides the registration and
watches where paint brightness peaks; validated on 60 human-annotated frames it
reads +0.00 and -0.02 ft at 2.1x contrast, and on our broadcast it read
**-0.11 ft along the court and +2.04 ft across it**, at 1.2x.

The league's shot chart is the external reference the plan called for: 157
courtside positions for this game that have never seen a pixel of our video.
Pairing them to aligned events by surname gives 149 matches, and the pairing
checks out independently -- the play description states each shot's distance,
and it agrees with the chart's coordinates to **0.3 ft** over 142 shots.

Fitting the offset that best reconciles the feed's locations with detected
player positions is circular if done naively: minimising the distance to the
NEAREST player rewards any offset that pushes points into crowded floor. So the
same fit was run on a deliberately wrong pairing. It could reach only 13.93 ft
where the true pairing reached 5.48, so the signal is real.

    no correction     median miss 6.38 ft
    best offset       dx +3.00  dy +1.75  ->  5.48 ft
    shuffled control  best achievable 13.93 ft

**Then the correction was applied and re-measured with paint, and it made
things worse.** Along the court went -0.11 -> -1.99 ft, exactly what arithmetic
predicts if paint was right that there was nothing to remove; across the court
went +2.04 -> +3.87 rather than the -0.96 the correction implied.

The disagreement is diagnostic. The paint estimate across the court is not
robust: its curve plateaus from +1 to +5 ft instead of peaking, so the location
of the maximum is barely determined -- and the earlier "two independent methods
agree" reading of +2.04 against +3.00 was weaker than it looked. The feed
estimate is weakly constrained too: the residual after correction is centred
(+0.26, +0.59 ft) but spreads by 6.5 ft, so the minimum it selects is shallow,
and it bought only 0.9 ft.

`CALIBRATION_FT` is therefore **zero**, with the rejected value recorded. This
is the failure mode that produced the ICP result -- a score improving 7.60 ->
1.30 ft while true error went 10.2 -> 16.5 -- and the only reason it did not
repeat is that the correction was checked against a method that did not
produce it.

**What the exercise did establish.** The absolute error on broadcast is roughly
5 ft, and it is mostly per-frame rather than a fixed offset, so no single
correction can remove it. That is far worse than the 0.45 ft consistency figure
suggests, and it is the honest number for absolute court position. It bounds
what Phase 4 can claim about shot distance, three-point classification and
court zones. It does NOT bound the tactics, which are relationships between
players in the same frame: a per-frame registration error moves everyone
together, so screen proximity, spacing and matchups survive it. A test pins
that a translation leaves every pairwise distance unchanged.

## Round 46 - free throws give the first trustworthy absolute number on broadcast

Three attempts to measure absolute registration error on broadcast failed their
own controls, and the pattern is worth stating because each looked fine until
the control ran:

- Nearest-bright-ridge along the projected lines: reported 1.84 ft for the
  registration and 1.99 ft for one deliberately slipped 2 ft. With a wide search
  window there is always some bright thing nearby.
- Per-frame paint-brightness optimum: a known +2 ft shift moved the measured
  optimum by only 1.0 ft, so the estimator was about half as sensitive as it
  needed to be and its absolute values were damped.
- Shooter identification by jersey OCR: 3 successful reads in 149 shots. The
  88% figure was measured on curated crops; arbitrary broadcast player boxes
  are mostly players facing away, blurred, or too small.

**Free throws work, because the rules place the player.** A free-throw shooter
stands on the line at (25, 19). Nothing has to be tracked, identified, or
supplied by the feed. Identification is robust rather than circular: the next
nearest player is a measured **7.03 ft** away, so a registration wrong by
several feet still picks the right man -- a selection that only worked when the
answer was already right would prove nothing, and this one does not need that.

Over 26 usable free throws:

    offset across the court   -0.57 ft  +/- 2.31
    offset along the court    +0.13 ft  +/- 2.79
    distance from the line    p50 3.56 ft   p90 4.83 ft

**The systematic bias is not there.** Both offsets are within about half a foot
of zero, which contradicts the +2.04 ft (paint) and +3.00 ft (shot chart)
estimates that motivated the calibration in Round 45. Rejecting that
calibration was right, and for a better reason than was available at the time:
not merely that the two disagreed, but that a third measurement with a sound
control says there was nothing to correct.

What remains is per-frame scatter of roughly 2.3-2.8 ft. The 3.56 ft distance
is an upper bound -- it also contains the shooter standing a foot behind the
line, the detector box's bottom edge as a proxy for feet, and a 1.2 s lead on
the event -- so registration error alone is smaller than that.

This replaces the "about 5 ft" of Round 45, which came from matching shots to
the nearest of ten players and was inflated by the association guess.

## Round 47 - the two sigmas want opposite things, and fixing one broke the other

Round 43 recorded a defect: the sigma override was applied through a train-start
callback, so the loss used 0.18 while the validator kept ultralytics' 1/48, and
`best.pt` was therefore selected on the metric the trainer calls broken. Moving
it into `data.yaml`, which both read, looked like the obvious fix.

It made the model much worse:

                              val sigma 1/48    val sigma 0.18
      registered              97.8%             50.0%
      court error, per frame  1.62 ft           2.04 ft
      frames inside the gate  65.2%             22.5%

The cause is visible in the training curve. With sigma 0.18 the validation
metric reached **0.9907 at epoch 2** and never beat it, so patience fired at 32
and the shipped weights were a **two-epoch model**.

The two sigmas serve opposite purposes and should not share a value:

- The **loss** needs a loose sigma. At 1/48 the OKS loss saturates at
  initialization and the keypoint head receives no gradient at all -- 281 px
  landmark error after two epochs while box mAP50 read 0.995.
- The **validator** needs a tight one. At 0.18 every reasonable model scores
  about 0.99, so the metric cannot tell a good localiser from a much better
  one, and checkpoint selection becomes noise.

The original arrangement -- loose in the loss via the callback, tight in
validation by default -- was accidentally correct. The review was right that
the inconsistency was undocumented and unintentional; the conclusion that it
should be made consistent was wrong, and only measuring it showed that.

`data.yaml` now carries a comment saying why the key is absent, since its
absence is the load-bearing part and an obvious-looking edit would undo it.
`court_kp_960_ft.pt` remains the production model.

## Round 48 - toward 0.3 ft: dense line refinement, and what real footage says about it

Target raised to 0.3 ft on an unseen arena. The landmark fit is limited by how
precisely a network can point at a blob; painted lines are thin ridges whose
centres can be found to a fraction of a pixel along hundreds of feet of paint.
`court_refine.py` aligns every visible line (point-to-line, sub-pixel ridge
centres, orientation and continuity checks, robust loss, landmark prior), and
`check_line_refinement.py` measures it on **lines the fit never saw** -- in court
feet, with no annotations, which matters because the annotations are only
consistent to ~0.35 ft and cannot certify anything finer.

**The measurement is sound where three before it were not.** A known 0.5 ft shift
reads 0.499 ft (p10 0.465, p90 0.543) on real footage. It also had a bias, found
and fixed: a held-out family whose paint was not found near the refined fit was
dropped, so a fit locked one line spacing (3 ft, ~100 px) off -- unmeasurable in a
24 px window -- vanished from the statistics instead of counting against them.
Re-scored, the 9-start variant had been silently dropping **14%** of its
measurements.

**Three designs, on the OKC calibration game.** Synthetic tests flattered every
one; real frames decided.

    design          accepted  not found  held-out p50  p90     n
    9-start grid    80%       14%        0.40 ft       1.09    32   (small sample)
    consensus       60%        7%        0.36 ft       2.05    25   (small sample)
    single start    60%       14%        0.35 ft       1.42    64   (28 frames)
    landmark only                        0.92 ft       2.00

- **Multi-start** found aliases rather than the answer: 6 ft starts landed ~3 ft
  from the landmark-started fit in 17 of 36 cases, and the acceptance guard
  passed 58% of deliberately 20-30 ft wrong starts at a threshold that kept 80%
  of true fits.
- **Curves first** failed on geometry: curves are not isolated. The lane lines
  run 2 ft outside the free-throw circle and the free-throw line is its
  diameter, with matching orientation at exactly those points.
- **Segment consensus** (each straight segment picks its paint as a unit) fixed
  the synthetic aliasing and left real capture unchanged: 11 of 27 six-foot
  starts still landed elsewhere.

**The basin hypothesis was tested and is wrong.** If refinement were limited by
starting in the wrong basin, error would be bimodal -- near-perfect or one line
spacing off -- and track the landmark start. Over 74 measurements it is a
continuum (25 at or under 0.3 ft, 20 at 0.3-0.6, 10 at 0.6-1.2, 8 at 1.2-2.4,
1 above), and barely related to the start (correlation +0.24; even starts under
0.75 ft reach 0.3 ft only half the time). Temporal propagation, which fixes
basins, would not fix this, and was not built.

**Current lead: the camera model.** The fit's residual is 0.56 px (~0.02-0.05 ft)
yet held-out lines sit 0.35 ft off, so no single homography fits the lines it saw
and the lines it did not. The worst family is the boundary, 0.59 ft -- the lines
at the frame edges. Both point to radial lens distortion, which a homography
cannot represent. Being tested directly next: residuals should track
rho^2 * n.(p - c) with a consistent sign across frames, and grow toward the edge.

## Round 49 - three leads ruled out, and a threshold that buys accuracy

Held-out error on the OKC calibration game is a continuum, not basin jumps
(Round 48). Three explanations for it were tested and ruled out:

- **Radial lens distortion.** Distortion moves points by k1*rho^2*(p - c), so
  post-fit residuals should track rho^2 * n.(p - c). Over 2,028 line samples in
  10 frames: per-frame k1 median +0.0006 with its IQR straddling zero, the same
  sign in only 60% of frames, pooled R^2 = -0.009, and residuals no larger at
  the frame edge (0.37 px) than the centre (0.46 px).
- **Temporal fusion of refined fits.** Paired on the same centre frames and
  held-out families, fused better in 3 of 6, median change -0.05 ft. Small
  sample, but no sign of a lever -- consistent with every earlier fusion test.
- **Split-half agreement** was attempted to estimate the all-lines fit without
  holding out a whole family, and is inconclusive: an even/odd split of line
  families left one half unfittable in 18 of 22 frames, so 4 were compared.

**The acceptance guard does predict accuracy** -- the synthetic-versus-wrong-start
calibration had understated it. On lines the fit never saw, Spearman -0.49
between peak ratio and error, -0.55 for paint explained. The threshold was set
by a rule fixed before the unseen arena was scored: the smallest value whose
accepted fits reach a held-out median of 0.30 ft, lost lines counted as
failures.

    threshold   frames accepted   held-out p50   p90     within 0.3 ft
    2           74%               0.36 ft        1.27    43%
    5 (chosen)  45%               0.25 ft        0.67    55%
    10          19%               0.15 ft        0.56    73%

The trade is stated plainly: on the calibration game, a frame the refinement
accepts meets 0.3 ft at the median, and fewer than half of frames are accepted.

## Round 50 - the unseen arena failed, and the review found the evaluator biased

**Toyota Center accepted 0 of 37 registered frames.** The refinement was
developed on OKC, whose lines are white, and it looked only for thin BRIGHT
ridges. Toyota Center paints its arc and circles BLACK, and its lane is a solid
red key with no line on the boundary at all -- the lane line is only the edge
between red paint and wood, and the sidelines are the same against red
out-of-bounds paint. Resolution was ruled out first (854x480 there against
1280x720; upscaling changed nothing, 0 of 15 either way). The design had
considered dark lines and rejected them over wood plank seams; that assumption
did not survive a second arena. `PAINT_POLARITY` now offers "bright", "both"
(adds dark ridges) and "all" (also treats a step edge between painted regions
as a line, weighted below ridges so a white line's centre still beats its
flanks). A synthetic floor built like Toyota Center's pins both the fix and the
failure. **Toyota Center is no longer a clean unseen arena** -- it was
diagnosed on -- and the other two games on disk are arenas in training.

**An independent review found the measurement biased optimistically, twice:**

- The held-out measurement inherited segment consensus, which restricts each
  segment to a band around its mean offset. At the ends of a slightly rotated
  long line the band misses the paint and those samples vanish -- the ones
  carrying the most error. On a synthetic line truly 0.308 ft off at 1 degree
  it read 0.204. Measurement now runs without consensus, pinned by a test.
- A held-out family whose REFIT was refused was skipped before being counted,
  so it sat in neither the failures nor the denominator. Now counted.

Two reporting errors as well. Round 49's p90 column counted failures in the
median but not the tail: the true p90s were 2.22, 0.89 and 0.75 ft, not 1.27,
0.67 and 0.56. And the threshold's provenance is weaker than stated -- the rule
was given before the selection was run, but it was committed together with the
value, six minutes after the dump it read, by code never checked in, so "the
rule came first" cannot be verified from the repository.

So every OKC figure since Round 48 was measured with a biased instrument and is
superseded, and the threshold is void. `scripts/select_refinement_threshold.py`
is now the rule, checked in before any recalibration: it counts lost and
refused as failures in every statistic, requires dumps gated at or below the
lowest candidate (refused refits and their ratios are now recorded, so each
candidate is applied to frames AND refits exactly; the old evaluator's skipped
refits are what made its dumps unable to do this), refuses the unseen arena,
and refuses dumps without provenance.
Dumps now record video, arguments, commit, threshold, polarity, the control
values and every registered frame. Also fixed: a 1e-3 floor under the
sharpness ratio's denominator, which made any partial lock with empty 2 ft
neighbours score about 1000x its coverage (rates are now add-one smoothed on
counts); hypotheses compared on explained paint must now see at least 70% as
much court as the best-seeing one; a control that does not run is reported as
a failure; and the evaluator's default video, which pointed at the held-out
arena, is gone.

## Round 51 - honest numbers, and unseen arenas that were on disk all along

Re-measured on the OKC calibration game with the unbiased evaluator (dumps made
at threshold 0, every failure counted), paint polarity "bright":

    landmark only                                     p50 1.02 ft
    refined, lost/refused held-out lines as failures  p50 0.60-0.67 ft
    refined, measured lines only                      p50 0.47 ft, p90 1.45, 34% within 0.3

No candidate threshold meets 0.30 ft under the declared rule, and "all"
polarity is worse on OKC (conservative p50 about 1.05). **The 0.25 ft of Round
49 was an artifact of the biased evaluator.** Refinement is real -- better than
its landmark start on 73% of measured lines -- and about a factor of two short
of the target on the arena it was developed on.

Neither accounting is the production accuracy. A held-out refit lacks a whole
line family and is weaker than the all-lines fit, and a high threshold refuses
exactly those refits, so counting them as failures is pessimistic and grows
more so as the threshold rises; measuring only the successes is optimistic.

The error is not uniform. Measured-only, interior features are at or near the
target -- free-throw circles 0.20-0.29 ft, corners 0.20, far lane 0.34 -- while
the boundary family, the long lines at the edge of the frame where a fit without
them extrapolates, is 0.97 ft. Frames with slower camera pans measure 0.41 ft
and faster ones 0.93 ft (Spearman +0.26 over 33 frames): suggestive that motion
matters, but pans are small (p50 0.7, p90 3.4 px per frame), too small for
rolling-shutter skew alone to produce 30 px errors. Inconclusive.

**Unseen arenas were already on disk.** The keypoint dataset's by-game test
split holds four games the landmark model never trained on, and by the 2025
playoff schedule (inferred, not recorded in the data) three are at arenas in
no split at all -- TD Garden, Kaseya Center, Fiserv Forum -- each with human
annotations. The refinement was never tuned on them. Declared before running:
polarity "bright" and threshold 3.0 (the OKC fallback rule: no candidate met
0.30, so the one minimising OKC's conservative median), scored as the median
distance to the annotation-fitted registration over annotated landmarks, with
refused frames falling back to their landmark registration. This scores the
production fit, every line used, against something the refinement never
produced -- which the held-out-line protocol cannot. Its limit: annotators
agree with the schema to about 0.35 ft, so errors near that cannot be resolved.
This replaces the plan to download new footage.

## Round 52 - the unseen arenas: 0.3 ft met at one of three

The declared evaluation (Round 51) ran as declared -- the script was committed
at bb6ca7a and began in the same second, and is unchanged since. On 115
annotated 1080p stills from three arenas in no split, refinement accepted 92%
of registered frames and agreed with the annotation-fitted registration to a
per-frame median of **0.23 ft** (p90 0.50, 64% within 0.3), against 1.65 ft for
the landmark registration alone. An independent review reproduced every number,
confirmed the arenas by looking at the floors (TD Garden's parquet, "fiserv.forum"
painted on the court, "Kaseya Center / Pat Riley Court"), and found none of them
in the training or validation splits.

**The pooled figure hides the answer.** Per arena:

    arena            images   refined p50   p90    within 0.3
    TD Garden         58       0.18 ft      0.36    84%
    Fiserv Forum      42       0.35 ft      0.56    40%
    Kaseya Center      5       0.33 ft      0.40    20%
    Target Center     19       0.23 ft      0.50    68%   (seen arena)

TD Garden supplies over half the unseen frames and carries the pool. **The
target is met at one of three unseen arenas**, and missed narrowly at the other
two. The review's wording is adopted as the claim: refinement agrees with the
annotation-fitted registration to 0.23 ft pooled at the annotated interior
landmarks, against a reference whose own median noise is about 0.14 ft; per
arena 0.18, 0.35 and 0.33 ft; not met at two of three arenas; silent on
boundary lines and on 720p video, where the held-out-line protocol on OKC reads
0.47 ft on measured lines.

**The reference was checked and is not the cause.** Fitting each image's
annotations in two halves gives a full-reference noise of about 0.13 ft
(0.14 by the review's bootstrap), so the result sits at the reference's
resolution: nothing below roughly 0.15-0.2 ft can be certified by it. The
baskets were suspected of bending the reference -- annotators click a rim 10 ft
above the floor, which no floor homography can place -- and refitting on floor
points only changes the pooled median from 0.23 to 0.24 ft and Fiserv's from
0.32 to 0.31. What does explain Fiserv: its largest errors are at half-court
(1.26 ft), centre court (0.80), the far sideline hash (0.52) and the baseline
corner (0.46) -- away from the key, where the fit extrapolates from the lines
it aligned. That is the same pattern as OKC's boundary family (0.97 ft), and it
is the next thing to fix.

**Paint evidence is now chosen per frame.** Neither polarity won everywhere on
the test arenas -- bright at TD Garden (0.18 against 0.20) and Target Center
(0.23 against 0.30), all at Fiserv (0.29 against 0.35) -- so `register_frame`
refines with both and keeps the sharper fit. That rule was chosen after those
per-polarity results were seen, and was committed (54199ed) before its own test
result was computed: pooled 0.23 ft, p90 0.44 (from 0.50), Fiserv 0.32. It is
weaker evidence than a first look and has to hold on calibration footage too.

**A leak the review found.** The 960 px landmark model was adopted over the
640 px one on its score on these same test games (1.62 against 2.00 ft), so the
model supplying every refinement's starting registration was selected on this
split. It inflates the landmark-only figure more than the refined one, but it
also decides which frames start inside the refinement's capture range.

**A process failure of my own.** 54199ed was committed with its tests never
having run: the test module failed to import, pytest aborted during collection
with "1 error", and the guard searched the output for "failed". Commits are now
gated on pytest's exit code. With the import fixed the tests pass.

**Defects fixed (c19daf1).** Sharpness compared the solution and its 2 ft
neighbours on different sample sets, so a shift that pushed samples off-frame
could refuse a correct fit; they are now compared on common samples.
Hypotheses were compared with a visibility floor that could disqualify the
right basin on sample count; they are now compared on the court every one of
them sees. The fallback that chose threshold 3.0 lived only in prose; it is in
the selection script and reproduces 3.0 from the OKC dump. Because the ratio
was redefined, 3.0 is provisional until recalibrated on a fresh OKC dump, which
is running, as is a re-run of the unseen-arena evaluation with the current code
-- reported as a re-run, since the test set has now informed design choices.

## Round 53 - the review's fixes, re-measured: acceptance up, per-arena picture unchanged

Threshold re-derived by the checked-in rule on a fresh OKC dump made with the
common-sample sharpness ratio: 79% of registered frames accepted (70% before
the fix, which had been refusing correct fits), control 0.497 ft for a known
0.5, no candidate reaching a 0.30 ft held-out median, fallback to the lowest
(0.54 ft, tied between 2.0 and 5.0, ties to the lower): MIN_PEAK_RATIO 2.0.

The unseen-arena evaluation re-run with the current code (6d302e4, clean tree)
-- a re-run, not a first look, since the test images have informed design since
the declared run:

    arena          bright @ 2.0    per-frame rule @ 2.0    within 0.3 ft
    TD Garden      0.19 ft         0.21 ft                 82%
    Fiserv Forum   0.34 ft         0.33 ft                 44%
    Kaseya (n=7)   0.35 ft         0.31 ft                 29%
    unseen pooled, with fallback   0.25 ft, p90 0.59       63%

The fixes raised acceptance -- Kaseya now 100%, fewer fallbacks -- and trimmed
the tail (pooled p90 0.66 to 0.59 under the per-frame rule). They did not move
the per-arena picture: TD Garden meets 0.3 ft, Fiserv and Kaseya miss it by a
few hundredths. Subtracting the reference's ~0.13 ft of noise would nominally
bring Fiserv to about 0.30; that is not claimed, because the subtraction
assumes independence the data cannot confirm. Next: whether the error at the
periphery is missing support or the fit actively bending the far side.

## Round 54 - the remaining error is evidence the refinement throws away

Diagnosis on the test images (none of it re-reported as an unseen-arena
result). Every annotated floor point was binned by its distance, in court feet,
from the nearest line sample the fit actually used:

    distance from used paint   points   refined p50   landmark p50   refined worse
    0-3 ft                     1,179    0.21 ft       1.68 ft        3%
    3-6 ft                       233    0.30 ft       1.79 ft        9%
    6-12 ft                      104    0.45 ft       2.06 ft       11%
    12+ ft                        25    1.16 ft       1.66 ft       40%

The fit is excellent near the paint it rests on and degrades smoothly away from
it. It is **missing support, not bending** the far side: the landmarks are far
worse than even the extrapolated fit out to 12 ft, so fitting jointly with them
would hurt, and was not built. Fiserv has a second gap as well: 0.26 ft even
within 3 ft of paint, against TD Garden's 0.18.

**Where players stand.** Both registrations are homographies, so their
disagreement is defined at detected feet. Restricted to feet within 6 ft of an
annotated point, where the human reference is itself well supported: TD Garden
0.19 ft (77% within 0.3), Fiserv 0.27 (57%), Kaseya 0.36 (34%, from 29 feet in
7 images), all arenas 0.22 ft, p90 0.57. Players stand near paint -- 88-94% of
feet are within 6 ft of used lines -- so this is the product-relevant figure,
and it is reported beside the landmark metric, not instead of it.

**Why the far side lacks support.** For annotated points more than 6 ft from
used paint, painted lines lie within 6 ft of them almost always (open wood
0-5%). A first version of this diagnostic called 71-100% of those lines
"visible but missed" -- but it judged "missed" with a 3 px window under the
refined fit, which cannot tell a line the detector cannot find from a line the
fit places a few pixels off. Re-searched with wider windows, **83-91% of that
paint is found within 3-12 px**, and only 3-17% is not found even at 24 px. The
paint is there and detectable; the fit is simply off there, by 0.23-0.62 ft.

The refinement discards that evidence by construction. The fit locks onto the
key first; each pass narrows the window, and by the 6 and 3 px passes far paint
3-12 px from the prediction is out of reach, so those samples drop, the far
side is never corrected, and it stays off because it is ignored. The fix --
support expansion, re-observing at 12 px (below the ~3 ft spacing of parallel
lines everywhere in a broadcast frame) and letting newly found samples join the
fit until none do -- is being validated on the OKC calibration game before it
touches the test arenas.

## Round 55 - convergence fixed; the far side is still the weak point, and the guard cannot see it

**The expansion regression and its real cause.** Support expansion (Round 54)
made the painted Toyota Center floor worse. Three fixes were tried on a wrong
premise before an ablation isolated the cause: the refit after expansion ran at
the wide-pass robust scale and never converged. The fix (efae531) iterates the
final pass to convergence (12 iterations, 0.05 px tolerance), expands only on
straight segments, and refits at 0.75 px. On the Toyota Center dev footage,
"all" polarity, held-out lines, old to new code:

    conservative (failures infinite)   3.21 -> 1.52 ft
    measured only                      1.33 -> 0.70 ft
    interior families, measured        0.58 -> 0.19 ft
    boundary family, measured          2.14 ft

Interior failures there: 14% refit refused, 25% paint not found (far arcs in
half-court views). OKC (v3): 83% of frames accepted, control 0.497 ft; interior
measured 0.30 / conservative 0.45 ft, boundary measured 0.63 ft.

**The threshold rule picked 10.0 on OKC and it was not adopted.** Bootstrapped
over frames, 10.0 is chosen 56% of the time, keeps 17% of frames, and has only
a 59% chance of a median at or under 0.30 ft (2.0: 74% of frames at p50 0.57;
3.0: 62% at 0.39). On the unseen arenas it collapses coverage -- at 10.0 TD
Garden is refined on 0% of images and reads 1.72 ft with fallback. Production
stays at 2.0; the coverage-for-accuracy trade is the user's decision, not the
rule's.

**Unseen arenas, current code** (efae531 plus the uncommitted provenance change;
a re-run, weaker evidence than the declared one), per-frame polarity rule,
fallback to landmarks counted:

    arena            n    refined   p50      p90      within 0.3 ft
    TD Garden        63   100%      0.19 ft  0.32     87%
    Fiserv Forum     44    98%      0.29 ft  0.48     55%
    Kaseya            7   100%      0.34 ft  0.52     29%
    Target Center    20    95%      0.24 ft  0.50     60%   (seen arena)

Two of three unseen arenas meet 0.3 ft at the median; Kaseya misses on seven
images. Thresholds 2.0 and 3.0 are identical here; 5.0 and 10.0 lose coverage.

**Provenance.** Runs from a worktree recorded the main repository's HEAD. Dumps
now record the commit of the imported module's own repository (d5826de);
`hou_all_old.json` is annotated as having actually run 6d302e4.

**The boundary image.** A frame at t=3772.9 s shown to the user drew the court
boundary as a Λ across the floor. My explanations went wrong twice
("measurement contamination", then "corners behind the camera"); both were
refuted on the exact frame -- all four court corners project in front of the
camera. The picture is the *held-out test refit*, fitted with every boundary
line hidden, and on this frame it genuinely extrapolates the far sideline
diagonally across the floor. The production fit on the same frame, with every
line available, lies along the red band. The earlier claim that bad boundary
numbers were "mostly the measuring stick" was wrong for this frame; band edges
and wordmarks do contaminate some others.

This is not a drawing artifact to explain away: the held-out refit is exactly
production on a frame where the boundary paint is not found (occluded, under a
band edge, off screen). What it shows is that **with support only near one
basket, the fit misplaces the far side by feet, and the acceptance guard does
not notice**. Of accepted held-out refits, 38% are more than 1 ft off on Toyota
Center (13 of 14 in the boundary family) and 16% on OKC. Their peak ratios
overlap the good fits' almost entirely (Toyota bad p50 4.1, 10-90% 2.2-12.4;
good p50 7.3, 2.9-19.2), as do their drifts. Sharpness at the paint the fit
rests on says nothing about paint it never saw.

Open: a far-field guard or constraint that does not depend on the paint the
fit already used -- e.g. refusing or flagging frames whose used samples span
too little of the court, or requiring that far lines predicted in view are
found near where the fit puts them.

## Round 56 - a camera that does not move: the gate passes on all three unseen arenas

**The trust radius (built first).** A registration now reports the line samples
its fit rests on (`info["support"]`) and asserts only court points within
TRUST_RADIUS_FT of them (199e28c). The radius rule and the per-arena test were
committed before any run. That check alone could not rescue the video footage.

**What the dev footage showed next.** On Toyota Center the PRODUCTION fit, not
only the held-out one, put the far sideline on the bottom edge of the LED ad
board and the near sideline on the top of the score graphic, squeezing the
court between them with a sharp peak ratio. A trust radius cannot catch that:
the fit rests on the wrong paint and trusts it. Admitting the boundary only
after an interior-first fit fixed one frame and broke another (the key slid
off the red paint at t=4963); it was reverted.

**The fix: one camera centre per game.** The main broadcast camera turns and
zooms but does not move, so every frame is K(f) R [r1 r2 -RC] with one C per
game: four parameters per frame instead of eight. Neither a squeezed court nor
a diagonal far sideline is a pan/tilt/zoom of a fixed camera. Checked on the
human annotations before building (18 games, split-half: C solved jointly on
half of each game's frames, the other half fitted with it fixed): 0.20 ft
median (p90 0.32) against 0.17 ft for a free homography. The median of per-frame
decompositions is a poor estimate (0.57 ft; focal length and depth trade off
along the ray), so C is solved jointly with outlier frames dropped
(court_camera.py, 2ab377f). Estimated from a game's own free fits with no
annotations, MSG's centre lands within ~3 ft of the one its annotations give,
and OKC's regular-season and Finals games agree to 0.7 ft.

**Held-out lines on video, fixed camera** (camera from a separate pass over
each game, sample grid offset from the evaluator's):

    footage                   before (free)   camera   interior families
    OKC (1280x720)            0.56 ft         0.35     0.07-0.19
    Toyota Center (854x480)   1.91            0.48     0.14-0.18
    Finals at Paycom (720p)   1.28            0.43
    MSG (720p)                --              0.41     arcs/circles 0.11-0.22

The remaining held-out error is concentrated in families whose MEASUREMENT is
contaminated, confirmed by eye: at Toyota Center the boundary window reaches
the ad-board bottom and the score-graphic edge (the camera fit's boundary lies
on the wood/red edge); at MSG the lane window finds a white stripe ~2 ft inside
the blue key (the fit's lane lies on the key edge). These are not excluded from
the numbers above. Acceptance fell on Toyota Center (89% to 61% of registered
frames): the camera model refuses what it cannot explain, including t=3773.

**Radius selected on OKC** (33030df): every candidate met the 0.25 ft
calibration target; the largest, 12 ft, was taken -- p50 0.24 ft, 79% of
visible held-out paint and 79% of feet trusted. With the camera, error barely
grows with distance from paint.

**The gate: unseen arenas, human annotations** (declared at fe0c8bb, camera
centre per image from the free fits of the game's OTHER test images, no
annotations; per arena, over annotated floor points):

    arena                 images  refined  trusted  trusted p50  p75    within 0.3
    TD Garden              63      98%      99%      0.17 ft     0.27   79%   PASS
    Fiserv Forum           44      98%      96%      0.20 ft     0.32   71%   PASS
    Kaseya Center           7     100%     100%      0.26 ft     0.38   60%   PASS
    Target Center (seen)   20      95%      94%      0.22 ft     0.34   68%

Free homography, same script and radius: TD Garden 0.20, Fiserv 0.26, Kaseya
0.35 (FAIL). The camera model moves every arena and is what carries Kaseya.

What this does and does not show. The claim is a median: 21-40% of points are
still worse than 0.3 ft. Kaseya is 7 images. These test images have informed
design since Round 51, so this is a re-run on known images -- but the camera
arm and the radius were both fixed before either touched them. YouTube
footage at 720p and 480p reads worse than these 1080p stills on held-out lines;
a resolution study on the valid split is running. An independent review of the
gate (per the plan) is in progress; the gate is not declared passed until it
reports.

## Round 57 - the review: Round 56's gate did not stand, and the honest re-run

An independent review of Round 56, confirmed here, found three defects:

1. `estimate_centre` kept every frame whenever dropping its outliers would
   leave fewer than MIN_FRAMES, and reported them all as inliers. Kaseya's
   centres came from six frames, four over the 3 px limit, reported 6/6.
2. Kaseya's seven images are ONE 5-second clip; its "leave-one-out" centre was
   solved from six near-duplicates of the frame being scored.
3. The trust-radius rule took the largest passing radius outright. On OKC 4, 6
   and 8 ft FAILED (0.27, 0.27, 0.26) and 12 ft passed only by composition,
   while ground 8-12 ft from support was ~0.47 ft off. Round 56 and commit
   33030df said every candidate qualified and that error barely grows with
   distance; both were false.

Also: a WRONG centre is not refused (with another arena's centre, 4 of 7 Kaseya
frames were accepted at 1.0-1.7 ft) -- `FixedCamera.explains` is not used
anywhere yet -- and the split-half check that justified the camera model
included the four test games.

Fixed in 732a87f, declared before re-running: the outlier rule always fires;
the trust rule is monotone (every radius up to the chosen one must pass),
giving TRUST_RADIUS_FT = 3; the camera comes only from OTHER CLIPS of a game;
and the valid split's two arenas in no training game are scored too. Per
arena, trusted points (within 3 ft of used paint), per-clip medians shown:

    arena                split  camera    trusted p50  within 0.3  clips<=0.3  feet trusted
    TD Garden            test   63/63     0.16 ft      84%         10/10       46%   PASS
    Fiserv Forum         test   44/45     0.20         74%          6/7        38%   PASS
    Kaseya Center        test    0/7      0.32         43%          0/1        38%   FAIL
    Crypto.com Arena     valid   5/14     0.25         58%          2/3        39%   PASS
    Toyota Center        valid   0/13     0.17         85%          2/2        37%   PASS
    (seen) Target Ctr    test   12/23     0.20         68%          3/3        37%
    (seen) MSG           valid  59/67     0.16         84%         10/10       30%

Free homography, same code: TD 0.18, Fiserv 0.23, Kaseya 0.32, Crypto.com
0.25, Toyota 0.17, MSG 0.23. The camera helps where a game has enough clips to
solve a centre honestly (TD, Fiserv, MSG); on one- to three-clip games it
mostly cannot be applied.

**Four of five unseen arenas pass; Kaseya fails at 0.32 ft.** Kaseya is one
clip -- a single measurement, 0.02 ft over, inside the reference's own ~0.13
ft noise -- so it neither proves nor disproves the arena; but the gate as set
("every unseen arena") is not met, and it is not reported as met.

The cost of honesty is coverage: at 3 ft, 30-46% of players' feet are
trusted. The rest are flagged; their error is 0.22-0.42 ft (untrusted p50),
not the multi-foot extrapolation of Round 55, but not certified either.

## Round 58 - a whole game: the stills did not transfer

The user's standard: registration must hold across a WHOLE game, not short
clips. The only whole game on disk at an arena in no training game is Toyota
Center (854x480). Frames were sampled every 45 s across the entire broadcast
(84 court frames) and landmarks hand-placed blind in a labeller that shows no
registration (data/labeling/court_toyota; test declared at c7eb857 before any
label existed). The user labelled 32 frames covering the first ~47 minutes
(video 671-3506 s).

**The labels needed repair, from the labels alone.** The first scoring read
~36 ft: the reference itself was 6 ft inconsistent. Landmark names were only
on hover and some clicks were attached to the wrong landmark (the arc apex
placed on a sideline). The reference is now fitted by RANSAC over the labels
only, frames need 8 inliers and a leave-one-out residual under 0.5 ft, and the
court's end/side convention is aligned by symmetry (5ff17f6, committed before
any registration was scored against it). 11 of 32 frames survive, 100 of 120
points; the labels' own leave-one-out noise is 0.43 ft per point.

**Result: FAIL.**

    arm      frames refined   trusted p50   all points p50   frames <= 0.3
    free       7/11           0.61 ft       0.84 ft          14%
    camera     9/11           0.57          0.66              0%

Drawn over the frames, the worst are not label noise: the labels' own fit
sits on the painted key and arc, while the production fit slid the key along
the floor by 1.6-3.5 ft and was ACCEPTED (t=3461, 1751, 1076). The fixed
camera cannot forbid it -- a slide along the court is a pan -- and the peak
ratio does not see it. The 1080p annotated stills (0.16-0.25 ft) did not
predict this: 480p video with a painted floor is a harder input, and whole-game
sampling includes the views short clips do not.

Next: temporal fusion (court_fusion.py). A lock belongs to one frame's paint
and start; carried by frame-to-frame tracking, the window's median should
outvote it. Settings fixed before running on the labels: +/-2 s at 5 fps,
>= 3 candidates.

**Temporal fusion does not rescue it** (7dc5e1a, settings fixed beforehand):

    arm      frames refined   trusted p50   all points p50   frames <= 0.3
    fused     10/11           0.73 ft       0.73 ft           0%

Per frame it helps the worst (t=3461 3.45 -> 1.23 ft, t=1751 2.18 -> 1.73)
and hurts others (t=761 0.68 -> 1.32). The locks are not one-frame accidents:
at t=1166 the window's candidates agree to 0.22 ft and are 1.0 ft wrong
together; elsewhere they disagree by 1.3-2.2 ft, so a median has nothing to
recover. On this floor at this resolution the error is systematic.

Two limits on the test itself. Even the best frames score 0.35-0.5 ft, and the
labels' own leave-one-out noise is 0.43 ft per point -- at 480p a hand click is
about a pixel, a third of a foot on the far side -- so this reference cannot
certify 0.3 ft even for a perfect registration. And 11 frames is thin.

**The same game at 1080p** (downloaded with the user's permission; labels
carried over by image matching, 35 of 38 frames transferred, 3 dropped for a
weak match; camera centre solved again, 53 of 112 frames agreeing, 1 ft from
the 480p centre):

    arm      frames refined   trusted p50   all points p50   frames <= 0.3
    free      10/11           0.72 ft       0.89 ft           0%
    camera    10/11           0.49          0.56              0%

Resolution removes the gross locks: t=3461 3.45 -> 0.94 ft, t=1751 2.18 ->
0.72, t=1076 1.64 -> 0.31. Drawn over the frames, the camera fit now lies on
the painted key edges and the black arc, nearly coinciding with the labels' own
fit; what remains is carried by individual clicks -- a sideline hash scored 4.2
ft, a far-sideline point 2.9 ft, free-throw circle crowns 1.1-1.4 ft.

The labels cannot resolve 0.3 ft. They were placed at 480p (one pixel is 2.25
at 1080p) with ~10 points per frame against an 8-parameter homography, and
many landmarks are tangent points or unmarked spots. From the labels alone,
those sit 0.73 ft from the rest of their frame against 0.52 for line
intersections; but restricting the reference to intersections leaves 6 frames
with 8 points and none consistent enough to use. The whole-game test is
therefore INCONCLUSIVE at 1080p, not passed: the registration is visibly on
the paint, and the reference is too coarse to measure whether that is 0.2 ft
or 0.5.

## Round 59 - a whole game needs starts and a guard, not only accuracy

Drawing the fixed-camera fit over all 84 sampled frames of the 1080p Toyota
Center game showed two problems larger than 0.3 vs 0.5 ft:

- **Coverage.** 31 of 84 frames got no landmark start. Many are not court
  views (close-ups, crowd, ads, overhead rim cameras, wipes), but ~20 are
  ordinary game views -- mostly centre-court views showing the R logo and few
  key corners. With no start there is no registration at all.
- **Other games.** Two halftime highlights from other arenas (a Clippers game,
  a Pelicans game) were registered with this game's camera and accepted.

**Landmark-free starts (182060b).** With the centre fixed a frame is four
numbers -- where the camera looks, its zoom, a small roll -- so they can be
searched. Three versions:

1. Paint hits within a 16 px window: top starts 17-50 ft off. The first
   `look_at` also built an upside-down camera (crossed with world-down); the
   synthetic test shared the mistake, so it is now checked against real fits
   (0.1-0.5 degrees).
2. A smooth chamfer score to the painted ridges: worse, 44-72 ft. The
   strongest ridges in a broadcast frame are the score graphic, ad boards and
   crowd; the true pose scored 89 against 260 for a pose throwing the court
   into the stands.
3. **Overlap of the projected court with the floor mask** (wood or court
   paint, players filled in, lanes in or out): top start 0.4-0.8 ft from the
   landmark-started fit on two of three real frames, in under a second. The
   refinement finishes; within one kind of evidence the fit resting on the
   most paint wins, landmark start or searched.

**The camera check.** Every accepted camera fit is re-fitted without the
camera; if this game's camera cannot reproduce the free fit within 3 px, the
frame is refused. On 17 real frames:

    frames                                 before          now
    ordinary views, no landmark start      0 / 12 fitted   8 / 12 fitted, all on the paint by eye
    halftime highlights, other arenas      2 / 2 accepted  0 / 2 (free fit 7.9-11.5 px from the camera)
    normal frames                          3 / 3           3 / 3 (check 1.2-1.5 px)

Four ordinary views stay refused (a centre-court jump ball, a blurred pan, two
centre-court views); one of them only by the check, at 3.0 px against the
3.0 limit. Genuine frames checked at 0.9-2.9 px and foreign ones at 7.9-11.5,
so the limit sits close to the genuine side; it was not moved.

**Over the whole game the camera check did more harm than good**, and is now
off by default. Search and check together, all 84 sampled frames: 50 fitted,
the same count as before. Gained 10 (8 with no landmark start, 2 previously
refused); lost 10 -- the two halftime highlights, correctly, and eight genuine
frames whose camera fits were on the paint, refused at 3.2-12.2 px against the
foreign frames' 7.9-11.5. The check trusts the free fit to judge the camera
fit, and the free fit is the one known to lock onto ad boards. With the check
off the search alone should give ~58 of 84 (to be re-measured), and frames
from other games need a test that does not rest on geometry -- the floor's
own colours (a red key, a navy one) are the obvious witness.

Against the whole-game labels at 1080p the search changes little (camera arm
0.51 ft trusted, 0.52 all), as expected: those frames mostly had landmark
starts.

**Coverage with the search and no geometric check:** 63 of 84 sampled frames
fitted (from 50), 13 gained, none lost; every new fit checked by eye lies on
the paint. Of the 21 still unfitted, ~15 are not main-camera game views
(close-ups, crowd, dancers, ads, overhead and rim cameras, wipes) and one is
the baseline camera; ~5 are main-camera views the search could not start --
centre-court shots and motion-blurred pans. Main-camera coverage is roughly
61 of 66, ~92%, against ~75% before.

**Other games, by the floor (e9812ac).** A halftime highlight from another
arena is a near pan/tilt/zoom of this camera -- its fit lies neatly on that
other court -- so geometry cannot refuse it without refusing good frames too.
Its paint can: the median CIELAB colour of the keys under each fit sat within
25 units of the game's for every genuine frame and at 106-115 for the two
highlights. estimate_camera now records each game's floor signature; a fit
whose key is more than 60 units away is refused (set with those values in
view, midway in the gap). No signature, or no key in view, never refuses.

**With the floor check, all 84 frames: 61 fitted.** Both halftime highlights
refused by key colour (106-115 units); no genuine frame refused by it (the
largest genuine distance was 25). Two frames flipped for another reason when
the centre was re-solved (136.86 -> 136.81 ft): t02426, a centre-court view,
lost its fit; t02876 gained one -- and that fit is WRONG. t02876 is a tight
close-up from a courtside camera under the basket; the search placed a
main-camera pose whose court lines happen to meet a few painted edges, and the
refinement accepted it (ratio 38) resting on only 38 samples. The key under it
was covered by players, so the floor check had nothing to judge.

Two guards were tested on data before being built, and both failed:

- Zoom range: the wrong fit's focal length is 2.46x the image width, in the
  middle of the 63 good fits (1.82-3.34x).
- Floor silhouette: its projected court overlaps the floor mask at 0.66,
  inside the good fits' 0.60-0.90 (the mask excludes the red key, so a
  close-up's floor looks court-shaped).

Next measured: how many samples good fits rest on, and which start won.

**What separates the wrong close-up fit: how much paint it rests on.** All 84
frames re-registered, recording each fit's samples and winning start:

    start that won     fits   samples: min   p10   median   max
    landmark            26            145    176     208     293
    search              35             38    104     201     350

The wrong fit (t02876) rests on 38; the next fewest of all 61 has 98. A fit
whose start came only from the search -- nothing else vouching for it -- now
needs 80 samples (SEARCH_MIN_SAMPLES, set with these values in view, midway in
the gap); landmark-started fits are unchanged. Note that the search now wins
35 of 61 frames, including frames that had a landmark start: within one kind
of evidence the fit on the most paint wins, whichever start it came from.

**Whole-game standing, 84 frames sampled every 45 s from 150 minutes:**
60 fitted, every fit checked by eye on the paint; both other-arena highlights
refused; ~16 frames are not main-camera views (close-ups, crowd, dancers, ads,
overhead and rim cameras, wipes, one baseline camera); ~6 main-camera views
unfitted (centre court, motion blur). Main-camera coverage ~60/66, ~91%.
Accuracy on this game remains measured only as ~0.5 ft against labels that
cannot resolve 0.3.

## Round 60 - a second labeller, the same failure, and the diagram that caused it

A second labeller used the precise 1080p labeller (line intersections only,
names on screen) on 42 frames across the whole Toyota Center game: 25 frames
with 8+ clicks, 227 clicks. Every frame's clicks disagreed with themselves by
feet -- ~20 ft leave-one-out -- and no frame gave a usable reference.

The clicks were mostly on the right spots with the wrong names. Sidelines and
corners were named consistently, but the two sides of the lane (x = 17 and 33)
the other way round; elsewhere a centre-court click carried the far sideline's
id, a baseline corner the opposite corner's. Label-only repairs, each committed
before scoring any registration: a per-frame lane/side swap (7f87f6b) chose a
corrected reading on 22 of 25 frames but left conventions mixed within frames;
a per-click reading among each landmark's mirror images (47052c1) found only 49
of 226 clicks agreeing on one reading.

Two eval bugs surfaced on the way and are fixed: frames with no landmark start
were skipped instead of registered by the search (ae24f63), and a frame with
no start and no accepted fit crashed the script (b697bfb). A reference-free
DIRECT metric was added (3df083e): each click through the registration against
its landmark, after repair and symmetry alignment. On the swap-repaired clicks
it read 0.66 ft with the camera (1.29 free), but that number is carried by
labels: on the three worst frames (10-50 ft) the camera fit lies on the paint
and the clicks carry the wrong landmark.

**Why both labellers mirrored.** The labeller drew the court with its length
vertical; the broadcast shows it horizontal, near sideline at the bottom. Every
click meant rotating the diagram by 90 degrees in one's head, and both people
did it inconsistently. The camera model says which way this game's court lies
in every main-camera frame -- court length to the right, the far sideline
(x = 0) at the top -- so the v3 labeller (data/labeling/court_toyota_1080_v3)
draws the diagram exactly so and names every landmark by where it is in the
picture ("left free-throw line: top end", "bottom sideline: right hash").

## Round 61 - measuring the labels, and a lens the model ignored

**How noisy are the hand labels?** Twelve frames were labelled by both people;
on 75 spots both clicked (within 25 px of each other) they disagree by 6.1 px
median at 1080p, 0.34 ft -- so one click carries roughly 0.24 ft of its own
noise (less for the second labeller, who clicked at 1080p; the first
labeller's clicks were carried up from 480p). On the second labeller's most
self-consistent clicks (per-click mirror repair: 49 clicks, 6 frames) the
camera fit reads 0.49 ft median; the first labeller's gave 0.49-0.52. Taking
the click noise out leaves a registration error of roughly 0.4 ft over the
whole game -- not 0.3. The labels do not explain it away.

**Where it sits: the frame edges.** By distance from the image centre (0 to 1
of the half-diagonal) the clean clicks read 0.40, 0.39, 0.44 and 0.65 ft.
The paint says the same without any label: long painted lines (boundary,
half-court) lie ~0 px from the straight-line model in the middle of the frame,
+2.2 px at 0.70-0.85 and +3.5 px beyond, always outward -- a zoom lens bending
straight lines, which no homography can represent. One radial coefficient
describes it (u_d = c + (u - c)(1 + k1 r^2)): from paint alone +0.0052 and
+0.0054 on two halves of the game's frames, +0.0052 and +0.0055 on wide and
tight zooms. Held out, it cut the edge offsets by 20-30% even before any
re-registration.

**Built (commit above):** estimate_camera measures k1 per game from its own
fits; register_frame undistorts the frame and boxes first and returns a matrix
on pinhole pixels; pixels go to the court through `to_court`. The centre of
the frame is still ~0.4 ft on the labels, so the lens is not the whole story.

**The lens correction, measured.** Estimated the committed way -- on the
game's own fixed-camera fits, since a free homography bends to follow the
lines and hid most of the distortion (k1 0.0007 on free fits) -- k1 is 0.0037
for this game. Against the second labeller's cleanest 49 clicks it changes
nothing that the labels can see:

    distance from frame centre      centre   mid    outer   edge    all
    no lens correction  (ft)        0.40     0.39   0.44    0.65    0.49
    lens correction     (ft)        0.41     0.39   0.37    0.63    0.50

The edge error that motivated it did not come down. 49 clicks is a small
sample, and the label-free straightness check (long painted lines against the
straight model, by radius) is the better test of whether the correction is
right; that result is recorded below when it lands. Either way the headline
stands: over the whole game the fixed-camera registration measures ~0.4-0.5 ft
against hand labels whose own noise is ~0.24 ft a click -- not the 0.3 ft
goal, and not a number these labels can refine.

## Round 62 - the reoriented labeller works; the labels' noise floor is ~0.25 ft

The third label set came from the labeller whose diagram is drawn as the
camera sees the court (v3). As given, with no repair, 163 of 203 clicks (80%)
agree with their frame's consensus within 1 ft; the second labeller's set,
from the old diagram, had whole frames mixing two conventions. The diagram was
the cause.

Against it the whole-game test (registration with camera, floor check and
landmark-free search; lens correction k1 = 0.0037):

    measure                               frames  clicks   camera p50   free p50
    DIRECT, all clicks                      23      193      0.54 ft     0.95 ft
    DIRECT-CONSENSUS (declared, cf36518)     6       48      0.44 ft     0.91 ft

Per-frame medians on the consensus frames: 0.24, 0.37, 0.40, 0.57, 0.65,
0.69 ft. FAIL. The consensus measure covers only six frames because the
declared rule needs 8 agreeing clicks and most frames have 7 of 8.

**The labels' own floor.** The two 1080p labellers clicked 176 of the same
spots; they disagree by 5.6 px, 0.35 ft median, so one click carries ~0.25 ft
of noise -- the same figure as the first pair (0.24). Against one labeller a
perfect registration would still score ~0.25 ft, so the 0.3 ft goal sits almost
on the labels' noise floor. Removing it from 0.44 leaves ~0.35-0.4 ft of
registration error over the whole game: above the goal, by this estimate.

Next, declared in scripts/eval_two_labellers.py before running: score against
the average of the two labellers' clicks on the spots both made (noise
~0.25 / sqrt 2 = 0.18 ft), keeping each frame's label consensus.

**Against the average of two labellers (declared, eval_two_labellers.py):**
19 frames, 132 spots where both 1080p labellers clicked and the averaged spots
agree with their frame's consensus. Camera registration (one-term lens,
k1 = 0.0037): **p50 0.39 ft, p75 0.62, 35% within 0.3 -- FAIL.** Per-frame
medians 0.20-0.58 ft, one frame 1.6. The averaged clicks carry ~0.18 ft of
their own noise, so the registration's own error over the whole game is
~0.35 ft: close to the goal, not under it. This is the cleanest whole-game
number the labels can give.

**Where the excess comes from: the frame edges, and not the boundary.** On the
stored fixed-camera fits of all 63 fitted frames, straight painted lines lie
~0 px from the straight model out to 0.6 of the half-diagonal and then bow
outward -- interior straights as much as the boundary (+1.95 / +5.6 px beyond
0.7 / 0.85, against +2.4 / +3.2 for the boundary), on the left (+1.9), right
(+2.9) and lower edge (+3.4) alike, and NOT along the score graphic's band
(-0.4). The one-term lens (k1 = 0.0037, estimated on the boundary alone) barely
moved it (2.32 -> 1.97 px, 3.25 -> 2.93 px): the bend rises far faster than
r^3. A two-term model (k1 r^2 + k2 r^4), fitted on boundary and interior
straights from two halves of the frames, agrees between halves (k1 -0.0018 /
-0.0008, k2 +0.0117 / +0.0128) and takes the held-out far-edge offsets from ~4
px to 1-2 px. It is now the lens model; estimate_camera fits both terms.

**With the two-term lens** (k1 -0.0021, k2 +0.0108, from paint, agreeing with
both split halves), against the average of the two labellers: p50 0.39 ft,
within 0.3 38% (from 35%) -- the lens barely moves the whole-game number.
Broken down, the error is not uniform: free-throw-line corners 0.24 ft (34
spots), baseline points 0.37 (61), far corners 0.45 (17), near corners 0.60
(5), sideline hashes and half-court points 1.54 ft (14). Drawn over the worst
frames, the key and lanes sit within 0.1-0.45 ft of the clicks while spots
20-40 ft down the court, near the frame edge, miss by 1.2-2.6 ft: the error
grows with distance from the paint the fit rests on, as in Round 54.

**Trusted score** (declared before its first output, acaf18e): of 132 spots,
107 lie within 3 ft of paint the fit used.

    spots        n     p50      p75      within 0.3
    trusted     107    0.34 ft  0.61     42%
    flagged      25    0.56 ft
    all         132    0.39 ft  0.62     38%

FAIL as measured -- 0.34 is above 0.30. The averaged clicks carry ~0.18 ft of
their own noise, so the registration's own error on trusted ground is roughly
0.29 ft: at the goal, but that is an estimate from the noise model, not a
measurement, and it is not reported as a pass. Over a whole game at Toyota
Center, the honest reading is: ~0.3 ft where the system vouches for a point,
~0.35-0.4 over everything, and the far sideline off by feet.

**Straightness with the two-term lens, re-registered (label-free):** straight
painted lines' median outward offset by radius, 60 fitted frames:

    radius (of half-diagonal)   0-0.3   0.3-0.5   0.5-0.7   0.7-0.85   0.85+
    no lens                     ~0      ~0        +0.2-0.3  +1.95-2.4  +3.2-5.6 px
    two-term lens               -0.20   -0.23     +0.21     +1.66      +2.66 px

The edges straighten somewhat but a ~2 px outward bow remains beyond 0.7 of
the half-diagonal. A two-term radial model centred on the image centre does
not account for all of it; an off-centre principal point, tangential terms, or
how paint is found near the frame edge remain open. On the whole-game labels
the lens changed nothing measurable (0.39 ft before and after).

## Round 63 - Phase 2 opens: the rim is projected, and the timeline is read

Phase 1 was accepted at ~0.3 ft on trusted ground over a whole game. Phase 2
(the ball at the rim during a shot) starts from two things Phase 1 makes
possible.

**The rim projects (129d2f5).** A homography maps the floor and nothing else,
but with the game's centre fixed a frame's registration IS a camera, so any 3D
court point projects -- the rim at 10 ft included, through the game's lens.
Checked against the four-class detector's own rim boxes over the whole 1080p
Toyota Center game, with no labels: 84 sampled frames, 60 fitted, the detector
sees a rim on 53 and the projected rim is in view on all 53; over 52 pairs the
projected rim centre sits **5.7 px from the detected one (p75 7.4), 0.13 ft
(p75 0.18), every pair inside one rim width.** That is both the Phase 2 unlock
-- rim availability goes from the detector's 0.364 of frames to every fitted
frame -- and a label-free check of the registration at a point no homography
can place.

**The timeline reads (c27e453).** The official shot chart is keyed to period
and game clock, so the video's own clock has to be read. scripts/read_game_clock
locates the clock (the region that reads as digits and ticks), learns its
digits from the seconds counting down, and reads every second: 3,707 of 9,356
sampled seconds on Finals G7, four periods, game time rising with video time on
99.9% of steps, spanning 125-2869 s against the chart's 16-2855.

Four period rules were tried and three failed on real footage: any upward jump
as a reset gave 20 periods (replays show an earlier clock); requiring the
previous reading near zero merged three quarters (the broadcast cuts away);
requiring the new quarter at 12:00 missed one first seen at 11:24; and
resolving the last minute's "35.9" by preferring tenths under a minute turned
"1:05" into 10.5 and split every quarter. What works uses only that the clock
falls within a period.

**Alignment:** all 157 official attempts map onto the video and **126 (80%)
sit within 2 s of a clock reading**, their video times rising with game time on
every consecutive pair. Those 126 are the evaluation set; the rest fall where
the clock is not on screen. Tuning will use the first half, scoring the second,
as the 0.396 measurement did.

## Round 64 - Phase 2's gate, on a game nothing was tuned on

Shot detection from the ball's path past the rim, scored against the official
play-by-play at +/- 3 s.

**The two findings that did the work.**

*Ball confidence.* The cache is written at 0.10 so no real ball is lost, but at
that level the four-class detector puts ~2.9 "ball" boxes on every frame, many
on the rim and net themselves. The nearest candidate to the rim was then junk:
over official shot windows the closest approach read 0.33 rim widths against
0.40 at random moments -- no signal at all. At 0.25 it reads 1.41 against 3.91,
and the ball comes within one rim width in 42% of shot windows against 17% of
random ones. Filtering by box SIZE instead destroys it: the detector's genuine
ball boxes are loose (~32 px at 720p against the rim's 39) and a 22 px cap
drops coverage from 53% of frames to 10%.

*Live play.* 76 of 181 second-half calls sat more than 30 s from any official
attempt: replays of a basket look exactly like the basket, free throws are
shots a field-goal chart does not contain, and warm-ups put balls through rims
too. All three happen with the clock stopped, while 98% of official attempts
happen with it running. Keeping only calls made while the clock ticks took
Finals G7 from F1 0.402 to 0.619 held out. That idea came from looking at where
the test half's false alarms fell, so it is declared in the script: the 0.619
is optimistic.

**The frozen test.** Thresholds fixed at what the sweep chose on Finals G7's
first half (approach 2.6 rim widths, far 5.0, merge 6 s, ball confidence 0.25),
committed before the second game's data existed (cf5e499), then run once on
Finals G1 -- a different game, 180 official attempts, 150 placed within 2 s of
a clock reading:

    game                         tol   predicted  official   P      R      F1
    Finals G1 (nothing tuned)    3 s      194       150     0.546  0.707  0.616
    Finals G1                    5 s      194       150     0.608  0.787  0.686
    Finals G7 (tuned on its 1st) 3 s      174       126     0.563  0.778  0.653

**PASS: 0.616 at +/- 3 s on a game nothing was tuned on**, against the 0.60
gate and the 0.396 this project measured before. The two games agree, which is
the point of the frozen run. [SUPERSEDED by Round 65: a clock misread had
misplaced 30 of G1's attempts, the framing here is too strong, and the stands
claim below is wrong. Corrected numbers and the narrower claim are in Round 65.]

**The gate's second half** -- no ball detections on people in the stands --
[WRONG, see Round 65: the test behind this flagged 0 of 78,293 boxes and
certified nothing; the condition is not met] is met where it matters: of the ball detections the rule uses (near the rim), 0 of
16 sampled sit off the court; 8.3% of the raw stream does, almost all of it
below 0.45 confidence. cache_detections now marks every ball box with whether
court lies beneath it (3494611).

**Not yet used: the projected rim.** The rim was seen by the detector on 0.372
of Finals G7's frames and 0.499 of G1's; gap-filling brings the rule's ball+rim
coverage to 0.35-0.47. The camera model puts a rim on every fitted frame,
0.13 ft from the detected one (Round 63), which is the obvious next lever for
recall -- and the measurement above is the baseline it has to beat.

## Round 65 - the review overturns Round 64's framing, and one misread cost a quarter

Round 64's gate was reviewed adversarially before being declared passed, as
every phase gate in this plan is. It did not survive as written. Four findings,
each checked here rather than taken on faith:

**1. The clever part of the rule does nothing.** The rule is described as
"far, then near, then far" -- an arc. A control that fires on proximity alone,
with no approach-or-recede test, returns the IDENTICAL 286 events on Finals G1
and 317 of the same 319 on G7. The test rejects two candidate events across two
whole games and moves F1 on neither. The ball is never parked at the rim for
the five seconds the windows span, so the condition never binds. The detector
is, honestly stated: *a ball-like box came within 2.6 rim widths of the rim
while the game clock was running.* The code keeps the test; its docstring now
carries the measurement instead of the story.

**2. One misread captured 738 s of Finals G1.** At video 1971 s the clock read
6:07 as "367". The resolver's rule was "take the largest reading not above the
previous one", so every later frame -- whose two candidates were "8:06" and
"80.6" -- had to fall below 367, and the whole rest of the quarter resolved to
tenths. Q2 was cut in two, a spurious fifth period appeared, and 30 official
attempts were dropped or misplaced. The fix is confirmation: a reading that
continues nothing is believed only once the NEXT reading continues it. A
misread survives one frame; a real jump (a new period) is still there on the
next one. `_continues` allows the clock to hold or fall at real time, +/-
CONTINUE_TOL_S = 3.0 for a dropped frame.

    game    periods found        attempts placed within 2 s
    G1      5  ->  4 (correct)     150  ->  173 of 180
    G7      8  ->  4 (correct)     126  ->  126 of 157

G7 had been splitting every quarter in two at the last-minute tenths. Both
games now resolve to exactly four periods running 12:00 -> 0:10. Re-resolving
costs no video pass (`--from-raw`), which is why the raw candidates are saved.

**3. The score, restated.** With the truth correctly placed:

    game                         tol   predicted  official   P      R      F1
    Finals G1 (nothing tuned)    3 s      200       173     0.605  0.699  0.649
    Finals G7 (tuned on its 1st) 3 s      174       126     0.563  0.778  0.653

A block bootstrap over 120 s blocks (resampling whole blocks preserves the
matching; resampling individual attempts does not, and an earlier attempt at
this reported a meaningless 0.375-0.456) gives G1 0.597-0.689, 4% of resamples
below the gate; G7 0.576-0.675, 14% below.

**The honest headline is narrower than Round 64's.** The number is F1 ~0.65
*within the stretches where the game clock is readable and running* -- 48% of
G1's video, 27% of G7's. The live-play filter is partly circular: official
attempts can only be placed where the clock is readable, and the filter allows
calls only in those same stretches. Without it, G1 is 0.495 and G7 0.447. The
comparison to "0.396 before" is therefore not like-for-like, and the period
rule (NEW_PERIOD_SHARE) was added after seeing G1. Phase 2 reads:
**0.65 within clock-live play, provisional** -- not a clean pass.

**4. The stands condition is NOT met, and Round 64's evidence for it was
vacuous.** `over_court` asked whether any court pixel lay BELOW the ball in its
column. In a broadcast frame the crowd sits above the floor, so this is true of
almost everything: it flagged 0 of 78,293 ball boxes as off court and certified
nothing. Replacing it with a real overlap test (does the box touch the floor
silhouette?) flags 34.8% of 400 sampled boxes at >=0.5 confidence -- but that
condemns every genuine ball in flight, so it does not measure the gate either.
The court-region mask cannot settle this question in either direction.

What can: looking. Of 40 randomly sampled off-silhouette detections at >=0.5
confidence, ~23 are one static object -- the spare ball on the rack at the
scorer's table, among seated courtside people -- ~7 are genuine balls in flight
or held, and ~10 are on people: heads, arms, torsos, including one
unmistakably on a spectator's bald head. Scaled to the stream that is roughly
8-9% of high-confidence ball detections landing on people, matching the
review's independent 8.7%/10.3%. **The gate's second half fails.** It does not
appear to cost the shot rule much -- a static rack ball never approaches the
rim, and the filter keeps calls to live play -- but the claim in Round 64 was
wrong and the test behind it measured nothing.

**Still not used: the projected rim**, the next lever for recall, unchanged
from Round 64.

## Round 66 - the rim comes off the camera, and a denominator written first

The owner's gate for Phase 2: rim accuracy 95%, ball accuracy 95%. Round 65
had just shown that a gate condition can be "met" by a test that cannot fail,
so the denominator is written before any number exists.

**The metric** (`scripts/eval_rim_and_ball.py`, declared before it was run):
frames on a fixed 25 s grid over the WHOLE video -- not around shots, not where
registration worked, not where the clock was readable. Every sampled frame is
labelled and counted. A frame where the object is not visible is not a miss --
nothing can be found there -- so it moves to the false-alarm denominator
instead, and both numbers are always printed:

    accuracy    = located / visible
    false alarm = reported where nothing was

"Located" is within one object width of the true centre: a rim is 1.5 ft
across, a ball 0.79 ft, so the tolerance follows the zoom. A point on the WRONG
basket, or on a spectator's head, is a miss AND a false alarm. Both baskets
count separately when both are shown, because picking "the" rim would have
meant choosing which one counted after seeing which one was found. What counts
as visible is fixed in `label_rim_and_ball.py` before labelling, replays and
other cameras included: a system that cannot register the baseline camera has
MISSED the rim in it, and excusing those frames would measure the main camera
and call it the game. Pre-game frames stay in; the report splits at the span
between the first and last clock reading, which this pipeline never touches.

**The rim now comes from the camera, not the detector.** Phase 2 had never used
what Phase 1 built. The detector sees a rim on 36.5% of G7's frames and 48.5%
of G1's, and dropping its confidence floor from 0.25 to 0.10 adds 0.7% and
1.4% -- it is saturated, not thresholded. Registering each frame from its own
paint costs 5-35 s and succeeds on about a third. But the centre is fixed, so
`track_camera.py` anchors rarely and carries the pose cheaply:

    method                                   pose      within 1 rim width
    per-frame registration                    36%      --
    anchor + chained frame-to-frame hops      69%      82%
    anchor + DIRECT hops, half-scale ORB      66%      99.5%

Three things were tried and rejected on measurement, not argument:

- *Chaining* consecutive frames: 0.47 rim widths median, 0 of 11 chained
  frames within one. Matching each frame to its ANCHOR is one hop of error
  instead of N.
- *Two-phase, bidirectional*: read each segment, then let every frame use the
  best anchor in either direction. It should rescue a frame whose camera move
  began moments earlier. It measured 65.3% and 90.8% against 66.1% and 99.5%;
  the poses it added came from hops across larger gaps, and capping the gap
  only cost coverage. Reverted, with the reason in the docstring.
- *A lower landmark-confidence floor*, to widen the gate anchors pass through:
  0.6 to 0.2 moved starts from 61% to 63% of frames. The landmark model is not
  being thresholded out; it simply does not fire on the rest.

What did work: half-scale ORB matching is both quicker AND slightly more
accurate (the area-averaged shrink denoises); re-fitting the carried
homography to the camera's four parameters every frame, so drift in the other
four is discarded instead of accumulated; and refusing any pose the DETECTOR
contradicts by more than 1.5 rim widths -- the one audit using evidence the
tracker never touches. Hops stay accurate to about 2-3 s and are then rejected
by the snap cost on their own (0.49 px at 0.2 s, 7.3 at 2 s, 18.3 at 4 s), so
the range limits itself rather than being set by hand.

**The ball is a selection problem before it is a detection one.** At the
cache's 0.10 floor a ball box appears on 80-83% of frames with ~3 candidates
each; at 0.25 it is 53-62%. On a first labelling sheet the chosen candidate
was repeatedly the spare ball on the rack at the scorer's table. Two tests,
both available only because the centre is fixed:

- A RAY, not a pixel: with the centre fixed, a world direction is the one
  description of an image point that camera motion cannot change. This is
  exactly what defeated the earlier image-space tracker, which scored
  smoothness in pixels and so made a stationary object look fast whenever the
  camera panned.
- FIXTURES: a direction producing ball boxes all game long is furniture. The
  rack ball is a real ball, correctly detected, and never the game ball.

No selection rule can beat the candidates it is given, so the ceiling gets
measured before any more work on choosing: `detect_ball_grid.py` re-detects the
grid at a larger inference size, the lever that took ball coverage 0.733 to
0.892 in an earlier round.

## Round 67 - the rim measured, and exactly what the remaining 17% is

The gate: rim 95%, ball 95%. Measured on the declared grid, by eye, against
`outputs/rim_ball/fullgame_grid.json` (verdicts in
`data/labeling/rim_ball/verdicts_fullgame.txt`):

    denominator                       visible  located  accuracy   95% CI
    whole video                          33      25      0.758   0.590-0.872
    in game (680-7276 s)                 23      19      0.826   0.629-0.930
    in game, main camera only            19      19      1.000   0.832-1.000

**FAIL on the gate as written.** But the third line is the whole story: every
single in-game miss is a frame the main camera did not shoot. The four are an
under-basket camera (888 s), a baseline camera (1062 s), a rim close-up that
fills half the picture (1388 s), and a replay inside a picture-in-picture
graphic (1812 s). Where the game's camera applies at all, the rim was found on
every labelled frame.

The third line is reported as context and NOT as the headline, deliberately. A
"main camera during live play" denominator is arguably the right one for Phase
2 -- the live-play filter already discards replays -- but it is a denominator
that would have been chosen after seeing which frames failed, which is exactly
the move that produced Round 64's false pass.

**Why those frames fail, measured rather than guessed.** Both halves of the rim
pipeline are blind to them, for the same reason:

- The landmark keypoint model returns ZERO keypoints on every alternate-camera
  frame tested (888, 1062, 1388, 2562, 5338, 5788 s). So neither the fixed
  camera nor a free homography can register them -- which also retires the
  free-homography projection built in Round 66 for exactly this case. It was
  built on a guess and the measurement says it does not apply.
- The four-class detector returns ZERO rim boxes on the same frames at
  confidence 0.01 and inference size 2560 -- and at 1280, 640, 320 and 160,
  so it is not a threshold or a resolution setting.

Part of the gap is plainly SCALE, and that part is reproducible on the main
camera: crop a frame around its own projected rim and enlarge it, and the
detector holds at 3x (confidence 0.47-0.78) and collapses to nothing at 6x.
The same rim, the same pixels, only bigger. The rest is viewpoint -- an
under-basket camera sees the ring from below, through the net -- and
shrinking those frames does not recover them, so crop-and-zoom augmentation
alone will not close it.

A label-free orange-ring finder was prototyped for the close-ups and rejected:
it finds the rim at 888 s and 1062 s but fires 30,744 px of "rim" on a branded
title card, and Indiana's gold kit sits next to rim orange in hue. This project
has been burnt by colour heuristics before.

**What would actually meet the gate**: rim boxes on alternate-camera frames,
which no existing signal in this repo can supply, so they have to be labelled
by hand. A rim is unambiguous to label -- unlike a screen, which is what the
original plan was right to refuse -- so this is a defensible place to spend
hand labels. Scale augmentation from the projected rim is free and should ride
along with them, since the 3x/6x result says scale is a real part of it.

**The ball is not yet measured.** At the 640 px panel the sheets render, a ball
is ~10 px: 29 of 66 frames could not be judged by eye at all, which makes the
2/7 reading meaningless and it is not reported as a number. The sheets now
magnify every candidate, which fixes the judging of a CLAIM, but finding a ball
the detector never proposed still needs the frame at full resolution. That pass
is owed before any ball figure is quoted.

## Round 68 - the projection is not the best rim, and the ball's problem is detection

Two corrections to Round 67, both found by looking harder rather than by
changing anything.

**The projection is beaten by the detector where the detector fires.** Round 66
reported the projected rim agreeing with a confident detection on 99.5% of a
180 s probe window. On the uniform GRID it is 90.6%. The probe window was one
steady stretch of main-camera play; the grid is the honest sample, and the
difference is a reminder that a window chosen for a smoke test is not a
measurement.

Of the 12 grid frames where they disagree by more than a rim width, all 8
inspected by eye had the DETECTOR on the rim and the projection about one rim
width above it, on the backboard. They are anchors, not drifted hops, several
at a snap cost of 0.0 -- so it is not tracking error. It is the weakness of
extrapolating a point 10 ft up from a homography fitted to the FLOOR: a pose
can match the floor lines with a slightly wrong tilt and only betray it away
from the floor. Re-fitting the rim's height against the detector puts the
best median at exactly 10.0 ft, the true height, so it is a biased subset and
not a global calibration to dial out. The order is now: believe a confident
detection, fall back to the projection, which still carries most frames and
the second basket.

Also measured: the snap cost predicts a hop's error well -- 100% within a rim
width below 2 px, 29% between 4 and 8 -- though capping it costs more coverage
than it buys.

**The labelling resolution was changing the answer.** At the 640 px panel of
the first pass, rims that are small or far read as "not in shot", which
quietly shrank the denominator in the system's favour. Re-labelled at 900 px
against the detector-first system, in game:

    object   visible  located  accuracy   95% CI
    rim        13       10      0.769   0.497-0.918
    ball        7        4      0.571   0.250-0.842

Both FAIL. The rim number did not move much (0.826 -> 0.769) but it moved the
wrong way under better looking, which is the direction that matters: the first
pass was flattering.

**The ball's problem is DETECTION, not selection.** All three ball misses had
no correct candidate on offer at any confidence -- the detector never proposed
the ball at all. The ray geometry, the fixture rule and the continuity
preference built in Round 66 all address SELECTION, and selection is not the
bottleneck. They are not wasted (the fixture rule alone removes the
scorer's-table ball from 34 frames) but they cannot move this number. Raising
the ceiling needs a better ball detector, and the same crop-and-zoom test that
diagnosed the rim should be run for the ball before assuming which axis of it
is wrong.

**Where the gate stands.** Neither object is at 95%, and the honest gap is not
a threshold anywhere:

- rim: frames the main camera did not shoot, where the keypoint model returns
  no keypoints and the detector no boxes. Needs hand-labelled rim boxes on
  those views -- a rim is unambiguous to label, so this is a defensible place
  to spend them -- plus scale augmentation, which is free from the projection.
- ball: frames where the detector proposes nothing. Needs the ceiling measured
  at higher inference size and, on that evidence, either a re-detection pass
  or a retrained ball class.

## Round 69 - a rim detector trained on rims it labelled itself

Round 68 left the rim at 0.769 and named the gap: frames where the landmark
model returns no keypoints AND the four-class detector returns no rim box, at
confidence 0.01 and at every inference size from 160 to 2560. Nothing left in
the repo to bootstrap from.

**The training data was made, not labelled.** Part of the gap is scale, and
that part reproduces on the main camera: crop a frame around its own rim and
enlarge it and the detector holds at 3x (confidence 0.47-0.78) and collapses to
nothing at 6x. The same rim, the same pixels, only bigger. So on frames where
the detector is already confident, its own box is crop-zoomed to make a
correctly labelled picture of a rim at a size it has never seen -- 8,256
training crops and 1,638 validation, 4,494 of them negatives so a single-class
model cannot simply learn to fire on anything orange.

Single class deliberately: folding rim-only crops into the four-class detector
would teach it "no ball and no player here" on every crop, the fault
`prepare_detector_dataset.py` already recorded for SportsMOT.

**Scored on the frames it cannot have seen** -- they are frames where the OLD
detector found nothing, and the training set was built from frames where it was
confident:

    fires on 3 of 7 frames that nothing else in the repo locates
    0 false alarms across 8 frames known to hold no rim
    main camera: 39 of 40 frames, p50 0.04 rim widths from the old detector's box

Every floor from 0.10 to 0.30 gives exactly that, so the threshold is on a
plateau and not a point fitted to those 15 frames. An UNDER-TRAINED checkpoint
reached 6 of 7 at a lower floor but also fired on a referee's red patch, a
water cooler, a graphic and an orange shoe -- the colour failure mode, learnt
rather than hand-written -- so the recall is not free and the finished model is
the conservative one.

The three it recovers are WIDE main-camera views where the rim is small and
far, each checked by eye and on the rim. That is the miss population that only
became visible when the labelling panel went from 640 to 900 px. The four it
does not are the true alternate viewpoints -- under-basket, baseline, a rim
close-up -- which crop-and-zoom cannot synthesise, and which still need hand
labels.

**Where the gate stands**, on 41 hand-judged in-game frames across two disjoint
uniform samples:

    object   visible  located  accuracy   95% CI       gate
    rim        25       21      0.840   0.653-0.936    0.95
    ball        7        4      0.571   0.250-0.842    0.95

The rim has gone 0.769 -> 0.840 and its interval now reaches the gate, on a
sample too small to settle it. Both still FAIL.

**The ball is detection-limited and the fix is known but unfinished.** At an
inference size of 2560 a ball box appears on 93% of grid frames against 80%,
and on a frame where the cached pass proposed nothing within 55 px of the ball
a candidate now sits on it -- the ceiling rises. But the same setting proposes
18 candidates a frame instead of 3, so choosing becomes the limit.

A geometric "is it in play" filter for that choosing was built twice and
removed both times. The reason is not a bug: worked by hand for a real
candidate, its ray sits between 10.2 and 17.3 ft above the floor for the whole
time it is over the court, an ordinary high arc. A ball 15 ft up over the far
side and a spectator behind it are ON THE SAME RAY. One frame carries no depth
to separate them; only motion does. That needs dense frames -- a 5 fps
re-detection pass at 2560 and dense poses to go with it, several hours each,
started and not finished here.

## Round 70 - two ball rules built, measured, and falsified

Both were meant to answer the ray ambiguity from Round 69: a ball 15 ft up over
the far side and a spectator behind it lie on the same ray, and only motion can
separate them. Motion does NOT need the camera's pose -- ORB between a frame
and its neighbours removes the camera's own movement directly, which turns a
six-hour dense-pose job into ten minutes. That part worked: across the grid,
1,220 of 2,717 candidates stand still once the camera is undone, and they are
heads, shoulders and logos.

**The rule built on it is wrong, because its premise is wrong.** "The game ball
is never still for a fifth of a second" is false: a player holding the ball at
the top of the key is very nearly static.

- As a FILTER it discarded held balls and ball coverage fell from 86.4% of
  frames to 78.3%.
- Rewritten as a PREFERENCE it still lost them: at 4412 s the ball sits plainly
  in a player's hands and the claim moved onto a defender; at 6912 s a
  free-throw shooter holds the ball and the claim moved across the floor.

The common case is the opposite of the premise -- the ball is HELD and still
while spectators and players move. Motion is off by default, kept behind
`--use-motion` with the finding attached.

**Raising the ceiling made the end-to-end result worse.** At 2560 a correct
candidate exists far more often (93% of frames carry a ball box against 80%,
and a frame that proposed nothing within 55 px of the ball now has one on it).
But it proposes 18 candidates a frame instead of 3, and confidence alone cannot
pick among them: at 6912 s the large-inference system picks (79, 81) -- the
top-left corner of the picture, in the crowd -- where the cached detector's
narrower candidate set had the ball. So the cached detections remain the
default for the ball.

That is worth stating plainly because it is the opposite of the obvious
conclusion from Round 69's ceiling measurement. A better ceiling is only worth
having with a selector that can use it, and this one cannot.

**Where the gate stands.** Rim 0.840 (21/25), ball 0.571 (4/7), both FAIL. Two
selection rules are now falsified with evidence rather than argued about, which
narrows what is left: the ball needs a selector that separates a ball from a
head WITHOUT assuming the ball moves -- appearance, or agreement between two
inference scales, or continuity over dense frames where 25 s of grid spacing
cannot help.

## Round 71 - the ball figure was inflated by its own truth set

Round 68's ball accuracy of 0.571 is wrong, and the fault is in how its truth
was built. Of the seven frames, four had their "true" ball position taken from
a v2 claim that had been judged correct. A system scores zero error against
truth copied from itself, so four of the seven were decided before they were
scored.

**Re-measured without that.** Frames chosen by position on the grid, the ball
located by eye on an unmarked 1300 px panel BEFORE any claim was looked at:

    system                                    located   accuracy   95% CI
    A  cached detections (the shipped one)      2/5       0.400   0.118-0.769
    B  2560 by confidence                       0/5       0.000   0.000-0.434
    C  2560 with two-scale agreement            0/5       0.000   0.000-0.434

Five confirmed balls is far too few to settle anything, and the intervals say
so. What it does settle is the direction: the ball is nowhere near 0.571, and
the two large-inference variants are worse than the one in use, not better. On
the frames they miss they are not close -- at 1062 s the ball sits plainly in a
player's hands and all three claims are 80-90 px away; at 4362 s it is being
dribbled and all three are in the crowd.

Both of the large-inference variants also fire on empty frames where the
shipped one correctly says nothing: on a player's hair at 4662 s and on a nose
at 5562 s.

**What this changes.** Three ball approaches have now been built, measured and
rejected -- motion (Round 70), raw large-inference selection (Round 70), and
two-scale agreement (here) -- and the measured position is worse than was
reported. The detector proposes the ball; on the frames checked at a 0.03 floor
it sits 3-16 px from the truth. It is buried at confidence 0.05-0.11 among 60
to 95 candidates, and nothing tried so far can dig it out.

That is the honest state: the RIM is at 0.840 and improving on a method that
works, the BALL is around 0.4 on a sample too small to pin, and its selection
problem is unsolved rather than nearly solved.

## Round 72 - seven ways to pick the ball, and what the failures add up to

Everything here was built, measured against hand-located balls, and rejected.
They are listed because the list is now the most useful thing known about this
problem.

    1  court-volume ray test        rejected 0 candidates: a test that cannot fail
    2  motion as a filter           premise false; held balls are still (86->78% cover)
    3  motion as a preference       still picked a moving defender over a held ball
    4  large inference + confidence worse end to end; picks the corner of the picture
    5  two-scale agreement          0 of 5, and fires on hair and on a nose
    6  handler-box proximity        rank unchanged or worse
    7  a learned patch ranker       mean rank of the true ball 1.0 -> 2.0

The ranker is the interesting failure. It was trained on 290 ball labels found
by motion-compensated tracking -- the Round 62 Viterbi idea with the fix that
it now runs in camera-compensated pixels rather than raw ones -- against the
other candidates in the same frames as hard negatives, split by TIME so it
could not be scored on what it memorised. It learns something real: validation
average precision 0.247 against a 0.145 baseline. It still ranks the true ball
WORSE than the detector's own confidence does.

Getting those labels took two attempts of its own, and both were caught only by
rendering a sample and looking at it:

- labelled from the shot chart, using the rim from the nearest grid frame up to
  30 s away while the camera pans: every label was on a shirt or in the crowd.
- labelled from smooth tracks at a 14 px/frame speed floor: 58% correct,
  because A RUNNING PLAYER'S SHOULDER traces just as smooth a path as a ball.
  At a 40 px floor -- only a ball in flight -- 83% correct.

**What the failures add up to.** On five balls located by eye on frames chosen
by grid position, the cached detector has NO candidate within 28 px on three of
them (92, 36 and 190 px away) and ranks the ball FIRST on the other two. So on
that detector the shipped selection is already optimal, and 0.400 is its
ceiling rather than its shortfall. Run large the ceiling is higher -- a
candidate 3-16 px from the truth on five of seven frames -- but buried at
confidence 0.05-0.11 among 60-95 proposals, and none of the seven rules above
can dig it out.

That is the shape of the problem, stated as plainly as it can be: the ball is
small, often held still, often occluded, and surrounded by objects that look
like it at the resolution it occupies. The rim is at 0.840 and rising on a
method that works. The ball is at 0.400 and every method tried on it has been
measured and rejected.

## Round 73 - retraining ball DETECTION, and the end of the ideas

Round 72 established that a ranker cannot help: on three of five hand-located
balls the detector proposed nothing within 28 px, so there was nothing to
re-score. Only a better detector fixes that, so one was trained.

**Labels without a labeller.** 602 ball positions from motion-compensated
tracks across both games -- a ball in flight moves faster than any player can
run, which is the one thing about its motion a player cannot imitate -- checked
by eye at 83% correct. Crops cut WITHOUT resizing, so the ball stays the 15-25
px it will be at inference, with the other candidates riding along inside each
crop as the hard negatives they are. Single class, alongside the four-class
detector rather than folded into it.

    validation on its own tracks: precision 0.51, recall 0.45, mAP50 0.464

**On the hand-located balls, which no track label touches: 2 of 5. Unchanged.**

    frame     old detector            new detector
    1062.5    92.3 px  rank 1/6       nothing proposed
    1662.5    13.5 px  rank 1/3       14.6 px  rank 1/1
    2562.5    20.3 px  rank 1/2        6.4 px  rank 1/3
    2862.5    35.5 px  rank 3/4      106.4 px  rank 1/1
    4362.5   189.5 px  rank 1/3      910.7 px  rank 1/1

It is a different detector, not a better one. It proposes one to three
candidates where the old proposes two to six, and it is tighter when right
(6.4 px against 20.3). It misses the same hard balls and replaces the old
one's scattered wrong guesses with one confident wrong guess.

**The full ledger for the ball**, every entry built and measured here:

    court-volume ray test        vacuous -- rejected 0 candidates
    motion as a filter           premise false; held balls are still
    motion as a preference       picks a moving defender over a held ball
    large inference + confidence worse end to end
    two-scale agreement          0 of 5; fires on hair and on a nose
    handler-box proximity        no change
    learned patch ranker         true ball's mean rank 1.0 -> 2.0
    retrained detector           2 of 5, unchanged

**The measured position.** Rim 0.840 and rising on a method that works. Ball
0.400, and the gap is not a rule waiting to be found: on the frames it misses,
no detector in this repo -- original, large-inference, or purpose-trained --
proposes anything within 90 px of the ball. The object is 15-25 px across, is
frequently held still, is frequently occluded by the hands holding it, and
shares a broadcast with dozens of objects its own size and colour.

What would move it is a ball detector trained on thousands of hand-drawn boxes
from this footage. Track labels cannot substitute: they only ever find the ball
where it is already easy -- in free flight, unoccluded, moving fast -- which is
precisely not where the misses are. That is the honest boundary of what was
reachable here.

## Round 74 - the ball measured properly: 0.300, and why that is the answer

Every earlier ball figure was too small a sample or drawn from truth the system
had supplied. This one is neither: ten ball positions located by eye on a
1280 px panel with a 50 px grid, on frames chosen by position in the evaluation
grid, recorded BEFORE any claim was looked at
(`data/labeling/rim_ball/ball_truth_handlocated.json`).

    system                      located   accuracy   95% CI
    shipped (cached detector)     3/10      0.300   0.108-0.603
    retrained ball detector       3/10      0.300   0.108-0.603

They succeed on DIFFERENT frames -- the shipped one at 1662, 2162 and 2562 s,
the retrained one at 1262, 1662 and 2562 s -- so a perfect oracle choosing
between them would reach 4 of 10. Choosing between them is the selection
problem that eight rules have now failed at, so 0.400 is the ceiling of every
combination available here, and 0.300 is what is actually delivered.

**A note on the labelling itself, which is evidence too.** Of ten in-game
frames looked at at full resolution with a measuring grid, the ball could be
confidently located on five, was plainly absent on two, and could not be found
at all on three -- by a careful human looking as long as it took. The object
is 15-25 px across, frequently held still inside the hands holding it, and
shares the picture with dozens of objects of its size and colour. A detector
being wrong on frames a person cannot solve either is not a tuning failure.

**Final position of the Phase 2 gate.**

    object   accuracy   95% CI          gate    over the session
    rim        0.840   0.653-0.936      0.95    0.769 -> 0.840
    ball       0.300   0.108-0.603      0.95    measured properly for the first time

The rim improved on a method that works and its remaining gap is known and
laborious rather than uncertain: hand-drawn boxes on under-basket, baseline and
rim-close-up cameras, which crop-and-zoom augmentation cannot synthesise
because the viewpoint, not the scale, is what differs.

The ball did not improve, and the ledger of what was tried is in Round 72 and
Round 73. The measured reason it did not: on the frames it misses, no detector
in this repository -- the original, the same one run at 2560, or one trained
from scratch on 602 labels from this exact footage -- proposes anything within
90 px of the ball. There is nothing for a selection rule to select. Closing it
needs a detector trained on thousands of hand-drawn boxes ON THE HARD FRAMES,
and the automatic label sources cannot supply those: the shot chart finds the
ball only at the rim, and tracking finds it only in free flight, unoccluded and
fast. Both find the ball exactly where it is already found.

## Round 75 - gap-filling raises the ceiling to 0.500 and delivers 0.300

A ninth approach, and the first to move the ceiling on the unbiased truth set.

`rim_track` has filled the rim's gaps from neighbouring frames since Round 63.
The ball needs the same thing and one extra step: a rim does not move, so its
gaps interpolate directly, while a ball does, so the neighbours' candidates are
carried into this frame's pixels by ORB before being pooled.

The detector HAS the balls it misses, just not on the frame being asked about:
at 4362 s its best candidate on the scored frame is 189 px away and 19 px away
a fifth of a second later; at 2862 s, 36 px and 14 px.

    on ten hand-located balls          located   accuracy
    shipped                              3/10      0.300
    pooled, by confidence                2/10      0.200
    pooled, own candidates first         3/10      0.300
    pooled, CEILING (an oracle choosing) 5/10      0.500

So pooling does what it was built to do -- a correct candidate now exists on
five frames instead of three -- and delivers nothing, because the two it adds
sit at rank 13 of 18 and rank 13 of 30 by confidence. The three the system
already found sit at ranks 3, 2 and 1. There is no signal in confidence or in
age that separates a rank-13 true ball from the twelve things above it; that is
the same wall every selection rule since Round 70 has hit.

It is kept anyway, off the delivered path, because it is the only thing that
has moved the ceiling and a future selector would need it.

**The ledger, complete:**

    1  court-volume ray test        vacuous
    2  motion as a filter           premise false; held balls are still
    3  motion as a preference       picks a moving defender over a held ball
    4  large inference + confidence worse end to end
    5  two-scale agreement          0 of 5
    6  handler-box proximity        no change
    7  learned patch ranker         true ball's mean rank 1.0 -> 2.0
    8  retrained ball detector      3 of 10, unchanged
    9  temporal gap-filling         ceiling 0.300 -> 0.500, delivered unchanged

**Final measured state of the Phase 2 gate:**

    object   accuracy   95% CI          gate
    rim        0.840   0.653-0.936      0.95
    ball       0.300   0.108-0.603      0.95

Every approach that could be built from signals inside this repository has been
built and measured. What remains is not an idea but a quantity: hand-drawn ball
boxes on the frames that are hard, in the thousands, which no automatic source
can supply because every one of them -- the shot chart, tracking, the rim --
finds the ball only where it is already found.

## Round 76 - what the hand-labelling actually costs, measured

Round 75 ended by saying the only thing left is hand-drawn boxes on the hard
frames. That is a claim about cost, so it was measured rather than asserted.

`mine_hard_ball_frames.py` defines hard WITHOUT knowing the answer, so the set
cannot be circular: a frame is hard when the detector's most confident ball
candidate is below 0.35. There are 560 such frames in G7's game span at 11 s
spacing, with a median top confidence of 0.23.

Nine of them were rendered at full resolution with a 50 px measuring grid and
worked through by eye:

    1 frame had no ball in it at all
    3 frames had a ball that could be located confidently
    5 frames had a ball that could NOT be located, by a person looking as long
      as they liked

So the yield is about one usable label per frame-triple, or roughly ONE LABEL
PER SHEET READ. Getting the few thousand that a detector needs is therefore a
few thousand careful looks -- and more than half of the hard frames cannot be
labelled at all, because the thing that makes them hard for the detector (the
ball is 15 px, held inside a hand, blurred by a pan, or lost against a crowd
of its own colour) makes them hard for a person too.

That is the honest cost of the remaining work, and it is why it did not get
done here rather than a judgement that it should not be. The three labels that
were obtained are kept in `data/labeling/rim_ball/ball_truth_hard.json`; they
are the only labels in this repository that come from the population the
detector actually fails on.

**The gate, finally:**

    object   accuracy   95% CI          gate    this session
    rim        0.840   0.653-0.936      0.95    0.769 -> 0.840
    ball       0.300   0.108-0.603      0.95    measured honestly for the first time

Nine ball approaches built and measured, two of which moved the ceiling (large
inference, gap-filling) and none of which moved delivery. The rim improved on a
method that works and has a known, laborious path to the rest.

## Round 77 - the rim's remaining work, priced

The ball's remaining work was priced in Round 76 and found to be worse than
expensive: more than half the frames that need labels cannot be labelled,
because what makes them hard for the detector makes them hard for a person.

The rim is the opposite case and worth pricing separately. Rims ARE labelable
-- large, orange, unambiguous, exactly the property that made hand labels
defensible here in the first place. Twenty mined frames, on which nothing in
this repository can find a rim (no keypoints, so no pose; no detector box at
any threshold), yielded six labels:

    overhead camera, rim and net large      1723 s
    under-basket camera                     1844 s
    rim at the frame's left edge            2597 s
    top edge, partially cut                 1594 s
    wide view, small and distant            5086 s
    high wide view                          6951 s

They are in `data/labeling/rim_ball/rim_truth_alternate_cameras.json`, and they
are the only rim labels here drawn from the population the misses come from.

The yield is about a third -- the mined set is diluted by close-ups and
graphics holding no rim at all -- so the 50 to 100 labels a retrain would want
is 150 to 300 frames looked at. That is a bounded, ordinary piece of work with
a plausible payoff, and it is the difference between the rim's 0.840 and the
gate. It is the one thing left in Phase 2 that is limited by effort rather than
by whether the information is in the picture.

**Phase 2, final:**

    object   accuracy   95% CI          gate    limited by
    rim        0.840   0.653-0.936      0.95    effort: ~200 frames of labelling
    ball       0.300   0.108-0.603      0.95    information: half the hard frames
                                                cannot be labelled by anyone

## Round 78 - the rim's gap, quantified on labels from the population itself

Round 77 priced the rim's remaining work. This does it: 15 rim boxes hand-located
on frames where nothing in this repository can find one -- no keypoints so no
pose, and no detector box at any threshold. Overhead cameras, under-basket
cameras where the ring fills half the picture, baseline cameras, a court inset
inside a graphic, and distant wides.

    on those 15 alternate-camera rims, at one rim width
      the scale-trained rim detector    1 / 15
      the four-class detector           0 / 15

That is the whole of the rim's remaining 16%, isolated and measured on labels
drawn from the failing population rather than inferred from the frames that
work. Two of them are rings 512 and 608 px across -- a rim filling half the
frame, found by nothing.

The labelling rate is about 1.3 usable labels per six-frame sheet, because the
mined set is dominated by close-ups and adverts holding no rim. So the 100 or
so labels a retrain would want is roughly 75 more sheets read by eye: bounded,
mechanical, and the honest remaining cost of the rim half of the gate.

`data/labeling/rim_ball/rim_truth_alternate_cameras.json`

**Phase 2 as it stands:**

    object   accuracy   95% CI          gate    what limits it
    rim        0.840   0.653-0.936      0.95    ~75 more sheets of labelling,
                                                then a retrain. Mechanical.
    ball       0.300   0.108-0.603      0.95    more than half the frames that
                                                need labels cannot be labelled
                                                by anyone; the information is
                                                not in the picture.

## Round 79 - 23 hand labels, trained, and measured WORSE

Round 78 isolated the rim's remaining gap on 29 rim boxes hand-located from the
cameras nothing here can register, and said the fix was training on them. It
was done: 23 of them (six held out for sitting within 30 s of an evaluation
frame), each written out as 26 crops at random offsets with half flipped, mixed
into the 8,256 synthetic crops, and trained for 14 epochs.

**It learns the viewpoint.** At 887.5 s -- an under-basket camera with the ring
filling half the picture, which no detector in this repository has ever fired
on -- the new model puts a box on the ring's lower-front arc, about 0.2 rim
widths from its centre, checked by eye. Main-camera accuracy is untouched
(p50 0.03 rim widths, 100% within one) and its false alarms fall from 1 to 0.

**And it is worse overall.**

    model                                 located   accuracy   95% CI
    scale crops only                       21/25     0.840   0.653-0.936
    scale crops + 23 hand labels           19/25     0.760   0.566-0.885

It gained the under-basket class and LOST two distant wides it used to find --
2012 s and 3412 s, both frames where the rim is small. Twenty-three examples
over-weighted twenty-six-fold taught it the viewpoints it saw and cost it a
size it already had. The shipped model is reverted to the better-measured one;
the hand-trained checkpoint is kept beside it as `checkpoints/rim_scale_hand.pt`.

Worth saying plainly: the verdict files record judgements against a particular
system, so scoring v7 with them gave a stale 12/13 until the two changed frames
were checked by hand. The number only fell to 0.760 because that check was
made. A verdict is not truth; it is truth-about-a-claim.

**What it means for the labelling plan.** The plan was right that these labels
are the gap and wrong that a few dozen would close it. 23 labels move the model
from one failure mode to another. Closing it needs enough to cover every
alternate camera AND hold the sizes already learnt -- hundreds, not dozens, at
the measured rate of about 1.6 usable labels per six-frame sheet.

**Phase 2, measured:**

    object   accuracy   95% CI          gate
    rim        0.840   0.653-0.936      0.95
    ball       0.300   0.108-0.603      0.95

## Round 80 - 41 hand labels, retrained, and the rim does not move

Round 79 trained on 23 hand-located alternate-camera rims and measured worse:
over-weighted 26-fold they taught the model the viewpoints they showed and cost
it a rim size it already had. So the set was extended and rebalanced.

**Mining was fixed first.** Filtering mined frames through `has_court` -- the
wood-fraction gate that already existed to refuse frames before a registration
search -- raised the labelling yield from about 30% to about 50%. Without it,
the frames "nothing can find a rim in" are dominated by close-ups, adverts and
studio shots that hold no rim at all, and most of a sheet is wasted.

41 labels now, 34 trainable and 7 held out for sitting within 30 s of an
evaluation frame, and the size spread is deliberate: 24 rims under 100 px, 10
between 100 and 300, and 7 over 300 px. Over-weighting cut from 26 to 12.

**The model improves on the held-out failures and the system does not.**

    on the 7 frames nothing could locate      scale only  23 labels  41 labels
                                                  3/7        2/7       3/7
      (mid-training, before the best epoch)                             4/7

    on the 25 labelled evaluation rims        0.840      0.760      0.840

Mid-training it reached 4 of 7 with 3 false alarms; the checkpoint selected by
validation mAP is the conservative one, 3 of 7 with 1. On the evaluation grid
exactly one labelled frame changed -- #28, which is labelled "no rim in shot",
so the change is a FALSE ALARM and not a gain.

**What 41 labels bought: nothing, and a direction.** 23 labels moved the model
backwards, 41 move it back to level with a better mid-training peak. The
trajectory says the labels are the right lever and that the quantity is still
wrong by roughly an order of magnitude -- hundreds, spread across every
alternate camera and every rim size, not dozens.

**Phase 2, measured, final for this session:**

    object   accuracy   95% CI          gate    over the session
    rim        0.840   0.653-0.936      0.95    0.769 -> 0.840
    ball       0.300   0.108-0.603      0.95    0.571 (inflated) -> 0.300 (honest)

## Round 81 - a leak in my own held-out rule, and the corrected count

While extending the rim labels to 49, the held-out rule turned out to be wrong.
Rounds 79 and 80 excluded labels within 30 s of the seven `rim_scale` test
frames -- but the GATE is scored on the 25 labelled grid frames, which is a
different set. Eight labels sat within 30 s of a grid frame and were used in
training anyway.

It did not inflate anything: those runs measured 0.760 and 0.840 against a
0.840 baseline, so the leak bought nothing. But the criterion was wrong and it
is now the union of both sets, which cuts the trainable labels from 35 to 27:

    49 hand labels
    27 trainable  (15 small under 100 px, 7 from 100-300, 5 over 300)
    22 held out   within 30 s of a frame the gate is scored on

And that is its own finding. The behind-backboard and under-basket views -- the
biggest rims, the hardest class, the ones worth most -- CLUSTER around the
frames the evaluation already flagged as failures, because both are picked out
by the same thing: a replay. So the labels most worth having are the ones most
often disqualified, and a bigger labelled set has to come from a wider sweep of
the game rather than from more looks near the known failures.

**Phase 2, final for this session:**

    object   accuracy   95% CI          gate    over the session
    rim        0.840   0.653-0.936      0.95    0.769 -> 0.840
    ball       0.300   0.108-0.603      0.95    0.571 (inflated) -> 0.300 (honest)

Neither meets the gate. The rim's remaining work is bounded and mechanical and
now has a measured rate: ~1.6 usable labels per six-frame sheet, roughly half
of them disqualified by the held-out rule, so the few hundred trainable labels
a retrain needs is on the order of a thousand frames looked at by eye. The
ball's remaining work is not bounded the same way: on the frames it misses no
detector here proposes anything within 90 px, and more than half of those
frames have a ball that cannot be located by a person at full resolution.

## Round 82 - proposal verification, hard negatives, and a fourth honest no

Labelling by locating rims yields ~1.6 a sheet and half are disqualified.
Verifying the detector's OWN low-confidence proposals is twelve judgements a
sheet instead of two locations, and an accepted box arrives with coordinates
attached. `propose_rim_labels.py` renders them in a numbered grid.

**The rejections were worth more than the acceptances.** Of 35 proposals
verified, 4 were rims. The other 31 are dominated by ONE mistake: the shooter's
square inside the backboard -- a bright rectangle a foot above the ring --
which the model reads as a rim at confidences up to 0.80. That is a specific,
nameable error, and 68 crops of it went in as explicit hard negatives: a crop
with an empty label file is the model being told this rectangle is not a basket.

**And it made the model worse.**

    model                                    held-out rims found   false alarms
    scale crops only                               3 / 7                0
    + 23 hand labels                               2 / 7                1
    + 41 hand labels                               3 / 7                1
    + 27 hand labels + 68 hard negatives           1 / 7                1

The negatives suppressed the thing they were meant to sharpen. A rim seen from
under the basket and a shooter's square seen from the stands are not far apart
in a 640 px crop, and with 68 examples saying "not this" against 27 saying
"yes that", the model learnt caution. Shipped model reverted to the scale-only
one for the fourth time; the others are kept beside it.

**Four training runs, four honest measurements, one unchanged number.** The rim
has sat at 0.840 through 23, 41 and 27+68 labels. Every run improves something
visible -- the under-basket viewpoint, the false-alarm rate, the validation
mAP -- and none improves the frames the gate is scored on. That is what a
label count an order of magnitude too small looks like from the inside.

**Phase 2, as measured:**

    object   accuracy   95% CI          gate
    rim        0.840   0.653-0.936      0.95
    ball       0.300   0.108-0.603      0.95

## Round 83 - the evaluation could not see the gate, so it was widened

Before running a fifth training cycle, a check on the measurement itself. At
n=25 visible rims, a system whose true accuracy is 0.90 reads as 0.700-0.958:

    n      a true 0.90 reads as      width
    25       0.700 - 0.958          0.258
    50       0.786 - 0.957          0.170
    100      0.826 - 0.945          0.119
    200      0.851 - 0.934          0.084

The 0.95 gate sat INSIDE the interval. The evaluation could not have told a
passing system from a failing one, which makes four training runs judged
against it worth less than they looked.

So 20 more grid frames were labelled for the rim -- a third sample, frames not
previously touched, judged on 900 px panels:

    RIM over three samples   31/38 = 0.816   95% CI 0.666-0.908

The gate is now OUTSIDE the interval, which is the first time this measurement
has been able to say so. The point estimate also fell, from 0.840 to 0.816, as
a bigger sample usually does when the first one was small.

Two of the new misses are worth naming because they are not the alternate-camera
class at all:

- 1638 s: the projection lands on the BACKBOARD'S TOP EDGE, a rim width above
  the ring, on an ordinary main-camera close-up.
- 3588 s: the same, with the detector's own box sitting correctly lower and
  losing to the projection because it was under the trust threshold.

That is the Round 68 finding again -- extrapolating a point 10 ft up from a
homography fitted to the floor -- and it means part of the remaining 18% is a
geometry problem on the MAIN camera, not only a training-data problem on the
others.

**Phase 2, measured on the widest sample yet:**

    object   located   accuracy   95% CI          gate
    rim       31/38     0.816   0.666-0.908      0.95
    ball       3/10     0.300   0.108-0.603      0.95

## Round 84 - three verdicts were mine, and the ball detector was measured at the wrong scale

### The rim was understated, because I judged the wrong circle

The labelling sheets draw TWO circles on every frame: the rim projected from
the camera pose, and the rim the detector reports. Only the second is what the
system claims and only the second is scored. On three frames I read the
projected circle, saw it off the ring, and wrote "miss" while the claim the
system actually makes sat on the rim.

Re-rendering every frame marked "miss" against the reported claim list:

    65   (1638 s)   2 claims, claim 0 on the orange ring     -> ok
    122  (3062 s)   1 claim, on the rim                      -> ok
    143  (3588 s)   1 claim, on the rim                      -> ok
    35   (888 s)    0 claims, under-basket camera            -> miss
    84   (2112 s)   0 claims, high wide view                 -> miss
    178  (4462 s)   0 claims, close-up drive                 -> miss
    218  (5462 s)   0 claims, a TV-schedule graphic          -> miss

    RIM over three samples   34/38 = 0.895   95% CI 0.759-0.958

The 0.95 gate is now INSIDE the interval, which it was not at 0.816. The point
estimate is still below it and the gate is not met. Frame 218 is a full-screen
broadcast graphic with a live court inset; it is left as a miss rather than
reclassified, because a rule invented after seeing which frame it helps is not
a rule.

### The four real misses are one failure, and crop-and-zoom cannot fix it

All four have `rim: []`, `projected: []` and `detected: []` -- no pose and no
box. On 887.5 s, an overhead-behind-backboard shot with a ring filling a
quarter of the picture, the four-class detector proposes NOTHING at conf 0.02
at 1280, 2560 and 3840 px, and the scale model nothing over a 3x3 tiling.

That is not a threshold. It is a viewpoint the detector has never seen, and
`build_rim_dataset.py` structurally cannot supply one: it crops around the
detector's own confident boxes, so every crop it makes comes from a camera the
detector already handles. Four training runs and one unchanged number follow
from that directly.

### Carrying the hand labels instead of collecting more

A rim is bolted to the floor and these cameras are bolted to the building, so
one rim located by hand can be carried by ORB to every other frame that camera
shot. 49 anchors became 158 labels, and two gates had to be written on the way.

The hold-out first refused everything -- the grid samples every 25 s, so a 30 s
radius covers all 9,355 s. Widening it would have been a threshold moved after
seeing the result. What a hold-out is FOR is that the model must not train on a
picture of an evaluation frame, and ORB states that exactly: a candidate is
dropped when an evaluation frame registers to it. It removed 30 of 49 anchors,
because these fixed cameras register to each other across the whole night, and
0 of 158 labels share a shot with the rim_scale test frames either.

Then 2 of 24 sampled boxes landed in open crowd -- once where a moving
under-basket camera let parallax break a fit the background still supported,
once across a hard cut where RANSAC found a consensus among false matches.
Both had no inlier near the ring. A homography is only trustworthy where it has
evidence, so the fit must now have inliers within 3 rim widths of the anchor's
ring. All 24 samples on the re-run are correct.

Also corrected: `data/rim_scale`, called "scale crops only" in Round 82's
table, already held 314 crops from 16 hand labels. The new set strips every
earlier hand crop and adds the 158 propagated ones at weight 8.

### The ball: recall is nearly solved, colour points the wrong way

Two entries for the ledger.

TILING. Slicing the frame into tiles and detecting in each is not the "large
inference" rejected in Round 71: upscaling the whole frame keeps a 15 px ball
in a crowd of 1,280 px, while a tile gives it its own context at twice the size.

    config                  within tol   the ball's rank by confidence
    whole 2560                 6/10      9, 15, 44, 52, 54, 62
    whole 3840                 4/10      6, 17, 40, 60
    tile 320/0.3 @640          7/10      19, 36, 58, 59, 66, 75, 97
    tile 256/0.4 @640          5/10      16, 59, 95, 191, 213
    tile 192/0.4 @640          6/10      23, 49, 51, 112, 124, 170
    pooled                     9/10

In every configuration the true ball is the top-confidence candidate ZERO
times. More proposals make recall better and ranking harder: 189 candidates a
frame to find a ball at rank 170.

COLOUR does not merely fail, it points backwards.

    true balls   n=7    orange share  min 0.000  p50 0.075  max 0.650
    false above  n=403  orange share             p50 0.000  p90 0.511

Three of seven true balls contain no orange pixel at all. A 15 px
motion-blurred ball over bright maple is a grey smudge, while the floor,
players' skin and Indiana's gold kit all sit in the orange band.

Together: AT THIS SIZE A SINGLE PATCH CARRIES ALMOST NO INFORMATION, which is
why the learned patch ranker moved the true ball's mean rank from 1.0 to 2.0.

### The ball detector had been measured at twice its training scale

`build_ball_detector_dataset.py` cuts 640 px crops WITHOUT resizing, so the
model learns balls at native size. It was then evaluated at imgsz 2560 on a
1280x720 frame -- every ball twice the size it was trained on. Round 74's "3 of
10, unchanged" is that mismatch.

    model       inference      within tol  top-1  candidates  ranks
    shipped     whole@2560        6/10       0        59      9...62
    ball_track  whole@2560        3/10       2         2      1, 1, 2
    ball_track  whole@1280        6/10       3         3      1,1,1,2,2,2

At its own scale the single-class detector proposes THREE candidates instead of
59, and the ball is rank 1 or 2 on every frame it finds. Ranking stops being a
search through a hundred distractors and becomes a choice between two.

It also cannot be trusted as measured, because it was trained on track labels
mined from this same video: the nearest sits 1.7 s from an evaluation frame,
and 1262.5 s -- one of its rank-1 successes -- is one of those. The same-shot
filter drops 8-13% of each label file, and the detector is being retrained on
what is left.

Time, tried on those three candidates, nets to zero: requiring a companion at a
ball-like displacement in neighbouring frames fixes 737.5 s and breaks
1262.5 s.

**Phase 2, as measured:**

    object   located   accuracy   95% CI          gate
    rim       34/38     0.895   0.759-0.958      0.95
    ball       3/10     0.300   0.108-0.603      0.95

## Round 85 - the fifth rim run's "new find" was an advertising hoarding

The 158 propagated labels trained cleanly (mAP50 0.875) and `eval_rim_scale.py`
reported the first improvement in five runs:

    model                    held-out rims found   false alarms   main camera
    shipped (scale crops)          3 / 7               0          0.04 widths
    + 158 propagated labels        4 / 7               0          0.03 widths

That script's own docstring says "positions still need an eye". Zoomed 4x, the
new find at 2112.5 s is a red State Farm advertising board in the stands -- a
horizontal red bar at rim scale, nothing to do with a basket.

The real result is 3 of 7, unchanged, plus one false alarm and one main-camera
frame that lost its box. Reverted, for the fifth time. What IS real: the three
true finds became much more confident, 0.31 -> 0.72, 0.33 -> 0.46, 0.76 -> 0.82.

**A count of boxes is not a measurement of rims.** Third time in this project a
metric has moved the right way while the truth did not, and the only thing that
caught it was rendering the claim and looking at it.

### Why the two biggest rims stay missed, which is now known rather than guessed

887.5 s and 1387.5 s are close-ups with rings 400-600 px across. The anchors
most like them are 885.0 s (608 px) and 888.0 s (432 px) -- and those are
exactly the ones the same-shot rule DROPS, because they share a shot with an
evaluation frame.

That is the hold-out working. Those frames are a fair test of generalising to
an unseen close-up viewpoint, and the model does not generalise to it. Only 28
of 158 propagated labels carry a rim over 200 px wide, and the scale dataset's
big rims are UPSCALED small ones -- blurry where a real close-up is sharp with
visible net cord. The domain gap is in the texture, not only the size.

**Phase 2, unchanged:**

    object   located   accuracy   95% CI          gate
    rim       34/38     0.895   0.759-0.958      0.95
    ball       3/10     0.300   0.108-0.603      0.95

## Round 86 - the ball moves for the first time, by fixing two measurement faults

Not a new idea. Two faults in how the existing ball detector was built and
measured, and the number moves once both are repaired.

**The scale mismatch.** `build_ball_detector_dataset.py` cuts 640 px crops
WITHOUT resizing, on purpose -- "the ball is 15-25 px across and its whole
difficulty is its size". The model was then evaluated at imgsz 2560 on a
1280x720 frame, showing it every ball at twice the size it learned.

**The leak.** Its track labels were mined from the same video the gate is
measured on. The nearest sits 1.7 s from an evaluation frame, and 1262.5 s --
one of the detector's rank-1 successes -- is one of those.

`filter_labels_by_shot.py` dropped 8-13% of each label file and the detector
was retrained on what was left, at yolo11n batch 8 rather than yolo11s batch
16, because the machine had been in swap at 15.9 GB of 16 GB and both earlier
runs were crawling for that reason and not from GPU contention.

    on 13 hand-located balls        ceiling   delivered   ranks
    shipped (4-class @2560)          -          0.300     -
    ball_track, leaky, @1280        6/13       3/13 0.231  1,1,1,2,2,2
    ball_clean, leak-free, @1280    7/13       4/13 0.308  1,1,1,1,2,3,5

Two things worth separating, which one figure had been hiding. At native scale
the single-class detector proposes FOUR candidates a frame instead of the
four-class detector's fifty-nine, and the ball is rank 1 on four of the seven
frames where it is proposed. Ranking stops being a search through a hundred
distractors and becomes a choice among four.

Honestly: 4/13 against 3/13 is one frame, and at n=13 the intervals are
0.127-0.576 and 0.082-0.503. The direction is right and the leak is gone; the
size of the gain is not measurable at this sample size.

### The twelfth approach, and why it looked right

Rendered at 4x, the candidates that outscore the true ball are HUMAN HEADS --
a bald head at 0.82 against the ball's 0.34, a spectator's head at 0.52 against
0.05. At 20 px a head is round, skin-toned and textureless. That is also the
explanation for Round 84's colour result: heads are orange.

What separates them to a person is the crowd around them, so the ranker was
rebuilt on a 192 px NEIGHBOURHOOD instead of the 30 px of context the Round 72
ranker had. It reached validation average precision 0.313 against a class
balance of 0.171 -- it learned something -- and it DELIVERED WORSE:

    delivered, by confidence    4/10  0.400   ranks 1, 1, 1, 1, 2, 3
    delivered, by the ranker    3/10  0.300   ranks 1, 1, 1, 2, 4, 4

It broke 1262.5 s from rank 1 to 2 and pushed two others down. Twelfth
approach, rejected, kept beside the eleven others.

**The ceiling is the thing to notice.** With this detector a correct candidate
exists on 7 of 13 frames, so 0.538 is what a PERFECT selector would deliver.
The 0.95 gate is not reachable by selection at all; it needs recall, on an
object that is 12-18 px in a 1280x720 broadcast. There is no higher-resolution
copy of this game -- the 1080p file on disk is a different fixture, Houston in
the regular season.

**Phase 2, as measured:**

    object   located   accuracy   95% CI          gate
    rim       34/38     0.895   0.759-0.958      0.95
    ball       4/13     0.308   0.127-0.576      0.95
    ball ceiling with this detector    0.538

## Round 87 - a fifth rim sample takes 0.958 back down to 0.932

The fourth sample put the in-game rim at 46/48 = 0.958 with an interval of
0.860-0.988: the gate met at the point estimate, on a sample far too small to
say so. A fifth sample of 24 more in-game grid frames was drawn to find out
whether it held.

It did not.

    samples 1-3, in game      34/36   0.944   CI 0.819-0.985
    samples 1-4, in game      46/48   0.958   CI 0.860-0.988
    samples 1-5, IN GAME      55/59   0.932   CI 0.838-0.973
    samples 1-5, whole video  55/61   0.902   CI 0.802-0.954

0.958 was a small-sample fluctuation and widening the sample removed it. This
is the fourth time in this project that a number good enough to stop on has
dissolved under more measurement, and the only defence has been to keep
measuring after the number looked right rather than before.

### The residual is one camera, named

The four in-game misses are no longer a miscellany:

    35   (888 s)   overhead behind the backboard, ring filling a quarter of
                   the frame                                     0 claims
    115  (2887 s)  low under-basket, backboard and ring and net in plain view
                   with a player shooting                         0 claims
    231  (5787 s)  the same camera, ring and net in plain view    0 claims
    218  (5462 s)  a TV-schedule graphic with a live inset        0 claims

Three of four are the low/under-basket camera. On all three, every model this
project owns returns nothing at confidence 0.05 across inference sizes 320,
640, 960, 1280 and 1920 -- the four-class detector, the scale-trained model and
the propagation-trained model alike. One exception at 0.07, which is noise.

That is worth stating precisely because it is not a tuning problem and not a
scale problem. Round 85 established the same for the giant close-ups by running
them from 128 px to 960 px and finding nothing anywhere. It is a viewpoint the
training distribution does not contain, and the frames most like the
evaluation's examples of it are exactly the ones the same-shot hold-out
correctly withholds.

### What the verdicts cost to get right

Five judgements in sample four were made twice, and three changed -- two
towards the system and two against it. Reading a verdict off a contact sheet
has now been wrong five times in this project. Reading it off a magnified pane
of the claim the system actually reports has not yet been.

`render_rim_sheets.py` therefore draws ONE thing: every entry in the frame's
reported `rim` list. The old diagnostic sheets drew the projected rim beside
the detected one and that cost three frames in Round 84. Anything else on the
picture is something to mistake for the answer.

**Phase 2, as measured:**

    object   located   accuracy   95% CI          gate
    rim       55/59     0.932   0.838-0.973      0.95    in game
    rim       55/61     0.902   0.802-0.954      0.95    whole video

## Round 88 - the thirteenth ball approach: dribbles harvested, and no gain

Every ball label this project owns is a ball IN FLIGHT, because
`find_ball_tracks.py` requires a motion-compensated 40 px/frame and says why:
below that a running player's shoulder traces just as smooth a path. So the
detector had never seen a ball at rest in someone's hands, while the evaluation
samples the game uniformly, where most balls are held or dribbled -- the truth
set's own notes say "held by the dribbler", "loose-ball scramble".

`harvest_ball_tracks.py` attacks that with the one signature the missing
population has and the poisoning population does not: A DRIBBLED BALL BOUNCES.
With the camera removed by ORB, a dribble is a periodic vertical reversal of
20-40 px at 1-3 Hz. A head does not do that. 95 labels over a whole game.

    on 13 hand-located balls     ceiling   delivered   validation mAP50
    ball_clean                    7/13     4/13 0.308       0.539
    ball_dribble (+95 dribbles)   6/13     4/13 0.308       0.580

Validation mAP rose and the gate number did not move -- and the CEILING FELL.
The frame it lost is 1055.6 s, whose truth note reads "held by the dribbler",
which is precisely the population the 95 labels were harvested to teach.

Reverted to ball_clean, which delivers the same and proposes more.

Two things the harvest did establish, and both are kept:

- The flight branch is OFF. It is mis-calibrated at a 0.1 s step (40 px/frame
  is 400 px/s where the flight miner meant 200), and the single flight chain it
  accepted in a whole game was A COACH'S HEAD on a close-up sideline shot --
  shallow depth, ORB cannot compensate the camera, so everything appears to
  move fast at once and "fast and smooth" describes the whole picture. The
  dribble rule is immune to that failure, because a compensation failure makes
  everything drift together and drift is not a periodic reversal.
- A sample sheet caught it. Six of eight dribble labels sit on a player's hands
  with the ball; the flight one was the head.

**Ledger, complete at thirteen:**

     1  court-volume ray test         vacuous
     2  motion as a filter            premise false; held balls are still
     3  motion as a preference        picks a moving defender over a held ball
     4  large inference + confidence  worse end to end
     5  two-scale agreement           0 of 5
     6  handler-box proximity         no change
     7  learned patch ranker          true ball's mean rank 1.0 -> 2.0
     8  retrained ball detector       3 of 10, unchanged (measured at 2x scale)
     9  temporal gap-filling          ceiling 0.300 -> 0.500, delivered nothing
    10  tiled inference               recall 6->7 of 10, top-1 still zero
    11  orange colour prior           true balls are LESS orange than the false
    12  192 px context ranker         val AP 0.313 vs 0.171, delivered 3/10
    13  harvested dribbles            val mAP 0.539 -> 0.580, ceiling 7 -> 6

The two things that DID move the ball were not ideas at all. They were a scale
mismatch -- 640 px crops cut without resizing, evaluated at imgsz 2560 -- and a
label leak, training frames 1.7 s from an evaluation frame. 0.300 to 0.308, one
frame, on intervals that overlap almost completely.

**Phase 2, as measured:**

    object   located   accuracy   95% CI          gate
    rim       55/59     0.932   0.838-0.973      0.95   in game
    ball       4/13     0.308   0.127-0.576      0.95
    ball ceiling with the best detector       0.538

### And the rim cannot be carried in from a neighbour either

The obvious remaining move on the under-basket misses was inference-time
propagation: the camera is bolted to the building, so if the rim is found
anywhere in the same shot, ORB carries it to the scored frame. That is what
`propagate_rim_labels.py` does for labels, and Round 63's rim gap-filling
already does it across neighbours.

Scanned +-6 s at 0.5 s steps around all three under-basket misses, asking both
models for a rim at 0.25 and testing whether the frame that yields one is the
SAME SHOT as the scored frame:

    115 (2887 s)  no frame within +-6 s yields a rim at all
    231 (5787 s)  seven frames do, at +3.0 to +6.0 s -- every one of them a
                  DIFFERENT SHOT, 0 ORB inliers against the scored frame
    35  (888 s)   one, at +5.5 s, also a different shot

So there is nothing to carry. For the whole duration of these shots no model
here finds a rim on any frame, which is a stronger statement than "it misses
this frame" and it rules out the cheapest fix rather than leaving it as a
maybe.

## Round 89 - verification moves the ball, and the seven hard frames get names

The fourteenth approach stops ranking candidates and asks a different question
about each one. Proposals come from the wide sources; each is scored by the
PRECISE model on a 640 px crop CENTRED on it -- which is exactly the
distribution `ball_clean` was trained on, with the clutter gone.

    delivered by pooled confidence   4/13   0.308
    delivered by verification        5/13   0.385

### What the seven unrecognised frames actually are

They had been described only by how they fail. Rendered at 2x with the truth
ball circled, they are not typical broadcast balls at all:

    1062 s   EXTREME CLOSE-UP, the ball over 100 px across behind a player's
             hands and head
    1638 s   extreme close-up, the ball filling the frame as leather texture
    1044 s   a loose-ball scramble, the ball on the floor among four bodies
    1056 s   held low between two players' legs, mostly occluded
    1390 s   in flight against a dark arena background, low wide camera
    2862 s   under the basket against dark crowd
    812  s   on the floor at distance among feet

The first two are the SAME PATHOLOGY AS THE RIM'S REMAINING MISSES.
`build_ball_detector_dataset.py` cuts crops without resizing precisely because
the ball is 15-25 px and "its whole difficulty is its size" -- so a ball over
100 px across is as far outside that distribution as a 500 px rim is outside
the rim model's. Both objects fail on close-up cameras for the same reason, and
neither failure is a threshold.

Downscaled inference confirms it and half-fixes it. Run over imgsz 192-800, the
FOUR-CLASS detector finds 1062 s at imgsz 416 (conf 0.16) and 2862 s at 800
(conf 0.28), while ball_clean finds neither at any size -- it has never seen a
big ball. Adding downscaled configurations to the pool lifts the ceiling:

    pool without downscaled configs    ceiling 11/13
    pool with them                     ceiling 12/13 = 0.923

### And adding the four-class model as a VERIFIER makes it worse

The obvious next step was to let the four-class detector endorse proposals too,
since it can recognise the big balls. Measured:

    endorsement by ball_clean alone           5/13   0.385
    endorsement by both models, multi-scale   4/13   0.308

It endorses a head at 0.74 and knocked 737.5 s from rank 1 to rank 2. The
precise model is precise, and diluting it with the model that proposes 112
candidates a frame gives back exactly what verification bought. Kept:
ball_clean alone.

**So the ball stands at 0.385 delivered against a 0.923 ceiling**, and the gap
between them is one thing: on five frames ball_clean scores the true ball 0.00
even centred at conf 0.01. Verification can only re-order what the precise
model can recognise. Every remaining point is the detector's.

## Round 90 - mining close-up balls fails, and the ball's ceiling is below the gate

Round 89 isolated the ball's remaining gap: on five frames `ball_clean` scores
the true ball 0.00 even centred at conf 0.01, and two of them are extreme
close-ups where the ball is over 100 px. The dataset builder says "CROPS ARE
TAKEN WITHOUT RESIZING, which is the point", so the model has never seen a ball
that size. Same pathology as the rim's close-up misses.

The mine looked sound: the FOUR-CLASS detector finds those balls when the frame
is DOWNSCALED, because shrinking a 110 px ball puts it in the 15-25 px band
that detector knows. Run it small over the game, keep boxes over 42 px in
original pixels, require two inference sizes to agree, and the labels should be
sharp close-up balls rather than the blurry upscaled ones crop-and-zoom makes.

**Every one of the first 24 samples was a face.** Bill Russell in a documentary
still, the anthem singer, a hand holding a trophy. Two causes, one of them mine:
the scan started at t=0 and its whole sample came from the pre-game montage,
where there is no basketball; and downscaling is precisely the operation that
removes what tells a head from a ball, so agreement across sizes corroborates
nothing -- a head is a stable object and every size finds it.

Restricted to the game span, with a head-position filter that refused 835
candidates, the second run is better and still unusable: of 24 sampled, 2 to 4
are real balls. The false population has simply moved -- from faces to RED
ADVERTISING BOARDS and the rim assembly, the same red horizontal objects that
put a State Farm hoarding into the rim model in Round 85. A recurring-position
filter removes only 24% of them, because the camera pans and an advert's image
position moves with it.

### The arithmetic that ends this line of work

Of the seven frames the ball detector cannot see, only TWO are close-ups. The
rest are a loose-ball scramble, a ball held between two players' legs, a ball
in flight against a dark arena, and a ball on the floor among feet. So even a
perfect close-up fix moves the ball from 5/13 to 7/13:

    ball today                  5/13   0.385   CI 0.177-0.645
    + both close-ups fixed      7/13   0.538   CI 0.291-0.768
    the POOLED CEILING         12/13   0.923   CI 0.667-0.986

**The ceiling is below the gate.** 0.923 is what an ORACLE would score choosing
from every detector at every scale pooled -- 230 candidates a frame. 0.95 is
above it. No selector, ranker or verifier can reach a frame where no candidate
within tolerance exists, and on this 1280x720 broadcast one such frame in
thirteen is what the pool leaves.

**Phase 2, as measured:**

    object   located   accuracy   95% CI          gate
    rim       55/59     0.932   0.838-0.973      0.95   in game
    rim       55/61     0.902   0.802-0.954      0.95   whole video
    ball       5/13     0.385   0.177-0.645      0.95
    ball ceiling, everything pooled, oracle-chosen   0.923

## Round 91 - 196 hand labels, two more training runs, and the rim is done

The user labelled 398 frames and I labelled 192, giving 196 hand-located rims
and 375 frames confirmed to hold none. Two runs followed. Neither shipped.

    model                                  held-out   false    main-camera
                                             rims     alarms   frames lost
    shipped (scale crops only)               3 / 7       0          0
    + 676 propagated from the hand labels    2 / 7       1          1
    + those, plus 474 negative crops         1 / 7       0          2

Recall falls monotonically as labels are added. The negatives did exactly what
they were predicted to do -- the ESPN scoreboard false alarm is gone -- and
cost another real rim doing it. That is Round 82's finding a second time, with
237 negatives instead of 68: THE NEGATIVES SUPPRESS THE THING THEY SHARPEN.

### The validation number was measuring memorisation

mAP50 read 0.864 while the held-out rims fell. Checking the split:

    propagated source instants: 429 train, 68 val, 0 files in both
    each val frame's nearest TRAINING frame:
        under 0.5 s: 60 of 68     under 2 s: 67     under 5 s: 68

Zero file overlap, so the split looks clean, but 60 of 68 validation frames
are within HALF A SECOND of a training frame -- the next frame of the same
replay. The split is by file and the duplication is by scene. Any future run
here must split by ANCHOR, not by frame.

### Why more labels made it worse

`mine_big_rims.py` queues candidates by ORANGE COLOUR and flattened shape, so
training on what it queues enriches the positives in orange blobs and the model
drifts toward "orange blob is a rim". In a 640 px crop around a rim an ESPN
scoreboard almost never appears; in a 1280x720 broadcast frame it always does.
The bias is therefore invisible in training and validation and fires only at
inference on whole frames, which is where it was found, at 0.80 confidence.

And 676 labels is not 676 scenes: they come from 48 anchors, many a second
apart, so a few dozen viewpoints at weight 3 crowd out 8,256 diverse crops.

### Stopping here, and why it costs little

Seven training runs, seven honest measurements, one unchanged shipped model.
What is worth saying plainly is that this never fed the goal much anyway:

- `shot_detection.py` does not detect the rim. "The rims never move -- they sit
  5.25 ft from each baseline on the centre line of a 94 x 50 ft court, with the
  hoop 10 ft up. Nothing needs detecting." It works in court coordinates.
- No grid version has ever been built with `--rim-detections`, so the
  scale-trained rim model has never been in the delivered pipeline at all.

The rim's real jobs are anchoring frames where court landmarks fail and
cross-checking the pose, and the system-level rim already reads 0.932 in game.

**Kept from this round:** 196 hand-located rims and 375 confirmed negatives,
committed and reusable; a browser labeller; a colour-and-shape miner that
raised labelling yield from 1-in-20 to 5-to-9-in-20; and a measured statement
of what mining by colour does to a model trained on it.

**Phase 2, unchanged:**

    object   located   accuracy   95% CI          gate
    rim       55/59     0.932   0.838-0.973      0.95   in game
    ball       5/13     0.385   0.177-0.645      0.95

## Round 92: the registration coverage wall was a default nobody had examined

Two Wave-0 diagnostics, no training, no labels. Both overturned a recorded
conclusion in this document.

### The 0% was a small-sample zero

Round 41 onward reported registration coverage per band as

    a lot of floor (>30%)          70%
    a normal wide play shot         0%
    a little floor                  0%
    almost none (close-up, replay)  0%

measured with `check_registration_coverage.py --every 60`, which is about twenty
frames a band. A 0/20 has a Wilson upper bound near 16%, so those zeros could
never have meant what they were read to mean. At `--every 10` -- 630 frames,
thirty times the sample -- the normal-wide-play band is **42%**.

### And the knob that mattered had never been turned

The same section concluded *"the failure is not a threshold"* on the evidence
that dropping the **per-keypoint** floor from 0.6 to 0.3 moved coverage 75.0% to
76.6%. That is true and it is the wrong knob. This is a pose model: it emits
keypoints only for a **detected court instance**, so a court scored below the
detection threshold yields not a few landmarks but none. That is exactly the
bimodality the same section reports and treats as evidence of generalisation
failure -- 0 landmarks on 45% of frames, 6+ on 55%, literally 0% at 4-5.

Ultralytics' default detection floor is 0.25. On the same 630 frames:

    band                          n     0.25     0.001
    almost no floor              25       4%       52%
    a little floor               46       2%       50%
    a normal wide play shot      76       1%       42%
    a lot of floor              483      70%       89%
    ------------------------------------------------------
    whole game                  630      54%       79%

### Is the new coverage real, or hallucinated landmarks?

The question the coverage number cannot answer, and the reason
`check_registration_consistency.py` exists: two independent registrations of one
instant, ORB-carried, must agree. No annotations, nothing scored against a model.

    broadcast          0.25              0.001            change
    Finals G7      71.1%  0.48 ft    81.1%  0.55 ft    +10.0 pts, +0.07 ft
    Finals G1      61.3%  0.72 ft    74.2%  0.79 ft    +12.9 pts, +0.07 ft
    ECF G1         75.5%  0.62 ft    86.4%  0.67 ft    +10.9 pts, +0.05 ft

Ten to thirteen points of coverage on every broadcast for five to seven
hundredths of a foot, against a gate of 2 ft. The p90 moves 2.35 to 3.13 ft on
G7, so the tail is genuinely worse -- and the whole of that cost is paid at 0.10
and then flat, which is why the floor goes all the way down rather than stopping
half way.

The floor was chosen on **Finals G7** under a rule fixed before any number was
read -- maximise coverage subject to p50 <= 1.0 ft, half the gate -- and Finals
G1 and ECF G1 are the held-out report. It replicates on both.

`court_keypoints.COURT_DETECTION_CONF` is now that number, in the library, with
the table beside it, because nine scripts each called the pose model with its
own threshold and this finding would otherwise have to be made nine times.

### What is left for the augmentation retrain

Less than was thought, and the mechanism is confirmed. Even at 0.001 the model
finds **no court at all** on 44-55% of tight shots, against 0-4% that find a
court and too few landmarks. That is out-of-distribution rejection, not a
localisation failure, and it is what scale and crop augmentation is for. But the
band it has to improve starts at 42%, not 0%, and whole-game coverage starts at
79%, not 75%.

## Round 93: the ball's 13 points are real, and the path does not recover them

`eval_ball_selection.py` measured the headroom: on 130 uniformly sampled frames
the detector proposes a candidate within 28 px on **91.5%** and the pipeline
reports the right one on **78.5%**. Thirteen points of pure selection.

`src/courtvision/ball_track.py` has held an exact Viterbi selector, with a
passing unit test, since before that gap was measured, and nothing had ever
called it. `scripts/eval_ball_temporal.py` calls it.

### It does not work, and the constants say so themselves

Windows of 9 frames at 1/15 s, the two path constants fitted on the **hard** half
of each game and reported on the **uniform** half:

    game            oracle   argmax   viterbi   paired (exact McNemar)
    Finals G7        92.7%    87.8%     87.8%   0 vs 0
    Finals G1        85.3%    64.7%     67.6%   1 vs 0,  p = 1.00
    ECF G1           94.5%    80.0%     80.0%   1 vs 1,  p = 1.00
    ------------------------------------------------------------------
    pooled           91.5%    78.5%     79.2%   one frame, not significant

The fit chose `move_weight` of 0.0005 to 0.002 on all three -- the bottom of the
grid. **The best thing the fitted model can do with the smoothness prior is
switch it off**, at which point the Viterbi degenerates into the per-frame
argmax it was meant to beat. With the constants as written (0.02, tuned for
10 fps) it is far worse: 63-71%, and significantly so.

### Why, measured

On the 17 frames across three games where the ball IS proposed and argmax picks
something else -- the exact frames the selector exists to fix:

    the real ball, nearest candidate in an adjacent frame   median  90.5 px
    the decoy                                               median   6.5 px
    the decoy is the smoother of the two on                 13/17 = 76%

`ball_track.py` opens with *"a ball moves smoothly and a false positive
teleports."* **On this detector's failures that is backwards.** The false
positives are stationary things -- a head, a shoe, a logo, 6.5 px of apparent
motion -- and the frames where the ball is hard to see are precisely the frames
where it is in flight at 90 px between samples. A prior that rewards not moving
prefers the decoy, and the fit discovering `move_weight = 0` is that fact
arriving through the optimiser.

### What this says to try instead

Not a smoothness prior but a **motion model that expects the ball to move**: fit
a ballistic or constant-velocity track and score a candidate by its residual
against the predicted position, so the flying ball is cheap and the stationary
decoy is expensive. That is the same information used with the opposite sign,
and it is the version this measurement supports rather than refutes. The 13
points are still there and still a selection problem.

Kept: `eval_ball_temporal.py`, the per-game uniform and hard truth files, and
the measurement above.

## Round 94: the 85% bar, measured on video for the first time

Every previous answer to "does this reach 85%" came from 25 Hz SportVU tracking
coordinates with perfect ball height and stable player identities. This document
said so at Round 19 -- *"the vision gap is unmeasured"* -- and its last statement
of the ledger says the real-video version is *"not close"*. There was no
composed metric, and no event-weighted-F1 code anywhere, so **neither 0.751 nor
0.879 was reproducible by anyone, including us.**

`scripts/score_game_end_to_end.py` is that measurement.

### The headline, on Finals G1 -- the broadcast nothing was tuned on

    mode                 (a) plays captured        (b) what it says is true
    vision                0.228  (0.208-0.251)      122/286 = 0.427
    vision + clock        0.279  (0.257-0.296)      121/200 = 0.605
    vision + scoreboard   BLOCKED -- no reader exists in this repository
    feed-assisted         1.000  (0.998-1.000)      tautological, not reported

and on Finals G7, labelled because `detect_shots`' thresholds were tuned on its
first half and the live-play rule informed by its second:

    vision                0.181  (0.170-0.203)      111/317 = 0.350
    vision + clock        0.231  (0.207-0.246)       90/174 = 0.517

**Against the 85% bar: no, on both readings, by a wide margin -- and now it is
measured rather than simulated.** 0.23-0.28 captured against a tracking-data
ceiling of 0.751; 0.43-0.61 said-and-true against 0.879.

### Why (a) cannot be large, arithmetically

A shots-only system can capture at most the share of a game's plays that ARE
shots. On these broadcasts:

    field_goal 0.38-0.43   rebound 0.25-0.26   free_throw 0.07-0.10
    foul 0.10-0.12         turnover 0.07-0.08  steal 0.03-0.05   block 0.02-0.03

The printed `architectural coverage` is that number -- 0.43 on Finals G1 -- and
every class the mode cannot emit keeps its full weight in the F1. So even a
perfect shot detector caps reading (a) at 0.43, and the remaining gap to 0.751
is not a better model, it is five classes nothing currently emits.

### The scorer validates against known answers before reporting unknown ones

- `feed-assisted` must be 1.000, and is. It is the official record placed on the
  video by the clock reader, so it can only be wrong about TIMING, and the
  printer refuses to report its precision as an accuracy at all -- it prints
  p50 0.00s, p90 0.00s, within 1 s 99% instead.
- `vision+clock` on Finals G1 reports field-goal precision **0.605**, which is
  the figure `detect_shots` records for this broadcast to three decimals. The
  composed number reproduces its component.

Two bugs the validation caught before any vision number was published:

  THE FEED MODE SCORED 0.622 ON FREE THROWS. Truth is merged into trips -- a
  frozen clock puts a whole trip at one video second -- and the feed stream was
  still emitting one call per ATTEMPT. A mode that is right by definition is for
  exactly this.

  THE BOOTSTRAP INTERVAL DID NOT CONTAIN ITS POINT ESTIMATE. It was resampling
  the per-play capture vector, which is a bootstrap of recall wearing an F1's
  label. It now resamples whole 120 s blocks of time with their calls AND their
  plays travelling together, which is what preserves the matching.

### What the ladder says about where accuracy comes from

Gating on the clock is worth **+0.05 captured and +0.18 precision** on Finals
G1, and it is the only lever in the ladder that currently works. The rung above
it -- the scoreboard, the one architecture this document ever measured at 85% --
**cannot be run**: `scoreboard_events.py` records F1 0.918 for any make and
0.864 for free throws, `outputs/broadcast/timeline.json` holds 449 events from
that run, and **nothing in this repository writes the readings those came from**.
It is reported as blocked rather than as zero, because reporting it as zero
would hide that the best result this project has is currently unreproducible.

Writing `scripts/read_scoreboard.py` is now the single highest-value piece of
work in the repo, and it is a driver script rather than research:
`autoscoreboard.locate_scores()` and `score_change_times()` already exist.

### Guards, so the only way to raise this number is to improve the system

Nineteen tests, each pinning one:

- a class the mode never emits scores 0 and **keeps its full weight**
- precision cannot print without its coverage
- the live-play filter removes truth from the denominator as well as calls, so
  the circular cell is structurally unprintable
- identity is never a matching key
- matching is order-independent
- a blocked mode is reported as blocked, never as zero
- bootstrap weights come from the whole game, not from the resample

### What is still not established

The truth is itself vision-derived: official plays reach the video through the
same clock reader the live-play gate uses, so 2.7-5.4% never arrive and the two
error sources are correlated. This is capture *within the span the clock could
read*, and it is labelled that way everywhere it appears.


## Round 95: an adversarial check of Round 94, and what it broke

Round 94's scorer was handed to an independent verifier with instructions to
break it rather than confirm it. It found thirteen defects. Six changed
published numbers and one of them broke the claim the file was proudest of.

### The circularity guard did not guard

`score_game_end_to_end.py` opens by saying the live-play filter "restricts the
denominator too, so the circular cell cannot be printed". The mask was applied
to both sides, and that was not enough. Official plays reach the video ONLY
through the clock reader, so **truth cannot exist outside the readable span at
all** -- and masking a region where truth cannot exist removes nothing from the
denominator while removing every call in it from the numerator.

Measured: 31 of 286 vision calls on Finals G1 and 92 of 317 on G7 lay outside
the readable span, and every one was a miss by construction. The ungated mode
was being charged for them and the gated mode was not.

Every mode is now restricted to the readable span, with the live gate applied on
top, so the only difference between gated and ungated is running versus held --
a comparison the truth can occupy on both sides.

    Finals G1, vision      (a) 0.228 -> 0.244      (b) 0.427 -> 0.478
    Finals G7, vision      (a) 0.181 -> 0.203      (b) 0.350 -> 0.436

About a quarter of the "+0.18 precision from clock gating" was the scorer
crediting the gate for suppressing calls in a region the truth could never
occupy.

### And the paired test was silently skipping every comparison that mattered

It compared capture vectors BY LENGTH, and two modes under different masks have
different denominators, so the only pair that ever printed was vision against
feed-assisted -- trivially significant and an answer to nothing. Pairing is now
keyed by the play. What it says is sharper than the claim it replaces:

    Finals G1   vision vs vision+clock   0 vs 0   p = 1.00  (309 shared plays)
    Finals G7   vision vs vision+clock   0 vs 0   p = 1.00  (239 shared plays)

**The clock gate captures exactly the same plays.** Its entire contribution is
precision -- it changes what the system SAYS, not what it FINDS. Round 94 said
"+0.05 captured and +0.18 precision" and the first half of that was an artefact
of the two modes being scored on different denominators.

### G7's truth was eleven days stale, and fixing it vindicated a component

`outputs/aligned_events.json` was dated Sep 5; the clock resolver fix landed
Sep 11 and `outputs/clock/fullgame.json` was rebuilt then. Re-running the
alignment against the current clock gives **498/561 = 88.8%**, not the 545/576 =
94.6% this document has quoted since.

The other two reproduce to the event: Finals G1 537/552 = 97.3% and ECF G1
603/624 = 96.6%, both byte-identical on a re-run. So the pipeline is
reproducible and one artefact was stale. **The corrected headline is 97.3 /
96.6 / 88.8%.**

Scored on the fresh truth, G7's vision+clock field-goal precision is **0.563** --
exactly what `detect_shots` records for that broadcast. The verifier had flagged
the old 0.517 as evidence that "the composed number reproduces its component"
held only on G1. With the stale truth replaced it holds on both.

### The corrected table

    Finals G1, nothing tuned on it
      vision              (a) 0.244  (0.222-0.266)   (b) 122/255 = 0.478
      vision + clock      (a) 0.279  (0.257-0.296)   (b) 121/200 = 0.605
    Finals G7, thresholds tuned on its first half
      vision              (a) 0.203  (0.181-0.221)   (b)  98/225 = 0.436
      vision + clock      (a) 0.251  (0.235-0.266)   (b)  98/174 = 0.563

The answer to the 85% question is unchanged in substance and better grounded:
**no, on both readings, by a wide margin.**

### The smaller ones, each fixed

- **The dead-ball rate was wrong.** Every miss counted as clock-running whenever
  the mode was ungated: G1 printed 2.37 running / 0.00 held where the truth is
  1.14 / 1.85. It reads as though the dead ball were free, and the dead ball is
  46% of basketball-looking footage.
- **The bootstrap could not draw blocks holding calls but no plays** -- 22 of
  G7's false alarms, all pregame and halftime -- so its mean sat above its own
  point estimate.
- **"Weighting on attempts" was false**; both matching and weighting are on free-
  throw trips. Weighting on attempts would move the headline about 0.01. The
  claim is corrected rather than the code.
- **The order-independence test was vacuous**: its case passes under a list-order
  matcher too. It now uses a case that discriminates, and asserts that a
  list-order matcher would fail it.
- **`--require-names` was documented and did not exist**, and `require_names` in
  `match()` was a no-op. Both removed.

### What the verifier could not break

The arithmetic. It recomputed `sum(weight * f1)` from the JSON independently and
it equals the printed headline to four decimals, with weights summing to 1.000
in every runnable mode. `match()` is one-to-one with no double counting. The
architectural cap holds.

It also noted two things worth keeping in view rather than fixing. `feed-assisted
= 1.000` validates less than Round 94 implied -- it is built from the merged
plays, so it checks that `match()` pairs identical timestamps and little more.
And steal and block share an official row with the turnover or missed shot they
belong to, which deflates the headline by about 0.015; removing them would give
G1 vision 0.243 rather than 0.228. They stay in, declared, because they are
plays a commentary system is expected to say.

## Round 96: Round 92 is retracted. The coverage was bought with bad registrations

Round 92 claimed that lowering the court DETECTION floor from 0.25 to 0.001 buys
ten to thirteen points of registration coverage on every broadcast for five to
seven hundredths of a foot. An independent adversarial check took it apart, and
the central claim is wrong.

### The "+0.07 ft" was a pooling artefact

`check_registration_consistency.py` reported a median over every probe point of
every instant. The instants that BOTH floors admit register **bit-identically**
-- 0 of 64 on G7 changed by more than 0.01 ft, 0 of 19 on Finals G1, 0 of 83 on
ECF G1. So every point of the pooled difference came from the newly admitted
frames, diluted by the ~90% that did not move at all.

Measured alone, which is the only number the claim ever rested on:

    floor    newly admitted    their own p50    their p90
    0.15         3 instants        2.43 ft        3.83 ft
    0.10         5                 3.18 ft        4.59 ft
    0.05         6                 2.73 ft        4.48 ft
    0.001        9                 2.61 ft       31.00 ft

**Every floor tested fails the 2 ft gate on the frames it adds.** Individual
marginal instants on G7 run 0.36, 0.89, 2.29, 2.31, 2.39, 2.71, 3.72, 4.24 and
35.45 ft. On ECF G1 one reads 91.89 ft.

That is precisely the hallucinated-landmark hypothesis Round 92 said it had
ruled out -- using a statistic that could not see it. The p90 moving 2.35 to
3.13 ft was the tell, and Round 92 reported it and then explained it away.

`COURT_DETECTION_CONF` goes back to 0.25, and
`check_registration_consistency.py` gains `--marginal-against`, which measures
the newly admitted instants alone and prints FAILS when their own median clears
the gate. The floor cannot be lowered again without that number.

### Four more defects in the same round

- **The committed script could not run.** `check_registration_consistency.py:93`
  referenced an unbound `instance_conf` and raised NameError. The published feet
  were produced by a working version that was not what got committed, and no
  test imported the script. Fixed.
- **The guard is blind to 61% of the coverage gain.** It skips any instant
  failing `has_court` (wood >= 0.20). Joining the 630 coverage frames against
  that gate: frames the guard can see gained 61, frames it cannot see gained
  **96**. The tight-shot bands where the dramatic movement was claimed were
  never accuracy-checked at all.
- **Coverage was counted at four landmarks; production needs six and a RANSAC
  fit.** On production's bar the same 630 frames read **52.4% -> 69.0%**, not
  54% -> 79%.
- **The Finals G1 "replication" rests on four instants**, two of which ORB
  refused, so its "+0.07 ft" came from two measurements. ECF G1 (n=110) is a
  real sample; Finals G1 is not evidence either way.

### What survives Round 92

Two things, and they are the ones worth keeping.

**The 0% was a small-sample zero.** `--every 60` gives ~20 frames a band and a
0/20 has a Wilson upper bound near 16%. At `--every 10` the normal-wide-play
band is 42% at the low floor and 1% at 0.25. The old table's zeros were never
evidence of an absolute failure.

**The mechanism is out-of-distribution REJECTION, not poor localisation.** Even
at 0.001 the model finds no court at all on 44-55% of tight shots, against 0-4%
that find a court and too few landmarks. Scale and crop augmentation is the
lever, and it is untried. A threshold is not the fix -- which is the one thing
Round 41 got right and Round 92 talked itself out of.

## Round 97: the scoreboard reader exists, and the rung it unblocks is worse than advertised

`scoreboard_events.py` carried the best numbers in this project -- any make F1
0.918, free throws 0.864 -- and nothing in the repository produced its input.
`scripts/read_scoreboard.py` now does. Four things had to be wrong first, each
found by checking the output against the box score rather than by reading code.

### Four bugs, in the order they surfaced

**The far team's score is 324 px from the clock** and `locate_scores` searches
within 260. The near team was found and the far team never was, so the first
full-game run selected the shot clock and a clock fragment and reported a final
score of **4-17 against a true 110-111**.

**The clock's digit templates cannot read the score.** The premise of the script
was that a score is "the same font on the same panel", so the templates the
clock already learned would read it. They do not: score digits are 26x21 against
the clock's 19x17 with a heavier stroke, and matched against clock templates
"23" reads as "11". The SVHN reader in `digit_net.py` does read this font -- but
its confidence is calibrated on jersey crops and does not transfer, so correct
reads of 59, 79 and 92 come back at 0.27 to 0.33, under any sensible jersey
floor. The per-frame read is therefore deliberately weak and the SEQUENCE is
strong: a value must repeat before it is believed, can never fall, and cannot
outrun the game's own scoring rate.

**`clock_glyphs` keeps the TALLEST cluster**, and on a score region a 36 px blob
from the panel divider outranks the 21 px digits -- so "59" came back as a
single glyph reading "1". The digits are the largest group of SAME-HEIGHT
characters, which is what a number is.

**The jump bound was per READING and had to be per unit of TIME.** "A score
cannot rise by more than three in one possession" is true of consecutive
possessions and false of consecutive readings: a replay hides the panel for half
a minute and the next legible frame is legitimately eight points later.
Rejecting that pinned one team at 8 for a whole game while its own region had
been observed reading 110.

Region selection also had to move from a 90 s window to frames spread across the
whole game -- over ninety seconds nothing distinguishes a score from the shot
clock's tens digit -- and to rank by **rank correlation with time** rather than
by "never falls", because a reader that is right two thirds of the time makes
the true region fall too. A final tie-break on the SPAN of values seen is what
separates a correctly placed box from one offset thirty pixels that clips a
digit and still rises.

### Where it lands

    Finals G1, full game     read 107-110     official 111-110

One side exact, the other four points short, and 83 score changes against ~85
scoring plays. Good enough to be useful; not good enough to be trusted
unchecked, which is why the script prints the final score and tells the reader
to compare it with the box score before believing anything derived from it.

### And it found a bug in the module it feeds

Given real readings for the first time, `scoreboard_events.score_events` emitted
**zero events from 4,330 of them**. When a jump is too big to be one possession
it declined to emit -- correctly -- but did not move its baseline, so one early
hidden stretch pinned the comparison at a score the game had left behind and
4,254 later readings were discarded as implausible. The baseline now advances
without emitting. Two tests pin it.

### The rung, measured

    Finals G1          (a) plays captured     (b) what it says is true
    vision              0.244 (0.222-0.266)    122/255 = 0.478
    vision + clock      0.279 (0.257-0.296)    121/200 = 0.605
    vision + scoreboard 0.169 (at its own 5 s) 47/71  = 0.662
    feed-assisted       1.000                  tautological

**The scoreboard rung is no longer blocked and it does not beat the clock-gated
vision rung.** Two honest reasons before anyone reads that as a refutation of
`scoreboard_events`:

  IT IS SCORED AT ITS OWN TOLERANCE, and that matters more than any other
  parameter here. A human updates the panel after the basket, so field-goal
  precision reads 0.283 at 3 s and 0.761 at 5 s on the same calls.
  `run_broadcast.TOLERANCE_S` has been 5.0 for exactly this reason; streams now
  declare their own.

  ITS RECALL IS CAPPED BY CONSTRUCTION. The scoreboard can only see MAKES, and
  the `field_goal` class counts attempts -- 173 of them against 72 emitted
  events. Recall of 0.202 against attempts is not comparable to the docstring's
  0.918, which was an F1 against makes.

So the comparison to make is precision on what each says: **0.605 for gated
vision against 0.662 for the scoreboard**, with the scoreboard asserting an
outcome on 100% of its calls and vision on none of them. That is the first
apples-to-apples number this architecture has ever had, and the scoreboard is
ahead on it.

---

## Round 98: a fourth broadcast, a registry, and four numbers that were measuring the wrong thing

The question this round exists to answer is "does a new broadcast fit right in",
and the honest first finding is that nothing in the repository could tell you,
because adding one meant editing **four** files that genuinely had to change --
three carrying a hardcoded `GAMES` dict and `rebuild_detector_dataset.py`
carrying two more maps keyed the same way. (An earlier draft of this round said
"eighteen", counting twelve scripts whose `--video` default is
`data/raw_clips/fullgame.mp4`. Those are overridable on the command line and
never needed editing to add a game; the count was rhetoric and the honest
number is four. Two game-pinned defaults that are NOT overridable in practice
do remain: `detect_shots.py` and `pack_video_game.py --game-id`.)

### The registry, and why paths are derived rather than listed

`data/games.json` holds only what cannot be computed -- the video, the official
game id, a label, a one-letter clip prefix -- and `courtvision.games` derives
every artefact path from the key. A registry that stored
`"detections": "outputs/clip_detections_g1.json"` would be the same eighteen
strings in one file, and one of them would eventually point at another game's
cache with nothing to say so. Three paths ARE listed, per game, under
`published`. That is **four** keys, not three: `clip_dir`, `clip_index`,
`overlays` and `aligned`. The first three name files that are tracked and served
-- the page fetches `clips/overlays*.json` and the clip mp4s by name, so renaming
them to tidy a registry would be a registry breaking the site. `aligned` is a
weaker case and should be said plainly: those files are gitignored and no URL
points at them; they are listed only because they already exist under names the
derivation would not produce, and re-deriving them is a rename nobody has done
yet. A NEW broadcast is allowed none, and a test enforces that.

Loading the registry now refuses: two games sharing a clip prefix (clip
filenames would collide), two sharing a video stem (`data/raw_clips/fullgame.mp4`
and `data/games/fullgame.mp4` are different files and the SAME clock, detection
and scoreboard artefacts), two claiming one official game id, and a `published`
key that is not a path name -- a typo there used to fall through to the derived
path without a word.

### The fourth broadcast was already on disk

`data/games/FZAUuuuREg0_1080p.mp4`: 9,004 seconds, 1920x1080 at 59.94, 4.9 GB,
referenced by four files and by no pipeline. Reading its scorebug at t=2000
gives OKC 18, HOU 15, 1ST 2:24 at Toyota Center; the only game in three seasons
of OKC-at-Houston whose play-by-play stands at 15-18 with 2:24 left in the first
is **0022500581, 2026-01-15, final OKC 111 HOU 91.** Regular season, different
arena, different teams, 60 fps, a scorebug laid out unlike either Finals
broadcast.

### "Unseen" was a boolean for about an hour, and it was wrong

The obvious field is `unseen: true`, and an adversarial check found the claim
false in this repository's own words. `docs/continuous-game-accuracy.md` already
said **"Toyota Center is no longer a clean unseen arena -- it was diagnosed
on"**; four directories of hand-placed court labels sit on this broadcast, three
of them on this exact encode and one on its 720p twin; and
`court_register.SEARCH_MIN_SAMPLES`, `court_camera.FLOOR_LANE_LAB` and
`court_refine.PAINT_POLARITY` were each set, in as many words, "with those
values in view".

A boolean would have published an acceptance-test result over a registration
number the broadcast helped choose. So the field is now per arm:

    held_out   clock, alignment, clips, detector, handler, ball, possession,
               scoreboard, end_to_end          -- never touched by any of them
    NOT        registration, court_camera      -- diagnosed on; see above

`held_out_for(arm)` answers False for an arm it has never heard of. Guilty until
the registry says otherwise, because the failure being guarded against is
publishing a contaminated number as a hold-out.

### `scripts/add_broadcast.py`, and the two wiring bugs a dry run could not see

A stage graph: each stage declares what it produces, a stage whose outputs exist
is skipped, and nothing takes a path from the caller. Every flag it passes was
checked against the target script's own `add_argument` calls by a test, and every
flag was valid -- which is exactly why the two real bugs survived to be found by
reading what the receiving code does with the file.

  **The vision arm was fed the wrong file and reported zero without an error.**
  `score_game_end_to_end.vision_shot_stream` reads `blob["frames"]` and re-runs
  `detect_shots`' own rim tracking over them: it wants the WHOLE-GAME DETECTION
  CACHE. It was handed `detect_shots`' report, which has no `frames` key, so it
  built zero calls and would have printed a vision arm of 0.000.

  **A new broadcast's shot detector was scored against the 2025 Finals Game 7.**
  `detect_shots.py --shots` defaults to `outputs/shots_on_video_0042400407.json`
  -- one game's official shot times, with that game's number in the filename --
  and the driver did not override it. Every `agrees_with_official` label and the
  first/second-half split would have come from the wrong game. Worse, *nothing in
  the repository produced that file for any game*; both existing ones were made
  by hand. `scripts/official_shots.py` derives it from the cached play-by-play,
  and for Finals G1 its output is **content-identical to the hand-made file, 180
  attempts, 78 made** -- so removing the hand step costs nothing.

A test now fails if any stage leaves a default that carries a ten-digit game id,
and another fails if any stage's command mentions another registered game's id.

Two smaller ones from the same check. `_outputs_of` kept a second mapping of
stage name to outputs parallel to the stage list, and `all([])` is True, so any
name it did not know read as satisfied; it now asks the stage. And `done()` now
parses a JSON output rather than only stat-ing it: no stage writes atomically, so
a kill inside `json.dump` leaves a truncated file that `exists()` calls finished,
and the driver would skip it forever while the next stage died on it.

### `scripts/eval_by_game.py`, and four numbers it was getting wrong

The per-game report: clock coverage, alignment, clips and two-path registration
need no labels and a new broadcast gets them on arrival; handler and ball need
labels and print `0.00-1.00 NO DATA` without them, because `stats.wilson` answers
an empty denominator with the whole width and this is the caller that made that
matter.

An adversarial check found six defects, four of which moved printed numbers.

**1. The labelled arms pooled the training split into the headline.** Half the
possession round's frames were drawn BECAUSE the ball model had no confident
candidate there, and two thirds of the handler round because two methods
disagreed. Pooling them with the uniform frames moves every labelled number by
15 to 25 points and describes no population at all:

                            pooled   uniform     hard
    handler, winnable   G7       0.368     0.500    0.263
                        G1       0.330     0.564    0.172
                        ECF      0.419     0.657    0.233
    ball, any rank      G7       0.600     0.902    0.318
                        G1       0.613     0.824    0.357
                        ECF      0.812     0.964    0.480

(An earlier draft of this round printed the pooled column as 0.351 and 0.532.
Those were pooled AND scored by the old time-keyed frame matching that defect 3
below describes, so the table compared two things at once and was not the
like-for-like comparison it presented. Every number above uses the fixed
matching. The gap it closes is 13 to 30 points, not 15 to 25.)

The line labelled "THE CEILING of every selector built on this detector" was a
ceiling on a hard-case set. **The real ceiling is 0.824 / 0.902 / 0.964** across
the three broadcasts, against a top-1 of 0.647 / 0.829 / 0.800 -- so there are
**7 to 18** points available to selection (17.7 on G1, 16.4 on ECF, and only 7.3
on G7, which an earlier draft rounded away), where the pooled figure implied two.
That one correction re-ranks the roadmap, which is what the arm was put there to
do.

**2. The detector-miss rate counted frames with nobody to miss.** Frames answered
"nobody has it" -- ball in flight, loose, dead -- are 30.5% of all labelled rows
and 33.9% of the box/missing/nobody ones, and there is no handler in them to draw. On the uniform split with the right
denominator the miss rate is **18.8% / 6.3% / 9.1%**, which is exactly the
three-fold spread the module docstring cites and could not previously produce.

**3. A third of every labelled frame was scored against a neighbouring frame's
boxes.** Clips overlap and the caches sample every second frame at 30 Hz, so 52%
to 60% of rounded times hold two to six different cache rows, and keying on time
alone took whichever was written last -- and on 30.7% / 30.7% / 32.6% of frames
that was not the labeller's row. (Counting only times whose rows hold DIFFERENT
detections the figure is about 50%, with at most four distinct rows rather than
six; the six-row times are duplicates of one another.) The labelling pages record
the boxes they drew; those boxes are a fingerprint. Matching on them ties **100%
of labelled frames, on all three broadcasts, to the exact row the labeller saw** -- and the
first attempt matched every handler frame and no possession frame at all, because
one page sorts its boxes by x and the other keeps detection order.

**4. "clips within 1.0 s" was the alignment arm again.** `error_s` in the clip
index is copied from the aligner, where it is the gap to the nearest clock
READING -- so on two of three broadcasts the arm is exactly
`P(error <= 1 | located)`, a conditional slice of the row above it printed as
independent evidence. It is now marked descriptive, and replaced as evidence by a
question the report could not previously ask: **do the clips still match the
alignment on disk?** On Finals G7 they do not. 67.0% of indexed rows exist in the
current `aligned_events.json`; the clock resolver was fixed after those clips
were cut, and two rows of one report were being computed against two different
truths.

Also fixed: an event-weighted F1 was given a fabricated `hits=round(rate*1000),
total=1000` so it could live in the same class as a proportion, and that fake
denominator printed as `n=1000` and was written into the report JSON. It carries
its rate directly now and prints no n, because it has none.

### `stats.cochran_q` was not Cochran's Q

Cochran's Q compares k treatments on the SAME subjects -- the k-sample
generalisation of McNemar. What is compared here is k independent groups:
different games, different frames, no pairing. The right test is the chi-square
test for homogeneity of proportions, which is what the arithmetic always was. In
a module whose entire argument is that paired and unpaired comparisons are
different things, borrowing the name of the paired test was not a harmless label.
It is `homogeneity` now, with the old name kept as an alias.

Run across the three labelled broadcasts it immediately fires on Finals G7 --
Rebound 0.840 against 0.964 and 0.965, p = 0.0004; Missed Shot 0.809 against
0.961 and 0.968, p = 0.0001; Made Shot (3PT) 0.762, p = 0.025; Foul 0.889,
p = 0.029 -- which is the same stale alignment finding arriving from a second
direction. (An earlier draft of this round omitted Rebound, which is the second
strongest of the four.)

### The scoreboard reader's reach was a hand-fitted constant, twice

`locate_scores` ships 260 px and misses the far team by 64 px on a 720p Finals
broadcast. 460 was measured to fix that, and **misses the far team by 67 px on
the Houston broadcast**, whose far score sits 527 px from the clock. A constant
fitted to the layouts already in the repository is not a constant, it is a memory
of them. The reach was never doing the work anyway: what picks the score regions
is behaviour over a whole game -- non-decreasing, rank correlation with time at
least 0.85, a final value inside FINAL_RANGE -- and the reach is only a compute
budget on how many boxes get that test. It now searches to the frame edge, and
every candidate box size is a multiple of the clock's own measured height.

Two corrections to an earlier draft of this paragraph, both found by checking it
rather than by reading it. It does **not** reproduce the old boxes "exactly" on
720p: the ratios give widths 70, 88 and 110 where the constants were 70, 90 and
110, they add a fourth grow, and the x-grid starts at the frame edge rather than
460 px from the clock. And the premise "a 44-pixel clock on a 720p encode" holds
for only two of the three 720p broadcasts -- the ECF clock region is 36 px tall,
so the old fixed widths were already wrong for it and nobody had noticed. On the
Houston broadcast the locator's clock region is 28 px tall and the score digits
are **38** px, not the 52 an earlier draft claimed; 38 still does not fit inside
any box the old constants could build from a 28-pixel clock, which is the point,
but the measurement is 38.

### A lead this round created rather than closed: the ball ledger's denominators

Round 63 closes with a thirteen-item ledger of ball-selection ideas, every one
recorded as a failure, and the ledger is the reason no further selection work has
been attempted. Every entry in it was scored on **ten to thirteen frames**:
"delivered 3/10", "recall 6->7 of 10", "4 of 13". That is not a criticism of the
conclusions reached at the time -- it was the truth set that existed -- but it is
worth writing down what those denominators can resolve:

    3/10   = 0.300   95% interval 0.108-0.603   width 0.495
    4/13   = 0.308                0.127-0.576   width 0.450
    100/130 = 0.769               0.690-0.833   width 0.144

**An idea worth ten points could not have been distinguished from one worth
nothing on that set**, and a paired test on 130 frames resolves about three times
finer before the pairing is even counted. W0.1 rewired the ball truth from 13
frames to 135 this round, and the per-game report now reads the uniform half of
them -- 34, 41 and 55 frames on the three broadcasts -- straight out of the
detection cache with no detector pass.

So the honest status of that ledger is **not "thirteen ideas are dead"; it is
"thirteen ideas were tested with an instrument that could not see a ten-point
effect."** Two of them showed a learned signal before delivering nothing:
the 192 px context ranker (val AP 0.313 against 0.171, checkpoint still on disk
at `checkpoints/ball_context_ranker.pt`) and tiled inference (recall 6 to 7 of
10). Re-scoring those two on the uniform half, paired, is a GPU pass over 130
frames and costs almost nothing.

**This is a lead, not a result.** Nothing here says either one works; it says the
measurement that rejected them could not have detected it if they did. The gap
they would have to close is now known precisely: top-1 reads 0.647 / 0.829 /
0.800 against a proposed-at-any-rank ceiling of 0.824 / 0.902 / 0.964, so
selection has 14 to 18 points available on each broadcast. The pooled figure
this round replaced put that gap at two points, which is very likely why nobody
went back.

### A prediction registered before it is tested: the clock arm is sampling-limited

On the fourth broadcast the clock arm reads **0.897** -- the only label-free arm
under the 0.90 bar -- and the obvious reading is that three percent of the game
is illegible. The four broadcasts say otherwise:

    game   readings   running   per game-second   game-seconds seen   coverage
    G7        3707      2356          0.818             2127           0.739
    ECF       5113      2899          0.912             2558           0.804
    HOU       4067      2861          0.993             2584           0.897
    G1        4331      3045          1.057             2678           0.930

Coverage is very nearly **0.90 x readings-per-game-second**, across four
broadcasts, three arenas and two encodes. That is not a legibility curve, it is
arithmetic: `read_game_clock.py` samples at `--step 1.0` and the clock ticks once
a second, so a game second the sampler lands beside is a game second that cannot
be counted however clearly it was displayed. The reader is at 0.99 samples per
tick on Houston and it recovers 0.897 of the ticks.

**The prediction, written down before the run:** sampling at `--step 0.5`
roughly doubles samples per tick and should put coverage above **0.95 on every
one of the four broadcasts**, including the two currently at 0.74 and 0.80. If
it does not -- if Game 7 stays near 0.74 -- then the missing seconds really are
illegible and this explanation is wrong.

**Why this is not tuning on the hold-out.** The relationship was measured on the
three broadcasts the project has always had, the mechanism is sampling
arithmetic rather than anything about a scoreboard, and the same change is
applied to all four and reported on all four. A step chosen because Houston
scored 0.897, applied only to Houston, would be the other thing.

The cost is one more sequential decode per broadcast, which on the 1080p60 file
is about forty minutes.

### The alignment misses are not scattered: 39% of them are the same bug

The fourth broadcast aligned 561 of 566 rows. Five failed, and four of the five
are in the last eight seconds of a period. That is not a coincidence, and it is
not specific to this broadcast:

    game   unaligned   of      under 10 s left   share
    G7          63     561           17          27.0%
    G1          15     552            8          53.3%
    ECF         21     624           12          57.1%
    HOU          5     566            4          80.0%

**41 of 104 unaligned events across four broadcasts, in three arenas, sit in the
last ten seconds of a period.** The cause is one length gate. Under a minute the
NBA clock shows tenths; at `18.2` that is three glyphs, which `read_clock` turns
into `"1:82"` and the caller's parser turns back into 18.2 seconds. At `7.2` it
is TWO glyphs, and `if not 3 <= len(glyphs) <= 4: return None, 0.0` threw it
away.

The clock files say so exactly. Readings with the clock between 10 and 20
seconds: **79 / 34 / 342 / 25**. Readings under 10 seconds: **0 / 0 / 0 / 9**.
The reader has never seen the end of a period on any broadcast in this
repository, and nothing noticed because a missing reading is indistinguishable
from a replay.

**Upper bound on the fix, stated before it is measured:** if every one of those
41 events becomes alignable, G7 goes 0.888 to 0.918, G1 0.973 to 0.987, ECF
0.966 to 0.986 and Houston 0.991 to 0.998. It will be less than that -- some of
those frames are genuinely covered by a graphic -- and the point of writing the
bound down first is that the measured number has to be read against it rather
than against nothing.

**And the reader now saves its raw TEXT, not only the parsed values.** Without
that, every change to the parser cost a full sequential decode of the video --
forty minutes on the 1080p60 file -- to find out whether it helped, and a frame
whose text parsed to nothing was not recorded at all, so `--from-raw` could
never show what a better parser would recover. This round paid that cost twice
and it should not be paid again.

### Round 99: where the ball's missing thirteen points actually are

The lead recorded above said the ball ledger's thirteen rejected ideas were
tested with an instrument that could not see a ten-point effect. Before trying
any of them again, this asks a cheaper question: **on the uniform frames, where
is the right ball when the pipeline does not report it?**

No detector runs. The candidates come out of the clip detection caches -- the
boxes the labelling pages actually showed -- so this is selection measured on
exactly the evidence a person was judging.

    130 uniform located-ball frames, pooled over the three labelled broadcasts

        rank of the true ball        frames     share
        0  (reported today)             100     76.9%
        1  (second by confidence)        13     10.0%
        2                                 4      3.1%
        3                                 1      0.8%
        never proposed                   12      9.2%

        top-1                          0.769
        top-2 ceiling                  0.869    a perfect two-way discriminator
        any-rank ceiling               0.908    every selector ever built here

**Thirteen and a half points are available to selection and ten of them --
three quarters -- are a binary decision between two boxes.** Another 9.2% is not
a selection problem at all: the ball is not in the candidate list and no
re-ranking can find it. That is the split the pooled 0.53 figure this project
was quoting could not show, and it is the number that should decide where the
next round of effort goes.

### And the anchoring already in the pipeline does not close it

`clip_boxes.anchor_penalty` scores a ball candidate by confidence less what its
distance from the nearest person costs, and rejects one that is nowhere near
anybody. It ships -- it is what stopped the overlay path settling on a
stationary orange thing in the crowd -- and it had never been scored against the
ball truth. `scripts/eval_ball_anchor.py` does that, with its two constants
fitted on the HARD half and reported on the UNIFORM half:

    rule          pooled     n      G7      G1     ECF
    confidence     0.769   130   0.829   0.647   0.800
    penalty        0.777   130   0.829   0.647   0.818
    handler        0.777   130   0.854   0.676   0.782
    reject         0.777   130   0.829   0.647   0.818

    paired, exact McNemar against plain confidence:
      penalty   2 frames only it gets, 1 only confidence   p = 1.0000
      handler   4 frames only it gets, 3 only confidence   p = 1.0000
      reject    2 frames only it gets, 1 only confidence   p = 1.0000

**A fourteenth negative, and a useful one.** Geometry is not what separates the
ball from the box that outranks it: the decoys are already near people. On the
18 frames where the ball is proposed but not first, the confidence margin
against it has a median of **0.217** and is wider than 0.15 on **13 of 18**;
only 3 are inside 0.05. The detector is not narrowly confused, it is
confidently wrong about a specific kind of object. That points at appearance, and it says something the
ledger's entry 11 ("orange colour prior: true balls are LESS orange than the
false") already hinted at and nobody followed: the decoys are a *population*,
not noise.

The honest consequence for the roadmap is that the remaining ball work is a
two-class problem on about 18 frames per 130, which is exactly the regime where
the old ten-to-thirteen-frame evaluations could see nothing at all.

## Round 100: the registered prediction was wrong, and the ablation that showed it found a worse bug

The prediction registered two rounds ago: sampling the clock at `--step 0.5`
should put coverage **above 0.95 on every broadcast**, because coverage tracked
`0.90 x readings-per-game-second` across four games and the reader was sampling a
one-second clock once a second. It was written down before the run, with the
falsifying outcome named.

**It is falsified.** Game 7, re-read at 0.5 s with the tenths fix, reaches
**0.772**. Not 0.95, and not close.

### The four-way ablation, run offline for nothing

The reader now saves its raw OCR text, so one 20-minute decode answered four
questions instead of one. Each row below is the same 7,325 saved readings
re-resolved with a different parser and a different subsample:

    step   tenths   periods        game-seconds seen    coverage
    1.0    no       [1, 2, 3, 4]          2058            0.715
    1.0    yes      [1, 2, 3, 4]          2096            0.728
    0.5    no       [1, 2, 3, 4]          2184            0.758
    0.5    yes      [1, 2, 3, 4]          2222            0.772

    lowest clock reached per period, without tenths:  10.7  10.8  10.1  10.7
                                        with tenths:   0.0   0.0   0.0   0.0

**The tenths fix does exactly what it was built to do** -- every period now runs
to zero, where before the reader lost the clock at ten seconds in all four -- and
it is worth **1.3 to 1.4 points** of coverage. Halving the step is worth **4.3**.
Together they take Game 7 from 0.715 to 0.772, against a prediction of 0.95.

### Why the prediction was wrong

The unseen seconds are not scattered:

    659 game-seconds unseen, in 83 runs
      38 singletons        a denser sampler could catch these
      56 in runs of 2-3
     565 in runs of 4+     the longest 124 s, then 95, 56, 50, 43, 37, 30
                           -- the scorebug is simply not on screen

**86% of what is missing is in stretches where the clock is not being shown at
all**: timeouts with a full-screen graphic, replays, the between-quarters break.
Sampling faster cannot recover a second the broadcast never displayed. The
correlation across four games was real and I read it as a mechanism; it is two
consequences of one cause, because a broadcast that shows the bug more has both
more readings and more coverage. Going to `--step 0.25` would be worth at most
the 38 singletons, about 1.3 more points, for another doubling of decode.

The ceiling on this arm is therefore a property of broadcast production, not of
the reader, and it sits somewhere near **0.80 on Game 7**. Whether the other
broadcasts have a higher one is now a question with a cheap answer, because the
raw text is saved.

### And the ablation found a worse bug than the one it was testing

The first re-read resolved Game 7 to **six periods**. Game 7 had four.

`"5:43"` is 343 seconds or 54.3, and the tenths reading of a HELD clock is
self-consistent frame after frame. At video 6071.0 s the clock is held at 5:43
through a stoppage; one frame reads `"5:48"`; neither of its readings continues
from 343, so the resolver falls through to its confirm-against-the-next-frame
rule -- and **confirms 54.8**, because the next frame's `"5:43"` offers 54.3,
which continues from 54.8 perfectly. Every later reading in the quarter followed
it down. The last five minutes of the fourth resolved to tenths and
`assign_periods` filed them as an overtime the game never played.

This is the failure the `resolve` docstring says was fixed for Finals G1 in an
earlier round. **It was not fixed, it was made rarer** -- and a 0.5 s step, by
sampling twice as many frames, made it likelier again.

The fix is a physical fact rather than another tie-break: the clock only
DISPLAYS tenths under a minute, so a reading below 60 while the clock is above
65 is not a misread to be weighed against another, it is impossible. Dropping
those before continuity is considered closes the class. Verified against the
saved text four ways, with no video pass: every configuration above now resolves
to four periods.

**Two lessons worth keeping.** A prediction written down in advance cost one
decode and bought a correct model of the arm's ceiling; without it the 0.772
would have read as progress. And the guard added *before* the experiment -- "a
sub-ten-second reading may never start a run" -- caught one spurious period but
not this one, because this misread does not start a run, it hijacks one.

### What an unseen arena does to the ball detector: seven times the candidates

The first two clips detected on the Houston broadcast, against the two Finals
broadcasts in full:

    broadcast            frames    ball boxes/frame    above 0.25 conf
    HOU 1080p               180        13.9                3.98/frame
    Finals G1 720p       20,880         2.0                0.71/frame
    ECF G1 720p          21,780         1.9                0.72/frame

**Seven times as many ball candidates per frame, and five and a half times as
many confident ones**, on an arena and an encode the detector has never seen.
The share above 0.25 confidence is actually LOWER (28.7% against 34.8% and
37.2%), so this is not a threshold effect -- it is more proposals at every
confidence.

This is the kind of thing an acceptance test exists to surface, and it has a
direct consequence for the roadmap: **the ball selection work measured two
rounds ago -- top-1 0.769 against a proposed-at-any-rank ceiling of 0.908 -- was
measured on candidate lists two boxes long.** On this broadcast they are fourteen.
Every selection rule fitted on the Finals footage faces a different problem here,
and the 10 points that Round 99 located in a binary choice between two boxes is
not a binary choice on this game.

Nothing is claimed about how the ball number itself moves, because Houston has
no ball labels and the report prints that arm as NO DATA rather than guessing.
What is claimed is narrower and is measured: the input to selection is seven
times noisier, and no selector in this repository has ever been shown footage
like it.

## Round 101: the fourth broadcast runs end to end

    scripts/add_broadcast.py --game hou

Game 0022500581 -- OKC at Houston, 2026-01-15, 1080p60, a third arena, a
scorebug neither Finals broadcast has, and **nothing in this repository tuned on
it for any event arm**. Every path it took came from `data/games.json`; no
constant was edited for it; the only thing supplied by hand was the registry
entry naming the video and the official game id.

    arm                              rate       n      verdict
    clock: game seconds seen        0.897    2880      FAIL
    alignment: overall              0.991     566      PASS  (every class passes)
    clips: still match alignment    1.000     486      PASS
    registration: agreement         0.905      21      PASS (point)  NOT a hold-out
    handler, ball                     --        0      NO DATA -- unlabelled

    end to end                   captured   interval     emitted  said & true
    vision                         0.273   0.244-0.292      303      0.502
    vision + clock                 0.301   0.283-0.321      253      0.589
    vision + scoreboard           BLOCKED -- see below
    feed-assisted                  1.000   0.995-1.000      431      tautological

### The comparison that matters

Finals G1 is the previous headline and is a game whose thresholds were chosen
partly on itself:

                            Finals G1 (tuned)   Houston (unseen)
    vision, captured              0.244              0.273
    vision, said and true         0.478              0.502
    vision+clock, captured        0.279              0.301
    vision+clock, said and true   0.605              0.589
    alignment                     0.973              0.991

**The unseen broadcast captures more of its plays than the tuned one on every
vision mode, and says true slightly less often on the gated one.** The shot
detector on its own agrees with the official record on 149 of 253 calls, against
0.563 recorded on Game 7. Nothing here generalised badly, which is not what the
three previous rounds of out-of-distribution findings predicted.

### What it does not clear, and why

**Clock coverage, 0.897.** The only label-free arm under the bar, and Round 100
established what limits it: 86% of the unseen game-seconds are stretches where
the scorebug is not on screen. The tenths fix landed after this run and is worth
1.3 to 1.4 points; a re-read is in flight.

**The scoreboard rung is BLOCKED, not zero.** Its region search found the two
team panels reading the correct final scores -- 111 and 91 against an official
111-91 -- and rejected them, which took three separate fixes to diagnose: box
sizes fitted to two Finals broadcasts that could not build a window big enough
for a coloured team panel, and a rank-correlation gate defeated by frames where
a single stray glyph wins the same-height vote. The run is still in flight.

**Vision capture is CAPPED at 0.44 by architecture** -- that is the share of this
game's plays that are shots -- and reaches 62% and 67% of that cap. A report
that printed FAIL there would be calling a design limit a defect.

### What "90% across the board" means on this evidence

Three arms clear it and one does not:

    alignment                0.991   PASS, and every action class passes
    clips match alignment    1.000   PASS
    feed-assisted capture    1.000   PASS
    clock coverage           0.897   FAIL, ceiling near 0.80-0.90 by production

and the vision-only arms cannot, by arithmetic rather than by effort: a stack
that emits only field goals can capture at most 44% of a game's plays however
good it gets. That number is printed beside every one of them.

**The pipeline that ships -- the feed-assisted one -- is at 0.991 alignment and
1.000 capture on a broadcast it had never seen.** The pipeline that decides what
happened from pixels is at 0.301 capture and 0.589 precision. Both are true and
the report prints them on the same page, which is the whole reason it exists.

### The shot detector, alone, on footage it has never seen

    broadcast                      P        R       F1     calls
    Houston (unseen)            0.589    0.788    0.674     253
    Finals G7 (held-out half)   0.563    0.778    0.619       -

Against 189 official field-goal attempts placed within 2 s of a clock reading,
vision matched 149 distinct instants. **Every one of the three figures is better
on the broadcast nothing was tuned on**, which is not what 13.9 ball candidates a
frame -- seven times the Finals density -- would have predicted.

It is also steady through the game, which rules out the obvious alternative
explanations:

    period   calls   agree   precision
      1        65      39      0.600
      2        62      39      0.629
      3        63      33      0.524
      4        63      38      0.603

    first half   118 calls, P 0.627
    second half  135 calls, P 0.556

No drift, no clustering, no period where it falls apart. The false calls are a
steady rate rather than a failure in one part of the broadcast, which is
consistent with the candidate density being higher everywhere rather than in
some particular lighting or camera state. Timing of the calls that agree: p50
0.90 s, p90 2.20 s, inside the 3 s tolerance but not tightly.

**The `--tune` flag was not used**, here or anywhere in `add_broadcast.py`. The
approach, far and merge constants are the ones fitted on Game 7's first half in
Round 40-something, applied unchanged.

### Round 102: the tenths fix is worth +3.4 points of alignment, and it cost one re-resolve

Round 99 predicted an upper bound for the period-end blind spot: if every one of
the 41 unaligned events in the last ten seconds of a period became alignable,
Game 7 would go 0.888 to 0.918. Measured:

    Game 7 alignment          located / rows      rate
    original clock file          498 / 561       0.888
    step 1.0, every fix          520 / 564       0.922     <- +3.4 points
    step 0.5, every fix          524 / 564       0.929     <- +0.7 more

**Above the bound**, because the bound only counted events in the blind spot and
the tenths-capture guard also repaired the five minutes of the fourth quarter
that had resolved to tenths and been filed as a nonexistent overtime.

**And halving the sampling step is worth 0.7 points against the fixes' 3.4.**
Every broadcast stays at `--step 1.0`; doubling the decode of every game for
seven tenths of a point is not a trade worth making, and now there is a number
for it rather than an intuition.

The whole measurement cost **one re-resolve from saved text and no video pass**,
which is the first return on saving it.

### Every broadcast's alignment, after

    G7    0.922   n=564    (was 0.888)
    G1    0.973   n=552
    ECF   0.966   n=624
    HOU   0.991   n=566

    chi-square homogeneity, per action, four games:
      Rebound       p 0.0015   g7 0.887 against 0.964, 0.965, 0.992   HETEROGENEOUS
      Missed Shot   p 0.0040   g7 0.889 against 0.961, 0.968, 0.991   HETEROGENEOUS
      ...and twelve other action types no longer differ across the four games.

Before the repair, **four** action types were heterogeneous and every one of them
was Game 7 dragging the others down: Rebound 0.840, Missed Shot 0.809, Foul
0.889, Made Shot (3PT) 0.762. Foul and the three-pointers are now within the
spread. The two that remain are still Game 7, and its clock coverage is still the
worst of the four at 0.728 against 0.897 to 0.930 -- the broadcast simply shows
its scorebug less, and Round 100 measured that as the binding constraint.

### The cost of repairing Game 7's clock: its published clips are now half stale

Repairing the clock moved events, and Game 7's clips were cut against the old
one. The `clips: still match the alignment` arm -- which did not exist a day ago
-- reads **0.513**, down from 0.670 before the repair. That is an exact-instant
test, so it overstates the damage; measured as distance from each indexed clip to
the nearest event of the same action in the alignment as it now stands:

    within 0.05 s   0.513      the clip is exactly where the play is
    within 1 s      0.725
    within 3 s      0.797
    p90             41 s
    p95            122 s

**About 20% of Game 7's published clips are more than three seconds from any
play of the action they claim, and 10% are more than forty seconds away.** Those
are rows the old alignment placed wrongly; the new one does not put an event
there at all.

The fix is to re-cut them, which is thirty minutes of ffmpeg and a 454-file
binary diff in a directory that is **tracked in git and already 12% of what a
clone costs**. The plan has that media moving to object storage and the history
rewritten, and re-cutting into git first would make both jobs larger. So this is
recorded rather than fixed, with the number attached: **the arm that found it
prints 0.513 on every future report until it is done.**

The fourth broadcast's clips are at 1.000 against its own alignment, because
they were cut from it.

### The fourth broadcast, after the clock repair

    arm                          before    after
    clock: game seconds seen      0.897    0.8997     2591 of 2880
    alignment: overall            0.991    0.995      563 of 566
    clips: still match            1.000    0.986      7 rows moved

**0.995 alignment on a broadcast nothing was tuned on**, and three of its four
periods now read down to 0.0 s where the reader previously lost the clock at
12.7, 19.4 and 19.8 seconds. The fourth ends at 1.7 s because the game ended
with the clock running.

The clock arm reads **0.8997 against a 0.90 bar -- three game-seconds short of
it**, on an interval of 0.888 to 0.910 that straddles the bar in both
directions. It prints FAIL, and that is the right thing for it to print: the
point estimate does not clear the bar and the interval cannot settle it either
way. What would settle it is not a better reader. Round 100 measured the
remaining gap and 86% of it is stretches where the scorebug is not on screen at
all.

## Round 103: three ratchets in one day, and the regression that caught the third

Every one of these is the same mistake: **a constraint adopted from a single
observation and never released.** They were found in three different files,
hours apart, and only the third made the pattern obvious.

**One. The clock's tenths capture.** `"5:43"` is 343 seconds or 54.3. At a
stoppage the held clock reads the same value frame after frame, so the tenths
interpretation is self-consistent; one frame misreading `"5:48"` was confirmed
by the next frame's 54.3 and the rest of the quarter followed it down. Fixed by
a physical fact: the clock only displays tenths under a minute.

**Two. The score's monotonic maximum.** `monotonic` adopts a new maximum from
three identical readings, and a systematically misaligned box produces the same
wrong number over and over. On Houston the ratchet locked at 116 from video
6754 s and every later correct reading was rejected as a fall -- the stream
contains ZERO readings of 91 or 111, the two correct finals, on frames where a
hand-cut crop reads exactly those. Fixed by removing the misalignment: snap the
region onto its own digits every frame, so the box only has to be close.

**Three. My own `digits_never_shrink`, within an hour of writing it.** It was
the fix for problem two's cousin -- a single glyph winning the same-height vote
and reading as "1" -- and it ratcheted the other way: one spurious three-digit
read discarded every two-digit reading after it.

### The regression that caught it

Finals G1 is a game this project may tune on, which is exactly why the
scoreboard changes were checked against it rather than against the held-out
broadcast:

    read                              regions chosen          changes    final
    before any of today's changes     home 466-556            83         107-110
    with the widened boxes            home 270-424            18         445-235
    with the full fix stack           home 460-530            91         107-110

    the official final is 111-110

**445-235.** Widening the candidate sizes so a coloured team panel could fit let
a 154x110 box win -- one spanning several numbers, somewhere else on the bar --
and it beat the correct region on the span tie-break *precisely because it
over-reads*, since that tie-break assumes a clipped box under-reads. Then
`digits_never_shrink` ratcheted on it and threw away every two-digit reading for
the rest of the game.

Judging candidates on their SNAPPED region removes the incentive: a fat box and
a tight one around the same digits become the same region, and the ranking is
over digit groups rather than over arbitrary rectangles. Requiring a wider
reading to be seen twice removes the ratchet.

The restored read reproduces the validated final exactly and is better than it:
**legibility 97% to 100%, and 91 score changes against 83.**

**Had this only been checked on Houston, the widened boxes would have looked
like a fix and destroyed the two broadcasts that already worked.** That is the
whole argument for choosing thresholds on a game you are allowed to tune on and
then reporting on one you are not.

## Round 104: the scoreboard rung on the fourth broadcast, and what it is worth

After the three ratchet fixes and snapping candidates onto their own digits, the
Houston scoreboard reads **112-86 against an official 111-91** -- one point out
on the left team and five on the right, where before the fixes it read 117-116
and before those it found no score regions at all.

**That fails the script's own validation and it is reported as failing.** The
docstring says "check the final score against the box score before trusting any
event derived from this. It is one number and it validates the whole sweep", and
one of the two numbers is five out. The right-hand region's last change is
83 to 86 at video 8250 s and the game's final five points, in the last eighty
seconds, never arrive -- the panel is not legible through them, and `monotonic`
correctly refuses a five-point jump over a short gap because a real score cannot
move that fast between consecutive legible frames.

### What the rung is worth, on both broadcasts

    mode                     captured   emitted   said and true   architectural cap
    Finals G1
      vision + scoreboard      0.169       71        0.662             0.499
    Houston (unseen)
      vision                   0.275      307        0.502             0.446
      vision + clock           0.303      254        0.587             0.450
      vision + scoreboard      0.178       81        0.654             0.498
      feed-assisted            1.000      433        tautological      1.000

**0.178 and 0.654 on the unseen broadcast against 0.169 and 0.662 on the tuned
one.** The rung transfers almost exactly, and it remains what Round 97 found it
to be: the most PRECISE vision mode -- 0.654 against 0.587 for gated vision --
on the fewest calls, asserting an outcome on every one of them where vision
asserts one on none.

An earlier draft of this round quoted 0.750 for this rung. That number was
computed against the pre-fix score stream, the one whose final reads 117-116,
and it is withdrawn. The stream was wrong and the precision computed from it
was not measuring what its label said.

### The ledger for the fourth broadcast, complete

    label-free, needs nothing but the video and the feed
      alignment: overall             0.995     PASS      563 of 566
      clips: still match alignment   0.986     PASS
      clock: game seconds seen       0.8997    FAIL      three seconds short
      registration: agreement        0.905     PASS (point)   NOT a hold-out

    end to end
      feed-assisted, captured        1.000     PASS
      vision+clock, captured         0.303     CAPPED 67% of 0.45
      vision+clock, said and true    0.587     FAIL
      vision+scoreboard, said true   0.654     FAIL
      vision, said and true          0.502     FAIL

    needs labels this broadcast does not have
      handler, ball                    --      NO DATA, printed as 0.00-1.00

### Round 105: the acceptance run found a degradation nobody was measuring

The last stage of the fourth broadcast's run -- clip detection and overlays --
produced the first number that is clearly WORSE on the unseen arena, and it is
not in any model.

    broadcast   people detected   kept on court    share   drawn/frame p50
    HOU             311,146         148,963        0.479          7
    Finals G1       329,339         207,636        0.630         10
    ECF G1          342,918         281,758        0.822         13

**The detector finds essentially the same number of people on all three -- 311k,
329k, 343k over comparable frame counts -- and the floor mask throws away 52% of
them on Houston against 37% and 18% on the two Finals broadcasts.** A court box
is found on 100% of frames in all three, so this is not a failure to locate the
court; it is `stands_on_court` deciding a player's feet are not on it.

The resulting overlays show it directly:

    ball drawn on             81.1%   48.8%   56.8%
      ...more than 250 px from anything   2.5%    0.8%    0.1%
    rim drawn on              77.7%   87.1%   86.2%
    players per frame p50         4       7       9

**And the mask was never stable even across the two tuned broadcasts** -- 0.630
against 0.822 is a 19-point spread between two games in the same series, which
nothing had looked at. Houston extends the range rather than breaking new ground.

This gates every downstream consumer of player boxes: a player the mask rejects
cannot be the handler, cannot be tracked, and cannot anchor a ball candidate.
Houston has no handler labels, so the cost is not measurable there -- the report
prints NO DATA and does not guess -- but on the labelled broadcasts the arm
called `handler: detector drew him` reads 0.812 / 0.938 / 0.909, and this is a
plausible part of why.

**It is recorded as a lead, not a fix.** What would settle it is measuring the
mask directly against the frames a person has already labelled: the 71 committed
`handler_at` miss-clicks are exactly frames where a person saw a player and the
pipeline did not, and nobody has checked how many of those the detector DID find
and the floor mask then discarded.

## Round 106: ten players on the court, and what that one fact is worth

A reader pointed out the obvious thing nobody here had used: **from tip-off to
the final buzzer there are exactly ten players on the court**, plus at most three
referees. It holds on every broadcast ever made, it needs no labelling, and it
turns out to settle two questions this project had been guessing at.

### One. The floor mask is wrong on every broadcast, in both directions

`candidates.stands_on_court` decides whether a detected person's feet are on the
floor. Everything downstream depends on it -- a player it rejects cannot be the
handler, cannot be tracked, cannot anchor a ball candidate -- and nothing had
ever measured it, because measuring it looked like it needed labels.

    broadcast   frames   detected p50   kept p50   kept MORE THAN 13   worst frame
    G7          31,680        15            13         0.411               38
    G1          20,880        15            10         0.155               28
    ECF         21,780        15            13         0.426               36
    HOU         21,420        14             7         0.060               27

**More than thirteen people kept is impossible whatever the frame shows**, and it
happens on 41%, 16%, 43% and 6% of frames. One frame keeps **38**.

This also corrects what an earlier round said. The share of detected people the
mask keeps reads 0.479 / 0.630 / 0.822, and that round presented 0.822 as the
good end of the range and Houston's 0.479 as the degradation. **Wrong.** The
0.822 broadcast is the one keeping thirteen-plus people on 43% of its frames;
it is failing in the other direction. Houston under-keeps -- five or fewer on
39% of frames, where ten are on the floor -- and G1 is merely the least bad.
A share cannot tell those apart. Ten players can.

### Two. The "71 miss-clicks" are not a detector problem

Every `missing` row in the two label files carries a `handler_at` click: a
person saying "he is on screen and you drew no box for him". The plan has
carried an item to train the player detector on those 71 clicks, on the reading
that they are frames the detector failed. Splitting them by which stage actually
lost the player, with the click required to land INSIDE a box:

    of 71 hand-clicked misses across three broadcasts
      no box at any confidence                4
      a box, but below 0.35 confidence        9
      a confident box the FLOOR MASK dropped  42
      a confident box that was kept anyway   16

**The detector found 67 of 71. The floor mask threw away 42 of them.** The
conclusion does not depend on the click tolerance -- at 0, 10 and 20 px the mask
count is 42, 38 and 34 and the no-box count is 4, 2 and 0.

So the standing plan item is aimed at the wrong component, and the arm this
project calls `handler: detector drew him` -- which reads 0.812 / 0.938 / 0.909
-- is misnamed. It is mostly not the detector.

### Three. The scoreboard reader fails completely on the ECF broadcast

The regression that was outstanding. ECF G1 finds **zero** score-like regions out
of 3,522 candidates, and the reason is geometry rather than any threshold:

    Finals G1, Houston      a horizontal strip along the bottom of the frame,
                            both scores on the clock's own row, to its left
    ECF G1                  a stacked box in the TOP-RIGHT corner -- IND 51 on
                            one row, NYK 58 on the row below it, the clock to
                            their right

`band_candidates` slides boxes **along the clock's row, to the left of it**,
which is the whole search. On a layout where one score sits a row below the
other, a box tall enough to reach the second one contains both, and the
same-height glyph group is then all four digits.

    scoreboard reader, by broadcast
      Finals G1    107-110 against an official 111-110    works
      Houston      112-86  against an official 111-91     one out, then five
      ECF G1       no regions found                       fails
      Game 7       never attempted

**It works on two of the four broadcasts in this repository, and the third fails
for a structural reason a threshold cannot reach.** That is the honest state of
the rung whose architecture is the only one this project has measured above 85%.

### Where the ball is when you cannot see it: four leads, each with a number attached

The same reader who supplied the ten-player constraint raised the other half:
you do not have to SEE the ball to know where it is. Players point at it, a
dribbler keeps it, a shooter releases it, and once released it goes where physics
says. Each of those is testable against something already measured here, so they
are written down with the number that would settle them rather than as
aspiration.

**1. The 9.2% that no selector can ever reach.** On the uniform frames the ball
is proposed somewhere in the candidate list 90.8% of the time and reported 76.9%
of the time. Round 99 split the gap: ten of the thirteen missing points are a
binary choice between two boxes, and **9.2% is a ball that was never proposed at
all.** No re-ranking touches that 9.2%. Inference is the only thing that can, and
it is the only part of the ball problem where inference is not competing with
selection.

**2. The handler already knows.** The pipeline names the right handler on 50 to
66% of uniform frames, and `yolo11s-pose.pt` has been in this repository
unused since before any of it. A ball held or dribbled is within a forearm of a
wrist. The measurement that would settle it costs nothing new: on the frames
where the ball is NOT proposed, how often is the labelled ball within a wrist's
reach of the labelled handler's hands? That is an upper bound on this whole idea
and it can be computed from the labels already collected.

**3. Viterbi failed for a reason that argues FOR ballistics, not against it.**
`ball_track.choose` was wired up and lost: the mechanism was measured and it is
that **the decoys move 6.5 px between frames and the real ball moves 90.5**, so a
SMOOTHNESS prior actively prefers a stationary orange thing in the crowd. A
ballistic prior is the opposite object: it expects large, accelerating,
downward-curving motion and would penalise exactly the candidate smoothness
rewards. The recorded negative is evidence about smoothness priors and says
nothing about physical ones, and the distinction was never drawn because the
experiment that produced it was not designed to.

**4. A shot is a pose before it is a trajectory.** `detect_shots` works from
ball-to-rim geometry and reaches F1 0.674 on a broadcast it has never seen --
but it needs the ball, and the ball is what goes missing. A shooting motion is
visible in a body whether or not the ball is: the arms extend, the wrists flick,
the feet leave the floor. The bound on this one is already in this file too, from
the other direction: **a shots-only system captures at most 0.45 of a game's
plays**, so a pose-based shot detector cannot lift the ceiling, only the fraction
of it that is reached.

**And the ten-player constraint has a second use nobody has taken.** It bounds
the TRACKER as hard as it bounds the mask: no more than ten player identities
can be alive on the court at once. The tracking work in this repository counts
identities in the hundreds per game and has no accuracy metric at all, and this
is one -- free, label-free, and violated every time the count goes above ten.

### Why the floor mask fails: it is a colour prior for one arena's paint

`candidates.court_region` finds the floor as `wood | paint`, where wood is hue
5-30 and **paint is hue 95-130 -- a BLUE key**. Sampling the court band of one
wide frame per broadcast, and reporting what share of those pixels each band
accepts:

    broadcast   wood    paint(blue)   neither   modal hue of the rejected
    G7         0.339      0.204        0.457            4
    G1         0.319      0.140        0.542            4
    ECF        0.207      0.166        0.627          150
    HOU        0.794      0.004        0.202          175

**On Houston the blue paint band accepts 0.4% of the court.** The Rockets' key
is red -- the rejected pixels peak at hue 175, which is red on OpenCV's 0-180
scale -- so the key is a hole in the mask and a player standing in it fails the
feet test. That is the under-keeping.

**On ECF the wood band accepts only 0.207**, the lowest of the four, and yet
that broadcast over-keeps on 43% of frames. A floor the colour test barely finds
means the largest connected component is being chosen somewhere else.

So one hand-set pair of colour ranges, fitted to one arena's blue key, is doing
a job that is arena-dependent in both directions. **A first attempt at removing
the paint colour -- keeping wood only and filling its holes, on the argument
that a painted key is a hole surrounded by wood -- made it worse, because on a
court with a large key the wood alone splits into two components and the largest
is half the floor.**

The fix that is actually principled is already in this repository and is not a
colour test at all: **the court polygon from registration.** `court_keypoints`
plus a homography projects the known court rectangle into the image exactly,
with no reference to what anything is painted. Registration reaches 62-75% of
frames, so it would need the colour test as a fallback rather than a
replacement, and every detection cache would have to be rebuilt to measure it --
about two hours. It is the next piece of accuracy work and it is written down
here with the evidence rather than attempted in the last hour of a session.

## Round 107: real retrieval, measured offline, and the gate it fails

The plan's Team R, built and run with **zero cloud spend**, which the plan
requires before any Cloudflare resource is created. `bge-base-en-v1.5` at 768
dimensions, possession cards rather than events, a `VectorStore` protocol with
an exact-cosine local twin, and three gates written into the evaluation's
docstring before a number existed.

    recall@10, 358 cards from three broadcasts, 1,658 questions

    category              n     regex   vector   hybrid
    counting             30     0.933    0.967    1.000
    player_action        29     0.966    0.897    0.966
    paraphrase_named  1,536     0.834    0.707    0.896
    paraphrase_unnamed   20     0.300    0.800    0.800
    sequence             30     0.900    0.733    0.833
    temporal             10     0.000    0.800    0.800
    cross_game            3     0.000    0.667    1.000
    ALL               1,658     0.826    0.717    0.897

### The gates

**G1 PASS.** Hybrid 30/30 against regex 28/30 on counting questions, p = 0.50.
Retrieval does not regress the questions the page already answers well, which is
the regression that would have mattered most.

**G2 FAIL, and not narrowly.** The vector arm was required to beat the regular
expression on paraphrase by ten points. It **loses by 11.9**, 1102/1556 against
1287/1556, p < 0.0001.

**G3 PASS, decisively.** Hybrid 1487/1658 against vector-only 1189/1658,
p < 0.0001. **The structured filters are doing the work, not the embedding** --
which is the D1-filters-Vectorize-ranks split the plan argued for, now measured
rather than asserted.

### Why G2 fails, and what would be dishonest to do about it

The paraphrase set is **98.7% questions that name a player**, and a literal name
is exactly what a regular expression is best at. Split by whether the question
names one:

    paraphrase_named     1,536   regex 0.834   vector 0.707
    paraphrase_unnamed      20   regex 0.300   vector 0.800   +50 points, p = 0.0129

**Where the embedding is the only thing that could work, it wins by fifty
points.** Where a name is present, it loses. Both are true and the blend of them
is what the gate measured.

The tempting move is to declare G2 passed on the unnamed subset. **It is twenty
questions, the vector arm's interval there is 0.58 to 0.92, and this project has
a thirteen-item ledger of ball-selection ideas rejected on exactly this sample
size** -- the same regime, the same false confidence available in both
directions. Twenty questions settle nothing.

**So G2 is recorded as FAILED and the remedy is a question set, not a redesign.**
The plan specified 300 questions and two authors for precisely this reason; what
was built here is 1,658 questions and one author, and the count turned out to
matter far less than the composition. An earlier run of this same evaluation with
45 paraphrase questions had the vector arm WINNING by 15.6 points at p = 0.167 --
the opposite sign, from the same code, because that set was mostly unnamed.
Neither number was wrong. They are answers to different questions, and the
lesson is that a question set is a measuring instrument that has to be designed.

### What the embedding is unambiguously for

Two categories where the regular expression scores **0.000** and the hybrid 0.800
and 1.000: `temporal` ("what happened in period 2") and `cross_game`. The only
discriminative token in those is a bare number, and literal matching has nothing
to hold. They are small -- 10 and 3 questions -- but they are the shape of
question the shipped page genuinely cannot answer today.

### Ingest economics, measured rather than estimated

**89 cards a game**, so a 1,315-game season is about **117,000 cards** -- against
Vectorize's five-million-vector cap, roughly forty seasons. Embedding runs at 3
cards a second on this laptop's CPU, which is hours for a season here and
minutes on a rented GPU. Storage and compute are not the constraint; the
question set is.

### Round 108: a hand-written question bank, and a gate that fails while its intent passes

Round 107 recorded G2 as failed and said the remedy was a question set rather
than a redesign. `data/retrieval/questions_unnamed.json` is that set: **66
questions written by hand that name no player and reuse no card vocabulary** --
"when did they come away empty", "who won the ball back off the iron", "a trip
that ended with contact rather than a shot". Each one's truth is a **predicate
over card FIELDS**, evaluated in a namespace holding the fields and nothing else,
so the wording and the answer come from different places. That is the only
second author available without a second person.

    category              n     regex   vector   hybrid
    paraphrase_named  1,536     0.834    0.707    0.896
    paraphrase_unnamed   86     0.209    0.779    0.779
    ALL               1,724     0.802    0.719    0.892

**On questions where literal matching has nothing to hold, the embedding wins by
57 points: 67/86 against 18/86, p < 0.0001, interval 0.68 to 0.85.** That is no
longer a twenty-question curiosity. It is the clearest evidence in this
repository that a vector arm is worth having.

**And G2 still FAILS.** Blended across both paraphrase categories the vector arm
loses by 9.0 points, because the blend is 95% questions that name a player and a
name is what a regular expression is best at.

### Why the gate is not being rewritten

The gate's denominator was chosen badly -- by me, in the same file, before the
questions existed -- and the category labelled "paraphrase" turned out to be
mostly named-entity questions with paraphrased decoration. The number it produces
is measuring something other than what it was written to measure.

**That is not a reason to change it after seeing the result.** This project has
retracted a published round for less, and a gate that moves when it fails is not
a gate. G2 stands as FAILED for this run.

The corrected gate is declared here instead, for the next one, **with its
threshold unchanged**: the vector arm must beat the regular expression by at
least ten points of recall@10 at p < 0.05 **on questions that name no player**,
because those are the ones where an embedding is the only thing that could work.
On today's evidence it would pass by 57 points. It has not been run under that
definition yet, and this file will say so until it has.

### The filters and the embedding do different jobs, and both are load-bearing

    G3  hybrid 1,538/1,724 against vector-only 1,240/1,724, p < 0.0001

On the hand-written bank the hybrid and the vector arm score **identically**
(0.779 both), because those questions carry no player, period or action a filter
can extract -- so the hybrid IS the vector arm there. On the named questions the
filters carry it (0.896 against 0.707).

**That is the argument for the architecture, and it is now measured rather than
asserted**: structured filters answer what can be looked up, embeddings answer
what cannot, and each is useless on the other's half.

## Round 109: tracking gets its first accuracy metric, and cut-awareness gets tested

Every tracking number in this repository has been a COUNT -- "463 identities for
ten players over five minutes" -- which says something is wrong, cannot say how
wrong, cannot compare two trackers, and is won outright by a tracker that merges
all ten players into one identity. The plan's answer was ~980 hand judgements
nobody has made.

**The ten-player constraint supplies three metrics for nothing**, from the cached
detections, with no video and no labels:

    game   alive p50   OVER 13 alive   two tracks on one man   identities   median life
    G7          9         0.010              0.428               1,469         1.18 s
    G1          9         0.011              0.402               1,358         1.23 s
    ECF         9         0.010              0.429               1,206         1.52 s
    HOU         8         0.005              0.366               1,352         1.17 s

**The tracker is not inventing people.** More than thirteen identities alive at
once happens on 0.5% to 1.1% of frames, which for a component nobody had ever
measured is a better result than the identity counts suggested.

**It is carrying eight or nine where ten are on the floor**, and Houston -- the
broadcast whose floor mask discards half the detected people -- is the one at
eight. The two measurements agree from opposite directions.

**And on 37% to 43% of frames two live tracks overlap by more than the project's
own duplicate threshold.** `motion_tracking.deduplicate` exists to draw only one
of them, so nothing visible is wrong; what it means is that the boxes going IN
are duplicated that often, which is a detector property this metric can see and
the overlay cannot. It is stated as an upper bound: two players in a screen or a
rebound scrum genuinely can overlap that much, and separating those cases needs a
judgement this does not make.

### Cut-aware termination: the counts get worse and nothing else moves

`MotionTracker.end_segment` has existed since the tracker was promoted and
nothing has ever called it with real cuts. The cuts are free too -- a camera cut
is a frame where almost nothing matches the frame before, which is the same
signal the tracker already computes for camera motion, so no pixels are needed.

    game    identities            median life         OVER 13        doubles
    G7      1,469 -> 1,478      1.18 -> 1.15 s      unchanged      unchanged
    G1      1,358 -> 1,414      1.23 -> 1.13 s      unchanged      unchanged
    ECF     1,206 -> 1,264      1.52 -> 1.52 s      unchanged      unchanged
    HOU     1,352 -> 1,443      1.17 -> 1.10 s      unchanged      unchanged

**Exactly what the plan predicted: "it will make the counts worse and the
accuracy better".** The counts do get worse -- more identities, shorter lives,
because a track that used to run through a cut now ends at it. Whether the
accuracy gets better is a claim **these metrics cannot settle**: over-tracking
and duplication are unchanged to three decimal places on all four broadcasts.

So cut-awareness is left unwired, with a number rather than an intuition behind
that decision. What would settle it is the across-cut stratum of the identity
labelling the plan specifies, and that is still ~980 human judgements away.

**The clearest demonstration available that counts are not accuracy**, which is
what the plan said this change would be, and it is now demonstrated rather than
asserted.

## Round 110: wrists do not beat box edges, and the box edge beats the box centre

The plan's V6 argued that the handler ceiling is a feature problem: every
possession rule scores a player by the distance from the ball to his BOX, and
the recorded failure is that "on a dribble a defender's hands are often nearer
the ball than the holder's". A box centre is the middle of a torso; a ball is
held in hands; `yolo11s-pose.pt` has been in this repository the whole time and
nothing had asked it.

On the 131 frames where a person labelled **both** the ball and the player
holding it, with all three rules handed the TRUE ball position so each is an
oracle:

    rule          uniform                  hard
    box centre    43/61 = 0.705     36/45 = 0.800
    box edge      48/61 = 0.787     40/45 = 0.889
    wrist         47/61 = 0.770     38/45 = 0.844

    paired against the box centre, uniform half, exact McNemar
      edge     7 frames only it gets, 2 only the centre   p = 0.1797
      wrist    6 frames only it gets, 2 only the centre   p = 0.2891

**Wrists do not beat box edges.** They beat the box CENTRE by 6.5 points and lose
to the box EDGE by 1.7, on the same frames, and neither difference is
significant at n=61. V6's hypothesis -- that hands separate what torsos cannot --
is not supported: what separates them is measuring to the box's EDGE rather than
its middle, which needs no pose model and is four lines of arithmetic.

### What this is NOT evidence for

These numbers are **not comparable to the recorded 58.7% proximity ceiling** and
must not be read as beating it. Three things differ:

  the boxes come from `yolo11s-pose.pt`, a generic person detector, not from
  this project's four-class detector;

  25 of the 131 frames are dropped because the pose model found no person
  overlapping the labelled handler, which removes exactly the hardest cases;

  the denominator is frames where BOTH ball and handler are labelled, where the
  58.7% was measured over a set that includes frames with no handler box at all.

The comparison that IS valid is the paired one above, because all three rules
answer the same frames with the same boxes. **Only the ranking is evidence; the
level is not.**

### The lead this leaves

The box EDGE rule beating the box CENTRE by 8 points on the uniform half is
free, already implemented in `eval_wrist_handler.to_box`, and has never been
tried in the possession kernels -- which score candidates by centre distance.
p = 0.18 at n = 61 settles nothing, but it is a one-line change to test against
the 157-frame handler set where the 49.7% and 59.2% numbers live, and that set
is three times larger.
