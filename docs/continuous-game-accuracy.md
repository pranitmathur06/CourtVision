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
