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
refused as failures in every statistic, requires dumps made at threshold 0 (so
a candidate can judge frames AND refits exactly -- a dump gated at 2 cannot
simulate 5), refuses the unseen arena, and refuses dumps without provenance.
Dumps now record video, arguments, commit, threshold, polarity, the control
values and every registered frame. Also fixed: a 1e-3 floor under the
sharpness ratio's denominator, which made any partial lock with empty 2 ft
neighbours score about 1000x its coverage (rates are now add-one smoothed on
counts); hypotheses compared on explained paint must now see at least 70% as
much court as the best-seeing one; a control that does not run is reported as
a failure; and the evaluator's default video, which pointed at the held-out
arena, is gone.
