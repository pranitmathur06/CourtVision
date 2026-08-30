# Making it accurate on a whole game: what worked, what did not, what is blocking

The action classifier scores 0.813 on held-out clips and called 67% of a game a
rebound. This is the record of chasing that down.

## The root cause is the representation, and that part is settled

`crop_player` crops tightly to the ball-handler. A montage of `rebound` clips
beside ordinary-play clips is indistinguishable, and **neither shows a
rebound** — a rebound is the ball coming off the rim with players converging,
and the event happens outside the crop. `rebound` therefore became the model's
label for "generic player crop".

Separating a rebound from ordinary play, 240 windows:

    player crop   0.688   +0.042 over chance
    full frame    0.758   +0.113          — 2.7x

This explains the rest of the table too. dribble 0.90, block 0.86 and shot 0.87
are ball-proximate: the ball is inside the crop, so those classes were never
broken. Rebound is the one class the representation cannot express. It also
explains why widening the crop margin did nothing earlier — 0.25, 1.0 and 2.0
were all centred on the wrong thing.

## Two fixes that work in isolation

**A background class.** Ordinary play, taken from the gaps between labelled
actions, so the model has somewhere to put windows that are not events. Held-out
accuracy held at 0.813 across 8 classes (was 0.816 across 7).

**A full-frame rim detector.** Binary, rebound against ordinary play, on the
view that can actually see a rebound: **0.860** on 200 balanced held-out
windows, against a 0.500 baseline, and balanced across both classes
(background 87/100, rebound 85/100).

Both are real. Neither fixed the game.

## What did not work, stated plainly

On windows sampled uniformly along real game footage:

    crop only        rebound 80%   background 18%
    two-view t=0.5   rebound 92%   background  8%
    two-view t=0.7   rebound 89%   background 11%
    two-view t=0.9   rebound 78%   background 21%

The two-view system is **worse** than the crop model at the default threshold.
A 0.860 detector that says rebound to 92% of a game is not calibrated for that
game, whatever its held-out number says.

Prior-shift calibration behaves as a dial and cannot rescue it either: pushing
rebound down far enough (strength 1.5) drives background to 93% and takes every
shot with it — 0% shots at every strength tried, in a game containing 180.

## Why, and what is actually blocking

Every negative example available comes from BARD, and **BARD only contains
footage cut around events**. Its clips run 18.8 s and carry 1.6 actions each:

    events per minute of BARD footage   5.1
    events per minute of a real feed    2.8
    BARD is 1.8x more event-dense

1.8x does not explain a 27x error, so the over-prediction is real rather than a
sampling artefact. But it does mean the "ordinary play" the models were trained
on is not ordinary: a 1.6 s window sampled inside an 18.8 s event-cut clip sits
on or beside an action far more often than one sampled from a live broadcast.
There is no dead time in this data at all — no inbounds, no free-throw
routines, no timeouts, no walking the ball up after a whistle — and dead time is
most of what a real game is.

**The blocker is footage, not modelling.** Calibrating a classifier for
continuous play requires continuous play to calibrate against, and every source
here is highlight-selected. The standard remedy is a dataset of untrimmed games
— Stanford's NCAA set is 257 games of roughly 1.5 hours each with timestamped
events — but it is distributed as YouTube links rather than video, so obtaining
it is a decision about downloading broadcast footage, not a technical step.

## What stands

  * the representation diagnosis, independently measured
  * a rim detector that works on its own distribution (0.860)
  * a background class that costs nothing in clip accuracy
  * `scripts/evaluate_game.py` — scores a run against BARD's labels for that game
  * `scripts/tune_prior.py` — sweeps calibration on game windows in minutes
  * `courtvision/calibration.py` — prior shift, defaulting to off

What does not stand is any claim that this is accurate over a full game. It is
not, and the honest reason is that it has never been trained on one.

---

# Round two: continuous footage, and the confound that runs both ways

## The failure is symmetric, and both directions are corpus detection

Trained with background from BARD gaps, the model called **80–92% of a game a
rebound**. Trained with background from a real 141-minute broadcast, it calls
**100% of a game background** — from both models, at every threshold:

    crop only        background 100%
    rim alone        background 100%
    two-view t=0.5   background 100%
    two-view t=0.9   background 100%

The rim detector scored **0.992** on its own held-out set, with training loss
reaching 0.0000 and background 160/160. That is not learning; it is a shortcut.
Its rebounds came from BARD and its background from the live TNT broadcast, so
recognising which broadcast it was looking at solved the task perfectly.

The confound audit predicted exactly this before training (+0.107 to +0.187,
OPEN at every size) because background was the only class drawn from the live
corpus. Training anyway was deliberate — the empirical answer was worth having —
and it is unambiguous.

**The rule this establishes:** live footage has to contribute to *every* class,
not just the negative one. Otherwise "which corpus is this" is always a better
hypothesis than "what is happening", and at serve time every window is live, so
the model answers with whatever label the live corpus taught it.

## Removing rebound from the crop model helped everything else

Accidentally, then deliberately. Moving rebound out to the full-frame set left
the crop model with seven classes it can actually see:

    V7 PASS — 0.879 on 767 held-out clips, 7 classes, no rebound
    (was 0.813 across 8 classes with rebound, 0.816 across 7 with it)

Dropping the one class the representation cannot express raised held-out
accuracy by 0.066. That is the two-view split earning its keep on the crop side.

## What was actually unlocked: the join key

A full 141-minute broadcast now exists — 253,516 frames of continuous play with
the dead time no dataset here had — and the clock reader works on it:

    readable            97/120 sampled frames (81%)
    counts DOWN         96/96 consecutive reads
    stoppage            held at 8:25 across six reads, as a real clock does

Two fixes were needed. The module was written for a dark-on-bright scoreboard
and TNT draws white on black, which returned zero glyphs from a crop where the
clock is plainly legible — `normalise_polarity` decides by the median and
`read_clock` applies it. And templates built from one frame covered only
{1,2,5} and read 1 frame in 60; built from seven frames across the game they
cover {0,1,2,3,4,5,6,8}.

Video time -> period and game clock -> official play-by-play is the join that
makes live footage **labelled**. Every attempt before this one lacked it.

## The remaining path, concretely

1. Read the clock across the game, building a video-time to game-clock map.
2. Fetch the official play-by-play for this game (`nba_feed` already does this).
3. Label live windows by joining on (period, clock) — giving live-corpus
   examples of shot, rebound, steal, block, and genuine background.
4. Retrain both views on live-corpus data, where corpus no longer predicts label.
5. Score against the official box score, which is exact ground truth.

Accuracy over a continuous game is **not fixed**. What changed is that the
blocker is no longer missing data or a missing join — both now exist — and the
remaining work is labelling and retraining rather than searching for a method.
