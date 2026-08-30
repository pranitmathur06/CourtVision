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
