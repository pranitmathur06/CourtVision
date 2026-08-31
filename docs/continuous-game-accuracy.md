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

**Steal: 33.8x from the classifier, 2.39x derived at the 6 s default.** A 14x
improvement, and the first time that class has been anywhere near usable.

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
