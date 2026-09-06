# Hand-labelled jersey crops

72 player crops from an uncut NBA broadcast, each labelled with the jersey
number a person can read on it. `labels_by_eye.json` maps filename -> number.

## Why these are labelled by eye

The obvious automatic labelling is circular and was measured to be wrong about
nine times in ten. It pairs a crop with the shooter's number by taking the
player nearest the eventual shot location -- but at any moment before the
release that is usually a DIFFERENT player, and establishing which player is
which is precisely the problem the labels are meant to validate.

## What they were used to measure

On the twelve crops carrying a number legible to a person, scored by each
reader's single best answer rather than by whether the truth appears anywhere
among many guesses:

    a person                      100%
    localised-digit nearest       36%   (majority-class baseline 33%)
    whole-crop nearest            23%
    easyocr, modal answer         17%
    rendered-font templates        8%

The information is present at 720p. No off-the-shelf reader gets close, because
jersey digits sit on curved fabric in a team typeface under stadium light,
unlike the flat aligned glyphs a scoreboard reader handles at 94.9%.

## What this set is for

It is the seed of the training data a domain-specific model needs. Reaching 85%
requires hundreds to thousands of labelled crops, and this is the format and
the method: annotation grids of the largest crops, read by eye, ambiguous ones
skipped rather than guessed.


## Scaling measured, not assumed

Labels were added in five annotation rounds and the reader re-scored each time.
Doubling the set did not move accuracy, though it did widen the lift over
guessing:

    39 labels, 13 classes    36% correct, majority baseline 33%
    72 labels, 15 classes    36% correct, majority baseline 23%

Nearest neighbour with about four examples per class is the constraint, not the
annotation. Reaching 85% needs on the order of 50-100 examples per class --
750-1500 labels, or 25-50 more rounds of the same manual reading -- and
probably a trained model rather than nearest neighbour on top of that.

Note 00 and 0 are DISTINCT classes here. Mathurin wears 00 and Haliburton
wears 0; the obvious `lstrip("0")` normalisation merges two different players
and was a real bug in an earlier version of this work.
