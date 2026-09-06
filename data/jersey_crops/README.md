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


## Both automatic labelling routes were measured, and both fail

**Nearest player to a shot location.** Before the release the nearest player to
where the shot will be taken is usually somebody else. About nine in ten labels
disagreed with the visible jersey.

**Nearest player to the free-throw line.** This looked airtight -- the shooter
stands alone at a FIXED court position while the other nine are along the lane,
so no association is being inferred. It fails anyway, and the reason is worth
recording: the camera is frequently not showing the line at the moment the feed
timestamps the attempt. It cuts to a replay, a bench reaction, or another
angle, and the homography then places whoever happens to be in frame at those
court coordinates. Of 24 crops inspected, roughly 2 labels were right; the set
included a referee wearing 58 and a person in yellow who is not a player.

The two routes fail for one underlying reason. Producing a label requires
knowing which detected person is a given player, which is precisely what the
labels exist to establish. Anchoring to a different event does not escape the
circle -- it only moves it.

That leaves human annotation as the only trustworthy source, which is why the
72 labels here were read by eye, and why reaching 85% is a data-collection
project rather than a modelling one.


## Refusing uncertain crops helps, and how much depends on honest evaluation

Extraction, not classification, is the weak step. Visualising what the
classifier receives shows the split plainly: about half the crops yield a crisp,
obviously readable silhouette (22, 33, 8, 9, 23, 0 come out clean) and the rest
yield thin streaks or scattered fragments.

Hand-designed quality gates did not separate those. An ink-fraction gate
rejected 3 of 72 crops and moved accuracy from 36% to 38%. The classifier's own
confidence does separate them:

    keep by MARGIN over runner-up      n     accuracy
    all                               59        36%
    top 50%                           29        52%
    top 25%                           14        86%

That 86% is NOT the honest number. The threshold was chosen as a percentile of
the same fourteen samples it was scored on. Re-run with a FIXED threshold, and
evaluated per player appearance rather than per crop -- which is the unit a
product uses -- it becomes:

    margin floor   answered   coverage   precision
        0.00          34        100%        29%
        0.15          11         32%        64%
        0.20           8         24%        62%

So the defensible figure is about 64% precision at 32% coverage, not 86%.

## Where jersey reading stands

    rendered templates                        8%
    easyocr, modal answer                    17%
    whole-crop nearest neighbour             23%
    localised digits, grey                   36%
    silhouette + digit-count prior           36%
    + confidence gating, honest evaluation   64% at 32% coverage

Real progress -- eightfold over the starting point -- and still short of 85%.
Refusing to answer is what buys the accuracy, so the remaining gap is coverage
as much as precision.
