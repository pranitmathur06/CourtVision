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


## Colour-based extraction: plausible, and measured worse

Grey Otsu thresholds whatever contrast dominates a crop -- jersey against skin,
jersey against floor -- rather than the number against the shirt, so segmenting
by colour distance from the KIT looked like the right fix. The kit is the
torso's modal colour, and digits are the pixels far from it in CIELAB, which
covers white-on-blue and dark-on-yellow under one rule.

On individual crops it produces visibly cleaner digits. Measured end to end it
is worse:

                        per crop    per appearance
    grey Otsu             36%         64% at 32% coverage
    colour distance       21%         29% at 32% coverage

The percentile threshold picks up shadows, seams and chest logos, and the
"modal colour is the kit" assumption fails whenever a crop carries much
background. Kept as a recorded negative: the cleaner examples were not
representative of the set.


## Classify DIGITS, not numbers

Treating "43" as its own class wastes the labels: fifteen classes, about four
examples each, and accuracy that did not move when the set doubled. But 43 is a
4 and a 3, and 22 is two 2s. The same 72 labels give 48 digit instances across
ten classes, and digit recognition is the better-posed problem -- ten shapes
rather than an open set of combinations.

    per digit, all                          50%   (whole-number: 36%)
    per digit, top 50% by margin            75%
    per digit, top 35% by margin            94%

Reconstructing the whole number needs every digit right, so those do not carry
over directly. End to end, fixed thresholds, per player appearance:

    margin   answered   coverage   precision
     0.00       24        100%        42%
     0.15       11         46%        73%
     0.20        9         38%        78%
     0.25        2          8%       100%   (n=2, meaningless)

78% at 38% coverage, against 64% for whole-number classification.

## Why more labels are now worth collecting

Whole-number classification was flat from 39 to 72 labels because each label
added one example to a fifteen-class problem. Per-digit turns each label into
one or two examples across ten classes, so the same annotation effort is worth
several times more. The scaling that stalled before should not stall here --
which makes annotation the productive path again rather than a last resort.


## Where it stopped, and why

80 labels over six annotation rounds. Adding the last eight moved nothing:

    72 labels   78% precision at 38% coverage
    80 labels   78% precision at 35% coverage

Per-digit scaling did not rescue it, and the run says why: only 44 of 80 crops
are usable, because extraction is rejected whenever the blob count disagrees
with the label's digit count. Extraction, not label volume, is the wall -- the
same conclusion the silhouette pictures reached, and it survived both attempts
to fix it (colour segmentation measured worse, ink-fraction gating moved 36% to
38%).

Final honest state of jersey reading:

    rendered templates                    8%
    easyocr                              17%
    whole-crop nearest neighbour         23%
    localised digits, whole number       36%
    whole number + confidence gating     64% at 32% coverage
    per digit + confidence gating        78% at 35% coverage

Roughly tenfold over the start, and short of 85%. The next real gain is digit
SEGMENTATION on curved fabric, which is a vision problem in its own right
rather than a threshold or a bigger label set.


## Final state, eight annotation rounds

104 labels read by eye. Splitting merged digits -- the fix clock_reader already
uses, where two touching glyphs binarise as one blob -- took usable crops from
55% to 68%.

    labels   precision   coverage
      80        78%         35%
      94        82%         32%
     104        82%         31%

Plateaued at 82% precision on eleven answered appearances. At that sample size
82% is 9 of 11, and 85% is not distinguishable from it -- one more correct
answer would read as 91%. The honest statement is that the point estimate is
82% and the measurement cannot resolve the difference from 85%.

Constraining the reading to a roster number, which sounded obviously right --
nobody wears "39" -- measured WORSE, 73% against 82%, because forcing validity
overrides the digit evidence when the classifier is correct but the combination
is uncommon.

## The whole arc

    rendered templates                     8%
    easyocr                               17%
    whole-crop nearest neighbour          23%
    localised digits, whole number        36%
    whole number + confidence gating      64% at 32% coverage
    per digit + confidence gating         78% at 35% coverage
    + splitting merged digits             82% at 31% coverage
    + roster constraint                   73%   (worse, reverted)

Tenfold over the start. What remains is digit segmentation on curved fabric --
still rejecting a third of crops -- and enough labels to measure past the noise
floor. Both are real work; neither is a threshold.


## Extraction failures, diagnosed and mostly not recoverable

Of 104 labelled crops, 33 are rejected because the blob count disagrees with
the label:

    label 2d, found 1 : 12   merged digits the splitter missed
    label 1d, found 2 :  9   a spurious blob beside the number
    label 1d, found 0 :  7   nothing found
    label 2d, found 0 :  5   nothing found

Targeting the first two -- lowering the split threshold from w>h to w>0.85h,
and discarding blobs far less substantial than their neighbour -- was NEUTRAL:
usable went 68% to 67% and precision stayed at 82%. Each change fixed some
crops and broke others.

That is where this stops: 82% precision at 31% coverage, eleven answered
appearances, across eight annotation rounds and roughly ten distinct reader
designs.
