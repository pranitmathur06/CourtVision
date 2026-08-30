# v3 — real player identities from live/official data

## The reframe

Spec §7.3 defers jersey-number OCR to v2 and §10 calls it "a genuinely hard,
actively-researched problem" — small text, motion blur, partial occlusion.

But **you do not have to recognise a player to name them.** Basketball already
produces an authoritative, timestamped event stream: the official play-by-play.
Given the game and roughly where in the game clock a moment sits, naming becomes
a **join**, not a recognition problem.

## Two regimes

| | archived game | live game |
|---|---|---|
| Game identity | you chose it, so you know the GameID | known |
| Event stream | stats.nba.com play-by-play, free (`nba_api` wraps it) | live feed, commercial (Sportradar, Genius Sports) |
| Join key | game clock from the scoreboard overlay | same |
| Latency | none | seconds, fine for commentary |

**The scoreboard clock is the enabling trick.** It is fixed position, large,
high-contrast digits — vastly easier OCR than a jersey number on a moving player.
Read the clock, look up what the play-by-play says happened at that moment, and
the names come for free.

## What is implemented

`courtvision.enrichment` does the join, and is tested against real BARD metadata:

```
K. Johnson     rebound  K. Johnson REBOUND (Off:0 Def:1)
Williams       shot     Williams 2' Running Dunk (2 PTS) (K. Johnson 1 AST)
```

BARD needs no OCR at all — each clip's metadata already carries `GameID`,
`GameEventID` and the official description, which makes it the ideal harness for
building and testing the naming layer before wiring up a live feed.

* `parse_nba_url` — BARD metadata → `PlayByPlayEvent`
* `extract_player` / `extract_action` — official description → name and action
* `align` — attaches names to our events **only on a confident match**

## The safety property

A misalignment does not produce a vague answer. It produces a **confidently wrong
name** — a false statement about a real, identifiable person. So:

1. `align` refuses to guess. No action agreement, no name; the event keeps its
   anonymous `Player N`.
2. `validate_commentary` enforces it in **both** directions:
   * a real name in the text when the event carries none → error
   * a *different* real name than the event's → error

That second check was missing at first and a fabricated "Jokic" passed
validation. It now reports:

```
line 0: names 'Jokic', but the event's player is 'K. Johnson'
```

The capitalisation check is an **allowlist**, not a blocklist: an unrecognised
capitalised word is assumed to be a name and flagged. A false flag costs one
retry; a missed fabrication puts a false claim about a real person into output.

## What vision still contributes

If the API already says "Cunningham made a layup at 10:57", why run the pipeline?
Because play-by-play is *discrete events* and vision is *continuous*: where
players were, how the play developed, who was defending, off-ball movement.
Fused, they produce commentary neither could alone — that is the actual case for
the system.

## Play recognition — what is built, and what is not

`courtvision.court` and `courtvision.formation` are the first two layers.

**Built: court registration.** `court.register()` fits an image-to-court
homography from named NBA landmarks and reports its error in feet. It does not
find the court by itself — robust line detection through crowds, glare, floor
logos and a panning camera is a research problem, and a silently wrong
homography is worse than none, because it places players plausibly but
incorrectly and every formation claim inherits the error.

Note the trap it guards: a homography has 8 degrees of freedom and 4 point pairs
give 8 equations, so with exactly four correspondences the fit is exact and
reports zero error *however wrong the inputs were*. Corrupting one by 300 px
still reports 0.000 ft while the true held-out error passes 2 ft. Supply more
than four, and check `is_usable()` before trusting anything downstream.

**Built: formation classification.** `formation.classify_formation()` names
Horns, five-out, post-up and isolation from court coordinates, and returns
`unknown` — with a reason — when the arrangement is not one it can justify. It
also reports spacing as mean nearest-neighbour distance, which is what "they
have no room" actually means.

**Built: screen actions over time.** `plays.detect_screens()` reports
pick-and-roll, pick-and-pop, dribble hand-off and unresolved ball screens;
`detect_off_ball_screens()` covers screens away from the ball. These need no
labelled data because they are definitions rather than categories: two
offensive players converge, one holding the ball, and afterwards the screener
either cuts to the rim (roll), steps out beyond the arc (pop), or takes the
ball (hand-off). Every term is measurable in court feet.

**Built: named sets that are compositions.** `plays.detect_sets()` reports
Spain pick-and-roll, double drag and re-screen. Calling all named sets a data
gap was too quick: some are conventions with no geometric content, but these
three are definitions built from primitives already detected. A Spain
pick-and-roll IS a pick-and-roll plus a third player back-screening the roller
while the roll is still happening — a co-occurrence with a timing and identity
constraint, needing no labels.

**Built: off-ball screens named by direction.** I had filed these under
"needs labels" too, and that was wrong for the same reason. A flare screen is
not a name someone assigned, it is a direction. The types differ by where the
cutter ends up relative to two fixed things, the rim and the ball:

| type | cutter goes |
| --- | --- |
| back screen | toward the rim |
| flare | away from the ball, staying on the perimeter |
| pin down | up toward the ball, away from the rim |
| cross screen | across the lane without changing depth |

Both distances are court feet, so no labels are needed. A cutter who does not
commit anywhere is left as a plain `off_ball_screen` rather than assigned a
direction it did not take.

**Built: shape-plus-action sets.** `detect_sets()` takes the per-frame
formation and names `horns_flare` (the horns alignment, then a flare screen
out of it) and `horns_set` (horns, then a ball screen).

"Horns Flare" was the example cited three times in this document as needing
labelled play types. It does not. Horns is an arrangement visible in one
frame, a flare is a direction a cutter takes, and the set is their conjunction
inside a two-second window. Each time the claim was re-examined it turned out
to be a composition rather than a label.

**Still not built: names that are calls rather than shapes.** A set call is
coaching vocabulary — teams differ on it and no camera can see it. That
genuinely needs labelled play types, which BARD does not carry: its captions
are event-level, listing jersey number, colour and one of nine action types.
The distinction that survives: "flare" and "horns" describe what bodies did,
while a call describes what a coach shouted.

BARD does not have them. Its captions were checked directly: they are
event-level, giving jersey number, jersey colour and one of nine action types
(Turnover, Foul, Block, Rebound, Steal, 2PT Shot, 3PT Shot, Free Throw,
Violation), with no play names anywhere. SpaceJam is action clips only. So this
is a data gap, not an implementation gap, and it is the honest limit of what
this layer can claim.

**Partly built: automatic registration.** `court_lines.court_line_mask()`
finds the painted lines, and `alignment_score()` scores a homography by
projecting the canonical court into the image and measuring how much of it
lands on detected lines. That closes the gap `court.register()` could not: its
own error only measures how well the fit reproduces the points it was handed,
which with four of them is zero however wrong they were.

The detector came from measurement, and the measurement overturned the obvious
approach. The Pistons floor lines look navy, but thresholding blue (hue
100-135) finds the bench area, the floor advertising and the crowd while
missing every arc. The lines actually sample at hue 155-175, and far more
reliably at grey 52-110 against a local median of 198-225. Finding them as dark
pixels relative to a local median also survives the floor being brightly lit at
the far sideline and shadowed near the camera. On real broadcast frames it
returns ~2.8-3.0% of the frame as line pixels and 57-71 Hough segments of 60 px
or more.

Contamination is honest and known: players are dark too, and their edges and
jersey numbers survive the blob filter. That biases `alignment_score` toward
accepting a homography, so it is an upper bound and its threshold should stay
strict.

**Built: the search.** `search_registration()` recovers the camera
automatically — differential evolution over six PHYSICAL parameters (camera
position, aim point, focal length) rather than a homography's eight free ones,
because every point in the physical space is a camera that could exist and most
of the 8-dimensional space is not.

On a synthetic court with a known camera it recovers court coordinates to
**0.36 ft**. On the real Pistons broadcast it scores 0.42 and puts **10 of 11
detected players on the court** — an independent check that never touches the
alignment score.

Two failures on the way, both worth keeping in mind:

*The one-directional score was not enough.* Asking only "do the model's lines
land on detected lines" returned a camera scoring **0.998 whose court
coordinates were hundreds of feet wrong**: it had zoomed onto a patch where two
arcs coincided, so every projected point sat on a line. Measured against the
true camera — recall 1.000 / coverage 0.801 versus recall 0.998 / coverage
0.023. The score is now the harmonic mean of both directions.

*Random restarts do not find it.* 600 restarts plus Nelder-Mead stalled at
0.233 against the true camera's 0.888. The good basin is narrow. Differential
evolution finds it, but needs its budget: at maxiter 25 and 60 it returns 0.198
and 0.229, so a cheap run is a wrong answer rather than a fast one. Pass
`bounds` when the camera's rough placement is known.

**Measured, not asserted: V11.** The alignment score cannot say whether the
resulting COURT COORDINATES are right, and hand-annotating landmarks would only
move the problem — my pixel estimates become the ground truth and could be wrong
in the same way the registration is. So `scripts/validate_registration.py`
checks physics instead: players do not teleport, so the implied speeds must look
like basketball.

    12 frames registered, median score 0.397; 14 tracks, 102 steps
    speed ft/s: p50 7.9   p90 21.7   p95 25.0   max 33.9
    within sprint (25 ft/s): 95.1%
    court extent: x -6.3..54.3 (court 0..50), y -1.9..39.0 (0..47)

A median of 7.9 ft/s is what sustained NBA movement actually looks like, and a
registration wrong in scale would inflate every one of these. So scale and
frame-to-frame stability are sound.

**What V11 still cannot see.** A court offset by a constant would pass it
unchanged — this measures relative motion, not absolute position. The x extent
running -6.3 to 54.3 against a 50 ft court is a real hint of that: some of it is
players genuinely out of bounds, some is offset error. And 5% of steps still
exceed sprint speed, part registration jitter and part track ID switches.

**Caveat that has not gone away.** 0.40 on real footage is far from the 0.89 the
synthetic court reaches. Registration works on this footage; it is not something
to run unattended on arbitrary broadcasts without checking the score, the
on-court player count, and V11.

## Wiring up a real game

1. `pip install nba_api`
2. Fetch play-by-play for the GameID (`playbyplayv3`).
3. OCR the scoreboard clock — fixed ROI, digits only.
4. Map video time → game clock → play-by-play window.
5. Feed the plays to `align` alongside pipeline events.

Steps 1, 2 and 5 are ready. Step 3 is the only new vision work, and it is the
easy kind.
