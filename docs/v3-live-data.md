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

**Not built: plays that are conventions rather than geometry.** "Spain
pick-and-roll" is a back-screen on the roller by a third player, and set calls
like "Horns Flare" are coaching vocabulary invisible to a camera; teams differ
on the names. Those need labelled play types.

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

**Still not built: the search.** With a scoring function in place, automatic
registration becomes an optimisation over camera parameters rather than a
perception problem. That search is not written, so the homography is still
supplied per camera.

## Wiring up a real game

1. `pip install nba_api`
2. Fetch play-by-play for the GameID (`playbyplayv3`).
3. OCR the scoreboard clock — fixed ROI, digits only.
4. Map video time → game clock → play-by-play window.
5. Feed the plays to `align` alongside pipeline events.

Steps 1, 2 and 5 are ready. Step 3 is the only new vision work, and it is the
easy kind.
