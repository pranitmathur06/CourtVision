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

**Not built: plays.** A formation is an arrangement at one instant; a play is a
sequence. Horns is visible in a single frame, which is why it can be named.
Spain pick-and-roll is a screen, then a back-screen on the screener, then a
roll — no snapshot contains it. Getting there needs formation sequences over
time and labelled play types, and neither BARD nor SpaceJam carries play-type
labels. That is still the honest next hard problem.

**Also not built: automatic registration.** Until court lines are detected
without help, the homography must be supplied per camera.

## Wiring up a real game

1. `pip install nba_api`
2. Fetch play-by-play for the GameID (`playbyplayv3`).
3. OCR the scoreboard clock — fixed ROI, digits only.
4. Map video time → game clock → play-by-play window.
5. Feed the plays to `align` alongside pipeline events.

Steps 1, 2 and 5 are ready. Step 3 is the only new vision work, and it is the
easy kind.
