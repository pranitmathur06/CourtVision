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

## Not done: play recognition

"What play are they running" (Horns, Spain pick-and-roll) needs court
registration — a homography mapping players onto court coordinates — then
formation classification over all ten players. That is real research, well past
this layer, and it is the honest next hard problem.

## Wiring up a real game

1. `pip install nba_api`
2. Fetch play-by-play for the GameID (`playbyplayv3`).
3. OCR the scoreboard clock — fixed ROI, digits only.
4. Map video time → game clock → play-by-play window.
5. Feed the plays to `align` alongside pipeline events.

Steps 1, 2 and 5 are ready. Step 3 is the only new vision work, and it is the
easy kind.
