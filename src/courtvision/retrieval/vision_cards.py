"""Cards built from what the CAMERA saw, with no play-by-play anywhere in them.

WHY THIS FILE EXISTS. `cards.py` builds a card per possession out of
`stream_*.json`, which is the NBA's own play-by-play with a timestamp attached.
Those cards are useful and they are honest -- each one says in its own text that
it came from the official record -- but **they are not vision-based retrieval.**
Embedding the feed and searching it is a better index over a record somebody else
produced. The goal is film analysis: asking the footage what happened in it.

WHAT THE VISION STACK ACTUALLY LEAVES BEHIND, per clip, in the overlay files:

    p   the players it tracked standing on the court
    b   one ball, chosen as a path through the candidates
    r   the rim, tracked rather than pinned
    s   the SUBJECT -- the player it believes is carrying the ball

Nothing in that came from a feed. From it a card can say how many players were
on the floor, whether a carrier was identified and for how much of the play,
whether the ball changed hands, how far the carrier moved and in which
direction, and how close the ball came to the rim -- **facts with no keyword**,
which is the entire justification for a vector arm over a regular expression.

EVERYTHING IS IN RIM WIDTHS, not pixels. A rim is 18 inches across and the stack
tracks it, so it is the one object in frame whose real size is known. Distances
in pixels mean different things on a 720p Finals broadcast and a 1080p regular
season one, and this repository has already been bitten twice by constants that
were secretly resolution-dependent.

AND EVERY CARD SAYS IT MIGHT BE WRONG. The handler this is built on is right
about 50 to 66% of the time on uniformly sampled frames. A card that asserts
"the carrier drove to the rim" without saying who says so teaches a reader, and
a model reading it back, to treat a coin-flip as a fact.
"""

from __future__ import annotations

import math
import statistics

from courtvision.retrieval.cards import Card

#: A rim is 18 inches. Distances below are in rim widths so a number means the
#: same thing on every broadcast.
INCHES_PER_RIM = 18.0
#: Below this, the ball is at the basket.
AT_THE_RIM = 1.5
#: A carrier held for less of the clip than this is not really identified.
CARRIER_SHARE = 0.25


def _centre(box):
    return ((box[1] + box[3]) / 2.0, (box[2] + box[4]) / 2.0)


def _feet(box):
    return ((box[1] + box[3]) / 2.0, float(box[4]))


def describe(frames, fps: float) -> dict:
    """Vision-only facts about one clip. No feed, no labels, no names."""
    players, balls, rims, subjects = [], [], [], []
    for index, frame in enumerate(frames):
        rows = {"p": [], "b": [], "r": [], "s": []}
        for row in frame:
            if row[0] in rows:
                rows[row[0]].append(row)
        players.append(len(rows["p"]))
        balls.append((index, _centre(rows["b"][0])) if rows["b"] else None)
        rims.append(_centre(rows["r"][0]) if rows["r"] else None)
        subjects.append(_feet(rows["s"][0]) if rows["s"] else None)

    seen_rims = [r for r in rims if r]
    widths = [row[3] - row[1] for frame in frames for row in frame if row[0] == "r"]
    rim_width = statistics.median(widths) if widths else 0.0
    rim = (statistics.median([r[0] for r in seen_rims]),
           statistics.median([r[1] for r in seen_rims])) if seen_rims else None

    def in_rims(pixels: float) -> float:
        return pixels / rim_width if rim_width > 1 else float("nan")

    held = [s for s in subjects if s]
    carrier_share = len(held) / max(len(frames), 1)
    # How far the carrier travelled, and whether it was toward the basket.
    travel = toward = float("nan")
    if len(held) >= 2 and rim:
        steps = [math.dist(a, b) for a, b in zip(held, held[1:])]
        travel = in_rims(sum(steps))
        toward = in_rims(math.dist(held[0], rim) - math.dist(held[-1], rim))
    # Did the carrier change? A jump the length of a person between consecutive
    # subject boxes is a different player, not the same one moving.
    changes = 0
    if rim_width > 1:
        for a, b in zip(held, held[1:]):
            if in_rims(math.dist(a, b)) > 3.0:
                changes += 1

    nearest = float("nan")
    if rim:
        gaps = [in_rims(math.dist(b[1], rim)) for b in balls if b]
        nearest = min(gaps) if gaps else float("nan")
    ball_share = sum(1 for b in balls if b) / max(len(frames), 1)

    return {"seconds": len(frames) / max(fps, 1e-9),
            "players_p50": statistics.median(players) if players else 0,
            "carrier_share": carrier_share, "carrier_changes": changes,
            "carrier_travel_rims": travel, "carrier_toward_rims": toward,
            "ball_share": ball_share, "ball_nearest_rim_rims": nearest,
            "rim_seen_share": len(seen_rims) / max(len(frames), 1)}


def _sentence(facts: dict, label: str) -> str:
    """The card a person reads, and the one that gets embedded."""
    parts = [f"{label}. Six seconds of film."
             if facts["seconds"] > 5 else f"{label}. A short clip."]
    parts.append(f"The camera has {facts['players_p50']:.0f} players on the "
                 f"floor.")
    if facts["carrier_share"] < CARRIER_SHARE:
        parts.append("No ball carrier is identified for most of it -- the ball "
                     "is loose, in the air, or the stack cannot tell who has it.")
    else:
        parts.append(f"One player is carrying for "
                     f"{facts['carrier_share']:.0%} of it")
        if facts["carrier_changes"]:
            parts[-1] += (f", and the ball changes hands "
                          f"{facts['carrier_changes']} times.")
        else:
            parts[-1] += " without changing hands."
        if not math.isnan(facts["carrier_toward_rims"]):
            if facts["carrier_toward_rims"] > 2:
                parts.append("The carrier drives toward the basket, closing "
                             f"{facts['carrier_toward_rims']:.0f} rim widths.")
            elif facts["carrier_toward_rims"] < -2:
                parts.append("The carrier moves away from the basket, out by "
                             f"{abs(facts['carrier_toward_rims']):.0f} rim widths.")
            else:
                parts.append("The carrier holds his ground near the perimeter.")
    if not math.isnan(facts["ball_nearest_rim_rims"]):
        if facts["ball_nearest_rim_rims"] <= AT_THE_RIM:
            parts.append("The ball reaches the rim, so a shot or a tip went up.")
        elif facts["ball_nearest_rim_rims"] <= 6:
            parts.append("The ball works into the middle without reaching the "
                         "rim.")
        else:
            parts.append("The ball never comes near the basket.")
    if facts["rim_seen_share"] < 0.3:
        parts.append("The basket is mostly out of shot, so anything about "
                     "distance to it is weak here.")
    parts.append("Recorded from the VISION pipeline and nothing else -- no "
                 "play-by-play was used. The carrier it names is right about "
                 "half to two thirds of the time on uniformly sampled frames, "
                 "so treat who had the ball as a guess and the geometry as "
                 "better than the identity.")
    return " ".join(parts)


def cards_from_overlays(overlays: dict, game_id: str, label: str,
                        index: dict | None = None) -> list[Card]:
    """One card per clip, from the vision stack's own output."""
    fps = float(overlays.get("fps") or 30.0)
    starts = {c["clip"]: c.get("start_s") for c in (index or {}).get("clips", [])
              if c.get("clip")}
    out: list[Card] = []
    for name in sorted(overlays.get("clips", {})):
        frames = overlays["clips"][name]
        if not frames:
            continue
        facts = describe(frames, fps)
        actions = []
        if facts["ball_nearest_rim_rims"] <= AT_THE_RIM:
            actions.append("ball reaches rim")
        if facts["carrier_changes"]:
            actions.append("ball changes hands")
        if facts["carrier_share"] < CARRIER_SHARE:
            actions.append("no carrier")
        if facts["carrier_toward_rims"] > 2:
            actions.append("drive")
        out.append(Card(
            card_id=f"{game_id}:vision:{name}", game_id=game_id,
            period=0, start_clock=str(starts.get(name, "")),
            end_clock="", team="", players=[], actions=actions,
            points=0, source="the vision pipeline",
            text=_sentence(facts, f"{label}, clip {name}"),
            rows=[]))
    return out
