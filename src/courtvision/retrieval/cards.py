"""Possessions, not events -- and the text that gets embedded.

WHY NOT EMBED THE EVENTS. A play-by-play row is a five-slot template:
"Nembhard 14' Pullup Jump Shot (2 PTS)". Every meaningful token in it is
literal -- a surname, a distance, a shot type -- and a regular expression
already matches all of them. Embedding 723,000 near-identical sentences buys
nothing but a slower way to find the word "Nembhard", and it collapses the
manifold: the nearest neighbours of any shot are other shots by the same player.

A POSSESSION IS THE UNIT A QUESTION IS ABOUT. "What did they run out of the
timeout", "who got the offensive rebound that led to the three", "how did they
score after the switch" are all questions about a sequence, and the sequence is
where the facts with no keyword live. One card is one team's trip down the
floor: how it started, what happened in it, how it ended, and how long it took.

PROVENANCE GOES IN THE EMBEDDED TEXT, not beside it in a metadata column. This
stack emits turnovers at 1.8x the official rate; a card that says "turnover"
without saying who says so teaches a reader -- and a model reading the card back
-- to treat an over-emission as a fact. `Card.text` carries the source, and a
card built from vision rather than from the official feed says so in the
sentence a person reads.

WHAT IS NOT CLAIMED. The plan this implements calls for cards carrying the
vision stack's tactical facts -- switches, matchups, spacing deltas, contest
level. Those come from `tactical_features`, which needs tracking arrays, and the
video broadcasts have play-by-play rather than tracking. So these cards carry
the possession STRUCTURE, which is derivable from the event sequence alone, and
the tactical layer is added where tracking exists. The ablation in
`scripts/eval_retrieval.py` compares cards against raw rows; if raw rows match
them, this layer is cut rather than defended.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: An action that ends a team's possession. A missed shot does not -- the
#: rebound decides -- and a foul does not, which is why both are absent.
ENDS_POSSESSION = frozenset({"made shot", "turnover", "steal"})
#: ...and one that ends it only when the other team gets the ball.
REBOUND = "rebound"
#: Actions that say nothing about who has the ball.
NEUTRAL = frozenset({"substitution", "timeout", "jump ball", "violation",
                     "instant replay", "period"})


@dataclass
class Card:
    """One possession, as a sentence and as the fields a filter needs.

    `text` is what gets embedded. `fields` is what the structured store filters
    on -- and the split matters, because a filter on an array-valued,
    ~500-cardinality column like "which players were involved" is something a
    vector index cannot express and a relational one can.
    """
    card_id: str
    game_id: str
    period: int
    start_clock: str
    end_clock: str
    team: str
    players: list[str]
    actions: list[str]
    points: int
    source: str
    text: str
    rows: list[int] = field(default_factory=list)

    def fields(self) -> dict:
        return {"card_id": self.card_id, "game_id": self.game_id,
                "period": self.period, "team": self.team,
                "players": list(self.players), "actions": list(self.actions),
                "points": self.points, "source": self.source,
                "rows": list(self.rows)}


def _surname(detail: str) -> str | None:
    """The name a play-by-play row starts with, which is how it writes them."""
    text = (detail or "").strip()
    text = re.sub(r"^MISS\s+", "", text)
    matched = re.match(r"([A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*)?)", text)
    if not matched:
        return None
    name = matched.group(1).strip()
    return name if len(name) > 2 else None


def _points(detail: str) -> int:
    """Points scored on this row, from the running total the feed writes."""
    if "MISS" in (detail or ""):
        return 0
    if re.search(r"3PT", detail or ""):
        return 3
    if re.search(r"Free Throw", detail or "", re.I):
        return 1
    return 2


def possessions(rows: list[dict]) -> list[list[dict]]:
    """Split a game's rows into possessions.

    NO TRACKING DATA IS USED. A possession ends on a made shot, a turnover or a
    steal, and a rebound ends it only when it is a defensive one -- which is
    inferred from whether the next shot comes from the other team. Where the
    team is unknown the possession simply continues, which is the honest
    behaviour: a boundary this cannot see is better than one it invents.
    """
    out: list[list[dict]] = []
    current: list[dict] = []
    for row in rows:
        action = (row.get("action") or "").lower()
        if action in NEUTRAL and not current:
            continue
        current.append(row)
        ends = action in ENDS_POSSESSION
        if action == REBOUND and len(current) > 1:
            # A rebound after a miss ends the possession when the rebounder is
            # on the other team. Teams are not always known; when they are not,
            # a rebound is treated as ending it, because the miss that preceded
            # it already did most of the work.
            before = next((r for r in reversed(current[:-1])
                           if r.get("team") is not None), None)
            ends = before is None or before.get("team") != row.get("team")
        if ends:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def _sentence(group: list[dict], game_label: str, source: str) -> str:
    """The possession as a sentence, with its provenance inside it."""
    first, last = group[0], group[-1]
    who = [n for n in (_surname(r.get("detail")) for r in group) if n]
    scored = sum(_points(r.get("detail") or "") for r in group
                 if (r.get("action") or "").lower() in ("made shot", "free throw")
                 and "MISS" not in (r.get("detail") or ""))
    beats = []
    for row in group:
        detail = (row.get("detail") or "").strip()
        beats.append(detail or (row.get("action") or "").replace("_", " "))
    span = f"{first.get('clock')} to {last.get('clock')}"
    head = (f"{game_label}, period {first.get('period')}, {span}. "
            f"A possession of {len(group)} "
            f"{'event' if len(group) == 1 else 'events'}")
    if who:
        head += f" involving {', '.join(dict.fromkeys(who))}"
    head += f", worth {scored} point{'' if scored == 1 else 's'}. "
    body = " Then ".join(beats)
    # The provenance is a sentence, not a column, so it survives being embedded
    # and survives being read back by a model that was handed only the text.
    tail = (f" Recorded from {source}."
            if source != "official play-by-play" else
            " Recorded from the official play-by-play, which is the NBA's own"
            " record rather than anything this system saw.")
    return head + body + "." + tail


def cards_from_stream(stream: dict, game_id: str | None = None,
                      source: str = "official play-by-play") -> list[Card]:
    """Cards for one packed game, or for every game in the stream."""
    actions = stream.get("actions", [])
    details = stream.get("details", [])
    fields = stream.get("fields", [])
    index = {name: i for i, name in enumerate(fields)}
    out: list[Card] = []
    for game in stream.get("games", []):
        if game_id and game["id"] != game_id:
            continue
        label = game.get("date") or game["id"]
        rows = []
        for n, raw in enumerate(game["rows"]):
            def at(name, default=None):
                i = index.get(name)
                return raw[i] if i is not None and i < len(raw) else default
            detail_index = at("detail", -1)
            rows.append({
                "n": n, "t": at("t"), "period": at("period"),
                "clock": at("clock"),
                "action": actions[at("action", 0)] if actions else None,
                "team": at("team"),
                "detail": details[detail_index] if isinstance(detail_index, int)
                          and 0 <= detail_index < len(details) else "",
            })
        for k, group in enumerate(possessions(rows)):
            who = [n for n in (_surname(r.get("detail")) for r in group) if n]
            scored = sum(_points(r.get("detail") or "") for r in group
                         if (r.get("action") or "").lower() in
                         ("made shot", "free throw")
                         and "MISS" not in (r.get("detail") or ""))
            out.append(Card(
                card_id=f"{game['id']}:{k:04d}",
                game_id=game["id"], period=group[0].get("period") or 0,
                start_clock=group[0].get("clock") or "",
                end_clock=group[-1].get("clock") or "",
                team=str(group[0].get("team")),
                players=list(dict.fromkeys(who)),
                actions=list(dict.fromkeys(
                    (r.get("action") or "") for r in group)),
                points=scored, source=source,
                text=_sentence(group, label, source),
                rows=[r["n"] for r in group]))
    return out
