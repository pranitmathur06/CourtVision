"""The hybrid: structured filters choose the candidates, vectors rank them.

THREE ARMS, so the evaluation can tell which part is doing the work:

    regex     what the page does today -- literal matching over the card text
    vector    nearest neighbours, whole corpus, no filter
    hybrid    filters resolve a candidate set, vectors rank inside it

If `hybrid` does not beat `vector`, the filters are decoration and should be
deleted. If `vector` does not beat `regex` on the questions with no keyword, the
embedding layer is not earning its place. Both of those are gates in
`scripts/eval_retrieval.py` and both are stated before any number was produced.

THE FILTERS ARE DELIBERATELY DUMB. They are what a relational store can do in
one statement -- equality on a game, a period, a team; membership in an
array-valued player column -- because that is the division of labour the plan
argues for and the point is to test it, not to hide a second ranker inside the
filter and call the result a hybrid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from courtvision.retrieval.store import Hit, VectorStore

#: Words that are never a player, a team or an action, so a name-like token in a
#: question is not confused with them.
STOPWORDS = frozenset("""
who what when where which how many much did was were is are the a an of in on at
and or to for from did do does done game quarter period half first second third
fourth ot overtime points point score scored shot shots make made makes miss
missed team play plays possession possessions time times most least best worst
after before during with without any all
""".split())


@dataclass
class Query:
    """A question, and the structure pulled out of it by the filters."""
    text: str
    game_id: str | None = None
    period: int | None = None
    players: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()


ACTION_WORDS = {
    "rebound": "rebound", "rebounds": "rebound", "board": "rebound",
    "turnover": "turnover", "turnovers": "turnover", "steal": "steal",
    "steals": "steal", "block": "block", "blocks": "block",
    "assist": "assist", "assists": "assist", "foul": "foul", "fouls": "foul",
    "three": "made shot", "threes": "made shot", "dunk": "made shot",
    "layup": "made shot", "jumper": "made shot",
    "free throw": "free throw", "free throws": "free throw",
}


def parse(question: str, known_players: set[str] | None = None,
          known_games: set[str] | None = None) -> Query:
    """Pull the filterable structure out of a question. Everything else ranks."""
    lowered = question.lower()
    period = None
    for word, number in (("first quarter", 1), ("second quarter", 2),
                         ("third quarter", 3), ("fourth quarter", 4),
                         ("1st", 1), ("2nd", 2), ("3rd", 3), ("4th", 4),
                         ("overtime", 5)):
        if word in lowered:
            period = number
            break
    actions = tuple(dict.fromkeys(
        v for k, v in ACTION_WORDS.items() if re.search(rf"\b{k}\b", lowered)))
    players: tuple[str, ...] = ()
    if known_players:
        found = [p for p in known_players
                 if re.search(rf"\b{re.escape(p.lower())}\b", lowered)]
        players = tuple(dict.fromkeys(found))
    game = None
    if known_games:
        game = next((g for g in known_games if g.lower() in lowered), None)
    return Query(question, game_id=game, period=period, players=players,
                 actions=actions)


def candidates(query: Query, fields: dict[str, dict]) -> list[str] | None:
    """The card ids a relational store would return. None means no filter fired.

    Returning None rather than every id is deliberate: "no filter applied" and
    "a filter matched everything" are different states, and a store that cannot
    tell them apart cannot report that its filters did nothing.
    """
    if not (query.game_id or query.period or query.players or query.actions):
        return None
    out = []
    for card_id, f in fields.items():
        if query.game_id and f.get("game_id") != query.game_id:
            continue
        if query.period and f.get("period") != query.period:
            continue
        if query.players and not (set(query.players) & set(f.get("players", []))):
            continue
        if query.actions and not (set(query.actions) & set(f.get("actions", []))):
            continue
        out.append(card_id)
    return out


def regex_search(question: str, texts: dict[str, str], limit: int = 10) -> list[Hit]:
    """What the page does today: literal tokens, scored by how many match.

    The baseline the embedding has to beat. It is not a straw man -- on a
    question naming a player and an action it is close to perfect, which is
    exactly why gate G1 exists.
    """
    tokens = [t for t in re.findall(r"[a-z0-9']+", question.lower())
              if t not in STOPWORDS and len(t) > 2]
    if not tokens:
        return []
    scored = []
    for card_id, text in texts.items():
        low = text.lower()
        hits = sum(1 for t in tokens if t in low)
        if hits:
            scored.append((hits / len(tokens), card_id))
    scored.sort(key=lambda e: (-e[0], e[1]))
    return [Hit(card_id, score, {}) for score, card_id in scored[:limit]]


def vector_search(question: str, store: VectorStore, limit: int = 10,
                  restrict: list[str] | None = None) -> list[Hit]:
    from courtvision.retrieval.embed import encode_query

    return store.search(encode_query(question), limit=limit, candidates=restrict)


def hybrid_search(question: str, store: VectorStore, fields: dict[str, dict],
                  known_players: set[str] | None = None,
                  known_games: set[str] | None = None,
                  limit: int = 10) -> list[Hit]:
    """Filters choose, vectors rank. The order is the whole argument."""
    query = parse(question, known_players, known_games)
    allowed = candidates(query, fields)
    if allowed is not None and not allowed:
        # A filter that matches nothing returns nothing. Falling back to the
        # whole corpus here would answer a question about one game with rows
        # from another and look like a good retrieval while doing it.
        return []
    return vector_search(question, store, limit=limit, restrict=allowed)
