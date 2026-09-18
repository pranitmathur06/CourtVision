"""The retrieval layer's non-negotiables.

Every one of these is a property the plan named before the code existed, and
each is a way the layer could look like it works while being wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.retrieval.cards import Card, cards_from_stream, possessions
from courtvision.retrieval.retriever import (Query, candidates, parse,
                                             regex_search)
from courtvision.retrieval.store import Hit, LocalStore, VectorStore


def _stream():
    return {
        "actions": ["made shot", "rebound", "turnover", "assist"],
        "details": ["Haliburton 26' 3PT Jump Shot (3 PTS)",
                    "MISS Turner 12' Hook Shot",
                    "Nesmith REBOUND (Off:1 Def:2)",
                    "Brunson Lost Ball Turnover"],
        "fields": ["t", "period", "clock", "action", "team", "detail"],
        "games": [{"id": "G", "date": "A Game", "rows": [
            [10.0, 1, "11:46", 0, "IND", 0],
            [30.0, 1, "11:20", 1, "IND", 1],
            [32.0, 1, "11:18", 1, "IND", 2],
            [50.0, 1, "10:58", 2, "NYK", 3],
        ]}],
    }


# -- cards -------------------------------------------------------------------

def test_a_made_shot_ends_a_possession_and_a_miss_does_not():
    rows = [{"action": "shot", "team": "A", "detail": "MISS", "clock": "1"},
            {"action": "rebound", "team": "A", "detail": "", "clock": "2"},
            {"action": "made shot", "team": "A", "detail": "", "clock": "3"}]
    groups = possessions(rows)
    assert len(groups) == 1 and len(groups[0]) == 3, (
        "an offensive rebound keeps the ball; the possession continues")


def test_a_defensive_rebound_ends_the_possession():
    rows = [{"action": "shot", "team": "A", "detail": "MISS", "clock": "1"},
            {"action": "rebound", "team": "B", "detail": "", "clock": "2"},
            {"action": "made shot", "team": "B", "detail": "", "clock": "3"}]
    groups = possessions(rows)
    assert len(groups) == 2, "the other team rebounding is a change of possession"


def test_provenance_is_inside_the_embedded_text_not_beside_it():
    """This stack emits turnovers at 1.8x the official rate. A card that says
    "turnover" without saying who says so teaches a model reading it back to
    narrate an over-emission as fact."""
    cards = cards_from_stream(_stream(), source="the vision pipeline")
    assert cards
    for card in cards:
        assert "the vision pipeline" in card.text
    official = cards_from_stream(_stream())
    assert "official play-by-play" in official[0].text
    assert "NBA's own record" in official[0].text


def test_a_card_carries_the_rows_it_came_from():
    """Aggregation must be able to go back to the whole log; a card that cannot
    say which rows it summarises makes that impossible."""
    cards = cards_from_stream(_stream())
    seen = [n for c in cards for n in c.rows]
    assert sorted(seen) == [0, 1, 2, 3], "every row belongs to exactly one card"


# -- the store ---------------------------------------------------------------

def test_the_local_store_is_a_protocol_implementation():
    assert isinstance(LocalStore(), VectorStore)


def test_a_candidate_set_is_honoured_and_an_empty_one_returns_nothing():
    """A retriever that silently ranks the whole season when it was asked about
    one game returns plausible rows from the wrong night."""
    store = LocalStore()
    vectors = np.eye(3, 8, dtype=np.float32)
    store.add(["a", "b", "c"], vectors, [{}, {}, {}])
    assert [h.card_id for h in store.search(vectors[1], limit=3)][0] == "b"
    restricted = store.search(vectors[1], limit=3, candidates=["a", "c"])
    assert [h.card_id for h in restricted] == ["a", "c"]
    assert store.search(vectors[1], limit=3, candidates=[]) == []
    assert store.search(vectors[1], limit=3, candidates=["nope"]) == []


def test_scores_are_cosines():
    store = LocalStore()
    store.add(["a"], np.array([[3.0, 4.0]], dtype=np.float32), [{}])
    hit = store.search(np.array([3.0, 4.0], dtype=np.float32))[0]
    assert hit.score == pytest.approx(1.0, abs=1e-6)


def test_adding_the_same_card_twice_does_not_duplicate_it():
    store = LocalStore()
    v = np.ones((1, 4), dtype=np.float32)
    store.add(["a"], v, [{}])
    store.add(["a"], v, [{}])
    assert len(store) == 1


# -- the filters -------------------------------------------------------------

def test_no_filter_firing_is_different_from_a_filter_matching_everything():
    """"No filter applied" and "a filter matched all of them" are different
    states, and a store that cannot tell them apart cannot report that its
    filters did nothing -- which is gate G3."""
    fields = {"a": {"game_id": "G", "period": 1, "players": ["X"], "actions": []}}
    assert candidates(Query("anything at all"), fields) is None
    assert candidates(Query("q", period=1), fields) == ["a"]
    assert candidates(Query("q", period=4), fields) == []


def test_a_player_filter_matches_on_membership_not_equality():
    """Involvement is array-valued, which is the reason the plan puts filtering
    in the relational store and ranking in the vector index."""
    fields = {"a": {"players": ["Haliburton", "Turner"], "actions": [],
                    "game_id": "G", "period": 1}}
    assert candidates(Query("q", players=("Turner",)), fields) == ["a"]
    assert candidates(Query("q", players=("Brunson",)), fields) == []


def test_the_question_parser_finds_periods_and_actions():
    q = parse("how many rebounds in the third quarter", known_players=set())
    assert q.period == 3 and "rebound" in q.actions


def test_regex_search_is_the_baseline_and_actually_works():
    """It is not a straw man: on a question naming a player it is close to
    perfect, which is exactly why gate G1 exists."""
    texts = {"a": "Haliburton hit a three", "b": "Turner grabbed a rebound"}
    assert [h.card_id for h in regex_search("Haliburton three", texts)] == ["a"]
    assert regex_search("who what the", texts) == [], "stopwords alone match nothing"


# -- the hand-written question bank ------------------------------------------

def test_the_question_bank_names_no_player():
    """Its whole purpose is to be the half of the evaluation where a regular
    expression has nothing literal to hold. One capitalised token undoes that."""
    import json
    import re
    from pathlib import Path
    bank = (Path(__file__).resolve().parent.parent
            / "data" / "retrieval" / "questions_unnamed.json")
    questions = json.loads(bank.read_text())["questions"]
    assert len(questions) >= 60
    for item in questions:
        assert not re.search(r"\b[A-Z][a-z]+\b", item["q"]), item["q"]


def test_every_banked_question_has_a_predicate_over_fields_only():
    """The truth is computed from the card's FIELDS, never its text, so a
    question cannot be graded against the words it happens to share with an
    answer."""
    import json
    from pathlib import Path
    bank = (Path(__file__).resolve().parent.parent
            / "data" / "retrieval" / "questions_unnamed.json")
    allowed = {"points", "actions", "players", "period", "team", "rows",
               "len", "set", "and", "or", "not", "in"}
    for item in json.loads(bank.read_text())["questions"]:
        assert "text" not in item["where"], item
        names = {w for w in item["where"].replace("(", " ").replace(")", " ")
                 .split() if w.isalpha()}
        assert names <= allowed, (item["where"], names - allowed)
