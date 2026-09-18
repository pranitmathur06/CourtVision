"""Does the embedding earn its place? Three gates, declared before any number.

    G1  the hybrid must NOT LOSE to the regular expression on counting
        questions. The page answers those correctly today by reading the whole
        log, and a retrieval layer that makes them worse is a regression however
        good it is elsewhere.
    G2  the vector arm must BEAT the regular expression on PARAPHRASE questions
        by at least ten points of recall@10, at p < 0.05. An embedding that only
        matches literal names is a slower regex and should be redesigned rather
        than shipped.
    G3  vector-only must LOSE to the hybrid. If it does not, the structured
        filters are decoration and should be deleted.

HOW THE QUESTIONS ARE MADE, AND WHAT THAT COSTS. They are synthesised from the
cards, so their ground truth is exact -- the set of cards that genuinely satisfy
each question is computed from the card fields, not judged. The price is the
thing the plan warned about: **a question written by the card generator shares
its vocabulary.** That is why the paraphrase category exists and why it is
reported separately. Its wording comes from a hand-written synonym table using
words that are NOT in any card -- "gave the ball away", "cleaned up the miss",
"from way out" -- which is the only second author available here.

So the literal categories measure whether retrieval works at all, and only the
PARAPHRASE category is evidence about the embedding, because only there do the
question's words fail to appear in the answer. A reader should discount the rest
accordingly, and the report prints them separately for that reason rather than
blending them into one figure.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.retrieval.cards import cards_from_stream  # noqa: E402
from courtvision.retrieval.retriever import (hybrid_search,  # noqa: E402
                                             regex_search, vector_search)
from courtvision.retrieval.store import LocalStore  # noqa: E402
from courtvision.stats import mcnemar, wilson  # noqa: E402

#: Words a fan would type that appear in NO card. The second vocabulary.
PARAPHRASE = {
    "made shot": ["put it in", "finished at the rim", "converted",
                  "got a bucket", "scored"],
    "rebound": ["cleaned up the miss", "grabbed the carom",
                "won the ball off the rim"],
    "turnover": ["gave the ball away", "coughed it up", "lost possession"],
    "steal": ["picked his pocket", "took it away"],
    "block": ["sent it back", "rejected at the rim", "denied the shot"],
    "free throw": ["from the charity stripe", "at the line"],
    "foul": ["was whistled", "picked up a personal"],
}
#: Six categories. `counting` and `paraphrase` carry the gates.
#: `paraphrase` is split by whether the question contains a literal player
#: name, because that decides whether the regular expression has anything to
#: match at all. Blending them hides the only comparison that is about the
#: embedding: the first version of this file asked 45 paraphrase questions,
#: mostly unnamed, and the vector arm won by 15.6 points; expanding to 1,556,
#: almost all NAMED, flipped it to a 11.9-point loss. Neither number was wrong.
#: They are answers to different questions.
CATEGORIES = ("counting", "player_action", "paraphrase_named",
              "paraphrase_unnamed", "sequence", "temporal", "cross_game")


def build_questions(cards, seed: int = 0) -> list[dict]:
    """(question, the card ids that genuinely answer it, category)."""
    rng = random.Random(seed)
    by_player = defaultdict(list)
    by_action = defaultdict(list)
    by_period = defaultdict(list)
    by_game = defaultdict(list)
    for c in cards:
        for p in c.players:
            by_player[p].append(c.card_id)
        for a in c.actions:
            if a:
                by_action[a].append(c.card_id)
        by_period[c.period].append(c.card_id)
        by_game[c.game_id].append(c.card_id)

    out: list[dict] = []
    players = [p for p, v in by_player.items() if len(v) >= 3]

    for player in rng.sample(players, min(30, len(players))):
        out.append({"q": f"how many possessions involved {player}",
                    "truth": set(by_player[player]), "cat": "counting"})
    for player in rng.sample(players, min(30, len(players))):
        for action in ("rebound", "turnover", "made shot"):
            ids = set(by_player[player]) & set(by_action.get(action, []))
            if ids:
                out.append({"q": f"{player} {action}",
                            "truth": ids, "cat": "player_action"})
                break
    for action, phrases in PARAPHRASE.items():
        ids = set(by_action.get(action, []))
        if not ids:
            continue
        for phrase in phrases:
            out.append({"q": f"when did somebody {phrase}",
                        "truth": ids, "cat": "paraphrase_unnamed"})
    # Every player crossed with every paraphrase they actually have a card for.
    # The first version of this took one phrase per player and produced 45
    # paraphrase questions; the gate asks for p < 0.05 and 45 cannot deliver it
    # at any effect size this layer is likely to have. The plan says 300
    # questions for exactly this reason.
    for player in players:
        for action, phrases in PARAPHRASE.items():
            ids = set(by_player[player]) & set(by_action.get(action, []))
            if not ids:
                continue
            for phrase in phrases:
                out.append({"q": f"{player} {phrase}",
                            "truth": ids, "cat": "paraphrase_named"})
    multi = [c for c in cards if len(c.rows) >= 4]
    for c in rng.sample(multi, min(30, len(multi))):
        out.append({"q": f"a long possession in period {c.period} "
                         f"involving {', '.join(c.players[:2])}",
                    "truth": {c.card_id}, "cat": "sequence"})
    for period, ids in by_period.items():
        if period and len(ids) >= 5:
            for word in (f"period {period}", f"in the {period} quarter"):
                out.append({"q": f"what happened in {word}",
                            "truth": set(ids), "cat": "temporal"})
    for game, ids in by_game.items():
        if len(ids) >= 5:
            out.append({"q": f"possessions in game {game}",
                        "truth": set(ids), "cat": "cross_game"})
    return out


def recall_at(hits, truth, k: int = 10) -> bool:
    return any(h.card_id in truth for h in hits[:k])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", default="outputs/games/stream_3games.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--raw-detail", action="store_true",
                        help="ABLATION 1: embed the raw event text instead of "
                             "the possession card, on the same questions. If "
                             "this matches, the card layer is cut.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    stream = json.loads(Path(args.stream).read_text())
    cards = cards_from_stream(stream)
    if args.raw_detail:
        for c in cards:
            details = [stream["details"][i] for i in range(len(stream["details"]))]
            c.text = " ".join(details[r] if r < len(details) else ""
                              for r in c.rows) or c.text
    print(f"  {len(cards)} cards from {args.stream}")

    from courtvision.retrieval.embed import encode_documents
    vectors = encode_documents([c.text for c in cards])
    store = LocalStore()
    store.add([c.card_id for c in cards], vectors, [c.fields() for c in cards])
    texts = {c.card_id: c.text for c in cards}
    fields = {c.card_id: c.fields() for c in cards}
    known_players = {p for c in cards for p in c.players}
    known_games = {c.game_id for c in cards}

    questions = build_questions(cards)
    print(f"  {len(questions)} questions over {len(CATEGORIES)} categories\n")

    arms = {"regex": [], "vector": [], "hybrid": []}
    per_cat = {a: defaultdict(list) for a in arms}
    for item in questions:
        got = {
            "regex": regex_search(item["q"], texts, args.limit),
            "vector": vector_search(item["q"], store, args.limit),
            "hybrid": hybrid_search(item["q"], store, fields, known_players,
                                    known_games, args.limit),
        }
        for arm, hits in got.items():
            ok = recall_at(hits, item["truth"], args.limit)
            arms[arm].append(ok)
            per_cat[arm][item["cat"]].append(ok)

    print(f"  recall@{args.limit}, per category, NEVER blended")
    print(f"  {'category':<14}{'n':>5}{'regex':>9}{'vector':>9}{'hybrid':>9}")
    for cat in CATEGORIES:
        n = len(per_cat["regex"].get(cat, []))
        if not n:
            continue
        row = "".join(f"{sum(per_cat[a][cat]) / n:>9.3f}" for a in
                      ("regex", "vector", "hybrid"))
        print(f"  {cat:<14}{n:>5}{row}")
    n = len(questions)
    print(f"  {'ALL':<14}{n:>5}"
          + "".join(f"{sum(arms[a]) / n:>9.3f}" for a in ("regex", "vector", "hybrid")))

    print("\n  GATES, declared before any of this was run")
    verdicts = {}

    count_regex = per_cat["regex"].get("counting", [])
    count_hybrid = per_cat["hybrid"].get("counting", [])
    only_h, only_r, p = mcnemar(count_hybrid, count_regex)
    passed = sum(count_hybrid) >= sum(count_regex) or p >= 0.05
    verdicts["G1"] = passed
    print(f"    G1  hybrid must not lose to regex on COUNTING: "
          f"{sum(count_hybrid)}/{len(count_hybrid)} against "
          f"{sum(count_regex)}/{len(count_regex)}, p = {p:.4f}  "
          f"-> {'PASS' if passed else 'FAIL'}")

    para_regex = (per_cat["regex"].get("paraphrase_named", [])
                  + per_cat["regex"].get("paraphrase_unnamed", []))
    para_vector = (per_cat["vector"].get("paraphrase_named", [])
                   + per_cat["vector"].get("paraphrase_unnamed", []))
    only_v, only_r, p = mcnemar(para_vector, para_regex)
    gap = ((sum(para_vector) - sum(para_regex)) / max(len(para_regex), 1))
    passed = gap >= 0.10 and p < 0.05
    verdicts["G2"] = passed
    print(f"    G2  vector must beat regex on PARAPHRASE by >=10 points at "
          f"p<0.05:\n        {sum(para_vector)}/{len(para_vector)} against "
          f"{sum(para_regex)}/{len(para_regex)} = {gap:+.3f}, p = {p:.4f}  "
          f"-> {'PASS' if passed else 'FAIL'}")

    only_h, only_v, p = mcnemar(arms["hybrid"], arms["vector"])
    passed = sum(arms["hybrid"]) > sum(arms["vector"])
    verdicts["G3"] = passed
    print(f"    G3  vector-only must lose to hybrid: "
          f"{sum(arms['hybrid'])}/{n} against {sum(arms['vector'])}/{n}, "
          f"p = {p:.4f}  -> {'PASS' if passed else 'FAIL'}")

    # The diagnostic the gate cannot carry: on the questions with NO literal
    # name, the regular expression has nothing to match and the comparison is
    # about the embedding alone.
    un_r = per_cat["regex"].get("paraphrase_unnamed", [])
    un_v = per_cat["vector"].get("paraphrase_unnamed", [])
    if un_r:
        only_v, only_r, p = mcnemar(un_v, un_r)
        gap = (sum(un_v) - sum(un_r)) / len(un_r)
        print(f"\n    diagnostic, NOT a gate -- paraphrase questions naming no "
              f"player:\n        vector {sum(un_v)}/{len(un_v)} against regex "
              f"{sum(un_r)}/{len(un_r)} = {gap:+.3f}, p = {p:.4f}")
        low, high = wilson(sum(un_v), len(un_v))
        print(f"        the vector arm's interval there is {low:.2f}-{high:.2f}, "
              f"so this subset is too small to settle anything on its own")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"cards": len(cards), "questions": n, "gates": verdicts,
             "per_category": {a: {c: [int(b) for b in v] for c, v in d.items()}
                              for a, d in per_cat.items()}}, indent=1))
        print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
