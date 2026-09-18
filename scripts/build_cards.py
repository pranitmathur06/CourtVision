"""Possession cards for a packed stream, embedded and indexed. No cloud.

The ingest half of the retrieval layer. `scripts/eval_retrieval.py` is the
measurement half, and the plan requires the measurement to pass offline on three
games before any Cloudflare resource is created -- so this writes a local index
and nothing else.

    scripts/build_cards.py --stream outputs/games/stream_3games.json \
                           --out outputs/games/cards.json

Embedding 358 cards takes seconds on this laptop. A season is 263,000 and would
be minutes on a rented GPU, which is the number that decides whether the card
layer is affordable at the scale this project is aimed at: it is.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.retrieval.cards import cards_from_stream  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--game", default=None, help="one game id, or all")
    parser.add_argument("--source", default="official play-by-play",
                        help="goes INSIDE the embedded text, not beside it")
    parser.add_argument("--vectors", default=None,
                        help="also embed and write a .npy beside the cards")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    stream = json.loads(Path(args.stream).read_text())
    cards = cards_from_stream(stream, args.game, args.source)
    if not cards:
        print("FAIL - no cards; is this a packed stream?")
        return 1
    payload = [{"text": c.text, **c.fields()} for c in cards]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    lengths = sorted(len(c.rows) for c in cards)
    print(f"{len(cards)} cards -> {out}")
    print(f"  events per card p50 {lengths[len(lengths) // 2]}, "
          f"p90 {lengths[int(0.9 * len(lengths))]}, longest {lengths[-1]}")
    print(f"  {sum(1 for c in cards if c.points)} scored, "
          f"{len({p for c in cards for p in c.players})} distinct players named")

    if args.vectors:
        import numpy as np

        from courtvision.retrieval.embed import MODEL, encode_documents
        began = time.time()
        vectors = encode_documents([c.text for c in cards])
        np.save(args.vectors, vectors)
        rate = len(cards) / max(time.time() - began, 1e-9)
        print(f"  {MODEL}: {vectors.shape} -> {args.vectors} "
              f"({rate:.0f} cards a second)")
        print(f"  a 1,315-game season at ~{len(cards) // max(len(set(c.game_id for c in cards)), 1)} "
              f"cards a game is {1315 * (len(cards) // max(len(set(c.game_id for c in cards)), 1)):,} "
              f"cards, about {1315 * (len(cards) // max(len(set(c.game_id for c in cards)), 1)) / rate / 60:.0f} "
              f"minutes on this laptop and far less on a GPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
