"""Save one game's official play-by-play to disk, once.

`data/pbp_cache/` has held 1,230 games of 2015-16 actions since the tracking-data
work and NOTHING HAS EVER READ IT. Meanwhile `align_game_events.py` and
`score_game_end_to_end.py` each fetch `stats.nba.com` on every run, which is
fine for three games and is not fine at the scale this is headed for: the
endpoint rate-limits, times out, and occasionally answers with HTML, and a
pipeline stage that fails for a reason outside the repository is a stage that
cannot be part of an acceptance test.

The cached shape is exactly what `playbyplayv3` returns under `game.actions` --
the same shape the 2015-16 files are already in -- so a reader needs no branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(game_id: str, cache: Path | str | None = None,
         timeout: int = 60, allow_fetch: bool = True) -> list[dict]:
    """This game's actions, from disk if they are there and the network if not.

    The cache is written only after a fetch returns something non-empty: an
    empty file would be indistinguishable from a game with no plays and would
    poison every later run.
    """
    path = Path(cache or f"data/pbp_cache/{game_id}.json")
    if path.exists():
        actions = json.loads(path.read_text())
        if actions:
            return actions
    if not allow_fetch:
        raise FileNotFoundError(f"{path} is missing and fetching is disabled")
    from nba_api.stats.endpoints import playbyplayv3
    actions = playbyplayv3.PlayByPlayV3(
        game_id=game_id, timeout=timeout).get_dict()["game"]["actions"]
    if actions:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actions))
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--refresh", action="store_true",
                        help="fetch even if the cache has it")
    args = parser.parse_args()
    path = Path(args.out or f"data/pbp_cache/{args.game_id}.json")
    if args.refresh and path.exists():
        path.unlink()
    actions = load(args.game_id, path)
    periods = sorted({a.get("period") for a in actions if a.get("period")})
    print(f"{len(actions)} actions, periods {periods} -> {path}")
    return 0 if actions else 1


if __name__ == "__main__":
    raise SystemExit(main())
