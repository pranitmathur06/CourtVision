"""Full names for a game, so a question can use the name a person would type.

The play-by-play writes surnames -- "Gilgeous-Alexander 15' Pullup Jump Shot" --
and a reader types "Shai Gilgeous-Alexander". The box score has both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from nba_api.stats.endpoints import boxscoretraditionalv3

    box = boxscoretraditionalv3.BoxScoreTraditionalV3(
        game_id=args.game_id, timeout=60).get_dict()["boxScoreTraditional"]
    roster: dict[str, dict[str, str]] = {}
    for side in ("homeTeam", "awayTeam"):
        team = box[side]
        code = team.get("teamTricode") or side
        roster[code] = {
            str(p.get("jerseyNum") or p["personId"]):
                f"{p['firstName']} {p['familyName']}".strip()
            for p in team["players"]
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(roster, open(args.out, "w"), indent=1)
    print(f"{sum(len(v) for v in roster.values())} players "
          f"across {', '.join(roster)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
