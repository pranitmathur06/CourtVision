"""Every field-goal attempt in a game, as [{period, clock_s, made}].

The input `align_shots_to_video.py` needs, and the one file on the shot-detection
path that nothing in the repository produced. `data/pbp/shots_0042400407.json`
and `shots_0042400401.json` were made by hand, which is why
`detect_shots.py:182` defaults to a path with a specific game's number in it --
and why a fourth broadcast scored its vision shot detector against the 2025
Finals Game 7's shot times. That is not a small error: the first/second-half
split, every `agrees_with_official` label and the PASS/FAIL line all come from
this file.

Made and missed only. Free throws are a different class with a different clock
behaviour (several attempts share one clock value) and `detect_shots` does not
claim them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache_pbp import load  # noqa: E402

CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")


def attempts(actions: list[dict]) -> list[dict]:
    out = []
    for action in actions:
        kind = (action.get("actionType") or "").strip()
        if kind not in ("Made Shot", "Missed Shot"):
            continue
        matched = CLOCK.fullmatch((action.get("clock") or "").strip())
        if not matched or not action.get("period"):
            continue
        out.append({"period": int(action["period"]),
                    "clock_s": int(matched.group(1)) * 60 + float(matched.group(2)),
                    "made": kind == "Made Shot",
                    "description": action.get("description") or ""})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    shots = attempts(load(args.game_id))
    if not shots:
        print(f"FAIL - no field goals in {args.game_id}'s play-by-play")
        return 1
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(shots, indent=1))
    made = sum(1 for s in shots if s["made"])
    print(f"{len(shots)} field-goal attempts ({made} made) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
