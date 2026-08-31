"""Score a full-game run against the OFFICIAL play-by-play for that game.

evaluate_game.py scores against BARD's labels, which only exist for footage
BARD cut around events. This scores a continuous broadcast against the NBA's
own record of what happened in it — exact ground truth, every event, including
all the stretches where nothing happened.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def main() -> int:
    from courtvision.nba_feed import fetch_game_plays

    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="outputs/live/commentary.json")
    parser.add_argument("--game-id", default="0042400301")
    parser.add_argument("--max-ratio", type=float, default=2.0)
    args = parser.parse_args()

    plays = fetch_game_plays(args.game_id)
    truth = collections.Counter(p.action for p in plays)
    events = json.loads(Path(args.events).read_text())["events"]
    emitted = collections.Counter(e["action"] for e in events)

    print(f"  game {args.game_id}: {len(plays)} official plays, "
          f"{len(events)} events emitted\n")
    print(f"  {'action':<9}{'emitted':>9}{'official':>10}{'ratio':>9}")
    worst, worst_action = 1.0, None
    for action in sorted(set(truth) | set(emitted)):
        want = truth.get(action, 0)
        got = emitted.get(action, 0)
        if not want:
            print(f"  {action:<9}{got:>9}{'—':>10}{'—':>9}")
            continue
        ratio = got / want
        off = ratio > args.max_ratio or ratio < 1 / args.max_ratio
        print(f"  {action:<9}{got:>9}{want:>10}{ratio:>8.2f}x{'  <-- off' if off else ''}")
        if off:
            severity = max(ratio, 1 / max(ratio, 1e-9))
            if severity > worst:
                worst, worst_action = severity, action

    total_ratio = len(events) / max(len(plays), 1)
    print(f"\n  overall event volume: {total_ratio:.2f}x official")
    if worst_action:
        print(f"  WORST {worst_action} at {worst:.1f}x — tolerance {args.max_ratio:.1f}x")
        print("  FAIL")
        return 1
    print(f"  every action within {args.max_ratio:.1f}x — PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
