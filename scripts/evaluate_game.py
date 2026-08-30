"""Score a full-game run against BARD's own labels for that game.

Clip-level accuracy said 0.816 while the same model called 67% of a game a
rebound. Both were true: held-out clips are a balanced mix of seven actions
because each was CUT to contain one, and a game is mostly ordinary play. Nothing
in the clip metric can see that, so this scores against the game.

The concatenated game video is exactly the clips BARD annotates, so the number
of each action in it is known. This compares what the pipeline emitted against
what is actually there — a count ratio per action, and a single worst-ratio
summary to tune against.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import sys
from pathlib import Path

# BARD action names -> our classes. Free Throw, Foul and Turnover have no class
# of their own; they are part of what "other" has to absorb.
TRUTH_MAP = {
    "2PT Shot": "shot",
    "3PT Shot": "shot",
    "Rebound": "rebound",
    "Steal": "steal",
    "Block": "block",
    "Free Throw": "other",
    "Foul": "other",
    "Turnover": "other",
}


def truth_counts(metadata_csv: str, game: str) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    with open(metadata_csv) as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if game not in row["urls"]:
                continue
            try:
                annotations = ast.literal_eval(row["actions"])
            except (ValueError, SyntaxError):
                continue
            for annotation in annotations:
                mapped = TRUTH_MAP.get(annotation.get("action"))
                if mapped:
                    counts[mapped] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="outputs/quarter/commentary.json")
    parser.add_argument("--game", default="chi-vs-tor-0022401223")
    parser.add_argument("--metadata", default="data/labeled/bard_meta/dataset_paths.csv")
    parser.add_argument("--max-ratio", type=float, default=2.0,
                        help="worst tolerated over- or under-emission")
    args = parser.parse_args()

    truth = truth_counts(args.metadata, args.game)
    if not truth:
        print(f"FAIL — no ground truth for {args.game}")
        return 1
    events = json.loads(Path(args.events).read_text())["events"]
    emitted = collections.Counter(e["action"] for e in events)

    print(f"  {args.game}: {sum(truth.values())} labelled actions, "
          f"{len(events)} events emitted\n")
    print(f"  {'action':<9}{'emitted':>9}{'truth':>8}{'ratio':>9}")
    worst, worst_action = 1.0, None
    # dribble and pass are continuous play, not discrete box-score events, so
    # BARD does not label them and they cannot be scored this way.
    for action in ("shot", "rebound", "steal", "block", "other"):
        got, want = emitted.get(action, 0), truth.get(action, 0)
        if not want:
            continue
        ratio = got / want
        flag = ""
        if ratio > args.max_ratio or ratio < 1 / args.max_ratio:
            flag = "  <-- off"
            if max(ratio, 1 / max(ratio, 1e-9)) > worst:
                worst = max(ratio, 1 / max(ratio, 1e-9))
                worst_action = action
        print(f"  {action:<9}{got:>9}{want:>8}{ratio:>8.2f}x{flag}")
    unscored = {a: emitted.get(a, 0) for a in ("dribble", "pass")}
    print(f"\n  not scorable (BARD labels no dribble/pass): {unscored}")

    if worst_action:
        print(f"\n  WORST {worst_action} at {worst:.1f}x — tolerance is "
              f"{args.max_ratio:.1f}x")
        print("  FAIL")
        return 1
    print(f"\n  every scorable action within {args.max_ratio:.1f}x — PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
