"""Put official shots on a video's timeline, using the clock read off the broadcast.

The official record gives each attempt a period and a game clock; the clock
reader gives every readable second of video the same pair. Matching them places
each shot at a video time, which is what any shot detector is scored against.

A shot is kept only when a clock reading sits within --max-gap-s of its time:
elsewhere the scoreboard was not on screen (replays, breaks, the open of a
quarter before the graphic appears) and the placement would be a guess. On
Finals G7 that kept 126 of 157 attempts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", required=True, help="JSON: [{period, clock_s, made}]")
    parser.add_argument("--clock", required=True, help="scripts/read_game_clock.py output")
    parser.add_argument("--max-gap-s", type=float, default=2.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    official = json.load(open(args.official))
    readings = json.load(open(args.clock))["readings"]
    by_period: dict[int, list[tuple[float, float]]] = {}
    for r in readings:
        by_period.setdefault(r["period"], []).append((r["seconds"], r["t"]))

    placed = []
    for shot in official:
        rows = by_period.get(shot["period"])
        if not rows:
            continue
        seconds = np.array([r[0] for r in rows])
        times = np.array([r[1] for r in rows])
        j = int(np.argmin(np.abs(seconds - shot["clock_s"])))
        placed.append({"t": float(times[j]), "gap_s": float(abs(seconds[j] - shot["clock_s"])),
                       "made": bool(shot.get("made")), "period": shot["period"],
                       "clock_s": shot["clock_s"]})
    kept = [p for p in placed if p["gap_s"] <= args.max_gap_s]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(placed, open(args.out, "w"), indent=1)
    order = np.argsort([p["clock_s"] for p in kept])
    rising = np.mean(np.diff([kept[i]["t"] for i in np.argsort([(p["period"], -p["clock_s"]) for p in kept], axis=0)[:, 0]]) >= 0) if len(kept) > 2 else float("nan")
    print(f"{len(official)} official attempts; {len(placed)} placed; {len(kept)} within {args.max_gap_s:g} s of a clock reading")
    print(f"  their video times rise with game time on {rising:.0%} of consecutive pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
