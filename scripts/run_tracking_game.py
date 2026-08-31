"""Derive events from tracking coordinates and score them against the NBA's own record.

This measures the CEILING, not the vision stack: what the event logic achieves
when perception is perfect. Tracking covers 2015-10-27 to 2016-01-23 and no
release pairs it with broadcast video, so the two never meet on one game.

Everything downstream of perception is reused unchanged — `possession.py`,
`derived_events.py` and `events.py` never needed pixels.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import sys
from dataclasses import replace
from pathlib import Path

# Possession attribution is nearest-player, which is wrong while the ball is in
# flight: mid-pass the ball can be nearest an opponent and invent a turnover.
# Requiring a low ball took steal from 6.08x to 3.62x of the official count and
# lifted its precision from 0.13 to 0.19 on 0021500492.
MAX_HELD_BALL_FT = 9.0
# Court feet, not body-heights of a projected pixel box. Swept: 0.35 gave
# rebound 0.94x where 1.0 gave 0.63x.
POSSESSION_MAX_NORM_DIST = 0.35


def score(predicted: list[float], truth: list[float], tolerance: float = 3.0):
    """Precision and recall by nearest unused match within `tolerance` seconds.

    Counts alone can be hit by accident — steal reached 1.54x of official while
    90% of the emitted events were the wrong moments — so both are reported.
    """
    used: set[int] = set()
    hits = 0
    for t in predicted:
        candidates = [(abs(t - x), i) for i, x in enumerate(truth)
                      if abs(t - x) <= tolerance and i not in used]
        if candidates:
            used.add(min(candidates)[1])
            hits += 1
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(truth) if truth else 0.0
    return precision, recall


def run_one(path: Path, target_hz: float = 10.0) -> dict:
    from courtvision.config import Config
    from courtvision.derived_events import derive
    from courtvision.nba_feed import fetch_game_plays
    from courtvision.possession import possession_timeline
    from courtvision.shot_detection import shots as detect_shots
    from courtvision.tracking_data import elapsed_seconds, load_game

    game = load_game(path, target_hz=target_hz)
    times = [f.time_s for f in game.frames]

    shot_events = detect_shots(game.frames, game.ball_z)
    config = replace(Config(), possession_max_norm_dist=POSSESSION_MAX_NORM_DIST)
    raw = possession_timeline(game.frames, config)
    holders = [
        h if (h is None or (game.ball_z[i] == game.ball_z[i]
                            and game.ball_z[i] <= MAX_HELD_BALL_FT)) else None
        for i, h in enumerate(raw)
    ]

    positions: dict[int, dict[float, tuple[float, float]]] = {}
    for frame in game.frames:
        key = round(frame.time_s, 1)
        for track in frame.players():
            positions.setdefault(track.track_id, {})[key] = track.box.center

    derived = derive(times, holders, game.teams, shot_events,
                     min_seconds=1.0, positions=positions)
    events = sorted(list(shot_events) + derived, key=lambda e: e.time_s)

    plays = fetch_game_plays(game.game_id)
    truth = collections.defaultdict(list)
    for play in plays:
        if play.period is not None and play.clock_seconds is not None:
            truth[play.action].append(elapsed_seconds(play.period, play.clock_seconds))

    result = {"game_id": game.game_id, "date": game.game_date, "classes": {}}
    for action in ("shot", "rebound", "steal"):
        emitted = sorted(e.time_s for e in events if e.action == action)
        actual = sorted(truth.get(action, []))
        precision, recall = score(emitted, actual)
        result["classes"][action] = {
            "emitted": len(emitted), "official": len(actual),
            "ratio": len(emitted) / len(actual) if actual else None,
            "precision": round(precision, 3), "recall": round(recall, 3),
        }
    result["events"] = [{"time_s": round(e.time_s, 2), "action": e.action,
                         "track_id": e.track_id, "team": e.team,
                         "possession_change": e.possession_change}
                        for e in events]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="data/tracking",
                        help="directory of extracted SportVU JSON")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="outputs/tracking")
    args = parser.parse_args()

    paths = sorted(Path(args.games).glob("*.json"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"FAIL — no tracking JSON under {args.games}")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        try:
            row = run_one(path)
        except Exception as error:            # one bad game must not stop ten
            print(f"  {path.name}: {type(error).__name__} {error}", flush=True)
            continue
        rows.append(row)
        summary = "  ".join(
            f"{a} {row['classes'][a]['ratio']:.2f}x" if row['classes'][a]['ratio']
            else f"{a} -" for a in ("shot", "rebound", "steal"))
        print(f"  {row['game_id']} {row['date']}  {summary}", flush=True)
        (out / f"{row['game_id']}.json").write_text(json.dumps(row))

    if not rows:
        print("FAIL — no game produced a result")
        return 1

    print(f"\n  {len(rows)} games\n")
    print(f"  {'action':<9}{'median x':>10}{'precision':>11}{'recall':>9}{'bar':>6}")
    failed = []
    for action in ("shot", "rebound", "steal"):
        ratios = sorted(r["classes"][action]["ratio"] for r in rows
                        if r["classes"][action]["ratio"])
        precisions = [r["classes"][action]["precision"] for r in rows]
        recalls = [r["classes"][action]["recall"] for r in rows]
        if not ratios:
            continue
        median = ratios[len(ratios) // 2]
        met = 0.5 <= median <= 2.0
        if not met:
            failed.append(action)
        print(f"  {action:<9}{median:>9.2f}x"
              f"{sum(precisions)/len(precisions):>11.2f}"
              f"{sum(recalls)/len(recalls):>9.2f}"
              f"{'met' if met else 'MISSED':>6}")
    print("\n  block is not emitted — see shot_detection.py for the measurement")
    if failed:
        print(f"  CEILING MISSED for {', '.join(failed)}")
        return 1
    print("  CEILING MET on count for every emitted class")
    return 0


if __name__ == "__main__":
    sys.exit(main())
