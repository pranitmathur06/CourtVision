"""Produce a complete, timestamped event timeline for a game.

Every event type at once -- steals, blocks, turnovers, substitutions, timeouts,
assists -- by taking event IDENTITY from the official feed and event TIMING
from the clock reader. Detecting these from pixels has measured ceilings this
project established the hard way (steal 0.271 on perfect tracking data, block
at or below chance), while alignment is a solved problem here.

Prints the per-class rate so a caller can see which parts of a game are usable
rather than trusting an aggregate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from courtvision.event_alignment import align, coverage, elapsed_seconds  # noqa: E402

CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")


def clock_seconds(text: str) -> float | None:
    matched = CLOCK.fullmatch((text or "").strip())
    return (int(matched.group(1)) * 60 + float(matched.group(2))
            if matched else None)


def classify(action: dict) -> list[str]:
    """Event labels for one play-by-play action.

    A single action can carry more than one: a missed shot that was blocked is
    both a Missed Shot and a Block, and a turnover that was stolen is both. The
    official feed encodes the second one only in the description text.
    """
    kind = (action.get("actionType") or "").strip()
    note = ((action.get("description") or "") + " "
            + (action.get("subType") or "")).upper()
    labels: list[str] = []
    if kind == "Made Shot":
        labels.append("Made Shot (3PT)" if "3PT" in note else "Made Shot (2PT)")
    elif kind == "Free Throw":
        labels.append("Free Throw (miss)" if "MISS" in note
                      else "Free Throw (made)")
    elif kind:
        labels.append(kind)
    if "STEAL" in note:
        labels.append("Steal")
    if "BLOCK" in note:
        labels.append("Block")
    if "AST)" in note or "ASSIST" in note:
        labels.append("Assist")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--clock", required=True,
                        help="clock reader output: per-frame t/elapsed/period")
    parser.add_argument("--tolerance", type=float, default=3.0)
    parser.add_argument("--out", default="outputs/aligned_events.json")
    args = parser.parse_args()

    from nba_api.stats.endpoints import playbyplayv3

    readings = json.loads(Path(args.clock).read_text())
    actions = playbyplayv3.PlayByPlayV3(
        game_id=args.game_id, timeout=60).get_dict()["game"]["actions"]

    readable = [r["elapsed"] for r in readings if r.get("elapsed") is not None]
    if not readable:
        print("FAIL — the clock was never read; nothing can be aligned")
        return 1
    low, high = min(readable), max(readable)

    events: list[tuple[float, str]] = []
    notes: list[str] = []
    for action in actions:
        seconds = clock_seconds(action.get("clock") or "")
        period = action.get("period")
        if seconds is None or period is None:
            continue
        moment = elapsed_seconds(period, seconds)
        if not (low - 3 <= moment <= high + 3):
            continue
        for label in classify(action):
            events.append((moment, label))
            notes.append(action.get("description") or "")

    aligned = align(events, readings, args.tolerance, notes)
    report = coverage(events, aligned)

    print(f"  {len(events)} official events in the clock-read window\n")
    print(f"  {'action type':<22}{'total':>7}{'located':>9}{'rate':>8}")
    below = []
    for action, (found, total) in sorted(report.items(),
                                         key=lambda kv: -kv[1][1]):
        rate = found / total if total else 0.0
        if rate < 0.85:
            below.append(action)
        print(f"  {action:<22}{total:>7}{found:>9}{rate:>7.1%}"
              f"{'  <-- below 85%' if rate < 0.85 else ''}")
    overall = len(aligned) / len(events) if events else 0.0
    print(f"\n  OVERALL {len(aligned)}/{len(events)} = {overall:.1%}")
    if below:
        print(f"  below 85%: {', '.join(below)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "game_id": args.game_id,
        "overall_rate": overall,
        "per_action": {k: {"located": v[0], "total": v[1]}
                       for k, v in report.items()},
        "events": [{"elapsed_s": round(e.elapsed_s, 1),
                    "video_s": round(e.video_s, 2),
                    "action": e.action,
                    "error_s": round(e.error_s, 2),
                    "description": e.description} for e in aligned],
    }, indent=1))
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
