"""Wire the scoreboard sensors into one event stream and score it.

This is the observed-sensor path end to end: per-frame clock, score and
shot-clock readings in, a typed event timeline out, scored against the official
play-by-play. No ball detection, no tracking, no homography anywhere in it.

The point of keeping this separate from `run_pipeline.py` is that the two
answer different questions. The vision pipeline asks what can be inferred from
pixels; this asks what the broadcast simply states. Measured on an uncut
broadcast, the stated answer is better for every class it covers -- typed
scoring reaches 0.86-0.98 where deriving field goals from a true 25 Hz ball
trajectory tops out at 0.859.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from courtvision.scoreboard_events import (fouls, missed_shots,  # noqa: E402
                                           possession_changes, score_events,
                                           stoppages)

# Matching tolerance against the official record. The play-by-play clock is
# recorded when the play is LOGGED, which trails the action by a beat -- the
# same lag that put the "release" of every shot under the basket earlier in
# this project -- so a tolerance below a few seconds measures the scorer's
# reaction time rather than the sensor.
TOLERANCE_S = 5.0


def match(predicted, truth, tolerance: float = TOLERANCE_S):
    """Precision, recall and F1 by nearest unused match.

    Count ratios are not used anywhere here. A class can hit 1.0x of the
    official count with every event at the wrong moment; that happened in this
    project at 1.54x with 90% of events misplaced.
    """
    used: set[int] = set()
    hits = 0
    for t in truth:
        candidates = [(abs(t - p), i) for i, p in enumerate(predicted)
                      if abs(t - p) <= tolerance and i not in used]
        if candidates:
            used.add(min(candidates)[1])
            hits += 1
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(truth) if truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return precision, recall, f1


def build(clock_rows, score_rows, shot_rows):
    """The event timeline from three streams of per-frame readings."""
    # (elapsed, home, away) for the score reader
    scores = score_events([(r["elapsed"], r.get("home"), r.get("away"))
                           for r in score_rows])
    makes = [e.elapsed_s for e in scores]
    free_throws = [e.elapsed_s for e in scores if e.points == 1]

    stops = stoppages([(r["video_s"], r.get("elapsed")) for r in clock_rows])
    changes = possession_changes([(r["elapsed"], r.get("shot_clock"))
                                  for r in shot_rows])
    misses = missed_shots(changes, makes)

    events = []
    for e in scores:
        events.append({"elapsed_s": e.elapsed_s, "action": e.action,
                       "points": e.points, "team": e.team})
    for t in misses:
        events.append({"elapsed_s": t, "action": "missed_shot"})
    for t in fouls(stops, free_throws):
        events.append({"elapsed_s": t, "action": "foul"})
    for t, _ in stops:
        events.append({"elapsed_s": t, "action": "dead_ball"})
    for t in changes:
        events.append({"elapsed_s": t, "action": "possession_change"})
    events.sort(key=lambda e: e["elapsed_s"])
    return events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readings", required=True,
                        help="JSON with clock/score/shot_clock reading streams")
    parser.add_argument("--truth", help="official play-by-play JSON, to score")
    parser.add_argument("--out", default="outputs/broadcast/timeline.json")
    args = parser.parse_args()

    payload = json.loads(Path(args.readings).read_text())
    events = build(payload["clock"], payload["score"], payload["shot_clock"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"events": events}, indent=1))
    print(f"  {len(events)} events -> {out}")

    if not args.truth:
        return 0
    truth = json.loads(Path(args.truth).read_text())
    by_action = collections.defaultdict(list)
    for e in events:
        by_action[e["action"]].append(e["elapsed_s"])

    print(f"\n  {'class':<20}{'pred':>6}{'offic':>7}{'P':>8}{'R':>8}{'F1':>8}{'85%':>6}")
    met = 0
    total = 0
    for name, key in (("three_point_make", "three_point_make"),
                      ("two_point_make", "two_point_make"),
                      ("free_throw", "free_throw"),
                      ("missed_shot", "missed_shot"),
                      ("foul", "foul")):
        actual = truth.get(key, [])
        if not actual:
            continue
        precision, recall, f1 = match(sorted(by_action.get(name, [])), actual)
        total += 1
        met += f1 >= 0.85
        print(f"  {name:<20}{len(by_action.get(name, [])):>6}{len(actual):>7}"
              f"{precision:>8.3f}{recall:>8.3f}{f1:>8.3f}"
              f"{'YES' if f1 >= 0.85 else 'no':>6}")
    scoring = sorted(t for name in ("three_point_make", "two_point_make",
                                    "free_throw") for t in by_action.get(name, []))
    any_make = sorted(t for key in ("three_point_make", "two_point_make",
                                    "free_throw") for t in truth.get(key, []))
    if any_make:
        precision, recall, f1 = match(scoring, any_make)
        print(f"  {'any make':<20}{len(scoring):>6}{len(any_make):>7}"
              f"{precision:>8.3f}{recall:>8.3f}{f1:>8.3f}"
              f"{'YES' if f1 >= 0.85 else 'no':>6}")
    print(f"\n  {met}/{total} classes at or above 85%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
