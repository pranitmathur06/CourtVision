"""The first accuracy metric this project's tracking has ever had.

EVERY TRACKING NUMBER HERE IS A COUNT. "463 identities for ten players over five
minutes" says something is wrong, and it cannot say how wrong, cannot compare two
trackers, and is trivially gamed -- a tracker that merges all ten players into one
identity scores best on it. `docs/` says so plainly and the plan's answer was
~980 hand judgements that nobody has made.

THE TEN-PLAYER CONSTRAINT MAKES THREE METRICS FREE. From tip-off to the buzzer
there are ten players on the court and at most three referees, so:

    over-tracking   more than 13 identities alive at once is wrong for certain,
                    whatever the frame shows. Every one above 13 is a ghost the
                    tracker is still carrying or a player it has split in two.

    doubles         two live identities whose boxes overlap by more than the
                    duplicate threshold are one player wearing two numbers.
                    `motion_tracking.deduplicate` exists to draw only one of
                    them; this counts how often it has to.

    churn           identities per clip and median track life. NOT an accuracy
                    -- it is the count this replaces -- reported beside the
                    other two so the difference between them is visible.

NO VIDEO AND NO LABELS. Everything runs from the cached clip detections, and the
camera cuts are found in the cache too: a cut is a frame where almost nothing
matches the frame before it, which is the same signal the tracker already
computes to estimate camera motion.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.games import get, registry  # noqa: E402
from courtvision.motion_tracking import MotionTracker, TrackerConfig, iou  # noqa: E402
from courtvision.stats import wilson  # noqa: E402

ON_COURT, REFEREES = 10, 3
IMPOSSIBLE_ABOVE = ON_COURT + REFEREES
#: A frame matching this little of the previous one is a camera cut. Measured
#: rather than chosen: on these caches the match rate between consecutive frames
#: of one shot sits above 0.6 and across a cut below 0.15.
CUT_MATCH_SHARE = 0.25


def cuts_from_boxes(per_frame, threshold: float = CUT_MATCH_SHARE) -> list[int]:
    """Frames where almost nothing matches the frame before. No pixels needed."""
    out = []
    for i in range(1, len(per_frame)):
        before, now = per_frame[i - 1], per_frame[i]
        if not before or not now:
            continue
        matched = sum(1 for b in now if any(iou(b, a) >= 0.2 for a in before))
        if matched / max(len(now), 1) < threshold:
            out.append(i)
    return out


def measure(per_frame, fps: float, use_cuts: bool) -> dict:
    """Run the tracker over one clip and count what the constraint forbids."""
    tracker = MotionTracker(fps=fps, config=TrackerConfig())
    breaks = set(cuts_from_boxes(per_frame)) if use_cuts else set()
    alive, doubles, lives = [], [], {}
    identities = set()
    for index, boxes in enumerate(per_frame):
        if index in breaks:
            tracker.end_segment()
        drawn = tracker.update(boxes, index)
        here = [(tid, box) for tid, box in drawn]
        alive.append(len(here))
        identities.update(t for t, _ in here)
        for t, _ in here:
            lives.setdefault(t, [index, index])[1] = index
        pairs = 0
        for a in range(len(here)):
            for b in range(a + 1, len(here)):
                if iou(here[a][1], here[b][1]) > TrackerConfig().duplicate_iou:
                    pairs += 1
        doubles.append(pairs)
    spans = [(b - a + 1) / fps for a, b in lives.values()]
    return {"frames": len(per_frame), "alive": alive, "doubles": doubles,
            "identities": len(identities),
            "median_life_s": statistics.median(spans) if spans else 0.0,
            "cuts": len(breaks)}


def report(key: str, rows: list[dict], label: str) -> dict:
    frames = sum(r["frames"] for r in rows)
    over = sum(1 for r in rows for a in r["alive"] if a > IMPOSSIBLE_ABOVE)
    doubled = sum(1 for r in rows for d in r["doubles"] if d)
    alive = [a for r in rows for a in r["alive"]]
    low, high = wilson(over, frames)
    print(f"  {key:5} {label:<14}{frames:>7} frames"
          f"   alive p50 {statistics.median(alive):>4.0f}"
          f"   OVER 13 {over / frames:>6.3f} ({low:.3f}-{high:.3f})"
          f"   doubles {doubled / frames:>6.3f}")
    print(f"        {sum(r['identities'] for r in rows):>6} identities over "
          f"{len(rows)} clips, median life "
          f"{statistics.median([r['median_life_s'] for r in rows]):.2f}s, "
          f"{sum(r['cuts'] for r in rows)} cuts found")
    return {"frames": frames, "over_13": over, "over_13_rate": over / frames,
            "doubles_rate": doubled / frames,
            "identities": sum(r["identities"] for r in rows),
            "median_life_s": statistics.median(
                [r["median_life_s"] for r in rows])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--clips", type=int, default=60,
                        help="how many clips to run; the whole cache is slow")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"\n  eval_tracking.py -- ten players plus at most {REFEREES} referees,"
          f" so more than {IMPOSSIBLE_ABOVE} alive is wrong for certain\n")
    out = {}
    for key in (args.game or list(registry())):
        game = get(key)
        if not game.clip_detections.exists():
            continue
        cache = json.loads(game.clip_detections.read_text())
        fps = float(cache.get("rate") or 15.0)
        names = sorted(cache["clips"])[:args.clips]
        # PLAYERS ONLY. The detector has a separate `handler` class, so the
        # player with the ball carries BOTH a `p` box and an `h` box on almost
        # every frame. Feeding both to the tracker made it look as though it
        # held two identities on one player on 77% of frames; that was this
        # script double-counting one man, not the tracker duplicating him.
        # `clip_boxes.assemble` tracks `d["p"]` alone for the same reason.
        clips = [[[b[2:6] for b in frame["d"]
                   if b[0] == "p" and b[1] >= 0.35]
                  for frame in cache["clips"][name]] for name in names]
        out[key] = {}
        for label, use_cuts in (("as it ships", False), ("cut-aware", True)):
            rows = [measure(c, fps, use_cuts) for c in clips if c]
            if rows:
                out[key][label] = report(key, rows, label)
        print()
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
