"""Run the pipeline over a game's clips independently, then aggregate.

Joining clips end to end was the wrong way to assemble a game. Every join is a
discontinuity the pipeline has to survive: tracking restarts, possession
fragments, and a 16-frame window landing on a join holds two unrelated scenes.
Shot detection fell to 9 of 97 that way, while the same checkpoint finds 246 of
262 on genuinely continuous broadcast.

Each clip IS continuous broadcast footage. Processing them separately and
summing the events keeps that property and still covers the whole game, which
is what the ground truth is stated over. Models load once and are reused.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def main() -> int:
    import cv2

    from courtvision.action_classifier import VideoMaeClassifier, classify_windows
    from courtvision.config import Config
    from courtvision.derived_events import derive
    from courtvision.detection import load_pipeline_detector
    from courtvision.device import resolve_device
    from courtvision.events import build_events
    from courtvision.extraction import extract_frames
    from courtvision.possession import possession_timeline
    from courtvision.shot_boundaries import cut_frames, segments
    from courtvision.team_assignment import assign_teams, collect_samples
    from courtvision.tracking import PlayerTracker
    from courtvision.types import Frame

    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", required=True, help="directory of .mp4 clips")
    parser.add_argument("--out", default="outputs/game_clips")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--derive-possession", action="store_true")
    args = parser.parse_args()

    config = Config()
    device = resolve_device()
    detector = load_pipeline_detector("checkpoints/detector.pt", device,
                                      config.detector_conf, config.ball_conf)
    classifier = VideoMaeClassifier("checkpoints/action_classifier", device)

    paths = sorted(Path(args.clips).glob("*.mp4"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.limit:
        paths = paths[: args.limit]
    print(f"  {len(paths)} clips", flush=True)

    all_events = []
    offset = 0.0
    tallies: collections.Counter = collections.Counter()
    for index, path in enumerate(paths):
        tracker = PlayerTracker()
        images, frames = [], []
        for i, time_s, image in extract_frames(str(path), config.target_fps):
            images.append(image)
            frames.append(Frame(i, time_s, tuple(tracker.update(detector.detect(image)))))
        if len(frames) < config.action_window_frames:
            continue
        teams = assign_teams(collect_samples(images, frames))
        holders = possession_timeline(frames, config)
        boundaries = segments(len(images), cut_frames(images))
        windows = classify_windows(images, frames, classifier, config, holders,
                                   boundaries=boundaries)
        events = build_events(windows, holders, teams)
        if args.derive_possession:
            kept = [e for e in events if e.action not in ("steal", "rebound")]
            shots = [e for e in events if e.action == "shot"]
            events = sorted(kept + derive([f.time_s for f in frames], holders,
                                          teams, shots),
                            key=lambda e: e.time_s)
        for e in events:
            tallies[e.action] += 1
            all_events.append({"time_s": round(offset + e.time_s, 2),
                               "action": e.action, "track_id": e.track_id,
                               "team": e.team,
                               "possession_change": e.possession_change})
        offset += frames[-1].time_s + 0.1
        if (index + 1) % 20 == 0:
            print(f"    {index + 1}/{len(paths)} clips, "
                  f"{len(all_events)} events, {dict(tallies.most_common(4))}",
                  flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "commentary.json").write_text(json.dumps({"events": all_events}))
    print(f"\n  {len(all_events)} events over {offset / 60:.1f} min of footage")
    print(f"  {dict(tallies.most_common())}")
    print(f"  wrote {out / 'commentary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
