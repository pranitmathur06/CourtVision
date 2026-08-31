"""Full end-to-end pipeline: clip in, annotated video + commentary log out.

Per spec §9.3 this exists only after V1-V8 pass individually. Each stage remains
its own module; this script only sequences them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from courtvision.action_classifier import VideoMaeClassifier, classify_windows
from courtvision.commentary import AnthropicNarrator, generate_commentary
from courtvision.config import Config
from courtvision.detection import load_pipeline_detector
from courtvision.device import resolve_device
from courtvision.enrichment import align, enforce_identity_consistency, plays_for_clip
from courtvision.events import build_events
from courtvision.extraction import extract_frames
from courtvision.framestore import FrameStore
from courtvision.possession import possession_timeline
from courtvision.render import render_video, write_log
from courtvision.team_assignment import assign_teams, collect_samples
from courtvision.tracking import PlayerTracker
from courtvision.types import Frame

DETECTOR = Path("checkpoints/detector.pt")
ACTION_MODEL = Path("checkpoints/action_classifier")
# Official play-by-play, used to name real players when the clip's game is known.
BARD_METADATA = Path("data/labeled/bard_meta/dataset.csv")


def run_pipeline(clip_path: str, out_dir: str, config: Config,
                 narrate: bool = True, prior_strength: float = 0.0) -> dict:
    device = resolve_device()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    # Stages 1-3: extract, detect, track.
    start = time.perf_counter()
    detector = load_pipeline_detector(
        str(DETECTOR), device, config.detector_conf, config.ball_conf
    )
    tracker = PlayerTracker()
    # Frames go to disk, not into a list. Holding them cost 2.8 MB each, so a
    # 19-minute clip would have needed 31.9 GB and the pipeline was quietly
    # limited to the ten-second clips every test used. FrameStore is a
    # Sequence, so the stages below are unchanged.
    images, frames = FrameStore(), []
    for index, time_s, image in extract_frames(clip_path, config.target_fps):
        images.append(image)
        frames.append(
            Frame(index, time_s, tuple(tracker.update(detector.detect(image))))
        )
    timings["1-3 extract/detect/track"] = time.perf_counter() - start
    print(f"  stages 1-3: {len(frames)} frames tracked")

    # Stage 4: team assignment.
    start = time.perf_counter()
    teams = assign_teams(collect_samples(images, frames))
    timings["4 team assignment"] = time.perf_counter() - start
    print(f"  stage 4: {len(teams)} tracks assigned to teams")

    # Stage 5: possession.
    start = time.perf_counter()
    holders = possession_timeline(frames, config)
    timings["5 possession"] = time.perf_counter() - start
    held = sum(1 for h in holders if h is not None)
    print(f"  stage 5: possession resolved on {held}/{len(holders)} frames")

    # Stage 6: action classification, cropped to the ball-handler and batched.
    start = time.perf_counter()
    # Training is class-balanced because every clip was cut to contain an
    # action; a game is ~90% ordinary play. Left uncorrected the classifier
    # emitted 7.6x the official event volume. The shift is per class and
    # additive in log space, so overwhelming evidence is untouched.
    train_counts = {d.name: len(list(d.glob("*.mp4")))
                    for d in Path("data/labeled/actions").iterdir() if d.is_dir()}
    classifier = VideoMaeClassifier(str(ACTION_MODEL), device,
                                    prior_strength=prior_strength,
                                    train_counts=train_counts or None)
    windows = classify_windows(images, frames, classifier, config, holders)
    timings["6 action classification"] = time.perf_counter() - start
    print(f"  stage 6: {len(windows)} action windows classified")

    # Stage 7: event structuring.
    start = time.perf_counter()
    events = build_events(windows, holders, teams)
    timings["7 events"] = time.perf_counter() - start
    print(f"  stage 7: {len(events)} events")

    # Stage 7b (v3): name real players from official play-by-play, when the
    # clip's game is identifiable. align() refuses to guess, so events with no
    # confident match keep their anonymous track id.
    plays = plays_for_clip(clip_path, str(BARD_METADATA))
    if plays:
        events = align(events, plays)
        # One track is one person, and one person is on one team.
        events = enforce_identity_consistency(events)
        named = sum(1 for e in events if e.player_name)
        print(f"  stage 7b: {len(plays)} official plays for this game, "
              f"{named}/{len(events)} events named")
    else:
        print("  stage 7b: no play-by-play for this clip; narrating anonymously")

    # Stage 8: commentary.
    lines, errors = [], []
    if narrate and events:
        start = time.perf_counter()
        lines, errors = generate_commentary(events, AnthropicNarrator(config), config)
        timings["8 commentary"] = time.perf_counter() - start
        print(f"  stage 8: {len(lines)} lines, {len(errors)} fabrication errors")
    else:
        print("  stage 8: skipped")

    # Stage 9: render.
    start = time.perf_counter()
    video_path = out / "annotated.mp4"
    log_path = out / "commentary.json"
    render_video(images, frames, teams, holders, str(video_path), config.target_fps)
    write_log(events, lines, str(log_path))
    timings["9 render"] = time.perf_counter() - start
    print(f"  stage 9: wrote {video_path} and {log_path}")

    images.close()
    total = sum(timings.values())
    print(f"\n  {'stage':<28}{'seconds':>9}{'%':>7}")
    for name, seconds in timings.items():
        print(f"  {name:<28}{seconds:>9.2f}{100*seconds/total:>6.1f}%")
    print(f"  {'TOTAL':<28}{total:>9.2f}{100.0:>6.1f}%")

    return {
        "n_frames": len(frames),
        "n_tracks": len(teams),
        "n_windows": len(windows),
        "n_events": len(events),
        "n_lines": len(lines),
        "errors": errors,
        "video_path": str(video_path),
        "log_path": str(log_path),
        "seconds": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CourtVision pipeline")
    parser.add_argument("clip", help="path to an input .mp4")
    parser.add_argument("--out", default="outputs/run", help="output directory")
    parser.add_argument("--prior-strength", type=float, default=0.0,
                        help="correct the train/serve prior mismatch; 0 disables")
    parser.add_argument("--no-narrate", action="store_true",
                        help="skip stage 8 (no API key needed)")
    args = parser.parse_args()

    if not Path(args.clip).exists():
        print(f"no such clip: {args.clip}")
        return 1
    for path in (DETECTOR, ACTION_MODEL):
        if not path.exists():
            print(f"missing model: {path}; run validate_v3.py and validate_v7.py first")
            return 1

    summary = run_pipeline(args.clip, args.out, Config(),
                           narrate=not args.no_narrate,
                           prior_strength=args.prior_strength)
    print(f"\n{summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
