"""Pick wide-angle court clips from BARD for the pipeline's sample and holdout.

BARD clips are broadcast segments, so a random one may be a close-up, a replay or
a free-throw framing — none of which exercise multi-player tracking, team
assignment or possession. Rather than pick blind, this scores candidates by how
many people stock YOLO finds per frame and keeps the widest shots.

Median person count is a cheap, honest proxy for "wide court shot": a close-up
yields 1-3, a wide shot yields 6+.
"""

from __future__ import annotations

import argparse
import random
import shutil
import statistics
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from courtvision.config import Config
from courtvision.detection import COCO_CLASS_MAP, YoloDetector
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.types import PLAYER
from scripts.fetch_bard_subset import META_DIR, REPO, load_index

CLIPS_DIR = Path("data/raw_clips")
FRAMES_SCORED = 6


def score_clip(detector: YoloDetector, path: str, target_fps: int) -> float:
    """Median count of detected people across a few evenly-spread frames."""
    counts = []
    for index, _, image in extract_frames(path, target_fps):
        if index >= FRAMES_SCORED:
            break
        counts.append(sum(1 for d in detector.detect(image) if d.label == PLAYER))
    return statistics.median(counts) if counts else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Select wide-angle clips")
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = Config()
    detector = YoloDetector(
        "yolo11n.pt", resolve_device(), config.detector_conf, COCO_CLASS_MAP
    )

    pairs = load_index()
    rng = random.Random(args.seed)
    chosen = rng.sample([v for v, _ in pairs], args.candidates)

    scored: list[tuple[float, str, str]] = []
    for video in chosen:
        local = hf_hub_download(
            REPO, video, repo_type="dataset", local_dir=str(META_DIR / "clips")
        )
        score = score_clip(detector, local, config.target_fps)
        scored.append((score, video, local))
        print(f"  {score:>5.1f} people/frame  {video}")

    scored.sort(reverse=True, key=lambda t: t[0])
    if not scored or scored[0][0] < 4:
        print(f"\nFAIL — best clip only had {scored[0][0] if scored else 0} people per "
              "frame; no wide-angle shot found. Re-run with more --candidates.")
        return 1

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    best = scored[0]
    shutil.copy(best[2], CLIPS_DIR / "sample.mp4")
    print(f"\nsample.mp4  <- {best[1]}  ({best[0]:.1f} people/frame)")

    # Holdout must come from a different game so it is genuinely unseen footage.
    best_game = best[1].split("/")[0]
    holdout = next((s for s in scored[1:] if s[1].split("/")[0] != best_game), None)
    if holdout is None:
        print("WARNING — no candidate from a different game; holdout not written")
        return 1
    shutil.copy(holdout[2], CLIPS_DIR / "holdout.mp4")
    print(f"holdout.mp4 <- {holdout[1]}  ({holdout[0]:.1f} people/frame)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
