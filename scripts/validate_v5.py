"""V5 — Team assignment sanity check (spec §6).

Clusters jersey colours over a real clip and writes a contact sheet of torso crops
grouped by assigned team, so the two clusters can be eyeballed against the real
jersey colours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from courtvision.config import Config
from courtvision.detection import load_finetuned
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.team_assignment import assign_teams, collect_samples, torso_crop
from courtvision.tracking import PlayerTracker
from courtvision.types import Frame

CLIP = Path("data/raw_clips/sample.mp4")
CHECKPOINT = Path("checkpoints/detector.pt")
OUT_DIR = Path("outputs/v5_teams")
CROP_SIZE = (64, 64)


def main() -> int:
    if not CLIP.exists() or not CHECKPOINT.exists():
        print(f"V5 FAIL — need clip at {CLIP} and checkpoint at {CHECKPOINT}")
        return 1

    config = Config()
    detector = load_finetuned(str(CHECKPOINT), resolve_device(), config.detector_conf)
    tracker = PlayerTracker()

    images, frames = [], []
    for index, time_s, image in extract_frames(str(CLIP), config.target_fps):
        tracks = tracker.update(detector.detect(image))
        images.append(image)
        frames.append(Frame(index, time_s, tuple(tracks)))

    teams = assign_teams(collect_samples(images, frames))
    if not teams:
        print("V5 FAIL — no player tracks to cluster")
        return 1

    # Contact sheet: one row per team, one column per sampled crop.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for team in ("A", "B"):
        crops = []
        for image, frame in zip(images, frames):
            for track in frame.players():
                if teams.get(track.track_id) != team:
                    continue
                crop = torso_crop(image, track.box)
                if crop.size:
                    crops.append(cv2.resize(crop, CROP_SIZE))
            if len(crops) >= 24:
                break
        if crops:
            cv2.imwrite(str(OUT_DIR / f"team_{team}.png"), np.hstack(crops))

    counts = {team: sum(1 for t in teams.values() if t == team) for team in ("A", "B")}
    # Both teams must actually be populated — a 1-vs-rest split means clustering failed.
    ok = counts["A"] >= 2 and counts["B"] >= 2
    verdict = "PASS" if ok else "FAIL"
    print(
        f"V5 {verdict} — {len(teams)} tracks split A={counts['A']} B={counts['B']}; "
        f"compare {OUT_DIR}/team_A.png and team_B.png against the real jerseys"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
