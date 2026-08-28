"""Add a `rebound` class to the action dataset from BARD, in SpaceJam's format.

SpaceJam has no rebound class; BARD has 5,062 rebound annotations. They cannot
simply be mixed: SpaceJam clips are 16-frame crops around ONE player, while BARD
clips are full 720p broadcast frames. A classifier fed both would learn "which
camera framing is this" — a dataset-bias shortcut that scores well and means
nothing.

So BARD rebound clips are converted into SpaceJam's format here: sample 16
frames, run the detector, crop to the ball-handler (or the player nearest the
ball when no handler fires), and write a 16-frame clip at 10 fps. That is
exactly what `classify_windows` produces at inference, so train and inference
domains match.

Label quality note: BARD clips are multi-label and a rebound almost always
follows a missed shot, so only 101 of 14,676 clips are UNAMBIGUOUSLY rebound.
Those are used first, then clips whose headline annotation is a rebound.
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

from courtvision.action_classifier import FRAME_SIZE, crop_player
from courtvision.config import Config
from courtvision.detection import load_pipeline_detector
from courtvision.device import resolve_device
from courtvision.types import BALL, HANDLER, PLAYER

REPO = "GabrieleGiudici/BARD"
META = Path("data/labeled/bard_meta")
OUT = Path("data/labeled/actions/rebound")
N_FRAMES = 16
OUT_FPS = 10


def rebound_clips() -> tuple[list[str], list[str]]:
    """Return (unambiguous, headline-only) rebound clip paths."""
    path = hf_hub_download(REPO, "dataset_paths.csv", repo_type="dataset",
                           local_dir=str(META))
    pure, headline = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            try:
                anns = ast.literal_eval(row["actions"])
            except (ValueError, SyntaxError):
                continue
            if not anns:
                continue
            actions = {a.get("action") for a in anns}
            if actions == {"Rebound"}:
                pure.append(row["urls"])
            elif anns[0].get("action") == "Rebound":
                headline.append(row["urls"])
    return pure, headline


def sample_frames(path: str) -> list[np.ndarray]:
    capture = cv2.VideoCapture(path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return []
    wanted = set(np.linspace(0, total - 1, N_FRAMES).round().astype(int).tolist())
    frames, index = [], 0
    while True:
        if not capture.grab():
            break
        if index in wanted:
            ok, image = capture.retrieve()
            if ok:
                frames.append(image)
        index += 1
    capture.release()
    while frames and len(frames) < N_FRAMES:
        frames.append(frames[-1])
    return frames[:N_FRAMES]


def main() -> int:
    parser = argparse.ArgumentParser(description="Add BARD rebound clips")
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()

    config = Config()
    detector = load_pipeline_detector(
        "checkpoints/detector.pt", resolve_device(),
        config.detector_conf, config.ball_conf,
    )

    pure, headline = rebound_clips()
    print(f"BARD rebound clips: {len(pure)} unambiguous, {len(headline)} headline-only")
    chosen = (pure + headline)[: args.count]

    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    for video in chosen:
        local = hf_hub_download(REPO, video, repo_type="dataset",
                                local_dir=str(META / "clips"))
        frames = sample_frames(local)
        if len(frames) < N_FRAMES:
            continue

        crops, last_box = [], None
        for image in frames:
            detections = detector.detect(image)
            handlers = [d for d in detections if d.label == HANDLER]
            box = None
            if handlers:
                box = max(handlers, key=lambda d: d.conf).box
            else:
                balls = [d for d in detections if d.label == BALL]
                players = [d for d in detections if d.label == PLAYER]
                if balls and players:
                    ball = max(balls, key=lambda d: d.conf)
                    bx, by = ball.box.center
                    box = min(
                        players,
                        key=lambda p: (p.box.center[0] - bx) ** 2
                        + (p.box.center[1] - by) ** 2,
                    ).box
            box = box or last_box
            last_box = box or last_box
            if box is None:
                break
            crops.append(crop_player(image, box))

        if len(crops) < N_FRAMES:
            continue

        name = video.replace("/", "__")
        writer = cv2.VideoWriter(str(OUT / name), cv2.VideoWriter_fourcc(*"mp4v"),
                                 OUT_FPS, (FRAME_SIZE, FRAME_SIZE))
        for crop in crops:
            writer.write(crop)
        writer.release()
        written += 1
        if written % 25 == 0:
            print(f"  {written}/{len(chosen)} written")

    print(f"\n{written} rebound clips written to {OUT}, cropped to the ball-handler "
          "so they match SpaceJam's format")
    return 0


if __name__ == "__main__":
    sys.exit(main())
