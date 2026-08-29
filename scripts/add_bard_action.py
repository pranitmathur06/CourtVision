"""Add BARD-sourced action classes (rebound, steal) in SpaceJam's format.

SpaceJam has no rebound or steal class; BARD has both. They cannot simply be
mixed: SpaceJam clips are 16-frame crops around ONE player, while BARD clips are
full 720p broadcast frames. A classifier fed both raw would learn "which camera
framing is this" — a dataset-bias shortcut that scores well and means nothing.

So BARD clips are converted into SpaceJam's format: sample 16 frames, run the
detector, crop to the ball-handler (or the player nearest the ball when no
handler fires) at SpaceJam's aspect ratio, and write a 16-frame clip at 10 fps.
That is exactly what `classify_windows` produces at inference, so train and
inference domains match.

Selecting clean clips per action:

* **rebound** — a rebound follows a missed shot, so it co-occurs with shots
  constantly; only 101 of 14,676 clips are unambiguously rebound. Those are used
  first, then clips whose headline annotation is a rebound.
* **steal** — 435 clips are exactly {Steal, Turnover}. That pair is NOT two
  confounded actions: it is one steal event annotated from both sides, the
  stealer and the player dispossessed, and in 891 of 929 cases they are
  different players. An earlier version of this project wrongly rejected these
  as "ambiguous" and concluded steal was unavailable. It is not.
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
ACTIONS_DIR = Path("data/labeled/actions")
N_FRAMES = 16
OUT_FPS = 10


# For each output class: the action sets that count as a clean example, then a
# looser headline fallback.
SELECTORS: dict[str, tuple[list[set[str]], str]] = {
    "rebound": ([{"Rebound"}], "Rebound"),
    # {Steal, Turnover} is one steal seen from both sides, not two actions.
    "steal": ([{"Steal", "Turnover"}, {"Steal"}], "Steal"),
}


def select_clips(action: str) -> tuple[list[str], list[str]]:
    """Return (clean, headline-only) clip paths for one output action."""
    exact_sets, headline_action = SELECTORS[action]
    path = hf_hub_download(REPO, "dataset_paths.csv", repo_type="dataset",
                           local_dir=str(META))
    clean, headline = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            try:
                anns = ast.literal_eval(row["actions"])
            except (ValueError, SyntaxError):
                continue
            if not anns:
                continue
            actions = {a.get("action") for a in anns}
            if any(actions == wanted for wanted in exact_sets):
                clean.append(row["urls"])
            elif anns[0].get("action") == headline_action:
                headline.append(row["urls"])
    return clean, headline


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
    parser = argparse.ArgumentParser(description="Add a BARD-sourced action class")
    parser.add_argument("--action", choices=sorted(SELECTORS), required=True)
    parser.add_argument("--count", type=int, default=400)
    args = parser.parse_args()
    out = ACTIONS_DIR / args.action

    config = Config()
    detector = load_pipeline_detector(
        "checkpoints/detector.pt", resolve_device(),
        config.detector_conf, config.ball_conf,
    )

    clean, headline = select_clips(args.action)
    print(f"BARD {args.action} clips: {len(clean)} clean, {len(headline)} headline-only")
    chosen = (clean + headline)[: args.count]

    out.mkdir(parents=True, exist_ok=True)
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
        writer = cv2.VideoWriter(str(out / name), cv2.VideoWriter_fourcc(*"mp4v"),
                                 OUT_FPS, (FRAME_SIZE, FRAME_SIZE))
        for crop in crops:
            writer.write(crop)
        writer.release()
        written += 1
        if written % 25 == 0:
            print(f"  {written}/{len(chosen)} written")

    print(f"\n{written} {args.action} clips written to {out}, cropped to the "
          "ball-handler so they match SpaceJam's format")
    return 0


if __name__ == "__main__":
    sys.exit(main())
