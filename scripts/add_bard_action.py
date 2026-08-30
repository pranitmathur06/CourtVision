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
# Fraction of the source clip the frames are drawn from, centred. See
# sample_frames for the measurement behind it.
WINDOW = 0.2

# BARD action names that map onto each output class, for the positional
# selector. Unlike SELECTORS these do NOT require the clip to be unambiguous:
# the action's index in the sequence says where to look, so a rebound inside a
# shot-then-rebound clip is perfectly usable. That is the difference between
# 223 rebound clips and the 4,709 that contain one.
TARGET_ACTIONS: dict[str, set[str]] = {
    "rebound": {"Rebound"},
    "steal": {"Steal"},
    "block": {"Block"},
    "shot": {"2PT Shot", "3PT Shot"},
}


# For each output class: the action sets that count as a clean example, then a
# looser headline fallback.
SELECTORS: dict[str, tuple[list[set[str]], str]] = {
    "rebound": ([{"Rebound"}], "Rebound"),
    # {Steal, Turnover} is one steal seen from both sides, not two actions.
    "steal": ([{"Steal", "Turnover"}, {"Steal"}], "Steal"),
    # `shot` and `other` exist in SpaceJam too, and that is the point. While
    # rebound/steal came only from BARD and every other class only from
    # SpaceJam, corpus membership PREDICTED the label for 660 of 2,660 clips,
    # so a model could score on those by recognising the dataset. Augmentation
    # cannot fix that — a logistic regression on nine cheap image statistics
    # still told the corpora apart 95% of the time after blur, brightness and
    # contrast jitter, and 98% before. Populating a class from BOTH corpora
    # removes the shortcut instead of trying to hide it.
    #
    # Free Throws are excluded despite being BARD's second-largest shot pool:
    # SpaceJam's shoot class is field goals, and a set shot from the line is a
    # different action wearing the same label.
    "shot": ([{"2PT Shot"}, {"3PT Shot"}], "2PT Shot"),
    "other": ([{"Foul"}, {"Turnover"}, {"Violation"}], "Foul"),
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


def select_background(count: int) -> list[tuple[str, float]]:
    """Windows of ordinary play, taken from the gaps between labelled actions.

    The classifier was trained only on clips CUT to contain an action, so every
    window it ever saw was an action and it has no way to say "nothing here".
    Run on a game it forced all 6,287 windows into an action class and rebound
    absorbed the slack: 2,253 rebounds against 83 real ones, 27x, at a mean
    confidence of 0.955 on random game windows. A threshold cannot fix a model
    that is confident, so it needs to be taught the negative.

    A BARD clip runs 8-10 s and its labelled actions occupy about 1.6 s, so most
    of every clip is a player bringing the ball up, resetting, or spacing — real
    broadcast footage from exactly the distribution the pipeline is served, and
    exactly what must stop reading as a rebound. This picks, per clip, the point
    furthest from every labelled action.

    What BARD cannot supply is DEAD time: timeouts, free throws, inbounds. A
    real game has more nothing-happening than this, so training on it is
    conservative rather than complete.
    """
    path = hf_hub_download(REPO, "dataset_paths.csv", repo_type="dataset",
                           local_dir=str(META))
    out: list[tuple[str, float]] = []
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            try:
                anns = ast.literal_eval(row["actions"])
            except (ValueError, SyntaxError):
                continue
            if not anns:
                continue
            spots = [(i + 0.5) / len(anns) for i in range(len(anns))]
            # Furthest point from every action, kept away from the clip edges
            # where a cut often lands mid-motion.
            best, best_gap = None, -1.0
            for candidate in [x / 20 for x in range(3, 18)]:
                gap = min(abs(candidate - s_) for s_ in spots)
                if gap > best_gap:
                    best, best_gap = candidate, gap
            # Only worth taking when it is genuinely clear of the action.
            if best is not None and best_gap >= 0.18:
                out.append((row["urls"], best))
            if len(out) >= count:
                break
    return out


def select_clips_positioned(action: str) -> list[tuple[str, float]]:
    """Every clip containing this action, with WHERE in it the action falls.

    BARD carries no timestamps, only an ordered list of actions per clip. That
    ordering places the action well enough to window on: the nth of m actions
    sits at about (n + 0.5) / m through the clip. It turns 223 usable rebound
    clips into thousands, because the exclusion was never about the clips being
    wrong — it was that a midpoint window looked at the shot instead.
    """
    wanted = TARGET_ACTIONS[action]
    path = hf_hub_download(REPO, "dataset_paths.csv", repo_type="dataset",
                           local_dir=str(META))
    out: list[tuple[str, float]] = []
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            try:
                anns = ast.literal_eval(row["actions"])
            except (ValueError, SyntaxError):
                continue
            actions = [a.get("action") for a in anns]
            hit = next((i for i, a in enumerate(actions) if a in wanted), None)
            if hit is None:
                continue
            out.append((row["urls"], (hit + 0.5) / len(actions)))
    return out


def sample_frames(path: str, window: float = WINDOW,
                  position: float = 0.5) -> list[np.ndarray]:
    """Sample N_FRAMES from the middle `window` fraction of the clip.

    Sampling across the WHOLE clip was the single biggest defect in this
    dataset. BARD clips run 8-10 s, so 16 even samples sit half a second apart
    while the labelled action lasts about one second: fourteen of the sixteen
    frames showed unrelated play, and a rebound clip and a steal clip were
    mostly the same footage. Measured on 240 clips, rebound vs steal:

        window 1.0 (whole clip)   0.621   lift +0.121
        window 0.4               0.675   lift +0.175
        window 0.2               0.713   lift +0.213

    Nearly double the lift, and it held at two different crop margins. Crop
    width, by contrast, changed nothing — the problem was never spatial.

    `position` says WHERE in the clip to look, as a fraction. BARD gives an
    ordered list of actions per clip but no timestamps, and that ordering is
    enough: in 3,127 clips the rebound is the second of two actions, so it sits
    around three quarters of the way through while the shot that caused it sits
    early. Centring every window at 0.5 finds the shot, not the rebound — which
    is why only the 101 single-action clips were usable and 4,709 clips contain
    a rebound.
    """
    capture = cv2.VideoCapture(path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return []
    half = max(total * window / 2.0, N_FRAMES / 2.0)
    centre = total * position
    low = max(int(centre - half), 0)
    high = min(int(centre + half), total - 1)
    if high - low < N_FRAMES:                    # clamped at an edge
        low = max(min(low, total - N_FRAMES), 0)
        high = min(low + max(int(2 * half), N_FRAMES), total - 1)
    wanted = set(np.linspace(low, high, N_FRAMES).round().astype(int).tolist())
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
    parser.add_argument("--action",
                        choices=sorted(set(SELECTORS) | {"background"}),
                        required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--full-frame", action="store_true",
                        help="keep the whole frame instead of cropping to the "
                             "ball-handler; required for rim events")
    parser.add_argument("--positional", action="store_true",
                        help="use every clip containing the action, windowed by "
                             "where the action sequence says it falls")
    args = parser.parse_args()
    out = ACTIONS_DIR / args.action

    config = Config()
    detector = load_pipeline_detector(
        "checkpoints/detector.pt", resolve_device(),
        config.detector_conf, config.ball_conf,
    )

    if args.action == "background":
        pairs = select_background(args.count)
        print(f"BARD background: {len(pairs)} windows of ordinary play, taken "
              f"from the gaps between labelled actions")
        chosen_pairs = pairs[: args.count]
    elif args.positional:
        pairs = select_clips_positioned(args.action)
        print(f"BARD {args.action}: {len(pairs)} clips contain this action, "
              f"windowed where the sequence says it falls")
        chosen_pairs = pairs[: args.count]
    else:
        clean, headline = select_clips(args.action)
        print(f"BARD {args.action} clips: {len(clean)} clean, "
              f"{len(headline)} headline-only")
        chosen_pairs = [(u, 0.5) for u in (clean + headline)[: args.count]]
    chosen = [u for u, _ in chosen_pairs]
    positions = {u: pos for u, pos in chosen_pairs}

    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for video in chosen:
        local = hf_hub_download(REPO, video, repo_type="dataset",
                                local_dir=str(META / "clips"))
        frames = sample_frames(local, position=positions.get(video, 0.5))
        if len(frames) < N_FRAMES:
            continue

        if args.full_frame:
            # A rebound is the ball coming off the rim and players converging on
            # it. crop_player keeps ONE player, so the event is outside the
            # frame: `rebound` and ordinary-play clips are visually
            # indistinguishable, and the class became the model's label for
            # "generic player crop" — 82% of a game. Measured on 240 windows,
            # separating rebound from ordinary play scores +0.042 over chance
            # from a player crop and +0.113 from the whole frame.
            crops = [cv2.resize(image, (FRAME_SIZE, FRAME_SIZE))
                     for image in frames]
        else:
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
