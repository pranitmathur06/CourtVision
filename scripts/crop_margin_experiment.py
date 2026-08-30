"""Does a wider crop help rebound vs steal? Changing ONLY the margin.

V7 on a GPU collapsed rebound to 0.03, and the rim appears in 0 of 40 sampled
clips of every class, so the crop was the obvious suspect. An earlier attempt to
test that was confounded: it cropped to the highest-confidence player while the
original clips crop to the BALL-HANDLER, so it changed which person was centred
as well as how much court was visible. A crop centred on the wrong player loses
accuracy by itself, and the comparison could not separate the two.

This holds the selection identical to add_bard_action — ball-handler, else the
player nearest the ball, else the last known box — and varies only the margin.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

BASE = "MCG-NJU/videomae-base-finetuned-kinetics"
SOURCE = Path("data/labeled/bard_meta/clips")
N_FRAMES = 16


def handler_box(detections, last_box):
    """Exactly add_bard_action's rule, so only the margin differs."""
    from courtvision.types import BALL, HANDLER, PLAYER

    handlers = [d for d in detections if d.label == HANDLER]
    if handlers:
        return max(handlers, key=lambda d: d.conf).box
    balls = [d for d in detections if d.label == BALL]
    players = [d for d in detections if d.label == PLAYER]
    if balls and players:
        bx, by = max(balls, key=lambda d: d.conf).box.center
        return min(players, key=lambda p: (p.box.center[0] - bx) ** 2
                   + (p.box.center[1] - by) ** 2).box
    return last_box


def main() -> int:
    import cv2
    import torch
    from transformers import VideoMAEImageProcessor

    from courtvision.action_classifier import crop_player
    from courtvision.config import Config
    from courtvision.detection import load_pipeline_detector
    from courtvision.device import resolve_device
    from courtvision.videomae import load_videomae_classifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--margins", default="0.25,1.0,2.0")
    parser.add_argument("--per-class", type=int, default=120)
    args = parser.parse_args()
    margins = [float(m) for m in args.margins.split(",")]

    config = Config()
    device = resolve_device()
    detector = load_pipeline_detector("checkpoints/detector.pt", device,
                                      config.detector_conf, config.ball_conf)
    processor = VideoMAEImageProcessor.from_pretrained(BASE)
    model, _ = load_videomae_classifier(BASE)
    model = model.to(device).eval()

    chosen: list[tuple[Path, int]] = []
    for label, action in enumerate(("rebound", "steal")):
        clips = sorted(Path(f"data/labeled/actions/{action}").glob("*__*.mp4"))
        random.Random(0).shuffle(clips)
        taken = 0
        for clip in clips:
            game, number = clip.stem.split("__")
            source = SOURCE / game / f"{number}.mp4"
            if source.exists():
                chosen.append((source, label))
                taken += 1
            if taken >= args.per_class:
                break
    print(f"{len(chosen)} source clips; margins {margins}", flush=True)

    # Detect once per frame and reuse across margins: the boxes do not depend on
    # the crop, and detection is the expensive half.
    results: dict[float, tuple[list, list]] = {m: ([], []) for m in margins}
    with torch.no_grad():
        for index, (source, label) in enumerate(chosen):
            capture = cv2.VideoCapture(str(source))
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                capture.release()
                continue
            wanted = np.linspace(0, total - 1, N_FRAMES).round().astype(int)
            frames, boxes, last = [], [], None
            for want in wanted:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(want))
                ok, image = capture.read()
                if not ok:
                    break
                box = handler_box(detector.detect(image), last)
                last = box or last
                if box is None:
                    break
                frames.append(image)
                boxes.append(box)
            capture.release()
            if len(frames) < N_FRAMES:
                continue

            for margin in margins:
                crops = [cv2.cvtColor(crop_player(f, b, margin=margin),
                                      cv2.COLOR_BGR2RGB)
                         for f, b in zip(frames, boxes)]
                inputs = {k: v.to(device) for k, v in
                          processor(crops, return_tensors="pt").items()}
                pooled = model.videomae(**inputs).last_hidden_state.mean(1)
                results[margin][0].append(pooled.squeeze(0).float().cpu().numpy())
                results[margin][1].append(label)
            if (index + 1) % 40 == 0:
                print(f"  {index + 1}/{len(chosen)}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print(f"\n  {'margin':>8}{'n':>6}{'baseline':>10}{'accuracy':>10}{'lift':>8}"
          f"   rim visible")
    for margin in margins:
        X = np.array(results[margin][0])
        y = np.array(results[margin][1])
        if len(y) < 30:
            print(f"  {margin:>8} too few clips")
            continue
        base = max((y == 0).mean(), (y == 1).mean())
        model_ = make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=4000,
                                                  class_weight="balanced"))
        acc = cross_val_score(model_, X, y, cv=5, scoring="accuracy").mean()
        print(f"  {margin:>8}{len(y):>6}{base:>10.3f}{acc:>10.3f}{acc - base:>+8.3f}")
    print("\n  Only the margin differs; the ball-handler selection is identical\n"
          "  to add_bard_action, which is what the earlier attempt got wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
