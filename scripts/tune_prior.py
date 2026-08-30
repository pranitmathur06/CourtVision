"""Measure what the classifier emits on continuous game footage, per prior strength.

Held-out clip accuracy cannot see the failure this exists to fix: the model
scored 0.816 on clips while calling 67% of a game a rebound. This samples
windows the way the pipeline does — uniformly along real game footage, not at
labelled action positions — and reports the label mix against what BARD says is
actually in that game.

It runs the classifier only, not the whole pipeline, so a full sweep of prior
strengths costs minutes instead of an hour per setting.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import sys

import numpy as np

N_FRAMES = 16


def main() -> int:
    import cv2
    from huggingface_hub import hf_hub_download

    from courtvision.action_classifier import VideoMaeClassifier, crop_player
    from courtvision.config import Config
    from courtvision.detection import load_pipeline_detector
    from courtvision.device import resolve_device
    from courtvision.types import HANDLER, PLAYER

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="chi-vs-tor-0022401223")
    parser.add_argument("--clips", type=int, default=60)
    parser.add_argument("--windows-per-clip", type=int, default=4)
    parser.add_argument("--strengths", default="0,0.5,1.0")
    parser.add_argument("--model", default="checkpoints/action_classifier")
    args = parser.parse_args()
    strengths = [float(s) for s in args.strengths.split(",")]

    meta = hf_hub_download("GabrieleGiudici/BARD", "dataset_paths.csv",
                           repo_type="dataset",
                           local_dir="data/labeled/bard_meta")
    urls, truth = [], collections.Counter()
    with open(meta) as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if args.game not in row["urls"]:
                continue
            urls.append(row["urls"])
            try:
                for a in ast.literal_eval(row["actions"]):
                    truth[a.get("action")] += 1
            except (ValueError, SyntaxError):
                pass
    urls = urls[: args.clips]
    print(f"  {args.game}: sampling {args.clips} clips x "
          f"{args.windows_per_clip} windows", flush=True)

    config = Config()
    device = resolve_device()
    detector = load_pipeline_detector("checkpoints/detector.pt", device,
                                      config.detector_conf, config.ball_conf)

    # Decode every window once; only the classifier changes between strengths.
    all_crops = []
    for i, url in enumerate(urls):
        local = hf_hub_download("GabrieleGiudici/BARD", url, repo_type="dataset",
                                local_dir="/workspace/bard-tune")
        capture = cv2.VideoCapture(local)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 60:
            capture.release()
            continue
        # Uniformly along the clip — the pipeline does not know where actions are.
        for centre in np.linspace(0.15, 0.85, args.windows_per_clip):
            span = max(int(total * 0.18), N_FRAMES)
            low = max(int(total * centre - span / 2), 0)
            picks = np.linspace(low, min(low + span, total - 1),
                                N_FRAMES).round().astype(int)
            frames = []
            for p in picks:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(p))
                ok, image = capture.read()
                if ok:
                    frames.append(image)
            if len(frames) < N_FRAMES:
                continue
            box = None
            for f in frames[:4]:
                found = detector.detect(f)
                cands = ([d for d in found if d.label == HANDLER]
                         or [d for d in found if d.label == PLAYER])
                if cands:
                    box = max(cands, key=lambda d: d.conf).box
                    break
            if box is None:
                continue
            all_crops.append([cv2.cvtColor(crop_player(f, box), cv2.COLOR_BGR2RGB)
                              for f in frames[:N_FRAMES]])
        capture.release()
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(urls)} clips, {len(all_crops)} windows",
                  flush=True)

    counts = {a: len(list(__import__("pathlib").Path(
        f"data/labeled/actions/{a}").glob("*.mp4")))
        for a in ("dribble", "pass", "shot", "rebound", "block", "steal",
                  "other", "background")}
    print(f"\n  {len(all_crops)} windows from real game footage")
    print(f"  BARD says this game contains: "
          f"{dict(truth.most_common(5))}\n")

    for strength in strengths:
        clf = VideoMaeClassifier(args.model, device, prior_strength=strength,
                                 train_counts=counts)
        labels = [lab for lab, _ in clf.classify_batch(all_crops)]
        mix = collections.Counter(labels)
        share = {k: f"{v / len(labels):.0%}" for k, v in mix.most_common()}
        print(f"  strength {strength:<4} -> {share}")
        del clf
    return 0


if __name__ == "__main__":
    sys.exit(main())
