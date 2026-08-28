"""Download a small, class-balanced subset of the BARD dataset.

BARD (CC BY 4.0, Gabriele Giudici 2025) is ~182 GB in full — far more than the
spec wants for validation, which asks for 50-200 items (spec §5). This pulls a
balanced subset plus two full clips to act as the pipeline's sample and holdout.

IMPORTANT — BARD does not label `dribble` or `pass`. Its nine action labels are
2PT Shot, 3PT Shot, Free Throw, Rebound, Foul, Turnover, Steal, Block, Violation.
Mapped onto the spec's five ACTIONS that yields only three populated classes:
`shot`, `rebound`, `other`. See ACTION_MAP below and the note printed at the end.

BARD also carries no bounding boxes, so it cannot serve V3 (detector fine-tuning).
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import random
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "GabrieleGiudici/BARD"
META_DIR = Path("data/labeled/bard_meta")
ACTIONS_DIR = Path("data/labeled/actions")
CLIPS_DIR = Path("data/raw_clips")

# BARD label -> spec ACTIONS label. `dribble` and `pass` have no BARD source.
ACTION_MAP = {
    "2PT Shot": "shot",
    "3PT Shot": "shot",
    "Free Throw": "shot",
    "Rebound": "rebound",
    "Foul": "other",
    "Turnover": "other",
    "Steal": "other",
    "Block": "other",
    "Violation": "other",
}
UNSUPPORTED = ("dribble", "pass")


def load_index() -> list[tuple[str, str]]:
    """Return [(video_path, mapped_action)] using each clip's first annotation."""
    path = hf_hub_download(
        REPO, "dataset_paths.csv", repo_type="dataset", local_dir=str(META_DIR)
    )
    pairs: list[tuple[str, str]] = []
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            try:
                annotations = ast.literal_eval(row["actions"])
            except (ValueError, SyntaxError):
                continue
            if not annotations:
                continue
            # The first annotation is the clip's headline event.
            mapped = ACTION_MAP.get(annotations[0].get("action"))
            if mapped:
                pairs.append((row["urls"], mapped))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a BARD subset")
    parser.add_argument("--per-class", type=int, default=40,
                        help="clips to download per mapped action class")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs = load_index()
    print(f"BARD index: {len(pairs)} annotated clips")

    by_action: dict[str, list[str]] = collections.defaultdict(list)
    for video, action in pairs:
        by_action[action].append(video)
    print("available per mapped class:",
          {a: len(v) for a, v in sorted(by_action.items())})

    rng = random.Random(args.seed)
    downloaded = 0
    for action, videos in sorted(by_action.items()):
        target = ACTIONS_DIR / action
        target.mkdir(parents=True, exist_ok=True)
        chosen = rng.sample(videos, min(args.per_class, len(videos)))
        for video in chosen:
            local = hf_hub_download(
                REPO, video, repo_type="dataset", local_dir=str(META_DIR / "clips")
            )
            dest = target / video.replace("/", "__")
            if not dest.exists():
                shutil.copy(local, dest)
            downloaded += 1
        print(f"  {action}: {len(chosen)} clips -> {target}")

    # Two full game clips for the pipeline itself: sample and holdout. Chosen from
    # different games so the holdout is genuinely unseen footage.
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    games = sorted({v.split("/")[0] for v, _ in pairs})
    picks = []
    for game in (games[0], games[len(games) // 2]):
        candidates = [v for v, _ in pairs if v.startswith(game + "/")]
        if candidates:
            picks.append(rng.choice(candidates))
    for name, video in zip(("sample.mp4", "holdout.mp4"), picks):
        local = hf_hub_download(
            REPO, video, repo_type="dataset", local_dir=str(META_DIR / "clips")
        )
        shutil.copy(local, CLIPS_DIR / name)
        print(f"  {name} <- {video}")

    print(f"\ndownloaded {downloaded} action clips")
    print("\nNOTE — BARD labels no `dribble` or `pass`, so those two of the spec's")
    print("five ACTIONS have zero training examples. Populated classes:",
          sorted(by_action))
    print("NOTE — BARD has no bounding boxes; V3 (detector fine-tuning) needs")
    print("another source or hand-labelling.")
    print("\nAttribution (CC BY 4.0): BARD dataset, Gabriele Giudici, 2025 —")
    print("https://github.com/GabrieleGiudic/BARD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
