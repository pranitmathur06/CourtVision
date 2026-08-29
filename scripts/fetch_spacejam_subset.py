"""Build a balanced action-classifier dataset from SpaceJam.

SpaceJam (Simone Francia, MIT) is the dataset the spec §5 recommends. The
author's Google Drive link is dead — three open issues on the repo report it —
but a 2021 mirror survives on Kaggle as `antocommii/spacejam-action-recognition`.

Why it is the right source, where BARD was not:

* It labels **dribble (3,490) and pass (1,070)**, which BARD never annotates.
* Labels are single and unambiguous; 37% of BARD clips carried two or more
  different actions, and its `rebound` class was confounded with `shot` (only
  101 clean examples in 14,676 clips).
* Clips are already 16 frames at 10 fps — exactly `Config.action_window_frames`
  and `Config.target_fps`.
* Clips are cropped to ONE player, which is what the classifier should see:
  "what is this player doing", not "what is happening somewhere in this frame".
  `classify_windows` crops to the possession holder to match.

SpaceJam has no `rebound` class — that comes from BARD via add_bard_rebound.py,
converted to this same crop format so the two sources are visually comparable.
"""

from __future__ import annotations

import argparse
import ast
import collections
import random
import shutil
import sys
import zipfile
from pathlib import Path

ARCHIVE = Path("data/labeled/spacejam/spacejam-action-recognition.zip")
OUT = Path("data/labeled/actions")

# SpaceJam class id -> spec ACTIONS label. Everything not a dribble, pass or
# shot is court activity without a distinct action, so it forms `other`.
REMAP = {
    3: "dribble",
    1: "pass",
    4: "shot",
    0: "block",
    2: "other",   # run
    5: "other",   # ball in hand
    6: "other",   # defense
    7: "other",   # pick
    8: "other",   # no_action
    9: "other",   # walk
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V7 dataset from SpaceJam")
    parser.add_argument("--per-class", type=int, default=400,
                        help="clips per class; capped by the rarest class (shot=426)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not ARCHIVE.exists():
        print(f"FAIL — no SpaceJam archive at {ARCHIVE}")
        return 1

    archive = zipfile.ZipFile(ARCHIVE)
    annotations = ast.literal_eval(archive.read("annotation_dict.json").decode())

    by_action: dict[str, list[str]] = collections.defaultdict(list)
    for clip_id, class_id in annotations.items():
        label = REMAP.get(class_id)
        if label:
            by_action[label].append(clip_id)
    print("available:", {k: len(v) for k, v in sorted(by_action.items())})

    if OUT.exists():
        shutil.rmtree(OUT)

    rng = random.Random(args.seed)
    written = 0
    for label, clip_ids in sorted(by_action.items()):
        target = OUT / label
        target.mkdir(parents=True, exist_ok=True)
        chosen = rng.sample(clip_ids, min(args.per_class, len(clip_ids)))
        for clip_id in chosen:
            member = f"examples/{clip_id}.mp4"
            try:
                data = archive.read(member)
            except KeyError:
                continue
            (target / f"{clip_id}.mp4").write_bytes(data)
            written += 1
        print(f"  {label}: {len(chosen)} clips -> {target}")

    print(f"\n{written} clips written to {OUT}")
    print("NOTE — SpaceJam has no `rebound` class; run add_bard_rebound.py for it.")
    print("\nSource: SpaceJam (Simone Francia, MIT), Kaggle mirror "
          "antocommii/spacejam-action-recognition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
