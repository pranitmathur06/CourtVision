"""Re-split the keypoint dataset by source game, because the shipped one leaks.

Roboflow split at the frame level. Every one of the 107 clips in its `test`
split also appears in `train`, and the clips are 5-second segments sampled at a
few frames each -- so a "held out" frame is a near-duplicate of one the model
trained on, taken a fraction of a second earlier. Scored that way the model
reads 100% registered at 0.87 ft, which measures memorisation.

This regroups every image by the game it came from and assigns whole games to
each split, so nothing in test shares a camera, an arena, a lighting setup or a
possession with anything in train. That is the same rule the screen detector
needed -- it was scored per source video for exactly this reason.

The original download is left untouched; the new split is written beside it.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

SOURCE = Path("data/labeled/court_keypoints")
TARGET = Path("data/labeled/court_keypoints_by_game")
#: Two naming conventions are in use, and handling only the first silently
#: turns every frame of the second into its own "game" -- which puts adjacent
#: frames back on both sides of the split, the exact leak this script exists to
#: remove:
#:     ...-game-1-q1-01_54-01_48_mp4-0005_jpg.rf.<hash>.jpg
#:     ...-game-1-09_49-09_44_mp4-0004_jpg.rf.<hash>.jpg
_CLIP = re.compile(r"_mp4.*$")
_QUARTER = re.compile(r"-q\d.*$")
_TIMESTAMP = re.compile(r"-\d+_\d+-\d+_\d+$")


def game_of(name: str) -> str:
    """The game a frame came from, with clip and frame identifiers removed."""
    name = _CLIP.sub("", name)
    name = _QUARTER.sub("", name)
    return _TIMESTAMP.sub("", name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-games", type=int, default=4)
    parser.add_argument("--valid-games", type=int, default=3)
    args = parser.parse_args()

    by_game: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        for image in sorted((SOURCE / split / "images").glob("*.jpg")):
            label = SOURCE / split / "labels" / (image.stem + ".txt")
            if label.exists():
                by_game[game_of(image.name)].append((image, label))

    games = sorted(by_game)
    if len(games) < args.test_games + args.valid_games + 2:
        print(f"FAIL - only {len(games)} games")
        return 1
    # Deterministic and declared here rather than chosen after seeing a score:
    # every third game to test, then every third of the rest to valid.
    test = games[::len(games) // args.test_games][:args.test_games]
    rest = [g for g in games if g not in test]
    valid = rest[::len(rest) // args.valid_games][:args.valid_games]
    train = [g for g in rest if g not in valid]

    for split, chosen in (("train", train), ("valid", valid), ("test", test)):
        for kind in ("images", "labels"):
            path = TARGET / split / kind
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True)
        for game in chosen:
            for image, label in by_game[game]:
                shutil.copy2(image, TARGET / split / "images" / image.name)
                shutil.copy2(label, TARGET / split / "labels" / label.name)
        print(f"{split:6s} {len(chosen):2d} games, "
              f"{sum(len(by_game[g]) for g in chosen):4d} images")
        for game in chosen:
            print(f"         {game}")

    yaml = (SOURCE / "data.yaml").read_text()
    yaml = re.sub(r"^path: .*$", f"path: {TARGET.resolve()}", yaml, flags=re.M)
    (TARGET / "data.yaml").write_text(yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
