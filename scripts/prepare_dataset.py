"""Write the ultralytics data.yaml for the fine-tuning subset.

Class order here MUST match courtvision.detection.FINETUNED_CLASS_MAP.
"""

from __future__ import annotations

import sys
from pathlib import Path

from courtvision.detection import FINETUNED_CLASS_MAP

ROOT = Path("data/labeled/detector")


def main() -> int:
    names = [FINETUNED_CLASS_MAP[i] for i in sorted(FINETUNED_CLASS_MAP)]
    missing = [
        str(ROOT / sub)
        for sub in ("images/train", "images/val", "labels/train", "labels/val")
        if not (ROOT / sub).is_dir()
    ]
    if missing:
        print(f"FAIL — missing dataset directories: {missing}")
        return 1

    yaml_path = ROOT / "data.yaml"
    yaml_path.write_text(
        f"path: {ROOT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    n_train = len(list((ROOT / "images/train").glob("*")))
    n_val = len(list((ROOT / "images/val").glob("*")))
    print(f"OK — wrote {yaml_path}; {n_train} train / {n_val} val images; names={names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
