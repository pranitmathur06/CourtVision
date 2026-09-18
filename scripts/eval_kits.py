#!/usr/bin/env python3
"""Is the kit model right, measured without a single label.

THE RULE COMES FROM THE SPORT: neither team may have six players on the court.
So every frame where the model gives six boxes the same kit is a model error,
and the error rate needs no labels, no feed and no hand judgement. A model that
calls everybody one kit breaks the rule on every frame with six boxes.

A ONE-SIDED TEST CAN BE PASSED BY SAYING NOTHING, so the rejected share is
printed beside it and has an arithmetic prediction to answer to: three
officials among thirteen people on a live court is 0.231.

Five-a-side is reported too and is a DIAGNOSTIC, not the score. Ten boxes is
almost never the ten players -- see `courtvision.kits` for why that test was
abandoned as a headline.

SPLIT DISCIPLINE. The centres are fitted on the even-numbered clips of a
broadcast and every number is measured on the odd-numbered ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.games import get, registry  # noqa: E402
from courtvision.kits import (  # noqa: E402
    KitModel,
    over_five,
    sample_clip,
    split_report,
)
from courtvision.stats import wilson  # noqa: E402

#: Detection rows sampled per clip when fitting the three centres.
FIT_ROWS_PER_CLIP = 3
#: Detection rows sampled per clip when measuring.
EVAL_ROWS_PER_CLIP = 12
#: Frames with fewer boxes than this cannot break the six-a-side rule.
MIN_BOXES_FOR_RULE = 6
#: Three officials among thirteen people on a live court.
EXPECTED_OFFICIAL_SHARE = 3.0 / 13.0


def _harvest(broadcast, clips: dict, names: list[str], source_size, want: int):
    frames = []
    for name in names:
        path = ROOT / broadcast.clip_dir / name
        if not path.exists():
            continue
        for _row, _boxes, colours in sample_clip(path, clips[name], source_size,
                                                 want=want):
            frames.append(colours)
    return frames


def evaluate(key: str, *, limit: int | None = None) -> dict:
    broadcast = get(key)
    cache = json.loads((ROOT / broadcast.clip_detections).read_text())
    clips = cache["clips"]
    source_size = cache.get("source_size") or [1280, 720]
    names = sorted(clips)
    if limit:
        names = names[:limit]

    fit_frames = _harvest(broadcast, clips, names[0::2], source_size,
                          FIT_ROWS_PER_CLIP)
    model = KitModel.fit([c for frame in fit_frames for c in frame if c is not None])
    if model is None:
        return {"game": key, "label": broadcast.label, "fitted": False}

    frames = _harvest(broadcast, clips, names[1::2], source_size,
                      EVAL_ROWS_PER_CLIP)

    counts: list[tuple[int, int]] = []
    splits: list[int] = []
    rejected = boxes = 0
    for colours in frames:
        kits = [model.kit(c)[0] for c in colours]
        boxes += len(kits)
        rejected += sum(1 for k in kits if k is None)
        zero, one = kits.count(0), kits.count(1)
        if len(kits) >= MIN_BOXES_FOR_RULE:
            counts.append((zero, one))
        if zero + one == 10:
            splits.append(min(zero, one))

    rule = over_five(counts)
    diagnostic = split_report(splits)
    low, high = wilson(rule["n"] - rule["violations"], rule["n"])
    return {
        "game": key,
        "label": broadcast.label,
        "fitted": True,
        "held_out": bool(broadcast.held_out_for("kits")),
        "fit_clips": len(names[0::2]),
        "eval_frames": len(frames),
        "obeys_five_a_side": 1.0 - rule["rate"],
        "rule_n": rule["n"],
        "ci": [low, high],
        "rejected_share": rejected / max(1, boxes),
        "separation_lab": model.separation(),
        "centres": model.centres.tolist(),
        "officials": model.officials.tolist(),
        "five_five": diagnostic["exact"],
        "five_five_n": diagnostic["n"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N clips (a fast smoke run)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    keys = args.game or list(registry())
    results = []
    print()
    print("  eval_kits.py -- no labels anywhere: the sport is the ground truth")
    print()
    print(f"  {'game':<6} {'<=5 a side':>11} {'n':>6} {'95% CI':>12} "
          f"{'reject':>7} {'sep':>6} {'5-5':>6} {'n':>5}")
    print("  " + "-" * 72)
    for key in keys:
        row = evaluate(key, limit=args.limit)
        results.append(row)
        if not row.get("fitted"):
            print(f"  {key:<6} not enough torso samples to fit")
            continue
        low, high = row["ci"]
        print(f"  {key:<6} {row['obeys_five_a_side']:>11.3f} {row['rule_n']:>6} "
              f"{low:>5.2f}-{high:<5.2f} {row['rejected_share']:>7.3f} "
              f"{row['separation_lab']:>6.1f} {row['five_five']:>6.3f} "
              f"{row['five_five_n']:>5}")
    print()
    print("  <=5 a side  share of frames where NEITHER kit was given six players.")
    print("              The sport forbids six, so every miss is the model's.")
    print(f"  reject      share of boxes called officials. Expect about "
          f"{EXPECTED_OFFICIAL_SHARE:.3f}")
    print("              (three officials among thirteen people). Far above it")
    print("              means the model is buying the rule by abstaining.")
    print("  sep         CIELAB distance between the kit centres. A broadcast")
    print("              where both teams wear the same colour has a small one.")
    print("  5-5         DIAGNOSTIC ONLY. Ten boxes is usually ten of thirteen")
    print("              people, not the ten players. Never quote it as accuracy.")
    print()
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
