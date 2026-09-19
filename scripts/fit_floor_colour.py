#!/usr/bin/env python3
"""Learn what colour THIS arena's floor is, from the broadcast, with no labels.

`candidates.court_region` accepts hue 5-30 as wood and 95-130 as paint -- tan
and blue, which is a list of colours somebody had seen. Toyota Center's court
is red: measured under the feet of the players standing on it, hue 174 and
saturation 230. The rule accepts none of it, 76% of the frames where the mask
drops the man holding the ball there have ZERO floor under his feet, and the
mask keeps him on 0.572 of frames against 0.860 and 0.906 on the two tan-and-
blue arenas.

A PLAYER STANDS ON THE FLOOR. So the strip of pixels just below a player's box
is floor by the definition of standing, and the person detector that already
ran over this broadcast hands over hundreds of samples a minute. No labels, no
new model, and it answers for whatever the next arena's court looks like.

Read from the SOURCE broadcast, like everything else here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.broadcast import SourceReader, clip_starts  # noqa: E402
from courtvision.floor_colour import collect, learn  # noqa: E402
from courtvision.games import get, registry  # noqa: E402
from courtvision.kits import MIN_BOX_CONF  # noqa: E402

#: Where the learned floors live. Under `data/` because they are an input the
#: pipeline depends on, not a result: `outputs/` is ignored by git, and a floor
#: living there would silently not exist on another machine.
FLOOR_FILE = "data/floor_colour.json"
#: One detection row in this many is sampled for colours.
ROW_STRIDE = 30
#: A floor model accepting more than this share of the colour space is not a
#: floor model. It would mask most of a frame and the caller should refuse it.
MAX_COLOUR_SHARE = 0.25


def fit(key: str, *, clips_wanted: int | None = None) -> dict:
    broadcast = get(key)
    cache = json.loads((ROOT / broadcast.clip_detections).read_text())
    starts = clip_starts(ROOT / broadcast.clip_index)
    step = max(1, int(cache.get("step") or 2))
    names = [n for n in sorted(cache["clips"]) if n in starts]
    if clips_wanted:
        names = names[:clips_wanted]

    samples = []
    reader = SourceReader(ROOT / broadcast.video)
    if not reader.ok:
        return {"game": key, "learned": False, "why": "the video would not open"}
    try:
        for name in names:
            rows = cache["clips"][name]
            wanted = {position: position
                      for position in range(0, len(rows), ROW_STRIDE)}
            for position, image in reader.frames(starts[name], wanted, step):
                boxes = [b[2:] for b in rows[position]["d"]
                         if b[0] == "p" and b[1] >= MIN_BOX_CONF]
                samples.extend(collect(image, boxes))
    finally:
        reader.close()

    floor = learn(samples)
    if floor is None:
        return {"game": key, "learned": False,
                "why": f"only {len(samples)} usable samples"}
    if floor.share() > MAX_COLOUR_SHARE:
        return {"game": key, "learned": False,
                "why": f"accepts {floor.share():.3f} of colour space"}
    return {"game": key, "label": broadcast.label, "learned": True,
            "samples": floor.samples, "colour_share": floor.share(),
            "bins": floor.bins.astype(int).tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--clips", type=int, default=60,
                        help="clips sampled per broadcast")
    parser.add_argument("--out", default=FLOOR_FILE)
    args = parser.parse_args()

    path = ROOT / args.out
    existing = json.loads(path.read_text()) if path.exists() else {}
    print()
    print("  fit_floor_colour.py -- a player stands on the floor, so the strip")
    print("  below his box IS the floor. No labels, no new model.")
    print()
    print(f"  {'game':<6} {'samples':>8} {'colour space':>13}  verdict")
    print("  " + "-" * 52)
    for key in (args.game or list(registry())):
        got = fit(key, clips_wanted=args.clips)
        if got.get("learned"):
            existing[key] = {"bins": got["bins"], "samples": got["samples"]}
            print(f"  {key:<6} {got['samples']:>8} {got['colour_share']:>13.4f}"
                  f"  learned")
        else:
            print(f"  {key:<6} {'--':>8} {'--':>13}  NOT learned: {got['why']}")
    path.write_text(json.dumps(existing))
    print()
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
