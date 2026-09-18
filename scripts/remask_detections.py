#!/usr/bin/env python3
"""Recompute the on-court mask of a cached detection file. No model runs.

A detection cache holds two different things: what the detector saw, which
costs a GPU pass over the whole broadcast, and which of those boxes stood on
the floor, which is a few morphology operations on the same frames. Changing
the mask has meant rebuilding both, so it has effectively never been changed.

This rewrites only `on`. It reads the clips that are already on disk, and it is
safe to point at a NEW output file and compare the two, which is what the
before-and-after in `eval_court_mask.py` needs.

The erosion comes from `fit_court_mask.py`'s per-broadcast choice, which is made
against two bounds that need no labels: more than thirteen people kept is
impossible, and the man holding the ball must be kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.candidates import (  # noqa: E402
    COURT_ERODE_FILE,
    COURT_ERODE_SHARE,
    court_region,
    stands_on_court,
)
from courtvision.games import get  # noqa: E402
from courtvision.kits import KitModel, sample_clip, torso_lab  # noqa: E402

#: The floor moves slowly, so the region is re-found every this many rows and
#: reused between -- the same cadence `clip_detect_raw.py` uses.
COURT_EVERY = 5


def fit_kits(broadcast, cache, names, source) -> KitModel | None:
    colours = []
    for name in names[0::2][:80]:
        path = ROOT / broadcast.clip_dir / name
        if not path.exists():
            continue
        for _row, _boxes, sampled in sample_clip(path, cache["clips"][name],
                                                 source, want=3):
            colours.extend(c for c in sampled if c is not None)
    return KitModel.fit(colours)


def remask(key: str, erode_share: float, out_path: Path, *,
           limit: int | None = None, kit_max_lab: float | None = None) -> dict:
    broadcast = get(key)
    cache = json.loads((ROOT / broadcast.clip_detections).read_text())
    source = cache.get("source_size") or [1280, 720]
    names = sorted(cache["clips"])
    model = (fit_kits(broadcast, cache, names, source)
             if kit_max_lab is not None else None)
    if limit:
        names = names[:limit]
    changed = frames = missing = 0
    for index, name in enumerate(names):
        path = ROOT / broadcast.clip_dir / name
        rows = cache["clips"][name]
        if not path.exists():
            missing += 1
            continue
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            missing += 1
            continue
        scale_x = capture.get(cv2.CAP_PROP_FRAME_WIDTH) / source[0]
        scale_y = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / source[1]
        region = None
        image = None
        # Read the clip FORWARD rather than seeking to each wanted frame.
        # Eighteen random seeks into a 360-frame clip each land on a keyframe
        # and decode forward from it anyway, so the decoder does much of the
        # sequential work regardless and pays the seek on top. Not benchmarked
        # against the seeking version under a quiet machine, so this is a
        # reasoned change rather than a measured speedup -- the OUTPUT is
        # identical either way, which is what the tests check.
        wanted = {int(row["f"]): position for position, row in enumerate(rows)
                  if position % COURT_EVERY == 0}
        at = 0
        last = max(wanted) if wanted else -1
        frames_by_index: dict[int, object] = {}
        while at <= last:
            ok, frame_image = capture.read()
            if not ok:
                break
            if at in wanted:
                frames_by_index[at] = frame_image
            at += 1
        try:
            for position, row in enumerate(rows):
                people = [b for b in row["d"] if b[0] in ("p", "h")]
                if position % COURT_EVERY == 0 or region is None:
                    frame_image = frames_by_index.get(int(row["f"]))
                    if frame_image is not None:
                        image = frame_image
                        region = court_region(image, erode_px=None,
                                              erode_share=erode_share)
                if not people:
                    row["on"] = []
                    continue
                boxes = np.array([[b[2] * scale_x, b[3] * scale_y,
                                   b[4] * scale_x, b[5] * scale_y]
                                  for b in people], dtype=float)
                keep = (stands_on_court(region, boxes).tolist()
                        if region is not None else [False] * len(people))
                if model is not None and any(keep):
                    # The kit gate needs THIS row's frame. The region is
                    # refreshed every COURT_EVERY rows, so the decoded image is
                    # this row's own on those and stale in between; a seek per
                    # row would be twenty thousand seeks a broadcast, so a
                    # stale frame within a fifth of a second is read instead
                    # and the box is looked up where it stands now. Players
                    # move a few pixels in that time and a jersey does not
                    # change colour.
                    if image is not None:
                        for index, box in enumerate(boxes):
                            if not keep[index]:
                                continue
                            colour = torso_lab(image, box)
                            if not model.belongs_on_court(colour, kit_max_lab):
                                keep[index] = False
                frames += 1
                if keep != (row.get("on") or []):
                    changed += 1
                row["on"] = keep
        finally:
            capture.release()
        if (index + 1) % 40 == 0:
            print(f"  {index + 1}/{len(names)} clips", flush=True)
    cache["mask_erode_share"] = erode_share
    cache["mask_kit_max_lab"] = kit_max_lab
    cache["mask_rebuilt_from"] = str(broadcast.clip_detections)
    out_path.write_text(json.dumps(cache))
    return {"frames": frames, "changed": changed, "clips_missing": missing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--erode-share", type=float, default=None,
                        help="defaults to this broadcast's fitted value")
    parser.add_argument("--kit-max-lab", type=float, default=None,
                        help="drop a kept box whose torso is further than this "
                             "in CIELAB from both kits AND the officials -- a "
                             "spectator. Defaults to the fitted value.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True,
                        help="a NEW file. Writing over the input would make "
                             "the before-and-after unmeasurable.")
    args = parser.parse_args()

    share, gate = args.erode_share, args.kit_max_lab
    fitted = ROOT / COURT_ERODE_FILE
    if fitted.exists():
        picked = json.loads(fitted.read_text()).get(args.game) or {}
        if isinstance(picked, dict):
            if share is None:
                share = picked.get("erode_share")
            if gate is None:
                gate = picked.get("kit_max_lab")
        elif share is None:
            share = picked
    if share is None:
        share = COURT_ERODE_SHARE
        print(f"  no fitted erosion for {args.game}; using the shipped "
              f"{share:.4f} of frame height")
    else:
        print(f"  erosion {float(share):.4f} of frame height")

    out = Path(args.out)
    if out.resolve() == (ROOT / get(args.game).clip_detections).resolve():
        print("  refusing to overwrite the input cache")
        return 2
    if gate is not None:
        print(f"  kit gate {float(gate):.0f} CIELAB")
    got = remask(args.game, float(share), out, limit=args.limit,
                 kit_max_lab=None if gate is None else float(gate))
    print(f"  {got['frames']} frames remasked, {got['changed']} changed, "
          f"{got['clips_missing']} clips not on disk")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
