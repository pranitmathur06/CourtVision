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
#: REUSED between. This said it was "the same cadence clip_detect_raw.py uses"
#: and was 5 where that file uses 3, so every before-and-after run through here
#: handicapped the new mask against the old one by two extra frames of
#: staleness -- and the staleness is not free, because the camera pans between
#: the frame the floor was found on and the frames its mask is applied to.
COURT_EVERY = 3


def source_frames_wanted(broadcast, cache, names, court_every: int):
    """{source frame index: [(clip, row position)]} for every floor needed.

    A clip is cut from the source at `start_s` and keeps its frame rate, so the
    source frame is `round(start_s * fps) + f`. Recomputing the floor from the
    854x480 published clip instead of the source costs 3.6 points of kept ball
    carrier, measured by rebuilding Finals G7 with the setting its own pipeline
    already used and watching the number fall.
    """
    index = json.loads((ROOT / broadcast.clip_index).read_text())
    rows_index = index["clips"] if isinstance(index, dict) else index
    starts = {row["clip"]: float(row["start_s"])
              for row in rows_index if row.get("clip")}
    fps = float(cache.get("fps") or broadcast.fps)
    wanted: dict[int, list] = {}
    for name in names:
        start = starts.get(name)
        if start is None:
            continue
        rows = cache["clips"][name]
        for position in range(0, len(rows), court_every):
            frame = int(round(start * fps)) + int(rows[position]["f"])
            wanted.setdefault(frame, []).append((name, position))
    return wanted


def regions_from_source(broadcast, cache, names, *, erode_share: float,
                        fill_holes: bool, court_every: int):
    """Floors computed from the SOURCE video, in one sequential pass.

    Seeking a two-hour file once per sampled frame is thousands of seeks, each
    landing on a keyframe and decoding forward anyway. Reading the file through
    once and picking off the wanted frames costs one decode of the broadcast.
    """
    wanted = source_frames_wanted(broadcast, cache, names, court_every)
    if not wanted:
        return {}
    capture = cv2.VideoCapture(str(ROOT / broadcast.video))
    if not capture.isOpened():
        print(f"  could not open {broadcast.video}; falling back to the clips")
        return {}
    out: dict[tuple[str, int], object] = {}
    last = max(wanted)
    at = 0
    try:
        while at <= last:
            ok = capture.grab()
            if not ok:
                break
            if at in wanted:
                ok, image = capture.retrieve()
                if ok:
                    region = court_region(image, erode_px=None,
                                          erode_share=erode_share,
                                          fill_holes=fill_holes)
                    for key in wanted[at]:
                        out[key] = region
            at += 1
            if at % 100000 == 0:
                print(f"  {at}/{last} source frames", flush=True)
    finally:
        capture.release()
    return out


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
           limit: int | None = None, kit_max_lab: float | None = None,
           court_every: int = COURT_EVERY, fill_holes: bool = True,
           from_source: bool = False) -> dict:
    broadcast = get(key)
    cache = json.loads((ROOT / broadcast.clip_detections).read_text())
    source = cache.get("source_size") or [1280, 720]
    names = sorted(cache["clips"])
    model = (fit_kits(broadcast, cache, names, source)
             if kit_max_lab is not None else None)
    source_regions = (regions_from_source(broadcast, cache, names,
                                          erode_share=erode_share,
                                          fill_holes=fill_holes,
                                          court_every=court_every)
                      if from_source else {})
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
                  if position % court_every == 0}
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
                if position % court_every == 0 or region is None:
                    frame_image = frames_by_index.get(int(row["f"]))
                    if frame_image is not None:
                        image = frame_image
                    if source_regions:
                        found = source_regions.get((name, position))
                        if found is not None:
                            region = found
                    elif frame_image is not None:
                        region = court_region(image, erode_px=None,
                                              erode_share=erode_share,
                                              fill_holes=fill_holes)
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
    cache["mask_court_every"] = court_every
    cache["mask_fill_holes"] = fill_holes
    cache["mask_from_source"] = from_source
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
    parser.add_argument("--from-source", action="store_true",
                        help="compute the floor from the SOURCE video rather "
                             "than the published 854x480 clip. Costs one "
                             "sequential pass over the broadcast and is worth "
                             "3.6 points of kept ball carrier, measured.")
    parser.add_argument("--fill", type=int, default=None,
                        help="1 to take what the wood encloses, 0 not to. "
                             "Defaults to this broadcast's fitted value.")
    parser.add_argument("--court-every", type=int, default=COURT_EVERY,
                        help="re-find the floor every N detection rows and "
                             "reuse it between. 1 costs the most and is the "
                             "only setting with no staleness in it.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True,
                        help="a NEW file. Writing over the input would make "
                             "the before-and-after unmeasurable.")
    args = parser.parse_args()

    share, gate = args.erode_share, args.kit_max_lab
    fill = None if args.fill is None else bool(args.fill)
    fitted = ROOT / COURT_ERODE_FILE
    if fitted.exists():
        picked = json.loads(fitted.read_text()).get(args.game) or {}
        if isinstance(picked, dict):
            if share is None:
                share = picked.get("erode_share")
            if gate is None:
                gate = picked.get("kit_max_lab")
            if fill is None:
                fill = picked.get("fill_holes")
        elif share is None:
            share = picked
    if fill is None:
        fill = True
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
    print(f"  hole filling {'on' if fill else 'off'}")
    got = remask(args.game, float(share), out, limit=args.limit,
                 kit_max_lab=None if gate is None else float(gate),
                 court_every=args.court_every, fill_holes=bool(fill),
                 from_source=args.from_source)
    print(f"  {got['frames']} frames remasked, {got['changed']} changed, "
          f"{got['clips_missing']} clips not on disk")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
