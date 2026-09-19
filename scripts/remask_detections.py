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


def clip_starts(broadcast) -> dict[str, float]:
    """{clip name: its start in the SOURCE video, in seconds}."""
    index = json.loads((ROOT / broadcast.clip_index).read_text())
    rows = index["clips"] if isinstance(index, dict) else index
    return {row["clip"]: float(row["start_s"])
            for row in rows if row.get("clip")}


def regions_from_source(broadcast, cache, name, start_s, *, erode_share: float,
                        fill_holes: bool, court_every: int, step: int,
                        capture) -> dict[int, object]:
    """Floors for one clip, computed from the SOURCE video.

    EXACTLY AS THE PIPELINE READS IT. `clip_detect_raw.py` seeks the source
    with `CAP_PROP_POS_MSEC` and then reads forward, so a clip's frame `f` is
    the `f`-th frame after that seek and NOT `round(start_s * fps) + f`. A
    POS_MSEC seek lands on a decodable frame, which is also what ffmpeg did
    when it cut the clip, so the two agree -- and the arithmetic does not.
    Computing the mapping instead of copying it put the floor 24 frames away
    from the boxes it was applied to and scored 0.400 kept ball carrier against
    the shipped 0.872.

    One seek per clip, then a short sequential read, so a broadcast costs its
    own clip footage rather than a pass over the whole file.
    """
    capture.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    rows = cache["clips"][name]
    wanted = {position * step: position
              for position in range(0, len(rows), court_every)}
    out: dict[int, object] = {}
    if not wanted:
        return out
    last = max(wanted)
    for index in range(last + 1):
        ok = capture.grab()
        if not ok:
            break
        position = wanted.get(index)
        if position is None:
            continue
        ok, image = capture.retrieve()
        if not ok:
            continue
        out[position] = court_region(image, erode_px=None,
                                     erode_share=erode_share,
                                     fill_holes=fill_holes)
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
           from_source: bool = True) -> dict:
    broadcast = get(key)
    cache = json.loads((ROOT / broadcast.clip_detections).read_text())
    source = cache.get("source_size") or [1280, 720]
    names = sorted(cache["clips"])
    model = (fit_kits(broadcast, cache, names, source)
             if kit_max_lab is not None else None)
    starts = clip_starts(broadcast) if from_source else {}
    broadcast_video = None
    if from_source:
        broadcast_video = cv2.VideoCapture(str(ROOT / broadcast.video))
        if not broadcast_video.isOpened():
            print(f"  could not open {broadcast.video}; falling back to clips")
            broadcast_video = None
    # The cache records the frame step it was built with -- 2 on a 30 fps
    # broadcast, 4 on a 60 fps one -- so row `position` is source frame
    # `position * step` after the seek. Recomputing it from the rate would
    # reintroduce the frames-versus-seconds confusion this repository has now
    # made four times.
    step = max(1, int(cache.get("step") or 2))
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
        source_regions = {}
        if broadcast_video is not None and name in starts:
            source_regions = regions_from_source(
                broadcast, cache, name, starts[name], erode_share=erode_share,
                fill_holes=fill_holes, court_every=court_every, step=step,
                capture=broadcast_video)
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
                        found = source_regions.get(position)
                        if found is not None:
                            region = found
                    elif frame_image is not None:
                        region = court_region(image, erode_px=None,
                                              erode_share=erode_share,
                                              fill_holes=fill_holes)
                if not people:
                    row["on"] = []
                    continue
                # The boxes are in SOURCE pixels. Scale them to the clip only
                # when the floor came from the clip; a source floor is 1280x720
                # and testing 854x480 feet against it puts every player in the
                # top-left corner of the court -- which read as the mask having
                # got tighter, and cost 0.872 -> 0.551 before it was found.
                if source_regions:
                    boxes = np.array([[b[2], b[3], b[4], b[5]]
                                      for b in people], dtype=float)
                else:
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
                            # `image` is always the CLIP frame, so a source-
                            # coordinate box has to come back down for it.
                            crop_box = ([box[0] * scale_x, box[1] * scale_y,
                                         box[2] * scale_x, box[3] * scale_y]
                                        if source_regions else box)
                            colour = torso_lab(image, crop_box)
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
    cache["mask_from_source"] = from_source and broadcast_video is not None
    if broadcast_video is not None:
        broadcast_video.release()
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
    parser.add_argument("--from-clips", action="store_true",
                        help="compute the floor from the published 854x480 "
                             "CLIP rather than the source broadcast. Cheaper "
                             "and WRONG: measured on Finals G7 with the "
                             "setting its own pipeline uses, the clips score "
                             "0.836 kept ball carrier against the source's "
                             "0.872. Only for reproducing an old run.")
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
    print(f"  floor from the {'published clips' if args.from_clips else 'SOURCE broadcast'}")
    got = remask(args.game, float(share), out, limit=args.limit,
                 kit_max_lab=None if gate is None else float(gate),
                 court_every=args.court_every, fill_holes=bool(fill),
                 from_source=not args.from_clips)
    print(f"  {got['frames']} frames remasked, {got['changed']} changed, "
          f"{got['clips_missing']} clips not on disk")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
