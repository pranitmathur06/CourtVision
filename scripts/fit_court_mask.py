#!/usr/bin/env python3
"""How hard to erode the court, chosen per broadcast with no labels at all.

THE MASK HAS TWO ERRORS AND ONE CONSTANT TRADES THEM. Erode too little and the
front row stands on the court; erode too much and a player is deleted before
any kernel is asked about him. The library's `COURT_ERODE_PX = 45` was swept
once, on one broadcast, against the first error only; `clip_detect_raw.py` then
shadowed it with its own `COURT_ERODE_PX = 15`, chosen against the second error
on 39 frames of each of two broadcasts counted by hand. So every cached mask
was built by a number that was never the documented one, and neither number had
a bound in either direction. The second error is the larger: Houston's mask
drops the man holding the ball on 55% of frames.

BOTH SIDES ARE MEASURABLE WITHOUT LABELS, which is what makes this fittable on
a broadcast nobody has touched:

    over-keeping    ten players and at most three officials, so more than
                    thirteen people kept is impossible
    under-keeping   whoever is holding the ball is playing, so a mask that
                    drops him is wrong

Either alone picks a degenerate mask -- keep nobody and never exceed thirteen,
keep everybody and never drop the carrier. Together they have an interior
answer.

THE RULE IS DECLARED HERE, BEFORE THE NUMBERS: take the erosion that keeps the
ball carrier most often, among those whose over-keeping stays at or above
`OVER_FLOOR`. Ties go to the larger erosion, because admitting the front row
costs more downstream than it costs here.

EROSION IS A SHARE OF FRAME HEIGHT, not pixels. 45 px is 6.25% of a 720p frame
and 4.2% of a 1080p one, so the shipped constant silently means two different
things on two broadcasts in this registry.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from courtvision.candidates import (  # noqa: E402
    COURT_ERODE_FILE,
    carrier_of,
    court_region,
    stands_on_court,
)
from courtvision.games import get, registry  # noqa: E402
from courtvision.broadcast import SourceReader, clip_starts  # noqa: E402
from courtvision.kits import KitModel, sample_broadcast, torso_lab  # noqa: E402
from courtvision.stats import iou, wilson  # noqa: E402

#: Erosions to try, as a share of frame height. 0.0625 is the shipped 45 px on
#: a 720p broadcast.
SHARES = [0.0, 0.015, 0.03, 0.045, 0.0625]
#: How far a torso may sit from the nearest kit or the officials and still be
#: somebody who belongs on the court. `None` is the filter switched off.
#: `candidates.MAX_KIT_DISTANCE_LAB` is 26.0 and is where the middle value
#: comes from.
KIT_DISTANCES = [None, 40.0, 26.0, 18.0]
#: Whether to take what the wood encloses. A red key needs it; a blue one is
#: already read by the colour rule, and filling is a larger mask that admits
#: more of the front row. Swept rather than assumed.
FILL_HOLES = [True, False]
#: The over-keeping rate a mask must hold. Below this it is admitting the crowd.
OVER_FLOOR = 0.95
#: More than this many people on a court is impossible.
IMPOSSIBLE_ABOVE = 13
#: Two boxes overlapping this much are one person seen by two detectors.
SAME_PERSON_IOU = 0.5
#: One detection row in this many starts a sampled group.
ROW_STRIDE = 30
#: How many consecutive rows each sampled floor is scored against -- the same
#: cadence `clip_detect_raw.py` applies, because the pipeline finds the floor
#: once and REUSES it while the camera pans. Scoring only the frame the floor
#: was found on measures a mask nothing ever applies, and it does so unevenly:
#: a tighter mask has less margin, so the same pan pushes more feet outside it
#: and the settings the fit likes most are the ones staleness damages most.
#: Measured on Finals G7, the same setting reads 0.852 scored fresh every frame
#: and 0.819 scored the way the pipeline uses it.
COURT_EVERY = 3


def distinct(boxes) -> int:
    """How many PEOPLE these boxes are. `p` and `h` draw the same players."""
    kept: list = []
    for box in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if any(iou(box, other) >= SAME_PERSON_IOU for other in kept):
            continue
        kept.append(box)
    return len(kept)


def carrier_index(row, people) -> int | None:
    """Which box is holding the ball, or None if nobody plainly is."""
    balls = [b for b in row["d"] if b[0] == "b"]
    if not balls or not people:
        return None
    ball = max(balls, key=lambda b: b[1])
    centre = ((ball[2] + ball[4]) / 2.0, (ball[3] + ball[5]) / 2.0)
    return centre, None


def fit_kits(broadcast, cache, names, reader, starts, step) -> KitModel | None:
    """Kit centres for this broadcast, from the BROADCAST's own torsos."""
    colours = []
    for name in names[0::2][:80]:
        if name not in starts:
            continue
        for _row, _boxes, sampled in sample_broadcast(
                reader, starts[name], cache["clips"][name], step, want=3):
            colours.extend(c for c in sampled if c is not None)
    return KitModel.fit(colours)


def _score_row(row, image, regions, model, carrier, over):
    """Score one detection row against an ALREADY-FOUND floor, per setting.

    Everything here is in SOURCE pixels -- the regions, the boxes and the
    `image` the torso colour is read from -- so nothing is scaled and the clips
    are not opened at all. A torso sampled off an 854x480 clip is a coarser
    colour than the detector saw, and the clips are not the broadcast.
    """
    people = [[b[2], b[3], b[4], b[5]]
              for b in row["d"] if b[0] in ("p", "h")]
    if not people:
        return False
    held = carrier_of(row, people)
    boxes = np.array(people, dtype=float)
    colours = ([torso_lab(image, box) for box in people]
               if model is not None else [None] * len(people))
    for (share, fill), region in regions.items():
        on_floor = (stands_on_court(region, boxes) if region is not None
                    else np.zeros(len(people), dtype=bool))
        for max_lab in KIT_DISTANCES:
            keep = list(on_floor)
            if max_lab is not None and model is not None:
                keep = [k and model.belongs_on_court(colours[i], max_lab)
                        for i, k in enumerate(keep)]
            count = distinct([people[i] for i in range(len(people)) if keep[i]])
            setting = (share, max_lab, fill)
            over[setting][0 if count <= IMPOSSIBLE_ABOVE else 1] += 1
            if held is not None:
                carrier[setting][0 if keep[held] else 1] += 1
    return True


def measure(key: str, *, frames_wanted: int) -> dict:
    """Score every setting ON THE BROADCAST, not on the published clips.

    THE FLOOR COMES FROM THE SOURCE VIDEO, read exactly the way
    `clip_detect_raw.py` reads it: seek with `CAP_PROP_POS_MSEC` to the clip's
    start and read forward, so row `position` is source frame
    `position * step` after the seek. Computing that index instead of copying
    the seek is wrong by up to 24 frames. Measured on Finals G7, a floor taken
    from the 854x480 clip scores 0.836 kept ball carrier where the same setting
    on the source scores 0.866, so a fit run on clips is choosing between
    settings by a number three points below the one that ships.

    The player boxes stay in SOURCE pixels, because that is what they are. Only
    the torso crop goes down to the clip, which is where the decoded image for
    it comes from.
    """
    broadcast = get(key)
    cache = json.loads((ROOT / broadcast.clip_detections).read_text())
    starts = clip_starts(ROOT / broadcast.clip_index)
    step = max(1, int(cache.get("step") or 2))
    reader = SourceReader(ROOT / broadcast.video)
    if not reader.ok:
        raise SystemExit(f"  could not open {broadcast.video}")
    try:
        model = fit_kits(broadcast, cache, sorted(cache["clips"]), reader,
                         starts, step)
    finally:
        reader.close()

    video = cv2.VideoCapture(str(ROOT / broadcast.video))
    if not video.isOpened():
        raise SystemExit(f"  could not open {broadcast.video}")

    # Split by CLIP, not by frame: frames within a clip are the same camera on
    # the same possession, so a frame split would put near-duplicates on both
    # sides and the report half would confirm whatever the fit half chose.
    carrier = {half: defaultdict(lambda: [0, 0]) for half in ("fit", "report")}
    over = {half: defaultdict(lambda: [0, 0]) for half in ("fit", "report")}
    seen = 0
    try:
        for index, name in enumerate(sorted(cache["clips"])):
            half = "fit" if index % 2 == 0 else "report"
            if seen >= frames_wanted:
                break
            if name not in starts:
                continue
            rows = cache["clips"][name]
            video.set(cv2.CAP_PROP_POS_MSEC, starts[name] * 1000.0)
            at = 0
            try:
                for start in range(0, len(rows), ROW_STRIDE):
                    if seen >= frames_wanted:
                        break
                    while at < start * step:
                        if not video.grab():
                            break
                        at += 1
                    ok, image = video.read()
                    if not ok:
                        break
                    at += 1
                    # ONE floor, scored against this row and the next few --
                    # what the pipeline does with it. Scoring only the frame it
                    # was found on measures a mask nothing ever applies.
                    regions = {(share, fill):
                               court_region(image, erode_px=None,
                                            erode_share=share, fill_holes=fill)
                               for share in SHARES for fill in FILL_HOLES}
                    for row in rows[start:start + COURT_EVERY]:
                        if seen >= frames_wanted:
                            break
                        if _score_row(row, image, regions, model,
                                      carrier[half], over[half]):
                            seen += 1
            finally:
                pass
    finally:
        video.release()

    rows_out = {"fit": {}, "report": {}}
    for half in ("fit", "report"):
        for share in SHARES:
            for max_lab in KIT_DISTANCES:
              for fill in FILL_HOLES:
                setting = (share, max_lab, fill)
                kept, dropped = carrier[half][setting]
                fine, bad = over[half][setting]
                rows_out[half][setting] = {
                    "carrier_kept": (kept / (kept + dropped)
                                     if kept + dropped else math.nan),
                    "carrier_n": kept + dropped,
                    "over_ok": fine / (fine + bad) if fine + bad else math.nan,
                    "over_n": fine + bad,
                }
    return {"game": key, "label": broadcast.label, "frames": seen,
            "shares": rows_out, "kits_fitted": model is not None,
            "court_every": COURT_EVERY, "from_source": True}


def choose(rows: dict) -> float | None:
    """The declared rule, applied. None when nothing clears the floor."""
    allowed = [s for s, r in rows.items()
               if r["over_n"] and r["over_ok"] >= OVER_FLOOR]
    if not allowed:
        return None
    # Ties go to the larger erosion and then to the tighter kit gate: both
    # exclusions cost more downstream when they are too loose than here.
    return max(allowed, key=lambda s: (rows[s]["carrier_kept"], s[0],
                                       -(s[1] or 1e9), not s[2]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--frames", type=int, default=200,
                        help="frames sampled per broadcast. 250 was NOT "
                             "enough: it over-estimated its own chosen "
                             "setting by 7 points on one broadcast and 12 on "
                             "another, and on the second the setting it chose "
                             "was a regression the sample could not see.")
    parser.add_argument("--share", type=float, action="append", default=None,
                        help="erosion to try, repeatable. Narrows the grid so "
                             "more FRAMES can be afforded, which is the axis "
                             "that was short.")
    parser.add_argument("--fill", type=int, action="append", default=None,
                        help="1 to take what the wood encloses, 0 not to. "
                             "Repeatable; both by default.")
    parser.add_argument("--gate", type=float, action="append", default=None,
                        help="kit gate to try, repeatable. Pass a negative "
                             "number for 'no gate'.")
    parser.add_argument("--out", default=str(COURT_ERODE_FILE))
    args = parser.parse_args()

    global SHARES, KIT_DISTANCES, FILL_HOLES
    if args.share:
        SHARES = sorted(args.share)
    if args.gate:
        KIT_DISTANCES = [None if g < 0 else g for g in args.gate]
    if args.fill:
        FILL_HOLES = [bool(f) for f in args.fill]
    keys = args.game or list(registry())
    chosen = {}
    print()
    print("  fit_court_mask.py -- two bounds from the sport, no labels")
    print(f"  keep the carrier as often as possible, while more than "
          f"{IMPOSSIBLE_ABOVE} kept stays under {1 - OVER_FLOOR:.2f}")
    for key in keys:
        got = measure(key, frames_wanted=args.frames)
        pick = choose(got["shares"]["fit"])
        chosen[key] = (None if pick is None
                       else {"erode_share": pick[0], "kit_max_lab": pick[1],
                             "fill_holes": pick[2]})
        print()
        print(f"  {got['label']}  ({key}, {got['frames']} frames)")
        print(f"    {'erode/height':<13} {'kit gate':>9} {'fill':>5} "
              f"{'carrier (fit)':>13} {'<=13 (fit)':>11} "
              f"{'carrier (rep)':>13} {'<=13 (rep)':>11}")
        for setting in got["shares"]["fit"]:
            fit_row = got["shares"]["fit"][setting]
            rep_row = got["shares"]["report"][setting]
            mark = "  <- chosen" if setting == pick else ""
            share, max_lab, fill = setting
            gate = "off" if max_lab is None else f"{max_lab:.0f}"
            print(f"    {share:<13.4f} {gate:>9} {str(fill):>5} "
                  f"{fit_row['carrier_kept']:>13.3f} "
                  f"{fit_row['over_ok']:>11.3f} {rep_row['carrier_kept']:>13.3f} "
                  f"{rep_row['over_ok']:>11.3f}{mark}")
        if pick is not None:
            fit_row = got["shares"]["fit"][pick]
            rep_row = got["shares"]["report"][pick]
            gap = fit_row["carrier_kept"] - rep_row["carrier_kept"]
            print(f"    the chosen setting reads {gap:+.3f} on the half it was "
                  f"chosen on. A large positive gap is the")
            print("    grid having picked this setting's sampling luck, and is "
                  "the whole reason for the split.")
        if pick is None:
            print("    nothing clears the over-keeping floor on this broadcast")
    Path(args.out).write_text(json.dumps(chosen, indent=2))
    print()
    print(f"  -> {args.out}")
    print("  clip_detect_raw.py reads this. The cached detections were built")
    print("  with the old constant and do not change until they are rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
