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
    HOLD_GATE,
    court_region,
    stands_on_court,
    to_box,
)
from courtvision.games import get, registry  # noqa: E402
from courtvision.kits import KitModel, sample_clip, torso_lab  # noqa: E402
from courtvision.stats import iou, wilson  # noqa: E402

#: Erosions to try, as a share of frame height. 0.0625 is the shipped 45 px on
#: a 720p broadcast.
SHARES = [0.0, 0.015, 0.03, 0.045, 0.0625]
#: How far a torso may sit from the nearest kit or the officials and still be
#: somebody who belongs on the court. `None` is the filter switched off.
#: `candidates.MAX_KIT_DISTANCE_LAB` is 26.0 and is where the middle value
#: comes from.
KIT_DISTANCES = [None, 40.0, 26.0, 18.0]
#: The over-keeping rate a mask must hold. Below this it is admitting the crowd.
OVER_FLOOR = 0.95
#: More than this many people on a court is impossible.
IMPOSSIBLE_ABOVE = 13
#: Two boxes overlapping this much are one person seen by two detectors.
SAME_PERSON_IOU = 0.5
#: One detection row in this many is sampled.
ROW_STRIDE = 30


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


def fit_kits(broadcast, cache, names, source) -> KitModel | None:
    """Kit centres for this broadcast, from a sample of its own clips."""
    colours = []
    for name in names[0::2][:80]:
        path = ROOT / broadcast.clip_dir / name
        if not path.exists():
            continue
        for _row, _boxes, sampled in sample_clip(path, cache["clips"][name],
                                                 source, want=3):
            colours.extend(c for c in sampled if c is not None)
    return KitModel.fit(colours)


def measure(key: str, *, frames_wanted: int) -> dict:
    broadcast = get(key)
    cache = json.loads((ROOT / broadcast.clip_detections).read_text())
    source = cache.get("source_size") or [1280, 720]
    model = fit_kits(broadcast, cache, sorted(cache["clips"]), source)
    # Split by CLIP, not by frame: frames within a clip are the same camera on
    # the same possession, so an even/odd frame split would put near-duplicates
    # on both sides and the report half would confirm whatever the fit half
    # chose. This is the split discipline every other arm in this project uses,
    # and its absence here is what let a 250-frame fit propose a setting that
    # cost twelve points over the whole broadcast.
    carrier = {"fit": defaultdict(lambda: [0, 0]), "report": defaultdict(lambda: [0, 0])}
    over = {"fit": defaultdict(lambda: [0, 0]), "report": defaultdict(lambda: [0, 0])}
    seen = 0
    for index, name in enumerate(sorted(cache["clips"])):
        half = "fit" if index % 2 == 0 else "report"
        if seen >= frames_wanted:
            break
        path = ROOT / broadcast.clip_dir / name
        if not path.exists():
            continue
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            continue
        scale_x = capture.get(cv2.CAP_PROP_FRAME_WIDTH) / source[0]
        scale_y = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / source[1]
        try:
            for row in cache["clips"][name][::ROW_STRIDE]:
                if seen >= frames_wanted:
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["f"]))
                ok, image = capture.read()
                if not ok:
                    continue
                people = [[b[2] * scale_x, b[3] * scale_y,
                           b[4] * scale_x, b[5] * scale_y]
                          for b in row["d"] if b[0] in ("p", "h")]
                if not people:
                    continue
                held = None
                balls = [b for b in row["d"] if b[0] == "b"]
                if balls:
                    ball = max(balls, key=lambda b: b[1])
                    centre = ((ball[2] + ball[4]) / 2.0 * scale_x,
                              (ball[3] + ball[5]) / 2.0 * scale_y)
                    nearest = min(range(len(people)),
                                  key=lambda i: to_box(centre, people[i]))
                    height = people[nearest][3] - people[nearest][1]
                    if height > 0 and to_box(centre, people[nearest]) <= HOLD_GATE * height:
                        held = nearest
                seen += 1
                boxes = np.array(people, dtype=float)
                colours = ([torso_lab(image, box) for box in people]
                           if model is not None else [None] * len(people))
                for share in SHARES:
                    region = court_region(image, erode_px=None, erode_share=share)
                    on_floor = (stands_on_court(region, boxes)
                                if region is not None
                                else np.zeros(len(people), dtype=bool))
                    for max_lab in KIT_DISTANCES:
                        keep = list(on_floor)
                        if max_lab is not None and model is not None:
                            keep = [k and model.belongs_on_court(colours[i], max_lab)
                                    for i, k in enumerate(keep)]
                        count = distinct([people[i] for i in range(len(people))
                                          if keep[i]])
                        setting = (share, max_lab)
                        over[half][setting][0 if count <= IMPOSSIBLE_ABOVE else 1] += 1
                        if held is not None:
                            carrier[half][setting][0 if keep[held] else 1] += 1
        finally:
            capture.release()
    rows = {"fit": {}, "report": {}}
    for half in ("fit", "report"):
        for share in SHARES:
            for max_lab in KIT_DISTANCES:
                setting = (share, max_lab)
                kept, dropped = carrier[half][setting]
                fine, bad = over[half][setting]
                rows[half][setting] = {
                    "carrier_kept": (kept / (kept + dropped)
                                     if kept + dropped else math.nan),
                    "carrier_n": kept + dropped,
                    "over_ok": fine / (fine + bad) if fine + bad else math.nan,
                    "over_n": fine + bad,
                }
    return {"game": key, "label": broadcast.label, "frames": seen, "shares": rows,
            "kits_fitted": model is not None}


def choose(rows: dict) -> float | None:
    """The declared rule, applied. None when nothing clears the floor."""
    allowed = [s for s, r in rows.items()
               if r["over_n"] and r["over_ok"] >= OVER_FLOOR]
    if not allowed:
        return None
    # Ties go to the larger erosion and then to the tighter kit gate: both
    # exclusions cost more downstream when they are too loose than here.
    return max(allowed, key=lambda s: (rows[s]["carrier_kept"], s[0],
                                       -(s[1] or 1e9)))


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
    parser.add_argument("--gate", type=float, action="append", default=None,
                        help="kit gate to try, repeatable. Pass a negative "
                             "number for 'no gate'.")
    parser.add_argument("--out", default=str(COURT_ERODE_FILE))
    args = parser.parse_args()

    global SHARES, KIT_DISTANCES
    if args.share:
        SHARES = sorted(args.share)
    if args.gate:
        KIT_DISTANCES = [None if g < 0 else g for g in args.gate]
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
                       else {"erode_share": pick[0], "kit_max_lab": pick[1]})
        print()
        print(f"  {got['label']}  ({key}, {got['frames']} frames)")
        print(f"    {'erode/height':<13} {'kit gate':>9} "
              f"{'carrier (fit)':>13} {'<=13 (fit)':>11} "
              f"{'carrier (rep)':>13} {'<=13 (rep)':>11}")
        for setting in got["shares"]["fit"]:
            fit_row = got["shares"]["fit"][setting]
            rep_row = got["shares"]["report"][setting]
            mark = "  <- chosen" if setting == pick else ""
            share, max_lab = setting
            gate = "off" if max_lab is None else f"{max_lab:.0f}"
            print(f"    {share:<13.4f} {gate:>9} {fit_row['carrier_kept']:>13.3f} "
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
