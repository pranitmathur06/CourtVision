"""Join the event stream to court positions: the layer a film-study tool sits on.

Two halves of this project now clear their bars separately. The scoreboard says
WHAT happened and when (typed scoring at F1 0.854-0.955). The painted-key
registration says WHERE everyone was (85.7% of court frames, 94.6% of players
landing on the floor). Neither is useful alone for coaching: an event with no
positions cannot explain itself, and positions with no events are a soup.

This joins them into one record per event, carrying the court coordinates of
everyone on the floor at that moment. That is the substrate a question like
"why was that shot open" is answered from -- and it is deliberately just data,
because the honest thing to put under a natural-language interface is a model
whose numbers have been measured, not a narrator over guesses.

What it does NOT do, and must not pretend to: identify players. Jersey OCR does
not exist here yet, so positions are anonymous and the model can say "the
nearest defender was 4.2 ft away" but never "Haliburton was open".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from courtvision.court import BASKET, COURT_WIDTH  # noqa: E402
from courtvision.court_key import (detect_paint_hue,  # noqa: E402
                                   key_homography, key_matches_rim,
                                   key_quad)
from courtvision.court_tracking import has_court  # noqa: E402

# How far either side of an event to gather positions. A shot's context is the
# moment of release, not the moment the scorer logged it -- the play-by-play
# clock trails the action, which is what put every "release" under the basket
# earlier in this project.
CONTEXT_S = 2.0
# Offsets tried around an event, nearest-in-time first: the release precedes
# the logged time, so earlier moments are preferred over later ones.
OFFSETS_S = (-2.0, -1.0, -3.0, -0.5, -4.0, 0.5, -5.0, 1.5)
COURT_LENGTH_FT = 94.0


def load_rims(path: str) -> dict[float, tuple[float, float]]:
    rims: dict[float, tuple[float, float]] = {}
    for row in json.loads(Path(path).read_text())["frames"]:
        if row.get("rim"):
            best = max(row["rim"], key=lambda r: r[2])
            rims[round(row["t"], 1)] = (best[0], best[1])
    return rims


def rim_at(rims, seconds: float):
    for probe in (round(seconds, 1), round(seconds + 0.1, 1),
                  round(seconds - 0.1, 1)):
        if probe in rims:
            return rims[probe]
    return None


def court_positions(image, rim_px, boxes, paint_hue=None,
                    gate: bool = True) -> np.ndarray | None:
    """Court coordinates of every detected person's feet, or None.

    `gate` applies `key_matches_rim`. It is a real trade, and an expensive one:

        gate            per-frame coverage   p50 error   within 3 ft
        none                       92.3%       2.84 ft       50.7%
        <= 220 px                  18.5%       1.72 ft       79.1%

    This function called `key_homography` bare, so every position it has
    produced came from the ungated row while the 1.72 ft figure was quoted
    downstream as though it applied.

    What the gate costs at EVENT level, measured on this script's own 30
    events rather than assumed: **30/30 with context ungated, 14/30 gated**.
    The eight offsets do not rescue it, because they span 6.5 s of one shot
    and the gate rejects on camera framing -- when the framing is wrong it is
    wrong for all eight. An earlier commit had already measured 51.4% of shots
    and that number was not carried forward.

    And 1.72 ft is a proxy, not established truth. The gate variable (key
    centre to rim, in pixels) is mechanically coupled to the metric that
    scored it (projected rim against BASKET, in feet), so the gate largely
    selects on its own evaluation and says nothing about error at the far arc.
    Round 26 measured gate-passing registrations against each other and found
    them disagreeing by 5.8 ft, where 1.72 ft registrations would disagree by
    about 2.4. Treat this as "rejects obviously-wrong keys", not "1.72 ft".

    `paint_hue` is the arena's own key colour. Left None, the hardcoded blue
    range runs; on the one arena in four that paints its key at hue 174 that
    is 10% of court frames registered instead of 87% with calibration.
    """
    if not has_court(image):
        return None
    if gate:
        quad = key_quad(image, paint_hue=paint_hue)
        if quad is None or not key_matches_rim(quad, rim_px):
            return None
    matrix = key_homography(image, rim_px, paint_hue=paint_hue)
    if matrix is None or boxes is None or len(boxes) == 0:
        return None
    feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
    homogeneous = np.hstack([feet, np.ones((len(feet), 1))])
    projected = homogeneous @ matrix.T
    valid = np.abs(projected[:, 2]) > 1e-9
    if not valid.any():
        return None
    return projected[valid, :2] / projected[valid, 2:3]


def on_court(points: np.ndarray, margin: float = 8.0) -> np.ndarray:
    """Drop anyone outside the floor: benches, officials, photographers."""
    keep = ((points[:, 0] >= -margin) & (points[:, 0] <= COURT_WIDTH + margin)
            & (points[:, 1] >= -margin)
            & (points[:, 1] <= COURT_LENGTH_FT + margin))
    return points[keep]


def describe(points: np.ndarray) -> dict:
    """The measurable context of a moment. No player identities -- see module docstring."""
    if points is None or len(points) == 0:
        return {}
    floor = on_court(points)
    if len(floor) == 0:
        return {}
    distances = np.hypot(floor[:, 0] - BASKET[0], floor[:, 1] - BASKET[1])
    order = np.argsort(distances)
    nearest = floor[order[0]]
    others = floor[order[1:]]
    record = {
        "people_on_floor": int(len(floor)),
        "closest_to_basket_ft": round(float(distances[order[0]]), 1),
        "closest_position": [round(float(nearest[0]), 1),
                             round(float(nearest[1]), 1)],
        "spread_ft": round(float(np.hypot(floor[:, 0].std(),
                                          floor[:, 1].std())), 1),
    }
    if len(others):
        gaps = np.hypot(others[:, 0] - nearest[0], others[:, 1] - nearest[1])
        record["nearest_other_ft"] = round(float(gaps.min()), 1)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/raw_clips/fullgame.mp4")
    parser.add_argument("--timeline", required=True,
                        help="event timeline from run_broadcast.py")
    parser.add_argument("--rims", required=True, help="cached rim detections")
    parser.add_argument("--clock", required=True,
                        help="per-frame clock readings, to map game time to video")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--out", default="outputs/game_model.json")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    events = json.loads(Path(args.timeline).read_text())["events"]
    rims = load_rims(args.rims)
    clock = [r for r in json.loads(Path(args.clock).read_text())
             if r.get("elapsed") is not None]
    video_t = np.array([r["t"] for r in clock])
    elapsed = np.array([r["elapsed"] for r in clock])

    def to_video(game_seconds: float):
        index = int(np.argmin(np.abs(elapsed - game_seconds)))
        return (float(video_t[index])
                if abs(elapsed[index] - game_seconds) <= 3.0 else None)

    scoring = [e for e in events
               if e["action"] in ("three_point_make", "two_point_make",
                                  "free_throw")][: args.limit]
    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    model = YOLO("yolo11x.pt")

    # Live-play moments, for the hue calibration below. Every scoring event has
    # a video time, and a few seconds before one is basketball by construction.
    shot_times = [t for t in (to_video(e["elapsed_s"]) for e in scoring)
                  if t is not None and t > 5.0]

    # Calibrate the key colour from this arena's own footage, sampling LIVE
    # PLAY -- not the opening minutes.
    #
    # A first version swept 60-600 s. The game runs 540-7209 s, so ten of
    # twelve probes were pre-game, and of the five frames that passed
    # `has_court` three were the anthem line-up, a player introduction and a
    # coach close-up: skin and warm-ups fall inside the "wood" hue range, which
    # is the failure `court_region` already documents. It reached the right
    # answer by luck. Aligned shot times are known live play, which is the same
    # fix adopted elsewhere in this project for the same problem.
    sampled = []
    if shot_times:
        for probe in shot_times[:: max(1, len(shot_times) // 16)][:16]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int((probe - 3.0) * fps))
            ok, image = capture.read()
            if ok and has_court(image):
                sampled.append(image)
    candidate = detect_paint_hue(sampled) if sampled else None

    # A wrong calibration is strictly worse than the default, and silent: if
    # the dominant hue lands on a courtside advertising board, `key_quad`
    # returns None on every frame of the game while this still prints success.
    # So the calibrated range only replaces the default if it actually finds
    # more keys on the frames we already have.
    paint_hue = None
    if candidate and sampled:
        with_default = sum(key_quad(f) is not None for f in sampled)
        with_candidate = sum(key_quad(f, paint_hue=candidate) is not None
                             for f in sampled)
        if with_candidate > with_default:
            paint_hue = candidate
        print(f"  paint hue {candidate} finds {with_candidate}/{len(sampled)} "
              f"keys against {with_default}/{len(sampled)} for the default "
              f"-> {'using it' if paint_hue else 'keeping the default'}")
    else:
        print(f"  paint hue not calibrated ({len(sampled)} live frames); "
              f"keeping the default range")

    records = []
    described = 0
    for event in scoring:
        seconds = to_video(event["elapsed_s"])
        if seconds is None:
            continue
        # One frame is a bad bet: an event's own timestamp often lands on a
        # replay or a close-up, which is why sampling only (t - 2s) registered
        # 57% of events while 85.7% of court frames register overall. Try a
        # spread of nearby moments and keep the first that resolves.
        context = {}
        for offset in OFFSETS_S:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int((seconds + offset) * fps))
            ok, image = capture.read()
            if not ok:
                continue
            detected = model(image, verbose=False, conf=0.35, classes=[0])[0]
            boxes = (detected.boxes.xyxy.cpu().numpy()
                     if len(detected.boxes) else None)
            points = court_positions(image, rim_at(rims, seconds + offset),
                                     boxes, paint_hue=paint_hue)
            if points is None:
                continue
            context = describe(points)
            if context:
                break
        if context:
            described += 1
        records.append({**event, "video_s": round(seconds, 1),
                        "context": context})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"events": records}, indent=1))
    print(f"  {len(records)} scoring events, {described} with court context "
          f"({described / max(len(records), 1):.0%})")
    print(f"  -> {out}")
    for record in records[:6]:
        context = record["context"]
        if not context:
            print(f"    {record['elapsed_s']:7.1f}s {record['action']:<18} "
                  f"no registration")
            continue
        print(f"    {record['elapsed_s']:7.1f}s {record['action']:<18} "
              f"{context['people_on_floor']:2d} on floor | "
              f"closest to rim {context['closest_to_basket_ft']:5.1f} ft | "
              f"nearest other {context.get('nearest_other_ft', float('nan')):4.1f} ft")
    return 0


if __name__ == "__main__":
    sys.exit(main())
