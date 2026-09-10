"""Measure the registration's absolute error against the league's own shot chart.

Every previous check on broadcast footage was internal. Consistency compares
two registrations of one instant, so a systematic error passes it. Line
agreement compares against court geometry, which is symmetric and forgiving.
Neither could see the +2.04 ft across-court bias that `check_registration_bias`
found.

The shot chart is outside the vision stack entirely: 157 court positions for
this game, recorded courtside, that have never seen a pixel of our video. At
each shot one of the players on screen is the shooter and the feed says where
he stood, so the offset that best reconciles the two is the registration's
absolute error.

The obvious way to fit that is circular. Minimising the distance from each
feed location to the NEAREST detected player rewards any offset that pushes
points into crowded parts of the floor, and would report a confident number
from pure player density. So the same fit is run on a deliberately WRONG
pairing -- each shot matched to another shot's frame. If the true pairing is
not clearly better, the result is density, not calibration, and this script
says so.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

SHOTS = Path("data/pbp/shotchart_0042400407.json")
EVENTS = Path("outputs/aligned_events.json")
#: The feed times a shot at its outcome; the release is a moment earlier.
RELEASE_LEAD_S = 0.6
GRID = np.arange(-6.0, 6.01, 0.25)
MATCH_FT = 6.0


def _court(loc_x: float, loc_y: float, far: bool) -> tuple[float, float]:
    """Shot-chart tenths of feet -> court feet, at whichever end is in play.

    The chart normalises every shot onto one half. Which physical end that is
    depends on the period and the team, so the end is taken from where the
    players actually are -- a decision a two-foot offset cannot flip.
    """
    x, y = 25.0 + loc_x / 10.0, 5.25 + loc_y / 10.0
    return (50.0 - x, 94.0 - y) if far else (x, y)


def _surname(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.split()[-1].lower())


def _pair(shots, events):
    """Match feed shots to aligned video events by surname, in time order."""
    used, pairs = set(), []
    for row in shots:
        want = _surname(row["PLAYER_NAME"])
        for n, event in enumerate(events):
            if n in used:
                continue
            if want and want in re.sub(r"[^a-z]", "", event["description"].lower()):
                pairs.append((row, event))
                used.add(n)
                break
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/raw_clips/fullgame.mp4")
    parser.add_argument("--weights",
                        default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--detector",
                        default="runs/detect/outputs/train/detector/weights/best.pt")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--cache", default="outputs/feed_calibration.json",
                        help="observations are expensive to gather (two models "
                             "per shot, seeking a 2 GB video); cached so the "
                             "analysis can be redone without re-reading it")
    parser.add_argument("--lead", type=float, default=RELEASE_LEAD_S)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    from courtvision.device import resolve_device

    if not SHOTS.exists():
        print(f"FAIL - no shot chart at {SHOTS}")
        return 1
    shots = json.load(SHOTS.open())
    events = [e for e in json.load(EVENTS.open())["events"]
              if "Shot" in (e.get("action") or "")]
    pairs = _pair(shots, events)
    print(f"{len(shots)} feed shots, {len(events)} aligned events, "
          f"{len(pairs)} paired by surname")

    # Independent check on the pairing: the play description states the shot's
    # distance in feet, and the chart's coordinates imply one.
    stated = []
    for row, event in pairs:
        m = re.search(r"(\d+)'", event["description"])
        if m:
            said = float(m.group(1))
            implied = np.hypot(row["LOC_X"], row["LOC_Y"]) / 10.0
            stated.append(abs(said - implied))
    if stated:
        print(f"  pairing check: |stated - implied| distance "
              f"p50 {np.median(stated):.1f} ft over {len(stated)} shots")

    model = YOLO(args.weights)
    detector = YOLO(args.detector)
    device = resolve_device()
    capture = cv2.VideoCapture(args.video)

    observations = []          # (feed_xy, players_xy) per usable shot
    for row, event in pairs[:args.limit]:
        at = event["video_s"] - args.lead
        if at < 60:
            continue
        capture.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        result = model.predict(frame, device=device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        xy = result.keypoints.xy[0].cpu().numpy()
        conf = result.keypoints.conf[0].cpu().numpy()
        seen = {i: tuple(xy[i]) for i in range(len(xy))
                if i in KEYPOINTS and conf[i] >= args.conf and (xy[i] > 0).all()}
        matrix, _ = homography_from_keypoints(seen)
        if matrix is None:
            continue
        boxes = detector.predict(frame, device=device, verbose=False)[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        keep = boxes.cls.cpu().numpy() == 0                      # players
        xyxy = boxes.xyxy.cpu().numpy()[keep]
        if len(xyxy) < 4:
            continue
        # A player's court position is where the feet meet the floor.
        feet = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)
        players = cv2.perspectiveTransform(
            feet.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(-1, 2)
        on_court = players[(players[:, 0] > -5) & (players[:, 0] < 55)
                           & (players[:, 1] > -5) & (players[:, 1] < 99)]
        if len(on_court) < 4:
            continue
        far = float(np.median(on_court[:, 1])) > 47.0
        observations.append((np.array(_court(row["LOC_X"], row["LOC_Y"], far)),
                             on_court))

    if args.cache:
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
        json.dump([{"feed": f.tolist(), "players": p.tolist()}
                   for f, p in observations], open(args.cache, "w"))
    print(f"  {len(observations)} shots with a registration and players")
    if len(observations) < 20:
        print("FAIL - too few to calibrate")
        return 1

    def score(offset, shuffled=False):
        """Median distance from each feed location to the nearest player."""
        distances = []
        for n, (feed, players) in enumerate(observations):
            others = observations[(n + 7) % len(observations)][1] if shuffled else players
            d = np.hypot(*(others - (feed + offset)).T)
            distances.append(d.min())
        return float(np.median(distances)), distances

    best, best_value = None, np.inf
    surface = {}
    for dx in GRID:
        for dy in GRID:
            value, _ = score(np.array([dx, dy]))
            surface[(dx, dy)] = value
            if value < best_value:
                best, best_value = np.array([dx, dy]), value
    zero, _ = score(np.array([0.0, 0.0]))

    shuffled_best = min(score(np.array([dx, dy]), shuffled=True)[0]
                        for dx in GRID[::2] for dy in GRID[::2])

    print(f"\n  no correction        median miss {zero:.2f} ft")
    print(f"  best offset          dx {best[0]:+.2f}  dy {best[1]:+.2f} ft"
          f"   -> median miss {best_value:.2f} ft")
    print(f"  SHUFFLED pairing     best achievable {shuffled_best:.2f} ft"
          f"   (if this is close to {best_value:.2f}, the fit is player density,"
          f" not calibration)")
    _, at_best = score(best)
    print(f"  shots within {MATCH_FT:.0f} ft of a player after correction: "
          f"{np.mean(np.array(at_best) < MATCH_FT):.0%}")

    # Is the residual a fixed offset, or does it move frame to frame? Take the
    # vector to the nearest player per shot: a systematic error clusters, a
    # per-frame one scatters. A global correction can only ever remove the
    # clustered part, which is why the best offset above helped so little.
    residuals = []
    for feed, players in observations:
        d = players - (feed + best)
        residuals.append(d[np.hypot(*d.T).argmin()])
    residuals = np.array(residuals)
    print(f"  residual after correction   dx {np.median(residuals[:,0]):+.2f} "
          f"+/- {residuals[:,0].std():.2f} ft   "
          f"dy {np.median(residuals[:,1]):+.2f} +/- {residuals[:,1].std():.2f} ft")
    print("  (a large spread means the error is per-frame, so no single "
          "offset can fix it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
