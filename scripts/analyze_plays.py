"""Run the whole v3 chain on real footage: pixels -> court -> formation -> plays.

Every piece of v3 was built and tested on its own. This composes them, which is
the only way to find out whether they work together on a real broadcast.

The camera pans, so a single homography does not hold across a clip. Frame one
pays for the full global search; after that the camera has barely moved, so each
frame is searched inside tight bounds around the previous solution. That is
about five times faster and far more reliable than starting cold every time.

Nothing here is a gate. It reports what the chain produces and how much of it
is trustworthy, because a homography that scores poorly puts players in
plausible-looking but wrong places and everything downstream inherits that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from courtvision.config import Config
from courtvision.court import COURT_WIDTH, HALF_COURT_LENGTH
from courtvision.court_lines import DEFAULT_CAMERA_BOUNDS, search_registration
from courtvision.detection import load_pipeline_detector
from courtvision.device import resolve_device
from courtvision.extraction import extract_frames
from courtvision.formation import classify_formation
from courtvision.possession import possession_timeline
from courtvision.team_assignment import assign_teams, collect_samples
from courtvision.plays import (detect_off_ball_screens, detect_screens,
                               detect_sets)
from courtvision.tracking import PlayerTracker
from courtvision.types import HANDLER, PLAYER, RIM, Frame

CLIP = Path("data/raw_clips/sample.mp4")
CHECKPOINT = Path("checkpoints/detector.pt")
MIN_SCORE = 0.30          # below this the registration is not worth using


def bounds_around(params: np.ndarray) -> list[tuple[float, float]]:
    """Tight search bounds around a previous solution, for the next frame."""
    spread = np.array([6.0, 6.0, 4.0, 5.0, 5.0, 150.0])
    lows = params - spread
    highs = params + spread
    return [(float(a), float(b)) for a, b in zip(lows, highs)]


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 end to end on a clip")
    parser.add_argument("--frames", type=int, default=8,
                        help="how many sampled frames to register")
    parser.add_argument("--stride", type=int, default=4,
                        help="sample every Nth extracted frame")
    args = parser.parse_args()

    if not CLIP.exists() or not CHECKPOINT.exists():
        print(f"FAIL — need {CLIP} and {CHECKPOINT}")
        return 1

    config = Config()
    detector = load_pipeline_detector(str(CHECKPOINT), resolve_device(),
                                      config.detector_conf, config.ball_conf)
    tracker = PlayerTracker()

    print("detecting and tracking...")
    frames: list[Frame] = []
    images: list[np.ndarray] = []
    for index, time_s, image in extract_frames(str(CLIP), config.target_fps):
        frames.append(Frame(index, time_s,
                            tuple(tracker.update(detector.detect(image)))))
        images.append(image)
    print(f"  {len(frames)} frames")

    # Split the teams. Formation and screens are claims about ONE side: five
    # offensive players make a shape, and a screen is set by a TEAMMATE. Fed
    # both teams, formation never matches — ten players are not a five-player
    # shape — and spacing collapses, because a defender guards at three to six
    # feet and drags every nearest-neighbour distance down with him. Screens
    # are worse than useless: a defender closing out on the ball handler
    # converges exactly like a screener and gets reported as one.
    teams = assign_teams(collect_samples(images, frames))
    print(f"  teams assigned for {len(teams)} tracks")

    # Take the handler from the SMOOTHED possession timeline, not from whichever
    # frames the detector happened to fire its handler class on. The raw label
    # is missing in about a third of frames, and every one of those is a frame
    # where the play layer can claim nothing: no handler means no offence to
    # identify, so no formation and no screen. The timeline carries possession
    # across those gaps, which is what it was built to do — and it distinguishes
    # a ball merely unseen from a ball visibly loose, so it does not carry
    # possession through a pass.
    timeline = possession_timeline(frames, config)
    raw_hits = sum(1 for f in frames if f.handler() is not None)
    print(f"  handler present in {raw_hits}/{len(frames)} frames raw, "
          f"{sum(1 for h in timeline if h is not None)}/{len(timeline)} smoothed")

    chosen = list(range(0, len(frames), args.stride))[: args.frames]
    print(f"\nregistering {len(chosen)} frames "
          f"(first is a full global search, the rest are local)")

    positions: list[dict[int, tuple[float, float]]] = []
    shapes: list[str | None] = []
    handlers: list[int | None] = []
    times: list[float] = []
    scores: list[float] = []
    previous: np.ndarray | None = None

    for step, index in enumerate(chosen):
        image, frame = images[index], frames[index]
        bounds = None if previous is None else bounds_around(previous)
        rims = [k for k in frame.tracks if k.label == RIM]
        rim_px = None
        if rims:
            best = max(rims, key=lambda k: k.conf)
            rim_px = ((best.box.x1 + best.box.x2) / 2,
                      (best.box.y1 + best.box.y2) / 2)
        # Mask players out of the line detector: they are dark against the wood
        # exactly like the paint is. frame.tracks may be non-empty while the
        # filtered list is not, so test the filtered list.
        people = [[k.box.x1, k.box.y1, k.box.x2, k.box.y2]
                  for k in frame.tracks if k.label in (PLAYER, HANDLER)]
        boxes = np.array(people) if people else None
        matrix, score, params = _search(image, bounds, rim_px, boxes)
        scores.append(score)
        if matrix is None or score < MIN_SCORE:
            print(f"  t={frame.time_s:>5.2f}s  score {score:.3f}  SKIPPED "
                  f"(below {MIN_SCORE})")
            continue    # positions/shapes/handlers/times all skip together
        previous = params

        players = [t for t in frame.tracks if t.label in (PLAYER, HANDLER)]
        feet = np.array([[(t.box.x1 + t.box.x2) / 2, t.box.y2] for t in players])
        if len(feet) == 0:
            continue
        homogeneous = np.hstack([feet, np.ones((len(feet), 1))])
        projected = homogeneous @ matrix.T
        court = projected[:, :2] / projected[:, 2:3]

        on = ((court[:, 0] >= -5) & (court[:, 0] <= COURT_WIDTH + 5)
              & (court[:, 1] >= -5) & (court[:, 1] <= HALF_COURT_LENGTH + 5)
              & ~np.isnan(court).any(axis=1))
        frame_positions = {t.track_id: (float(c[0]), float(c[1]))
                           for t, c, ok in zip(players, court, on) if ok}
        handler_id = timeline[frame.index]

        # Keep only the side with the ball. Without a handler there is no
        # offence to speak of, so nothing is claimed for that frame.
        offense_team = teams.get(handler_id) if handler_id is not None else None
        if offense_team is None:
            offense = {}
        else:
            offense = {tid: xy for tid, xy in frame_positions.items()
                       if teams.get(tid) == offense_team}

        positions.append(offense)
        handlers.append(handler_id)
        times.append(frame.time_s)

        formation = (classify_formation(np.array(list(offense.values())))
                     if offense else None)
        shapes.append(formation.name if formation else None)
        if formation is None:
            print(f"  t={frame.time_s:>5.2f}s  score {score:.3f}  "
                  f"{on.sum()}/{len(players)} on court  no ball handler, "
                  f"nothing claimed")
        else:
            print(f"  t={frame.time_s:>5.2f}s  score {score:.3f}  "
                  f"{on.sum()}/{len(players)} on court  "
                  f"{len(offense)} on offence  "
                  f"spacing {formation.spacing_ft:>4.1f}ft  {formation}")

    if not positions:
        print("\nno frame registered well enough to use")
        return 1

    print(f"\nregistration score: median {np.median(scores):.3f}, "
          f"best {max(scores):.3f}, {sum(s >= MIN_SCORE for s in scores)}"
          f"/{len(scores)} usable")

    # Cache the court positions: registration costs ~8 s a frame, and every
    # question about thresholds afterwards should be answerable without paying
    # that again.
    cache = Path("outputs/play_positions.json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {"times": times, "handlers": handlers, "shapes": shapes,
         "positions": [{str(k): v for k, v in f.items()} for f in positions]},
        indent=1))
    print(f"\npositions cached to {cache}")

    screens = detect_screens(positions, handlers, times)
    off_ball = detect_off_ball_screens(positions, handlers, times)
    print(f"\nscreen actions: {len(screens)} on-ball, {len(off_ball)} off-ball")
    for play in screens + off_ball:
        print(f"  {play}")
    if not screens and not off_ball:
        print("  none detected in this window")

    sets = detect_sets(positions, handlers, times, shapes)
    print(f"\nnamed sets: {len(sets)}")
    for play in sets:
        print(f"  {play}")
    if not sets:
        print("  none — these are compositions of screen actions, so they need "
              "the\n  primitives above to fire first")

    print("\nRead this as a demonstration, not a gate. Formation and play output "
          "are only\nas good as the registration underneath them.")
    return 0


def _search(image, bounds, rim_px=None, exclude_boxes=None):
    """Register a frame, returning the matrix, its score, and the parameters.

    The parameters matter: they seed the next frame's bounds. An earlier version
    tried to recover them by intercepting the objective's last call, which
    returns whatever the optimiser evaluated last rather than its best point.
    """
    from courtvision.court_lines import homography_from_camera, search_camera

    params, score = search_camera(
        image, seed=0, max_iterations=60 if bounds else 250, bounds=bounds,
        rim_px=rim_px, exclude_boxes=exclude_boxes)
    if params is None:
        return None, score, None
    matrix = homography_from_camera(params, image.shape[:2])
    return np.linalg.inv(matrix), score, params


if __name__ == "__main__":
    sys.exit(main())
