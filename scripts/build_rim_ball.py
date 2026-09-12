"""Rim and ball per frame, from tracked camera poses and cached detections.

No video is read here. `track_camera.py` has already put a pose on every frame
it could, and `cache_detections.py` has already run the detector, so this pass
is a join -- which means the rules below can be changed and re-measured in
seconds instead of hours.

THE RIM prefers the DETECTOR where it is confident, and falls back to the pair
of baskets projected through the pose.

That order is the opposite of the obvious one, and it is measured. The
projection agrees with a confident detector box within one rim width on 90.6%
of grid frames; on the 12 that disagree, all 8 inspected by eye had the
detector on the rim and the projection about one rim width ABOVE it, on the
backboard. They are ANCHORS, not drifted hops -- several at a snap cost of
0.0 -- so this is not tracking error but the known weakness of extrapolating a
point 10 ft up from a homography fitted to the FLOOR: the floor lines can be
matched well by a pose whose tilt is slightly wrong, and the error shows up
only away from the floor. Fitting the rim's height to the detector confirms
the model is right on typical frames (the median error is smallest at exactly
10.0 ft) and that a subset is biased, so it is not a global calibration error
to be dialled out.

A detection is direct evidence and a projection is inference. The projection
earns its place on the frames where the detector sees nothing at all -- which
is most of them -- and on the second basket when the detector found only one.

THE BALL is chosen from the detector's candidates, and the choosing is the
whole problem: at the cache's 0.10 floor there are ~3 "ball" boxes a frame and
the right one is not reliably the most confident. Two tests, both of which
exist only because the camera centre is fixed:

- A RAY, not a pixel. With the centre fixed, every image point has a world
  direction, and camera motion cannot change it. This is what the earlier
  image-space tracker lacked: it scored smoothness in pixels, so a stationary
  object looked fast whenever the camera panned, and a shot -- the fastest the
  ball ever moves -- lost to a head that happened to be sitting still. In ray
  space a stationary object IS stationary.
- MOTION, off by default, and the reason is below. As a PREFERENCE, not a filter. A candidate that stands still
  between neighbouring frames, once ORB has removed the camera's own movement,
  is usually a head, a shoulder or a logo -- and motion is the only answer to
  the ray ambiguity, since a ball 15 ft up and a spectator behind it are on the
  same ray and only movement tells them apart.

  It was written on the premise that "the game ball is never still for a fifth
  of a second". THE PREMISE IS FALSE, and the measurement said so twice:

  - As a FILTER it threw away held balls. A player at the top of the key is
    very nearly static, and ball coverage fell from 86.4% of frames to 78.3%.
  - Rewritten as a PREFERENCE it still picked a moving distractor over a held
    ball: at 4412 s the ball sits plainly in a player's hands and the claim
    moved onto a defender; at 6912 s a free-throw shooter holds the ball and
    the claim moved onto a player across the floor.

  The common case is the opposite of the premise: the ball is HELD and still,
  while spectators and players move. Motion is still the only thing that can
  answer the ray ambiguity, but it cannot be used this way, so it is off by
  default and kept behind --use-motion with this note attached.
- FIXTURES. A direction that keeps producing ball boxes all game long is
  furniture: the spare ball on the rack at the scorer's table is a real ball,
  correctly detected, and never the game ball. Measured by eye in Round 65 it
  was ~23 of 40 sampled off-court detections. It holds one ray for the whole
  game; the game ball holds none.

A GEOMETRIC "is it in play" TEST WAS TRIED AND DOES NOT EXIST. The idea was
that a ball must lie inside the court footprint raised to a high arc, so a
candidate on a face in the tenth row could be thrown out. Two versions were
built and both rejected:

- "does the ray pass over the court at SOME height" rejected NOTHING. This
  camera sits 64 ft off the sideline and only 26 ft up, so its rays are
  shallow: sweeping a ray's height from 0 to 18 ft drags its landing point
  from x = -75 ft to x = -3 ft, and nearly every ray crosses the court box
  somewhere. A test that cannot fail is the exact fault Round 64 shipped.
- the silhouette of that box, projected into the picture, is degenerate --
  its far corners pass near the horizon and project to coordinates like
  (-17592, 520), so the hull swallows the frame.

The reason is not a bug to fix. Worked by hand for one real candidate: its ray
is between 10.2 and 17.3 ft above the floor for the whole time it is over the
court, which is an ordinary high arc. A ball 15 ft up over the far side and a
spectator's head behind it are ON THE SAME RAY. One frame does not carry the
depth to separate them; only motion does, and motion needs neighbouring
frames, which the 25 s evaluation grid does not have.

Among the survivors the candidate nearest the previous frame's choice wins if
it is within MAX_STEP_DEG, and otherwise the most confident does -- the ball
does leave and re-enter, so continuity is a preference and never a cage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: The cache floor. Selection, not detection, is meant to be the limit here.
MIN_CONF = 0.10
#: A detector rim this confident is believed ahead of the projection.
RIM_TRUST_CONF = 0.40
#: The dedicated rim detector's floor. Trained out, it finds 3 of 7 frames
#: that nothing else in the repo locates and raises NO false alarm on 8 frames
#: known to hold no rim. Every floor from 0.10 to 0.30 gives exactly that, so
#: this sits on a plateau rather than on a point fitted to those 15 frames;
#: an under-trained checkpoint at 0.20 reached 6 of 7 but also fired on a
#: referee's red patch, a water cooler and an orange shoe. It still goes BELOW
#: the four-class detector, and its false alarms are reported, never netted off.
RIM_SCALE_CONF = 0.25
#: A projected rim further than this from a believed detection is the OTHER
#: basket, and is kept; nearer than this it is the same rim seen twice.
SAME_RIM_WIDTHS = 2.0
#: A ball 47 ft away moving 60 ft/s sweeps ~73 deg/s; at 5 fps that is ~15 deg.
MAX_STEP_DEG = 15.0
#: A choice older than this tells you nothing about this frame -- a cut, a
#: replay or a gap in the poses has intervened.
MAX_CONTINUITY_GAP_S = 1.0
#: Ray directions are counted in cells this wide.
CELL_DEG = 0.6
#: A cell producing candidates on more than this share of posed frames is
#: furniture. Set from the SHAPE of the counts, which the labels never enter:
#: ranked by occupancy the cells on G7's grid run 34, 12, 9, 8, 7, ... and the
#: one holding 34 spans the whole game, 638 s to 7088 s. The threshold goes in
#: that gap. See scripts/show_ball_fixtures.py.
FIXTURE_SHARE = 0.15
#: ...and never fewer than this many frames, so a short pose run cannot
#: manufacture a fixture out of two detections.
FIXTURE_MIN_FRAMES = 8
#: The playing surface, in feet.
COURT_W, COURT_L = 50.0, 94.0
#: A ball in play is over the floor and under a high arc. The margin is
#: generous: a ball can be held out of bounds for a throw-in, and the pose
#: itself carries error, so this is meant to exclude the tenth row and not to
#: adjudicate the sideline.
PLAY_MARGIN_FT = 12.0
PLAY_MAX_HEIGHT_FT = 18.0
#: A direction this close to a basket is NEVER called furniture. The rim sits
#: still in the world exactly as the scorer's-table ball does, and the game
#: ball visits it on every attempt -- so on a dense pass the rule would learn
#: to throw away the ball at the rim, which is the one place Phase 2 needs it.
RIM_GUARD_DEG = 4.0


def ray_directions(params, centre, size, pixels, k1=None, k2=None):
    """World unit vectors for recorded-image pixels, under a pose's parameters.

    `params` is the rotation vector and log focal saved by `track_camera.py`.
    A pixel u maps to R^T K^-1 u, normalised: where the camera was looking when
    that pixel was lit. Independent of pan, tilt, roll and zoom by construction.
    """
    import cv2

    from courtvision.court_camera import undistort_points
    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
    if not len(pixels):
        return np.zeros((0, 3))
    pinhole = undistort_points(pixels, k1, size, k2)
    rotation = cv2.Rodrigues(np.asarray(params[:3], np.float64))[0]
    focal = float(np.exp(params[3]))
    width, height = size
    camera_rays = np.c_[(pinhole[:, 0] - width / 2.0) / focal,
                        (pinhole[:, 1] - height / 2.0) / focal,
                        np.ones(len(pinhole))]
    world = camera_rays @ rotation                      # R^T applied on the right
    norms = np.linalg.norm(world, axis=1, keepdims=True)
    return world / np.maximum(norms, 1e-12)


def cell_of(direction, cell_deg=CELL_DEG):
    """A ray direction as a coarse (azimuth, elevation) cell."""
    x, y, z = direction
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return (int(np.floor(azimuth / cell_deg)), int(np.floor(elevation / cell_deg)))


def angle_between(a, b):
    """Degrees between two unit vectors."""
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def rim_directions(centre):
    """Unit vectors from the camera centre to each basket."""
    from courtvision.court_camera import RIMS_3D
    out = []
    for point in RIMS_3D:
        d = np.asarray(point, np.float64) - np.asarray(centre, np.float64)
        out.append(d / np.linalg.norm(d))
    return out


def find_fixtures(cells_per_frame, posed_frames, share=FIXTURE_SHARE,
                  minimum=FIXTURE_MIN_FRAMES, protect=(), cell_deg=CELL_DEG,
                  guard_deg=RIM_GUARD_DEG):
    """Direction cells that keep producing candidates -- furniture, not the ball.

    Directions within `guard_deg` of a basket are protected however busy they
    are: a rim is as still in the world as the scorer's-table ball, and the
    game ball visits it on every attempt.
    """
    counts: dict[tuple[int, int], int] = {}
    for cells in cells_per_frame:
        for cell in set(cells):
            counts[cell] = counts.get(cell, 0) + 1
    floor = max(minimum, int(share * max(posed_frames, 1)))
    found = {cell for cell, n in counts.items() if n >= floor}
    if not protect:
        return found
    safe = set()
    for cell in found:
        azimuth = (cell[0] + 0.5) * cell_deg
        elevation = (cell[1] + 0.5) * cell_deg
        direction = np.array([np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)),
                              np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth)),
                              np.sin(np.radians(elevation))])
        if min(angle_between(direction, r) for r in protect) > guard_deg:
            safe.add(cell)
    return safe


def choose_balls(rows, fixtures, max_step_deg, use_continuity=True,
                 max_gap_s=MAX_CONTINUITY_GAP_S):
    """Per frame, (chosen | None, survivors), and how many candidates were dropped.

    Fixtures go first, then continuity decides among what is left, and
    confidence decides when continuity has nothing to say. Continuity is a
    preference and never a cage: the ball does leave the picture and come back,
    so a frame with no candidate near the last one still takes its best.
    """
    out, dropped, standing, previous, previous_t = [], 0, 0, None, None
    for row in rows:
        # Continuity only means something across a short gap. After a cut, a
        # replay or a stretch with no pose, the last choice says nothing about
        # this frame, and letting it pull would be worse than not looking.
        if previous_t is not None and row["t"] - previous_t > max_gap_s:
            previous, previous_t = None, None
        keep = []
        for i, candidate in enumerate(row["candidates"]):
            if row["cells"] and row["cells"][i] in fixtures:
                dropped += 1
                continue

            keep.append((i, candidate))
        moving = [(i, c) for i, c in keep if not c.get("still")]
        if moving and len(moving) < len(keep):
            standing += len(keep) - len(moving)
            keep = moving

        chosen = None
        if keep:
            if previous is not None and row["dirs"] is not None and use_continuity:
                near = [(angle_between(row["dirs"][i], previous), i, c) for i, c in keep]
                near = [n for n in near if n[0] <= max_step_deg]
                if near:
                    _, i, candidate = min(near, key=lambda n: n[0])
                    chosen = (i, candidate)
            if chosen is None:
                chosen = max(keep, key=lambda pair: pair[1]["conf"])
        if chosen is not None and row["dirs"] is not None:
            previous, previous_t = row["dirs"][chosen[0]], row["t"]
        elif chosen is None:
            previous, previous_t = None, None
        out.append((chosen, [c for _, c in keep]))
    return out, dropped, standing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", required=True, help="track_camera.py output")
    parser.add_argument("--detections", required=True)
    parser.add_argument("--rim-detections", default=None,
                        help="a dedicated rim detector's boxes (train_rim_scale.py). "
                             "Believed after the four-class detector and before the "
                             "projection: on main-camera frames it agrees with the "
                             "four-class boxes to 0.04 rim widths against the "
                             "projection's 0.13, and it has no upward bias.")
    parser.add_argument("--rim-conf", type=float, default=RIM_SCALE_CONF)
    parser.add_argument("--motion", default=None,
                        help="ball_motion.py output. Candidates that stand still "
                             "between neighbouring frames, once ORB has removed the "
                             "camera's own movement, are heads and furniture: the "
                             "game ball is never still for a fifth of a second.")
    parser.add_argument("--no-motion", action="store_true", default=True,
                        help="DEFAULT. Motion measured worse; see the module docstring")
    parser.add_argument("--use-motion", dest="no_motion", action="store_false",
                        help="turn the motion preference back on")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--min-conf", type=float, default=MIN_CONF)
    parser.add_argument("--max-step-deg", type=float, default=MAX_STEP_DEG)
    parser.add_argument("--no-fixtures", action="store_true",
                        help="control arm: keep every candidate")
    parser.add_argument("--no-continuity", action="store_true",
                        help="control arm: most confident survivor, always")
    parser.add_argument("--grid-only", action="store_true",
                        help="emit only the evaluation grid's frames")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from courtvision.court_camera import FixedCamera

    spec = json.load(open(args.camera))
    centre = np.array(spec["centre"], np.float64)
    size = tuple(spec["size"])
    k1, k2 = spec.get("k1", 0.0), spec.get("k2", 0.0)
    camera = FixedCamera(centre, size, floor=spec.get("floor"), k1=k1, k2=k2)

    poses = json.load(open(args.poses))
    cache = json.load(open(args.detections))
    by_time = {round(row["t"], 3): row for row in cache["frames"]}
    cache_times = np.array(sorted(by_time)) if by_time else np.array([])

    rim_extra, rim_extra_times = {}, np.array([])
    if args.rim_detections:
        found = json.load(open(args.rim_detections))
        rim_extra = {round(r["t"], 3): r for r in found["frames"]}
        rim_extra_times = np.array(sorted(rim_extra))

    def boxes_near(t, kind, floor):
        if not len(cache_times):
            return []
        j = int(np.argmin(np.abs(cache_times - t)))
        if abs(cache_times[j] - t) > 0.3:
            return []
        return [b for b in by_time[cache_times[j]]["boxes"]
                if b["cls"] == kind and b["conf"] >= floor]

    def centre_of(box):
        x1, y1, x2, y2 = box["xyxy"]
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    motion = {}
    if args.motion and not args.no_motion:
        found = json.load(open(args.motion))
        motion = {round(r["t"], 3): r["candidates"] for r in found["frames"]}

    def still_near(t, centre, tolerance=6.0):
        """Was a candidate at this point judged to be standing still?"""
        for c in motion.get(round(t, 3), ()):
            if abs(c["centre"][0] - centre[0]) <= tolerance \
                    and abs(c["centre"][1] - centre[1]) <= tolerance:
                return bool(c.get("still"))
        return False

    # Pass 1: candidates and their ray directions.
    rows = []
    posed = 0
    for row in poses["frames"]:
        t = row["t"]
        balls = boxes_near(t, "ball", args.min_conf)
        entry = {"t": t, "rims": row.get("rims") or [], "source": row.get("source"),
                 "candidates": [{"centre": centre_of(b), "conf": b["conf"],
                                 "still": still_near(t, centre_of(b))} for b in balls],
                 "cells": [], "dirs": None, "hull": None}
        if row.get("params") and balls:
            posed += 1
            directions = ray_directions(row["params"], centre, size,
                                        [c["centre"] for c in entry["candidates"]], k1, k2)
            entry["dirs"] = directions
            entry["cells"] = [cell_of(d) for d in directions]
        elif row.get("params"):
            posed += 1
        rows.append(entry)

    fixtures = set() if args.no_fixtures else \
        find_fixtures([r["cells"] for r in rows], posed,
                      protect=rim_directions(centre))

    # Pass 2: choose one ball per frame.
    decided, dropped, standing = choose_balls(
        rows, fixtures, args.max_step_deg,
        use_continuity=not args.no_continuity)

    frames = []
    for row, (chosen, survivors) in zip(rows, decided):
        detected_rims = [centre_of(b) for b in boxes_near(row["t"], "rim", 0.25)]
        trusted = boxes_near(row["t"], "rim", RIM_TRUST_CONF)
        if not trusted and len(rim_extra_times):
            j = int(np.argmin(np.abs(rim_extra_times - row["t"])))
            if abs(rim_extra_times[j] - row["t"]) <= 0.3:
                trusted = [b for b in rim_extra[rim_extra_times[j]]["boxes"]
                           if b["conf"] >= args.rim_conf]
        rim = [centre_of(b) for b in trusted]
        widths = [max(b["xyxy"][2] - b["xyxy"][0], 1.0) for b in trusted]
        for point in row["rims"]:
            # Keep a projected basket only where no believed detection already
            # stands: the other end of the floor, or a rim the detector missed.
            if all(np.hypot(point[0] - r[0], point[1] - r[1]) > SAME_RIM_WIDTHS * w
                   for r, w in zip(rim, widths)):
                rim.append(point)
        frames.append({"t": row["t"],
                       "rim": rim or detected_rims,
                       "ball": chosen[1]["centre"] if chosen else None,
                       "rim_source": ("detector+projected" if trusted and row["rims"]
                                      else "detector" if trusted
                                      else "projected" if row["rims"]
                                      else "detector-weak" if detected_rims else None),
                       # Every candidate the detector offered, kept so a
                       # labelling pass can measure the CEILING as well as the
                       # choice: a miss because no candidate existed and a miss
                       # because the wrong one was taken need different work.
                       "projected": row["rims"],
                       "detected": detected_rims,
                       "candidates": [c["centre"] for c in row["candidates"]],
                       "survivors": [c["centre"] for c in survivors],
                       "n_candidates": len(row["candidates"])})

    if args.grid_only:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_rim_and_ball import SAMPLE_EVERY_S, sample_times
        span = max(f["t"] for f in frames) if frames else 0.0
        wanted = sample_times(span + SAMPLE_EVERY_S, SAMPLE_EVERY_S)
        available = np.array([f["t"] for f in frames])
        picked, seen = [], set()
        for g in wanted:
            j = int(np.argmin(np.abs(available - g)))
            if abs(available[j] - g) <= 0.5 and j not in seen:
                seen.add(j)
                picked.append({**frames[j], "t": g})
        frames = picked

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"poses": args.poses, "detections": args.detections,
               "min_conf": args.min_conf, "fixtures": sorted(map(list, fixtures)),
               "frames": frames}, open(out, "w"))
    n = max(len(frames), 1)
    print(f"{len(frames)} frames; rim on {sum(1 for f in frames if f['rim'])} "
          f"({sum(1 for f in frames if f['rim']) / n:.1%}); "
          f"ball on {sum(1 for f in frames if f['ball'])} "
          f"({sum(1 for f in frames if f['ball']) / n:.1%}); "
          f"{len(fixtures)} fixture directions dropped {dropped} candidates, "
          f"{standing} more were passed over for standing still")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
