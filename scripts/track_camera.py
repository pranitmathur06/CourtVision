"""The camera's pose on every frame of a game, by anchoring rarely and tracking cheaply.

Registering each frame from its own paint costs 5-35 s and succeeds on about a
third of them, which is neither fast enough nor complete enough to put a rim on
every frame. But the camera is FIXED: one centre for the whole game, and each
frame only a pan, tilt, roll and zoom about it. So the pose can be carried from
frame to frame instead of re-derived:

- ANCHOR (expensive, rare): a full `register_frame` on the painted lines. Only
  attempted where the landmark model gives a start, because without one the
  search adds 10-25 s and almost always fails anyway; that gate is a speed
  choice and costs no coverage the tracker cannot recover by other means.
- TRACK (cheap, every frame): ORB from the ANCHOR FRAME ITSELF to this one,
  players masked out, which `court_tracking.pairwise_homography` already does
  at ~0.5 px a hop and REFUSES when the evidence is thin rather than guessing.
  Matching to the anchor rather than to the previous frame is the difference
  between one hop of error and N: chaining consecutive frames was measured at
  0.47 rim widths median and only 82% within a rim width, against 0.08 and 96%
  for the anchors it started from. A chained fallback was tried and REMOVED: measured
  against the detector's own rims it landed 1.27 rim widths out and 0 of 11
  such frames were within one, so it bought about a percent of coverage at no
  accuracy at all. Where the view has panned off the anchor, the answer is a
  new anchor, not a longer chain.
- SNAP (every frame): the carried homography is re-fitted to the four
  parameters the fixed camera allows. A chain of hops drifts in eight degrees
  of freedom; the camera only has four, so projecting back onto the model each
  step throws the drift away instead of accumulating it.
- ALARM: where the detector independently sees a rim and the carried pose puts
  the rim more than DRIFT_WIDTHS rim widths away, the pose is stale -- the
  chain has slipped or the shot has cut -- and an anchor is forced. This is the
  one check that can catch a confidently wrong pose, and it uses evidence the
  tracker itself never touches.

Everything is done in PINHOLE pixels: frames are undistorted once up front, so
the hops, the poses and the camera model all live in the same coordinates.

Output: one row per sampled frame with the pose, how it was obtained, and the
projected rims -- enough for `eval_rim_and_ball.py` to score and for a later
pass to re-project anything else without walking the video again.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

#: Sampling step. 5 fps, matching the detection cache.
STEP_S = 0.2
#: A fresh anchor is attempted at least this often while tracking holds.
ANCHOR_EVERY_S = 20.0
#: With no pose, an anchor is attempted this often (cheap gate first).
ANCHOR_RETRY_S = 1.0
#: The detector's rim this far from the carried one means the pose is stale.
DRIFT_WIDTHS = 1.5
#: A snap costing more than this many pixels means the carried homography is
#: no longer a pose of this camera at all, and the pose is dropped.
SNAP_MAX_PX = 12.0
#: ORB runs on frames shrunk by this much. A hop good to ~0.5 px at full size
#: is good to ~1 px here, against a tolerance of a 40 px rim width, and the
#: matching is the whole cost of tracking.
ORB_SCALE = 0.5


def hop_at_scale(source, target, source_boxes, target_boxes, scale):
    """`pairwise_homography` on shrunk frames, returned in full-size pixels."""
    import cv2

    from courtvision.court_tracking import pairwise_homography

    if scale >= 0.999:
        return pairwise_homography(source, target, source_boxes, target_boxes)

    def small(image):
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def shrink(boxes):
        return None if boxes is None else np.asarray(boxes, np.float64) * scale

    found = pairwise_homography(small(source), small(target),
                                shrink(source_boxes), shrink(target_boxes))
    if found is None:
        return None
    to_small = np.diag([scale, scale, 1.0])
    return np.linalg.inv(to_small) @ found @ to_small


def snap(camera, image_to_court, size):
    """Re-fit the carried homography to the camera's four parameters.

    Returns (image_to_court, cost_px, params) or (None, None, None). The cost
    is how far the carried homography was from being a pose of this camera -- a
    chain that has slipped produces a large one, which is why it is reported
    and not hidden. `params` is the pan/tilt/roll rotation vector and log
    focal, saved so a later pass can turn any pixel into a world ray without
    re-solving: with the centre fixed, a ray direction is the one description
    of an image point that camera motion cannot change.
    """
    from courtvision.court_camera import ptz_matrix, ptz_params
    court_to_image = np.linalg.inv(image_to_court)
    grid = np.array([[x, y] for x in np.linspace(0, 50, 6)
                     for y in np.linspace(0, 94, 11)], np.float64)
    h = np.c_[grid, np.ones(len(grid))] @ court_to_image.T
    visible = grid[h[:, 2] > 0]
    if len(visible) < 6:
        return None, None, None
    params, cost = ptz_params(court_to_image, camera.centre, camera.size, visible)
    fitted = ptz_matrix(params, camera.centre, camera.size)
    if not np.isfinite(fitted).all() or abs(np.linalg.det(fitted)) < 1e-12:
        return None, None, None
    return np.linalg.inv(fitted), float(cost), [float(v) for v in params]


def project_rims(camera, image_to_court, size, margin=8.0):
    """Both baskets as recorded-image pixels, keeping those inside the picture."""
    from courtvision.court_camera import RIMS_3D, project_3d
    try:
        points = project_3d(camera, image_to_court, np.array(RIMS_3D, np.float64))
    except Exception:
        return []
    out = []
    for p in np.asarray(points, np.float64).reshape(-1, 2):
        if np.isfinite(p).all() and -margin <= p[0] <= size[0] + margin \
                and -margin <= p[1] <= size[1] + margin:
            out.append([float(p[0]), float(p[1])])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--weights", default="checkpoints/court_keypoints/court_kp_960_ft.pt")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--step-s", type=float, default=STEP_S)
    parser.add_argument("--anchor-every-s", type=float, default=ANCHOR_EVERY_S)
    parser.add_argument("--orb-scale", type=float, default=ORB_SCALE)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument("--grid-lead-s", type=float, default=None,
                        help="track only a lead-in window before each evaluation-grid "
                             "time instead of the whole video -- same algorithm, a "
                             "fraction of the wall clock, for iterating on the metric")
    parser.add_argument("--no-track", action="store_true",
                        help="control arm: anchor only, never carry a pose")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from courtvision.court_camera import FixedCamera, undistort_image
    from courtvision.court_keypoints import KEYPOINTS, homography_from_keypoints
    from courtvision.court_register import register_frame
    from courtvision.device import resolve_device
    from courtvision.provenance import code_provenance

    spec = json.load(open(args.camera))
    camera = FixedCamera(np.array(spec["centre"], np.float64), tuple(spec["size"]),
                         floor=spec.get("floor"), k1=spec.get("k1", 0.0),
                         k2=spec.get("k2", 0.0))
    cache = json.load(open(args.detections))
    by_time = {round(row["t"], 3): row for row in cache["frames"]}
    cache_times = np.array(sorted(by_time)) if by_time else np.array([])

    def boxes_near(t, kind):
        if not len(cache_times):
            return []
        j = int(np.argmin(np.abs(cache_times - t)))
        if abs(cache_times[j] - t) > 0.3:
            return []
        return [b for b in by_time[cache_times[j]]["boxes"] if b["cls"] == kind]

    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end = min(args.end_s or duration, duration)
    model, device = YOLO(args.weights), resolve_device()

    if args.grid_lead_s:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_rim_and_ball import SAMPLE_EVERY_S, sample_times
        segments = [(max(0.0, g - args.grid_lead_s), g + args.step_s / 2)
                    for g in sample_times(end, SAMPLE_EVERY_S) if g >= args.start_s]
    else:
        segments = [(args.start_s, end)]

    rows, anchors, tracked, alarms = [], 0, 0, 0
    started = time.time()

    for segment_start, segment_end in segments:
        # Each window starts cold: no pose may leak across a gap in the walk.
        pose, pose_source, anchored_at, last_try = None, None, -1e9, -1e9
        anchor_grey, anchor_boxes, anchor_pose = None, None, None
        t = segment_start
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        while t < segment_end:
            ok, frame = capture.read()
            if not ok:
                break
            size = (frame.shape[1], frame.shape[0])
            pinhole = undistort_image(frame, camera.k1, camera.k2)
            grey = cv2.cvtColor(pinhole, cv2.COLOR_BGR2GRAY)
            players = [b["xyxy"] for b in boxes_near(t, "player")]
            boxes = np.array(players, np.float64) if players else None

            # TRACK: carry a pose to this frame, from the anchor where possible.
            snap_cost, hops, pose_params = None, 0, None
            if args.no_track:
                pose, pose_source = None, None
            else:
                carried, source = None, None
                if anchor_pose is not None:
                    hop = hop_at_scale(anchor_grey, grey, anchor_boxes, boxes,
                                       args.orb_scale)
                    if hop is not None and abs(np.linalg.det(hop)) > 1e-12:
                        carried, source, hops = anchor_pose @ np.linalg.inv(hop), "anchored-hop", 1
                if carried is None:
                    pose, pose_source = None, None
                else:
                    pose, snap_cost, pose_params = snap(camera, carried, size)
                    if pose is None or snap_cost > SNAP_MAX_PX:
                        pose, pose_source = None, None
                    else:
                        pose_source = source
                        tracked += 1

            # ALARM: an independently detected rim far from the carried one.
            stale = False
            if pose is not None:
                rims = [b for b in boxes_near(t, "rim") if b["conf"] >= 0.4]
                if rims:
                    shown = project_rims(camera, pose, size)
                    if shown:
                        b = max(rims, key=lambda b: b["conf"])
                        centre = np.array([(b["xyxy"][0] + b["xyxy"][2]) / 2,
                                           (b["xyxy"][1] + b["xyxy"][3]) / 2])
                        width = max(b["xyxy"][2] - b["xyxy"][0], 1.0)
                        if min(np.hypot(*(np.array(p) - centre)) for p in shown) > \
                                DRIFT_WIDTHS * width:
                            stale = True
                            alarms += 1
                    else:
                        stale = True

            due = (pose is None and t - last_try >= ANCHOR_RETRY_S) or \
                  (pose is not None and t - anchored_at >= args.anchor_every_s) or stale
            if due:
                last_try = t
                result = model.predict(frame, device=device, verbose=False)[0]
                start = None
                if result.keypoints is not None and len(result.keypoints):
                    xy = result.keypoints.xy[0].cpu().numpy()
                    conf = result.keypoints.conf[0].cpu().numpy()
                    start, _ = homography_from_keypoints(
                        {i: tuple(xy[i]) for i in range(len(xy))
                         if i in KEYPOINTS and conf[i] >= args.conf and (xy[i] > 0).all()})
                if start is not None:
                    matrix, info = register_frame(frame, start, boxes=boxes,
                                                  camera=camera, search=False)
                    if matrix is not None and info.get("refined"):
                        _, _, pose_params = snap(camera, matrix, size)
                        pose, pose_source, anchored_at = matrix, "anchor", t
                        anchor_grey, anchor_boxes, anchor_pose = grey, boxes, matrix
                        hops, anchors = 0, anchors + 1
                    elif stale:
                        pose, pose_source = None, None
                elif stale:
                    pose, pose_source = None, None

            rows.append({"t": round(float(t), 3),
                         "pose": pose.tolist() if pose is not None else None,
                         "source": pose_source,
                         "hops": hops,
                     "params": pose_params,
                         "snap_px": round(snap_cost, 2) if snap_cost is not None else None,
                         "rims": project_rims(camera, pose, size) if pose is not None else []})

            t += args.step_s
            target = int(round(t * fps))
            if abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - target) > 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            if len(rows) % 500 == 0:
                have = sum(1 for r in rows if r["pose"])
                print(f"  {t / 60:6.1f} min  {len(rows)} frames, pose on {have} "
                      f"({have / len(rows):.0%}), {anchors} anchors, {alarms} alarms, "
                      f"{(time.time() - started) / 60:.0f} min elapsed", flush=True)
    capture.release()

    have = sum(1 for r in rows if r["pose"])
    with_rim = sum(1 for r in rows if r["rims"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"video": args.video, "camera": args.camera, "step_s": args.step_s,
               "anchor_every_s": args.anchor_every_s, "no_track": args.no_track,
               "anchors": anchors, "tracked": tracked, "alarms": alarms,
               "frames": rows, **code_provenance(__file__)}, open(out, "w"))
    print(f"{len(rows)} frames; pose on {have} ({have / max(len(rows),1):.1%}); "
          f"rim projected on {with_rim} ({with_rim / max(len(rows),1):.1%}); "
          f"{anchors} anchors, {tracked} tracked hops, {alarms} drift alarms; "
          f"{(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
