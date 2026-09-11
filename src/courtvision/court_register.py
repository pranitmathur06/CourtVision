"""From a frame's landmark registration to the one downstream phases use.

Courts are not painted alike, and the evidence that finds one floor's lines
misses another's. OKC and TD Garden paint white lines, which thin-bright-ridge
evidence ("bright") finds without the clutter that edges add. Toyota Center
paints black lines and a solid red key with no lane line on its boundary, which
only ridges of either polarity plus colour edges ("all") can see. On the held-
out test arenas neither setting won everywhere: bright was better at TD Garden
(0.18 against 0.20 ft) and Target Center (0.23 against 0.30), all at Fiserv
Forum (0.29 against 0.35).

So each frame is refined with both and keeps the sharper fit -- the higher
peak ratio, which within every test arena went with lower error on the human
annotations (Spearman -0.29 to -0.86). The rule has no free parameter to tune.
It was chosen after the test arenas' per-polarity results had been seen, which
is recorded here and in the commit that adds this file: a test-set figure for
it is weaker evidence than a first look would be, and it has to hold on the
calibration footage too.
"""

from __future__ import annotations

import numpy as np

from .court_refine import _structure, paint_response, refine

#: Evidence types tried per frame; on a tie the earlier one is kept.
POLARITIES = ("bright", "all")
#: A fit whose start came from the landmark-free search, with nothing else
#: vouching for it, must rest on at least this many line samples (0.5 ft
#: apart). Over a whole Toyota Center game the one wrong search fit -- a
#: courtside close-up given a main-camera pose -- rested on 38; the next
#: fewest of 61 fits had 98, and no landmark-started fit had under 145. Set
#: with those values in view, midway in the gap.
SEARCH_MIN_SAMPLES = 80


def register_frame(frame, landmark_matrix, boxes=None, polarities=POLARITIES,
                   camera=None, search=True, verify=False):
    """Refine a landmark registration with each kind of paint evidence.

    Returns `(matrix, info)`. When `info["refined"]` is False no evidence type
    produced an accepted fit and the landmark registration comes back
    unchanged -- the pipeline's fallback, never dressed as a refinement.
    `info["polarity"]` names the evidence used, and `info["tried"]` keeps every
    attempt's diagnostics.

    With the game's `camera`:
    - `search`: starts found from paint alone (court_camera.search_starts) are
      tried beside the landmark start, and are the only starts when the
      landmark model gives none -- on a whole game at Toyota Center it gave
      none on ~20 of ~65 ordinary game views. Within one kind of evidence the
      fit resting on the most paint wins.
    - `verify` (off by default): the accepted fit is re-fitted without the
      camera, and the frame refused if this game's camera cannot reproduce the
      free fit. Measured over a whole game it does not separate the cases it
      is for: two halftime highlights from other arenas read 7.9-11.5 px, but
      eight genuine frames whose camera fits sit on the paint read 3.2-12.2 px
      and were refused with them. The free fit is the less trustworthy of the
      two -- it is the one that locks onto ad boards and extrapolates the far
      side -- so its disagreement is not evidence against the camera.
    - The floor: when the camera carries its game's floor signature, a fit
      whose key is a different colour is refused (court_camera.same_floor).
      That is what catches another game's highlights: their geometry can be a
      near pan/tilt/zoom of this camera, their paint never is.
    """
    from .court_camera import same_floor, search_starts, undistort_image, undistort_points
    from .court_refine import _POINTS

    # With the game's lens distortion known, everything below works on the
    # frame a pinhole camera would have recorded, and the returned matrix maps
    # PINHOLE pixels to the court: callers undistort pixels first
    # (court_camera.undistort_points, or to_court). The landmark start was
    # found on the recorded frame; it is only a start, a few pixels off at the
    # edges, well inside the first 48 px pass.
    k1 = getattr(camera, "k1", None) if camera is not None else None
    if k1:
        size = (frame.shape[1], frame.shape[0])
        frame = undistort_image(frame, k1)
        if boxes is not None and len(boxes):
            corners = undistort_points(np.asarray(boxes, np.float64).reshape(-1, 2), k1, size)
            boxes = corners.reshape(-1, 4)

    best_matrix, best_info, tried = None, None, {}
    for polarity in polarities:
        response = paint_response(frame, polarity)
        prepared = (response, _structure(response))
        starts = [("landmark", landmark_matrix)] if landmark_matrix is not None else []
        if camera is not None and search:
            starts += [("search", s) for s in search_starts(camera, *prepared, boxes=boxes,
                                                            image=frame)]
        chosen = None
        for source, start in starts:
            matrix, info = refine(frame, start, boxes=boxes, prepared=prepared,
                                  camera=camera)
            if not info["refined"]:
                continue
            if source == "search" and info["samples"] < SEARCH_MIN_SAMPLES:
                continue
            if chosen is None or info["samples"] > chosen[1]["samples"]:
                chosen = (matrix, dict(info, start=source))
        if chosen is None:
            tried[polarity] = {"refined": False, "starts": len(starts)}
            continue
        matrix, info = chosen
        if camera is not None and verify:
            free, free_info = refine(frame, matrix, boxes=boxes, prepared=prepared,
                                     passes=(12, 6, 3), starts=((0.0, 0.0),))
            if free_info["refined"]:
                ok, cost = camera.explains(free, _POINTS[free_info["support"]])
                info = dict(info, camera_check_px=float(cost))
                if not ok:
                    tried[polarity] = dict(info, refined=False,
                                           reason=f"not this game's camera ({cost:.1f} px)")
                    continue
        ok, distance = same_floor(camera, frame, matrix, boxes)
        info = dict(info, floor_lab_distance=distance)
        if not ok:
            tried[polarity] = dict(info, refined=False,
                                   reason=f"not this game's floor (key colour {distance:.0f} Lab away)")
            continue
        tried[polarity] = info
        if best_info is None or info["peak_ratio"] > best_info["peak_ratio"]:
            best_matrix, best_info = matrix, dict(info, polarity=polarity)
    if best_info is None:
        return landmark_matrix, {"refined": False, "polarity": None, "tried": tried, "k1": k1}
    best_info["tried"] = tried
    best_info["k1"] = k1
    return best_matrix, best_info


def to_court(matrix, info, pixels, size):
    """Court feet of recorded-image pixels under a `register_frame` result."""
    from .court_camera import undistort_points
    pts = undistort_points(pixels, info.get("k1"), size)
    h = np.c_[pts, np.ones(len(pts))] @ np.asarray(matrix).T
    return h[:, :2] / h[:, 2:3]
