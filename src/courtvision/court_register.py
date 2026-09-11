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

from .court_refine import _structure, paint_response, refine

#: Evidence types tried per frame; on a tie the earlier one is kept.
POLARITIES = ("bright", "all")


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
    from .court_camera import same_floor, search_starts
    from .court_refine import _POINTS

    best_matrix, best_info, tried = None, None, {}
    for polarity in polarities:
        response = paint_response(frame, polarity)
        prepared = (response, _structure(response))
        starts = [landmark_matrix] if landmark_matrix is not None else []
        if camera is not None and search:
            starts += search_starts(camera, *prepared, boxes=boxes, image=frame)
        chosen = None
        for start in starts:
            matrix, info = refine(frame, start, boxes=boxes, prepared=prepared,
                                  camera=camera)
            if info["refined"] and (chosen is None or info["samples"] > chosen[1]["samples"]):
                chosen = (matrix, info)
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
        return landmark_matrix, {"refined": False, "polarity": None, "tried": tried}
    best_info["tried"] = tried
    return best_matrix, best_info
