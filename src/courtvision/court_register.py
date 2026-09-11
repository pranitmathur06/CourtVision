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
                   camera=None):
    """Refine a landmark registration with each kind of paint evidence.

    Returns `(matrix, info)`. When `info["refined"]` is False no evidence type
    produced an accepted fit and the landmark registration comes back
    unchanged -- the pipeline's fallback, never dressed as a refinement.
    `info["polarity"]` names the evidence used, and `info["tried"]` keeps every
    attempt's diagnostics.
    """
    best_matrix, best_info, tried = None, None, {}
    for polarity in polarities:
        response = paint_response(frame, polarity)
        matrix, info = refine(frame, landmark_matrix, boxes=boxes,
                              prepared=(response, _structure(response)),
                              camera=camera)
        tried[polarity] = info
        if info["refined"] and (best_info is None
                                or info["peak_ratio"] > best_info["peak_ratio"]):
            best_matrix, best_info = matrix, dict(info, polarity=polarity)
    if best_info is None:
        return landmark_matrix, {"refined": False, "polarity": None, "tried": tried}
    best_info["tried"] = tried
    return best_matrix, best_info
