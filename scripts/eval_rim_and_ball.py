"""Rim and ball accuracy on a whole game, against hand-checked truth.

Declared BEFORE the numbers exist, because Round 64 shipped a gate condition
whose test could not fail (`over_court` flagged 0 of 78,293 ball boxes) and it
took an adversarial review to notice.

THE DENOMINATOR is the thing that decides whether a number like "95%" means
anything, so it is fixed here and never chosen after the fact:

- Frames are sampled on a FIXED GRID over the whole video (every SAMPLE_EVERY_S
  seconds from t=0), not around shots, not where registration worked, not where
  the clock was readable. Every sampled frame is labelled and every sampled
  frame counts.
- A frame where the object is NOT VISIBLE (a close-up, a crowd shot, a replay
  wipe, the rim out of frame) is not a miss -- nothing can be found there. It
  moves to the false-alarm denominator instead.

So each object gets two numbers, and BOTH are reported always:

    accuracy    = located / visible          (of the frames where it is there,
                                              how often is it found, correctly)
    false alarm = reported / not visible     (how often is it claimed anyway)

"Located" means the reported centre lies within a tolerance of the true centre,
scaled to the object so it means the same at any zoom:

    rim   TOL_RIM  = 1.0 rim widths   (a rim is 1.5 ft across)
    ball  TOL_BALL = 1.0 ball widths  (a ball is 0.79 ft across)

A reported point on the WRONG rim, or on a spectator's head, is not located --
it is both a miss and a false alarm, and the report says so separately.

A wide view shows BOTH baskets, so the rim is a LIST per frame on both sides:
every truth rim must be matched by a reported one, and every reported rim that
matches no truth rim is a false alarm. Picking "the" rim would have meant
choosing which basket counted, after seeing which one the system found.

TRUTH comes from `label_rim_and_ball.py`: for each sampled frame a human says
whether each object is visible and where its centre is. Proposals from the
detector and from the projected rim are shown to make that a confirmation
rather than a click, but a proposal is only ever ACCEPTED or CORRECTED by the
labeller -- never assumed.

The gate for Phase 2, set by the project's owner: 95% on both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: Frames are labelled and scored every this many seconds of video. 25 s gives
#: ~290 frames a game -- enough that a 95% reading carries a +/-2.5% interval,
#: few enough that every one of them can be looked at by eye.
SAMPLE_EVERY_S = 25.0
#: A rim is 1.5 ft across, a ball 0.79 ft. Tolerance is one object width.
TOL_RIM_WIDTHS = 1.0
TOL_BALL_WIDTHS = 1.0
#: The gate.
GATE = 0.95


def sample_times(duration_s: float, every_s: float = SAMPLE_EVERY_S) -> list[float]:
    """The fixed grid. Offset half a step so it cannot land on t=0 exactly."""
    return [float(t) for t in np.arange(every_s / 2, duration_s, every_s)]


def located(reported, truth_centre, truth_width, tol_widths) -> bool:
    """Is the reported centre within `tol_widths` object widths of the truth?"""
    if reported is None or truth_centre is None:
        return False
    d = float(np.hypot(reported[0] - truth_centre[0], reported[1] - truth_centre[1]))
    return d <= tol_widths * float(truth_width)


def _objects(row, name):
    """Truth or system entry as a list -- one ball, zero to two rims.

    A bare `[x, y]` is one object, not two; a list of them is many.
    """
    value = row.get(name) if isinstance(row, dict) else None
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if value and all(isinstance(v, (int, float)) for v in value):
        return [list(value)]
    return [v for v in value if v is not None]


def score(truth_rows, system, tol_rim=TOL_RIM_WIDTHS, tol_ball=TOL_BALL_WIDTHS):
    """Per-object accuracy and false-alarm rate over the labelled grid.

    `truth_rows`: [{t, rim: [{centre, width}], ball: {centre, width} | None}]
    `system`:     {t: {"rim": [[x, y]], "ball": [x, y] | None}}
    """
    out = {}
    for name, tol in (("rim", tol_rim), ("ball", tol_ball)):
        visible = located_n = claimed = 0
        frames_absent = 0
        for row in truth_rows:
            truths = _objects(row, name)
            reports = [r for r in _objects(system.get(row["t"], {}), name) if r is not None]
            visible += len(truths)
            if not truths:
                frames_absent += 1
            taken = set()
            for truth in truths:
                for i, report in enumerate(reports):
                    if i in taken:
                        continue
                    if located(report, truth["centre"], truth["width"], tol):
                        taken.add(i)
                        located_n += 1
                        break
            claimed += len(reports) - len(taken)
        out[name] = {
            "visible": visible,
            "located": located_n,
            "missed": visible - located_n,
            "accuracy": located_n / visible if visible else float("nan"),
            "frames_without": frames_absent,
            "false_alarms": claimed,
        }
    return out


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion -- so 'we got 95%' carries its own error."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="label_rim_and_ball.py output")
    parser.add_argument("--system", required=True, help="{t: {rim, ball}} per frame")
    args = parser.parse_args()

    truth_rows = json.load(open(args.truth))["frames"]
    raw = json.load(open(args.system))
    system = {float(t): v for t, v in (raw["frames"].items() if isinstance(raw.get("frames"), dict)
                                       else ((r["t"], r) for r in raw["frames"]))}
    report = score(truth_rows, system)

    print(f"{len(truth_rows)} frames on a {SAMPLE_EVERY_S:.0f} s grid\n")
    passed = True
    for name in ("rim", "ball"):
        r = report[name]
        lo, hi = wilson(r["located"], r["visible"])
        mark = "PASS" if r["accuracy"] >= GATE else "FAIL"
        passed &= r["accuracy"] >= GATE
        print(f"{name.upper():5s} visible {r['visible']:4d}  located {r['located']:4d}  "
              f"missed {r['missed']:4d}")
        print(f"      accuracy {r['accuracy']:.3f}  (95% CI {lo:.3f}-{hi:.3f})  "
              f"gate {GATE:.2f}  {mark}")
        print(f"      false alarms {r['false_alarms']:4d}  "
              f"({r['frames_without']} frames had no {name} at all)\n")
    print("PASS - both objects at or above the gate" if passed else
          "FAIL - below the gate")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
