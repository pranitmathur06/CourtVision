"""Score the rim and ball from a labeller's verdicts on the rendered sheets.

`make_truth.py` builds reusable coordinate truth, which is the right thing when
the truth will be scored against many systems. It is also slow enough per frame
that a whole game's grid does not get labelled, and a coordinate read off a
magnified pane carries its own error. So the primary pass records VERDICTS --
what the sheet shows about the claim in front of it -- which are exact for the
system that drew the sheet, and cheap enough to cover every sampled frame.

The cost is that a verdict is tied to the system that produced the sheet: change
the system and the frames whose claim changed must be looked at again. That is
recorded honestly rather than worked around, and it is why the verdict file
names the system it was made against.

Per frame, for each object:

    ok       every visible one was claimed correctly
    miss     it is visible and was not claimed correctly
    ok+miss  two baskets shown, one found                  (rim only)
    -        it is not in this picture
    ?        it is in this picture somewhere, but cannot be found by eye

and for the ball only, a "+c" suffix on a miss -- as in `ball=miss+c` -- means
one of the magnified candidate panes DID hold the real ball. That separates the
ceiling from the choice: a miss with "+c" is a selection failure and costs
thought, a miss without it is a detection failure and costs a re-detection pass
over the whole game.

Accuracy is located/visible, exactly as `eval_rim_and_ball.py` defines it, and
a "?" is excluded from both sides and counted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GATE = 0.95


def parse(path):
    """Verdict lines -> {index: {"rim": verdict, "ball": verdict}}."""
    out = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        index = int(parts[0])
        row = {}
        for token in parts[1:]:
            key, _, value = token.partition("=")
            row[key] = value
        out[index] = row
    return out


def tally(verdicts, name):
    """(visible, located, unknown, absent, ceiling_recoverable) for one object."""
    visible = located = unknown = absent = recoverable = 0
    for row in verdicts.values():
        verdict = row.get(name, "-")
        if verdict == "?":
            unknown += 1
            continue
        if verdict == "-":
            absent += 1
            continue
        body = verdict.replace("+c", "")
        parts = [p for p in body.split("+") if p]
        visible += len(parts)
        located += sum(1 for p in parts if p == "ok")
        if "+c" in verdict:
            recoverable += 1
    return visible, located, unknown, absent, recoverable


def wilson(hits, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--clock", default=None)
    args = parser.parse_args()

    verdicts = parse(args.verdicts)
    rows = json.load(open(args.system))["frames"]

    spans = [("whole video", verdicts)]
    if args.clock:
        readings = json.load(open(args.clock))["readings"]
        first, last = min(r["t"] for r in readings), max(r["t"] for r in readings)
        inside = {i: v for i, v in verdicts.items()
                  if first <= rows[i]["t"] <= last}
        outside = {i: v for i, v in verdicts.items() if i not in inside}
        spans.append((f"in game ({first:.0f}-{last:.0f} s)", inside))
        spans.append(("outside the game", outside))

    passed = True
    for label, subset in spans:
        if not subset:
            continue
        print(f"--- {label}: {len(subset)} frames labelled")
        for name in ("rim", "ball"):
            visible, located, unknown, absent, recoverable = tally(subset, name)
            accuracy = located / visible if visible else float("nan")
            lo, hi = wilson(located, visible)
            mark = "PASS" if accuracy >= GATE else "FAIL"
            if label == "whole video":
                passed &= bool(accuracy >= GATE)
            print(f"  {name.upper():4s} visible {visible:4d}  located {located:4d}  "
                  f"missed {visible - located:4d}   accuracy {accuracy:.3f} "
                  f"(95% CI {lo:.3f}-{hi:.3f})  {mark}")
            print(f"       not in shot {absent:4d}   could not tell {unknown:3d}", end="")
            if name == "ball":
                lost = visible - located
                print(f"   of {lost} misses, {recoverable} had a right candidate "
                      f"on offer (selection) and {lost - recoverable} had none "
                      f"(detection)", end="")
            print()
        print()
    print("PASS - both objects at or above the gate" if passed else "FAIL - below the gate")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
