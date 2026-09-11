"""Choose TRUST_RADIUS_FT from a calibration dump, by a rule written here first.

A registration asserts only court points within the trust radius of the paint
its fit rests on (court_refine.TRUST_RADIUS_FT). This picks that radius on the
OKC calibration game and nowhere else.

The metric is the held-out one, restricted to trusted ground. For each
(frame, held-out family) whose frame was accepted: the family's samples within
R of the paint the refit -- which never saw the family -- rests on. If the
landmark registration finds paint on at least MIN_SAMPLES of them, the family
is visible there; its error is the median |offset| of the refit's found samples
among them, or a FAILURE (infinite) if the refit finds fewer than MIN_SAMPLES.
A refused refit asserts nothing, so it adds no error, but its visible paint
stays in the coverage denominator -- a radius cannot look good by refusing.

The rule: the largest radius R in CANDIDATES_FT such that EVERY candidate up to
and including R has a conservative median at or under CALIBRATION_TARGET_FT
(candidates with no measurable family are skipped). A first version took the
largest passing radius outright, and that is not monotone: on OKC 4, 6 and 8 ft
failed (0.27, 0.27, 0.26) while 12 ft passed at 0.24 only because the families
it newly admitted scored 0.19 and pulled the pooled median down -- the ground
between 8 and 12 ft was itself ~0.47 ft off. A radius certifies all the ground
inside it, so every radius inside it must pass. That is 0.05 ft inside the 0.30 ft goal,
because at equal distance from paint the test arenas differed by up to ~0.08 ft
(Round 54), and the calibration arena must leave room for an arena that is
worse. If no radius qualifies, the smallest is taken and marked FALLBACK --
which means the goal is not met at any radius, and says so.

`--report` scores a dump at every radius and selects nothing: that is how the
development arenas are read. The unseen arenas are scored by
scripts/eval_trusted_on_annotations.py, never here.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

CANDIDATES_FT = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
CALIBRATION_TARGET_FT = 0.25
TARGET_FT = 0.30
MIN_SAMPLES = 15
CALIBRATION = "fullgame.mp4"


def score(records, frames, radius):
    """Conservative trusted errors, and coverage of visible paint and of feet."""
    errors, trusted_visible, all_visible = [], 0, 0
    for r in records:
        base = set(r.get("base_idx", []))
        all_visible += len(base)
        if r.get("dist") is None:
            continue                          # refit refused: asserts nothing
        near = {i for i, d in zip(r["family_idx"], r["dist"]) if d <= radius}
        seen = base & near
        if len(seen) < MIN_SAMPLES:
            continue
        trusted_visible += len(seen)
        offsets = [o for i, o in zip(r["fit_idx"], r["fit_off"]) if i in near]
        errors.append(float(np.median(offsets)) if len(offsets) >= MIN_SAMPLES else np.inf)
    feet = sum(f.get("feet_n", 0) for f in frames)
    feet_trusted = sum(sum(d <= radius for d in f.get("feet_dist", []))
                       for f in frames if f.get("refined"))
    v = np.array(errors, dtype=float)
    return {"n": len(v),
            "p50": float(np.percentile(v, 50, method="nearest")) if len(v) else float("nan"),
            "p75": float(np.percentile(v, 75, method="nearest")) if len(v) else float("nan"),
            "within": float(np.mean(v <= TARGET_FT)) if len(v) else float("nan"),
            "failed": float(np.mean(~np.isfinite(v))) if len(v) else float("nan"),
            "paint_coverage": trusted_visible / max(all_visible, 1),
            "feet_coverage": feet_trusted / max(feet, 1)}


def choose(table):
    """The largest radius below which every measurable radius meets the target."""
    chosen = None
    for radius in sorted(table):
        p50 = table[radius]["p50"]
        if not np.isfinite(p50) and table[radius].get("n", 1) == 0:
            continue                      # nothing measurable at this radius
        if not p50 <= CALIBRATION_TARGET_FT:
            break
        chosen = radius
    if chosen is not None:
        return chosen, False
    return min(table), True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--report", action="store_true",
                        help="score every radius; select nothing")
    args = parser.parse_args()
    data = json.load(open(args.dump))
    if not isinstance(data, dict) or "meta" not in data or "commit" not in data["meta"]:
        print("FAIL - dump has no provenance; regenerate it with --dump")
        return 1
    meta, records = data["meta"], data["records"]
    if not records or "base_idx" not in records[0]:
        print("FAIL - dump predates per-sample distances; regenerate it")
        return 1
    if not args.report and not meta["video"].endswith(CALIBRATION):
        print(f"FAIL - {meta['video']} is not the calibration game; use --report")
        return 1
    frames = meta.get("frames", [])
    print(f"{meta['video']}  commit {meta['commit']}{' (dirty)' if meta.get('dirty') else ''}"
          f"  polarity {meta['polarity']}  threshold {meta['min_peak_ratio']}")
    print("  radius   families  p50      p75      within 0.3  failed   paint trusted  feet trusted")
    table = {radius: score(records, frames, radius) for radius in CANDIDATES_FT}
    for radius, s in table.items():
        print(f"  {radius:4.1f} ft  {s['n']:5d}    {s['p50']:5.2f}    {s['p75']:5.2f}    "
              f"{s['within']:5.0%}      {s['failed']:4.0%}    {s['paint_coverage']:5.0%}"
              f"          {s['feet_coverage']:5.0%}")
    if args.report:
        return 0
    radius, fallback = choose(table)
    print(f"  -> TRUST_RADIUS_FT = {radius}"
          + ("  (FALLBACK: no radius reaches the calibration target)" if fallback else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
