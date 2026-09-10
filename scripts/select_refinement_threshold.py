"""Choose MIN_PEAK_RATIO from a calibration dump, by a rule written here first.

The rule: the smallest threshold in CANDIDATES whose accepted measurements
reach a held-out median of TARGET_FT or better, where a held-out family that is
lost, or whose refit is refused, counts as a failure -- infinite error -- in
EVERY statistic. An earlier table counted failures in the median but not the
p90 and so reported a 0.67 ft tail that was really 0.89.

The simulation is exact only for a dump made at threshold 0: a candidate then
refuses a frame when the full fit's ratio falls below it, and refuses a
measurement when that family's refit's ratio does. A dump gated higher has
already discarded refits the candidate would need to judge, so it is refused.

It also refuses the unseen arena, whose footage must not choose anything, and
any dump without provenance. The previous threshold was set by code that was
never checked in, six minutes after the dump it read, which left "the rule came
first" unverifiable from the repository. This file is the rule.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

CANDIDATES = (2.0, 3.0, 5.0, 7.0, 10.0)
TARGET_FT = 0.30
UNSEEN = ("FZAUuuuREg0",)


def _stats(errors):
    v = np.array(errors, dtype=float)
    if not len(v):
        return float("nan"), float("nan"), float("nan"), float("nan")
    # "nearest" keeps an infinite error infinite instead of interpolating to NaN.
    return (float(np.percentile(v, 50, method="nearest")),
            float(np.percentile(v, 90, method="nearest")),
            float(np.mean(v <= TARGET_FT)), float(np.mean(~np.isfinite(v))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    args = parser.parse_args()
    data = json.load(open(args.dump))
    if not isinstance(data, dict) or "meta" not in data:
        print("FAIL - dump has no provenance; regenerate it with --dump")
        return 1
    meta, records = data["meta"], data["records"]
    if any(u in meta["video"] for u in UNSEEN):
        print(f"FAIL - {meta['video']} is the unseen arena; it may not choose a threshold")
        return 1
    if meta["min_peak_ratio"] > min(CANDIDATES):
        print(f"FAIL - dump gated at {meta['min_peak_ratio']}; regenerate with --min-peak-ratio 0")
        return 1
    registered = meta["registered"]
    frames = meta.get("frames", [])
    print(f"{meta['video']}  commit {meta['commit']}{' (dirty)' if meta.get('dirty') else ''}"
          f"  polarity {meta['polarity']}  {registered} registered frames")
    print("  threshold  frames accepted   measurements   p50       p90       within   failed")
    chosen = None
    for threshold in CANDIDATES:
        accepted = sum(1 for f in frames if f["refined"] and (f["peak_ratio"] or 0) >= threshold)
        errors = []
        for r in records:
            if (r["peak_ratio"] or 0) < threshold:
                continue                               # frame refused at this threshold
            if r["refined_err"] is None or (r.get("refit_ratio") or 0) < threshold:
                errors.append(np.inf)                  # lost, or refit refused
            else:
                errors.append(r["refined_err"])
        p50, p90, within, failed = _stats(errors)
        mark = ""
        if chosen is None and p50 <= TARGET_FT:
            chosen, mark = threshold, "  <- chosen"
        print(f"  {threshold:9.1f}  {accepted:4d} ({accepted/max(registered,1):4.0%})"
              f"      {len(errors):5d}       {p50:6.2f}    {p90:6.2f}    {within:5.0%}   {failed:5.0%}{mark}")
    if chosen is None:
        print(f"  no candidate reaches a held-out median of {TARGET_FT} ft")
        return 1
    print(f"MIN_PEAK_RATIO = {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
