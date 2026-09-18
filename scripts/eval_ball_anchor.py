"""Does anchoring the ball to the people beat taking the most confident box?

THE GAP THIS TESTS. On the uniform half of the labelled frames the detector
PROPOSES the ball somewhere in its candidate list on 0.824 / 0.902 / 0.964 of
frames and the shipped pipeline REPORTS the right one on 0.647 / 0.829 / 0.800.
Seven to eighteen points per broadcast sit between what is seen and what is
said, and every one of them is a selection problem rather than a detection one.

WHAT IS BEING TESTED IS ALREADY WRITTEN AND ALREADY SHIPS. `clip_boxes.anchor_penalty`
scores a candidate by confidence less what its distance from the nearest person
costs, and rejects one that is nowhere near anybody -- the term that stopped the
overlay path settling on a stationary orange thing in the crowd. It has never
been scored against the ball truth. The per-frame number this project quotes is
plain argmax over confidence.

NO DETECTOR RUNS HERE. Candidates come from the clip detection caches, which are
the boxes the labelling pages showed, so this measures selection on exactly the
evidence a person was judging.

DISCIPLINE, STATED BEFORE THE NUMBERS. The rules below were written down before
any of them was scored. Anything with a free parameter is fitted on the HARD
half -- the frames drawn because the ball model was failing -- and reported on
the UNIFORM half, which is the same split `eval_ball_temporal.py` uses and for
the same reason. Reported paired, with exact McNemar, because the rules answer
the same frames and an unpaired interval at n=130 cannot resolve the effects
this project produces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from courtvision.games import get, registry  # noqa: E402
from courtvision.stats import mcnemar, wilson  # noqa: E402
from eval_by_game import _cached_frames, _inside, _labels, _split  # noqa: E402


def centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def gap_to_box(point, box) -> float:
    """Distance from a point to a box, 0 inside it."""
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def anchors_of(dets, use_handler: bool):
    people = [d[2:6] for d in dets if d[0] == ("h" if use_handler else "p")]
    if use_handler and not people:
        people = [d[2:6] for d in dets if d[0] == "p"]
    return people


def scale_of(dets) -> float:
    """A player's median height, so distances are in bodies rather than pixels.

    A pixel means different things on a wide shot and a tight one, and the three
    broadcasts here are two resolutions. Every threshold below is a share of
    this."""
    heights = sorted(d[5] - d[3] for d in dets if d[0] in ("p", "h"))
    return heights[len(heights) // 2] if heights else 100.0


def rank(dets, rule: str, free: float, reject: float, use_handler: bool):
    """Candidate boxes best first under one rule."""
    balls = [(d[1], d[2:6]) for d in dets if d[0] == "b"]
    if rule == "confidence" or not balls:
        return [b for _, b in sorted(balls, key=lambda e: -e[0])]
    anchors = anchors_of(dets, use_handler)
    if not anchors:
        return [b for _, b in sorted(balls, key=lambda e: -e[0])]
    scale = scale_of(dets)
    scored = []
    for conf, box in balls:
        near = min(gap_to_box(centre(box), a) for a in anchors) / max(scale, 1.0)
        if rule == "reject" and near > reject:
            # Rejected outright, not merely penalised: a candidate two bodies
            # from every person on the floor is not a ball anyone is playing.
            # Kept at the back rather than dropped, so the any-rank ceiling is
            # unchanged and only the ORDER is being tested.
            scored.append((-1.0 - conf, box))
            continue
        penalty = max(0.0, near - free)
        scored.append((conf - penalty, box))
    return [b for _, b in sorted(scored, key=lambda e: -e[0])]


#: Declared before anything was scored. `free` is how far from a person a
#: candidate may be at no cost, in player heights; `reject` is where "reject"
#: stops believing it at all.
RULES = (
    ("confidence", "the most confident candidate -- what the 78.5% is"),
    ("penalty", "confidence less distance to the nearest PLAYER, in bodies"),
    ("handler", "...to the nearest HANDLER box, falling back to players"),
    ("reject", "...and a candidate beyond `reject` bodies is sent to the back"),
)


def score(game, rows, rule, free, reject, use_handler, frames):
    """Per-frame right/wrong for one rule, in a fixed order."""
    out = []
    for row in rows:
        dets = frames.get(round(float(row["t"]), 1))
        if dets is None:
            continue
        order = rank(dets, rule, free, reject, use_handler)
        radius = max(float(row.get("radius") or 10.0), 10.0)
        out.append(bool(order) and _inside(order[0], row["ball"], radius))
    return out


def gather(keys):
    """{game key: (uniform rows, hard rows, frames)} for every labelled game."""
    out = {}
    for key in keys:
        game = get(key)
        frames, _ = _cached_frames(game)
        if not frames:
            continue
        rows = [r for r in _labels(ROOT / "data/labels/possession_labels.json",
                                  game.label)
                if r.get("ball_verdict") == "ball" and r.get("ball")]
        uniform, hard = _split(rows)
        if uniform:
            out[key] = (uniform, hard, frames)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = gather(args.game or list(registry()))
    if not data:
        print("FAIL - no labelled broadcast has a clip detection cache")
        return 1

    # ---- fit the two constants on the HARD half only ------------------------
    grid = [(f, r) for f in (0.0, 0.25, 0.5, 1.0, 1.5)
            for r in (1.5, 2.0, 3.0, 5.0)]
    fitted = {}
    for rule, _ in RULES:
        if rule == "confidence":
            fitted[rule] = (0.0, 0.0)
            continue
        best, choice = -1.0, grid[0]
        for free, reject in grid:
            right = total = 0
            for key, (_, hard, frames) in data.items():
                got = score(key, hard, rule, free, reject, rule == "handler", frames)
                right += sum(got)
                total += len(got)
            if total and right / total > best:
                best, choice = right / total, (free, reject)
        fitted[rule] = choice
        print(f"  fitted {rule:11} free {choice[0]:.2f} bodies, reject "
              f"{choice[1]:.1f} -> {best:.3f} on the HARD half "
              f"(which is NOT reported below)")

    # ---- report on the UNIFORM half -----------------------------------------
    print(f"\n  UNIFORM frames only. Fitted on the hard half, reported here.\n")
    print(f"  {'rule':<12}{'pooled':>9}{'n':>6}   95% CI      "
          f"{'  '.join(f'{k:>7}' for k in data)}")
    vectors = {}
    for rule, note in RULES:
        free, reject = fitted[rule]
        per_game, pooled = {}, []
        for key, (uniform, _, frames) in data.items():
            got = score(key, uniform, rule, free, reject, rule == "handler", frames)
            per_game[key] = got
            pooled += got
        vectors[rule] = pooled
        low, high = wilson(sum(pooled), len(pooled))
        rates = "  ".join(f"{sum(v) / max(len(v), 1):>7.3f}"
                          for v in per_game.values())
        print(f"  {rule:<12}{sum(pooled) / max(len(pooled), 1):>9.3f}"
              f"{len(pooled):>6}   {low:.2f}-{high:.2f}   {rates}")
        print(f"      {note}")

    print("\n  PAIRED against plain confidence, exact McNemar on the frames "
          "where they differ:")
    base = vectors["confidence"]
    for rule, _ in RULES[1:]:
        only_a, only_b, p = mcnemar(vectors[rule], base)
        verdict = ("better" if only_a > only_b else
                   "worse" if only_b > only_a else "level")
        mark = "significant" if p < 0.05 else "not significant"
        print(f"    {rule:<12} {only_a:>3} frames only it gets, {only_b:>3} only "
              f"confidence   p = {p:.4f}   ({verdict}, {mark})")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"fitted": fitted,
             "uniform": {k: [int(b) for b in v] for k, v in vectors.items()}},
            indent=1))
        print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
