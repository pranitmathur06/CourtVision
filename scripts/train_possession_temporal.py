"""Train the possession operator and the scan over time, end to end.

The windows come from `build_possession_windows.py` and the split is the
sampling: frames drawn UNIFORMLY are the evaluation and are never trained on,
frames drawn because the model was struggling there are the training set. So the
number at the end estimates in-game accuracy, not performance on hard cases.

WHAT IS COMPARED, AND WHY BOTH NUMBERS COME OUT OF THIS SCRIPT. Kernel 1 alone
is the centre frame's softmax; kernels 1+2 is the posterior at the centre frame
after the scan. They are trained together and scored on the same frames in the
same run, so the temporal part has to justify itself against the per-frame part
rather than against a number remembered from another script.

TWO PHASES, FOR A REASON THAT IS ABOUT COST AND NOT ABOUT PRINCIPLE. Only the
sampling region makes the swept features depend on the pixels; everything else
is a function of the features. So phase one holds the region fixed and trains on
features computed once, which is fast enough to run hundreds of epochs, and
phase two unfreezes the region and pays for the pixels again on every step. The
gradient reaching the region is the same gradient in both -- phase one simply
does not spend it.

A CARRIED BOX IS NOT AN OBSERVED ONE. A track with no detection in a frame keeps
its last box, so the sweep reads the floor where the player used to be. That is
weak evidence rather than absent evidence, and `stale` is a trained penalty on
it -- one number, learned from the training half, rather than a rule about how
much to trust a stale box.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def wilson(hits, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def drawn_boxes():
    """The box a person drew, per frame. What everything is scored against.

    Scoring by "did it name the right index" is stricter than `eval_handler.py`,
    which asks whether the box a method points at overlaps the box a person
    drew. Players overlap, so two candidates can both clear IoU 0.5 on the same
    man; counting one right and the other wrong measures the box list, not the
    method, and it understated these kernels by seven frames against a baseline
    that was never scored that way.
    """
    out = {}
    for labels in ("data/labels/possession_labels.json",
                   "data/labels/handler_labels.json"):
        if Path(labels).exists():
            for row in json.load(open(labels))["frames"]:
                if row.get("handler_box"):
                    out[row["file"]] = np.asarray(row["handler_box"],
                                                  dtype=np.float64)
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def load_windows(folder, want_split=None):
    rows = []
    for path in sorted(Path(folder).glob("*.npz")):
        blob = np.load(path, allow_pickle=True)
        split = str(blob["split"])
        if want_split and split != want_split:
            continue
        rows.append({"path": path, "split": split, "target": int(blob["target"]),
                     "centre": int(blob["centre"]), "verdict": str(blob["verdict"]),
                     "name": str(blob["name"]), "game": str(blob["game"])})
    return rows


def open_window(row, torch, dtype, with_patches=True):
    """Patches, boxes, presence and ball for one window, as tensors.

    The patches are the whole cost: a window is about 20 MB once it is float32,
    and all 665 of them at once would be 13 GB on a 17 GB machine. So phase one
    asks for `with_patches=False`, keeps only the features it computed, and lets
    the pixels go; phase two reads them back from disk on every epoch, which is
    slower per step and the reason the two phases exist.
    """
    blob = np.load(row["path"], allow_pickle=True)
    origin = blob["origin"].astype(np.float32)               # (T, K, 2)
    local = blob["boxes"].astype(np.float32)                 # (T, K, 4)
    absolute = local + origin[:, :, [0, 1, 0, 1]]
    return {
        "patches": (torch.as_tensor(blob["patches"].astype(np.float32), dtype=dtype)
                    if with_patches else None),
        "shape": tuple(blob["patches"].shape),
        "local": torch.as_tensor(local, dtype=dtype),
        "absolute": torch.as_tensor(absolute, dtype=dtype),
        "present": torch.as_tensor(blob["present"].astype(np.float32), dtype=dtype),
        "ball": torch.as_tensor(blob["ball"].astype(np.float32), dtype=dtype),
        "target": row["target"], "centre": row["centre"],
    }


def sweep_features(window, region, torch, grid, beta, surround, eps):
    """Kernel 1's features for every frame and track at once.

    The same arithmetic as `possession_torch`, over a (T, K) batch instead of a
    single frame -- the patches make that possible, because each player's
    pixels are already cut out and a sample inside a patch reads exactly what
    the same sample would read in the whole frame.
    """
    patches = window["patches"]                               # (T, K, H, W)
    frames, tracks, height, width = patches.shape
    flat = patches.reshape(frames * tracks, height, width)
    boxes = window["local"].reshape(frames * tracks, 4)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    box_w = (x2 - x1).unsqueeze(1)
    box_h = (y2 - y1).unsqueeze(1)
    steps = (torch.arange(grid, dtype=patches.dtype) + 0.5) / grid
    fx = region[0] + (region[1] - region[0]) * steps
    fy = region[2] + (region[3] - region[2]) * steps
    xs = x1.unsqueeze(1) + box_w * fx.unsqueeze(0)            # (N, G)
    ys = y1.unsqueeze(1) + box_h * fy.unsqueeze(0)
    grid_x = xs.unsqueeze(1).expand(-1, grid, -1).reshape(len(boxes), -1)
    grid_y = ys.unsqueeze(2).expand(-1, -1, grid).reshape(len(boxes), -1)
    which = torch.arange(len(boxes)).unsqueeze(1)

    def read(px, py):
        px = px.clamp(0.0, width - 1.0 - 1e-6)
        py = py.clamp(0.0, height - 1.0 - 1e-6)
        ax, ay = px.floor(), py.floor()
        rx, ry = px - ax, py - ay
        axi, ayi = ax.long(), ay.long()
        bxi = (axi + 1).clamp(max=width - 1)
        byi = (ayi + 1).clamp(max=height - 1)
        return ((1 - rx) * (1 - ry) * flat[which, ayi, axi]
                + rx * (1 - ry) * flat[which, ayi, bxi]
                + (1 - rx) * ry * flat[which, byi, axi]
                + rx * ry * flat[which, byi, bxi])

    radius = (surround * (y2 - y1).clamp_min(eps)).unsqueeze(1)
    swept = read(grid_x, grid_y) - 0.25 * (read(grid_x - radius, grid_y)
                                           + read(grid_x + radius, grid_y)
                                           + read(grid_x, grid_y - radius)
                                           + read(grid_x, grid_y + radius))
    peak = (torch.logsumexp(beta * swept, dim=1)
            - math.log(swept.shape[1])) / beta
    mean = swept.mean(dim=1)

    absolute = window["absolute"].reshape(frames * tracks, 4)
    ax1, ay1, ax2, ay2 = absolute[:, 0], absolute[:, 1], absolute[:, 2], absolute[:, 3]
    body = (ay2 - ay1).clamp_min(eps)
    ball = window["ball"]                                     # (T, 3)
    ball_flat = ball.unsqueeze(1).expand(-1, tracks, -1).reshape(frames * tracks, 3)
    has_ball = (ball_flat[:, 2] > 0).to(patches.dtype)
    dx = (ball_flat[:, 0] - (ax1 + ax2) / 2.0) / body * has_ball
    dy = (ball_flat[:, 1] - (ay1 + ay2) / 2.0) / body * has_ball
    inside = ((ball_flat[:, 0] >= ax1) & (ball_flat[:, 0] <= ax2)
              & (ball_flat[:, 1] >= ay1) & (ball_flat[:, 1] <= ay2)
              ).to(patches.dtype) * has_ball
    features = torch.stack([dx, dy, torch.hypot(dx, dy) * has_ball, inside,
                            peak, mean, ball_flat[:, 2]], dim=1)
    return features.reshape(frames, tracks, -1)


def head_scores(features, parameters, torch, eps):
    standard = ((features - parameters["feature_mean"])
                / parameters["feature_scale"].clamp_min(eps))
    hidden = torch.tanh(standard @ parameters["first"].T + parameters["first_bias"])
    return hidden @ parameters["second"] + parameters["second_bias"]


def window_scores(window, parameters, torch, grid, beta, surround, eps,
                  features=None):
    """Per-frame log-scores over the tracks plus 'nobody'."""
    if features is None:
        features = sweep_features(window, parameters["region"], torch, grid,
                                  beta, surround, eps)
    players = head_scores(features, parameters, torch, eps)   # (T, K)
    players = players + parameters["stale"] * (1.0 - window["present"])
    nobody = parameters["nobody"].reshape(1, 1).expand(players.shape[0], 1)
    return torch.cat([players, nobody], dim=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--fast-epochs", type=int, default=400)
    parser.add_argument("--full-epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--region-lr", type=float, default=0.01)
    parser.add_argument("--out", default="checkpoints/possession/temporal.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cv", type=int, default=0,
                        help="cross-validate INSIDE the training half instead "
                             "of scoring the held-out frames. Five folds of 314 "
                             "windows give a selection signal with five times "
                             "the frames of one eval run, and -- the point -- "
                             "spend none of the evaluation set doing it. The "
                             "held-out number then means what it claims.")
    parser.add_argument("--init-region", default=None,
                        help="a weights.json to take the starting sampling "
                             "region from. The region matters more than its "
                             "four numbers suggest: trained end to end on the "
                             "single frames it moved from the upper body to "
                             "0.64-0.91 of the way down, which is where a "
                             "dribbled ball is, and phase one cannot find that "
                             "because it holds the region still.")
    parser.add_argument("--stay", default="learn", choices=["learn", "off"],
                        help="'off' pins the stay bonus to zero and freezes it, "
                             "which makes the scan the identity and trains "
                             "kernel 1 against its own per-frame objective -- "
                             "the only fair thing to compare kernels 1+2 with, "
                             "because a head trained THROUGH the scan is not "
                             "the same head")
    args = parser.parse_args()

    import torch

    from courtvision.kernels.possession import (BETA, EPS, FEATURES, GRID,
                                                SURROUND, default_parameters,
                                                temporal_torch)

    dtype = torch.float64
    torch.manual_seed(args.seed)
    rows = load_windows(args.windows)
    train = [r for r in rows if r["split"] == "train" and r["target"] >= 0]
    evaluate = [r for r in rows if r["split"] == "eval"]
    print(f"  {len(rows)} windows: {len(train)} to train on, {len(evaluate)} held out")

    start = time.time()
    # Everything except the pixels, which are read back only when needed.
    light = {r["name"]: open_window(r, torch, dtype, with_patches=False)
             for r in rows}
    by_name = {r["name"]: r for r in rows}
    print(f"  indexed {len(light)} windows in {time.time() - start:.0f}s")

    def with_pixels(name):
        return open_window(by_name[name], torch, dtype, with_patches=True)

    base = default_parameters(args.seed)
    if args.init_region:
        base["region"] = np.asarray(json.load(open(args.init_region))["region"],
                                    dtype=np.float64)
        print(f"  starting region from {args.init_region}: "
              f"{np.round(base['region'], 3)}")
    parameters = {k: torch.tensor(v, dtype=dtype) for k, v in base.items()}
    # The scan's own parameters. stay starts near zero -- softplus(0) = 0.69, a
    # mild preference for keeping the ball -- and stale starts at zero so the
    # penalty is learned rather than assumed.
    parameters["stay_raw"] = torch.tensor(0.0 if args.stay == "learn" else -60.0,
                                          dtype=dtype)
    parameters["stale"] = torch.tensor(0.0, dtype=dtype)

    fixed = {"feature_mean", "feature_scale"}
    if args.stay == "off":
        # softplus(-60) is 0 to every digit of float64, so the transition matrix
        # is all zeros and the posterior at the centre frame is that frame's own
        # softmax -- checked in test_a_zero_stay_bonus_leaves_every_frame_independent.
        fixed = fixed | {"stay_raw"}
        print("  stay bonus pinned off: this trains kernel 1 on its own")
    # Fitted on the CENTRE frames of the training windows only. Fitting it
    # across the whole window instead cost eleven points: the neighbours carry a
    # stale box whenever the detector lost a player, and those frames drag the
    # mean and the scale away from the distribution the answer is actually read
    # from. Evaluation frames are excluded either way, which is the part that
    # would be cheating rather than merely wrong.
    with torch.no_grad():
        seen = []
        for row in train:
            seen.append(sweep_features(with_pixels(row["name"]), parameters["region"],
                                       torch, GRID, BETA, SURROUND, EPS)
                        [row["centre"]].numpy())
        stacked = np.concatenate(seen, axis=0)
        parameters["feature_mean"] = torch.tensor(stacked.mean(axis=0), dtype=dtype)
        parameters["feature_scale"] = torch.tensor(stacked.std(axis=0) + 1e-3,
                                                   dtype=dtype)
    print(f"    feature scale {np.round(parameters['feature_scale'].numpy(), 4)}")

    for name in parameters:
        if name not in fixed:
            parameters[name].requires_grad_(True)

    def run_phase(epochs, with_region, label):
        if not epochs:
            return
        names = [n for n in parameters if n not in fixed
                 and (with_region or n != "region")]
        # Phase two is a warm restart, and Adam does not carry its moments
        # across one. Starting the head again at the full rate threw the loss
        # from 1.07 to 1.22 in four epochs -- not the region moving, which at
        # 0.001 barely moves at all, but the head being kicked out of the basin
        # phase one found. So the head continues at a tenth of the rate.
        head_lr = args.lr / 10.0 if with_region else args.lr
        groups = [{"params": [parameters[n] for n in names if n != "region"],
                   "lr": head_lr}]
        if with_region:
            groups.append({"params": [parameters["region"]], "lr": args.region_lr})
        optimiser = torch.optim.Adam(groups)
        cache = None
        if not with_region:
            with torch.no_grad():
                cache = {r["name"]: sweep_features(with_pixels(r["name"]),
                                                   parameters["region"], torch,
                                                   GRID, BETA, SURROUND, EPS)
                         for r in rows}
        began = time.time()
        for epoch in range(epochs):
            optimiser.zero_grad()
            total = 0.0
            for row in train:
                window = (light[row["name"]] if cache
                          else with_pixels(row["name"]))
                scores = window_scores(window, parameters, torch, GRID, BETA,
                                       SURROUND, EPS,
                                       features=cache[row["name"]] if cache else None)
                posterior = temporal_torch(scores, parameters["stay_raw"],
                                           row["centre"])
                total = total + -torch.log(posterior[row["target"]] + 1e-12)
            loss = total / max(len(train), 1)
            loss.backward()
            optimiser.step()
            with torch.no_grad():
                parameters["region"].clamp_(0.0, 1.0)
            if epoch % max(epochs // 6, 1) == 0 or epoch == epochs - 1:
                print(f"    {label} epoch {epoch:>4}  loss {float(loss):.4f}  "
                      f"stay {float(torch.nn.functional.softplus(parameters['stay_raw']).detach()):.3f}"
                      f"  stale {float(parameters['stale'].detach()):+.3f}"
                      f"  [{time.time() - began:.0f}s]")

    if args.cv:
        folds = args.cv
        order = np.random.default_rng(args.seed).permutation(len(train))
        scores_by_fold = []
        for fold in range(folds):
            held = {train[i]["name"] for i in order[fold::folds]}
            inner = [r for r in train if r["name"] not in held]
            outer = [r for r in train if r["name"] in held]
            fresh = {k: torch.tensor(np.asarray(v.detach().numpy()), dtype=dtype)
                     for k, v in parameters.items()}
            for name in fresh:
                if name not in fixed:
                    fresh[name].requires_grad_(True)
            cache = {}
            with torch.no_grad():
                for row in train:
                    cache[row["name"]] = sweep_features(
                        with_pixels(row["name"]), fresh["region"], torch, GRID,
                        BETA, SURROUND, EPS)
            optimiser = torch.optim.Adam(
                [v for k, v in fresh.items() if k not in fixed], lr=args.lr)
            for _ in range(args.fast_epochs):
                optimiser.zero_grad()
                total = 0.0
                for row in inner:
                    scores = window_scores(light[row["name"]], fresh, torch, GRID,
                                           BETA, SURROUND, EPS,
                                           features=cache[row["name"]])
                    posterior = temporal_torch(scores, fresh["stay_raw"],
                                               row["centre"])
                    total = total + -torch.log(posterior[row["target"]] + 1e-12)
                (total / max(len(inner), 1)).backward()
                optimiser.step()
            hits = 0
            for row in outer:
                with torch.no_grad():
                    scores = window_scores(light[row["name"]], fresh, torch, GRID,
                                           BETA, SURROUND, EPS,
                                           features=cache[row["name"]])
                    tracks = scores.shape[1] - 1
                    posterior = temporal_torch(scores, fresh["stay_raw"],
                                               row["centre"])
                    hits += int(int(torch.argmax(posterior[:tracks])) == row["target"])
                    # index equality here, not overlap: inside cross-validation
                    # the question is only which configuration learns more, and
                    # both are measured the same way.
            scores_by_fold.append(hits / max(len(outer), 1))
            print(f"    fold {fold + 1}/{folds}  {hits}/{len(outer)} = "
                  f"{scores_by_fold[-1]:.1%}")
        mean = float(np.mean(scores_by_fold))
        print(f"\n  cross-validated on the TRAINING half: {mean:.1%} "
              f"(+/- {np.std(scores_by_fold):.1%} across {folds} folds)")
        print("  the held-out frames were not touched")
        return 0

    run_phase(args.fast_epochs, False, "region fixed ")
    run_phase(args.full_epochs, True, "end to end   ")
    print(f"    region {np.round(parameters['region'].detach().numpy(), 3)}")

    # ---- the held-out numbers ---------------------------------------------
    truth = drawn_boxes()
    hits = {"kernel1": 0, "kernel1+2": 0}
    scored = nobox = 0
    per_case = {"kernel1": {}, "kernel1+2": {}}
    for row in evaluate:
        if row["verdict"] == "nobody":
            continue
        scored += 1
        if row["verdict"] == "missing" or row["target"] < 0:
            nobox += 1
            continue                     # no candidate is right; a miss for both
        window = with_pixels(row["name"])
        with torch.no_grad():
            scores = window_scores(window, parameters, torch, GRID, BETA,
                                   SURROUND, EPS)
            tracks = scores.shape[1] - 1
            alone = int(torch.argmax(scores[row["centre"], :tracks]))
            posterior = temporal_torch(scores, parameters["stay_raw"], row["centre"])
            together = int(torch.argmax(posterior[:tracks]))
        found = bool(float(window["ball"][row["centre"], 2]) > 0)
        wanted = truth.get(row["name"])
        centre_boxes = window["absolute"][row["centre"]].numpy()
        for key, pick in (("kernel1", alone), ("kernel1+2", together)):
            right = wanted is not None and iou(centre_boxes[pick], wanted) >= 0.5
            hits[key] += int(right)
            bucket = per_case[key].setdefault(found, [0, 0])
            bucket[0] += int(right)
            bucket[1] += 1

    print(f"\n  held out: {scored} frames where somebody had the ball "
          f"({nobox} with no box drawn for him, automatic misses)")
    for key, label in (("kernel1", "kernel 1 alone, the centre frame"),
                       ("kernel1+2", "kernels 1+2, the scan over time")):
        low, high = wilson(hits[key], scored)
        print(f"    {label:<36} {hits[key]:>3}/{scored} = {hits[key] / max(scored, 1):5.1%}"
              f"   (95% CI {low:.0%}-{high:.0%})")
    print("    the detector's handler class, same frames: 49.7%")
    print("\n  split by whether the ball detector found anything:")
    for key in ("kernel1", "kernel1+2"):
        parts = [f"ball {'found  ' if found else 'missing'} "
                 f"{v[0]}/{v[1]} = {v[0] / max(v[1], 1):.0%}"
                 for found, v in sorted(per_case[key].items(), reverse=True)]
        print(f"    {key:<10} " + "   ".join(parts))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: v.detach().numpy().tolist() for k, v in parameters.items()},
              open(out, "w"), indent=1)
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
