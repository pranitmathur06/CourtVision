"""Compile both possession kernels on a real GPU, check them, and time them.

This is the half of the verification that the host harness cannot do.
`verify_possession_numerics.py` proves the arithmetic and the chain rule on a
machine with no GPU; this proves the launch configuration, and measures.

SPEED IS REPORTED PER MACHINE AND NOT GENERALISED, which is the lesson of
`docs/v7-gpu-results.md`: the one kernel this project shipped before measured
1.4x faster than the unfused path on one box and 1.0x on another, and the
honest conclusion was that 1.4x was a property of the box. So this prints the
GPU's name, runs each timing three times, and reports all three.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np


def timed(call, repeats: int, inner: int):
    import torch

    runs = []
    for _ in range(repeats):
        for _ in range(3):                       # warm up, and compile any JIT
            call()
        torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(inner):
            call()
        torch.cuda.synchronize()
        runs.append((time.perf_counter() - began) / inner * 1e3)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, default=10)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inner", type=int, default=200)
    args = parser.parse_args()

    import torch

    from courtvision.kernels.possession import (BETA, default_parameters,
                                                load_fused_kernel,
                                                possession_torch,
                                                temporal_reference,
                                                temporal_torch)

    if not torch.cuda.is_available():
        print("SKIP - no CUDA on this machine")
        return 0
    name = torch.cuda.get_device_name(0)
    print(f"  GPU: {name}   torch {torch.__version__}")

    began = time.time()
    module = load_fused_kernel()
    if module is None:
        print("FAIL - load_fused_kernel() returned None with CUDA available")
        return 1
    print(f"  compiled possession.cu in {time.time() - began:.0f}s")

    rng = np.random.default_rng(5)
    height, width, players = 720, 1280, args.players
    image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    image[360:376, 640:656] = (20, 110, 240)
    boxes = np.stack([np.linspace(60, 1100, players),
                      np.full(players, 240.0),
                      np.linspace(60, 1100, players) + 80,
                      np.full(players, 420.0)], axis=1)
    ball = np.array([648.0, 368.0, 0.8])
    base = default_parameters()

    device = torch.device("cuda")
    gpu = {k: torch.tensor(v, dtype=torch.float32, device=device)
           for k, v in base.items()}
    image_t = torch.as_tensor(np.ascontiguousarray(image)).to(device)
    boxes_t = torch.tensor(boxes, dtype=torch.float32, device=device)
    ball_t = torch.tensor(ball, dtype=torch.float32, device=device)

    # ---- kernel 1: the fused path against the portable one -----------------
    probs, features = module.possession_forward(
        image_t, boxes_t, ball_t, gpu["first"], gpu["first_bias"], gpu["second"],
        gpu["second_bias"], gpu["nobody"], gpu["region"], gpu["feature_mean"],
        gpu["feature_scale"], float(BETA))
    want_probs, want_features = possession_torch(image, boxes, ball, gpu)
    print(f"  kernel 1 forward   probs {float((probs - want_probs).abs().max()):.3e}"
          f"   features {float((features - want_features).abs().max()):.3e}")

    grad_probs = torch.tensor(rng.normal(0.0, 1.0, size=players + 1),
                              dtype=torch.float32, device=device)
    grads = module.possession_backward(
        image_t, boxes_t, gpu["first"], gpu["first_bias"], gpu["second"],
        gpu["region"], gpu["feature_mean"], gpu["feature_scale"], features,
        probs, grad_probs, float(BETA))
    traced = {k: v.clone().requires_grad_(k not in ("feature_mean", "feature_scale"))
              for k, v in gpu.items()}
    portable, _ = possession_torch(image, boxes, ball, traced)
    portable.backward(grad_probs)
    worst = max(float((grads[i] - traced[n].grad.reshape(grads[i].shape)).abs().max())
                for i, n in enumerate(["first", "first_bias", "second",
                                       "second_bias", "nobody", "region"]))
    print(f"  kernel 1 backward  worst |diff| vs autograd {worst:.3e}")

    # ---- kernel 2: the same, for the scan ----------------------------------
    states = players + 1
    scores = rng.normal(0.0, 1.8, size=(args.frames, states))
    stay_raw = 0.4
    target, centre = 3, args.frames // 2
    scores_t = torch.tensor(scores, dtype=torch.float32, device=device)
    stay_t = torch.tensor([stay_raw], dtype=torch.float32, device=device)
    posterior, loss, grad_scores, grad_stay = module.temporal_forward(
        scores_t, stay_t, target, centre)
    want_post, want_loss, want_grads = temporal_reference(scores, stay_raw,
                                                          target, centre)
    print(f"  kernel 2 forward   posterior "
          f"{float((posterior.cpu() - torch.tensor(want_post)).abs().max()):.3e}"
          f"   loss {abs(float(loss[0]) - want_loss):.3e}")
    print(f"  kernel 2 backward  d/dscores "
          f"{float((grad_scores.cpu() - torch.tensor(want_grads['scores'])).abs().max()):.3e}"
          f"   d/dstay {abs(float(grad_stay[0]) - want_grads['stay_raw']):.3e}")

    # ---- speed, three runs, this machine only ------------------------------
    fused_1 = timed(lambda: module.possession_forward(
        image_t, boxes_t, ball_t, gpu["first"], gpu["first_bias"], gpu["second"],
        gpu["second_bias"], gpu["nobody"], gpu["region"], gpu["feature_mean"],
        gpu["feature_scale"], float(BETA)), args.repeats, args.inner)
    torch_1 = timed(lambda: possession_torch(image_t, boxes_t, ball_t, gpu),
                    args.repeats, args.inner)
    fused_2 = timed(lambda: module.temporal_forward(scores_t, stay_t, target, centre),
                    args.repeats, args.inner)

    def scan_and_grad():
        traced_scores = scores_t.detach().clone().requires_grad_(True)
        traced_stay = stay_t.detach().clone().reshape(()).requires_grad_(True)
        (-torch.log(temporal_torch(traced_scores, traced_stay, centre)[target])
         ).backward()

    torch_2 = timed(scan_and_grad, args.repeats, args.inner)

    print(f"\n  timings on {name}, {players} players, {args.frames} frames, "
          f"{args.inner} calls per run, milliseconds")
    for label, runs in (("kernel 1 fused (fwd)", fused_1),
                        ("kernel 1 torch (fwd)", torch_1),
                        ("kernel 2 fused (fwd+bwd)", fused_2),
                        ("kernel 2 torch (fwd+bwd)", torch_2)):
        print(f"    {label:<26} " + "  ".join(f"{v:.3f}" for v in runs)
              + f"   median {statistics.median(runs):.3f}")
    print(f"    kernel 1 speedup {statistics.median(torch_1) / statistics.median(fused_1):.2f}x"
          f"   kernel 2 speedup "
          f"{statistics.median(torch_2) / statistics.median(fused_2):.2f}x")
    print("    (one machine, one session -- not a claim about the kernels)")

    # ---- occupancy, and whether 'memory-bound' is true ---------------------
    # `ncu` cannot run here: performance counters are gated by the HOST driver
    # (ERR_NVGPUCTRPERM) and being root inside a rented container does not lift
    # it. These two numbers are the ones the runbook wanted, obtained without a
    # profiler -- occupancy from the runtime, and the bandwidth from bytes over
    # measured time, which is the claim 'memory-bound' actually makes.
    threads = 64
    occupancy = module.kernel_occupancy(threads, players, args.frames, states)[0]
    blocks_1, blocks_2, per_sm, sms, shared = [int(v) for v in occupancy]
    print(f"\n  occupancy at {threads} threads/block, from the runtime "
          f"(no profiler needed)")
    print(f"    kernel 1  {blocks_1} blocks/SM = {blocks_1 * threads} threads, "
          f"{blocks_1 * threads / per_sm:.0%} of the {per_sm} an SM can hold")
    print(f"    kernel 2  {blocks_2} blocks/SM = {blocks_2 * threads} threads, "
          f"{blocks_2 * threads / per_sm:.0%} of the {per_sm} an SM can hold")
    print(f"    {sms} SMs, {shared // 1024} KB shared memory per block")
    print("    One block per frame leaves an SM nearly idle -- the shape is "
          "built for\n    a batch of frames, and one frame at a time is the "
          "wrong way to launch it.")

    touched = image.nbytes
    seconds = statistics.median(fused_1) / 1e3
    print(f"\n  is kernel 1 memory-bound? it reads a "
          f"{height}x{width} frame = {touched / 1e6:.1f} MB")
    print(f"    {touched / seconds / 1e9:.0f} GB/s achieved if every byte is "
          f"read once, against ~768 GB/s peak on this card")
    print("    -- but the sweep touches only the players' boxes, so the bytes "
          "it really\n    reads are far fewer and this is an upper bound on "
          "the traffic, not a\n    measurement of it. Settling that needs "
          "counters, and counters need ncu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
