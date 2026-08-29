"""One command to verify all v2 work on a CUDA box.

Everything in v2 was written on an Apple M2, which has no CUDA. The kernel has
never been compiled and the disaggregated pipeline has never had two devices to
spread across. This runs the whole verification sequence and prints an
unambiguous PASS/FAIL per step, in the order the runbook prescribes:

    ./.venv/bin/python -m scripts.verify_v2_gpu

Nothing here is wired into the pipeline until it passes. A kernel that is subtly
wrong would shift team assignments silently, which is worse than a kernel that
is simply absent.
"""

from __future__ import annotations

import sys
import time

import numpy as np


def step(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  — ' + detail if detail else ''}")
    return ok


def main() -> int:
    import torch

    print("v2 GPU verification\n")
    results = []

    # 1. Hardware.
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    results.append(step("CUDA available", n > 0, f"{n} device(s): {', '.join(names) or 'none'}"))
    if n == 0:
        print("\nNo CUDA device. Nothing further can be verified here.")
        return 1
    step("two devices for §7.2", n >= 2,
         "one GPU: disaggregation will run but cannot overlap" if n < 2 else "")

    # 2. Kernel compiles.
    from courtvision.kernels.torso_color import (
        load_fused_kernel,
        mean_torso_lab_cuda,
        mean_torso_lab_reference,
        mean_torso_lab_torch,
        verify_fused_kernel,
    )

    compiled = False
    try:
        compiled = load_fused_kernel() is not None
        results.append(step("kernel compiles", compiled))
    except Exception as exc:  # noqa: BLE001 - first compiles usually fail somewhere
        results.append(step("kernel compiles", False, f"{type(exc).__name__}: {exc}"))

    # 3. Kernel is numerically correct against the OpenCV oracle.
    if compiled:
        results.append(step("kernel matches the OpenCV oracle", verify_fused_kernel()))

    # 4. Is it actually faster? A kernel that is correct but slower is not worth
    #    the risk of using.
    if compiled:
        from courtvision.types import Box

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
        boxes = [Box(float(x), float(y), float(x + 60), float(y + 150))
                 for x, y in rng.integers(0, [1200, 560], size=(10, 2))]

        def bench(fn, warmup=3, iters=20):
            for _ in range(warmup):
                fn(image, boxes)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                fn(image, boxes)
            torch.cuda.synchronize()
            return (time.perf_counter() - start) / iters * 1000.0

        reference_ms = bench(mean_torso_lab_reference)
        torch_ms = bench(lambda i, b: mean_torso_lab_torch(i, b, device="cuda"))
        fused_ms = bench(mean_torso_lab_cuda)
        print(f"\n    OpenCV reference {reference_ms:7.3f} ms/frame")
        print(f"    torch on CUDA    {torch_ms:7.3f} ms/frame")
        print(f"    fused kernel     {fused_ms:7.3f} ms/frame "
              f"({reference_ms / max(fused_ms, 1e-9):.1f}x vs reference)\n")
        results.append(step("fused kernel beats the reference", fused_ms < reference_ms))

    # 5. §7.2 — does splitting across devices actually overlap?
    from courtvision.config import Config
    from courtvision.serving import StagePlan, run_disaggregated
    from courtvision.types import ActionWindow, Frame

    config = Config()
    frames_n = 96
    images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(frames_n)]
    frames = [Frame(i, i / config.target_fps, ()) for i in range(frames_n)]

    def slow_classify(imgs, frs, cfg, holders):
        time.sleep(0.005)
        return [ActionWindow(frs[0].index, frs[-1].index, frs[0].time_s,
                             frs[-1].time_s, "dribble", 0.9)]

    windows, timing = run_disaggregated(
        images, frames, [1] * frames_n, slow_classify, config
    )
    results.append(step("disaggregated pipeline runs", len(windows) > 0,
                        f"{len(windows)} windows, "
                        f"waits {timing.waiting}"))
    print(f"\n    plan for this box: "
          f"{StagePlan('cuda:0', 'cuda:1') if n >= 2 else StagePlan.single('cuda:0')}")

    ok = all(results)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} — "
          f"{sum(results)}/{len(results)}")
    if ok:
        print("Safe to wire the fused kernel into team_assignment and to run the")
        print("split StagePlan in run_pipeline.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
