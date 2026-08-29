"""v2 §7.1 — fused torso-crop → CIELAB → mean colour.

Spec §7.1 says profile before writing CUDA, and profiling said something
counter-intuitive: do NOT target detection or the video model. Those are standard
forward passes already running vendor-tuned cuDNN/Metal kernels that hand-written
code will not beat.

The real candidate is team assignment. It was 4.8% of pipeline runtime at
5.8 ms/frame and is the largest stage with *no* optimised implementation behind
it — plain NumPy and OpenCV, one Python-level call per player per frame:

    crop the torso box -> cv2.cvtColor BGR2LAB -> reshape -> mean

For 10 players over 100 frames that is 1,000 tiny crops, 1,000 colour-space
conversions and 1,000 reductions, each with Python and OpenCV call overhead
dwarfing the arithmetic. It is embarrassingly parallel and memory-bound: exactly
what a fused kernel is for. One pass reads each pixel once, converts it in
registers, and reduces — never materialising the crops at all.

Three implementations, same numerics:

* `mean_torso_lab_reference` — NumPy/OpenCV, matches v1 exactly. The oracle.
* `mean_torso_lab_torch` — batched torch, runs on MPS/CUDA/CPU today.
* `mean_torso_lab_cuda` — the fused CUDA kernel in torso_color.cu, compiled on
  demand. Falls back to the torch path when CUDA is unavailable.

Everything is checked against the reference, so a wrong kernel fails a test
rather than quietly shifting team assignments.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# sRGB -> XYZ (D65), the matrix OpenCV uses.
_RGB_TO_XYZ = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=np.float64,
)
# D65 white point.
_WHITE = np.array([0.950456, 1.0, 1.088754], dtype=np.float64)
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def _torso_bounds(box, width: int, height: int, x_frac, y_frac):
    """Pixel bounds of the torso region, clamped. Mirrors team_assignment."""
    x1 = int(round(box.x1 + (box.x2 - box.x1) * x_frac[0]))
    x2 = int(round(box.x1 + (box.x2 - box.x1) * x_frac[1]))
    y1 = int(round(box.y1 + (box.y2 - box.y1) * y_frac[0]))
    y2 = int(round(box.y1 + (box.y2 - box.y1) * y_frac[1]))
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def mean_torso_lab_reference(image: np.ndarray, boxes: Sequence) -> np.ndarray:
    """NumPy/OpenCV oracle — exactly what v1's team_assignment does."""
    import cv2

    from courtvision.team_assignment import TORSO_X, TORSO_Y

    height, width = image.shape[:2]
    out = np.zeros((len(boxes), 3), dtype=np.float64)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = _torso_bounds(box, width, height, TORSO_X, TORSO_Y)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        out[index] = lab.reshape(-1, 3).mean(axis=0)
    return out


def _bgr_to_lab_torch(pixels):
    """BGR uint8 -> OpenCV-scaled 8-bit CIELAB, batched.

    float32 throughout: MPS has no float64 at all, and a real kernel would use
    float32 regardless. Lab values live in 0-255, so single precision is far more
    than this needs.
    """
    import torch

    rgb = pixels.flip(-1).to(torch.float32) / 255.0
    linear = torch.where(
        rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92
    )
    matrix = torch.tensor(_RGB_TO_XYZ, dtype=torch.float32, device=pixels.device)
    xyz = linear @ matrix.T
    scaled = xyz / torch.tensor(_WHITE, dtype=torch.float32, device=pixels.device)
    f = torch.where(
        scaled > _EPS, scaled.clamp(min=1e-12) ** (1.0 / 3.0), (_KAPPA * scaled + 16.0) / 116.0
    )
    lightness = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    # OpenCV's 8-bit Lab: L scaled by 255/100, a/b offset by 128.
    return torch.stack(
        [lightness * (255.0 / 100.0), a + 128.0, b + 128.0], dim=-1
    )


def mean_torso_lab_torch(image: np.ndarray, boxes: Sequence, device: str = "cpu"):
    """Batched torch implementation. Runs on MPS, CUDA or CPU."""
    import torch

    from courtvision.team_assignment import TORSO_X, TORSO_Y

    height, width = image.shape[:2]
    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    out = torch.zeros((len(boxes), 3), dtype=torch.float32, device=device)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = _torso_bounds(box, width, height, TORSO_X, TORSO_Y)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = tensor[y1:y2, x1:x2]
        out[index] = _bgr_to_lab_torch(crop).reshape(-1, 3).mean(dim=0)
    return out.cpu().numpy().astype(np.float64)


_FUSED_CACHE: list = []


def load_fused_kernel():
    """Compile torso_color.cu on first use. Returns None when CUDA is absent.

    UNVERIFIED ON HARDWARE: this project was developed on an Apple M2, which has
    no CUDA, so the kernel below has never been compiled or run. It is written to
    be correct and is checked against `mean_torso_lab_reference` by
    `verify_fused_kernel()` — run that first on any GPU box before trusting it.
    """
    import torch

    if _FUSED_CACHE:
        return _FUSED_CACHE[0]
    if not torch.cuda.is_available():
        return None

    from pathlib import Path

    from torch.utils.cpp_extension import load_inline

    source = (Path(__file__).with_suffix(".cu")).read_text()
    # The .cu carries its own includes and PYBIND11_MODULE, so it is passed as
    # the CUDA source with an empty C++ stub.
    module = load_inline(
        name="courtvision_torso_color",
        cpp_sources="",
        cuda_sources=source,
        functions=["torso_mean_lab"],
        verbose=False,
    )
    _FUSED_CACHE.append(module)
    return module


def mean_torso_lab_cuda(image: np.ndarray, boxes: Sequence) -> np.ndarray:
    """Fused CUDA path, falling back to the torch path when CUDA is absent."""
    import torch

    module = load_fused_kernel()
    if module is None:
        return mean_torso_lab_torch(image, boxes, device="cpu")

    from courtvision.team_assignment import TORSO_X, TORSO_Y

    height, width = image.shape[:2]
    bounds = np.array(
        [_torso_bounds(b, width, height, TORSO_X, TORSO_Y) for b in boxes],
        dtype=np.int32,
    ).reshape(-1, 4)
    image_gpu = torch.from_numpy(np.ascontiguousarray(image)).cuda()
    boxes_gpu = torch.from_numpy(bounds).cuda()
    return module.torso_mean_lab(image_gpu, boxes_gpu).cpu().numpy().astype(np.float64)


def verify_fused_kernel(trials: int = 5, tolerance: float = 2.0) -> bool:
    """Check the CUDA kernel against the OpenCV oracle. Run this on a GPU box.

    Returns True on agreement. Prints the worst disagreement either way, so a
    subtly wrong kernel is caught here rather than silently shifting team
    assignments in the pipeline.
    """
    import torch

    from courtvision.types import Box

    if not torch.cuda.is_available():
        print("verify_fused_kernel: no CUDA device; nothing to verify")
        return False

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(trials):
        image = rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8)
        boxes = [
            Box(float(x), float(y), float(x + 40), float(y + 100))
            for x, y in rng.integers(0, [560, 240], size=(8, 2))
        ]
        reference = mean_torso_lab_reference(image, boxes)
        fused = mean_torso_lab_cuda(image, boxes)
        worst = max(worst, float(np.abs(reference - fused).max()))

    ok = worst <= tolerance
    print(f"verify_fused_kernel: worst |diff| = {worst:.4f} "
          f"(tolerance {tolerance}) -> {'PASS' if ok else 'FAIL'}")
    return ok
