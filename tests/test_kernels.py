"""The fused torso-colour path must match the OpenCV oracle exactly enough."""

import numpy as np
import pytest

from courtvision.kernels.torso_color import (
    mean_torso_lab_reference,
    mean_torso_lab_torch,
)
from courtvision.types import Box


def synthetic_frame(seed: int = 0):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    boxes = [Box(20, 30, 60, 130), Box(120, 40, 165, 150), Box(200, 20, 250, 120)]
    return image, boxes


def test_torch_matches_the_opencv_oracle():
    image, boxes = synthetic_frame()
    reference = mean_torso_lab_reference(image, boxes)
    torch_result = mean_torso_lab_torch(image, boxes)
    # OpenCV's 8-bit LAB quantises to uint8 before averaging; the torch path stays
    # in float. A couple of levels of mean difference is that rounding, not a bug.
    assert np.allclose(reference, torch_result, atol=2.0), (
        f"max diff {np.abs(reference - torch_result).max():.3f}"
    )


def test_matches_on_flat_colour_patches():
    """Exact colours, where quantisation error is smallest."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :100] = (40, 40, 200)    # BGR red
    image[:, 100:] = (200, 60, 40)    # BGR blue
    boxes = [Box(10, 10, 90, 190), Box(110, 10, 190, 190)]
    reference = mean_torso_lab_reference(image, boxes)
    torch_result = mean_torso_lab_torch(image, boxes)
    assert np.allclose(reference, torch_result, atol=2.0)
    # And the two patches must be far apart, or team clustering is pointless.
    assert np.linalg.norm(reference[0] - reference[1]) > 30.0


def test_degenerate_boxes_yield_zeros_not_errors():
    image, _ = synthetic_frame()
    boxes = [Box(10, 10, 10, 10), Box(-50, -50, -10, -10)]
    reference = mean_torso_lab_reference(image, boxes)
    torch_result = mean_torso_lab_torch(image, boxes)
    assert reference.shape == (2, 3)
    assert np.allclose(reference, torch_result, atol=2.0)


def test_handles_no_boxes():
    image, _ = synthetic_frame()
    assert mean_torso_lab_reference(image, []).shape == (0, 3)
    assert mean_torso_lab_torch(image, []).shape == (0, 3)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_agreement_holds_across_random_frames(seed):
    image, boxes = synthetic_frame(seed)
    reference = mean_torso_lab_reference(image, boxes)
    torch_result = mean_torso_lab_torch(image, boxes)
    assert np.allclose(reference, torch_result, atol=2.0)


def test_torch_path_runs_on_the_accelerator():
    from courtvision.device import resolve_device

    image, boxes = synthetic_frame()
    device = resolve_device()
    result = mean_torso_lab_torch(image, boxes, device=device)
    assert result.shape == (len(boxes), 3)
    assert np.allclose(mean_torso_lab_reference(image, boxes), result, atol=2.0)


def test_cuda_path_falls_back_cleanly_without_a_gpu():
    """On a CUDA-less machine the fused path must still return correct numbers."""
    from courtvision.kernels.torso_color import load_fused_kernel, mean_torso_lab_cuda

    import torch

    image, boxes = synthetic_frame()
    if torch.cuda.is_available():
        pytest.skip("CUDA present; the fallback path is not what runs here")

    assert load_fused_kernel() is None
    result = mean_torso_lab_cuda(image, boxes)
    assert np.allclose(mean_torso_lab_reference(image, boxes), result, atol=2.0)


def test_cuda_source_exists_and_declares_the_entry_point():
    """The kernel is unverified on hardware, so at least pin its interface."""
    from pathlib import Path

    from courtvision.kernels import torso_color

    source = Path(torso_color.__file__).with_suffix(".cu").read_text()
    assert "torso_mean_lab" in source
    assert "__global__ void torso_mean_lab_kernel" in source
    assert "PYBIND11_MODULE" in source
