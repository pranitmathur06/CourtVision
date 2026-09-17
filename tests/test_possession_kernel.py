"""The fused possession operator: three implementations, one set of numbers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from courtvision.kernels.possession import (BETA, FEATURES, GRID,
                                            default_parameters,
                                            load_fused_kernel,
                                            possession_reference,
                                            possession_torch)

torch = pytest.importorskip("torch")


def scene(seed: int = 5, players: int = 3):
    """A frame with something orange in the first player's hands."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(120, 200, 3), dtype=np.uint8)
    image[58:66, 88:96] = (20, 110, 240)
    boxes = np.array([[80.0, 40.0, 110.0, 100.0],
                      [130.0, 38.0, 158.0, 98.0],
                      [30.0, 42.0, 58.0, 102.0]])[:players]
    ball = np.array([92.0, 62.0, 0.8])
    return image, boxes, ball


def as_tensors(parameters, grad=True):
    return {name: torch.tensor(np.asarray(value), dtype=torch.float64,
                               requires_grad=grad and name not in
                               ("feature_mean", "feature_scale"))
            for name, value in parameters.items()}


def test_the_reference_and_the_torch_path_agree():
    image, boxes, ball = scene()
    parameters = default_parameters()
    expected, _ = possession_reference(image, boxes, ball, parameters)
    got, _ = possession_torch(image, boxes, ball, as_tensors(parameters, grad=False))
    assert np.abs(expected - got.detach().numpy()).max() < 1e-12


def test_the_probabilities_are_a_distribution_over_players_and_nobody():
    image, boxes, ball = scene(players=3)
    probs, features = possession_reference(image, boxes, ball, default_parameters())
    assert probs.shape == (4,)
    assert features.shape == (3, FEATURES)
    assert probs.min() >= 0.0
    assert abs(probs.sum() - 1.0) < 1e-12


def test_the_sweep_sees_a_ball_that_the_geometry_cannot():
    """With no ball detection at all, the pixels still separate the players.

    This is the case the operator exists for: 65% of the frames where the ball
    model's best candidate is under 0.35 have a ball a person can see.
    """
    image, boxes, _ = scene()
    blind = np.array([0.0, 0.0, 0.0])          # the detector offered nothing
    _, features = possession_reference(image, boxes, blind, default_parameters())
    assert features[:, :4].max() == 0.0        # geometry is switched off
    assert features[0, 4] > features[1, 4]     # and the sweep still finds him
    assert features[0, 4] > features[2, 4]


def test_a_ball_on_the_floor_does_not_look_like_a_ball_in_the_hands():
    """Centre minus surround, because hardwood is orange too.

    Scoring orange-ness alone measured 0.099 on the floor against 0.037 on the
    crowd, and a sweep trained on it learned to point at whoever stood on the
    most visible wood -- 31.8% against a 49.7% baseline.
    """
    rng = np.random.default_rng(1)
    wooden = np.zeros((120, 200, 3), dtype=np.uint8)
    wooden[:, :] = (120, 170, 210)             # a floor, orange everywhere
    wooden += rng.integers(0, 6, size=wooden.shape, dtype=np.uint8)
    boxes = np.array([[80.0, 40.0, 110.0, 100.0]])
    _, flat = possession_reference(wooden, boxes, np.zeros(3), default_parameters())
    spotted = wooden.copy()
    spotted[58:66, 88:96] = (20, 110, 240)     # one compact ball on that floor
    _, blob = possession_reference(spotted, boxes, np.zeros(3), default_parameters())
    assert flat[0, 4] < 0.05                   # uniform wood answers near zero
    assert blob[0, 4] > 3 * max(flat[0, 4], 1e-3)


def test_the_backward_matches_a_numeric_gradient():
    """gradcheck over every trainable parameter, including where to look."""
    image, boxes, ball = scene(players=2)
    parameters = default_parameters()
    names = [n for n in parameters
             if n not in ("feature_mean", "feature_scale")]

    def run(*values):
        full = dict(zip(names, values))
        full["feature_mean"] = torch.tensor(parameters["feature_mean"])
        full["feature_scale"] = torch.tensor(parameters["feature_scale"])
        probs, _ = possession_torch(image, boxes, ball, full)
        return probs

    args = tuple(torch.tensor(np.asarray(parameters[n]), dtype=torch.float64,
                              requires_grad=True) for n in names)
    assert torch.autograd.gradcheck(run, args, eps=1e-6, atol=1e-7, rtol=1e-4)


def test_the_gradient_reaches_where_the_sweep_looks():
    """The region is learned, so it must receive gradient -- and not zero."""
    image, boxes, ball = scene()
    parameters = default_parameters()
    tensors = as_tensors(parameters)
    probs, _ = possession_torch(image, boxes, ball, tensors)
    (-torch.log(probs[0])).backward()
    assert tensors["region"].grad is not None
    assert float(tensors["region"].grad.abs().max()) > 0.0


def test_the_cuda_path_falls_back_cleanly_without_a_gpu():
    if torch.cuda.is_available():
        pytest.skip("CUDA present; the fallback path is not what runs here")
    assert load_fused_kernel() is None


def test_the_cuda_source_declares_what_the_loader_asks_for():
    source = Path("src/courtvision/kernels/possession.cu").read_text()
    for wanted in ("__global__ void possession_forward_kernel",
                   "__global__ void possession_backward_kernel",
                   "possession_forward", "possession_backward",
                   "PYBIND11_MODULE"):
        assert wanted in source


def test_the_kernel_has_no_warp_intrinsics_the_host_harness_cannot_stub():
    """verify_possession_numerics.py compiles this file as host C++.

    Warp intrinsics have no host equivalent, so using one would put the
    kernel's arithmetic beyond the reach of the only correctness check that
    runs without a GPU.
    """
    # The comments name these to explain why they are absent, so the check
    # reads the code and not the prose about the code.
    code = "\n".join(line for line in
                     Path("src/courtvision/kernels/possession.cu").read_text().split("\n")
                     if not line.lstrip().startswith("//"))
    for banned in ("__shfl", "atomicAdd", "__ballot", "cooperative_groups"):
        assert banned not in code


@pytest.mark.skipif(shutil.which("clang++") is None and shutil.which("g++") is None,
                    reason="no host C++ compiler")
def test_the_real_kernel_compiles_and_matches_the_oracle_on_the_host():
    """The whole of verify_possession_numerics, as a test."""
    result = subprocess.run([sys.executable, "scripts/verify_possession_numerics.py"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    assert "PASS" in result.stdout
