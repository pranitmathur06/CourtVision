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
                                            possession_torch,
                                            temporal_reference,
                                            temporal_torch)

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


# ---- kernel 2: the scan over time -----------------------------------------

def scores_over_time(frames=7, states=5, seed=17):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.8, size=(frames, states)), float(rng.normal(0.4, 0.8))


def test_the_scan_agrees_with_the_torch_path():
    scores, stay = scores_over_time()
    expected, _, _ = temporal_reference(scores, stay, target=-1)
    got = temporal_torch(torch.tensor(scores), torch.tensor(stay, dtype=torch.float64))
    assert np.abs(expected - got.detach().numpy()).max() < 1e-9


def test_the_hand_derived_gradient_is_free_minus_clamped():
    """The backward is two expectations, not a differentiated recursion.

    -log(posterior) is logZ - logZ_clamped, so its gradient is the difference of
    two log-partition gradients. Autograd through the portable scan is the
    independent witness; deriving it a second time by hand would only prove the
    same mistake twice.
    """
    scores, stay = scores_over_time()
    target = 2
    _, loss, grads = temporal_reference(scores, stay, target)
    tensor = torch.tensor(scores, requires_grad=True)
    raw = torch.tensor(stay, dtype=torch.float64, requires_grad=True)
    value = -torch.log(temporal_torch(tensor, raw)[target])
    value.backward()
    assert abs(loss - float(value.detach())) < 1e-8
    assert np.abs(grads["scores"] - tensor.grad.numpy()).max() < 1e-7
    assert abs(grads["stay_raw"] - float(raw.grad)) < 1e-7


def test_the_posterior_is_a_distribution_and_the_centre_is_what_is_read():
    scores, stay = scores_over_time()
    posterior, _, _ = temporal_reference(scores, stay, target=-1, centre=3)
    assert posterior.shape == (scores.shape[1],)
    assert posterior.min() >= 0.0
    assert abs(posterior.sum() - 1.0) < 1e-12


def test_a_zero_stay_bonus_leaves_every_frame_independent():
    """With no reward for keeping the ball the scan must do nothing at all.

    softplus(x) -> 0 as x -> -inf, and a transition matrix of all zeros makes
    the posterior at the centre the centre frame's own softmax. If the scan
    changed the answer here it would be adding something that is not in the
    model.
    """
    scores, _ = scores_over_time()
    posterior, _, _ = temporal_reference(scores, -60.0, target=-1, centre=3)
    alone = np.exp(scores[3] - scores[3].max())
    assert np.abs(posterior - alone / alone.sum()).max() < 1e-9


def test_a_large_stay_bonus_makes_one_state_win_every_frame():
    """Turned up, the scan should answer with the best state OVER THE WINDOW.

    This is the behaviour the kernel exists for: a frame where the evidence is
    poor gets the answer from its neighbours instead.
    """
    scores = np.array([[0.0, 3.0], [0.0, 3.0], [0.2, 0.0], [0.0, 3.0],
                       [0.0, 3.0], [0.0, 3.0], [0.0, 3.0]])
    alone = int(np.argmax(scores[2]))
    posterior, _, _ = temporal_reference(scores, 20.0, target=-1, centre=2)
    assert alone == 0                       # the centre frame on its own says 0
    assert int(np.argmax(posterior)) == 1   # the window says 1, and it is right


def test_the_stay_bonus_can_never_punish_keeping_the_ball():
    """softplus, so the transition is a bonus by construction.

    A negative stay bonus would be a model that believes the ball changes hands
    more often than it is kept, which is false in every broadcast, and training
    could still wander there on 314 frames.
    """
    from courtvision.kernels.possession import softplus
    assert float(softplus(-50.0)) >= 0.0
    assert float(softplus(0.0)) > 0.0
    assert abs(float(softplus(30.0)) - 30.0) < 1e-6


def test_the_cuda_source_declares_the_temporal_entry_points():
    source = Path("src/courtvision/kernels/possession.cu").read_text()
    for wanted in ("__global__ void temporal_kernel", "temporal_forward",
                   "forward_backward", "block_max", "block_sum"):
        assert wanted in source
