"""The CUDA kernel's arithmetic is checkable on the host; do it in CI."""

import shutil

import pytest


@pytest.mark.skipif(
    shutil.which("clang++") is None and shutil.which("g++") is None,
    reason="no host C++ compiler",
)
def test_cuda_kernel_compiles_and_matches_opencv():
    """Catches a broken kernel before anyone pays for a GPU to find out.

    nvcc does not exist for Apple Silicon, so torso_color.cu is otherwise
    unbuilt until it reaches rented hardware. Host stubs plus real threads and
    a real barrier run the actual source, so this covers the colour transform,
    the channel order, the Lab scaling AND the shared-memory reduction.
    """
    from scripts.verify_kernel_numerics import main

    assert main() == 0
