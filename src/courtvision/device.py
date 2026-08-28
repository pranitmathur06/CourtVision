"""Single source of truth for which torch device to use.

This machine is an Apple M2 — there is no CUDA. `cuda` is checked anyway so the
same code runs unmodified on a rented GPU box for v2.
"""

from __future__ import annotations


def resolve_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
