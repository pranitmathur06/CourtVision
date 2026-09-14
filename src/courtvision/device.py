"""Single source of truth for which torch device to use.

CUDA is preferred when it exists. This machine is an Apple M2 and falls back to
MPS, but the ordering matters for v2 on a rented GPU box: a discrete CUDA card
beats MPS decisively, and checking MPS first would have silently kept the slow
path on any machine that somehow offered both.

Half precision is a separate question from the device, and it lives in
`fast_detect.half_precision_ok`: CUDA benefits, MPS does not uniformly, and
every timing in this repository was taken on MPS in fp32.
"""

from __future__ import annotations


def resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe() -> str:
    """One line naming the device and what it is, for a run's own log."""
    import torch

    device = resolve_device()
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"cuda: {name}, {total:.0f} GB, {torch.cuda.device_count()} device(s)"
    if device == "mps":
        return "mps: Apple unified memory, fp32 (fp16 is not uniformly faster here)"
    return "cpu"
