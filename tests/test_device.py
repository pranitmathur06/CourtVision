"""Which device is chosen, and that CUDA wins when it is there."""

from __future__ import annotations

import sys
import types

from courtvision.device import resolve_device


def _torch(cuda: bool, mps: bool):
    module = types.SimpleNamespace()
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps))
    return module


def test_cuda_is_preferred_over_mps(monkeypatch):
    # The ordering that matters for a rented GPU box: a discrete card beats MPS.
    monkeypatch.setitem(sys.modules, "torch", _torch(cuda=True, mps=True))
    assert resolve_device() == "cuda"


def test_mps_is_used_when_there_is_no_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch(cuda=False, mps=True))
    assert resolve_device() == "mps"


def test_cpu_is_the_last_resort(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch(cuda=False, mps=False))
    assert resolve_device() == "cpu"
