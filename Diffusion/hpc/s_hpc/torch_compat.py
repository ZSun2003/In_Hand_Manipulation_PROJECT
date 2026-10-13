"""Narrow compatibility scopes for the pinned PyTorch CUDA runtime."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def unique_cuda_index_writes(torch, *, device_type: str) -> Iterator[None]:
    """Avoid PyTorch 2.7.1's deterministic CUDA advanced-index assertion.

    The reset path writes each selected world/field exactly once, so its
    result is deterministic even through PyTorch's regular CUDA indexing
    implementation.  PyTorch 2.7.1's alternate deterministic implementation
    can assert internally for a vector RHS on Blackwell.  Disable only the
    dispatcher flag around these disjoint writes and restore its exact prior
    mode immediately afterwards.
    """

    enabled = bool(torch.are_deterministic_algorithms_enabled())
    if device_type != "cuda" or not enabled:
        yield
        return

    warn_only_getter = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        None,
    )
    warn_only = bool(warn_only_getter()) if warn_only_getter else False
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
