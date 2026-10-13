"""Shared Gripper3/Gripper6 analytic-IK manipulation package."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = _PROJECT_ROOT / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "torch"))
if sys.pycache_prefix is None:
    sys.pycache_prefix = str(_CACHE_ROOT / "pycache")

from .config import GripperKind, Phase


def __getattr__(name: str) -> Any:
    if name == "G3G6ManipulationEnv":
        from .env import G3G6ManipulationEnv

        return G3G6ManipulationEnv
    raise AttributeError(name)


__all__ = ["G3G6ManipulationEnv", "GripperKind", "Phase"]
