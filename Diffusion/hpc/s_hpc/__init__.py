"""Native CUDA/MJWarp implementation of the SCREW task.

The package deliberately has no import-time dependency on CUDA, MuJoCo Warp,
or the parent ``SCREW`` directory.  Static validation and checkpoint
inspection therefore work on the user's Apple Silicon machine, while runtime
objects fail early with a precise message if they are not launched on a Linux
NVIDIA node.
"""

from .config import HPCConfig, load_config
from .contracts import (
    ACTION_DIM,
    DIFFUSION_SCHEMA_HASH,
    OBS_DIM,
    OBS_HORIZON,
    RESIDUAL_ACTION_DIM,
    SCHEMA_HASH,
    TQC_CONTROLLER_STATE_NAMES,
    TQC_OBS_DIM,
    TQC_SCHEMA_HASH,
)

__all__ = [
    "ACTION_DIM",
    "DIFFUSION_SCHEMA_HASH",
    "HPCConfig",
    "OBS_DIM",
    "OBS_HORIZON",
    "RESIDUAL_ACTION_DIM",
    "SCHEMA_HASH",
    "TQC_CONTROLLER_STATE_NAMES",
    "TQC_OBS_DIM",
    "TQC_SCHEMA_HASH",
    "load_config",
]
