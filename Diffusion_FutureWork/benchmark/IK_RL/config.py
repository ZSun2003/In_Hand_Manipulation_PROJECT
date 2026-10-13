"""Configuration shared by Gripper3 residual training, evaluation and analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final


PACKAGE_DIR: Final = Path(__file__).resolve().parent
GRAB_DIR: Final = PACKAGE_DIR.parent
PARENT_MODEL_XML: Final = GRAB_DIR / "G3_Project.xml"
MODEL_XML: Final = PARENT_MODEL_XML
PARENT_CONTROLLER: Final = GRAB_DIR / "G3_grab.py"
PARENT_PLANNER: Final = GRAB_DIR / "_g3_objects.py"
PARENT_FRUIT_MANIFEST: Final = GRAB_DIR / "fruit_capsules_5mm.json"
PARENT_FRUIT_COLLISION_XMLS: Final = tuple(
    GRAB_DIR / f"fruit_capsules_5mm_{fruit}.xml"
    for fruit in ("banana", "pear", "strawberry")
)
PARENT_FRUIT_MANIFEST_SCHEMA: Final = "ycb-fruit-capsules-5mm-v1"

OBJECTS: Final = (
    "cube",
    "cylinder",
    "banana",
    "pear",
    "strawberry",
    "cone",
)
FRUITS: Final = frozenset(("banana", "pear", "strawberry"))
TASKS: Final = ("translate_xy45_100mm", "rotate_z_45deg")

POSITION_TOLERANCE_M: Final = 0.005
RIGID_ROTATION_TOLERANCE_DEG: Final = 5.0
FRUIT_ROTATION_TOLERANCE_DEG: Final = 10.0

ACTUATOR_DIM: Final = 12
PLAN_RESIDUAL_DIM: Final = 2
ACTION_DIM: Final = ACTUATOR_DIM + PLAN_RESIDUAL_DIM
RESIDUAL_MAX_RAD: Final = 0.005
RESIDUAL_RATE_LIMIT_RAD_PER_STEP: Final = 0.0001
RESIDUAL_TARGET_BLEND: Final = 0.35

# The final two normalized action values are consumed only by the planning
# handshake. The geometry planner selects a nominal from 48 candidates;
# height and yaw residuals offset that selected plan, whose yaw is not fixed.
GRASP_HEIGHT_RESIDUAL_MAX_M: Final = 0.020
GRASP_YAW_RESIDUAL_MAX_RAD: Final = math.radians(90.0)
PLAN_ZERO_EPS: Final = 1.0e-12

CONTROL_HZ: Final = 50.0
TRANSLATION_VECTOR_M: Final = (
    0.100 / math.sqrt(2.0),
    0.100 / math.sqrt(2.0),
    0.0,
)
ROTATION_ANGLE_RAD: Final = math.radians(45.0)

TOTAL_TIMESTEPS: Final = 2_000_000
SEED: Final = 7
N_ENVS: Final = 8
VEC_ENV_TYPE: Final = "subproc"
USE_VEC_NORMALIZE: Final = True
NORM_OBS: Final = True
NORM_REWARD: Final = True
CLIP_OBS: Final = 10.0
CLIP_REWARD: Final = 10.0

ALGORITHM: Final = "TQC"
POLICY_NET_ARCH: Final = (256, 256)
LEARNING_RATE: Final = 1.0e-4
BUFFER_SIZE: Final = 1_000_000
LEARNING_STARTS: Final = 20_000
BATCH_SIZE: Final = 256
TAU: Final = 0.005
GAMMA: Final = 0.99
TRAIN_FREQ: Final = 1
GRADIENT_STEPS: Final = 8
TQC_N_CRITICS: Final = 4
TQC_N_QUANTILES: Final = 32
TQC_TOP_QUANTILES_TO_DROP_PER_NET: Final = 2
TQC_USE_SDE: Final = True
TQC_SDE_SAMPLE_FREQ: Final = 64
MAX_GRAD_NORM: Final = 10.0

DEFAULT_MODEL_DIR: Final = PACKAGE_DIR / "models"
DEFAULT_EVAL_PATH: Final = PACKAGE_DIR / "evaluation" / "paired_eval.json"
EVAL_SCHEMA: Final = "ik-residual-paired-v1"
EVAL_EPISODES_PER_PAIR: Final = 20


def rotation_tolerance_deg(object_kind: str) -> float:
    value = str(object_kind)
    if value not in OBJECTS:
        raise KeyError(f"unknown evaluation object: {value!r}")
    return (
        FRUIT_ROTATION_TOLERANCE_DEG
        if value in FRUITS
        else RIGID_ROTATION_TOLERANCE_DEG
    )


def validate() -> None:
    parent_files = (
        PARENT_MODEL_XML,
        PARENT_CONTROLLER,
        PARENT_PLANNER,
        PARENT_FRUIT_MANIFEST,
        *PARENT_FRUIT_COLLISION_XMLS,
    )
    for path in parent_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            path.resolve().relative_to(PACKAGE_DIR.resolve())
        except ValueError:
            pass
        else:
            raise ValueError(f"parent contract path entered IK_RL: {path}")
    if len(OBJECTS) != len(set(OBJECTS)):
        raise ValueError("OBJECTS contains duplicates")
    if len(TASKS) != len(set(TASKS)):
        raise ValueError("TASKS contains duplicates")
    if ACTION_DIM != 14 or ACTUATOR_DIM != 12:
        raise ValueError("G3 action contract must be 12 actuator + 2 plan values")
    if TOTAL_TIMESTEPS != 2_000_000:
        raise ValueError("formal training must use exactly 2,000,000 steps")
    if not (0.0 < RESIDUAL_TARGET_BLEND <= 1.0):
        raise ValueError("invalid residual target blend")


__all__ = [name for name in globals() if name.isupper()] + [
    "rotation_tolerance_deg",
    "validate",
]
