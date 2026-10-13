"""Fixed experiment contract for the G6 IK + residual comparison."""

from __future__ import annotations

import math
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
PARENT_MODEL_XML = PROJECT_ROOT / "_g6_project.xml"
MODEL_XML = PARENT_MODEL_XML
PARENT_CONTROLLER = PROJECT_ROOT / "_g6_grab.py"
PARENT_PLANNER = PROJECT_ROOT / "_g6_objects.py"
PARENT_FRUIT_MANIFEST = PROJECT_ROOT / "fruit_capsules_5mm.json"
PARENT_FRUIT_COLLISION_XMLS = tuple(
    PROJECT_ROOT / f"fruit_capsules_5mm_{fruit}.xml"
    for fruit in ("banana", "pear", "strawberry")
)
PARENT_FRUIT_MANIFEST_SCHEMA = "ycb-fruit-capsules-5mm-v1"


OBJECTS = (
    "cube",
    "cylinder",
    "banana",
    "pear",
    "strawberry",
    "cone",
)
PLATE_OBJECT = "plate"

TASK_TRANSLATE = "translate_xy45_100mm"
TASK_ROTATE = "rotate_z_45deg"
TASKS = (TASK_TRANSLATE, TASK_ROTATE)

ACTION_DIM = 16
LINEAR_ACTUATOR_INDICES = (0, 3, 6, 9)
HINGE_ACTUATOR_INDICES = tuple(
    index
    for index in range(ACTION_DIM)
    if index not in LINEAR_ACTUATOR_INDICES
)

RESIDUAL_MAX_RAD = 0.005
RESIDUAL_MAX_LINEAR_M = 0.0005
RESIDUAL_RATE_LIMIT_RAD_PER_STEP = 0.0001
RESIDUAL_RATE_LIMIT_LINEAR_M_PER_STEP = 0.00002
RESIDUAL_TARGET_BLEND = 0.35

POSITION_TOLERANCE_M = 0.005
RIGID_ROTATION_TOLERANCE_DEG = 5.0
FRUIT_ROTATION_TOLERANCE_DEG = 10.0
FRUIT_OBJECTS = frozenset(("banana", "pear", "strawberry"))
RIGID_OBJECTS = frozenset(("cube", "cylinder", "cone"))

CONTROL_HZ = 50.0
TRANSLATION_VECTOR_M = (
    0.100 / math.sqrt(2.0),
    0.100 / math.sqrt(2.0),
    0.0,
)
ROTATION_ANGLE_RAD = math.radians(45.0)

TOTAL_TIMESTEPS = 2_000_000
DEFAULT_SEED = 7
N_ENVS = 8

# Reward is dense throughout the fixed current-controller workflow.  Endpoint
# classification alone defines reported success.
POSITION_REWARD_SCALE_M = 0.025
ORIENTATION_REWARD_SCALE_DEG = 20.0
RESIDUAL_PENALTY_WEIGHT = 0.002
ENDPOINT_SUCCESS_BONUS = 20.0
ENDPOINT_FAILURE_PENALTY = 20.0


def rotation_tolerance_deg(object_kind: str) -> float:
    """Return the object-specific wrapped fixed-XYZ Euler L2 threshold."""
    if object_kind in FRUIT_OBJECTS:
        return FRUIT_ROTATION_TOLERANCE_DEG
    if object_kind in RIGID_OBJECTS:
        return RIGID_ROTATION_TOLERANCE_DEG
    raise ValueError(f"unsupported training object: {object_kind!r}")


def endpoint_success(
    object_kind: str,
    position_error_m: float,
    euler_l2_error_deg: float,
) -> bool:
    """Classify an endpoint using only the requested pose thresholds."""
    return bool(
        math.isfinite(position_error_m)
        and math.isfinite(euler_l2_error_deg)
        and position_error_m <= POSITION_TOLERANCE_M
        and euler_l2_error_deg <= rotation_tolerance_deg(object_kind)
    )


def validate_parent_paths() -> None:
    """Require every shared model artifact to live in the parent project."""
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
