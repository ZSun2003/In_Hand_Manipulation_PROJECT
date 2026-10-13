"""Authoritative experiment constants for the local G3/G6 project."""

from __future__ import annotations

from enum import IntEnum, StrEnum
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_VENV = PROJECT_ROOT.parent / "RL_env"
MODEL_XML = {
    "g3": PROJECT_ROOT / "model" / "g3" / "Project.xml",
    "g6": PROJECT_ROOT / "model" / "g6" / "Project.xml",
}


class GripperKind(StrEnum):
    G3 = "g3"
    G6 = "g6"


class Phase(IntEnum):
    GRASP_LIFT = 1
    TRANSLATE = 2
    ROTATE = 3


OBJECT_NAMES = ("cube", "cylinder", "triangular_prism", "cone")
ROOT_HEIGHT_M = 0.525
TABLE_TOP_M = 0.260
TARGET_GRASP_PLANE_Z_M = 0.360
# Phase 3 deliberately rotates closer to the table.  Keep this separate from
# the grasp/lift and translation target so their established 0.360 m contract
# does not change.
ROTATION_TARGET_GRASP_PLANE_Z_M = 0.325
OBJECT_MASS_KG = 0.200
HARD_TIP_RADIUS_M = 0.016
CONTACT_INDENTATION_M = 0.0015
PREGRASP_CLEARANCE_M = 0.025
# Small controller calibration offsets compensate the measured steady-state
# lift bias caused by the two grippers' different closed-chain compliance.
NOMINAL_GRASP_Z_BIAS_M = {"g3": -0.002, "g6": -0.004}

CONTROL_HZ = 50
CONTROL_DT_S = 1.0 / CONTROL_HZ
PHYSICS_TIMESTEP_S = 0.0005
PHYSICS_SUBSTEPS = int(round(CONTROL_DT_S / PHYSICS_TIMESTEP_S))
NO_SLIP_ITERATIONS = 10

# The two grippers have different native actuator counts, so they must be
# trained with separate policy heads.  ``ACTION_DIM`` remains a G3-compatible
# alias for older callers; new code must use :func:`action_dim`.
G3_ACTION_DIM = 12
G6_ACTION_DIM = 16
ACTION_DIM = G3_ACTION_DIM
ACTION_DIMS = {"g3": G3_ACTION_DIM, "g6": G6_ACTION_DIM}
MAX_ACTION_DIM = max(ACTION_DIMS.values())
G3_ANGULAR_ACTUATOR_INDICES = tuple(range(12))
G6_ANGULAR_ACTUATOR_INDICES = (1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15)
ANGULAR_ACTUATOR_INDICES = {
    "g3": G3_ANGULAR_ACTUATOR_INDICES,
    "g6": G6_ANGULAR_ACTUATOR_INDICES,
}
G6_LINEAR_ACTUATOR_INDICES = (0, 3, 6, 9)
RESIDUAL_ACTUATOR_INDICES = {
    "g3": tuple(range(G3_ACTION_DIM)),
    "g6": tuple(range(G6_ACTION_DIM)),
}

# Residual authority matches the established G3/G6 translation controllers.
# G3 exposes twelve rotary actuators.  G6 additionally exposes four linear
# slides, whose action components are expressed in metres rather than radians.
ANGULAR_RESIDUAL_LIMIT_RAD = 0.005
ANGULAR_RESIDUAL_RATE_RAD_PER_TICK = 0.0001
ANGULAR_RESIDUAL_BLEND = 0.35
LINEAR_RESIDUAL_LIMIT_M = 0.0005
LINEAR_RESIDUAL_RATE_M_PER_TICK = 0.00002
LINEAR_RESIDUAL_BLEND = 0.35

# Fixed-step phases.  There is intentionally no success-rate admission gate.
PHASE_TIMESTEPS = {
    Phase.GRASP_LIFT: 500_000,
    Phase.TRANSLATE: 2_000_000,
    Phase.ROTATE: 2_000_000,
}

POSITION_TOLERANCE_M_PER_AXIS = 0.005
ANGLE_TOLERANCE_DEG_PER_AXIS = 5.0
ROTATION_ANGLE_MIN_DEG = 10
ROTATION_ANGLE_MAX_DEG = 45
# Explicit opt-in ceiling.  The default/random task remains 10--45 deg; this
# wider bound is used by the 0--90 deg pure-IK benchmark and the dedicated
# weighted rotation-residual training profile only.
ROTATION_SWEEP_ANGLE_MAX_DEG = 90.0
ROTATION_AXIS_SETS = ("x", "y", "z", "xy", "xz", "yz", "xyz")
DEFAULT_ROTATION_SAMPLING_PROFILE = "default_10_45"
WEIGHTED_ROTATION_SAMPLING_PROFILE = "weighted_0_90"
ROTATION_SAMPLING_PROFILES = (
    DEFAULT_ROTATION_SAMPLING_PROFILE,
    WEIGHTED_ROTATION_SAMPLING_PROFILE,
)
WEIGHTED_ROTATION_MAGNITUDE_BINS_DEG = ((0, 30), (31, 60), (61, 90))
WEIGHTED_ROTATION_BIN_PROBABILITIES = (0.20, 0.50, 0.30)

# Every phase uses the same grasp/lift prefix.  Phase 2/3 begin their action
# on the very next tick after lift completion: there is no post-lift wait.
APPROACH_TICKS = 120
CONTACT_TICKS = 80
LIFT_TICKS = 160
PRE_ACTION_HOLD_TICKS = 0
PRE_ACTION_TICKS = APPROACH_TICKS + CONTACT_TICKS + LIFT_TICKS
# The upward-tapered cone needs a short desired-frame lift to avoid losing its
# shrinking cross-section.  This is a nominal cage calibration relative to the
# active phase's task height, not a second absolute task target.
CONE_FAST_LIFT_TICKS = 10
CONE_NOMINAL_CAGE_Z_OFFSET_M = -0.005
POST_ACTION_HOLD_TICKS = 25
PHASE1_TERMINAL_HOLD_TICKS = POST_ACTION_HOLD_TICKS
PHASE1_EPISODE_TICKS = PRE_ACTION_TICKS + PHASE1_TERMINAL_HOLD_TICKS
ACTION_MOTION_TICKS = 2_400
ACTION_SCHEDULE_TICKS = ACTION_MOTION_TICKS + POST_ACTION_HOLD_TICKS
FORMAL_EPISODE_TICKS = PRE_ACTION_TICKS + ACTION_SCHEDULE_TICKS

# Once the scheduled lift is complete, a grasp frame below one quarter of its
# object-specific lift distance is unambiguously back near the table.  A
# single absolute world-Z threshold cannot detect this because all four rest
# grasp planes are naturally above the table at different heights.
DROP_MIN_LIFT_FRACTION = 0.25
MAX_ABS_QVEL = 200.0
MAX_OBJECT_LINEAR_SPEED_M_S = 5.0
MAX_OBJECT_ANGULAR_SPEED_RAD_S = 100.0
REQUIRED_TIP_CONTACTS = 4

# Dense reward scales.  Endpoint success is reported, but never selects the
# next training phase or truncates the fixed training budget.
POSITION_REWARD_SCALE_M = 0.025
ORIENTATION_REWARD_SCALE_DEG = 20.0
CONTACT_REWARD_WEIGHT = 0.25
HEIGHT_REWARD_WEIGHT = 0.75
# Match the two translation reference controllers' different regularizers.
# G3 applies its coefficient to normalized RMS residual; G6 applies its
# coefficient to normalized mean-square residual.
G3_RESIDUAL_PENALTY_WEIGHT = 0.01
G6_RESIDUAL_PENALTY_WEIGHT = 0.002
RESIDUAL_PENALTY_WEIGHTS = {
    "g3": G3_RESIDUAL_PENALTY_WEIGHT,
    "g6": G6_RESIDUAL_PENALTY_WEIGHT,
}
# Backward-compatible scalar for historical manifest readers.  New code must
# use ``RESIDUAL_PENALTY_WEIGHTS[gripper]``.
RESIDUAL_PENALTY_WEIGHT = G3_RESIDUAL_PENALTY_WEIGHT
# Conservative TQC schedule.  The critic first learns around the nominal
# controller from tiny, safe action perturbations; actor updates begin only
# after that local action neighbourhood is represented in replay.
TQC_CRITIC_LEARNING_STARTS = 20_000
TQC_ACTOR_LEARNING_STARTS = 40_000
TQC_SAFE_ACTION_LIMIT = 1.0
TQC_ACTOR_RESIDUAL_L2_WEIGHT = 1.0
TQC_PHASE_ACTION_TRUST_SCALE = {
    Phase.GRASP_LIFT: 0.02,
    Phase.TRANSLATE: 0.20,
    Phase.ROTATE: 0.50,
}
TQC_PHASE_ACTOR_DELAY_STEPS = {
    Phase.GRASP_LIFT: 40_000,
    Phase.TRANSLATE: 20_000,
    Phase.ROTATE: 20_000,
}
ENDPOINT_SUCCESS_BONUS = 20.0

DEFAULT_SEED = 7
DEFAULT_N_ENVS = 4


def action_dim(gripper: str | GripperKind) -> int:
    """Return the policy action width for one independently trained gripper."""

    return ACTION_DIMS[GripperKind(gripper).value]


def validate_contract() -> None:
    if (
        ROOT_HEIGHT_M != 0.525
        or TARGET_GRASP_PLANE_Z_M != 0.360
        or ROTATION_TARGET_GRASP_PLANE_Z_M != 0.325
    ):
        raise AssertionError("root/grasp-plane contract changed")
    if PHASE_TIMESTEPS != {
        Phase.GRASP_LIFT: 500_000,
        Phase.TRANSLATE: 2_000_000,
        Phase.ROTATE: 2_000_000,
    }:
        raise AssertionError("formal phase budgets changed")
    if PHYSICS_SUBSTEPS != 40:
        raise AssertionError("50 Hz control must contain 40 x 0.5 ms steps")
    if ACTION_DIMS != {"g3": 12, "g6": 16} or MAX_ACTION_DIM != 16:
        raise AssertionError("native residual action dimensions changed")
    if RESIDUAL_ACTUATOR_INDICES != {
        "g3": tuple(range(12)),
        "g6": tuple(range(16)),
    }:
        raise AssertionError("native residual actuator mapping changed")
    if (
        ANGULAR_RESIDUAL_LIMIT_RAD != 0.005
        or ANGULAR_RESIDUAL_RATE_RAD_PER_TICK != 0.0001
        or ANGULAR_RESIDUAL_BLEND != 0.35
        or LINEAR_RESIDUAL_LIMIT_M != 0.0005
        or LINEAR_RESIDUAL_RATE_M_PER_TICK != 0.00002
        or LINEAR_RESIDUAL_BLEND != 0.35
    ):
        raise AssertionError("translation residual envelope changed")
    if (
        WEIGHTED_ROTATION_MAGNITUDE_BINS_DEG
        != ((0, 30), (31, 60), (61, 90))
        or WEIGHTED_ROTATION_BIN_PROBABILITIES != (0.20, 0.50, 0.30)
    ):
        raise AssertionError("weighted rotation sampling contract changed")
    if PRE_ACTION_HOLD_TICKS < 0:
        raise AssertionError("invalid pre-action schedule")
    if PRE_ACTION_TICKS != 360:
        raise AssertionError("Phase 2/3 action must start immediately after lift")
    if PHASE1_EPISODE_TICKS != 385 or FORMAL_EPISODE_TICKS != 2_785:
        raise AssertionError("25-tick terminal hold contract changed")
    for path in MODEL_XML.values():
        path.resolve().relative_to(PROJECT_ROOT.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)


def componentwise_pose_within_tolerance(
    position_error_m: object,
    angle_error_deg: object,
) -> bool:
    position = np.asarray(position_error_m, dtype=np.float64).reshape(3)
    angle = np.asarray(angle_error_deg, dtype=np.float64).reshape(3)
    return bool(
        np.all(np.isfinite(position))
        and np.all(np.isfinite(angle))
        and np.all(np.abs(position) <= POSITION_TOLERANCE_M_PER_AXIS)
        and np.all(np.abs(angle) <= ANGLE_TOLERANCE_DEG_PER_AXIS)
    )


validate_contract()
