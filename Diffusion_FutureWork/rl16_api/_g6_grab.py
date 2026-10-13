#!/usr/bin/env python3
"""Headless analytic controller used by archived YCB grasp training.

Contains the original grasp, translation and rotation control logic consumed
by IK_RL and MJWarp_RL. Interactive menus, viewers and image export are removed.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
LOCAL_ENV_DIR = Path(os.environ.get("IRP_PYTHON_ENV", str(Path(__file__).resolve().parents[2] / "RL_env")))
MODEL_XML = ROOT_DIR / "_g6_project.xml"







import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import statistics
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import mujoco
import numpy as np

from _g6_kinematics import (
    ACTUATOR_NAMES,
    FINGERS,
    LINEAR_CTRL_INDICES,
    IKTeacher,
    IKSolution,
    JOINT_KINDS,
    GripperIK,
    minimum_jerk,
    quat_to_euler_world_xyz_deg,
    quat_to_rot,
    rot_to_quat,
)
from _g6_objects import (
    OBJECT_KIND_ORDER,
    AdaptiveGraspPlanePlanner,
    DIRECT_GRASP_PLANE_HEIGHT_FRACTION,
    FRUIT_CAPSULE_MANIFEST,
    FRUIT_CAPSULE_MANIFEST_PATH,
    FRUIT_CAPSULE_MANIFEST_SHA256,
    FRUIT_CAPSULE_SCHEMA_VERSION,
    FRUIT_KINDS,
    GraspPlan,
    ObjectKind,
    ObjectSpec,
    get_object_spec,
)


def _fingerprint_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (candidate.resolve() for candidate in paths),
        key=lambda candidate: str(candidate.relative_to(ROOT_DIR)),
    ):
        relative = path.relative_to(ROOT_DIR)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


MODEL_DEPENDENCY_PATHS = (
    MODEL_XML,
    ROOT_DIR / "_g6_share.xml",
    ROOT_DIR / "_g6_gripper.xml",
    ROOT_DIR / "_g6_object_assets.xml",
    ROOT_DIR / "fruit_capsules_5mm_banana.xml",
    ROOT_DIR / "fruit_capsules_5mm_pear.xml",
    ROOT_DIR / "fruit_capsules_5mm_strawberry.xml",
    ROOT_DIR / "fruit_capsules_5mm_orange.xml",
    *tuple(
        sorted(
            path
            for path in (ROOT_DIR / "mesh").rglob("*")
            if path.is_file()
        )
    ),
)
MODEL_FINGERPRINT_SHA256 = _fingerprint_files(MODEL_DEPENDENCY_PATHS)
CONTROLLER_FINGERPRINT_SHA256 = _fingerprint_files(
    (
        Path(__file__),
        ROOT_DIR / "_g6_kinematics.py",
        ROOT_DIR / "_g6_objects.py",
        FRUIT_CAPSULE_MANIFEST_PATH,
    )
)




# Fixed model/task contract.
ROOT_HEIGHT_M = 0.560
TABLE_TOP_Z_M = 0.260
LIFT_HEIGHT_M = 0.050
START_JITTER_M = 0.003
CYLINDER_SYMMETRY_NUDGE_M = 0.0001
RESET_DROP_HEIGHT_M = 0.002
RESET_SETTLE_STEPS = 1000

# Native MuJoCo controller.
CONTROL_HZ = 50.0
CONTROL_DT_S = 1.0 / CONTROL_HZ
MODEL_TIMESTEP_S = 0.0005
PHYSICS_STEPS_PER_CONTROL = round(CONTROL_DT_S / MODEL_TIMESTEP_S)
MAX_CTRL_DELTA_RAD_PER_STEP = 0.055
MAX_LINEAR_CTRL_DELTA_M_PER_STEP = 0.002
CTRL_FILTER_ALPHA = 0.65

# Complete action: approach, grasp, settle, lift, then verify the hold.
PLANE_SELECTION_STEPS = 8
CLOSURE_STEPS = 90
CONTACT_SETTLE_STEPS = 60
PRELIFT_REQUIRED_CONTACT_STEPS = 5
MAX_PRELIFT_HORIZONTAL_DISPLACEMENT_M = 0.006
MAX_PRELIFT_ORIENTATION_DRIFT_RAD = math.radians(8.0)
LIFT_STEPS = 70
DEMO_ROTATION_ANGLE_RAD = math.radians(45.0)
DEMO_CAGE_PROPORTIONAL_GAIN = 0.025
DEMO_CAGE_MAX_STEP_RAD = math.radians(0.12)
DEMO_CAGE_LIMIT_RAD = math.radians(70.0)
DEMO_MAX_STEPS_PER_AXIS = 1000
DEMO_AXIS_STABLE_STEPS = 30
DEMO_AXIS_TOLERANCE_RAD = math.radians(2.0)
DEMO_ANGULAR_SPEED_TOLERANCE_RAD_S = 0.05
DEMO_CONTACT_LOSS_FAILURE_STEPS = 1
DEMO_NON_TIP_CONTACT_FAILURE_STEPS = 1
DEMO_MIN_REQUIRED_CONTACTS = len(FINGERS)
DEMO_ROTATION_TARGET_TOLERANCE_RAD = math.radians(12.0)
DEMO_CENTER_DRIFT_LIMIT_M = 0.025

# Post-lift translation capability.  The public demo keeps its historical
# 100 mm / 45-degree default, while callers may request any horizontal vector
# from 5 mm through 100 mm.  Quantized RL goals are represented by integer
# millimetres and integer degrees before conversion to Cartesian metres.
MIN_TRANSLATION_DISTANCE_M = 0.005
MAX_TRANSLATION_DISTANCE_M = 0.100
TRANSLATION_RADIUS_RESOLUTION_M = 0.001
TRANSLATION_THETA_RESOLUTION_DEG = 1
DEMO_TRANSLATION_DISTANCE_M = MAX_TRANSLATION_DISTANCE_M
DEMO_TRANSLATION_COMPONENT_M = (
    DEMO_TRANSLATION_DISTANCE_M / math.sqrt(2.0)
)
DEMO_TRANSLATION_VECTOR_M = (
    DEMO_TRANSLATION_COMPONENT_M,
    DEMO_TRANSLATION_COMPONENT_M,
    0.0,
)
DEMO_TRANSLATION_STEPS = 250
# The action endpoint is the end of the fixed minimum-jerk trajectory.  There
# is no additional settling gate because velocity is not a success criterion.
DEMO_TRANSLATION_MAX_HOLD_STEPS = 0
DEMO_FORCE_RAMP_STEPS = 50
DEMO_TRANSLATION_STABLE_STEPS = 30
DEMO_TRANSLATION_FEEDBACK_GAIN = 1.00
DEMO_TRANSLATION_FEEDBACK_LIMIT_M = 0.025
DEMO_TRANSLATION_GOAL_TOLERANCE_M = 0.012
DEMO_TRANSLATION_MIN_ALONG_TRACK_M = 0.095
DEMO_TRANSLATION_CROSS_TRACK_TOLERANCE_M = 0.012
DEMO_TRANSLATION_ORIENTATION_TOLERANCE_RAD = math.radians(10.0)
DEMO_TRANSLATION_ABORT_ORIENTATION_RAD = math.radians(25.0)
DEMO_TRANSLATION_LINEAR_SPEED_TOLERANCE_M_S = 0.020
DEMO_TRANSLATION_CONTACT_LOSS_FAILURE_STEPS = 1
DEMO_TRANSLATION_NON_TIP_CONTACT_FAILURE_STEPS = 1
DEMO_TRANSLATION_RENDER_FINAL_PAUSE_S = 0.8
HOLD_STEPS = 20
# Motion demos do not score pose or speed during grasp/lift.  They do require
# a bounded, mechanically valid static-pinch window before capturing the
# action reference; the speed values remain diagnostic telemetry.
MOTION_PRECONDITION_MAX_HOLD_STEPS = 70
MOTION_PRECONDITION_REQUIRED_STABLE_STEPS = 10
MOTION_PRECONDITION_LINEAR_SPEED_M_S = 0.003
MOTION_PRECONDITION_ANGULAR_SPEED_RAD_S = 0.10
MAX_TRIAL_STEPS = (
    PLANE_SELECTION_STEPS
    + CLOSURE_STEPS
    + CONTACT_SETTLE_STEPS
    + LIFT_STEPS
    + MOTION_PRECONDITION_MAX_HOLD_STEPS
    + 30
)

# Geometry and safety.
HARD_TIP_RADIUS_M = 0.016
PREGRASP_CLEARANCE_M = 0.012
TIP_TABLE_CLEARANCE_M = 0.002
CONTACT_FORCE_THRESHOLD_N = 0.035
CONTACT_LOCK_FORCE_THRESHOLD_N = 0.300
PINCH_MIN_FORCE_PER_TIP_N = 0.300
PINCH_MIN_PLANNED_NORMAL_COSINE = math.cos(math.radians(35.0))
PINCH_MAX_PLANNED_REGION_ERROR_M = 0.020
PINCH_MIN_SIDE_MARGIN_M = 0.0
PINCH_MAX_OPPOSING_NORMAL_DOT = -0.5
PINCH_MIN_OPPOSING_FORCE_RATIO = 0.25
STATIC_PINCH_MAX_UPWARD_NORMAL_WEIGHT_RATIO = 0.20
CONTACT_PRELOAD_M = 0.0008
PEAR_CONTACT_PRELOAD_M = 0.0003
CLOSE_COMPRESSION_M = 0.0
SETTLE_COMPRESSION_M = 0.0
CLOSE_COMPRESSION_BY_OBJECT_M: Dict[ObjectKind, float] = {}
SETTLE_COMPRESSION_BY_OBJECT_M: Dict[ObjectKind, float] = {}
LIFT_CONTACT_COMPRESSION_M: Dict[ObjectKind, float] = {}
TARGET_RUBBER_CONTACT_FORCE_N = 3.0
# Composite motion needs a larger friction reserve than the short formal lift.
# This changes preload only; the rubber contact compliance remains unchanged.
DEMO_TARGET_RUBBER_CONTACT_FORCE_N = 8.0
PAIR_FORCE_BALANCE_GAIN_M_PER_N = 0.000030
PAIR_PRELOAD_GAIN_M_PER_N = 0.000030
MAX_CONTACT_CENTER_TRIM_M = 0.012
MIN_PAIR_COMPRESSION_M = -0.0100
MAX_PAIR_COMPRESSION_M = 0.0100
FORMATION_COMPENSATION_GAIN = 0.30
BANANA_FORMATION_COMPENSATION_GAIN = 0.80
CONE_FORMATION_COMPENSATION_GAIN = 0.40
MAX_FORMATION_COMPENSATION_M = 0.008
POSE_STABILIZED_OBJECTS = FRUIT_KINDS
LIFT_FEEDBACK_OBJECTS = FRUIT_KINDS | frozenset(
    (ObjectKind.CONE,)
)
LIFT_POSITION_FEEDBACK_GAIN = 0.8
LIFT_POSITION_FEEDBACK_XY_LIMIT_M = 0.006
LIFT_POSITION_FEEDBACK_Z_LIMIT_M = 0.008
BANANA_LIFT_POSITION_FEEDBACK_GAIN = 0.83
BANANA_LIFT_POSITION_FEEDBACK_XY_LIMIT_M = 0.008
CONE_LIFT_POSITION_FEEDBACK_GAIN = 2.0
CONE_LIFT_POSITION_FEEDBACK_Z_LIMIT_M = 0.050
LIFT_ORIENTATION_FEEDBACK_GAIN = {
    ObjectKind.BANANA: 0.30,
    ObjectKind.PEAR: 0.40,
    ObjectKind.STRAWBERRY: 0.15,
    ObjectKind.ORANGE: 0.15,
}
LIFT_ORIENTATION_FEEDBACK_LIMIT_RAD = math.radians(10.0)
MIN_REQUIRED_CONTACTS = len(FINGERS)
TIP_COPLANARITY_TOLERANCE_M = 0.00075
TIP_PARALLELOGRAM_TOLERANCE_M = 0.00100
MAX_CONTACT_ONSET_SPAN_STEPS = 1
CONTACT_LOSS_FAILURE_STEPS = 1
MAX_ORIENTATION_DRIFT_RAD = math.radians(18.0)
DROP_MARGIN_M = 0.010
SUCCESS_POSITION_TOLERANCE_M = 0.012
SUCCESS_LINEAR_SPEED_M_S = 0.020
SUCCESS_ORIENTATION_DRIFT_RAD = math.radians(8.0)
MIN_SUCCESS_LIFT_M = 0.045
MAX_SUCCESS_HORIZONTAL_DRIFT_M = 0.006

OBJECT_NAMES = tuple(kind.value for kind in OBJECT_KIND_ORDER)
OBJECT_LABELS = {
    "cube": "立方体 cube",
    "cylinder": "圆柱 cylinder",
    "banana": "香蕉 banana",
    "pear": "梨 pear",
    "strawberry": "草莓 strawberry",
    "orange": "橙子 orange",
    "cone": "圆锥 cone",
    "plate": "薄片 plate",
}
THIN_OBJECT_FAILURE_REASON = "no_safe_grasp_plane_for_thin_object"
ROTATION_DEMO_CHOICES = ("x", "y", "z", "xy", "xz", "yz", "xyz")
ROTATION_AXIS_LABELS = {
    "x": "X",
    "y": "Y",
    "z": "Z",
}
ROTATION_DEMO_PLANE_Z_M = {
    ObjectKind.CUBE: (
        get_object_spec(ObjectKind.CUBE).local_bottom_z_m
        + DIRECT_GRASP_PLANE_HEIGHT_FRACTION[ObjectKind.CUBE]
        * get_object_spec(ObjectKind.CUBE).height_m
    ),
    ObjectKind.CYLINDER: (
        get_object_spec(ObjectKind.CYLINDER).local_bottom_z_m
        + DIRECT_GRASP_PLANE_HEIGHT_FRACTION[ObjectKind.CYLINDER]
        * get_object_spec(ObjectKind.CYLINDER).height_m
    ),
    ObjectKind.BANANA: 0.018,
    ObjectKind.PEAR: 0.0328,
    ObjectKind.STRAWBERRY: 0.0225,
    ObjectKind.ORANGE: 0.5 * get_object_spec(ObjectKind.ORANGE).height_m,
    ObjectKind.CONE: (
        get_object_spec(ObjectKind.CONE).local_bottom_z_m
        + DIRECT_GRASP_PLANE_HEIGHT_FRACTION[ObjectKind.CONE]
        * get_object_spec(ObjectKind.CONE).height_m
    ),
}
ROTATION_DEMO_PLANE_YAW_RAD = {
    ObjectKind.CUBE: 0.0,
    ObjectKind.CYLINDER: 0.0,
    # Axis-aligned side contacts give the straight banana collision proxy a
    # materially larger friction/torque reserve during post-lift motion.
    ObjectKind.BANANA: 0.0,
    ObjectKind.PEAR: 0.0,
    ObjectKind.STRAWBERRY: math.radians(45.0),
    ObjectKind.ORANGE: 0.0,
    ObjectKind.CONE: 0.0,
}
ROTATION_DEMO_MODE_YAW_RAD: Dict[tuple[ObjectKind, str], float] = {}
ROTATION_DEMO_LIFT_COMMAND_M = {
    ObjectKind.CUBE: LIFT_HEIGHT_M,
    ObjectKind.CYLINDER: LIFT_HEIGHT_M,
    ObjectKind.BANANA: LIFT_HEIGHT_M,
    ObjectKind.PEAR: LIFT_HEIGHT_M,
    ObjectKind.STRAWBERRY: LIFT_HEIGHT_M,
    ObjectKind.ORANGE: LIFT_HEIGHT_M,
    ObjectKind.CONE: LIFT_HEIGHT_M,
}
ROTATION_DEMO_MODE_LIFT_COMMAND_M: Dict[
    tuple[ObjectKind, str], float
] = {}


def _contact_preload_m(kind: ObjectKind) -> float:
    if kind is ObjectKind.PEAR:
        return PEAR_CONTACT_PRELOAD_M
    if kind is ObjectKind.STRAWBERRY:
        return 0.0003
    return CONTACT_PRELOAD_M


def _close_compression_m(kind: ObjectKind) -> float:
    return CLOSE_COMPRESSION_BY_OBJECT_M.get(kind, CLOSE_COMPRESSION_M)


def _settle_compression_m(kind: ObjectKind) -> float:
    return SETTLE_COMPRESSION_BY_OBJECT_M.get(kind, SETTLE_COMPRESSION_M)


class Phase(str):
    SELECT = "select"
    CLOSE = "close"
    SETTLE = "settle"
    LIFT = "lift"
    HOLD = "hold"


def _require_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"MuJoCo object not found: {name!r}")
    return value


def _minimum_jerk_fraction(index: int, count: int) -> float:
    if count <= 0:
        return 1.0
    return minimum_jerk(float(np.clip(index / count, 0.0, 1.0)))


def _ramped_demo_force_target_n(
    start_force_n: float,
    control_step: int,
) -> float:
    """Ramp post-lift grip force without an action-boundary preload step."""
    blend = _minimum_jerk_fraction(
        min(max(int(control_step), 0), DEMO_FORCE_RAMP_STEPS),
        DEMO_FORCE_RAMP_STEPS,
    )
    start = float(
        np.clip(
            start_force_n,
            TARGET_RUBBER_CONTACT_FORCE_N,
            DEMO_TARGET_RUBBER_CONTACT_FORCE_N,
        )
    )
    return float(
        start
        + blend * (DEMO_TARGET_RUBBER_CONTACT_FORCE_N - start)
    )


def _quat_angle(
    reference: Sequence[float],
    current: Sequence[float],
) -> float:
    first = np.asarray(reference, dtype=np.float64).reshape(4)
    second = np.asarray(current, dtype=np.float64).reshape(4)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    cosine = float(np.clip(abs(first @ second), 0.0, 1.0))
    return 2.0 * math.acos(cosine)


def _axis_rotation(axis: str, angle_rad: float) -> np.ndarray:
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    if axis == "x":
        return np.asarray(
            ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
            dtype=np.float64,
        )
    if axis == "y":
        return np.asarray(
            ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
            dtype=np.float64,
        )
    if axis == "z":
        return np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
    raise ValueError(f"unknown rotation axis: {axis!r}")


def _signed_local_axis_angle(
    reference_rotation: np.ndarray,
    current_quaternion: Sequence[float],
    axis: str,
) -> float:
    relative = (
        np.asarray(reference_rotation, dtype=np.float64).reshape(3, 3).T
        @ quat_to_rot(current_quaternion)
    )
    if axis == "x":
        numerator = relative[2, 1] - relative[1, 2]
        denominator = relative[1, 1] + relative[2, 2]
    elif axis == "y":
        numerator = relative[0, 2] - relative[2, 0]
        denominator = relative[0, 0] + relative[2, 2]
    elif axis == "z":
        numerator = relative[1, 0] - relative[0, 1]
        denominator = relative[0, 0] + relative[1, 1]
    else:
        raise ValueError(f"unknown rotation axis: {axis!r}")
    return math.atan2(float(numerator), float(denominator))


def _rotation_error_angle(
    target_rotation: np.ndarray,
    current_quaternion: Sequence[float],
) -> float:
    current_rotation = quat_to_rot(current_quaternion)
    relative = np.asarray(target_rotation, dtype=np.float64).T @ current_rotation
    cosine = float(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0))
    return math.acos(cosine)


def _fixed_xyz_euler_error_deg(
    target_rotation: np.ndarray,
    current_quaternion: Sequence[float],
) -> np.ndarray:
    """Return wrapped fixed-XYZ Euler error for ``target -> current``.

    The endpoint rotation criterion intentionally follows the user's requested
    XYZ Euler-vector L2 definition, rather than the SO(3) geodesic angle kept
    elsewhere as a diagnostic.  Components are wrapped to [-180, 180] degrees.
    """
    relative = (
        np.asarray(target_rotation, dtype=np.float64).reshape(3, 3).T
        @ quat_to_rot(current_quaternion)
    )
    return np.asarray(
        quat_to_euler_world_xyz_deg(rot_to_quat(relative)),
        dtype=np.float64,
    )


def _translation_endpoint_within_tolerance(
    error_xyz_m: Sequence[float],
) -> bool:
    """Return the sole translation-action success decision."""
    return bool(
        float(
            np.linalg.norm(
                np.asarray(error_xyz_m, dtype=np.float64).reshape(3)
            )
        )
        <= DEMO_TRANSLATION_GOAL_TOLERANCE_M + 1e-12
    )


def quantized_translation_vector_m(
    radius_mm: int,
    theta_deg: int,
) -> tuple[float, float, float]:
    """Return one horizontal goal from integer-mm/integer-degree coordinates."""

    if isinstance(radius_mm, (bool, np.bool_)) or not isinstance(
        radius_mm, (int, np.integer)
    ):
        raise TypeError("radius_mm must be an integer number of millimetres")
    if isinstance(theta_deg, (bool, np.bool_)) or not isinstance(
        theta_deg, (int, np.integer)
    ):
        raise TypeError("theta_deg must be an integer number of degrees")
    radius_value = int(radius_mm)
    theta_value = int(theta_deg)
    minimum_mm = round(1000.0 * MIN_TRANSLATION_DISTANCE_M)
    maximum_mm = round(1000.0 * MAX_TRANSLATION_DISTANCE_M)
    if not minimum_mm <= radius_value <= maximum_mm:
        raise ValueError(
            f"radius_mm must be in [{minimum_mm}, {maximum_mm}]"
        )
    if not 0 <= theta_value < 360:
        raise ValueError("theta_deg must be in [0, 359]")
    radius_m = radius_value * TRANSLATION_RADIUS_RESOLUTION_M
    theta_rad = math.radians(
        theta_value * TRANSLATION_THETA_RESOLUTION_DEG
    )
    return (
        float(radius_m * math.cos(theta_rad)),
        float(radius_m * math.sin(theta_rad)),
        0.0,
    )


def validate_translation_vector_m(
    requested_translation_m: Sequence[float],
) -> tuple[float, float, float]:
    """Validate and canonicalize a 5--100 mm horizontal translation vector."""

    try:
        requested = np.asarray(
            requested_translation_m,
            dtype=np.float64,
        ).reshape(3).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "requested_translation_m must contain exactly three values"
        ) from exc
    if not np.all(np.isfinite(requested)):
        raise ValueError("requested_translation_m must be finite")
    if abs(float(requested[2])) > 1.0e-12:
        raise ValueError("requested_translation_m must have dz=0")
    requested[2] = 0.0
    distance = float(np.linalg.norm(requested))
    if (
        distance < MIN_TRANSLATION_DISTANCE_M - 1.0e-12
        or distance > MAX_TRANSLATION_DISTANCE_M + 1.0e-12
    ):
        raise ValueError(
            "requested translation norm must be between 0.005 m and 0.100 m"
        )
    return tuple(float(value) for value in requested)


def _rotation_endpoint_within_tolerance(
    error_xyz_deg: Sequence[float],
) -> bool:
    """Return the sole rotation-action success decision."""
    return bool(
        float(
            np.linalg.norm(
                np.asarray(error_xyz_deg, dtype=np.float64).reshape(3)
            )
        )
        <= math.degrees(DEMO_ROTATION_TARGET_TOLERANCE_RAD) + 1e-10
    )


def _horizontal_parallelogram_targets(
    targets: Mapping[str, Sequence[float]],
) -> Dict[str, np.ndarray]:
    """Project four tip centres onto one XY-parallel parallelogram."""
    values = {
        finger: np.asarray(targets[finger], dtype=np.float64).reshape(3)
        for finger in FINGERS
    }
    center = 0.25 * sum(values.values(), start=np.zeros(3))
    u_half = 0.5 * (values["R"] - values["L"])
    v_half = 0.5 * (values["U"] - values["D"])
    u_half[2] = 0.0
    v_half[2] = 0.0
    return {
        "R": center + u_half,
        "L": center - u_half,
        "U": center + v_half,
        "D": center - v_half,
    }


def _xy_parallelogram_targets_preserving_tip_z(
    targets: Mapping[str, Sequence[float]],
) -> Dict[str, np.ndarray]:
    """Project only XY formation while preserving all four independent Zs."""
    values = {
        finger: np.asarray(targets[finger], dtype=np.float64).reshape(3)
        for finger in FINGERS
    }
    projected = _horizontal_parallelogram_targets(values)
    for finger in FINGERS:
        projected[finger][2] = values[finger][2]
    return projected


def _formation_targets(
    kind: ObjectKind,
    targets: Mapping[str, Sequence[float]],
) -> Dict[str, np.ndarray]:
    """Apply the object-specific cage contract without erasing fruit tip Z."""
    if kind in FRUIT_KINDS:
        return _xy_parallelogram_targets_preserving_tip_z(targets)
    return _horizontal_parallelogram_targets(targets)


def _tip_formation_within_tolerance(
    kind: ObjectKind,
    coplanarity_error_m: float,
    xy_parallelogram_error_m: float,
) -> bool:
    """Check XY formation for all objects and coplanarity for non-fruits."""
    return bool(
        (
            kind in FRUIT_KINDS
            or coplanarity_error_m <= TIP_COPLANARITY_TOLERANCE_M
        )
        and xy_parallelogram_error_m
        <= TIP_PARALLELOGRAM_TOLERANCE_M
    )


def _simultaneous_contact_targets_local(
    kind: ObjectKind,
    targets: Mapping[str, Sequence[float]],
) -> Dict[str, np.ndarray]:
    """Return detached local targets for the simultaneous-contact approach."""
    del kind
    return {
        finger: np.asarray(targets[finger], dtype=np.float64).copy()
        for finger in FINGERS
    }


def _normalize_rotation_axes(
    axes: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if axes is None:
        return ()
    values = tuple(axes) if not isinstance(axes, str) else tuple(axes.lower())
    if not values:
        return ()
    if (
        any(axis not in ROTATION_AXIS_LABELS for axis in values)
        or len(set(values)) != len(values)
        or values != tuple(axis for axis in "xyz" if axis in values)
    ):
        valid = ", ".join(ROTATION_DEMO_CHOICES)
        raise ValueError(f"rotation axes must be one of: {valid}")
    return values


def _rotation_execution_axes(
    axes: str | Sequence[str],
    object_kind: ObjectKind,
) -> tuple[str, ...]:
    normalized = _normalize_rotation_axes(axes)
    if normalized == ("x", "y", "z"):
        return ("x", "z", "y")
    if normalized == ("y", "z"):
        if object_kind in (
            ObjectKind.CUBE,
            ObjectKind.CYLINDER,
            ObjectKind.CONE,
        ):
            return ("y", "z")
        return ("z", "y")
    return normalized


@dataclass(frozen=True)
class TrialResult:
    object_kind: str
    seed: int
    success: bool
    termination_reason: str
    steps: int
    lift_mm: float
    position_error_mm: float
    orientation_drift_deg: float
    horizontal_drift_mm: float
    linear_speed_mm_s: float
    contact_count: int
    tip_coplanarity_error_mm: float
    tip_parallelogram_error_mm: float
    contact_onset_steps: tuple[int, ...]
    contact_onset_span_steps: int
    first_all_four_contact_step: int
    four_contact_stable_steps: int
    max_four_contact_stable_steps: int
    min_lift_contact_count: int
    max_lift_tip_coplanarity_error_mm: float
    max_lift_tip_parallelogram_error_mm: float
    max_normal_support_fraction: float
    final_contact_forces_n: tuple[float, ...]
    initial_position_m: tuple[float, ...]
    goal_position_m: tuple[float, ...]
    final_position_m: tuple[float, ...]


@dataclass(frozen=True)
class RotationDemoResult:
    object_kind: str
    axes: tuple[str, ...]
    execution_axes: tuple[str, ...]
    axis_rotations_deg: tuple[float, ...]
    axis_steps: tuple[int, ...]
    completed: bool
    termination_reason: str
    steps: int
    target_error_deg: float
    endpoint_euler_error_xyz_deg: tuple[float, ...]
    endpoint_euler_error_l2_deg: float
    composite_target_within_tolerance: bool
    achieved_net_rotation_deg: float
    center_drift_mm: float
    contact_count: int
    max_normal_support_fraction: float
    final_contact_forces_n: tuple[float, ...]


@dataclass(frozen=True)
class TranslationDemoResult:
    object_kind: str
    requested_translation_m: tuple[float, ...]
    requested_distance_mm: float
    completed: bool
    termination_reason: str
    steps: int
    achieved_translation_mm: float
    achieved_along_track_mm: float
    cross_track_error_mm: float
    target_error_mm: float
    endpoint_translation_error_xyz_mm: tuple[float, ...]
    endpoint_translation_error_l2_mm: float
    orientation_drift_deg: float
    terminal_speed_mm_s: float
    contact_count: int
    min_contact_count: int
    max_normal_support_fraction: float
    final_contact_forces_n: tuple[float, ...]
    start_com_position_m: tuple[float, ...]
    target_com_position_m: tuple[float, ...]
    final_com_position_m: tuple[float, ...]


class IKGrabber:
    """MuJoCo controller driven only by analytic IK."""

    def __init__(
        self,
        *,
        render: bool = False,
        root_height_m: float = ROOT_HEIGHT_M,
    ) -> None:
        self.root_height_m = float(root_height_m)
        if not math.isfinite(self.root_height_m):
            raise ValueError("root_height_m must be finite")
        model_path = MODEL_XML.resolve()
        try:
            model_path.relative_to(Path(__file__).resolve().parent)
        except ValueError as exc:
            raise RuntimeError("model XML escaped the GRAB directory") from exc

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        root_id = _require_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "root"
        )
        self.model.body_pos[root_id, 2] = self.root_height_m
        # MuJoCo's multiccd path can produce a degenerate flat
        # cylinder-on-table manifold with an enormous false impulse.  The
        # ordinary convex collision path is stable for the local proxies.
        nativeccd_flag = getattr(
            mujoco.mjtDisableBit,
            "mjDSBL_NATIVECCD",
            None,
        )
        if nativeccd_flag is not None:
            self.model.opt.disableflags |= int(nativeccd_flag)
        multiccd_disable_flag = getattr(
            mujoco.mjtDisableBit,
            "mjDSBL_MULTICCD",
            None,
        )
        if multiccd_disable_flag is not None:
            self.model.opt.disableflags |= int(multiccd_disable_flag)
        multiccd_enable_flag = getattr(
            mujoco.mjtEnableBit,
            "mjENBL_MULTICCD",
            None,
        )
        if multiccd_enable_flag is not None:
            self.model.opt.enableflags &= ~int(multiccd_enable_flag)
        if not math.isclose(
            float(self.model.opt.timestep),
            MODEL_TIMESTEP_S,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "_g6_project.xml timestep does not match _g6_grab.py"
            )
        if self.model.nu != len(ACTUATOR_NAMES):
            raise ValueError(
                f"expected {len(ACTUATOR_NAMES)} actuators, got "
                f"{self.model.nu}"
            )
        self._verify_fixed_root()

        base_ik = GripperIK.from_xml(MODEL_XML)
        root_shift = np.asarray(
            (0.0, 0.0, self.root_height_m - ROOT_HEIGHT_M),
            dtype=np.float64,
        )
        shifted_fingers = {
            finger: replace(
                geometry,
                p_A_world=np.asarray(geometry.p_A_world) + root_shift,
            )
            for finger, geometry in base_ik.fingers.items()
        }
        shifted_info = replace(
            base_ik.model_info,
            root_pos=np.asarray(base_ik.model_info.root_pos) + root_shift,
        )
        shifted_geometry_model = replace(
            base_ik.geometry_model,
            xml_info=shifted_info,
            fingers=shifted_fingers,
        )
        self.ik = GripperIK(
            shifted_geometry_model,
            linear_search_step_m=base_ik.linear_search_step_m,
            linear_ref_m=base_ik.linear_ref_m,
            filter_passive_limits=base_ik.filter_passive_limits,
        )
        self.teacher = IKTeacher(self.ik)
        self.planner = AdaptiveGraspPlanePlanner(
            hard_tip_radius_m=HARD_TIP_RADIUS_M,
            pregrasp_clearance_m=PREGRASP_CLEARANCE_M,
            maximum_tilt_rad=math.radians(12.0),
        )

        self._actuator_joint_ids = np.asarray(
            self.model.actuator_trnid[:, 0], dtype=np.int32
        )
        self._ctrl_low = np.asarray(
            self.model.actuator_ctrlrange[:, 0], dtype=np.float64
        )
        self._ctrl_high = np.asarray(
            self.model.actuator_ctrlrange[:, 1], dtype=np.float64
        )
        self._joint_qpos_adrs: Dict[tuple[str, str], int] = {}
        for finger in FINGERS:
            for kind in JOINT_KINDS:
                joint_name = f"robot0:{kind}_joint_{finger}"
                joint_id = _require_id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                self._joint_qpos_adrs[(finger, kind)] = int(
                    self.model.jnt_qposadr[joint_id]
                )

        self._tip_geom_ids = {
            finger: _require_id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"hard_tip_{finger}",
            )
            for finger in FINGERS
        }
        self._tip_geom_to_finger = {
            geom_id: finger
            for finger, geom_id in self._tip_geom_ids.items()
        }
        root_body_id = _require_id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "root",
        )
        gripper_geom_ids: set[int] = set()
        for geom_id in range(int(self.model.ngeom)):
            body_id = int(self.model.geom_bodyid[geom_id])
            while body_id != 0:
                if body_id == root_body_id:
                    gripper_geom_ids.add(geom_id)
                    break
                body_id = int(self.model.body_parentid[body_id])
        self._gripper_geom_ids = frozenset(gripper_geom_ids)
        if not set(self._tip_geom_ids.values()).issubset(
            self._gripper_geom_ids
        ):
            raise ValueError("hard tips must belong to the fixed gripper")
        self._table_geom_id = _require_id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "table_geom",
        )

        self._object_handles: Dict[ObjectKind, Dict[str, Any]] = {}
        for spec in (get_object_spec(kind) for kind in OBJECT_KIND_ORDER):
            joint_id = _require_id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                spec.joint_name,
            )
            body_id = _require_id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                spec.body_name,
            )
            geom_ids = tuple(
                _require_id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    name,
                )
                for name in spec.geom_names
            )
            self._object_handles[spec.kind] = {
                "joint_id": joint_id,
                "body_id": body_id,
                "qpos_adr": int(self.model.jnt_qposadr[joint_id]),
                "qvel_adr": int(self.model.jnt_dofadr[joint_id]),
                "geom_ids": geom_ids,
            }
            model_mass = float(self.model.body_mass[body_id])
            if not math.isclose(
                model_mass,
                spec.mass_kg,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{spec.kind.value} XML mass {model_mass} kg does not "
                    f"match analytic spec {spec.mass_kg} kg"
                )

        self._original_geom_contype = np.asarray(
            self.model.geom_contype, dtype=np.int32
        ).copy()
        self._original_geom_conaffinity = np.asarray(
            self.model.geom_conaffinity, dtype=np.int32
        ).copy()
        self._original_geom_rgba = np.asarray(
            self.model.geom_rgba, dtype=np.float32
        ).copy()

        self._visualize_current_trial = False

        self.object_spec = get_object_spec(ObjectKind.CUBE)
        self.phase = Phase.SELECT
        self.phase_step = 0
        self.trial_step = 0
        self.selected_plan: GraspPlan
        self._reset_runtime_fields()

    def _verify_fixed_root(self) -> None:
        root_id = _require_id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "root",
        )
        root_position = np.asarray(
            self.model.body_pos[root_id], dtype=np.float64
        )
        if int(self.model.body_jntnum[root_id]) != 0:
            raise ValueError("root must be fixed and have no joint")
        expected = np.array(
            (0.0, 0.0, self.root_height_m), dtype=np.float64
        )
        if not np.allclose(root_position, expected, atol=1e-12, rtol=0.0):
            raise ValueError(
                f"root position must be [0, 0, {self.root_height_m}] m; got "
                f"{root_position.tolist()}"
            )

    def _reset_runtime_fields(self) -> None:
        self.executed_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self.last_ctrl_delta = np.zeros(self.model.nu, dtype=np.float64)
        self.last_tip_targets_world = {
            finger: np.zeros(3, dtype=np.float64)
            for finger in FINGERS
        }
        # The uncompensated cage is kept separately from the motor-side target.
        # Post-lift demos must move this nominal cage and then apply the current
        # force/formation compensation exactly once.
        self.last_nominal_cage_targets_world = {
            finger: np.zeros(3, dtype=np.float64)
            for finger in FINGERS
        }
        self.last_contact_forces = np.zeros(
            len(FINGERS), dtype=np.float64
        )
        self.closure_fraction = np.zeros(
            len(FINGERS), dtype=np.float64
        )
        self.contact_locked = np.zeros(len(FINGERS), dtype=bool)
        self.locked_targets_local: Dict[str, np.ndarray] = {}
        self.lift_tip_offsets_world: Dict[str, np.ndarray] = {}
        self.contact_center_trim_world = np.zeros(3, dtype=np.float64)
        self.contact_pair_compression_m = np.zeros(
            2, dtype=np.float64
        )
        self.contact_formation_trim_world = {
            finger: np.zeros(3, dtype=np.float64)
            for finger in FINGERS
        }
        self.ik_failure_steps = 0
        self.total_ik_failure_steps = 0
        self.contact_loss_steps = 0
        self.prelift_contact_stable_steps = 0
        self.four_contact_stable_steps = 0
        self.max_four_contact_stable_steps = 0
        self.first_all_four_contact_step = -1
        self.contact_onset_steps = np.full(
            len(FINGERS), -1, dtype=np.int32
        )
        self.min_lift_contact_count = len(FINGERS)
        self.max_lift_tip_coplanarity_error_m = 0.0
        self.max_lift_tip_parallelogram_error_m = 0.0
        self.max_normal_support_fraction = 0.0
        self.prelift_failure_reason = ""
        self.stable_hold_steps = 0
        self.motion_precondition_stable_steps = 0
        self.entered_lift = False
        self.initial_body_pos = np.zeros(3, dtype=np.float64)
        self.initial_body_quat = np.array(
            (1.0, 0.0, 0.0, 0.0), dtype=np.float64
        )
        self.lift_start_body_pos = np.zeros(3, dtype=np.float64)
        self.lift_start_body_quat = np.array(
            (1.0, 0.0, 0.0, 0.0), dtype=np.float64
        )
        self.commanded_lift_height_m = LIFT_HEIGHT_M
        self.active_translation_vector_m = np.asarray(
            DEMO_TRANSLATION_VECTOR_M,
            dtype=np.float64,
        )
        self.goal_body_pos = np.zeros(3, dtype=np.float64)
        self.success = False
        self.termination_reason = ""
        self._pinch_plan_ready = False
        self._post_lift_demo_consumed = False
        self._unscored_motion_precondition = False

    @property
    def active_handle(self) -> Dict[str, Any]:
        return self._object_handles[self.object_spec.kind]

    def _object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        body_id = int(self.active_handle["body_id"])
        return (
            np.asarray(
                self.data.xpos[body_id], dtype=np.float64
            ).copy(),
            np.asarray(
                self.data.xquat[body_id], dtype=np.float64
            ).copy(),
        )

    def _object_com_position(self) -> np.ndarray:
        body_id = int(self.active_handle["body_id"])
        return np.asarray(
            self.data.xipos[body_id], dtype=np.float64
        ).copy()

    def _object_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        body_id = int(self.active_handle["body_id"])
        cvel = np.asarray(self.data.cvel[body_id], dtype=np.float64)
        return cvel[3:6].copy(), cvel[0:3].copy()

    def _set_free_body_pose(
        self,
        kind: ObjectKind,
        position: Sequence[float],
        quaternion: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        handle = self._object_handles[kind]
        qpos_adr = int(handle["qpos_adr"])
        qvel_adr = int(handle["qvel_adr"])
        self.data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(
            position, dtype=np.float64
        )
        quaternion_value = np.asarray(
            quaternion, dtype=np.float64
        ).reshape(4)
        quaternion_value /= max(
            float(np.linalg.norm(quaternion_value)), 1e-12
        )
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quaternion_value
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def _activate_object(
        self,
        spec: ObjectSpec,
        start_xy: np.ndarray,
    ) -> None:
        for index, kind in enumerate(OBJECT_KIND_ORDER):
            handle = self._object_handles[kind]
            for geom_id in handle["geom_ids"]:
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
                self.model.geom_rgba[geom_id, 3] = 0.0
            self._set_free_body_pose(
                kind,
                (1.25 + 0.20 * index, 0.80, -0.50),
            )

        active = self._object_handles[spec.kind]
        for geom_id in active["geom_ids"]:
            self.model.geom_contype[geom_id] = (
                self._original_geom_contype[geom_id]
            )
            self.model.geom_conaffinity[geom_id] = (
                self._original_geom_conaffinity[geom_id]
            )
            self.model.geom_rgba[geom_id] = (
                self._original_geom_rgba[geom_id]
            )
        self._set_free_body_pose(
            spec.kind,
            (
                float(start_xy[0]),
                float(start_xy[1]),
                spec.rest_body_z_m + RESET_DROP_HEIGHT_M,
            ),
        )

    def _set_solution_qpos(self, solution: IKSolution) -> None:
        if not solution.success:
            raise ValueError("cannot install an incomplete IK solution")
        for finger, finger_solution in solution.finger_solutions.items():
            for kind, value in finger_solution.joint_map.items():
                self.data.qpos[
                    self._joint_qpos_adrs[(finger, kind)]
                ] = value
        self.data.ctrl[:] = solution.ctrl
        self.executed_ctrl = np.asarray(
            solution.ctrl, dtype=np.float64
        ).copy()

    def reset(
        self,
        object_kind: str,
        *,
        seed: int,
        demo_grasp_plane_z_m: float | None = None,
        demo_grasp_plane_yaw_rad: float | None = None,
        demo_lift_command_m: float | None = None,
        initial_com_z_m: float | None = None,
    ) -> bool:
        spec = get_object_spec(object_kind)
        rng = np.random.default_rng(int(seed))

        mujoco.mj_resetData(self.model, self.data)
        self.model.geom_contype[:] = self._original_geom_contype
        self.model.geom_conaffinity[:] = (
            self._original_geom_conaffinity
        )
        self.model.geom_rgba[:] = self._original_geom_rgba
        self._reset_runtime_fields()
        if demo_lift_command_m is not None:
            self.commanded_lift_height_m = float(
                demo_lift_command_m
            )
        self.phase = Phase.SELECT
        self.phase_step = 0
        self.trial_step = 0
        self.object_spec = spec

        start_xy = rng.uniform(-START_JITTER_M, START_JITTER_M, size=2)
        if (
            spec.kind is ObjectKind.CYLINDER
            and float(np.linalg.norm(start_xy)) < 1e-12
        ):
            start_xy[0] += CYLINDER_SYMMETRY_NUDGE_M
        self._activate_object(spec, start_xy)
        mujoco.mj_forward(self.model, self.data)
        # Starting a flat object exactly on the contact margin can create a
        # degenerate, perfectly simultaneous table contact (especially for the
        # cylinder).  Drop it by 2 mm and let MuJoCo establish a stable resting
        # manifold before computing the grasp plane.
        self.data.ctrl[:] = 0.0
        for _ in range(RESET_SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Formal G6 Diffusion templates may start the free fruit above the
        # table.  Shift the free-joint translation only after the ordinary
        # table settle so its orientation remains deterministic, then plan IK
        # against the actual elevated pose.  No hidden weld/anchor is added.
        if initial_com_z_m is not None:
            target_com_z_m = float(initial_com_z_m)
            if not math.isfinite(target_com_z_m):
                raise ValueError("initial_com_z_m must be finite")
            qpos_adr = int(self.active_handle["qpos_adr"])
            qvel_adr = int(self.active_handle["qvel_adr"])
            current_com_z_m = float(self._object_com_position()[2])
            self.data.qpos[qpos_adr + 2] += (
                target_com_z_m - current_com_z_m
            )
            self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            actual_com_z_m = float(self._object_com_position()[2])
            if not math.isclose(
                actual_com_z_m,
                target_com_z_m,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ):
                raise RuntimeError(
                    "failed to place active object COM at requested world z"
                )

        body_position, body_quaternion = self._object_pose()
        self.initial_body_pos = body_position.copy()
        self.initial_body_quat = body_quaternion.copy()
        self.goal_body_pos = (
            self.initial_body_pos
            + np.array((0.0, 0.0, self.commanded_lift_height_m))
        )
        try:
            try:
                nominal_plan = self.planner.plan(
                    spec,
                    ik=self.ik,
                    body_world_pos=body_position,
                    body_world_quat=body_quaternion,
                    table_top_z=TABLE_TOP_Z_M,
                    table_clearance_m=TIP_TABLE_CLEARANCE_M,
                )
            except RuntimeError as nominal_exc:
                # A task-specific plane can be reachable even when the
                # geometry teacher's nominal yaw is not.  Build the nominal
                # geometry without an IK gate, then let apply_residual below
                # validate the requested task plane with the real IK solver.
                nominal_message = str(nominal_exc)
                nominal_reason = nominal_message.split(": ", 1)[-1]
                can_try_task_plane = (
                    demo_grasp_plane_z_m is not None
                    and nominal_message.startswith(
                        "no feasible adaptive grasp plane for "
                        f"{spec.kind.value}:"
                    )
                    and nominal_reason.startswith("ik_unreachable_")
                )
                if not can_try_task_plane:
                    raise
                nominal_plan = self.planner.plan(
                    spec,
                    ik=None,
                    body_world_pos=body_position,
                    body_world_quat=body_quaternion,
                    table_top_z=TABLE_TOP_Z_M,
                    table_clearance_m=TIP_TABLE_CLEARANCE_M,
                )
            self.selected_plan = nominal_plan
            if demo_grasp_plane_z_m is not None:
                nominal_center = np.asarray(
                    nominal_plan.plane.center_local,
                    dtype=np.float64,
                )
                residual = np.asarray(
                    (
                        spec.local_com_m[0] - nominal_center[0],
                        spec.local_com_m[1] - nominal_center[1],
                        float(demo_grasp_plane_z_m)
                        - nominal_center[2],
                        -nominal_plan.plane.roll_rad,
                        -nominal_plan.plane.pitch_rad,
                        (
                            float(demo_grasp_plane_yaw_rad)
                            - nominal_plan.plane.yaw_rad
                            if demo_grasp_plane_yaw_rad is not None
                            else 0.0
                        ),
                    ),
                    dtype=np.float64,
                )
                demo_plan = self.planner.apply_residual(
                    nominal_plan,
                    residual,
                    ik=self.ik,
                    body_world_pos=body_position,
                    body_world_quat=body_quaternion,
                    table_top_z=TABLE_TOP_Z_M,
                    table_clearance_m=TIP_TABLE_CLEARANCE_M,
                )
                if not demo_plan.valid:
                    reason = str(
                        demo_plan.diagnostics.get(
                            "reason",
                            "invalid_rotation_demo_grasp_plane",
                        )
                    )
                    raise RuntimeError(
                        "rotation demo grasp plane is invalid: "
                        f"{reason}"
                    )
                self.selected_plan = demo_plan
        except RuntimeError as exc:
            planner_message = str(exc)
            expected_plate_rejection = (
                spec.kind is ObjectKind.PLATE
                and planner_message.startswith(
                    "no feasible adaptive grasp plane for plate:"
                )
                and "tip_table_clearance" in planner_message
            )
            if not expected_plate_rejection:
                raise
            self.success = False
            self.termination_reason = THIN_OBJECT_FAILURE_REASON
            return False
        self._pinch_plan_ready = True
        rotation = quat_to_rot(body_quaternion)
        pregrasp_local = _simultaneous_contact_targets_local(
            self.object_spec.kind,
            self.selected_plan.pregrasp_targets_local,
        )
        pregrasp_targets = {
            finger: body_position + rotation @ pregrasp_local[finger]
            for finger in FINGERS
        }
        pregrasp_targets = _formation_targets(
            self.object_spec.kind,
            pregrasp_targets,
        )
        pregrasp_solution = self.ik.solve_tip_targets(
            pregrasp_targets,
            strict=True,
        )
        self._set_solution_qpos(pregrasp_solution)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.teacher.reset(self.executed_ctrl)
        self.last_tip_targets_world = {
            finger: target.copy()
            for finger, target in pregrasp_targets.items()
        }
        self.last_contact_forces = self._contact_forces()
        return True

    def _tip_contact_evidence(self) -> Dict[str, Dict[str, Any]]:
        accumulators: Dict[str, Dict[str, Any]] = {
            finger: {
                "force_n": 0.0,
                "weighted_position_world_m": np.zeros(
                    3, dtype=np.float64
                ),
                "outward_normal_force_world_n": np.zeros(
                    3, dtype=np.float64
                ),
            }
            for finger in FINGERS
        }
        active_geoms = set(self.active_handle["geom_ids"])
        wrench = np.zeros(6, dtype=np.float64)
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            finger: str | None = None
            outward_normal: np.ndarray | None = None
            if (
                geom1 in self._tip_geom_to_finger
                and geom2 in active_geoms
            ):
                finger = self._tip_geom_to_finger[geom1]
                outward_normal = -np.asarray(
                    contact.frame, dtype=np.float64
                ).reshape(3, 3)[0]
            elif (
                geom2 in self._tip_geom_to_finger
                and geom1 in active_geoms
            ):
                finger = self._tip_geom_to_finger[geom2]
                outward_normal = np.asarray(
                    contact.frame, dtype=np.float64
                ).reshape(3, 3)[0]
            if finger is None or outward_normal is None:
                continue
            wrench.fill(0.0)
            mujoco.mj_contactForce(
                self.model,
                self.data,
                contact_index,
                wrench,
            )
            force_n = abs(float(wrench[0]))
            normal_norm = float(np.linalg.norm(outward_normal))
            if normal_norm <= 1e-12:
                continue
            outward_normal = outward_normal / normal_norm
            accumulator = accumulators[finger]
            accumulator["force_n"] += force_n
            accumulator["weighted_position_world_m"] += (
                force_n
                * np.asarray(contact.pos, dtype=np.float64).reshape(3)
            )
            accumulator["outward_normal_force_world_n"] += (
                force_n * outward_normal
            )

        evidence: Dict[str, Dict[str, Any]] = {}
        for finger in FINGERS:
            accumulator = accumulators[finger]
            force_n = float(accumulator["force_n"])
            normal_force = np.asarray(
                accumulator["outward_normal_force_world_n"],
                dtype=np.float64,
            )
            normal_norm = float(np.linalg.norm(normal_force))
            evidence[finger] = {
                "force_n": force_n,
                "position_world_m": (
                    np.asarray(
                        accumulator["weighted_position_world_m"],
                        dtype=np.float64,
                    )
                    / force_n
                    if force_n > 1e-12
                    else np.zeros(3, dtype=np.float64)
                ),
                "outward_normal_world": (
                    normal_force / normal_norm
                    if normal_norm > 1e-12
                    else np.zeros(3, dtype=np.float64)
                ),
                "outward_normal_force_world_n": normal_force.copy(),
            }
        return evidence

    def _pinch_evidence(
        self,
        *,
        require_static: bool,
    ) -> Dict[str, Any]:
        contacts = self._tip_contact_evidence()
        forces = np.asarray(
            [contacts[finger]["force_n"] for finger in FINGERS],
            dtype=np.float64,
        )
        positions = np.asarray(
            [
                contacts[finger]["position_world_m"]
                for finger in FINGERS
            ],
            dtype=np.float64,
        )
        normals = np.asarray(
            [
                contacts[finger]["outward_normal_world"]
                for finger in FINGERS
            ],
            dtype=np.float64,
        )
        alignments = np.full(len(FINGERS), -1.0, dtype=np.float64)
        region_errors = np.full(
            len(FINGERS), 1.0, dtype=np.float64
        )
        side_margins = np.full(
            len(FINGERS), -1.0, dtype=np.float64
        )
        plan_ready = bool(
            self._pinch_plan_ready
            and hasattr(self, "selected_plan")
            and self.selected_plan.object_kind is self.object_spec.kind
        )
        if plan_ready:
            body_position, body_quaternion = self._object_pose()
            body_rotation = quat_to_rot(body_quaternion)
            object_com = self._object_com_position()
            for index, finger in enumerate(FINGERS):
                planned_normal_world = (
                    body_rotation
                    @ np.asarray(
                        self.selected_plan.contact_normals_local[finger],
                        dtype=np.float64,
                    )
                )
                planned_normal_world /= max(
                    float(np.linalg.norm(planned_normal_world)),
                    1e-12,
                )
                alignments[index] = float(
                    normals[index] @ planned_normal_world
                )
                contact_local = body_rotation.T @ (
                    positions[index] - body_position
                )
                region_errors[index] = float(
                    np.linalg.norm(
                        contact_local
                        - np.asarray(
                            self.selected_plan.surface_points_local[
                                finger
                            ],
                            dtype=np.float64,
                        )
                    )
                )
                planned_side_world = (
                    body_rotation
                    @ np.asarray(
                        self.selected_plan.outward_directions_local[
                            finger
                        ],
                        dtype=np.float64,
                    )
                )
                planned_side_world /= max(
                    float(np.linalg.norm(planned_side_world)),
                    1e-12,
                )
                side_margins[index] = float(
                    (positions[index] - object_com)
                    @ planned_side_world
                )

        finger_index = {
            finger: index for index, finger in enumerate(FINGERS)
        }
        force_ratio_ud = float(
            min(
                forces[finger_index["U"]],
                forces[finger_index["D"]],
            )
            / max(
                forces[finger_index["U"]],
                forces[finger_index["D"]],
                1e-12,
            )
        )
        force_ratio_lr = float(
            min(
                forces[finger_index["L"]],
                forces[finger_index["R"]],
            )
            / max(
                forces[finger_index["L"]],
                forces[finger_index["R"]],
                1e-12,
            )
        )
        opposition_dot_ud = float(
            normals[finger_index["U"]] @ normals[finger_index["D"]]
        )
        opposition_dot_lr = float(
            normals[finger_index["L"]] @ normals[finger_index["R"]]
        )
        object_touches_non_tip = self._object_touches_non_tip_gripper()
        gripper_touches_table = self._gripper_touches_table()
        finite_metrics = bool(
            np.all(np.isfinite(forces))
            and np.all(np.isfinite(positions))
            and np.all(np.isfinite(normals))
            and np.all(np.isfinite(alignments))
            and np.all(np.isfinite(region_errors))
            and np.all(np.isfinite(side_margins))
            and math.isfinite(opposition_dot_ud)
            and math.isfinite(opposition_dot_lr)
        )
        body_fixed_valid = bool(
            plan_ready
            and finite_metrics
            and np.all(forces >= PINCH_MIN_FORCE_PER_TIP_N)
            and np.all(
                alignments >= PINCH_MIN_PLANNED_NORMAL_COSINE
            )
            and np.all(
                region_errors <= PINCH_MAX_PLANNED_REGION_ERROR_M
            )
            and np.all(side_margins > PINCH_MIN_SIDE_MARGIN_M)
            and opposition_dot_ud <= PINCH_MAX_OPPOSING_NORMAL_DOT
            and opposition_dot_lr <= PINCH_MAX_OPPOSING_NORMAL_DOT
            and force_ratio_ud >= PINCH_MIN_OPPOSING_FORCE_RATIO
            and force_ratio_lr >= PINCH_MIN_OPPOSING_FORCE_RATIO
            and not object_touches_non_tip
            and not gripper_touches_table
        )

        gravity = np.asarray(
            self.model.opt.gravity, dtype=np.float64
        ).reshape(3)
        gravity_magnitude = float(np.linalg.norm(gravity))
        object_weight_n = (
            float(self.model.body_mass[int(self.active_handle["body_id"])])
            * gravity_magnitude
        )
        upward = (
            -gravity / gravity_magnitude
            if gravity_magnitude > 1e-12
            else np.zeros(3, dtype=np.float64)
        )
        outward_normal_force = sum(
            (
                np.asarray(
                    contacts[finger][
                        "outward_normal_force_world_n"
                    ],
                    dtype=np.float64,
                )
                for finger in FINGERS
            ),
            start=np.zeros(3, dtype=np.float64),
        )
        normal_force_on_object = -outward_normal_force
        upward_normal_force_n = max(
            float(normal_force_on_object @ upward),
            0.0,
        )
        upward_normal_weight_ratio = (
            upward_normal_force_n / object_weight_n
            if object_weight_n > 1e-12
            else float("inf")
        )
        static_valid = bool(
            body_fixed_valid
            and math.isfinite(upward_normal_weight_ratio)
            and upward_normal_weight_ratio
            <= STATIC_PINCH_MAX_UPWARD_NORMAL_WEIGHT_RATIO
        )

        if not plan_ready:
            failure_reason = "pinch_plan_unavailable"
        elif not finite_metrics:
            failure_reason = "non_finite_pinch_evidence"
        elif bool(np.any(forces < PINCH_MIN_FORCE_PER_TIP_N)):
            failure_reason = "insufficient_pinch_force"
        elif bool(
            np.any(
                alignments < PINCH_MIN_PLANNED_NORMAL_COSINE
            )
        ):
            failure_reason = "pinch_normal_misaligned"
        elif bool(
            np.any(
                region_errors > PINCH_MAX_PLANNED_REGION_ERROR_M
            )
        ):
            failure_reason = "pinch_contact_outside_planned_region"
        elif bool(
            np.any(side_margins <= PINCH_MIN_SIDE_MARGIN_M)
        ):
            failure_reason = "pinch_contact_on_wrong_side"
        elif (
            opposition_dot_ud > PINCH_MAX_OPPOSING_NORMAL_DOT
            or opposition_dot_lr > PINCH_MAX_OPPOSING_NORMAL_DOT
        ):
            failure_reason = "pinch_opposition_lost"
        elif (
            force_ratio_ud < PINCH_MIN_OPPOSING_FORCE_RATIO
            or force_ratio_lr < PINCH_MIN_OPPOSING_FORCE_RATIO
        ):
            failure_reason = "pinch_force_pair_imbalanced"
        elif object_touches_non_tip:
            failure_reason = "object_touches_non_tip_gripper"
        elif gripper_touches_table:
            failure_reason = "gripper_touches_table"
        elif (
            require_static
            and upward_normal_weight_ratio
            > STATIC_PINCH_MAX_UPWARD_NORMAL_WEIGHT_RATIO
        ):
            failure_reason = "normal_support_exceeded"
        else:
            failure_reason = ""

        return {
            "body_fixed_valid": body_fixed_valid,
            "static_valid": static_valid,
            "valid": (
                static_valid if require_static else body_fixed_valid
            ),
            "failure_reason": failure_reason,
            "forces_n": forces,
            "positions_world_m": positions,
            "outward_normals_world": normals,
            "planned_normal_cosines": alignments,
            "planned_region_errors_m": region_errors,
            "side_margins_m": side_margins,
            "opposition_dot_ud": opposition_dot_ud,
            "opposition_dot_lr": opposition_dot_lr,
            "force_ratio_ud": force_ratio_ud,
            "force_ratio_lr": force_ratio_lr,
            "upward_normal_force_n": upward_normal_force_n,
            "object_weight_n": object_weight_n,
            "upward_normal_weight_ratio": (
                upward_normal_weight_ratio
            ),
            "object_touches_non_tip_gripper": (
                object_touches_non_tip
            ),
            "gripper_touches_table": gripper_touches_table,
        }

    def _record_normal_support_fraction(
        self,
        pinch_evidence: Mapping[str, Any],
    ) -> None:
        ratio = float(pinch_evidence["upward_normal_weight_ratio"])
        if math.isfinite(ratio):
            self.max_normal_support_fraction = max(
                self.max_normal_support_fraction,
                ratio,
            )

    def _contact_forces(self) -> np.ndarray:
        evidence = self._tip_contact_evidence()
        return np.asarray(
            [evidence[finger]["force_n"] for finger in FINGERS],
            dtype=np.float64,
        )

    def _demo_force_ramp_start_n(self) -> float:
        """Use the established contact load as the demo force-ramp origin."""
        forces = np.asarray(self.last_contact_forces, dtype=np.float64)
        active = forces[
            np.isfinite(forces)
            & (forces >= CONTACT_FORCE_THRESHOLD_N)
        ]
        if active.size == 0:
            return TARGET_RUBBER_CONTACT_FORCE_N
        return float(
            np.clip(
                np.median(active),
                TARGET_RUBBER_CONTACT_FORCE_N,
                DEMO_TARGET_RUBBER_CONTACT_FORCE_N,
            )
        )

    def _prepare_post_lift_cage_reference(self) -> None:
        """Prepare one uncompensated reference for a post-lift action.

        Regular objects retain the nominal horizontal cage recorded before
        force compensation.  Fruit independent-Z contact geometry needs
        the already-loaded four-tip formation to become the new nominal;
        rebasing its compensation state preserves that formation without
        applying the accumulated trims a second time.
        """
        if self.object_spec.kind not in FRUIT_KINDS:
            return
        rebased = _formation_targets(
            self.object_spec.kind,
            {
                finger: self.last_tip_targets_world[finger].copy()
                for finger in FINGERS
            },
        )
        self.last_nominal_cage_targets_world = {
            finger: rebased[finger].copy()
            for finger in FINGERS
        }
        self.contact_center_trim_world.fill(0.0)
        self.contact_pair_compression_m.fill(0.0)
        for finger in FINGERS:
            self.contact_formation_trim_world[finger].fill(0.0)

    def _tip_cage_metrics(self) -> tuple[float, float]:
        centers = {
            finger: np.asarray(
                self.data.geom_xpos[self._tip_geom_ids[finger]],
                dtype=np.float64,
            )
            for finger in FINGERS
        }
        coplanarity_error = float(
            np.ptp([centers[finger][2] for finger in FINGERS])
        )
        parallelogram_error = float(
            np.linalg.norm(
                0.5
                * (
                    centers["R"]
                    + centers["L"]
                    - centers["U"]
                    - centers["D"]
                )[:2]
            )
        )
        return coplanarity_error, parallelogram_error

    def _contact_onset_span_steps(self) -> int:
        if bool(np.any(self.contact_onset_steps < 0)):
            return -1
        return int(
            np.max(self.contact_onset_steps)
            - np.min(self.contact_onset_steps)
        )

    def _object_touches_table(self) -> bool:
        active_geoms = set(self.active_handle["geom_ids"])
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (
                geom1 == self._table_geom_id
                and geom2 in active_geoms
            ) or (
                geom2 == self._table_geom_id
                and geom1 in active_geoms
            ):
                return True
        return False

    def _object_touches_non_tip_gripper(self) -> bool:
        active_geoms = set(self.active_handle["geom_ids"])
        tip_geoms = set(self._tip_geom_ids.values())
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in active_geoms:
                other = geom2
            elif geom2 in active_geoms:
                other = geom1
            else:
                continue
            if (
                other not in tip_geoms
                and other != self._table_geom_id
                and other not in active_geoms
            ):
                return True
        return False

    def _gripper_touches_table(self) -> bool:
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (
                geom1 == self._table_geom_id
                and geom2 in self._gripper_geom_ids
            ) or (
                geom2 == self._table_geom_id
                and geom1 in self._gripper_geom_ids
            ):
                return True
        return False

    def _physics_ctrl(self, ctrl: Sequence[float]) -> None:
        started = time.perf_counter()
        self.data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)
        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            mujoco.mj_step(self.model, self.data)

    def _limited_ctrl(self, desired: Sequence[float]) -> np.ndarray:
        desired_value = np.clip(
            np.asarray(desired, dtype=np.float64).reshape(self.model.nu),
            self._ctrl_low,
            self._ctrl_high,
        )
        filtered = self.executed_ctrl + CTRL_FILTER_ALPHA * (
            desired_value - self.executed_ctrl
        )
        delta_limit = np.full(
            self.model.nu,
            MAX_CTRL_DELTA_RAD_PER_STEP,
            dtype=np.float64,
        )
        delta_limit[list(LINEAR_CTRL_INDICES)] = (
            MAX_LINEAR_CTRL_DELTA_M_PER_STEP
        )
        delta = np.clip(
            filtered - self.executed_ctrl,
            -delta_limit,
            delta_limit,
        )
        limited = np.clip(
            self.executed_ctrl + delta,
            self._ctrl_low,
            self._ctrl_high,
        )
        self.last_ctrl_delta = limited - self.executed_ctrl
        self.executed_ctrl = limited
        self.teacher.sync_executed_ctrl(limited)
        return limited

    def _solve_targets(
        self,
        requested: Mapping[str, np.ndarray],
        *,
        nominal: Optional[Mapping[str, np.ndarray]] = None,
    ) -> tuple[
        IKSolution | None,
        Dict[str, np.ndarray],
    ]:
        requested_value = {
            finger: np.asarray(
                requested[finger], dtype=np.float64
            ).reshape(3)
            for finger in FINGERS
        }
        solution = self.teacher.solve_tip_targets(
            requested_value,
            update=False,
        )
        if solution.success:
            return solution, requested_value
        if nominal is None:
            return None, self.last_tip_targets_world

        nominal_value = {
            finger: np.asarray(
                nominal[finger], dtype=np.float64
            ).reshape(3)
            for finger in FINGERS
        }
        nominal_solution = self.teacher.solve_tip_targets(
            nominal_value,
            update=False,
        )
        if not nominal_solution.success:
            return None, self.last_tip_targets_world

        low = 0.0
        high = 1.0
        best_solution = nominal_solution
        best_targets = nominal_value
        for _ in range(24):
            middle = 0.5 * (low + high)
            candidate = {
                finger: nominal_value[finger]
                + middle
                * (
                    requested_value[finger]
                    - nominal_value[finger]
                )
                for finger in FINGERS
            }
            candidate_solution = self.teacher.solve_tip_targets(
                candidate,
                update=False,
            )
            if candidate_solution.success:
                low = middle
                best_solution = candidate_solution
                best_targets = candidate
            else:
                high = middle
        return best_solution, best_targets

    def _command_targets(
        self,
        requested: Mapping[str, np.ndarray],
        *,
        nominal: Optional[Mapping[str, np.ndarray]] = None,
        enforce_horizontal_cage: bool = True,
    ) -> None:
        requested_value = (
            _formation_targets(self.object_spec.kind, requested)
            if enforce_horizontal_cage
            else {
                finger: np.asarray(requested[finger], dtype=np.float64)
                for finger in FINGERS
            }
        )
        nominal_value = (
            _formation_targets(self.object_spec.kind, nominal)
            if enforce_horizontal_cage and nominal is not None
            else nominal
        )
        solution, applied_targets = self._solve_targets(
            requested_value,
            nominal=nominal_value,
        )
        if solution is None:
            self.ik_failure_steps += 1
            self.total_ik_failure_steps += 1
            ctrl = self.executed_ctrl
        else:
            self.ik_failure_steps = 0
            ctrl = self._limited_ctrl(solution.ctrl)
            self.last_tip_targets_world = {
                finger: applied_targets[finger].copy()
                for finger in FINGERS
            }
        self._physics_ctrl(ctrl)

    def _step_select(self) -> None:
        body_position, body_quaternion = self._object_pose()
        rotation = quat_to_rot(body_quaternion)
        local_targets = _simultaneous_contact_targets_local(
            self.object_spec.kind,
            self.selected_plan.pregrasp_targets_local,
        )
        targets = {
            finger: body_position + rotation @ local_targets[finger]
            for finger in FINGERS
        }
        self._command_targets(targets)
        self.phase_step += 1
        if self.phase_step >= PLANE_SELECTION_STEPS:
            self.phase = Phase.CLOSE
            self.phase_step = 0
            self.closure_fraction.fill(0.0)
            self.contact_locked.fill(False)
            self.locked_targets_local.clear()

    def _closure_targets_local(self) -> Dict[str, np.ndarray]:
        targets: Dict[str, np.ndarray] = {}
        blend = _minimum_jerk_fraction(
            self.phase_step + 1,
            CLOSURE_STEPS,
        )
        pregrasp_targets = _simultaneous_contact_targets_local(
            self.object_spec.kind,
            self.selected_plan.pregrasp_targets_local,
        )
        contact_targets = _simultaneous_contact_targets_local(
            self.object_spec.kind,
            self.selected_plan.contact_targets_local,
        )
        for finger in FINGERS:
            pregrasp = np.asarray(pregrasp_targets[finger])
            contact = np.asarray(contact_targets[finger])
            inward = -np.asarray(
                self.selected_plan.outward_directions_local[finger]
            )
            compressed = (
                contact
                + _close_compression_m(self.object_spec.kind) * inward
            )
            targets[finger] = (
                (1.0 - blend) * pregrasp + blend * compressed
            )
        self.closure_fraction.fill(blend)
        return targets

    def _lock_contact_group(
        self,
        local_targets: Mapping[str, np.ndarray],
    ) -> None:
        """Lock all four commanded cage vertices in one control sample."""
        self.contact_locked.fill(True)
        self.locked_targets_local = {
            finger: np.asarray(
                local_targets[finger], dtype=np.float64
            ).copy()
            for finger in FINGERS
        }

    def _update_formation_compensation(self) -> None:
        if not bool(
            np.all(
                self.last_contact_forces
                >= CONTACT_FORCE_THRESHOLD_N
            )
        ):
            return
        actual = {
            finger: np.asarray(
                self.data.geom_xpos[self._tip_geom_ids[finger]],
                dtype=np.float64,
            ).copy()
            for finger in FINGERS
        }
        projected = _formation_targets(self.object_spec.kind, actual)
        if self.object_spec.kind is ObjectKind.BANANA:
            compensation_gain = BANANA_FORMATION_COMPENSATION_GAIN
        elif self.object_spec.kind is ObjectKind.CONE:
            compensation_gain = CONE_FORMATION_COMPENSATION_GAIN
        else:
            compensation_gain = FORMATION_COMPENSATION_GAIN
        for finger in FINGERS:
            trim = (
                self.contact_formation_trim_world[finger]
                + compensation_gain
                * (projected[finger] - actual[finger])
            )
            norm = float(np.linalg.norm(trim))
            if norm > MAX_FORMATION_COMPENSATION_M:
                trim *= MAX_FORMATION_COMPENSATION_M / norm
            self.contact_formation_trim_world[finger] = trim

    def _symmetric_force_targets(
        self,
        world_targets: Mapping[str, np.ndarray],
        *,
        update_feedback: bool,
        target_force_n: float | None = None,
        freeze_feedback_on_contact_loss: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Balance rubber loads and compensate mechanism compliance.

        The only feedback degrees of freedom are a common in-plane
        translation and equal/opposite motion of each opposing pair.  A small
        motor-side correction may make the *command* non-planar.  Regular
        objects use that correction to retain a world-horizontal cage;
        fruits retain their per-tip Z and correct only XY formation.
        """
        preserve_tip_z = self.object_spec.kind in FRUIT_KINDS
        cage = _formation_targets(self.object_spec.kind, world_targets)
        self.last_nominal_cage_targets_world = {
            finger: np.asarray(cage[finger], dtype=np.float64).copy()
            for finger in FINGERS
        }
        center = 0.25 * sum(
            cage.values(), start=np.zeros(3, dtype=np.float64)
        )
        u_half = 0.5 * (cage["R"] - cage["L"])
        v_half = 0.5 * (cage["U"] - cage["D"])
        if preserve_tip_z:
            u_half[2] = 0.0
            v_half[2] = 0.0
        u_length = float(np.linalg.norm(u_half))
        v_length = float(np.linalg.norm(v_half))
        u_direction = u_half / max(u_length, 1e-12)
        v_direction = v_half / max(v_length, 1e-12)

        if update_feedback:
            desired_force_n = float(
                (
                    DEMO_TARGET_RUBBER_CONTACT_FORCE_N
                    if self._unscored_motion_precondition
                    else TARGET_RUBBER_CONTACT_FORCE_N
                )
                if target_force_n is None
                else target_force_n
            )
            forces = {
                finger: float(self.last_contact_forces[index])
                for index, finger in enumerate(FINGERS)
            }
            all_tip_contacts = all(
                force >= CONTACT_FORCE_THRESHOLD_N
                for force in forces.values()
            )
            feedback_allowed = (
                max(forces.values(), default=0.0) > 0.0
                and (
                    not freeze_feedback_on_contact_loss
                    or all_tip_contacts
                )
            )
            if feedback_allowed:
                self.contact_center_trim_world += (
                    PAIR_FORCE_BALANCE_GAIN_M_PER_N
                    * (
                        (forces["R"] - forces["L"]) * u_direction
                        + (forces["U"] - forces["D"]) * v_direction
                    )
                )
                u_trim = float(
                    np.clip(
                        self.contact_center_trim_world @ u_direction,
                        -MAX_CONTACT_CENTER_TRIM_M,
                        MAX_CONTACT_CENTER_TRIM_M,
                    )
                )
                v_trim = float(
                    np.clip(
                        self.contact_center_trim_world @ v_direction,
                        -MAX_CONTACT_CENTER_TRIM_M,
                        MAX_CONTACT_CENTER_TRIM_M,
                    )
                )
                self.contact_center_trim_world = (
                    u_trim * u_direction + v_trim * v_direction
                )

                pair_forces = (
                    (forces["R"], forces["L"]),
                    (forces["U"], forces["D"]),
                )
                pair_control_forces = np.asarray(
                    tuple(
                        (
                            min(pair)
                            if min(pair)
                            < CONTACT_FORCE_THRESHOLD_N
                            else 0.5 * sum(pair)
                        )
                        for pair in pair_forces
                    ),
                    dtype=np.float64,
                )
                self.contact_pair_compression_m += (
                    PAIR_PRELOAD_GAIN_M_PER_N
                    * (
                        desired_force_n
                        - pair_control_forces
                    )
                )
                self.contact_pair_compression_m[:] = np.clip(
                    self.contact_pair_compression_m,
                    MIN_PAIR_COMPRESSION_M,
                    MAX_PAIR_COMPRESSION_M,
                )

                self._update_formation_compensation()

        center += self.contact_center_trim_world
        u_half = (
            max(
                u_length - self.contact_pair_compression_m[0],
                HARD_TIP_RADIUS_M,
            )
            * u_direction
        )
        v_half = (
            max(
                v_length - self.contact_pair_compression_m[1],
                HARD_TIP_RADIUS_M,
            )
            * v_direction
        )
        planar_targets = {
            "R": center + u_half,
            "L": center - u_half,
            "U": center + v_half,
            "D": center - v_half,
        }
        if preserve_tip_z:
            for finger in FINGERS:
                planar_targets[finger][2] = cage[finger][2]
        return {
            finger: (
                planar_targets[finger]
                + self.contact_formation_trim_world[finger]
            )
            for finger in FINGERS
        }

    def _step_close(self) -> None:
        body_position, body_quaternion = self._object_pose()
        if self.object_spec.kind in POSE_STABILIZED_OBJECTS:
            body_position = self.initial_body_pos
            body_quaternion = self.initial_body_quat
        local_targets = self._closure_targets_local()
        rotation = quat_to_rot(body_quaternion)
        world_targets = {
            finger: body_position + rotation @ local_targets[finger]
            for finger in FINGERS
        }
        world_targets = self._symmetric_force_targets(
            world_targets,
            update_feedback=True,
        )
        self._command_targets(
            world_targets,
            enforce_horizontal_cage=False,
        )
        self.last_contact_forces = self._contact_forces()
        self.phase_step += 1
        if self.phase_step >= CLOSURE_STEPS:
            self.phase = Phase.SETTLE
            self.phase_step = 0
            self.prelift_contact_stable_steps = 0

    def _step_settle(self) -> None:
        body_position, body_quaternion = self._object_pose()
        if self.object_spec.kind in POSE_STABILIZED_OBJECTS:
            body_position = self.initial_body_pos
            body_quaternion = self.initial_body_quat
        rotation = quat_to_rot(body_quaternion)
        group_locked = bool(np.all(self.contact_locked))
        if bool(np.any(self.contact_locked)) != group_locked:
            raise RuntimeError("hard-tip contact group lock became asymmetric")
        local_targets = {}
        simultaneous_targets = _simultaneous_contact_targets_local(
            self.object_spec.kind,
            self.selected_plan.contact_targets_local,
        )
        for finger in FINGERS:
            contact = np.asarray(
                simultaneous_targets[finger]
            )
            inward = -np.asarray(
                self.selected_plan.outward_directions_local[finger]
            )
            local_targets[finger] = (
                contact
                + _settle_compression_m(self.object_spec.kind) * inward
            )
        targets = {
            finger: body_position + rotation @ local_targets[finger]
            for finger in FINGERS
        }
        targets = self._symmetric_force_targets(
            targets,
            update_feedback=True,
        )
        self._command_targets(
            targets,
            enforce_horizontal_cage=False,
        )
        self.last_contact_forces = self._contact_forces()
        contact_count = int(
            np.count_nonzero(
                self.last_contact_forces
                >= CONTACT_FORCE_THRESHOLD_N
            )
        )
        all_four_contacts = contact_count == len(FINGERS)
        (
            tip_coplanarity_error,
            tip_parallelogram_error,
        ) = self._tip_cage_metrics()
        formation_ok = _tip_formation_within_tolerance(
            self.object_spec.kind,
            tip_coplanarity_error,
            tip_parallelogram_error,
        )
        pinch_evidence = self._pinch_evidence(
            require_static=True
        )
        self._record_normal_support_fraction(pinch_evidence)
        precondition_sample_ok = (
            all_four_contacts
            and (
                formation_ok or self._unscored_motion_precondition
            )
            and bool(pinch_evidence["static_valid"])
        )
        self.prelift_contact_stable_steps = (
            self.prelift_contact_stable_steps + 1
            if precondition_sample_ok
            else 0
        )
        self.max_four_contact_stable_steps = max(
            self.max_four_contact_stable_steps,
            self.prelift_contact_stable_steps,
        )
        self.phase_step += 1
        contacts_ready = (
            self.prelift_contact_stable_steps
            >= PRELIFT_REQUIRED_CONTACT_STEPS
        )
        if contacts_ready and not group_locked:
            self._lock_contact_group(local_targets)
        settle_timed_out = self.phase_step >= CONTACT_SETTLE_STEPS
        if contacts_ready or settle_timed_out:
            body_position, body_quaternion = self._object_pose()
            (
                tip_coplanarity_error,
                tip_parallelogram_error,
            ) = self._tip_cage_metrics()
            horizontal_displacement = float(
                np.linalg.norm(
                    body_position[:2] - self.initial_body_pos[:2]
                )
            )
            orientation_drift = _quat_angle(
                self.initial_body_quat,
                body_quaternion,
            )
            if not contacts_ready:
                self.prelift_failure_reason = (
                    "prelift_invalid_side_pinch"
                    if not bool(pinch_evidence["static_valid"])
                    else "insufficient_grasp_contacts"
                )
            elif (
                not self._unscored_motion_precondition
                and
                self.object_spec.kind not in FRUIT_KINDS
                and tip_coplanarity_error
                > TIP_COPLANARITY_TOLERANCE_M
            ):
                self.prelift_failure_reason = (
                    "prelift_tip_coplanarity_exceeded"
                )
            elif (
                not self._unscored_motion_precondition
                and
                tip_parallelogram_error
                > TIP_PARALLELOGRAM_TOLERANCE_M
            ):
                self.prelift_failure_reason = (
                    "prelift_tip_parallelogram_error_exceeded"
                )
            elif (
                not self._unscored_motion_precondition
                and
                horizontal_displacement
                > MAX_PRELIFT_HORIZONTAL_DISPLACEMENT_M
            ):
                self.prelift_failure_reason = (
                    "prelift_horizontal_drift_exceeded"
                )
            elif (
                not self._unscored_motion_precondition
                and
                orientation_drift
                > MAX_PRELIFT_ORIENTATION_DRIFT_RAD
            ):
                self.prelift_failure_reason = (
                    "prelift_orientation_drift_exceeded"
                )
            else:
                self._freeze_lift_template()
                self.phase = Phase.LIFT
                self.phase_step = 0

    def _freeze_lift_template(self) -> None:
        self.entered_lift = True
        body_position, body_quaternion = self._object_pose()
        self.lift_start_body_pos = body_position.copy()
        self.lift_start_body_quat = body_quaternion.copy()
        lift_compression = LIFT_CONTACT_COMPRESSION_M.get(
            self.object_spec.kind,
            0.0,
        )
        targets = _formation_targets(
            self.object_spec.kind,
            self.last_tip_targets_world,
        )
        u_direction = targets["R"] - targets["L"]
        v_direction = targets["U"] - targets["D"]
        if self.object_spec.kind in FRUIT_KINDS:
            u_direction[2] = 0.0
            v_direction[2] = 0.0
        u_direction /= max(float(np.linalg.norm(u_direction)), 1e-12)
        v_direction /= max(float(np.linalg.norm(v_direction)), 1e-12)
        targets["R"] -= lift_compression * u_direction
        targets["L"] += lift_compression * u_direction
        targets["U"] -= lift_compression * v_direction
        targets["D"] += lift_compression * v_direction
        targets = _formation_targets(self.object_spec.kind, targets)
        self.lift_tip_offsets_world = {}
        for finger in FINGERS:
            self.lift_tip_offsets_world[finger] = (
                targets[finger] - body_position
            )
        self.contact_center_trim_world.fill(0.0)
        self.contact_pair_compression_m.fill(0.0)
        self.goal_body_pos = (
            self.lift_start_body_pos
            + np.array((0.0, 0.0, self.commanded_lift_height_m))
        )

    def _lift_targets(
        self,
        body_target: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        cage_center = np.asarray(body_target, dtype=np.float64).copy()
        body_position, body_quaternion = self._object_pose()
        if self.object_spec.kind in LIFT_FEEDBACK_OBJECTS:
            if self.object_spec.kind is ObjectKind.BANANA:
                position_gain = BANANA_LIFT_POSITION_FEEDBACK_GAIN
            elif self.object_spec.kind is ObjectKind.CONE:
                position_gain = CONE_LIFT_POSITION_FEEDBACK_GAIN
            else:
                position_gain = LIFT_POSITION_FEEDBACK_GAIN
            xy_limit = (
                BANANA_LIFT_POSITION_FEEDBACK_XY_LIMIT_M
                if self.object_spec.kind is ObjectKind.BANANA
                else LIFT_POSITION_FEEDBACK_XY_LIMIT_M
            )
            correction = position_gain * (
                cage_center - body_position
            )
            correction[:2] = np.clip(
                correction[:2],
                -xy_limit,
                xy_limit,
            )
            z_limit = (
                CONE_LIFT_POSITION_FEEDBACK_Z_LIMIT_M
                if self.object_spec.kind is ObjectKind.CONE
                else LIFT_POSITION_FEEDBACK_Z_LIMIT_M
            )
            correction[2] = float(
                np.clip(
                    correction[2],
                    -z_limit,
                    z_limit,
                )
            )
            cage_center += correction
        reference_rotation = quat_to_rot(self.lift_start_body_quat)
        reference_yaw = math.atan2(
            float(reference_rotation[1, 0]),
            float(reference_rotation[0, 0]),
        )
        orientation_gain = LIFT_ORIENTATION_FEEDBACK_GAIN.get(
            self.object_spec.kind,
            0.0,
        )
        correction_angle = 0.0
        if orientation_gain > 0.0:
            current_rotation = quat_to_rot(body_quaternion)
            current_yaw = math.atan2(
                float(current_rotation[1, 0]),
                float(current_rotation[0, 0]),
            )
            yaw_drift = math.atan2(
                math.sin(current_yaw - reference_yaw),
                math.cos(current_yaw - reference_yaw),
            )
            correction_angle = float(
                np.clip(
                    -orientation_gain * yaw_drift,
                    -LIFT_ORIENTATION_FEEDBACK_LIMIT_RAD,
                    LIFT_ORIENTATION_FEEDBACK_LIMIT_RAD,
                )
            )
        cosine = math.cos(correction_angle)
        sine = math.sin(correction_angle)
        rotation = np.asarray(
            (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        targets = {
            finger: (
                cage_center
                + rotation @ self.lift_tip_offsets_world[finger]
            )
            for finger in FINGERS
        }
        return self._symmetric_force_targets(
            targets,
            update_feedback=True,
        )

    def _step_lift(self) -> None:
        blend = _minimum_jerk_fraction(
            self.phase_step + 1,
            LIFT_STEPS,
        )
        body_target = (
            self.lift_start_body_pos
            + np.array(
                (0.0, 0.0, blend * self.commanded_lift_height_m)
            )
        )
        targets = self._lift_targets(body_target)
        self._command_targets(
            targets,
            nominal=targets,
            enforce_horizontal_cage=False,
        )
        self.phase_step += 1
        if self.phase_step >= LIFT_STEPS:
            self.phase = Phase.HOLD
            self.phase_step = 0

    def _step_hold(self) -> None:
        if (
            self._unscored_motion_precondition
            and self.phase_step >= HOLD_STEPS
        ):
            # The first HOLD_STEPS establish the loaded cage.  Keep the motor
            # command exactly fixed during the additional motion-precondition
            # settling window; continuing to integrate force compensation here
            # can itself create the oscillation that the window is meant to
            # remove.
            self._physics_ctrl(self.executed_ctrl)
        else:
            targets = self._lift_targets(self.goal_body_pos)
            self._command_targets(
                targets,
                nominal=targets,
                enforce_horizontal_cage=False,
            )
        self.phase_step += 1

    def _safety_and_success(
        self,
    ) -> tuple[bool, bool, str]:
        unscored_motion_precondition = (
            self._unscored_motion_precondition
        )
        if not (
            np.all(np.isfinite(self.data.qpos))
            and np.all(np.isfinite(self.data.qvel))
        ):
            return True, False, "non_finite_physics"
        if self.prelift_failure_reason:
            return True, False, self.prelift_failure_reason
        if (
            self.total_ik_failure_steps > 0
            and not unscored_motion_precondition
        ):
            return True, False, "analytic_ik_failure"

        body_position, body_quaternion = self._object_pose()
        orientation_drift = _quat_angle(
            self.initial_body_quat,
            body_quaternion,
        )
        if (
            orientation_drift > MAX_ORIENTATION_DRIFT_RAD
            and not unscored_motion_precondition
        ):
            return True, False, "orientation_drift_exceeded"
        if (
            float(np.linalg.norm(body_position[:2])) > 0.25
            and not unscored_motion_precondition
        ):
            return True, False, "object_left_task_region"
        estimated_bottom = (
            body_position[2] + self.object_spec.local_bottom_z_m
        )
        if (
            estimated_bottom < TABLE_TOP_Z_M - DROP_MARGIN_M
            and not unscored_motion_precondition
        ):
            return True, False, "object_dropped"
        contact_count = int(
            np.count_nonzero(
                self.last_contact_forces
                >= CONTACT_FORCE_THRESHOLD_N
            )
        )
        (
            tip_coplanarity_error,
            tip_parallelogram_error,
        ) = self._tip_cage_metrics()
        formation_ok = _tip_formation_within_tolerance(
            self.object_spec.kind,
            tip_coplanarity_error,
            tip_parallelogram_error,
        )
        require_static_pinch = self.phase in (Phase.LIFT, Phase.HOLD)
        pinch_evidence = self._pinch_evidence(
            require_static=require_static_pinch
        )
        if require_static_pinch:
            self._record_normal_support_fraction(pinch_evidence)
        all_four_contacts = contact_count == len(FINGERS)
        for index, force in enumerate(self.last_contact_forces):
            if (
                self.contact_onset_steps[index] < 0
                and force >= CONTACT_FORCE_THRESHOLD_N
            ):
                self.contact_onset_steps[index] = self.trial_step
        if all_four_contacts and self.first_all_four_contact_step < 0:
            self.first_all_four_contact_step = self.trial_step
        contact_onset_span_steps = self._contact_onset_span_steps()
        if (
            contact_onset_span_steps > MAX_CONTACT_ONSET_SPAN_STEPS
            and not unscored_motion_precondition
        ):
            return True, False, "tip_contact_onset_not_simultaneous"
        self.four_contact_stable_steps = (
            self.four_contact_stable_steps + 1
            if all_four_contacts and formation_ok
            else 0
        )
        self.max_four_contact_stable_steps = max(
            self.max_four_contact_stable_steps,
            self.four_contact_stable_steps,
        )
        if self.phase in (Phase.LIFT, Phase.HOLD):
            self.min_lift_contact_count = min(
                self.min_lift_contact_count,
                contact_count,
            )
            self.max_lift_tip_coplanarity_error_m = max(
                self.max_lift_tip_coplanarity_error_m,
                tip_coplanarity_error,
            )
            self.max_lift_tip_parallelogram_error_m = max(
                self.max_lift_tip_parallelogram_error_m,
                tip_parallelogram_error,
            )
            runtime_formation_ok = bool(
                formation_ok or unscored_motion_precondition
            )
            runtime_pinch_valid = bool(pinch_evidence["static_valid"])
            self.contact_loss_steps = (
                0
                if (
                    all_four_contacts
                    and runtime_formation_ok
                    and runtime_pinch_valid
                )
                else self.contact_loss_steps + 1
            )
            if self.contact_loss_steps >= CONTACT_LOSS_FAILURE_STEPS:
                if not all_four_contacts:
                    reason = "lost_grasp_contact"
                elif not bool(pinch_evidence["body_fixed_valid"]):
                    reason = "lost_side_pinch"
                elif not bool(pinch_evidence["static_valid"]):
                    reason = "normal_support_exceeded"
                else:
                    reason = "tip_cage_formation_lost"
                return True, False, reason

        linear_velocity, angular_velocity = self._object_velocity()
        position_error = float(
            np.linalg.norm(body_position - self.goal_body_pos)
        )
        measured_lift = float(
            body_position[2] - self.initial_body_pos[2]
        )
        horizontal_drift = float(
            np.linalg.norm(
                body_position[:2] - self.initial_body_pos[:2]
            )
        )
        if unscored_motion_precondition:
            # Pose and speed remain unscored diagnostics.  Mechanical
            # eligibility requires a static side pinch with the object clear
            # of the table for a consecutive hold window.
            if self.phase == Phase.HOLD:
                self.stable_hold_steps += 1
                settled_now = (
                    bool(pinch_evidence["static_valid"])
                    and not self._object_touches_table()
                )
                self.motion_precondition_stable_steps = (
                    self.motion_precondition_stable_steps + 1
                    if settled_now
                    else 0
                )
                settled = (
                    self.phase_step >= HOLD_STEPS
                    and self.motion_precondition_stable_steps
                    >= MOTION_PRECONDITION_REQUIRED_STABLE_STEPS
                )
                settle_timeout = (
                    self.phase_step
                    >= MOTION_PRECONDITION_MAX_HOLD_STEPS
                )
                if settled:
                    return (
                        True,
                        True,
                        "motion_precondition_complete_unscored",
                    )
                if settle_timeout:
                    return (
                        True,
                        False,
                        "motion_precondition_invalid_side_pinch",
                    )
            return False, False, ""
        stable_now = (
            self.phase == Phase.HOLD
            and self.phase_step > 0
            and position_error <= SUCCESS_POSITION_TOLERANCE_M
            and measured_lift >= MIN_SUCCESS_LIFT_M
            and horizontal_drift <= MAX_SUCCESS_HORIZONTAL_DRIFT_M
            and float(np.linalg.norm(linear_velocity))
            <= SUCCESS_LINEAR_SPEED_M_S
            and orientation_drift <= SUCCESS_ORIENTATION_DRIFT_RAD
            and all_four_contacts
            and formation_ok
            and bool(pinch_evidence["static_valid"])
            and 0
            <= contact_onset_span_steps
            <= MAX_CONTACT_ONSET_SPAN_STEPS
            and not self._object_touches_table()
            and not self._object_touches_non_tip_gripper()
            and self.total_ik_failure_steps == 0
        )
        if self.phase == Phase.HOLD:
            self.stable_hold_steps = (
                self.stable_hold_steps + 1 if stable_now else 0
            )
        if self.stable_hold_steps >= HOLD_STEPS:
            return True, True, "success"
        return False, False, ""

    def step(self) -> tuple[bool, Dict[str, Any]]:
        if self._post_lift_demo_consumed:
            raise RuntimeError(
                "a post-lift demonstration consumes the current trial; "
                "call reset() before step()"
            )
        if self.phase == Phase.SELECT:
            self._step_select()
        elif self.phase == Phase.CLOSE:
            self._step_close()
        elif self.phase == Phase.SETTLE:
            self._step_settle()
        elif self.phase == Phase.LIFT:
            self._step_lift()
        elif self.phase == Phase.HOLD:
            self._step_hold()
        else:
            raise RuntimeError(f"unknown phase: {self.phase}")

        self.trial_step += 1
        self.last_contact_forces = self._contact_forces()
        terminated, success, reason = self._safety_and_success()
        if (
            self.trial_step >= MAX_TRIAL_STEPS
            and not terminated
        ):
            terminated = True
            success = False
            reason = "time_limit"
        self.success = bool(success)
        if reason:
            self.termination_reason = reason
        return terminated, self.info()

    def info(self) -> Dict[str, Any]:
        body_position, body_quaternion = self._object_pose()
        linear_velocity, angular_velocity = self._object_velocity()
        pinch_evidence = self._pinch_evidence(require_static=True)
        (
            tip_coplanarity_error,
            tip_parallelogram_error,
        ) = self._tip_cage_metrics()
        contact_count = int(
            np.count_nonzero(
                self.last_contact_forces
                >= CONTACT_FORCE_THRESHOLD_N
            )
        )
        position_error = float(
            np.linalg.norm(body_position - self.goal_body_pos)
        )
        horizontal_drift = float(
            np.linalg.norm(
                body_position[:2] - self.initial_body_pos[:2]
            )
        )
        return {
            "object_kind": self.object_spec.kind.value,
            "phase": self.phase,
            "phase_step": self.phase_step,
            "trial_step": self.trial_step,
            "root_height_m": self.root_height_m,
            "requested_lift_m": LIFT_HEIGHT_M,
            "commanded_lift_m": self.commanded_lift_height_m,
            "active_translation_vector_m": (
                self.active_translation_vector_m.copy()
            ),
            "initial_position_m": self.initial_body_pos.copy(),
            "goal_position_m": self.goal_body_pos.copy(),
            "object_position_m": body_position,
            "object_quaternion_wxyz": body_quaternion,
            "object_linear_velocity_m_s": linear_velocity,
            "object_angular_velocity_rad_s": angular_velocity,
            "measured_lift_m": float(
                body_position[2] - self.initial_body_pos[2]
            ),
            "horizontal_drift_m": horizontal_drift,
            "position_error_m": position_error,
            "orientation_drift_rad": _quat_angle(
                self.initial_body_quat,
                body_quaternion,
            ),
            "contact_forces_n": self.last_contact_forces.copy(),
            "contact_count": contact_count,
            "all_four_contacts": contact_count == len(FINGERS),
            "normal_support_fraction": float(
                pinch_evidence["upward_normal_weight_ratio"]
            ),
            "max_normal_support_fraction": float(
                self.max_normal_support_fraction
            ),
            "side_pinch_valid": bool(
                pinch_evidence["static_valid"]
            ),
            "body_fixed_pinch_valid": bool(
                pinch_evidence["body_fixed_valid"]
            ),
            "gripper_touches_table": bool(
                pinch_evidence["gripper_touches_table"]
            ),
            "contact_normal_alignment": np.asarray(
                pinch_evidence["planned_normal_cosines"],
                dtype=np.float64,
            ).copy(),
            "contact_point_error_m": np.asarray(
                pinch_evidence["planned_region_errors_m"],
                dtype=np.float64,
            ).copy(),
            "contact_side_margin_m": np.asarray(
                pinch_evidence["side_margins_m"],
                dtype=np.float64,
            ).copy(),
            "pair_opposition_cos": np.asarray(
                (
                    pinch_evidence["opposition_dot_ud"],
                    pinch_evidence["opposition_dot_lr"],
                ),
                dtype=np.float64,
            ),
            "pair_force_balance_ratio": np.asarray(
                (
                    pinch_evidence["force_ratio_ud"],
                    pinch_evidence["force_ratio_lr"],
                ),
                dtype=np.float64,
            ),
            "pinch_contact_positions_world_m": np.asarray(
                pinch_evidence["positions_world_m"],
                dtype=np.float64,
            ).copy(),
            "pinch_contact_normals_world": np.asarray(
                pinch_evidence["outward_normals_world"],
                dtype=np.float64,
            ).copy(),
            "pinch_failure_reason": str(
                pinch_evidence["failure_reason"]
            ),
            "tip_coplanarity_error_m": tip_coplanarity_error,
            "tip_parallelogram_error_m": tip_parallelogram_error,
            "contact_onset_steps": self.contact_onset_steps.copy(),
            "contact_onset_span_steps": (
                self._contact_onset_span_steps()
            ),
            "four_contact_stable_steps": (
                self.four_contact_stable_steps
            ),
            "max_four_contact_stable_steps": (
                self.max_four_contact_stable_steps
            ),
            "first_all_four_contact_step": (
                self.first_all_four_contact_step
            ),
            "min_lift_contact_count": self.min_lift_contact_count,
            "entered_lift": self.entered_lift,
            "max_lift_tip_coplanarity_error_m": (
                self.max_lift_tip_coplanarity_error_m
            ),
            "max_lift_tip_parallelogram_error_m": (
                self.max_lift_tip_parallelogram_error_m
            ),
            "group_contact_locked": bool(np.all(self.contact_locked)),
            "contact_locked": self.contact_locked.copy(),
            "object_touches_table": self._object_touches_table(),
            "object_touches_non_tip_gripper": (
                self._object_touches_non_tip_gripper()
            ),
            "total_ik_failure_steps": self.total_ik_failure_steps,
            "stable_hold_steps": self.stable_hold_steps,
            "success": bool(self.success),
            "termination_reason": self.termination_reason,
            "post_lift_demo_consumed": self._post_lift_demo_consumed,
            "unscored_motion_precondition": bool(
                self._unscored_motion_precondition
            ),
            # Backward-compatible alias for older telemetry readers.
            "rotation_demo_consumed": self._post_lift_demo_consumed,
            "control": "analytic_ik_only",
        }

    def run_trial(
        self,
        object_kind: str,
        *,
        seed: int,
        visualize: bool = False,
    ) -> TrialResult:
        return self._run_trial(
            object_kind,
            seed=seed,
            visualize=visualize,
            demo_grasp_plane_z_m=None,
            demo_grasp_plane_yaw_rad=None,
            demo_lift_command_m=None,
            unscored_motion_precondition=False,
        )

    def run_rotation_ready_trial(
        self,
        object_kind: str,
        *,
        axes: str | Sequence[str],
        seed: int,
        visualize: bool = False,
    ) -> TrialResult:
        """Run the unscored grasp/lift initialization used by rotation."""
        normalized_axes = _normalize_rotation_axes(axes)
        spec = get_object_spec(object_kind)
        if spec.kind is ObjectKind.PLATE:
            # Keep the expected thin-object limitation structured and visible
            # in the same seven-object capability comparison.
            return self.run_trial(
                object_kind,
                seed=seed,
                visualize=visualize,
            )
        if (
            normalized_axes == ("z",)
            and spec.kind is not ObjectKind.BANANA
        ):
            return self._run_trial(
                object_kind,
                seed=seed,
                visualize=visualize,
                demo_grasp_plane_z_m=None,
                demo_grasp_plane_yaw_rad=None,
                demo_lift_command_m=None,
                unscored_motion_precondition=True,
            )
        return self._run_trial(
            object_kind,
            seed=seed,
            visualize=visualize,
            demo_grasp_plane_z_m=ROTATION_DEMO_PLANE_Z_M[spec.kind],
            demo_grasp_plane_yaw_rad=(
                ROTATION_DEMO_MODE_YAW_RAD.get(
                    (spec.kind, "".join(normalized_axes)),
                    ROTATION_DEMO_PLANE_YAW_RAD[spec.kind],
                )
            ),
            demo_lift_command_m=(
                ROTATION_DEMO_MODE_LIFT_COMMAND_M.get(
                    (spec.kind, "".join(normalized_axes)),
                    ROTATION_DEMO_LIFT_COMMAND_M[spec.kind],
                )
            ),
            unscored_motion_precondition=True,
        )

    def run_translation_ready_trial(
        self,
        object_kind: str,
        *,
        seed: int,
        visualize: bool = False,
    ) -> TrialResult:
        """Run an unscored 50 mm grasp/lift precondition for the 100 mm move."""
        spec = get_object_spec(object_kind)
        if spec.kind is ObjectKind.PLATE:
            return self.run_trial(
                object_kind,
                seed=seed,
                visualize=visualize,
            )
        return self._run_trial(
            object_kind,
            seed=seed,
            visualize=visualize,
            demo_grasp_plane_z_m=(
                ROTATION_DEMO_PLANE_Z_M[spec.kind]
                if spec.kind is ObjectKind.BANANA
                else None
            ),
            demo_grasp_plane_yaw_rad=(
                0.0 if spec.kind is ObjectKind.BANANA else None
            ),
            demo_lift_command_m=None,
            unscored_motion_precondition=True,
        )

    def _run_trial(
        self,
        object_kind: str,
        *,
        seed: int,
        visualize: bool,
        demo_grasp_plane_z_m: float | None,
        demo_grasp_plane_yaw_rad: float | None,
        demo_lift_command_m: float | None,
        unscored_motion_precondition: bool,
    ) -> TrialResult:
        self._visualize_current_trial = False
        ready = self.reset(
            object_kind,
            seed=seed,
            demo_grasp_plane_z_m=demo_grasp_plane_z_m,
            demo_grasp_plane_yaw_rad=demo_grasp_plane_yaw_rad,
            demo_lift_command_m=demo_lift_command_m,
        )
        if not ready:
            return self._trial_result(self.info(), seed=seed)
        self._unscored_motion_precondition = bool(
            unscored_motion_precondition
        )
        try:
            terminated = False
            info = self.info()
            while not terminated:
                terminated, info = self.step()
            return self._trial_result(info, seed=seed)
        finally:
            self._unscored_motion_precondition = False

    def run_rotation_demo(
        self,
        axes: str | Sequence[str],
    ) -> RotationDemoResult:
        """Rotate and score only the final wrapped XYZ-Euler error L2."""
        normalized_axes = _normalize_rotation_axes(axes)
        if not normalized_axes:
            raise ValueError("at least one rotation axis is required")
        if self.object_spec.kind is ObjectKind.PLATE:
            raise RuntimeError(
                "plate is the intentional thin-object grasping limitation"
            )
        if not self.success or self.phase != Phase.HOLD:
            raise RuntimeError(
                "rotation demo requires a completed unscored rotation-ready "
                "grasp-and-lift immediately beforehand"
            )
        if self._post_lift_demo_consumed:
            raise RuntimeError(
                "the current trial has already been consumed by a post-lift "
                "demo; call reset() first"
            )

        start_com = self._object_com_position()
        _start_body_position, start_quaternion = self._object_pose()
        start_rotation = quat_to_rot(start_quaternion)
        execution_axes = _rotation_execution_axes(
            normalized_axes,
            self.object_spec.kind,
        )
        active_rotation_axis = execution_axes[0]
        final_target_rotation = start_rotation.copy()
        for axis in execution_axes:
            final_target_rotation = (
                final_target_rotation
                @ _axis_rotation(axis, DEMO_ROTATION_ANGLE_RAD)
            )
        demo_force_start_n = self._demo_force_ramp_start_n()
        self._prepare_post_lift_cage_reference()

        formal_bookkeeping = {
            "phase": self.phase,
            "phase_step": self.phase_step,
            "trial_step": self.trial_step,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "stable_hold_steps": self.stable_hold_steps,
            "contact_loss_steps": self.contact_loss_steps,
            "ik_failure_steps": self.ik_failure_steps,
            "total_ik_failure_steps": self.total_ik_failure_steps,
        }
        demo_steps = 0
        reason = ""
        axis_rotations_deg: list[float] = []
        axis_steps: list[int] = []

        def command_cage(
            center: np.ndarray,
            target_rotation: np.ndarray,
            offsets_local: Mapping[str, np.ndarray],
        ) -> int:
            nonlocal demo_steps, reason
            requested = {
                finger: center + target_rotation @ offsets_local[finger]
                for finger in FINGERS
            }
            requested = self._symmetric_force_targets(
                requested,
                update_feedback=True,
                target_force_n=_ramped_demo_force_target_n(
                    demo_force_start_n,
                    demo_steps + 1,
                ),
                freeze_feedback_on_contact_loss=True,
            )
            nominal = {
                finger: self.last_tip_targets_world[finger].copy()
                for finger in FINGERS
            }
            self._command_targets(
                requested,
                nominal=nominal,
                enforce_horizontal_cage=False,
            )
            demo_steps += 1
            self.last_contact_forces = self._contact_forces()
            contact_count = int(
                np.count_nonzero(
                    self.last_contact_forces
                    >= CONTACT_FORCE_THRESHOLD_N
                )
            )

            if not (
                np.all(np.isfinite(self.data.qpos))
                and np.all(np.isfinite(self.data.qvel))
            ):
                reason = "not_evaluable_non_finite_physics"
            else:
                pinch_evidence = self._pinch_evidence(
                    require_static=active_rotation_axis == "z"
                )
                self._record_normal_support_fraction(pinch_evidence)
            if (
                not reason
                and (
                    not bool(pinch_evidence["valid"])
                    or self._object_touches_table()
                )
            ):
                reason = "not_evaluable_lost_pinch"
            return contact_count

        try:
            for axis in execution_axes:
                active_rotation_axis = axis
                segment_com = self._object_com_position()
                _body_position, segment_start_quaternion = (
                    self._object_pose()
                )
                segment_start_rotation = quat_to_rot(
                    segment_start_quaternion
                )
                cage_offsets_local = {
                    finger: (
                        segment_start_rotation.T
                        @ (
                            self.last_nominal_cage_targets_world[finger]
                            - segment_com
                        )
                    )
                    for finger in FINGERS
                }
                cage_angle = 0.0
                stable_steps = 0
                actual_axis_angle = 0.0
                segment_step_count = 0

                for _ in range(DEMO_MAX_STEPS_PER_AXIS):
                    _body_position, current_quaternion = (
                        self._object_pose()
                    )
                    actual_axis_angle = _signed_local_axis_angle(
                        segment_start_rotation,
                        current_quaternion,
                        axis,
                    )
                    axis_error = (
                        DEMO_ROTATION_ANGLE_RAD - actual_axis_angle
                    )
                    cage_angle += float(
                        np.clip(
                            DEMO_CAGE_PROPORTIONAL_GAIN * axis_error,
                            -DEMO_CAGE_MAX_STEP_RAD,
                            DEMO_CAGE_MAX_STEP_RAD,
                        )
                    )
                    cage_angle = float(
                        np.clip(
                            cage_angle,
                            -DEMO_CAGE_LIMIT_RAD,
                            DEMO_CAGE_LIMIT_RAD,
                        )
                    )
                    cage_rotation = (
                        segment_start_rotation
                        @ _axis_rotation(axis, cage_angle)
                    )
                    contact_count = command_cage(
                        segment_com,
                        cage_rotation,
                        cage_offsets_local,
                    )
                    segment_step_count += 1
                    if reason:
                        break

                    _body_position, current_quaternion = (
                        self._object_pose()
                    )
                    actual_axis_angle = _signed_local_axis_angle(
                        segment_start_rotation,
                        current_quaternion,
                        axis,
                    )
                    axis_error = (
                        DEMO_ROTATION_ANGLE_RAD - actual_axis_angle
                    )
                    # This is only an internal controller-convergence stop.
                    # Final success is evaluated later from the endpoint
                    # wrapped fixed-XYZ Euler L2 error alone.
                    stable_now = (
                        abs(axis_error) <= DEMO_AXIS_TOLERANCE_RAD
                    )
                    stable_steps = (
                        stable_steps + 1 if stable_now else 0
                    )
                    if stable_steps >= DEMO_AXIS_STABLE_STEPS:
                        break

                axis_rotations_deg.append(
                    math.degrees(actual_axis_angle)
                )
                axis_steps.append(segment_step_count)
                if reason:
                    break

            final_com = self._object_com_position()
            _final_body_position, final_quaternion = self._object_pose()
            final_forces = self._contact_forces()
            self.last_contact_forces = final_forces.copy()
            contact_count = int(
                np.count_nonzero(
                    final_forces >= CONTACT_FORCE_THRESHOLD_N
                )
            )
            terminal_pinch_evidence = self._pinch_evidence(
                require_static=active_rotation_axis == "z"
            )
            self._record_normal_support_fraction(
                terminal_pinch_evidence
            )
            if (
                not reason
                and (
                    not bool(terminal_pinch_evidence["valid"])
                    or self._object_touches_table()
                )
            ):
                reason = "not_evaluable_lost_pinch"
            if reason:
                endpoint_euler_error_xyz_deg = np.full(
                    3, np.nan, dtype=np.float64
                )
                endpoint_euler_error_l2_deg = float("nan")
                achieved_net_rotation_rad = float("nan")
            else:
                endpoint_euler_error_xyz_deg = (
                    _fixed_xyz_euler_error_deg(
                        final_target_rotation,
                        final_quaternion,
                    )
                )
                endpoint_euler_error_l2_deg = float(
                    np.linalg.norm(endpoint_euler_error_xyz_deg)
                )
                achieved_net_rotation_rad = _rotation_error_angle(
                    start_rotation,
                    final_quaternion,
                )
            center_drift_m = float(
                np.linalg.norm(final_com - start_com)
            )
            composite_target_within_tolerance = (
                not reason
                and _rotation_endpoint_within_tolerance(
                    endpoint_euler_error_xyz_deg
                )
            )

            if not reason:
                reason = (
                    "completed"
                    if composite_target_within_tolerance
                    else "endpoint_rotation_euler_l2_error_exceeded"
                )


            return RotationDemoResult(
                object_kind=self.object_spec.kind.value,
                axes=normalized_axes,
                execution_axes=execution_axes,
                axis_rotations_deg=tuple(axis_rotations_deg),
                axis_steps=tuple(axis_steps),
                completed=reason == "completed",
                termination_reason=reason,
                steps=demo_steps,
                target_error_deg=endpoint_euler_error_l2_deg,
                endpoint_euler_error_xyz_deg=tuple(
                    float(value)
                    for value in endpoint_euler_error_xyz_deg
                ),
                endpoint_euler_error_l2_deg=(
                    endpoint_euler_error_l2_deg
                ),
                composite_target_within_tolerance=(
                    composite_target_within_tolerance
                ),
                achieved_net_rotation_deg=math.degrees(
                    achieved_net_rotation_rad
                ),
                center_drift_mm=1000.0 * center_drift_m,
                contact_count=contact_count,
                max_normal_support_fraction=float(
                    self.max_normal_support_fraction
                ),
                final_contact_forces_n=tuple(
                    float(value) for value in final_forces
                ),
            )
        finally:
            for field, value in formal_bookkeeping.items():
                setattr(self, field, value)
            self._post_lift_demo_consumed = True

    def run_translation_demo(
        self,
        requested_translation_m: Sequence[float] | None = None,
    ) -> TranslationDemoResult:
        """Translate on the lift plane and score final XYZ position-error L2.

        The default remains the original 100 mm, world-XY 45-degree
        demonstration.  Explicit vectors must have ``dz == 0`` and a norm in
        the inclusive 5--100 mm range.
        """
        if self.object_spec.kind is ObjectKind.PLATE:
            raise RuntimeError(
                "plate is the intentional thin-object grasping limitation"
            )
        if not self.success or self.phase != Phase.HOLD:
            raise RuntimeError(
                "translation demo requires a completed unscored grasp-and-"
                "lift immediately beforehand"
            )
        if self._post_lift_demo_consumed:
            raise RuntimeError(
                "the current trial has already been consumed by a post-lift "
                "demo; call reset() first"
            )

        requested_translation = np.asarray(
            validate_translation_vector_m(
                DEMO_TRANSLATION_VECTOR_M
                if requested_translation_m is None
                else requested_translation_m
            ),
            dtype=np.float64,
        )
        self.active_translation_vector_m = requested_translation.copy()
        requested_distance = float(np.linalg.norm(requested_translation))
        if requested_distance <= 0.0:
            raise RuntimeError("translation vector must be non-zero")
        path_direction = requested_translation / requested_distance

        start_com = self._object_com_position()
        _start_body_position, start_quaternion = self._object_pose()
        start_rotation = quat_to_rot(start_quaternion)
        target_com = start_com + requested_translation
        demo_force_start_n = self._demo_force_ramp_start_n()
        self._prepare_post_lift_cage_reference()
        cage_offsets_local = {
            finger: (
                start_rotation.T
                @ (
                    self.last_nominal_cage_targets_world[finger]
                    - start_com
                )
            )
            for finger in FINGERS
        }

        formal_bookkeeping = {
            "phase": self.phase,
            "phase_step": self.phase_step,
            "trial_step": self.trial_step,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "stable_hold_steps": self.stable_hold_steps,
            "contact_loss_steps": self.contact_loss_steps,
            "ik_failure_steps": self.ik_failure_steps,
            "total_ik_failure_steps": self.total_ik_failure_steps,
        }
        demo_steps = 0
        min_contact_count = len(FINGERS)
        reason = ""

        def translation_metrics() -> tuple[
            np.ndarray,
            np.ndarray,
            float,
            float,
            float,
            float,
        ]:
            current_com = self._object_com_position()
            _body_position, current_quaternion = self._object_pose()
            displacement = current_com - start_com
            along_track = float(displacement @ path_direction)
            cross_track = float(
                np.linalg.norm(
                    displacement - along_track * path_direction
                )
            )
            goal_error = float(np.linalg.norm(current_com - target_com))
            orientation_drift = _quat_angle(
                start_quaternion,
                current_quaternion,
            )
            return (
                current_com,
                current_quaternion,
                along_track,
                cross_track,
                goal_error,
                orientation_drift,
            )

        def command_translation(blend: float) -> int:
            nonlocal demo_steps, min_contact_count, reason

            desired_com = start_com + float(blend) * requested_translation
            current_com = self._object_com_position()
            feedback = (
                DEMO_TRANSLATION_FEEDBACK_GAIN
                * (desired_com - current_com)
            )
            feedback_norm = float(np.linalg.norm(feedback))
            if feedback_norm > DEMO_TRANSLATION_FEEDBACK_LIMIT_M:
                feedback *= (
                    DEMO_TRANSLATION_FEEDBACK_LIMIT_M / feedback_norm
                )
            cage_center = desired_com + feedback
            requested_targets = {
                finger: (
                    cage_center
                    + start_rotation @ cage_offsets_local[finger]
                )
                for finger in FINGERS
            }
            requested_targets = self._symmetric_force_targets(
                requested_targets,
                update_feedback=True,
                target_force_n=_ramped_demo_force_target_n(
                    demo_force_start_n,
                    demo_steps + 1,
                ),
                freeze_feedback_on_contact_loss=True,
            )
            nominal_targets = {
                finger: self.last_tip_targets_world[finger].copy()
                for finger in FINGERS
            }
            self._command_targets(
                requested_targets,
                nominal=nominal_targets,
                enforce_horizontal_cage=False,
            )
            demo_steps += 1
            self.last_contact_forces = self._contact_forces()
            contact_count = int(
                np.count_nonzero(
                    self.last_contact_forces
                    >= CONTACT_FORCE_THRESHOLD_N
                )
            )
            min_contact_count = min(min_contact_count, contact_count)

            if not (
                np.all(np.isfinite(self.data.qpos))
                and np.all(np.isfinite(self.data.qvel))
            ):
                reason = "not_evaluable_non_finite_physics"
            else:
                pinch_evidence = self._pinch_evidence(
                    require_static=True
                )
                self._record_normal_support_fraction(pinch_evidence)
            if (
                not reason
                and (
                    not bool(pinch_evidence["static_valid"])
                    or self._object_touches_table()
                )
            ):
                reason = "not_evaluable_lost_pinch"
            return contact_count

        try:
            for index in range(DEMO_TRANSLATION_STEPS):
                blend = _minimum_jerk_fraction(
                    index + 1,
                    DEMO_TRANSLATION_STEPS,
                )
                command_translation(blend)
                if reason:
                    break

            if not reason:
                for _ in range(DEMO_TRANSLATION_MAX_HOLD_STEPS):
                    command_translation(1.0)
                    if reason:
                        break

            (
                final_com,
                _final_quaternion,
                achieved_along_track,
                cross_track_error,
                target_error,
                orientation_drift,
            ) = translation_metrics()
            final_forces = self._contact_forces()
            self.last_contact_forces = final_forces.copy()
            contact_count = int(
                np.count_nonzero(
                    final_forces >= CONTACT_FORCE_THRESHOLD_N
                )
            )
            terminal_pinch_evidence = self._pinch_evidence(
                require_static=True
            )
            self._record_normal_support_fraction(
                terminal_pinch_evidence
            )
            if (
                not reason
                and (
                    not bool(
                        terminal_pinch_evidence["static_valid"]
                    )
                    or self._object_touches_table()
                )
            ):
                reason = "not_evaluable_lost_pinch"
            linear_velocity, _angular_velocity = self._object_velocity()
            terminal_speed = float(np.linalg.norm(linear_velocity))
            endpoint_translation_error_xyz_mm = (
                1000.0 * (final_com - target_com)
            )
            endpoint_translation_error_l2_mm = float(
                np.linalg.norm(endpoint_translation_error_xyz_mm)
            )

            if not reason:
                reason = (
                    "completed"
                    if _translation_endpoint_within_tolerance(
                        final_com - target_com
                    )
                    else "endpoint_translation_xyz_error_exceeded"
                )


            return TranslationDemoResult(
                object_kind=self.object_spec.kind.value,
                requested_translation_m=tuple(
                    float(value) for value in requested_translation
                ),
                requested_distance_mm=1000.0 * requested_distance,
                completed=reason == "completed",
                termination_reason=reason,
                steps=demo_steps,
                achieved_translation_mm=1000.0
                * float(np.linalg.norm(final_com - start_com)),
                achieved_along_track_mm=1000.0 * achieved_along_track,
                cross_track_error_mm=1000.0 * cross_track_error,
                target_error_mm=1000.0 * target_error,
                endpoint_translation_error_xyz_mm=tuple(
                    float(value)
                    for value in endpoint_translation_error_xyz_mm
                ),
                endpoint_translation_error_l2_mm=(
                    endpoint_translation_error_l2_mm
                ),
                orientation_drift_deg=math.degrees(
                    orientation_drift
                ),
                terminal_speed_mm_s=1000.0 * terminal_speed,
                contact_count=contact_count,
                min_contact_count=min_contact_count,
                max_normal_support_fraction=float(
                    self.max_normal_support_fraction
                ),
                final_contact_forces_n=tuple(
                    float(value) for value in final_forces
                ),
                start_com_position_m=tuple(
                    float(value) for value in start_com
                ),
                target_com_position_m=tuple(
                    float(value) for value in target_com
                ),
                final_com_position_m=tuple(
                    float(value) for value in final_com
                ),
            )
        finally:
            for field, value in formal_bookkeeping.items():
                setattr(self, field, value)
            self._post_lift_demo_consumed = True

    def _trial_result(
        self,
        info: Mapping[str, Any],
        *,
        seed: int,
    ) -> TrialResult:
        return TrialResult(
            object_kind=str(info["object_kind"]),
            seed=int(seed),
            success=bool(info["success"]),
            termination_reason=str(info["termination_reason"]),
            steps=int(info["trial_step"]),
            lift_mm=1000.0 * float(info["measured_lift_m"]),
            position_error_mm=1000.0 * float(info["position_error_m"]),
            orientation_drift_deg=math.degrees(
                float(info["orientation_drift_rad"])
            ),
            horizontal_drift_mm=(
                1000.0 * float(info["horizontal_drift_m"])
            ),
            linear_speed_mm_s=(
                1000.0
                * float(
                    np.linalg.norm(
                        info["object_linear_velocity_m_s"]
                    )
                )
            ),
            contact_count=int(info["contact_count"]),
            tip_coplanarity_error_mm=(
                1000.0 * float(info["tip_coplanarity_error_m"])
            ),
            tip_parallelogram_error_mm=(
                1000.0 * float(info["tip_parallelogram_error_m"])
            ),
            contact_onset_steps=tuple(
                int(value) for value in info["contact_onset_steps"]
            ),
            contact_onset_span_steps=int(
                info["contact_onset_span_steps"]
            ),
            first_all_four_contact_step=int(
                info["first_all_four_contact_step"]
            ),
            four_contact_stable_steps=int(
                info["four_contact_stable_steps"]
            ),
            max_four_contact_stable_steps=int(
                info["max_four_contact_stable_steps"]
            ),
            min_lift_contact_count=(
                int(info["min_lift_contact_count"])
                if bool(info["entered_lift"])
                else 0
            ),
            max_lift_tip_coplanarity_error_mm=(
                1000.0
                * float(info["max_lift_tip_coplanarity_error_m"])
            ),
            max_lift_tip_parallelogram_error_mm=(
                1000.0
                * float(info["max_lift_tip_parallelogram_error_m"])
            ),
            max_normal_support_fraction=float(
                info["max_normal_support_fraction"]
            ),
            final_contact_forces_n=tuple(
                float(value) for value in info["contact_forces_n"]
            ),
            initial_position_m=tuple(
                float(value) for value in info["initial_position_m"]
            ),
            goal_position_m=tuple(
                float(value) for value in info["goal_position_m"]
            ),
            final_position_m=tuple(
                float(value) for value in info["object_position_m"]
            ),
        )


    def close(self) -> None:
        """Release a headless controller (MuJoCo state is Python-owned)."""
        return None


























