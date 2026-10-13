"""Native MuJoCo-Warp vector environment for Diffusion-commanded IK.

Physics is owned by :class:`MJWarpBackend`; policy, controller state, reward,
and observations are ordinary CUDA Torch tensors.  The learned 25-D command
modulates independent hard-tip IK targets; the all-zero command is therefore
the paired IK-only controller in the same model and backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

import numpy as np

from . import contract
from .curriculum import (
    PHASE,
    rotation_angle_bounds_deg,
)
from .hpc_config import DEFAULT_CONFIG, HPCConfig
from .mjwarp_backend import MJWarpBackend
from .platform_adapter import PlatformSpec, build_platform_spec


try:
    import torch
except ImportError as exc:  # Source inspection remains possible on macOS.
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: BaseException | None = exc
else:
    _TORCH_IMPORT_ERROR = None


CONTROLLER_PHASES = (
    "plan",
    "close",
    "settle",
    "lift",
    "hold",
    "action",
)

# Expert-certification and Cartesian hard-tip command contract.  These are
# kept next to the controller because they constrain measured runtime state,
# rather than the immutable task geometry in ``contract.py``.
IK_TRUST_STAGES = (
    "grasp",
    "lift",
    "path_start",
    "route",
)
EXPERT_STABLE_TICKS = contract.EXPERT_STABLE_TICKS
EXPERT_OBJECT_LINEAR_SPEED_MAX_M_S = (
    contract.EXPERT_OBJECT_LINEAR_SPEED_MAX_M_S
)
EXPERT_OBJECT_ANGULAR_SPEED_MAX_RAD_S = (
    contract.EXPERT_OBJECT_ANGULAR_SPEED_MAX_RAD_S
)
EXPERT_OBJECT_TABLE_CLEARANCE_M = contract.EXPERT_OBJECT_TABLE_CLEARANCE_M
EXPERT_MIN_CONTACTS = contract.EXPERT_MIN_CONTACTS
TIP_TARGET_SPEED_MAX_M_S = contract.TIP_TARGET_SPEED_MAX_M_S
TIP_TARGET_ACCELERATION_MAX_M_S2 = (
    contract.TIP_TARGET_ACCELERATION_MAX_M_S2
)
TIP_TARGET_JERK_MAX_M_S3 = contract.TIP_TARGET_JERK_MAX_M_S3
CAGE_ORIENTATION_FEEDBACK_GAIN = 0.50
CAGE_ORIENTATION_FEEDBACK_MAX_RAD = math.radians(10.0)
IK_CAGE_TARGET_ERROR_MAX_M = contract.IK_CAGE_TARGET_ERROR_MAX_M

# A lift is a one-way physical transition.  Do not start it from a transient
# or unproved grasp: the exact hard-tip/current-fruit contact certificate must
# hold continuously at the end of CLOSE/SETTLE.  This gate is deliberately
# shorter than the 25-tick entry/terminal stability certificate, which remains
# unchanged and is evaluated after lift/path-start motion.
PRELIFT_CONTACT_STABLE_TICKS = contract.PRELIFT_CONTACT_STABLE_TICKS
PRELIFT_CONTACT_STAGES = (
    "close",
    "settle",
)

TRANSLATION_ENTRY_PREDICATES = (
    "action_reference_valid",
    "object_not_dropped",
    "tip_height_valid",
    "minimum_tip_object_contacts",
    "linear_speed_valid",
    "angular_speed_valid",
    "object_lifted",
    "start_position_valid",
    "start_orientation_valid",
)

# Fixed-width Markov state shared by both mechanisms.  Low-level proprioception
# reserves four active slots and two passive slots per tip; Gripper3 pads the
# fourth active slot with zero and exposes a validity mask.
OBSERVATION_SCHEMA = (
    ("object_one_hot", len(contract.OBJECTS)),
    ("task_one_hot", len(contract.TASKS)),
    ("curriculum_one_hot", 1),
    ("phase_one_hot", len(CONTROLLER_PHASES)),
    ("translation_path_shape_one_hot", len(contract.TRANSLATION_PATH_SHAPES)),
    ("translation_path_waypoints_xy", 2 * contract.TRANSLATION_PATH_WAYPOINTS),
    ("translation_path_descriptor", 4),
    ("translation_path_current_target_world", 3),
    ("translation_path_current_target_error", 3),
    ("translation_path_progress", 5),
    ("goal_position_error", 3),
    ("goal_orientation_error", 4),
    ("object_velocity", 6),
    ("tip_position_object_relative", 12),
    ("tip_target_error_world", 12),
    ("tip_target_command_velocity_world", 12),
    ("tip_target_command_acceleration_world", 12),
    ("tip_velocity_world", 12),
    ("finger_contact_force", 4),
    ("preload_trim", 4),
    ("position_feedback", 3),
    ("previous_action", contract.ACTION_DIM),
    ("motion_progress", 2),
    ("object_body_position_world", 3),
    ("object_com_position_world", 3),
    ("object_quaternion_world", 4),
    ("goal_position_world", 3),
    ("goal_quaternion_world", 4),
    ("initial_body_position_world", 3),
    ("initial_com_position_world", 3),
    ("initial_quaternion_world", 4),
    ("action_reference_position_world", 3),
    ("action_reference_quaternion_world", 4),
    ("action_reference_valid", 1),
    ("action_start_cage_object_local", 12),
    ("action_start_action", contract.ACTION_DIM),
    ("action_start_position_feedback", 3),
    ("action_start_preload_trim", 4),
    ("tip_position_world", 12),
    ("tip_target_world", 12),
    ("object_local_height_geometry", 3),
    ("grasp_band_fraction", 2),
    ("tip_height_fraction", 4),
    ("grasp_plane_fraction", 1),
    ("grasp_plane_compliant", 1),
    ("actual_tip_height_fraction", 4),
    ("actual_grasp_plane_fraction", 1),
    ("actual_grasp_plane_within_bounds", 1),
    ("actual_grasp_plane_compliant", 1),
    ("template_index_normalized", 1),
    ("source_region_one_hot", 1),
    ("control_progress", 1),
    ("expert_certification_state", 12),
    ("hold_reference_valid", 1),
    ("hold_frozen_cage_world", 12),
    ("active_joint_qpos_padded", 16),
    ("active_joint_qvel_padded", 16),
    ("active_joint_ctrl_padded", 16),
    ("active_joint_valid_mask", 16),
    ("ctrl_velocity_padded", 16),
    ("ctrl_acceleration_padded", 16),
    ("last_certified_ctrl_padded", 16),
    ("passive_joint_qpos", 8),
    ("passive_joint_qvel", 8),
    ("qacc_warmstart_padded", 72),
    ("qacc_warmstart_valid_mask", 72),
)


def _observation_slices() -> dict[str, slice]:
    start = 0
    result: dict[str, slice] = {}
    for name, width in OBSERVATION_SCHEMA:
        result[name] = slice(start, start + width)
        start += width
    return result


OBSERVATION_SLICES = _observation_slices()
OBSERVATION_DIM = sum(width for _, width in OBSERVATION_SCHEMA)


def _path_tracking_certified(
    task_index: Any,
    sample_count: Any,
    axis_abs_max_m: Any,
    schedule_complete: Any,
    schedule_limit_valid: Any,
) -> Any:
    """Return the fail-closed route-wide certificate for each world."""

    if torch is None:
        raise RuntimeError("PyTorch is required for path certification")
    task_tensor = torch.as_tensor(task_index)
    count_tensor = torch.as_tensor(sample_count, device=task_tensor.device)
    error_tensor = torch.as_tensor(axis_abs_max_m, device=task_tensor.device)
    complete_tensor = torch.as_tensor(
        schedule_complete, device=task_tensor.device, dtype=torch.bool
    )
    limit_tensor = torch.as_tensor(
        schedule_limit_valid, device=task_tensor.device, dtype=torch.bool
    )
    if count_tensor.shape != task_tensor.shape:
        raise ValueError("path tracking sample count has the wrong shape")
    if error_tensor.shape != (*task_tensor.shape, 3):
        raise ValueError("path tracking per-axis maximum has the wrong shape")
    if complete_tensor.shape != task_tensor.shape:
        raise ValueError("path schedule completion has the wrong shape")
    if limit_tensor.shape != task_tensor.shape:
        raise ValueError("path schedule limit certificate has the wrong shape")
    translating = task_tensor == contract.TASKS.index(contract.TASK_TRANSLATE)
    translation_certified = (
        (count_tensor == contract.TRANSLATION_PATH_ROUTE_SAMPLES)
        & complete_tensor
        & limit_tensor
        & torch.isfinite(error_tensor).all(dim=-1)
        & (
            error_tensor
            <= contract.TRANSLATION_PATH_TRACKING_AXIS_ABS_TOLERANCE_M
        ).all(dim=-1)
    )
    return (~translating) | translation_certified


LIFT_HEIGHT_M = 0.050
MIN_TIP_CENTER_Z_M = 0.278
# Preserve the parent-controller contact preload for each trained object.  The
# order matches contract.OBJECTS: orange, banana, pear, strawberry.
NOMINAL_PRELOAD_BY_OBJECT_M = (
    0.0008,
    0.0008,
    0.0003,
    0.0003,
)
IK_DAMPING = 1.0e-5
# G6 has one exactly redundant closed-chain direction per finger.  A 1e-8
# multiplier block makes the resulting FP32 KKT system condition number about
# 1e8; 1e-5 is the first conditioning plateau and is behaviourally neutral for
# the full-rank task-space solution.
CONSTRAINT_DAMPING = 1.0e-5
IK_POSITION_GAIN = 1.00
IK_NEWTON_ITERATIONS = 3
IK_BOUND_TOLERANCE = 1.0e-6
POSITION_FEEDBACK_GAIN = 1.00
POSITION_FEEDBACK_XY_LIMIT_M = 0.025
POSITION_FEEDBACK_Z_LIMIT_M = 0.008
POSITION_FEEDBACK_SLEW_M_PER_TICK = 5.0e-4
# Match the parent-controller pickup contract: the 3 N grip is sufficient for
# the vertical lift, then the action phase ramps to the 8 N motion reserve.
# Using 8 N throughout made the nominal ramp a no-op and drove high-force
# candidates into feedback saturation before the learned task even started.
GRASP_CONTACT_FORCE_N = 3.0
ACTION_CONTACT_FORCE_N = 8.0
ACTION_FORCE_RAMP_TICKS = 50
FORCE_FEEDBACK_GAIN_M_PER_N = 3.0e-5
# Match the parent IK controller's full 10 mm outward-release authority.  A
# 2 mm limit saturates before it can cancel the policy's independent 3 mm
# inward tip residual and was observed on every tip before a formal-batch
# constraint-solver nonfinite event.
FORCE_FEEDBACK_MIN_M = -0.010
FORCE_FEEDBACK_MAX_M = 0.010
PRELOAD_TRIM_SLEW_M_PER_TICK = 5.0e-4
CONTACT_FORCE_THRESHOLD_N = 0.035
MOTION_HOLD_FEEDBACK_TICKS = 20
EPISODE_ACTION_ALPHA = 0.22
DYNAMIC_ACTION_ALPHA = 0.35
IK_BACKWARD_ERROR_LIMIT = 1.0e-4
CTRL_LINEAR_ACCELERATION_LIMIT = 4.0
CTRL_ROTARY_ACCELERATION_LIMIT = 8.0
CTRL_LINEAR_JERK_LIMIT = 40.0
CTRL_ROTARY_JERK_LIMIT = 80.0


def _ik_free_mask_at_bounds(
    qpos: Any, low: Any, high: Any, raw_delta: Any
) -> Any:
    """Return the IK active-set mask for active and passive joints alike."""

    blocked = (
        (qpos <= low[None, :] + IK_BOUND_TOLERANCE)
        & (raw_delta < 0.0)
    ) | (
        (qpos >= high[None, :] - IK_BOUND_TOLERANCE)
        & (raw_delta > 0.0)
    )
    return (~blocked).to(qpos.dtype)


@dataclass(frozen=True)
class StepBatch:
    """One vector-environment transition, entirely on the Torch device."""

    observation: Any
    reward: Any
    done: Any
    info: dict[str, Any]


def _object_drop_transition(
    absorbing: Any,
    finite: Any,
    estimated_bottom_z_m: Any,
    raw_reward: Any,
) -> tuple[Any, Any, Any]:
    """Resolve sticky drop state and its one-shot absorbing reward."""

    object_drop_new = (
        (~absorbing)
        & finite
        & (estimated_bottom_z_m < contract.OBJECT_DROP_THRESHOLD_Z_M)
    )
    object_dropped = absorbing | object_drop_new
    terminal_penalty = raw_reward.new_full(
        raw_reward.shape, contract.OBJECT_DROP_TERMINAL_REWARD
    )
    reward = torch.where(
        finite,
        torch.where(
            absorbing,
            torch.zeros_like(raw_reward),
            torch.where(object_drop_new, terminal_penalty, raw_reward),
        ),
        terminal_penalty,
    )
    return object_drop_new, object_dropped, reward


def require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "MJWarp_RL.env requires the pinned Torch runtime"
        ) from _TORCH_IMPORT_ERROR
    return torch


def _validate_runtime_config(config: HPCConfig) -> None:
    """Validate structural runtime invariants without enforcing run budgets."""

    contract.validate_task(config.task)
    for component_name in ("elite", "diffusion"):
        component = getattr(config, component_name, None)
        if component is not None and hasattr(component, "validate"):
            component.validate()
    if config.num_envs <= 0 or config.num_eval_envs <= 0:
        raise ValueError("environment counts must be positive")
    for name in (
        "physics_substeps_per_control_tick",
        "plan_ticks",
        "close_ticks",
        "settle_ticks",
        "lift_ticks",
        "hold_ticks",
        "translation_action_ticks",
        "rotation_action_ticks",
        "translation_path_lead_in_ticks",
        "translation_segment_ticks",
        "rotation_motion_ticks",
        "contacts_per_world",
        "constraints_per_world",
    ):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        config.translation_action_ticks != contract.FORMAL_ACTION_TICKS
        or config.rotation_action_ticks
        != contract.ROTATION_FORMAL_ACTION_TICKS
        or config.translation_path_lead_in_ticks
        != contract.TRANSLATION_PATH_LEAD_IN_TICKS
        or config.translation_segment_ticks
        != contract.TRANSLATION_PATH_SEGMENT_TICKS
        or config.rotation_motion_ticks != contract.ROTATION_MOTION_TICKS
    ):
        raise ValueError("runtime task-motion timing differs from the contract")
    if (
        config.translation_motion_ticks > config.translation_action_ticks
        or config.rotation_motion_ticks > config.rotation_action_ticks
    ):
        raise ValueError("runtime task motion exceeds the action horizon")


def _minimum_jerk(value: Any) -> Any:
    value = value.clamp(0.0, 1.0)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def _translation_path_waypoints_torch(
    path_shape_index: Any,
    translating: Any,
    *,
    dtype: Any,
) -> Any:
    """Gather the five immutable 21-anchor path descriptors on-device."""

    device = path_shape_index.device
    table = torch.as_tensor(
        contract.TRANSLATION_PATH_WAYPOINTS_XY_M,
        device=device,
        dtype=dtype,
    )
    waypoints = table.index_select(
        0,
        path_shape_index.clamp(0, len(contract.TRANSLATION_PATH_SHAPES) - 1),
    )
    return torch.where(
        translating[:, None, None], waypoints, torch.zeros_like(waypoints)
    )


def _translation_path_point_torch(
    path_shape_index: Any,
    progress: Any,
    translating: Any,
) -> Any:
    """Evaluate all five analytic paths at normalized arc-length progress."""

    p = progress.clamp(0.0, 1.0)
    dtype = p.dtype
    device = p.device
    worlds = int(p.numel())

    side = contract.TRANSLATION_SQUARE_SIDE_M
    half = side / 2.0
    distance = 4.0 * side * p
    square = torch.stack(
        (
            torch.where(
                distance <= side,
                half - distance,
                torch.where(
                    distance <= 2.0 * side,
                    torch.full_like(distance, -half),
                    torch.where(
                        distance <= 3.0 * side,
                        -half + distance - 2.0 * side,
                        torch.full_like(distance, half),
                    ),
                ),
            ),
            torch.where(
                distance <= side,
                torch.full_like(distance, half),
                torch.where(
                    distance <= 2.0 * side,
                    half - (distance - side),
                    torch.where(
                        distance <= 3.0 * side,
                        torch.full_like(distance, -half),
                        -half + distance - 3.0 * side,
                    ),
                ),
            ),
        ),
        dim=-1,
    )

    radius = contract.TRANSLATION_CIRCLE_RADIUS_M
    angle = 2.0 * math.pi * p
    circle = radius * torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    circle = torch.where(
        (p >= 1.0)[:, None],
        torch.tensor((radius, 0.0), device=device, dtype=dtype),
        circle,
    )

    letter_distance = contract.TRANSLATION_LETTER_PATH_LENGTH_M * p
    s_radius = 0.100 / math.pi
    top_angle = math.pi / 2.0 + letter_distance / s_radius
    bottom_angle = (
        math.pi / 2.0 - (letter_distance - 0.100) / s_radius
    )
    letter_s = torch.where(
        (letter_distance <= 0.100)[:, None],
        torch.stack(
            (
                s_radius * torch.cos(top_angle),
                s_radius + s_radius * torch.sin(top_angle),
            ),
            dim=-1,
        ),
        torch.stack(
            (
                s_radius * torch.cos(bottom_angle),
                -s_radius + s_radius * torch.sin(bottom_angle),
            ),
            dim=-1,
        ),
    )
    letter_s = torch.where(
        (p <= 0.0)[:, None],
        torch.tensor((0.0, 2.0 * s_radius), device=device, dtype=dtype),
        torch.where(
            (p >= 1.0)[:, None],
            torch.tensor((0.0, -2.0 * s_radius), device=device, dtype=dtype),
            letter_s,
        ),
    )

    u_radius = contract.TRANSLATION_U_RADIUS_M
    u_vertical = (
        contract.TRANSLATION_LETTER_PATH_LENGTH_M - math.pi * u_radius
    ) / 2.0
    u_center_y = (u_radius - u_vertical) / 2.0
    u_angle = math.pi + (letter_distance - u_vertical) / u_radius
    letter_u = torch.where(
        (letter_distance <= u_vertical)[:, None],
        torch.stack(
            (
                torch.full_like(p, -u_radius),
                u_center_y + u_vertical - letter_distance,
            ),
            dim=-1,
        ),
        torch.where(
            (letter_distance <= u_vertical + math.pi * u_radius)[:, None],
            torch.stack(
                (
                    u_radius * torch.cos(u_angle),
                    u_center_y + u_radius * torch.sin(u_angle),
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    torch.full_like(p, u_radius),
                    u_center_y
                    + letter_distance
                    - u_vertical
                    - math.pi * u_radius,
                ),
                dim=-1,
            ),
        ),
    )

    scale = contract.TRANSLATION_LETTER_PATH_LENGTH_M / 13.0
    width = 3.0 * scale
    height = 4.0 * scale
    diagonal = 5.0 * scale
    diagonal_distance = letter_distance - height
    letter_n = torch.where(
        (letter_distance <= height)[:, None],
        torch.stack(
            (
                torch.full_like(p, -width / 2.0),
                -height / 2.0 + letter_distance,
            ),
            dim=-1,
        ),
        torch.where(
            (letter_distance <= height + diagonal)[:, None],
            torch.stack(
                (
                    -width / 2.0 + width * diagonal_distance / diagonal,
                    height / 2.0 - height * diagonal_distance / diagonal,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    torch.full_like(p, width / 2.0),
                    -height / 2.0
                    + letter_distance
                    - height
                    - diagonal,
                ),
                dim=-1,
            ),
        ),
    )

    candidates = torch.stack((square, circle, letter_s, letter_u, letter_n), dim=1)
    selected = candidates[
        torch.arange(worlds, device=device),
        path_shape_index.clamp(0, len(contract.TRANSLATION_PATH_SHAPES) - 1),
    ]
    return torch.where(
        translating[:, None], selected, torch.zeros_like(selected)
    )


def _cage_rotation_fraction(
    linear_fraction: Any,
    smooth_fraction: Any,
    object_index: Any,
    task_index: Any,
) -> Any:
    """Return minimum-jerk task-space progress for every trained object."""

    del linear_fraction, object_index, task_index
    return smooth_fraction


def _reported_action_fraction(
    linear_fraction: Any,
    smooth_fraction: Any,
    object_index: Any,
    task_index: Any,
) -> Any:
    """Return the minimum-jerk execution progress in the observation."""

    del linear_fraction, object_index, task_index
    return smooth_fraction


def _hold_boundaries(config: HPCConfig) -> tuple[int, int]:
    hold_start = (
        config.plan_ticks
        + config.close_ticks
        + config.settle_ticks
        + config.lift_ticks
    )
    action_start = hold_start + config.hold_ticks
    freeze_start = min(
        hold_start + MOTION_HOLD_FEEDBACK_TICKS,
        action_start,
    )
    return freeze_start, action_start


def _hold_freeze_window(control_step: Any, config: HPCConfig) -> Any:
    freeze_start, action_start = _hold_boundaries(config)
    return (control_step >= freeze_start) & (control_step < action_start)


def _hold_reference_capture(
    next_step: Any,
    reference_valid: Any,
    config: HPCConfig,
) -> Any:
    freeze_start, _ = _hold_boundaries(config)
    return (next_step == freeze_start) & (~reference_valid)


def _quat_normalize(value: Any) -> Any:
    result = value / value.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return torch.where(result[..., :1] < 0.0, -result, result)


def _quat_conjugate(value: Any) -> Any:
    return torch.cat((value[..., :1], -value[..., 1:]), dim=-1)


def _quat_mul(left: Any, right: Any) -> Any:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return _quat_normalize(
        torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dim=-1,
        )
    )


def _quat_rotate(quaternion: Any, points: Any) -> Any:
    """Rotate batched ``(..., 3)`` points by wxyz quaternions."""

    q = _quat_normalize(quaternion)
    vector = q[..., 1:]
    scalar = q[..., :1]
    while vector.ndim < points.ndim:
        vector = vector.unsqueeze(-2)
        scalar = scalar.unsqueeze(-2)
    twice_cross = 2.0 * torch.cross(
        vector.expand_as(points), points, dim=-1
    )
    return (
        points
        + scalar * twice_cross
        + torch.cross(vector.expand_as(points), twice_cross, dim=-1)
    )


def _quat_slerp(start: Any, target: Any, fraction: Any) -> Any:
    start = _quat_normalize(start)
    target = _quat_normalize(target)
    dot = (start * target).sum(dim=-1, keepdim=True)
    target = torch.where(dot < 0.0, -target, target)
    dot = dot.abs().clamp(0.0, 1.0)
    angle = torch.acos(dot)
    sine = torch.sin(angle)
    weight = fraction.reshape(-1, 1)
    linear = _quat_normalize((1.0 - weight) * start + weight * target)
    spherical = (
        torch.sin((1.0 - weight) * angle) / sine.clamp_min(1.0e-7)
    ) * start + (
        torch.sin(weight * angle) / sine.clamp_min(1.0e-7)
    ) * target
    return torch.where(
        sine < 1.0e-5, linear, _quat_normalize(spherical)
    )


def _relative_euler_xyz_deg(goal: Any, actual: Any) -> Any:
    """Wrapped fixed-XYZ Euler error of ``goal -> actual`` in degrees."""

    q = _quat_mul(_quat_conjugate(goal), actual)
    w, x, y, z = q.unbind(dim=-1)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    beta = torch.asin((-r20).clamp(-1.0, 1.0))
    regular = torch.cos(beta).abs() > 1.0e-6
    alpha = torch.where(
        regular, torch.atan2(r21, r22), torch.zeros_like(beta)
    )
    gamma = torch.where(
        regular, torch.atan2(r10, r00), torch.atan2(-r01, r11)
    )
    return torch.stack((alpha, beta, gamma), dim=-1) * (
        180.0 / math.pi
    )


def _yaw_rotation(angle: Any) -> Any:
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    return torch.stack(
        (
            torch.stack((cosine, -sine, zero), dim=-1),
            torch.stack((sine, cosine, zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ),
        dim=-2,
    )


def _rotation_delta(
    rotation_angle_deg: Any, *, device: Any, dtype: Any
) -> Any:
    """Return one local-Z goal quaternion for every requested angle."""

    angle = torch.as_tensor(
        rotation_angle_deg, device=device, dtype=dtype
    ).reshape(-1)
    half = angle * (math.pi / 360.0)
    zero = torch.zeros_like(half)
    return torch.stack(
        (torch.cos(half), zero, zero, torch.sin(half)), dim=-1
    )


def _post_lift_reference(
    action_start_com: Any,
    action_start_quat: Any,
    commanded_cage_world: Any,
    translation_world: Any,
    task_index: Any,
    rotation_angle_deg: Any,
) -> tuple[Any, Any, Any]:
    """Return captured cage, final COM goal, and final quaternion goal.

    The cage is the previous control tick's *commanded* cage, expressed in the
    actual post-lift object frame.  This exactly preserves the loaded grasp at
    the phase boundary instead of baking hard-tip tracking error into the new
    nominal geometry.
    """

    cage_local = _quat_rotate(
        _quat_conjugate(action_start_quat),
        commanded_cage_world - action_start_com[:, None, :],
    )
    translating = task_index == 0
    goal_position = action_start_com + torch.where(
        translating[:, None],
        translation_world,
        torch.zeros_like(translation_world),
    )
    rotated = _quat_mul(
        action_start_quat,
        _rotation_delta(
            rotation_angle_deg,
            device=action_start_com.device,
            dtype=action_start_com.dtype,
        ),
    )
    goal_quaternion = torch.where(
        translating[:, None], action_start_quat, rotated
    )
    return cage_local, goal_position, goal_quaternion


def action_mask_from_observation(observation: Any) -> Any:
    """Mask object- and phase-inapplicable policy channels."""

    require_torch()
    if observation.shape[-1] != OBSERVATION_DIM:
        raise ValueError(
            f"observation must end in {OBSERVATION_DIM} values"
        )
    phase_slice = OBSERVATION_SLICES["phase_one_hot"]
    plan = observation[..., phase_slice.start] > 0.5
    object_slice = OBSERVATION_SLICES["object_one_hot"]
    learned_plane = torch.zeros_like(plan)
    for object_index, object_kind in enumerate(contract.OBJECTS):
        if contract.grasp_plane_is_learned(object_kind):
            learned_plane |= (
                observation[..., object_slice.start + object_index] > 0.5
            )
    mask = torch.ones(
        (*observation.shape[:-1], contract.ACTION_DIM),
        device=observation.device,
        dtype=observation.dtype,
    )
    mask[..., 0:5] *= learned_plane[..., None].to(mask.dtype)
    mask[..., 0:9] *= plan[..., None].to(mask.dtype)
    mask[..., contract.FORCE_SCALE_SLICE] *= plan[..., None].to(mask.dtype)
    return mask


def _construct_fail_closed(
    constructor: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Construct a CUDA owner while retaining a handle for partial cleanup."""

    if not isinstance(constructor, type):
        return constructor(*args, **kwargs)
    instance = constructor.__new__(constructor)
    try:
        constructor.__init__(instance, *args, **kwargs)
    except BaseException:
        try:
            close = getattr(instance, "close", None)
            if callable(close):
                close()
        except BaseException:
            # Preserve the constructor failure.  The process-wide memory
            # guard will retain any cleanup failure as elevated used bytes.
            pass
        raise
    return instance


class MJWarpIKResidualEnv:
    """One native MJWarp backend with ``worlds`` Torch-vectorized episodes."""

    def __init__(
        self,
        *,
        curriculum_phase: str,
        config: HPCConfig = DEFAULT_CONFIG,
        residual_enabled: bool = True,
        platform_spec: PlatformSpec | None = None,
        worlds: int | None = None,
        capacity_num_envs: int | None = None,
        device: str = "cuda:0",
        require_linux: bool = True,
        require_cuda: bool = True,
        fixed_object: int | None = None,
        fixed_task: int | None = None,
        fixed_path_seed: int | None = None,
        fixed_path_shape_index: int | None = None,
        fixed_rotation_angle_deg: int | None = None,
        fixed_template_index: int | None = None,
        backend: Any | None = None,
        seed: int | None = None,
        validate_official_config: bool = True,
    ) -> None:
        require_torch()
        if curriculum_phase != PHASE:
            raise ValueError(f"unknown curriculum phase: {curriculum_phase!r}")
        if validate_official_config:
            config.validate()
        else:
            _validate_runtime_config(config)
        self.config = config
        self.curriculum_phase = curriculum_phase
        self.residual_enabled = bool(residual_enabled)
        self.platform = platform_spec or build_platform_spec(config.task)
        if getattr(self.platform, "task", config.task) != config.task:
            raise ValueError("platform task disagrees with config.task")
        platform_fixed_object = getattr(
            self.platform, "fixed_object_index", None
        )
        if platform_fixed_object is not None:
            platform_fixed_object = int(platform_fixed_object)
            if fixed_object is None:
                fixed_object = platform_fixed_object
            elif int(fixed_object) != platform_fixed_object:
                raise ValueError(
                    "fixed_object disagrees with the single-object model"
                )
        self.fixed_object = fixed_object
        if fixed_task is None:
            fixed_task = config.task_index
        elif int(fixed_task) != config.task_index:
            raise ValueError("fixed_task disagrees with config.task")
        self.fixed_task = int(fixed_task)
        self.fixed_path_seed = fixed_path_seed
        self.fixed_path_shape_index = fixed_path_shape_index
        self.fixed_rotation_angle_deg = fixed_rotation_angle_deg
        self.fixed_template_index = fixed_template_index
        self._validate_fixed_episode()
        requested_worlds = int(
            worlds or capacity_num_envs or config.num_envs
        )
        self.backend = backend or _construct_fail_closed(
            MJWarpBackend,
            config,
            self.platform,
            worlds=requested_worlds,
            device=device,
            require_linux=require_linux,
            require_cuda=require_cuda,
        )
        self.torch = getattr(self.backend, "torch", torch)
        self.device = self.backend.device
        self.worlds = int(self.backend.worlds)
        if self.worlds != requested_worlds:
            raise ValueError("backend world count differs from environment")
        self.dtype = self.backend.tensor("qpos").dtype
        backend_control_dt = float(getattr(self.backend, "control_dt", math.nan))
        if not math.isclose(
            backend_control_dt,
            contract.TRANSLATION_CONTROL_DT_S,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "backend control period differs from the certified path schedule"
            )
        self._nv = int(self.platform.host_model.nv)
        self._rows = self.torch.arange(
            self.worlds, device=self.device, dtype=self.torch.long
        )
        self._generator = self.torch.Generator(device=self.device)
        self._generator.manual_seed(int(config.seed if seed is None else seed))

        def floating(value: Any) -> Any:
            return self.torch.as_tensor(
                np.asarray(value), device=self.device, dtype=self.dtype
            )

        def integer(value: Any) -> Any:
            return self.torch.as_tensor(
                np.asarray(value),
                device=self.device,
                dtype=self.torch.long,
            )

        self._qpos_templates = floating(
            self.platform.qpos_templates
        ).reshape(-1, int(self.platform.host_model.nq))
        self._ctrl_templates = floating(
            self.platform.ctrl_templates
        ).reshape(-1, int(self.platform.host_model.nu))
        self._pregrasp_local = floating(
            self.platform.pregrasp_targets_local
        ).reshape(-1, len(contract.TIP_ORDER), 3)
        self._contact_local = floating(
            self.platform.contact_targets_local
        ).reshape(-1, len(contract.TIP_ORDER), 3)
        self._outward_local = floating(
            self.platform.outward_directions_local
        ).reshape(-1, len(contract.TIP_ORDER), 3)
        self._object_local_bottom_z_m = floating(
            self.platform.object_local_bottom_z_m
        )
        self._object_local_top_z_m = floating(
            self.platform.object_local_top_z_m
        )
        self._grasp_plane_nominal_fraction_by_object = floating(
            [
                contract.grasp_plane_nominal_fraction(kind)
                for kind in contract.OBJECTS
            ]
        )
        self._grasp_plane_fraction_limits_by_object = floating(
            [
                contract.grasp_plane_fraction_limits(kind)
                for kind in contract.OBJECTS
            ]
        )
        self._grasp_plane_learned_by_object = self.torch.as_tensor(
            [
                contract.grasp_plane_is_learned(kind)
                for kind in contract.OBJECTS
            ],
            device=self.device,
            dtype=self.torch.bool,
        )
        self._fruit_by_object = self.torch.as_tensor(
            [kind in contract.FRUIT_OBJECTS for kind in contract.OBJECTS],
            device=self.device,
            dtype=self.torch.bool,
        )
        self._object_body_ids = integer(self.platform.object_body_ids)
        self._object_qpos_adrs = integer(self.platform.object_qpos_adrs)
        self._object_qvel_adrs = integer(self.platform.object_qvel_adrs)
        self._object_qpos_adr_values = tuple(
            int(value)
            for value in np.asarray(self.platform.object_qpos_adrs).reshape(-1)
        )
        self._object_qvel_adr_values = tuple(
            int(value)
            for value in np.asarray(self.platform.object_qvel_adrs).reshape(-1)
        )
        free_joint_ids = np.flatnonzero(
            np.asarray(self.platform.host_model.jnt_type) == 0
        )
        self._free_qpos_adr_values = tuple(
            int(value)
            for value in np.asarray(
                self.platform.host_model.jnt_qposadr
            )[free_joint_ids]
        )
        self._free_qvel_adr_values = tuple(
            int(value)
            for value in np.asarray(
                self.platform.host_model.jnt_dofadr
            )[free_joint_ids]
        )
        if not set(self._object_qpos_adr_values).issubset(
            self._free_qpos_adr_values
        ):
            raise ValueError("every training object must use a free joint")
        self._object_geom_mask = self.torch.as_tensor(
            np.asarray(self.platform.object_geom_mask),
            device=self.device,
            dtype=self.torch.bool,
        )
        self._tip_body_ids = integer(self.platform.tip_body_ids)
        self._tip_body_id_values = tuple(
            int(value)
            for value in np.asarray(self.platform.tip_body_ids).reshape(-1)
        )
        self._tip_geom_ids = integer(self.platform.tip_geom_ids)
        self._actuator_qpos_adrs = integer(
            self.platform.actuator_qpos_adrs
        )
        self._actuator_dof_adrs = integer(
            self.platform.actuator_dof_adrs
        )
        self._passive_qpos_adrs = integer(
            self.platform.passive_qpos_adrs
        )
        self._passive_dof_adrs = integer(
            self.platform.passive_dof_adrs
        )
        self._passive_qpos_low = floating(
            self.platform.passive_qpos_low
        )
        self._passive_qpos_high = floating(
            self.platform.passive_qpos_high
        )
        self._pin_site_ids = integer(self.platform.pin_site_ids)
        self._pin_site_id_values = tuple(
            tuple(int(value) for value in row)
            for row in np.asarray(self.platform.pin_site_ids)
        )
        self._pin_body_ids = integer(self.platform.pin_body_ids)
        self._finger_actuator_indices = integer(
            self.platform.finger_actuator_indices
        )
        self._finger_passive_indices = integer(
            self.platform.finger_passive_indices
        )
        active_slots = int(self._finger_actuator_indices.shape[1])
        if active_slots > 4:
            raise ValueError("at most four active joints per hard tip are supported")
        self._active_joint_slots = active_slots
        active_valid = self.torch.zeros(
            (len(contract.TIP_ORDER), 4),
            device=self.device,
            dtype=self.dtype,
        )
        active_valid[:, :active_slots] = 1.0
        self._active_joint_valid_mask = active_valid
        self._ctrl_low = floating(self.platform.ctrl_low)
        self._ctrl_high = floating(self.platform.ctrl_high)
        width = self._ctrl_high - self._ctrl_low
        self._ctrl_linear_mask = width < 0.20
        self._ctrl_rate = self.torch.where(
            self._ctrl_linear_mask,
            self.torch.full_like(width, 0.002),
            self.torch.full_like(width, 0.055),
        )
        self._ctrl_acceleration_limit = self.torch.where(
            self._ctrl_linear_mask,
            self.torch.full_like(width, CTRL_LINEAR_ACCELERATION_LIMIT),
            self.torch.full_like(width, CTRL_ROTARY_ACCELERATION_LIMIT),
        )
        self._ctrl_jerk_limit = self.torch.where(
            self._ctrl_linear_mask,
            self.torch.full_like(width, CTRL_LINEAR_JERK_LIMIT),
            self.torch.full_like(width, CTRL_ROTARY_JERK_LIMIT),
        )
        self._nominal_preload = floating(
            NOMINAL_PRELOAD_BY_OBJECT_M
        )
        self._translation_route_progress_table = floating(
            contract.TRANSLATION_ROUTE_PROGRESS_TABLE
        )
        self._translation_lead_progress_table = floating(
            contract.TRANSLATION_LEAD_PROGRESS_TABLE
        )
        self._translation_route_targets_xy_table = floating(
            contract.TRANSLATION_ROUTE_TARGETS_XY_M
        )
        self._translation_lead_offsets_xy_table = floating(
            contract.TRANSLATION_LEAD_OFFSETS_XY_M
        )
        self._world_tip_body_ids = self._tip_body_ids[None, :].expand(
            self.worlds, -1
        )
        self._world_pin_body_ids = self._pin_body_ids[None, :, :].expand(
            self.worlds, -1, -1
        )
        self._kkt_eye: dict[int, Any] = {}
        self._allocate_state()

    def _allocate_state(self) -> None:
        t = self.torch
        w = self.worlds
        f = len(contract.TIP_ORDER)
        self.control_step = t.zeros(w, device=self.device, dtype=t.long)
        self.object_index = t.zeros(w, device=self.device, dtype=t.long)
        self.template_index = t.zeros(w, device=self.device, dtype=t.long)
        self.task_index = t.zeros(w, device=self.device, dtype=t.long)
        self.source_region = t.zeros(w, device=self.device, dtype=t.long)
        self.path_seed = t.zeros(w, device=self.device, dtype=t.long)
        self.path_shape_index = t.zeros(w, device=self.device, dtype=t.long)
        self.path_step_m = t.zeros(w, device=self.device, dtype=self.dtype)
        self.path_length_m = t.zeros(w, device=self.device, dtype=self.dtype)
        self.path_waypoints_xy_m = t.zeros(
            (w, contract.TRANSLATION_PATH_WAYPOINTS, 2),
            device=self.device,
            dtype=self.dtype,
        )
        self.path_tracking_squared_error_sum = t.zeros(
            w, device=self.device, dtype=self.dtype
        )
        self.path_tracking_max_error_m = t.zeros_like(
            self.path_tracking_squared_error_sum
        )
        self.path_tracking_axis_abs_max_m = t.zeros(
            (w, 3), device=self.device, dtype=self.dtype
        )
        self.path_tracking_sample_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.path_tracking_current_error_m = t.zeros_like(
            self.path_tracking_squared_error_sum
        )
        # Route-schedule certification is deliberately independent of object
        # tracking.  It proves that every scheduled target was emitted in
        # order and stayed inside the hard velocity/acceleration envelope.
        self.path_schedule_previous_offset_xy_m = t.zeros(
            (w, 2), device=self.device, dtype=self.dtype
        )
        self.path_schedule_previous_velocity_xy_m_s = t.zeros_like(
            self.path_schedule_previous_offset_xy_m
        )
        self.path_schedule_progress = t.zeros(
            w, device=self.device, dtype=self.dtype
        )
        self.path_schedule_limit_valid = t.ones(
            w, device=self.device, dtype=t.bool
        )
        self.path_schedule_complete = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.path_schedule_max_speed_m_s = t.zeros_like(
            self.path_schedule_progress
        )
        self.path_schedule_max_acceleration_m_s2 = t.zeros_like(
            self.path_schedule_progress
        )
        # First route violation is retained per world so a strict route-wide
        # maximum certificate remains diagnosable after the episode endpoint.
        # Sample/control ticks are one-based; -1 and a zero error vector are
        # the explicit sentinels for a world that never exceeded tolerance.
        self.path_tracking_first_violation_sample_tick = t.full(
            (w,), -1, device=self.device, dtype=t.long
        )
        self.path_tracking_first_violation_control_tick = t.full(
            (w,), -1, device=self.device, dtype=t.long
        )
        self.path_tracking_first_violation_axis_index = t.full(
            (w,), -1, device=self.device, dtype=t.long
        )
        self.path_tracking_first_violation_error_xyz_abs_m = t.zeros(
            (w, 3), device=self.device, dtype=self.dtype
        )
        self.path_current_target = t.zeros(
            (w, 3), device=self.device, dtype=self.dtype
        )
        self.rotation_angle_deg = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.initial_body_pos = t.zeros(
            (w, 3), device=self.device, dtype=self.dtype
        )
        self.initial_com = t.zeros_like(self.initial_body_pos)
        self.initial_quat = t.zeros(
            (w, 4), device=self.device, dtype=self.dtype
        )
        self.initial_quat[:, 0] = 1.0
        self.translation = t.zeros_like(self.initial_body_pos)
        self.goal_pos = t.zeros_like(self.initial_body_pos)
        self.goal_quat = self.initial_quat.clone()
        self.filtered_action = t.zeros(
            (w, contract.ACTION_DIM),
            device=self.device,
            dtype=self.dtype,
        )
        self.frozen_action = t.zeros_like(self.filtered_action)
        self.previous_action = t.zeros_like(self.filtered_action)
        self.preload_trim_m = t.zeros(
            (w, f), device=self.device, dtype=self.dtype
        )
        self.position_feedback_m = t.zeros(
            (w, 3), device=self.device, dtype=self.dtype
        )
        self.ctrl_velocity = t.zeros(
            (w, int(self.platform.host_model.nu)),
            device=self.device,
            dtype=self.dtype,
        )
        self.ctrl_acceleration = t.zeros_like(self.ctrl_velocity)
        self.last_certified_ctrl = t.zeros_like(self.ctrl_velocity)
        self.ik_trust_region_projection_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.ik_trust_region_projection_count_by_stage = t.zeros(
            (w, len(IK_TRUST_STAGES)), device=self.device, dtype=t.long
        )
        self.ik_cage_projection_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.ik_cage_projection_count_by_stage = t.zeros(
            (w, len(IK_TRUST_STAGES)), device=self.device, dtype=t.long
        )
        self.ik_cage_projection_scale_min = t.ones(
            w, device=self.device, dtype=self.dtype
        )
        self.ik_cage_projection_scale_sum = t.zeros_like(
            self.ik_cage_projection_scale_min
        )
        self.ik_cage_projection_sample_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.position_feedback_slew_clip_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.preload_trim_slew_clip_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.ctrl_slew_clip_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.ctrl_certificate_fallback_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.position_error_m = t.zeros(
            w, device=self.device, dtype=self.dtype
        )
        self.euler_l2_error_deg = t.zeros_like(self.position_error_m)
        self.success = t.zeros(w, device=self.device, dtype=t.bool)
        self.time_out = t.zeros(w, device=self.device, dtype=t.bool)
        self.done = t.zeros(w, device=self.device, dtype=t.bool)
        self.object_dropped = t.zeros(w, device=self.device, dtype=t.bool)
        self.object_drop_new = t.zeros(w, device=self.device, dtype=t.bool)
        self.object_drop_first_control_step = t.full(
            (w,), -1, device=self.device, dtype=t.long
        )
        self.object_drop_first_position = t.zeros_like(
            self.initial_body_pos
        )
        self.object_drop_first_body_position = t.zeros_like(
            self.initial_body_pos
        )
        self.object_drop_first_quaternion = self.initial_quat.clone()
        self.object_drop_first_bottom_z_m = t.zeros(
            w, device=self.device, dtype=self.dtype
        )
        self.ik_solve_fallback_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self._last_ik_solve_feasible = t.ones(
            w, device=self.device, dtype=t.bool
        )
        self.action_reference_valid = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.action_start_com = t.zeros_like(self.initial_body_pos)
        self.action_start_quat = self.initial_quat.clone()
        self.action_start_cage_local = t.zeros(
            (w, f, 3), device=self.device, dtype=self.dtype
        )
        self.action_start_action = t.zeros_like(self.filtered_action)
        self.action_start_preload_trim_m = t.zeros_like(
            self.preload_trim_m
        )
        self.action_start_position_feedback_m = t.zeros_like(
            self.position_feedback_m
        )
        self.hold_reference_valid = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.hold_frozen_ctrl = t.zeros(
            (w, int(self.platform.host_model.nu)),
            device=self.device,
            dtype=self.dtype,
        )
        self.hold_frozen_cage_world = t.zeros(
            (w, f, 3), device=self.device, dtype=self.dtype
        )
        self._last_contact_force = t.zeros_like(self.preload_trim_m)
        self._last_force_decode_finite = t.ones(
            w, device=self.device, dtype=t.bool
        )
        self._last_tip_object_contact_raw_pair = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self._last_tip_object_contact_pair = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self._last_tip_object_contact_force_qualified_pair = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        # Fail-closed pre-lift admission.  State is kept per world so the
        # sharded runtime can report it by object without any host-side
        # synchronization.  Stage matrices preserve whether contact failed
        # while the cage was closing or while it was settling.
        self.prelift_contact_stable_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.prelift_contact_max_stable_count = t.zeros_like(
            self.prelift_contact_stable_count
        )
        self.prelift_contact_gate_latched = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.prelift_contact_gate_first_control_step = t.full(
            (w,), -1, device=self.device, dtype=t.long
        )
        self.prelift_lift_boundary_contact_valid = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.prelift_lift_permitted = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.prelift_lift_reference_valid = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.prelift_lift_start_body_pos = self.initial_body_pos.clone()
        self.prelift_lift_start_quat = self.initial_quat.clone()
        self.prelift_contact_count_current = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.prelift_tip_raw_pair_current = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self.prelift_tip_active_pair_current = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self.prelift_tip_contact_qualified_current = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self.prelift_contact_stage_index = t.full(
            (w,), -1, device=self.device, dtype=t.long
        )
        prelift_shape = (w, len(PRELIFT_CONTACT_STAGES))
        self.prelift_stage_observation_count = t.zeros(
            prelift_shape, device=self.device, dtype=t.long
        )
        self.prelift_stage_qualified_count = t.zeros_like(
            self.prelift_stage_observation_count
        )
        self.prelift_stage_failure_count = t.zeros_like(
            self.prelift_stage_observation_count
        )
        self.prelift_stage_max_contact_count = t.zeros_like(
            self.prelift_stage_observation_count
        )
        self.prelift_stage_first_failure_control_step = t.full(
            prelift_shape, -1, device=self.device, dtype=t.long
        )
        self.prelift_stage_last_failure_control_step = t.full_like(
            self.prelift_stage_first_failure_control_step, -1
        )
        self.prelift_tip_pair_missing_count_by_stage = t.zeros(
            (w, len(PRELIFT_CONTACT_STAGES), f),
            device=self.device,
            dtype=t.long,
        )
        self.prelift_tip_raw_pair_missing_count_by_stage = t.zeros_like(
            self.prelift_tip_pair_missing_count_by_stage
        )
        self.prelift_tip_efc_inactive_count_by_stage = t.zeros_like(
            self.prelift_tip_pair_missing_count_by_stage
        )
        self.prelift_tip_force_failure_count_by_stage = t.zeros_like(
            self.prelift_tip_pair_missing_count_by_stage
        )
        self.prelift_lift_boundary_tip_raw_pair = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self.prelift_lift_boundary_tip_active_pair = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self.prelift_lift_boundary_tip_contact_qualified = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self.prelift_lift_boundary_contact_force_n = t.zeros(
            (w, f), device=self.device, dtype=self.dtype
        )
        self._last_raw_observation_finite = t.ones(
            w, device=self.device, dtype=t.bool
        )
        self._last_targets = t.zeros(
            (w, f, 3), device=self.device, dtype=self.dtype
        )
        self.last_certified_tip_targets = t.zeros_like(self._last_targets)
        self.tip_target_velocity_world_m_s = t.zeros_like(
            self._last_targets
        )
        self.tip_target_acceleration_world_m_s2 = t.zeros_like(
            self._last_targets
        )
        self.tip_target_jerk_world_m_s3 = t.zeros_like(self._last_targets)
        self.tip_target_slew_clip_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.tip_target_certificate_fallback_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.translation_entry_stable_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.translation_entry_max_stable_count = t.zeros_like(
            self.translation_entry_stable_count
        )
        self.translation_entry_certificate_latched = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.translation_entry_certificate_first_control_step = t.full(
            (w,), -1, device=self.device, dtype=t.long
        )
        self.translation_route_admitted = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.translation_entry_route_start_safety_valid = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.translation_entry_gate_current = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.translation_entry_tip_height_valid = t.zeros_like(
            self.translation_entry_gate_current
        )
        self.translation_entry_lifted = t.zeros_like(
            self.translation_entry_gate_current
        )
        self.translation_entry_pose_valid = t.zeros_like(
            self.translation_entry_gate_current
        )
        self.translation_entry_start_position_valid = t.zeros_like(
            self.translation_entry_gate_current
        )
        self.translation_entry_start_orientation_valid = t.zeros_like(
            self.translation_entry_gate_current
        )
        self.translation_entry_linear_speed_valid = t.zeros_like(
            self.translation_entry_gate_current
        )
        self.translation_entry_angular_speed_valid = t.zeros_like(
            self.translation_entry_gate_current
        )
        self.translation_entry_predicate_current = t.zeros(
            (w, len(TRANSLATION_ENTRY_PREDICATES)),
            device=self.device,
            dtype=t.bool,
        )
        self.translation_entry_failure_count_by_predicate = t.zeros(
            (w, len(TRANSLATION_ENTRY_PREDICATES)),
            device=self.device,
            dtype=t.long,
        )
        self.translation_entry_first_failure_control_step = t.full(
            (w, len(TRANSLATION_ENTRY_PREDICATES)),
            -1,
            device=self.device,
            dtype=t.long,
        )
        self.translation_entry_last_failure_control_step = t.full_like(
            self.translation_entry_first_failure_control_step, -1
        )
        self.translation_entry_tip_height_fraction = t.zeros(
            (w, f), device=self.device, dtype=self.dtype
        )
        self.translation_entry_tip_object_pair = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self.translation_entry_tip_object_raw_pair = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self.translation_entry_tip_object_active_pair = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self.translation_entry_tip_contact_qualified = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self.translation_entry_tip_contact_force_n = t.zeros(
            (w, f), device=self.device, dtype=self.dtype
        )
        self.translation_entry_tip_force_valid = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self.translation_entry_tip_height_failure_count = t.zeros(
            (w, f), device=self.device, dtype=t.long
        )
        self.translation_entry_tip_pair_missing_count = t.zeros_like(
            self.translation_entry_tip_height_failure_count
        )
        self.translation_entry_tip_raw_pair_missing_count = t.zeros_like(
            self.translation_entry_tip_height_failure_count
        )
        self.translation_entry_tip_efc_inactive_count = t.zeros_like(
            self.translation_entry_tip_height_failure_count
        )
        self.translation_entry_tip_force_failure_count = t.zeros_like(
            self.translation_entry_tip_height_failure_count
        )
        self.translation_entry_position_error_xyz_abs_m = t.zeros(
            (w, 3), device=self.device, dtype=self.dtype
        )
        self.translation_entry_euler_error_xyz_abs_deg = t.zeros_like(
            self.translation_entry_position_error_xyz_abs_m
        )
        self.translation_entry_estimated_bottom_z_m = t.zeros(
            w, device=self.device, dtype=self.dtype
        )
        self.translation_entry_object_linear_speed_m_s = t.zeros(
            w, device=self.device, dtype=self.dtype
        )
        self.translation_entry_object_angular_speed_rad_s = t.zeros_like(
            self.translation_entry_object_linear_speed_m_s
        )
        self.expert_tip_object_contact = t.zeros(
            (w, f), device=self.device, dtype=t.bool
        )
        self.expert_contact_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.expert_four_tip_contact = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.expert_object_linear_speed_m_s = t.zeros(
            w, device=self.device, dtype=self.dtype
        )
        self.expert_object_angular_speed_rad_s = t.zeros_like(
            self.expert_object_linear_speed_m_s
        )
        self.terminal_stable_count = t.zeros(
            w, device=self.device, dtype=t.long
        )
        self.terminal_stable_certified = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.terminal_stability_gate_current = t.zeros(
            w, device=self.device, dtype=t.bool
        )
        self.cage_orientation_error_rad = t.zeros(
            w, device=self.device, dtype=self.dtype
        )
        # Each dense-Newton shard contains exactly one trained free object.
        self._parked_free_qpos = t.zeros(
            (w, len(getattr(self, "_free_qpos_adr_values", ())), 7),
            device=self.device,
            dtype=self.dtype,
        )
        self._object_drop_quarantine_qpos = t.zeros_like(
            self._parked_free_qpos
        )
        self._observation_tensor = t.zeros(
            (w, OBSERVATION_DIM), device=self.device, dtype=self.dtype
        )
        self._object_drop_terminal_observation = t.zeros_like(
            self._observation_tensor
        )
        self._ik_warmstart_backup = None
        if (
            getattr(self, "backend", None) is not None
            and self.backend.has_field("qacc_warmstart")
        ):
            self._ik_warmstart_backup = t.empty_like(
                self.backend.tensor("qacc_warmstart")
            )

    def _validate_fixed_episode(self) -> None:
        if self.fixed_object is not None and not (
            0 <= self.fixed_object < len(contract.OBJECTS)
        ):
            raise ValueError("fixed_object is outside contract.OBJECTS")
        if self.fixed_task is not None and self.fixed_task not in (0, 1):
            raise ValueError("fixed_task must be 0 (translate) or 1 (rotate)")
        if self.fixed_path_seed is not None and not (
            isinstance(self.fixed_path_seed, int)
            and contract.TRANSLATION_PATH_SEED_MIN
            <= self.fixed_path_seed
            <= contract.TRANSLATION_PATH_SEED_MAX
        ):
            raise ValueError("fixed_path_seed is outside the path seed contract")
        if self.fixed_path_shape_index is not None and not (
            isinstance(self.fixed_path_shape_index, int)
            and 0
            <= self.fixed_path_shape_index
            < len(contract.TRANSLATION_PATH_SHAPES)
        ):
            raise ValueError("fixed_path_shape_index is outside the shape table")
        rotation_low, rotation_high = rotation_angle_bounds_deg(
            self.curriculum_phase
        )
        if self.fixed_rotation_angle_deg is not None and not (
            isinstance(self.fixed_rotation_angle_deg, int)
            and rotation_low
            <= self.fixed_rotation_angle_deg
            <= rotation_high
        ):
            raise ValueError(
                "fixed_rotation_angle_deg must be an integer in "
                f"[{rotation_low}, {rotation_high}] for "
                f"{self.curriculum_phase}"
            )
        if self.fixed_template_index is not None and not (
            isinstance(self.fixed_template_index, int)
            and 0
            <= self.fixed_template_index
            < contract.RESET_TEMPLATES_PER_OBJECT
        ):
            raise ValueError(
                "fixed_template_index must address the reset template bank"
            )

    @property
    def observation_size(self) -> int:
        return OBSERVATION_DIM

    @property
    def action_size(self) -> int:
        return contract.ACTION_DIM

    @property
    def episode_length(self) -> int:
        return self.config.episode_length

    def _mask(self, mask: Any | None) -> Any:
        if mask is None:
            return self.torch.ones(
                self.worlds, device=self.device, dtype=self.torch.bool
            )
        result = self.torch.as_tensor(
            mask, device=self.device, dtype=self.torch.bool
        ).reshape(-1)
        if result.numel() != self.worlds:
            raise ValueError("reset mask must contain one value per world")
        return result

    def _goal_rows(
        self, value: Any, ids: Any, *, dtype: Any, label: str
    ) -> Any:
        tensor = self.torch.as_tensor(
            value, device=self.device, dtype=dtype
        ).reshape(-1)
        if tensor.numel() == 1:
            return tensor.expand(ids.numel())
        if tensor.numel() == self.worlds:
            return tensor.index_select(0, ids)
        if tensor.numel() == ids.numel():
            return tensor
        raise ValueError(
            f"{label} must be scalar, full-world, or selected-world length"
        )

    def _sample_goal_rows(
        self, ids: Any, goals: Mapping[str, Any] | None
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        count = int(ids.numel())
        t = self.torch
        object_index = t.randint(
            0,
            len(contract.OBJECTS),
            (count,),
            generator=self._generator,
            device=self.device,
        )
        template_index = t.randint(
            0,
            contract.RESET_TEMPLATES_PER_OBJECT,
            (count,),
            generator=self._generator,
            device=self.device,
        )
        task_index = t.randint(
            0,
            len(contract.TASKS),
            (count,),
            generator=self._generator,
            device=self.device,
        )
        source_region = t.full(
            (count,),
            0,
            device=self.device,
            dtype=t.long,
        )
        path_seed = t.randint(
            contract.TRANSLATION_PATH_SEED_MIN,
            contract.TRANSLATION_PATH_SEED_MAX + 1,
            (count,),
            generator=self._generator,
            device=self.device,
            dtype=t.long,
        )
        path_shape_index = t.randint(
            0,
            len(contract.TRANSLATION_PATH_SHAPES),
            (count,),
            generator=self._generator,
            device=self.device,
        )
        rotation_low, rotation_high = rotation_angle_bounds_deg(
            self.curriculum_phase
        )
        rotation_angle_deg = t.randint(
            rotation_low,
            rotation_high + 1,
            (count,),
            generator=self._generator,
            device=self.device,
        )
        fixed_object = getattr(self, "fixed_object", None)
        if fixed_object is not None:
            object_index.fill_(fixed_object)
        if self.fixed_template_index is not None:
            template_index.fill_(self.fixed_template_index)
        if self.fixed_task is not None:
            task_index.fill_(self.fixed_task)
        if self.fixed_path_seed is not None:
            path_seed.fill_(self.fixed_path_seed)
        if self.fixed_path_shape_index is not None:
            path_shape_index.fill_(self.fixed_path_shape_index)
        if self.fixed_rotation_angle_deg is not None:
            rotation_angle_deg.fill_(self.fixed_rotation_angle_deg)
        if goals is not None:
            fields = {
                "object_index": (object_index, t.long),
                "template_index": (template_index, t.long),
                "task_index": (task_index, t.long),
                "path_seed": (path_seed, t.long),
                "path_shape_index": (path_shape_index, t.long),
                "rotation_angle_deg": (rotation_angle_deg, t.long),
                "source_region": (source_region, t.long),
            }
            for name, (current, dtype) in fields.items():
                if name in goals:
                    current.copy_(
                        self._goal_rows(
                            goals[name],
                            ids,
                            dtype=dtype,
                            label=name,
                        )
                    )
        if fixed_object is not None and bool(
            (object_index != int(fixed_object)).any()
        ):
            raise ValueError("goals attempted to cross a fixed object shard")
        if self.fixed_task is not None and bool(
            (task_index != int(self.fixed_task)).any()
        ):
            raise ValueError("goals attempted to cross the fixed task bundle")
        translating = task_index == 0
        rotating = ~translating
        if goals is None or "source_region" not in goals:
            source_region.fill_(0)
        path_seed = t.where(translating, path_seed, t.zeros_like(path_seed))
        path_shape_index = t.where(
            translating, path_shape_index, t.zeros_like(path_shape_index)
        )
        rotation_angle_deg = t.where(
            rotating,
            rotation_angle_deg,
            t.zeros_like(rotation_angle_deg),
        )
        source_invalid = source_region != 0
        invalid = (
            (object_index < 0)
            | (object_index >= len(contract.OBJECTS))
            | (template_index < 0)
            | (
                template_index
                >= contract.RESET_TEMPLATES_PER_OBJECT
            )
            | (task_index < 0)
            | (task_index >= len(contract.TASKS))
            | (
                translating
                & (
                    (path_seed < contract.TRANSLATION_PATH_SEED_MIN)
                    | (path_seed > contract.TRANSLATION_PATH_SEED_MAX)
                )
            )
            | (rotating & (path_seed != 0))
            | (path_shape_index < 0)
            | (path_shape_index >= len(contract.TRANSLATION_PATH_SHAPES))
            | (
                rotating
                & (
                    (rotation_angle_deg < rotation_low)
                    | (rotation_angle_deg > rotation_high)
                )
            )
            | (translating & (rotation_angle_deg != 0))
            | source_invalid
        )
        if bool(invalid.any()):
            raise ValueError("explicit goal values are outside the contract")
        if fixed_object is not None:
            expected_object = int(fixed_object)
            wrong_object = object_index != expected_object
            if bool(wrong_object.any()):
                bad = self.torch.nonzero(
                    wrong_object, as_tuple=False
                ).flatten()[:8].tolist()
                raise ValueError(
                    "single-object shard goal routing mismatch: "
                    f"expected object_index={expected_object}, bad rows={bad}"
                )
        return (
            object_index,
            template_index,
            task_index,
            path_seed,
            path_shape_index,
            rotation_angle_deg,
            source_region,
        )

    def _template_rows(
        self,
        values: Any,
        object_index: Any | None = None,
        template_index: Any | None = None,
    ) -> Any:
        """Gather flattened ``[object, template, ...]`` bank rows."""

        objects = self.object_index if object_index is None else object_index
        templates = (
            self.template_index
            if template_index is None
            else template_index
        )
        flat_index = (
            objects * contract.RESET_TEMPLATES_PER_OBJECT + templates
        )
        return values.index_select(0, flat_index)

    def reset(
        self,
        mask: Any | None = None,
        goals: Mapping[str, Any] | None = None,
    ) -> Any:
        selected = self._mask(mask)
        ids = self.torch.nonzero(selected, as_tuple=False).flatten()
        if int(ids.numel()) == 0:
            return self._observation_tensor
        (
            object_index,
            template_index,
            task_index,
            path_seed,
            path_shape_index,
            rotation_angle_deg,
            source_region,
        ) = self._sample_goal_rows(ids, goals)
        fixed_object = getattr(self, "fixed_object", None)
        if fixed_object is not None and not bool(
            (object_index == int(fixed_object)).all()
        ):
            raise ValueError(
                "reset goals attempted to cross a fixed object-model shard"
            )
        qpos = self._template_rows(
            self._qpos_templates, object_index, template_index
        )
        ctrl = self._template_rows(
            self._ctrl_templates, object_index, template_index
        )
        self.backend.reset_worlds(selected, qpos=qpos, ctrl=ctrl)
        for slot, address in enumerate(
            getattr(self, "_free_qpos_adr_values", ())
        ):
            reset_free_qpos = qpos[:, address : address + 7]
            self._parked_free_qpos[:, slot].index_copy_(
                0, ids, reset_free_qpos
            )
            quarantine_qpos = reset_free_qpos.clone()
            quarantine_qpos[:, :3] = self.torch.as_tensor(
                contract.OBJECT_DROP_QUARANTINE_POSITION_M,
                device=self.device,
                dtype=self.dtype,
            )
            self._object_drop_quarantine_qpos[:, slot].index_copy_(
                0, ids, quarantine_qpos
            )

        xpos = self.backend.tensor("xpos")
        xipos = self.backend.tensor("xipos")
        xquat = self.backend.tensor("xquat")
        body_ids = self._object_body_ids.index_select(0, object_index)
        initial_body = xpos[ids, body_ids]
        initial_com = xipos[ids, body_ids]
        initial_quat = _quat_normalize(xquat[ids, body_ids])
        translating = task_index == contract.TASKS.index(
            contract.TASK_TRANSLATE
        )
        path_step_table = torch.as_tensor(
            contract.TRANSLATION_PATH_STEP_M_BY_SHAPE,
            device=self.device,
            dtype=self.dtype,
        )
        path_length_table = torch.as_tensor(
            contract.TRANSLATION_PATH_LENGTH_M_BY_SHAPE,
            device=self.device,
            dtype=self.dtype,
        )
        path_step_m = torch.where(
            translating,
            path_step_table.index_select(0, path_shape_index),
            torch.zeros((ids.numel(),), device=self.device, dtype=self.dtype),
        )
        path_waypoints = _translation_path_waypoints_torch(
            path_shape_index,
            translating,
            dtype=self.dtype,
        )
        waypoint_radius = path_waypoints.norm(dim=-1)
        path_invalid = translating & (
            (waypoint_radius.amax(dim=1)
             > contract.TRANSLATION_PATH_GENERATION_RADIUS_M + 2.0e-6)
            | (waypoint_radius[:, 0]
               > contract.TRANSLATION_PATH_START_RADIUS_MAX_M + 2.0e-6)
            | (~torch.isfinite(path_waypoints).all(dim=(1, 2)))
        )
        if bool(path_invalid.any()):
            raise RuntimeError("generated translation path violates its contract")
        path_length_m = torch.where(
            translating,
            path_length_table.index_select(0, path_shape_index),
            torch.zeros_like(path_step_m),
        )
        final_xy = path_waypoints[:, -1]
        translation = torch.cat(
            (final_xy, torch.zeros_like(final_xy[:, :1])), dim=-1
        )
        provisional_goal = initial_com + torch.tensor(
            (0.0, 0.0, LIFT_HEIGHT_M),
            device=self.device,
            dtype=self.dtype,
        )
        provisional_goal = provisional_goal + torch.where(
            (task_index == 0)[:, None],
            translation,
            torch.zeros_like(translation),
        )
        rotated_goal = _quat_mul(
            initial_quat,
            _rotation_delta(
                rotation_angle_deg,
                device=self.device,
                dtype=self.dtype,
            ),
        )
        provisional_quat = torch.where(
            (task_index == 1)[:, None], rotated_goal, initial_quat
        )

        def put(target: Any, value: Any) -> None:
            target.index_copy_(0, ids, value)

        put(self.object_index, object_index)
        put(self.template_index, template_index)
        put(self.task_index, task_index)
        put(self.path_seed, path_seed)
        put(self.path_shape_index, path_shape_index)
        put(self.path_step_m, path_step_m)
        put(self.path_length_m, path_length_m)
        put(self.path_waypoints_xy_m, path_waypoints)
        put(self.rotation_angle_deg, rotation_angle_deg)
        put(self.source_region, source_region)
        put(self.initial_body_pos, initial_body)
        put(self.initial_com, initial_com)
        put(self.initial_quat, initial_quat)
        put(self.translation, translation)
        put(self.goal_pos, provisional_goal)
        put(self.goal_quat, provisional_quat)
        self.control_step.index_fill_(0, ids, 0)
        for value in (
            self.filtered_action,
            self.frozen_action,
            self.previous_action,
            self.preload_trim_m,
            self.position_feedback_m,
            self.ctrl_velocity,
            self.ctrl_acceleration,
            self.action_start_cage_local,
            self.action_start_action,
            self.action_start_preload_trim_m,
            self.action_start_position_feedback_m,
            self._last_contact_force,
        ):
            value.index_fill_(0, ids, 0.0)
        self.last_certified_ctrl.index_copy_(0, ids, ctrl)
        self.action_start_com.index_copy_(0, ids, initial_com)
        self.action_start_quat.index_copy_(0, ids, initial_quat)
        self.action_reference_valid.index_fill_(0, ids, False)
        self.path_tracking_squared_error_sum.index_fill_(0, ids, 0.0)
        self.path_tracking_max_error_m.index_fill_(0, ids, 0.0)
        self.path_tracking_axis_abs_max_m.index_fill_(0, ids, 0.0)
        self.path_tracking_sample_count.index_fill_(0, ids, 0)
        self.path_tracking_current_error_m.index_fill_(0, ids, 0.0)
        self.path_schedule_previous_offset_xy_m.index_copy_(
            0, ids, path_waypoints[:, 0]
        )
        self.path_schedule_previous_velocity_xy_m_s.index_fill_(
            0, ids, 0.0
        )
        self.path_schedule_progress.index_fill_(0, ids, 0.0)
        self.path_schedule_limit_valid.index_fill_(0, ids, True)
        self.path_schedule_complete.index_fill_(0, ids, False)
        self.path_schedule_max_speed_m_s.index_fill_(0, ids, 0.0)
        self.path_schedule_max_acceleration_m_s2.index_fill_(0, ids, 0.0)
        self._reset_path_tracking_first_violation(ids)
        put(self.path_current_target, provisional_goal)
        self._reset_hold_reference(ids)
        self.position_error_m.index_copy_(
            0, ids, (provisional_goal - initial_com).norm(dim=-1)
        )
        initial_euler = _relative_euler_xyz_deg(
            provisional_quat, initial_quat
        ).norm(dim=-1)
        self.euler_l2_error_deg.index_copy_(0, ids, initial_euler)
        self.success.index_fill_(0, ids, False)
        self.time_out.index_fill_(0, ids, False)
        self.done.index_fill_(0, ids, False)
        self.object_dropped.index_fill_(0, ids, False)
        self.object_drop_new.index_fill_(0, ids, False)
        self.object_drop_first_control_step.index_fill_(0, ids, -1)
        put(self.object_drop_first_position, initial_com)
        put(self.object_drop_first_body_position, initial_body)
        put(self.object_drop_first_quaternion, initial_quat)
        local_bottom_bank = getattr(
            self, "_object_local_bottom_z_m", None
        )
        if local_bottom_bank is None:
            local_bottom_bank = self.torch.zeros(
                len(contract.OBJECTS),
                device=self.device,
                dtype=self.dtype,
            )
        initial_bottom = initial_body[:, 2] + local_bottom_bank.index_select(
            0, object_index
        )
        put(self.object_drop_first_bottom_z_m, initial_bottom)
        self.ik_solve_fallback_count.index_fill_(0, ids, 0)
        self.ik_trust_region_projection_count.index_fill_(0, ids, 0)
        self.ik_trust_region_projection_count_by_stage.index_fill_(
            0, ids, 0
        )
        self.ik_cage_projection_count.index_fill_(0, ids, 0)
        self.ik_cage_projection_count_by_stage.index_fill_(0, ids, 0)
        self.ik_cage_projection_scale_min.index_fill_(0, ids, 1.0)
        self.ik_cage_projection_scale_sum.index_fill_(0, ids, 0.0)
        self.ik_cage_projection_sample_count.index_fill_(0, ids, 0)
        self.position_feedback_slew_clip_count.index_fill_(0, ids, 0)
        self.preload_trim_slew_clip_count.index_fill_(0, ids, 0)
        self.ctrl_slew_clip_count.index_fill_(0, ids, 0)
        self.ctrl_certificate_fallback_count.index_fill_(0, ids, 0)
        self.tip_target_slew_clip_count.index_fill_(0, ids, 0)
        self.tip_target_certificate_fallback_count.index_fill_(0, ids, 0)
        self.translation_entry_stable_count.index_fill_(0, ids, 0)
        self.translation_entry_max_stable_count.index_fill_(0, ids, 0)
        self.translation_entry_certificate_latched.index_fill_(
            0, ids, False
        )
        self.translation_entry_certificate_first_control_step.index_fill_(
            0, ids, -1
        )
        self.translation_route_admitted.index_fill_(0, ids, False)
        self.translation_entry_route_start_safety_valid.index_fill_(
            0, ids, False
        )
        self.translation_entry_gate_current.index_fill_(0, ids, False)
        self.translation_entry_tip_height_valid.index_fill_(0, ids, False)
        self.translation_entry_lifted.index_fill_(0, ids, False)
        self.translation_entry_pose_valid.index_fill_(0, ids, False)
        self.translation_entry_start_position_valid.index_fill_(
            0, ids, False
        )
        self.translation_entry_start_orientation_valid.index_fill_(
            0, ids, False
        )
        self.translation_entry_linear_speed_valid.index_fill_(
            0, ids, False
        )
        self.translation_entry_angular_speed_valid.index_fill_(
            0, ids, False
        )
        self.translation_entry_predicate_current.index_fill_(0, ids, False)
        self.translation_entry_failure_count_by_predicate.index_fill_(
            0, ids, 0
        )
        self.translation_entry_first_failure_control_step.index_fill_(
            0, ids, -1
        )
        self.translation_entry_last_failure_control_step.index_fill_(
            0, ids, -1
        )
        self.translation_entry_tip_height_fraction.index_fill_(0, ids, 0.0)
        self.translation_entry_tip_object_pair.index_fill_(0, ids, False)
        self.translation_entry_tip_object_raw_pair.index_fill_(
            0, ids, False
        )
        self.translation_entry_tip_object_active_pair.index_fill_(
            0, ids, False
        )
        self.translation_entry_tip_contact_qualified.index_fill_(
            0, ids, False
        )
        self.translation_entry_tip_contact_force_n.index_fill_(0, ids, 0.0)
        self.translation_entry_tip_force_valid.index_fill_(0, ids, False)
        self.translation_entry_tip_height_failure_count.index_fill_(
            0, ids, 0
        )
        self.translation_entry_tip_pair_missing_count.index_fill_(0, ids, 0)
        self.translation_entry_tip_raw_pair_missing_count.index_fill_(
            0, ids, 0
        )
        self.translation_entry_tip_efc_inactive_count.index_fill_(
            0, ids, 0
        )
        self.translation_entry_tip_force_failure_count.index_fill_(
            0, ids, 0
        )
        self.translation_entry_position_error_xyz_abs_m.index_fill_(
            0, ids, 0.0
        )
        self.translation_entry_euler_error_xyz_abs_deg.index_fill_(
            0, ids, 0.0
        )
        self.translation_entry_estimated_bottom_z_m.index_fill_(
            0, ids, 0.0
        )
        self.translation_entry_object_linear_speed_m_s.index_fill_(
            0, ids, 0.0
        )
        self.translation_entry_object_angular_speed_rad_s.index_fill_(
            0, ids, 0.0
        )
        self._last_tip_object_contact_raw_pair.index_fill_(0, ids, False)
        self._last_tip_object_contact_pair.index_fill_(0, ids, False)
        self._last_tip_object_contact_force_qualified_pair.index_fill_(
            0, ids, False
        )
        self.prelift_contact_stable_count.index_fill_(0, ids, 0)
        self.prelift_contact_max_stable_count.index_fill_(0, ids, 0)
        self.prelift_contact_gate_latched.index_fill_(0, ids, False)
        self.prelift_contact_gate_first_control_step.index_fill_(0, ids, -1)
        self.prelift_lift_boundary_contact_valid.index_fill_(0, ids, False)
        self.prelift_lift_permitted.index_fill_(0, ids, False)
        self.prelift_lift_reference_valid.index_fill_(0, ids, False)
        put(self.prelift_lift_start_body_pos, initial_body)
        put(self.prelift_lift_start_quat, initial_quat)
        self.prelift_contact_count_current.index_fill_(0, ids, 0)
        self.prelift_tip_raw_pair_current.index_fill_(0, ids, False)
        self.prelift_tip_active_pair_current.index_fill_(0, ids, False)
        self.prelift_tip_contact_qualified_current.index_fill_(
            0, ids, False
        )
        self.prelift_contact_stage_index.index_fill_(0, ids, -1)
        self.prelift_stage_observation_count.index_fill_(0, ids, 0)
        self.prelift_stage_qualified_count.index_fill_(0, ids, 0)
        self.prelift_stage_failure_count.index_fill_(0, ids, 0)
        self.prelift_stage_max_contact_count.index_fill_(0, ids, 0)
        self.prelift_stage_first_failure_control_step.index_fill_(0, ids, -1)
        self.prelift_stage_last_failure_control_step.index_fill_(0, ids, -1)
        self.prelift_tip_pair_missing_count_by_stage.index_fill_(0, ids, 0)
        self.prelift_tip_raw_pair_missing_count_by_stage.index_fill_(
            0, ids, 0
        )
        self.prelift_tip_efc_inactive_count_by_stage.index_fill_(0, ids, 0)
        self.prelift_tip_force_failure_count_by_stage.index_fill_(0, ids, 0)
        self.prelift_lift_boundary_tip_raw_pair.index_fill_(
            0, ids, False
        )
        self.prelift_lift_boundary_tip_active_pair.index_fill_(
            0, ids, False
        )
        self.prelift_lift_boundary_tip_contact_qualified.index_fill_(
            0, ids, False
        )
        self.prelift_lift_boundary_contact_force_n.index_fill_(0, ids, 0.0)
        self.expert_tip_object_contact.index_fill_(0, ids, False)
        self.expert_contact_count.index_fill_(0, ids, 0)
        self.expert_four_tip_contact.index_fill_(0, ids, False)
        self.expert_object_linear_speed_m_s.index_fill_(0, ids, 0.0)
        self.expert_object_angular_speed_rad_s.index_fill_(0, ids, 0.0)
        self.terminal_stable_count.index_fill_(0, ids, 0)
        self.terminal_stable_certified.index_fill_(0, ids, False)
        self.terminal_stability_gate_current.index_fill_(0, ids, False)
        self.cage_orientation_error_rad.index_fill_(0, ids, 0.0)
        self._last_ik_solve_feasible.index_fill_(0, ids, True)
        targets = self._tip_targets(self.previous_action)
        self.last_certified_tip_targets.index_copy_(
            0, ids, targets.index_select(0, ids)
        )
        for value in (
            self.tip_target_velocity_world_m_s,
            self.tip_target_acceleration_world_m_s2,
            self.tip_target_jerk_world_m_s3,
        ):
            value.index_fill_(0, ids, 0.0)
        self._last_targets.index_copy_(
            0, ids, targets.index_select(0, ids)
        )
        self._last_contact_force.index_fill_(0, ids, 0.0)
        self._last_force_decode_finite.index_fill_(0, ids, True)
        self._last_raw_observation_finite.index_fill_(0, ids, True)
        reset_observation = self._observation(
            targets, self.previous_action, self._last_contact_force
        )
        self._observation_tensor = self.torch.where(
            selected[:, None],
            reset_observation,
            self._observation_tensor,
        )
        self._object_drop_terminal_observation = self.torch.where(
            selected[:, None],
            reset_observation,
            self._object_drop_terminal_observation,
        )
        return self._observation_tensor

    def reset_with_goals(
        self,
        object_index: Any,
        template_index: Any,
        task_index: Any,
        path_seed: Any,
        path_shape_index: Any,
        rotation_angle_deg: Any,
        source_region: Any = 0,
    ) -> Any:
        return self.reset(
            goals={
                "object_index": object_index,
                "template_index": template_index,
                "task_index": task_index,
                "path_seed": path_seed,
                "path_shape_index": path_shape_index,
                "rotation_angle_deg": rotation_angle_deg,
                "source_region": source_region,
            }
        )

    def _phase_index(self) -> Any:
        starts = torch.tensor(
            (
                self.config.plan_ticks,
                self.config.plan_ticks + self.config.close_ticks,
                self.config.plan_ticks
                + self.config.close_ticks
                + self.config.settle_ticks,
                self.config.plan_ticks
                + self.config.close_ticks
                + self.config.settle_ticks
                + self.config.lift_ticks,
                self.config.plan_ticks
                + self.config.close_ticks
                + self.config.settle_ticks
                + self.config.lift_ticks
                + self.config.hold_ticks,
            ),
            device=self.device,
            dtype=self.torch.long,
        )
        return (self.control_step[:, None] >= starts[None, :]).sum(dim=1)

    def _filter_action(self, action: Any) -> tuple[Any, Any, Any]:
        requested = action.clamp(-1.0, 1.0)
        learned_plane = self._grasp_plane_learned_by_object.index_select(
            0, self.object_index
        )
        object_mask = torch.ones_like(requested)
        object_mask[:, 0:5] = learned_plane[:, None].to(self.dtype)
        requested = requested * object_mask
        if not self.residual_enabled:
            requested = torch.zeros_like(requested)
        previous = self.filtered_action * object_mask
        filtered = previous.clone()
        filtered[:, :9] = previous[:, :9] + EPISODE_ACTION_ALPHA * (
            requested[:, :9] - previous[:, :9]
        )
        force_slice = contract.FORCE_SCALE_SLICE
        filtered[:, force_slice] = previous[
            :, force_slice
        ] + EPISODE_ACTION_ALPHA * (
            requested[:, force_slice] - previous[:, force_slice]
        )
        filtered[:, 9:21] = previous[:, 9:21] + DYNAMIC_ACTION_ALPHA * (
            requested[:, 9:21] - previous[:, 9:21]
        )
        freeze_now = self.control_step == self.config.plan_ticks - 1
        frozen = torch.where(
            freeze_now[:, None], filtered, self.frozen_action
        ) * object_mask
        planning = self.control_step < self.config.plan_ticks
        used = filtered.clone()
        used[:, :9] = torch.where(
            planning[:, None], filtered[:, :9], frozen[:, :9]
        )
        used[:, force_slice] = torch.where(
            planning[:, None],
            filtered[:, force_slice],
            frozen[:, force_slice],
        )
        return filtered * object_mask, frozen, used * object_mask

    def _translation_path_state(
        self,
        step: Any | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        """Return the bounded lead/settle/route target for every world."""

        control_step = (
            self.control_step.to(self.dtype)
            if step is None
            else step.to(device=self.device, dtype=self.dtype)
        )
        action_start = (
            self.config.plan_ticks
            + self.config.close_ticks
            + self.config.settle_ticks
            + self.config.lift_ticks
            + self.config.hold_ticks
        )
        elapsed = control_step - float(action_start)
        translating = self.task_index == contract.TASKS.index(
            contract.TASK_TRANSLATE
        )
        shapes = self.path_shape_index.clamp(
            0, len(contract.TRANSLATION_PATH_SHAPES) - 1
        )
        lead_index = (elapsed.to(torch.long) + 1).clamp(
            0, contract.TRANSLATION_PATH_LEAD_IN_TICKS
        )
        lead_progress = self._translation_lead_progress_table[
            shapes, lead_index
        ]
        route_start_elapsed = (
            contract.TRANSLATION_PATH_LEAD_IN_TICKS
            + contract.TRANSLATION_PATH_START_SETTLE_TICKS
        )
        path_elapsed = elapsed - float(route_start_elapsed)
        route_index = (path_elapsed.to(torch.long) + 1).clamp(
            0, contract.TRANSLATION_PATH_ROUTE_SAMPLES
        )
        route_progress = self._translation_route_progress_table[
            shapes, route_index
        ]
        route_admitted = getattr(
            self, "translation_route_admitted", self.action_reference_valid
        )
        following = (
            (path_elapsed >= 0.0)
            & (path_elapsed < float(contract.TRANSLATION_PATH_ROUTE_SAMPLES))
            & translating
            & self.action_reference_valid
            & route_admitted
        )
        path_xy = self._translation_route_targets_xy_table[
            shapes, route_index
        ]
        lead_xy = self._translation_lead_offsets_xy_table[
            shapes, lead_index
        ]
        start_xy = self.path_waypoints_xy_m[:, 0]
        endpoint_xy = self.path_waypoints_xy_m[:, -1]
        offset_xy = torch.where(
            (elapsed < float(contract.TRANSLATION_PATH_LEAD_IN_TICKS))[:, None],
            lead_xy,
            torch.where(
                (path_elapsed < 0.0)[:, None],
                start_xy,
                torch.where(
                    following[:, None],
                    path_xy,
                    torch.where(
                        route_admitted[:, None], endpoint_xy, start_xy
                    ),
                ),
            ),
        )
        translation_scalar = translating.to(self.dtype)
        offset_xy = torch.where(
            translating[:, None], offset_xy, torch.zeros_like(offset_xy)
        )
        offset = torch.cat(
            (offset_xy, torch.zeros_like(offset_xy[:, :1])), dim=-1
        )
        anchor_progress = (
            route_progress * float(contract.TRANSLATION_PATH_SEGMENTS)
        )
        selected_segment = torch.floor(anchor_progress).to(torch.long).clamp(
            0, contract.TRANSLATION_PATH_SEGMENTS - 1
        )
        segment_linear = (
            anchor_progress - selected_segment.to(self.dtype)
        ).clamp(0.0, 1.0)
        segment_linear = torch.where(
            route_progress >= 1.0,
            torch.ones_like(segment_linear),
            segment_linear,
        )
        # Observation anchors describe geometry only.  They do not restart a
        # minimum-jerk clock; local progress is the actual C1 route progress.
        segment_smooth = segment_linear
        path_segment_progress = route_progress * translation_scalar
        target = torch.where(
            translating[:, None],
            self.action_start_com + offset,
            torch.zeros_like(offset),
        )
        return (
            offset,
            target,
            lead_progress * translation_scalar,
            (
                selected_segment.to(self.dtype)
                / float(contract.TRANSLATION_PATH_SEGMENTS)
            ) * translation_scalar,
            segment_linear * translation_scalar,
            segment_smooth * translation_scalar,
            path_segment_progress,
            following,
        )

    def _reset_path_tracking_first_violation(self, ids: Any) -> None:
        """Restore the explicit no-violation sentinels for selected worlds."""

        self.path_tracking_first_violation_sample_tick.index_fill_(0, ids, -1)
        self.path_tracking_first_violation_control_tick.index_fill_(0, ids, -1)
        self.path_tracking_first_violation_axis_index.index_fill_(0, ids, -1)
        self.path_tracking_first_violation_error_xyz_abs_m.index_fill_(
            0, ids, 0.0
        )

    def _update_path_schedule_certificate(
        self,
        tracking_sample: Any,
        route_progress: Any,
        route_offset_xy_m: Any,
    ) -> None:
        """Certify route order, completion, speed, and acceleration on-device."""

        sample = self.torch.as_tensor(
            tracking_sample, device=self.device, dtype=self.torch.bool
        )
        progress = self.torch.as_tensor(
            route_progress, device=self.device, dtype=self.dtype
        )
        offset = self.torch.as_tensor(
            route_offset_xy_m, device=self.device, dtype=self.dtype
        )
        if tuple(sample.shape) != (self.worlds,):
            raise ValueError("path schedule sample mask has the wrong shape")
        if tuple(progress.shape) != (self.worlds,):
            raise ValueError("path schedule progress has the wrong shape")
        if tuple(offset.shape) != (self.worlds, 2):
            raise ValueError("path schedule XY offset has the wrong shape")

        next_sample_count = self.path_tracking_sample_count + 1
        bounded_sample_index = next_sample_count.clamp(
            0, contract.TRANSLATION_PATH_ROUTE_SAMPLES
        )
        shapes = self.path_shape_index.clamp(
            0, len(contract.TRANSLATION_PATH_SHAPES) - 1
        )
        expected_progress = self._translation_route_progress_table[
            shapes, bounded_sample_index
        ]
        expected_offset = self._translation_route_targets_xy_table[
            shapes, bounded_sample_index
        ]
        dt = contract.TRANSLATION_CONTROL_DT_S
        velocity = (
            offset - self.path_schedule_previous_offset_xy_m
        ) / dt
        acceleration = (
            velocity - self.path_schedule_previous_velocity_xy_m_s
        ) / dt
        speed = velocity.norm(dim=-1)
        acceleration_norm = acceleration.norm(dim=-1)
        epsilon = 1.0e-6
        finite = (
            self.torch.isfinite(progress)
            & self.torch.isfinite(offset).all(dim=-1)
            & self.torch.isfinite(speed)
            & self.torch.isfinite(acceleration_norm)
        )
        exact_schedule = (
            (next_sample_count >= 1)
            & (next_sample_count <= contract.TRANSLATION_PATH_ROUTE_SAMPLES)
            & ((progress - expected_progress).abs() <= epsilon)
            & ((offset - expected_offset).abs() <= epsilon).all(dim=-1)
            & (progress + epsilon >= self.path_schedule_progress)
        )
        kinematic_limits = (
            speed <= contract.TRANSLATION_ROUTE_HARD_MAX_SPEED_M_S + epsilon
        ) & (
            acceleration_norm
            <= contract.TRANSLATION_ROUTE_HARD_MAX_ACCELERATION_M_S2 + epsilon
        )
        # Completion is an integer schedule property.  Never infer it from a
        # float32 progress value: minimum-jerk samples immediately before the
        # endpoint can legitimately round to 1.0.  The canonical table lookup
        # above still proves the exact progress and offset for this integer
        # sample, while the count below rejects overrun fail-closed.
        integer_sample_in_range = (
            next_sample_count <= contract.TRANSLATION_PATH_ROUTE_SAMPLES
        )
        step_valid = (
            finite
            & exact_schedule
            & kinematic_limits
            & integer_sample_in_range
        )
        valid_after = self.path_schedule_limit_valid & self.torch.where(
            sample, step_valid, self.torch.ones_like(step_valid)
        )
        terminal = (
            sample
            & (next_sample_count == contract.TRANSLATION_PATH_ROUTE_SAMPLES)
            & ((progress - 1.0).abs() <= epsilon)
            & valid_after
        )

        self.path_schedule_previous_offset_xy_m.copy_(
            self.torch.where(
                sample[:, None],
                offset,
                self.path_schedule_previous_offset_xy_m,
            )
        )
        self.path_schedule_previous_velocity_xy_m_s.copy_(
            self.torch.where(
                sample[:, None],
                velocity,
                self.path_schedule_previous_velocity_xy_m_s,
            )
        )
        self.path_schedule_progress.copy_(
            self.torch.where(sample, progress, self.path_schedule_progress)
        )
        self.path_schedule_max_speed_m_s.copy_(
            self.torch.where(
                sample,
                self.torch.maximum(self.path_schedule_max_speed_m_s, speed),
                self.path_schedule_max_speed_m_s,
            )
        )
        self.path_schedule_max_acceleration_m_s2.copy_(
            self.torch.where(
                sample,
                self.torch.maximum(
                    self.path_schedule_max_acceleration_m_s2,
                    acceleration_norm,
                ),
                self.path_schedule_max_acceleration_m_s2,
            )
        )
        self.path_schedule_limit_valid.copy_(valid_after)
        self.path_schedule_complete.copy_(
            self.path_schedule_complete | terminal
        )

    def _record_path_tracking_first_violation(
        self,
        tracking_sample: Any,
        error_xyz_abs_m: Any,
    ) -> None:
        """Latch the first strict per-axis route-tolerance violation.

        ``sample_tick`` is one-based within path-following samples and
        ``control_tick`` is the one-based episode transition endpoint.  The
        method is tensor-only so it remains safe inside the CUDA-graph-shaped
        vector environment step.
        """

        sample = self.torch.as_tensor(
            tracking_sample, device=self.device, dtype=self.torch.bool
        )
        error = self.torch.as_tensor(
            error_xyz_abs_m, device=self.device, dtype=self.dtype
        )
        if tuple(sample.shape) != (self.worlds,):
            raise ValueError("path tracking sample mask has the wrong shape")
        if tuple(error.shape) != (self.worlds, 3):
            raise ValueError("path tracking XYZ error has the wrong shape")
        unseen = self.path_tracking_first_violation_sample_tick < 0
        exceeds = (
            error
            > contract.TRANSLATION_PATH_TRACKING_AXIS_ABS_TOLERANCE_M
        ).any(dim=1)
        capture = sample & unseen & exceeds
        sample_tick = self.path_tracking_sample_count + 1
        control_tick = self.control_step + 1
        axis_index = error.argmax(dim=1)
        self.path_tracking_first_violation_sample_tick.copy_(
            self.torch.where(
                capture,
                sample_tick,
                self.path_tracking_first_violation_sample_tick,
            )
        )
        self.path_tracking_first_violation_control_tick.copy_(
            self.torch.where(
                capture,
                control_tick,
                self.path_tracking_first_violation_control_tick,
            )
        )
        self.path_tracking_first_violation_axis_index.copy_(
            self.torch.where(
                capture,
                axis_index,
                self.path_tracking_first_violation_axis_index,
            )
        )
        self.path_tracking_first_violation_error_xyz_abs_m.copy_(
            self.torch.where(
                capture[:, None],
                error,
                self.path_tracking_first_violation_error_xyz_abs_m,
            )
        )

    def _desired_object_pose(
        self,
    ) -> tuple[Any, Any, Any, Any, Any]:
        step = self.control_step.to(self.dtype)
        lift_start = (
            self.config.plan_ticks
            + self.config.close_ticks
            + self.config.settle_ticks
        )
        # Keep the action boundary identical to the HOLD/capture contract.
        # This value is used by both the rotation interpolation and the
        # action-active gate below.
        _, action_start = _hold_boundaries(self.config)
        nominal_lift_fraction = _minimum_jerk(
            (step - lift_start + 1.0) / self.config.lift_ticks
        )
        # Unit-level callers constructed with ``object.__new__`` predate the
        # gate; the runtime always owns these tensors.  The fallback keeps the
        # pure pose helper backwards-compatible while preserving fail-closed
        # behavior in every real environment.
        lift_permitted = getattr(
            self,
            "prelift_lift_permitted",
            torch.ones_like(self.control_step, dtype=torch.bool),
        )
        lift_reference_valid = getattr(
            self,
            "prelift_lift_reference_valid",
            torch.ones_like(self.control_step, dtype=torch.bool),
        )
        permitted = lift_permitted & lift_reference_valid
        lift_fraction = nominal_lift_fraction * permitted.to(self.dtype)
        linear_action_fraction = (
            (step - action_start + 1.0) / self.config.rotation_motion_ticks
        ).clamp(0.0, 1.0)
        action_fraction = _minimum_jerk(linear_action_fraction)
        lift_origin = getattr(
            self, "prelift_lift_start_body_pos", self.initial_body_pos
        )
        lift_quaternion = getattr(
            self, "prelift_lift_start_quat", self.initial_quat
        )
        lifted_body = lift_origin + torch.tensor(
            (0.0, 0.0, LIFT_HEIGHT_M),
            device=self.device,
            dtype=self.dtype,
        ) * lift_fraction[:, None]
        (
            action_translation,
            _,
            _,
            _,
            _,
            _,
            path_progress,
            _,
        ) = self._translation_path_state()
        action_center = self.action_start_com + action_translation
        cage_rotation_fraction = _cage_rotation_fraction(
            linear_action_fraction,
            action_fraction,
            self.object_index,
            self.task_index,
        )
        rotation_reported_fraction = _reported_action_fraction(
            linear_action_fraction,
            action_fraction,
            self.object_index,
            self.task_index,
        )
        reported_action_fraction = torch.where(
            self.task_index == contract.TASKS.index(contract.TASK_TRANSLATE),
            path_progress,
            rotation_reported_fraction,
        )
        rotated = _quat_slerp(
            self.action_start_quat,
            self.goal_quat,
            cage_rotation_fraction,
        )
        action_quat = torch.where(
            (self.task_index == 1)[:, None],
            rotated,
            self.action_start_quat,
        )
        action_active = (
            (self.control_step >= action_start)
            & self.action_reference_valid
            & permitted
        )
        desired_position = torch.where(
            action_active[:, None], action_center, lifted_body
        )
        desired_quaternion = torch.where(
            action_active[:, None], action_quat, lift_quaternion
        )
        return (
            desired_position,
            desired_quaternion,
            lift_fraction,
            reported_action_fraction,
            action_active,
        )

    def _effective_outward(self) -> Any:
        outward = self._template_rows(self._outward_local)
        planar = outward.clone()
        planar[:, :, 2] = 0.0
        planar = planar / planar.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        fruit = self._fruit_by_object.index_select(0, self.object_index)
        return torch.where(fruit[:, None, None], planar, outward)

    def _object_height_rows(self) -> tuple[Any, Any, Any]:
        bottom = self._object_local_bottom_z_m.index_select(
            0, self.object_index
        )
        top = self._object_local_top_z_m.index_select(0, self.object_index)
        return bottom, top, (top - bottom).clamp_min(1.0e-8)

    def _grasp_plane_contract_rows(
        self,
    ) -> tuple[Any, Any, Any, Any]:
        """Gather nominal fraction, limits, and learnability per world."""

        nominal = self._grasp_plane_nominal_fraction_by_object.index_select(
            0, self.object_index
        )
        limits = self._grasp_plane_fraction_limits_by_object.index_select(
            0, self.object_index
        )
        learned = self._grasp_plane_learned_by_object.index_select(
            0, self.object_index
        )
        return nominal, limits[:, 0], limits[:, 1], learned

    def _requested_tip_height_fractions(self, action: Any) -> Any:
        """Decode the learned grasp-plane height of every training fruit."""

        nominal, _, _, learned = self._grasp_plane_contract_rows()
        normalized = action[:, contract.TIP_HEIGHT_SLICE]
        learned_fraction = contract.GRASP_PLANE_NOMINAL_FRACTION + torch.where(
            normalized <= 0.0,
            normalized
            * (
                contract.GRASP_PLANE_NOMINAL_FRACTION
                - contract.GRASP_PLANE_MIN_FRACTION
            ),
            normalized
            * (
                contract.GRASP_PLANE_MAX_FRACTION
                - contract.GRASP_PLANE_NOMINAL_FRACTION
            ),
        )
        return torch.where(
            learned[:, None],
            learned_fraction,
            nominal[:, None].expand(-1, len(contract.TIP_ORDER)),
        )

    def _targets_in_actual_object_frame(self, targets: Any) -> Any:
        """Express world targets in the measured, not desired, object frame."""

        body_ids = self._object_body_ids.index_select(0, self.object_index)
        body_position = self.backend.tensor("xpos")[self._rows, body_ids]
        body_quaternion = _quat_normalize(
            self.backend.tensor("xquat")[self._rows, body_ids]
        )
        return _quat_rotate(
            _quat_conjugate(body_quaternion),
            targets - body_position[:, None, :],
        )

    def _project_grasp_plane(self, targets: Any) -> Any:
        """Hard-project every commanded hard-tip height into its band.

        Projection is per tip, not merely on the plane mean.  Independent
        residuals remain executable inside the 40--60% fruit band, while no
        combination of residuals can hide an out-of-band tip behind the other
        three tips' average.
        """

        body_ids = self._object_body_ids.index_select(0, self.object_index)
        body_position = self.backend.tensor("xpos")[self._rows, body_ids]
        body_quaternion = _quat_normalize(
            self.backend.tensor("xquat")[self._rows, body_ids]
        )
        local = _quat_rotate(
            _quat_conjugate(body_quaternion),
            targets - body_position[:, None, :],
        )
        bottom, _, height = self._object_height_rows()
        nominal, lower, upper, learned = self._grasp_plane_contract_rows()
        minimum_z = bottom[:, None] + lower[:, None] * height[:, None]
        maximum_z = bottom[:, None] + upper[:, None] * height[:, None]
        learned_z = torch.minimum(
            torch.maximum(local[:, :, 2], minimum_z), maximum_z
        )
        fixed_z = (
            bottom[:, None] + nominal[:, None] * height[:, None]
        ).expand_as(learned_z)
        projected_z = torch.where(learned[:, None], learned_z, fixed_z)
        projected_local = local.clone()
        projected_local[:, :, 2] = projected_z
        return body_position[:, None, :] + _quat_rotate(
            body_quaternion, projected_local
        )

    def _grasp_plane_metrics(
        self, targets: Any, *, physical: bool = False
    ) -> tuple[Any, Any, Any]:
        local = self._targets_in_actual_object_frame(targets)
        bottom, _, height = self._object_height_rows()
        tip_fraction = (
            local[:, :, 2] - bottom[:, None]
        ) / height[:, None]
        plane_fraction = tip_fraction.mean(dim=1)
        finite = torch.isfinite(tip_fraction).all(dim=1) & torch.isfinite(
            plane_fraction
        )
        _, lower, upper, learned = self._grasp_plane_contract_rows()
        numerical_tolerance = torch.full_like(plane_fraction, 2.0e-6)
        physical_tolerance = torch.where(
            learned,
            numerical_tolerance,
            torch.full_like(
                plane_fraction,
                contract.FIXED_GRASP_PLANE_PHYSICAL_TOLERANCE_M,
            )
            / height,
        )
        tolerance = physical_tolerance if physical else numerical_tolerance
        compliant = (
            finite
            & (tip_fraction >= lower[:, None] - tolerance[:, None]).all(
                dim=1
            )
            & (tip_fraction <= upper[:, None] + tolerance[:, None]).all(
                dim=1
            )
        )
        return tip_fraction, plane_fraction, compliant

    def _actual_grasp_plane_metrics(
        self, finger_force: Any
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Measure the physical hard-tip plane with endpoint-aware gating.

        During plan/approach the hard tips intentionally occupy their pregrasp
        heights, so an out-of-band measurement is not yet a constraint
        failure.  The physical plane becomes relevant once all four hard tips
        report contact or once closing has completed.  Compliance is
        recomputed from the current frame (never latched), while a non-finite
        physical measurement always fails closed, even before relevance.
        """

        tip_position = self.backend.tensor("xpos").index_select(
            1, self._tip_body_ids
        )
        tip_fraction, plane_fraction, in_band = (
            self._grasp_plane_metrics(tip_position, physical=True)
        )
        finite = torch.isfinite(tip_fraction).all(dim=1) & torch.isfinite(
            plane_fraction
        )
        finite_force = torch.isfinite(finger_force).all(dim=1)
        all_contact = finite_force & (
            finger_force >= CONTACT_FORCE_THRESHOLD_N
        ).all(dim=1)
        close_complete = self.control_step >= (
            self.config.plan_ticks + self.config.close_ticks
        )
        # A dropped world is already a sticky failed terminal.  Its active
        # object is quarantined away from the hard tips, so the resulting
        # relative height is deliberately excluded from plane statistics.
        relevant = (close_complete | all_contact) & (~self.object_dropped)
        compliant = finite & ((~relevant) | in_band)
        return (
            tip_fraction,
            plane_fraction,
            compliant,
            relevant,
            in_band,
            finite,
        )

    def _expert_physical_metrics(
        self,
        finger_force: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        """Return fail-closed measurements used by expert certification.

        A qualified contact is an exact native MuJoCo-Warp geom pair between
        one hard tip and the currently active fruit, with finite decoded
        normal force at or above the shared threshold.  Contact direction is
        deliberately not part of the certificate.
        """

        tip_position = self.backend.tensor("xpos").index_select(
            1, self._tip_body_ids
        )
        tip_fraction, _, _ = self._grasp_plane_metrics(
            tip_position, physical=True
        )
        _, lower, upper, _ = self._grasp_plane_contract_rows()
        tip_height_valid = (
            torch.isfinite(tip_fraction).all(dim=1)
            & (tip_fraction >= lower[:, None]).all(dim=1)
            & (tip_fraction <= upper[:, None]).all(dim=1)
        )

        _, _, _, estimated_bottom = self._active_object_drop_measurement()
        force_valid = (
            self._last_force_decode_finite[:, None]
            & torch.isfinite(finger_force)
        )
        tip_object_contact = (
            self._last_tip_object_contact_pair
            & force_valid
            & (finger_force >= CONTACT_FORCE_THRESHOLD_N)
        )
        contact_count = tip_object_contact.to(torch.long).sum(dim=1)

        velocity = self._object_velocity()
        linear_speed = velocity[:, :3].norm(dim=1)
        angular_speed = velocity[:, 3:].norm(dim=1)
        velocity_finite = torch.isfinite(velocity).all(dim=1)
        linear_speed_valid = (
            velocity_finite
            & torch.isfinite(linear_speed)
            & (linear_speed <= EXPERT_OBJECT_LINEAR_SPEED_MAX_M_S)
        )
        angular_speed_valid = (
            velocity_finite
            & torch.isfinite(angular_speed)
            & (angular_speed <= EXPERT_OBJECT_ANGULAR_SPEED_MAX_RAD_S)
        )
        lifted = torch.isfinite(estimated_bottom) & (
            estimated_bottom
            >= contract.TABLE_TOP_Z_M + EXPERT_OBJECT_TABLE_CLEARANCE_M
        )
        return (
            tip_height_valid,
            tip_object_contact,
            contact_count,
            contact_count == len(contract.TIP_ORDER),
            linear_speed_valid,
            angular_speed_valid,
            lifted,
            linear_speed,
            angular_speed,
            tip_fraction,
            estimated_bottom,
        )

    def _update_prelift_contact_gate(
        self,
        finger_force: Any,
    ) -> None:
        """Certify the grasp before the first lift command is executed.

        The decoder has already proved exact hard-tip/current-fruit geom
        identity.  Here a contact is qualified only when that pair is active,
        the decoded force is finite, and it reaches the common force
        threshold.  At least three qualified tips must persist for five
        consecutive CLOSE/SETTLE control ticks *through the lift boundary*.

        Worlds that miss the boundary certificate keep a frozen, measured
        body pose and never capture a HOLD/action reference.  This prevents a
        time-only phase transition from blindly lifting a fruit that was
        never grasped.
        """

        close_start = self.config.plan_ticks
        close_end = close_start + self.config.close_ticks
        lift_start = close_end + self.config.settle_ticks
        step = self.control_step
        in_close = (step > close_start) & (step <= close_end)
        in_settle = (step > close_end) & (step <= lift_start)
        in_window = in_close | in_settle

        force_valid = (
            self._last_force_decode_finite[:, None]
            & torch.isfinite(finger_force)
            & (finger_force >= CONTACT_FORCE_THRESHOLD_N)
        )
        active_pair = self._last_tip_object_contact_pair
        raw_pair = getattr(
            self, "_last_tip_object_contact_raw_pair", active_pair
        )
        tip_qualified = getattr(
            self,
            "_last_tip_object_contact_force_qualified_pair",
            active_pair & force_valid,
        )
        contact_count = tip_qualified.to(torch.long).sum(dim=1)
        _, _, _, estimated_bottom = self._active_object_drop_measurement()
        object_live = (
            (~self.object_dropped)
            & torch.isfinite(estimated_bottom)
            & (estimated_bottom >= contract.OBJECT_DROP_THRESHOLD_Z_M)
        )
        contact_valid = object_live & (contact_count >= EXPERT_MIN_CONTACTS)

        next_count = torch.where(
            in_window,
            torch.where(
                contact_valid,
                (self.prelift_contact_stable_count + 1).clamp_max(
                    PRELIFT_CONTACT_STABLE_TICKS
                ),
                torch.zeros_like(self.prelift_contact_stable_count),
            ),
            self.prelift_contact_stable_count,
        )
        self.prelift_contact_stable_count.copy_(next_count)
        self.prelift_contact_max_stable_count.copy_(
            torch.maximum(
                self.prelift_contact_max_stable_count,
                torch.where(
                    in_window,
                    next_count,
                    self.prelift_contact_max_stable_count,
                ),
            )
        )
        reached = in_window & (
            next_count >= PRELIFT_CONTACT_STABLE_TICKS
        )
        latched_before = self.prelift_contact_gate_latched.clone()
        newly_latched = reached & (~latched_before)
        self.prelift_contact_gate_first_control_step.copy_(
            torch.where(
                newly_latched,
                step,
                self.prelift_contact_gate_first_control_step,
            )
        )
        self.prelift_contact_gate_latched.logical_or_(reached)

        stage_mask = torch.stack((in_close, in_settle), dim=1)
        qualified_by_stage = stage_mask & contact_valid[:, None]
        failed_by_stage = stage_mask & (~contact_valid[:, None])
        self.prelift_stage_observation_count.add_(stage_mask.to(torch.long))
        self.prelift_stage_qualified_count.add_(
            qualified_by_stage.to(torch.long)
        )
        self.prelift_stage_failure_count.add_(failed_by_stage.to(torch.long))
        stage_contacts = contact_count[:, None].expand_as(
            self.prelift_stage_max_contact_count
        )
        self.prelift_stage_max_contact_count.copy_(
            torch.where(
                stage_mask,
                torch.maximum(
                    self.prelift_stage_max_contact_count,
                    stage_contacts,
                ),
                self.prelift_stage_max_contact_count,
            )
        )
        step_matrix = step[:, None].expand_as(
            self.prelift_stage_first_failure_control_step
        )
        first_failure = failed_by_stage & (
            self.prelift_stage_first_failure_control_step < 0
        )
        self.prelift_stage_first_failure_control_step.copy_(
            torch.where(
                first_failure,
                step_matrix,
                self.prelift_stage_first_failure_control_step,
            )
        )
        self.prelift_stage_last_failure_control_step.copy_(
            torch.where(
                failed_by_stage,
                step_matrix,
                self.prelift_stage_last_failure_control_step,
            )
        )
        diagnostic_tips = stage_mask[:, :, None]
        raw = raw_pair[:, None, :]
        pair = active_pair[:, None, :]
        qualified = tip_qualified[:, None, :]
        self.prelift_tip_pair_missing_count_by_stage.add_(
            (diagnostic_tips & (~pair)).to(torch.long)
        )
        self.prelift_tip_raw_pair_missing_count_by_stage.add_(
            (diagnostic_tips & (~raw)).to(torch.long)
        )
        self.prelift_tip_efc_inactive_count_by_stage.add_(
            (diagnostic_tips & raw & (~pair)).to(torch.long)
        )
        self.prelift_tip_force_failure_count_by_stage.add_(
            (diagnostic_tips & pair & (~qualified)).to(torch.long)
        )
        stage_index = torch.where(
            in_close,
            torch.zeros_like(step),
            torch.ones_like(step),
        )
        self.prelift_contact_stage_index.copy_(
            torch.where(
                in_window,
                stage_index,
                self.prelift_contact_stage_index,
            )
        )
        self.prelift_contact_count_current.copy_(
            torch.where(
                in_window,
                contact_count,
                self.prelift_contact_count_current,
            )
        )
        self.prelift_tip_raw_pair_current.copy_(
            torch.where(
                in_window[:, None],
                raw_pair,
                self.prelift_tip_raw_pair_current,
            )
        )
        self.prelift_tip_active_pair_current.copy_(
            torch.where(
                in_window[:, None],
                active_pair,
                self.prelift_tip_active_pair_current,
            )
        )
        self.prelift_tip_contact_qualified_current.copy_(
            torch.where(
                in_window[:, None],
                tip_qualified,
                self.prelift_tip_contact_qualified_current,
            )
        )

        # The post-physics state at ``step == lift_start`` is the measured
        # state after the last SETTLE command and before any lift command.
        # Freeze it as the only permissible lift origin.
        at_lift_boundary = step == lift_start
        body_ids = self._object_body_ids.index_select(0, self.object_index)
        body_position = self.backend.tensor("xpos")[self._rows, body_ids]
        raw_quaternion = self.backend.tensor("xquat")[self._rows, body_ids]
        quaternion_norm = raw_quaternion.norm(dim=1)
        pose_valid = (
            torch.isfinite(body_position).all(dim=1)
            & torch.isfinite(raw_quaternion).all(dim=1)
            & torch.isfinite(quaternion_norm)
            & (quaternion_norm > 1.0e-8)
        )
        quaternion = _quat_normalize(raw_quaternion)
        capture_pose = at_lift_boundary & object_live & pose_valid
        self.prelift_lift_start_body_pos.copy_(
            torch.where(
                capture_pose[:, None],
                body_position,
                self.prelift_lift_start_body_pos,
            )
        )
        self.prelift_lift_start_quat.copy_(
            torch.where(
                capture_pose[:, None],
                quaternion,
                self.prelift_lift_start_quat,
            )
        )
        self.prelift_lift_reference_valid.copy_(
            torch.where(
                at_lift_boundary,
                capture_pose,
                self.prelift_lift_reference_valid,
            )
        )
        boundary_contact_valid = (
            contact_valid
            & (next_count >= PRELIFT_CONTACT_STABLE_TICKS)
        )
        self.prelift_lift_boundary_contact_valid.copy_(
            torch.where(
                at_lift_boundary,
                boundary_contact_valid,
                self.prelift_lift_boundary_contact_valid,
            )
        )
        self.prelift_lift_permitted.copy_(
            torch.where(
                at_lift_boundary,
                boundary_contact_valid & capture_pose,
                self.prelift_lift_permitted,
            )
        )
        self.prelift_lift_boundary_tip_raw_pair.copy_(
            torch.where(
                at_lift_boundary[:, None],
                raw_pair,
                self.prelift_lift_boundary_tip_raw_pair,
            )
        )
        self.prelift_lift_boundary_tip_active_pair.copy_(
            torch.where(
                at_lift_boundary[:, None],
                active_pair,
                self.prelift_lift_boundary_tip_active_pair,
            )
        )
        self.prelift_lift_boundary_tip_contact_qualified.copy_(
            torch.where(
                at_lift_boundary[:, None],
                tip_qualified,
                self.prelift_lift_boundary_tip_contact_qualified,
            )
        )
        self.prelift_lift_boundary_contact_force_n.copy_(
            torch.where(
                at_lift_boundary[:, None],
                finger_force,
                self.prelift_lift_boundary_contact_force_n,
            )
        )

    def _update_translation_entry_gate(
        self,
        finger_force: Any,
    ) -> None:
        """Admit the fixed route only after 25 consecutive certified ticks."""

        (
            tip_height_valid,
            tip_object_contact,
            contact_count,
            four_tip_contact,
            linear_speed_valid,
            angular_speed_valid,
            lifted,
            linear_speed,
            angular_speed,
            tip_fraction,
            estimated_bottom,
        ) = self._expert_physical_metrics(finger_force)
        body_ids = self._object_body_ids.index_select(0, self.object_index)
        com = self.backend.tensor("xipos")[self._rows, body_ids]
        quaternion = _quat_normalize(
            self.backend.tensor("xquat")[self._rows, body_ids]
        )
        start_xy = self.path_waypoints_xy_m[:, 0]
        start_position = self.action_start_com + torch.cat(
            (start_xy, torch.zeros_like(start_xy[:, :1])), dim=1
        )
        start_position_error = (start_position - com).abs()
        start_position_valid = (
            torch.isfinite(com).all(dim=1)
            & torch.isfinite(start_position_error).all(dim=1)
            & (
                start_position_error
                <= contract.TRANSLATION_POSITION_AXIS_ABS_TOLERANCE_M
            ).all(dim=1)
        )
        start_orientation_error = _relative_euler_xyz_deg(
            self.action_start_quat, quaternion
        ).abs()
        start_orientation_valid = (
            torch.isfinite(quaternion).all(dim=1)
            & torch.isfinite(start_orientation_error).all(dim=1)
            & (
                start_orientation_error
                <= contract.TRANSLATION_EULER_AXIS_ABS_TOLERANCE_DEG
            ).all(dim=1)
        )
        pose_valid = start_position_valid & start_orientation_valid
        enough_contacts = contact_count >= EXPERT_MIN_CONTACTS
        translating = self.task_index == contract.TASKS.index(
            contract.TASK_TRANSLATE
        )
        action_start = sum(
            (
                self.config.plan_ticks,
                self.config.close_ticks,
                self.config.settle_ticks,
                self.config.lift_ticks,
                self.config.hold_ticks,
            )
        )
        settle_start = action_start + contract.TRANSLATION_PATH_LEAD_IN_TICKS
        route_start = (
            settle_start + contract.TRANSLATION_PATH_START_SETTLE_TICKS
        )
        admission_window = (
            translating
            & (self.control_step >= settle_start)
            & (self.control_step <= route_start)
        )
        prelift_permitted = getattr(
            self,
            "prelift_lift_permitted",
            torch.ones_like(self.action_reference_valid),
        )
        prelift_reference_valid = getattr(
            self,
            "prelift_lift_reference_valid",
            torch.ones_like(self.action_reference_valid),
        )
        predicates = torch.stack(
            (
                (
                    self.action_reference_valid
                    & prelift_permitted
                    & prelift_reference_valid
                ),
                ~self.object_dropped,
                tip_height_valid,
                enough_contacts,
                linear_speed_valid,
                angular_speed_valid,
                lifted,
                start_position_valid,
                start_orientation_valid,
            ),
            dim=1,
        )
        safety_now = predicates.all(dim=1)
        current = admission_window & safety_now
        next_count = torch.where(
            current,
            (self.translation_entry_stable_count + 1).clamp_max(
                EXPERT_STABLE_TICKS
            ),
            torch.zeros_like(self.translation_entry_stable_count),
        )
        next_count = torch.where(
            admission_window,
            next_count,
            self.translation_entry_stable_count,
        )
        self.translation_entry_stable_count.copy_(
            next_count
        )
        self.translation_entry_max_stable_count.copy_(
            torch.maximum(
                self.translation_entry_max_stable_count, next_count
            )
        )
        reached = admission_window & (next_count >= EXPERT_STABLE_TICKS)
        latched_before = self.translation_entry_certificate_latched.clone()
        latched_after = latched_before | reached
        newly_latched = reached & (~latched_before)
        self.translation_entry_certificate_first_control_step.copy_(
            torch.where(
                newly_latched,
                self.control_step,
                self.translation_entry_certificate_first_control_step,
            )
        )
        self.translation_entry_certificate_latched.copy_(latched_after)
        at_route_start = translating & (self.control_step == route_start)
        admit_now = (
            at_route_start
            & (next_count >= EXPERT_STABLE_TICKS)
            & safety_now
            & prelift_permitted
        )
        self.translation_route_admitted.logical_or_(admit_now)
        self.translation_entry_route_start_safety_valid.copy_(
            torch.where(
                at_route_start,
                safety_now,
                self.translation_entry_route_start_safety_valid,
            )
        )

        failure = admission_window[:, None] & (~predicates)
        self.translation_entry_failure_count_by_predicate.add_(
            failure.to(torch.long)
        )
        first_failure = failure & (
            self.translation_entry_first_failure_control_step < 0
        )
        step_matrix = self.control_step[:, None].expand_as(
            self.translation_entry_first_failure_control_step
        )
        self.translation_entry_first_failure_control_step.copy_(
            torch.where(
                first_failure,
                step_matrix,
                self.translation_entry_first_failure_control_step,
            )
        )
        self.translation_entry_last_failure_control_step.copy_(
            torch.where(
                failure,
                step_matrix,
                self.translation_entry_last_failure_control_step,
            )
        )
        _, lower, upper, _ = self._grasp_plane_contract_rows()
        tip_height_ok = (
            torch.isfinite(tip_fraction)
            & (tip_fraction >= lower[:, None])
            & (tip_fraction <= upper[:, None])
        )
        force_ok = (
            self._last_force_decode_finite[:, None]
            & torch.isfinite(finger_force)
            & (finger_force >= CONTACT_FORCE_THRESHOLD_N)
        )
        active_pair = self._last_tip_object_contact_pair
        raw_pair = getattr(
            self, "_last_tip_object_contact_raw_pair", active_pair
        )
        qualified_pair = getattr(
            self,
            "_last_tip_object_contact_force_qualified_pair",
            active_pair & force_ok,
        )
        self.translation_entry_tip_height_failure_count.add_(
            (admission_window[:, None] & (~tip_height_ok)).to(torch.long)
        )
        self.translation_entry_tip_pair_missing_count.add_(
            (
                admission_window[:, None]
                & (~active_pair)
            ).to(torch.long)
        )
        self.translation_entry_tip_raw_pair_missing_count.add_(
            (admission_window[:, None] & (~raw_pair)).to(torch.long)
        )
        self.translation_entry_tip_efc_inactive_count.add_(
            (
                admission_window[:, None]
                & raw_pair
                & (~active_pair)
            ).to(torch.long)
        )
        self.translation_entry_tip_force_failure_count.add_(
            (
                admission_window[:, None]
                & active_pair
                & (~qualified_pair)
            ).to(torch.long)
        )
        diagnostic_worlds = admission_window[:, None]
        self.translation_entry_predicate_current.copy_(
            torch.where(
                diagnostic_worlds,
                predicates,
                self.translation_entry_predicate_current,
            )
        )
        for destination, value in (
            (self.translation_entry_gate_current, current),
            (self.translation_entry_tip_height_valid, tip_height_valid),
            (self.translation_entry_lifted, lifted),
            (self.translation_entry_pose_valid, pose_valid),
            (
                self.translation_entry_start_position_valid,
                start_position_valid,
            ),
            (
                self.translation_entry_start_orientation_valid,
                start_orientation_valid,
            ),
            (
                self.translation_entry_linear_speed_valid,
                linear_speed_valid,
            ),
            (
                self.translation_entry_angular_speed_valid,
                angular_speed_valid,
            ),
        ):
            destination.copy_(
                torch.where(admission_window, value, destination)
            )
        for destination, value in (
            (self.translation_entry_tip_height_fraction, tip_fraction),
            (
                self.translation_entry_tip_object_pair,
                active_pair,
            ),
            (self.translation_entry_tip_object_raw_pair, raw_pair),
            (self.translation_entry_tip_object_active_pair, active_pair),
            (
                self.translation_entry_tip_contact_qualified,
                qualified_pair,
            ),
            (self.translation_entry_tip_contact_force_n, finger_force),
            (self.translation_entry_tip_force_valid, force_ok),
            (
                self.translation_entry_position_error_xyz_abs_m,
                start_position_error,
            ),
            (
                self.translation_entry_euler_error_xyz_abs_deg,
                start_orientation_error,
            ),
        ):
            destination.copy_(
                torch.where(diagnostic_worlds, value, destination)
            )
        self.translation_entry_estimated_bottom_z_m.copy_(
            torch.where(
                admission_window,
                estimated_bottom,
                self.translation_entry_estimated_bottom_z_m,
            )
        )
        self.translation_entry_object_linear_speed_m_s.copy_(
            torch.where(
                admission_window,
                linear_speed,
                self.translation_entry_object_linear_speed_m_s,
            )
        )
        self.translation_entry_object_angular_speed_rad_s.copy_(
            torch.where(
                admission_window,
                angular_speed,
                self.translation_entry_object_angular_speed_rad_s,
            )
        )
        self.expert_tip_object_contact.copy_(tip_object_contact)
        self.expert_contact_count.copy_(contact_count)
        self.expert_four_tip_contact.copy_(four_tip_contact)
        self.expert_object_linear_speed_m_s.copy_(linear_speed)
        self.expert_object_angular_speed_rad_s.copy_(angular_speed)

    def _update_terminal_stability_gate(
        self,
        *,
        pose_success: Any,
        actual_plane_compliant: Any,
        finger_force: Any,
    ) -> Any:
        """Require the complete terminal certificate for 25 live ticks."""

        (
            tip_height_valid,
            tip_object_contact,
            contact_count,
            four_tip_contact,
            linear_speed_valid,
            angular_speed_valid,
            lifted,
            linear_speed,
            angular_speed,
            _,
            _,
        ) = self._expert_physical_metrics(finger_force)
        route_certified = _path_tracking_certified(
            self.task_index,
            self.path_tracking_sample_count,
            self.path_tracking_axis_abs_max_m,
            self.path_schedule_complete,
            self.path_schedule_limit_valid,
        )
        current = (
            pose_success
            & route_certified
            & actual_plane_compliant
            & tip_height_valid
            & (contact_count >= EXPERT_MIN_CONTACTS)
            & linear_speed_valid
            & angular_speed_valid
            & lifted
            & (~self.object_dropped)
        )
        next_count = torch.where(
            current,
            (self.terminal_stable_count + 1).clamp_max(
                EXPERT_STABLE_TICKS
            ),
            torch.zeros_like(self.terminal_stable_count),
        )
        self.terminal_stable_count.copy_(next_count)
        self.terminal_stable_certified.copy_(
            next_count >= EXPERT_STABLE_TICKS
        )
        self.terminal_stability_gate_current.copy_(current)
        self.expert_tip_object_contact.copy_(tip_object_contact)
        self.expert_contact_count.copy_(contact_count)
        self.expert_four_tip_contact.copy_(four_tip_contact)
        self.expert_object_linear_speed_m_s.copy_(linear_speed)
        self.expert_object_angular_speed_rad_s.copy_(angular_speed)
        return self.terminal_stable_certified

    def _park_inactive_objects(self) -> None:
        """Restore every non-selected free body to its finite reset parking.

        MJWarp batches share immutable model collision parameters, so the
        parent model keeps all object bodies compiled and parks the inactive
        ones per world.  Those disconnected free bodies still see gravity.
        Without this per-control-tick projection they fall roughly 500 m
        during the 10.96 s episode and can poison an otherwise independent
        batched solve.
        The selected object's state is never touched.
        """

        qpos = self.backend.tensor("qpos")
        qvel = self.backend.tensor("qvel")
        warmstart = (
            self.backend.tensor("qacc_warmstart")
            if self.backend.has_field("qacc_warmstart")
            else None
        )
        active_qpos_address = self._object_qpos_adrs.index_select(
            0, self.object_index
        )
        for slot, (qpos_address, qvel_address) in enumerate(
            zip(
                self._free_qpos_adr_values,
                self._free_qvel_adr_values,
            )
        ):
            inactive = active_qpos_address != qpos_address
            qpos_slice = qpos[:, qpos_address : qpos_address + 7]
            qpos_slice.copy_(
                self.torch.where(
                    inactive[:, None],
                    self._parked_free_qpos[:, slot],
                    qpos_slice,
                )
            )
            qvel[:, qvel_address : qvel_address + 6].masked_fill_(
                inactive[:, None], 0.0
            )
            if warmstart is not None:
                warmstart[
                    :, qvel_address : qvel_address + 6
                ].masked_fill_(inactive[:, None], 0.0)

    def _quarantine_dropped_active_object(self, dropped: Any) -> None:
        """Project sticky finite drops to a remote collision-free pose.

        The projection applies only on the control tick after the terminal
        was observed (and every later tick).  It keeps the CUDA batch shape
        fixed while preventing a detached free body from continuing to fall.
        Robot commands are independently held/braked by :meth:`step`.
        """

        dropped = dropped.to(self.torch.bool)
        qpos = self.backend.tensor("qpos")
        qvel = self.backend.tensor("qvel")
        warmstart = (
            self.backend.tensor("qacc_warmstart")
            if self.backend.has_field("qacc_warmstart")
            else None
        )
        active_qpos_address = self._object_qpos_adrs.index_select(
            0, self.object_index
        )
        for slot, (qpos_address, qvel_address) in enumerate(
            zip(self._free_qpos_adr_values, self._free_qvel_adr_values)
        ):
            selected = dropped & (active_qpos_address == qpos_address)
            qpos_slice = qpos[:, qpos_address : qpos_address + 7]
            qpos_slice.copy_(
                self.torch.where(
                    selected[:, None],
                    self._object_drop_quarantine_qpos[:, slot],
                    qpos_slice,
                )
            )
            qvel[:, qvel_address : qvel_address + 6].masked_fill_(
                selected[:, None], 0.0
            )
            if warmstart is not None:
                warmstart[
                    :, qvel_address : qvel_address + 6
                ].masked_fill_(selected[:, None], 0.0)
        # No host-side ``any`` or extra forward launch is needed: the fixed
        # step_control graph consumes the projected qpos/qvel directly.

    def _active_object_drop_measurement(
        self,
    ) -> tuple[Any, Any, Any, Any]:
        """Return live body pose, COM pose and conservative bottom height."""

        body_ids = self._object_body_ids.index_select(0, self.object_index)
        body_position = self.backend.tensor("xpos")[self._rows, body_ids]
        com_position = self.backend.tensor("xipos")[self._rows, body_ids]
        quaternion = _quat_normalize(
            self.backend.tensor("xquat")[self._rows, body_ids]
        )
        local_bottom = self._object_local_bottom_z_m.index_select(
            0, self.object_index
        )
        estimated_bottom = body_position[:, 2] + local_bottom
        return body_position, com_position, quaternion, estimated_bottom

    def _reported_object_drop_pose(
        self,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        """Return cached terminal pose plus separate live quarantine pose."""

        (
            live_body_position,
            live_position,
            live_quaternion,
            live_estimated_bottom,
        ) = self._active_object_drop_measurement()
        dropped = self.object_dropped
        body_position = torch.where(
            dropped[:, None],
            self.object_drop_first_body_position,
            live_body_position,
        )
        position = torch.where(
            dropped[:, None], self.object_drop_first_position, live_position
        )
        quaternion = torch.where(
            dropped[:, None],
            self.object_drop_first_quaternion,
            live_quaternion,
        )
        estimated_bottom = torch.where(
            dropped,
            self.object_drop_first_bottom_z_m,
            live_estimated_bottom,
        )
        return (
            body_position,
            position,
            quaternion,
            estimated_bottom,
            live_body_position,
            live_position,
            live_quaternion,
            live_estimated_bottom,
        )

    def _update_position_feedback(self) -> None:
        """Slew measured object feedback once per control tick.

        The prior stateless path could move every hard-tip target by the full
        feedback range in one frame.  Keeping this state update outside
        ``_tip_targets`` preserves that method as a pure calculation even
        though it is called more than once per transition.
        """

        (
            desired_position,
            desired_quaternion,
            _,
            _,
            action_active,
        ) = self._desired_object_pose()
        body_ids = self._object_body_ids.index_select(0, self.object_index)
        body_position = self.backend.tensor("xpos")[self._rows, body_ids]
        com_position = self.backend.tensor("xipos")[self._rows, body_ids]
        body_quaternion = _quat_normalize(
            self.backend.tensor("xquat")[self._rows, body_ids]
        )
        tracked_position = torch.where(
            action_active[:, None], com_position, body_position
        )
        raw_error = desired_position - tracked_position
        finite = torch.isfinite(raw_error).all(dim=1)
        error = torch.nan_to_num(raw_error, nan=0.0, posinf=0.0, neginf=0.0)
        scaled_error = POSITION_FEEDBACK_GAIN * error
        xy_error = scaled_error[:, :2]
        scale = (
            POSITION_FEEDBACK_XY_LIMIT_M
            / xy_error.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        ).clamp_max(1.0)
        xy_feedback = xy_error * scale
        z_feedback = scaled_error[:, 2:3].clamp(
            -POSITION_FEEDBACK_Z_LIMIT_M,
            POSITION_FEEDBACK_Z_LIMIT_M,
        )
        target = torch.cat((xy_feedback, z_feedback), dim=-1)
        lift_start = (
            self.config.plan_ticks
            + self.config.close_ticks
            + self.config.settle_ticks
        )
        action_start = (
            lift_start + self.config.lift_ticks + self.config.hold_ticks
        )
        # Both tasks use measured-COM correction for the complete cage.  No
        # hidden feedback state may integrate while HOLD has frozen the
        # physical cage and actuator commands.
        enabled = self.control_step >= lift_start
        target = torch.where(
            finite[:, None], target, self.position_feedback_m
        )
        target = torch.where(
            enabled[:, None], target, torch.zeros_like(target)
        )
        feedback_frozen = _hold_freeze_window(self.control_step, self.config)
        target = torch.where(
            feedback_frozen[:, None],
            self.position_feedback_m,
            target,
        )
        raw_delta = target - self.position_feedback_m
        self.position_feedback_slew_clip_count.add_(
            (
                enabled
                & finite
                & (
                    raw_delta.abs().amax(dim=1)
                    > POSITION_FEEDBACK_SLEW_M_PER_TICK
                )
            ).to(torch.long)
        )
        self.position_feedback_m.add_(
            raw_delta.clamp(
                -POSITION_FEEDBACK_SLEW_M_PER_TICK,
                POSITION_FEEDBACK_SLEW_M_PER_TICK,
            )
        )
        orientation_error = _quat_mul(
            desired_quaternion, _quat_conjugate(body_quaternion)
        )
        orientation_angle = 2.0 * torch.acos(
            orientation_error[:, 0].abs().clamp(0.0, 1.0)
        )
        self.cage_orientation_error_rad.copy_(
            torch.where(
                torch.isfinite(orientation_angle),
                orientation_angle,
                self.cage_orientation_error_rad,
            )
        )

    def _apply_whole_cage_orientation_feedback(
        self,
        targets: Any,
        desired_position: Any,
        desired_quaternion: Any,
    ) -> Any:
        """Rotate the complete cage from measured object attitude error."""

        body_ids = self._object_body_ids.index_select(0, self.object_index)
        actual_quaternion = _quat_normalize(
            self.backend.tensor("xquat")[self._rows, body_ids]
        )
        error_quaternion = _quat_mul(
            desired_quaternion, _quat_conjugate(actual_quaternion)
        )
        error_angle = 2.0 * torch.acos(
            error_quaternion[:, 0].abs().clamp(0.0, 1.0)
        )
        capped_gain = (
            CAGE_ORIENTATION_FEEDBACK_MAX_RAD
            / error_angle.clamp_min(1.0e-8)
        ).clamp_max(CAGE_ORIENTATION_FEEDBACK_GAIN)
        identity = torch.zeros_like(error_quaternion)
        identity[:, 0] = 1.0
        correction = _quat_slerp(
            identity, error_quaternion, capped_gain
        )
        center = desired_position + self.position_feedback_m
        corrected = center[:, None, :] + _quat_rotate(
            correction, targets - center[:, None, :]
        )
        lift_start = (
            self.config.plan_ticks
            + self.config.close_ticks
            + self.config.settle_ticks
        )
        enabled = (
            (self.control_step >= lift_start)
            & (~_hold_freeze_window(self.control_step, self.config))
            & torch.isfinite(error_quaternion).all(dim=1)
            & torch.isfinite(corrected).all(dim=(1, 2))
        )
        return torch.where(enabled[:, None, None], corrected, targets)

    def _tip_targets(self, action: Any) -> Any:
        pregrasp = self._template_rows(self._pregrasp_local)
        contact = self._template_rows(self._contact_local)
        step = self.control_step.to(self.dtype)
        close_fraction = _minimum_jerk(
            (step - self.config.plan_ticks + 1.0)
            / self.config.close_ticks
        )
        local = pregrasp + close_fraction[:, None, None] * (
            contact - pregrasp
        )
        bottom, _, object_height = self._object_height_rows()
        height_fractions = self._requested_tip_height_fractions(action)
        contact_height = (
            bottom[:, None] + height_fractions * object_height[:, None]
        )
        approach_height_delta = pregrasp[:, :, 2] - contact[:, :, 2]
        local[:, :, 2] = contact_height + (
            1.0 - close_fraction[:, None]
        ) * approach_height_delta
        _, _, _, learned_plane = self._grasp_plane_contract_rows()
        yaw = torch.where(
            learned_plane,
            action[:, contract.YAW_INDEX] * contract.YAW_LIMIT_RAD,
            torch.full_like(
                action[:, contract.YAW_INDEX],
                contract.FIXED_GRASP_PLANE_YAW_RAD,
            ),
        )
        preload_residual = (
            action[:, contract.TIP_PRELOAD_SLICE]
            * contract.TIP_PRELOAD_LIMIT_M
        )
        force_scale = (
            contract.FORCE_SCALE_NOMINAL
            + 0.5 * action[:, contract.FORCE_SCALE_SLICE]
        )
        nominal = self._nominal_preload.index_select(
            0, self.object_index
        )[:, None]
        preload = (
            nominal * force_scale
            + preload_residual
            + self.preload_trim_m
        ).clamp(FORCE_FEEDBACK_MIN_M, FORCE_FEEDBACK_MAX_M)
        outward = self._effective_outward()
        yaw_rotation = _yaw_rotation(yaw)
        center = local.mean(dim=1, keepdim=True)
        local = center + torch.bmm(
            local - center, yaw_rotation.transpose(1, 2)
        )
        outward = torch.bmm(outward, yaw_rotation.transpose(1, 2))
        local = local - close_fraction[:, None, None] * (
            preload[:, :, None] * outward
        )
        dynamic = (
            action[:, contract.TIP_XYZ_RESIDUAL_SLICE].reshape(
                self.worlds, 4, 3
            )
            * contract.TIP_XYZ_RESIDUAL_LIMIT_M
        )
        local = local + dynamic

        (
            desired_position,
            desired_quaternion,
            _,
            _,
            action_active,
        ) = self._desired_object_pose()
        plan_targets = desired_position[:, None, :] + _quat_rotate(
            desired_quaternion, local
        )
        start_dynamic = (
            self.action_start_action[
                :, contract.TIP_XYZ_RESIDUAL_SLICE
            ].reshape(self.worlds, 4, 3)
            * contract.TIP_XYZ_RESIDUAL_LIMIT_M
        )
        trim_delta = (
            self.preload_trim_m - self.action_start_preload_trim_m
        )
        action_local = (
            self.action_start_cage_local
            + dynamic
            - start_dynamic
            - trim_delta[:, :, None] * outward
        )
        action_targets = desired_position[:, None, :] + _quat_rotate(
            desired_quaternion, action_local
        )
        plan_targets = plan_targets + self.position_feedback_m[:, None, :]
        action_targets = action_targets + (
            self.position_feedback_m
            - self.action_start_position_feedback_m
        )[:, None, :]
        targets = torch.where(
            action_active[:, None, None], action_targets, plan_targets
        )
        targets = self._apply_whole_cage_orientation_feedback(
            targets, desired_position, desired_quaternion
        )
        result = targets.clone()
        result[:, :, 2] = result[:, :, 2].clamp_min(MIN_TIP_CENTER_Z_M)
        return self._project_grasp_plane(result)

    def _rate_limit_tip_targets(
        self,
        desired: Any,
        active_worlds: Any,
    ) -> Any:
        """Track hard-tip positions without crossing the projected target.

        The previous implementation integrated a clipped acceleration even
        after the target had stopped.  That is a velocity limiter, not a
        position servo: it can carry a tip straight through its contact anchor.
        This controller uses a critically damped third-order position law,
        starts braking at the jerk-limited stopping envelope, and clamps the
        final step at the target plane.  The grasp-plane projection is applied
        *after* the servo and the certificate is computed from that value, so
        the recorded v/a/jerk state always describes the command sent to IK.
        """

        desired = torch.as_tensor(
            desired, device=self.device, dtype=self.dtype
        )
        active = torch.as_tensor(
            active_worlds, device=self.device, dtype=torch.bool
        )
        if tuple(desired.shape) != (self.worlds, len(contract.TIP_ORDER), 3):
            raise ValueError("tip targets must be [worlds,4,3]")
        if tuple(active.shape) != (self.worlds,):
            raise ValueError("active_worlds must contain one value per world")

        def clip_norm(value: Any, limit: float) -> Any:
            norm = value.norm(dim=-1, keepdim=True)
            scale = (limit / norm.clamp_min(1.0e-12)).clamp_max(1.0)
            return value * scale

        previous_target = self.last_certified_tip_targets
        previous_velocity = self.tip_target_velocity_world_m_s
        previous_acceleration = self.tip_target_acceleration_world_m_s2
        dt = float(self.backend.control_dt)
        desired_finite = torch.isfinite(desired).all(dim=(1, 2))
        safe_desired = torch.where(
            desired_finite[:, None, None], desired, previous_target
        )
        error = safe_desired - previous_target
        distance = error.norm(dim=-1, keepdim=True)
        direction = error / distance.clamp_min(1.0e-12)

        # Symmetric jerk-limited S-curve stopping envelope, as used by the
        # FLUID route scheduler.  Above the envelope no further positive jerk
        # is allowed in the target direction; the damped position law then
        # releases velocity and acceleration before the endpoint.
        acceleration_limit = float(TIP_TARGET_ACCELERATION_MAX_M_S2)
        jerk_limit = float(TIP_TARGET_JERK_MAX_M_S3)
        transition_distance = acceleration_limit**3 / jerk_limit**2
        triangular_speed = (
            distance * math.sqrt(jerk_limit)
        ).clamp_min(0.0).pow(2.0 / 3.0)
        linear_term = acceleration_limit**2 / jerk_limit
        trapezoidal_speed = 0.5 * (
            -linear_term
            + torch.sqrt(
                linear_term**2
                + 8.0 * acceleration_limit * distance
            )
        )
        stopping_speed = torch.where(
            distance <= transition_distance,
            triangular_speed,
            trapezoidal_speed,
        ).clamp(0.0, TIP_TARGET_SPEED_MAX_M_S)

        # Three radians/second keeps the unsaturated critically damped response
        # inside the public 40 mm/s, 150 mm/s^2 and 1 m/s^3 caps for the
        # planner's normal cage moves.  A faster pole merely slams into those
        # caps and can leave no jerk-feasible braking step at the endpoint.
        natural_frequency_rad_s = 3.0
        requested_jerk = (
            natural_frequency_rad_s**3 * error
            - 3.0
            * natural_frequency_rad_s**2
            * previous_velocity
            - 3.0
            * natural_frequency_rad_s
            * previous_acceleration
        )
        speed_toward_target = (
            previous_velocity * direction
        ).sum(dim=-1, keepdim=True)
        directional_jerk = (
            requested_jerk * direction
        ).sum(dim=-1, keepdim=True)
        stopping_limited = speed_toward_target > stopping_speed
        requested_jerk = requested_jerk - torch.where(
            stopping_limited,
            direction * directional_jerk.clamp_min(0.0),
            torch.zeros_like(requested_jerk),
        )
        realized_jerk = clip_norm(
            requested_jerk, TIP_TARGET_JERK_MAX_M_S3
        )
        realized_acceleration = clip_norm(
            previous_acceleration + realized_jerk * dt,
            TIP_TARGET_ACCELERATION_MAX_M_S2,
        )
        realized_velocity = clip_norm(
            previous_velocity + realized_acceleration * dt,
            TIP_TARGET_SPEED_MAX_M_S,
        )
        candidate = previous_target + realized_velocity * dt

        # A discrete step is never allowed to cross the requested target
        # plane.  This clamp normally only removes a sub-micrometre tail after
        # braking; its exact finite differences are certified below.
        error_squared = error.square().sum(dim=-1, keepdim=True)
        # Finish the sub-two-micrometre tail exactly.  Below this scale float32
        # finite differences can flip the sign of a nanometre-sized step and
        # create a false one-micrometre overshoot/return cycle.
        at_target = error_squared <= (2.0e-6) ** 2
        crossed_target = (
            (~at_target)
            & (
                ((candidate - safe_desired) * error).sum(
                    dim=-1, keepdim=True
                )
                >= 0.0
            )
        )
        candidate = torch.where(
            at_target | crossed_target, safe_desired, candidate
        )

        # The current measured object frame may have moved since the previous
        # tick.  Re-project after rate limiting, then perform the crossing clamp
        # once more because projection itself can move a point along ``error``.
        servo_candidate = candidate
        projected_candidate = self._project_grasp_plane(candidate)
        crossed_after_projection = (
            (~at_target)
            & (
                ((projected_candidate - safe_desired) * error).sum(
                    dim=-1, keepdim=True
                )
                >= 0.0
            )
        )
        projected_candidate = torch.where(
            at_target | crossed_after_projection,
            safe_desired,
            projected_candidate,
        )

        # The measured fruit frame can settle by several millimetres in one
        # control tick before contact.  An instantaneous projection following
        # that moving frame is not always compatible with the v/a/jerk
        # certificate.  Accept the post-filter projection only when its *final*
        # finite differences are feasible; otherwise keep the monotone servo
        # step toward the already-projected desired target.  This avoids a
        # permanent fail-closed stall while still converging into the band.
        projected_velocity = (
            projected_candidate - previous_target
        ) / dt
        projected_acceleration = (
            projected_velocity - previous_velocity
        ) / dt
        projected_jerk = (
            projected_acceleration - previous_acceleration
        ) / dt
        projection_epsilon = 2.0e-5
        projection_feasible = (
            torch.isfinite(projected_candidate).all(dim=(1, 2))
            & torch.isfinite(projected_velocity).all(dim=(1, 2))
            & torch.isfinite(projected_acceleration).all(dim=(1, 2))
            & torch.isfinite(projected_jerk).all(dim=(1, 2))
            & (
                projected_velocity.norm(dim=-1)
                <= TIP_TARGET_SPEED_MAX_M_S + projection_epsilon
            ).all(dim=1)
            & (
                projected_acceleration.norm(dim=-1)
                <= TIP_TARGET_ACCELERATION_MAX_M_S2 + projection_epsilon
            ).all(dim=1)
            & (
                projected_jerk.norm(dim=-1)
                <= TIP_TARGET_JERK_MAX_M_S3 + projection_epsilon
            ).all(dim=1)
        )
        candidate = torch.where(
            projection_feasible[:, None, None],
            projected_candidate,
            servo_candidate,
        )

        # These are deliberately recomputed from the final post-projection
        # command.  Never retain pre-projection kinematics in the certificate.
        realized_velocity = (candidate - previous_target) / dt
        realized_acceleration = (
            realized_velocity - previous_velocity
        ) / dt
        realized_jerk = (
            realized_acceleration - previous_acceleration
        ) / dt
        # Position finite differences are evaluated in float32 on both CPU and
        # CUDA.  Ten microunits covers only round-off from the two divisions by
        # ``dt``; it does not relax any physical cap in a meaningful way.
        epsilon = 2.0e-5
        certified = (
            desired_finite
            & torch.isfinite(candidate).all(dim=(1, 2))
            & torch.isfinite(realized_velocity).all(dim=(1, 2))
            & torch.isfinite(realized_acceleration).all(dim=(1, 2))
            & torch.isfinite(realized_jerk).all(dim=(1, 2))
            & (
                realized_velocity.norm(dim=-1)
                <= TIP_TARGET_SPEED_MAX_M_S + epsilon
            ).all(dim=1)
            & (
                realized_acceleration.norm(dim=-1)
                <= TIP_TARGET_ACCELERATION_MAX_M_S2 + epsilon
            ).all(dim=1)
            & (
                realized_jerk.norm(dim=-1)
                <= TIP_TARGET_JERK_MAX_M_S3 + epsilon
            ).all(dim=1)
        )
        update = active & certified
        clipped = (
            (requested_jerk.norm(dim=-1) > TIP_TARGET_JERK_MAX_M_S3)
            | stopping_limited[:, :, 0]
            | crossed_target[:, :, 0]
            | crossed_after_projection[:, :, 0]
        ).any(dim=1)
        self.tip_target_slew_clip_count.add_((active & clipped).to(torch.long))
        self.tip_target_certificate_fallback_count.add_(
            (active & (~certified)).to(torch.long)
        )
        self.last_certified_tip_targets.copy_(
            torch.where(update[:, None, None], candidate, previous_target)
        )
        self.tip_target_velocity_world_m_s.copy_(
            torch.where(
                update[:, None, None],
                realized_velocity,
                previous_velocity,
            )
        )
        self.tip_target_acceleration_world_m_s2.copy_(
            torch.where(
                update[:, None, None],
                realized_acceleration,
                previous_acceleration,
            )
        )
        self.tip_target_jerk_world_m_s3.copy_(
            torch.where(
                update[:, None, None],
                realized_jerk,
                self.tip_target_jerk_world_m_s3,
            )
        )
        return self.last_certified_tip_targets

    def _jacobian(self, name: str, points: Any, body_ids: Any) -> Any:
        value = self.backend.jacobian(name, points, body_ids)
        if isinstance(value, (tuple, list)):
            value = value[0]
        if tuple(value.shape) == (self.worlds, 3, self._nv):
            return value
        if tuple(value.shape) == (self.worlds, self._nv, 3):
            return value.transpose(1, 2)
        raise ValueError(
            f"backend jacobian {name!r} has shape {tuple(value.shape)}"
        )

    def _solve_kkt(
        self,
        jacobian: Any,
        constraint: Any,
        error: Any,
        pin_gap: Any,
        free: Any,
    ) -> tuple[Any, Any]:
        joint_count = int(jacobian.shape[-1])
        masked_jacobian = jacobian * free[:, None, :]
        masked_constraint = constraint * free[:, None, :]
        if joint_count not in self._kkt_eye:
            self._kkt_eye[joint_count] = torch.eye(
                joint_count, device=self.device, dtype=self.dtype
            )
        hessian = (
            masked_jacobian.transpose(1, 2) @ masked_jacobian
            + IK_DAMPING * self._kkt_eye[joint_count][None, :, :]
        )
        kkt = torch.zeros(
            (self.worlds, joint_count + 3, joint_count + 3),
            device=self.device,
            dtype=self.dtype,
        )
        kkt[:, :joint_count, :joint_count] = hessian
        kkt[:, :joint_count, joint_count:] = masked_constraint.transpose(
            1, 2
        )
        kkt[:, joint_count:, :joint_count] = masked_constraint
        kkt[:, joint_count:, joint_count:] = (
            -CONSTRAINT_DAMPING
            * torch.eye(3, device=self.device, dtype=self.dtype)[
                None, :, :
            ]
        )
        rhs = torch.cat(
            (
                (masked_jacobian.transpose(1, 2) @ error[:, :, None])[
                    :, :, 0
                ],
                -pin_gap,
            ),
            dim=-1,
        )
        solved, info = torch.linalg.solve_ex(
            kkt, rhs[:, :, None], check_errors=False
        )
        solved_vector = solved[:, :, 0]
        residual = (kkt @ solved)[:, :, 0] - rhs
        matrix_inf = kkt.abs().sum(dim=2).amax(dim=1)
        solution_inf = solved_vector.abs().amax(dim=1)
        rhs_inf = rhs.abs().amax(dim=1)
        denominator = (
            matrix_inf * solution_inf
            + rhs_inf
            + torch.finfo(self.dtype).tiny
        )
        backward_error = residual.abs().amax(dim=1) / denominator
        feasible = (
            (info == 0)
            & torch.isfinite(kkt).all(dim=(1, 2))
            & torch.isfinite(rhs).all(dim=1)
            & torch.isfinite(solved).all(dim=(1, 2))
            & torch.isfinite(backward_error)
            & (backward_error <= IK_BACKWARD_ERROR_LIMIT)
        )
        # Match the proven SCREW controller's fail-closed semantics: an
        # uncertified per-world IK result contributes zero delta, so the
        # caller keeps the last finite actuator command.
        solved = torch.where(
            feasible[:, None, None], solved, torch.zeros_like(solved)
        )
        return free * solved[:, :joint_count, 0], feasible

    def _refine_kkt_active_set(
        self,
        jacobian: Any,
        constraint: Any,
        error: Any,
        pin_gap: Any,
        joint_qpos: Any,
        joint_low: Any,
        joint_high: Any,
        free: Any,
        delta: Any,
        solved_feasible: Any,
    ) -> tuple[Any, Any]:
        """Resolve one direction flip after the bound-constrained KKT solve."""

        refined_free = free * _ik_free_mask_at_bounds(
            joint_qpos, joint_low, joint_high, delta
        )
        has_new_blocker = (refined_free < free).any(dim=1)
        refined_delta, refined_feasible = self._solve_kkt(
            jacobian,
            constraint,
            error,
            pin_gap,
            refined_free,
        )
        # Always execute the fixed-shape third solve so CUDA Graph capture and
        # replay have identical work.  Only worlds whose active set tightened
        # consume it; a failed refinement therefore remains fail-closed.
        delta = torch.where(
            has_new_blocker[:, None], refined_delta, delta
        )
        solved_feasible = torch.where(
            has_new_blocker, refined_feasible, solved_feasible
        )
        return delta, solved_feasible

    def _trust_region_delta(
        self, delta: Any, limits: Any
    ) -> tuple[Any, Any]:
        """Scale a whole finger joint vector without changing its direction."""

        if limits.ndim == 1:
            limits = limits[None, :].expand_as(delta)
        if tuple(limits.shape) != tuple(delta.shape):
            raise ValueError("trust-region limits must match the joint delta")
        tiny = torch.finfo(self.dtype).tiny
        safe_limits = limits.clamp_min(tiny)
        ratio = delta.abs() / safe_limits
        ratio = torch.where(
            (limits <= tiny) & (delta.abs() > tiny),
            torch.full_like(ratio, torch.inf),
            ratio,
        )
        maximum_ratio = ratio.amax(dim=1)
        scale = maximum_ratio.clamp_min(1.0).reciprocal()
        return delta * scale[:, None], maximum_ratio > 1.0

    def _ik_trust_stage_index(self) -> Any:
        """Attribute trust projections to grasp/lift/start/route stages."""

        lift_start = (
            self.config.plan_ticks
            + self.config.close_ticks
            + self.config.settle_ticks
        )
        action_start = lift_start + self.config.lift_ticks + self.config.hold_ticks
        route_start = (
            action_start
            + contract.TRANSLATION_PATH_LEAD_IN_TICKS
            + contract.TRANSLATION_PATH_START_SETTLE_TICKS
        )
        grasp = self.control_step < lift_start
        lifting = (self.control_step >= lift_start) & (
            self.control_step < action_start
        )
        translating = self.task_index == contract.TASKS.index(
            contract.TASK_TRANSLATE
        )
        path_start = (
            translating
            & (self.control_step >= action_start)
            & (self.control_step < route_start)
        )
        result = torch.full_like(
            self.control_step, IK_TRUST_STAGES.index("route")
        )
        result = torch.where(
            path_start,
            torch.full_like(result, IK_TRUST_STAGES.index("path_start")),
            result,
        )
        result = torch.where(
            lifting,
            torch.full_like(result, IK_TRUST_STAGES.index("lift")),
            result,
        )
        return torch.where(
            grasp,
            torch.full_like(result, IK_TRUST_STAGES.index("grasp")),
            result,
        )

    def _project_ik_cage_targets(
        self, targets: Any, active_worlds: Any
    ) -> Any:
        """Apply one common workspace trust scale to all four hard tips.

        The public Cartesian command and its v/a/jerk history remain intact.
        Only the target consumed by the current Newton iteration is shortened
        when its largest actual tip error exceeds 3 mm.  A single per-world
        scale preserves the four independent target-error directions and
        prevents one finger from taking a much larger workspace step than the
        rest of the cage; the original joint trust region remains the final
        fail-closed guard.
        """

        tip_position = self.backend.tensor("xpos").index_select(
            1, self._tip_body_ids
        )
        delta = targets - tip_position
        finite = (
            torch.isfinite(targets).all(dim=(1, 2))
            & torch.isfinite(tip_position).all(dim=(1, 2))
            & torch.isfinite(delta).all(dim=(1, 2))
        )
        max_error = delta.norm(dim=-1).amax(dim=1)
        scale = torch.minimum(
            torch.ones_like(max_error),
            torch.full_like(max_error, IK_CAGE_TARGET_ERROR_MAX_M)
            / max_error.clamp_min(torch.finfo(self.dtype).tiny),
        )
        scale = torch.where(finite, scale, torch.ones_like(scale))
        projected = (
            active_worlds
            & finite
            & (scale < 1.0 - 1.0e-7)
        )
        self.ik_cage_projection_count.add_(projected.to(torch.long))
        stage_index = self._ik_trust_stage_index()
        stage_axis = torch.arange(
            len(IK_TRUST_STAGES), device=self.device, dtype=torch.long
        )
        self.ik_cage_projection_count_by_stage.add_(
            (
                projected[:, None]
                & (stage_index[:, None] == stage_axis[None, :])
            ).to(torch.long)
        )
        self.ik_cage_projection_scale_min.copy_(
            torch.where(
                active_worlds,
                torch.minimum(self.ik_cage_projection_scale_min, scale),
                self.ik_cage_projection_scale_min,
            )
        )
        self.ik_cage_projection_scale_sum.add_(
            torch.where(active_worlds, scale, torch.zeros_like(scale))
        )
        self.ik_cage_projection_sample_count.add_(
            active_worlds.to(torch.long)
        )
        shortened = tip_position + delta * scale[:, None, None]
        return torch.where(
            active_worlds[:, None, None], shortened, targets
        )

    def _constrained_ik_delta(
        self, targets: Any
    ) -> tuple[Any, Any, Any, Any]:
        qpos = self.backend.tensor("qpos")
        xpos = self.backend.tensor("xpos")
        site_xpos = self.backend.tensor("site_xpos")
        active_delta = torch.zeros(
            (self.worlds, self._ctrl_low.numel()),
            device=self.device,
            dtype=self.dtype,
        )
        passive_delta = torch.zeros(
            (self.worlds, self._passive_qpos_low.numel()),
            device=self.device,
            dtype=self.dtype,
        )
        active_count = int(self._finger_actuator_indices.shape[1])
        passive_count = int(self._finger_passive_indices.shape[1])
        joint_count = active_count + passive_count
        feasible = torch.ones(
            self.worlds, device=self.device, dtype=torch.bool
        )
        projected = torch.zeros_like(feasible)
        for tip_index in range(len(contract.TIP_ORDER)):
            tip_body = self._tip_body_id_values[tip_index]
            long_site, front_site = self._pin_site_id_values[tip_index]
            tip_jacobian = self._jacobian(
                f"tip_{tip_index}",
                xpos[:, tip_body],
                self._world_tip_body_ids[:, tip_index],
            )
            long_jacobian = self._jacobian(
                f"pin_long_{tip_index}",
                site_xpos[:, long_site],
                self._world_pin_body_ids[:, tip_index, 0],
            )
            front_jacobian = self._jacobian(
                f"pin_front_{tip_index}",
                site_xpos[:, front_site],
                self._world_pin_body_ids[:, tip_index, 1],
            )
            active_indices = self._finger_actuator_indices[tip_index]
            passive_indices = self._finger_passive_indices[tip_index]
            dof_indices = torch.cat(
                (
                    self._actuator_dof_adrs.index_select(
                        0, active_indices
                    ),
                    self._passive_dof_adrs.index_select(
                        0, passive_indices
                    ),
                )
            )
            jacobian = tip_jacobian.index_select(2, dof_indices)
            constraint_jacobian = (
                long_jacobian - front_jacobian
            ).index_select(2, dof_indices)
            error = IK_POSITION_GAIN * (
                targets[:, tip_index] - xpos[:, tip_body]
            )
            pin_gap = site_xpos[:, long_site] - site_xpos[:, front_site]
            free = torch.ones(
                (self.worlds, joint_count),
                device=self.device,
                dtype=self.dtype,
            )
            raw_delta, raw_feasible = self._solve_kkt(
                jacobian,
                constraint_jacobian,
                error,
                pin_gap,
                free,
            )
            feasible &= raw_feasible
            active_qpos = qpos.index_select(
                1,
                self._actuator_qpos_adrs.index_select(
                    0, active_indices
                ),
            )
            low = self._ctrl_low.index_select(0, active_indices)
            high = self._ctrl_high.index_select(0, active_indices)
            passive_qpos = qpos.index_select(
                1,
                self._passive_qpos_adrs.index_select(
                    0, passive_indices
                ),
            )
            passive_low = self._passive_qpos_low.index_select(
                0, passive_indices
            )
            passive_high = self._passive_qpos_high.index_select(
                0, passive_indices
            )
            joint_qpos = torch.cat((active_qpos, passive_qpos), dim=1)
            joint_low = torch.cat((low, passive_low))
            joint_high = torch.cat((high, passive_high))
            free.copy_(
                _ik_free_mask_at_bounds(
                    joint_qpos, joint_low, joint_high, raw_delta
                )
            )
            delta, solved_feasible = self._solve_kkt(
                jacobian,
                constraint_jacobian,
                error,
                pin_gap,
                free,
            )
            delta, solved_feasible = self._refine_kkt_active_set(
                jacobian,
                constraint_jacobian,
                error,
                pin_gap,
                joint_qpos,
                joint_low,
                joint_high,
                free,
                delta,
                solved_feasible,
            )
            base_limits = torch.cat(
                (
                    self._ctrl_rate.index_select(0, active_indices)
                    / float(IK_NEWTON_ITERATIONS),
                    torch.full(
                        (passive_count,),
                        0.055 / float(IK_NEWTON_ITERATIONS),
                        device=self.device,
                        dtype=self.dtype,
                    ),
                )
            )
            positive_room = torch.cat(
                (
                    high[None, :] - active_qpos,
                    passive_high[None, :] - passive_qpos,
                ),
                dim=1,
            ).clamp_min(0.0)
            negative_room = torch.cat(
                (
                    active_qpos - low[None, :],
                    passive_qpos - passive_low[None, :],
                ),
                dim=1,
            ).clamp_min(0.0)
            bound_room = torch.where(
                delta >= 0.0, positive_room, negative_room
            )
            limits = torch.minimum(
                base_limits[None, :].expand_as(delta), bound_room
            )
            unconstrained_delta_norm = delta.norm(dim=1)
            delta, finger_projected = self._trust_region_delta(
                delta, limits
            )
            bound_blocked = (
                (unconstrained_delta_norm > 1.0e-7)
                & (delta.norm(dim=1) <= 1.0e-9)
            )
            predicted_tip_error = error - (
                jacobian @ delta[:, :, None]
            )[:, :, 0]
            predicted_pin_gap = pin_gap + (
                constraint_jacobian @ delta[:, :, None]
            )[:, :, 0]
            descent = (
                predicted_tip_error.norm(dim=1)
                <= error.norm(dim=1) + 5.0e-5
            ) & (
                predicted_pin_gap.norm(dim=1)
                <= pin_gap.norm(dim=1) + 5.0e-5
            )
            finger_feasible = solved_feasible & descent & (~bound_blocked)
            feasible &= finger_feasible
            projected |= finger_projected
            delta = torch.where(
                finger_feasible[:, None], delta, torch.zeros_like(delta)
            )
            active_delta.index_copy_(
                1, active_indices, delta[:, :active_count]
            )
            passive_delta.index_copy_(
                1, passive_indices, delta[:, active_count:]
            )
        active_delta = torch.where(
            feasible[:, None], active_delta, torch.zeros_like(active_delta)
        )
        passive_delta = torch.where(
            feasible[:, None], passive_delta, torch.zeros_like(passive_delta)
        )
        return active_delta, passive_delta, feasible, projected

    def _ik_control(
        self, targets: Any, *, active_worlds: Any | None = None
    ) -> Any:
        """Apply and attribute IK only to worlds whose episode is active.

        Object-drop terminals remain in the fixed-size CUDA batch until the
        other worlds finish.  They must not keep re-entering IK after their
        object has been quarantined: doing so can turn harmless closed-chain
        drift in an already-terminal world into thousands of false fallback
        counts.  Inactive worlds therefore retain their last certified
        command and are reported feasible without contributing to counters.
        """

        qpos = self.backend.tensor("qpos")
        ctrl = self.backend.tensor("ctrl")
        if active_worlds is None:
            active_worlds = torch.ones(
                self.worlds, device=self.device, dtype=torch.bool
            )
        else:
            active_worlds = torch.as_tensor(
                active_worlds, device=self.device, dtype=torch.bool
            )
            if tuple(active_worlds.shape) != (self.worlds,):
                raise ValueError(
                    "active_worlds must contain one mask value per world"
                )
        original_active = qpos.index_select(
            1, self._actuator_qpos_adrs
        ).clone()
        original_passive = qpos.index_select(
            1, self._passive_qpos_adrs
        ).clone()
        feasible = torch.ones(
            self.worlds, device=self.device, dtype=torch.bool
        )
        warmstart = None
        if self._ik_warmstart_backup is not None:
            warmstart = self.backend.tensor("qacc_warmstart")
            self._ik_warmstart_backup.copy_(warmstart)
        try:
            for _ in range(IK_NEWTON_ITERATIONS):
                iteration_targets = self._project_ik_cage_targets(
                    targets, active_worlds
                )
                (
                    active_delta,
                    passive_delta,
                    iteration_feasible,
                    trust_projected,
                ) = self._constrained_ik_delta(iteration_targets)
                # A terminal world is deliberately outside the IK contract.
                # Treat it as feasible and force both temporary deltas to zero.
                iteration_feasible = iteration_feasible | (~active_worlds)
                feasible &= iteration_feasible
                self.ik_trust_region_projection_count.add_(
                    (trust_projected & active_worlds).to(torch.long)
                )
                stage_counts = getattr(
                    self,
                    "ik_trust_region_projection_count_by_stage",
                    None,
                )
                if stage_counts is not None:
                    stage_index = self._ik_trust_stage_index()
                    stage_axis = torch.arange(
                        len(IK_TRUST_STAGES),
                        device=self.device,
                        dtype=torch.long,
                    )
                    stage_counts.add_(
                        (
                            (trust_projected & active_worlds)[:, None]
                            & (stage_index[:, None] == stage_axis[None, :])
                        ).to(torch.long)
                    )
                active_delta = torch.where(
                    (feasible & active_worlds)[:, None],
                    active_delta,
                    torch.zeros_like(active_delta),
                )
                passive_delta = torch.where(
                    (feasible & active_worlds)[:, None],
                    passive_delta,
                    torch.zeros_like(passive_delta),
                )
                active = (
                    qpos.index_select(1, self._actuator_qpos_adrs)
                    + active_delta
                ).clamp(self._ctrl_low[None, :], self._ctrl_high[None, :])
                passive = (
                    qpos.index_select(1, self._passive_qpos_adrs)
                    + passive_delta
                ).clamp(
                    self._passive_qpos_low[None, :],
                    self._passive_qpos_high[None, :],
                )
                qpos.copy_(
                    qpos.scatter(
                        1,
                        self._actuator_qpos_adrs[None, :].expand(
                            self.worlds, -1
                        ),
                        active,
                    )
                )
                qpos.copy_(
                    qpos.scatter(
                        1,
                        self._passive_qpos_adrs[None, :].expand(
                            self.worlds, -1
                        ),
                        passive,
                    )
                )
                self.backend.forward()
            desired = qpos.index_select(
                1, self._actuator_qpos_adrs
            ).clone()
        except BaseException:
            qpos.copy_(
                qpos.scatter(
                    1,
                    self._actuator_qpos_adrs[None, :].expand(
                        self.worlds, -1
                    ),
                    original_active,
                )
            )
            qpos.copy_(
                qpos.scatter(
                    1,
                    self._passive_qpos_adrs[None, :].expand(
                        self.worlds, -1
                    ),
                    original_passive,
                )
            )
            if warmstart is not None:
                warmstart.copy_(self._ik_warmstart_backup)
            try:
                self.backend.forward()
            except BaseException:
                pass
            qpos.copy_(
                qpos.scatter(
                    1,
                    self._actuator_qpos_adrs[None, :].expand(
                        self.worlds, -1
                    ),
                    original_active,
                )
            )
            qpos.copy_(
                qpos.scatter(
                    1,
                    self._passive_qpos_adrs[None, :].expand(
                        self.worlds, -1
                    ),
                    original_passive,
                )
            )
            if warmstart is not None:
                warmstart.copy_(self._ik_warmstart_backup)
            raise
        qpos.copy_(
            qpos.scatter(
                1,
                self._actuator_qpos_adrs[None, :].expand(self.worlds, -1),
                original_active,
            )
        )
        qpos.copy_(
            qpos.scatter(
                1,
                self._passive_qpos_adrs[None, :].expand(
                    self.worlds, -1
                ),
                original_passive,
            )
        )
        if warmstart is not None:
            warmstart.copy_(self._ik_warmstart_backup)
        try:
            self.backend.forward()
        finally:
            qpos.copy_(
                qpos.scatter(
                    1,
                    self._actuator_qpos_adrs[None, :].expand(
                        self.worlds, -1
                    ),
                    original_active,
                )
            )
            qpos.copy_(
                qpos.scatter(
                    1,
                    self._passive_qpos_adrs[None, :].expand(
                        self.worlds, -1
                    ),
                    original_passive,
                )
            )
            if warmstart is not None:
                warmstart.copy_(self._ik_warmstart_backup)
        ctrl_finite = torch.isfinite(ctrl).all(dim=1)
        safe_ctrl = torch.where(
            ctrl_finite[:, None], ctrl, self.last_certified_ctrl
        )
        candidate_finite = (
            torch.isfinite(targets).all(dim=(1, 2))
            & torch.isfinite(desired).all(dim=1)
            & ctrl_finite
        )
        feasible &= candidate_finite | (~active_worlds)
        desired = torch.where(
            (feasible & active_worlds)[:, None],
            desired,
            self.last_certified_ctrl,
        )
        self._last_ik_solve_feasible.copy_(feasible)
        self.ik_solve_fallback_count.add_(
            (active_worlds & (~feasible)).to(torch.long)
        )
        command_delta = (
            0.65 * (desired - safe_ctrl)
        ).clamp(-self._ctrl_rate[None, :], self._ctrl_rate[None, :])
        candidate = (safe_ctrl + command_delta).clamp(
            self._ctrl_low[None, :], self._ctrl_high[None, :]
        )
        return torch.where(
            active_worlds[:, None], candidate, self.last_certified_ctrl
        )

    def _rate_limit_control(self, desired: Any, frozen: Any) -> Any:
        """Apply a certified second-order actuator command filter."""

        previous = self.backend.tensor("ctrl")
        previous_finite = torch.isfinite(previous).all(dim=1)
        previous_in_bounds = (
            (previous >= self._ctrl_low[None, :])
            & (previous <= self._ctrl_high[None, :])
        ).all(dim=1)
        previous_certified = previous_finite & previous_in_bounds
        safe_previous = torch.where(
            previous_certified[:, None], previous, self.last_certified_ctrl
        )
        desired_finite = torch.isfinite(desired).all(dim=1)
        frozen = frozen.to(torch.bool)
        braking = frozen | (~desired_finite)
        target = torch.where(
            desired_finite[:, None], desired, safe_previous
        ).clamp(self._ctrl_low[None, :], self._ctrl_high[None, :])
        dt = float(self.backend.control_dt)
        alpha = 1.0 - math.exp(-2.0 * math.pi * 10.0 * dt)
        filtered = safe_previous + alpha * (target - safe_previous)
        omega = min(max(2.0 * math.pi * 10.0, 2.0), 12.0)
        error = filtered - safe_previous
        desired_acceleration = (
            omega * omega * error - 2.0 * omega * self.ctrl_velocity
        )
        normal_acceleration = torch.maximum(
            torch.minimum(
                desired_acceleration,
                self.ctrl_acceleration + self._ctrl_jerk_limit[None, :] * dt,
            ),
            self.ctrl_acceleration - self._ctrl_jerk_limit[None, :] * dt,
        )
        normal_acceleration = torch.maximum(
            torch.minimum(
                normal_acceleration,
                self._ctrl_acceleration_limit[None, :],
            ),
            -self._ctrl_acceleration_limit[None, :],
        )
        desired_brake_acceleration = torch.maximum(
            torch.minimum(
                -self.ctrl_velocity / dt,
                self._ctrl_acceleration_limit[None, :],
            ),
            -self._ctrl_acceleration_limit[None, :],
        )
        brake_acceleration = torch.maximum(
            torch.minimum(
                desired_brake_acceleration,
                self.ctrl_acceleration
                + self._ctrl_jerk_limit[None, :] * dt,
            ),
            self.ctrl_acceleration
            - self._ctrl_jerk_limit[None, :] * dt,
        )
        brake_acceleration = torch.maximum(
            torch.minimum(
                brake_acceleration,
                self._ctrl_acceleration_limit[None, :],
            ),
            -self._ctrl_acceleration_limit[None, :],
        )
        acceleration = torch.where(
            braking[:, None], brake_acceleration, normal_acceleration
        )
        velocity = self.ctrl_velocity + acceleration * dt
        exact_stop_acceleration = -self.ctrl_velocity / dt
        exact_stop_jerk = (
            exact_stop_acceleration - self.ctrl_acceleration
        ) / dt
        can_stop_exactly = (
            exact_stop_acceleration.abs()
            <= self._ctrl_acceleration_limit[None, :] * 1.001 + 1.0e-5
        ) & (
            exact_stop_jerk.abs()
            <= self._ctrl_jerk_limit[None, :] * 1.001 + 1.0e-4
        )
        crosses_zero = (
            (self.ctrl_velocity * velocity <= 0.0)
            & (self.ctrl_velocity.abs() > torch.finfo(self.dtype).eps)
        )
        velocity = torch.where(
            braking[:, None] & crosses_zero & can_stop_exactly,
            torch.zeros_like(velocity),
            velocity,
        )
        velocity_limit = self._ctrl_rate / dt
        velocity = torch.maximum(
            torch.minimum(velocity, velocity_limit[None, :]),
            -velocity_limit[None, :],
        )
        candidate = (safe_previous + velocity * dt).clamp(
            self._ctrl_low[None, :], self._ctrl_high[None, :]
        )
        realized_velocity = (candidate - safe_previous) / dt
        realized_acceleration = (
            realized_velocity - self.ctrl_velocity
        ) / dt
        realized_jerk = (
            realized_acceleration - self.ctrl_acceleration
        ) / dt
        within_rate = (
            (candidate - safe_previous).abs()
            <= self._ctrl_rate[None, :] + 8.0 * torch.finfo(self.dtype).eps
        ).all(dim=1)
        within_acceleration = (
            realized_acceleration.abs()
            <= self._ctrl_acceleration_limit[None, :] * 1.001 + 1.0e-5
        ).all(dim=1)
        within_jerk = (
            realized_jerk.abs()
            <= self._ctrl_jerk_limit[None, :] * 1.001 + 1.0e-4
        ).all(dim=1)
        dynamic_certificate = (
            torch.isfinite(candidate).all(dim=1)
            & within_rate
            & within_acceleration
            & within_jerk
        )
        motion_certified = previous_certified & dynamic_certificate
        self.ctrl_certificate_fallback_count.add_(
            (~(motion_certified & desired_finite)).to(torch.long)
        )
        self.ctrl_slew_clip_count.add_(
            (
                (~frozen)
                & (
                    (filtered - safe_previous).abs()
                    > self._ctrl_rate[None, :]
                ).any(dim=1)
            ).to(torch.long)
        )
        if not bool(motion_certified.all()):
            bad = torch.nonzero(
                ~motion_certified, as_tuple=False
            ).flatten()
            first = int(bad[0].detach().cpu())
            diagnostic = {
                "world": first,
                "frozen": bool(frozen[first].detach().cpu()),
                "braking": bool(braking[first].detach().cpu()),
                "previous_certified": bool(
                    previous_certified[first].detach().cpu()
                ),
                "desired_finite": bool(
                    desired_finite[first].detach().cpu()
                ),
                "within_rate": bool(within_rate[first].detach().cpu()),
                "within_acceleration": bool(
                    within_acceleration[first].detach().cpu()
                ),
                "within_jerk": bool(within_jerk[first].detach().cpu()),
                "previous": safe_previous[first].detach().cpu().tolist(),
                "ctrl_low": self._ctrl_low.detach().cpu().tolist(),
                "ctrl_high": self._ctrl_high.detach().cpu().tolist(),
                "candidate": candidate[first].detach().cpu().tolist(),
                "velocity_before": self.ctrl_velocity[first]
                .detach()
                .cpu()
                .tolist(),
                "acceleration_before": self.ctrl_acceleration[first]
                .detach()
                .cpu()
                .tolist(),
                "velocity_after": realized_velocity[first]
                .detach()
                .cpu()
                .tolist(),
                "acceleration_after": realized_acceleration[first]
                .detach()
                .cpu()
                .tolist(),
                "jerk": realized_jerk[first].detach().cpu().tolist(),
            }
            raise FloatingPointError(
                "actuator filter could not construct a certified braking "
                f"trajectory for {int((~motion_certified).sum().detach().cpu())} "
                "world(s): "
                + json.dumps(diagnostic, sort_keys=True)
            )
        self.ctrl_velocity.copy_(realized_velocity)
        self.ctrl_acceleration.copy_(realized_acceleration)
        self.last_certified_ctrl.copy_(candidate)
        return candidate

    def _tensor_any(self, *names: str) -> Any:
        last: BaseException | None = None
        for name in names:
            try:
                return self.backend.tensor(name)
            except (AttributeError, KeyError) as exc:
                last = exc
        raise RuntimeError(
            "native MJWarp backend lacks required tensors: "
            + ", ".join(names)
        ) from last

    def _finger_contact_force(self) -> Any:
        """Decode exact hard-tip/current-fruit normal contact forces.

        The generic direct-force tensors exposed by some runtimes do not
        carry geom-pair identity, so they are intentionally not accepted by
        the expert certificate.  This decoder always proves both participants
        from the native contact records.  MuJoCo-Warp owns one global contact
        buffer and only ``contact[:nacon[0]]`` is live; entries in the unused
        suffix are scratch storage and may retain geom IDs, EFC addresses, and
        forces from an earlier step.

        Three deliberately separate facts are retained for diagnostics:

        * ``raw_pair``: the exact geom pair occurs in the live contact prefix;
        * ``pair``: that exact pair also owns at least one active EFC row;
        * ``force_qualified_pair``: its summed finite normal force reaches the
          shared expert threshold.

        ``_last_tip_object_contact_pair`` keeps its historical meaning: it is
        the active-EFC exact-pair flag, not merely a broadphase/raw contact.
        """

        geoms = self._tensor_any(
            "contact.geom", "contact_geom", "contact__geom"
        )
        worldid = self._tensor_any(
            "contact.worldid", "contact_worldid", "contact__worldid"
        )
        dimensions = self._tensor_any(
            "contact.dim", "contact_dim", "contact__dim"
        )
        try:
            contact_type = self._tensor_any(
                "contact.type", "contact_type", "contact__type"
            )
        except RuntimeError:
            # The pinned production backend always exposes Contact.type.  A
            # legacy compatibility backend without it cannot prove that a
            # reused address belongs to a constraint contact, so fail closed
            # instead of treating a raw/sensor pair as solver-active.
            contact_type = None
        addresses = self._tensor_any(
            "contact.efc_address",
            "contact_efc_address",
            "contact__efc_address",
        )
        force_array = self._tensor_any(
            "efc.force", "efc_force", "efc__force"
        )
        nacon = self._tensor_any("nacon").reshape(-1).to(torch.long)
        if int(nacon.numel()) != 1:
            raise RuntimeError(
                "native MJWarp contact decoder requires scalar global nacon"
            )
        if geoms.ndim == 3:
            worlds, slots = geoms.shape[:2]
            worldid = torch.arange(
                worlds, device=self.device, dtype=torch.long
            )[:, None].expand(worlds, slots)
        geoms = geoms.reshape(-1, 2).to(torch.long)
        worldid = worldid.reshape(-1).to(torch.long)
        dimensions = dimensions.reshape(-1).to(torch.long)
        if contact_type is None:
            contact_type = torch.zeros_like(dimensions)
        else:
            contact_type = contact_type.reshape(-1).to(torch.long)
        entry_count = int(geoms.shape[0])
        if int(worldid.numel()) != entry_count or int(
            dimensions.numel()
        ) != entry_count or int(contact_type.numel()) != entry_count:
            raise RuntimeError(
                "native MJWarp contact arrays have inconsistent capacities"
            )
        live_count = nacon[0]
        count_valid = (live_count >= 0) & (live_count <= entry_count)
        slot_index = getattr(self, "_contact_slot_index", None)
        if slot_index is None or int(slot_index.numel()) != entry_count:
            slot_index = torch.arange(
                entry_count, device=self.device, dtype=torch.long
            )
            self._contact_slot_index = slot_index
        live_prefix = (
            slot_index < live_count.clamp(0, entry_count)
        ) & count_valid

        # Pinned mujoco-warp 3.10.0.3 exposes efc_address as
        # (naconmax, nmaxpyramid).  Keep a narrow compatibility path for a
        # one-address tensor, but never infer subsequent addresses when the
        # native matrix is available: constraint overflow can make later
        # pyramid rows -1 independently.
        if addresses.ndim == 1:
            base_address = addresses.reshape(-1).to(torch.long)
            if int(base_address.numel()) != entry_count:
                raise RuntimeError(
                    "native MJWarp contact EFC addresses have inconsistent "
                    "capacity"
                )
            offsets = torch.arange(
                10, device=self.device, dtype=torch.long
            )
            addresses = base_address[:, None] + offsets[None, :]
            legacy_base_active = base_address[:, None] >= 0
        else:
            address_width = int(addresses.shape[-1])
            addresses = addresses.reshape(-1, address_width).to(torch.long)
            if int(addresses.shape[0]) != entry_count:
                raise RuntimeError(
                    "native MJWarp contact EFC addresses have inconsistent "
                    "capacity"
                )
            offsets = torch.arange(
                address_width, device=self.device, dtype=torch.long
            )
            legacy_base_active = None
        direction_count = torch.where(
            dimensions == 1,
            torch.ones_like(dimensions),
            2 * (dimensions - 1).clamp_min(0),
        )
        valid_world = (worldid >= 0) & (worldid < self.worlds)
        safe_world = worldid.clamp(0, self.worlds - 1)
        if force_array.ndim == 2:
            capacity = int(force_array.shape[1])
            safe_address = addresses.clamp(0, capacity - 1)
            values = force_array[safe_world[:, None], safe_address]
        else:
            capacity = int(self.config.constraints_per_world)
            flat_address = safe_world[:, None] * capacity + addresses
            safe_address = flat_address.clamp(0, force_array.numel() - 1)
            values = force_array.reshape(-1)[safe_address]
        active_address = (
            (offsets[None, :] < direction_count[:, None])
            & (addresses >= 0)
            & (addresses < capacity)
        )
        if legacy_base_active is not None:
            active_address &= legacy_base_active
        normal = torch.where(
            active_address, values, torch.zeros_like(values)
        ).sum(dim=1)
        raw_entry = live_prefix & valid_world
        active_entry = (
            raw_entry
            # ContactType.CONSTRAINT is bit zero in pinned MuJoCo-Warp.
            # Sensor-only contacts skip EFC allocation and can otherwise
            # expose stale addresses retained at a reused live slot.
            & ((contact_type & 1) != 0)
            & (dimensions > 0)
            & active_address.any(dim=1)
        )
        force_finite = torch.ones(
            self.worlds, device=self.device, dtype=torch.bool
        )
        force_finite.logical_and_(count_valid)
        bad_force_worlds = safe_world[
            active_entry & (~torch.isfinite(normal))
        ]
        if int(bad_force_worlds.numel()) > 0:
            force_finite.index_fill_(0, bad_force_worlds, False)
        self._last_force_decode_finite.copy_(force_finite)
        valid = active_entry & torch.isfinite(normal)
        object_mask = self._object_geom_mask.index_select(
            0, self.object_index
        )
        first_geom_valid = (geoms[:, 0] >= 0) & (
            geoms[:, 0] < object_mask.shape[1]
        )
        second_geom_valid = (geoms[:, 1] >= 0) & (
            geoms[:, 1] < object_mask.shape[1]
        )
        first = geoms[:, 0].clamp(0, object_mask.shape[1] - 1)
        second = geoms[:, 1].clamp(0, object_mask.shape[1] - 1)
        first_is_object = first_geom_valid & object_mask[safe_world, first]
        second_is_object = second_geom_valid & object_mask[safe_world, second]
        result = torch.zeros(
            (self.worlds, 4), device=self.device, dtype=self.dtype
        )
        self._last_tip_object_contact_raw_pair.zero_()
        self._last_tip_object_contact_pair.zero_()
        order = torch.argsort(safe_world, stable=True)
        sorted_world = safe_world.index_select(0, order)
        sorted_entry_count = int(sorted_world.numel())
        end_mask = torch.ones(
            sorted_entry_count, device=self.device, dtype=torch.bool
        )
        if sorted_entry_count > 1:
            end_mask[:-1] = sorted_world[:-1] != sorted_world[1:]
        ends = torch.nonzero(end_mask, as_tuple=False).flatten()
        starts = torch.cat(
            (
                torch.zeros(1, device=self.device, dtype=torch.long),
                ends[:-1] + 1,
            )
        )
        for tip_index in range(4):
            tip_geom = self._tip_geom_ids[tip_index]
            matches = (
                (geoms[:, 0] == tip_geom) & second_is_object
            ) | ((geoms[:, 1] == tip_geom) & first_is_object)
            raw_pair_count = torch.zeros(
                self.worlds, device=self.device, dtype=torch.long
            )
            raw_pair_count.index_add_(
                0,
                safe_world,
                (raw_entry & matches).to(torch.long),
            )
            self._last_tip_object_contact_raw_pair[:, tip_index].copy_(
                raw_pair_count > 0
            )
            pair_count = torch.zeros(
                self.worlds, device=self.device, dtype=torch.long
            )
            pair_count.index_add_(
                0,
                safe_world,
                (active_entry & matches).to(torch.long),
            )
            self._last_tip_object_contact_pair[:, tip_index].copy_(
                pair_count > 0
            )
            values_by_contact = torch.where(
                valid & matches,
                normal.clamp_min(0.0).to(self.dtype),
                torch.zeros_like(normal, dtype=self.dtype),
            ).index_select(0, order).to(torch.float64)
            # A float32 prefix across every world loses low-order contact
            # force bits when two large prefixes are subtracted for a high
            # world index.  Accumulate the deterministic sorted reduction in
            # float64, then cast the small per-world segment back once.
            prefix = values_by_contact.cumsum(dim=0)
            before_start = torch.where(
                starts > 0,
                prefix.index_select(0, (starts - 1).clamp_min(0)),
                torch.zeros_like(starts, dtype=torch.float64),
            )
            segment_sum = (
                prefix.index_select(0, ends) - before_start
            ).to(self.dtype)
            result[:, tip_index].index_copy_(
                0, sorted_world.index_select(0, ends), segment_sum
            )
        self._last_tip_object_contact_force_qualified_pair.copy_(
            self._last_tip_object_contact_pair
            & self._last_force_decode_finite[:, None]
            & torch.isfinite(result)
            & (result >= CONTACT_FORCE_THRESHOLD_N)
        )
        return result

    def _target_contact_force(self, action: Any) -> Any:
        action_start = (
            self.config.plan_ticks
            + self.config.close_ticks
            + self.config.settle_ticks
            + self.config.lift_ticks
            + self.config.hold_ticks
        )
        ramp = _minimum_jerk(
            (
                self.control_step.to(self.dtype)
                - action_start
                + 1.0
            )
            / ACTION_FORCE_RAMP_TICKS
        )
        nominal = GRASP_CONTACT_FORCE_N + ramp * (
            ACTION_CONTACT_FORCE_N - GRASP_CONTACT_FORCE_N
        )
        force_scale = (
            contract.FORCE_SCALE_NOMINAL
            + action[:, contract.FORCE_SCALE_SLICE]
            * (contract.FORCE_SCALE_MAX - contract.FORCE_SCALE_MIN)
            / 2.0
        )
        return nominal[:, None] * force_scale

    def _hold_command_targets(self, requested_targets: Any) -> tuple[Any, Any]:
        frozen = (
            _hold_freeze_window(self.control_step, self.config)
            & self.hold_reference_valid
        )
        targets = torch.where(
            frozen[:, None, None],
            self.hold_frozen_cage_world,
            requested_targets,
        )
        return targets, frozen

    def _hold_control(self, proposed_ctrl: Any, frozen: Any) -> Any:
        return torch.where(
            frozen[:, None],
            self.hold_frozen_ctrl,
            proposed_ctrl,
        )

    def _updated_preload_trim(
        self, action: Any, force: Any
    ) -> Any:
        target = self._target_contact_force(action)
        force_finite = torch.isfinite(force).all(dim=1)
        safe_force = torch.nan_to_num(
            force, nan=0.0, posinf=0.0, neginf=0.0
        )
        raw_delta = FORCE_FEEDBACK_GAIN_M_PER_N * (target - safe_force)
        delta = torch.nan_to_num(
            raw_delta, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(
            -PRELOAD_TRIM_SLEW_M_PER_TICK,
            PRELOAD_TRIM_SLEW_M_PER_TICK,
        )
        updated = (self.preload_trim_m + delta).clamp(
            FORCE_FEEDBACK_MIN_M, FORCE_FEEDBACK_MAX_M
        )
        feedback_start = (
            self.config.plan_ticks
            + (4 * self.config.close_ticks) // 5
        )
        _, action_start = _hold_boundaries(self.config)
        in_action = self.control_step >= action_start
        action_relative_bound = in_action & self.action_reference_valid
        relative_low = (
            self.action_start_preload_trim_m
            - contract.TIP_PRELOAD_LIMIT_M
        ).clamp(FORCE_FEEDBACK_MIN_M, FORCE_FEEDBACK_MAX_M)
        relative_high = (
            self.action_start_preload_trim_m
            + contract.TIP_PRELOAD_LIMIT_M
        ).clamp(FORCE_FEEDBACK_MIN_M, FORCE_FEEDBACK_MAX_M)
        relative_bounded = torch.maximum(
            torch.minimum(updated, relative_high), relative_low
        )
        # The loaded action-start cage already contains the pre-action force
        # correction.  Bound only the subsequent independent per-tip motion;
        # a global [-10,+10] mm state clamp alone permits a 20 mm action delta
        # when the state crosses from one pole to the other.
        updated = torch.where(
            action_relative_bound[:, None], relative_bounded, updated
        )
        all_contacts = (force >= CONTACT_FORCE_THRESHOLD_N).all(dim=1)
        any_measured_contact = force_finite & (force.amax(dim=1) > 0)
        # Before action, one finite positive measurement is enough to enable
        # the bounded search.  During action, retain the stricter four-tip gate
        # so a missing tip cannot make the other tips chase force dynamically.
        contact_allows_feedback = any_measured_contact & (
            (~in_action) | all_contacts
        )
        enabled = (
            (self.control_step >= feedback_start)
            & contact_allows_feedback
            & (~_hold_freeze_window(self.control_step, self.config))
        )
        self.preload_trim_slew_clip_count.add_(
            (
                enabled
                & (
                    raw_delta.abs().amax(dim=1)
                    > PRELOAD_TRIM_SLEW_M_PER_TICK
                )
            ).to(torch.long)
        )
        return torch.where(
            enabled[:, None], updated, self.preload_trim_m
        )

    def _capture_hold_reference(
        self,
        capture: Any,
        commanded_targets: Any,
    ) -> None:
        actual_ctrl = self.backend.tensor("ctrl")
        self.hold_frozen_ctrl.copy_(
            torch.where(
                capture[:, None],
                actual_ctrl,
                self.hold_frozen_ctrl,
            )
        )
        self.hold_frozen_cage_world.copy_(
            torch.where(
                capture[:, None, None],
                commanded_targets,
                self.hold_frozen_cage_world,
            )
        )
        self.hold_reference_valid.logical_or_(capture)

    def _reset_hold_reference(self, ids: Any) -> None:
        self.hold_frozen_ctrl.index_fill_(0, ids, 0.0)
        self.hold_frozen_cage_world.index_fill_(0, ids, 0.0)
        self.hold_reference_valid.index_fill_(0, ids, False)

    def _capture_action_reference(
        self,
        capture: Any,
        commanded_targets: Any,
        action: Any,
        trim_at_command: Any,
    ) -> None:
        body_ids = self._object_body_ids.index_select(0, self.object_index)
        com = self.backend.tensor("xipos")[self._rows, body_ids]
        quaternion = _quat_normalize(
            self.backend.tensor("xquat")[self._rows, body_ids]
        )
        cage, goal_position, goal_quaternion = _post_lift_reference(
            com,
            quaternion,
            commanded_targets,
            self.translation,
            self.task_index,
            self.rotation_angle_deg,
        )
        self.action_start_com = torch.where(
            capture[:, None], com, self.action_start_com
        )
        self.action_start_quat = torch.where(
            capture[:, None], quaternion, self.action_start_quat
        )
        self.action_start_cage_local = torch.where(
            capture[:, None, None], cage, self.action_start_cage_local
        )
        self.action_start_action = torch.where(
            capture[:, None], action, self.action_start_action
        )
        self.action_start_preload_trim_m = torch.where(
            capture[:, None],
            trim_at_command,
            self.action_start_preload_trim_m,
        )
        self.action_start_position_feedback_m = torch.where(
            capture[:, None],
            self.position_feedback_m,
            self.action_start_position_feedback_m,
        )
        self.goal_pos = torch.where(
            capture[:, None], goal_position, self.goal_pos
        )
        self.goal_quat = torch.where(
            capture[:, None], goal_quaternion, self.goal_quat
        )
        self.action_reference_valid |= capture

    def _pose_metrics(self) -> tuple[Any, Any, Any, Any]:
        body_ids = self._object_body_ids.index_select(0, self.object_index)
        position = self.backend.tensor("xipos")[self._rows, body_ids]
        quaternion = self.backend.tensor("xquat")[self._rows, body_ids]
        position_error_xyz = self.goal_pos - position
        # Retain L2 norms for dense reward and telemetry only.  Formal success
        # is the closed axis-aligned XYZ box requested by the experiment.
        position_error = position_error_xyz.norm(dim=-1)
        euler = _relative_euler_xyz_deg(self.goal_quat, quaternion)
        euler_l2 = euler.norm(dim=-1)
        translating = self.task_index == contract.TASKS.index(
            contract.TASK_TRANSLATE
        )
        position_tolerance = torch.where(
            translating,
            torch.full_like(
                position_error,
                contract.TRANSLATION_POSITION_AXIS_ABS_TOLERANCE_M,
            ),
            torch.full_like(
                position_error,
                contract.ROTATION_POSITION_AXIS_ABS_TOLERANCE_M,
            ),
        )
        euler_tolerance = torch.where(
            translating,
            torch.full_like(
                euler_l2,
                contract.TRANSLATION_EULER_AXIS_ABS_TOLERANCE_DEG,
            ),
            torch.full_like(
                euler_l2,
                contract.ROTATION_EULER_AXIS_ABS_TOLERANCE_DEG,
            ),
        )
        success = (
            (position_error_xyz.abs() <= position_tolerance[:, None]).all(
                dim=-1
            )
            & (euler.abs() <= euler_tolerance[:, None]).all(dim=-1)
        )
        return position_error, euler_l2, success, euler

    def _object_velocity(self) -> Any:
        qvel = self.backend.tensor("qvel")
        addresses = self._object_qvel_adrs.index_select(
            0, self.object_index
        )
        indices = addresses[:, None] + torch.arange(
            6, device=self.device, dtype=torch.long
        )[None, :]
        return torch.gather(qvel, 1, indices)

    def _low_level_proprioception(
        self,
    ) -> tuple[
        Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any
    ]:
        """Return fixed-width per-tip mechanism state for a Markov observation."""

        qpos = self.backend.tensor("qpos")
        qvel = self.backend.tensor("qvel")
        ctrl = self.backend.tensor("ctrl")
        active_qpos = qpos.index_select(
            1, self._actuator_qpos_adrs
        )[:, self._finger_actuator_indices]
        active_qvel = qvel.index_select(
            1, self._actuator_dof_adrs
        )[:, self._finger_actuator_indices]
        active_ctrl = ctrl[:, self._finger_actuator_indices]
        pad = 4 - self._active_joint_slots
        active_qpos = torch.nn.functional.pad(active_qpos, (0, pad))
        active_qvel = torch.nn.functional.pad(active_qvel, (0, pad))
        active_ctrl = torch.nn.functional.pad(active_ctrl, (0, pad))

        def padded_actuator_state(value: Any) -> Any:
            grouped = value[:, self._finger_actuator_indices]
            return torch.nn.functional.pad(grouped, (0, pad)).reshape(
                self.worlds, -1
            )

        velocity_scale = (self._ctrl_rate / float(self.backend.control_dt))[
            None, :
        ]
        normalized_velocity = self.ctrl_velocity / velocity_scale
        normalized_acceleration = (
            self.ctrl_acceleration
            / self._ctrl_acceleration_limit[None, :]
        )
        ctrl_midpoint = 0.5 * (self._ctrl_low + self._ctrl_high)
        ctrl_half_range = 0.5 * (self._ctrl_high - self._ctrl_low)
        normalized_certified_ctrl = (
            self.last_certified_ctrl - ctrl_midpoint[None, :]
        ) / ctrl_half_range.clamp_min(1.0e-8)[None, :]
        active_valid = self._active_joint_valid_mask[None, :, :].expand(
            self.worlds, -1, -1
        )
        passive_qpos = qpos.index_select(
            1, self._passive_qpos_adrs
        )[:, self._finger_passive_indices]
        passive_qvel = qvel.index_select(
            1, self._passive_dof_adrs
        )[:, self._finger_passive_indices]
        if self.backend.has_field("qacc_warmstart"):
            qacc_warmstart = self.backend.tensor("qacc_warmstart")
            qacc_width = int(qacc_warmstart.shape[1])
            if qacc_width > 72:
                raise RuntimeError(
                    "qacc_warmstart exceeds the shared 72-DoF observation pad"
                )
            qacc_pad = 72 - qacc_width
            qacc_warmstart = torch.nn.functional.pad(
                torch.tanh(qacc_warmstart / 100.0), (0, qacc_pad)
            )
            qacc_valid = torch.nn.functional.pad(
                torch.ones(
                    (self.worlds, qacc_width),
                    device=self.device,
                    dtype=self.dtype,
                ),
                (0, qacc_pad),
            )
        else:
            qacc_warmstart = torch.zeros(
                (self.worlds, 72),
                device=self.device,
                dtype=self.dtype,
            )
            qacc_valid = torch.zeros_like(qacc_warmstart)
        return (
            active_qpos.reshape(self.worlds, -1),
            active_qvel.reshape(self.worlds, -1),
            active_ctrl.reshape(self.worlds, -1),
            active_valid.reshape(self.worlds, -1),
            padded_actuator_state(normalized_velocity),
            padded_actuator_state(normalized_acceleration),
            padded_actuator_state(normalized_certified_ctrl),
            passive_qpos.reshape(self.worlds, -1),
            passive_qvel.reshape(self.worlds, -1),
            qacc_warmstart,
            qacc_valid,
        )

    def _observation(
        self, targets: Any, action: Any, finger_force: Any
    ) -> Any:
        body_ids = self._object_body_ids.index_select(0, self.object_index)
        body_position = self.backend.tensor("xpos")[self._rows, body_ids]
        position = self.backend.tensor("xipos")[self._rows, body_ids]
        quaternion = _quat_normalize(
            self.backend.tensor("xquat")[self._rows, body_ids]
        )
        velocity = self._object_velocity()
        tip_position = self.backend.tensor("xpos").index_select(
            1, self._tip_body_ids
        )
        tip_velocity = self.backend.tensor("cvel").index_select(
            1, self._tip_body_ids
        )[:, :, 3:]
        relative_q = _quat_mul(
            _quat_conjugate(self.goal_quat), quaternion
        )
        relative_angle = 2.0 * torch.acos(
            relative_q[:, :1].abs().clamp(0.0, 1.0)
        )
        phase = self._phase_index()
        _, _, lift_progress, action_progress, _ = (
            self._desired_object_pose()
        )
        (
            _,
            path_target,
            path_lead_progress,
            path_segment_index,
            path_segment_linear,
            path_segment_smooth,
            path_distance_progress,
            _,
        ) = self._translation_path_state()
        translating = self.task_index == contract.TASKS.index(
            contract.TASK_TRANSLATE
        )
        translation_mask = translating.to(self.dtype)[:, None]
        curriculum = torch.zeros_like(self.object_index)
        bottom, top, object_height = self._object_height_rows()
        (
            tip_height_fraction,
            grasp_plane_fraction,
            grasp_plane_compliant,
        ) = self._grasp_plane_metrics(targets)
        (
            actual_tip_height_fraction,
            actual_grasp_plane_fraction,
            actual_grasp_plane_compliant,
            _,
            actual_grasp_plane_within_bounds,
            _,
        ) = self._actual_grasp_plane_metrics(finger_force)
        _, plane_lower, plane_upper, _ = self._grasp_plane_contract_rows()
        low_level = self._low_level_proprioception()
        observation_fields = (
            torch.nn.functional.one_hot(
                self.object_index, len(contract.OBJECTS)
            ).to(self.dtype),
            torch.nn.functional.one_hot(
                self.task_index, len(contract.TASKS)
            ).to(self.dtype),
            torch.nn.functional.one_hot(curriculum, 1).to(self.dtype),
            torch.nn.functional.one_hot(
                phase, len(CONTROLLER_PHASES)
            ).to(self.dtype),
            torch.nn.functional.one_hot(
                self.path_shape_index,
                len(contract.TRANSLATION_PATH_SHAPES),
            ).to(self.dtype) * translation_mask,
            (
                self.path_waypoints_xy_m
                / contract.TRANSLATION_PATH_REGION_RADIUS_M
            ).reshape(self.worlds, -1),
            torch.stack(
                (
                    self.path_seed.to(self.dtype)
                    / float(contract.TRANSLATION_PATH_SEED_MAX),
                    self.path_step_m
                    / contract.TRANSLATION_PATH_MAX_STEP_M,
                    self.path_length_m
                    / contract.TRANSLATION_PATH_MAX_LENGTH_M,
                    self.action_reference_valid.to(self.dtype),
                ),
                dim=-1,
            ) * translation_mask,
            path_target * translation_mask,
            ((path_target - position) / 0.10) * translation_mask,
            torch.stack(
                (
                    path_lead_progress,
                    path_segment_index,
                    path_segment_linear,
                    path_segment_smooth,
                    path_distance_progress,
                ),
                dim=-1,
            ) * translation_mask,
            (self.goal_pos - position) / 0.10,
            torch.cat(
                (relative_q[:, 1:], relative_angle / math.pi),
                dim=-1,
            ),
            torch.cat(
                (velocity[:, :3] / 0.25, velocity[:, 3:] / 4.0),
                dim=-1,
            ),
            ((tip_position - position[:, None, :]) / 0.15).reshape(
                self.worlds, -1
            ),
            ((targets - tip_position) / 0.03).reshape(
                self.worlds, -1
            ),
            (
                self.tip_target_velocity_world_m_s
                / TIP_TARGET_SPEED_MAX_M_S
            ).reshape(self.worlds, -1),
            (
                self.tip_target_acceleration_world_m_s2
                / TIP_TARGET_ACCELERATION_MAX_M_S2
            ).reshape(self.worlds, -1),
            (tip_velocity / 0.50).reshape(self.worlds, -1),
            finger_force / 10.0,
            self.preload_trim_m / FORCE_FEEDBACK_MAX_M,
            self.position_feedback_m
            / torch.tensor(
                (
                    POSITION_FEEDBACK_XY_LIMIT_M,
                    POSITION_FEEDBACK_XY_LIMIT_M,
                    POSITION_FEEDBACK_Z_LIMIT_M,
                ),
                device=self.device,
                dtype=self.dtype,
            )[None, :],
            action,
            torch.stack((lift_progress, action_progress), dim=-1),
            body_position,
            position,
            quaternion,
            self.goal_pos,
            self.goal_quat,
            self.initial_body_pos,
            self.initial_com,
            self.initial_quat,
            self.action_start_com,
            self.action_start_quat,
            self.action_reference_valid.to(self.dtype)[:, None],
            self.action_start_cage_local.reshape(self.worlds, -1),
            self.action_start_action,
            self.action_start_position_feedback_m
            / torch.tensor(
                (
                    POSITION_FEEDBACK_XY_LIMIT_M,
                    POSITION_FEEDBACK_XY_LIMIT_M,
                    POSITION_FEEDBACK_Z_LIMIT_M,
                ),
                device=self.device,
                dtype=self.dtype,
            )[None, :],
            self.action_start_preload_trim_m / FORCE_FEEDBACK_MAX_M,
            tip_position.reshape(self.worlds, -1),
            targets.reshape(self.worlds, -1),
            torch.stack((bottom, top, object_height), dim=-1),
            torch.stack(
                (plane_lower, plane_upper),
                dim=-1,
            ),
            tip_height_fraction,
            grasp_plane_fraction[:, None],
            grasp_plane_compliant.to(self.dtype)[:, None],
            actual_tip_height_fraction,
            actual_grasp_plane_fraction[:, None],
            actual_grasp_plane_within_bounds.to(self.dtype)[:, None],
            actual_grasp_plane_compliant.to(self.dtype)[:, None],
            (
                self.template_index.to(self.dtype)
                / max(contract.RESET_TEMPLATES_PER_OBJECT - 1, 1)
            )[:, None],
            torch.nn.functional.one_hot(
                self.source_region, 1
            ).to(self.dtype),
            (
                self.control_step.to(self.dtype)
                / float(self.config.episode_length)
            ).clamp(0.0, 1.0)[:, None],
            torch.stack(
                (
                    self.translation_entry_stable_count.to(self.dtype)
                    / float(EXPERT_STABLE_TICKS),
                    self.translation_route_admitted.to(self.dtype),
                    self.translation_entry_certificate_latched.to(self.dtype),
                    self.translation_entry_tip_height_valid.to(self.dtype),
                    self.translation_entry_lifted.to(self.dtype),
                    self.translation_entry_pose_valid.to(self.dtype),
                    self.expert_contact_count.to(self.dtype)
                    / float(len(contract.TIP_ORDER)),
                    self.expert_four_tip_contact.to(self.dtype),
                    self.expert_object_linear_speed_m_s
                    / EXPERT_OBJECT_LINEAR_SPEED_MAX_M_S,
                    self.expert_object_angular_speed_rad_s
                    / EXPERT_OBJECT_ANGULAR_SPEED_MAX_RAD_S,
                    self.terminal_stable_count.to(self.dtype)
                    / float(EXPERT_STABLE_TICKS),
                    self.terminal_stable_certified.to(self.dtype),
                ),
                dim=-1,
            ),
            self.hold_reference_valid.to(self.dtype)[:, None],
            self.hold_frozen_cage_world.reshape(self.worlds, -1),
            *low_level,
        )
        for (name, width), value in zip(
            OBSERVATION_SCHEMA, observation_fields
        ):
            if value.ndim != 2 or value.shape[1] != width:
                raise RuntimeError(
                    f"observation field {name!r} has shape "
                    f"{tuple(value.shape)}, expected [worlds,{width}]"
                )
        observation = torch.cat(observation_fields, dim=-1)
        if observation.shape[1] != OBSERVATION_DIM:
            raise RuntimeError(
                f"observation width is {observation.shape[1]}, "
                f"expected {OBSERVATION_DIM}"
            )
        self._last_raw_observation_finite.copy_(
            torch.isfinite(observation).all(dim=1)
        )
        return torch.nan_to_num(
            observation, nan=0.0, posinf=10.0, neginf=-10.0
        )

    def snapshot(self) -> dict[str, Any]:
        (
            body_position,
            position,
            quaternion,
            estimated_bottom,
            live_body_position,
            live_position,
            live_quaternion,
            live_estimated_bottom,
        ) = self._reported_object_drop_pose()
        euler = _relative_euler_xyz_deg(self.goal_quat, quaternion)
        from_initial = _relative_euler_xyz_deg(
            self.initial_quat, quaternion
        )
        from_action = _relative_euler_xyz_deg(
            self.action_start_quat, quaternion
        )
        (
            tip_height_fraction,
            grasp_plane_fraction,
            grasp_plane_compliant,
        ) = self._grasp_plane_metrics(self._last_targets)
        (
            actual_tip_height_fraction,
            actual_grasp_plane_fraction,
            actual_grasp_plane_compliant,
            actual_grasp_plane_relevant,
            actual_grasp_plane_in_band,
            actual_grasp_plane_finite,
        ) = self._actual_grasp_plane_metrics(self._last_contact_force)
        _, _, object_height = self._object_height_rows()
        nominal_fraction, plane_lower, plane_upper, plane_learned = (
            self._grasp_plane_contract_rows()
        )
        requested_tip_height_fraction = self._requested_tip_height_fractions(
            self.previous_action
        )
        fixed_physical_tolerance_m = torch.where(
            plane_learned,
            torch.zeros_like(object_height),
            torch.full_like(
                object_height,
                contract.FIXED_GRASP_PLANE_PHYSICAL_TOLERANCE_M,
            ),
        )
        (
            _,
            path_target,
            path_lead_progress,
            path_segment_index,
            path_segment_linear,
            path_segment_smooth,
            path_distance_progress,
            path_following,
        ) = self._translation_path_state()
        path_count = self.path_tracking_sample_count.to(self.dtype)
        path_rmse = torch.where(
            path_count > 0.0,
            torch.sqrt(
                self.path_tracking_squared_error_sum
                / path_count.clamp_min(1.0)
            ),
            torch.zeros_like(path_count),
        )
        path_tracking_certified = _path_tracking_certified(
            self.task_index,
            self.path_tracking_sample_count,
            self.path_tracking_axis_abs_max_m,
            self.path_schedule_complete,
            self.path_schedule_limit_valid,
        )
        final_position_error_xyz = self.goal_pos - position
        final_position_error_l2 = final_position_error_xyz.norm(dim=-1)
        desired_final_displacement = self.goal_pos - self.action_start_com
        actual_final_displacement = position - self.action_start_com
        translating = self.task_index == contract.TASKS.index(
            contract.TASK_TRANSLATE
        )
        path_target_error = torch.where(
            translating[:, None],
            path_target - position,
            torch.zeros_like(path_target),
        )
        return {
            "object_index": self.object_index,
            "template_index": self.template_index,
            "position": position,
            "body_position": body_position,
            "quaternion": quaternion,
            "live_position": live_position,
            "live_body_position": live_body_position,
            "live_quaternion": live_quaternion,
            "goal_position": self.goal_pos,
            "goal_quaternion": self.goal_quat,
            "rotation_angle_deg": self.rotation_angle_deg,
            "position_error_xyz": final_position_error_xyz,
            "position_error": final_position_error_l2,
            "final_position_error_xyz_m": final_position_error_xyz,
            "final_position_error_l2_m": final_position_error_l2,
            "desired_final_displacement_xyz_m": desired_final_displacement,
            "desired_final_displacement_l2_m": (
                desired_final_displacement.norm(dim=-1)
            ),
            "actual_final_displacement_xyz_m": actual_final_displacement,
            "actual_final_displacement_l2_m": (
                actual_final_displacement.norm(dim=-1)
            ),
            # Sign convention is desired minus actual.
            "final_displacement_error_xyz_m": (
                desired_final_displacement - actual_final_displacement
            ),
            "final_displacement_error_l2_m": (
                desired_final_displacement - actual_final_displacement
            ).norm(dim=-1),
            "euler_error_xyz": euler,
            "euler_error": euler.norm(dim=-1),
            "final_euler_error_xyz_deg": euler,
            "final_euler_error_l2_deg": euler.norm(dim=-1),
            "path_seed": self.path_seed,
            "path_shape_index": self.path_shape_index,
            "path_step_m": self.path_step_m,
            "path_length_m": self.path_length_m,
            "path_waypoints_xy_m": self.path_waypoints_xy_m,
            "path_current_target_world_m": path_target,
            "path_current_target_error_xyz_m": path_target_error,
            "path_tracking_current_error_m": (
                self.path_tracking_current_error_m
            ),
            "path_tracking_rmse_m": path_rmse,
            "path_tracking_max_error_m": self.path_tracking_max_error_m,
            "path_tracking_axis_abs_max_m": (
                self.path_tracking_axis_abs_max_m
            ),
            "path_tracking_sample_count": self.path_tracking_sample_count,
            "path_schedule_progress": self.path_schedule_progress,
            "path_schedule_limit_valid": self.path_schedule_limit_valid,
            "path_schedule_complete": self.path_schedule_complete,
            "path_schedule_max_speed_m_s": (
                self.path_schedule_max_speed_m_s
            ),
            "path_schedule_max_acceleration_m_s2": (
                self.path_schedule_max_acceleration_m_s2
            ),
            "path_tracking_first_violation_sample_tick": (
                self.path_tracking_first_violation_sample_tick
            ),
            "path_tracking_first_violation_control_tick": (
                self.path_tracking_first_violation_control_tick
            ),
            "path_tracking_first_violation_axis_index": (
                self.path_tracking_first_violation_axis_index
            ),
            "path_tracking_first_violation_error_xyz_abs_m": (
                self.path_tracking_first_violation_error_xyz_abs_m
            ),
            "path_tracking_certified": path_tracking_certified,
            "path_following_active": path_following,
            "path_progress": torch.stack(
                (
                    path_lead_progress,
                    path_segment_index,
                    path_segment_linear,
                    path_segment_smooth,
                    path_distance_progress,
                ),
                dim=-1,
            ),
            "rotation_from_initial_xyz": from_initial,
            "rotation_from_action_reference_xyz": from_action,
            "velocity": torch.where(
                self.object_dropped[:, None],
                torch.zeros_like(self._object_velocity()),
                self._object_velocity(),
            ),
            "contact_forces": self._last_contact_force,
            "tip_object_contact_raw_pair": (
                self._last_tip_object_contact_raw_pair
            ),
            # Historical pair-present semantics: exact hard-tip/current-fruit
            # identity plus an active EFC row, independent of force threshold.
            "tip_object_contact_pair": self._last_tip_object_contact_pair,
            "tip_object_contact_force_qualified_pair": (
                self._last_tip_object_contact_force_qualified_pair
            ),
            "contact_force_decode_finite": self._last_force_decode_finite,
            "position_feedback_m": self.position_feedback_m,
            "action_start_position_feedback_m": (
                self.action_start_position_feedback_m
            ),
            "ctrl_velocity": self.ctrl_velocity,
            "ctrl_acceleration": self.ctrl_acceleration,
            "last_certified_ctrl": self.last_certified_ctrl,
            "last_certified_tip_targets_world_m": (
                self.last_certified_tip_targets
            ),
            "tip_target_velocity_world_m_s": (
                self.tip_target_velocity_world_m_s
            ),
            "tip_target_acceleration_world_m_s2": (
                self.tip_target_acceleration_world_m_s2
            ),
            "tip_target_jerk_world_m_s3": (
                self.tip_target_jerk_world_m_s3
            ),
            "tip_target_slew_clip_count": self.tip_target_slew_clip_count,
            "tip_target_certificate_fallback_count": (
                self.tip_target_certificate_fallback_count
            ),
            "action": self.previous_action,
            "requested_tip_height_fraction": (
                requested_tip_height_fraction
            ),
            "tip_height_fraction": tip_height_fraction,
            "commanded_tip_height_fraction": tip_height_fraction,
            "tip_height_m": (
                tip_height_fraction
                - nominal_fraction[:, None]
            ) * object_height[:, None],
            "grasp_plane_fraction": grasp_plane_fraction,
            "commanded_grasp_plane_fraction": grasp_plane_fraction,
            "commanded_grasp_plane_error_m": (
                grasp_plane_fraction - nominal_fraction
            ) * object_height,
            "grasp_plane_compliant": grasp_plane_compliant,
            "commanded_grasp_plane_compliant": grasp_plane_compliant,
            "actual_tip_height_fraction": actual_tip_height_fraction,
            "actual_tip_height_m": (
                actual_tip_height_fraction
                - nominal_fraction[:, None]
            ) * object_height[:, None],
            "actual_grasp_plane_fraction": (
                actual_grasp_plane_fraction
            ),
            "actual_grasp_plane_error_m": (
                actual_grasp_plane_fraction - nominal_fraction
            ) * object_height,
            "actual_grasp_plane_compliant": (
                actual_grasp_plane_compliant
            ),
            "actual_grasp_plane_relevant": (
                actual_grasp_plane_relevant
            ),
            "actual_grasp_plane_within_bounds": (
                actual_grasp_plane_in_band
            ),
            "actual_grasp_plane_in_band": actual_grasp_plane_in_band,
            "actual_grasp_plane_finite": actual_grasp_plane_finite,
            "object_height_m": object_height,
            "grasp_plane_nominal_fraction": nominal_fraction,
            "grasp_plane_fraction_limits": torch.stack(
                (plane_lower, plane_upper), dim=-1
            ),
            "grasp_plane_is_learned": plane_learned,
            "fixed_grasp_plane_physical_tolerance_m": (
                fixed_physical_tolerance_m
            ),
            "tip_force_scale": (
                contract.FORCE_SCALE_NOMINAL
                + self.previous_action[:, contract.FORCE_SCALE_SLICE]
                * (contract.FORCE_SCALE_MAX - contract.FORCE_SCALE_MIN)
                / 2.0
            ),
            "success": self.success,
            "object_dropped": self.object_dropped,
            "object_drop_new": self.object_drop_new,
            "object_drop_first_control_step": (
                self.object_drop_first_control_step
            ),
            "object_drop_first_position": self.object_drop_first_position,
            "object_drop_first_body_position": (
                self.object_drop_first_body_position
            ),
            "object_drop_first_quaternion": (
                self.object_drop_first_quaternion
            ),
            "object_drop_first_bottom_z_m": (
                self.object_drop_first_bottom_z_m
            ),
            "object_estimated_bottom_z_m": estimated_bottom,
            "object_live_estimated_bottom_z_m": live_estimated_bottom,
            "object_drop_threshold_z_m": torch.full_like(
                estimated_bottom, contract.OBJECT_DROP_THRESHOLD_Z_M
            ),
            "object_drop_quarantined": self.object_dropped & (
                self.control_step > self.object_drop_first_control_step
            ),
            "action_reference_position": self.action_start_com,
            "action_reference_quaternion": self.action_start_quat,
            "action_reference_valid": self.action_reference_valid,
            "prelift_contact_required_stable_ticks": torch.full_like(
                self.prelift_contact_stable_count,
                PRELIFT_CONTACT_STABLE_TICKS,
            ),
            "prelift_contact_stage_names": PRELIFT_CONTACT_STAGES,
            "prelift_contact_stage_index": (
                self.prelift_contact_stage_index
            ),
            "prelift_contact_stable_count": (
                self.prelift_contact_stable_count
            ),
            "prelift_contact_max_stable_count": (
                self.prelift_contact_max_stable_count
            ),
            "prelift_contact_gate_latched": (
                self.prelift_contact_gate_latched
            ),
            "prelift_contact_gate_first_control_step": (
                self.prelift_contact_gate_first_control_step
            ),
            "prelift_contact_count_current": (
                self.prelift_contact_count_current
            ),
            "prelift_tip_raw_pair_current": (
                self.prelift_tip_raw_pair_current
            ),
            "prelift_tip_active_pair_current": (
                self.prelift_tip_active_pair_current
            ),
            "prelift_tip_contact_qualified_current": (
                self.prelift_tip_contact_qualified_current
            ),
            "prelift_stage_observation_count": (
                self.prelift_stage_observation_count
            ),
            "prelift_stage_qualified_count": (
                self.prelift_stage_qualified_count
            ),
            "prelift_stage_failure_count": (
                self.prelift_stage_failure_count
            ),
            "prelift_stage_max_contact_count": (
                self.prelift_stage_max_contact_count
            ),
            "prelift_stage_first_failure_control_step": (
                self.prelift_stage_first_failure_control_step
            ),
            "prelift_stage_last_failure_control_step": (
                self.prelift_stage_last_failure_control_step
            ),
            "prelift_tip_pair_missing_count_by_stage": (
                self.prelift_tip_pair_missing_count_by_stage
            ),
            "prelift_tip_raw_pair_missing_count_by_stage": (
                self.prelift_tip_raw_pair_missing_count_by_stage
            ),
            "prelift_tip_efc_inactive_count_by_stage": (
                self.prelift_tip_efc_inactive_count_by_stage
            ),
            "prelift_tip_force_failure_count_by_stage": (
                self.prelift_tip_force_failure_count_by_stage
            ),
            "prelift_lift_boundary_contact_valid": (
                self.prelift_lift_boundary_contact_valid
            ),
            "prelift_lift_boundary_tip_raw_pair": (
                self.prelift_lift_boundary_tip_raw_pair
            ),
            "prelift_lift_boundary_tip_active_pair": (
                self.prelift_lift_boundary_tip_active_pair
            ),
            "prelift_lift_boundary_tip_contact_qualified": (
                self.prelift_lift_boundary_tip_contact_qualified
            ),
            "prelift_lift_boundary_contact_force_n": (
                self.prelift_lift_boundary_contact_force_n
            ),
            "prelift_lift_permitted": self.prelift_lift_permitted,
            "prelift_lift_reference_valid": (
                self.prelift_lift_reference_valid
            ),
            "prelift_lift_start_body_position": (
                self.prelift_lift_start_body_pos
            ),
            "prelift_lift_start_quaternion": (
                self.prelift_lift_start_quat
            ),
            "hold_reference_valid": self.hold_reference_valid,
            "hold_frozen_ctrl": self.hold_frozen_ctrl,
            "hold_frozen_cage_world": self.hold_frozen_cage_world,
            "ik_solve_feasible": self._last_ik_solve_feasible,
            "ik_solve_fallback_count": self.ik_solve_fallback_count,
            "ik_trust_region_projection_count": (
                self.ik_trust_region_projection_count
            ),
            "ik_trust_region_projection_count_by_stage": (
                self.ik_trust_region_projection_count_by_stage
            ),
            "ik_trust_stage_names": IK_TRUST_STAGES,
            "ik_cage_target_error_limit_m": torch.full_like(
                self.expert_object_linear_speed_m_s,
                IK_CAGE_TARGET_ERROR_MAX_M,
            ),
            "ik_cage_projection_count": self.ik_cage_projection_count,
            "ik_cage_projection_count_by_stage": (
                self.ik_cage_projection_count_by_stage
            ),
            "ik_cage_projection_scale_min": (
                self.ik_cage_projection_scale_min
            ),
            "ik_cage_projection_scale_mean": torch.where(
                self.ik_cage_projection_sample_count > 0,
                self.ik_cage_projection_scale_sum
                / self.ik_cage_projection_sample_count.to(self.dtype)
                .clamp_min(1.0),
                torch.ones_like(self.ik_cage_projection_scale_sum),
            ),
            "ik_cage_projection_sample_count": (
                self.ik_cage_projection_sample_count
            ),
            "translation_entry_stable_count": (
                self.translation_entry_stable_count
            ),
            "translation_entry_max_stable_count": (
                self.translation_entry_max_stable_count
            ),
            "translation_entry_certificate_latched": (
                self.translation_entry_certificate_latched
            ),
            "translation_entry_certificate_first_control_step": (
                self.translation_entry_certificate_first_control_step
            ),
            "translation_route_admitted": self.translation_route_admitted,
            "translation_entry_route_start_safety_valid": (
                self.translation_entry_route_start_safety_valid
            ),
            "translation_entry_gate_current": (
                self.translation_entry_gate_current
            ),
            "translation_entry_tip_height_valid": (
                self.translation_entry_tip_height_valid
            ),
            "translation_entry_lifted": self.translation_entry_lifted,
            "translation_entry_pose_valid": (
                self.translation_entry_pose_valid
            ),
            "translation_entry_start_position_valid": (
                self.translation_entry_start_position_valid
            ),
            "translation_entry_start_orientation_valid": (
                self.translation_entry_start_orientation_valid
            ),
            "translation_entry_linear_speed_valid": (
                self.translation_entry_linear_speed_valid
            ),
            "translation_entry_angular_speed_valid": (
                self.translation_entry_angular_speed_valid
            ),
            "translation_entry_predicate_names": (
                TRANSLATION_ENTRY_PREDICATES
            ),
            "translation_entry_predicate_current": (
                self.translation_entry_predicate_current
            ),
            "translation_entry_failure_count_by_predicate": (
                self.translation_entry_failure_count_by_predicate
            ),
            "translation_entry_first_failure_control_step": (
                self.translation_entry_first_failure_control_step
            ),
            "translation_entry_last_failure_control_step": (
                self.translation_entry_last_failure_control_step
            ),
            "translation_entry_tip_height_fraction": (
                self.translation_entry_tip_height_fraction
            ),
            "translation_entry_tip_object_pair": (
                self.translation_entry_tip_object_pair
            ),
            "translation_entry_tip_object_raw_pair": (
                self.translation_entry_tip_object_raw_pair
            ),
            "translation_entry_tip_object_active_pair": (
                self.translation_entry_tip_object_active_pair
            ),
            "translation_entry_tip_contact_qualified": (
                self.translation_entry_tip_contact_qualified
            ),
            "translation_entry_tip_contact_force_n": (
                self.translation_entry_tip_contact_force_n
            ),
            "translation_entry_tip_force_valid": (
                self.translation_entry_tip_force_valid
            ),
            "translation_entry_tip_height_failure_count": (
                self.translation_entry_tip_height_failure_count
            ),
            "translation_entry_tip_pair_missing_count": (
                self.translation_entry_tip_pair_missing_count
            ),
            "translation_entry_tip_raw_pair_missing_count": (
                self.translation_entry_tip_raw_pair_missing_count
            ),
            "translation_entry_tip_efc_inactive_count": (
                self.translation_entry_tip_efc_inactive_count
            ),
            "translation_entry_tip_force_failure_count": (
                self.translation_entry_tip_force_failure_count
            ),
            "translation_entry_position_error_xyz_abs_m": (
                self.translation_entry_position_error_xyz_abs_m
            ),
            "translation_entry_euler_error_xyz_abs_deg": (
                self.translation_entry_euler_error_xyz_abs_deg
            ),
            "translation_entry_estimated_bottom_z_m": (
                self.translation_entry_estimated_bottom_z_m
            ),
            "translation_entry_object_linear_speed_m_s": (
                self.translation_entry_object_linear_speed_m_s
            ),
            "translation_entry_object_angular_speed_rad_s": (
                self.translation_entry_object_angular_speed_rad_s
            ),
            "expert_tip_object_contact": self.expert_tip_object_contact,
            "expert_contact_count": self.expert_contact_count,
            "expert_four_tip_contact": (
                self.expert_four_tip_contact
            ),
            "expert_object_linear_speed_m_s": (
                self.expert_object_linear_speed_m_s
            ),
            "expert_object_angular_speed_rad_s": (
                self.expert_object_angular_speed_rad_s
            ),
            "terminal_stable_count": self.terminal_stable_count,
            "terminal_stable_certified": self.terminal_stable_certified,
            "terminal_stability_gate_current": (
                self.terminal_stability_gate_current
            ),
            "cage_orientation_error_rad": self.cage_orientation_error_rad,
            "position_feedback_slew_clip_count": (
                self.position_feedback_slew_clip_count
            ),
            "preload_trim_slew_clip_count": (
                self.preload_trim_slew_clip_count
            ),
            "ctrl_slew_clip_count": self.ctrl_slew_clip_count,
            "ctrl_certificate_fallback_count": (
                self.ctrl_certificate_fallback_count
            ),
        }

    def _info(
        self,
        *,
        tip_error: Any,
        force_imbalance: Any,
        euler_xyz: Any,
        transition_finite: Any,
    ) -> dict[str, Any]:
        result = self.snapshot()
        result.update(
            {
                "object_index": self.object_index,
                "template_index": self.template_index,
                "task_index": self.task_index,
                "source_region": self.source_region,
                "path_seed": self.path_seed,
                "path_shape_index": self.path_shape_index,
                "rotation_angle_deg": self.rotation_angle_deg,
                "control_step": self.control_step,
                "phase_index": self._phase_index(),
                "position_error_m": self.position_error_m,
                "euler_l2_error_deg": self.euler_l2_error_deg,
                "euler_error_xyz_deg": euler_xyz,
                "tip_error_m": tip_error,
                "force_imbalance": force_imbalance,
                "preload_trim_m": self.preload_trim_m,
                "time_out": self.time_out,
                "done": self.done,
                "action_start_cage_local": self.action_start_cage_local,
                "transition_finite": transition_finite,
            }
        )
        return result

    def step(self, action: Any) -> StepBatch:
        with self.torch.no_grad():
            action = self.torch.as_tensor(
                action, device=self.device, dtype=self.dtype
            )
            if action.ndim == 1 and self.worlds == 1:
                action = action[None, :]
            if tuple(action.shape) != (self.worlds, contract.ACTION_DIM):
                raise ValueError(
                    f"action must be [{self.worlds},{contract.ACTION_DIM}]"
                )
            action = torch.nan_to_num(
                action, nan=0.0, posinf=1.0, neginf=-1.0
            )
            absorbing = self.object_dropped.clone()
            self.object_drop_new.zero_()
            self._park_inactive_objects()
            old_previous = self.previous_action.clone()
            trim_at_command = self.preload_trim_m.clone()
            filtered, frozen, used = self._filter_action(action)
            filtered = torch.where(
                absorbing[:, None], self.filtered_action, filtered
            )
            frozen = torch.where(
                absorbing[:, None], self.frozen_action, frozen
            )
            used = torch.where(
                absorbing[:, None], self.previous_action, used
            )
            old_position_feedback = self.position_feedback_m.clone()
            self._update_position_feedback()
            self.position_feedback_m = torch.where(
                absorbing[:, None],
                old_position_feedback,
                self.position_feedback_m,
            )
            requested_targets = self._tip_targets(used)
            commanded_targets, hold_frozen = (
                self._hold_command_targets(requested_targets)
            )
            commanded_targets = self._project_grasp_plane(
                commanded_targets
            )
            commanded_targets = self._rate_limit_tip_targets(
                commanded_targets, ~absorbing
            )
            current_tip_position = self.backend.tensor("xpos").index_select(
                1, self._tip_body_ids
            )
            commanded_targets = torch.where(
                absorbing[:, None, None],
                current_tip_position,
                commanded_targets,
            )
            (
                instantaneous_target,
                _,
                _,
                _,
                instantaneous_action_active,
            ) = self._desired_object_pose()
            (
                path_offset_before_step,
                _,
                _,
                _,
                _,
                _,
                path_progress_before_step,
                path_following_before_step,
            ) = self._translation_path_state()
            ctrl = self._ik_control(
                commanded_targets, active_worlds=~absorbing
            )
            ctrl = self._hold_control(ctrl, hold_frozen)
            ctrl = torch.where(
                absorbing[:, None], self.last_certified_ctrl, ctrl
            )
            ctrl = self._rate_limit_control(
                ctrl, hold_frozen | absorbing
            )
            self._quarantine_dropped_active_object(absorbing)
            prephysics_finite = (
                torch.isfinite(commanded_targets).all(dim=(1, 2))
                & torch.isfinite(ctrl).all(dim=1)
                & torch.isfinite(self.backend.tensor("qpos")).all(dim=1)
                & torch.isfinite(self.backend.tensor("qvel")).all(dim=1)
                & (ctrl >= self._ctrl_low[None, :]).all(dim=1)
                & (ctrl <= self._ctrl_high[None, :]).all(dim=1)
            )
            if self.backend.has_field("qacc_warmstart"):
                prephysics_finite &= torch.isfinite(
                    self.backend.tensor("qacc_warmstart")
                ).all(dim=1)
            if not bool(prephysics_finite.all()):
                raise FloatingPointError(
                    "pre-physics IK/control certificate failed for "
                    f"{int((~prephysics_finite).sum().detach().cpu())} world(s)"
                )
            self.backend.tensor("ctrl").copy_(ctrl)
            self.backend.step_control()
            active_body_ids = self._object_body_ids.index_select(
                0, self.object_index
            )
            post_body_position = self.backend.tensor("xpos")[
                self._rows, active_body_ids
            ]
            post_com_position = self.backend.tensor("xipos")[
                self._rows, active_body_ids
            ]
            instantaneous_position = torch.where(
                instantaneous_action_active[:, None],
                post_com_position,
                post_body_position,
            )
            instantaneous_tracking_error_xyz_abs = (
                instantaneous_target - instantaneous_position
            ).abs()
            instantaneous_tracking_error = (
                instantaneous_tracking_error_xyz_abs.norm(dim=-1)
            )
            translating = self.task_index == contract.TASKS.index(
                contract.TASK_TRANSLATE
            )
            instantaneous_tracking_error = torch.where(
                translating & (~absorbing),
                instantaneous_tracking_error,
                self.path_tracking_current_error_m,
            )
            tracking_sample = (
                path_following_before_step & translating & (~absorbing)
            )
            self._update_path_schedule_certificate(
                tracking_sample,
                path_progress_before_step,
                path_offset_before_step[:, :2],
            )
            self._record_path_tracking_first_violation(
                tracking_sample,
                instantaneous_tracking_error_xyz_abs,
            )
            self.path_tracking_squared_error_sum.add_(
                torch.where(
                    tracking_sample,
                    instantaneous_tracking_error.square(),
                    torch.zeros_like(instantaneous_tracking_error),
                )
            )
            self.path_tracking_max_error_m.copy_(
                torch.where(
                    tracking_sample,
                    torch.maximum(
                        self.path_tracking_max_error_m,
                        instantaneous_tracking_error,
                    ),
                    self.path_tracking_max_error_m,
                )
            )
            self.path_tracking_axis_abs_max_m.copy_(
                torch.where(
                    tracking_sample[:, None],
                    torch.maximum(
                        self.path_tracking_axis_abs_max_m,
                        instantaneous_tracking_error_xyz_abs,
                    ),
                    self.path_tracking_axis_abs_max_m,
                )
            )
            self.path_tracking_sample_count.add_(tracking_sample.to(torch.long))
            self.path_tracking_current_error_m.copy_(
                instantaneous_tracking_error
            )
            force = self._finger_contact_force()
            next_trim = self._updated_preload_trim(used, force)
            next_step = self.control_step + 1
            self.filtered_action = filtered
            self.frozen_action = frozen
            self.previous_action = used
            self.preload_trim_m = torch.where(
                absorbing[:, None], self.preload_trim_m, next_trim
            )
            self.control_step = next_step
            self._update_prelift_contact_gate(force)
            action_start = (
                self.config.plan_ticks
                + self.config.close_ticks
                + self.config.settle_ticks
                + self.config.lift_ticks
                + self.config.hold_ticks
            )
            capture = (
                (next_step == action_start)
                & (~self.action_reference_valid)
                & self.prelift_lift_permitted
                & self.prelift_lift_reference_valid
                & (~absorbing)
            )
            hold_capture = (
                _hold_reference_capture(
                    next_step,
                    self.hold_reference_valid,
                    self.config,
                )
                & self.prelift_lift_permitted
                & self.prelift_lift_reference_valid
                & (~absorbing)
            )
            self._capture_hold_reference(
                hold_capture,
                commanded_targets,
            )
            self._capture_action_reference(
                capture,
                commanded_targets,
                used,
                trim_at_command,
            )
            self._update_translation_entry_gate(force)
            self.path_current_target.copy_(
                self._translation_path_state()[1]
            )
            # The observation and expert gates report the last physically
            # certified Cartesian command.  Re-running the stateful limiter
            # for an observation preview would consume a second v/a/j tick.
            next_targets = self.last_certified_tip_targets
            position_error, euler_l2, success, euler_xyz = (
                self._pose_metrics()
            )
            cached_euler_xyz = _relative_euler_xyz_deg(
                self.goal_quat, self.object_drop_first_quaternion
            )
            position_error = torch.where(
                absorbing, self.position_error_m, position_error
            )
            euler_l2 = torch.where(
                absorbing, self.euler_l2_error_deg, euler_l2
            )
            euler_xyz = torch.where(
                absorbing[:, None], cached_euler_xyz, euler_xyz
            )
            (
                _,
                _,
                actual_plane_compliant,
                _,
                _,
                _,
            ) = self._actual_grasp_plane_metrics(force)
            # The object-specific physical hard-tip plane is a task invariant,
            # not the excluded anti-support/pinch heuristic.  Every active
            # fruit uses the learned 40--60% hard-tip height band.
            success = (
                success
                & actual_plane_compliant
                & self.prelift_lift_permitted
                & self.prelift_lift_reference_valid
                & self.action_reference_valid
            )
            success = success & self._update_terminal_stability_gate(
                pose_success=success,
                actual_plane_compliant=actual_plane_compliant,
                finger_force=force,
            )
            tip_position = self.backend.tensor("xpos").index_select(
                1, self._tip_body_ids
            )
            tip_error = (next_targets - tip_position).norm(
                dim=-1
            ).mean(dim=-1)
            action_delta = used - old_previous
            force_imbalance = force.std(dim=-1, unbiased=False) / (
                1.0 + force.mean(dim=-1)
            )
            dense_position_error = torch.where(
                translating, instantaneous_tracking_error, position_error
            )
            dense = (
                1.5 * torch.exp(-dense_position_error / 0.030)
                + 0.8 * torch.exp(-euler_l2 / 15.0)
                + 1.0 * torch.exp(-tip_error / 0.012)
                - 0.002 * (used * used).sum(dim=-1)
                - 0.010 * (action_delta * action_delta).sum(dim=-1)
                - 0.20 * force_imbalance
            )
            final_tick = next_step >= self.config.episode_length
            raw_reward = dense + torch.where(
                final_tick,
                torch.where(
                    success,
                    torch.full_like(dense, 20.0),
                    torch.full_like(dense, -10.0),
                ),
                torch.zeros_like(dense),
            )
            self.position_error_m = position_error
            self.euler_l2_error_deg = euler_l2
            self._last_contact_force = torch.where(
                absorbing[:, None], self._last_contact_force, force
            )
            self._last_targets = torch.where(
                absorbing[:, None, None], self._last_targets, next_targets
            )
            raw_observation = self._observation(
                next_targets, used, force
            )
            (
                drop_body_position,
                drop_position,
                drop_quaternion,
                estimated_bottom,
            ) = self._active_object_drop_measurement()
            drop_measurement_finite = (
                torch.isfinite(drop_body_position).all(dim=1)
                & torch.isfinite(drop_position).all(dim=1)
                & torch.isfinite(drop_quaternion).all(dim=1)
                & torch.isfinite(estimated_bottom)
            )
            finite = (
                torch.isfinite(self.backend.tensor("qpos")).all(dim=1)
                & torch.isfinite(self.backend.tensor("qvel")).all(dim=1)
                & torch.isfinite(position_error)
                & torch.isfinite(instantaneous_tracking_error)
                & torch.isfinite(euler_l2)
                & torch.isfinite(next_targets).all(dim=(1, 2))
                & torch.isfinite(force).all(dim=1)
                & self._last_force_decode_finite
                & torch.isfinite(tip_error)
                & torch.isfinite(force_imbalance)
                & torch.isfinite(raw_reward)
                & self._last_raw_observation_finite
                & drop_measurement_finite
            )
            if self.backend.has_field("qacc_warmstart"):
                finite &= torch.isfinite(
                    self.backend.tensor("qacc_warmstart")
                ).all(dim=1)
            object_drop_new, object_dropped, reward = (
                _object_drop_transition(
                    absorbing, finite, estimated_bottom, raw_reward
                )
            )
            self.object_drop_first_control_step = torch.where(
                object_drop_new,
                next_step,
                self.object_drop_first_control_step,
            )
            self.object_drop_first_position = torch.where(
                object_drop_new[:, None],
                drop_position,
                self.object_drop_first_position,
            )
            self.object_drop_first_body_position = torch.where(
                object_drop_new[:, None],
                drop_body_position,
                self.object_drop_first_body_position,
            )
            self.object_drop_first_quaternion = torch.where(
                object_drop_new[:, None],
                drop_quaternion,
                self.object_drop_first_quaternion,
            )
            self.object_drop_first_bottom_z_m = torch.where(
                object_drop_new,
                estimated_bottom,
                self.object_drop_first_bottom_z_m,
            )
            self._object_drop_terminal_observation = torch.where(
                object_drop_new[:, None],
                raw_observation,
                self._object_drop_terminal_observation,
            )
            self._observation_tensor = torch.where(
                object_dropped[:, None],
                self._object_drop_terminal_observation,
                raw_observation,
            )
            success = success & finite & (~object_dropped)
            done = final_tick | (~finite) | object_dropped
            self.success = success
            self.time_out = final_tick
            self.done = done
            self.object_dropped = object_dropped
            self.object_drop_new = object_drop_new
            self.backend.accumulate_external_health(
                position_error,
                euler_l2,
                next_targets,
                force,
                tip_error,
                force_imbalance,
                raw_reward,
                self._observation_tensor,
                torch.where(
                    self._last_raw_observation_finite
                    & self._last_force_decode_finite,
                    torch.zeros_like(raw_reward),
                    torch.full_like(raw_reward, torch.nan),
                ),
            )
            info = self._info(
                tip_error=tip_error,
                force_imbalance=force_imbalance,
                euler_xyz=euler_xyz,
                transition_finite=finite,
            )
            return StepBatch(
                observation=self._observation_tensor,
                reward=reward,
                done=done,
                info=info,
            )

    def close(self) -> None:
        self.backend.close()


def make_env(
    curriculum_phase: str,
    *,
    config: HPCConfig = DEFAULT_CONFIG,
    residual_enabled: bool = True,
    platform_spec: PlatformSpec | None = None,
    worlds: int | None = None,
    capacity_num_envs: int | None = None,
    device: str = "cuda:0",
    require_linux: bool = True,
    require_cuda: bool = True,
    validate_official_config: bool = True,
    **kwargs: Any,
) -> Any:
    resolved_platform = platform_spec or build_platform_spec(config.task)
    if hasattr(resolved_platform, "shards"):
        from .sharded_env import ShardedMJWarpEnv

        return ShardedMJWarpEnv(
            curriculum_phase=curriculum_phase,
            config=config,
            residual_enabled=residual_enabled,
            platform_spec=resolved_platform,
            worlds=worlds,
            capacity_num_envs=capacity_num_envs,
            device=device,
            require_linux=require_linux,
            require_cuda=require_cuda,
            validate_official_config=validate_official_config,
            **kwargs,
        )
    return _construct_fail_closed(
        MJWarpIKResidualEnv,
        curriculum_phase=curriculum_phase,
        config=config,
        residual_enabled=residual_enabled,
        platform_spec=resolved_platform,
        worlds=worlds,
        capacity_num_envs=capacity_num_envs,
        device=device,
        require_linux=require_linux,
        require_cuda=require_cuda,
        validate_official_config=validate_official_config,
        **kwargs,
    )


__all__ = [
    "CONTROLLER_PHASES",
    "MJWarpIKResidualEnv",
    "OBSERVATION_DIM",
    "OBSERVATION_SCHEMA",
    "OBSERVATION_SLICES",
    "StepBatch",
    "action_mask_from_observation",
    "make_env",
    "require_torch",
]
