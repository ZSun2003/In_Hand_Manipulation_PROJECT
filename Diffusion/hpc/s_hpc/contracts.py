"""Stable model, action, observation, and checkpoint contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from screw_config import DEFAULT_CONFIG
from screw_control import (
    CAPTURE_GATE_STEP,
    DIFFUSION_TO_ANCHOR_BLEND_STEPS,
    PREINSERT_GATE_STEP,
    THREAD_GATE_STEP,
    THREAD_REGRASP_GRASP_HEIGHT_FROM_TIP_M,
    THREAD_REGRASP_RELEASED_DISTANCE_M,
    THREAD_REGRASP_STEP,
    YAW_EQUIVARIANT_ALIGN_BOUNDARY_STEP,
    YAW_EQUIVARIANT_ALIGN_STEPS,
    YAW_EQUIVARIANT_RELEASED_DISTANCE_M,
    YAW_EQUIVARIANT_ROLL_GAIN,
    YAW_EQUIVARIANT_ROLL_BOUNDARY_STEP,
    YAW_EQUIVARIANT_ROLL_STEPS,
    YAW_EQUIVARIANT_STRATEGY_VERSION,
)


BACKEND = "mjwarp-native-v3"
CHECKPOINT_FORMAT = "screw-hpc-checkpoint-v3"
OBS_SCHEMA = "screw-omniscient-state-v7"
OBS_DIM = 191
OBS_HORIZON = 2
ACTION_DIM = 14
# The Diffusion policy contract is intentionally frozen: existing H110
# checkpoints were trained with a historical six-dimensional task-space
# residual field in their metadata.  The executable residual controller now
# acts on all 16 actuators after IK, without changing the 191-dimensional
# Diffusion observation.
DIFFUSION_RESIDUAL_ACTION_DIM = 6
RESIDUAL_ACTION_DIM = 16
TQC_OBS_SCHEMA = "screw-post-ik-actuator-residual-state-v1"
TQC_CONTROLLER_STATE_NAMES: Tuple[str, ...] = (
    "diffusion_plan_cursor_fraction",
    "anchor_handover_active",
    "anchor_handover_progress",
)
TQC_OBS_DIM = (
    OBS_DIM
    + 3 * RESIDUAL_ACTION_DIM
    + len(TQC_CONTROLLER_STATE_NAMES)
)
FINGERS: Tuple[str, ...] = ("U", "D", "L", "R")
PHASES: Tuple[str, ...] = (
    "PREGRASP",
    "CLOSE",
    "LIFT",
    "ALIGN",
    "INSERT",
    "ENGAGE",
    "POWER",
    "LR_OPEN",
    "LR_RESET",
    "LR_CLOSE",
    "UD_OPEN",
    "UD_RESET",
    "UD_CLOSE",
    "FINAL_HOLD",
    "COMPLETE",
)


def observation_names(nq: int = 31, nv: int = 30, nu: int = 16) -> Tuple[str, ...]:
    names = []
    names.extend(f"qpos_{index}" for index in range(nq))
    names.extend(f"qvel_{index}" for index in range(nv))
    names.extend(f"ctrl_{index}" for index in range(nu))
    names.extend(f"actuator_force_{index}" for index in range(nu))
    names.extend(
        (
            "screw_x", "screw_y", "screw_z",
            "screw_qw", "screw_qx", "screw_qy", "screw_qz",
            "screw_vx", "screw_vy", "screw_vz",
            "screw_wx", "screw_wy", "screw_wz",
            "tip_socket_x", "tip_socket_y", "depth",
            "axis_error_x", "axis_error_y",
            "marker_sin", "marker_cos", "twist_unwrapped", "turn_count",
            "helical_error", "helical_velocity_error",
            "thread_resistance", "axial_preload",
            "socket_angle_sin", "socket_angle_cos",
            "socket_mouth_x", "socket_mouth_y",
            "socket_opening_x", "socket_opening_y",
        )
    )
    for finger in FINGERS:
        names.extend(
            (
                f"force_{finger}", f"contact_{finger}",
                f"tip_rel_x_{finger}", f"tip_rel_y_{finger}",
                f"tip_rel_z_{finger}", f"tip_vx_{finger}",
                f"tip_vy_{finger}", f"tip_vz_{finger}",
                f"material_sin_{finger}", f"material_cos_{finger}",
                f"distance_{finger}",
            )
        )
    names.extend(f"phase_{phase.lower()}" for phase in PHASES)
    names.extend(
        (
            "target_depth", "target_twist", "depth_remaining",
            "twist_remaining", "episode_fraction", "holder_active", "engaged",
        )
    )
    result = tuple(names)
    if len(result) != OBS_DIM:
        raise RuntimeError(f"observation contract has {len(result)} != {OBS_DIM} fields")
    return result


OBSERVATION_NAMES = observation_names()
TQC_OBSERVATION_NAMES = (
    OBSERVATION_NAMES
    + tuple(
        f"normalised_nominal_ctrl_{index}"
        for index in range(RESIDUAL_ACTION_DIM)
    )
    + tuple(
        f"normalised_previous_realized_residual_{index}"
        for index in range(RESIDUAL_ACTION_DIM)
    )
    + tuple(
        f"residual_action_mask_{index}"
        for index in range(RESIDUAL_ACTION_DIM)
    )
    + TQC_CONTROLLER_STATE_NAMES
)
if len(TQC_OBSERVATION_NAMES) != TQC_OBS_DIM:
    raise RuntimeError(
        "TQC observation contract has "
        f"{len(TQC_OBSERVATION_NAMES)} != {TQC_OBS_DIM} fields"
    )
_ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
TASK_ASSET_HASHES = {
    name: hashlib.sha256((_ASSET_ROOT / name).read_bytes()).hexdigest()
    for name in ("Project_hpc.xml", "Gripper.xml", "Share_hpc.xml")
}
_ROLL_APPLICATION = (
    "disabled-direct-final-frame"
    if YAW_EQUIVARIANT_ROLL_GAIN == 0.0
    else (
        "post-interpolation-right-multiply-Rz("
        f"{YAW_EQUIVARIANT_ROLL_GAIN:g}*delta*minimum_jerk("
        f"(step-{YAW_EQUIVARIANT_ROLL_BOUNDARY_STEP})/"
        f"{YAW_EQUIVARIANT_ROLL_STEPS}));"
        "controller-boundaries-imported-from-screw-control"
    )
)


CONTROLLER_STRATEGY = {
    "version": YAW_EQUIVARIANT_STRATEGY_VERSION,
    "released_distance_m": YAW_EQUIVARIANT_RELEASED_DISTANCE_M,
    "thread_regrasp_released_distance_m": (
        THREAD_REGRASP_RELEASED_DISTANCE_M
    ),
    "roll_gain": YAW_EQUIVARIANT_ROLL_GAIN,
    "scripted_grasp_height_from_tip_m": (
        DEFAULT_CONFIG.screw_grasp_height_from_tip_m
    ),
    "diffusion_to_anchor_blend": {
        "steps": DIFFUSION_TO_ANCHOR_BLEND_STEPS,
        "profile": "minimum-jerk",
        "rotation": "so3-geodesic",
        "routes": (
            "preinsert-gate",
            "capture-gate",
            "thread-gate",
            "thread-regrasp-gate",
            "regrasp-phases",
            "contact-recovery",
            "unmeasured-engage",
        ),
    },
    "thread_regrasp_height": {
        "from_tip_m": THREAD_REGRASP_GRASP_HEIGHT_FROM_TIP_M,
        "switch_step": THREAD_REGRASP_STEP + 1,
        "mode": "direct-at-regrasp",
    },
    "align_boundary_step": YAW_EQUIVARIANT_ALIGN_BOUNDARY_STEP,
    "align_steps": YAW_EQUIVARIANT_ALIGN_STEPS,
    "roll_boundary_step": YAW_EQUIVARIANT_ROLL_BOUNDARY_STEP,
    "roll_steps": YAW_EQUIVARIANT_ROLL_STEPS,
    "yaw_decomposition": (
        "diagnostic-only;actual-socket-yaw-used-directly"
    ),
    "finger_role_mapping": (
        "identity-physical-U/D/L/R;yaw-invariant;"
        "material-fixed-until-measured-thread-engagement"
    ),
    "pose_transform": (
        "direct-SLERP(identity,aligned-actual-socket-frame);"
        "tip=Q@canonical-tip+mouth"
    ),
    "roll_application": _ROLL_APPLICATION,
    "preinsert_gate": {
        "leading_face_standoff_m": DEFAULT_CONFIG.preinsert_standoff_m,
        "lateral_tolerance_m": (
            DEFAULT_CONFIG.preinsert_lateral_tolerance_m
        ),
        "orientation_tolerance_deg": (
            DEFAULT_CONFIG.preinsert_orientation_tolerance_deg
        ),
        "depth_tolerance_m": DEFAULT_CONFIG.preinsert_depth_tolerance_m,
        "tip_speed_tolerance_m_s": (
            DEFAULT_CONFIG.preinsert_tip_speed_tolerance_m_s
        ),
        "lateral_speed_tolerance_m_s": (
            DEFAULT_CONFIG.preinsert_lateral_speed_tolerance_m_s
        ),
        "angular_speed_tolerance_deg_s": (
            DEFAULT_CONFIG.preinsert_angular_speed_tolerance_deg_s
        ),
        "stable_steps": DEFAULT_CONFIG.preinsert_stable_steps,
        "support_separation_deg": (
            DEFAULT_CONFIG.preinsert_support_separation_deg
        ),
    },
    "capture_handoff": {
        "preinsert_gate_step": PREINSERT_GATE_STEP,
        "capture_gate_step": CAPTURE_GATE_STEP,
        "thread_gate_step": THREAD_GATE_STEP,
        "thread_regrasp_step": THREAD_REGRASP_STEP,
        "capture_depth_m": DEFAULT_CONFIG.capture_depth_m,
        "head_bottom_standoff_m": (
            DEFAULT_CONFIG.screw_shaft_length_m
            - DEFAULT_CONFIG.capture_depth_m
        ),
        "capture_lateral_tolerance_m": (
            DEFAULT_CONFIG.capture_lateral_tolerance_m
        ),
        "capture_orientation_tolerance_deg": (
            DEFAULT_CONFIG.capture_orientation_tolerance_deg
        ),
        "capture_depth_tolerance_m": (
            DEFAULT_CONFIG.capture_depth_tolerance_m
        ),
        "capture_tip_speed_tolerance_m_s": (
            DEFAULT_CONFIG.capture_tip_speed_tolerance_m_s
        ),
        "capture_lateral_speed_tolerance_m_s": (
            DEFAULT_CONFIG.capture_lateral_speed_tolerance_m_s
        ),
        "capture_angular_speed_tolerance_deg_s": (
            DEFAULT_CONFIG.capture_angular_speed_tolerance_deg_s
        ),
        "capture_stable_steps": DEFAULT_CONFIG.capture_stable_steps,
        "all_four_contact_stable_steps": (
            DEFAULT_CONFIG.all_four_contact_stable_steps
        ),
        "contact_force_threshold_n": (
            DEFAULT_CONFIG.contact_force_threshold_n
        ),
        "contact_force_release_threshold_n": (
            DEFAULT_CONFIG.contact_force_release_threshold_n
        ),
        "contact_loss_grace_steps": (
            DEFAULT_CONFIG.contact_loss_grace_steps
        ),
        "minimum_contact_count_after_capture": (
            DEFAULT_CONFIG.minimum_contact_count_after_capture
        ),
        "minimum_contact_count_before_capture": (
            DEFAULT_CONFIG.minimum_contact_count_before_capture
        ),
        "commanded_prethread_grasp": "all-four-at-0.039m",
        "diagnostic_primary_pair": "physical-L/R",
        "material_angles": "fixed-from-pickup-through-thread-gate",
        "thread_engagement_depth_m": (
            DEFAULT_CONFIG.thread_engagement_depth_m
        ),
        "thread_depth_tolerance_m": (
            DEFAULT_CONFIG.thread_depth_tolerance_m
        ),
        "thread_lateral_tolerance_m": (
            DEFAULT_CONFIG.thread_lateral_tolerance_m
        ),
        "thread_orientation_tolerance_deg": (
            DEFAULT_CONFIG.thread_orientation_tolerance_deg
        ),
        "thread_stable_steps": DEFAULT_CONFIG.thread_stable_steps,
        "prethread_regrasp": False,
    },
}
for _name in (
    "preinsert_feedback_gain",
    "preinsert_command_lateral_limit_m",
    "preinsert_feedback_depth_limit_m",
    "preinsert_feedback_lateral_slew_m",
    "preinsert_feedback_depth_slew_m",
):
    if hasattr(DEFAULT_CONFIG, _name):
        CONTROLLER_STRATEGY["preinsert_gate"][_name] = getattr(
            DEFAULT_CONFIG, _name
        )
TASK_GEOMETRY_HASH = hashlib.sha256(
    json.dumps(
        {
            "root_height_m": 0.620,
            "task": DEFAULT_CONFIG.as_dict(),
            "assets": TASK_ASSET_HASHES,
            "controller_strategy": CONTROLLER_STRATEGY,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
PHASE_OBSERVATION_SLICE = slice(
    OBSERVATION_NAMES.index("phase_pregrasp"),
    OBSERVATION_NAMES.index("phase_complete") + 1,
)
ENGAGED_OBSERVATION_INDEX = OBSERVATION_NAMES.index("engaged")
DIFFUSION_SCHEMA_HASH = hashlib.sha256(
    json.dumps(
        {
            "schema": OBS_SCHEMA,
            "names": OBSERVATION_NAMES,
            "action_dim": ACTION_DIM,
            # Keep this legacy value in the Diffusion hash so previously
            # trained checkpoints remain byte-for-byte schema compatible.
            "residual_action_dim": DIFFUSION_RESIDUAL_ACTION_DIM,
        },
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
TQC_SCHEMA_HASH = hashlib.sha256(
    json.dumps(
        {
            "schema": TQC_OBS_SCHEMA,
            "names": TQC_OBSERVATION_NAMES,
            "base_schema_hash": DIFFUSION_SCHEMA_HASH,
            "obs_dim": TQC_OBS_DIM,
            "action_dim": RESIDUAL_ACTION_DIM,
        },
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

# Public compatibility alias used by demonstrations, preflight, adaptive
# capacity selection, and monitoring.  Those artifacts all describe the
# frozen Diffusion observation/action contract.
SCHEMA_HASH = DIFFUSION_SCHEMA_HASH


def checkpoint_metadata(kind: str, config: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint_kind = str(kind)
    is_tqc = checkpoint_kind == "tqc"
    return {
        "format": CHECKPOINT_FORMAT,
        "backend": BACKEND,
        "kind": checkpoint_kind,
        "obs_schema": TQC_OBS_SCHEMA if is_tqc else OBS_SCHEMA,
        "schema_hash": TQC_SCHEMA_HASH if is_tqc else DIFFUSION_SCHEMA_HASH,
        "obs_dim": TQC_OBS_DIM if is_tqc else OBS_DIM,
        "obs_horizon": 1 if is_tqc else OBS_HORIZON,
        "action_dim": RESIDUAL_ACTION_DIM if is_tqc else ACTION_DIM,
        "diffusion_action_dim": ACTION_DIM,
        "residual_action_dim": (
            RESIDUAL_ACTION_DIM
            if is_tqc
            else DIFFUSION_RESIDUAL_ACTION_DIM
        ),
        "diffusion_schema_hash": DIFFUSION_SCHEMA_HASH,
        "task_geometry_hash": TASK_GEOMETRY_HASH,
        "config": config,
    }


def validate_checkpoint(payload: Dict[str, Any], *, kind: str | None = None) -> None:
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not an hpc checkpoint")
    if payload.get("backend") != BACKEND:
        raise ValueError("checkpoint backend is incompatible with native MJWarp")
    expected_kind = str(kind) if kind is not None else str(payload.get("kind"))
    expected_schema_hash = (
        TQC_SCHEMA_HASH if expected_kind == "tqc" else DIFFUSION_SCHEMA_HASH
    )
    if payload.get("schema_hash") != expected_schema_hash:
        raise ValueError("checkpoint observation/action schema hash mismatch")
    if payload.get("task_geometry_hash") != TASK_GEOMETRY_HASH:
        raise ValueError("checkpoint task geometry hash mismatch")
    if kind is not None and payload.get("kind") != kind:
        raise ValueError(
            f"expected {kind!r} checkpoint, got {payload.get('kind')!r}"
        )


__all__ = [
    "ACTION_DIM",
    "BACKEND",
    "CHECKPOINT_FORMAT",
    "CONTROLLER_STRATEGY",
    "FINGERS",
    "ENGAGED_OBSERVATION_INDEX",
    "DIFFUSION_RESIDUAL_ACTION_DIM",
    "DIFFUSION_SCHEMA_HASH",
    "OBSERVATION_NAMES",
    "OBS_DIM",
    "OBS_HORIZON",
    "OBS_SCHEMA",
    "PHASE_OBSERVATION_SLICE",
    "PHASES",
    "RESIDUAL_ACTION_DIM",
    "SCHEMA_HASH",
    "TQC_OBSERVATION_NAMES",
    "TQC_CONTROLLER_STATE_NAMES",
    "TQC_OBS_DIM",
    "TQC_OBS_SCHEMA",
    "TQC_SCHEMA_HASH",
    "TASK_GEOMETRY_HASH",
    "TASK_ASSET_HASHES",
    "checkpoint_metadata",
    "validate_checkpoint",
]
