"""Deterministic trained-residual 0--90 degree rotation sweep.

The runner deliberately keeps rollout execution on the raw Gym environment.
``VecNormalize`` is used only as a frozen observation transform, so terminal
state, actuator residuals, and controller diagnostics cannot be overwritten by
an automatic vector-environment reset before they are recorded.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import config
from .env import G3G6ManipulationEnv, OBSERVATION_DIM
from .rotation_sweep import (
    EXPECTED_SWEEP_SAMPLES,
    MAX_SWEEP_ANGLE_DEG,
    RotationSweepSpec,
    build_rotation_sweep_specs,
)
from .rotations import ROTATION_MOTION_TICKS, build_rotation_euler_schedule


TRAINING_SCHEMA = "g3g6-full-residual-tqc-training-v1"
TRACE_SCHEMA = "g3g6-rotation-residual-sweep-trace-v1"
RUN_SCHEMA = "g3g6-rotation-residual-sweep-run-v1"
CONTROLLER_KIND = "trained_tqc_ik_residual"
PADDED_ACTION_DIM = config.MAX_ACTION_DIM
MODEL_FILENAME = "final_model.zip"
VEC_NORMALIZE_FILENAME = "vec_normalize.pkl"


@dataclass(frozen=True)
class TrainingArtifacts:
    run_dir: Path
    manifest: Mapping[str, Any]
    gripper: str
    action_dim: int
    observation_dim: int
    rotation_sampling_profile: str
    residual_nominal_feedback: bool
    residual_after_preaction_only: bool
    model_path: Path
    vec_normalize_path: Path
    model_sha256: str
    vec_normalize_sha256: str


@dataclass
class LoadedController:
    model: Any
    normalizer: Any
    close_callback: Callable[[], None] | None = None

    def close(self) -> None:
        if self.close_callback is not None:
            callback, self.close_callback = self.close_callback, None
            callback()


def _inside_project(path: Path) -> Path:
    value = (
        (config.PROJECT_ROOT / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )
    try:
        value.relative_to(config.PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("residual sweep inputs and outputs must stay inside IK-residual") from exc
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_residual_contract(gripper: str) -> dict[str, Any]:
    """Return the exact unit-aware residual manifest emitted by the trainer."""

    normalized = config.GripperKind(gripper).value
    payload: dict[str, Any] = {
        "action_dimension": config.action_dim(normalized),
        "meaning": "normalized_direct_actuator_residual_after_nominal_ik",
        "nominal_ik_pose_feedback": True,
        "activation_by_phase": {
            "1": "all_ticks",
            "2": "after_preaction_tick_360",
            "3": "after_preaction_tick_360",
        },
        "blend": float(config.ANGULAR_RESIDUAL_BLEND),
        "rotary": {
            "count": 12,
            "unit": "rad",
            "limit": float(config.ANGULAR_RESIDUAL_LIMIT_RAD),
            "rate_per_control_tick": float(
                config.ANGULAR_RESIDUAL_RATE_RAD_PER_TICK
            ),
            "actuator_indices": list(config.ANGULAR_ACTUATOR_INDICES[normalized]),
        },
        "reward_penalty_weight": float(
            config.RESIDUAL_PENALTY_WEIGHTS[normalized]
        ),
    }
    if normalized == "g6":
        payload["linear"] = {
            "count": 4,
            "unit": "m",
            "limit": float(config.LINEAR_RESIDUAL_LIMIT_M),
            "rate_per_control_tick": float(
                config.LINEAR_RESIDUAL_RATE_M_PER_TICK
            ),
            "blend": float(config.LINEAR_RESIDUAL_BLEND),
            "actuator_indices": list(config.G6_LINEAR_ACTUATOR_INDICES),
        }
    else:
        payload["linear"] = {"count": 0, "actuator_indices": []}
    return payload


def _require_exact_mapping(
    actual: object, expected: Mapping[str, Any], *, label: str
) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError(f"{label} must be an object")
    if set(actual) != set(expected):
        raise ValueError(
            f"{label} keys mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, list) and expected_value and isinstance(
            expected_value[0], (int, float)
        ):
            try:
                numeric = np.asarray(actual_value, dtype=np.float64)
                target = np.asarray(expected_value, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label}.{key} must be numeric") from exc
            if numeric.shape != target.shape or not np.allclose(
                numeric, target, rtol=0.0, atol=1.0e-15
            ):
                raise ValueError(f"{label}.{key} does not match the runtime contract")
        elif actual_value != expected_value:
            raise ValueError(
                f"{label}.{key} mismatch: expected {expected_value!r}, "
                f"got {actual_value!r}"
            )


def load_training_artifacts(run_dir: Path) -> TrainingArtifacts:
    directory = _inside_project(run_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TRAINING_SCHEMA:
        raise ValueError(
            f"unsupported residual training schema: {manifest.get('schema')!r}"
        )
    try:
        gripper = config.GripperKind(str(manifest["gripper"])).value
    except (KeyError, ValueError) as exc:
        raise ValueError("training manifest has an invalid gripper") from exc
    if int(manifest.get("phase", -1)) != int(config.Phase.ROTATE):
        raise ValueError("residual training manifest must describe Phase 3")
    if str(manifest.get("algorithm", "")) != "TQC":
        raise ValueError("residual training manifest algorithm must be TQC")
    if manifest.get("status") != "complete":
        raise ValueError("residual training run is not complete")
    observation_dim = int(manifest.get("observation_dimension", -1))
    if observation_dim != 149 or observation_dim != OBSERVATION_DIM:
        raise ValueError(
            f"residual model observation_dim must equal runtime OBSERVATION_DIM=149, "
            f"got {observation_dim}"
        )
    task = manifest.get("task")
    if not isinstance(task, Mapping):
        raise ValueError("training manifest must contain a task object")
    residual = manifest.get("residual")
    if not isinstance(residual, Mapping):
        raise ValueError("training manifest must contain a residual object")
    action_dim = int(residual.get("action_dimension", -1))
    if action_dim != config.action_dim(gripper):
        raise ValueError(
            f"residual model action_dim mismatch for {gripper}: {action_dim}"
        )
    target_height = float(
        task.get("target_grasp_plane_z_m", float("nan"))
    )
    if not math.isfinite(target_height) or not math.isclose(
        target_height,
        config.ROTATION_TARGET_GRASP_PLANE_Z_M,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("residual training target height must be 0.325 m")
    rotation_profile = str(task.get("sampling_profile", ""))
    if rotation_profile != config.WEIGHTED_ROTATION_SAMPLING_PROFILE:
        raise ValueError("residual training must use weighted_0_90 sampling")
    residual_nominal_feedback = bool(residual.get("nominal_ik_pose_feedback", False))
    if not residual_nominal_feedback:
        raise ValueError("trained residual must use the pure-IK feedback nominal")
    activation = residual.get("activation_by_phase")
    if not isinstance(activation, Mapping) or activation.get("3") != (
        "after_preaction_tick_360"
    ):
        raise ValueError(
            "trained Phase-3 residual must be disabled through the pre-action prefix"
        )
    expected_budgets = {"1": 500_000, "2": 2_000_000, "3": 2_000_000}
    if manifest.get("phase_budgets") != expected_budgets:
        raise ValueError("residual training phase budgets do not match 500k/2M/2M")
    if int(manifest.get("total_timesteps", -1)) != 4_500_000:
        raise ValueError("residual training must contain 4,500,000 steps per gripper")
    if int(manifest.get("n_envs", -1)) != 72:
        raise ValueError("residual training must use 72 vector environments")
    if int(manifest.get("expected_executed_timesteps", -1)) != 4_500_072:
        raise ValueError("72-env residual training executed-step contract changed")
    _require_exact_mapping(
        residual,
        expected_residual_contract(gripper),
        label="residual",
    )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("training manifest must contain an artifacts object")
    required_artifact_keys = {"model", "vec_normalize"}
    if set(artifacts) != required_artifact_keys:
        raise ValueError("training artifacts keys do not match the v1 contract")
    model_artifact = artifacts["model"]
    normalizer_artifact = artifacts["vec_normalize"]
    if not isinstance(model_artifact, Mapping) or not isinstance(
        normalizer_artifact, Mapping
    ):
        raise ValueError("training artifacts must contain nested path/SHA objects")
    if set(model_artifact) != {"path", "sha256"} or set(normalizer_artifact) != {
        "path",
        "sha256",
    }:
        raise ValueError("training artifact entries must contain only path and sha256")
    if model_artifact["path"] != MODEL_FILENAME:
        raise ValueError(f"training model must be named {MODEL_FILENAME}")
    if normalizer_artifact["path"] != VEC_NORMALIZE_FILENAME:
        raise ValueError(
            f"training normalizer must be named {VEC_NORMALIZE_FILENAME}"
        )
    model_path = directory / MODEL_FILENAME
    vec_path = directory / VEC_NORMALIZE_FILENAME
    if not model_path.is_file() or not vec_path.is_file():
        raise FileNotFoundError("training model or VecNormalize artifact is missing")
    model_hash = sha256_file(model_path)
    vec_hash = sha256_file(vec_path)
    for label, declared, computed in (
        ("model", model_artifact["sha256"], model_hash),
        ("vec_normalize", normalizer_artifact["sha256"], vec_hash),
    ):
        if not isinstance(declared, str) or declared.lower() != computed:
            raise ValueError(f"{label} SHA256 mismatch")
    return TrainingArtifacts(
        run_dir=directory,
        manifest=MappingProxyType(dict(manifest)),
        gripper=gripper,
        action_dim=action_dim,
        observation_dim=observation_dim,
        rotation_sampling_profile=rotation_profile,
        residual_nominal_feedback=residual_nominal_feedback,
        residual_after_preaction_only=True,
        model_path=model_path,
        vec_normalize_path=vec_path,
        model_sha256=model_hash,
        vec_normalize_sha256=vec_hash,
    )


def _validate_space_shape(space: object, shape: tuple[int, ...], label: str) -> None:
    actual = tuple(getattr(space, "shape", ()))
    if actual != shape:
        raise ValueError(f"{label} shape mismatch: expected {shape}, got {actual}")


def load_controller(
    artifacts: TrainingArtifacts, *, device: str = "auto"
) -> LoadedController:
    """Load TQC and a frozen VecNormalize transform without stepping its VecEnv."""

    from sb3_contrib import TQC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    probe = DummyVecEnv(
        [
            lambda: G3G6ManipulationEnv(
                gripper=artifacts.gripper,
                phase=config.Phase.ROTATE,
                object_name="cube",
                rotation_axes="x",
                rotation_sampling_profile=artifacts.rotation_sampling_profile,
                residual_nominal_feedback=artifacts.residual_nominal_feedback,
                residual_after_preaction_only=artifacts.residual_after_preaction_only,
            )
        ]
    )
    try:
        normalizer = VecNormalize.load(str(artifacts.vec_normalize_path), probe)
        normalizer.training = False
        normalizer.norm_reward = False
        model = TQC.load(str(artifacts.model_path), device=device)
        _validate_space_shape(
            model.observation_space, (artifacts.observation_dim,), "model observation"
        )
        _validate_space_shape(model.action_space, (artifacts.action_dim,), "model action")
        _validate_space_shape(
            normalizer.observation_space,
            (artifacts.observation_dim,),
            "VecNormalize observation",
        )
    except BaseException:
        probe.close()
        raise
    return LoadedController(model=model, normalizer=normalizer, close_callback=probe.close)


def _normalized_policy_action(
    model: Any,
    normalizer: Any,
    observation: np.ndarray,
    *,
    action_dim: int,
) -> np.ndarray:
    raw_observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    normalized = np.asarray(normalizer.normalize_obs(raw_observation), dtype=np.float32)
    action, _state = model.predict(normalized, deterministic=True)
    value = np.asarray(action, dtype=np.float64)
    if value.size != action_dim:
        raise RuntimeError(
            f"TQC action width changed: expected {action_dim}, got {value.shape}"
        )
    value = value.reshape(action_dim)
    if not np.all(np.isfinite(value)):
        raise RuntimeError("TQC produced a non-finite residual action")
    return value


def _pad_native(value: object, native_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(native_dim)
    if not np.all(np.isfinite(array)):
        raise RuntimeError("non-finite residual-controller vector")
    result = np.zeros(PADDED_ACTION_DIM, dtype=np.float64)
    result[:native_dim] = array
    return result


def pair_id(spec: RotationSweepSpec) -> str:
    return (
        f"{spec.gripper}:{spec.object_name}:{spec.rotation_axes}:"
        f"seed={spec.seed}:goal={spec.max_angle_deg:g}"
    )


def _record_tick(
    spec: RotationSweepSpec,
    env: G3G6ManipulationEnv,
    artifacts: TrainingArtifacts,
    info: dict[str, Any],
    euler_schedule: np.ndarray,
    raw_action: np.ndarray,
    *,
    terminated: bool,
) -> dict[str, Any]:
    if int(info.get("phase", -1)) != int(config.Phase.ROTATE):
        raise RuntimeError("trained residual sweep left Phase 3")
    if str(info.get("task")) != "rotate":
        raise RuntimeError("trained residual sweep task is not rotate")
    if str(info.get("control_mode")) != "residual_action":
        raise RuntimeError("trained residual sweep did not use env.step(action)")
    if info.get("rotation_sampling_profile") != artifacts.rotation_sampling_profile:
        raise RuntimeError("runtime rotation sampling profile differs from training")
    if bool(info.get("residual_nominal_feedback")) != artifacts.residual_nominal_feedback:
        raise RuntimeError("runtime nominal feedback differs from training")
    if bool(info.get("residual_after_preaction_only")) is not True:
        raise RuntimeError("runtime residual prefix mask differs from training")
    target_height = float(info.get("task_target_grasp_plane_z_m", float("nan")))
    if not math.isclose(
        target_height,
        config.ROTATION_TARGET_GRASP_PLANE_Z_M,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("trained residual sweep target height is not 0.325 m")

    native_dim = artifacts.action_dim
    requested = np.asarray(info["residual_requested_action"], dtype=np.float64).reshape(
        native_dim
    )
    if not np.allclose(requested, raw_action, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("environment requested action differs from TQC prediction")
    executed = np.asarray(info["residual_executed_action"], dtype=np.float64).reshape(
        native_dim
    )
    residual = np.asarray(info["actuator_residual"], dtype=np.float64).reshape(
        native_dim
    )
    if not np.all(np.isfinite(executed)) or not np.all(np.isfinite(residual)):
        raise RuntimeError("environment emitted a non-finite residual diagnostic")

    command_tick = int(info["step"]) - 1
    action_tick = command_tick - config.PRE_ACTION_TICKS
    residual_active = bool(info["residual_active"])
    expected_active = action_tick >= 0
    if residual_active != expected_active:
        raise RuntimeError("residual pre-action mask changed")
    if not residual_active and (
        np.any(executed != 0.0) or np.any(residual != 0.0)
    ):
        raise RuntimeError("residual was applied before the rotation action")

    is_baseline = action_tick == -1 and not terminated
    is_motion = 0 <= action_tick < ROTATION_MOTION_TICKS
    is_hold = ROTATION_MOTION_TICKS <= action_tick < euler_schedule.shape[0]
    if is_baseline:
        commanded_angle = 0.0
        segment = "reference_baseline"
    elif action_tick >= 0:
        index = min(action_tick, euler_schedule.shape[0] - 1)
        commanded_angle = float(
            euler_schedule[index, "xyz".index(spec.rotation_axes[0])]
        )
        segment = "rotation_motion" if is_motion else "terminal_hold"
    elif command_tick < config.APPROACH_TICKS:
        commanded_angle, segment = 0.0, "approach"
    elif command_tick < config.APPROACH_TICKS + config.CONTACT_TICKS:
        commanded_angle, segment = 0.0, "contact"
    else:
        commanded_angle, segment = 0.0, "lift"

    angle_error = np.asarray(info["angle_error_deg_per_axis"], dtype=np.float64)
    if is_baseline:
        angle_error = np.zeros(3, dtype=np.float64)
    actual_position = np.asarray(
        info["object_grasp_frame_position_m"], dtype=np.float64
    )
    target_position = np.asarray(info["target_position_m"], dtype=np.float64)
    if is_baseline:
        target_position = np.array(
            (actual_position[0], actual_position[1], target_height),
            dtype=np.float64,
        )
    padded_raw = _pad_native(raw_action, native_dim)
    padded_executed = _pad_native(executed, native_dim)
    padded_residual = _pad_native(residual, native_dim)
    ctrl_difference = float(
        np.max(np.abs(env.nominal_ctrl - env.applied_ctrl), initial=0.0)
    )
    row: dict[str, Any] = {
        "schema": TRACE_SCHEMA,
        "controller": CONTROLLER_KIND,
        "pair_id": pair_id(spec),
        "episode_id": f"{spec.episode_id}_trained_residual",
        "gripper": spec.gripper,
        "object": spec.object_name,
        "phase": int(info["phase"]),
        "task": str(info["task"]),
        "rotation_axes": spec.rotation_axes,
        "seed": spec.seed,
        "max_angle_deg": spec.max_angle_deg,
        "rotation_target_grasp_plane_z_m": target_height,
        "rotation_sampling_profile": artifacts.rotation_sampling_profile,
        "residual_nominal_feedback": int(artifacts.residual_nominal_feedback),
        "residual_after_preaction_only": 1,
        "residual_active": int(residual_active),
        "model_sha256": artifacts.model_sha256,
        "vec_normalize_sha256": artifacts.vec_normalize_sha256,
        "step": int(info["step"]),
        "command_tick": command_tick,
        "action_tick": action_tick,
        "time_s": float((command_tick + 1) * config.CONTROL_DT_S),
        "segment": segment,
        "target_active_component_angle_deg": commanded_angle,
        "sweep_sample": int(is_baseline or is_motion),
        "reference_baseline": int(is_baseline),
        "rotation_motion_sample": int(is_motion),
        "terminal_hold_sample": int(is_hold),
        "rotation_error_l2_deg": float(np.linalg.norm(angle_error)),
        "object_grasp_frame_height_m": float(actual_position[2]),
        "policy_action_raw_l2": float(np.linalg.norm(raw_action)),
        "policy_action_raw_linf": float(np.max(np.abs(raw_action), initial=0.0)),
        "policy_action_executed_l2": float(np.linalg.norm(executed)),
        "policy_action_executed_linf": float(
            np.max(np.abs(executed), initial=0.0)
        ),
        "actuator_residual_l2": float(np.linalg.norm(residual)),
        "actuator_residual_linf": float(np.max(np.abs(residual), initial=0.0)),
        "max_abs_actuator_residual": float(
            np.max(np.abs(residual), initial=0.0)
        ),
        "nominal_applied_ctrl_linf": ctrl_difference,
        "tip_contact_count": int(info["tip_contact_count"]),
        "ik_exact": int(bool(info["ik_exact"])),
        "ik_projection_fraction": float(info["ik_projection_fraction"]),
        "terminated": int(terminated),
        "is_success": int(bool(info.get("is_success", False))),
        "termination_reason": str(info["termination_reason"]),
    }
    for index, axis in enumerate("xyz"):
        row[f"angle_error_deg_{axis}"] = float(angle_error[index])
        row[f"actual_position_m_{axis}"] = float(actual_position[index])
        row[f"target_position_m_{axis}"] = float(target_position[index])
    for index in range(PADDED_ACTION_DIM):
        row[f"policy_action_raw_{index}"] = float(padded_raw[index])
        row[f"policy_action_executed_{index}"] = float(padded_executed[index])
        row[f"actuator_residual_{index}"] = float(padded_residual[index])
    return row


def _summary(
    spec: RotationSweepSpec,
    artifacts: TrainingArtifacts,
    rows: list[dict[str, Any]],
    trace_path: Path,
) -> dict[str, Any]:
    sweep = [row for row in rows if bool(row["sweep_sample"])]
    motion = [row for row in rows if bool(row["rotation_motion_sample"])]
    baseline = [row for row in rows if bool(row["reference_baseline"])]
    hold = [row for row in rows if bool(row["terminal_hold_sample"])]
    projected = [row for row in motion if not bool(row["ik_exact"])]
    final = rows[-1]
    reached = (
        max(float(row["target_active_component_angle_deg"]) for row in sweep)
        if sweep
        else None
    )
    return {
        "pair_id": pair_id(spec),
        "episode_id": str(final["episode_id"]),
        "trace": str(trace_path),
        "controller": CONTROLLER_KIND,
        "gripper": spec.gripper,
        "object": spec.object_name,
        "rotation_axes": spec.rotation_axes,
        "goal_euler_deg": list(spec.goal_euler_deg),
        "max_angle_deg": spec.max_angle_deg,
        "seed": spec.seed,
        "rotation_target_grasp_plane_z_m": (
            config.ROTATION_TARGET_GRASP_PLANE_Z_M
        ),
        "model_sha256": artifacts.model_sha256,
        "vec_normalize_sha256": artifacts.vec_normalize_sha256,
        "ticks": len(rows),
        "sweep_samples": len(sweep),
        "expected_sweep_samples": EXPECTED_SWEEP_SAMPLES,
        "baseline_recorded": len(baseline) == 1,
        "motion_samples": len(motion),
        "expected_motion_samples": ROTATION_MOTION_TICKS,
        "terminal_hold_samples": len(hold),
        "expected_terminal_hold_samples": config.POST_ACTION_HOLD_TICKS,
        "sweep_started": bool(motion),
        "completed_motion_sweep": bool(
            len(motion) == ROTATION_MOTION_TICKS
            and reached is not None
            and math.isclose(reached, spec.max_angle_deg, abs_tol=1.0e-12)
        ),
        "reached_angle_deg": reached,
        "ended_early": bool(
            str(final["termination_reason"]) != "horizon"
            or reached is None
            or reached < spec.max_angle_deg - 1.0e-6
        ),
        "is_success": bool(final["is_success"]),
        "termination_reason": str(final["termination_reason"]),
        "maximum_rotation_error_l2_deg": (
            max(float(row["rotation_error_l2_deg"]) for row in sweep)
            if sweep
            else None
        ),
        "minimum_object_height_m": (
            min(float(row["object_grasp_frame_height_m"]) for row in sweep)
            if sweep
            else None
        ),
        "first_projection_angle_deg": (
            float(projected[0]["target_active_component_angle_deg"])
            if projected
            else None
        ),
        "projected_sweep_ticks": len(projected),
        "maximum_policy_action_raw_linf": max(
            float(row["policy_action_raw_linf"]) for row in rows
        ),
        "maximum_policy_action_executed_linf": max(
            float(row["policy_action_executed_linf"]) for row in rows
        ),
        "maximum_actuator_residual_linf": max(
            float(row["actuator_residual_linf"]) for row in rows
        ),
        "early_stop_policy": "trace ends at the last physical sample; no extrapolation",
    }


def run_residual_rotation_sweep_episode(
    spec: RotationSweepSpec,
    *,
    artifacts: TrainingArtifacts,
    controller: LoadedController,
    traces_dir: Path,
    env_factory: Callable[..., G3G6ManipulationEnv] = G3G6ManipulationEnv,
) -> dict[str, Any]:
    if spec.gripper != artifacts.gripper:
        raise ValueError("rotation spec and residual model grippers differ")
    if not math.isclose(spec.max_angle_deg, MAX_SWEEP_ANGLE_DEG, abs_tol=0.0):
        raise ValueError("trained residual comparison sweep is fixed at 90 degrees")
    goal = np.asarray(spec.goal_euler_deg, dtype=np.float64)
    schedule = build_rotation_euler_schedule(goal)
    env = env_factory(
        gripper=spec.gripper,
        phase=config.Phase.ROTATE,
        object_name=spec.object_name,
        rotation_axes=spec.rotation_axes,
        rotation_sampling_profile=artifacts.rotation_sampling_profile,
        residual_nominal_feedback=artifacts.residual_nominal_feedback,
        residual_after_preaction_only=artifacts.residual_after_preaction_only,
    )
    rows: list[dict[str, Any]] = []
    try:
        observation, reset_info = env.reset(
            seed=spec.seed,
            options={
                "rotation_axes": spec.rotation_axes,
                "rotation_euler_deg": goal,
            },
        )
        if np.asarray(observation).shape != (artifacts.observation_dim,):
            raise RuntimeError("runtime observation width differs from training")
        if reset_info.get("rotation_sampling_profile") != artifacts.rotation_sampling_profile:
            raise RuntimeError("reset profile differs from the trained profile")
        terminated = truncated = False
        while not (terminated or truncated):
            raw_action = _normalized_policy_action(
                controller.model,
                controller.normalizer,
                observation,
                action_dim=artifacts.action_dim,
            )
            observation, _reward, terminated, truncated, info = env.step(raw_action)
            row = _record_tick(
                spec,
                env,
                artifacts,
                info,
                schedule,
                raw_action,
                terminated=terminated or truncated,
            )
            rows.append(row)
    finally:
        env.close()
    if not rows:
        raise RuntimeError(f"residual sweep produced no rows: {spec.episode_id}")
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"{spec.episode_id}_trained_residual.csv"
    temporary = trace_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(trace_path)
    return _summary(
        spec,
        artifacts,
        rows,
        trace_path.relative_to(traces_dir.parent),
    )


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_residual_rotation_sweep(
    *,
    specs: Sequence[RotationSweepSpec],
    training_runs: Mapping[str, Path],
    output_dir: Path,
    device: str = "auto",
    artifact_loader: Callable[[Path], TrainingArtifacts] = load_training_artifacts,
    controller_loader: Callable[..., LoadedController] = load_controller,
    env_factory: Callable[..., G3G6ManipulationEnv] = G3G6ManipulationEnv,
) -> dict[str, Any]:
    spec_list = tuple(specs)
    if not spec_list:
        raise ValueError("at least one residual rotation sweep is required")
    if len({spec.episode_id for spec in spec_list}) != len(spec_list):
        raise ValueError("duplicate residual rotation sweep episode ids")
    if any(
        not math.isclose(spec.max_angle_deg, MAX_SWEEP_ANGLE_DEG, abs_tol=0.0)
        for spec in spec_list
    ):
        raise ValueError("all trained residual sweeps must target exactly 90 degrees")
    grippers = tuple(dict.fromkeys(spec.gripper for spec in spec_list))
    if set(training_runs) != set(grippers):
        raise ValueError("provide exactly one residual training run per requested gripper")
    artifacts_by_gripper = {
        gripper: artifact_loader(Path(training_runs[gripper])) for gripper in grippers
    }
    for gripper, artifacts in artifacts_by_gripper.items():
        if artifacts.gripper != gripper:
            raise ValueError("training-run map contains a mismatched gripper")

    destination = _inside_project(output_dir)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(f"residual sweep destination is not empty: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    traces_dir = destination / "traces"
    manifest: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "controller": CONTROLLER_KIND,
        "deterministic_policy": True,
        "normalization_frozen": True,
        "phase": int(config.Phase.ROTATE),
        "rotation_target_grasp_plane_z_m": config.ROTATION_TARGET_GRASP_PLANE_Z_M,
        "max_angle_deg": MAX_SWEEP_ANGLE_DEG,
        "expected_experiments": len(spec_list),
        "completed_experiments": 0,
        "training_runs": {
            gripper: {
                "path": str(artifacts.run_dir),
                "schema": TRAINING_SCHEMA,
                "action_dim": artifacts.action_dim,
                "observation_dim": artifacts.observation_dim,
                "rotation_sampling_profile": artifacts.rotation_sampling_profile,
                "residual_nominal_feedback": artifacts.residual_nominal_feedback,
                "residual_after_preaction_only": artifacts.residual_after_preaction_only,
                "model_sha256": artifacts.model_sha256,
                "vec_normalize_sha256": artifacts.vec_normalize_sha256,
            }
            for gripper, artifacts in artifacts_by_gripper.items()
        },
        "pairing": "same RotationSweepSpec object/axis/goal/seed as pure-IK v2",
        "early_stop_policy": "curves stop at the last physical sample; no extrapolation",
        "experiments": [],
    }
    manifest_path = destination / "manifest.json"
    _write_manifest(manifest_path, manifest)
    controllers: dict[str, LoadedController] = {}
    active: RotationSweepSpec | None = None
    try:
        for gripper in grippers:
            controllers[gripper] = controller_loader(
                artifacts_by_gripper[gripper], device=device
            )
        for index, spec in enumerate(spec_list, start=1):
            active = spec
            print(
                f"[{index}/{len(spec_list)}] {spec.episode_id}: deterministic TQC residual",
                flush=True,
            )
            summary = run_residual_rotation_sweep_episode(
                spec,
                artifacts=artifacts_by_gripper[spec.gripper],
                controller=controllers[spec.gripper],
                traces_dir=traces_dir,
                env_factory=env_factory,
            )
            manifest["experiments"].append(summary)
            manifest["completed_experiments"] = len(manifest["experiments"])
            _write_manifest(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "episode_id": active.episode_id if active is not None else None,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_manifest(manifest_path, manifest)
        raise
    finally:
        for controller in controllers.values():
            controller.close()
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    _write_manifest(manifest_path, manifest)
    return manifest


def _expand(
    values: Sequence[str], allowed: Sequence[str], label: str
) -> tuple[str, ...]:
    if tuple(values) == ("all",):
        return tuple(allowed)
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"unknown {label}: {', '.join(invalid)}")
    return tuple(dict.fromkeys(values))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic trained-residual Phase-3 sweeps at 0.325 m."
    )
    parser.add_argument("--gripper", choices=("g3", "g6", "both"), default="both")
    parser.add_argument("--g3-training-run", type=Path)
    parser.add_argument("--g6-training-run", type=Path)
    parser.add_argument("--objects", nargs="+", default=("all",))
    parser.add_argument("--axes", nargs="+", default=("all",))
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation")
        / datetime.now().strftime("rotation_residual_sweep_%Y%m%d_%H%M%S"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    grippers = ("g3", "g6") if args.gripper == "both" else (args.gripper,)
    training_paths = {
        "g3": args.g3_training_run,
        "g6": args.g6_training_run,
    }
    missing = [gripper for gripper in grippers if training_paths[gripper] is None]
    if missing:
        raise SystemExit(
            "missing training run argument(s): "
            + ", ".join(f"--{gripper}-training-run" for gripper in missing)
        )
    try:
        objects = _expand(args.objects, config.OBJECT_NAMES, "object")
        axes = _expand(args.axes, config.ROTATION_AXIS_SETS, "rotation axis set")
        specs = build_rotation_sweep_specs(
            grippers=grippers,
            objects=objects,
            axes=axes,
            max_angle_deg=MAX_SWEEP_ANGLE_DEG,
            seed=args.seed,
        )
        manifest = run_residual_rotation_sweep(
            specs=specs,
            training_runs={
                gripper: training_paths[gripper] for gripper in grippers
            },
            output_dir=args.output_dir,
            device=args.device,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"completed {len(manifest['experiments'])} trained-residual sweeps: "
        f"{_inside_project(args.output_dir)}"
    )


if __name__ == "__main__":
    main()
