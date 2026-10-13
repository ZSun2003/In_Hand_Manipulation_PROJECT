"""Deterministic Phase-2 evaluation of the final trained IK-residual policy.

The output trace is deliberately a strict superset of :mod:`g3g6.ik_control`
so that every pure-translation and coupled translation/rotation case can be
paired by its unchanged ``episode_id`` and ``seed``.  ``VecNormalize`` is used
only as a frozen observation transform; rollout execution stays on the raw
Gym environment so an early terminal sample is recorded before any auto-reset.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import config, trajectories
from .env import G3G6ManipulationEnv
from .ik_control import (
    RUN_SCHEMA as IK_RUN_SCHEMA,
    ExperimentSpec,
    _record_tick as _record_ik_tick,
    _summary as _ik_summary,
    add_route_distances,
    build_experiment_specs,
)
from .rotation_residual_sweep import (
    LoadedController,
    TrainingArtifacts,
    _normalized_policy_action,
    _pad_native,
    _validate_space_shape,
    load_training_artifacts as _load_full_training_artifacts,
)


TRACE_SCHEMA = "g3g6-translation-residual-trace-v1"
RUN_SCHEMA = "g3g6-translation-residual-run-v1"
CONTROLLER_KIND = "trained_tqc_ik_residual"
PAIRING_CONTRACT = "same ExperimentSpec episode_id/object/path/axes/goal/seed as pure IK"


def _inside_project(path: Path) -> Path:
    value = (
        (config.PROJECT_ROOT / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )
    try:
        value.relative_to(config.PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            "translation residual inputs and outputs must stay inside IK-residual"
        ) from exc
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_artifacts(run_dir: Path) -> TrainingArtifacts:
    """Validate the full final artifact and its exact Phase-2 task contract."""

    artifacts = _load_full_training_artifacts(run_dir)
    phase_tasks = artifacts.manifest.get("phase_tasks")
    if not isinstance(phase_tasks, Mapping):
        raise ValueError("training manifest must contain phase_tasks")
    phase_two = phase_tasks.get("2")
    expected = {
        "name": "translate",
        "target_grasp_plane_z_m": config.TARGET_GRASP_PLANE_Z_M,
        "objects_uniform": list(config.OBJECT_NAMES),
        "paths_uniform": list(trajectories.PATH_NAMES),
        "residual_activation": "after_preaction_tick_360",
    }
    if phase_two != expected:
        raise ValueError("training Phase-2 task contract differs from runtime")
    residual = artifacts.manifest.get("residual")
    activation = residual.get("activation_by_phase") if isinstance(residual, Mapping) else None
    if not isinstance(activation, Mapping) or activation.get("2") != (
        "after_preaction_tick_360"
    ):
        raise ValueError(
            "trained Phase-2 residual must be disabled through the pre-action prefix"
        )
    if not artifacts.residual_nominal_feedback:
        raise ValueError("trained Phase-2 residual must use feedback nominal IK")
    if not artifacts.residual_after_preaction_only:
        raise ValueError("trained Phase-2 residual prefix mask is not enabled")
    return artifacts


def load_controller(
    artifacts: TrainingArtifacts, *, device: str = "auto"
) -> LoadedController:
    """Load TQC with a frozen normalizer attached to a Phase-2 probe only."""

    from sb3_contrib import TQC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    probe = DummyVecEnv(
        [
            lambda: G3G6ManipulationEnv(
                gripper=artifacts.gripper,
                phase=config.Phase.TRANSLATE,
                object_name="cube",
                path_name="circle",
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
            model.observation_space,
            (artifacts.observation_dim,),
            "model observation",
        )
        _validate_space_shape(
            model.action_space, (artifacts.action_dim,), "model action"
        )
        _validate_space_shape(
            normalizer.observation_space,
            (artifacts.observation_dim,),
            "VecNormalize observation",
        )
        if bool(normalizer.training) or bool(normalizer.norm_reward):
            raise RuntimeError("VecNormalize was not frozen for evaluation")
    except BaseException:
        probe.close()
        raise
    return LoadedController(
        model=model, normalizer=normalizer, close_callback=probe.close
    )


def pair_id(spec: ExperimentSpec) -> str:
    axes = spec.rotation_axes or "none"
    goal = ",".join(f"{value:g}" for value in spec.rotation_goal_euler_deg)
    return (
        f"{spec.gripper}:{spec.object_name}:{spec.path_name}:{axes}:"
        f"seed={spec.seed}:goal={goal}"
    )


def _record_tick(
    spec: ExperimentSpec,
    env: G3G6ManipulationEnv,
    artifacts: TrainingArtifacts,
    info: dict[str, Any],
    raw_action: np.ndarray,
    *,
    terminated: bool,
) -> dict[str, Any]:
    """Record the pure-IK columns unchanged, then append RL diagnostics."""

    if int(info.get("phase", -1)) != int(config.Phase.TRANSLATE):
        raise RuntimeError("trained residual evaluation left Phase 2")
    expected_task = "translation" if spec.rotation_axes is None else "translate_rotate"
    # The environment names the non-coupled task after the enum (``translate``),
    # whereas ExperimentSpec calls that analysis case ``translation``.
    runtime_task = "translate" if spec.rotation_axes is None else "translate_rotate"
    if str(info.get("task")) != runtime_task:
        raise RuntimeError("trained residual evaluation task differs from its spec")
    if str(info.get("object")) != spec.object_name:
        raise RuntimeError("runtime object differs from its translation spec")
    if str(info.get("path")) != spec.path_name:
        raise RuntimeError("runtime path differs from its translation spec")
    runtime_axes = info.get("rotation_axes")
    if runtime_axes != spec.rotation_axes:
        raise RuntimeError("runtime rotation axes differ from the translation spec")
    runtime_goal = np.asarray(info.get("rotation_goal_euler_deg"), dtype=np.float64)
    if runtime_goal.shape != (3,) or not np.allclose(
        runtime_goal,
        np.asarray(spec.rotation_goal_euler_deg, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("runtime Euler goal differs from the translation spec")
    if str(info.get("control_mode")) != "residual_action":
        raise RuntimeError("trained residual evaluation did not use env.step(action)")
    if bool(info.get("residual_nominal_feedback")) is not True:
        raise RuntimeError("runtime nominal controller is not feedback IK")
    if bool(info.get("residual_after_preaction_only")) is not True:
        raise RuntimeError("runtime residual prefix mask differs from training")
    if not math.isclose(
        float(info.get("task_target_grasp_plane_z_m", float("nan"))),
        config.TARGET_GRASP_PLANE_Z_M,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("trained translation target height is not 0.360 m")

    native_dim = artifacts.action_dim
    raw = np.asarray(raw_action, dtype=np.float64).reshape(native_dim)
    requested = np.asarray(
        info["residual_requested_action"], dtype=np.float64
    ).reshape(native_dim)
    if not np.allclose(requested, raw, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("environment requested action differs from TQC prediction")
    executed = np.asarray(
        info["residual_executed_action"], dtype=np.float64
    ).reshape(native_dim)
    residual = np.asarray(info["actuator_residual"], dtype=np.float64).reshape(
        native_dim
    )
    if not (
        np.all(np.isfinite(raw))
        and np.all(np.isfinite(executed))
        and np.all(np.isfinite(residual))
    ):
        raise RuntimeError("environment emitted a non-finite residual diagnostic")

    command_tick = int(info["step"]) - 1
    action_tick = command_tick - config.PRE_ACTION_TICKS
    residual_active = bool(info["residual_active"])
    if residual_active != (action_tick >= 0):
        raise RuntimeError("residual pre-action mask changed")
    if not residual_active and (
        np.any(executed != 0.0) or np.any(residual != 0.0)
    ):
        raise RuntimeError("residual was applied before the translation action")

    row = _record_ik_tick(spec, env, info, terminated=terminated)
    row["schema"] = TRACE_SCHEMA
    row.update(
        {
            "controller": CONTROLLER_KIND,
            "pair_id": pair_id(spec),
            "source_ik_episode_id": spec.episode_id,
            "phase": int(config.Phase.TRANSLATE),
            "seed": int(spec.seed),
            "model_sha256": artifacts.model_sha256,
            "vec_normalize_sha256": artifacts.vec_normalize_sha256,
            "residual_nominal_feedback": 1,
            "residual_after_preaction_only": 1,
            "residual_active": int(residual_active),
            "policy_action_raw_l2": float(np.linalg.norm(raw)),
            "policy_action_raw_linf": float(np.max(np.abs(raw), initial=0.0)),
            "policy_action_executed_l2": float(np.linalg.norm(executed)),
            "policy_action_executed_linf": float(
                np.max(np.abs(executed), initial=0.0)
            ),
            "actuator_residual_l2": float(np.linalg.norm(residual)),
            "actuator_residual_linf": float(
                np.max(np.abs(residual), initial=0.0)
            ),
        }
    )
    padded_raw = _pad_native(raw, native_dim)
    padded_executed = _pad_native(executed, native_dim)
    padded_residual = _pad_native(residual, native_dim)
    for index in range(config.MAX_ACTION_DIM):
        row[f"policy_action_raw_{index}"] = float(padded_raw[index])
        row[f"policy_action_executed_{index}"] = float(padded_executed[index])
        row[f"actuator_residual_{index}"] = float(padded_residual[index])
    # ``expected_task`` documents the analysis vocabulary and makes a future
    # accidental change to ExperimentSpec.task observable here.
    if row["task"] != expected_task:
        raise RuntimeError("IK-compatible task label changed")
    return row


def _summary(
    spec: ExperimentSpec,
    artifacts: TrainingArtifacts,
    rows: list[dict[str, Any]],
    trace_path: Path,
) -> dict[str, Any]:
    result = _ik_summary(spec, rows, trace_path)
    final = rows[-1]
    result.update(
        {
            "pair_id": pair_id(spec),
            "source_ik_episode_id": spec.episode_id,
            "controller": CONTROLLER_KIND,
            "model_sha256": artifacts.model_sha256,
            "vec_normalize_sha256": artifacts.vec_normalize_sha256,
            "ended_early": str(final["termination_reason"]) != "horizon",
            "maximum_policy_action_raw_linf": max(
                float(row["policy_action_raw_linf"]) for row in rows
            ),
            "maximum_policy_action_executed_linf": max(
                float(row["policy_action_executed_linf"]) for row in rows
            ),
            "maximum_actuator_residual_linf": max(
                float(row["actuator_residual_linf"]) for row in rows
            ),
            "early_stop_policy": (
                "trace ends at the last physical sample; no extrapolation"
            ),
        }
    )
    return result


def run_residual_translation_episode(
    spec: ExperimentSpec,
    *,
    artifacts: TrainingArtifacts,
    controller: LoadedController,
    traces_dir: Path,
    position_feedback_gain: float = 1.0,
    orientation_feedback_gain: float = 4.0,
    max_position_correction_m: float = 0.015,
    max_angle_correction_deg: float = 45.0,
    env_factory: Callable[..., G3G6ManipulationEnv] = G3G6ManipulationEnv,
) -> dict[str, Any]:
    if spec.gripper != artifacts.gripper:
        raise ValueError("translation spec and residual model grippers differ")
    kwargs: dict[str, Any] = {}
    if spec.rotation_axes is not None:
        kwargs.update(
            coupled_rotation_axes=spec.rotation_axes,
            coupled_rotation_euler_deg=spec.rotation_goal_euler_deg,
        )
    env = env_factory(
        gripper=spec.gripper,
        phase=config.Phase.TRANSLATE,
        object_name=spec.object_name,
        path_name=spec.path_name,
        residual_nominal_feedback=True,
        residual_after_preaction_only=True,
        ik_position_feedback_gain=position_feedback_gain,
        ik_orientation_feedback_gain=orientation_feedback_gain,
        ik_max_position_correction_m=max_position_correction_m,
        ik_max_angle_correction_deg=max_angle_correction_deg,
        **kwargs,
    )
    rows: list[dict[str, Any]] = []
    try:
        observation, reset_info = env.reset(seed=spec.seed)
        if np.asarray(observation).shape != (artifacts.observation_dim,):
            raise RuntimeError("runtime observation width differs from training")
        if str(reset_info.get("object")) != spec.object_name:
            raise RuntimeError("reset object differs from the paired spec")
        if str(reset_info.get("path")) != spec.path_name:
            raise RuntimeError("reset path differs from the paired spec")
        terminated = truncated = False
        while not (terminated or truncated):
            raw_action = _normalized_policy_action(
                controller.model,
                controller.normalizer,
                observation,
                action_dim=artifacts.action_dim,
            )
            observation, _reward, terminated, truncated, info = env.step(raw_action)
            rows.append(
                _record_tick(
                    spec,
                    env,
                    artifacts,
                    info,
                    raw_action,
                    terminated=terminated or truncated,
                )
            )
    finally:
        env.close()
    if not rows:
        raise RuntimeError(f"residual translation produced no rows: {spec.episode_id}")

    add_route_distances(rows)
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"{spec.episode_id}.csv"
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


def load_matching_ik_specs(run_dirs: Sequence[Path]) -> list[ExperimentSpec]:
    """Load exact specs from pure-IK manifests, preserving their seeds."""

    specs: list[ExperimentSpec] = []
    seen: set[str] = set()
    for requested in run_dirs:
        source = _inside_project(Path(requested))
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"IK manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != IK_RUN_SCHEMA:
            raise ValueError(f"not a pure-IK run manifest: {manifest_path}")
        for item in manifest.get("experiments", []):
            axes_value = item.get("rotation_axes")
            axes = None if axes_value in (None, "none") else str(axes_value)
            if axes is not None and axes not in config.ROTATION_AXIS_SETS:
                raise ValueError(
                    f"invalid paired IK rotation axes in {manifest_path}: {axes!r}"
                )
            goal_raw = item.get("rotation_goal_euler_deg")
            try:
                goal_array = np.asarray(goal_raw, dtype=np.float64).reshape(3)
                object_name = str(item["object"])
                if object_name not in config.OBJECT_NAMES:
                    raise ValueError(f"unknown object: {object_name}")
                spec = ExperimentSpec(
                    gripper=config.GripperKind(str(item["gripper"])).value,
                    object_name=object_name,
                    path_name=trajectories.normalize_path_name(str(item["path"])),
                    rotation_axes=axes,
                    rotation_goal_euler_deg=tuple(float(value) for value in goal_array),
                    seed=int(item["seed"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid paired IK experiment in {manifest_path}"
                ) from exc
            if not np.all(np.isfinite(goal_array)):
                raise ValueError("paired IK Euler goal contains non-finite values")
            active = np.asarray(
                [axis in axes if axes is not None else False for axis in "xyz"]
            )
            if np.any(goal_array[~active] != 0.0):
                raise ValueError("paired IK goal activates an unselected Euler axis")
            if axes is None and np.any(goal_array != 0.0):
                raise ValueError("paired pure-translation goal must be zero")
            if str(item.get("task")) != spec.task:
                raise ValueError("paired IK task label does not match its fields")
            if str(item.get("episode_id")) != spec.episode_id:
                raise ValueError("paired IK episode_id does not match its fields")
            if spec.episode_id in seen:
                raise ValueError(f"duplicate paired IK episode: {spec.episode_id}")
            seen.add(spec.episode_id)
            specs.append(spec)
    if not specs:
        raise ValueError("paired IK manifests contain no experiments")
    return specs


def run_residual_translation_suite(
    *,
    specs: Sequence[ExperimentSpec],
    training_runs: Mapping[str, Path],
    output_dir: Path,
    device: str = "auto",
    position_feedback_gain: float = 1.0,
    orientation_feedback_gain: float = 4.0,
    max_position_correction_m: float = 0.015,
    max_angle_correction_deg: float = 45.0,
    pairing_sources: Sequence[Path] = (),
    artifact_loader: Callable[[Path], TrainingArtifacts] = load_training_artifacts,
    controller_loader: Callable[..., LoadedController] = load_controller,
    env_factory: Callable[..., G3G6ManipulationEnv] = G3G6ManipulationEnv,
) -> dict[str, Any]:
    spec_list = tuple(specs)
    if not spec_list:
        raise ValueError("at least one residual translation episode is required")
    episode_ids = [spec.episode_id for spec in spec_list]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("duplicate residual translation episode ids")
    feedback_values = np.asarray(
        (
            position_feedback_gain,
            orientation_feedback_gain,
            max_position_correction_m,
            max_angle_correction_deg,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(feedback_values)) or np.any(feedback_values < 0.0):
        raise ValueError("feedback gains and limits must be finite and non-negative")
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
            raise FileExistsError(
                f"residual translation destination is not empty: {destination}"
            )
    else:
        destination.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "trace_schema": TRACE_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "controller": CONTROLLER_KIND,
        "deterministic_policy": True,
        "normalization_frozen": True,
        "phase": int(config.Phase.TRANSLATE),
        "target_grasp_plane_z_m": config.TARGET_GRASP_PLANE_Z_M,
        "expected_experiments": len(spec_list),
        "completed_experiments": 0,
        "training_runs": {
            gripper: {
                "path": str(artifacts.run_dir),
                "schema": artifacts.manifest["schema"],
                "action_dim": artifacts.action_dim,
                "observation_dim": artifacts.observation_dim,
                "residual_nominal_feedback": True,
                "residual_after_preaction_only": True,
                "model_sha256": artifacts.model_sha256,
                "vec_normalize_sha256": artifacts.vec_normalize_sha256,
            }
            for gripper, artifacts in artifacts_by_gripper.items()
        },
        "pairing": PAIRING_CONTRACT,
        "pairing_sources": [str(_inside_project(Path(path))) for path in pairing_sources],
        "early_stop_policy": "traces stop at the last physical sample; no extrapolation",
        "feedback": {
            "enabled": True,
            "position_gain": float(position_feedback_gain),
            "orientation_gain": float(orientation_feedback_gain),
            "max_position_correction_m_per_axis": float(max_position_correction_m),
            "max_angle_correction_deg_per_axis": float(max_angle_correction_deg),
        },
        "target_route_distance_definition": (
            "cumulative target XY arc length over route"
        ),
        "actual_route_distance_definition": (
            "cumulative measured grasp-frame XY arc length over route"
        ),
        "height_definition": "object grasp-frame/body-origin world Z",
        "rotation_error_definition": "wrapped XYZ Euler of target^-1 times actual",
        "position_tolerance_m_per_axis": config.POSITION_TOLERANCE_M_PER_AXIS,
        "angle_tolerance_deg_per_axis": config.ANGLE_TOLERANCE_DEG_PER_AXIS,
        "experiments": [],
    }
    manifest_path = destination / "manifest.json"
    _write_manifest(manifest_path, manifest)
    traces_dir = destination / "traces"
    controllers: dict[str, LoadedController] = {}
    active: ExperimentSpec | None = None
    try:
        for gripper in grippers:
            controller = controller_loader(
                artifacts_by_gripper[gripper], device=device
            )
            if bool(getattr(controller.normalizer, "training", False)):
                raise RuntimeError("loaded VecNormalize is not frozen")
            if bool(getattr(controller.normalizer, "norm_reward", False)):
                raise RuntimeError("reward normalization must be disabled at evaluation")
            controllers[gripper] = controller
        for index, spec in enumerate(spec_list, start=1):
            active = spec
            print(
                f"[{index}/{len(spec_list)}] {spec.episode_id}: deterministic TQC residual",
                flush=True,
            )
            summary = run_residual_translation_episode(
                spec,
                artifacts=artifacts_by_gripper[spec.gripper],
                controller=controllers[spec.gripper],
                traces_dir=traces_dir,
                position_feedback_gain=position_feedback_gain,
                orientation_feedback_gain=orientation_feedback_gain,
                max_position_correction_m=max_position_correction_m,
                max_angle_correction_deg=max_angle_correction_deg,
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
        description=(
            "Run deterministic final-TQC residual evaluation for Phase-2 "
            "translation and coupled translation/rotation cases."
        )
    )
    parser.add_argument("--gripper", choices=("g3", "g6", "both"), default="both")
    parser.add_argument("--g3-training-run", type=Path)
    parser.add_argument("--g6-training-run", type=Path)
    parser.add_argument("--objects", nargs="+", default=("all",))
    parser.add_argument("--paths", nargs="+", default=("all",))
    parser.add_argument("--axes", nargs="+", default=("all",))
    parser.add_argument(
        "--suite", choices=("translation", "coupled", "full"), default="full"
    )
    parser.add_argument("--rotation-angle-deg", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument(
        "--match-ik-run",
        type=Path,
        nargs="+",
        help=(
            "use exact ExperimentSpecs from one or more pure-IK manifests; "
            "selection and seed-generation arguments are then ignored"
        ),
    )
    parser.add_argument("--position-feedback-gain", type=float, default=1.0)
    parser.add_argument("--orientation-feedback-gain", type=float, default=4.0)
    parser.add_argument("--max-position-correction-m", type=float, default=0.015)
    parser.add_argument("--max-angle-correction-deg", type=float, default=45.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation")
        / datetime.now().strftime("translation_residual_%Y%m%d_%H%M%S"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.match_ik_run:
            specs = load_matching_ik_specs(args.match_ik_run)
            pairing_sources = tuple(args.match_ik_run)
        else:
            grippers = (
                ("g3", "g6") if args.gripper == "both" else (args.gripper,)
            )
            objects = _expand(args.objects, config.OBJECT_NAMES, "object")
            paths = _expand(args.paths, trajectories.PATH_NAMES, "path")
            axes = _expand(args.axes, config.ROTATION_AXIS_SETS, "rotation axis set")
            specs = build_experiment_specs(
                grippers=grippers,
                objects=objects,
                paths=paths,
                axes=axes,
                suite=args.suite,
                rotation_angle_deg=args.rotation_angle_deg,
                seed=args.seed,
            )
            pairing_sources = ()
        requested_grippers = tuple(dict.fromkeys(spec.gripper for spec in specs))
        training_paths = {
            "g3": args.g3_training_run,
            "g6": args.g6_training_run,
        }
        missing = [
            gripper
            for gripper in requested_grippers
            if training_paths[gripper] is None
        ]
        if missing:
            raise ValueError(
                "missing training run argument(s): "
                + ", ".join(
                    f"--{gripper}-training-run" for gripper in missing
                )
            )
        manifest = run_residual_translation_suite(
            specs=specs,
            training_runs={
                gripper: training_paths[gripper]
                for gripper in requested_grippers
            },
            output_dir=args.output_dir,
            device=args.device,
            position_feedback_gain=args.position_feedback_gain,
            orientation_feedback_gain=args.orientation_feedback_gain,
            max_position_correction_m=args.max_position_correction_m,
            max_angle_correction_deg=args.max_angle_correction_deg,
            pairing_sources=pairing_sources,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"completed {len(manifest['experiments'])} trained-residual translation "
        f"episodes: {_inside_project(args.output_dir)}"
    )


if __name__ == "__main__":
    main()
