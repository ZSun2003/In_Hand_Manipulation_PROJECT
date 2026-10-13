"""Pure analytic-IK experiment runner with per-tick trace recording."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import config, trajectories
from .env import G3G6ManipulationEnv
from .math3d import (
    quat_conjugate,
    quat_multiply,
    quat_to_euler_xyz_deg,
)


TRACE_SCHEMA = "g3g6-pure-ik-trace-v1"
RUN_SCHEMA = "g3g6-pure-ik-run-v1"
ROUTE_BASE_ACTION_TICK = trajectories.PHASE_BOUNDS["route"][0] - 1
ROUTE_LAST_ACTION_TICK = trajectories.PHASE_BOUNDS["route"][1] - 1


@dataclass(frozen=True)
class ExperimentSpec:
    gripper: str
    object_name: str
    path_name: str
    rotation_axes: str | None
    rotation_goal_euler_deg: tuple[float, float, float]
    seed: int

    @property
    def task(self) -> str:
        return "translation" if self.rotation_axes is None else "translate_rotate"

    @property
    def episode_id(self) -> str:
        suffix = "translation" if self.rotation_axes is None else f"rot_{self.rotation_axes}"
        return f"{self.gripper}_{self.object_name}_{self.path_name}_{suffix}"


def _inside_project(path: Path) -> Path:
    value = (config.PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        value.relative_to(config.PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("IK outputs must stay inside the IK-residual project") from exc
    return value


def _segment_for_command_tick(command_tick: int) -> str:
    if command_tick < config.APPROACH_TICKS:
        return "approach"
    if command_tick < config.APPROACH_TICKS + config.CONTACT_TICKS:
        return "contact"
    if command_tick < config.PRE_ACTION_TICKS:
        return "lift"
    action_tick = command_tick - config.PRE_ACTION_TICKS
    if 0 <= action_tick < trajectories.TOTAL_TICKS:
        return trajectories.phase_for_tick(action_tick)
    return "terminal"


def quaternion_geodesic_error_deg(
    actual_wxyz: Iterable[float], target_wxyz: Iterable[float]
) -> float:
    actual = np.asarray(tuple(actual_wxyz), dtype=np.float64).reshape(4)
    target = np.asarray(tuple(target_wxyz), dtype=np.float64).reshape(4)
    actual /= np.linalg.norm(actual)
    target /= np.linalg.norm(target)
    dot = float(np.clip(abs(np.dot(actual, target)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _relative_euler_deg(quaternion: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return quat_to_euler_xyz_deg(
        quat_multiply(quat_conjugate(reference), quaternion)
    )


def _xyz_columns(prefix: str, value: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(value), dtype=np.float64).reshape(3)
    return {f"{prefix}_{axis}": float(array[index]) for index, axis in enumerate("xyz")}


def _quat_columns(prefix: str, value: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(value), dtype=np.float64).reshape(4)
    return {
        f"{prefix}_{component}": float(array[index])
        for index, component in enumerate(("w", "x", "y", "z"))
    }


def _record_tick(
    spec: ExperimentSpec,
    env: G3G6ManipulationEnv,
    info: dict[str, Any],
    *,
    terminated: bool,
) -> dict[str, Any]:
    command_tick = int(info["step"]) - 1
    action_tick = command_tick - config.PRE_ACTION_TICKS
    actual_position = np.asarray(info["object_grasp_frame_position_m"], dtype=np.float64)
    target_position = np.asarray(info["target_position_m"], dtype=np.float64)
    command_position = np.asarray(info["ik_command_position_m"], dtype=np.float64)
    actual_quaternion = np.asarray(info["object_quaternion_wxyz"], dtype=np.float64)
    target_quaternion = np.asarray(info["target_quaternion_wxyz"], dtype=np.float64)
    command_quaternion = np.asarray(
        info["ik_command_quaternion_wxyz"], dtype=np.float64
    )
    reference_quaternion = np.asarray(
        info["action_reference_quaternion_wxyz"], dtype=np.float64
    )
    position_error = np.asarray(info["position_error_m_per_axis"], dtype=np.float64)
    angle_error = np.asarray(info["angle_error_deg_per_axis"], dtype=np.float64)
    actual_euler = _relative_euler_deg(actual_quaternion, reference_quaternion)
    target_euler = _relative_euler_deg(target_quaternion, reference_quaternion)
    command_euler = _relative_euler_deg(command_quaternion, reference_quaternion)

    row: dict[str, Any] = {
        "schema": TRACE_SCHEMA,
        "episode_id": spec.episode_id,
        "gripper": spec.gripper,
        "object": spec.object_name,
        "path": spec.path_name,
        "task": spec.task,
        "rotation_axes": spec.rotation_axes or "none",
        "step": int(info["step"]),
        "command_tick": command_tick,
        "action_tick": action_tick,
        "time_s": float((command_tick + 1) * config.CONTROL_DT_S),
        "segment": _segment_for_command_tick(command_tick),
        "target_route_distance_m": 0.0,
        "actual_route_distance_m": 0.0,
        "object_grasp_frame_height_m": float(actual_position[2]),
        "position_error_norm_m": float(np.linalg.norm(position_error)),
        "angle_error_linf_deg": float(np.max(np.abs(angle_error))),
        "orientation_geodesic_error_deg": quaternion_geodesic_error_deg(
            actual_quaternion, target_quaternion
        ),
        "tip_contact_count": int(info["tip_contact_count"]),
        "ik_exact": int(bool(info["ik_exact"])),
        "ik_projection_fraction": float(info["ik_projection_fraction"]),
        "max_abs_actuator_residual": float(info["max_abs_actuator_residual"]),
        "nominal_applied_ctrl_linf": float(
            np.max(np.abs(env.nominal_ctrl - env.applied_ctrl), initial=0.0)
        ),
        "terminated": int(terminated),
        "is_success": int(bool(info.get("is_success", False))),
        "termination_reason": str(info["termination_reason"]),
    }
    row.update(_xyz_columns("actual_position_m", actual_position))
    row.update(_xyz_columns("target_position_m", target_position))
    row.update(_xyz_columns("ik_command_position_m", command_position))
    row.update(_xyz_columns("position_error_m", position_error))
    row.update(_xyz_columns("angle_error_deg", angle_error))
    row.update(_xyz_columns("actual_euler_deg", actual_euler))
    row.update(_xyz_columns("target_euler_deg", target_euler))
    row.update(_xyz_columns("ik_command_euler_deg", command_euler))
    row.update(_quat_columns("actual_quaternion", actual_quaternion))
    row.update(_quat_columns("target_quaternion", target_quaternion))
    row.update(_quat_columns("ik_command_quaternion", command_quaternion))
    return row


def add_route_distances(rows: list[dict[str, Any]]) -> None:
    """Add cumulative target/actual XY route lengths in-place."""

    if not rows:
        return
    base_index = next(
        (
            index
            for index, row in enumerate(rows)
            if int(row["action_tick"]) >= ROUTE_BASE_ACTION_TICK
        ),
        None,
    )
    if base_index is None:
        return

    target_distance = 0.0
    actual_distance = 0.0
    previous_target = np.array(
        (
            rows[base_index]["target_position_m_x"],
            rows[base_index]["target_position_m_y"],
        ),
        dtype=np.float64,
    )
    previous_actual = np.array(
        (
            rows[base_index]["actual_position_m_x"],
            rows[base_index]["actual_position_m_y"],
        ),
        dtype=np.float64,
    )
    for index in range(base_index, len(rows)):
        row = rows[index]
        action_tick = int(row["action_tick"])
        current_target = np.array(
            (row["target_position_m_x"], row["target_position_m_y"]),
            dtype=np.float64,
        )
        current_actual = np.array(
            (row["actual_position_m_x"], row["actual_position_m_y"]),
            dtype=np.float64,
        )
        if ROUTE_BASE_ACTION_TICK < action_tick <= ROUTE_LAST_ACTION_TICK:
            target_distance += float(np.linalg.norm(current_target - previous_target))
            actual_distance += float(np.linalg.norm(current_actual - previous_actual))
        row["target_route_distance_m"] = target_distance
        row["actual_route_distance_m"] = actual_distance
        previous_target = current_target
        previous_actual = current_actual


def _route_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if ROUTE_BASE_ACTION_TICK <= int(row["action_tick"]) <= ROUTE_LAST_ACTION_TICK
    ]


def _summary(
    spec: ExperimentSpec,
    rows: list[dict[str, Any]],
    trace_path: Path,
) -> dict[str, Any]:
    final = rows[-1]
    route = _route_rows(rows) or rows

    def maximum(columns: Sequence[str]) -> list[float]:
        return [max(abs(float(row[column])) for row in route) for column in columns]

    def rms(columns: Sequence[str]) -> list[float]:
        return [
            float(np.sqrt(np.mean([float(row[column]) ** 2 for row in route])))
            for column in columns
        ]

    position_columns = [f"position_error_m_{axis}" for axis in "xyz"]
    angle_columns = [f"angle_error_deg_{axis}" for axis in "xyz"]
    return {
        "episode_id": spec.episode_id,
        "trace": str(trace_path),
        "gripper": spec.gripper,
        "object": spec.object_name,
        "path": spec.path_name,
        "task": spec.task,
        "rotation_axes": spec.rotation_axes,
        "rotation_goal_euler_deg": list(spec.rotation_goal_euler_deg),
        "seed": spec.seed,
        "ticks": len(rows),
        "is_success": bool(final["is_success"]),
        "termination_reason": str(final["termination_reason"]),
        "endpoint_position_error_m_per_axis": [
            float(final[column]) for column in position_columns
        ],
        "endpoint_angle_error_deg_per_axis": [
            float(final[column]) for column in angle_columns
        ],
        "route_max_abs_position_error_m_per_axis": maximum(position_columns),
        "route_rms_position_error_m_per_axis": rms(position_columns),
        "route_max_abs_angle_error_deg_per_axis": maximum(angle_columns),
        "route_rms_angle_error_deg_per_axis": rms(angle_columns),
        "route_max_geodesic_error_deg": max(
            float(row["orientation_geodesic_error_deg"]) for row in route
        ),
        "route_min_tip_contacts": min(int(row["tip_contact_count"]) for row in route),
        "ik_projected_ticks": sum(not bool(row["ik_exact"]) for row in rows),
        "minimum_ik_projection_fraction": min(
            float(row["ik_projection_fraction"]) for row in rows
        ),
        "target_route_distance_m": float(route[-1]["target_route_distance_m"]),
        "actual_route_distance_m": float(route[-1]["actual_route_distance_m"]),
        "maximum_abs_actuator_residual": max(
            float(row["max_abs_actuator_residual"]) for row in rows
        ),
        "maximum_nominal_applied_ctrl_difference": max(
            float(row["nominal_applied_ctrl_linf"]) for row in rows
        ),
    }


def run_episode(
    spec: ExperimentSpec,
    *,
    traces_dir: Path,
    feedback: bool,
    position_feedback_gain: float,
    orientation_feedback_gain: float,
    max_position_correction_m: float,
    max_angle_correction_deg: float,
) -> dict[str, Any]:
    environment_kwargs: dict[str, Any] = {}
    if spec.rotation_axes is not None:
        environment_kwargs.update(
            coupled_rotation_axes=spec.rotation_axes,
            coupled_rotation_euler_deg=spec.rotation_goal_euler_deg,
        )
    env = G3G6ManipulationEnv(
        gripper=spec.gripper,
        phase=config.Phase.TRANSLATE,
        object_name=spec.object_name,
        path_name=spec.path_name,
        ik_position_feedback_gain=position_feedback_gain,
        ik_orientation_feedback_gain=orientation_feedback_gain,
        ik_max_position_correction_m=max_position_correction_m,
        ik_max_angle_correction_deg=max_angle_correction_deg,
        **environment_kwargs,
    )
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=spec.seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _observation, _reward, terminated, truncated, info = env.step_ik(
                feedback=feedback
            )
            row = _record_tick(spec, env, info, terminated=terminated or truncated)
            if row["max_abs_actuator_residual"] != 0.0:
                raise RuntimeError("pure IK produced a nonzero actuator residual")
            if row["nominal_applied_ctrl_linf"] != 0.0:
                raise RuntimeError("pure IK applied control differs from nominal IK")
            rows.append(row)
    finally:
        env.close()

    add_route_distances(rows)
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"{spec.episode_id}.csv"
    with trace_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return _summary(spec, rows, trace_path.relative_to(traces_dir.parent))


def _expand(values: Sequence[str], allowed: Sequence[str], label: str) -> tuple[str, ...]:
    if tuple(values) == ("all",):
        return tuple(allowed)
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"unknown {label}: {', '.join(invalid)}")
    return tuple(dict.fromkeys(values))


def build_experiment_specs(
    *,
    grippers: Sequence[str],
    objects: Sequence[str],
    paths: Sequence[str],
    axes: Sequence[str],
    suite: str,
    rotation_angle_deg: float,
    seed: int,
) -> list[ExperimentSpec]:
    if not config.ROTATION_ANGLE_MIN_DEG <= abs(rotation_angle_deg) <= config.ROTATION_ANGLE_MAX_DEG:
        raise ValueError("rotation angle magnitude must be inside [10, 45] degrees")
    include_translation = suite in ("translation", "full")
    include_coupled = suite in ("coupled", "full")
    specs: list[ExperimentSpec] = []
    episode_seed = int(seed)
    for gripper in grippers:
        for object_name in objects:
            for path_name in paths:
                if include_translation:
                    specs.append(
                        ExperimentSpec(
                            gripper=gripper,
                            object_name=object_name,
                            path_name=path_name,
                            rotation_axes=None,
                            rotation_goal_euler_deg=(0.0, 0.0, 0.0),
                            seed=episode_seed,
                        )
                    )
                    episode_seed += 1
                if include_coupled:
                    for axis_set in axes:
                        goal = tuple(
                            float(rotation_angle_deg) if axis in axis_set else 0.0
                            for axis in "xyz"
                        )
                        specs.append(
                            ExperimentSpec(
                                gripper=gripper,
                                object_name=object_name,
                                path_name=path_name,
                                rotation_axes=axis_set,
                                rotation_goal_euler_deg=goal,
                                seed=episode_seed,
                            )
                        )
                        episode_seed += 1
    return specs


def run_suite(
    *,
    specs: Sequence[ExperimentSpec],
    output_dir: Path,
    feedback: bool,
    position_feedback_gain: float,
    orientation_feedback_gain: float,
    max_position_correction_m: float,
    max_angle_correction_deg: float,
) -> dict[str, Any]:
    destination = _inside_project(output_dir)
    if (destination / "manifest.json").exists():
        raise FileExistsError(f"run already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    traces_dir = destination / "traces"
    manifest: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "controller": "rate_limited_analytic_ik",
        "requires_training": False,
        "uses_learned_model": False,
        "uses_actuator_residual": False,
        "feedback": {
            "enabled": bool(feedback),
            "position_gain": float(position_feedback_gain),
            "orientation_gain": float(orientation_feedback_gain),
            "max_position_correction_m_per_axis": float(max_position_correction_m),
            "max_angle_correction_deg_per_axis": float(max_angle_correction_deg),
        },
        "target_route_distance_definition": "cumulative target XY arc length over route",
        "actual_route_distance_definition": "cumulative measured grasp-frame XY arc length over route",
        "height_definition": "object grasp-frame/body-origin world Z",
        "rotation_error_definition": "wrapped XYZ Euler of target^-1 times actual",
        "position_tolerance_m_per_axis": config.POSITION_TOLERANCE_M_PER_AXIS,
        "angle_tolerance_deg_per_axis": config.ANGLE_TOLERANCE_DEG_PER_AXIS,
        "experiments": [],
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] {spec.episode_id}: pure IK",
            flush=True,
        )
        summary = run_episode(
            spec,
            traces_dir=traces_dir,
            feedback=feedback,
            position_feedback_gain=position_feedback_gain,
            orientation_feedback_gain=orientation_feedback_gain,
            max_position_correction_m=max_position_correction_m,
            max_angle_correction_deg=max_angle_correction_deg,
        )
        manifest["experiments"].append(summary)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "    "
            f"success={summary['is_success']} "
            f"projected_ticks={summary['ik_projected_ticks']} "
            f"termination={summary['termination_reason']}",
            flush=True,
        )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run G3/G6 manipulation with analytic IK only; no training or learned-policy files."
    )
    parser.add_argument("--gripper", choices=("g3", "g6", "both"), default="g3")
    parser.add_argument("--objects", nargs="+", default=("cube",))
    parser.add_argument("--paths", nargs="+", default=("circle",))
    parser.add_argument("--axes", nargs="+", default=("all",))
    parser.add_argument(
        "--suite", choices=("translation", "coupled", "full"), default="full"
    )
    parser.add_argument("--rotation-angle-deg", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument(
        "--feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use bounded object-pose feedback before each IK solve",
    )
    parser.add_argument("--position-feedback-gain", type=float, default=1.0)
    parser.add_argument("--orientation-feedback-gain", type=float, default=4.0)
    parser.add_argument("--max-position-correction-m", type=float, default=0.015)
    parser.add_argument("--max-angle-correction-deg", type=float, default=45.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation")
        / datetime.now().strftime("pure_ik_%Y%m%d_%H%M%S"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    feedback_values = (
        args.position_feedback_gain,
        args.orientation_feedback_gain,
        args.max_position_correction_m,
        args.max_angle_correction_deg,
    )
    if not np.all(np.isfinite(feedback_values)) or any(value < 0 for value in feedback_values):
        raise SystemExit("feedback gains and limits must be finite and non-negative")
    grippers = ("g3", "g6") if args.gripper == "both" else (args.gripper,)
    try:
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
        manifest = run_suite(
            specs=specs,
            output_dir=args.output_dir,
            feedback=args.feedback,
            position_feedback_gain=args.position_feedback_gain,
            orientation_feedback_gain=args.orientation_feedback_gain,
            max_position_correction_m=args.max_position_correction_m,
            max_angle_correction_deg=args.max_angle_correction_deg,
        )
        destination = _inside_project(args.output_dir)
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"completed {len(manifest['experiments'])} pure-IK episodes: {destination}")


if __name__ == "__main__":
    main()
