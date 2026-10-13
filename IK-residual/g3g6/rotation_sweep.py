"""Pure-IK in-place rotation sweep from zero to an explicit Euler goal."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import config
from .env import G3G6ManipulationEnv
from .rotations import (
    ROTATION_MOTION_TICKS,
    build_rotation_euler_schedule,
    validate_axis_set,
)


# Version 2 changes the physical task: Phase.ROTATE now runs at a dedicated
# 0.325 m grasp-plane height instead of inheriting the 0.360 m translation
# height.  Keeping the schema boundary explicit prevents old 0.360 m traces
# from being relabelled as the new task during re-analysis.
TRACE_SCHEMA = "g3g6-pure-ik-rotation-sweep-trace-v2"
RUN_SCHEMA = "g3g6-pure-ik-rotation-sweep-run-v2"
# Public alias retained for the CLI and callers, while the authoritative hard
# limit lives with the rest of the experiment contract.
MAX_SWEEP_ANGLE_DEG = config.ROTATION_SWEEP_ANGLE_MAX_DEG
EXPECTED_SWEEP_SAMPLES = ROTATION_MOTION_TICKS + 1  # exact 0-degree baseline


@dataclass(frozen=True)
class RotationSweepSpec:
    gripper: str
    object_name: str
    rotation_axes: str
    max_angle_deg: float
    seed: int

    def __post_init__(self) -> None:
        try:
            gripper = config.GripperKind(str(self.gripper)).value
        except ValueError as exc:
            raise ValueError(f"unknown gripper: {self.gripper!r}") from exc
        object_name = str(self.object_name)
        if object_name not in config.OBJECT_NAMES:
            raise ValueError(f"unknown object: {object_name!r}")
        axes = validate_axis_set(self.rotation_axes)
        maximum = float(self.max_angle_deg)
        if (
            not np.isfinite(maximum)
            or not config.ROTATION_ANGLE_MIN_DEG
            <= maximum
            <= config.ROTATION_SWEEP_ANGLE_MAX_DEG
        ):
            raise ValueError(
                "rotation sweep maximum must be inside "
                f"[{config.ROTATION_ANGLE_MIN_DEG}, "
                f"{config.ROTATION_SWEEP_ANGLE_MAX_DEG:g}] degrees"
            )
        try:
            seed = int(self.seed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("rotation sweep seed must be an integer") from exc

        object.__setattr__(self, "gripper", gripper)
        object.__setattr__(self, "object_name", object_name)
        object.__setattr__(self, "rotation_axes", axes)
        object.__setattr__(self, "max_angle_deg", maximum)
        object.__setattr__(self, "seed", seed)

    @property
    def episode_id(self) -> str:
        return f"{self.gripper}_{self.object_name}_rotation_sweep_{self.rotation_axes}"

    @property
    def goal_euler_deg(self) -> tuple[float, float, float]:
        return tuple(
            self.max_angle_deg if axis in self.rotation_axes else 0.0
            for axis in "xyz"
        )


def _inside_project(path: Path) -> Path:
    value = (
        (config.PROJECT_ROOT / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )
    try:
        value.relative_to(config.PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("rotation-sweep outputs must stay inside IK-residual") from exc
    return value


def _expand(
    values: Sequence[str], allowed: Sequence[str], label: str
) -> tuple[str, ...]:
    if tuple(values) == ("all",):
        return tuple(allowed)
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"unknown {label}: {', '.join(invalid)}")
    return tuple(dict.fromkeys(values))


def _validate_feedback_values(
    position_feedback_gain: float,
    orientation_feedback_gain: float,
    max_position_correction_m: float,
    max_angle_correction_deg: float,
) -> None:
    values = np.asarray(
        (
            position_feedback_gain,
            orientation_feedback_gain,
            max_position_correction_m,
            max_angle_correction_deg,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("feedback gains and limits must be finite and non-negative")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically replace the checkpoint manifest for interruption safety."""

    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def build_rotation_sweep_specs(
    *,
    grippers: Sequence[str],
    objects: Sequence[str],
    axes: Sequence[str],
    max_angle_deg: float,
    seed: int,
) -> list[RotationSweepSpec]:
    maximum = float(max_angle_deg)
    if (
        not np.isfinite(maximum)
        or not config.ROTATION_ANGLE_MIN_DEG <= maximum <= MAX_SWEEP_ANGLE_DEG
    ):
        raise ValueError(
            f"rotation sweep maximum must be inside "
            f"[{config.ROTATION_ANGLE_MIN_DEG}, {MAX_SWEEP_ANGLE_DEG:g}] degrees"
        )
    normalized_axes = tuple(validate_axis_set(axis_set) for axis_set in axes)
    specs: list[RotationSweepSpec] = []
    base_seed = int(seed)
    for gripper in grippers:
        for object_index, object_name in enumerate(objects):
            for axis_index, axis_set in enumerate(normalized_axes):
                # Reuse the same seed for the same object/axis pair across G3
                # and G6.  Gripper comparison must not be confounded by a
                # different randomized simulator reset.
                paired_seed = (
                    base_seed + object_index * len(normalized_axes) + axis_index
                )
                specs.append(
                    RotationSweepSpec(
                        gripper=str(gripper),
                        object_name=str(object_name),
                        rotation_axes=axis_set,
                        max_angle_deg=maximum,
                        seed=paired_seed,
                    )
                )
    return specs


def _record_tick(
    spec: RotationSweepSpec,
    env: G3G6ManipulationEnv,
    info: dict[str, Any],
    euler_schedule: np.ndarray,
    *,
    terminated: bool,
) -> dict[str, Any]:
    rotation_target_height_m = float(info["task_target_grasp_plane_z_m"])
    if (
        not np.isfinite(rotation_target_height_m)
        or not np.isclose(
            rotation_target_height_m,
            env.task_target_grasp_plane_z_m,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise RuntimeError(
            "rotation trace target height differs from the environment task height"
        )
    command_tick = int(info["step"]) - 1
    action_tick = command_tick - config.PRE_ACTION_TICKS
    # The post-lift state becomes the exact zero-degree reference only if the
    # episode survives long enough to enter the rotation phase.  A failure on
    # the final lift tick must not be presented as a started sweep.
    is_baseline = action_tick == -1 and not terminated
    is_motion = 0 <= action_tick < ROTATION_MOTION_TICKS
    is_hold = ROTATION_MOTION_TICKS <= action_tick < euler_schedule.shape[0]
    if is_baseline:
        commanded_angle = 0.0
    elif action_tick >= 0:
        schedule_index = min(action_tick, euler_schedule.shape[0] - 1)
        first_axis = "xyz".index(spec.rotation_axes[0])
        commanded_angle = float(euler_schedule[schedule_index, first_axis])
    else:
        commanded_angle = 0.0

    if is_baseline:
        segment = "reference_baseline"
    elif is_motion:
        segment = "rotation_motion"
    elif is_hold:
        segment = "terminal_hold"
    elif command_tick < config.APPROACH_TICKS:
        segment = "approach"
    elif command_tick < config.APPROACH_TICKS + config.CONTACT_TICKS:
        segment = "contact"
    else:
        segment = "lift"

    angle_error = np.asarray(info["angle_error_deg_per_axis"], dtype=np.float64)
    # The exact zero-angle target is the measured pose captured at the first
    # action tick.  Re-express the immediately preceding row as that reference
    # instead of comparing it with the pre-action world-identity target.
    if is_baseline:
        angle_error = np.zeros(3, dtype=np.float64)
    actual_position = np.asarray(
        info["object_grasp_frame_position_m"], dtype=np.float64
    )
    target_position = np.asarray(info["target_position_m"], dtype=np.float64)
    if is_baseline:
        # Match the position reference that Phase.ROTATE captures on the next
        # command: measured X/Y with the environment task's exact grasp-plane
        # height.  Do not reuse the 0.360 m translation/lift constant here.
        target_position = np.array(
            (
                actual_position[0],
                actual_position[1],
                rotation_target_height_m,
            ),
            dtype=np.float64,
        )
    row: dict[str, Any] = {
        "schema": TRACE_SCHEMA,
        "episode_id": spec.episode_id,
        "gripper": spec.gripper,
        "object": spec.object_name,
        "phase": int(info["phase"]),
        "task": str(info["task"]),
        "rotation_axes": spec.rotation_axes,
        "rotation_target_grasp_plane_z_m": rotation_target_height_m,
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
    for index, axis in enumerate("xyz"):
        row[f"angle_error_deg_{axis}"] = float(angle_error[index])
        row[f"actual_position_m_{axis}"] = float(actual_position[index])
        row[f"target_position_m_{axis}"] = float(target_position[index])
    return row


def run_rotation_sweep_episode(
    spec: RotationSweepSpec,
    *,
    traces_dir: Path,
    feedback: bool,
    position_feedback_gain: float,
    orientation_feedback_gain: float,
    max_position_correction_m: float,
    max_angle_correction_deg: float,
) -> dict[str, Any]:
    goal = np.asarray(spec.goal_euler_deg, dtype=np.float64)
    euler_schedule = build_rotation_euler_schedule(goal)
    if euler_schedule.shape != (config.ACTION_SCHEDULE_TICKS, 3):
        raise RuntimeError("rotation sweep schedule length diverged from the phase contract")
    env = G3G6ManipulationEnv(
        gripper=spec.gripper,
        phase=config.Phase.ROTATE,
        object_name=spec.object_name,
        rotation_axes=spec.rotation_axes,
        rotation_goal_max_deg=spec.max_angle_deg,
        ik_position_feedback_gain=position_feedback_gain,
        ik_orientation_feedback_gain=orientation_feedback_gain,
        ik_max_position_correction_m=max_position_correction_m,
        ik_max_angle_correction_deg=max_angle_correction_deg,
    )
    rows: list[dict[str, Any]] = []
    try:
        _reset_observation, reset_info = env.reset(
            seed=spec.seed,
            options={
                "rotation_axes": spec.rotation_axes,
                "rotation_euler_deg": goal,
            },
        )
        if (
            int(reset_info["phase"]) != int(config.Phase.ROTATE)
            or str(reset_info["task"]) != "rotate"
        ):
            raise RuntimeError("rotation sweep did not enter the Phase.ROTATE task")
        reset_target_height_m = float(
            reset_info["task_target_grasp_plane_z_m"]
        )
        if (
            not np.isfinite(reset_target_height_m)
            or not np.isclose(
                reset_target_height_m,
                env.task_target_grasp_plane_z_m,
                rtol=0.0,
                atol=1.0e-12,
            )
            or not np.isclose(
                reset_target_height_m,
                config.ROTATION_TARGET_GRASP_PLANE_Z_M,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise RuntimeError(
                "rotation sweep environment did not expose the 0.325 m task height"
            )
        terminated = truncated = False
        while not (terminated or truncated):
            _observation, _reward, terminated, truncated, info = env.step_ik(
                feedback=feedback
            )
            if int(info["phase"]) != int(config.Phase.ROTATE):
                raise RuntimeError("rotation sweep phase changed during the episode")
            row = _record_tick(
                spec,
                env,
                info,
                euler_schedule,
                terminated=terminated or truncated,
            )
            if row["max_abs_actuator_residual"] != 0.0:
                raise RuntimeError("pure IK produced a nonzero actuator residual")
            if row["nominal_applied_ctrl_linf"] != 0.0:
                raise RuntimeError("pure IK applied control differs from nominal IK")
            rows.append(row)
    finally:
        env.close()

    if not rows:
        raise RuntimeError(f"rotation sweep produced no trace rows: {spec.episode_id}")

    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"{spec.episode_id}.csv"
    temporary_trace_path = trace_path.with_suffix(".csv.tmp")
    with temporary_trace_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_trace_path.replace(trace_path)

    sweep_rows = [row for row in rows if bool(row["sweep_sample"])]
    motion_rows = [row for row in rows if bool(row["rotation_motion_sample"])]
    baseline_rows = [row for row in rows if bool(row["reference_baseline"])]
    hold_rows = [row for row in rows if bool(row["terminal_hold_sample"])]
    recorded_target_heights = {
        float(row["rotation_target_grasp_plane_z_m"]) for row in rows
    }
    if len(recorded_target_heights) != 1:
        raise RuntimeError(
            f"rotation task height changed during the episode: {spec.episode_id}"
        )
    rotation_target_height_m = recorded_target_heights.pop()
    # Baseline IK status belongs to the preceding lift command.  Rotation
    # reachability statistics therefore use motion rows only.
    projected = [row for row in motion_rows if not bool(row["ik_exact"])]
    final = rows[-1]
    reached_angle = (
        max(float(row["target_active_component_angle_deg"]) for row in sweep_rows)
        if sweep_rows
        else None
    )
    maximum_rotation_error = (
        max(float(row["rotation_error_l2_deg"]) for row in sweep_rows)
        if sweep_rows
        else None
    )
    minimum_object_height = (
        min(float(row["object_grasp_frame_height_m"]) for row in sweep_rows)
        if sweep_rows
        else None
    )
    minimum_tip_contacts = (
        min(int(row["tip_contact_count"]) for row in sweep_rows)
        if sweep_rows
        else None
    )
    minimum_projection_fraction = (
        min(float(row["ik_projection_fraction"]) for row in motion_rows)
        if motion_rows
        else None
    )
    completed_motion_sweep = bool(
        len(motion_rows) == ROTATION_MOTION_TICKS
        and reached_angle is not None
        and np.isclose(reached_angle, spec.max_angle_deg, rtol=0.0, atol=1.0e-12)
    )
    return {
        "episode_id": spec.episode_id,
        "trace": str(trace_path.relative_to(traces_dir.parent)),
        "gripper": spec.gripper,
        "object": spec.object_name,
        "rotation_axes": spec.rotation_axes,
        "rotation_target_grasp_plane_z_m": rotation_target_height_m,
        "goal_euler_deg": list(spec.goal_euler_deg),
        "max_angle_deg": spec.max_angle_deg,
        "seed": spec.seed,
        "ticks": len(rows),
        "sweep_samples": len(sweep_rows),
        "expected_sweep_samples": EXPECTED_SWEEP_SAMPLES,
        "baseline_recorded": len(baseline_rows) == 1,
        "motion_samples": len(motion_rows),
        "expected_motion_samples": ROTATION_MOTION_TICKS,
        "terminal_hold_samples": len(hold_rows),
        "expected_terminal_hold_samples": config.POST_ACTION_HOLD_TICKS,
        "sweep_started": bool(motion_rows),
        "completed_motion_sweep": completed_motion_sweep,
        "reached_angle_deg": reached_angle,
        "is_success": bool(final["is_success"]),
        "termination_reason": str(final["termination_reason"]),
        "maximum_rotation_error_l2_deg": maximum_rotation_error,
        "minimum_object_height_m": minimum_object_height,
        "first_projection_angle_deg": (
            float(projected[0]["target_active_component_angle_deg"])
            if projected
            else None
        ),
        "projected_sweep_ticks": len(projected),
        "minimum_ik_projection_fraction": minimum_projection_fraction,
        "minimum_tip_contacts": minimum_tip_contacts,
        "maximum_abs_actuator_residual": max(
            float(row["max_abs_actuator_residual"]) for row in rows
        ),
        "maximum_nominal_applied_ctrl_difference": max(
            float(row["nominal_applied_ctrl_linf"]) for row in rows
        ),
    }


def run_rotation_sweep(
    *,
    specs: Sequence[RotationSweepSpec],
    output_dir: Path,
    feedback: bool,
    position_feedback_gain: float,
    orientation_feedback_gain: float,
    max_position_correction_m: float,
    max_angle_correction_deg: float,
) -> dict[str, Any]:
    _validate_feedback_values(
        position_feedback_gain,
        orientation_feedback_gain,
        max_position_correction_m,
        max_angle_correction_deg,
    )
    spec_list = tuple(specs)
    if not spec_list:
        raise ValueError("at least one rotation sweep experiment is required")
    if not all(isinstance(spec, RotationSweepSpec) for spec in spec_list):
        raise TypeError("all rotation sweep specs must be RotationSweepSpec instances")
    episode_ids = [spec.episode_id for spec in spec_list]
    duplicate_ids = sorted(
        episode_id
        for episode_id in set(episode_ids)
        if episode_ids.count(episode_id) > 1
    )
    if duplicate_ids:
        raise ValueError(
            "duplicate rotation sweep episode ids: " + ", ".join(duplicate_ids)
        )

    destination = _inside_project(output_dir)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(f"run destination is not empty: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    traces_dir = destination / "traces"
    manifest: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "expected_experiments": len(spec_list),
        "completed_experiments": 0,
        "controller": "rate_limited_analytic_ik",
        "task": "in_place_rotation_sweep",
        "requires_training": False,
        "uses_learned_model": False,
        "uses_actuator_residual": False,
        "target_position_motion": "fixed",
        "phase": int(config.Phase.ROTATE),
        "phase_name": config.Phase.ROTATE.name.lower(),
        "rotation_goal_default_max_deg": config.ROTATION_ANGLE_MAX_DEG,
        "rotation_sweep_hard_max_deg": config.ROTATION_SWEEP_ANGLE_MAX_DEG,
        "rotation_target_grasp_plane_z_m": (
            config.ROTATION_TARGET_GRASP_PLANE_Z_M
        ),
        "seed_pairing": "same object/axis pair reuses its seed across grippers",
        "timing": {
            "pre_action_ticks": config.PRE_ACTION_TICKS,
            "rotation_motion_ticks": ROTATION_MOTION_TICKS,
            "terminal_hold_ticks": config.POST_ACTION_HOLD_TICKS,
            "episode_ticks": config.FORMAL_EPISODE_TICKS,
            "expected_sweep_samples_including_zero_baseline": (
                EXPECTED_SWEEP_SAMPLES
            ),
        },
        "angle_axis_definition": (
            "common scheduled Euler component theta for every active axis; "
            "x=(theta,0,0), xy=(theta,theta,0), xyz=(theta,theta,theta)"
        ),
        "rotation_error_definition": (
            "L2 norm of wrapped EulerXYZ(target^-1 * actual)"
        ),
        "height_definition": "object grasp-frame/body-origin world Z",
        "target_position_definition": (
            "measured post-lift X/Y with fixed Phase.ROTATE grasp-plane Z "
            f"of {config.ROTATION_TARGET_GRASP_PLANE_Z_M:.3f} m"
        ),
        "feedback": {
            "enabled": bool(feedback),
            "position_gain": float(position_feedback_gain),
            "orientation_gain": float(orientation_feedback_gain),
            "max_position_correction_m_per_axis": float(max_position_correction_m),
            "max_angle_correction_deg_per_axis": float(max_angle_correction_deg),
        },
        "experiments": [],
    }
    manifest_path = destination / "manifest.json"
    _write_manifest(manifest_path, manifest)
    active_spec: RotationSweepSpec | None = None
    try:
        for index, spec in enumerate(spec_list, start=1):
            active_spec = spec
            print(
                f"[{index}/{len(spec_list)}] {spec.episode_id}: "
                "pure IK rotation sweep",
                flush=True,
            )
            summary = run_rotation_sweep_episode(
                spec,
                traces_dir=traces_dir,
                feedback=feedback,
                position_feedback_gain=position_feedback_gain,
                orientation_feedback_gain=orientation_feedback_gain,
                max_position_correction_m=max_position_correction_m,
                max_angle_correction_deg=max_angle_correction_deg,
            )
            manifest["experiments"].append(summary)
            manifest["completed_experiments"] = len(manifest["experiments"])
            _write_manifest(manifest_path, manifest)
            reached = summary["reached_angle_deg"]
            reached_label = "not-started" if reached is None else f"{reached:.3f}deg"
            print(
                "    "
                f"reached={reached_label} "
                f"success={summary['is_success']} "
                f"projected_ticks={summary['projected_sweep_ticks']} "
                f"termination={summary['termination_reason']}",
                flush=True,
            )
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["completed_experiments"] = len(manifest["experiments"])
        manifest["failure"] = {
            "episode_id": active_spec.episode_id if active_spec is not None else None,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    _write_manifest(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a no-training pure-IK in-place rotation sweep."
    )
    parser.add_argument("--gripper", choices=("g3", "g6", "both"), default="both")
    parser.add_argument("--objects", nargs="+", default=("all",))
    parser.add_argument("--axes", nargs="+", default=("all",))
    parser.add_argument("--max-angle-deg", type=float, default=MAX_SWEEP_ANGLE_DEG)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument(
        "--feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--position-feedback-gain", type=float, default=1.0)
    parser.add_argument("--orientation-feedback-gain", type=float, default=4.0)
    parser.add_argument("--max-position-correction-m", type=float, default=0.015)
    parser.add_argument("--max-angle-correction-deg", type=float, default=45.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation")
        / datetime.now().strftime("rotation_sweep_%Y%m%d_%H%M%S"),
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
    if not np.all(np.isfinite(feedback_values)) or any(
        value < 0 for value in feedback_values
    ):
        raise SystemExit("feedback gains and limits must be finite and non-negative")
    grippers = ("g3", "g6") if args.gripper == "both" else (args.gripper,)
    try:
        objects = _expand(args.objects, config.OBJECT_NAMES, "object")
        axes = _expand(args.axes, config.ROTATION_AXIS_SETS, "rotation axis set")
        specs = build_rotation_sweep_specs(
            grippers=grippers,
            objects=objects,
            axes=axes,
            max_angle_deg=args.max_angle_deg,
            seed=args.seed,
        )
        manifest = run_rotation_sweep(
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
    print(f"completed {len(manifest['experiments'])} rotation sweeps: {destination}")


if __name__ == "__main__":
    main()
