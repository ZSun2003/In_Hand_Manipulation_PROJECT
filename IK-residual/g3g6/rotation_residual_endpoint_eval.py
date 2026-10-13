"""Independent 28-sample endpoint evaluation for the final rotation policies.

Each object/axis condition receives 28 fresh Phase-3 episodes.  Magnitudes are
deterministically stratified as 6/14/8 draws from the weighted 0--30, 31--60,
and 61--90 degree bins.  G3 and G6 use exactly the same seed, signed Euler
goal, object, axis set, and sample index.  No point is extracted from a
continuous 90-degree sweep.

The runner is shardable for parallel CPU execution and checkpoint-resumable.
It stores one summary row per independent episode rather than multi-million-row
tick traces; endpoint success and early termination remain the environment's
physical terminal results.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import config
from .env import G3G6ManipulationEnv
from .rotation_residual_sweep import (
    CONTROLLER_KIND,
    LoadedController,
    TrainingArtifacts,
    _normalized_policy_action,
    load_controller,
    load_training_artifacts,
)
from .rotations import (
    ROTATION_MOTION_TICKS,
    build_rotation_euler_schedule,
    sample_goal_euler_deg,
)


SHARD_SCHEMA = "g3g6-rotation-residual-endpoint-28x-shard-v1"
MERGED_SCHEMA = "g3g6-rotation-residual-endpoint-28x-run-v1"
SAMPLING_SCHEMA = "weighted-0-90-stratified-6-14-8-v1"
SAMPLES_PER_CONDITION = 28
BIN_QUOTAS = (6, 14, 8)
BIN_RANGES_DEG = tuple(config.WEIGHTED_ROTATION_MAGNITUDE_BINS_DEG)
EVAL_AXIS_ORDER = ("x", "y", "z", "xz", "xy", "yz", "xyz")
OBJECT_ORDER = tuple(config.OBJECT_NAMES)
GRIPPER_ORDER = ("g3", "g6")
DEFAULT_BASE_SEED = 20_260_812
DEFAULT_PARTS_ROOT = Path("evaluation/rotation_residual_endpoint_28x_parts")
DEFAULT_MERGED_OUTPUT = Path("evaluation/rotation_residual_endpoint_28x_latest")


@dataclass(frozen=True)
class EndpointEvalSpec:
    gripper: str
    object_name: str
    rotation_axes: str
    sample_index: int
    magnitude_bin: int
    seed: int
    goal_euler_deg: tuple[float, float, float]

    @property
    def goal_magnitude_deg(self) -> float:
        active = [
            abs(float(self.goal_euler_deg["xyz".index(axis)]))
            for axis in self.rotation_axes
        ]
        if not active or len(set(active)) != 1:
            raise ValueError("endpoint-evaluation active axes must share one magnitude")
        return active[0]

    @property
    def pair_id(self) -> str:
        return (
            f"{self.object_name}:{self.rotation_axes}:"
            f"sample={self.sample_index:02d}"
        )

    @property
    def episode_id(self) -> str:
        return (
            f"{self.gripper}_{self.object_name}_rotation_endpoint_"
            f"{self.rotation_axes}_sample{self.sample_index:02d}"
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
        raise ValueError("endpoint-evaluation paths must stay inside IK-residual") from exc
    return value


def magnitude_bin(magnitude_deg: float) -> int:
    value = float(magnitude_deg)
    for index, (lower, upper) in enumerate(BIN_RANGES_DEG):
        if lower <= value <= upper:
            return index
    raise ValueError(f"rotation magnitude is outside the weighted bins: {value}")


def sample_bin_for_index(sample_index: int) -> int:
    index = int(sample_index)
    if not 0 <= index < SAMPLES_PER_CONDITION:
        raise ValueError(f"sample index must be in [0, {SAMPLES_PER_CONDITION - 1}]")
    if index < BIN_QUOTAS[0]:
        return 0
    if index < BIN_QUOTAS[0] + BIN_QUOTAS[1]:
        return 1
    return 2


def _paired_seed_and_goal(
    *,
    object_index: int,
    axis_index: int,
    sample_index: int,
    base_seed: int,
) -> tuple[int, tuple[float, float, float]]:
    target_bin = sample_bin_for_index(sample_index)
    group_index = object_index * len(EVAL_AXIS_ORDER) + axis_index
    first_candidate = int(base_seed) + group_index * 1_000_000 + sample_index * 10_000
    axes = EVAL_AXIS_ORDER[axis_index]
    for offset in range(10_000):
        seed = first_candidate + offset
        _selected, goal = sample_goal_euler_deg(
            np.random.default_rng(seed),
            axes,
            profile=config.WEIGHTED_ROTATION_SAMPLING_PROFILE,
        )
        active_magnitude = abs(float(goal["xyz".index(axes[0])]))
        if magnitude_bin(active_magnitude) == target_bin:
            return seed, tuple(float(value) for value in goal)
    raise RuntimeError("could not find a deterministic seed for the requested bin")


def build_endpoint_specs(
    gripper: str,
    *,
    base_seed: int = DEFAULT_BASE_SEED,
) -> tuple[EndpointEvalSpec, ...]:
    normalized = config.GripperKind(gripper).value
    specs: list[EndpointEvalSpec] = []
    for object_index, object_name in enumerate(OBJECT_ORDER):
        for axis_index, axes in enumerate(EVAL_AXIS_ORDER):
            for sample_index in range(SAMPLES_PER_CONDITION):
                seed, goal = _paired_seed_and_goal(
                    object_index=object_index,
                    axis_index=axis_index,
                    sample_index=sample_index,
                    base_seed=base_seed,
                )
                spec = EndpointEvalSpec(
                    gripper=normalized,
                    object_name=object_name,
                    rotation_axes=axes,
                    sample_index=sample_index,
                    magnitude_bin=sample_bin_for_index(sample_index),
                    seed=seed,
                    goal_euler_deg=goal,
                )
                if magnitude_bin(spec.goal_magnitude_deg) != spec.magnitude_bin:
                    raise AssertionError("stratified sample landed in the wrong bin")
                specs.append(spec)
    expected = len(OBJECT_ORDER) * len(EVAL_AXIS_ORDER) * SAMPLES_PER_CONDITION
    if len(specs) != expected or len({spec.episode_id for spec in specs}) != expected:
        raise AssertionError("endpoint-evaluation grid is incomplete or duplicated")
    return tuple(specs)


def paired_sampling_contract(base_seed: int = DEFAULT_BASE_SEED) -> None:
    g3 = build_endpoint_specs("g3", base_seed=base_seed)
    g6 = build_endpoint_specs("g6", base_seed=base_seed)
    for first, second in zip(g3, g6):
        if (
            first.pair_id != second.pair_id
            or first.seed != second.seed
            or first.goal_euler_deg != second.goal_euler_deg
        ):
            raise AssertionError("G3/G6 endpoint samples are not exactly paired")


def _command_magnitude(
    schedule: np.ndarray,
    axes: str,
    action_tick: int,
) -> float:
    if action_tick < 0:
        return 0.0
    schedule_index = min(action_tick, schedule.shape[0] - 1)
    return abs(float(schedule[schedule_index, "xyz".index(axes[0])]))


def run_endpoint_episode(
    spec: EndpointEvalSpec,
    *,
    artifacts: TrainingArtifacts,
    controller: LoadedController,
    env_factory: Callable[..., G3G6ManipulationEnv] = G3G6ManipulationEnv,
) -> dict[str, Any]:
    if artifacts.gripper != spec.gripper:
        raise ValueError("endpoint spec and policy grippers differ")
    environment = env_factory(
        gripper=spec.gripper,
        phase=config.Phase.ROTATE,
        object_name=spec.object_name,
        rotation_axes=spec.rotation_axes,
        rotation_sampling_profile=artifacts.rotation_sampling_profile,
        residual_nominal_feedback=artifacts.residual_nominal_feedback,
        residual_after_preaction_only=artifacts.residual_after_preaction_only,
    )
    started = time.perf_counter()
    try:
        observation, reset_info = environment.reset(
            seed=spec.seed,
            options={"rotation_axes": spec.rotation_axes},
        )
        sampled_goal = tuple(
            float(value) for value in np.asarray(
                reset_info["rotation_goal_euler_deg"], dtype=np.float64
            )
        )
        if sampled_goal != spec.goal_euler_deg:
            raise RuntimeError(
                "environment did not reproduce the paired deterministic goal: "
                f"expected {spec.goal_euler_deg}, got {sampled_goal}"
            )
        if reset_info.get("rotation_sampling_profile") != (
            config.WEIGHTED_ROTATION_SAMPLING_PROFILE
        ):
            raise RuntimeError("endpoint evaluation did not use weighted_0_90")
        schedule = build_rotation_euler_schedule(sampled_goal)
        terminated = truncated = False
        reached = 0.0
        motion_samples = 0
        projected_motion_ticks = 0
        maximum_position_linf_m = 0.0
        maximum_euler_linf_deg = 0.0
        minimum_height_m = math.inf
        info: dict[str, Any] = dict(reset_info)
        while not (terminated or truncated):
            action = _normalized_policy_action(
                controller.model,
                controller.normalizer,
                observation,
                action_dim=artifacts.action_dim,
            )
            observation, _reward, terminated, truncated, info = environment.step(action)
            if str(info.get("control_mode")) != "residual_action":
                raise RuntimeError("endpoint evaluation left residual-action control")
            command_tick = int(info["step"]) - 1
            action_tick = command_tick - config.PRE_ACTION_TICKS
            commanded = _command_magnitude(schedule, spec.rotation_axes, action_tick)
            reached = max(reached, commanded)
            if 0 <= action_tick < ROTATION_MOTION_TICKS:
                motion_samples += 1
                position_error = np.asarray(
                    info["position_error_m_per_axis"], dtype=np.float64
                )
                angle_error = np.asarray(
                    info["angle_error_deg_per_axis"], dtype=np.float64
                )
                maximum_position_linf_m = max(
                    maximum_position_linf_m,
                    float(np.max(np.abs(position_error), initial=0.0)),
                )
                maximum_euler_linf_deg = max(
                    maximum_euler_linf_deg,
                    float(np.max(np.abs(angle_error), initial=0.0)),
                )
                minimum_height_m = min(
                    minimum_height_m,
                    float(np.asarray(info["object_grasp_frame_position_m"])[2]),
                )
                if float(info["ik_projection_fraction"]) < 1.0 - 1.0e-12:
                    projected_motion_ticks += 1

        endpoint_position = np.asarray(
            info["position_error_m_per_axis"], dtype=np.float64
        )
        endpoint_angle = np.asarray(
            info["angle_error_deg_per_axis"], dtype=np.float64
        )
        contacts = int(info["tip_contact_count"])
        reason = str(info["termination_reason"])
        environment_success = bool(info.get("is_success", False))
        independently_recomputed_success = bool(
            reason == "horizon"
            and np.all(
                np.abs(endpoint_position)
                <= config.POSITION_TOLERANCE_M_PER_AXIS
            )
            and np.all(
                np.abs(endpoint_angle)
                <= config.ANGLE_TOLERANCE_DEG_PER_AXIS
            )
            and contacts >= config.REQUIRED_TIP_CONTACTS
        )
        if environment_success != independently_recomputed_success:
            raise RuntimeError("environment and recomputed strict success disagree")
        if not math.isfinite(minimum_height_m):
            minimum_height_m = float(
                np.asarray(info["object_grasp_frame_position_m"])[2]
            )
        return {
            "pair_id": spec.pair_id,
            "episode_id": spec.episode_id,
            "controller": CONTROLLER_KIND,
            "gripper": spec.gripper,
            "object": spec.object_name,
            "rotation_axes": spec.rotation_axes,
            "sample_index": spec.sample_index,
            "magnitude_bin": spec.magnitude_bin,
            "magnitude_bin_lower_deg": BIN_RANGES_DEG[spec.magnitude_bin][0],
            "magnitude_bin_upper_deg": BIN_RANGES_DEG[spec.magnitude_bin][1],
            "seed": spec.seed,
            "goal_euler_deg_x": sampled_goal[0],
            "goal_euler_deg_y": sampled_goal[1],
            "goal_euler_deg_z": sampled_goal[2],
            "goal_magnitude_deg": spec.goal_magnitude_deg,
            "ticks": int(info["step"]),
            "motion_samples": motion_samples,
            "reached_command_magnitude_deg": reached,
            "completed_motion": int(
                motion_samples == ROTATION_MOTION_TICKS
                and math.isclose(
                    reached, spec.goal_magnitude_deg, rel_tol=0.0, abs_tol=1.0e-9
                )
            ),
            "ended_early": int(reason != "horizon"),
            "termination_reason": reason,
            "is_success": int(environment_success),
            "object_drop": int(reason == "object_dropped"),
            "endpoint_position_error_m_x": float(endpoint_position[0]),
            "endpoint_position_error_m_y": float(endpoint_position[1]),
            "endpoint_position_error_m_z": float(endpoint_position[2]),
            "endpoint_euler_error_deg_x": float(endpoint_angle[0]),
            "endpoint_euler_error_deg_y": float(endpoint_angle[1]),
            "endpoint_euler_error_deg_z": float(endpoint_angle[2]),
            "endpoint_position_linf_mm": 1000.0
            * float(np.max(np.abs(endpoint_position), initial=0.0)),
            "endpoint_euler_linf_deg": float(
                np.max(np.abs(endpoint_angle), initial=0.0)
            ),
            "endpoint_tip_contact_count": contacts,
            "maximum_motion_position_linf_mm": 1000.0
            * maximum_position_linf_m,
            "maximum_motion_euler_linf_deg": maximum_euler_linf_deg,
            "minimum_motion_object_height_mm": 1000.0 * minimum_height_m,
            "projected_motion_ticks": projected_motion_ticks,
            "model_sha256": artifacts.model_sha256,
            "vec_normalize_sha256": artifacts.vec_normalize_sha256,
            "wall_time_s": time.perf_counter() - started,
        }
    finally:
        environment.close()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if rows:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        temporary.write_text("", encoding="utf-8")
    temporary.replace(path)


def _shard_specs(
    specs: Sequence[EndpointEvalSpec], shard_index: int, num_shards: int
) -> tuple[EndpointEvalSpec, ...]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard index must be inside [0, num_shards)")
    return tuple(
        spec for index, spec in enumerate(specs) if index % num_shards == shard_index
    )


def _validate_resume_artifacts(
    manifest: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
    artifacts: TrainingArtifacts,
) -> None:
    """Prevent a resumed shard from mixing policies or normalizers."""

    expected = {
        "training_run": str(artifacts.run_dir),
        "model_sha256": artifacts.model_sha256,
        "vec_normalize_sha256": artifacts.vec_normalize_sha256,
    }
    for key, value in expected.items():
        actual = manifest.get(key)
        if actual is not None and actual != value:
            raise ValueError(
                f"existing shard {key} mismatch: expected {value!r}, got {actual!r}"
            )
        if experiments and actual is None:
            raise ValueError(f"resumed non-empty shard is missing {key}")
    for row in experiments:
        if row.get("model_sha256") != artifacts.model_sha256:
            raise ValueError("resumed shard contains a case from another model")
        if row.get("vec_normalize_sha256") != artifacts.vec_normalize_sha256:
            raise ValueError("resumed shard contains a case from another normalizer")


def _validate_row_against_spec(
    row: Mapping[str, Any], spec: EndpointEvalSpec
) -> None:
    expected: Mapping[str, Any] = {
        "episode_id": spec.episode_id,
        "pair_id": spec.pair_id,
        "gripper": spec.gripper,
        "object": spec.object_name,
        "rotation_axes": spec.rotation_axes,
        "sample_index": spec.sample_index,
        "magnitude_bin": spec.magnitude_bin,
        "seed": spec.seed,
        "goal_euler_deg_x": spec.goal_euler_deg[0],
        "goal_euler_deg_y": spec.goal_euler_deg[1],
        "goal_euler_deg_z": spec.goal_euler_deg[2],
        "goal_magnitude_deg": spec.goal_magnitude_deg,
    }
    mismatches = {
        key: (value, row.get(key))
        for key, value in expected.items()
        if row.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"endpoint row does not match rebuilt spec {spec.episode_id}: {mismatches}"
        )


def run_endpoint_shard(
    *,
    gripper: str,
    training_run: Path,
    output_dir: Path,
    shard_index: int,
    num_shards: int,
    base_seed: int = DEFAULT_BASE_SEED,
    device: str = "cpu",
    limit: int | None = None,
    artifact_loader: Callable[[Path], TrainingArtifacts] = load_training_artifacts,
    controller_loader: Callable[..., LoadedController] = load_controller,
    env_factory: Callable[..., G3G6ManipulationEnv] = G3G6ManipulationEnv,
) -> Mapping[str, Any]:
    normalized = config.GripperKind(gripper).value
    all_specs = build_endpoint_specs(normalized, base_seed=base_seed)
    selected = _shard_specs(all_specs, int(shard_index), int(num_shards))
    production_contract = limit is None
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit must be positive")
        selected = selected[: int(limit)]
    destination = _inside_project(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    cases_path = destination / "cases.csv"
    expected_ids = [spec.episode_id for spec in selected]
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "schema": SHARD_SCHEMA,
            "gripper": normalized,
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
            "base_seed": int(base_seed),
            "expected_episode_ids": expected_ids,
        }
        mismatch = {
            key: (expected, manifest.get(key))
            for key, expected in required.items()
            if manifest.get(key) != expected
        }
        if mismatch:
            raise ValueError(f"existing shard manifest contract mismatch: {mismatch}")
        experiments = list(manifest.get("experiments", []))
    else:
        manifest = {
            "schema": SHARD_SCHEMA,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "running",
            "controller": CONTROLLER_KIND,
            "deterministic_policy": True,
            "normalization_frozen": True,
            "phase": int(config.Phase.ROTATE),
            "rotation_target_grasp_plane_z_m": (
                config.ROTATION_TARGET_GRASP_PLANE_Z_M
            ),
            "sampling_schema": SAMPLING_SCHEMA,
            "samples_per_object_axis": SAMPLES_PER_CONDITION,
            "bin_quotas": list(BIN_QUOTAS),
            "bin_ranges_deg": [list(value) for value in BIN_RANGES_DEG],
            "axis_order": list(EVAL_AXIS_ORDER),
            "object_order": list(OBJECT_ORDER),
            "gripper": normalized,
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
            "base_seed": int(base_seed),
            "production_contract": production_contract,
            "full_grid_experiments_per_gripper": len(all_specs),
            "expected_experiments": len(selected),
            "expected_episode_ids": expected_ids,
            "completed_experiments": 0,
            "training_run": str(_inside_project(training_run)),
            "experiments": [],
        }
        experiments = []
        _atomic_json(manifest_path, manifest)
    completed = {str(row["episode_id"]) for row in experiments}
    if not completed.issubset(set(expected_ids)):
        raise ValueError("shard contains an episode outside its deterministic spec")
    controller: LoadedController | None = None
    active: EndpointEvalSpec | None = None
    try:
        artifacts = artifact_loader(Path(training_run))
        if artifacts.gripper != normalized:
            raise ValueError("training run belongs to the other gripper")
        _validate_resume_artifacts(manifest, experiments, artifacts)
        expected_by_id = {spec.episode_id: spec for spec in selected}
        for row in experiments:
            _validate_row_against_spec(row, expected_by_id[str(row["episode_id"])])
        manifest["status"] = "running"
        manifest["training_run"] = str(artifacts.run_dir)
        manifest["model_sha256"] = artifacts.model_sha256
        manifest["vec_normalize_sha256"] = artifacts.vec_normalize_sha256
        _atomic_json(manifest_path, manifest)
        controller = controller_loader(artifacts, device=device)
        for spec in selected:
            if spec.episode_id in completed:
                continue
            active = spec
            ordinal = len(experiments) + 1
            print(
                f"[{ordinal}/{len(selected)}] {spec.episode_id} "
                f"goal={spec.goal_euler_deg}",
                flush=True,
            )
            result = run_endpoint_episode(
                spec,
                artifacts=artifacts,
                controller=controller,
                env_factory=env_factory,
            )
            experiments.append(result)
            completed.add(spec.episode_id)
            manifest["experiments"] = experiments
            manifest["completed_experiments"] = len(experiments)
            _atomic_json(manifest_path, manifest)
            _atomic_csv(cases_path, experiments)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "episode_id": active.episode_id if active else None,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _atomic_json(manifest_path, manifest)
        raise
    finally:
        if controller is not None:
            controller.close()
    manifest["status"] = "complete"
    manifest.pop("failure", None)
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    manifest["completed_experiments"] = len(experiments)
    _atomic_json(manifest_path, manifest)
    _atomic_csv(cases_path, experiments)
    return manifest


def _read_complete_shards(parts_root: Path) -> list[tuple[Path, Mapping[str, Any]]]:
    root = _inside_project(parts_root)
    manifests: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted(root.glob("*/*/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != SHARD_SCHEMA:
            continue
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete endpoint shard: {path}")
        if payload.get("production_contract") is not True:
            continue
        manifests.append((path, payload))
    return manifests


def merge_endpoint_shards(
    *,
    parts_root: Path = DEFAULT_PARTS_ROOT,
    output_dir: Path = DEFAULT_MERGED_OUTPUT,
) -> Mapping[str, Any]:
    shards = _read_complete_shards(parts_root)
    if not shards:
        raise ValueError("no complete production endpoint shards were found")
    base_seeds = {int(payload["base_seed"]) for _path, payload in shards}
    if len(base_seeds) != 1:
        raise ValueError(f"endpoint shards use multiple base seeds: {base_seeds}")
    artifact_provenance: dict[str, dict[str, str]] = {}
    for gripper in GRIPPER_ORDER:
        selected = [
            payload for _path, payload in shards if payload.get("gripper") == gripper
        ]
        shard_counts = {int(payload["num_shards"]) for payload in selected}
        if len(shard_counts) != 1:
            raise ValueError(f"{gripper} shards use inconsistent shard counts")
        shard_count = next(iter(shard_counts), 0)
        if {int(payload["shard_index"]) for payload in selected} != set(
            range(shard_count)
        ):
            raise ValueError(f"{gripper} shard-index coverage is incomplete")
        provenance: dict[str, str] = {}
        for field in ("training_run", "model_sha256", "vec_normalize_sha256"):
            values = {str(payload.get(field)) for payload in selected}
            if len(values) != 1 or "None" in values:
                raise ValueError(
                    f"{gripper} shards mix endpoint provenance for {field}: {values}"
                )
            provenance[field] = next(iter(values))
        artifact_provenance[gripper] = provenance
    rows: list[dict[str, Any]] = []
    for path, payload in shards:
        model_hash = payload.get("model_sha256")
        vec_hash = payload.get("vec_normalize_sha256")
        if not isinstance(model_hash, str) or not isinstance(vec_hash, str):
            raise ValueError(f"endpoint shard lacks artifact hashes: {path}")
        shard_rows = [dict(row) for row in payload.get("experiments", [])]
        for row in shard_rows:
            if row.get("model_sha256") != model_hash:
                raise ValueError(f"shard case/model hash mismatch: {path}")
            if row.get("vec_normalize_sha256") != vec_hash:
                raise ValueError(f"shard case/normalizer hash mismatch: {path}")
        rows.extend(shard_rows)
    if len({str(row["episode_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate endpoint episode across shards")
    expected_per_gripper = (
        len(OBJECT_ORDER) * len(EVAL_AXIS_ORDER) * SAMPLES_PER_CONDITION
    )
    expected_total = len(GRIPPER_ORDER) * expected_per_gripper
    counts = {
        gripper: sum(str(row["gripper"]) == gripper for row in rows)
        for gripper in GRIPPER_ORDER
    }
    if len(rows) != expected_total or any(
        count != expected_per_gripper for count in counts.values()
    ):
        raise ValueError(
            f"endpoint grid incomplete: total={len(rows)}/{expected_total}, "
            f"by_gripper={counts}"
        )
    rebuilt_specs = {
        spec.episode_id: spec
        for gripper in GRIPPER_ORDER
        for spec in build_endpoint_specs(gripper, base_seed=next(iter(base_seeds)))
    }
    actual_ids = {str(row["episode_id"]) for row in rows}
    if actual_ids != set(rebuilt_specs):
        raise ValueError(
            "merged endpoint episode IDs differ from the full rebuilt 1,568-case grid"
        )
    for row in rows:
        _validate_row_against_spec(row, rebuilt_specs[str(row["episode_id"])])
    for gripper in GRIPPER_ORDER:
        for object_name in OBJECT_ORDER:
            for axes in EVAL_AXIS_ORDER:
                condition = [
                    row
                    for row in rows
                    if str(row["gripper"]) == gripper
                    and str(row["object"]) == object_name
                    and str(row["rotation_axes"]) == axes
                ]
                bin_counts = tuple(
                    sum(int(row["magnitude_bin"]) == index for row in condition)
                    for index in range(3)
                )
                if len(condition) != SAMPLES_PER_CONDITION or bin_counts != BIN_QUOTAS:
                    raise ValueError(
                        "endpoint stratification changed for "
                        f"{gripper}/{object_name}/{axes}: "
                        f"n={len(condition)}, bins={bin_counts}"
                    )
    paired: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        paired.setdefault(str(row["pair_id"]), []).append(row)
    if len(paired) != expected_per_gripper:
        raise ValueError("paired endpoint key count is incomplete")
    for pair_id, group in paired.items():
        if {str(row["gripper"]) for row in group} != set(GRIPPER_ORDER):
            raise ValueError(f"missing G3/G6 member for {pair_id}")
        reference = group[0]
        for candidate in group[1:]:
            for field in (
                "object",
                "rotation_axes",
                "sample_index",
                "magnitude_bin",
                "seed",
                "goal_euler_deg_x",
                "goal_euler_deg_y",
                "goal_euler_deg_z",
                "goal_magnitude_deg",
            ):
                if candidate[field] != reference[field]:
                    raise ValueError(f"paired sample mismatch for {pair_id}/{field}")
    axis_rank = {value: index for index, value in enumerate(EVAL_AXIS_ORDER)}
    object_rank = {value: index for index, value in enumerate(OBJECT_ORDER)}
    gripper_rank = {value: index for index, value in enumerate(GRIPPER_ORDER)}
    rows.sort(
        key=lambda row: (
            axis_rank[str(row["rotation_axes"])],
            object_rank[str(row["object"])],
            int(row["sample_index"]),
            gripper_rank[str(row["gripper"])],
        )
    )
    destination = _inside_project(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"merged endpoint destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    success_by_gripper_axis = {
        gripper: {
            axes: sum(
                str(row["gripper"]) == gripper
                and str(row["rotation_axes"]) == axes
                and bool(int(row["is_success"]))
                for row in rows
            )
            for axes in EVAL_AXIS_ORDER
        }
        for gripper in GRIPPER_ORDER
    }
    manifest: dict[str, Any] = {
        "schema": MERGED_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "complete",
        "controller": CONTROLLER_KIND,
        "deterministic_policy": True,
        "normalization_frozen": True,
        "phase": int(config.Phase.ROTATE),
        "rotation_target_grasp_plane_z_m": config.ROTATION_TARGET_GRASP_PLANE_Z_M,
        "sampling_schema": SAMPLING_SCHEMA,
        "sampling_profile": config.WEIGHTED_ROTATION_SAMPLING_PROFILE,
        "samples_per_object_axis_gripper": SAMPLES_PER_CONDITION,
        "attempts_per_axis_gripper": len(OBJECT_ORDER) * SAMPLES_PER_CONDITION,
        "bin_quotas_per_object_axis_gripper": list(BIN_QUOTAS),
        "bin_quota_percentages_per_object_axis_gripper": [
            100.0 * value / SAMPLES_PER_CONDITION for value in BIN_QUOTAS
        ],
        "bin_allocation_note": (
            "28 samples cannot realize 20/50/30 percent exactly; 6/14/8 is "
            "the nearest integer stratification (21.4/50.0/28.6 percent), "
            "applied separately to every object-axis-gripper condition"
        ),
        "conditional_draw_note": (
            "paired seeds are deterministically rejection-selected by target bin; "
            "the environment's unchanged weighted_0_90 sampler then draws the "
            "integer magnitude and independent active-axis signs within that bin"
        ),
        "bin_ranges_deg": [list(value) for value in BIN_RANGES_DEG],
        "axis_order": list(EVAL_AXIS_ORDER),
        "object_order": list(OBJECT_ORDER),
        "gripper_order": list(GRIPPER_ORDER),
        "base_seed": next(iter(base_seeds)),
        "expected_experiments": expected_total,
        "completed_experiments": len(rows),
        "paired_conditions": len(paired),
        "pairing": (
            "G3 and G6 use identical object, axis set, sample index, seed, "
            "signed Euler goal, and magnitude bin"
        ),
        "training_artifacts": artifact_provenance,
        "independent_episode_contract": True,
        "continuous_sweep_points_used": False,
        "goal_sampling_audit": {
            "attempts_at_0_to_9_deg": sum(
                float(row["goal_magnitude_deg"]) <= 9.0 for row in rows
            ),
            "attempts_at_exactly_0_deg": sum(
                float(row["goal_magnitude_deg"]) == 0.0 for row in rows
            ),
            "low_magnitude_samples_filtered": False,
            "vertical_jitter_authorized": False,
        },
        "strict_success_definition": (
            "horizon without physical failure; endpoint position and every "
            "wrapped Euler component within 5 mm/5 deg; at least four tip contacts"
        ),
        "source_shards": [str(path.parent) for path, _payload in shards],
        "success_by_gripper_axis": success_by_gripper_axis,
        "experiments": rows,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    _atomic_csv(destination / "cases.csv", rows)
    _atomic_json(destination / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-shard", help="run or resume one evaluation shard")
    run.add_argument("--gripper", choices=GRIPPER_ORDER, required=True)
    run.add_argument("--training-run", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--num-shards", type=int, required=True)
    run.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    run.add_argument("--device", default="cpu")
    run.add_argument("--limit", type=int)
    merge = subparsers.add_parser("merge", help="validate and merge all production shards")
    merge.add_argument("--parts-root", type=Path, default=DEFAULT_PARTS_ROOT)
    merge.add_argument("--output-dir", type=Path, default=DEFAULT_MERGED_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    if arguments.command == "run-shard":
        result = run_endpoint_shard(
            gripper=arguments.gripper,
            training_run=arguments.training_run,
            output_dir=arguments.output_dir,
            shard_index=arguments.shard_index,
            num_shards=arguments.num_shards,
            base_seed=arguments.base_seed,
            device=arguments.device,
            limit=arguments.limit,
        )
    else:
        result = merge_endpoint_shards(
            parts_root=arguments.parts_root,
            output_dir=arguments.output_dir,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
