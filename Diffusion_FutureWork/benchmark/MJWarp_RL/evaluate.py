"""Strict paired IK-only versus IK+Diffusion evaluation on native MJWarp."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import contract
from .curriculum import (
    PHASE,
    REGION_ALL,
    rotation_angle_bounds_deg,
    sample_translation,
    translation_path_length_m,
    translation_path_step_m,
    translation_path_waypoints_m,
)
from .diffusion_policy import DiffusionPolicy, load_policy_checkpoint
from .diffusion_runtime import (
    BatchedDiffusionController,
    BatchedObservationHistory,
)
from .env import OBSERVATION_DIM, action_mask_from_observation, make_env
from .hpc_config import (
    DiffusionSettings,
    EliteSearchSettings,
    HPCConfig,
)
from .platform_adapter import PROJECT_ROOT, build_platform_spec
from .sharded_env import ordered_shard_manifest
from .runtime_fingerprint import runtime_code_fingerprint_sha256


SCHEMA = "mjwarp-task-bound-paired-ik-diffusion-eval-v6"
TRAINING_SCHEMA = (
    "native-mjwarp-sharded-certified-ik-diffusion-training-v5"
)
SOURCE_REGION_INDEX = {
    REGION_ALL: 0,
}
FRUIT_MANIFEST = PROJECT_ROOT / "fruit_capsules_5mm.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    """Normalize tuple/list differences across JSON and Torch metadata."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _success_tolerances() -> dict[str, dict[str, float]]:
    return {
        contract.TASK_TRANSLATE: {
            "position_axis_abs_tolerance_m": (
                contract.TRANSLATION_POSITION_AXIS_ABS_TOLERANCE_M
            ),
            "euler_axis_abs_tolerance_deg": (
                contract.TRANSLATION_EULER_AXIS_ABS_TOLERANCE_DEG
            ),
        },
        contract.TASK_ROTATE: {
            "position_axis_abs_tolerance_m": (
                contract.ROTATION_POSITION_AXIS_ABS_TOLERANCE_M
            ),
            "euler_axis_abs_tolerance_deg": (
                contract.ROTATION_EULER_AXIS_ABS_TOLERANCE_DEG
            ),
        },
    }


def _case_success_tolerance(object_kind: str, task: str) -> dict[str, float]:
    contract.validate_object(object_kind)
    task_tolerances = _success_tolerances()[task]
    return {
        "position_axis_abs_tolerance_m": task_tolerances[
            "position_axis_abs_tolerance_m"
        ],
        "euler_axis_abs_tolerance_deg": task_tolerances[
            "euler_axis_abs_tolerance_deg"
        ],
    }


def _torch_checkpoint(torch: Any, path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Diffusion checkpoint must contain a dictionary")
    return checkpoint


def _checkpoint_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Diffusion checkpoint is missing metadata")
    return dict(metadata)


def _config_from_mapping(raw_config: dict[str, Any]) -> HPCConfig:
    """Restore only dataclass inputs, ignoring serialized derived fields."""

    raw = dict(raw_config)
    for derived in (
        "action_ticks",
        "episode_length",
        "control_ticks",
        "remainder_worlds",
        "groups_per_vector_batch",
        "object_world_counts",
        "object_group_counts",
        "remainder_object_world_counts",
        "maximum_demo_episodes",
        "task_index",
    ):
        raw.pop(derived, None)
    elite = EliteSearchSettings(**dict(raw.pop("elite", {})))
    diffusion = DiffusionSettings(**dict(raw.pop("diffusion", {})))
    config = HPCConfig(elite=elite, diffusion=diffusion, **raw)
    config.validate()
    return config


def _checkpoint_trained_task(
    checkpoint: Mapping[str, Any],
    *,
    requested_task: str | None,
) -> str:
    """Infer one immutable task from a checkpoint, rejecting mixed bindings."""

    metadata = _checkpoint_metadata(checkpoint)
    trained_task = contract.validate_task(str(metadata.get("trained_task")))
    if metadata.get("task") != trained_task:
        raise ValueError("policy task and trained_task mismatch")
    expected_index = contract.TASKS.index(trained_task)
    if metadata.get("task_index") != expected_index:
        raise ValueError("policy task_index mismatch")
    raw_config = metadata.get("training_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("policy is missing its task-bound training config")
    checkpoint_config = _config_from_mapping(dict(raw_config))
    if (
        checkpoint_config.task != trained_task
        or checkpoint_config.task_index != expected_index
    ):
        raise ValueError("policy training config task binding mismatch")
    if requested_task is not None:
        selected = contract.validate_task(requested_task)
        if selected != trained_task:
            raise ValueError(
                "requested evaluation task differs from checkpoint trained_task"
            )
    return trained_task


def _validate_task_binding(
    record: Mapping[str, Any],
    *,
    label: str,
    config: HPCConfig,
    platform_spec: Any,
) -> None:
    """Require record, config, and platform to describe one identical task."""

    expected_task = contract.validate_task(config.task)
    expected_index = contract.TASKS.index(expected_task)
    expected_root = float(platform_spec.root_height_m)
    platform_expected = {
        "task": expected_task,
        "task_index": expected_index,
        "root_height_m": expected_root,
    }
    for name, wanted in platform_expected.items():
        if getattr(platform_spec, name, None) != wanted:
            raise ValueError(f"platform/config task binding mismatch for {name}")
    expected_record = {
        "platform": platform_spec.platform,
        "trained_task": expected_task,
        "task": expected_task,
        "task_index": expected_index,
        "root_height_m": expected_root,
    }
    for name, wanted in expected_record.items():
        if record.get(name) != wanted:
            raise ValueError(f"{label} task binding mismatch for {name}")


def _load_training_config(
    manifest_path: Path | None,
    *,
    checkpoint: dict[str, Any],
    model_path: Path,
    platform_spec: Any,
) -> HPCConfig:
    metadata = _checkpoint_metadata(checkpoint)
    checkpoint_config = metadata.get("training_config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Diffusion checkpoint is missing training_config")
    checkpoint_run_config = _config_from_mapping(dict(checkpoint_config))
    _validate_task_binding(
        metadata,
        label="policy",
        config=checkpoint_run_config,
        platform_spec=platform_spec,
    )
    if manifest_path is None:
        return checkpoint_run_config

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TRAINING_SCHEMA:
        raise ValueError("training manifest schema mismatch")
    if manifest.get("algorithm") != contract.ALGORITHM:
        raise ValueError("training manifest algorithm mismatch")
    if manifest.get("algorithm_signature") != contract.ALGORITHM_SIGNATURE:
        raise ValueError("training manifest algorithm signature mismatch")
    if manifest.get("runtime_code_fingerprint_sha256") != (
        runtime_code_fingerprint_sha256()
    ):
        raise ValueError("training/evaluation shared runtime code mismatch")
    if manifest.get("physics_backend") != contract.PHYSICS_BACKEND:
        raise ValueError("training manifest physics backend mismatch")
    if (
        manifest.get("torch_warp_exchange")
        != contract.TORCH_WARP_EXCHANGE
    ):
        raise ValueError("training manifest Torch/Warp exchange mismatch")
    if (
        manifest.get("total_mjwarp_expert_steps")
        != contract.TOTAL_TRAINING_STEPS
    ):
        raise ValueError("training manifest is not an exact-60M expert run")
    if manifest.get("anti_support_gate_used") is not False:
        raise ValueError("training manifest unexpectedly used anti-support")
    if manifest.get("physical_hard_tip_plane_success_invariant") is not True:
        raise ValueError(
            "training manifest lacks the physical hard-tip plane invariant"
        )
    if manifest.get("independent_hard_tip_control") is not True:
        raise ValueError("training manifest lacks independent hard-tip control")
    if _canonical_json(manifest.get("translation_path_contract")) != (
        _canonical_json(contract.translation_path_contract_payload())
    ):
        raise ValueError("training manifest translation path contract mismatch")
    expected_rotation_distribution = {
        "bounds": [
            contract.ROTATION_ANGLE_MIN_DEG,
            contract.ROTATION_ANGLE_MAX_DEG,
        ],
        "integer_closed_intervals": True,
        "maximum_successful_demo_required_per_object": True,
    }
    if (
        manifest.get("rotation_angle_distribution_deg")
        != expected_rotation_distribution
    ):
        raise ValueError("training manifest rotation distribution mismatch")
    expected_bounds = [
        contract.GRASP_PLANE_MIN_FRACTION,
        contract.GRASP_PLANE_MAX_FRACTION,
    ]
    if manifest.get("grasp_plane_fraction_bounds") != expected_bounds:
        raise ValueError(
            "training manifest learned-fruit grasp-plane bounds mismatch"
        )
    expected_learned_objects = [
        object_kind
        for object_kind in contract.OBJECTS
        if object_kind in contract.LEARNED_GRASP_PLANE_OBJECTS
    ]
    if (
        manifest.get("learned_grasp_plane_objects")
        != expected_learned_objects
    ):
        raise ValueError(
            "training manifest learned grasp-plane object list mismatch"
        )
    expected_fixed_fractions = {
        object_kind: float(
            contract.FIXED_GRASP_PLANE_FRACTIONS[object_kind]
        )
        for object_kind in contract.OBJECTS
        if object_kind in contract.FIXED_GRASP_PLANE_FRACTIONS
    }
    if (
        manifest.get("fixed_grasp_plane_fractions")
        != expected_fixed_fractions
    ):
        raise ValueError(
            "training manifest fixed grasp-plane fraction mapping mismatch"
        )
    if (
        manifest.get("fixed_grasp_plane_yaw_rad")
        != contract.FIXED_GRASP_PLANE_YAW_RAD
    ):
        raise ValueError("training manifest fixed grasp-plane yaw mismatch")
    if (
        manifest.get("fixed_grasp_plane_physical_tolerance_m")
        != contract.FIXED_GRASP_PLANE_PHYSICAL_TOLERANCE_M
    ):
        raise ValueError(
            "training manifest fixed grasp-plane physical tolerance mismatch"
        )
    if (
        manifest.get("reset_templates_per_object")
        != contract.RESET_TEMPLATES_PER_OBJECT
    ):
        raise ValueError("training manifest reset template bank mismatch")
    config = _config_from_mapping(dict(manifest.get("config", {})))
    _validate_task_binding(
        manifest,
        label="training manifest",
        config=config,
        platform_spec=platform_spec,
    )
    if manifest.get("physics_layout") != contract.PHYSICS_LAYOUT:
        raise ValueError("training manifest physics layout mismatch")
    if manifest.get("global_worlds") != config.num_envs:
        raise ValueError("training manifest global world count mismatch")
    if manifest.get("object_world_counts") != list(
        config.object_world_counts
    ):
        raise ValueError("training manifest object-world allocation mismatch")
    if (
        manifest.get("remainder_worlds") != config.remainder_worlds
        or manifest.get("remainder_object_world_counts")
        != list(config.remainder_object_world_counts)
    ):
        raise ValueError("training manifest remainder allocation mismatch")
    expected_shards = ordered_shard_manifest(
        platform_spec, config.object_world_counts
    )
    actual_shards = manifest.get("model_shards")
    if not isinstance(actual_shards, list) or len(actual_shards) != len(
        expected_shards
    ):
        raise ValueError("training manifest four-shard inventory is missing")
    bound_fields = (
        "object_index",
        "object",
        "model_xml_sha256",
        "nq",
        "nv",
        "nu",
        "worlds",
        "task",
        "task_index",
        "root_height_m",
    )
    for actual, expected in zip(actual_shards, expected_shards):
        if not isinstance(actual, Mapping) or any(
            actual.get(name) != expected.get(name) for name in bound_fields
        ):
            raise ValueError(
                "training/evaluation single-object model shard mismatch"
            )
    if (
        manifest.get("parent_model_fingerprint_sha256")
        != platform_spec.parent_model_fingerprint_sha256
    ):
        raise ValueError(
            "training/evaluation complete parent model fingerprint mismatch"
        )
    if (
        manifest.get("parent_controller_fingerprint_sha256")
        != platform_spec.parent_controller_fingerprint_sha256
    ):
        raise ValueError(
            "training/evaluation parent controller fingerprint mismatch"
        )
    if (
        manifest.get("platform_adapter_fingerprint_sha256")
        != platform_spec.platform_adapter_fingerprint_sha256
    ):
        raise ValueError(
            "training/evaluation platform adapter fingerprint mismatch"
        )
    fruit_hash = (
        _sha256(FRUIT_MANIFEST) if FRUIT_MANIFEST.is_file() else None
    )
    if manifest.get("fruit_manifest_sha256") != fruit_hash:
        raise ValueError("training/evaluation fruit capsule manifest mismatch")
    if manifest.get("final_model") != model_path.name:
        raise ValueError("training manifest final model name mismatch")
    if manifest.get("final_model_sha256") != _sha256(model_path):
        raise ValueError("training manifest/final policy hash mismatch")

    if checkpoint_run_config.canonical_json() != config.canonical_json():
        raise ValueError("training manifest/checkpoint config mismatch")
    diffusion_summary = manifest.get("diffusion")
    if not isinstance(diffusion_summary, Mapping):
        raise ValueError("training manifest is missing Diffusion summary")
    if (
        diffusion_summary.get("dataset_metadata_sha256")
        != metadata.get("dataset_metadata_sha256")
    ):
        raise ValueError("manifest/checkpoint demonstration hash mismatch")
    return config


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    policy: DiffusionPolicy,
    config: HPCConfig,
    platform_spec: Any,
) -> None:
    metadata = _checkpoint_metadata(checkpoint)
    _validate_task_binding(
        metadata,
        label="policy",
        config=config,
        platform_spec=platform_spec,
    )
    if metadata.get("algorithm") != contract.ALGORITHM:
        raise ValueError("policy algorithm mismatch")
    if metadata.get("algorithm_signature") != contract.ALGORITHM_SIGNATURE:
        raise ValueError("policy algorithm signature mismatch")
    if metadata.get("runtime_code_fingerprint_sha256") != (
        runtime_code_fingerprint_sha256()
    ):
        raise ValueError("policy shared runtime code mismatch")
    if metadata.get("platform") != platform_spec.platform:
        raise ValueError("policy was trained for another gripper")
    if (
        metadata.get("platform_adapter_fingerprint_sha256")
        != platform_spec.platform_adapter_fingerprint_sha256
    ):
        raise ValueError("policy platform adapter fingerprint mismatch")
    if metadata.get("physics_layout") != contract.PHYSICS_LAYOUT:
        raise ValueError("policy physics layout mismatch")
    if _canonical_json(metadata.get("translation_path_contract")) != (
        _canonical_json(contract.translation_path_contract_payload())
    ):
        raise ValueError("policy translation path contract mismatch")
    if metadata.get("rotation_angle_distribution_deg") != [
        contract.ROTATION_ANGLE_MIN_DEG,
        contract.ROTATION_ANGLE_MAX_DEG,
    ]:
        raise ValueError("policy rotation distribution mismatch")
    if metadata.get("global_worlds") != config.num_envs:
        raise ValueError("policy global world count mismatch")
    if metadata.get("object_world_counts") != list(
        config.object_world_counts
    ):
        raise ValueError("policy object-world allocation mismatch")
    if (
        metadata.get("remainder_worlds") != config.remainder_worlds
        or metadata.get("remainder_object_world_counts")
        != list(config.remainder_object_world_counts)
    ):
        raise ValueError("policy remainder allocation mismatch")
    expected_shards = ordered_shard_manifest(
        platform_spec, config.object_world_counts
    )
    metadata_shards = metadata.get("model_shards")
    if not isinstance(metadata_shards, list) or len(metadata_shards) != len(
        expected_shards
    ):
        raise ValueError("policy four-shard inventory is missing")
    for actual, expected in zip(metadata_shards, expected_shards):
        if not isinstance(actual, Mapping) or any(
            actual.get(name) != expected.get(name)
            for name in (
                "object_index",
                "object",
                "model_xml_sha256",
                "nq",
                "nv",
                "nu",
                "worlds",
                "task",
                "task_index",
                "root_height_m",
            )
        ):
            raise ValueError("policy single-object model shard mismatch")
    if (
        metadata.get("total_mjwarp_expert_steps")
        != contract.TOTAL_TRAINING_STEPS
    ):
        raise ValueError("policy is not from the final exact-60M expert run")
    if metadata.get("epoch") != config.diffusion.epochs:
        raise ValueError("policy is not the final Diffusion epoch")
    if metadata.get("epochs") != config.diffusion.epochs:
        raise ValueError("policy Diffusion epoch schedule mismatch")
    if int(metadata.get("updates", 0)) <= 0:
        raise ValueError("policy has no completed Diffusion updates")
    if checkpoint.get("ema_state") is None:
        raise ValueError("final Diffusion checkpoint is missing EMA weights")

    settings = config.diffusion
    expected = {
        "obs_dim": OBSERVATION_DIM,
        "action_dim": contract.ACTION_DIM,
        "obs_horizon": settings.obs_horizon,
        "horizon": settings.action_horizon,
        "num_diffusion_steps": settings.diffusion_steps,
        "hidden_dim": settings.hidden_dim,
        "depth": settings.depth,
        "dropout": settings.dropout,
    }
    actual = policy.model_config()
    for name, value in expected.items():
        if actual.get(name) != value:
            raise ValueError(
                f"policy model configuration mismatch for {name}: "
                f"expected {value!r}, found {actual.get(name)!r}"
            )


def _cases(
    phase: str,
    *,
    episodes_per_case: int,
    seed: int,
    task: str | None = None,
) -> list[dict[str, Any]]:
    if not 0 < episodes_per_case <= contract.RESET_TEMPLATES_PER_OBJECT:
        raise ValueError(
            "episodes_per_case must be in [1, "
            f"{contract.RESET_TEMPLATES_PER_OBJECT}] so reset templates "
            "remain unique"
        )
    cases: list[dict[str, Any]] = []
    selected_tasks = (
        contract.TASKS
        if task is None
        else (contract.validate_task(task),)
    )
    for object_index, object_kind in enumerate(contract.OBJECTS):
        for case_task in selected_tasks:
            task_index = contract.TASKS.index(case_task)
            for episode in range(episodes_per_case):
                template_index = episode
                episode_seed = (
                    int(seed)
                    + object_index * 100_000
                    + task_index * 10_000
                    + episode
                )
                if case_task == contract.TASK_TRANSLATE:
                    sample = sample_translation(
                        phase, random.Random(episode_seed)
                    )
                    path_seed = sample.path_seed
                    path_shape_index = episode % len(
                        contract.TRANSLATION_PATH_SHAPES
                    )
                    path_step_m = translation_path_step_m(
                        phase, path_shape_index
                    )
                    path_length_m = translation_path_length_m(
                        phase, path_shape_index
                    )
                    path_waypoints_xy_m = [
                        list(waypoint)
                        for waypoint in translation_path_waypoints_m(
                            phase, path_shape_index, path_seed
                        )
                    ]
                    source_region = SOURCE_REGION_INDEX[
                        sample.source_region
                    ]
                    rotation_angle_deg = 0
                else:
                    path_seed = 0
                    path_shape_index = 0
                    path_step_m = 0.0
                    path_length_m = 0.0
                    path_waypoints_xy_m = [
                        [0.0, 0.0]
                        for _ in range(contract.TRANSLATION_PATH_WAYPOINTS)
                    ]
                    source_region = 0
                    rotation_low, rotation_high = rotation_angle_bounds_deg(
                        phase
                    )
                    rotation_angle_deg = rotation_low + (
                        episode % (rotation_high - rotation_low + 1)
                    )
                cases.append(
                    {
                        "object_index": object_index,
                        "object": object_kind,
                        "task_index": task_index,
                        "task": case_task,
                        "phase": phase,
                        "episode": episode,
                        "seed": episode_seed,
                        "template_index": template_index,
                        "template_seed": contract.reset_template_seed(
                            object_index, template_index
                        ),
                        "path_seed": path_seed,
                        "path_shape_index": path_shape_index,
                        "path_shape": (
                            contract.TRANSLATION_PATH_SHAPES[
                                path_shape_index
                            ]
                            if case_task == contract.TASK_TRANSLATE
                            else None
                        ),
                        "path_step_m": path_step_m,
                        "path_length_m": path_length_m,
                        "path_waypoints_xy_m": path_waypoints_xy_m,
                        "rotation_angle_deg": rotation_angle_deg,
                        "source_region": source_region,
                    }
                )
    return cases


def _translation_path_case_coverage(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate deterministic shape cycling and summarize paired coverage."""

    case_tasks = {case.get("task") for case in cases}
    if case_tasks == {contract.TASK_ROTATE}:
        return {
            "active": False,
            "reason": "evaluation_task_is_rotation",
            "assignment": None,
            "shape_counts_by_object": {},
            "all_shapes_covered": None,
        }
    if contract.TASK_TRANSLATE not in case_tasks:
        raise ValueError("evaluation cases contain no recognized translation task")
    shape_count = len(contract.TRANSLATION_PATH_SHAPES)
    by_object: dict[str, dict[str, int]] = {}
    for object_kind in contract.OBJECTS:
        translation_cases = [
            case
            for case in cases
            if case["object"] == object_kind
            and case["task"] == contract.TASK_TRANSLATE
        ]
        if not translation_cases:
            raise ValueError(
                f"paired evaluation has no translation cases for {object_kind}"
            )
        if len(translation_cases) != contract.RESET_TEMPLATES_PER_OBJECT:
            raise ValueError(
                "formal paired evaluation requires all 32 reset templates"
            )
        counts = [0] * shape_count
        for case in translation_cases:
            phase = str(case["phase"])
            expected_source = 0
            if int(case["source_region"]) != expected_source:
                raise ValueError("paired translation source region is invalid")
            expected_shape = int(case["episode"]) % shape_count
            if int(case["path_shape_index"]) != expected_shape:
                raise ValueError("paired translation shape cycle is not deterministic")
            if case.get("path_shape") != contract.TRANSLATION_PATH_SHAPES[
                expected_shape
            ]:
                raise ValueError("paired translation shape name/index mismatch")
            path_seed = int(case["path_seed"])
            if not (
                contract.TRANSLATION_PATH_SEED_MIN
                <= path_seed
                <= contract.TRANSLATION_PATH_SEED_MAX
            ):
                raise ValueError("paired translation path seed is invalid")
            if not math.isclose(
                float(case["path_step_m"]),
                translation_path_step_m(phase, expected_shape),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                float(case["path_length_m"]),
                translation_path_length_m(phase, expected_shape),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("paired translation path scale/length mismatch")
            expected_waypoints = translation_path_waypoints_m(
                phase,
                expected_shape,
                path_seed,
            )
            actual_waypoints = np.asarray(
                case["path_waypoints_xy_m"], dtype=np.float64
            )
            if (
                actual_waypoints.shape
                != (contract.TRANSLATION_PATH_WAYPOINTS, 2)
                or not np.isfinite(actual_waypoints).all()
            ):
                raise ValueError("paired translation path geometry is invalid")
            if not np.allclose(
                actual_waypoints,
                np.asarray(expected_waypoints, dtype=np.float64),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError("paired translation path is not reproducible")
            if np.linalg.norm(actual_waypoints, axis=1).max() > (
                contract.TRANSLATION_PATH_REGION_RADIUS_M + 1.0e-12
            ):
                raise ValueError("paired translation path geometry is invalid")
            counts[expected_shape] += 1
        by_object[object_kind] = {
            name: counts[index]
            for index, name in enumerate(contract.TRANSLATION_PATH_SHAPES)
        }
        expected_counts = [
            contract.RESET_TEMPLATES_PER_OBJECT // shape_count
            + (index < contract.RESET_TEMPLATES_PER_OBJECT % shape_count)
            for index in range(shape_count)
        ]
        if counts != expected_counts:
            raise ValueError(
                "formal paired evaluation path assignment is not balanced"
            )
    return {
        "active": True,
        "reason": None,
        "assignment": "episode_modulo_shape_count",
        "shape_counts_by_object": by_object,
        "all_shapes_covered": all(
            count > 0
            for object_counts in by_object.values()
            for count in object_counts.values()
        ),
    }


def _segment(step: int, config: HPCConfig) -> str:
    boundaries = (
        config.plan_ticks,
        config.plan_ticks + config.close_ticks,
        config.plan_ticks + config.close_ticks + config.settle_ticks,
        config.plan_ticks
        + config.close_ticks
        + config.settle_ticks
        + config.lift_ticks,
        config.plan_ticks
        + config.close_ticks
        + config.settle_ticks
        + config.lift_ticks
        + config.hold_ticks,
    )
    for boundary, name in zip(
        boundaries, ("plan", "close", "settle", "lift", "hold")
    ):
        if step < boundary:
            return name
    return "action"


def _goal_tensors(
    torch: Any,
    cases: list[dict[str, Any]],
    device: Any,
) -> dict[str, Any]:
    def values(name: str) -> Any:
        return torch.tensor(
            [case[name] for case in cases],
            device=device,
            dtype=torch.long,
        )

    return {
        "object_index": values("object_index"),
        "template_index": values("template_index"),
        "task_index": values("task_index"),
        "path_seed": values("path_seed"),
        "path_shape_index": values("path_shape_index"),
        "rotation_angle_deg": values("rotation_angle_deg"),
        "source_region": values("source_region"),
    }


def _cpu_snapshot(snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, value in snapshot.items():
        if hasattr(value, "detach"):
            result[key] = value.detach().cpu().numpy()
        else:
            result[key] = np.asarray(value)
    return result


def _safe_policy_action(
    action: Any,
    observation: Any,
    *,
    torch: Any,
) -> Any:
    """Apply the public normalized-action contract before entering MJWarp."""

    safe = torch.nan_to_num(
        action.to(device=observation.device, dtype=observation.dtype),
        nan=0.0,
        posinf=contract.NORMALIZED_ACTION_MAX,
        neginf=contract.NORMALIZED_ACTION_MIN,
    ).clamp(
        contract.NORMALIZED_ACTION_MIN,
        contract.NORMALIZED_ACTION_MAX,
    )
    mask = action_mask_from_observation(observation).to(safe.dtype)
    return safe * mask


def _unexpected_early_done(transition: Any, torch: Any) -> Any:
    """Classify only non-horizon, non-drop or non-finite terminals."""

    done = transition.done.to(torch.bool)
    dropped = torch.as_tensor(
        transition.info.get("object_dropped", torch.zeros_like(done)),
        device=done.device,
        dtype=torch.bool,
    )
    time_out = torch.as_tensor(
        transition.info.get("time_out", torch.zeros_like(done)),
        device=done.device,
        dtype=torch.bool,
    )
    transition_finite = torch.as_tensor(
        transition.info.get("transition_finite", torch.ones_like(done)),
        device=done.device,
        dtype=torch.bool,
    )
    return (done & (~dropped) & (~time_out)) | (~transition_finite)


def _rollout_mode(
    *,
    environment: Any,
    policy: DiffusionPolicy,
    goals: dict[str, Any],
    use_diffusion: bool,
    config: HPCConfig,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    import torch

    observation = environment.reset_with_goals(**goals)
    initial = _cpu_snapshot(environment.snapshot())
    initial.update(
        {
            f"backend.{name}": value.detach().cpu().numpy()
            for name, value in environment.backend.state_dict().items()
        }
    )
    history = None
    controller = None
    if use_diffusion:
        history = BatchedObservationHistory(
            worlds=environment.worlds,
            obs_horizon=config.diffusion.obs_horizon,
            obs_dim=OBSERVATION_DIM,
            device=environment.device,
        )
        all_worlds = torch.ones(
            environment.worlds,
            device=environment.device,
            dtype=torch.bool,
        )
        history.reset(all_worlds, observation)
        controller = BatchedDiffusionController(
            policy,
            worlds=environment.worlds,
            inference_steps=config.diffusion.inference_steps,
            replan_interval=config.diffusion.replan_interval,
            seed=seed,
            action_clip=True,
        )
        controller.reset(all_worlds, history.values)

    rows: dict[str, list[np.ndarray]] = {}
    for step in range(config.episode_length):
        with torch.no_grad():
            if controller is None:
                requested = torch.zeros(
                    (environment.worlds, contract.ACTION_DIM),
                    device=environment.device,
                    dtype=torch.float32,
                )
            else:
                if history is None:  # pragma: no cover - construction invariant.
                    raise RuntimeError("Diffusion controller has no history")
                requested = controller.act(history.values)
            action = _safe_policy_action(
                requested, observation, torch=torch
            )
        transition = environment.step(action)
        transition_finite = transition.info.get("transition_finite")
        if transition_finite is None or not bool(
            torch.as_tensor(transition_finite, dtype=torch.bool).all()
        ):
            raise FloatingPointError(
                "paired evaluation encountered a non-finite MJWarp transition"
            )
        done = transition.done.to(torch.bool)
        unexplained_done = _unexpected_early_done(transition, torch)
        if step + 1 < config.episode_length and bool(
            unexplained_done.any()
        ):
            raise RuntimeError(
                "paired evaluation had a non-drop terminal before the horizon"
            )
        if step + 1 == config.episode_length and not bool(done.all()):
            raise RuntimeError("paired evaluation did not terminate at the horizon")
        observation = transition.observation
        if history is not None:
            history.append(observation)
        snapshot = _cpu_snapshot(environment.snapshot())
        snapshot["requested_action"] = action.detach().cpu().numpy()
        for name, value in snapshot.items():
            rows.setdefault(name, []).append(value)
    return (
        {name: np.stack(values, axis=0) for name, values in rows.items()},
        initial,
    )


def _rollout_phase(
    *,
    environment: Any,
    policy: DiffusionPolicy,
    cases: list[dict[str, Any]],
    config: HPCConfig,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    import torch

    goals = _goal_tensors(torch, cases, environment.device)
    ik_trace, ik_initial = _rollout_mode(
        environment=environment,
        policy=policy,
        goals=goals,
        use_diffusion=False,
        config=config,
        seed=seed,
    )
    diffusion_trace, diffusion_initial = _rollout_mode(
        environment=environment,
        policy=policy,
        goals=goals,
        use_diffusion=True,
        config=config,
        seed=seed,
    )
    if ik_initial.keys() != diffusion_initial.keys():
        raise RuntimeError("paired reset exposed different state fields")
    for key in ik_initial:
        if not np.array_equal(ik_initial[key], diffusion_initial[key]):
            raise RuntimeError(
                f"paired reset diverged before controller action: {key}"
            )
    return {"ik_only": ik_trace, "ik_diffusion": diffusion_trace}


def _trace_or_default(
    trace: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    value: float | bool,
    *,
    dtype: Any = float,
) -> np.ndarray:
    found = trace.get(name)
    if found is not None:
        return found
    return np.full(shape, value, dtype=dtype)


def _result_from_trace(
    trace: dict[str, np.ndarray],
    *,
    case_index: int,
    object_kind: str,
    task: str,
    config: HPCConfig,
    trajectory_stride: int,
) -> dict[str, Any]:
    endpoint_index = config.episode_length - 1
    action_start = (
        config.plan_ticks
        + config.close_ticks
        + config.settle_ticks
        + config.lift_ticks
        + config.hold_ticks
    )
    precondition_index = max(0, action_start - 1)
    reference_index = min(
        config.plan_ticks + config.close_ticks + config.settle_ticks - 1,
        endpoint_index,
    )
    initial_position = trace["position"][reference_index, case_index]
    precondition_position = trace["position"][
        precondition_index, case_index
    ]
    precondition_rotation = trace["rotation_from_initial_xyz"][
        precondition_index, case_index
    ]
    action_values = trace["action"][:, case_index]
    requested_action = trace.get("requested_action", trace["action"])[
        :, case_index
    ]
    ticks = trace["action"].shape[0]
    worlds = trace["action"].shape[1]
    plane_bounds = contract.grasp_plane_fraction_limits(object_kind)
    plane_target = contract.grasp_plane_nominal_fraction(object_kind)
    plane_is_learned = contract.grasp_plane_is_learned(object_kind)
    tip_fraction = _trace_or_default(
        trace,
        "tip_height_fraction",
        (ticks, worlds, len(contract.TIP_ORDER)),
        plane_target,
    )
    commanded_plane_fraction = _trace_or_default(
        trace,
        "grasp_plane_fraction",
        (ticks, worlds),
        plane_target,
    )
    commanded_plane_compliant_value = trace.get(
        "grasp_plane_compliant"
    )
    if commanded_plane_compliant_value is None:
        numerical_tolerance = 2.0e-6 if not plane_is_learned else 0.0
        commanded_plane_compliant = (
            (commanded_plane_fraction >= plane_bounds[0] - numerical_tolerance)
            & (commanded_plane_fraction <= plane_bounds[1] + numerical_tolerance)
        )
    else:
        commanded_plane_compliant = commanded_plane_compliant_value.astype(
            bool
        )
    actual_tip_fraction = _trace_or_default(
        trace,
        "actual_tip_height_fraction",
        (ticks, worlds, len(contract.TIP_ORDER)),
        plane_target,
    )
    actual_tip_height_m = _trace_or_default(
        trace,
        "actual_tip_height_m",
        (ticks, worlds, len(contract.TIP_ORDER)),
        0.0,
    )
    plane_fraction = _trace_or_default(
        trace,
        "actual_grasp_plane_fraction",
        (ticks, worlds),
        plane_target,
    )
    plane_compliant = _trace_or_default(
        trace,
        "actual_grasp_plane_compliant",
        (ticks, worlds),
        False,
        dtype=bool,
    ).astype(bool)
    plane_relevant = _trace_or_default(
        trace,
        "actual_grasp_plane_relevant",
        (ticks, worlds),
        True,
        dtype=bool,
    ).astype(bool)
    force_scale = _trace_or_default(
        trace,
        "tip_force_scale",
        (ticks, worlds, len(contract.TIP_ORDER)),
        contract.FORCE_SCALE_NOMINAL,
    )
    object_dropped = _trace_or_default(
        trace,
        "object_dropped",
        (ticks, worlds),
        False,
        dtype=bool,
    ).astype(bool)
    object_drop_new = _trace_or_default(
        trace,
        "object_drop_new",
        (ticks, worlds),
        False,
        dtype=bool,
    ).astype(bool)
    object_drop_first_control_step = _trace_or_default(
        trace,
        "object_drop_first_control_step",
        (ticks, worlds),
        -1,
        dtype=np.int64,
    )
    object_drop_first_bottom_z_m = _trace_or_default(
        trace,
        "object_drop_first_bottom_z_m",
        (ticks, worlds),
        contract.OBJECT_DROP_THRESHOLD_Z_M,
    )
    object_drop_first_position = _trace_or_default(
        trace,
        "object_drop_first_position",
        (ticks, worlds, 3),
        0.0,
    )
    path_seed = _trace_or_default(
        trace, "path_seed", (ticks, worlds), 0, dtype=np.int64
    )
    path_shape_index = _trace_or_default(
        trace, "path_shape_index", (ticks, worlds), 0, dtype=np.int64
    )
    path_step_m = _trace_or_default(
        trace, "path_step_m", (ticks, worlds), 0.0
    )
    path_length_m = _trace_or_default(
        trace, "path_length_m", (ticks, worlds), 0.0
    )
    path_waypoints_xy_m = _trace_or_default(
        trace,
        "path_waypoints_xy_m",
        (ticks, worlds, contract.TRANSLATION_PATH_WAYPOINTS, 2),
        0.0,
    )
    path_current_target_world_m = trace.get(
        "path_current_target_world_m", trace["goal_position"]
    )
    path_current_target_error_xyz_m = trace.get(
        "path_current_target_error_xyz_m", trace["position_error_xyz"]
    )
    path_tracking_current_error_m = trace.get(
        "path_tracking_current_error_m", trace["position_error"]
    )
    path_tracking_rmse_m = _trace_or_default(
        trace, "path_tracking_rmse_m", (ticks, worlds), 0.0
    )
    path_tracking_max_error_m = _trace_or_default(
        trace, "path_tracking_max_error_m", (ticks, worlds), 0.0
    )
    path_tracking_sample_count = _trace_or_default(
        trace,
        "path_tracking_sample_count",
        (ticks, worlds),
        0,
        dtype=np.int64,
    )
    path_following_active = _trace_or_default(
        trace,
        "path_following_active",
        (ticks, worlds),
        False,
        dtype=bool,
    ).astype(bool)
    path_progress = _trace_or_default(
        trace, "path_progress", (ticks, worlds, 5), 0.0
    )
    desired_final_displacement_xyz_m = trace.get(
        "desired_final_displacement_xyz_m", trace["goal_position"]
    )
    actual_final_displacement_xyz_m = trace.get(
        "actual_final_displacement_xyz_m", trace["position"]
    )
    desired_final_displacement_l2_m = trace.get(
        "desired_final_displacement_l2_m",
        np.linalg.norm(desired_final_displacement_xyz_m, axis=-1),
    )
    actual_final_displacement_l2_m = trace.get(
        "actual_final_displacement_l2_m",
        np.linalg.norm(actual_final_displacement_xyz_m, axis=-1),
    )
    final_displacement_error_xyz_m = trace.get(
        "final_displacement_error_xyz_m", trace["position_error_xyz"]
    )
    final_displacement_error_l2_m = trace.get(
        "final_displacement_error_l2_m", trace["position_error"]
    )
    final_position_error_xyz_m = trace.get(
        "final_position_error_xyz_m", trace["position_error_xyz"]
    )
    final_position_error_l2_m = trace.get(
        "final_position_error_l2_m", trace["position_error"]
    )
    final_euler_error_xyz_deg = trace.get(
        "final_euler_error_xyz_deg", trace["euler_error_xyz"]
    )
    final_euler_error_l2_deg = trace.get(
        "final_euler_error_l2_deg", trace["euler_error"]
    )

    trajectory: list[dict[str, Any]] = []
    selected_steps = list(
        range(0, config.episode_length, max(1, trajectory_stride))
    )
    if endpoint_index not in selected_steps:
        selected_steps.append(endpoint_index)
    for step in selected_steps:
        segment = _segment(step, config)
        if segment == "action":
            segment = "translate" if task == "translate" else "rotate"
        dropped_at_step = bool(object_dropped[step, case_index])
        trajectory.append(
            {
                "control_step": step + 1,
                "time_s": (step + 1) / 50.0,
                "segment": segment,
                "object_xyz_m": trace["position"][step, case_index].tolist(),
                "final_goal_xyz_m": trace["goal_position"][
                    step, case_index
                ].tolist(),
                "path_current_target_world_m": (
                    path_current_target_world_m[step, case_index].tolist()
                ),
                "path_current_target_error_xyz_m": (
                    path_current_target_error_xyz_m[
                        step, case_index
                    ].tolist()
                ),
                "path_tracking_current_error_m": float(
                    path_tracking_current_error_m[step, case_index]
                ),
                "path_tracking_rmse_m": float(
                    path_tracking_rmse_m[step, case_index]
                ),
                "path_tracking_max_error_m": float(
                    path_tracking_max_error_m[step, case_index]
                ),
                "path_tracking_sample_count": int(
                    path_tracking_sample_count[step, case_index]
                ),
                "path_following_active": bool(
                    path_following_active[step, case_index]
                ),
                "path_progress": path_progress[step, case_index].tolist(),
                "position_error_m": float(
                    trace["position_error"][step, case_index]
                ),
                "tracking_position_error_xyz_m": trace[
                    "position_error_xyz"
                ][step, case_index].tolist(),
                "euler_l2_error_deg": float(
                    trace["euler_error"][step, case_index]
                ),
                "tracking_euler_error_xyz_deg": trace[
                    "euler_error_xyz"
                ][step, case_index].tolist(),
                "desired_final_displacement_xyz_m": (
                    desired_final_displacement_xyz_m[
                        step, case_index
                    ].tolist()
                ),
                "actual_final_displacement_xyz_m": (
                    actual_final_displacement_xyz_m[
                        step, case_index
                    ].tolist()
                ),
                "desired_final_displacement_l2_m": float(
                    desired_final_displacement_l2_m[step, case_index]
                ),
                "actual_final_displacement_l2_m": float(
                    actual_final_displacement_l2_m[step, case_index]
                ),
                "final_displacement_error_xyz_m": (
                    final_displacement_error_xyz_m[
                        step, case_index
                    ].tolist()
                ),
                "final_displacement_error_l2_m": float(
                    final_displacement_error_l2_m[step, case_index]
                ),
                "rotation_progress_deg": float(
                    trace["rotation_from_action_reference_xyz"][
                        step, case_index, 2
                    ]
                ),
                "contact_forces_n": trace["contact_forces"][
                    step, case_index
                ].tolist(),
                "applied_action": trace["action"][
                    step, case_index
                ].tolist(),
                "requested_action_after_mask": requested_action[
                    step
                ].tolist(),
                "tip_height_m": actual_tip_height_m[
                    step, case_index
                ].tolist(),
                "commanded_tip_height_m": trace["tip_height_m"][
                    step, case_index
                ].tolist(),
                "tip_height_fraction": tip_fraction[
                    step, case_index
                ].tolist(),
                "actual_tip_height_fraction": actual_tip_fraction[
                    step, case_index
                ].tolist(),
                "grasp_plane_fraction": float(
                    plane_fraction[step, case_index]
                ),
                "grasp_plane_compliant": bool(
                    plane_compliant[step, case_index]
                ),
                "actual_grasp_plane_relevant": bool(
                    plane_relevant[step, case_index]
                ),
                "commanded_grasp_plane_fraction": float(
                    commanded_plane_fraction[step, case_index]
                )
                if not dropped_at_step
                else None,
                "commanded_grasp_plane_compliant": (
                    bool(commanded_plane_compliant[step, case_index])
                    if not dropped_at_step
                    else None
                ),
                "object_dropped": dropped_at_step,
                "object_drop_new": bool(
                    object_drop_new[step, case_index]
                ),
            }
        )
    case_plane_fraction = plane_fraction[:, case_index]
    case_plane_compliant = plane_compliant[:, case_index]
    case_plane_relevant = plane_relevant[:, case_index]
    case_dropped = object_dropped[:, case_index]
    relevant_mask = case_plane_relevant & (~case_dropped)
    relevant_fraction = case_plane_fraction[relevant_mask]
    relevant_compliance = case_plane_compliant[relevant_mask]
    commanded_live_mask = ~case_dropped
    live_commanded_fraction = commanded_plane_fraction[
        commanded_live_mask, case_index
    ]
    live_commanded_compliance = commanded_plane_compliant[
        commanded_live_mask, case_index
    ]
    has_relevant_plane = bool(relevant_fraction.size)
    has_live_command = bool(live_commanded_fraction.size)
    endpoint_plane_relevant = bool(relevant_mask[endpoint_index])
    endpoint_is_live = not bool(case_dropped[endpoint_index])
    plan_index = max(config.plan_ticks - 1, 0)
    endpoint_path_shape_index = int(
        path_shape_index[endpoint_index, case_index]
    )
    return {
        "success": bool(trace["success"][endpoint_index, case_index]),
        "success_tolerance": _case_success_tolerance(object_kind, task),
        "translation_path": {
            "seed": int(path_seed[endpoint_index, case_index]),
            "shape_index": endpoint_path_shape_index,
            "shape": (
                contract.TRANSLATION_PATH_SHAPES[
                    endpoint_path_shape_index
                ]
                if task == contract.TASK_TRANSLATE
                else None
            ),
            "step_m": float(
                path_step_m[endpoint_index, case_index]
            ),
            "length_m": float(
                path_length_m[endpoint_index, case_index]
            ),
            "waypoints_xy_m": path_waypoints_xy_m[
                endpoint_index, case_index
            ].tolist(),
            "tracking_rmse_m": float(
                path_tracking_rmse_m[endpoint_index, case_index]
            ),
            "tracking_max_error_m": float(
                path_tracking_max_error_m[endpoint_index, case_index]
            ),
            "tracking_sample_count": int(
                path_tracking_sample_count[endpoint_index, case_index]
            ),
        },
        "object_drop": {
            "dropped": bool(object_dropped[endpoint_index, case_index]),
            "first_control_step": int(
                object_drop_first_control_step[endpoint_index, case_index]
            ),
            "first_position_m": object_drop_first_position[
                endpoint_index, case_index
            ].tolist(),
            "first_estimated_bottom_z_m": float(
                object_drop_first_bottom_z_m[endpoint_index, case_index]
            ),
            "threshold_z_m": contract.OBJECT_DROP_THRESHOLD_Z_M,
        },
        "endpoint": {
            "final_position_error_l2_m": float(
                final_position_error_l2_m[endpoint_index, case_index]
            ),
            "final_euler_error_l2_deg": float(
                final_euler_error_l2_deg[endpoint_index, case_index]
            ),
            "final_position_axis_abs_max_error_m": float(
                np.abs(
                    final_position_error_xyz_m[
                        endpoint_index, case_index
                    ]
                ).max()
            ),
            "final_euler_axis_abs_max_error_deg": float(
                np.abs(
                    final_euler_error_xyz_deg[
                        endpoint_index, case_index
                    ]
                ).max()
            ),
            "final_position_error_xyz_m": final_position_error_xyz_m[
                endpoint_index, case_index
            ].tolist(),
            "final_euler_error_xyz_deg": final_euler_error_xyz_deg[
                endpoint_index, case_index
            ].tolist(),
            "desired_final_displacement_xyz_m": (
                desired_final_displacement_xyz_m[
                    endpoint_index, case_index
                ].tolist()
            ),
            "actual_final_displacement_xyz_m": (
                actual_final_displacement_xyz_m[
                    endpoint_index, case_index
                ].tolist()
            ),
            "desired_final_displacement_l2_m": float(
                desired_final_displacement_l2_m[
                    endpoint_index, case_index
                ]
            ),
            "actual_final_displacement_l2_m": float(
                actual_final_displacement_l2_m[
                    endpoint_index, case_index
                ]
            ),
            "final_displacement_error_xyz_m": (
                final_displacement_error_xyz_m[
                    endpoint_index, case_index
                ].tolist()
            ),
            "final_displacement_error_l2_m": float(
                final_displacement_error_l2_m[endpoint_index, case_index]
            ),
            "path_tracking_rmse_m": float(
                path_tracking_rmse_m[endpoint_index, case_index]
            ),
            "path_tracking_max_error_m": float(
                path_tracking_max_error_m[endpoint_index, case_index]
            ),
            "path_tracking_sample_count": int(
                path_tracking_sample_count[endpoint_index, case_index]
            ),
            # Compatibility aliases retained within the new schema.
            "position_error_m": float(
                final_position_error_l2_m[endpoint_index, case_index]
            ),
            "euler_l2_error_deg": float(
                final_euler_error_l2_deg[endpoint_index, case_index]
            ),
            "position_error_xyz_m": final_position_error_xyz_m[
                endpoint_index, case_index
            ].tolist(),
            "euler_error_xyz_deg": final_euler_error_xyz_deg[
                endpoint_index, case_index
            ].tolist(),
            "contact_forces_n": trace["contact_forces"][
                endpoint_index, case_index
            ].tolist(),
            "actual_tip_height_m": actual_tip_height_m[
                endpoint_index, case_index
            ].tolist(),
            "actual_tip_height_fraction": actual_tip_fraction[
                endpoint_index, case_index
            ].tolist(),
            "grasp_plane_fraction": float(
                case_plane_fraction[endpoint_index]
            )
            if endpoint_plane_relevant
            else None,
            "grasp_plane_compliant": (
                bool(case_plane_compliant[endpoint_index])
                if endpoint_plane_relevant
                else None
            ),
            "commanded_grasp_plane_fraction": float(
                commanded_plane_fraction[endpoint_index, case_index]
            )
            if endpoint_is_live
            else None,
            "commanded_grasp_plane_compliant": (
                bool(commanded_plane_compliant[endpoint_index, case_index])
                if endpoint_is_live
                else None
            ),
        },
        "precondition": {
            "position_error_mm": float(
                1000.0
                * trace["position_error"][
                    precondition_index, case_index
                ]
            ),
            "position_error_xyz_mm": (
                1000.0
                * trace["position_error_xyz"][
                    precondition_index, case_index
                ]
            ).tolist(),
            "lift_mm": float(
                1000.0
                * (precondition_position[2] - initial_position[2])
            ),
            "horizontal_drift_mm": float(
                1000.0
                * np.linalg.norm(
                    precondition_position[:2] - initial_position[:2]
                )
            ),
            "orientation_drift_deg": float(
                np.linalg.norm(precondition_rotation)
            ),
            "linear_speed_mm_s": float(
                1000.0
                * np.linalg.norm(
                    trace["velocity"][precondition_index, case_index, :3]
                )
            ),
            "final_contact_forces_n": trace["contact_forces"][
                precondition_index, case_index
            ].tolist(),
        },
        "grasp_plane": {
            "measurement": "physical_hard_tip_positions",
            "mode": "learned" if plane_is_learned else "fixed",
            "fraction_bounds": list(plane_bounds),
            "target_fraction": plane_target,
            "fixed_yaw_rad": (
                None
                if plane_is_learned
                else contract.FIXED_GRASP_PLANE_YAW_RAD
            ),
            "physical_tolerance_m": (
                None
                if plane_is_learned
                else contract.FIXED_GRASP_PLANE_PHYSICAL_TOLERANCE_M
            ),
            "minimum_fraction": (
                float(relevant_fraction.min()) if has_relevant_plane else None
            ),
            "maximum_fraction": (
                float(relevant_fraction.max()) if has_relevant_plane else None
            ),
            "mean_fraction": (
                float(relevant_fraction.mean()) if has_relevant_plane else None
            ),
            "compliance_rate": (
                float(relevant_compliance.mean())
                if has_relevant_plane
                else None
            ),
            "relevant_sample_count": int(relevant_fraction.size),
            "all_relevant_control_ticks_compliant": (
                bool(relevant_compliance.all())
                if has_relevant_plane
                else None
            ),
            "commanded": {
                "minimum_fraction": (
                    float(live_commanded_fraction.min())
                    if has_live_command
                    else None
                ),
                "maximum_fraction": (
                    float(live_commanded_fraction.max())
                    if has_live_command
                    else None
                ),
                "compliance_rate": (
                    float(live_commanded_compliance.mean())
                    if has_live_command
                    else None
                ),
                "sample_count": int(live_commanded_fraction.size),
            },
        },
        "action_summary": {
            "mean_abs_applied_normalized": np.mean(
                np.abs(action_values), axis=0
            ).tolist(),
            "mean_abs_requested_normalized": np.mean(
                np.abs(requested_action), axis=0
            ).tolist(),
            "final_applied_normalized": action_values[
                endpoint_index
            ].tolist(),
        },
        "action_physical": {
            "tip_height_m": trace["tip_height_m"][
                plan_index, case_index
            ].tolist(),
            "actual_tip_height_m": actual_tip_height_m[
                plan_index, case_index
            ].tolist(),
            "tip_height_fraction": tip_fraction[
                plan_index, case_index
            ].tolist(),
            "actual_tip_height_fraction": actual_tip_fraction[
                plan_index, case_index
            ].tolist(),
            "tip_force_scale": force_scale[
                plan_index, case_index
            ].tolist(),
        },
        "trajectory": trajectory,
    }


def evaluate_paired(
    *,
    model_path: Path,
    output_path: Path,
    training_manifest: Path | None,
    episodes_per_case: int,
    seed: int,
    trajectory_stride: int,
    task: str | None = None,
) -> Path:
    """Evaluate one checkpoint-bound task across all objects and phases."""

    import torch

    if episodes_per_case != contract.RESET_TEMPLATES_PER_OBJECT:
        raise ValueError(
            "formal paired evaluation requires episodes_per_case == "
            f"{contract.RESET_TEMPLATES_PER_OBJECT}"
        )
    if trajectory_stride <= 0:
        raise ValueError("trajectory_stride must be positive")
    if model_path.name != "final_diffusion_model.pt":
        raise ValueError(
            "paired evaluation requires final_diffusion_model.pt"
        )
    checkpoint = _torch_checkpoint(torch, model_path)
    trained_task = _checkpoint_trained_task(
        checkpoint, requested_task=task
    )
    platform_spec = build_platform_spec(trained_task)
    config = _load_training_config(
        training_manifest,
        checkpoint=checkpoint,
        model_path=model_path,
        platform_spec=platform_spec,
    )
    policy, loaded_checkpoint = load_policy_checkpoint(
        model_path,
        device=config.device,
        use_ema=True,
        eval_mode=True,
        require_environment_dimensions=True,
    )
    _validate_checkpoint(
        loaded_checkpoint,
        policy=policy,
        config=config,
        platform_spec=platform_spec,
    )

    pairs: list[dict[str, Any]] = []
    translation_path_coverage: dict[str, dict[str, Any]] = {}
    for phase_index, phase in enumerate((PHASE,)):
        phase_cases = _cases(
            phase,
            episodes_per_case=episodes_per_case,
            seed=seed,
            task=trained_task,
        )
        translation_path_coverage[phase] = (
            _translation_path_case_coverage(phase_cases)
        )
        environment = make_env(
            phase,
            config=config,
            platform_spec=platform_spec,
            worlds=len(phase_cases),
            device=config.device,
            require_linux=True,
        )
        try:
            if config.cuda_graph:
                # Capture begins only after one valid free-joint reset.
                environment.reset()
                environment.backend.prepare_cuda_graph()
            traces = _rollout_phase(
                environment=environment,
                policy=policy,
                cases=phase_cases,
                config=config,
                seed=seed + phase_index * 1_000_000,
            )
        finally:
            environment.close()
        for case_index, case in enumerate(phase_cases):
            pairs.append(
                {
                    "pair_id": (
                        f"{case['object']}:{case['task']}:{phase}:"
                        f"{case['episode']:05d}"
                    ),
                    "object": case["object"],
                    "task": case["task"],
                    "phase": phase,
                    "seed": case["seed"],
                    "template_index": case["template_index"],
                    "template_seed": case["template_seed"],
                    "path_seed": case["path_seed"],
                    "path_shape_index": case["path_shape_index"],
                    "path_shape": case["path_shape"],
                    "path_step_m": case["path_step_m"],
                    "path_length_m": case["path_length_m"],
                    "path_waypoints_xy_m": case[
                        "path_waypoints_xy_m"
                    ],
                    "source_region": case["source_region"],
                    "rotation_angle_deg": case["rotation_angle_deg"],
                    "ik_only": _result_from_trace(
                        traces["ik_only"],
                        case_index=case_index,
                        object_kind=case["object"],
                        task=case["task"],
                        config=config,
                        trajectory_stride=trajectory_stride,
                    ),
                    "ik_diffusion": _result_from_trace(
                        traces["ik_diffusion"],
                        case_index=case_index,
                        object_kind=case["object"],
                        task=case["task"],
                        config=config,
                        trajectory_stride=trajectory_stride,
                    ),
                }
            )
    payload = {
        "schema": SCHEMA,
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform_spec.platform,
            "trained_task": trained_task,
            "task": trained_task,
            "task_index": contract.TASKS.index(trained_task),
            "root_height_m": float(platform_spec.root_height_m),
            "physics_backend": contract.PHYSICS_BACKEND,
            "torch_warp_exchange": contract.TORCH_WARP_EXCHANGE,
            "algorithm": contract.ALGORITHM,
            "algorithm_signature": contract.ALGORITHM_SIGNATURE,
            "runtime_code_fingerprint_sha256": (
                runtime_code_fingerprint_sha256()
            ),
            "control_hierarchy": (
                "Diffusion RL -> independent hard-tip IK commands -> "
                "per-tip constrained IK/KKT -> actuators"
            ),
            "controllers": ["ik_only", "ik_diffusion"],
            "observation_semantics": (
                "fully_observed mechanism-independent Markov state"
            ),
            "physics_layout": contract.PHYSICS_LAYOUT,
            "model_shards": ordered_shard_manifest(
                platform_spec, config.object_world_counts
            ),
            "training_global_worlds": config.num_envs,
            "training_object_world_counts": list(
                config.object_world_counts
            ),
            "platform_adapter_fingerprint_sha256": (
                platform_spec.platform_adapter_fingerprint_sha256
            ),
            "training_remainder_worlds": config.remainder_worlds,
            "training_remainder_object_world_counts": list(
                config.remainder_object_world_counts
            ),
            "policy_path": str(model_path),
            "policy_sha256": _sha256(model_path),
            "policy_weights": "EMA",
            "same_parent_model_for_both_controllers": True,
            "paired_reset_is_bit_exact": True,
            "plate_evaluated": False,
            "anti_support_gate_used": False,
            "object_drop_termination": {
                "measurement": "body_xpos_z_plus_object_local_bottom_z",
                "table_top_z_m": contract.TABLE_TOP_Z_M,
                "margin_m": contract.DROP_MARGIN_M,
                "threshold_z_m": contract.OBJECT_DROP_THRESHOLD_Z_M,
                "continuation": "sticky_absorbing_quarantine_to_horizon",
            },
            "independent_hard_tip_control": True,
            "grasp_plane_fraction_bounds": [
                contract.GRASP_PLANE_MIN_FRACTION,
                contract.GRASP_PLANE_MAX_FRACTION,
            ],
            "learned_grasp_plane_objects": [
                object_kind
                for object_kind in contract.OBJECTS
                if object_kind in contract.LEARNED_GRASP_PLANE_OBJECTS
            ],
            "fixed_grasp_plane_fractions": {
                object_kind: float(
                    contract.FIXED_GRASP_PLANE_FRACTIONS[object_kind]
                )
                for object_kind in contract.OBJECTS
                if object_kind in contract.FIXED_GRASP_PLANE_FRACTIONS
            },
            "fixed_grasp_plane_yaw_rad": (
                contract.FIXED_GRASP_PLANE_YAW_RAD
            ),
            "fixed_grasp_plane_physical_tolerance_m": (
                contract.FIXED_GRASP_PLANE_PHYSICAL_TOLERANCE_M
            ),
            "episodes_per_object_task": episodes_per_case,
            "reset_templates_per_object": (
                contract.RESET_TEMPLATES_PER_OBJECT
            ),
            "reset_template_seed_scheme": (
                "base + object_index * object_stride + template_index"
            ),
            "reset_template_seed_base": contract.RESET_TEMPLATE_SEED_BASE,
            "reset_template_object_stride": (
                contract.RESET_TEMPLATE_OBJECT_STRIDE
            ),
            "translation_reference": contract.ACTION_REFERENCE_CAPTURE,
            "translation_path_contract": (
                contract.translation_path_contract_payload()
            ),
            "translation_path_case_coverage": translation_path_coverage,
            "success_tolerances": _success_tolerances(),
            "rotation_angle_distribution_deg": {
                "bounds": [
                    contract.ROTATION_ANGLE_MIN_DEG,
                    contract.ROTATION_ANGLE_MAX_DEG,
                ],
                "sampling": "deterministic_episode_ordinal_modulo_span",
            },
        },
        "pairs": pairs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="path to final_diffusion_model.pt",
    )
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--task", choices=contract.TASKS, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("paired_evaluation.json")
    )
    parser.add_argument("--episodes-per-case", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20_260_729)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    args = parser.parse_args(argv)
    output = evaluate_paired(
        model_path=args.model.resolve(),
        output_path=args.output.resolve(),
        training_manifest=(
            args.training_manifest.resolve()
            if args.training_manifest is not None
            else None
        ),
        episodes_per_case=args.episodes_per_case,
        seed=args.seed,
        trajectory_stride=args.trajectory_stride,
        task=args.task,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
