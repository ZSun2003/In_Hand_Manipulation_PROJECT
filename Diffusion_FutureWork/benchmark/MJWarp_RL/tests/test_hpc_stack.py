from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import time
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from MJWarp_RL import contract
from MJWarp_RL.env import (
    ACTION_CONTACT_FORCE_N,
    ACTION_FORCE_RAMP_TICKS,
    CONSTRAINT_DAMPING,
    CONTACT_FORCE_THRESHOLD_N,
    EXPERT_OBJECT_ANGULAR_SPEED_MAX_RAD_S,
    EXPERT_OBJECT_LINEAR_SPEED_MAX_M_S,
    EXPERT_STABLE_TICKS,
    FORCE_FEEDBACK_MAX_M,
    FORCE_FEEDBACK_MIN_M,
    GRASP_CONTACT_FORCE_N,
    MJWarpIKResidualEnv,
    NOMINAL_PRELOAD_BY_OBJECT_M,
    OBSERVATION_DIM,
    OBSERVATION_SCHEMA,
    OBSERVATION_SLICES,
    POSITION_FEEDBACK_SLEW_M_PER_TICK,
    PRELOAD_TRIM_SLEW_M_PER_TICK,
    TIP_TARGET_ACCELERATION_MAX_M_S2,
    TIP_TARGET_JERK_MAX_M_S3,
    TIP_TARGET_SPEED_MAX_M_S,
    _cage_rotation_fraction,
    _hold_boundaries,
    _hold_reference_capture,
    _ik_free_mask_at_bounds,
    _object_drop_transition,
    _post_lift_reference,
    _quat_rotate,
    _reported_action_fraction,
    _relative_euler_xyz_deg,
    action_mask_from_observation,
)
from MJWarp_RL.capacity_scan import (
    CANDIDATES,
    REPORT_SCHEMA,
    _capture_native_pre_step_diagnostic,
    _diagnostic_run_lifecycle,
    _first_failure_transition_diagnostic,
    _run_lifecycle_counts,
    _unexpected_early_done as _capacity_unexpected_early_done,
    _validate_diagnostic_config,
    _worker_config,
    _worker_lifecycle_metadata,
)
from MJWarp_RL.curriculum import PHASE
from MJWarp_RL.evaluate import (
    _unexpected_early_done as _evaluation_unexpected_early_done,
)
from MJWarp_RL.train import (
    _rotation_success_breakdown,
    _success_breakdown_rates,
    _unexpected_early_done as _training_unexpected_early_done,
)
from MJWarp_RL.hpc_config import (
    CAPACITY_REPORT_SCHEMA,
    DEFAULT_CONFIG,
    FORMAL_WORLD_CANDIDATES,
)
from MJWarp_RL.newton_regularization import (
    NEWTON_ABSOLUTE_DIAGONAL_SHIFT,
    NEWTON_REGULARIZATION_SCHEMA,
    NEWTON_RELATIVE_DIAGONAL_SHIFT,
    PATCH_TARGET,
    PINNED_BLOCK_CHOLESKY_DIM,
    PINNED_SOLVER_SOURCE_SHA256,
    formal_newton_regularization_metadata,
    formal_newton_regularization_metadata_matches,
)
from MJWarp_RL.platform_adapter import (
    EXPECTED_ACTUATORS,
    MODEL_XMLS_BY_TASK,
    PACKAGE_DIR,
    PARENT_MODEL_XML,
    PLATFORM,
    PlatformSpec,
    ShardedPlatformSpec,
    TASK_GRIPPER_XMLS,
    TASK_ROOT_HEIGHT_M,
    model_xmls_for_task,
)


def _install_grasp_plane_contract_tensors(
    environment: object, torch_module: object
) -> None:
    """Install constructor-owned plane tensors on a lightweight fake env."""

    torch = torch_module
    environment._grasp_plane_nominal_fraction_by_object = torch.tensor(
        [
            contract.grasp_plane_nominal_fraction(kind)
            for kind in contract.OBJECTS
        ],
        dtype=torch.float32,
    )
    environment._grasp_plane_fraction_limits_by_object = torch.tensor(
        [
            contract.grasp_plane_fraction_limits(kind)
            for kind in contract.OBJECTS
        ],
        dtype=torch.float32,
    )
    environment._grasp_plane_learned_by_object = torch.tensor(
        [
            contract.grasp_plane_is_learned(kind)
            for kind in contract.OBJECTS
        ],
        dtype=torch.bool,
    )
    environment._fruit_by_object = torch.tensor(
        [kind in contract.FRUIT_OBJECTS for kind in contract.OBJECTS],
        dtype=torch.bool,
    )


def _fake_model_shards(
    world_counts: tuple[int, ...],
    *,
    digest_character: str = "e",
    task: str = contract.TASK_TRANSLATE,
    root_height_m: float | None = None,
) -> list[dict[str, object]]:
    task = contract.validate_task(task)
    task_index = contract.TASKS.index(task)
    root_height_m = float(
        TASK_ROOT_HEIGHT_M[task]
        if root_height_m is None
        else root_height_m
    )
    return [
        {
            "object_index": index,
            "object": object_name,
            "task": task,
            "task_index": task_index,
            "root_height_m": root_height_m,
            "model_xml": f"/fake/{object_name}.xml",
            "model_xml_sha256": digest_character * 64,
            "nq": 27,
            "nv": 26,
            "nu": EXPECTED_ACTUATORS,
            "worlds": int(worlds),
        }
        for index, (object_name, worlds) in enumerate(
            zip(contract.OBJECTS, world_counts)
        )
    ]


def _fake_sharded_health(
    model_fingerprint: str = "a" * 64,
    controller_fingerprint: str = "b" * 64,
    world_counts: tuple[int, ...] = contract.FORMAL_OBJECT_WORLD_COUNTS,
    adapter_fingerprint: str = "c" * 64,
) -> dict[str, object]:
    shard = {
        **formal_newton_regularization_metadata(),
        "finite": True,
        "model_nv": 26,
        "solver_option": 2,
        "solver_name": "Newton",
        "requested_solver_tolerance": 1.0e-10,
        "effective_solver_tolerance": 1.0e-6,
        "jacobian_option": 0,
        "resolved_sparse_jacobian": False,
        "sticky_nonfinite": 0,
        "sticky_overflow": 0,
        "sticky_contacts_full": 0,
        "sticky_constraints_full": 0,
        "sticky_external_nonfinite": 0,
    }
    return {
        **shard,
        "worlds": sum(world_counts),
        "shard_count": contract.OBJECT_SHARD_COUNT,
        "world_allocation": dict(
            zip(contract.OBJECTS, world_counts)
        ),
        "by_object": {
            name: dict(shard) for name in contract.OBJECTS
        },
        "parent_model_fingerprint_sha256": model_fingerprint,
        "parent_controller_fingerprint_sha256": controller_fingerprint,
        "platform_adapter_fingerprint_sha256": adapter_fingerprint,
    }


def _fake_platform_spec(
    model_fingerprint: str,
    controller_fingerprint: str,
    adapter_fingerprint: str = "c" * 64,
    rl_model_fingerprint: str = "9" * 64,
    task: str = contract.TASK_TRANSLATE,
) -> SimpleNamespace:
    task = contract.validate_task(task)
    task_index = contract.TASKS.index(task)
    root_height_m = float(TASK_ROOT_HEIGHT_M[task])

    def contact_bank(index: int) -> np.ndarray:
        bank = np.full(
            (
                len(contract.OBJECTS),
                contract.RESET_TEMPLATES_PER_OBJECT,
                len(contract.TIP_ORDER),
                3,
            ),
            np.nan,
        )
        bank[index, :, :, 2] = 0.5
        return bank

    shards = tuple(
        SimpleNamespace(
            task=task,
            task_index=task_index,
            root_height_m=root_height_m,
            fixed_object_index=index,
            model_xml=Path(f"/fake/{name}.xml"),
            model_xml_sha256="e" * 64,
            host_model=SimpleNamespace(
                nq=27, nv=26, nu=EXPECTED_ACTUATORS
            ),
            planned_grasp_plane_fraction=np.where(
                np.arange(len(contract.OBJECTS))[:, None] == index,
                np.full(
                    (
                        len(contract.OBJECTS),
                        contract.RESET_TEMPLATES_PER_OBJECT,
                    ),
                    0.5,
                ),
                np.nan,
            ),
            planned_grasp_plane_yaw_rad=np.where(
                np.arange(len(contract.OBJECTS))[:, None] == index,
                np.zeros(
                    (
                        len(contract.OBJECTS),
                        contract.RESET_TEMPLATES_PER_OBJECT,
                    )
                ),
                np.nan,
            ),
            contact_targets_local=contact_bank(index),
            object_local_bottom_z_m=np.zeros(len(contract.OBJECTS)),
            object_local_top_z_m=np.ones(len(contract.OBJECTS)),
        )
        for index, name in enumerate(contract.OBJECTS)
    )
    return SimpleNamespace(
        platform=PLATFORM,
        task=task,
        task_index=task_index,
        root_height_m=root_height_m,
        parent_model_fingerprint_sha256=model_fingerprint,
        parent_controller_fingerprint_sha256=controller_fingerprint,
        platform_adapter_fingerprint_sha256=adapter_fingerprint,
        rl_model_fingerprint_sha256=rl_model_fingerprint,
        shards=shards,
    )


_CONTROLLER_COUNTER_NAMES = (
    "ik_solve_fallback_count",
    "ik_trust_region_projection_count",
    "position_feedback_slew_clip_count",
    "preload_trim_slew_clip_count",
    "ctrl_slew_clip_count",
    "ctrl_certificate_fallback_count",
)


def _fake_controller_counters(**overrides: int) -> dict[str, int]:
    counters = {name: 0 for name in _CONTROLLER_COUNTER_NAMES}
    counters.update(overrides)
    return counters


def _fake_translation_path_coverage(
    phase: str,
    world_counts: tuple[int, ...] = contract.FORMAL_OBJECT_WORLD_COUNTS,
    *,
    task: str = contract.TASK_TRANSLATE,
) -> dict[str, object]:
    task = contract.validate_task(task)
    if task == contract.TASK_ROTATE:
        from MJWarp_RL.capacity_scan import _inactive_goal_coverage

        return _inactive_goal_coverage(
            coverage="translation_path",
            task=task,
            phase=phase,
            world_counts=world_counts,
        )
    if phase != PHASE:
        raise ValueError(f"unsupported curriculum phase: {phase!r}")
    source = 0
    shape_count = len(contract.TRANSLATION_PATH_SHAPES)
    by_object: dict[str, object] = {}
    total_translation = 0
    total_rotation = 0
    for object_name, world_count in zip(contract.OBJECTS, world_counts):
        translation_count = int(world_count)
        rotation_count = 0
        base, extra = divmod(translation_count, shape_count)
        shape_counts = {
            name: base + (index < extra)
            for index, name in enumerate(contract.TRANSLATION_PATH_SHAPES)
        }
        by_object[object_name] = {
            "world_count": int(world_count),
            "translation_goal_count": translation_count,
            "rotation_goal_count": rotation_count,
            "shape_index_values": list(range(shape_count)),
            "shape_counts_by_name": shape_counts,
            "seed_min": contract.TRANSLATION_PATH_SEED_MIN,
            "seed_max": contract.TRANSLATION_PATH_SEED_MAX,
            "seed_unique_count": 1,
            "source_region_values": [source],
            "rotation_source_region_values": [],
            "path_step_min_m": min(contract.TRANSLATION_PATH_STEP_M_BY_SHAPE),
            "path_step_max_m": max(contract.TRANSLATION_PATH_STEP_M_BY_SHAPE),
            "path_length_min_m": min(contract.TRANSLATION_PATH_LENGTH_M_BY_SHAPE),
            "path_length_max_m": max(contract.TRANSLATION_PATH_LENGTH_M_BY_SHAPE),
            "waypoint_count": contract.TRANSLATION_PATH_WAYPOINTS,
            "maximum_waypoint_radius_m": 0.05,
            "maximum_start_radius_m": contract.TRANSLATION_PATH_START_RADIUS_MAX_M,
            "maximum_arc_step_error_m": 0.0,
            "maximum_arc_length_error_m": 0.0,
            "maximum_canonical_waypoint_error_m": 0.0,
            "path_fields_finite": True,
            "rotation_path_fields_are_zero": True,
        }
        total_translation += translation_count
        total_rotation += rotation_count
    return {
        "active": True,
        "coverage": "translation_path",
        "task": task,
        "task_index": contract.TASKS.index(task),
        "phase": phase,
        "path_family": contract.TRANSLATION_PATH_FAMILY,
        "segments": contract.TRANSLATION_PATH_SEGMENTS,
        "waypoints": contract.TRANSLATION_PATH_WAYPOINTS,
        "shape_names": list(contract.TRANSLATION_PATH_SHAPES),
        "expected_source_region": source,
        "path_seed_provenance_sentinel": [
            contract.TRANSLATION_PATH_SEED_MIN,
            contract.TRANSLATION_PATH_SEED_MAX,
        ],
        "expected_arc_step_m_by_shape": list(
            contract.TRANSLATION_PATH_STEP_M_BY_SHAPE
        ),
        "expected_arc_length_m_by_shape": list(
            contract.TRANSLATION_PATH_LENGTH_M_BY_SHAPE
        ),
        "fixed_geometry_phase_invariant": True,
        "waypoint_radius_limit_m": contract.TRANSLATION_PATH_REGION_RADIUS_M,
        "generation_radius_limit_m": (
            contract.TRANSLATION_PATH_GENERATION_RADIUS_M
        ),
        "start_radius_limit_m": contract.TRANSLATION_PATH_START_RADIUS_MAX_M,
        "equal_segment_tolerance_m": 2.0e-6,
        "translation_goal_count": total_translation,
        "rotation_goal_count": total_rotation,
        "worlds": sum(world_counts),
        "rotation_path_fields_are_zero": True,
        "by_object": by_object,
    }


def _fake_rotation_goal_coverage(
    phase: str,
    world_counts: tuple[int, ...] = contract.FORMAL_OBJECT_WORLD_COUNTS,
    *,
    task: str = contract.TASK_TRANSLATE,
) -> dict[str, object]:
    task = contract.validate_task(task)
    if task == contract.TASK_TRANSLATE:
        from MJWarp_RL.capacity_scan import _inactive_goal_coverage

        return _inactive_goal_coverage(
            coverage="rotation",
            task=task,
            phase=phase,
            world_counts=world_counts,
        )
    if phase != PHASE:
        raise ValueError(f"unsupported curriculum phase: {phase!r}")
    low = contract.ROTATION_ANGLE_MIN_DEG
    high = contract.ROTATION_ANGLE_MAX_DEG
    source = 0
    return {
        "active": True,
        "coverage": "rotation",
        "task": task,
        "task_index": contract.TASKS.index(task),
        "phase": phase,
        "worlds": sum(world_counts),
        "closed_integer_bounds_deg": [low, high],
        "translation_angles_are_zero": True,
        "by_object": {
            name: {
                "active": True,
                "world_count": int(worlds),
                "angle_min_deg": low,
                "angle_max_deg": high,
                "angle_unique_count": high - low + 1,
                "maximum_target_count": 1,
                "source_region_values": [source],
            }
            for name, worlds in zip(contract.OBJECTS, world_counts)
        },
    }


def _fake_certified_candidate_coverage(
    phase: str,
    *,
    task: str = contract.TASK_TRANSLATE,
    world_counts: tuple[int, ...] = contract.FORMAL_OBJECT_WORLD_COUNTS,
    population: int = DEFAULT_CONFIG.elite.population,
) -> dict[str, object]:
    task = contract.validate_task(task)
    group_counts = tuple(worlds // population for worlds in world_counts)
    templates = list(range(contract.RESET_TEMPLATES_PER_OBJECT))
    shapes = list(range(len(contract.TRANSLATION_PATH_SHAPES)))
    if phase != PHASE:
        raise ValueError(f"unsupported curriculum phase: {phase!r}")
    rotation_maximum = contract.ROTATION_ANGLE_MAX_DEG
    return {
        "passed": True,
        "phase": phase,
        "task": task,
        "task_index": contract.TASKS.index(task),
        "population": population,
        "groups": sum(group_counts),
        "formal_coverage_required": sum(world_counts)
        == contract.FORMAL_GLOBAL_WORLDS,
        "selection_rule": "first_valid_replica_no_reward",
        "valid_definition": (
            "success & path_tracking_certified & "
            "actual_grasp_plane_compliant & !object_dropped"
        ),
        "valid_world_count": sum(group_counts),
        "valid_group_count": sum(group_counts),
        "by_object": {
            name: {
                "group_count": int(groups),
                "valid_group_count": int(groups),
                "valid_template_indices": templates,
                "valid_template_count": len(templates),
                "all_reset_templates_covered": True,
                "translation_path_shapes": (
                    {
                        "active": True,
                        "valid_shape_indices": shapes,
                        "all_shapes_covered": True,
                    }
                    if task == contract.TASK_TRANSLATE
                    else {
                        "active": False,
                        "reason": "inactive_for_task",
                        "valid_shape_indices": [],
                        "all_shapes_covered": False,
                    }
                ),
                "rotation_maximum_angle": (
                    {
                        "active": True,
                        "maximum_angle_deg": rotation_maximum,
                        "valid_group_count": 1,
                        "covered": True,
                    }
                    if task == contract.TASK_ROTATE
                    else {
                        "active": False,
                        "reason": "inactive_for_task",
                        "maximum_angle_deg": None,
                        "valid_group_count": 0,
                        "covered": False,
                    }
                ),
            }
            for name, groups in zip(contract.OBJECTS, group_counts)
        },
    }


_FAKE_GPU_TOTAL_BYTES = 96 * 1024**3


def _fake_memory_segment() -> dict[str, object]:
    used = _FAKE_GPU_TOTAL_BYTES // 2
    return {
        "minimum_free_bytes": _FAKE_GPU_TOTAL_BYTES - used,
        "actual_peak_device_used_bytes": used,
        "actual_peak_device_used_gb": used / (1024.0**3),
        "actual_peak_device_memory_fraction": 0.5,
    }


def _fake_release_certificate(label: str) -> dict[str, object]:
    total = _FAKE_GPU_TOTAL_BYTES
    free = 80 * 1024**3
    baseline = {
        "label": f"pre_allocation:{label}",
        "free_bytes": free,
        "device_used_bytes": total - free,
        "torch_allocated_bytes": 0,
        "torch_reserved_bytes": 0,
    }
    current = {**baseline, "label": f"post_release:{label}"}
    return {
        "passed": True,
        "label": label,
        "measurement": "actual device free plus current Torch allocation",
        "total_device_bytes": total,
        "baseline": baseline,
        "post_release": current,
        "device_unrecovered_bytes": 0,
        "device_tolerance_bytes": int(total * 0.01),
        "torch_allocated_unrecovered_bytes": 0,
        "torch_allocated_tolerance_bytes": 64 * 1024**2,
        "torch_reserved_unrecovered_bytes": 0,
        "torch_reserved_tolerance_bytes": 256 * 1024**2,
    }


def _fake_stage_memory_certificate(
    stage: str,
    *,
    segments: dict[str, object],
    extra_samples: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    total = _FAKE_GPU_TOTAL_BYTES
    baseline_free = 80 * 1024**3
    minimum_free = min(
        int(segment["minimum_free_bytes"])
        for segment in segments.values()
    )
    peak_used = total - minimum_free
    baseline = {
        "label": "baseline",
        "free_bytes": baseline_free,
        "device_used_bytes": total - baseline_free,
        "torch_allocated_bytes": 0,
        "torch_reserved_bytes": 0,
    }
    checkpoints = [baseline, *extra_samples]
    details = {
        "measurement": (
            "background torch.cuda.mem_get_info device-wide polling"
        ),
        "includes_non_torch_cuda_allocations": True,
        "stage": stage,
        "total_bytes": total,
        "baseline_free_bytes": baseline_free,
        "minimum_free_bytes": minimum_free,
        "actual_peak_device_used_bytes": peak_used,
        "actual_peak_device_used_gb": peak_used / (1024.0**3),
        "actual_peak_device_memory_fraction": peak_used / total,
        "stage_peak_increment_bytes": baseline_free - minimum_free,
        "stage_peak_increment_gb": (
            baseline_free - minimum_free
        ) / (1024.0**3),
        "limit_fraction": 0.90,
        "poll_interval_s": 0.02,
        "background_sample_count": 2,
        "checkpoint_sample_count": len(checkpoints),
        "sample_count": len(checkpoints) + 2,
        "checkpoint_samples": checkpoints,
        "segments": segments,
        "poll_thread_errors": [],
        "valid": True,
        "validation_error": None,
    }
    return {
        "passed": True,
        "actual_device_memory": details,
        "actual_peak_device_used_bytes": peak_used,
        "actual_peak_device_used_gb": peak_used / (1024.0**3),
        "actual_peak_device_memory_fraction": peak_used / total,
    }


def _fake_native_protocol(
    model_fingerprint: str,
    controller_fingerprint: str,
    adapter_fingerprint: str,
    *,
    task: str = contract.TASK_TRANSLATE,
) -> dict[str, object]:
    task = contract.validate_task(task)
    task_config = replace(DEFAULT_CONFIG, task=task)
    zero = _fake_controller_counters()
    control_ticks = int(task_config.control_ticks)
    complete_episode_count, final_tail_ticks = divmod(
        control_ticks, task_config.episode_length
    )
    additional_complete_episodes = [
        {
            "episode_index": episode_index,
            "validated_ticks": task_config.episode_length,
            "complete": True,
            "transition_finite": True,
            "terminated_worlds": contract.FORMAL_GLOBAL_WORLDS,
            "object_drop_count": 0,
            "controller_safety_counters": dict(zero),
        }
        for episode_index in range(1, complete_episode_count)
    ]
    final_tail = {
        "episode_index": complete_episode_count if final_tail_ticks else None,
        "validated_ticks": final_tail_ticks,
        "complete": False,
        "transition_finite": True,
        "terminated_worlds": 0,
        "object_drop_count": 0,
        "controller_safety_counters": dict(zero),
    }
    label = "formal_global_batch"
    certificates = {label: _fake_release_certificate(label)}
    segments = {label: _fake_memory_segment()}
    full_episode = {
        "passed": True,
        "phase": PHASE,
        "worlds": contract.FORMAL_GLOBAL_WORLDS,
        "object_world_counts": list(contract.FORMAL_OBJECT_WORLD_COUNTS),
        "episode_control_ticks": task_config.episode_length,
        "run_control_ticks": control_ticks,
        "configured_run_control_ticks": control_ticks,
        "complete_episode_count": complete_episode_count,
        "final_tail_ticks": final_tail_ticks,
        "rotation_goal_coverage": _fake_rotation_goal_coverage(
            PHASE, task=task
        ),
        "certified_candidate_coverage": _fake_certified_candidate_coverage(
            PHASE, task=task
        ),
        "translation_path_goal_coverage": _fake_translation_path_coverage(
            PHASE, task=task
        ),
        "backend_health": _fake_sharded_health(
            model_fingerprint,
            controller_fingerprint,
            adapter_fingerprint=adapter_fingerprint,
        ),
        "controller_safety_counters": dict(zero),
        "completed_episode_controller_safety_counters": dict(zero),
        "post_extraction_lifecycle": {
            "passed": True,
            "additional_complete_episode_count": complete_episode_count - 1,
            "additional_complete_episodes": additional_complete_episodes,
            "final_tail": final_tail,
            "controller_safety_counters": dict(zero),
        },
        "actual_device_memory_segment": segments[label],
    }
    checkpoint_samples = tuple(
        sample
        for certificate in certificates.values()
        for sample in (
            certificate["baseline"],
            certificate["post_release"],
        )
    )
    return {
        **_fake_stage_memory_certificate(
            "native_search",
            segments=segments,
            extra_samples=checkpoint_samples,
        ),
        "backend_health": _fake_sharded_health(
            model_fingerprint,
            controller_fingerprint,
            adapter_fingerprint=adapter_fingerprint,
        ),
        "protocol_order": [label, f"{label}_release"],
        "remainder": None,
        "full_episode": full_episode,
        "post_release_baseline_certificates": certificates,
        "controller_safety_counters_total": dict(zero),
    }


class SharedHPCStackTests(unittest.TestCase):
    def test_pose_metrics_use_closed_per_axis_bounds_not_l2(self) -> None:
        import math
        import torch

        worlds = 2 * len(contract.OBJECTS)
        object_index = torch.arange(len(contract.OBJECTS)).repeat(2)
        xipos = torch.zeros((worlds, len(contract.OBJECTS), 3))
        xquat = torch.zeros((worlds, len(contract.OBJECTS), 4))
        xquat[..., 0] = 1.0
        half = math.radians(8.0) / 2.0
        rotated = torch.tensor(
            (math.cos(half), math.sin(half), 0.0, 0.0)
        )
        xquat[torch.arange(worlds), object_index] = rotated

        class FakeBackend:
            def tensor(self, name: str) -> torch.Tensor:
                return {"xipos": xipos, "xquat": xquat}[name]

        environment = MJWarpIKResidualEnv.__new__(MJWarpIKResidualEnv)
        environment.backend = FakeBackend()
        environment._rows = torch.arange(worlds)
        environment._object_body_ids = torch.arange(len(contract.OBJECTS))
        environment.object_index = object_index
        environment.task_index = torch.tensor(
            [contract.TASKS.index(contract.TASK_TRANSLATE)]
            * len(contract.OBJECTS)
            + [contract.TASKS.index(contract.TASK_ROTATE)]
            * len(contract.OBJECTS)
        )
        environment.goal_pos = torch.full((worlds, 3), 0.010)
        environment.goal_quat = torch.zeros((worlds, 4))
        environment.goal_quat[:, 0] = 1.0
        environment._fruit_by_object = torch.tensor(
            [kind in contract.FRUIT_OBJECTS for kind in contract.OBJECTS]
        )

        position_l2, euler_l2, success, _ = environment._pose_metrics()
        self.assertTrue(
            bool(
                (
                    position_l2
                    > contract.TRANSLATION_POSITION_AXIS_ABS_TOLERANCE_M
                ).all()
            )
        )
        self.assertTrue(torch.allclose(euler_l2, torch.full_like(euler_l2, 8.0)))
        self.assertTrue(bool(success.all()))

        environment.goal_pos[0, 0] = 0.0101
        _, _, success, _ = environment._pose_metrics()
        self.assertFalse(bool(success[0]))
        self.assertTrue(bool(success[1:].all()))

    def test_rotation_success_logging_is_split_by_requested_angle(self) -> None:
        import torch

        rotate = contract.TASKS.index(contract.TASK_ROTATE)
        translate = contract.TASKS.index(contract.TASK_TRANSLATE)
        breakdown = _rotation_success_breakdown(
            torch.tensor((True, False, True, True, True)),
            torch.tensor((rotate, rotate, translate, rotate, rotate)),
            torch.tensor((10, 30, 0, 30, 20)),
            phase=PHASE,
        )
        self.assertEqual(
            breakdown["10"], {"success_count": 1, "episode_count": 1}
        )
        self.assertEqual(
            breakdown["20"], {"success_count": 1, "episode_count": 1}
        )
        self.assertEqual(
            breakdown["30"], {"success_count": 1, "episode_count": 2}
        )
        self.assertEqual(
            breakdown["11"], {"success_count": 0, "episode_count": 0}
        )
        rates = _success_breakdown_rates(breakdown)
        self.assertEqual(rates["30"]["success_rate"], 0.5)
        self.assertIsNone(rates["11"]["success_rate"])
        with self.assertRaisesRegex(RuntimeError, "violate"):
            _rotation_success_breakdown(
                torch.tensor((True,)),
                torch.tensor((rotate,)),
                torch.tensor((contract.ROTATION_ANGLE_MAX_DEG + 1,)),
                phase=PHASE,
            )

    def test_native_failure_diagnostic_keeps_compact_pre_step_evidence(
        self,
    ) -> None:
        import torch

        views = {
            "qpos": torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
            ),
            "qvel": torch.tensor([[0.5, -0.25], [1.5, -2.0]]),
            "qacc_warmstart": torch.tensor(
                [[0.1, 0.2], [-0.3, 0.4]]
            ),
        }

        class FakeBackend:
            def tensor(self, name: str) -> torch.Tensor:
                return views[name]

            def has_field(self, name: str) -> bool:
                return name in views

        requested = torch.tensor(
            [[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]]
        )
        environment = SimpleNamespace(
            backend=FakeBackend(),
            _last_contact_force=torch.tensor(
                [[1.0, 2.0], [5.0, 6.0]]
            ),
            _last_force_decode_finite=torch.tensor([True, True]),
            filtered_action=torch.zeros_like(requested),
            previous_action=torch.zeros_like(requested),
        )
        pre_step = _capture_native_pre_step_diagnostic(
            torch=torch,
            environment=environment,
            requested_action=requested,
        )

        # The cache retains per-world summaries, not full qpos/qvel/qacc.
        for name in ("qpos", "qvel", "qacc_warmstart"):
            self.assertEqual(tuple(pre_step[name]["finite"].shape), (2,))
            self.assertEqual(tuple(pre_step[name]["max_abs"].shape), (2,))

        views["qpos"][1, 1] = float("nan")
        views["qpos"][1, 3] = float("inf")
        views["qvel"][1, 0] = float("nan")
        environment._last_contact_force.fill_(0.0)
        environment._last_force_decode_finite[1] = False
        environment.filtered_action[1] = torch.tensor([0.7, 0.8, 0.9])
        environment.previous_action[1] = torch.tensor([-0.7, -0.8, -0.9])
        requested.fill_(1.0)  # Prove requested action has independent storage.

        diagnostic = _first_failure_transition_diagnostic(
            torch=torch,
            environment=environment,
            pre_step=pre_step,
            first_world=1,
        )

        self.assertEqual(
            diagnostic["first_done_qpos_nonfinite_indices"], [1, 3]
        )
        self.assertEqual(
            diagnostic["first_done_qvel_nonfinite_indices"], [0]
        )
        self.assertTrue(diagnostic["first_done_previous_qpos_finite"])
        self.assertEqual(
            diagnostic["first_done_previous_qpos_max_abs"], 4.0
        )
        self.assertTrue(diagnostic["first_done_previous_qvel_finite"])
        self.assertEqual(
            diagnostic["first_done_previous_qvel_max_abs"], 2.0
        )
        self.assertTrue(
            diagnostic["first_done_previous_qacc_warmstart_finite"]
        )
        self.assertAlmostEqual(
            diagnostic["first_done_previous_qacc_warmstart_max_abs"],
            0.4,
        )
        self.assertEqual(
            diagnostic["first_done_previous_contact_force_n"], [5.0, 6.0]
        )
        self.assertTrue(
            diagnostic["first_done_previous_force_decode_finite"]
        )
        self.assertFalse(diagnostic["first_done_force_decode_finite"])
        np.testing.assert_allclose(
            diagnostic["first_done_requested_action"], [-0.4, 0.5, -0.6]
        )
        np.testing.assert_allclose(
            diagnostic["first_done_filtered_action"], [0.7, 0.8, 0.9]
        )
        np.testing.assert_allclose(
            diagnostic["first_done_used_action"], [-0.7, -0.8, -0.9]
        )

    def test_nominal_preload_exactly_matches_parent_controllers(self) -> None:
        self.assertEqual(
            dict(zip(contract.OBJECTS, NOMINAL_PRELOAD_BY_OBJECT_M)),
            {
                "orange": 0.0008,
                "banana": 0.0008,
                "pear": 0.0003,
                "strawberry": 0.0003,
            },
        )

    def test_kkt_regularization_and_per_world_fallback_are_finite(self) -> None:
        import torch

        self.assertEqual(CONSTRAINT_DAMPING, 1.0e-5)
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.torch = torch
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = 2
        environment._kkt_eye = {}
        jacobian = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            ]
        )
        constraint = torch.zeros_like(jacobian)
        error = torch.tensor(
            [[0.1, -0.2, 0.0], [float("nan"), 0.0, 0.0]]
        )
        pin_gap = torch.zeros((2, 3))
        free = torch.ones((2, 2))
        delta, feasible = environment._solve_kkt(
            jacobian, constraint, error, pin_gap, free
        )
        self.assertEqual(feasible.tolist(), [True, False])
        self.assertTrue(bool(torch.isfinite(delta).all()))
        self.assertTrue(bool((delta[1] == 0.0).all()))

    def test_finger_trust_region_preserves_direction_and_caps_ratio(self) -> None:
        import torch

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.dtype = torch.float32
        delta = torch.tensor(
            [[0.006, -0.110, 0.020], [0.0001, 0.001, -0.002]],
            dtype=torch.float32,
        )
        limits = torch.tensor(
            [0.002 / 3.0, 0.055 / 3.0, 0.055 / 3.0],
            dtype=torch.float32,
        )
        scaled, projected = environment._trust_region_delta(delta, limits)
        self.assertEqual(projected.tolist(), [True, False])
        self.assertTrue(
            bool(((scaled.abs() / limits[None, :]).amax(dim=1) <= 1.0).all())
        )
        scale = scaled[0, 0] / delta[0, 0]
        self.assertTrue(torch.allclose(scaled[0], delta[0] * scale))
        self.assertTrue(torch.equal(scaled[1], delta[1]))

    def test_second_order_control_limiter_fails_closed(self) -> None:
        import torch

        class FakeBackend:
            control_dt = 0.02

            def __init__(self) -> None:
                self.ctrl = torch.zeros((2, 2), dtype=torch.float32)

            def tensor(self, name: str) -> torch.Tensor:
                if name != "ctrl":
                    raise AssertionError(name)
                return self.ctrl

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = FakeBackend()
        environment.dtype = torch.float32
        environment._ctrl_low = torch.tensor([-1.0, -1.0])
        environment._ctrl_high = torch.tensor([1.0, 1.0])
        environment._ctrl_rate = torch.tensor([0.002, 0.055])
        environment._ctrl_acceleration_limit = torch.tensor([4.0, 8.0])
        environment._ctrl_jerk_limit = torch.tensor([40.0, 80.0])
        environment.ctrl_velocity = torch.zeros((2, 2))
        environment.ctrl_acceleration = torch.zeros((2, 2))
        environment.last_certified_ctrl = torch.zeros((2, 2))
        environment.ctrl_slew_clip_count = torch.zeros(2, dtype=torch.long)
        environment.ctrl_certificate_fallback_count = torch.zeros(
            2, dtype=torch.long
        )
        desired = torch.tensor([[1.0, 1.0], [float("nan"), 1.0]])
        value = environment._rate_limit_control(
            desired, torch.zeros(2, dtype=torch.bool)
        )
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertTrue(
            torch.allclose(value[0], torch.tensor([0.00032, 0.00064]))
        )
        self.assertTrue(torch.equal(value[1], torch.zeros(2)))
        self.assertTrue(torch.equal(environment.ctrl_velocity[1], torch.zeros(2)))
        self.assertEqual(
            environment.ctrl_certificate_fallback_count.tolist(), [0, 1]
        )

    def test_cartesian_tip_target_limiter_is_vector_bounded_and_fail_closed(
        self,
    ) -> None:
        import torch

        class FakeBackend:
            control_dt = 0.02

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = FakeBackend()
        environment.torch = torch
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = 2
        shape = (2, 4, 3)
        environment.last_certified_tip_targets = torch.zeros(shape)
        environment.tip_target_velocity_world_m_s = torch.zeros(shape)
        environment.tip_target_acceleration_world_m_s2 = torch.zeros(shape)
        environment.tip_target_jerk_world_m_s3 = torch.zeros(shape)
        environment.tip_target_slew_clip_count = torch.zeros(
            2, dtype=torch.long
        )
        environment.tip_target_certificate_fallback_count = torch.zeros(
            2, dtype=torch.long
        )
        environment._project_grasp_plane = MethodType(
            lambda self, value: value, environment
        )
        environment._grasp_plane_metrics = MethodType(
            lambda self, value: (
                torch.zeros(self.worlds),
                torch.zeros((self.worlds, len(contract.TIP_ORDER))),
                torch.ones(self.worlds, dtype=torch.bool),
            ),
            environment,
        )
        desired = torch.ones(shape)
        desired[1, 0, 0] = torch.nan
        previous = environment.last_certified_tip_targets.clone()
        value = environment._rate_limit_tip_targets(
            desired, torch.ones(2, dtype=torch.bool)
        )

        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertTrue(
            bool(
                (
                    environment.tip_target_velocity_world_m_s.norm(dim=-1)
                    <= TIP_TARGET_SPEED_MAX_M_S + 1.0e-6
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    environment.tip_target_acceleration_world_m_s2.norm(dim=-1)
                    <= TIP_TARGET_ACCELERATION_MAX_M_S2 + 1.0e-6
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    environment.tip_target_jerk_world_m_s3.norm(dim=-1)
                    <= TIP_TARGET_JERK_MAX_M_S3 + 1.0e-6
                ).all()
            )
        )
        torch.testing.assert_close(value[1], previous[1])
        self.assertEqual(
            environment.tip_target_certificate_fallback_count.tolist(),
            [0, 1],
        )

    def test_cartesian_tip_target_servo_never_crosses_a_stopped_target(
        self,
    ) -> None:
        import torch

        class FakeBackend:
            control_dt = 0.02

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = FakeBackend()
        environment.torch = torch
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = 1
        shape = (1, len(contract.TIP_ORDER), 3)
        environment.last_certified_tip_targets = torch.zeros(shape)
        environment.tip_target_velocity_world_m_s = torch.zeros(shape)
        environment.tip_target_acceleration_world_m_s2 = torch.zeros(shape)
        environment.tip_target_jerk_world_m_s3 = torch.zeros(shape)
        environment.tip_target_slew_clip_count = torch.zeros(1, dtype=torch.long)
        environment.tip_target_certificate_fallback_count = torch.zeros(
            1, dtype=torch.long
        )
        environment._project_grasp_plane = MethodType(
            lambda self, value: value, environment
        )
        environment._grasp_plane_metrics = MethodType(
            lambda self, value: (
                torch.zeros(self.worlds),
                torch.zeros((self.worlds, len(contract.TIP_ORDER))),
                torch.ones(self.worlds, dtype=torch.bool),
            ),
            environment,
        )
        desired = torch.zeros(shape)
        desired[:, :, 0] = 0.012
        active = torch.ones(1, dtype=torch.bool)

        maximum = environment.last_certified_tip_targets.clone()
        for _ in range(500):
            previous_target = environment.last_certified_tip_targets.clone()
            previous_velocity = (
                environment.tip_target_velocity_world_m_s.clone()
            )
            previous_acceleration = (
                environment.tip_target_acceleration_world_m_s2.clone()
            )
            value = environment._rate_limit_tip_targets(desired, active)
            self.assertTrue(bool((value[:, :, 0] <= 0.012 + 1.0e-9).all()))
            self.assertTrue(
                bool(
                    (
                        value[:, :, 0]
                        >= previous_target[:, :, 0] - 1.0e-9
                    ).all()
                )
            )
            maximum = torch.maximum(maximum, value)
            realized_velocity = (
                value - previous_target
            ) / environment.backend.control_dt
            realized_acceleration = (
                realized_velocity - previous_velocity
            ) / environment.backend.control_dt
            realized_jerk = (
                realized_acceleration - previous_acceleration
            ) / environment.backend.control_dt
            torch.testing.assert_close(
                environment.tip_target_velocity_world_m_s,
                realized_velocity,
            )
            torch.testing.assert_close(
                environment.tip_target_acceleration_world_m_s2,
                realized_acceleration,
            )
            torch.testing.assert_close(
                environment.tip_target_jerk_world_m_s3,
                realized_jerk,
            )

        self.assertLess(
            float((desired - value).norm(dim=-1).max()), 1.0e-5
        )
        self.assertLessEqual(float(maximum[:, :, 0].max()), 0.012 + 1.0e-9)
        self.assertEqual(
            environment.tip_target_certificate_fallback_count.tolist(), [0]
        )

    def test_translation_entry_and_terminal_gates_require_25_consecutive_ticks(
        self,
    ) -> None:
        import torch

        class FakeBackend:
            def __init__(self) -> None:
                self.views = {
                    "xipos": torch.zeros((1, 1, 3)),
                    "xquat": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
                }

            def tensor(self, name: str) -> torch.Tensor:
                return self.views[name]

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = FakeBackend()
        environment.torch = torch
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = 1
        environment.config = DEFAULT_CONFIG
        environment._rows = torch.tensor([0])
        environment._object_body_ids = torch.tensor([0, 0, 0, 0])
        environment.object_index = torch.tensor([0])
        environment.task_index = torch.tensor(
            [contract.TASKS.index(contract.TASK_TRANSLATE)]
        )
        environment.action_reference_valid = torch.tensor([True])
        environment.action_start_com = torch.zeros((1, 3))
        environment.action_start_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]]
        )
        environment.path_waypoints_xy_m = torch.zeros((1, 21, 2))
        environment.object_dropped = torch.tensor([False])
        environment.translation_entry_stable_count = torch.zeros(
            1, dtype=torch.long
        )
        environment.translation_entry_max_stable_count = torch.zeros(
            1, dtype=torch.long
        )
        environment.translation_entry_certificate_latched = torch.tensor([False])
        environment.translation_entry_certificate_first_control_step = torch.full(
            (1,), -1, dtype=torch.long
        )
        environment.translation_route_admitted = torch.tensor([False])
        environment.translation_entry_route_start_safety_valid = torch.tensor([False])
        environment.translation_entry_gate_current = torch.tensor([False])
        environment.translation_entry_tip_height_valid = torch.tensor([False])
        environment.translation_entry_lifted = torch.tensor([False])
        environment.translation_entry_pose_valid = torch.tensor([False])
        for name in (
            "translation_entry_start_position_valid",
            "translation_entry_start_orientation_valid",
            "translation_entry_linear_speed_valid",
            "translation_entry_angular_speed_valid",
        ):
            setattr(environment, name, torch.tensor([False]))
        environment.translation_entry_predicate_current = torch.zeros(
            (1, 9), dtype=torch.bool
        )
        environment.translation_entry_failure_count_by_predicate = torch.zeros(
            (1, 9), dtype=torch.long
        )
        environment.translation_entry_first_failure_control_step = torch.full(
            (1, 9), -1, dtype=torch.long
        )
        environment.translation_entry_last_failure_control_step = torch.full(
            (1, 9), -1, dtype=torch.long
        )
        for name in (
            "translation_entry_tip_height_fraction",
            "translation_entry_tip_contact_force_n",
        ):
            setattr(environment, name, torch.zeros((1, 4)))
        for name in (
            "translation_entry_tip_object_pair",
            "translation_entry_tip_object_raw_pair",
            "translation_entry_tip_object_active_pair",
            "translation_entry_tip_contact_qualified",
            "translation_entry_tip_force_valid",
        ):
            setattr(environment, name, torch.zeros((1, 4), dtype=torch.bool))
        for name in (
            "translation_entry_tip_height_failure_count",
            "translation_entry_tip_pair_missing_count",
            "translation_entry_tip_raw_pair_missing_count",
            "translation_entry_tip_efc_inactive_count",
            "translation_entry_tip_force_failure_count",
        ):
            setattr(environment, name, torch.zeros((1, 4), dtype=torch.long))
        environment.translation_entry_position_error_xyz_abs_m = torch.zeros((1, 3))
        environment.translation_entry_euler_error_xyz_abs_deg = torch.zeros((1, 3))
        for name in (
            "translation_entry_estimated_bottom_z_m",
            "translation_entry_object_linear_speed_m_s",
            "translation_entry_object_angular_speed_rad_s",
        ):
            setattr(environment, name, torch.zeros(1))
        environment._last_force_decode_finite = torch.tensor([True])
        environment._last_tip_object_contact_pair = torch.ones(
            (1, 4), dtype=torch.bool
        )
        environment.expert_tip_object_contact = torch.zeros(
            (1, 4), dtype=torch.bool
        )
        environment.expert_contact_count = torch.zeros(1, dtype=torch.long)
        environment.expert_four_tip_contact = torch.tensor([False])
        environment.expert_object_linear_speed_m_s = torch.zeros(1)
        environment.expert_object_angular_speed_rad_s = torch.zeros(1)

        stable = True

        def physical_metrics(self, force):
            value = torch.tensor([stable])
            tip_contact = torch.tensor(
                [[True, True, True, stable]], dtype=torch.bool
            )
            contact_count = tip_contact.to(torch.long).sum(dim=1)
            return (
                value,
                tip_contact,
                contact_count,
                contact_count == 4,
                value,
                value,
                value,
                torch.tensor(
                    [0.5 * EXPERT_OBJECT_LINEAR_SPEED_MAX_M_S]
                ),
                torch.tensor(
                    [0.5 * EXPERT_OBJECT_ANGULAR_SPEED_MAX_RAD_S]
                ),
                torch.full((1, 4), 0.5),
                torch.tensor([contract.TABLE_TOP_Z_M + 0.010]),
            )

        environment._expert_physical_metrics = MethodType(
            physical_metrics, environment
        )
        environment._grasp_plane_contract_rows = MethodType(
            lambda self: (
                torch.tensor([0.5]),
                torch.tensor([0.4]),
                torch.tensor([0.6]),
                torch.tensor([True]),
            ),
            environment,
        )
        action_start = sum(
            (
                DEFAULT_CONFIG.plan_ticks,
                DEFAULT_CONFIG.close_ticks,
                DEFAULT_CONFIG.settle_ticks,
                DEFAULT_CONFIG.lift_ticks,
                DEFAULT_CONFIG.hold_ticks,
            )
        )
        route_start = (
            action_start
            + contract.TRANSLATION_PATH_LEAD_IN_TICKS
            + contract.TRANSLATION_PATH_START_SETTLE_TICKS
        )
        for tick in range(route_start - EXPERT_STABLE_TICKS + 1, route_start + 1):
            environment.control_step = torch.tensor([tick])
            environment._update_translation_entry_gate(torch.ones((1, 4)))
        self.assertEqual(
            environment.translation_entry_stable_count.item(),
            EXPERT_STABLE_TICKS,
        )
        self.assertTrue(environment.translation_entry_certificate_latched.item())
        self.assertTrue(environment.translation_route_admitted.item())

        environment.task_index.fill_(contract.TASKS.index(contract.TASK_ROTATE))
        environment.path_tracking_sample_count = torch.zeros(1, dtype=torch.long)
        environment.path_tracking_axis_abs_max_m = torch.zeros((1, 3))
        environment.path_schedule_complete = torch.tensor([False])
        environment.path_schedule_limit_valid = torch.tensor([False])
        environment.terminal_stable_count = torch.zeros(1, dtype=torch.long)
        environment.terminal_stable_certified = torch.tensor([False])
        environment.terminal_stability_gate_current = torch.tensor([False])
        for _ in range(EXPERT_STABLE_TICKS):
            certified = environment._update_terminal_stability_gate(
                pose_success=torch.tensor([True]),
                actual_plane_compliant=torch.tensor([True]),
                finger_force=torch.ones((1, 4)),
            )
        self.assertTrue(certified.item())
        stable = False
        certified = environment._update_terminal_stability_gate(
            pose_success=torch.tensor([True]),
            actual_plane_compliant=torch.tensor([True]),
            finger_force=torch.ones((1, 4)),
        )
        self.assertFalse(certified.item())
        self.assertEqual(environment.terminal_stable_count.item(), 0)

    def test_invalid_control_holds_current_command_without_jump(self) -> None:
        import torch

        class FakeBackend:
            control_dt = 0.02

            def __init__(self) -> None:
                self.ctrl = torch.tensor([[0.9]], dtype=torch.float32)

            def tensor(self, name: str) -> torch.Tensor:
                if name != "ctrl":
                    raise AssertionError(name)
                return self.ctrl

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = FakeBackend()
        environment.dtype = torch.float32
        environment._ctrl_low = torch.tensor([-1.0])
        environment._ctrl_high = torch.tensor([1.0])
        environment._ctrl_rate = torch.tensor([0.055])
        environment._ctrl_acceleration_limit = torch.tensor([8.0])
        environment._ctrl_jerk_limit = torch.tensor([80.0])
        environment.ctrl_velocity = torch.zeros((1, 1))
        environment.ctrl_acceleration = torch.zeros((1, 1))
        environment.last_certified_ctrl = torch.zeros((1, 1))
        environment.ctrl_slew_clip_count = torch.zeros(1, dtype=torch.long)
        environment.ctrl_certificate_fallback_count = torch.zeros(
            1, dtype=torch.long
        )
        value = environment._rate_limit_control(
            torch.tensor([[float("nan")]]), torch.tensor([False])
        )
        self.assertTrue(torch.equal(value, torch.tensor([[0.9]])))
        self.assertEqual(
            environment.ctrl_certificate_fallback_count.tolist(), [1]
        )

    def test_frozen_control_must_pass_dynamic_certificate(self) -> None:
        import torch

        class FakeBackend:
            control_dt = 0.02

            def __init__(self) -> None:
                self.ctrl = torch.tensor([[0.4], [0.4]], dtype=torch.float32)

            def tensor(self, name: str) -> torch.Tensor:
                if name != "ctrl":
                    raise AssertionError(name)
                return self.ctrl

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = FakeBackend()
        environment.dtype = torch.float32
        environment._ctrl_low = torch.tensor([-1.0])
        environment._ctrl_high = torch.tensor([1.0])
        environment._ctrl_rate = torch.tensor([0.055])
        environment._ctrl_acceleration_limit = torch.tensor([8.0])
        environment._ctrl_jerk_limit = torch.tensor([80.0])
        environment.ctrl_velocity = torch.tensor([[0.001], [0.2]])
        environment.ctrl_acceleration = torch.zeros((2, 1))
        environment.last_certified_ctrl = environment.backend.ctrl.clone()
        environment.ctrl_slew_clip_count = torch.zeros(2, dtype=torch.long)
        environment.ctrl_certificate_fallback_count = torch.zeros(
            2, dtype=torch.long
        )

        value = environment._rate_limit_control(
            environment.backend.ctrl.clone(),
            torch.ones(2, dtype=torch.bool),
        )

        self.assertTrue(torch.equal(value[0], environment.backend.ctrl[0]))
        self.assertGreater(float(value[1]), float(environment.backend.ctrl[1]))
        self.assertEqual(
            environment.ctrl_certificate_fallback_count.tolist(), [0, 0]
        )
        self.assertTrue(
            torch.allclose(
                environment.ctrl_acceleration[0], torch.tensor([-0.05])
            )
        )
        self.assertTrue(torch.equal(environment.ctrl_velocity[0], torch.zeros(1)))
        self.assertGreater(float(environment.ctrl_velocity[1]), 0.0)
        self.assertLess(float(environment.ctrl_velocity[1]), 0.2)

    def test_inactive_free_bodies_are_parked_without_touching_active_state(
        self,
    ) -> None:
        import torch

        worlds = 2
        free_body_count = len(contract.OBJECTS) + 1  # Includes parent plate.
        qpos_width = free_body_count * 7
        qvel_width = free_body_count * 6
        qpos = (
            torch.arange(worlds * qpos_width, dtype=torch.float32)
            .reshape(worlds, qpos_width)
            .add_(0.125)
        )
        qvel = (
            torch.arange(worlds * qvel_width, dtype=torch.float32)
            .reshape(worlds, qvel_width)
            .add_(10.25)
        )
        qacc_warmstart = qvel.clone().add_(100.0)

        class FakeBackend:
            def __init__(self) -> None:
                self.views = {
                    "qpos": qpos,
                    "qvel": qvel,
                    "qacc_warmstart": qacc_warmstart,
                }

            def tensor(self, name: str) -> torch.Tensor:
                return self.views[name]

            def has_field(self, name: str) -> bool:
                return name in self.views

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.torch = torch
        environment.backend = FakeBackend()
        environment.object_index = torch.tensor([0, 3], dtype=torch.long)
        environment._object_qpos_adrs = torch.arange(
            len(contract.OBJECTS), dtype=torch.long
        ) * 7
        environment._free_qpos_adr_values = tuple(
            7 * index for index in range(free_body_count)
        )
        environment._free_qvel_adr_values = tuple(
            6 * index for index in range(free_body_count)
        )
        environment._parked_free_qpos = (
            torch.arange(
                worlds * free_body_count * 7, dtype=torch.float32
            )
            .reshape(worlds, free_body_count, 7)
            .add_(1_000.5)
        )

        active_slots = (0, 3)
        active_before = []
        for world, slot in enumerate(active_slots):
            qpos_address = environment._free_qpos_adr_values[slot]
            qvel_address = environment._free_qvel_adr_values[slot]
            active_before.append(
                (
                    qpos[world, qpos_address : qpos_address + 7].clone(),
                    qvel[world, qvel_address : qvel_address + 6].clone(),
                    qacc_warmstart[
                        world, qvel_address : qvel_address + 6
                    ].clone(),
                )
            )

        environment._park_inactive_objects()

        for world, active_slot in enumerate(active_slots):
            for slot, (qpos_address, qvel_address) in enumerate(
                zip(
                    environment._free_qpos_adr_values,
                    environment._free_qvel_adr_values,
                )
            ):
                if slot == active_slot:
                    expected_qpos, expected_qvel, expected_qacc = (
                        active_before[world]
                    )
                    self.assertTrue(
                        torch.equal(
                            qpos[world, qpos_address : qpos_address + 7],
                            expected_qpos,
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            qvel[world, qvel_address : qvel_address + 6],
                            expected_qvel,
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            qacc_warmstart[
                                world, qvel_address : qvel_address + 6
                            ],
                            expected_qacc,
                        )
                    )
                    continue
                self.assertTrue(
                    torch.equal(
                        qpos[world, qpos_address : qpos_address + 7],
                        environment._parked_free_qpos[world, slot],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        qvel[world, qvel_address : qvel_address + 6],
                        torch.zeros(6),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        qacc_warmstart[
                            world, qvel_address : qvel_address + 6
                        ],
                        torch.zeros(6),
                    )
                )

        plate_slot = free_body_count - 1
        plate_qpos = environment._free_qpos_adr_values[plate_slot]
        plate_qvel = environment._free_qvel_adr_values[plate_slot]
        self.assertTrue(
            torch.equal(
                qpos[:, plate_qpos : plate_qpos + 7],
                environment._parked_free_qpos[:, plate_slot],
            )
        )
        self.assertTrue(
            torch.equal(qvel[:, plate_qvel : plate_qvel + 6], torch.zeros((2, 6)))
        )
        self.assertTrue(
            torch.equal(
                qacc_warmstart[:, plate_qvel : plate_qvel + 6],
                torch.zeros((2, 6)),
            )
        )

    def test_object_drop_terminal_is_sticky_and_reward_is_one_shot(self) -> None:
        import torch

        absorbing = torch.tensor([False, True, False, False])
        finite = torch.tensor([True, True, False, True])
        estimated_bottom = torch.tensor([0.249, 0.249, float("nan"), 0.250])
        raw_reward = torch.full((4,), 5.0)

        new, dropped, reward = _object_drop_transition(
            absorbing, finite, estimated_bottom, raw_reward
        )

        self.assertEqual(new.tolist(), [True, False, False, False])
        self.assertEqual(dropped.tolist(), [True, True, False, False])
        torch.testing.assert_close(
            reward,
            torch.tensor(
                [contract.OBJECT_DROP_TERMINAL_REWARD, 0.0,
                 contract.OBJECT_DROP_TERMINAL_REWARD, 5.0]
            ),
        )

    def test_absorbing_world_does_not_accumulate_ik_fallback(self) -> None:
        import torch

        qpos = torch.zeros((2, 2), dtype=torch.float32)
        ctrl = torch.zeros((2, 1), dtype=torch.float32)

        class FakeBackend:
            views = {"qpos": qpos, "ctrl": ctrl}

            def tensor(self, name: str) -> torch.Tensor:
                return self.views[name]

            def forward(self) -> None:
                return None

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.worlds = 2
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.backend = FakeBackend()
        environment._actuator_qpos_adrs = torch.tensor([0])
        environment._passive_qpos_adrs = torch.tensor([1])
        environment._ik_warmstart_backup = None
        environment._ctrl_low = torch.tensor([-1.0])
        environment._ctrl_high = torch.tensor([1.0])
        environment._ctrl_rate = torch.tensor([1.0])
        environment._passive_qpos_low = torch.tensor([-1.0])
        environment._passive_qpos_high = torch.tensor([1.0])
        environment.last_certified_ctrl = torch.tensor([[0.1], [0.2]])
        environment._last_ik_solve_feasible = torch.ones(
            2, dtype=torch.bool
        )
        environment.ik_solve_fallback_count = torch.zeros(
            2, dtype=torch.long
        )
        environment.ik_trust_region_projection_count = torch.zeros(
            2, dtype=torch.long
        )

        def failed_delta(self, targets):
            return (
                torch.full((2, 1), 0.25),
                torch.full((2, 1), 0.25),
                torch.zeros(2, dtype=torch.bool),
                torch.ones(2, dtype=torch.bool),
            )

        environment._constrained_ik_delta = MethodType(
            failed_delta, environment
        )
        environment._project_ik_cage_targets = MethodType(
            lambda self, targets, active_worlds: targets, environment
        )
        result = environment._ik_control(
            torch.zeros((2, 4, 3)),
            active_worlds=torch.tensor([False, True]),
        )

        self.assertEqual(
            environment._last_ik_solve_feasible.tolist(), [True, False]
        )
        self.assertEqual(
            environment.ik_solve_fallback_count.tolist(), [0, 1]
        )
        self.assertEqual(
            environment.ik_trust_region_projection_count.tolist(), [0, 3]
        )
        torch.testing.assert_close(result[0], environment.last_certified_ctrl[0])
        torch.testing.assert_close(qpos, torch.zeros_like(qpos))

    def test_ik_active_set_blocks_passive_joints_at_bounds(self) -> None:
        import torch

        qpos = torch.tensor([[0.5, 0.0, 1.0], [0.0, 1.0, 0.5]])
        low = torch.zeros(3)
        high = torch.ones(3)
        raw_delta = torch.tensor([[0.1, -0.1, 0.1], [-0.1, 0.1, -0.1]])

        free = _ik_free_mask_at_bounds(qpos, low, high, raw_delta)

        torch.testing.assert_close(
            free, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        )

    def test_ik_active_set_refines_direction_flip_per_world(self) -> None:
        import torch

        environment = object.__new__(MJWarpIKResidualEnv)
        seen_free = []

        def refined_solve(
            self, jacobian, constraint, error, pin_gap, free
        ):
            seen_free.append(free.clone())
            refined_feasible = torch.tensor([True, False, False])
            candidate = torch.tensor(
                [[7.0, -0.05], [8.0, 9.0], [10.0, 11.0]]
            )
            solved = torch.where(
                refined_feasible[:, None],
                candidate,
                torch.zeros_like(candidate),
            )
            return free * solved, refined_feasible

        environment._solve_kkt = MethodType(refined_solve, environment)
        joint_qpos = torch.tensor(
            [[1.0, 0.5], [1.0, 0.5], [1.0, 0.5]]
        )
        low = torch.zeros(2)
        high = torch.ones(2)
        raw_delta = torch.tensor(
            [[-0.25, 0.10], [-0.25, 0.10], [-0.25, -0.10]]
        )
        free = _ik_free_mask_at_bounds(
            joint_qpos, low, high, raw_delta
        )
        second_delta = torch.tensor(
            [[0.20, 0.30], [-0.20, 0.40], [0.10, -0.10]]
        )
        second_feasible = torch.ones(3, dtype=torch.bool)

        delta, feasible = environment._refine_kkt_active_set(
            torch.empty((3, 3, 2)),
            torch.empty((3, 3, 2)),
            torch.empty((3, 3)),
            torch.empty((3, 3)),
            joint_qpos,
            low,
            high,
            free,
            second_delta,
            second_feasible,
        )

        self.assertEqual(len(seen_free), 1)
        torch.testing.assert_close(
            seen_free[0],
            torch.tensor([[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]),
        )
        torch.testing.assert_close(
            delta,
            torch.tensor(
                [[0.0, -0.05], [-0.20, 0.40], [0.0, 0.0]]
            ),
        )
        self.assertEqual(feasible.tolist(), [True, True, False])

    def test_drop_quarantine_is_remote_and_clears_only_dropped_active_body(
        self,
    ) -> None:
        import torch

        qpos = torch.arange(28, dtype=torch.float32).reshape(2, 14)
        qvel = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        warmstart = qvel.clone().add_(100.0)

        class FakeBackend:
            views = {
                "qpos": qpos,
                "qvel": qvel,
                "qacc_warmstart": warmstart,
            }

            def tensor(self, name: str) -> torch.Tensor:
                return self.views[name]

            def has_field(self, name: str) -> bool:
                return name in self.views

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.torch = torch
        environment.backend = FakeBackend()
        environment.object_index = torch.tensor([0, 1])
        environment._object_qpos_adrs = torch.tensor([0, 7])
        environment._free_qpos_adr_values = (0, 7)
        environment._free_qvel_adr_values = (0, 6)
        environment._object_drop_quarantine_qpos = torch.tensor(
            [
                [[2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                 [2.0, 2.0, 1.0, 0.7, 0.1, 0.2, 0.3]],
                [[2.0, 2.0, 1.0, 0.6, 0.2, 0.3, 0.4],
                 [2.0, 2.0, 1.0, 0.5, 0.3, 0.4, 0.5]],
            ],
            dtype=torch.float32,
        )
        before_qpos = qpos.clone()
        before_qvel = qvel.clone()
        before_warmstart = warmstart.clone()

        environment._quarantine_dropped_active_object(
            torch.tensor([True, False])
        )

        torch.testing.assert_close(
            qpos[0, :7], environment._object_drop_quarantine_qpos[0, 0]
        )
        self.assertEqual(
            qpos[0, :3].tolist(),
            list(contract.OBJECT_DROP_QUARANTINE_POSITION_M),
        )
        self.assertTrue(torch.equal(qvel[0, :6], torch.zeros(6)))
        self.assertTrue(torch.equal(warmstart[0, :6], torch.zeros(6)))
        self.assertTrue(torch.equal(qpos[0, 7:], before_qpos[0, 7:]))
        self.assertTrue(torch.equal(qpos[1], before_qpos[1]))
        self.assertTrue(torch.equal(qvel[1], before_qvel[1]))
        self.assertTrue(torch.equal(warmstart[1], before_warmstart[1]))

    def test_drop_snapshot_pose_remains_first_failure_not_live_quarantine(
        self,
    ) -> None:
        import torch

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.object_dropped = torch.tensor([True, False])
        environment.object_drop_first_body_position = torch.tensor(
            [[0.1, 0.2, 0.24], [0.0, 0.0, 0.0]]
        )
        environment.object_drop_first_position = torch.tensor(
            [[0.1, 0.2, 0.27], [0.0, 0.0, 0.0]]
        )
        environment.object_drop_first_quaternion = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        )
        environment.object_drop_first_bottom_z_m = torch.tensor([0.249, 0.0])
        live = (
            torch.tensor([[2.0, 2.0, 1.0], [0.3, 0.4, 0.5]]),
            torch.tensor([[2.0, 2.0, 1.1], [0.3, 0.4, 0.6]]),
            torch.tensor(
                [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            ),
            torch.tensor([0.98, 0.48]),
        )
        environment._active_object_drop_measurement = MethodType(
            lambda self: live, environment
        )

        reported = environment._reported_object_drop_pose()

        torch.testing.assert_close(
            reported[0][0], environment.object_drop_first_body_position[0]
        )
        torch.testing.assert_close(
            reported[1][0], environment.object_drop_first_position[0]
        )
        torch.testing.assert_close(
            reported[2][0], environment.object_drop_first_quaternion[0]
        )
        self.assertAlmostEqual(float(reported[3][0]), 0.249)
        torch.testing.assert_close(reported[0][1], live[0][1])
        torch.testing.assert_close(reported[1][1], live[1][1])

    def test_early_done_classification_covers_horizon_drop_and_nonfinite(
        self,
    ) -> None:
        import torch

        transition = SimpleNamespace(
            done=torch.tensor([True, True, True, True]),
            info={
                "object_dropped": torch.tensor([False, False, True, True]),
                "time_out": torch.tensor([True, False, False, True]),
                "transition_finite": torch.tensor([True, True, True, False]),
            },
        )
        expected = torch.tensor([False, True, False, True])
        for classify in (
            _training_unexpected_early_done,
            _capacity_unexpected_early_done,
            _evaluation_unexpected_early_done,
        ):
            self.assertTrue(torch.equal(classify(transition, torch), expected))

    def test_contact_force_target_ramps_from_grasp_three_to_action_eight(
        self,
    ) -> None:
        import torch

        self.assertEqual(GRASP_CONTACT_FORCE_N, 3.0)
        self.assertEqual(ACTION_CONTACT_FORCE_N, 8.0)
        action_start = (
            DEFAULT_CONFIG.plan_ticks
            + DEFAULT_CONFIG.close_ticks
            + DEFAULT_CONFIG.settle_ticks
            + DEFAULT_CONFIG.lift_ticks
            + DEFAULT_CONFIG.hold_ticks
        )
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.config = DEFAULT_CONFIG
        environment.dtype = torch.float32
        environment.control_step = torch.tensor(
            [
                action_start - 1,
                action_start,
                action_start + ACTION_FORCE_RAMP_TICKS // 2 - 1,
                action_start + ACTION_FORCE_RAMP_TICKS - 1,
                action_start + ACTION_FORCE_RAMP_TICKS + 5,
            ],
            dtype=torch.long,
        )
        action = torch.zeros(
            (environment.control_step.numel(), contract.ACTION_DIM),
            dtype=torch.float32,
        )

        target = environment._target_contact_force(action)
        first_action_fraction = 1.0 / ACTION_FORCE_RAMP_TICKS
        first_action_ramp = (
            10.0 * first_action_fraction**3
            - 15.0 * first_action_fraction**4
            + 6.0 * first_action_fraction**5
        )
        first_action_target = GRASP_CONTACT_FORCE_N + first_action_ramp * (
            ACTION_CONTACT_FORCE_N - GRASP_CONTACT_FORCE_N
        )

        torch.testing.assert_close(
            target[:, 0],
            torch.tensor(
                [
                    GRASP_CONTACT_FORCE_N,
                    first_action_target,
                    0.5
                    * (GRASP_CONTACT_FORCE_N + ACTION_CONTACT_FORCE_N),
                    ACTION_CONTACT_FORCE_N,
                    ACTION_CONTACT_FORCE_N,
                ]
            ),
        )
        self.assertGreater(float(target[1, 0]), GRASP_CONTACT_FORCE_N)
        self.assertLess(float(target[1, 0]), float(target[2, 0]))
        self.assertTrue(torch.equal(target, target[:, :1].expand_as(target)))

    def test_all_fruit_pre_action_feedback_slews_during_lift(self) -> None:
        import torch

        object_indices = torch.tensor([0, 0, 1, 2, 3], dtype=torch.long)
        worlds = int(object_indices.numel())
        lift_start = (
            DEFAULT_CONFIG.plan_ticks
            + DEFAULT_CONFIG.close_ticks
            + DEFAULT_CONFIG.settle_ticks
        )
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.config = DEFAULT_CONFIG
        environment.dtype = torch.float32
        environment.worlds = worlds
        environment._rows = torch.arange(worlds, dtype=torch.long)
        environment.object_index = object_indices
        environment.task_index = torch.zeros(worlds, dtype=torch.long)
        environment._object_body_ids = torch.zeros(
            len(contract.OBJECTS), dtype=torch.long
        )
        environment.control_step = torch.full(
            (worlds,), lift_start, dtype=torch.long
        )
        environment.position_feedback_m = torch.zeros((worlds, 3))
        environment.position_feedback_slew_clip_count = torch.zeros(
            worlds, dtype=torch.long
        )
        environment.cage_orientation_error_rad = torch.zeros(worlds)
        desired_position = torch.tensor(
            [[0.020, -0.020, 0.020]] * worlds, dtype=torch.float32
        )

        def desired_pose(self: MJWarpIKResidualEnv):
            return (
                desired_position,
                torch.zeros((worlds, 4)),
                torch.ones(worlds),
                torch.zeros(worlds),
                torch.zeros(worlds, dtype=torch.bool),
            )

        class FakeBackend:
            views = {
                "xpos": torch.zeros((worlds, 1, 3)),
                "xipos": torch.zeros((worlds, 1, 3)),
                "xquat": torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0]]] * worlds
                ),
            }

            def tensor(self, name: str) -> torch.Tensor:
                return self.views[name]

        environment._desired_object_pose = MethodType(
            desired_pose, environment
        )
        environment.backend = FakeBackend()

        environment._update_position_feedback()

        torch.testing.assert_close(
            environment.position_feedback_m,
            torch.tensor(
                [
                    [
                        POSITION_FEEDBACK_SLEW_M_PER_TICK,
                        -POSITION_FEEDBACK_SLEW_M_PER_TICK,
                        POSITION_FEEDBACK_SLEW_M_PER_TICK,
                    ]
                ]
                * worlds
            ),
        )
        self.assertEqual(
            environment.position_feedback_slew_clip_count.tolist(),
            [1] * worlds,
        )

    def test_both_tasks_use_feedback_while_hold_freezes_it(
        self,
    ) -> None:
        import torch

        freeze_start, action_start = _hold_boundaries(DEFAULT_CONFIG)
        worlds = 3
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.config = DEFAULT_CONFIG
        environment.dtype = torch.float32
        environment.worlds = worlds
        environment._rows = torch.arange(worlds, dtype=torch.long)
        environment.object_index = torch.full(
            (worlds,), contract.OBJECTS.index("banana"), dtype=torch.long
        )
        environment.task_index = torch.tensor(
            [
                contract.TASKS.index(contract.TASK_ROTATE),
                contract.TASKS.index(contract.TASK_TRANSLATE),
                contract.TASKS.index(contract.TASK_TRANSLATE),
            ],
            dtype=torch.long,
        )
        environment._object_body_ids = torch.zeros(
            len(contract.OBJECTS), dtype=torch.long
        )
        environment.control_step = torch.tensor(
            [action_start, action_start, freeze_start], dtype=torch.long
        )
        initial = torch.tensor(
            [[0.001, -0.002, 0.003], [0.0, 0.0, 0.0], [0.004, 0.0, 0.0]],
            dtype=torch.float32,
        )
        environment.position_feedback_m = initial.clone()
        environment.position_feedback_slew_clip_count = torch.zeros(
            worlds, dtype=torch.long
        )
        environment.cage_orientation_error_rad = torch.zeros(worlds)
        desired_position = torch.full((worlds, 3), 0.020)

        def desired_pose(self: MJWarpIKResidualEnv):
            return (
                desired_position,
                torch.zeros((worlds, 4)),
                torch.ones(worlds),
                torch.zeros(worlds),
                torch.tensor([True, True, False]),
            )

        class FakeBackend:
            views = {
                "xpos": torch.zeros((worlds, 1, 3)),
                "xipos": torch.zeros((worlds, 1, 3)),
                "xquat": torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0]]] * worlds
                ),
            }

            def tensor(self, name: str) -> torch.Tensor:
                return self.views[name]

        environment._desired_object_pose = MethodType(
            desired_pose, environment
        )
        environment.backend = FakeBackend()
        environment._update_position_feedback()

        self.assertFalse(
            torch.equal(environment.position_feedback_m[0], initial[0])
        )
        self.assertLessEqual(
            float(
                (environment.position_feedback_m[0] - initial[0])
                .abs()
                .amax()
            ),
            POSITION_FEEDBACK_SLEW_M_PER_TICK + 1.0e-8,
        )
        torch.testing.assert_close(
            environment.position_feedback_m[1],
            torch.full((3,), POSITION_FEEDBACK_SLEW_M_PER_TICK),
        )
        torch.testing.assert_close(environment.position_feedback_m[2], initial[2])

    def test_single_task_budget_and_diffusion_profile_are_exact(self) -> None:
        config = DEFAULT_CONFIG
        config.validate()
        self.assertEqual(config.total_steps, 60_000_000)
        self.assertEqual(config.num_envs, contract.FORMAL_GLOBAL_WORLDS)
        self.assertEqual(config.control_ticks, 3_000)
        self.assertEqual(config.remainder_worlds, 0)
        self.assertEqual(
            config.remainder_object_world_counts, (0,) * len(contract.OBJECTS)
        )
        self.assertEqual(config.object_world_counts, (5_000,) * 4)
        self.assertEqual(
            (
                config.diffusion.obs_horizon,
                config.diffusion.action_horizon,
                config.diffusion.diffusion_steps,
                config.diffusion.inference_steps,
                config.diffusion.replan_interval,
            ),
            (2, 16, 100, 16, 4),
        )

        self.assertEqual(config.diffusion.batch_size, 4_096)
        self.assertEqual(config.diffusion.hidden_dim, 256)
        self.assertEqual(config.diffusion.depth, 6)
        self.assertEqual(config.diffusion.epochs, 60)
        self.assertEqual(config.diffusion.updates_per_epoch, 1_100)
        self.assertTrue(
            config.diffusion.stratified_object_task_sampling
        )
        for completed_demos in (1, 24, 10_000):
            self.assertEqual(
                config.diffusion_batches_per_epoch(completed_demos),
                1_100,
            )
        with self.assertRaisesRegex(ValueError, "at least one demo"):
            config.diffusion_batches_per_epoch(0)
        self.assertEqual(config.elite.population, 10)
        self.assertEqual(config.physics_substeps_per_control_tick, 40)
        self.assertEqual(config.episode_length, 3_000)
        with self.assertRaisesRegex(ValueError, "complete endpoint"):
            replace(config, num_envs=120_000).validate()

    def test_120k_worlds_is_capacity_only_not_formal_60m_training(self) -> None:
        config = _worker_config(
            num_envs=120_000,
            device="cuda:0",
            task=contract.TASK_TRANSLATE,
            diagnostic_only=True,
        )
        _validate_diagnostic_config(config)
        with self.assertRaisesRegex(
            ValueError, "official search"
        ):
            config.validate()
        self.assertEqual(config.control_ticks, 3_000)
        self.assertEqual(config.episode_length, 3_000)

    def test_diagnostic_worker_budget_is_exactly_one_complete_episode_per_run(
        self,
    ) -> None:
        for task in contract.TASKS:
            with self.subTest(task=task):
                worlds = (
                    200
                    if task == contract.TASK_TRANSLATE
                    else 1_440
                )
                config = _worker_config(
                    num_envs=worlds,
                    device="cuda:0",
                    task=task,
                    diagnostic_only=True,
                )
                episode_ticks = config.episode_length
                run_steps = worlds * episode_ticks
                self.assertEqual(config.total_steps, run_steps)
                self.assertEqual(config.control_ticks, episode_ticks)
                self.assertEqual(
                    _run_lifecycle_counts(config),
                    (episode_ticks, 1, 0),
                )
                lifecycle = _diagnostic_run_lifecycle(config)
                self.assertEqual(
                    lifecycle,
                    {
                        "total_environment_steps": run_steps,
                        "episode_control_ticks": episode_ticks,
                        "run": {
                            "environment_steps": run_steps,
                            "control_ticks": episode_ticks,
                            "complete_episode_count": 1,
                            "final_tail_ticks": 0,
                        },
                    },
                )
                metadata = _worker_lifecycle_metadata(
                    config, diagnostic_only=True
                )
                self.assertEqual(
                    metadata["episode_control_ticks"], episode_ticks
                )
                self.assertEqual(
                    metadata["diagnostic_run_lifecycle"], lifecycle
                )
                _validate_diagnostic_config(config)
                with self.assertRaisesRegex(ValueError, "official search"):
                    config.validate()
                with self.assertRaisesRegex(
                    ValueError, "exactly one complete episode"
                ):
                    _validate_diagnostic_config(
                        replace(
                            config,
                            total_steps=config.total_steps + config.num_envs,
                        )
                    )

        formal = _worker_config(
            num_envs=20_000,
            device="cuda:0",
            task=contract.TASK_TRANSLATE,
            diagnostic_only=False,
        )
        formal.validate()
        self.assertEqual(formal.total_steps, 60_000_000)
        self.assertEqual(formal.control_ticks, 3_000)
        self.assertEqual(
            _run_lifecycle_counts(formal), (3_000, 1, 0)
        )
        self.assertEqual(
            _worker_lifecycle_metadata(formal, diagnostic_only=False),
            {"episode_control_ticks": 3_000},
        )

    def test_capacity_env_official_validation_follows_diagnostic_mode(
        self,
    ) -> None:
        import MJWarp_RL.capacity_scan as capacity

        class StopAtMakeEnv(RuntimeError):
            pass

        validation_modes: list[bool] = []

        def stop_at_make_env(*args: object, **kwargs: object) -> object:
            validation_modes.append(
                bool(kwargs["validate_official_config"])
            )
            raise StopAtMakeEnv

        diagnostic = _worker_config(
            num_envs=200,
            device="cuda:0",
            task=contract.TASK_TRANSLATE,
            diagnostic_only=True,
        )
        _validate_diagnostic_config(diagnostic)
        formal = _worker_config(
            num_envs=20_000,
            device="cuda:0",
            task=contract.TASK_TRANSLATE,
            diagnostic_only=False,
        )
        formal.validate()

        with mock.patch.object(
            capacity, "make_env", side_effect=stop_at_make_env
        ):
            for config, diagnostic_only in (
                (diagnostic, True),
                (formal, False),
            ):
                with self.subTest(diagnostic_only=diagnostic_only):
                    with self.assertRaises(StopAtMakeEnv):
                        capacity._run_full_search_episode_peak(
                            torch=object(),
                            config=config,
                            platform_spec=object(),
                            device="cuda:0",
                            memory=object(),
                            teacher=object(),
                            phase=PHASE,
                            validate_official_config=not diagnostic_only,
                            diagnostic_only=diagnostic_only,
                        )

        self.assertEqual(validation_modes, [False, True])
        native_source = inspect.getsource(capacity._run_native_search_peak)
        self.assertIn(
            "validate_official_config = not diagnostic_only",
            native_source,
        )
        self.assertEqual(
            native_source.count(
                "validate_official_config=validate_official_config"
            ),
            2,
        )
        worker_source = inspect.getsource(capacity._worker)
        self.assertIn("diagnostic_only=diagnostic_only", worker_source)

    def test_failed_diagnostic_coverage_is_reported_without_relaxing_formal(
        self,
    ) -> None:
        import torch
        import MJWarp_RL.capacity_scan as capacity

        config = _worker_config(
            num_envs=40,
            device="cpu",
            task=contract.TASK_TRANSLATE,
            diagnostic_only=True,
        )
        population = config.elite.population
        goals = {
            "object_index": torch.arange(len(contract.OBJECTS)).repeat_interleave(
                population
            ),
            "template_index": torch.zeros(config.num_envs, dtype=torch.long),
            "task_index": torch.full(
                (config.num_envs,),
                contract.TASKS.index(contract.TASK_TRANSLATE),
                dtype=torch.long,
            ),
            "path_shape_index": torch.zeros(
                config.num_envs, dtype=torch.long
            ),
            "rotation_angle_deg": torch.zeros(
                config.num_envs, dtype=torch.long
            ),
        }
        valid_world = torch.zeros(config.num_envs, dtype=torch.bool)

        diagnostic = capacity._certified_candidate_coverage(
            torch,
            config=config,
            goals=goals,
            phase=PHASE,
            valid_world=valid_world,
            raise_on_failure=False,
        )
        self.assertFalse(diagnostic["passed"])
        self.assertEqual(diagnostic["valid_world_count"], 0)
        self.assertEqual(diagnostic["valid_group_count"], 0)
        with self.assertRaisesRegex(
            RuntimeError, "insufficient certified candidates"
        ):
            capacity._certified_candidate_coverage(
                torch,
                config=config,
                goals=goals,
                phase=PHASE,
                valid_world=valid_world,
                raise_on_failure=True,
            )

        full_source = inspect.getsource(
            capacity._run_full_search_episode_peak
        )
        self.assertIn(
            "raise_on_failure=not diagnostic_only", full_source
        )

    def test_planner_and_dense_rl_model_are_explicitly_separated(self) -> None:
        from xml.etree import ElementTree

        self.assertEqual(tuple(MODEL_XMLS_BY_TASK), contract.TASKS)
        self.assertEqual(tuple(TASK_GRIPPER_XMLS), contract.TASKS)
        self.assertEqual(tuple(TASK_ROOT_HEIGHT_M), contract.TASKS)
        self.assertTrue(PARENT_MODEL_XML.is_file())
        shard_count = 0
        for task in contract.TASKS:
            model_xmls = model_xmls_for_task(task)
            self.assertEqual(tuple(model_xmls), contract.OBJECTS)
            self.assertTrue(TASK_GRIPPER_XMLS[task].is_file())
            for object_name, model_xml in model_xmls.items():
                shard_count += 1
                with self.subTest(task=task, object_name=object_name):
                    self.assertTrue(model_xml.is_file())
                    self.assertNotEqual(model_xml, PARENT_MODEL_XML)
                    self.assertNotEqual(model_xml.parent, PACKAGE_DIR)
                    root = ElementTree.parse(model_xml).getroot()
                    option = root.find("option")
                    self.assertIsNotNone(option)
                    self.assertEqual(option.get("solver"), "Newton")
                    self.assertEqual(option.get("jacobian"), "dense")
                    self.assertEqual(float(option.get("tolerance")), 1.0e-10)
                    object_joints = {
                        joint.get("name")
                        for joint in root.findall(".//joint")
                        if str(joint.get("name", "")).startswith(
                            "joint:object_"
                        )
                    }
                    self.assertEqual(
                        object_joints, {f"joint:object_{object_name}"}
                    )
        self.assertEqual(shard_count, 8)
        parent_option = ElementTree.parse(PARENT_MODEL_XML).getroot().find(
            "option"
        )
        self.assertIsNotNone(parent_option)
        self.assertEqual(parent_option.get("solver"), "CG")
        self.assertEqual(parent_option.get("jacobian"), "sparse")
        self.assertIn(EXPECTED_ACTUATORS, (12, 16))
        fields = set(PlatformSpec.__dataclass_fields__)
        for field in (
            "fixed_object_index",
            "task",
            "task_index",
            "root_height_m",
            "model_xml",
            "model_xml_sha256",
            "qpos_templates",
            "ctrl_templates",
            "contact_targets_local",
            "pregrasp_targets_local",
            "planned_grasp_plane_fraction",
            "planned_grasp_plane_yaw_rad",
            "initial_object_pos",
            "initial_object_quat",
            "object_local_bottom_z_m",
            "object_local_top_z_m",
            "finger_actuator_indices",
            "finger_passive_indices",
            "planner_model_fingerprint_sha256",
            "rl_model_fingerprint_sha256",
        ):
            self.assertIn(field, fields)
        self.assertIn("shards", ShardedPlatformSpec.__dataclass_fields__)
        import MJWarp_RL.platform_adapter as adapter

        source = inspect.getsource(adapter)
        self.assertIn("RESET_TEMPLATES_PER_OBJECT", source)
        self.assertIn("demo_grasp_plane_z_m", source)
        self.assertIn("GRASP_PLANE_NOMINAL_FRACTION", source)
        self.assertIn("enforce_anti_support = False", source)
        self.assertIn("mjJAC_SPARSE", source)
        self.assertIn("mjSOL_NEWTON", source)
        self.assertIn("mjJAC_DENSE", source)
        self.assertIn("_project_qpos", source)
        self.assertTrue(
            "_fail_closed_bank" in source or "_single_slot_bank" in source
        )
        self.assertIn("np.nan", source)
        self.assertNotIn("plate", contract.OBJECTS)
        self.assertFalse(any(PACKAGE_DIR.glob("*.xml")))

    def test_grasp_object_contact_profile_matches_stable_screw_contract(self) -> None:
        from xml.etree import ElementTree

        model_xml = model_xmls_for_task(contract.TASK_TRANSLATE)[
            contract.OBJECTS[0]
        ]
        root = ElementTree.parse(model_xml).getroot()
        share_name = next(
            include.get("file")
            for include in root.findall("include")
            if "share" in str(include.get("file", "")).lower()
        )
        share_root = ElementTree.parse(model_xml.parent / share_name).getroot()
        geom = share_root.find(".//default[@class='grasp_object']/geom")
        self.assertIsNotNone(geom)
        self.assertEqual(geom.get("condim"), "4")
        self.assertEqual(geom.get("solref"), "0.004 1")

    def test_checkpoint_binds_complete_parent_model_and_controller(self) -> None:
        import MJWarp_RL.evaluate as evaluation
        import MJWarp_RL.train as training

        for source in (
            inspect.getsource(training),
            inspect.getsource(evaluation),
        ):
            self.assertIn("parent_model_fingerprint_sha256", source)
            self.assertIn("parent_controller_fingerprint_sha256", source)

    def test_grasp_plane_projection_hard_clamps_every_tip(self) -> None:
        import torch

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.object_index = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        environment._rows = torch.arange(4, dtype=torch.long)
        environment._object_body_ids = torch.zeros(4, dtype=torch.long)
        environment._object_local_bottom_z_m = torch.zeros(4)
        environment._object_local_top_z_m = torch.ones(4)
        _install_grasp_plane_contract_tensors(environment, torch)

        class FakeBackend:
            views = {
                "xpos": torch.zeros((4, 1, 3)),
                "xquat": torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0]]] * 4
                ),
            }

            def tensor(self, name: str) -> torch.Tensor:
                return self.views[name]

        environment.backend = FakeBackend()
        one_row = torch.tensor(
            [[0.0, 0.0, 0.10], [0.0, 0.0, 0.15],
             [0.0, 0.0, 0.20], [0.0, 0.0, 0.25]]
        )
        below = one_row[None, :, :].expand(4, -1, -1).clone()
        projected = environment._project_grasp_plane(below)
        fractions, planes, compliant = environment._grasp_plane_metrics(
            projected
        )
        torch.testing.assert_close(
            fractions,
            torch.full((4, 4), 0.4),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            planes,
            torch.full((4,), 0.4),
            atol=1.0e-6,
            rtol=0.0,
        )
        self.assertTrue(bool(compliant.all()))
        self.assertEqual(fractions.shape, (4, 4))

    def test_all_fruit_physical_planes_use_the_learned_band(self) -> None:
        import torch

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.object_index = torch.arange(4, dtype=torch.long)
        environment._rows = torch.arange(4, dtype=torch.long)
        environment._object_body_ids = torch.zeros(4, dtype=torch.long)
        environment._object_local_bottom_z_m = torch.zeros(4)
        environment._object_local_top_z_m = torch.tensor(
            [0.12, 0.04, 0.0655, 0.05]
        )
        _install_grasp_plane_contract_tensors(environment, torch)

        class FakeBackend:
            views = {
                "xpos": torch.zeros((4, 1, 3)),
                "xquat": torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0]]] * 4
                ),
            }

            def tensor(self, name: str) -> torch.Tensor:
                return self.views[name]

        environment.backend = FakeBackend()
        fractions = torch.tensor(
            [0.50, 0.40 - 1.0e-5, 0.60, 0.60 + 1.0e-5]
        )
        targets = torch.zeros((4, 4, 3))
        targets[:, :, 2] = (
            fractions * environment._object_local_top_z_m
        )[:, None]
        _, measured, commanded = environment._grasp_plane_metrics(targets)
        _, _, physical = environment._grasp_plane_metrics(
            targets, physical=True
        )
        torch.testing.assert_close(measured, fractions)
        self.assertEqual(commanded.tolist(), [True, False, True, False])
        self.assertEqual(physical.tolist(), commanded.tolist())

    def test_actual_grasp_plane_is_physical_finite_and_endpoint_gated(self) -> None:
        import torch

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.object_index = torch.zeros(4, dtype=torch.long)
        environment._rows = torch.arange(4, dtype=torch.long)
        environment._object_body_ids = torch.tensor([0], dtype=torch.long)
        environment._tip_body_ids = torch.tensor(
            [1, 2, 3, 4], dtype=torch.long
        )
        environment._object_local_bottom_z_m = torch.tensor([0.0])
        environment._object_local_top_z_m = torch.tensor([1.0])
        _install_grasp_plane_contract_tensors(environment, torch)
        environment.control_step = torch.tensor([0, 30, 30, 0])
        environment.object_dropped = torch.tensor(
            [False, False, True, False]
        )
        environment.config = SimpleNamespace(plan_ticks=10, close_ticks=20)

        positions = torch.zeros((4, 5, 3))
        positions[:2, 1:, 2] = 0.70
        positions[2, 1:, 2] = 10.0
        positions[3, 1:, 2] = 0.50
        positions[3, 1, 2] = float("nan")

        class FakeBackend:
            views = {
                "xpos": positions,
                "xquat": torch.tensor(
                    [
                        [[1.0, 0.0, 0.0, 0.0]],
                        [[1.0, 0.0, 0.0, 0.0]],
                        [[1.0, 0.0, 0.0, 0.0]],
                        [[1.0, 0.0, 0.0, 0.0]],
                    ]
                ),
            }

            def tensor(self, name: str) -> torch.Tensor:
                return self.views[name]

        environment.backend = FakeBackend()
        (
            tip_fraction,
            plane_fraction,
            compliant,
            relevant,
            in_band,
            finite,
        ) = environment._actual_grasp_plane_metrics(torch.zeros((4, 4)))
        self.assertEqual(tip_fraction.shape, (4, 4))
        torch.testing.assert_close(plane_fraction[:2], torch.tensor([0.7, 0.7]))
        self.assertEqual(relevant.tolist(), [False, True, False, False])
        self.assertEqual(in_band.tolist(), [False, False, False, False])
        self.assertEqual(finite.tolist(), [True, True, True, False])
        self.assertEqual(compliant.tolist(), [True, False, True, False])
        self.assertIn(
            "& actual_plane_compliant",
            inspect.getsource(MJWarpIKResidualEnv.step),
        )

    def test_paired_evaluation_uses_each_template_once(self) -> None:
        from MJWarp_RL.evaluate import _cases

        cases = _cases(
            PHASE,
            episodes_per_case=contract.RESET_TEMPLATES_PER_OBJECT,
            seed=123,
        )
        for object_index in range(len(contract.OBJECTS)):
            for task_index in range(len(contract.TASKS)):
                group = [
                    case
                    for case in cases
                    if case["object_index"] == object_index
                    and case["task_index"] == task_index
                ]
                self.assertEqual(
                    [case["template_index"] for case in group],
                    list(range(contract.RESET_TEMPLATES_PER_OBJECT)),
                )
                self.assertEqual(
                    [case["template_seed"] for case in group],
                    [
                        contract.reset_template_seed(
                            object_index, template_index
                        )
                        for template_index in range(
                            contract.RESET_TEMPLATES_PER_OBJECT
                        )
                    ],
                )
                if contract.TASKS[task_index] == contract.TASK_ROTATE:
                    angles = [case["rotation_angle_deg"] for case in group]
                    self.assertEqual(
                        min(angles), contract.ROTATION_ANGLE_MIN_DEG
                    )
                    self.assertEqual(
                        max(angles),
                        contract.ROTATION_ANGLE_MIN_DEG
                        + contract.RESET_TEMPLATES_PER_OBJECT
                        - 1,
                    )
                    self.assertEqual(
                        len(set(angles)),
                        contract.RESET_TEMPLATES_PER_OBJECT,
                    )
                    self.assertLessEqual(
                        max(angles), contract.ROTATION_ANGLE_MAX_DEG
                    )
                    self.assertEqual(
                        {case["source_region"] for case in group}, {0}
                    )
                else:
                    self.assertEqual(
                        {case["rotation_angle_deg"] for case in group},
                        {0},
                    )
        with self.assertRaises(ValueError):
            _cases(
                PHASE,
                episodes_per_case=(
                    contract.RESET_TEMPLATES_PER_OBJECT + 1
                ),
                seed=123,
            )
        with self.assertRaisesRegex(ValueError, "unsupported curriculum phase"):
            _cases(
                "phase2",
                episodes_per_case=contract.RESET_TEMPLATES_PER_OBJECT,
                seed=123,
            )

    def test_paired_result_records_exact_action_start_position_error(
        self,
    ) -> None:
        from MJWarp_RL.evaluate import _result_from_trace

        ticks = DEFAULT_CONFIG.episode_length
        precondition_index = (
            DEFAULT_CONFIG.plan_ticks
            + DEFAULT_CONFIG.close_ticks
            + DEFAULT_CONFIG.settle_ticks
            + DEFAULT_CONFIG.lift_ticks
            + DEFAULT_CONFIG.hold_ticks
            - 1
        )
        trace = {
            "position": np.zeros((ticks, 1, 3)),
            "goal_position": np.zeros((ticks, 1, 3)),
            "position_error": np.zeros((ticks, 1)),
            "position_error_xyz": np.zeros((ticks, 1, 3)),
            "euler_error": np.zeros((ticks, 1)),
            "euler_error_xyz": np.zeros((ticks, 1, 3)),
            "rotation_from_initial_xyz": np.zeros((ticks, 1, 3)),
            "rotation_from_action_reference_xyz": np.zeros(
                (ticks, 1, 3)
            ),
            "contact_forces": np.zeros((ticks, 1, 4)),
            "action": np.zeros((ticks, 1, contract.ACTION_DIM)),
            "tip_height_m": np.zeros((ticks, 1, 4)),
            "success": np.zeros((ticks, 1), dtype=bool),
            "velocity": np.zeros((ticks, 1, 6)),
        }
        trace["position_error"][precondition_index, 0] = 0.0123
        trace["position_error_xyz"][precondition_index, 0] = (
            0.003,
            -0.004,
            0.011,
        )
        result = _result_from_trace(
            trace,
            case_index=0,
            object_kind="orange",
            task=contract.TASK_TRANSLATE,
            config=DEFAULT_CONFIG,
            trajectory_stride=ticks,
        )
        self.assertAlmostEqual(
            result["precondition"]["position_error_mm"], 12.3
        )
        self.assertEqual(
            result["precondition"]["position_error_xyz_mm"],
            [3.0, -4.0, 11.0],
        )

    def test_paired_result_does_not_fallback_to_drop_quarantine_plane(
        self,
    ) -> None:
        from MJWarp_RL.evaluate import _result_from_trace

        ticks = DEFAULT_CONFIG.episode_length
        trace = {
            "position": np.zeros((ticks, 1, 3)),
            "goal_position": np.zeros((ticks, 1, 3)),
            "position_error": np.zeros((ticks, 1)),
            "position_error_xyz": np.zeros((ticks, 1, 3)),
            "euler_error": np.zeros((ticks, 1)),
            "euler_error_xyz": np.zeros((ticks, 1, 3)),
            "rotation_from_initial_xyz": np.zeros((ticks, 1, 3)),
            "rotation_from_action_reference_xyz": np.zeros(
                (ticks, 1, 3)
            ),
            "contact_forces": np.zeros((ticks, 1, 4)),
            "action": np.zeros((ticks, 1, contract.ACTION_DIM)),
            "tip_height_m": np.zeros((ticks, 1, 4)),
            "success": np.zeros((ticks, 1), dtype=bool),
            "velocity": np.zeros((ticks, 1, 6)),
            "object_dropped": np.ones((ticks, 1), dtype=bool),
            "actual_grasp_plane_relevant": np.zeros(
                (ticks, 1), dtype=bool
            ),
            # Deliberately impossible quarantine-derived values: no fallback
            # may admit them into either physical or commanded statistics.
            "actual_grasp_plane_fraction": np.full((ticks, 1), 99.0),
            "actual_grasp_plane_compliant": np.zeros(
                (ticks, 1), dtype=bool
            ),
            "grasp_plane_fraction": np.full((ticks, 1), -99.0),
        }
        result = _result_from_trace(
            trace,
            case_index=0,
            object_kind="orange",
            task=contract.TASK_TRANSLATE,
            config=DEFAULT_CONFIG,
            trajectory_stride=ticks,
        )
        plane = result["grasp_plane"]
        self.assertEqual(plane["relevant_sample_count"], 0)
        for key in (
            "minimum_fraction",
            "maximum_fraction",
            "mean_fraction",
            "compliance_rate",
            "all_relevant_control_ticks_compliant",
        ):
            self.assertIsNone(plane[key])
        self.assertEqual(plane["commanded"]["sample_count"], 0)
        self.assertIsNone(plane["commanded"]["minimum_fraction"])
        self.assertIsNone(result["endpoint"]["grasp_plane_fraction"])

    def test_partial_reset_selects_bank_row_and_preserves_other_worlds(
        self,
    ) -> None:
        import torch

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.torch = torch
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = 3
        environment.curriculum_phase = PHASE
        environment.platform = SimpleNamespace(
            host_model=SimpleNamespace(nu=1)
        )
        environment._allocate_state()
        environment._object_body_ids = torch.zeros(6, dtype=torch.long)
        row_count = (
            len(contract.OBJECTS)
            * contract.RESET_TEMPLATES_PER_OBJECT
        )
        environment._qpos_templates = torch.arange(
            row_count * 2, dtype=torch.float32
        ).reshape(row_count, 2)
        environment._ctrl_templates = torch.arange(
            row_count, dtype=torch.float32
        ).reshape(row_count, 1)
        template_values = torch.arange(
            row_count, dtype=torch.float32
        ).reshape(row_count, 1)

        class FakeBackend:
            def __init__(self) -> None:
                self.views = {
                    "xpos": torch.zeros((3, 1, 3)),
                    "xipos": torch.zeros((3, 1, 3)),
                    "xquat": torch.tensor(
                        [[[1.0, 0.0, 0.0, 0.0]]] * 3
                    ),
                }
                self.reset_qpos = None
                self.reset_ctrl = None

            def reset_worlds(
                self, mask: torch.Tensor, *, qpos: torch.Tensor, ctrl: torch.Tensor
            ) -> None:
                self.reset_qpos = qpos.clone()
                self.reset_ctrl = ctrl.clone()

            def tensor(self, name: str) -> torch.Tensor:
                return self.views[name]

        environment.backend = FakeBackend()
        environment._sample_goal_rows = MethodType(
            lambda self, ids, goals: (
                torch.tensor([2]),
                torch.tensor([7]),
                torch.tensor([0]),
                torch.tensor([5]),
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([0]),
            ),
            environment,
        )

        def fake_targets(self: MJWarpIKResidualEnv, action: torch.Tensor) -> torch.Tensor:
            values = self._template_rows(template_values)
            return values[:, None, :].expand(-1, 4, 3)

        def fake_observation(
            self: MJWarpIKResidualEnv,
            targets: torch.Tensor,
            action: torch.Tensor,
            force: torch.Tensor,
        ) -> torch.Tensor:
            result = torch.zeros((3, OBSERVATION_DIM))
            result[:, 0] = targets[:, 0, 0]
            return result

        environment._tip_targets = MethodType(fake_targets, environment)
        environment._observation = MethodType(
            fake_observation, environment
        )
        environment._observation_tensor.fill_(11.0)
        environment._last_targets.fill_(9.0)
        environment.previous_action.fill_(0.75)
        selected = torch.tensor([False, True, False])
        observation = environment.reset(selected)
        flat_index = 2 * contract.RESET_TEMPLATES_PER_OBJECT + 7
        self.assertTrue(
            torch.equal(
                environment.backend.reset_qpos,
                environment._qpos_templates[flat_index : flat_index + 1],
            )
        )
        self.assertTrue(
            torch.equal(
                environment.backend.reset_ctrl,
                environment._ctrl_templates[flat_index : flat_index + 1],
            )
        )
        self.assertEqual(environment.template_index.tolist(), [0, 7, 0])
        self.assertTrue(torch.all(observation[0] == 11.0))
        self.assertTrue(torch.all(observation[2] == 11.0))
        self.assertEqual(float(observation[1, 0]), float(flat_index))
        self.assertTrue(torch.all(environment._last_targets[0] == 9.0))
        self.assertTrue(torch.all(environment._last_targets[2] == 9.0))
        self.assertTrue(torch.all(environment.previous_action[0] == 0.75))
        self.assertTrue(torch.all(environment.previous_action[2] == 0.75))
        self.assertTrue(torch.all(environment.previous_action[1] == 0.0))

    def test_zero_action_is_neutral_ik_and_tip_channels_are_independent(self) -> None:
        zero = contract.decode_action(
            [0.0] * contract.ACTION_DIM, "banana"
        )
        self.assertEqual(zero.tip_height_fraction, (0.5,) * 4)
        self.assertEqual(zero.force_scale, (1.0,) * 4)
        action = [0.0] * contract.ACTION_DIM
        action[:4] = [1.0, -1.0, 0.5, -0.5]
        action[21:25] = [-1.0, -0.5, 0.5, 1.0]
        decoded = contract.decode_action(action, "pear")
        for actual, expected in zip(
            decoded.tip_height_fraction, (0.60, 0.40, 0.55, 0.45)
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            decoded.force_scale,
            (0.5, 0.75, 1.25, 1.5),
        )
        orange = contract.decode_action(action, "orange")
        self.assertEqual(
            orange.tip_height_fraction, decoded.tip_height_fraction
        )
        for infeasible in contract.INFEASIBLE_OBJECTS:
            with self.assertRaises(ValueError):
                contract.decode_action(action, infeasible)
        self.assertEqual(orange.force_scale, decoded.force_scale)

    def test_policy_mask_enables_episode_channels_for_every_fruit(self) -> None:
        import torch

        observation = torch.zeros((2, OBSERVATION_DIM))
        object_slice = OBSERVATION_SLICES["object_one_hot"]
        phase_slice = OBSERVATION_SLICES["phase_one_hot"]
        observation[0, object_slice.start] = 1.0  # orange
        observation[1, object_slice.start + 1] = 1.0  # banana
        observation[:, phase_slice.start] = 1.0  # plan phase
        mask = action_mask_from_observation(observation)
        self.assertTrue(mask[:, :9].all())
        self.assertTrue(mask[:, 21:25].all())
        self.assertEqual(mask.sum(dim=1).tolist(), [25.0, 25.0])
        observation[:, phase_slice.start] = 0.0
        observation[:, phase_slice.stop - 1] = 1.0  # action phase
        mask = action_mask_from_observation(observation)
        self.assertFalse(mask[:, :9].any())
        self.assertFalse(mask[:, 21:25].any())
        self.assertTrue(mask[:, 9:21].all())
        self.assertEqual(mask.sum(dim=1).tolist(), [12.0, 12.0])

    def test_environment_filter_preserves_all_fruit_episode_channels(self) -> None:
        import torch

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.torch = torch
        environment.dtype = torch.float32
        environment.worlds = 3
        environment.object_index = torch.tensor([0, 1, 3])
        environment.residual_enabled = True
        environment.filtered_action = torch.zeros((3, contract.ACTION_DIM))
        environment.frozen_action = torch.zeros_like(
            environment.filtered_action
        )
        environment.control_step = torch.zeros(3, dtype=torch.long)
        environment.config = SimpleNamespace(plan_ticks=8)
        _install_grasp_plane_contract_tensors(environment, torch)

        filtered, _, used = environment._filter_action(
            torch.ones((3, contract.ACTION_DIM))
        )
        for rows in (filtered, used):
            self.assertTrue(bool((rows[:, :5] > 0.0).all()))
            self.assertTrue(bool((rows[:, 5:9] > 0.0).all()))
            self.assertTrue(bool((rows[:, 9:21] > 0.0).all()))
            self.assertTrue(bool((rows[:, 21:25] > 0.0).all()))

    def test_post_lift_translation_roundtrips_commanded_cage(self) -> None:
        import torch

        dtype = torch.float64
        com = torch.tensor([[0.20, -0.10, 0.35]], dtype=dtype)
        half_yaw = np.deg2rad(17.0)
        quaternion = torch.tensor(
            [[np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]],
            dtype=dtype,
        )
        cage_local = torch.tensor(
            [
                [
                    [0.00, 0.04, 0.01],
                    [0.00, -0.04, -0.01],
                    [-0.05, 0.00, 0.02],
                    [0.05, 0.00, -0.02],
                ]
            ],
            dtype=dtype,
        )
        commanded = com[:, None, :] + _quat_rotate(
            quaternion, cage_local
        )
        translation = torch.tensor([[0.025, 0.0, 0.0]], dtype=dtype)
        captured, goal_position, goal_quaternion = _post_lift_reference(
            com,
            quaternion,
            commanded,
            translation,
            torch.tensor([0]),
            torch.tensor([0]),
        )
        reconstructed = com[:, None, :] + _quat_rotate(
            quaternion, captured
        )
        self.assertTrue(
            torch.allclose(reconstructed, commanded, atol=1.0e-12, rtol=0.0)
        )
        self.assertTrue(
            torch.allclose(
                goal_position, com + translation, atol=1.0e-12, rtol=0.0
            )
        )
        self.assertTrue(
            torch.allclose(
                goal_quaternion, quaternion, atol=1.0e-12, rtol=0.0
            )
        )

    def test_post_lift_rotation_uses_each_worlds_local_z_angle(self) -> None:
        import torch

        dtype = torch.float64
        half_roll = np.deg2rad(45.0)
        quaternion = torch.tensor(
            [[np.cos(half_roll), np.sin(half_roll), 0.0, 0.0]],
            dtype=dtype,
        )
        com = torch.tensor([[0.10, 0.20, 0.30]], dtype=dtype)
        cage_local = torch.tensor(
            [
                [
                    [0.00, 0.04, 0.00],
                    [0.00, -0.04, 0.00],
                    [-0.05, 0.00, 0.00],
                    [0.05, 0.00, 0.00],
                ]
            ],
            dtype=dtype,
        )
        commanded = com[:, None, :] + _quat_rotate(
            quaternion, cage_local
        )
        _, goal_position, goal_quaternion = _post_lift_reference(
            com,
            quaternion,
            commanded,
            torch.tensor([[0.09, -0.04, 0.03]], dtype=dtype),
            torch.tensor([1]),
            torch.tensor([30]),
        )
        local_euler = _relative_euler_xyz_deg(
            quaternion, goal_quaternion
        )
        self.assertTrue(
            torch.allclose(goal_position, com, atol=1.0e-12, rtol=0.0)
        )
        self.assertTrue(
            torch.allclose(
                local_euler,
                torch.tensor([[0.0, 0.0, 30.0]], dtype=dtype),
                atol=1.0e-10,
                rtol=0.0,
            )
        )

    def test_all_active_objects_use_minimum_jerk_rotation_geometry(self) -> None:
        import torch

        ticks = DEFAULT_CONFIG.action_ticks
        linear = torch.arange(1, ticks + 1, dtype=torch.float64) / ticks
        smooth = linear**3 * (10.0 - 15.0 * linear + 6.0 * linear**2)
        rotate = torch.full(
            (ticks,), contract.TASKS.index(contract.TASK_ROTATE), dtype=torch.long
        )
        for object_kind in contract.OBJECTS:
            object_index = torch.full(
                (ticks,),
                contract.OBJECTS.index(object_kind),
                dtype=torch.long,
            )
            with self.subTest(object_kind=object_kind):
                torch.testing.assert_close(
                    _cage_rotation_fraction(
                        linear, smooth, object_index, rotate
                    ),
                    smooth,
                )
                torch.testing.assert_close(
                    _reported_action_fraction(
                        linear, smooth, object_index, rotate
                    ),
                    smooth,
                )

        banana = torch.full(
            (ticks,), contract.OBJECTS.index("banana"), dtype=torch.long
        )
        translating = torch.full_like(
            rotate, contract.TASKS.index(contract.TASK_TRANSLATE)
        )
        torch.testing.assert_close(
            _cage_rotation_fraction(linear, smooth, banana, translating),
            smooth,
        )
        torch.testing.assert_close(
            _reported_action_fraction(
                linear, smooth, banana, translating
            ),
            smooth,
        )

    def test_motion_hold_freezes_actual_cage_and_ctrl_after_20_ticks(
        self,
    ) -> None:
        import torch

        freeze_start, action_start = _hold_boundaries(DEFAULT_CONFIG)
        self.assertEqual((freeze_start, action_start), (248, 298))
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.config = DEFAULT_CONFIG
        environment.control_step = torch.tensor(
            [
                freeze_start - 1,
                freeze_start,
                action_start - 1,
                action_start,
            ],
            dtype=torch.long,
        )
        environment.hold_reference_valid = torch.ones(4, dtype=torch.bool)
        environment.hold_frozen_cage_world = torch.full((4, 4, 3), 7.0)
        environment.hold_frozen_ctrl = torch.full((4, 3), 9.0)
        requested = torch.arange(48, dtype=torch.float32).reshape(4, 4, 3)
        proposed_ctrl = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        commanded, frozen = environment._hold_command_targets(requested)
        selected_ctrl = environment._hold_control(proposed_ctrl, frozen)
        self.assertEqual(frozen.tolist(), [False, True, True, False])
        self.assertTrue(torch.equal(commanded[0], requested[0]))
        self.assertTrue(
            torch.equal(commanded[1], environment.hold_frozen_cage_world[1])
        )
        self.assertTrue(
            torch.equal(commanded[2], environment.hold_frozen_cage_world[2])
        )
        self.assertTrue(torch.equal(commanded[3], requested[3]))
        self.assertTrue(torch.equal(selected_ctrl[0], proposed_ctrl[0]))
        self.assertTrue(
            torch.equal(selected_ctrl[1], environment.hold_frozen_ctrl[1])
        )
        self.assertTrue(
            torch.equal(selected_ctrl[2], environment.hold_frozen_ctrl[2])
        )
        self.assertTrue(torch.equal(selected_ctrl[3], proposed_ctrl[3]))

        valid = torch.tensor([False, False, True], dtype=torch.bool)
        capture = _hold_reference_capture(
            torch.tensor(
                [freeze_start - 1, freeze_start, freeze_start],
                dtype=torch.long,
            ),
            valid,
            DEFAULT_CONFIG,
        )
        self.assertEqual(capture.tolist(), [False, True, False])

        class FakeBackend:
            def __init__(self, ctrl: torch.Tensor) -> None:
                self.ctrl = ctrl

            def tensor(self, name: str) -> torch.Tensor:
                if name != "ctrl":
                    raise AssertionError(name)
                return self.ctrl

        captured_environment = object.__new__(MJWarpIKResidualEnv)
        actual_ctrl = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=torch.float32,
        )
        commanded_targets = torch.arange(
            36, dtype=torch.float32
        ).reshape(3, 4, 3)
        captured_environment.backend = FakeBackend(actual_ctrl)
        captured_environment.hold_reference_valid = torch.tensor(
            [False, False, True], dtype=torch.bool
        )
        captured_environment.hold_frozen_ctrl = torch.full((3, 2), -1.0)
        captured_environment.hold_frozen_cage_world = torch.full(
            (3, 4, 3), -2.0
        )
        ctrl_storage = captured_environment.hold_frozen_ctrl
        cage_storage = captured_environment.hold_frozen_cage_world
        valid_storage = captured_environment.hold_reference_valid
        captured_environment._capture_hold_reference(
            torch.tensor([False, True, False], dtype=torch.bool),
            commanded_targets,
        )
        self.assertIs(captured_environment.hold_frozen_ctrl, ctrl_storage)
        self.assertIs(
            captured_environment.hold_frozen_cage_world, cage_storage
        )
        self.assertIs(
            captured_environment.hold_reference_valid, valid_storage
        )
        self.assertTrue(
            torch.equal(captured_environment.hold_frozen_ctrl[1], actual_ctrl[1])
        )
        self.assertTrue(
            torch.equal(
                captured_environment.hold_frozen_cage_world[1],
                commanded_targets[1],
            )
        )
        self.assertEqual(
            captured_environment.hold_reference_valid.tolist(),
            [False, True, True],
        )

        reset_ids = torch.tensor([0, 2], dtype=torch.long)
        captured_environment._reset_hold_reference(reset_ids)
        self.assertIs(captured_environment.hold_frozen_ctrl, ctrl_storage)
        self.assertIs(
            captured_environment.hold_frozen_cage_world, cage_storage
        )
        self.assertIs(
            captured_environment.hold_reference_valid, valid_storage
        )
        self.assertTrue(
            torch.equal(
                captured_environment.hold_frozen_ctrl,
                torch.tensor(
                    [[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertEqual(
            captured_environment.hold_reference_valid.tolist(),
            [False, True, False],
        )
        for name in (
            '"hold_reference_valid"',
            '"hold_frozen_ctrl"',
            '"hold_frozen_cage_world"',
        ):
            self.assertIn(name, inspect.getsource(MJWarpIKResidualEnv.snapshot))

    def test_force_trim_contact_gates_before_and_during_action(self) -> None:
        import torch

        freeze_start, action_start = _hold_boundaries(DEFAULT_CONFIG)
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.config = DEFAULT_CONFIG
        environment.dtype = torch.float32
        environment.control_step = torch.tensor(
            [
                freeze_start - 1,
                freeze_start,
                action_start,
                action_start,
            ],
            dtype=torch.long,
        )
        environment.preload_trim_m = torch.zeros((4, 4), dtype=torch.float32)
        environment.action_start_preload_trim_m = torch.zeros_like(
            environment.preload_trim_m
        )
        environment.action_reference_valid = torch.tensor(
            [False, False, True, True], dtype=torch.bool
        )
        environment.preload_trim_slew_clip_count = torch.zeros(
            4, dtype=torch.long
        )
        action = torch.zeros((4, contract.ACTION_DIM), dtype=torch.float32)
        all_contact = CONTACT_FORCE_THRESHOLD_N + 0.1
        forces = torch.tensor(
            [
                [all_contact, 0.0, 0.0, 0.0],
                [all_contact, all_contact, all_contact, all_contact],
                [all_contact, 0.0, all_contact, all_contact],
                [all_contact, all_contact, all_contact, all_contact],
            ],
            dtype=torch.float32,
        )
        updated = environment._updated_preload_trim(action, forces)
        self.assertGreater(float(updated[0].abs().sum()), 0.0)
        self.assertTrue(torch.equal(updated[1], environment.preload_trim_m[1]))
        self.assertTrue(torch.equal(updated[2], environment.preload_trim_m[2]))
        self.assertGreater(float(updated[3].abs().sum()), 0.0)

    def test_zero_contact_leaves_preload_trim_unchanged(self) -> None:
        import torch

        feedback_start = (
            DEFAULT_CONFIG.plan_ticks
            + (4 * DEFAULT_CONFIG.close_ticks) // 5
        )
        settle_start = DEFAULT_CONFIG.plan_ticks + DEFAULT_CONFIG.close_ticks
        lift_start = settle_start + DEFAULT_CONFIG.settle_ticks
        _, action_start = _hold_boundaries(DEFAULT_CONFIG)
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.config = DEFAULT_CONFIG
        environment.dtype = torch.float32
        environment.control_step = torch.tensor(
            [
                feedback_start - 1,
                feedback_start,
                settle_start,
                lift_start,
                feedback_start,
                action_start,
            ],
            dtype=torch.long,
        )
        environment.preload_trim_m = torch.zeros(
            (6, 4), dtype=torch.float32
        )
        environment.preload_trim_m[4].fill_(FORCE_FEEDBACK_MAX_M)
        environment.action_start_preload_trim_m = torch.zeros_like(
            environment.preload_trim_m
        )
        environment.action_reference_valid = torch.zeros(
            6, dtype=torch.bool
        )
        environment.preload_trim_slew_clip_count = torch.zeros(
            6, dtype=torch.long
        )
        action = torch.zeros((6, contract.ACTION_DIM), dtype=torch.float32)
        zero_force = torch.zeros((6, 4), dtype=torch.float32)

        updated = environment._updated_preload_trim(action, zero_force)

        self.assertTrue(torch.equal(updated, environment.preload_trim_m))

    def test_preload_trim_rejects_nonfinite_force_and_limits_slew(self) -> None:
        import torch

        _, action_start = _hold_boundaries(DEFAULT_CONFIG)
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.config = DEFAULT_CONFIG
        environment.dtype = torch.float32
        environment.control_step = torch.tensor(
            [action_start, action_start], dtype=torch.long
        )
        environment.preload_trim_m = torch.zeros((2, 4), dtype=torch.float32)
        environment.action_start_preload_trim_m = torch.zeros_like(
            environment.preload_trim_m
        )
        environment.action_reference_valid = torch.ones(2, dtype=torch.bool)
        environment.preload_trim_slew_clip_count = torch.zeros(
            2, dtype=torch.long
        )
        action = torch.zeros((2, contract.ACTION_DIM), dtype=torch.float32)
        force = torch.tensor(
            [[1.0e6, 1.0e6, 1.0e6, 1.0e6], [float("nan"), 1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
        updated = environment._updated_preload_trim(action, force)
        self.assertLessEqual(
            float(updated[0].abs().amax()),
            PRELOAD_TRIM_SLEW_M_PER_TICK + 1.0e-9,
        )
        self.assertTrue(torch.equal(updated[1], environment.preload_trim_m[1]))

    def test_preload_trim_has_parent_ik_outward_release_authority(self) -> None:
        import torch

        feedback_start = (
            DEFAULT_CONFIG.plan_ticks
            + (4 * DEFAULT_CONFIG.close_ticks) // 5
        )
        self.assertEqual(FORCE_FEEDBACK_MIN_M, -0.010)
        self.assertEqual(FORCE_FEEDBACK_MAX_M, 0.010)
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.config = DEFAULT_CONFIG
        environment.dtype = torch.float32
        environment.control_step = torch.tensor(
            [feedback_start], dtype=torch.long
        )
        environment.preload_trim_m = torch.full(
            (1, 4), -0.003, dtype=torch.float32
        )
        environment.action_start_preload_trim_m = torch.zeros_like(
            environment.preload_trim_m
        )
        environment.action_reference_valid = torch.zeros(1, dtype=torch.bool)
        environment.preload_trim_slew_clip_count = torch.zeros(
            1, dtype=torch.long
        )
        action = torch.zeros((1, contract.ACTION_DIM), dtype=torch.float32)
        force = torch.full((1, 4), 1.0e6, dtype=torch.float32)
        updated = environment._updated_preload_trim(action, force)
        self.assertTrue((updated < -0.003).all())
        for _ in range(32):
            environment.preload_trim_m = updated
            updated = environment._updated_preload_trim(action, force)
        torch.testing.assert_close(
            updated,
            torch.full_like(updated, FORCE_FEEDBACK_MIN_M),
        )

    def test_action_preload_trim_is_bounded_relative_to_action_start(self) -> None:
        import torch

        _, action_start = _hold_boundaries(DEFAULT_CONFIG)
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.config = DEFAULT_CONFIG
        environment.dtype = torch.float32
        environment.control_step = torch.full(
            (2,), action_start, dtype=torch.long
        )
        environment.action_reference_valid = torch.ones(2, dtype=torch.bool)
        environment.action_start_preload_trim_m = torch.zeros(
            (2, 4), dtype=torch.float32
        )
        limit = contract.TIP_PRELOAD_LIMIT_M
        environment.preload_trim_m = torch.tensor(
            [[limit - 1.0e-5] * 4, [-limit + 1.0e-5] * 4],
            dtype=torch.float32,
        )
        environment.preload_trim_slew_clip_count = torch.zeros(
            2, dtype=torch.long
        )
        action = torch.zeros((2, contract.ACTION_DIM), dtype=torch.float32)
        force = torch.tensor(
            [[0.1] * 4, [1.0e6] * 4], dtype=torch.float32
        )

        updated = environment._updated_preload_trim(action, force)

        torch.testing.assert_close(
            updated[0], torch.full((4,), limit, dtype=torch.float32)
        )
        torch.testing.assert_close(
            updated[1], torch.full((4,), -limit, dtype=torch.float32)
        )

    def test_source_is_native_mjwarp_torch_without_jax_or_brax(self) -> None:
        import MJWarp_RL.env as environment
        import MJWarp_RL.mjwarp_backend as backend
        import MJWarp_RL.preflight as preflight

        env_source = inspect.getsource(environment)
        backend_source = inspect.getsource(backend)
        preflight_source = inspect.getsource(preflight.run_smoke)
        self.assertIn("MJWarpBackend", env_source)
        self.assertIn("backend.step_control()", env_source)
        self.assertNotIn("import jax", env_source)
        self.assertNotIn("import brax", env_source)
        self.assertIn("wp.to_torch", backend_source)
        self.assertIn("wp.from_torch", backend_source)
        self.assertIn("stream_from_torch", backend_source)
        self.assertIn("wait_stream", backend_source)
        self.assertIn("self.device.index is None", backend_source)
        self.assertIn("resolved_sparse_jacobian", backend_source)
        self.assertIn("mjSOL_NEWTON", backend_source)
        self.assertIn('"solver_name"', backend_source)
        self.assertIn("contract.FORMAL_SOLVER", backend_source)
        self.assertIn("mjJAC_DENSE", backend_source)
        self.assertIn("requires an explicitly dense", backend_source)
        self.assertIn("FORMAL_DENSE_NEWTON_MAX_NV", backend_source)
        self.assertIn("graph_replays = 4", preflight_source)
        self.assertIn("torch_warp_stream_handle_match", preflight_source)
        self.assertIn("zero_copy_pointer_match", preflight_source)
        self.assertEqual(OBSERVATION_DIM, 607)
        self.assertEqual(
            sum(width for _, width in OBSERVATION_SCHEMA),
            OBSERVATION_DIM,
        )
        for name in (
            "object_body_position_world",
            "object_com_position_world",
            "object_quaternion_world",
            "goal_position_world",
            "goal_quaternion_world",
            "tip_position_world",
            "tip_target_world",
            "object_local_height_geometry",
            "grasp_band_fraction",
            "tip_height_fraction",
            "grasp_plane_fraction",
            "actual_tip_height_fraction",
            "actual_grasp_plane_fraction",
            "actual_grasp_plane_within_bounds",
            "actual_grasp_plane_compliant",
            "active_joint_qpos_padded",
            "active_joint_qvel_padded",
            "active_joint_ctrl_padded",
            "position_feedback",
            "action_start_position_feedback",
            "action_start_preload_trim",
            "ctrl_velocity_padded",
            "ctrl_acceleration_padded",
            "last_certified_ctrl_padded",
            "passive_joint_qpos",
            "passive_joint_qvel",
            "qacc_warmstart_padded",
            "qacc_warmstart_valid_mask",
        ):
            self.assertIn(name, OBSERVATION_SLICES)
        covered = [
            index
            for value in OBSERVATION_SLICES.values()
            for index in range(value.start, value.stop)
        ]
        self.assertEqual(covered, list(range(OBSERVATION_DIM)))

    def test_capacity_report_selects_largest_passing_candidate(self) -> None:
        import MJWarp_RL.train as training
        from MJWarp_RL.train import _resolve_num_envs

        self.assertEqual(
            CANDIDATES,
            FORMAL_WORLD_CANDIDATES,
        )
        self.assertEqual(REPORT_SCHEMA, CAPACITY_REPORT_SCHEMA)
        self.assertEqual(
            CAPACITY_REPORT_SCHEMA, "grab-mjwarp-capacity-scan-v13"
        )
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "capacity.json"
            payload = {
                "schema": REPORT_SCHEMA,
                "is_common_report": True,
                "common_platforms": ["gripper3", "gripper6"],
                "algorithm_signature": contract.ALGORITHM_SIGNATURE,
                "config_canonical_sha256": training._config_sha256(
                    DEFAULT_CONFIG
                ),
                "candidate_config_set_sha256": (
                    training._capacity_config_sha256(
                        DEFAULT_CONFIG.task, DEFAULT_CONFIG.device
                    )
                ),
                "candidate_config_sha256": {
                    str(candidate): training._config_sha256(
                        replace(
                            DEFAULT_CONFIG,
                            num_envs=candidate,
                            device=DEFAULT_CONFIG.device,
                        )
                    )
                    for candidate in FORMAL_WORLD_CANDIDATES
                },
                "observation_dim": OBSERVATION_DIM,
                "action_dim": contract.ACTION_DIM,
                "episode_control_ticks": DEFAULT_CONFIG.episode_length,
                "candidates": list(FORMAL_WORLD_CANDIDATES),
                "passing_candidates": [contract.FORMAL_GLOBAL_WORLDS],
                "selected_num_envs": contract.FORMAL_GLOBAL_WORLDS,
            }
            report.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            self.assertEqual(
                _resolve_num_envs("auto", report, parser),
                contract.FORMAL_GLOBAL_WORLDS,
            )
            payload["selected_num_envs"] = 12_000
            report.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                parser, "error", side_effect=ValueError
            ):
                with self.assertRaises(ValueError):
                    _resolve_num_envs("auto", report, parser)

    def test_device_memory_certificate_polls_beyond_torch_allocator(self) -> None:
        import MJWarp_RL.capacity_scan as capacity

        class FakeCuda:
            def __init__(self) -> None:
                self.calls = 0

            def synchronize(self, device: object) -> None:
                del device

            def set_device(self, device: object) -> None:
                del device

            def mem_get_info(self, device: object) -> tuple[int, int]:
                del device
                free_values = (900, 860, 810, 760, 720, 700)
                value = free_values[min(self.calls, len(free_values) - 1)]
                self.calls += 1
                return value, 1_000

        fake_torch = SimpleNamespace(cuda=FakeCuda())
        tracker = capacity._DeviceMemoryTracker(
            fake_torch,
            "cuda:0",
            stage="unit",
            poll_interval_s=0.001,
        )
        tracker.begin_segment("remainder")
        time.sleep(0.01)
        report = tracker.report()
        self.assertTrue(report["valid"])
        self.assertTrue(report["includes_non_torch_cuda_allocations"])
        self.assertGreater(report["background_sample_count"], 0)
        self.assertEqual(report["poll_thread_errors"], [])
        self.assertEqual(report["minimum_free_bytes"], 700)
        self.assertIn("remainder", report["segments"])

    def test_ik_expert_episode_state_is_explicitly_released(self) -> None:
        import torch

        from MJWarp_RL.ik_expert import MJWarpIKExpert

        episode_fields = (
            "episode_action",
            "episode_planned_grasp_plane_fraction",
            "episode_planned_grasp_plane_yaw_rad",
            "dynamic_knots",
            "goals",
            "group_object",
            "group_task",
            "group_template",
        )
        for task in contract.TASKS:
            config = replace(
                DEFAULT_CONFIG, task=task, num_envs=80, device="cpu"
            )
            expert = MJWarpIKExpert(
                config,
                platform_spec=_fake_platform_spec(
                    "a" * 64, "b" * 64, task=task
                ),
                device="cpu",
            )
            goals = expert.begin_episode(PHASE, 0)
            self.assertTrue(
                bool(
                    (
                        goals["task_index"]
                        == contract.TASKS.index(task)
                    ).all()
                )
            )
            static_fraction_bank = expert.planned_grasp_plane_fraction.clone()
            expert.clear_episode_state()
            for name in episode_fields:
                self.assertIsNone(getattr(expert, name))
            self.assertTrue(
                torch.equal(
                    expert.planned_grasp_plane_fraction,
                    static_fraction_bank,
                )
            )

            original_group_goals = expert._group_goals

            def checked_group_goals(
                instance: object, phase: str, episode_index: int
            ) -> dict[str, object]:
                for name in episode_fields:
                    self.assertIsNone(getattr(instance, name))
                return original_group_goals(phase, episode_index)

            for name in episode_fields:
                setattr(expert, name, object())
            expert._group_goals = MethodType(checked_group_goals, expert)
            expert.begin_episode(PHASE, 1)
            winners = (
                torch.arange(expert.groups, dtype=torch.long)
                * expert.population
            )
            before = expert.episode_action.clone()
            expert.update_centroids(winners)
            self.assertTrue(torch.equal(expert.episode_action, before))

    def test_demo_gate_requires_every_translation_path_shape(self) -> None:
        from MJWarp_RL.train import _validate_demo_distribution

        distribution = {
            f"{PHASE}/{object_name}/{contract.TASK_TRANSLATE}": 6
            for object_name in contract.OBJECTS
        }
        coverage = {
            "template_counts": {
                object_name: {
                    str(index): 1
                    for index in range(contract.RESET_TEMPLATES_PER_OBJECT)
                }
                for object_name in contract.OBJECTS
            },
            "translation_path_by_object": {
                object_name: {
                    "shape_success_counts": {
                        shape_name: 1
                        for shape_name in contract.TRANSLATION_PATH_SHAPES
                    }
                }
                for object_name in contract.OBJECTS
            },
            "rotation_angle_by_object": {
                object_name: {
                    "maximum_target_deg": contract.ROTATION_ANGLE_MAX_DEG,
                    "maximum_target_count": 1,
                    "maximum_target_success_count": 1,
                }
                for object_name in contract.OBJECTS
            },
        }

        class Store:
            completed = 24

            @staticmethod
            def distribution() -> dict[str, int]:
                return distribution

            @staticmethod
            def coverage_distribution() -> dict[str, object]:
                return coverage

        self.assertEqual(
            _validate_demo_distribution(Store(), DEFAULT_CONFIG),
            distribution,
        )
        coverage["translation_path_by_object"]["orange"][
            "shape_success_counts"
        ][contract.TRANSLATION_PATH_SHAPES[0]] = 0
        with self.assertRaisesRegex(
            RuntimeError, "every path shape"
        ):
            _validate_demo_distribution(Store(), DEFAULT_CONFIG)

    def test_capacity_v19_native_protocol_validators_fail_closed(self) -> None:
        import MJWarp_RL.capacity_scan as capacity
        import MJWarp_RL.train as training

        model = "a" * 64
        controller = "b" * 64
        adapter = "c" * 64

        def report(
            task: str = DEFAULT_CONFIG.task,
        ) -> dict[str, object]:
            return {
                "task": task,
                "task_index": contract.TASKS.index(task),
                "root_height_m": TASK_ROOT_HEIGHT_M[task],
                "gpu": {
                    "name": "NVIDIA RTX PRO 6000",
                    "memory_bytes": _FAKE_GPU_TOTAL_BYTES,
                    "compute_capability": "12.0",
                },
                "object_world_counts": list(
                    contract.FORMAL_OBJECT_WORLD_COUNTS
                ),
                "remainder_worlds": contract.FORMAL_REMAINDER_WORLDS,
                "remainder_object_world_counts": list(
                    contract.FORMAL_REMAINDER_OBJECT_WORLD_COUNTS
                ),
                "parent_model_fingerprint_sha256": model,
                "parent_controller_fingerprint_sha256": controller,
                "platform_adapter_fingerprint_sha256": adapter,
                "stages": {
                    "native_search": _fake_native_protocol(
                        model, controller, adapter, task=task
                    )
                },
            }

        pristine = report()
        self.assertTrue(capacity._native_stage_uses_expected_jacobian(pristine))
        self.assertTrue(training._native_stage_uses_expected_jacobian(pristine))
        for task in contract.TASKS:
            self.assertEqual(
                capacity._run_lifecycle_counts(
                    replace(DEFAULT_CONFIG, task=task),
                ),
                (3000, 1, 0),
            )

        def clone() -> dict[str, object]:
            return json.loads(json.dumps(pristine))

        tampered = clone()
        tampered["stages"]["native_search"]["full_episode"]["worlds"] -= 10
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        tampered = clone()
        tampered["stages"]["native_search"]["full_episode"][
            "final_tail_ticks"
        ] = 1
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        rotation = report(contract.TASK_ROTATE)
        self.assertTrue(capacity._native_stage_uses_expected_jacobian(rotation))
        self.assertTrue(training._native_stage_uses_expected_jacobian(rotation))
        rotation_run = rotation["stages"]["native_search"]["full_episode"]
        self.assertEqual(rotation_run["phase"], PHASE)
        self.assertEqual(rotation_run["complete_episode_count"], 1)
        self.assertEqual(rotation_run["final_tail_ticks"], 0)
        self.assertEqual(
            len(
                rotation_run["post_extraction_lifecycle"][
                    "additional_complete_episodes"
                ]
            ),
            0,
        )

        tampered_rotation = json.loads(json.dumps(rotation))
        tampered_rotation["stages"]["native_search"]["full_episode"][
            "complete_episode_count"
        ] = 2
        self.assertFalse(
            capacity._native_stage_uses_expected_jacobian(tampered_rotation)
        )
        self.assertFalse(
            training._native_stage_uses_expected_jacobian(tampered_rotation)
        )

        tampered = clone()
        tampered["stages"]["native_search"]["full_episode"][
            "rotation_goal_coverage"
        ]["reason"] = "wrong"
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        tampered = clone()
        path_coverage = tampered["stages"]["native_search"]["full_episode"][
            "translation_path_goal_coverage"
        ]
        path_coverage["by_object"]["orange"]["shape_counts_by_name"][
            contract.TRANSLATION_PATH_SHAPES[-1]
        ] = 0
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        tampered = clone()
        tampered["stages"]["native_search"]["remainder"] = {}
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        tampered = clone()
        details = tampered["stages"]["native_search"]["actual_device_memory"]
        details["checkpoint_samples"][0]["free_bytes"] = (
            _FAKE_GPU_TOTAL_BYTES + 1
        )
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        tampered = clone()
        details = tampered["stages"]["native_search"]["actual_device_memory"]
        details["checkpoint_samples"][0]["device_used_bytes"] += 1
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        tampered = clone()
        tampered["stages"]["native_search"][
            "actual_peak_device_memory_fraction"
        ] = 0.10
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        tampered = clone()
        native = tampered["stages"]["native_search"]
        certificate = native["post_release_baseline_certificates"][
            "formal_global_batch"
        ]
        old_label = certificate["baseline"]["label"]
        certificate["baseline"]["label"] = "wrong"
        for sample in native["actual_device_memory"]["checkpoint_samples"]:
            if sample["label"] == old_label:
                sample["label"] = "wrong"
                break
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        tampered = clone()
        native = tampered["stages"]["native_search"]
        checkpoints = native["actual_device_memory"]["checkpoint_samples"]
        target = "pre_allocation:formal_global_batch"
        native["actual_device_memory"]["checkpoint_samples"] = [
            sample for sample in checkpoints if sample["label"] != target
        ]
        native["actual_device_memory"]["checkpoint_sample_count"] -= 1
        native["actual_device_memory"]["sample_count"] -= 1
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        tampered = clone()
        tampered["stages"]["native_search"]["full_episode"][
            "actual_device_memory_segment"
        ] = {
            "minimum_free_bytes": 49 * 1024**3,
            "actual_peak_device_used_bytes": 47 * 1024**3,
            "actual_peak_device_used_gb": 47.0,
            "actual_peak_device_memory_fraction": 47.0 / 96.0,
        }
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        tampered = clone()
        tampered["stages"]["native_search"]["full_episode"][
            "post_extraction_lifecycle"
        ]["controller_safety_counters"][
            "ctrl_certificate_fallback_count"
        ] = 1
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        tampered = clone()
        tampered["stages"]["native_search"][
            "controller_safety_counters_total"
        ]["ik_solve_fallback_count"] = 1
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

        tampered = clone()
        tampered["stages"]["native_search"][
            "post_release_baseline_certificates"
        ]["formal_global_batch"]["passed"] = False
        self.assertFalse(capacity._native_stage_uses_expected_jacobian(tampered))
        self.assertFalse(training._native_stage_uses_expected_jacobian(tampered))

    def test_capacity_hashes_shards_and_coverage_are_task_bound(self) -> None:
        import MJWarp_RL.capacity_scan as capacity
        import MJWarp_RL.train as training

        self.assertNotEqual(
            capacity._scan_config_sha256(
                contract.TASK_TRANSLATE, DEFAULT_CONFIG.device
            ),
            capacity._scan_config_sha256(
                contract.TASK_ROTATE, DEFAULT_CONFIG.device
            ),
        )
        self.assertNotEqual(
            training._capacity_config_sha256(
                contract.TASK_TRANSLATE, DEFAULT_CONFIG.device
            ),
            training._capacity_config_sha256(
                contract.TASK_ROTATE, DEFAULT_CONFIG.device
            ),
        )
        task = contract.TASK_TRANSLATE
        root_height_m = float(TASK_ROOT_HEIGHT_M[task])
        manifest = _fake_model_shards(
            contract.FORMAL_OBJECT_WORLD_COUNTS,
            task=task,
            root_height_m=root_height_m,
        )
        self.assertTrue(
            capacity._valid_shard_manifest(
                manifest,
                expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
                expected_task=task,
                expected_root_height_m=root_height_m,
            )
        )
        self.assertFalse(
            capacity._valid_shard_manifest(
                manifest,
                expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
                expected_task=contract.TASK_ROTATE,
                expected_root_height_m=root_height_m,
            )
        )

        for selected_task in contract.TASKS:
            with self.subTest(task=selected_task):
                translation = _fake_translation_path_coverage(
                    PHASE, task=selected_task
                )
                rotation = _fake_rotation_goal_coverage(
                    PHASE, task=selected_task
                )
                certified = _fake_certified_candidate_coverage(
                    PHASE, task=selected_task
                )
                self.assertTrue(
                    capacity._translation_path_goal_coverage_is_valid(
                        translation,
                        PHASE,
                        task=selected_task,
                        expected_world_counts=(
                            contract.FORMAL_OBJECT_WORLD_COUNTS
                        ),
                    )
                )
                if selected_task == contract.TASK_TRANSLATE:
                    sampled = json.loads(json.dumps(translation))
                    sampled_shape_index = 1
                    sampled_shape = contract.TRANSLATION_PATH_SHAPES[
                        sampled_shape_index
                    ]
                    for entry in sampled["by_object"].values():
                        entry["shape_index_values"] = [sampled_shape_index]
                        entry["shape_counts_by_name"] = {
                            shape: (
                                entry["world_count"]
                                if shape == sampled_shape
                                else 0
                            )
                            for shape in contract.TRANSLATION_PATH_SHAPES
                        }
                        sampled_step = (
                            contract.TRANSLATION_PATH_STEP_M_BY_SHAPE[
                                sampled_shape_index
                            ]
                        )
                        sampled_length = (
                            contract.TRANSLATION_PATH_LENGTH_M_BY_SHAPE[
                                sampled_shape_index
                            ]
                        )
                        entry["path_step_min_m"] = sampled_step
                        entry["path_step_max_m"] = sampled_step
                        entry["path_length_min_m"] = sampled_length
                        entry["path_length_max_m"] = sampled_length
                    self.assertFalse(
                        capacity._translation_path_goal_coverage_is_valid(
                            sampled,
                            PHASE,
                            task=selected_task,
                            expected_world_counts=(
                                contract.FORMAL_OBJECT_WORLD_COUNTS
                            ),
                        )
                    )
                    self.assertTrue(
                        capacity._translation_path_goal_coverage_is_valid(
                            sampled,
                            PHASE,
                            task=selected_task,
                            expected_world_counts=(
                                contract.FORMAL_OBJECT_WORLD_COUNTS
                            ),
                            require_complete_shape_coverage=False,
                        )
                    )
                self.assertTrue(
                    capacity._rotation_goal_coverage_is_valid(
                        rotation,
                        PHASE,
                        task=selected_task,
                        expected_world_counts=(
                            contract.FORMAL_OBJECT_WORLD_COUNTS
                        ),
                    )
                )
                self.assertTrue(
                    capacity._certified_candidate_coverage_is_valid(
                        certified,
                        task=selected_task,
                        phase=PHASE,
                        expected_world_counts=(
                            contract.FORMAL_OBJECT_WORLD_COUNTS
                        ),
                        population=DEFAULT_CONFIG.elite.population,
                    )
                )
                certified["valid_group_count"] = 0
                self.assertFalse(
                    capacity._certified_candidate_coverage_is_valid(
                        certified,
                        task=selected_task,
                        phase=PHASE,
                        expected_world_counts=(
                            contract.FORMAL_OBJECT_WORLD_COUNTS
                        ),
                        population=DEFAULT_CONFIG.elite.population,
                    )
                )

    def test_capacity_cli_requires_explicit_task(self) -> None:
        import contextlib
        import io
        import MJWarp_RL.capacity_scan as capacity

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                capacity.main([])
        self.assertEqual(raised.exception.code, 2)

    def test_remainder_transition_reports_and_enforces_hard_counters(self) -> None:
        import torch

        import MJWarp_RL.capacity_scan as capacity

        config = replace(DEFAULT_CONFIG, device="cpu", num_envs=40_010)
        allocation = tuple(config.remainder_object_world_counts)

        class FakeBackend:
            def health(self) -> dict[str, object]:
                return _fake_sharded_health(world_counts=allocation)

        class FakeEnvironment:
            def __init__(self, ik_fallbacks: int) -> None:
                self.backend = FakeBackend()
                self.closed = False
                for name in _CONTROLLER_COUNTER_NAMES:
                    setattr(self, name, torch.zeros(1, dtype=torch.int64))
                self.ik_solve_fallback_count[0] = ik_fallbacks

            def reset(self) -> torch.Tensor:
                return torch.zeros((sum(allocation), 1))

            def step(self, action: torch.Tensor) -> SimpleNamespace:
                worlds = action.shape[0]
                return SimpleNamespace(
                    done=torch.zeros(worlds, dtype=torch.bool),
                    reward=torch.zeros(worlds),
                    info={
                        "transition_finite": torch.ones(
                            worlds, dtype=torch.bool
                        )
                    },
                )

            def close(self) -> None:
                self.closed = True

        memory = SimpleNamespace(sample=lambda *args, **kwargs: None)
        valid_environment = FakeEnvironment(0)
        with mock.patch.object(
            capacity, "make_env", return_value=valid_environment
        ), mock.patch.object(
            capacity,
            "_translation_path_goal_coverage",
            return_value=_fake_translation_path_coverage(
                PHASE, allocation
            ),
        ):
            result = capacity._run_remainder_transition(
                torch=torch,
                config=config,
                platform_spec=object(),
                device=torch.device("cpu"),
                memory=memory,
                phase=PHASE,
                validate_official_config=True,
            )
        self.assertEqual(
            result["controller_safety_counters"],
            _fake_controller_counters(),
        )
        self.assertTrue(valid_environment.closed)

        failed_environment = FakeEnvironment(1)
        with mock.patch.object(
            capacity, "make_env", return_value=failed_environment
        ), mock.patch.object(
            capacity,
            "_translation_path_goal_coverage",
            return_value=_fake_translation_path_coverage(
                PHASE, allocation
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "ik_solve_fallback_count"):
                capacity._run_remainder_transition(
                    torch=torch,
                    config=config,
                    platform_spec=object(),
                    device=torch.device("cpu"),
                    memory=memory,
                    phase=PHASE,
                    validate_official_config=True,
                )
        self.assertTrue(failed_environment.closed)

    def test_capacity_manifests_require_complete_exact_dynamics_fields(self) -> None:
        import MJWarp_RL.capacity_scan as capacity
        import MJWarp_RL.train as training

        manifest = _fake_model_shards(contract.FORMAL_OBJECT_WORLD_COUNTS)
        self.assertTrue(
            capacity._valid_shard_manifest(
                manifest,
                expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
                expected_task=DEFAULT_CONFIG.task,
                expected_root_height_m=TASK_ROOT_HEIGHT_M[DEFAULT_CONFIG.task],
            )
        )
        self.assertTrue(
            training._capacity_shards_are_valid(
                manifest,
                expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
                expected_task=DEFAULT_CONFIG.task,
                expected_root_height_m=TASK_ROOT_HEIGHT_M[DEFAULT_CONFIG.task],
            )
        )
        for missing in ("nq", "nv", "nu"):
            tampered = json.loads(json.dumps(manifest))
            del tampered[0][missing]
            self.assertFalse(
                capacity._valid_shard_manifest(
                    tampered,
                    expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
                    expected_task=DEFAULT_CONFIG.task,
                    expected_root_height_m=TASK_ROOT_HEIGHT_M[
                        DEFAULT_CONFIG.task
                    ],
                )
            )
            self.assertFalse(
                training._capacity_shards_are_valid(
                    tampered,
                    expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
                    expected_task=DEFAULT_CONFIG.task,
                    expected_root_height_m=TASK_ROOT_HEIGHT_M[
                        DEFAULT_CONFIG.task
                    ],
                )
            )
        mismatched = json.loads(json.dumps(manifest))
        mismatched[0]["model_xml_sha256"] = "f" * 64
        self.assertFalse(
            capacity._valid_shard_manifest(
                mismatched,
                expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
                expected_task=DEFAULT_CONFIG.task,
                expected_root_height_m=TASK_ROOT_HEIGHT_M[DEFAULT_CONFIG.task],
                expected_manifest=manifest,
            )
        )
        self.assertFalse(
            training._capacity_shards_are_valid(
                mismatched,
                expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
                expected_task=DEFAULT_CONFIG.task,
                expected_root_height_m=TASK_ROOT_HEIGHT_M[DEFAULT_CONFIG.task],
                expected_manifest=manifest,
            )
        )

    def test_runtime_code_fingerprint_binds_every_critical_module(self) -> None:
        from MJWarp_RL.runtime_fingerprint import (
            RUNTIME_CODE_FILES,
            RUNTIME_CODE_SCHEMA,
            runtime_code_fingerprint_sha256,
            runtime_code_manifest,
        )

        required = {
            "contract.py",
            "hpc_config.py",
            "memory_certificate.py",
            "runtime_memory_guard.py",
            "curriculum.py",
            "success.py",
            "env.py",
            "newton_regularization.py",
            "mjwarp_backend.py",
            "sharded_env.py",
            "ik_expert.py",
            "demos.py",
            "diffusion_policy.py",
            "diffusion_runtime.py",
            "preflight.py",
            "train.py",
            "capacity_scan.py",
            "evaluate.py",
            "runtime_fingerprint.py",
        }
        self.assertEqual(RUNTIME_CODE_SCHEMA, "grab-shared-runtime-code-v5")
        self.assertEqual(required, set(RUNTIME_CODE_FILES))
        current = runtime_code_fingerprint_sha256()
        self.assertEqual(len(current), 64)
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory)
            package = Path(__file__).resolve().parents[1]
            for relative in RUNTIME_CODE_FILES:
                shutil.copy2(package / relative, copied / relative)
            self.assertEqual(
                runtime_code_fingerprint_sha256(copied), current
            )
            regularization_copy = copied / "newton_regularization.py"
            regularization_copy.write_bytes(
                regularization_copy.read_bytes()
                + b"\n# fingerprint mutation\n"
            )
            self.assertNotEqual(
                runtime_code_fingerprint_sha256(copied), current
            )
            manifest = runtime_code_manifest(copied)
            self.assertEqual(set(manifest["files"]), set(RUNTIME_CODE_FILES))

    def test_capacity_and_training_release_every_shard_boundary(self) -> None:
        import MJWarp_RL.capacity_scan as capacity
        import MJWarp_RL.train as training

        capacity_source = inspect.getsource(capacity._run_native_search_peak)
        self.assertIn("phase = PHASE", capacity_source)
        self.assertNotIn("PHASE1", capacity_source)
        self.assertNotIn("PHASE2", capacity_source)
        self.assertIn("remainder_result", capacity_source)
        self.assertIn("full_episode", capacity_source)
        self.assertIn("post_release_baseline_certificates", capacity_source)
        remainder_call = capacity_source.index("_run_remainder_transition")
        full_call = capacity_source.index("_run_full_search_episode_peak")
        self.assertLess(
            remainder_call,
            full_call,
        )
        self.assertIn(
            "_post_release_baseline_certificate",
            capacity_source[remainder_call:full_call],
        )
        full_source = inspect.getsource(
            capacity._run_full_search_episode_peak
        )
        self.assertLess(
            full_source.index("trajectory_observation = torch.empty"),
            full_source.index("teacher.begin_episode"),
        )
        self.assertIn("_run_lifecycle_counts", full_source)
        self.assertIn(
            "for episode_index in range(1, complete_episode_count)",
            full_source,
        )
        self.assertIn("final_tail_ticks", full_source)
        self.assertNotIn("second_episode_tail_ticks", full_source)
        training_source = inspect.getsource(training._train_two_phase_impl)
        task_start = training_source.index("full_task = _run_expert_task")
        diffusion_admission = training_source.index('stage="diffusion"')
        self.assertIn(
            "_release_cuda_stage_memory",
            training_source[task_start:diffusion_admission],
        )
        self.assertNotIn("_run_expert_phase", training_source)
        self.assertLess(
            training_source.index("del teacher"),
            training_source.rindex("_release_cuda_stage_memory"),
        )

    def test_cuda_release_trims_retired_graph_memory(self) -> None:
        import MJWarp_RL.capacity_scan as capacity
        import MJWarp_RL.train as training

        for release in (
            capacity._release_cuda_stage_memory,
            training._release_cuda_stage_memory,
        ):
            events: list[tuple[str, object]] = []
            wp_device = SimpleNamespace(ordinal=7)
            fake_warp = ModuleType("warp")
            fake_warp.get_device = lambda value: (
                events.append(("get_device", value)) or wp_device
            )
            fake_warp.set_mempool_release_threshold = (
                lambda device, threshold: events.append(
                    ("threshold", (device, threshold))
                )
            )
            fake_warp.synchronize_device = lambda device: events.append(
                ("warp_sync", device)
            )
            fake_warp._src = SimpleNamespace(
                context=SimpleNamespace(
                    runtime=SimpleNamespace(
                        core=SimpleNamespace(
                            wp_cuda_device_graph_mem_trim=lambda ordinal: (
                                events.append(("graph_trim", ordinal))
                            )
                        )
                    )
                )
            )
            fake_torch = SimpleNamespace(
                device=lambda value: value,
                cuda=SimpleNamespace(
                    synchronize=lambda device: events.append(
                        ("torch_sync", device)
                    ),
                    empty_cache=lambda: events.append(
                        ("torch_empty", None)
                    ),
                ),
            )
            with (
                mock.patch.dict(sys.modules, {"warp": fake_warp}),
                mock.patch.object(
                    capacity.time,
                    "sleep",
                    side_effect=lambda delay: events.append(
                        ("sleep", delay)
                    ),
                ),
            ):
                release(fake_torch, "cuda:0")

            trim_index = events.index(("graph_trim", 7))
            self.assertLess(events.index(("sleep", 0.1)), trim_index)
            self.assertGreaterEqual(
                sum(
                    1
                    for name, _ in events[:trim_index]
                    if name == "warp_sync"
                ),
                2,
            )
            self.assertEqual(events.count(("graph_trim", 7)), 1)
            self.assertEqual(events[trim_index + 1], ("warp_sync", wp_device))

    def test_training_wrapper_always_finalizes_cuda_memory_guard(self) -> None:
        import MJWarp_RL.train as training

        class FakeGuard:
            def __init__(self) -> None:
                self.report_calls = 0
                self.close_calls = 0

            def report(self) -> dict[str, object]:
                self.report_calls += 1
                return {"valid": True, "closed": True, "errors": []}

            def close(self) -> None:
                self.close_calls += 1

        class FakeLogger:
            def __init__(self) -> None:
                self.last_total_steps = 123
                self.records: list[dict[str, object]] = []

            def write(self, **record: object) -> dict[str, object]:
                self.records.append(dict(record))
                return dict(record)

        success_guard = FakeGuard()
        success_logger = FakeLogger()
        expected = Path("/tmp/final_diffusion_model.pt")

        def successful_impl(*args: object, **kwargs: object) -> Path:
            del args
            kwargs["memory_guard_sink"].append(success_guard)
            kwargs["logger_sink"].append(success_logger)
            return expected

        with mock.patch.object(
            training, "_train_two_phase_impl", side_effect=successful_impl
        ):
            actual = training.train_two_phase(
                DEFAULT_CONFIG,
                Path("/tmp/mock-success"),
                capacity_report=Path("/tmp/capacity.json"),
            )
        self.assertEqual(actual, expected)
        self.assertEqual(success_guard.report_calls, 0)
        self.assertEqual(success_guard.close_calls, 1)
        self.assertEqual(success_logger.records, [])

        failure_guard = FakeGuard()
        failure_logger = FakeLogger()

        def failing_impl(*args: object, **kwargs: object) -> Path:
            del args
            kwargs["memory_guard_sink"].append(failure_guard)
            kwargs["logger_sink"].append(failure_logger)
            raise ValueError("mock training failure")

        with mock.patch.object(
            training, "_train_two_phase_impl", side_effect=failing_impl
        ):
            with self.assertRaisesRegex(ValueError, "mock training failure"):
                training.train_two_phase(
                    DEFAULT_CONFIG,
                    Path("/tmp/mock-failure"),
                    capacity_report=Path("/tmp/capacity.json"),
                )
        self.assertEqual(failure_guard.report_calls, 1)
        self.assertEqual(failure_guard.close_calls, 1)
        self.assertEqual(len(failure_logger.records), 1)
        failure_record = failure_logger.records[0]
        self.assertEqual(failure_record["stage"], "failed")
        self.assertEqual(failure_record["total_steps"], 123)
        self.assertEqual(
            failure_record["training_device_memory"]["valid"], True
        )
        self.assertIn("ValueError", failure_record["error"])
        self.assertIn(
            "training_device_memory = memory_guard.report()",
            inspect.getsource(training._train_two_phase_impl),
        )

    def test_teacher_constructor_oom_is_observed_by_an_existing_guard(
        self,
    ) -> None:
        import MJWarp_RL.train as training

        class FakeTorch:
            @staticmethod
            def device(value: str) -> str:
                return value

        class FakeGuard:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                self.report_calls = 0
                self.close_calls = 0
                self.labels: list[str] = []

            def raise_if_violated(self, label: str) -> None:
                self.labels.append(label)

            def report(self) -> dict[str, object]:
                self.report_calls += 1
                return {"valid": True, "closed": True, "errors": []}

            def close(self) -> None:
                self.close_calls += 1

        guard = FakeGuard()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "teacher-oom"
            with (
                mock.patch.object(
                    training,
                    "inspect_runtime",
                    return_value={"ok": True, "errors": []},
                ),
                mock.patch.object(
                    training, "build_platform_spec", return_value=object()
                ),
                mock.patch.object(
                    training,
                    "_validate_capacity_selection",
                    return_value={
                        "gpu": {"memory_bytes": 1_000},
                        "certified_stage_peak_increment_by_stage": {
                            "native_search": 100,
                            "diffusion": 200,
                        },
                    },
                ),
                mock.patch.object(
                    training, "_configure_torch", return_value=FakeTorch()
                ),
                mock.patch.object(
                    training,
                    "TrainingDeviceMemoryGuard",
                    return_value=guard,
                ),
                mock.patch.object(
                    training,
                    "MJWarpIKExpert",
                    side_effect=MemoryError("teacher constructor OOM"),
                ),
            ):
                with self.assertRaisesRegex(
                    MemoryError, "teacher constructor OOM"
                ):
                    training.train_two_phase(
                        DEFAULT_CONFIG,
                        output,
                        capacity_report=Path("/tmp/capacity.json"),
                    )
        self.assertEqual(guard.report_calls, 1)
        self.assertEqual(guard.close_calls, 1)
        self.assertEqual(guard.labels, ["pre_teacher"])
        source = inspect.getsource(training._train_two_phase_impl)
        guard_constructor = source.index("TrainingDeviceMemoryGuard(")
        pre_teacher = source.index(
            'memory_guard.raise_if_violated("pre_teacher")'
        )
        teacher_constructor = source.index("teacher = MJWarpIKExpert(")
        self.assertLess(guard_constructor, pre_teacher)
        self.assertLess(pre_teacher, teacher_constructor)

    def test_progress_logger_persists_terminal_failed_state(self) -> None:
        import MJWarp_RL.train as training

        with tempfile.TemporaryDirectory() as directory:
            logger = training.ProgressLogger(
                Path(directory), total_steps=50_000_000
            )
            with mock.patch("builtins.print"):
                payload = logger.write(
                    stage="failed",
                    phase="failed",
                    total_steps=123,
                    error="mock failure",
                )
            self.assertEqual(payload["state"], "failed")
            self.assertEqual(logger.last_total_steps, 123)
            persisted = json.loads(
                logger.status_path.read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["state"], "failed")

    def test_progress_logger_serializes_endpoint_vector_tensors(self) -> None:
        import torch
        import MJWarp_RL.train as training

        expected = [0.125, -0.25, 0.5]
        endpoint_vector_names = (
            "translation_endpoint_desired_final_displacement_xyz_mean_m",
            "translation_endpoint_actual_final_displacement_xyz_mean_m",
            "translation_endpoint_final_displacement_error_xyz_mean_m",
            "translation_endpoint_final_euler_error_xyz_mean_deg",
        )
        with tempfile.TemporaryDirectory() as directory:
            logger = training.ProgressLogger(
                Path(directory), total_steps=50_000_000
            )
            with mock.patch("builtins.print"):
                payload = logger.write(
                    stage="ik_expert",
                    phase=PHASE,
                    total_steps=24_738_000,
                    nested={"vector": torch.tensor(expected)},
                    scalar=torch.tensor(1.5),
                    **{
                        name: torch.tensor(expected, dtype=torch.float32)
                        for name in endpoint_vector_names
                    },
                )

            for name in endpoint_vector_names:
                self.assertEqual(payload[name], expected)
            self.assertEqual(payload["nested"]["vector"], expected)
            self.assertEqual(payload["scalar"], 1.5)
            persisted = json.loads(
                logger.status_path.read_text(encoding="utf-8")
            )
            jsonl = [
                json.loads(line)
                for line in logger.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(jsonl), 1)
            for name in endpoint_vector_names:
                self.assertEqual(persisted[name], expected)
                self.assertEqual(jsonl[0][name], expected)

    def test_full_search_allocation_failure_closes_partial_environment(
        self,
    ) -> None:
        import torch
        import MJWarp_RL.train as training

        class FakeTeacher:
            def __init__(self) -> None:
                self.clear_calls = 0

            def clear_episode_state(self) -> None:
                self.clear_calls += 1

            def begin_episode(self, *args: object) -> object:
                del args
                raise AssertionError("teacher allocation must not be reached")

        class FakeEnvironment:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1
                raise RuntimeError("mock environment close failure")

        class FakeGuard:
            def raise_if_violated(self, label: str) -> None:
                self.label = label

        teacher = FakeTeacher()
        environment = FakeEnvironment()
        release = mock.Mock(
            side_effect=RuntimeError("mock allocator release failure")
        )
        with (
            mock.patch.object(
                training, "_consume_remainder_shard", return_value=None
            ),
            mock.patch.object(training, "make_env", return_value=environment),
            mock.patch.object(
                training, "_release_cuda_stage_memory", release
            ),
            mock.patch.object(
                torch, "empty", side_effect=MemoryError("mock CUDA OOM")
            ),
        ):
            with self.assertRaisesRegex(MemoryError, "mock CUDA OOM"):
                training._run_expert_task(
                    phase=PHASE,
                    steps=DEFAULT_CONFIG.num_envs,
                    transitions_before_run=0,
                    config=DEFAULT_CONFIG,
                    platform_spec=object(),
                    teacher=teacher,
                    store=object(),
                    logger=object(),
                    global_success={
                        "controller_safety_counters": (
                            training._zero_controller_counters()
                        )
                    },
                    memory_guard=FakeGuard(),
                )
        self.assertEqual(environment.close_calls, 1)
        self.assertGreaterEqual(teacher.clear_calls, 2)
        release.assert_called_once()

    def test_diffusion_cache_allocation_failure_drops_every_cuda_owner(
        self,
    ) -> None:
        import MJWarp_RL.train as training

        instances: dict[str, object] = {}

        class FakeSampler:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                self.device_observations = None
                self.device_actions = None
                instances["sampler"] = self

            def normalization(self) -> dict[str, object]:
                return {}

            def cache_on_device(self, device: str) -> None:
                del device
                self.device_observations = object()
                self.device_actions = object()
                raise MemoryError("mock cache OOM")

        class FakePolicy:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                self._parameters = {"weight": object()}
                self._buffers = {"normalization": object()}
                instances["policy"] = self

            def to(self, device: str) -> "FakePolicy":
                del device
                return self

            def set_normalization(self, value: object) -> None:
                del value

            def modules(self) -> list["FakePolicy"]:
                return [self]

        class FakeGuard:
            def raise_if_violated(self, label: str) -> None:
                self.label = label

        release = mock.Mock(
            side_effect=RuntimeError("mock allocator release failure")
        )
        with (
            mock.patch.object(training, "DemoWindowSampler", FakeSampler),
            mock.patch.object(training, "DiffusionPolicy", FakePolicy),
            mock.patch.object(
                training, "_release_cuda_stage_memory", release
            ),
        ):
            with self.assertRaisesRegex(MemoryError, "mock cache OOM"):
                training._train_diffusion(
                    config=DEFAULT_CONFIG,
                    dataset=Path("/tmp/mock-dataset"),
                    output_dir=Path("/tmp/mock-output"),
                    platform_spec=object(),
                    logger=object(),
                    memory_guard=FakeGuard(),
                )
        sampler = instances["sampler"]
        policy = instances["policy"]
        self.assertIsNone(sampler.device_observations)
        self.assertIsNone(sampler.device_actions)
        self.assertEqual(policy._parameters, {})
        self.assertEqual(policy._buffers, {})
        release.assert_called_once()

    def test_diffusion_ema_constructor_oom_clears_unwound_frames(self) -> None:
        import gc
        import traceback as traceback_module
        import weakref

        import torch
        import MJWarp_RL.train as training

        finalized: list[str] = []

        class Owner:
            pass

        class FakeSampler:
            episodes = 1

            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                self.device_observations = None
                self.device_actions = None

            def normalization(self) -> dict[str, object]:
                return {}

            def cache_on_device(self, device: str) -> None:
                del device

        class FakePolicy:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                self._parameters: dict[str, object] = {}
                self._buffers: dict[str, object] = {}

            def to(self, device: str) -> "FakePolicy":
                del device
                return self

            def set_normalization(self, value: object) -> None:
                del value

            def parameters(self) -> list[object]:
                return []

            def modules(self) -> list["FakePolicy"]:
                return [self]

        class FakeOptimizer:
            def __init__(self) -> None:
                self.state: dict[str, object] = {}
                self.param_groups: list[object] = []

        class FakeScheduler:
            def __init__(self, optimizer: object) -> None:
                self.optimizer = optimizer

        class FailingEMA:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                owner = Owner()
                weakref.finalize(owner, finalized.append, "released")
                self.shadow = {"partial": owner}
                raise MemoryError("EMA constructor OOM")

        class FakeGuard:
            def raise_if_violated(self, label: str) -> None:
                self.label = label

        optimizer = FakeOptimizer()
        real_clear_frames = traceback_module.clear_frames
        with (
            mock.patch.object(training, "DemoWindowSampler", FakeSampler),
            mock.patch.object(training, "DiffusionPolicy", FakePolicy),
            mock.patch.object(
                training, "ExponentialMovingAverage", FailingEMA
            ),
            mock.patch.object(torch.optim, "AdamW", return_value=optimizer),
            mock.patch.object(
                torch.optim.lr_scheduler,
                "CosineAnnealingLR",
                side_effect=lambda value, **_: FakeScheduler(value),
            ),
            mock.patch.object(
                training.traceback,
                "clear_frames",
                wraps=real_clear_frames,
            ) as clear_frames,
            mock.patch.object(
                training,
                "_release_cuda_stage_memory",
                side_effect=lambda *args: gc.collect(),
            ),
        ):
            with self.assertRaisesRegex(MemoryError, "EMA constructor OOM"):
                training._train_diffusion(
                    config=DEFAULT_CONFIG,
                    dataset=Path("/tmp/mock-dataset"),
                    output_dir=Path("/tmp/mock-output"),
                    platform_spec=object(),
                    logger=object(),
                    memory_guard=FakeGuard(),
                )
        gc.collect()
        self.assertTrue(clear_frames.called)
        self.assertEqual(finalized, ["released"])
        self.assertEqual(optimizer.state, {})
        self.assertEqual(optimizer.param_groups, [])

    def test_diffusion_runtime_failure_releases_traceback_temporaries(
        self,
    ) -> None:
        import gc
        import weakref

        import MJWarp_RL.train as training

        finalized: list[str] = []

        class Owner:
            pass

        def failing_body(**kwargs: object) -> tuple[Path, dict[str, object]]:
            del kwargs
            temporary = Owner()
            weakref.finalize(temporary, finalized.append, "released")
            raise RuntimeError("optimizer runtime failure")

        with (
            mock.patch.object(
                training, "_train_diffusion_body", side_effect=failing_body
            ),
            mock.patch.object(
                training,
                "_release_cuda_stage_memory",
                side_effect=lambda *args: gc.collect(),
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "optimizer runtime failure"
            ):
                training._train_diffusion(
                    config=DEFAULT_CONFIG,
                    dataset=Path("/tmp/mock-dataset"),
                    output_dir=Path("/tmp/mock-output"),
                    platform_spec=object(),
                    logger=object(),
                    memory_guard=object(),
                )
        gc.collect()
        self.assertEqual(finalized, ["released"])

    def test_native_and_diffusion_have_separate_baseline_admissions(self) -> None:
        import MJWarp_RL.train as training

        class FakeGuard:
            def __init__(self) -> None:
                self.admitted: list[int] = []
                self.labels: list[str] = []

            def admit(self, increment: int) -> dict[str, object]:
                self.admitted.append(increment)
                return {"passed": True, "total_bytes": 1_000}

            def raise_if_violated(self, label: str) -> None:
                self.labels.append(label)

        selection = {
            "gpu": {"memory_bytes": 1_000},
            "certified_stage_peak_increment_by_stage": {
                "native_search": 100,
                "diffusion": 200,
            },
            "training_memory_admissions": {},
        }
        guard = FakeGuard()
        training._admit_training_stage(
            memory_guard=guard,
            capacity_selection=selection,
            stage="native_search",
        )
        training._admit_training_stage(
            memory_guard=guard,
            capacity_selection=selection,
            stage="diffusion",
        )
        self.assertEqual(guard.admitted, [100, 200])
        self.assertEqual(
            set(selection["training_memory_admissions"]),
            {"native_search", "diffusion"},
        )
        self.assertEqual(
            selection["training_memory_admissions"]["diffusion"][
                "capacity_stage"
            ],
            "diffusion",
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate CUDA admission"):
            training._admit_training_stage(
                memory_guard=guard,
                capacity_selection=selection,
                stage="diffusion",
            )

        source = inspect.getsource(training._train_two_phase_impl)
        native_admission = source.index('stage="native_search"')
        expert_run = source.index("full_task = _run_expert_task")
        expert_release = source.index(
            'raise_if_violated("expert_release_complete")'
        )
        diffusion_admission = source.index('stage="diffusion"')
        diffusion_training = source.index("final_path, diffusion_summary")
        self.assertLess(native_admission, expert_run)
        self.assertLess(expert_release, diffusion_admission)
        self.assertLess(diffusion_admission, diffusion_training)

    def test_common_capacity_report_is_hash_bound_to_both_grippers(
        self,
    ) -> None:
        import MJWarp_RL.train as training
        from MJWarp_RL.runtime_fingerprint import (
            runtime_code_fingerprint_sha256,
        )

        selected = contract.FORMAL_GLOBAL_WORLDS
        config = replace(DEFAULT_CONFIG, num_envs=selected)
        config.validate()
        other_platform = (
            "gripper6" if PLATFORM == "gripper3" else "gripper3"
        )
        current_root_height_m = float(TASK_ROOT_HEIGHT_M[config.task])
        other_root_height_m = 0.525
        gpu = {
            "name": "NVIDIA RTX PRO 6000",
            "memory_bytes": 96 * 1024**3,
            "compute_capability": "12.0",
        }
        candidate_hashes = {
            str(candidate): training._config_sha256(
                replace(
                    DEFAULT_CONFIG,
                    num_envs=candidate,
                    device=config.device,
                )
            )
            for candidate in FORMAL_WORLD_CANDIDATES
        }

        current_model = "a" * 64
        current_controller = "b" * 64
        other_model = "c" * 64
        other_controller = "d" * 64
        current_adapter = "1" * 64
        other_adapter = "2" * 64
        current_rl_model = "3" * 64
        other_rl_model = "4" * 64
        runtime_fingerprint = runtime_code_fingerprint_sha256()

        def result(
            model_fingerprint: str,
            controller_fingerprint: str,
            adapter_fingerprint: str,
            *,
            shard_digest: str,
            platform_name: str,
            root_height_m: float,
            rl_model_fingerprint: str,
        ) -> dict[str, object]:
            diffusion_memory = _fake_stage_memory_certificate(
                "diffusion",
                segments={"diffusion_training": _fake_memory_segment()},
            )
            return {
                "num_envs": selected,
                "passed": True,
                "worker_reported_passed": True,
                "returncode": 0,
                "error": None,
                "validation_errors": [],
                "runtime_errors": [],
                "diagnostic_only": False,
                "formal_training_eligible": True,
                "formal_training_ineligibility_reason": None,
                "platform": platform_name,
                "task": config.task,
                "task_index": contract.TASKS.index(config.task),
                "root_height_m": root_height_m,
                "algorithm_signature": contract.ALGORITHM_SIGNATURE,
                "gpu": gpu,
                "runtime": {"torch": {}, "warp": {}},
                "runtime_code_fingerprint_sha256": runtime_fingerprint,
                "physics_layout": contract.PHYSICS_LAYOUT,
                "object_world_counts": list(
                    contract.FORMAL_OBJECT_WORLD_COUNTS
                ),
                "remainder_worlds": config.remainder_worlds,
                "remainder_object_world_counts": list(
                    config.remainder_object_world_counts
                ),
                "observation_dim": OBSERVATION_DIM,
                "action_dim": contract.ACTION_DIM,
                "episode_control_ticks": config.episode_length,
                "candidate_config_canonical_sha256": (
                    training._config_sha256(config)
                ),
                "model_shards": _fake_model_shards(
                    contract.FORMAL_OBJECT_WORLD_COUNTS,
                    digest_character=shard_digest,
                    task=config.task,
                    root_height_m=root_height_m,
                ),
                "parent_model_fingerprint_sha256": model_fingerprint,
                "parent_controller_fingerprint_sha256": (
                    controller_fingerprint
                ),
                "platform_adapter_fingerprint_sha256": adapter_fingerprint,
                "rl_model_fingerprint_sha256": rl_model_fingerprint,
                "stages": {
                    "native_search": _fake_native_protocol(
                        model_fingerprint,
                        controller_fingerprint,
                        adapter_fingerprint,
                    ),
                    "diffusion": diffusion_memory,
                },
            }

        bindings = {
            "schema": CAPACITY_REPORT_SCHEMA,
            "task": config.task,
            "task_index": contract.TASKS.index(config.task),
            "algorithm_signature": contract.ALGORITHM_SIGNATURE,
            "runtime_code_fingerprint_sha256": runtime_fingerprint,
            "physics_layout": contract.PHYSICS_LAYOUT,
            "object_world_counts": list(
                contract.FORMAL_OBJECT_WORLD_COUNTS
            ),
            "remainder_worlds": config.remainder_worlds,
            "remainder_object_world_counts": list(
                config.remainder_object_world_counts
            ),
            "config_canonical_sha256": training._config_sha256(
                replace(DEFAULT_CONFIG, task=config.task)
            ),
            "candidate_config_set_sha256": (
                training._capacity_config_sha256(config.task, config.device)
            ),
            "candidate_config_sha256": candidate_hashes,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": contract.ACTION_DIM,
            "episode_control_ticks": config.episode_length,
            "device": config.device,
            "gpu": gpu,
            "runtime_errors": [],
            "candidates": list(FORMAL_WORLD_CANDIDATES),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "other.json"
            source = {
                **bindings,
                "is_common_report": False,
                "platform": other_platform,
                "root_height_m": other_root_height_m,
                "rl_model_fingerprint_sha256": other_rl_model,
                "parent_model_fingerprint_sha256": other_model,
                "parent_controller_fingerprint_sha256": other_controller,
                "platform_adapter_fingerprint_sha256": other_adapter,
                "model_shards": _fake_model_shards(
                    contract.FORMAL_OBJECT_WORLD_COUNTS,
                    digest_character="f",
                    task=config.task,
                    root_height_m=other_root_height_m,
                ),
                "common_platforms": [other_platform],
                "platform_root_heights_m": {
                    other_platform: other_root_height_m
                },
                "platform_rl_model_fingerprints_sha256": {
                    other_platform: other_rl_model
                },
                "passing_candidates": [selected],
                "results": [
                    result(
                        other_model,
                        other_controller,
                        other_adapter,
                        shard_digest="f",
                        platform_name=other_platform,
                        root_height_m=other_root_height_m,
                        rl_model_fingerprint=other_rl_model,
                    )
                ],
                "selected_num_envs": selected,
                "common_with": None,
                "common_with_sha256": None,
            }
            source_path.write_text(
                json.dumps(source, sort_keys=True),
                encoding="utf-8",
            )
            source_sha256 = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
            report_path = root / "common.json"
            report = {
                **bindings,
                "is_common_report": True,
                "platform": PLATFORM,
                "root_height_m": current_root_height_m,
                "rl_model_fingerprint_sha256": current_rl_model,
                "parent_model_fingerprint_sha256": current_model,
                "parent_controller_fingerprint_sha256": (
                    current_controller
                ),
                "platform_adapter_fingerprint_sha256": current_adapter,
                "model_shards": _fake_model_shards(
                    contract.FORMAL_OBJECT_WORLD_COUNTS,
                    digest_character="e",
                    task=config.task,
                    root_height_m=current_root_height_m,
                ),
                "common_platforms": ["gripper3", "gripper6"],
                "platform_root_heights_m": {
                    PLATFORM: current_root_height_m,
                    other_platform: other_root_height_m,
                },
                "platform_rl_model_fingerprints_sha256": {
                    PLATFORM: current_rl_model,
                    other_platform: other_rl_model,
                },
                "passing_candidates": [selected],
                "selected_num_envs": selected,
                "selection_rule": "largest common passing candidate",
                "common_with": str(source_path.resolve()),
                "common_with_sha256": source_sha256,
                "results": [
                    result(
                        current_model,
                        current_controller,
                        current_adapter,
                        shard_digest="e",
                        platform_name=PLATFORM,
                        root_height_m=current_root_height_m,
                        rl_model_fingerprint=current_rl_model,
                    )
                ],
            }
            valid_source = json.loads(json.dumps(source))
            valid_report = json.loads(json.dumps(report))
            report_path.write_text(
                json.dumps(report, sort_keys=True),
                encoding="utf-8",
            )
            selection = training._validate_capacity_selection(
                report_path=report_path,
                config=config,
                runtime={"torch": gpu},
                platform_spec=_fake_platform_spec(
                    current_model,
                    current_controller,
                    current_adapter,
                    current_rl_model,
                ),
            )
            report["results"][0]["stages"]["native_search"][
                "backend_health"
            ]["solver_name"] = "CG"
            report_path.write_text(
                json.dumps(report, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                training._validate_capacity_selection(
                    report_path=report_path,
                    config=config,
                    runtime={"torch": gpu},
                    platform_spec=_fake_platform_spec(
                        current_model,
                        current_controller,
                        current_adapter,
                        current_rl_model,
                    ),
                )

            def write_pair(
                source_payload: dict[str, object],
                report_payload: dict[str, object],
            ) -> None:
                source_path.write_text(
                    json.dumps(source_payload, sort_keys=True),
                    encoding="utf-8",
                )
                report_payload["common_with_sha256"] = hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest()
                report_path.write_text(
                    json.dumps(report_payload, sort_keys=True),
                    encoding="utf-8",
                )

            source_tampered = json.loads(json.dumps(valid_source))
            report_tampered = json.loads(json.dumps(valid_report))
            source_tampered["passing_candidates"] = []
            source_tampered["selected_num_envs"] = None
            write_pair(source_tampered, report_tampered)
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                training._validate_capacity_selection(
                    report_path=report_path,
                    config=config,
                    runtime={"torch": gpu},
                    platform_spec=_fake_platform_spec(
                        current_model,
                        current_controller,
                        current_adapter,
                        current_rl_model,
                    ),
                )

            source_tampered = json.loads(json.dumps(valid_source))
            report_tampered = json.loads(json.dumps(valid_report))
            del source_tampered["results"][0]["runtime"]
            write_pair(source_tampered, report_tampered)
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                training._validate_capacity_selection(
                    report_path=report_path,
                    config=config,
                    runtime={"torch": gpu},
                    platform_spec=_fake_platform_spec(
                        current_model,
                        current_controller,
                        current_adapter,
                        current_rl_model,
                    ),
                )

            source_tampered = json.loads(json.dumps(valid_source))
            report_tampered = json.loads(json.dumps(valid_report))
            source_tampered["results"][0]["model_shards"][0][
                "model_xml_sha256"
            ] = "0" * 64
            write_pair(source_tampered, report_tampered)
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                training._validate_capacity_selection(
                    report_path=report_path,
                    config=config,
                    runtime={"torch": gpu},
                    platform_spec=_fake_platform_spec(
                        current_model,
                        current_controller,
                        current_adapter,
                        current_rl_model,
                    ),
                )
            source_path.write_text(
                json.dumps(valid_source, sort_keys=True), encoding="utf-8"
            )
            report["results"][0]["stages"]["native_search"][
                "backend_health"
            ]["resolved_sparse_jacobian"] = False
            report["results"][0]["stages"]["native_search"][
                "backend_health"
            ]["solver_name"] = "Newton"
            report["results"][0][
                "runtime_code_fingerprint_sha256"
            ] = "0" * 64
            report_path.write_text(
                json.dumps(report, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                training._validate_capacity_selection(
                    report_path=report_path,
                    config=config,
                    runtime={"torch": gpu},
                    platform_spec=_fake_platform_spec(
                        current_model,
                        current_controller,
                        current_adapter,
                        current_rl_model,
                    ),
                )
            report["results"][0]["stages"]["native_search"][
                "backend_health"
            ]["solver_name"] = "Newton"
            report["results"][0][
                "runtime_code_fingerprint_sha256"
            ] = runtime_fingerprint
            report["results"][0]["stages"]["native_search"][
                "backend_health"
            ]["resolved_sparse_jacobian"] = True
            report_path.write_text(
                json.dumps(report, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                training._validate_capacity_selection(
                    report_path=report_path,
                    config=config,
                    runtime={"torch": gpu},
                    platform_spec=_fake_platform_spec(
                        current_model,
                        current_controller,
                        current_adapter,
                        current_rl_model,
                    ),
                )
        self.assertEqual(selection["selected_num_envs"], selected)
        self.assertEqual(
            selection["platforms"],
            ["gripper3", "gripper6"],
        )
        self.assertEqual(
            selection["selected_config_sha256"],
            candidate_hashes[str(selected)],
        )

    def test_dense_newton_health_contract_is_fail_closed(self) -> None:
        import MJWarp_RL.capacity_scan as capacity
        import MJWarp_RL.train as training

        health = _fake_sharded_health()
        self.assertTrue(
            capacity._health_is_valid(
                health,
                expected_world_counts=(
                    contract.FORMAL_OBJECT_WORLD_COUNTS
                ),
            )
        )
        training._assert_backend_health(health, context="test")
        invalid_values = {
            "model_nv": PINNED_BLOCK_CHOLESKY_DIM + 1,
            "solver_option": 1,
            "solver_name": "CG",
            "requested_solver_tolerance": 1.0e-8,
            "effective_solver_tolerance": 1.0e-5,
            "jacobian_option": 1,
            "resolved_sparse_jacobian": True,
        }
        for field, value in invalid_values.items():
            with self.subTest(field=field):
                invalid = {**health, field: value}
                self.assertFalse(
                    capacity._health_is_valid(
                        invalid,
                        expected_world_counts=(
                            contract.FORMAL_OBJECT_WORLD_COUNTS
                        ),
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "health failed"):
                    training._assert_backend_health(
                        invalid,
                        context="test",
                    )

    def test_newton_regularization_contract_and_source_are_pinned(self) -> None:
        import MJWarp_RL.mjwarp_backend as backend
        import MJWarp_RL.newton_regularization as regularization

        self.assertEqual(
            NEWTON_REGULARIZATION_SCHEMA,
            "grab-mjwarp-dense-newton-tile-regularization-v1",
        )
        self.assertEqual(PINNED_BLOCK_CHOLESKY_DIM, 32)
        self.assertEqual(NEWTON_RELATIVE_DIAGONAL_SHIFT, 1.0e-6)
        self.assertEqual(NEWTON_ABSOLUTE_DIAGONAL_SHIFT, 1.0e-8)
        self.assertEqual(
            PINNED_SOLVER_SOURCE_SHA256,
            "f07a7c33e038a312ea5c8cab4ec718b24d5e7d163dc8435a592118c9fc0e07fb",
        )
        self.assertEqual(
            PATCH_TARGET,
            "mujoco_warp._src.solver._update_gradient_cholesky",
        )
        metadata = formal_newton_regularization_metadata()
        self.assertTrue(
            formal_newton_regularization_metadata_matches(metadata)
        )
        wrong_type = dict(metadata)
        wrong_type["newton_regularization_max_nv"] = 32.0
        self.assertFalse(
            formal_newton_regularization_metadata_matches(wrong_type)
        )

        source = inspect.getsource(regularization)
        before_installer = source.split(
            "def install_formal_newton_regularization", 1
        )[0]
        self.assertNotIn("import warp", before_installer)
        self.assertIn("@cache_kernel", source)
        self.assertIn("wp.tile_max(abs_tile)[0]", source)
        self.assertIn("wp.tile_diag_add(mat_tile, diagonal)", source)
        self.assertIn(
            "wp.float32(1.0e-6) * scale",
            source,
        )
        self.assertNotIn("wp.launch(", source)
        backend_source = inspect.getsource(backend._configure_runtime)
        self.assertLess(
            backend_source.index("if formal:"),
            backend_source.index("install_formal_newton_regularization"),
        )
        self.assertLess(
            backend_source.index("install_formal_newton_regularization"),
            backend_source.index("elif allow_embedded_fallback:"),
        )

    def test_newton_factory_install_is_idempotent_with_fake_modules(
        self,
    ) -> None:
        import MJWarp_RL.newton_regularization as regularization

        fake_solver = ModuleType("mujoco_warp._src.solver")

        def upstream_factory(
            tile_size: int, skip_noflip: bool = False
        ) -> object:
            return (tile_size, skip_noflip)

        fake_solver._update_gradient_cholesky = upstream_factory
        fake_solver.solve_search_sums = object()
        fake_warp = ModuleType("warp")
        fake_warp_util = ModuleType("mujoco_warp._src.warp_util")
        fake_warp_util.cache_kernel = lambda factory: factory
        fake_src = ModuleType("mujoco_warp._src")
        fake_src.__path__ = []
        fake_src.solver = fake_solver
        fake_mujoco_warp = ModuleType("mujoco_warp")
        fake_mujoco_warp.__path__ = []
        fake_mujoco_warp._src = fake_src
        fake_modules = {
            "warp": fake_warp,
            "mujoco_warp": fake_mujoco_warp,
            "mujoco_warp._src": fake_src,
            "mujoco_warp._src.solver": fake_solver,
            "mujoco_warp._src.warp_util": fake_warp_util,
        }
        with (
            mock.patch.object(regularization, "_verify_solver_source"),
            mock.patch.object(regularization, "wp", None),
            mock.patch.object(regularization, "solve_search_sums", None),
            mock.patch.object(regularization, "cache_kernel", None),
            mock.patch.dict(sys.modules, fake_modules),
        ):
            first_metadata = (
                regularization.install_formal_newton_regularization(
                    fake_solver
                )
            )
            installed = fake_solver._update_gradient_cholesky
            second_metadata = (
                regularization.install_formal_newton_regularization(
                    fake_solver
                )
            )
            self.assertIs(
                fake_solver._update_gradient_cholesky, installed
            )
            self.assertEqual(first_metadata, second_metadata)
            self.assertEqual(
                tuple(inspect.signature(installed).parameters),
                ("tile_size", "skip_noflip"),
            )
            self.assertTrue(
                formal_newton_regularization_metadata_matches(
                    installed._grab_newton_regularization_metadata
                )
            )

    def test_newton_metadata_must_match_aggregate_and_every_shard(self) -> None:
        from MJWarp_RL.sharded_env import formal_sharded_health_is_valid

        health = _fake_sharded_health()
        expected_counts = contract.FORMAL_OBJECT_WORLD_COUNTS
        self.assertTrue(
            formal_sharded_health_is_valid(
                health, expected_world_counts=expected_counts
            )
        )
        for name, expected in formal_newton_regularization_metadata().items():
            with self.subTest(scope="aggregate", name=name):
                tampered = json.loads(json.dumps(health))
                tampered[name] = None
                self.assertFalse(
                    formal_sharded_health_is_valid(
                        tampered, expected_world_counts=expected_counts
                    )
                )
            with self.subTest(scope="shard", name=name):
                tampered = json.loads(json.dumps(health))
                tampered["by_object"][contract.OBJECTS[0]][name] = None
                self.assertFalse(
                    formal_sharded_health_is_valid(
                        tampered, expected_world_counts=expected_counts
                    )
                )

    def test_exact_60m_accounting_includes_each_remainder_shard(self) -> None:
        for worlds in CANDIDATES:
            config = replace(DEFAULT_CONFIG, num_envs=worlds)
            config.validate()
            self.assertEqual(
                config.control_ticks * worlds + config.remainder_worlds,
                config.total_steps,
            )
            self.assertEqual(
                config.total_steps,
                60_000_000,
            )
            self.assertEqual(
                sum(config.remainder_object_world_counts),
                config.remainder_worlds,
            )

    def test_diffusion_manifest_config_roundtrip_ignores_derived_fields(
        self,
    ) -> None:
        from MJWarp_RL.evaluate import _config_from_mapping

        restored = _config_from_mapping(DEFAULT_CONFIG.as_dict())
        self.assertEqual(
            restored.canonical_json(),
            DEFAULT_CONFIG.canonical_json(),
        )

    def test_only_final_diffusion_model_is_named_as_training_output(self) -> None:
        import MJWarp_RL.train as training

        source = inspect.getsource(training)
        signature = inspect.signature(training.train_two_phase)
        capacity_parameter = signature.parameters["capacity_report"]
        self.assertEqual(
            capacity_parameter.kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        self.assertIs(
            capacity_parameter.default,
            inspect.Parameter.empty,
        )
        self.assertIn("final_diffusion_model.pt", source)
        self.assertNotIn("phase1_model", source)
        self.assertIn('"success_gate": True', source)
        self.assertIn('"anti_support_gate_used": False', source)
        self.assertIn(
            '"physical_hard_tip_plane_success_invariant": True',
            source,
        )
        self.assertIn('"capacity_selection": capacity_selection', source)
        self.assertIn(
            '"certified_demo_coverage": coverage_distribution', source
        )
        self.assertNotIn("from .tqc import", source)


if __name__ == "__main__":
    unittest.main()
