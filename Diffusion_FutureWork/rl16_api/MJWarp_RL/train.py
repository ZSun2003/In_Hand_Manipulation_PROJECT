"""Certified closed-loop IK demonstrations followed by Diffusion training."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any, Mapping

from . import contract
from .capacity_scan import (
    _certified_candidate_coverage_is_valid,
    _native_stage_uses_expected_jacobian as _capacity_native_stage_uses_expected_jacobian,
    _rotation_goal_coverage_is_valid as _capacity_rotation_coverage_is_valid,
    _translation_path_goal_coverage_is_valid,
)
from .curriculum import PHASE, rotation_angle_bounds_deg
from .demos import CertifiedIKDemoStore, DemoWindowSampler, sha256_file
from .diffusion_policy import (
    DiffusionPolicy,
    ExponentialMovingAverage,
    policy_checkpoint,
    seed_everything,
)
from .ik_expert import MJWarpIKExpert
from .env import OBSERVATION_DIM, OBSERVATION_SLICES, make_env
from .hpc_config import (
    CAPACITY_REPORT_SCHEMA,
    DEFAULT_CONFIG,
    FORMAL_WORLD_CANDIDATES,
    HPCConfig,
)
from .memory_certificate import (
    post_release_certificate_is_valid,
    stage_has_actual_memory_certificate,
)
from .platform_adapter import PROJECT_ROOT, build_platform_spec
from .preflight import configure_cublas_workspace, inspect_runtime
from .sharded_env import (
    formal_sharded_health_is_valid,
    ordered_shard_manifest,
)
from .runtime_fingerprint import runtime_code_fingerprint_sha256
from .runtime_memory_guard import TrainingDeviceMemoryGuard


_PLATFORM_FINGERPRINT_FIELDS = (
    "parent_model_fingerprint_sha256",
    "parent_controller_fingerprint_sha256",
    "platform_adapter_fingerprint_sha256",
)
_RL_MODEL_FINGERPRINT_FIELD = "rl_model_fingerprint_sha256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_sha256(config: HPCConfig) -> str:
    return hashlib.sha256(config.canonical_json().encode("utf-8")).hexdigest()


def _capacity_shards_are_valid(
    value: Any,
    *,
    expected_world_counts: tuple[int, ...],
    expected_task: str,
    expected_root_height_m: float,
    expected_manifest: list[dict[str, Any]] | None = None,
) -> bool:
    try:
        selected_task = contract.validate_task(expected_task)
    except ValueError:
        return False
    selected_task_index = contract.TASKS.index(selected_task)
    if (
        not isinstance(value, list)
        or len(value) != contract.OBJECT_SHARD_COUNT
        or len(expected_world_counts) != contract.OBJECT_SHARD_COUNT
    ):
        return False
    fields = (
        "object_index",
        "object",
        "task",
        "task_index",
        "root_height_m",
        "model_xml_sha256",
        "nq",
        "nv",
        "nu",
        "worlds",
    )
    for index, (entry, worlds) in enumerate(
        zip(value, expected_world_counts)
    ):
        if not isinstance(entry, Mapping):
            return False
        nq = entry.get("nq")
        nv = entry.get("nv")
        nu = entry.get("nu")
        if not (
            entry.get("object_index") == index
            and entry.get("object") == contract.OBJECTS[index]
            and entry.get("task") == selected_task
            and entry.get("task_index") == selected_task_index
            and isinstance(entry.get("root_height_m"), (int, float))
            and not isinstance(entry.get("root_height_m"), bool)
            and math.isclose(
                float(entry["root_height_m"]),
                float(expected_root_height_m),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and entry.get("worlds") == int(worlds)
            and _is_sha256(entry.get("model_xml_sha256"))
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in (nq, nv, nu)
            )
            and 0 < nv <= contract.FORMAL_DENSE_NEWTON_MAX_NV
        ):
            return False
        if expected_manifest is not None and any(
            entry.get(name) != expected_manifest[index].get(name)
            for name in fields
        ):
            return False
    return True


def _capacity_config_sha256(task: str, device: str) -> str:
    selected_task = contract.validate_task(task)
    return _canonical_sha256(
        {
            str(candidate): replace(
                DEFAULT_CONFIG,
                task=selected_task,
                num_envs=int(candidate),
                device=str(device),
            ).canonical_json()
            for candidate in FORMAL_WORLD_CANDIDATES
        }
    )


def _controller_counters_are_formally_valid(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(_CONTROLLER_COUNTER_FIELDS)
        and all(
            isinstance(value[name], int)
            and not isinstance(value[name], bool)
            and value[name] >= 0
            for name in _CONTROLLER_COUNTER_FIELDS
        )
        and all(
            value[name] == 0
            for name in _CONTROLLER_HARD_FAILURE_FIELDS
        )
    )


def _rotation_goal_coverage_is_valid(
    value: Any,
    phase: str,
    *,
    task: str,
    expected_world_counts: tuple[int, ...],
) -> bool:
    return _capacity_rotation_coverage_is_valid(
        value,
        phase,
        task=task,
        expected_world_counts=expected_world_counts,
    )


def _native_stage_uses_expected_jacobian(report: Mapping[str, Any]) -> bool:
    # Keep formal-training admission bound to the canonical capacity-report
    # validator.  Duplicating this protocol here previously let lifecycle
    # fields drift when task-specific episode lengths changed.
    return _capacity_native_stage_uses_expected_jacobian(report)



def _capacity_candidate_passed(
    report: Mapping[str, Any], candidate: int, *, task: str
) -> bool:
    try:
        selected_task = contract.validate_task(task)
    except ValueError:
        return False
    report_root_height_m = report.get("root_height_m")
    if not (
        isinstance(report_root_height_m, (int, float))
        and not isinstance(report_root_height_m, bool)
        and math.isfinite(float(report_root_height_m))
        and report.get("task") == selected_task
        and report.get("task_index") == contract.TASKS.index(selected_task)
        and _is_sha256(report.get(_RL_MODEL_FINGERPRINT_FIELD))
    ):
        return False
    report_gpu = report.get("gpu")
    expected_total_bytes = (
        report_gpu.get("memory_bytes")
        if isinstance(report_gpu, Mapping)
        else None
    )
    if not (
        isinstance(expected_total_bytes, int)
        and not isinstance(expected_total_bytes, bool)
        and expected_total_bytes > 0
    ):
        return False
    if report.get("runtime_errors") != []:
        return False
    passing_values = report.get("passing_candidates")
    if not isinstance(passing_values, list) or any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in passing_values
    ):
        return False
    passing = set(passing_values)
    if (
        len(passing) != len(passing_values)
        or int(candidate) not in passing
        or report.get("selected_num_envs") != max(passing, default=None)
    ):
        return False
    results = report.get("results")
    if not isinstance(results, list) or len(results) != len(
        FORMAL_WORLD_CANDIDATES
    ):
        return False
    if any(not isinstance(result, Mapping) for result in results):
        return False
    result_ids = [result.get("num_envs") for result in results]
    if (
        len(set(result_ids)) != len(result_ids)
        or set(result_ids) != set(FORMAL_WORLD_CANDIDATES)
    ):
        return False
    matches = [
        result for result in results if result.get("num_envs") == int(candidate)
    ]
    if len(matches) != 1:
        return False
    result = matches[0]
    device = report.get("device")
    if not isinstance(device, str) or not device:
        return False
    expected_config = replace(
        DEFAULT_CONFIG,
        task=selected_task,
        num_envs=int(candidate),
        device=device,
    )
    top_manifest = report.get("model_shards")
    if not _capacity_shards_are_valid(
        top_manifest,
        expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
        expected_task=selected_task,
        expected_root_height_m=float(report_root_height_m),
    ):
        return False
    stages = result.get("stages")
    runtime = result.get("runtime")
    expected_bindings = {
        "num_envs": int(candidate),
        "platform": report.get("platform"),
        "task": selected_task,
        "task_index": contract.TASKS.index(selected_task),
        "root_height_m": float(report_root_height_m),
        "algorithm_signature": contract.ALGORITHM_SIGNATURE,
        "runtime_code_fingerprint_sha256": report.get(
            "runtime_code_fingerprint_sha256"
        ),
        "physics_layout": contract.PHYSICS_LAYOUT,
        "object_world_counts": list(expected_config.object_world_counts),
        "remainder_worlds": expected_config.remainder_worlds,
        "remainder_object_world_counts": list(
            expected_config.remainder_object_world_counts
        ),
        "observation_dim": OBSERVATION_DIM,
        "action_dim": contract.ACTION_DIM,
        "episode_control_ticks": DEFAULT_CONFIG.episode_length,
        "candidate_config_canonical_sha256": _config_sha256(
            expected_config
        ),
        "gpu": report_gpu,
        _RL_MODEL_FINGERPRINT_FIELD: report.get(
            _RL_MODEL_FINGERPRINT_FIELD
        ),
        "diagnostic_only": False,
        "formal_training_eligible": True,
        "formal_training_ineligibility_reason": None,
    }
    return (
        all(result.get(name) == value for name, value in expected_bindings.items())
        and result.get("passed") is True
        and result.get("worker_reported_passed") is True
        and result.get("error") is None
        and result.get("runtime_errors") == []
        and result.get("validation_errors") == []
        and result.get("returncode") == 0
        and isinstance(runtime, Mapping)
        and set(runtime) == {"torch", "warp"}
        and all(isinstance(runtime[name], Mapping) for name in runtime)
        and isinstance(stages, Mapping)
        and set(stages) == {"native_search", "diffusion"}
        and all(
            stage_has_actual_memory_certificate(
                stages[name],
                expected_stage=name,
                expected_total_bytes=expected_total_bytes,
            )
            for name in ("native_search", "diffusion")
        )
        and _capacity_shards_are_valid(
            result.get("model_shards"),
            expected_world_counts=expected_config.object_world_counts,
            expected_task=selected_task,
            expected_root_height_m=float(report_root_height_m),
            expected_manifest=top_manifest,
        )
        and _native_stage_uses_expected_jacobian(result)
        and all(
            _is_sha256(report.get(name))
            and result.get(name) == report.get(name)
            for name in _PLATFORM_FINGERPRINT_FIELDS
        )
    )


def _validate_capacity_selection(
    *,
    report_path: Path,
    config: HPCConfig,
    runtime: Mapping[str, Any],
    platform_spec: Any,
) -> dict[str, Any]:
    """Fail closed unless one hash-bound G6-only report selects config."""

    selected_task = contract.validate_task(config.task)
    expected_task_index = contract.TASKS.index(selected_task)
    platform_root_height_m = float(
        getattr(platform_spec, "root_height_m", float("nan"))
    )
    if (
        getattr(platform_spec, "platform", None) != "gripper6"
        or getattr(platform_spec, "task", None) != selected_task
        or int(getattr(platform_spec, "task_index", -1))
        != expected_task_index
        or not math.isfinite(platform_root_height_m)
        or not math.isclose(
            platform_root_height_m, 0.56, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        raise RuntimeError("training G6 task/root binding mismatch")

    base_config = replace(DEFAULT_CONFIG, task=selected_task)
    resolved = report_path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(
            f"G6 capacity report is missing or not a regular file: {resolved}"
        )
    report_bytes = resolved.read_bytes()
    report = json.loads(report_bytes.decode("utf-8"))
    if not isinstance(report, Mapping):
        raise RuntimeError("G6 capacity report is not a JSON object")
    expected_gpu = {
        "name": runtime["torch"].get("name"),
        "memory_bytes": runtime["torch"].get("memory_bytes"),
        "compute_capability": runtime["torch"].get("compute_capability"),
    }
    expected = {
        "schema": CAPACITY_REPORT_SCHEMA,
        "platform": "gripper6",
        "platforms": ["gripper6"],
        "task": selected_task,
        "task_index": expected_task_index,
        "root_height_m": 0.56,
        "platform_root_heights_m": {"gripper6": 0.56},
        "algorithm_signature": contract.ALGORITHM_SIGNATURE,
        "runtime_code_fingerprint_sha256": (
            runtime_code_fingerprint_sha256()
        ),
        "physics_layout": contract.PHYSICS_LAYOUT,
        "config_canonical_sha256": _config_sha256(base_config),
        "candidate_config_set_sha256": _capacity_config_sha256(
            selected_task, config.device
        ),
        "observation_dim": OBSERVATION_DIM,
        "action_dim": contract.ACTION_DIM,
        "episode_control_ticks": config.episode_length,
        "device": config.device,
        "gpu": expected_gpu,
        "candidates": list(FORMAL_WORLD_CANDIDATES),
        "selected_num_envs": config.num_envs,
        "object_world_counts": list(config.object_world_counts),
        "remainder_worlds": config.remainder_worlds,
        "remainder_object_world_counts": list(
            config.remainder_object_world_counts
        ),
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise RuntimeError(
                f"G6 capacity report binding mismatch for {name}: "
                f"expected {value!r}, found {report.get(name)!r}"
            )

    rl_fingerprints = report.get(
        "platform_rl_model_fingerprints_sha256"
    )
    expected_rl_fingerprint = str(
        platform_spec.rl_model_fingerprint_sha256
    )
    if (
        not isinstance(rl_fingerprints, Mapping)
        or dict(rl_fingerprints)
        != {"gripper6": expected_rl_fingerprint}
        or not _is_sha256(expected_rl_fingerprint)
        or report.get(_RL_MODEL_FINGERPRINT_FIELD)
        != expected_rl_fingerprint
    ):
        raise RuntimeError("G6 capacity report RL model pins are invalid")
    for name in _PLATFORM_FINGERPRINT_FIELDS:
        expected_fingerprint = str(getattr(platform_spec, name))
        if (
            not _is_sha256(expected_fingerprint)
            or report.get(name) != expected_fingerprint
        ):
            raise RuntimeError(
                f"G6 capacity report platform mismatch for {name}"
            )

    candidate_hashes = report.get("candidate_config_sha256")
    if not isinstance(candidate_hashes, Mapping) or candidate_hashes.get(
        str(config.num_envs)
    ) != _config_sha256(config):
        raise RuntimeError("G6 capacity report selected-config hash mismatch")
    passing_values = report.get("passing_candidates")
    if not isinstance(passing_values, list) or any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in passing_values
    ):
        raise RuntimeError("G6 capacity report passing candidates are invalid")
    passing = set(passing_values)
    if len(passing) != len(passing_values) or (
        config.num_envs not in passing
        or config.num_envs != max(passing, default=-1)
    ):
        raise RuntimeError(
            "G6 capacity report did not select its largest passing worlds"
        )

    expected_shards = ordered_shard_manifest(
        platform_spec, config.object_world_counts
    )
    if not _capacity_shards_are_valid(
        report.get("model_shards"),
        expected_world_counts=config.object_world_counts,
        expected_task=selected_task,
        expected_root_height_m=0.56,
        expected_manifest=expected_shards,
    ):
        raise RuntimeError(
            "G6 capacity report does not bind the four model shards"
        )
    if not _capacity_candidate_passed(
        report, config.num_envs, task=selected_task
    ):
        raise RuntimeError(
            f"gripper6 did not pass {config.num_envs} worlds"
        )
    results = report.get("results")
    if not isinstance(results, list):
        raise RuntimeError("G6 capacity report has no worker results")
    current_workers = [
        result
        for result in results
        if isinstance(result, Mapping)
        and result.get("num_envs") == config.num_envs
    ]
    if len(current_workers) != 1:
        raise RuntimeError("G6 capacity report has no unique selected worker")
    certified_increment_by_stage = {
        name: int(
            current_workers[0]["stages"][name]["actual_device_memory"][
                "stage_peak_increment_bytes"
            ]
        )
        for name in ("native_search", "diffusion")
    }
    return {
        "report_path": str(resolved),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "platforms": ["gripper6"],
        "task": selected_task,
        "task_index": expected_task_index,
        "root_height_m": platform_root_height_m,
        "platform_root_heights_m": {"gripper6": 0.56},
        "platform_rl_model_fingerprints_sha256": {
            "gripper6": expected_rl_fingerprint
        },
        "gpu": expected_gpu,
        "selected_num_envs": config.num_envs,
        "object_world_counts": list(config.object_world_counts),
        "remainder_worlds": config.remainder_worlds,
        "remainder_object_world_counts": list(
            config.remainder_object_world_counts
        ),
        "model_shards": expected_shards,
        "passing_candidates": sorted(passing),
        "selection_rule": report.get("selection_rule"),
        "config_canonical_sha256": report["config_canonical_sha256"],
        "candidate_config_set_sha256": report[
            "candidate_config_set_sha256"
        ],
        "selected_config_sha256": _config_sha256(config),
        "certified_stage_peak_increment_by_stage": (
            certified_increment_by_stage
        ),
        "runtime_code_fingerprint_sha256": (
            runtime_code_fingerprint_sha256()
        ),
        **{
            name: str(getattr(platform_spec, name))
            for name in _PLATFORM_FINGERPRINT_FIELDS
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.ndim == 0 else tensor.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(torch: Any, payload: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class ProgressLogger:
    """Append JSONL and maintain an atomically readable live status snapshot."""

    def __init__(self, output_dir: Path, *, total_steps: int) -> None:
        self.path = output_dir / "training.jsonl"
        self.status_path = output_dir / "run_status.json"
        self.started = time.perf_counter()
        self.total_steps = int(total_steps)
        self.sequence = 0
        self.last_total_steps = 0

    def write(self, **record: Any) -> dict[str, Any]:
        self.sequence += 1
        elapsed = time.perf_counter() - self.started
        total_steps = int(record.get("total_steps", 0))
        self.last_total_steps = total_steps
        throughput = total_steps / elapsed if elapsed > 0.0 else None
        eta = (
            (self.total_steps - total_steps) / throughput
            if throughput and total_steps < self.total_steps
            else 0.0
            if total_steps >= self.total_steps
            else None
        )
        payload = {
            "schema": "grab-diffusion-live-v1",
            "sequence": self.sequence,
            "state": (
                "complete"
                if record.get("stage") == "complete"
                else "failed"
                if record.get("stage") == "failed"
                else "running"
            ),
            "target_steps": self.total_steps,
            "elapsed_s": elapsed,
            "throughput_transitions_s": throughput,
            "eta_search_s": eta,
            **_jsonable(record),
        }
        line = json.dumps(payload, sort_keys=True, allow_nan=False)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        _atomic_json(payload, self.status_path)
        return payload


def _configure_torch(config: HPCConfig) -> Any:
    configure_cublas_workspace(deterministic=config.deterministic_torch)
    import torch

    torch.cuda.set_device(torch.device(config.device))
    seed_everything(config.seed, deterministic=config.deterministic_torch)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch


def _fraction_compliance_by_object(
    info: Mapping[str, Any],
    fraction: Any,
    torch: Any,
) -> Any:
    """Fail-closed per-object compliance fallback for legacy telemetry."""

    value = torch.as_tensor(fraction)
    object_index = info.get("object_index")
    if object_index is None:
        return torch.zeros_like(value, dtype=torch.bool)
    indices = torch.as_tensor(
        object_index, device=value.device, dtype=torch.long
    )
    if indices.shape != value.shape:
        return torch.zeros_like(value, dtype=torch.bool)
    compliant = torch.zeros_like(value, dtype=torch.bool)
    for index, object_kind in enumerate(contract.OBJECTS):
        low, high = contract.grasp_plane_fraction_limits(object_kind)
        tolerance = 2.0e-6 if low == high else 0.0
        selected = indices == index
        valid = (
            torch.isfinite(value)
            & (value >= float(low) - tolerance)
            & (value <= float(high) + tolerance)
        )
        compliant |= selected & valid
    return compliant


def _plane_compliance(info: Mapping[str, Any], torch: Any) -> Any:
    for key in (
        "actual_grasp_plane_compliant",
        "actual_grasp_plane_within_bounds",
    ):
        if key in info:
            return torch.as_tensor(info[key], dtype=torch.bool)
    fraction = info.get("actual_grasp_plane_fraction")
    if fraction is None:
        fraction = info.get("grasp_plane_fraction")
    if fraction is None:
        raise KeyError("environment info has no grasp-plane compliance telemetry")
    return _fraction_compliance_by_object(info, fraction, torch)


def _commanded_plane_compliance(
    info: Mapping[str, Any], torch: Any
) -> Any:
    for key in (
        "commanded_grasp_plane_compliant",
        "grasp_plane_compliant",
        "grasp_plane_within_bounds",
    ):
        if key in info:
            return torch.as_tensor(info[key], dtype=torch.bool)
    fraction = info.get(
        "commanded_grasp_plane_fraction",
        info.get("grasp_plane_fraction"),
    )
    if fraction is None:
        raise KeyError("environment info has no commanded plane telemetry")
    return _fraction_compliance_by_object(info, fraction, torch)


def _actual_plane_relevant(info: Mapping[str, Any], torch: Any) -> Any:
    value = info.get("actual_grasp_plane_relevant")
    if value is None:
        reference = _plane_compliance(info, torch)
        relevant = torch.ones_like(reference, dtype=torch.bool)
    else:
        relevant = torch.as_tensor(value, dtype=torch.bool)
    dropped = info.get("object_dropped")
    if dropped is None:
        return relevant
    dropped_value = torch.as_tensor(
        dropped, device=relevant.device, dtype=torch.bool
    )
    if dropped_value.shape != relevant.shape:
        return torch.zeros_like(relevant, dtype=torch.bool)
    return relevant & (~dropped_value)


def _episode_freeze_certificate(
    trajectory_observation: Any,
    trajectory_action: Any,
    info: Mapping[str, Any],
    frozen_episode_action: Any,
    frozen_planner_grasp_plane_yaw_rad: Any,
    *,
    expected_task_index: int,
    torch: Any,
) -> dict[str, Any]:
    """Certify immutable path, grasp-plane command, and hard-tip anchors.

    The endpoint success flag alone is insufficient demonstration provenance.
    This certificate binds every stored trajectory to the task/path fixed by
    reset, the episode-level grasp-plane action fixed by the expert, and the
    four object-local hard-tip anchors captured before action motion begins.
    """

    observation = torch.as_tensor(trajectory_observation)
    action = torch.as_tensor(
        trajectory_action,
        device=observation.device,
        dtype=torch.float32,
    )
    episode_action = torch.as_tensor(
        frozen_episode_action,
        device=observation.device,
        dtype=torch.float32,
    )
    planner_yaw = torch.as_tensor(
        frozen_planner_grasp_plane_yaw_rad,
        device=observation.device,
        dtype=torch.float32,
    )
    if observation.ndim != 3 or observation.shape[-1] != OBSERVATION_DIM:
        raise ValueError("trajectory_observation has an invalid shape")
    worlds = int(observation.shape[0])
    if tuple(action.shape) != (
        worlds,
        int(observation.shape[1]) - 1,
        contract.ACTION_DIM,
    ):
        raise ValueError("trajectory_action has an invalid shape")
    if tuple(episode_action.shape) != (worlds, contract.ACTION_DIM):
        raise ValueError("frozen_episode_action has an invalid shape")
    if tuple(planner_yaw.shape) != (worlds,):
        raise ValueError("frozen planner grasp-plane yaw has an invalid shape")

    path_slice = OBSERVATION_SLICES["translation_path_waypoints_xy"]
    path_history = observation[:, :, path_slice].to(torch.float32)
    path_reference = path_history[:, :1]
    path_frozen = torch.isfinite(path_history).all(dim=(1, 2)) & torch.isclose(
        path_history,
        path_reference,
        rtol=0.0,
        atol=2.0e-3,
    ).all(dim=(1, 2))
    endpoint_path = torch.as_tensor(
        info["path_waypoints_xy_m"],
        device=observation.device,
        dtype=torch.float32,
    )
    expected_path_shape = (
        worlds,
        contract.TRANSLATION_PATH_WAYPOINTS,
        2,
    )
    if tuple(endpoint_path.shape) != expected_path_shape:
        raise ValueError("endpoint path_waypoints_xy_m has an invalid shape")
    normalized_endpoint_path = (
        endpoint_path / contract.TRANSLATION_PATH_REGION_RADIUS_M
    ).reshape(worlds, -1)
    path_matches_endpoint = torch.isclose(
        path_history[:, -1],
        normalized_endpoint_path,
        rtol=0.0,
        atol=2.0e-3,
    ).all(dim=1)
    path_frozen &= path_matches_endpoint

    episode_indices = torch.tensor(
        contract.EPISODE_LEVEL_ACTION_INDICES,
        device=observation.device,
        dtype=torch.long,
    )
    episode_history = action.index_select(2, episode_indices)
    episode_reference = episode_action.index_select(1, episode_indices)
    phase_slice = OBSERVATION_SLICES["phase_one_hot"]
    plan_tick = observation[:, :-1, phase_slice.start].to(torch.float32) > 0.5
    plan_equal = torch.isclose(
        episode_history,
        episode_reference[:, None, :],
        rtol=0.0,
        atol=2.0e-3,
    )
    plan_equal = torch.where(
        plan_tick[:, :, None],
        plan_equal,
        torch.ones_like(plan_equal),
    )
    action_start_slice = OBSERVATION_SLICES["action_start_action"]
    action_start = observation[:, -1, action_start_slice].to(torch.float32)
    frozen_action_start = action_start.index_select(1, episode_indices)
    plane_action_frozen = (
        torch.isfinite(episode_history).all(dim=(1, 2))
        & torch.isfinite(episode_reference).all(dim=1)
        & plan_tick.any(dim=1)
        & plan_equal.all(dim=(1, 2))
        & torch.isclose(
            frozen_action_start,
            episode_reference,
            rtol=0.0,
            atol=2.0e-3,
        ).all(dim=1)
    )

    reference_valid_slice = OBSERVATION_SLICES["action_reference_valid"]
    reference_valid = (
        observation[:, :, reference_valid_slice].to(torch.float32).squeeze(-1)
        > 0.5
    )
    anchor_slice = OBSERVATION_SLICES["action_start_cage_object_local"]
    anchor_history = observation[:, :, anchor_slice].to(torch.float32)
    frozen_anchors = anchor_history[:, -1]
    active_anchor_equal = torch.isclose(
        anchor_history,
        frozen_anchors[:, None, :],
        rtol=0.0,
        atol=2.0e-4,
    )
    active_anchor_equal = torch.where(
        reference_valid[:, :, None],
        active_anchor_equal,
        torch.ones_like(active_anchor_equal),
    )
    anchors_frozen = (
        reference_valid[:, -1]
        & torch.isfinite(frozen_anchors).all(dim=1)
        & active_anchor_equal.all(dim=(1, 2))
    )

    task_index = torch.as_tensor(
        info["task_index"],
        device=observation.device,
        dtype=torch.long,
    )
    if tuple(task_index.shape) != (worlds,):
        raise ValueError("endpoint task_index has an invalid shape")
    task_frozen = task_index == int(expected_task_index)
    frozen = path_frozen & plane_action_frozen & anchors_frozen & task_frozen

    tip_height_fraction = torch.as_tensor(
        info["requested_tip_height_fraction"],
        device=observation.device,
        dtype=torch.float32,
    )
    grasp_plane_fraction = torch.as_tensor(
        info["grasp_plane_fraction"],
        device=observation.device,
        dtype=torch.float32,
    )
    if tuple(tip_height_fraction.shape) != (
        worlds,
        len(contract.TIP_ORDER),
    ):
        raise ValueError("requested_tip_height_fraction has an invalid shape")
    if tuple(grasp_plane_fraction.shape) != (worlds,):
        raise ValueError("grasp_plane_fraction has an invalid shape")
    artifacts_finite = (
        torch.isfinite(endpoint_path).all(dim=(1, 2))
        & torch.isfinite(tip_height_fraction).all(dim=1)
        & torch.isfinite(grasp_plane_fraction)
        & torch.isfinite(planner_yaw)
    )
    frozen &= artifacts_finite
    return {
        "path_frozen": path_frozen,
        "plane_action_frozen": plane_action_frozen,
        "anchors_frozen": anchors_frozen,
        "task_frozen": task_frozen,
        "trajectory_frozen": frozen,
        "frozen_path_waypoints_xy_m": endpoint_path,
        "frozen_episode_action": episode_action,
        "frozen_tip_height_fraction": tip_height_fraction,
        "frozen_grasp_plane_fraction": grasp_plane_fraction,
        "frozen_grasp_plane_yaw_rad": planner_yaw,
        "frozen_hard_tip_anchors_object_local_m": frozen_anchors.reshape(
            worlds, len(contract.TIP_ORDER), 3
        ),
    }


def _unexpected_early_done(transition: Any, torch: Any) -> Any:
    """Return terminals that are neither horizon nor explicit finite drops."""

    done = transition.done.to(torch.bool)
    dropped = transition.info.get("object_dropped")
    if dropped is None:
        dropped = torch.zeros_like(done)
    else:
        dropped = torch.as_tensor(
            dropped, dtype=torch.bool, device=done.device
        )
    time_out = transition.info.get("time_out")
    if time_out is None:
        time_out = torch.zeros_like(done)
    else:
        time_out = torch.as_tensor(
            time_out, dtype=torch.bool, device=done.device
        )
    transition_finite = transition.info.get("transition_finite")
    if transition_finite is None:
        transition_finite = torch.ones_like(done)
    else:
        transition_finite = torch.as_tensor(
            transition_finite, dtype=torch.bool, device=done.device
        )
    return (done & (~dropped) & (~time_out)) | (~transition_finite)


def _success_breakdown(
    success: Any,
    object_index: Any,
    task_index: Any,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    by_object: dict[str, dict[str, int]] = {}
    by_task: dict[str, dict[str, int]] = {}
    for index, name in enumerate(contract.OBJECTS):
        selected = object_index == index
        by_object[name] = {
            "success_count": int(success[selected].sum().item()),
            "episode_count": int(selected.sum().item()),
        }
    for index, name in enumerate(contract.TASKS):
        selected = task_index == index
        by_task[name] = {
            "success_count": int(success[selected].sum().item()),
            "episode_count": int(selected.sum().item()),
        }
    return by_object, by_task


def _zero_breakdown(names: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {
        name: {"success_count": 0, "episode_count": 0}
        for name in names
    }


def _merge_breakdown(
    target: dict[str, dict[str, int]],
    source: Mapping[str, Mapping[str, int]],
) -> None:
    for name, counts in source.items():
        entry = target.setdefault(
            str(name), {"success_count": 0, "episode_count": 0}
        )
        entry["success_count"] += int(counts["success_count"])
        entry["episode_count"] += int(counts["episode_count"])


def _zero_rotation_success_breakdown(
    phase: str,
) -> dict[str, dict[str, int]]:
    low, high = rotation_angle_bounds_deg(phase)
    return {
        str(angle): {"success_count": 0, "episode_count": 0}
        for angle in range(low, high + 1)
    }


def _rotation_success_breakdown(
    success: Any,
    task_index: Any,
    rotation_angle_deg: Any,
    *,
    phase: str,
) -> dict[str, dict[str, int]]:
    """Count strict endpoint success for every requested rotation angle."""

    low, high = rotation_angle_bounds_deg(phase)
    rotating = task_index == contract.TASKS.index(contract.TASK_ROTATE)
    angles = rotation_angle_deg[rotating].to(dtype=task_index.dtype)
    if int(angles.numel()) == 0 or bool(
        ((angles < low) | (angles > high)).any()
    ):
        raise RuntimeError("endpoint rotation targets violate the curriculum")
    relative = angles - low
    width = high - low + 1
    episode_counts = relative.bincount(minlength=width)
    successful_counts = relative[
        success[rotating].bool()
    ].bincount(minlength=width)
    episodes = episode_counts.detach().cpu().tolist()
    successes = successful_counts.detach().cpu().tolist()
    return {
        str(low + offset): {
            "success_count": int(successes[offset]),
            "episode_count": int(episodes[offset]),
        }
        for offset in range(width)
    }


def _success_breakdown_rates(
    breakdown: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int | float | None]]:
    return {
        str(name): {
            "success_count": int(counts["success_count"]),
            "episode_count": int(counts["episode_count"]),
            "success_rate": (
                int(counts["success_count"]) / int(counts["episode_count"])
                if int(counts["episode_count"])
                else None
            ),
        }
        for name, counts in breakdown.items()
    }


def _drop_breakdown(
    dropped: Any,
    object_index: Any,
    task_index: Any,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    def rows(names: tuple[str, ...], indices: Any) -> dict[str, dict[str, int]]:
        return {
            name: {
                "drop_count": int(dropped[indices == index].sum().item()),
                "episode_count": int((indices == index).sum().item()),
            }
            for index, name in enumerate(names)
        }

    return rows(contract.OBJECTS, object_index), rows(contract.TASKS, task_index)


def _zero_drop_breakdown(names: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {
        name: {"drop_count": 0, "episode_count": 0}
        for name in names
    }


def _merge_drop_breakdown(
    target: dict[str, dict[str, int]],
    source: Mapping[str, Mapping[str, int]],
) -> None:
    for name, counts in source.items():
        entry = target.setdefault(
            str(name), {"drop_count": 0, "episode_count": 0}
        )
        entry["drop_count"] += int(counts["drop_count"])
        entry["episode_count"] += int(counts["episode_count"])


def _drop_breakdown_rates(
    breakdown: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int | float | None]]:
    return {
        name: {
            "drop_count": int(counts["drop_count"]),
            "episode_count": int(counts["episode_count"]),
            "drop_rate": (
                int(counts["drop_count"]) / int(counts["episode_count"])
                if int(counts["episode_count"])
                else None
            ),
        }
        for name, counts in breakdown.items()
    }


def _gpu_metrics(torch: Any, device: str) -> dict[str, float]:
    index = torch.device(device)
    return {
        "gpu_memory_allocated_gb": (
            torch.cuda.memory_allocated(index) / (1024.0**3)
        ),
        "gpu_memory_reserved_gb": (
            torch.cuda.memory_reserved(index) / (1024.0**3)
        ),
    }


_STICKY_HEALTH_FAILURES = (
    "sticky_nonfinite",
    "sticky_overflow",
    "sticky_contacts_full",
    "sticky_constraints_full",
    "sticky_external_nonfinite",
)


def _assert_backend_health(
    health: Mapping[str, Any],
    *,
    context: str,
    expected_world_counts: tuple[int, ...] | None = None,
) -> None:
    failed = [
        name for name in _STICKY_HEALTH_FAILURES if health.get(name, False)
    ]
    if health.get("solver_option") != 2:
        failed.append("newton_solver_option_not_certified")
    if health.get("solver_name") != contract.FORMAL_SOLVER:
        failed.append("newton_solver_name_not_certified")
    if (
        health.get("requested_solver_tolerance")
        != contract.FORMAL_XML_SOLVER_TOLERANCE
    ):
        failed.append("xml_solver_tolerance_not_certified")
    if (
        health.get("effective_solver_tolerance")
        != contract.FORMAL_EFFECTIVE_SOLVER_TOLERANCE
    ):
        failed.append("effective_solver_tolerance_not_certified")
    if (
        health.get("jacobian_option") != 0
        or health.get("resolved_sparse_jacobian") is not False
    ):
        failed.append("explicit_dense_jacobian_not_certified")
    model_nv = health.get("model_nv")
    if (
        not isinstance(model_nv, int)
        or isinstance(model_nv, bool)
        or not 0 < model_nv <= contract.FORMAL_DENSE_NEWTON_MAX_NV
    ):
        failed.append("dense_newton_model_nv_not_certified")
    if not formal_sharded_health_is_valid(
        health,
        expected_world_counts=(
            expected_world_counts
            if expected_world_counts is not None
            else (
                contract.FORMAL_OBJECT_WORLD_COUNTS
                if int(health.get("worlds", -1))
                == contract.FORMAL_GLOBAL_WORLDS
                else None
            )
        ),
    ):
        failed.append("four_object_shards_not_certified")
    if any(
        not _is_sha256(health.get(name))
        for name in _PLATFORM_FINGERPRINT_FIELDS
    ):
        failed.append("platform_fingerprint_not_certified")
    if not health.get("finite", False) or failed:
        detail = ", ".join(failed) if failed else "finite=false"
        raise RuntimeError(f"{context} health failed ({detail}): {health}")


_CONTROLLER_COUNTER_FIELDS = (
    "ik_solve_fallback_count",
    "ik_trust_region_projection_count",
    "position_feedback_slew_clip_count",
    "preload_trim_slew_clip_count",
    "ctrl_slew_clip_count",
    "ctrl_certificate_fallback_count",
)
_CONTROLLER_HARD_FAILURE_FIELDS = (
    "ik_solve_fallback_count",
    "ctrl_certificate_fallback_count",
)


def _zero_controller_counters() -> dict[str, int]:
    return {name: 0 for name in _CONTROLLER_COUNTER_FIELDS}


def _controller_counter_totals(environment: Any) -> dict[str, int]:
    return {
        name: int(getattr(environment, name).sum().detach().cpu())
        for name in _CONTROLLER_COUNTER_FIELDS
    }


def _controller_counter_delta(
    current: Mapping[str, int], previous: Mapping[str, int]
) -> dict[str, int]:
    delta = {
        name: int(current[name]) - int(previous[name])
        for name in _CONTROLLER_COUNTER_FIELDS
    }
    if any(value < 0 for value in delta.values()):
        raise RuntimeError(
            "controller safety counters decreased without an episode reset"
        )
    return delta


def _add_controller_counters(
    target: dict[str, int], source: Mapping[str, int]
) -> None:
    for name in _CONTROLLER_COUNTER_FIELDS:
        target[name] += int(source[name])


def _assert_controller_safety(
    environment: Any,
    counters: Mapping[str, int],
    *,
    context: str,
) -> None:
    failures = {
        name: int(counters[name])
        for name in _CONTROLLER_HARD_FAILURE_FIELDS
        if int(counters[name])
    }
    if not failures:
        return
    import torch

    per_world_failure = torch.zeros_like(
        environment.ik_solve_fallback_count, dtype=torch.bool
    )
    for name in _CONTROLLER_HARD_FAILURE_FIELDS:
        per_world_failure |= getattr(environment, name) > 0
    bad = torch.nonzero(per_world_failure, as_tuple=False).flatten()
    first = int(bad[0].detach().cpu()) if int(bad.numel()) else -1
    diagnostic: dict[str, Any] = {
        "context": context,
        "failures": failures,
        "affected_worlds": int(per_world_failure.sum().detach().cpu()),
        "first_world": first,
    }
    if first >= 0:
        diagnostic.update(
            {
                "object_index": int(
                    environment.object_index[first].detach().cpu()
                ),
                "task_index": int(
                    environment.task_index[first].detach().cpu()
                ),
                "template_index": int(
                    environment.template_index[first].detach().cpu()
                ),
                "ik_solve_feasible": bool(
                    environment._last_ik_solve_feasible[first]
                    .detach()
                    .cpu()
                ),
                "world_counters": {
                    name: int(
                        getattr(environment, name)[first].detach().cpu()
                    )
                    for name in _CONTROLLER_COUNTER_FIELDS
                },
            }
        )
    raise RuntimeError(
        "controller safety certificate failed: "
        + json.dumps(diagnostic, sort_keys=True)
    )


def _release_cuda_stage_memory(torch: Any, device: str) -> None:
    """Release Torch, Warp mempool, and retired CUDA Graph allocations."""

    selected = torch.device(device)
    import warp as wp

    wp_device = wp.get_device(str(selected))
    wp.set_mempool_release_threshold(wp_device, 0)
    gc.collect()
    torch.cuda.synchronize(selected)
    torch.cuda.empty_cache()
    wp.synchronize_device(wp_device)
    time.sleep(0.1)
    wp.synchronize_device(wp_device)
    graph_mem_trim = getattr(
        wp._src.context.runtime.core,
        "wp_cuda_device_graph_mem_trim",
        None,
    )
    if not callable(graph_mem_trim):
        raise RuntimeError(
            "pinned Warp runtime lacks CUDA graph-memory trim support"
        )
    graph_mem_trim(wp_device.ordinal)
    wp.synchronize_device(wp_device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(selected)


def _consume_remainder_shard(
    *,
    phase: str,
    worlds: int,
    config: HPCConfig,
    platform_spec: Any,
    logger: ProgressLogger,
    transitions_before_phase: int,
    object_world_counts: tuple[int, ...],
    memory_guard: TrainingDeviceMemoryGuard,
) -> dict[str, Any] | None:
    """Consume the run's non-divisible tail without rounding the 90M budget."""

    if int(worlds) == 0:
        return None
    import torch

    environment = make_env(
        phase,
        config=config,
        platform_spec=platform_spec,
        worlds=int(worlds),
        object_world_counts=object_world_counts,
        device=config.device,
        require_linux=True,
    )
    try:
        observation = environment.reset()
        action = torch.zeros(
            (int(worlds), contract.ACTION_DIM),
            device=config.device,
            dtype=observation.dtype,
        )
        transition = environment.step(action)
        memory_guard.raise_if_violated(f"{phase}_remainder_transition")
        controller_counters = _controller_counter_totals(environment)
        try:
            _assert_controller_safety(
                environment,
                controller_counters,
                context=f"{phase} remainder",
            )
        except RuntimeError as exc:
            logger.write(
                stage="controller_safety_failure",
                phase=phase,
                phase_steps=int(worlds),
                total_steps=transitions_before_phase + int(worlds),
                controller_safety_counters=controller_counters,
                error=str(exc),
            )
            raise
        if not bool(transition.info["transition_finite"].all()):
            raise FloatingPointError(
                "remainder MJWarp transition failed its finite certificate"
            )
        compliance = _plane_compliance(transition.info, torch)
        relevant = _actual_plane_relevant(transition.info, torch)
        relevant_compliance = compliance[relevant]
        commanded_compliance = _commanded_plane_compliance(
            transition.info, torch
        )
        dropped = torch.as_tensor(
            transition.info.get(
                "object_dropped", torch.zeros_like(transition.done)
            ),
            device=commanded_compliance.device,
            dtype=torch.bool,
        )
        live_commanded_compliance = commanded_compliance[~dropped]
        health = environment.backend.health()
        _assert_backend_health(
            health,
            context="remainder MJWarp",
            expected_world_counts=tuple(environment.world_counts),
        )
        if bool(_unexpected_early_done(transition, torch).any()):
            raise RuntimeError(
                "remainder MJWarp world had an unexplained early terminal"
            )
        logger.write(
            stage="ik_expert_remainder",
            phase=phase,
            phase_steps=int(worlds),
            total_steps=transitions_before_phase + int(worlds),
            remainder_worlds=int(worlds),
            endpoint_success_rate=None,
            grasp_plane_compliance_rate=(
                relevant_compliance.to(torch.float32).mean()
                if int(relevant_compliance.numel())
                else None
            ),
            grasp_plane_sample_count=int(relevant_compliance.numel()),
            commanded_grasp_plane_compliance_rate=(
                live_commanded_compliance.to(torch.float32).mean()
                if int(live_commanded_compliance.numel())
                else None
            ),
            commanded_grasp_plane_sample_count=int(
                live_commanded_compliance.numel()
            ),
            position_error_m=transition.info["position_error_m"].mean(),
            euler_l2_error_deg=transition.info[
                "euler_l2_error_deg"
            ].mean(),
            reward_mean=transition.reward.mean(),
            object_drop_new_count=int(
                torch.as_tensor(
                    transition.info.get(
                        "object_drop_new", torch.zeros_like(transition.done)
                    ),
                    dtype=torch.bool,
                ).sum().item()
            ),
            controller_safety_counters=controller_counters,
            **_gpu_metrics(torch, config.device),
        )
        return {
            "backend_health": _jsonable(health),
            "object_world_counts": list(environment.world_counts),
            "controller_safety_counters": controller_counters,
            "object_drop_count": int(
                torch.as_tensor(
                    transition.info.get(
                        "object_dropped", torch.zeros_like(transition.done)
                    ),
                    dtype=torch.bool,
                ).sum().item()
            ),
        }
    finally:
        environment.close()


def _run_expert_task(
    *,
    phase: str,
    steps: int,
    transitions_before_run: int,
    config: HPCConfig,
    platform_spec: Any,
    teacher: MJWarpIKExpert,
    store: CertifiedIKDemoStore,
    logger: ProgressLogger,
    global_success: dict[str, Any],
    memory_guard: TrainingDeviceMemoryGuard,
) -> dict[str, Any]:
    import torch

    memory_guard.raise_if_violated(f"{phase}_expert_task_start")
    # No world-sized episode tensors may survive into a remainder shard.
    teacher.clear_episode_state()
    remainder_worlds = steps % config.num_envs
    remainder_object_world_counts = config.remainder_object_world_counts
    remainder_diagnostics = _consume_remainder_shard(
        phase=phase,
        worlds=remainder_worlds,
        config=config,
        platform_spec=platform_spec,
        logger=logger,
        transitions_before_phase=transitions_before_run,
        object_world_counts=remainder_object_world_counts,
        memory_guard=memory_guard,
    )
    if remainder_diagnostics is not None:
        # The formal capacity protocol certifies this exact boundary: destroy
        # the remainder systems and release both allocators before the
        # four full-batch systems are constructed.
        _release_cuda_stage_memory(torch, config.device)
    environment: Any | None = None
    trajectory_observation: Any | None = None
    trajectory_action: Any | None = None
    episode_return: Any | None = None
    goals: Any | None = None
    observation: Any | None = None
    requested_action: Any | None = None
    transition: Any | None = None
    next_observation: Any | None = None
    done: Any | None = None
    info: Any | None = None

    def cleanup_full_search_allocations() -> None:
        nonlocal environment
        nonlocal trajectory_observation, trajectory_action, episode_return
        nonlocal goals, observation, requested_action, transition
        nonlocal next_observation, done, info
        errors: list[str] = []

        def cleanup(label: str, operation: Any) -> None:
            try:
                operation()
            except BaseException as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")

        cleanup("teacher episode state", teacher.clear_episode_state)
        try:
            if environment is not None:
                cleanup("sharded environment", environment.close)
        finally:
            environment = None
            trajectory_observation = None
            trajectory_action = None
            episode_return = None
            goals = None
            observation = None
            requested_action = None
            transition = None
            next_observation = None
            done = None
            info = None
            cleanup(
                "CUDA allocator release",
                lambda: _release_cuda_stage_memory(torch, config.device),
            )
        if errors:
            raise RuntimeError(
                "native-search CUDA cleanup failed: " + "; ".join(errors)
            )

    # Start the lifecycle boundary before the first full-shard allocation.
    # This also covers OOM during environment, trajectory, or teacher state
    # construction rather than only errors after reset.
    try:
        environment = make_env(
            phase,
            config=config,
            platform_spec=platform_spec,
            worlds=config.num_envs,
            device=config.device,
            require_linux=True,
        )
        episode_length = config.episode_length
        control_ticks = steps // config.num_envs
        trajectory_observation = torch.empty(
            (
                config.num_envs,
                episode_length + 1,
                OBSERVATION_DIM,
            ),
            device=config.device,
            dtype=torch.float16,
        )
        trajectory_action = torch.empty(
            (config.num_envs, episode_length, contract.ACTION_DIM),
            device=config.device,
            dtype=torch.float16,
        )
        episode_return = torch.zeros(
            config.num_envs, device=config.device, dtype=torch.float32
        )
        episode_index = 0
        endpoint_success_count = 0
        endpoint_episode_count = 0
        interval_endpoint_success_count = 0
        interval_endpoint_episode_count = 0
        valid_group_count = 0
        valid_group_episode_count = 0
        interval_valid_group_count = 0
        interval_valid_group_episode_count = 0
        latest_endpoint_success_rate: float | None = None
        latest_translation_endpoint_metrics: dict[str, Any] = (
            {
                "translation_endpoint_desired_final_displacement_xyz_mean_m": None,
                "translation_endpoint_desired_final_displacement_l2_mean_m": None,
                "translation_endpoint_actual_final_displacement_xyz_mean_m": None,
                "translation_endpoint_actual_final_displacement_l2_mean_m": None,
                "translation_endpoint_final_displacement_error_xyz_mean_m": None,
                "translation_endpoint_final_displacement_error_l2_mean_m": None,
                "translation_endpoint_final_euler_error_xyz_mean_deg": None,
                "translation_endpoint_final_euler_error_l2_mean_deg": None,
                "translation_endpoint_path_tracking_rmse_mean_m": None,
                "translation_endpoint_path_tracking_max_mean_m": None,
                "translation_endpoint_path_tracking_max_m": None,
            }
            if config.task == contract.TASK_TRANSLATE
            else {}
        )
        certified_trajectory_count = 0
        remainder_object_drop_count = int(
            remainder_diagnostics.get("object_drop_count", 0)
            if remainder_diagnostics is not None
            else 0
        )
        object_drop_count = remainder_object_drop_count
        global_success["object_drop_count"] += remainder_object_drop_count
        interval_object_drop_count = 0
        endpoint_object_drop_count = 0
        plane_compliant_count = 0
        plane_sample_count = 0
        commanded_plane_compliant_count = 0
        commanded_plane_sample_count = 0
        interval_breakdown_object = _zero_breakdown(contract.OBJECTS)
        interval_breakdown_task = _zero_breakdown(contract.TASKS)
        interval_rotation_breakdown = _zero_rotation_success_breakdown(
            phase
        )
        phase_rotation_breakdown = _zero_rotation_success_breakdown(phase)
        interval_drop_by_object = _zero_drop_breakdown(contract.OBJECTS)
        interval_drop_by_task = _zero_drop_breakdown(contract.TASKS)
        phase_drop_by_object = _zero_drop_breakdown(contract.OBJECTS)
        phase_drop_by_task = _zero_drop_breakdown(contract.TASKS)
        phase_controller_counters = _zero_controller_counters()
        interval_controller_counters = _zero_controller_counters()
        previous_episode_controller_counters = _zero_controller_counters()
        if remainder_diagnostics is not None:
            remainder_counters = remainder_diagnostics[
                "controller_safety_counters"
            ]
            _add_controller_counters(
                phase_controller_counters, remainder_counters
            )
            _add_controller_counters(
                interval_controller_counters, remainder_counters
            )
            _add_controller_counters(
                global_success["controller_safety_counters"],
                remainder_counters,
            )
        goals = teacher.begin_episode(phase, episode_index)
        memory_guard.raise_if_violated(f"{phase}_full_allocation_complete")
    except BaseException:
        try:
            cleanup_full_search_allocations()
        except BaseException:
            # Preserve the allocation failure.  The outer memory guard still
            # produces a device-wide failure report.
            pass
        raise
    active_search_exception: BaseException | None = None
    try:
        observation = environment.reset(goals=goals)
        if config.cuda_graph:
            environment.backend.prepare_cuda_graph()
            observation = environment.reset(goals=goals)
        trajectory_observation[:, 0].copy_(observation.to(torch.float16))
        local_tick = 0
        last_status_time = time.monotonic()
        for control_tick in range(control_ticks):
            with torch.no_grad():
                requested_action = teacher.action(local_tick, observation)
                transition = environment.step(requested_action)
            memory_guard.raise_if_violated(
                f"{phase}_control_tick_{control_tick + 1}"
            )
            current_controller_counters = _controller_counter_totals(
                environment
            )
            counter_delta = _controller_counter_delta(
                current_controller_counters,
                previous_episode_controller_counters,
            )
            _add_controller_counters(phase_controller_counters, counter_delta)
            _add_controller_counters(
                interval_controller_counters, counter_delta
            )
            _add_controller_counters(
                global_success["controller_safety_counters"], counter_delta
            )
            previous_episode_controller_counters = (
                current_controller_counters
            )
            try:
                _assert_controller_safety(
                    environment,
                    current_controller_counters,
                    context=f"{phase} control tick {control_tick + 1}",
                )
            except RuntimeError as exc:
                logger.write(
                    stage="controller_safety_failure",
                    phase=phase,
                    phase_steps=(
                        remainder_worlds
                        + (control_tick + 1) * config.num_envs
                    ),
                    total_steps=(
                        transitions_before_run
                        + remainder_worlds
                        + (control_tick + 1) * config.num_envs
                    ),
                    control_tick=control_tick + 1,
                    controller_safety_counters=(
                        current_controller_counters
                    ),
                    error=str(exc),
                )
                raise
            if not bool(transition.info["transition_finite"].all()):
                raise FloatingPointError(
                    "native MJWarp transition failed its finite certificate "
                    f"at control tick {control_tick + 1}"
                )
            next_observation = transition.observation
            # Supervise the policy command, not the environment's filtered
            # and frozen action.  At deployment the predicted command passes
            # through that same environment filter exactly once.
            trajectory_action[:, local_tick].copy_(
                requested_action.to(torch.float16)
            )
            trajectory_observation[:, local_tick + 1].copy_(
                next_observation.to(torch.float16)
            )
            episode_return.add_(transition.reward)
            compliance = _plane_compliance(transition.info, torch).to(
                device=config.device
            )
            relevant = _actual_plane_relevant(
                transition.info, torch
            ).to(device=config.device)
            relevant_compliance = compliance[relevant]
            plane_compliant_count += int(
                relevant_compliance.sum().item()
            )
            plane_sample_count += int(relevant_compliance.numel())
            commanded_compliance = _commanded_plane_compliance(
                transition.info, torch
            ).to(device=config.device)
            dropped_now = transition.info["object_dropped"].to(torch.bool)
            live_commanded_compliance = commanded_compliance[~dropped_now]
            commanded_plane_compliant_count += int(
                live_commanded_compliance.sum().item()
            )
            commanded_plane_sample_count += int(
                live_commanded_compliance.numel()
            )
            new_drop_count = int(
                transition.info["object_drop_new"]
                .to(torch.bool)
                .sum()
                .item()
            )
            object_drop_count += new_drop_count
            interval_object_drop_count += new_drop_count
            global_success["object_drop_count"] += new_drop_count

            done = transition.done.to(torch.bool)
            expected_terminal = local_tick + 1 == episode_length
            unexplained_done = _unexpected_early_done(transition, torch)
            if bool(unexplained_done.any()) and not expected_terminal:
                raise RuntimeError(
                    "MJWarp world had a non-drop terminal before the fixed "
                    "episode endpoint"
                )
            if expected_terminal:
                if not bool(done.all()):
                    raise RuntimeError(
                        "synchronized IK-expert population did not terminate"
                    )
                _assert_backend_health(
                    environment.backend.health(),
                    context=(
                        f"native MJWarp {phase} endpoint "
                        f"episode {episode_index}"
                    ),
                    expected_world_counts=tuple(environment.world_counts),
                )
                info = transition.info
                success = info["success"].to(torch.bool)
                expected_task_index = contract.TASKS.index(config.task)
                endpoint_task_index = info["task_index"].to(torch.long)
                if not bool((endpoint_task_index == expected_task_index).all()):
                    raise RuntimeError(
                        "formal endpoint batch mixed tasks in a single-task run"
                    )
                if config.task == contract.TASK_TRANSLATE:
                    translation_endpoint = torch.ones_like(
                        success, dtype=torch.bool
                    )
                    latest_translation_endpoint_metrics = {
                        "translation_endpoint_desired_final_displacement_xyz_mean_m": (
                            info["desired_final_displacement_xyz_m"].mean(dim=0)
                        ),
                        "translation_endpoint_desired_final_displacement_l2_mean_m": (
                            info["desired_final_displacement_l2_m"].mean()
                        ),
                        "translation_endpoint_actual_final_displacement_xyz_mean_m": (
                            info["actual_final_displacement_xyz_m"].mean(dim=0)
                        ),
                        "translation_endpoint_actual_final_displacement_l2_mean_m": (
                            info["actual_final_displacement_l2_m"].mean()
                        ),
                        "translation_endpoint_final_displacement_error_xyz_mean_m": (
                            info["final_displacement_error_xyz_m"].mean(dim=0)
                        ),
                        "translation_endpoint_final_displacement_error_l2_mean_m": (
                            info["final_displacement_error_l2_m"].mean()
                        ),
                        "translation_endpoint_final_euler_error_xyz_mean_deg": (
                            info["final_euler_error_xyz_deg"].mean(dim=0)
                        ),
                        "translation_endpoint_final_euler_error_l2_mean_deg": (
                            info["final_euler_error_l2_deg"].mean()
                        ),
                        "translation_endpoint_path_tracking_rmse_mean_m": (
                            info["path_tracking_rmse_m"].mean()
                        ),
                        "translation_endpoint_path_tracking_max_mean_m": (
                            info["path_tracking_max_error_m"].mean()
                        ),
                        "translation_endpoint_path_tracking_max_m": (
                            info["path_tracking_max_error_m"].max()
                        ),
                    }
                dropped = info["object_dropped"].to(torch.bool)
                endpoint_object_drop_count += int(dropped.sum().item())
                global_success["completed_object_drop_count"] += int(
                    dropped.sum().item()
                )
                drop_by_object, drop_by_task = _drop_breakdown(
                    dropped, info["object_index"], info["task_index"]
                )
                _merge_drop_breakdown(
                    interval_drop_by_object, drop_by_object
                )
                _merge_drop_breakdown(interval_drop_by_task, drop_by_task)
                _merge_drop_breakdown(phase_drop_by_object, drop_by_object)
                _merge_drop_breakdown(phase_drop_by_task, drop_by_task)
                _merge_drop_breakdown(
                    global_success["drop_by_object"], drop_by_object
                )
                _merge_drop_breakdown(
                    global_success["drop_by_task"], drop_by_task
                )
                if (
                    teacher.episode_action is None
                    or teacher.episode_planned_grasp_plane_yaw_rad is None
                ):
                    raise RuntimeError("IK expert lost its frozen episode action")
                freeze_certificate = _episode_freeze_certificate(
                    trajectory_observation,
                    trajectory_action,
                    info,
                    teacher.episode_action,
                    teacher.episode_planned_grasp_plane_yaw_rad,
                    expected_task_index=expected_task_index,
                    torch=torch,
                )
                route_certified = info["path_tracking_certified"].to(
                    torch.bool
                )
                certified_candidates = (
                    success
                    & compliance
                    & (~dropped)
                    & route_certified
                    & freeze_certificate["trajectory_frozen"]
                )
                winner = teacher.select_winners(
                    episode_return, certified_candidates, compliance
                )
                teacher.update_centroids(winner)
                winner_success = success.index_select(0, winner)
                winner_compliance = compliance.index_select(0, winner)
                winner_not_dropped = (~dropped).index_select(0, winner)
                winner_route_certified = route_certified.index_select(
                    0, winner
                )
                winner_frozen = freeze_certificate[
                    "trajectory_frozen"
                ].index_select(0, winner)
                valid = (
                    winner_success
                    & winner_compliance
                    & winner_not_dropped
                    & winner_route_certified
                    & winner_frozen
                )
                valid_group = (
                    certified_candidates
                    .reshape(config.groups_per_vector_batch, config.elite.population)
                    .any(dim=1)
                )
                valid_groups = int(valid_group.sum().item())
                total_groups = int(valid_group.numel())
                valid_group_count += valid_groups
                valid_group_episode_count += total_groups
                interval_valid_group_count += valid_groups
                interval_valid_group_episode_count += total_groups
                global_success["valid_group_count"] += valid_groups
                global_success["valid_group_episode_count"] += total_groups
                selected = winner[valid]
                if int(selected.numel()):
                    count = int(selected.numel())
                    phase_value = 0
                    selected_task = info["task_index"].index_select(
                        0, selected
                    )
                    translating = (
                        selected_task
                        == contract.TASKS.index(contract.TASK_TRANSLATE)
                    )
                    selected_path_seed = info["path_seed"].index_select(
                        0, selected
                    )
                    selected_path_shape = info[
                        "path_shape_index"
                    ].index_select(0, selected)
                    selected_rotation_angle = info[
                        "rotation_angle_deg"
                    ].index_select(0, selected)
                    store.append(
                        trajectory_observation.index_select(0, selected)
                        .cpu()
                        .numpy(),
                        trajectory_action.index_select(0, selected)
                        .cpu()
                        .numpy(),
                        returns=episode_return.index_select(0, selected)
                        .cpu()
                        .numpy(),
                        success=winner_success[valid].cpu().numpy(),
                        object_index=info["object_index"]
                        .index_select(0, selected)
                        .cpu()
                        .numpy(),
                        task_index=selected_task.cpu().numpy(),
                        curriculum_phase=(
                            torch.full(
                                (count,),
                                phase_value,
                                device=config.device,
                                dtype=torch.int8,
                            )
                            .cpu()
                            .numpy()
                        ),
                        template_index=info["template_index"]
                        .index_select(0, selected)
                        .cpu()
                        .numpy(),
                        source_region=info["source_region"]
                        .index_select(0, selected)
                        .cpu()
                        .numpy(),
                        path_seed=torch.where(
                            translating,
                            selected_path_seed,
                            torch.zeros_like(selected_path_seed),
                        )
                        .cpu()
                        .numpy(),
                        path_shape_index=torch.where(
                            translating,
                            selected_path_shape,
                            torch.zeros_like(selected_path_shape),
                        )
                        .cpu()
                        .numpy(),
                        path_tracking_rmse_m=info[
                            "path_tracking_rmse_m"
                        ].index_select(0, selected).cpu().numpy(),
                        path_tracking_max_error_m=info[
                            "path_tracking_max_error_m"
                        ].index_select(0, selected).cpu().numpy(),
                        rotation_angle_deg=torch.where(
                            translating,
                            torch.zeros_like(selected_rotation_angle),
                            selected_rotation_angle,
                        )
                        .cpu()
                        .numpy(),
                        position_error_m=info["position_error_m"]
                        .index_select(0, selected)
                        .cpu()
                        .numpy(),
                        euler_error_deg=info["euler_l2_error_deg"]
                        .index_select(0, selected)
                        .cpu()
                        .numpy(),
                        position_error_xyz_m=info[
                            "final_position_error_xyz_m"
                        ].index_select(0, selected).cpu().numpy(),
                        euler_error_xyz_deg=info[
                            "final_euler_error_xyz_deg"
                        ].index_select(0, selected).cpu().numpy(),
                        plane_compliance=winner_compliance[valid].cpu().numpy(),
                        object_dropped=(
                            ~winner_not_dropped[valid]
                        ).cpu().numpy(),
                        path_frozen=freeze_certificate["path_frozen"]
                        .index_select(0, selected).cpu().numpy(),
                        plane_action_frozen=freeze_certificate[
                            "plane_action_frozen"
                        ].index_select(0, selected).cpu().numpy(),
                        anchors_frozen=freeze_certificate["anchors_frozen"]
                        .index_select(0, selected).cpu().numpy(),
                        trajectory_certified=(
                            certified_candidates.index_select(0, selected)
                        ).cpu().numpy(),
                        frozen_path_waypoints_xy_m=freeze_certificate[
                            "frozen_path_waypoints_xy_m"
                        ].index_select(0, selected).cpu().numpy(),
                        frozen_episode_action=freeze_certificate[
                            "frozen_episode_action"
                        ].index_select(0, selected).cpu().numpy(),
                        frozen_tip_height_fraction=freeze_certificate[
                            "frozen_tip_height_fraction"
                        ].index_select(0, selected).cpu().numpy(),
                        frozen_grasp_plane_fraction=freeze_certificate[
                            "frozen_grasp_plane_fraction"
                        ].index_select(0, selected).cpu().numpy(),
                        frozen_grasp_plane_yaw_rad=freeze_certificate[
                            "frozen_grasp_plane_yaw_rad"
                        ].index_select(0, selected).cpu().numpy(),
                        frozen_hard_tip_anchors_object_local_m=(
                            freeze_certificate[
                                "frozen_hard_tip_anchors_object_local_m"
                            ].index_select(0, selected).cpu().numpy()
                        ),
                    )
                    certified_trajectory_count += count
                endpoint_success_count += int(success.sum().item())
                endpoint_episode_count += int(success.numel())
                interval_endpoint_success_count += int(success.sum().item())
                interval_endpoint_episode_count += int(success.numel())
                latest_endpoint_success_rate = float(
                    success.to(torch.float32).mean().item()
                )
                breakdown_object, breakdown_task = _success_breakdown(
                    success,
                    info["object_index"],
                    info["task_index"],
                )
                _merge_breakdown(interval_breakdown_object, breakdown_object)
                _merge_breakdown(interval_breakdown_task, breakdown_task)
                _merge_breakdown(global_success["by_object"], breakdown_object)
                _merge_breakdown(global_success["by_task"], breakdown_task)
                if config.task == contract.TASK_ROTATE:
                    rotation_breakdown = _rotation_success_breakdown(
                        success,
                        info["task_index"],
                        info["rotation_angle_deg"],
                        phase=phase,
                    )
                    _merge_breakdown(
                        interval_rotation_breakdown, rotation_breakdown
                    )
                    _merge_breakdown(
                        phase_rotation_breakdown, rotation_breakdown
                    )
                global_success["success_count"] += int(success.sum().item())
                global_success["episode_count"] += int(success.numel())
                teacher.clear_episode_state()
                episode_index += 1
                episode_return.zero_()
                local_tick = 0
                if control_tick + 1 < control_ticks:
                    goals = teacher.begin_episode(phase, episode_index)
                    observation = environment.reset(goals=goals)
                    previous_episode_controller_counters = (
                        _zero_controller_counters()
                    )
                    trajectory_observation[:, 0].copy_(
                        observation.to(torch.float16)
                    )
            else:
                observation = next_observation
                local_tick += 1

            if (
                (control_tick + 1) % config.log_every_control_ticks == 0
                or time.monotonic() - last_status_time
                >= config.status_every_seconds
                or control_tick + 1 == control_ticks
            ):
                completed = (
                    remainder_worlds
                    + (control_tick + 1) * config.num_envs
                )
                info = transition.info
                rotation_status: dict[str, Any] = {}
                if config.task == contract.TASK_ROTATE:
                    live_rotation_targets = info["rotation_angle_deg"]
                    rotation_status = {
                        "rotation_target_angle_bounds_deg": list(
                            rotation_angle_bounds_deg(phase)
                        ),
                        "rotation_target_angle_deg_min": (
                            live_rotation_targets.min()
                        ),
                        "rotation_target_angle_deg_max": (
                            live_rotation_targets.max()
                        ),
                        "rotation_target_angle_deg_mean": (
                            live_rotation_targets.to(torch.float32).mean()
                        ),
                        "success_by_rotation_target_angle_deg": (
                            _success_breakdown_rates(
                                interval_rotation_breakdown
                            )
                        ),
                        "phase_cumulative_success_by_rotation_target_angle_deg": (
                            _success_breakdown_rates(
                                phase_rotation_breakdown
                            )
                        ),
                    }
                logger.write(
                    stage="ik_expert",
                    phase=phase,
                    phase_steps=completed,
                    total_steps=transitions_before_run + completed,
                    control_tick=control_tick + 1,
                    endpoint_success_rate=(
                        interval_endpoint_success_count
                        / interval_endpoint_episode_count
                        if interval_endpoint_episode_count
                        else None
                    ),
                    candidate_endpoint_success_rate=(
                        interval_endpoint_success_count
                        / interval_endpoint_episode_count
                        if interval_endpoint_episode_count
                        else None
                    ),
                    valid_group_rate=(
                        interval_valid_group_count
                        / interval_valid_group_episode_count
                        if interval_valid_group_episode_count
                        else None
                    ),
                    cumulative_valid_group_rate=(
                        global_success["valid_group_count"]
                        / global_success["valid_group_episode_count"]
                        if global_success["valid_group_episode_count"]
                        else None
                    ),
                    valid_group_count=interval_valid_group_count,
                    valid_group_episode_count=(
                        interval_valid_group_episode_count
                    ),
                    cumulative_valid_group_count=global_success[
                        "valid_group_count"
                    ],
                    cumulative_valid_group_episode_count=global_success[
                        "valid_group_episode_count"
                    ],
                    latest_endpoint_success_rate=latest_endpoint_success_rate,
                    cumulative_endpoint_success_rate=(
                        global_success["success_count"]
                        / global_success["episode_count"]
                        if global_success["episode_count"]
                        else None
                    ),
                    endpoint_success_count=interval_endpoint_success_count,
                    endpoint_episode_count=interval_endpoint_episode_count,
                    cumulative_endpoint_success_count=global_success[
                        "success_count"
                    ],
                    cumulative_endpoint_episode_count=global_success[
                        "episode_count"
                    ],
                    phase_certified_trajectory_count=(
                        certified_trajectory_count
                    ),
                    certified_trajectory_count=store.completed,
                    demos_written=store.completed,
                    success_by_object=interval_breakdown_object,
                    success_by_task=interval_breakdown_task,
                    cumulative_success_by_object=global_success["by_object"],
                    cumulative_success_by_task=global_success["by_task"],
                    **rotation_status,
                    drop_by_object=_drop_breakdown_rates(
                        interval_drop_by_object
                    ),
                    drop_by_task=_drop_breakdown_rates(
                        interval_drop_by_task
                    ),
                    cumulative_drop_by_object=_drop_breakdown_rates(
                        global_success["drop_by_object"]
                    ),
                    cumulative_drop_by_task=_drop_breakdown_rates(
                        global_success["drop_by_task"]
                    ),
                    grasp_plane_compliance_rate=(
                        plane_compliant_count / plane_sample_count
                        if plane_sample_count
                        else None
                    ),
                    grasp_plane_sample_count=plane_sample_count,
                    commanded_grasp_plane_compliance_rate=(
                        commanded_plane_compliant_count
                        / commanded_plane_sample_count
                        if commanded_plane_sample_count
                        else None
                    ),
                    commanded_grasp_plane_sample_count=(
                        commanded_plane_sample_count
                    ),
                    position_error_m=info["position_error_m"].mean(),
                    euler_l2_error_deg=info["euler_l2_error_deg"].mean(),
                    **latest_translation_endpoint_metrics,
                    reward_mean=transition.reward.mean(),
                    active_object_drop_rate=(
                        info["object_dropped"].to(torch.float32).mean()
                    ),
                    interval_object_drop_count=interval_object_drop_count,
                    cumulative_object_drop_count=global_success[
                        "object_drop_count"
                    ],
                    interval_controller_safety_counters=(
                        interval_controller_counters
                    ),
                    cumulative_controller_safety_counters=(
                        global_success["controller_safety_counters"]
                    ),
                    **_gpu_metrics(torch, config.device),
                )
                plane_compliant_count = 0
                plane_sample_count = 0
                commanded_plane_compliant_count = 0
                commanded_plane_sample_count = 0
                interval_endpoint_success_count = 0
                interval_endpoint_episode_count = 0
                interval_valid_group_count = 0
                interval_valid_group_episode_count = 0
                interval_breakdown_object = _zero_breakdown(contract.OBJECTS)
                interval_breakdown_task = _zero_breakdown(contract.TASKS)
                interval_rotation_breakdown = (
                    _zero_rotation_success_breakdown(phase)
                )
                interval_drop_by_object = _zero_drop_breakdown(
                    contract.OBJECTS
                )
                interval_drop_by_task = _zero_drop_breakdown(contract.TASKS)
                interval_controller_counters = (
                    _zero_controller_counters()
                )
                interval_object_drop_count = 0
                last_status_time = time.monotonic()
        health = environment.backend.health()
        _assert_backend_health(
            health,
            context="native MJWarp search",
            expected_world_counts=tuple(environment.world_counts),
        )
        memory_guard.raise_if_violated(f"{phase}_expert_phase_complete")
        return {
            "phase": phase,
            "steps": steps,
            "control_ticks": control_ticks,
            "remainder_worlds": remainder_worlds,
            "remainder_object_world_counts": list(
                remainder_object_world_counts
            ),
            "remainder_backend_health": (
                remainder_diagnostics["backend_health"]
                if remainder_diagnostics is not None
                else None
            ),
            "remainder_controller_safety_counters": (
                remainder_diagnostics["controller_safety_counters"]
                if remainder_diagnostics is not None
                else None
            ),
            "controller_safety_counters_total": phase_controller_counters,
            "completed_population_episodes": episode_index,
            "endpoint_success_count": endpoint_success_count,
            "endpoint_episode_count": endpoint_episode_count,
            "cumulative_endpoint_success_rate": (
                endpoint_success_count / endpoint_episode_count
                if endpoint_episode_count
                else None
            ),
            "certified_trajectory_count": certified_trajectory_count,
            "object_drop_count": endpoint_object_drop_count,
            "object_drop_event_count": object_drop_count,
            "remainder_object_drop_count": remainder_object_drop_count,
            "object_drop_rate": (
                endpoint_object_drop_count / endpoint_episode_count
                if endpoint_episode_count
                else None
            ),
            "drop_by_object": _drop_breakdown_rates(
                phase_drop_by_object
            ),
            "drop_by_task": _drop_breakdown_rates(
                phase_drop_by_task
            ),
            "valid_group_count": valid_group_count,
            "valid_group_episode_count": valid_group_episode_count,
            "valid_group_rate": (
                valid_group_count / valid_group_episode_count
                if valid_group_episode_count
                else None
            ),
            **(
                {
                    "rotation_target_angle_bounds_deg": list(
                        rotation_angle_bounds_deg(phase)
                    ),
                    "success_by_rotation_target_angle_deg": (
                        _success_breakdown_rates(phase_rotation_breakdown)
                    ),
                }
                if config.task == contract.TASK_ROTATE
                else {}
            ),
            "backend_health": _jsonable(health),
        }
    except BaseException as exc:
        active_search_exception = exc
        raise
    finally:
        try:
            cleanup_full_search_allocations()
        except BaseException:
            if active_search_exception is None:
                raise


def _validate_demo_distribution(
    store: CertifiedIKDemoStore, config: HPCConfig
) -> dict[str, int]:
    distribution = store.distribution()
    coverage = store.coverage_distribution()
    if store.completed < config.elite.minimum_successful_demos:
        raise RuntimeError(
            "certified IK expert produced only "
            f"{store.completed} trajectories; requires at least "
            f"{config.elite.minimum_successful_demos}"
        )
    required = {
        f"all/{object_name}/{task_name}"
        for object_name in contract.OBJECTS
        for task_name in (config.task,)
    }
    minimum = config.elite.minimum_demos_per_object_task
    missing = sorted(
        key for key in required if int(distribution.get(key, 0)) < minimum
    )
    if missing:
        raise RuntimeError(
            "certified IK demonstrations do not meet the minimum per "
            "controlled "
            f"stratum ({minimum}): "
            + ", ".join(missing)
        )
    if config.elite.require_all_reset_templates:
        missing_templates = [
            f"{object_name}/template{template}"
            for object_name, counts in coverage["template_counts"].items()
            for template in range(contract.RESET_TEMPLATES_PER_OBJECT)
            if int(counts[str(template)]) <= 0
        ]
        if missing_templates:
            raise RuntimeError(
                "certified IK demonstrations do not cover every object reset "
                "template: " + ", ".join(missing_templates)
            )
    if (
        config.task == contract.TASK_TRANSLATE
        and config.elite.require_all_translation_path_shapes
    ):
        missing_shapes = [
            f"{object_name}/{shape_name}"
            for object_name, values in coverage[
                "translation_path_by_object"
            ].items()
            for shape_name, count in values["shape_success_counts"].items()
            if int(count) <= 0
        ]
        if missing_shapes:
            raise RuntimeError(
                "certified translation demonstrations do not cover "
                "every path shape in every object stratum: "
                + ", ".join(missing_shapes)
            )
    if config.task == contract.TASK_ROTATE:
        missing_rotation_maxima = [
            f"{object_name}/{values['maximum_target_deg']}deg"
            for object_name, values in coverage[
                "rotation_angle_by_object"
            ].items()
            if int(values["maximum_target_success_count"]) <= 0
        ]
        if missing_rotation_maxima:
            raise RuntimeError(
                "certified rotation demonstrations do not cover the maximum "
                "task angle for every object stratum: "
                + ", ".join(missing_rotation_maxima)
            )
    return distribution


def _train_diffusion_body(
    *,
    config: HPCConfig,
    dataset: Path,
    output_dir: Path,
    platform_spec: Any,
    logger: ProgressLogger,
    memory_guard: TrainingDeviceMemoryGuard,
    cleanup_resources: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    import torch

    memory_guard.raise_if_violated("diffusion_stage_start")
    settings = config.diffusion
    sampler = DemoWindowSampler(dataset, seed=config.seed)
    cleanup_resources["sampler"] = sampler
    policy = DiffusionPolicy(
        obs_dim=OBSERVATION_DIM,
        action_dim=contract.ACTION_DIM,
        obs_horizon=settings.obs_horizon,
        horizon=settings.action_horizon,
        num_diffusion_steps=settings.diffusion_steps,
        hidden_dim=settings.hidden_dim,
        depth=settings.depth,
        dropout=settings.dropout,
    )
    cleanup_resources["policy"] = policy
    policy = policy.to(config.device)
    cleanup_resources["policy"] = policy
    policy.set_normalization(sampler.normalization())
    if settings.cache_demos_on_cuda:
        sampler.cache_on_device(config.device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    cleanup_resources["optimizer"] = optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=settings.epochs,
        eta_min=settings.learning_rate * 0.05,
    )
    cleanup_resources["scheduler"] = scheduler
    ema = ExponentialMovingAverage(policy, decay=settings.ema_decay)
    cleanup_resources["ema"] = ema
    batches_per_epoch = config.diffusion_batches_per_epoch(sampler.episodes)
    updates = 0
    best_loss = math.inf
    final_path = output_dir / "final_diffusion_model.pt"
    dataset_hash = sha256_file(dataset / "metadata.json")
    use_bfloat16 = (
        settings.use_bfloat16
        and torch.cuda.get_device_properties(torch.device(config.device)).major >= 8
    )
    training_started = time.perf_counter()
    last_status_time = time.monotonic()
    for epoch in range(settings.epochs):
        policy.train()
        loss_sum = 0.0
        gradient_norm = 0.0
        epoch_started = time.perf_counter()
        for batch_index in range(batches_per_epoch):
            sample_method = (
                sampler.sample_stratified
                if settings.stratified_object_task_sampling
                else sampler.sample
            )
            observation, action = sample_method(
                settings.batch_size,
                settings.action_horizon,
                device=config.device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                loss = policy.compute_loss(observation, action)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Diffusion loss became non-finite")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), settings.max_grad_norm
            )
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Diffusion gradient became non-finite")
            optimizer.step()
            ema.update(policy)
            memory_guard.raise_if_violated(
                f"diffusion_epoch_{epoch + 1}_update_{batch_index + 1}"
            )
            updates += 1
            loss_sum += float(loss.detach().cpu())
            gradient_norm = float(norm.detach().cpu())
            if (
                (batch_index + 1) % max(1, config.log_every_control_ticks) == 0
                or time.monotonic() - last_status_time
                >= config.status_every_seconds
                or batch_index + 1 == batches_per_epoch
            ):
                elapsed = time.perf_counter() - training_started
                completed_batches = epoch * batches_per_epoch + batch_index + 1
                total_batches = settings.epochs * batches_per_epoch
                batch_rate = completed_batches / max(elapsed, 1e-9)
                logger.write(
                    stage="diffusion",
                    phase="diffusion",
                    total_steps=config.total_steps,
                    epoch=epoch + 1,
                    diffusion_epochs=settings.epochs,
                    batch_in_epoch=batch_index + 1,
                    batches_per_epoch=batches_per_epoch,
                    diffusion_updates=updates,
                    diffusion_loss=float(loss.detach().cpu()),
                    diffusion_epoch_loss=loss_sum / (batch_index + 1),
                    diffusion_lr=optimizer.param_groups[0]["lr"],
                    diffusion_gradient_norm=gradient_norm,
                    diffusion_eta_s=(
                        (total_batches - completed_batches) / batch_rate
                    ),
                    demos_written=sampler.episodes,
                    **_gpu_metrics(torch, config.device),
                )
                last_status_time = time.monotonic()
        scheduler.step()
        epoch_loss = loss_sum / batches_per_epoch
        best_loss = min(best_loss, epoch_loss)
        metadata = {
            "algorithm": contract.ALGORITHM,
            "algorithm_signature": contract.ALGORITHM_SIGNATURE,
            "runtime_code_fingerprint_sha256": (
                runtime_code_fingerprint_sha256()
            ),
            "platform": platform_spec.platform,
            "trained_task": config.task,
            "task": config.task,
            "task_index": contract.TASKS.index(config.task),
            "root_height_m": float(platform_spec.root_height_m),
            "platform_adapter_fingerprint_sha256": (
                platform_spec.platform_adapter_fingerprint_sha256
            ),
            "physics_layout": contract.PHYSICS_LAYOUT,
            "global_worlds": config.num_envs,
            "object_world_counts": list(config.object_world_counts),
            "remainder_worlds": config.remainder_worlds,
            "remainder_object_world_counts": list(
                config.remainder_object_world_counts
            ),
            "model_shards": ordered_shard_manifest(
                platform_spec, config.object_world_counts
            ),
            "total_mjwarp_expert_steps": config.total_steps,
            "expert": "deterministic-certified-closed-loop-ik-v1",
            "rotation_angle_distribution_deg": [
                contract.ROTATION_ANGLE_MIN_DEG,
                contract.ROTATION_ANGLE_MAX_DEG,
            ],
            "translation_path_contract": (
                contract.translation_path_contract_payload()
            ),
            "epoch": epoch + 1,
            "epochs": settings.epochs,
            "updates": updates,
            "best_loss": best_loss,
            "dataset_metadata_sha256": dataset_hash,
            "training_config": config.as_dict(),
        }
        if (
            (epoch + 1) % settings.checkpoint_every_epochs == 0
            or epoch + 1 == settings.epochs
        ):
            checkpoint = policy_checkpoint(policy, ema=ema, metadata=metadata)
            checkpoint.update(
                {
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                }
            )
            _atomic_torch_save(torch, checkpoint, final_path)
        logger.write(
            stage="diffusion_epoch",
            phase="diffusion",
            total_steps=config.total_steps,
            epoch=epoch + 1,
            diffusion_epochs=settings.epochs,
            diffusion_updates=updates,
            diffusion_loss=epoch_loss,
            diffusion_best_loss=best_loss,
            diffusion_lr=optimizer.param_groups[0]["lr"],
            diffusion_gradient_norm=gradient_norm,
            epoch_seconds=time.perf_counter() - epoch_started,
            demos_written=sampler.episodes,
            **_gpu_metrics(torch, config.device),
        )
        memory_guard.raise_if_violated(
            f"diffusion_epoch_{epoch + 1}_complete"
        )
    memory_guard.raise_if_violated("diffusion_stage_complete")
    return final_path, {
        "epochs": settings.epochs,
        "updates": updates,
        "batches_per_epoch": batches_per_epoch,
        "best_loss": best_loss,
        "dataset_metadata_sha256": dataset_hash,
    }


def _release_diffusion_training_resources(
    *,
    resources: dict[str, Any],
    torch: Any,
    device: str,
) -> None:
    """Destroy every persistent CUDA owner, even after partial allocation."""

    errors: list[str] = []

    def cleanup(label: str, operation: Any) -> None:
        try:
            operation()
        except BaseException as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    sampler = resources.get("sampler")
    if sampler is not None:
        cleanup(
            "sampler cache",
            lambda: setattr(sampler, "device_observations", None),
        )
        cleanup(
            "sampler action cache",
            lambda: setattr(sampler, "device_actions", None),
        )
    optimizer = resources.get("optimizer")
    if optimizer is not None:
        cleanup("optimizer state", optimizer.state.clear)
        cleanup("optimizer parameter groups", optimizer.param_groups.clear)
    scheduler = resources.get("scheduler")
    if scheduler is not None:
        cleanup(
            "scheduler optimizer reference",
            lambda: setattr(scheduler, "optimizer", None),
        )
    ema = resources.get("ema")
    if ema is not None:
        cleanup("EMA shadow", ema.shadow.clear)
    policy = resources.get("policy")
    if policy is not None:
        def clear_policy_state() -> None:
            for module in policy.modules():
                module._parameters.clear()
                module._buffers.clear()

        cleanup("policy parameters and buffers", clear_policy_state)
    resources.clear()
    cleanup(
        "CUDA allocator release",
        lambda: _release_cuda_stage_memory(torch, device),
    )
    if errors:
        raise RuntimeError(
            "Diffusion CUDA cleanup failed: " + "; ".join(errors)
        )


def _train_diffusion(
    *,
    config: HPCConfig,
    dataset: Path,
    output_dir: Path,
    platform_spec: Any,
    logger: ProgressLogger,
    memory_guard: TrainingDeviceMemoryGuard,
) -> tuple[Path, dict[str, Any]]:
    """Train Diffusion with fail-safe cleanup of partial CUDA allocations."""

    import torch

    resources: dict[str, Any] = {}
    failed = False
    try:
        return _train_diffusion_body(
            config=config,
            dataset=dataset,
            output_dir=output_dir,
            platform_spec=platform_spec,
            logger=logger,
            memory_guard=memory_guard,
            cleanup_resources=resources,
        )
    except BaseException as exc:
        failed = True
        # Unwound Diffusion/torch frames otherwise retain batch activations
        # and partially constructed EMA tensors until after guard reporting.
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        try:
            _release_diffusion_training_resources(
                resources=resources,
                torch=torch,
                device=config.device,
            )
        except BaseException:
            if not failed:
                raise


def _admit_training_stage(
    *,
    memory_guard: TrainingDeviceMemoryGuard,
    capacity_selection: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    """Admit one stage against its own certified post-cleanup baseline."""

    increments = capacity_selection.get(
        "certified_stage_peak_increment_by_stage"
    )
    if not isinstance(increments, Mapping) or set(increments) != {
        "native_search",
        "diffusion",
    }:
        raise RuntimeError("capacity selection lacks exact stage increments")
    if stage not in increments:
        raise RuntimeError(f"uncertified training stage: {stage}")
    admissions = capacity_selection.setdefault(
        "training_memory_admissions", {}
    )
    if not isinstance(admissions, dict) or stage in admissions:
        raise RuntimeError(f"duplicate CUDA admission for stage {stage}")
    certificate = memory_guard.admit(int(increments[stage]))
    gpu = capacity_selection.get("gpu")
    expected_total = gpu.get("memory_bytes") if isinstance(gpu, Mapping) else None
    if certificate.get("total_bytes") != expected_total:
        raise RuntimeError(
            "training CUDA memory total differs from the capacity report"
        )
    certificate = dict(certificate)
    certificate["capacity_stage"] = stage
    admissions[stage] = certificate
    memory_guard.raise_if_violated(f"{stage}_post_admission")
    return certificate


def _train_two_phase_impl(
    config: HPCConfig,
    output_dir: Path,
    *,
    capacity_report: Path,
    memory_guard_sink: list[TrainingDeviceMemoryGuard],
    logger_sink: list[ProgressLogger],
) -> Path:
    """Collect certified IK trajectories and train one task-specific policy."""

    config.validate()
    if output_dir.exists():
        raise FileExistsError(
            f"training output already exists; choose a new directory: {output_dir}"
        )
    runtime = inspect_runtime(
        strict=True,
        device=config.device,
        deterministic_torch=config.deterministic_torch,
    )
    if not runtime["ok"]:
        raise RuntimeError(
            "HPC runtime inspection failed:\n"
            + "\n".join(str(item) for item in runtime["errors"])
        )
    platform_spec = build_platform_spec(config.task)
    capacity_selection = _validate_capacity_selection(
        report_path=capacity_report,
        config=config,
        runtime=runtime,
        platform_spec=platform_spec,
    )
    torch = _configure_torch(config)
    memory_guard = TrainingDeviceMemoryGuard(
        torch,
        torch.device(config.device),
        limit_fraction=0.90,
        poll_interval_s=0.02,
    )
    memory_guard_sink.append(memory_guard)
    memory_guard.raise_if_violated("pre_teacher")
    # Start device-wide polling before the first training CUDA owner.  The
    # native admission still samples after persistent teacher centroids exist,
    # exactly matching the capacity certificate's post-teacher baseline.
    teacher = MJWarpIKExpert(
        config,
        platform_spec=platform_spec,
        device=config.device,
    )
    capacity_selection = dict(capacity_selection)
    capacity_selection["training_memory_admissions"] = {}
    _admit_training_stage(
        memory_guard=memory_guard,
        capacity_selection=capacity_selection,
        stage="native_search",
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = ProgressLogger(output_dir, total_steps=config.total_steps)
    logger_sink.append(logger)
    dataset = output_dir / "certified_ik_demos"
    provenance = {
        "algorithm_signature": contract.ALGORITHM_SIGNATURE,
        "runtime_code_fingerprint_sha256": (
            runtime_code_fingerprint_sha256()
        ),
        "platform": platform_spec.platform,
        "trained_task": config.task,
        "task": config.task,
        "task_index": contract.TASKS.index(config.task),
        "root_height_m": float(platform_spec.root_height_m),
        "seed": config.seed,
        "config": config.as_dict(),
        "capacity_selection": capacity_selection,
        "expert": "deterministic-certified-closed-loop-ik-v1",
        "action_supervision": (
            "requested normalized closed-loop IK-expert command before "
            "environment low-pass filtering and episode-channel freezing"
        ),
    }
    store = CertifiedIKDemoStore(
        dataset,
        capacity=config.maximum_demo_episodes,
        steps=config.episode_length,
        create=True,
        provenance=provenance,
    )
    global_success: dict[str, Any] = {
        "success_count": 0,
        "episode_count": 0,
        "valid_group_count": 0,
        "valid_group_episode_count": 0,
        "by_object": _zero_breakdown(contract.OBJECTS),
        "by_task": _zero_breakdown(contract.TASKS),
        "controller_safety_counters": _zero_controller_counters(),
        "object_drop_count": 0,
        "completed_object_drop_count": 0,
        "drop_by_object": _zero_drop_breakdown(contract.OBJECTS),
        "drop_by_task": _zero_drop_breakdown(contract.TASKS),
    }
    full_task = _run_expert_task(
        phase=PHASE,
        steps=config.total_steps,
        transitions_before_run=0,
        config=config,
        platform_spec=platform_spec,
        teacher=teacher,
        store=store,
        logger=logger,
        global_success=global_success,
        memory_guard=memory_guard,
    )
    teacher.clear_episode_state()
    _release_cuda_stage_memory(torch, config.device)
    memory_guard.raise_if_violated("expert_release_complete")
    del teacher
    _release_cuda_stage_memory(torch, config.device)
    # The Diffusion certificate was measured from the post-native cleanup
    # baseline, so it must be admitted again against the actual current used
    # bytes before any sampler cache, policy, EMA, or optimizer is created.
    _admit_training_stage(
        memory_guard=memory_guard,
        capacity_selection=capacity_selection,
        stage="diffusion",
    )
    distribution = _validate_demo_distribution(store, config)
    coverage_distribution = store.coverage_distribution()
    demo_hashes = store.finalize()
    final_path, diffusion_summary = _train_diffusion(
        config=config,
        dataset=dataset,
        output_dir=output_dir,
        platform_spec=platform_spec,
        logger=logger,
        memory_guard=memory_guard,
    )
    memory_guard.raise_if_violated("training_stages_complete")
    training_device_memory = memory_guard.report()
    if training_device_memory.get("valid") is not True:
        raise RuntimeError(
            "training device-memory guard ended invalid: "
            + json.dumps(training_device_memory, sort_keys=True)
        )
    fruit_manifest = PROJECT_ROOT / "fruit_capsules_5mm.json"

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    manifest = {
        "schema": "native-mjwarp-sharded-certified-ik-diffusion-training-v6",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": contract.ALGORITHM,
        "algorithm_signature": contract.ALGORITHM_SIGNATURE,
        "runtime_code_fingerprint_sha256": (
            runtime_code_fingerprint_sha256()
        ),
        "control_hierarchy": (
            "Diffusion imitation -> independent hard-tip IK commands -> "
            "per-tip constrained IK/KKT -> actuators"
        ),
        "physics_backend": contract.PHYSICS_BACKEND,
        "torch_warp_exchange": contract.TORCH_WARP_EXCHANGE,
        "platform": platform_spec.platform,
        "trained_task": config.task,
        "task": config.task,
        "task_index": contract.TASKS.index(config.task),
        "root_height_m": float(platform_spec.root_height_m),
        "expert": "deterministic-certified-closed-loop-ik-v1",
        "total_mjwarp_expert_steps": config.total_steps,
        "expert_steps": config.total_steps,
        "rotation_angle_distribution_deg": {
            "bounds": [
                contract.ROTATION_ANGLE_MIN_DEG,
                contract.ROTATION_ANGLE_MAX_DEG,
            ],
            "integer_closed_intervals": True,
            "maximum_successful_demo_required_per_object": True,
        },
        "translation_path_contract": (
            contract.translation_path_contract_payload()
        ),
        "diffusion_epochs": config.diffusion.epochs,
        "success_gate": True,
        "trajectory_certification": (
            "endpoint_success_and_route_tracking_per_axis_10mm_and_plane_"
            "compliance_and_not_dropped_and_frozen_path_plane_four_anchors"
        ),
        "anti_support_gate_used": False,
        "physical_hard_tip_plane_success_invariant": True,
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
        "independent_hard_tip_control": True,
        "diffusion_action_supervision": (
            "requested normalized closed-loop IK-expert command before "
            "environment low-pass filtering and episode-channel freezing"
        ),
        "plate_trained": False,
        "reset_templates_per_object": contract.RESET_TEMPLATES_PER_OBJECT,
        "physics_layout": contract.PHYSICS_LAYOUT,
        "global_worlds": config.num_envs,
        "object_world_counts": list(config.object_world_counts),
        "remainder_worlds": config.remainder_worlds,
        "remainder_object_world_counts": list(
            config.remainder_object_world_counts
        ),
        "model_shards": ordered_shard_manifest(
            platform_spec, config.object_world_counts
        ),
        "parent_model_fingerprint_sha256": (
            platform_spec.parent_model_fingerprint_sha256
        ),
        "parent_controller_fingerprint_sha256": (
            platform_spec.parent_controller_fingerprint_sha256
        ),
        "platform_adapter_fingerprint_sha256": (
            platform_spec.platform_adapter_fingerprint_sha256
        ),
        "fruit_manifest_sha256": (
            _sha256(fruit_manifest) if fruit_manifest.is_file() else None
        ),
        "maximum_shard_nv": max(
            int(shard.host_model.nv) for shard in platform_spec.shards
        ),
        "observation_dim": OBSERVATION_DIM,
        "action_dim": contract.ACTION_DIM,
        "config": config.as_dict(),
        "capacity_selection": capacity_selection,
        "runtime": runtime,
        "training_device_memory": training_device_memory,
        "packages": {
            name: package_version(name)
            for name in ("torch", "mujoco", "mujoco-warp", "warp-lang")
        },
        "expert_final": full_task,
        "controller_safety": {
            "fail_closed": True,
            "hard_failure_fields": list(
                _CONTROLLER_HARD_FAILURE_FIELDS
            ),
            "all": full_task["controller_safety_counters_total"],
            "combined_totals": global_success[
                "controller_safety_counters"
            ],
        },
        "object_drop": {
            "detector": "finite_body_xpos_z_plus_local_bottom_below_table_margin",
            "table_top_z_m": contract.TABLE_TOP_Z_M,
            "margin_m": contract.DROP_MARGIN_M,
            "threshold_z_m": contract.OBJECT_DROP_THRESHOLD_Z_M,
            "terminal_reward": contract.OBJECT_DROP_TERMINAL_REWARD,
            "continuation": "sticky_absorbing_quarantine_to_fixed_horizon",
            "count": global_success["completed_object_drop_count"],
            "event_count_including_partial_episodes": global_success[
                "object_drop_count"
            ],
            "rate": (
                global_success["completed_object_drop_count"]
                / global_success["episode_count"]
                if global_success["episode_count"]
                else None
            ),
            "by_object": _drop_breakdown_rates(
                global_success["drop_by_object"]
            ),
            "by_task": _drop_breakdown_rates(
                global_success["drop_by_task"]
            ),
        },
        "certified_demo_count": store.completed,
        "certified_demo_distribution": distribution,
        "certified_demo_coverage": coverage_distribution,
        "certified_demo_hashes": demo_hashes,
        "diffusion": diffusion_summary,
        "final_model": final_path.name,
        "final_model_sha256": _sha256(final_path),
    }
    _atomic_json(manifest, output_dir / "training_manifest.json")
    logger.write(
        stage="complete",
        phase="complete",
        total_steps=config.total_steps,
        demos_written=store.completed,
        diffusion_updates=diffusion_summary["updates"],
        controller_safety_counters=global_success[
            "controller_safety_counters"
        ],
        object_drop_count=global_success["completed_object_drop_count"],
        training_device_memory=training_device_memory,
        final_model=final_path.name,
    )
    return final_path


def train_two_phase(
    config: HPCConfig,
    output_dir: Path,
    *,
    capacity_report: Path,
) -> Path:
    """Run formal training and always finalize its CUDA memory guard.

    The implementation publishes a newly created guard and logger through
    private sinks.  Keeping cleanup in this thin public boundary guarantees
    that admission failures, training failures, and finalization failures all
    stop the background sampler and preserve a final diagnostic when a logger
    is already available.
    """

    guards: list[TrainingDeviceMemoryGuard] = []
    loggers: list[ProgressLogger] = []
    failed = False
    try:
        return _train_two_phase_impl(
            config,
            output_dir,
            capacity_report=capacity_report,
            memory_guard_sink=guards,
            logger_sink=loggers,
        )
    except BaseException as exc:
        failed = True
        # Release all locals owned by unwound constructors/training frames
        # before taking the final synchronized device-memory sample.
        traceback.clear_frames(exc.__traceback__)
        if guards:
            guard = guards[-1]
            try:
                memory_report: dict[str, Any] = guard.report()
            except BaseException as report_exc:
                memory_report = {
                    "valid": False,
                    "closed": False,
                    "errors": [
                        "memory-guard report failed during exception cleanup: "
                        f"{type(report_exc).__name__}: {report_exc}"
                    ],
                }
            if loggers:
                logger = loggers[-1]
                try:
                    logger.write(
                        stage="failed",
                        phase="failed",
                        total_steps=int(
                            getattr(logger, "last_total_steps", 0)
                        ),
                        error=f"{type(exc).__name__}: {exc}",
                        training_device_memory=memory_report,
                    )
                except BaseException:
                    # Preserve the original training failure.  The guard is
                    # still closed by the finally block below.
                    pass
        raise
    finally:
        if guards:
            try:
                guards[-1].close()
            except BaseException:
                # Never replace the original failure, but fail closed if
                # cleanup itself is the only failing operation.
                if not failed:
                    raise


def _default_run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"diffusion_ik_native_mjwarp_60m_seed"
        f"{DEFAULT_CONFIG.seed}_{stamp}"
    )


def _resolve_num_envs(
    value: str,
    capacity_report: Path,
    parser: argparse.ArgumentParser,
) -> int:
    if value != "auto":
        parser.error(
            "formal training requires --num-envs auto and a hash-bound "
            "G6 capacity report; use capacity_scan for debug worlds"
        )
    report = capacity_report.expanduser().resolve()
    if not report.is_file():
        parser.error(
            "automatic world selection requires a capacity report; run "
            "`python -m MJWarp_RL.capacity_scan` for G6 first"
        )
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("schema") != CAPACITY_REPORT_SCHEMA:
            raise ValueError("incompatible capacity report schema")
        if payload.get("platform") != "gripper6":
            raise ValueError("formal training requires a G6 capacity report")
        if payload.get("platforms") != ["gripper6"]:
            raise ValueError("capacity report is not G6-only")
        if payload.get("root_height_m") != 0.56:
            raise ValueError("capacity report G6 root height is stale")
        if payload.get("algorithm_signature") != contract.ALGORITHM_SIGNATURE:
            raise ValueError("capacity report algorithm signature is stale")
        if int(payload.get("observation_dim", -1)) != OBSERVATION_DIM:
            raise ValueError("capacity report observation dimension is stale")
        if int(payload.get("action_dim", -1)) != contract.ACTION_DIM:
            raise ValueError("capacity report action dimension is stale")
        if int(payload.get("episode_control_ticks", -1)) != (
            DEFAULT_CONFIG.episode_length
        ):
            raise ValueError("capacity report episode allocation is stale")
        if tuple(payload.get("candidates", ())) != FORMAL_WORLD_CANDIDATES:
            raise ValueError("capacity report candidates are not formal")
        selected = int(payload["selected_num_envs"])
        passing = {int(item) for item in payload["passing_candidates"]}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"invalid capacity report {report}: {exc}")
    if selected not in passing or selected != max(passing):
        parser.error("capacity report does not select its largest passing candidate")
    if not passing.issubset(set(FORMAL_WORLD_CANDIDATES)):
        parser.error("capacity report contains a non-formal passing candidate")
    if selected not in FORMAL_WORLD_CANDIDATES:
        parser.error("capacity report selected a world count outside the formal scan")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "artifacts"
            / _default_run_name()
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument(
        "--task",
        choices=contract.TASKS,
        required=True,
        help="train exactly one task; translation and rotation are separate runs",
    )
    parser.add_argument(
        "--num-envs",
        default="auto",
        choices=("auto",),
        help="formal runs always use the largest qualified G6 worlds",
    )
    parser.add_argument(
        "--capacity-report",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "artifacts"
            / "capacity_scan.json"
        ),
    )
    parser.add_argument("--device", default=DEFAULT_CONFIG.device)
    args = parser.parse_args(argv)
    num_envs = _resolve_num_envs(
        args.num_envs, args.capacity_report, parser
    )
    config = replace(
        DEFAULT_CONFIG,
        seed=args.seed,
        num_envs=num_envs,
        device=args.device,
        task=args.task,
    )
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    final_path = train_two_phase(
        config,
        args.output.resolve(),
        capacity_report=args.capacity_report,
    )
    print(f"final model: {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
