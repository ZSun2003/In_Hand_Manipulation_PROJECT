"""Subprocess-isolated RTX PRO 6000 world-capacity scan for formal training."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from . import contract
from .curriculum import PHASE
from .diffusion_policy import DiffusionPolicy, ExponentialMovingAverage
from .diffusion_runtime import (
    BatchedDiffusionController,
    BatchedObservationHistory,
)
from .ik_expert import MJWarpIKExpert
from .env import OBSERVATION_DIM, make_env
from .hpc_config import (
    CAPACITY_REPORT_SCHEMA,
    DEFAULT_CONFIG,
    FORMAL_WORLD_CANDIDATES,
)
from .memory_certificate import (
    POST_RELEASE_DEVICE_TOLERANCE_FRACTION,
    POST_RELEASE_DEVICE_TOLERANCE_MIN_BYTES,
    POST_RELEASE_TORCH_ALLOCATED_TOLERANCE_BYTES,
    POST_RELEASE_TORCH_RESERVED_TOLERANCE_BYTES,
    post_release_certificate_is_valid,
    stage_has_actual_memory_certificate,
)
from .platform_adapter import PLATFORM, TASK_ROOT_HEIGHT_M, build_platform_spec
from .preflight import configure_cublas_workspace, inspect_runtime
from .sharded_env import (
    formal_sharded_health_is_valid,
    ordered_shard_manifest,
)
from .runtime_fingerprint import runtime_code_fingerprint_sha256


CANDIDATES = FORMAL_WORLD_CANDIDATES
REPORT_SCHEMA = CAPACITY_REPORT_SCHEMA
ROTATION_DIAGNOSTIC_MIN_WORLDS = (
    len(contract.OBJECTS)
    * (contract.ROTATION_ANGLE_MAX_DEG - contract.ROTATION_ANGLE_MIN_DEG + 1)
    * DEFAULT_CONFIG.elite.population
)
_PLATFORM_FINGERPRINT_FIELDS = (
    "parent_model_fingerprint_sha256",
    "parent_controller_fingerprint_sha256",
    "platform_adapter_fingerprint_sha256",
)
_RL_MODEL_FINGERPRINT_FIELD = "rl_model_fingerprint_sha256"
_CONTROLLER_COUNTER_FIELDS = (
    "ik_solve_fallback_count",
    "ik_trust_region_projection_count",
    "position_feedback_slew_clip_count",
    "preload_trim_slew_clip_count",
    "ctrl_slew_clip_count",
    "ctrl_certificate_fallback_count",
)
_HARD_CONTROLLER_COUNTER_FIELDS = (
    "ik_solve_fallback_count",
    "ctrl_certificate_fallback_count",
)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    return _sha256_bytes(encoded)


def _valid_shard_manifest(
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
    for index, (entry, worlds) in enumerate(
        zip(value, expected_world_counts)
    ):
        if not isinstance(entry, dict):
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
        if expected_manifest is not None:
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
            if (
                len(expected_manifest) != contract.OBJECT_SHARD_COUNT
                or not isinstance(expected_manifest[index], dict)
                or any(
                    entry.get(name) != expected_manifest[index].get(name)
                    for name in fields
                )
            ):
                return False
    return True


def _config_sha256(config: Any) -> str:
    return _sha256_bytes(config.canonical_json().encode("utf-8"))


def _scan_config_sha256(task: str, device: str) -> str:
    """Bind the report to every runtime-derived formal candidate config."""

    selected_task = contract.validate_task(task)
    return _canonical_sha256(
        {
            str(candidate): replace(
                DEFAULT_CONFIG,
                task=selected_task,
                num_envs=int(candidate),
                device=str(device),
            ).canonical_json()
            for candidate in CANDIDATES
        }
    )


def _gpu_binding(torch_runtime: dict[str, Any]) -> dict[str, Any]:
    """Return the hardware fields that must match across controlled scans."""

    capability = torch_runtime.get("compute_capability")
    return {
        "name": torch_runtime.get("name"),
        "memory_bytes": torch_runtime.get("memory_bytes"),
        "compute_capability": (
            list(capability) if capability is not None else None
        ),
    }


def _memory_gb(torch: Any, device: Any) -> tuple[float, float]:
    return (
        torch.cuda.max_memory_allocated(device) / (1024.0**3),
        torch.cuda.max_memory_reserved(device) / (1024.0**3),
    )


def _allocated_gb(torch: Any, device: Any) -> tuple[float, float]:
    return (
        torch.cuda.memory_allocated(device) / (1024.0**3),
        torch.cuda.memory_reserved(device) / (1024.0**3),
    )


_ACTIVE_DEVICE_MEMORY_TRACKER: "_DeviceMemoryTracker | None" = None


class _DeviceMemoryTracker:
    """Sample device-wide CUDA use so Warp allocations cannot be hidden.

    Torch's allocator counters deliberately exclude allocations owned by
    Warp/MJWarp. ``cudaMemGetInfo`` is allocator-agnostic and therefore binds
    the capacity certificate to the actual device pressure seen by both
    runtimes.  Samples are synchronized checkpoints; the external 90% HPC
    watchdog remains the asynchronous fallback between checkpoints.
    """

    def __init__(
        self,
        torch: Any,
        device: Any,
        *,
        stage: str,
        poll_interval_s: float = 0.02,
    ) -> None:
        global _ACTIVE_DEVICE_MEMORY_TRACKER
        self.torch = torch
        self.device = device
        self.stage = str(stage)
        self.poll_interval_s = float(poll_interval_s)
        if self.poll_interval_s <= 0.0:
            raise ValueError("CUDA memory poll interval must be positive")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread_errors: list[str] = []
        self._background_sample_count = 0
        self._active_segment = "stage_setup"
        self._segment_minimum_free_bytes: dict[str, int] = {}
        self._stopped = False
        self._cached_report: dict[str, Any] | None = None
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        self.total_bytes = int(total_bytes)
        self.baseline_free_bytes = int(free_bytes)
        self.minimum_free_bytes = int(free_bytes)
        self.samples: list[dict[str, Any]] = []
        self.sample("baseline", synchronize=False)
        self._thread = threading.Thread(
            target=self._poll_device_memory,
            name=f"cuda-memory-{self.stage}",
            daemon=True,
        )
        self._thread.start()
        _ACTIVE_DEVICE_MEMORY_TRACKER = self

    def _record_free_bytes(self, free_bytes: int) -> None:
        with self._lock:
            self.minimum_free_bytes = min(
                self.minimum_free_bytes, int(free_bytes)
            )
            current = self._segment_minimum_free_bytes.get(
                self._active_segment, int(free_bytes)
            )
            self._segment_minimum_free_bytes[self._active_segment] = min(
                current, int(free_bytes)
            )

    def begin_segment(self, name: str) -> None:
        """Start attributing background samples to one protocol segment."""

        label = str(name)
        self.sample(f"segment_start:{label}")
        with self._lock:
            current_free = int(self.samples[-1]["free_bytes"])
            self._active_segment = label
            self._segment_minimum_free_bytes.setdefault(
                label, current_free
            )

    def _poll_device_memory(self) -> None:
        try:
            self.torch.cuda.set_device(self.device)
            while not self._stop.is_set():
                free_bytes, total_bytes = self.torch.cuda.mem_get_info(
                    self.device
                )
                if int(total_bytes) != self.total_bytes:
                    raise RuntimeError(
                        "CUDA device capacity changed during background poll"
                    )
                self._record_free_bytes(int(free_bytes))
                with self._lock:
                    self._background_sample_count += 1
                self._stop.wait(self.poll_interval_s)
        except BaseException as exc:
            with self._lock:
                self._thread_errors.append(
                    f"{type(exc).__name__}: {exc}"
                )
            self._stop.set()

    def sample(
        self, label: str, *, synchronize: bool = True
    ) -> dict[str, Any]:
        if synchronize:
            self.torch.cuda.synchronize(self.device)
        free_bytes, total_bytes = self.torch.cuda.mem_get_info(self.device)
        if int(total_bytes) != self.total_bytes:
            raise RuntimeError("CUDA device capacity changed during scan")
        free = int(free_bytes)
        self._record_free_bytes(free)
        cuda = self.torch.cuda
        allocated = int(
            cuda.memory_allocated(self.device)
            if hasattr(cuda, "memory_allocated")
            else 0
        )
        reserved = int(
            cuda.memory_reserved(self.device)
            if hasattr(cuda, "memory_reserved")
            else allocated
        )
        sample = {
            "label": str(label),
            "free_bytes": free,
            "device_used_bytes": self.total_bytes - free,
            "torch_allocated_bytes": allocated,
            "torch_reserved_bytes": reserved,
        }
        with self._lock:
            self.samples.append(sample)
        return dict(sample)

    def _stop_polling(self) -> None:
        if self._stopped:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.poll_interval_s * 10.0))
        if self._thread.is_alive():
            with self._lock:
                self._thread_errors.append(
                    "background CUDA memory poll thread did not stop"
                )
        self._stopped = True

    def _report(self, *, enforce: bool) -> dict[str, Any]:
        if self._cached_report is not None:
            result = dict(self._cached_report)
            if enforce and not result["valid"]:
                raise RuntimeError(result["validation_error"])
            return result
        self._stop_polling()
        with self._lock:
            minimum_free = int(self.minimum_free_bytes)
            background_samples = int(self._background_sample_count)
            thread_errors = list(self._thread_errors)
            checkpoints = list(self.samples)
            segment_minimums = dict(self._segment_minimum_free_bytes)
        peak_used = self.total_bytes - minimum_free
        stage_delta = max(
            0, self.baseline_free_bytes - minimum_free
        )
        fraction = peak_used / self.total_bytes
        validation_errors = list(thread_errors)
        if fraction > 0.90:
            validation_errors.append(
                f"actual CUDA use reached {fraction:.2%}, exceeding 90%"
            )
        segments = {
            name: {
                "minimum_free_bytes": free,
                "actual_peak_device_used_bytes": self.total_bytes - free,
                "actual_peak_device_used_gb": (
                    (self.total_bytes - free) / (1024.0**3)
                ),
                "actual_peak_device_memory_fraction": (
                    (self.total_bytes - free) / self.total_bytes
                ),
            }
            for name, free in sorted(segment_minimums.items())
        }
        result = {
            "measurement": (
                "background torch.cuda.mem_get_info device-wide polling"
            ),
            "includes_non_torch_cuda_allocations": True,
            "stage": self.stage,
            "total_bytes": self.total_bytes,
            "baseline_free_bytes": self.baseline_free_bytes,
            "minimum_free_bytes": minimum_free,
            "actual_peak_device_used_bytes": peak_used,
            "actual_peak_device_used_gb": peak_used / (1024.0**3),
            "actual_peak_device_memory_fraction": fraction,
            "stage_peak_increment_bytes": stage_delta,
            "stage_peak_increment_gb": stage_delta / (1024.0**3),
            "limit_fraction": 0.90,
            "poll_interval_s": self.poll_interval_s,
            "background_sample_count": background_samples,
            "checkpoint_sample_count": len(checkpoints),
            "sample_count": background_samples + len(checkpoints),
            "checkpoint_samples": checkpoints,
            "segments": segments,
            "poll_thread_errors": thread_errors,
            "valid": not validation_errors,
            "validation_error": (
                "; ".join(validation_errors)
                if validation_errors
                else None
            ),
        }
        self._cached_report = dict(result)
        if enforce and validation_errors:
            raise RuntimeError(
                f"{self.stage} device-memory certificate failed: "
                + result["validation_error"]
            )
        return result

    def report(self) -> dict[str, Any]:
        return self._report(enforce=True)

    def failure_report(self) -> dict[str, Any]:
        return self._report(enforce=False)


def _release_cuda_stage_memory(torch: Any, device: Any) -> None:
    """Release Torch, Warp mempool, and retired CUDA Graph allocations."""

    import warp as wp

    wp_device = wp.get_device(str(device))
    wp.set_mempool_release_threshold(wp_device, 0)
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    wp.synchronize_device(wp_device)
    # CUDA graph destruction completes through an asynchronous callback.
    # A second synchronization lets Warp process deferred destructors before
    # trimming CUDA's separate graph-memory pool.  The pinned Warp 1.14
    # runtime exposes this binding even though it is not a public top-level
    # function.
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
    torch.cuda.synchronize(device)


def _post_release_baseline_certificate(
    *,
    memory: _DeviceMemoryTracker,
    label: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless a released stage returns near its actual baseline."""

    current = memory.sample(f"post_release:{label}")
    device_tolerance = max(
        POST_RELEASE_DEVICE_TOLERANCE_MIN_BYTES,
        int(memory.total_bytes * POST_RELEASE_DEVICE_TOLERANCE_FRACTION),
    )
    device_unrecovered = max(
        0, int(baseline["free_bytes"]) - int(current["free_bytes"])
    )
    torch_allocated_unrecovered = max(
        0,
        int(current["torch_allocated_bytes"])
        - int(baseline["torch_allocated_bytes"]),
    )
    torch_reserved_unrecovered = max(
        0,
        int(current["torch_reserved_bytes"])
        - int(baseline["torch_reserved_bytes"]),
    )
    passed = (
        device_unrecovered <= device_tolerance
        and torch_allocated_unrecovered
        <= POST_RELEASE_TORCH_ALLOCATED_TOLERANCE_BYTES
        and torch_reserved_unrecovered
        <= POST_RELEASE_TORCH_RESERVED_TOLERANCE_BYTES
    )
    certificate = {
        "passed": passed,
        "label": str(label),
        "measurement": "actual device free plus current Torch allocation",
        "total_device_bytes": memory.total_bytes,
        "baseline": dict(baseline),
        "post_release": current,
        "device_unrecovered_bytes": device_unrecovered,
        "device_tolerance_bytes": device_tolerance,
        "torch_allocated_unrecovered_bytes": torch_allocated_unrecovered,
        "torch_allocated_tolerance_bytes": (
            POST_RELEASE_TORCH_ALLOCATED_TOLERANCE_BYTES
        ),
        "torch_reserved_unrecovered_bytes": torch_reserved_unrecovered,
        "torch_reserved_tolerance_bytes": (
            POST_RELEASE_TORCH_RESERVED_TOLERANCE_BYTES
        ),
    }
    if not passed:
        raise RuntimeError(
            "post-release CUDA baseline certificate failed: "
            + json.dumps(certificate, sort_keys=True)
        )
    return certificate


def _controller_counter_totals(environment: Any) -> dict[str, int]:
    return {
        name: int(getattr(environment, name).sum().detach().cpu())
        for name in _CONTROLLER_COUNTER_FIELDS
    }


def _controller_counters_are_formally_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(_CONTROLLER_COUNTER_FIELDS)
        and all(
            isinstance(value[name], int)
            and not isinstance(value[name], bool)
            and value[name] >= 0
            for name in _CONTROLLER_COUNTER_FIELDS
        )
        and all(value[name] == 0 for name in _HARD_CONTROLLER_COUNTER_FIELDS)
    )


def _assert_hard_controller_counters_zero(
    counters: dict[str, int], *, context: str
) -> None:
    failures = {
        name: counters[name]
        for name in _HARD_CONTROLLER_COUNTER_FIELDS
        if counters[name] != 0
    }
    if failures:
        raise RuntimeError(
            f"{context} required uncertified controller fallback(s): "
            + json.dumps(failures, sort_keys=True)
        )


def _validate_diagnostic_config(config: Any) -> None:
    """Validate a capacity-only config without authorizing a 60M run."""

    config.elite.validate()
    config.diffusion.validate()
    if config.num_envs <= 0 or config.num_eval_envs <= 0:
        raise ValueError("environment counts must be positive")
    if config.num_envs % config.elite.population:
        raise ValueError("num_envs must be divisible by elite population")
    for name in (
        "physics_substeps_per_control_tick",
        "plan_ticks",
        "close_ticks",
        "settle_ticks",
        "lift_ticks",
        "hold_ticks",
        "action_ticks",
        "contacts_per_world",
        "constraints_per_world",
        "log_every_control_ticks",
    ):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.status_every_seconds <= 0.0:
        raise ValueError("status_every_seconds must be positive")
    expected_steps = int(config.num_envs) * int(config.episode_length)
    if int(config.total_steps) != expected_steps:
        raise ValueError(
            "diagnostic run must contain exactly one complete episode"
        )
    if _run_lifecycle_counts(config) != (
        int(config.episode_length),
        1,
        0,
    ):
        raise ValueError(
            "diagnostic lifecycle must contain one complete episode and no tail"
        )


def _run_lifecycle_counts(config: Any) -> tuple[int, int, int]:
    """Return configured ticks, complete episodes, and the final tail."""

    configured_ticks = int(config.control_ticks)
    episode_ticks = int(config.episode_length)
    if configured_ticks <= 0 or episode_ticks <= 0:
        raise ValueError("run and episode control ticks must be positive")
    complete_episodes, final_tail_ticks = divmod(
        configured_ticks, episode_ticks
    )
    if complete_episodes <= 0:
        raise ValueError("formal run must contain a complete episode")
    return configured_ticks, complete_episodes, final_tail_ticks


def _worker_config(
    *,
    num_envs: int,
    device: str,
    task: str,
    diagnostic_only: bool,
) -> Any:
    """Build a formal config or one bounded capacity-only diagnostic config.

    A diagnostic world count must not inherit the formal 60M-step budget:
    doing so makes runtime grow inversely with the requested world count.  Each
    diagnostic run instead owns exactly one task-specific complete episode.
    """

    selected_task = contract.validate_task(task)
    config = replace(
        DEFAULT_CONFIG,
        task=selected_task,
        num_envs=int(num_envs),
        device=device,
    )
    if not diagnostic_only:
        return config
    if (
        selected_task == contract.TASK_ROTATE
        and int(num_envs) < ROTATION_DIAGNOSTIC_MIN_WORLDS
    ):
        raise ValueError(
            "rotation diagnostic requires at least "
            f"{ROTATION_DIAGNOSTIC_MIN_WORLDS} worlds to cover every "
            "10..45 degree target for all four objects at population=10"
        )
    return replace(
        config,
        total_steps=int(num_envs) * int(config.episode_length),
    )


def _diagnostic_run_lifecycle(config: Any) -> dict[str, Any]:
    """Return explicit one-episode evidence for a diagnostic report."""

    control_ticks, complete_episodes, final_tail_ticks = (
        _run_lifecycle_counts(config)
    )
    return {
        "total_environment_steps": int(config.total_steps),
        "episode_control_ticks": int(config.episode_length),
        "run": {
            "environment_steps": int(config.total_steps),
            "control_ticks": control_ticks,
            "complete_episode_count": complete_episodes,
            "final_tail_ticks": final_tail_ticks,
        },
    }


def _worker_lifecycle_metadata(
    config: Any, *, diagnostic_only: bool
) -> dict[str, Any]:
    """Return task-specific lifecycle fields for the worker report."""

    result: dict[str, Any] = {
        "episode_control_ticks": int(config.episode_length),
    }
    if diagnostic_only:
        result["diagnostic_run_lifecycle"] = (
            _diagnostic_run_lifecycle(config)
        )
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.ndim == 0 else tensor.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _unexpected_early_done(transition: Any, torch: Any) -> Any:
    """Keep finite object-drop terminals out of native failure diagnostics."""

    done = transition.done.to(torch.bool)
    dropped = transition.info.get("object_dropped")
    if dropped is None:
        dropped = torch.zeros_like(done)
    else:
        dropped = torch.as_tensor(
            dropped, device=done.device, dtype=torch.bool
        )
    time_out = transition.info.get("time_out")
    if time_out is None:
        time_out = torch.zeros_like(done)
    else:
        time_out = torch.as_tensor(
            time_out, device=done.device, dtype=torch.bool
        )
    transition_finite = transition.info.get("transition_finite")
    if transition_finite is None:
        transition_finite = torch.ones_like(done)
    else:
        transition_finite = torch.as_tensor(
            transition_finite, device=done.device, dtype=torch.bool
        )
    return (done & (~dropped) & (~time_out)) | (~transition_finite)


_TRANSLATION_PATH_NUMERIC_TOLERANCE_M = 2.0e-6
_INACTIVE_COVERAGE_REASON = "inactive_for_task"


def _inactive_goal_coverage(
    *,
    coverage: str,
    task: str,
    phase: str,
    world_counts: tuple[int, ...],
) -> dict[str, Any]:
    """Return an explicit, fail-closed certificate for the inactive task."""

    selected_task = contract.validate_task(task)
    return {
        "active": False,
        "coverage": coverage,
        "task": selected_task,
        "task_index": contract.TASKS.index(selected_task),
        "phase": phase,
        "reason": _INACTIVE_COVERAGE_REASON,
        "worlds": int(sum(world_counts)),
        "by_object": {
            name: {
                "active": False,
                "world_count": int(worlds),
            }
            for name, worlds in zip(contract.OBJECTS, world_counts)
        },
    }


def _inactive_goal_coverage_is_valid(
    value: Any,
    *,
    coverage: str,
    task: str,
    phase: str,
    expected_world_counts: tuple[int, ...],
) -> bool:
    try:
        selected_task = contract.validate_task(task)
    except ValueError:
        return False
    by_object = value.get("by_object") if isinstance(value, dict) else None
    return (
        isinstance(value, dict)
        and value.get("active") is False
        and value.get("coverage") == coverage
        and value.get("task") == selected_task
        and value.get("task_index") == contract.TASKS.index(selected_task)
        and value.get("phase") == phase
        and value.get("reason") == _INACTIVE_COVERAGE_REASON
        and value.get("worlds") == sum(expected_world_counts)
        and isinstance(by_object, dict)
        and set(by_object) == set(contract.OBJECTS)
        and all(
            isinstance(by_object[name], dict)
            and by_object[name]
            == {
                "active": False,
                "world_count": int(expected_world_counts[index]),
            }
            for index, name in enumerate(contract.OBJECTS)
        )
    )


def _translation_path_phase_contract(
    phase: str,
) -> int:
    if phase == PHASE:
        return 0
    raise ValueError(f"unsupported translation-path phase: {phase!r}")


def _translation_path_goal_coverage_is_valid(
    value: Any,
    phase: str,
    *,
    task: str,
    expected_world_counts: tuple[int, ...] | None = None,
    require_complete_shape_coverage: bool = True,
) -> bool:
    """Fail-closed validation for one capacity path-coverage certificate."""

    if not isinstance(require_complete_shape_coverage, bool):
        return False

    try:
        source_region = _translation_path_phase_contract(phase)
    except ValueError:
        return False
    try:
        selected_task = contract.validate_task(task)
    except ValueError:
        return False
    if expected_world_counts is not None and (
        len(expected_world_counts) != len(contract.OBJECTS)
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            for count in expected_world_counts
        )
    ):
        return False
    if expected_world_counts is None:
        if not isinstance(value, dict):
            return False
        by_object = value.get("by_object")
        if not isinstance(by_object, dict) or set(by_object) != set(
            contract.OBJECTS
        ):
            return False
        try:
            expected_world_counts = tuple(
                int(by_object[name]["world_count"])
                for name in contract.OBJECTS
            )
        except (KeyError, TypeError, ValueError):
            return False
    if selected_task == contract.TASK_ROTATE:
        return _inactive_goal_coverage_is_valid(
            value,
            coverage="translation_path",
            task=selected_task,
            phase=phase,
            expected_world_counts=expected_world_counts,
        )
    if not isinstance(value, dict):
        return False
    by_object = value.get("by_object")
    if not (
        value.get("active") is True
        and value.get("coverage") == "translation_path"
        and value.get("task") == selected_task
        and value.get("task_index") == contract.TASKS.index(selected_task)
        and value.get("phase") == phase
        and value.get("path_family") == contract.TRANSLATION_PATH_FAMILY
        and value.get("segments") == contract.TRANSLATION_PATH_SEGMENTS
        and value.get("waypoints") == contract.TRANSLATION_PATH_WAYPOINTS
        and value.get("shape_names")
        == list(contract.TRANSLATION_PATH_SHAPES)
        and value.get("expected_source_region") == source_region
        and value.get("path_seed_provenance_sentinel")
        == [
            contract.TRANSLATION_PATH_SEED_MIN,
            contract.TRANSLATION_PATH_SEED_MAX,
        ]
        and value.get("expected_arc_step_m_by_shape")
        == list(contract.TRANSLATION_PATH_STEP_M_BY_SHAPE)
        and value.get("expected_arc_length_m_by_shape")
        == list(contract.TRANSLATION_PATH_LENGTH_M_BY_SHAPE)
        and value.get("fixed_geometry_phase_invariant") is True
        and value.get("waypoint_radius_limit_m")
        == contract.TRANSLATION_PATH_REGION_RADIUS_M
        and value.get("generation_radius_limit_m")
        == contract.TRANSLATION_PATH_GENERATION_RADIUS_M
        and value.get("start_radius_limit_m")
        == contract.TRANSLATION_PATH_START_RADIUS_MAX_M
        and value.get("equal_segment_tolerance_m")
        == _TRANSLATION_PATH_NUMERIC_TOLERANCE_M
        and value.get("rotation_path_fields_are_zero") is True
        and isinstance(value.get("translation_goal_count"), int)
        and not isinstance(value.get("translation_goal_count"), bool)
        and value["translation_goal_count"] > 0
        and isinstance(value.get("rotation_goal_count"), int)
        and not isinstance(value.get("rotation_goal_count"), bool)
        and value["rotation_goal_count"] == 0
        and isinstance(value.get("worlds"), int)
        and not isinstance(value.get("worlds"), bool)
        and value["worlds"] > 0
        and (
            expected_world_counts is None
            or value["worlds"] == sum(expected_world_counts)
        )
        and isinstance(by_object, dict)
        and set(by_object) == set(contract.OBJECTS)
    ):
        return False

    all_shape_indices = list(
        range(len(contract.TRANSLATION_PATH_SHAPES))
    )
    total_translation = 0
    total_rotation = 0
    for object_index, name in enumerate(contract.OBJECTS):
        entry = by_object.get(name)
        if not isinstance(entry, dict):
            return False
        shape_counts = entry.get("shape_counts_by_name")
        translation_count = entry.get("translation_goal_count")
        rotation_count = entry.get("rotation_goal_count")
        world_count = entry.get("world_count")
        numeric_values = (
            entry.get("path_step_min_m"),
            entry.get("path_step_max_m"),
            entry.get("path_length_min_m"),
            entry.get("path_length_max_m"),
            entry.get("maximum_waypoint_radius_m"),
            entry.get("maximum_start_radius_m"),
            entry.get("maximum_arc_step_error_m"),
            entry.get("maximum_arc_length_error_m"),
            entry.get("maximum_canonical_waypoint_error_m"),
        )
        if not isinstance(shape_counts, dict) or set(shape_counts) != set(
            contract.TRANSLATION_PATH_SHAPES
        ):
            return False
        if not all(
            isinstance(shape_counts[shape], int)
            and not isinstance(shape_counts[shape], bool)
            and shape_counts[shape] >= 0
            for shape in contract.TRANSLATION_PATH_SHAPES
        ):
            return False
        sampled_shape_indices = [
            index
            for index, shape in enumerate(contract.TRANSLATION_PATH_SHAPES)
            if shape_counts[shape] > 0
        ]
        expected_shape_indices = (
            all_shape_indices
            if require_complete_shape_coverage
            else sampled_shape_indices
        )
        expected_steps = [
            contract.TRANSLATION_PATH_STEP_M_BY_SHAPE[index]
            for index in sampled_shape_indices
        ]
        expected_lengths = [
            contract.TRANSLATION_PATH_LENGTH_M_BY_SHAPE[index]
            for index in sampled_shape_indices
        ]
        if not (
            isinstance(translation_count, int)
            and not isinstance(translation_count, bool)
            and translation_count > 0
            and isinstance(rotation_count, int)
            and not isinstance(rotation_count, bool)
            and rotation_count == 0
            and isinstance(world_count, int)
            and not isinstance(world_count, bool)
            and world_count == translation_count + rotation_count
            and (
                expected_world_counts is None
                or world_count == expected_world_counts[object_index]
            )
            and entry.get("shape_index_values") == expected_shape_indices
            and bool(sampled_shape_indices)
            and (
                not require_complete_shape_coverage
                or sampled_shape_indices == all_shape_indices
            )
            and sum(shape_counts.values()) == translation_count
            and isinstance(entry.get("seed_min"), int)
            and not isinstance(entry.get("seed_min"), bool)
            and isinstance(entry.get("seed_max"), int)
            and not isinstance(entry.get("seed_max"), bool)
            and contract.TRANSLATION_PATH_SEED_MIN
            <= entry["seed_min"]
            <= entry["seed_max"]
            <= contract.TRANSLATION_PATH_SEED_MAX
            and isinstance(entry.get("seed_unique_count"), int)
            and not isinstance(entry.get("seed_unique_count"), bool)
            and 0 < entry["seed_unique_count"] <= translation_count
            and entry.get("source_region_values") == [source_region]
            and entry.get("rotation_source_region_values") == []
            and entry.get("waypoint_count")
            == contract.TRANSLATION_PATH_WAYPOINTS
            and entry.get("path_fields_finite") is True
            and entry.get("rotation_path_fields_are_zero") is True
            and all(
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and math.isfinite(float(observed))
                for observed in numeric_values
            )
            and math.isclose(
                float(entry["path_step_min_m"]),
                min(expected_steps),
                rel_tol=0.0,
                abs_tol=_TRANSLATION_PATH_NUMERIC_TOLERANCE_M,
            )
            and math.isclose(
                float(entry["path_step_max_m"]),
                max(expected_steps),
                rel_tol=0.0,
                abs_tol=_TRANSLATION_PATH_NUMERIC_TOLERANCE_M,
            )
            and math.isclose(
                float(entry["path_length_min_m"]),
                min(expected_lengths),
                rel_tol=0.0,
                abs_tol=_TRANSLATION_PATH_NUMERIC_TOLERANCE_M,
            )
            and math.isclose(
                float(entry["path_length_max_m"]),
                max(expected_lengths),
                rel_tol=0.0,
                abs_tol=_TRANSLATION_PATH_NUMERIC_TOLERANCE_M,
            )
            and float(entry["maximum_waypoint_radius_m"])
            >= 0.0
            and float(entry["maximum_waypoint_radius_m"])
            <= contract.TRANSLATION_PATH_GENERATION_RADIUS_M
            + _TRANSLATION_PATH_NUMERIC_TOLERANCE_M
            and float(entry["maximum_start_radius_m"])
            >= 0.0
            and float(entry["maximum_start_radius_m"])
            <= contract.TRANSLATION_PATH_START_RADIUS_MAX_M
            + _TRANSLATION_PATH_NUMERIC_TOLERANCE_M
            and float(entry["maximum_arc_step_error_m"]) >= 0.0
            and float(entry["maximum_arc_step_error_m"])
            <= _TRANSLATION_PATH_NUMERIC_TOLERANCE_M
            and float(entry["maximum_arc_length_error_m"]) >= 0.0
            and float(entry["maximum_arc_length_error_m"])
            <= _TRANSLATION_PATH_NUMERIC_TOLERANCE_M
            and float(entry["maximum_canonical_waypoint_error_m"]) >= 0.0
            and float(entry["maximum_canonical_waypoint_error_m"])
            <= _TRANSLATION_PATH_NUMERIC_TOLERANCE_M
        ):
            return False
        total_translation += translation_count
        total_rotation += rotation_count
    return (
        total_translation == value["translation_goal_count"]
        and total_rotation == value["rotation_goal_count"]
        and total_translation + total_rotation == value["worlds"]
    )


def _translation_path_goal_coverage(
    torch: Any,
    environment: Any,
    phase: str,
    *,
    task: str,
    require_complete_shape_coverage: bool = True,
) -> dict[str, Any]:
    """Certify every generated translation path without global tensor copies."""

    selected_task = contract.validate_task(task)
    shards = getattr(environment, "shards", None)
    if not isinstance(shards, (list, tuple)) or len(shards) != len(
        contract.OBJECTS
    ):
        raise RuntimeError(
            "translation path coverage requires all four object shards"
        )
    world_counts = tuple(int(shard.worlds) for shard in shards)
    if selected_task == contract.TASK_ROTATE:
        rotate_index = contract.TASKS.index(contract.TASK_ROTATE)
        for name, shard in zip(contract.OBJECTS, shards):
            if not bool(
                (shard.task_index == rotate_index).all()
                and (shard.path_seed == 0).all()
                and (shard.path_shape_index == 0).all()
                and (shard.path_step_m == 0.0).all()
                and (shard.path_length_m == 0.0).all()
                and (shard.path_waypoints_xy_m == 0.0).all()
            ):
                raise RuntimeError(
                    f"{phase} {name} inactive translation coverage is invalid"
                )
        return _inactive_goal_coverage(
            coverage="translation_path",
            task=selected_task,
            phase=phase,
            world_counts=world_counts,
        )
    source_region = _translation_path_phase_contract(phase)
    result: dict[str, Any] = {
        "active": True,
        "coverage": "translation_path",
        "task": selected_task,
        "task_index": contract.TASKS.index(selected_task),
        "phase": phase,
        "path_family": contract.TRANSLATION_PATH_FAMILY,
        "segments": contract.TRANSLATION_PATH_SEGMENTS,
        "waypoints": contract.TRANSLATION_PATH_WAYPOINTS,
        "shape_names": list(contract.TRANSLATION_PATH_SHAPES),
        "expected_source_region": source_region,
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
        "waypoint_radius_limit_m": (
            contract.TRANSLATION_PATH_REGION_RADIUS_M
        ),
        "generation_radius_limit_m": (
            contract.TRANSLATION_PATH_GENERATION_RADIUS_M
        ),
        "start_radius_limit_m": (
            contract.TRANSLATION_PATH_START_RADIUS_MAX_M
        ),
        "equal_segment_tolerance_m": (
            _TRANSLATION_PATH_NUMERIC_TOLERANCE_M
        ),
        "translation_goal_count": 0,
        "rotation_goal_count": 0,
        "worlds": int(environment.worlds),
        "rotation_path_fields_are_zero": True,
        "by_object": {},
    }
    translate_index = contract.TASKS.index(contract.TASK_TRANSLATE)
    for name, shard in zip(contract.OBJECTS, shards):
        translating = shard.task_index == translate_index
        rotating = ~translating
        translation_count = int(translating.sum().detach().cpu())
        rotation_count = int(rotating.sum().detach().cpu())
        if translation_count != int(shard.worlds) or rotation_count != 0:
            raise RuntimeError(
                f"{phase} {name} capacity batch must contain only translation"
            )
        seeds = shard.path_seed[translating]
        shapes = shard.path_shape_index[translating]
        sources = shard.source_region[translating]
        steps = shard.path_step_m[translating]
        lengths = shard.path_length_m[translating]
        waypoints = shard.path_waypoints_xy_m[translating]
        if tuple(waypoints.shape[1:]) != (
            contract.TRANSLATION_PATH_WAYPOINTS,
            2,
        ):
            raise RuntimeError(
                f"{phase} {name} translation waypoint tensor has wrong shape"
            )
        waypoint_radii = waypoints.norm(dim=-1)
        step_table = torch.as_tensor(
            contract.TRANSLATION_PATH_STEP_M_BY_SHAPE,
            device=steps.device,
            dtype=steps.dtype,
        )
        length_table = torch.as_tensor(
            contract.TRANSLATION_PATH_LENGTH_M_BY_SHAPE,
            device=lengths.device,
            dtype=lengths.dtype,
        )
        waypoint_table = torch.as_tensor(
            contract.TRANSLATION_PATH_WAYPOINTS_XY_M,
            device=waypoints.device,
            dtype=waypoints.dtype,
        )
        expected_steps = step_table.index_select(0, shapes)
        expected_lengths = length_table.index_select(0, shapes)
        expected_waypoints = waypoint_table.index_select(0, shapes)
        step_error = (steps - expected_steps).abs()
        length_error = (lengths - expected_lengths).abs()
        waypoint_error = (waypoints - expected_waypoints).abs()
        unique_shapes = torch.unique(shapes, sorted=True)
        shape_counts = {
            shape_name: int(
                (shapes == shape_index).sum().detach().cpu()
            )
            for shape_index, shape_name in enumerate(
                contract.TRANSLATION_PATH_SHAPES
            )
        }
        rotation_fields_are_zero = bool(
            (shard.path_seed[rotating] == 0).all()
            and (shard.path_shape_index[rotating] == 0).all()
            and (shard.path_step_m[rotating] == 0.0).all()
            and (shard.path_length_m[rotating] == 0.0).all()
            and (shard.path_waypoints_xy_m[rotating] == 0.0).all()
        )
        entry = {
            "world_count": int(shard.worlds),
            "translation_goal_count": translation_count,
            "rotation_goal_count": rotation_count,
            "shape_index_values": [
                int(item) for item in unique_shapes.detach().cpu()
            ],
            "shape_counts_by_name": shape_counts,
            "seed_min": int(seeds.min().detach().cpu()),
            "seed_max": int(seeds.max().detach().cpu()),
            "seed_unique_count": int(torch.unique(seeds).numel()),
            "source_region_values": [
                int(item)
                for item in torch.unique(sources, sorted=True).detach().cpu()
            ],
            "rotation_source_region_values": [
                int(item)
                for item in torch.unique(
                    shard.source_region[rotating], sorted=True
                )
                .detach()
                .cpu()
            ],
            "path_step_min_m": float(steps.min().detach().cpu()),
            "path_step_max_m": float(steps.max().detach().cpu()),
            "path_length_min_m": float(lengths.min().detach().cpu()),
            "path_length_max_m": float(lengths.max().detach().cpu()),
            "waypoint_count": int(waypoints.shape[1]),
            "maximum_waypoint_radius_m": float(
                waypoint_radii.max().detach().cpu()
            ),
            "maximum_start_radius_m": float(
                waypoint_radii[:, 0].max().detach().cpu()
            ),
            "maximum_arc_step_error_m": float(
                step_error.max().detach().cpu()
            ),
            "maximum_arc_length_error_m": float(
                length_error.max().detach().cpu()
            ),
            "maximum_canonical_waypoint_error_m": float(
                waypoint_error.max().detach().cpu()
            ),
            "path_fields_finite": bool(
                torch.isfinite(steps).all()
                and torch.isfinite(lengths).all()
                and torch.isfinite(waypoints).all()
            ),
            "rotation_path_fields_are_zero": rotation_fields_are_zero,
        }
        result["by_object"][name] = entry
        result["translation_goal_count"] += translation_count
        result["rotation_goal_count"] += rotation_count
        result["rotation_path_fields_are_zero"] = bool(
            result["rotation_path_fields_are_zero"]
            and rotation_fields_are_zero
        )
    if not _translation_path_goal_coverage_is_valid(
        result,
        phase,
        task=selected_task,
        expected_world_counts=world_counts,
        require_complete_shape_coverage=require_complete_shape_coverage,
    ):
        raise RuntimeError(
            f"{phase} capacity batch violates translation path coverage"
        )
    return result


def _rotation_goal_coverage_is_valid(
    value: Any,
    phase: str,
    *,
    task: str,
    expected_world_counts: tuple[int, ...],
) -> bool:
    try:
        selected_task = contract.validate_task(task)
    except ValueError:
        return False
    if selected_task == contract.TASK_TRANSLATE:
        return _inactive_goal_coverage_is_valid(
            value,
            coverage="rotation",
            task=selected_task,
            phase=phase,
            expected_world_counts=expected_world_counts,
        )
    if phase == PHASE:
        low = contract.ROTATION_ANGLE_MIN_DEG
        high = contract.ROTATION_ANGLE_MAX_DEG
        source_region = 0
    else:
        return False
    by_object = value.get("by_object") if isinstance(value, dict) else None
    return (
        isinstance(value, dict)
        and value.get("active") is True
        and value.get("coverage") == "rotation"
        and value.get("task") == selected_task
        and value.get("task_index") == contract.TASKS.index(selected_task)
        and value.get("phase") == phase
        and value.get("worlds") == sum(expected_world_counts)
        and value.get("closed_integer_bounds_deg") == [low, high]
        and value.get("translation_angles_are_zero") is True
        and isinstance(by_object, dict)
        and set(by_object) == set(contract.OBJECTS)
        and all(
            isinstance(by_object[name], dict)
            and by_object[name].get("active") is True
            and by_object[name].get("world_count")
            == expected_world_counts[index]
            and by_object[name].get("angle_min_deg") == low
            and by_object[name].get("angle_max_deg") == high
            and by_object[name].get("angle_unique_count") == high - low + 1
            and isinstance(
                by_object[name].get("maximum_target_count"), int
            )
            and not isinstance(
                by_object[name].get("maximum_target_count"), bool
            )
            and by_object[name]["maximum_target_count"] > 0
            and by_object[name].get("source_region_values") == [source_region]
            for index, name in enumerate(contract.OBJECTS)
        )
    )


def _rotation_goal_coverage(
    torch: Any,
    goals: dict[str, Any],
    phase: str,
    *,
    task: str,
    expected_world_counts: tuple[int, ...],
) -> dict[str, Any]:
    selected_task = contract.validate_task(task)
    if selected_task == contract.TASK_TRANSLATE:
        translate_index = contract.TASKS.index(contract.TASK_TRANSLATE)
        task_index = torch.as_tensor(goals["task_index"])
        angle = torch.as_tensor(goals["rotation_angle_deg"])
        if not bool(
            (task_index == translate_index).all() and (angle == 0).all()
        ):
            raise RuntimeError(
                f"{phase} inactive rotation coverage is invalid"
            )
        return _inactive_goal_coverage(
            coverage="rotation",
            task=selected_task,
            phase=phase,
            world_counts=expected_world_counts,
        )
    if phase != PHASE:
        raise ValueError(f"unsupported rotation phase: {phase!r}")
    low, high = (
        contract.ROTATION_ANGLE_MIN_DEG,
        contract.ROTATION_ANGLE_MAX_DEG,
    )
    object_index = goals["object_index"]
    task_index = goals["task_index"]
    angle = goals["rotation_angle_deg"]
    source = goals["source_region"]
    translating = task_index == contract.TASKS.index(contract.TASK_TRANSLATE)
    rotating = ~translating
    if bool(translating.any()) or int(rotating.sum().detach().cpu()) != int(
        sum(expected_world_counts)
    ):
        raise RuntimeError(f"{phase} capacity batch must contain only rotation")
    result = {
        "active": True,
        "coverage": "rotation",
        "task": selected_task,
        "task_index": contract.TASKS.index(selected_task),
        "phase": phase,
        "worlds": int(angle.numel()),
        "closed_integer_bounds_deg": [low, high],
        "translation_angles_are_zero": bool((angle[translating] == 0).all()),
        "by_object": {},
    }
    for index, name in enumerate(contract.OBJECTS):
        selected = rotating & (object_index == index)
        values = angle[selected]
        regions = torch.unique(source[selected], sorted=True)
        world_count = int(values.numel())
        if world_count != expected_world_counts[index]:
            raise RuntimeError(
                f"{phase} {name} rotation capacity allocation mismatch"
            )
        result["by_object"][name] = {
            "active": True,
            "world_count": world_count,
            "angle_min_deg": int(values.min().detach().cpu()),
            "angle_max_deg": int(values.max().detach().cpu()),
            "angle_unique_count": int(torch.unique(values).numel()),
            "maximum_target_count": int((values == high).sum().detach().cpu()),
            "source_region_values": [int(value) for value in regions.cpu()],
        }
    if not _rotation_goal_coverage_is_valid(
        result,
        phase,
        task=selected_task,
        expected_world_counts=expected_world_counts,
    ):
        raise RuntimeError(
            f"{phase} formal batch does not cover its rotation curriculum"
        )
    return result


_CERTIFIED_VALID_DEFINITION = (
    "success & path_tracking_certified & actual_grasp_plane_compliant & "
    "!object_dropped"
)

_PRELIFT_STAGE_NAMES = ("close", "settle")
_PRELIFT_INFO_KEYS = (
    "prelift_contact_required_stable_ticks",
    "prelift_contact_stage_names",
    "prelift_contact_stage_index",
    "prelift_contact_stable_count",
    "prelift_contact_max_stable_count",
    "prelift_contact_gate_latched",
    "prelift_contact_gate_first_control_step",
    "prelift_contact_count_current",
    "prelift_tip_raw_pair_current",
    "prelift_tip_active_pair_current",
    "prelift_tip_contact_qualified_current",
    "prelift_stage_observation_count",
    "prelift_stage_qualified_count",
    "prelift_stage_failure_count",
    "prelift_stage_max_contact_count",
    "prelift_stage_first_failure_control_step",
    "prelift_stage_last_failure_control_step",
    "prelift_tip_pair_missing_count_by_stage",
    "prelift_tip_raw_pair_missing_count_by_stage",
    "prelift_tip_efc_inactive_count_by_stage",
    "prelift_tip_force_failure_count_by_stage",
    "prelift_lift_boundary_contact_valid",
    "prelift_lift_boundary_tip_raw_pair",
    "prelift_lift_boundary_tip_active_pair",
    "prelift_lift_boundary_tip_contact_qualified",
    "prelift_lift_boundary_contact_force_n",
    "prelift_lift_permitted",
    "prelift_lift_reference_valid",
    "prelift_lift_start_body_position",
    "prelift_lift_start_quaternion",
)

_TRANSLATION_ENTRY_CONTACT_LAYER_KEYS = (
    "translation_entry_tip_object_raw_pair",
    "translation_entry_tip_object_active_pair",
    "translation_entry_tip_contact_qualified",
    "translation_entry_tip_raw_pair_missing_count",
    "translation_entry_tip_efc_inactive_count",
)


def _optional_prelift_diagnostics(
    torch: Any,
    *,
    info: dict[str, Any],
    device: Any,
    world_shape: tuple[int, ...],
) -> dict[str, Any] | None:
    """Decode the optional pre-lift telemetry as an all-or-none schema."""

    present = tuple(name for name in _PRELIFT_INFO_KEYS if name in info)
    if not present:
        return None
    missing = tuple(name for name in _PRELIFT_INFO_KEYS if name not in info)
    if missing:
        raise ValueError(
            "certification funnel pre-lift diagnostic schema is incomplete; "
            f"missing={list(missing)!r}"
        )

    stage_names = tuple(
        str(name) for name in info["prelift_contact_stage_names"]
    )
    if stage_names != _PRELIFT_STAGE_NAMES:
        raise ValueError("certification funnel pre-lift stage schema is invalid")

    def tensor(name: str, *, dtype: Any | None = None) -> Any:
        return torch.as_tensor(info[name], device=device, dtype=dtype)

    values = {
        "required_ticks": tensor(
            "prelift_contact_required_stable_ticks", dtype=torch.long
        ),
        "stage_index": tensor(
            "prelift_contact_stage_index", dtype=torch.long
        ),
        "stable_count": tensor(
            "prelift_contact_stable_count", dtype=torch.long
        ),
        "max_stable_count": tensor(
            "prelift_contact_max_stable_count", dtype=torch.long
        ),
        "gate_latched": tensor(
            "prelift_contact_gate_latched", dtype=torch.bool
        ),
        "first_latch_step": tensor(
            "prelift_contact_gate_first_control_step", dtype=torch.long
        ),
        "current_contact_count": tensor(
            "prelift_contact_count_current", dtype=torch.long
        ),
        "current_tip_raw_pair": tensor(
            "prelift_tip_raw_pair_current", dtype=torch.bool
        ),
        "current_tip_active_pair": tensor(
            "prelift_tip_active_pair_current", dtype=torch.bool
        ),
        "current_tip_qualified": tensor(
            "prelift_tip_contact_qualified_current", dtype=torch.bool
        ),
        "stage_observations": tensor(
            "prelift_stage_observation_count", dtype=torch.long
        ),
        "stage_qualified": tensor(
            "prelift_stage_qualified_count", dtype=torch.long
        ),
        "stage_failures": tensor(
            "prelift_stage_failure_count", dtype=torch.long
        ),
        "stage_max_contacts": tensor(
            "prelift_stage_max_contact_count", dtype=torch.long
        ),
        "stage_first_failure": tensor(
            "prelift_stage_first_failure_control_step", dtype=torch.long
        ),
        "stage_last_failure": tensor(
            "prelift_stage_last_failure_control_step", dtype=torch.long
        ),
        "tip_pair_missing": tensor(
            "prelift_tip_pair_missing_count_by_stage", dtype=torch.long
        ),
        "tip_raw_pair_missing": tensor(
            "prelift_tip_raw_pair_missing_count_by_stage", dtype=torch.long
        ),
        "tip_efc_inactive": tensor(
            "prelift_tip_efc_inactive_count_by_stage", dtype=torch.long
        ),
        "tip_force_failures": tensor(
            "prelift_tip_force_failure_count_by_stage", dtype=torch.long
        ),
        "boundary_valid": tensor(
            "prelift_lift_boundary_contact_valid", dtype=torch.bool
        ),
        "boundary_tip_raw_pair": tensor(
            "prelift_lift_boundary_tip_raw_pair", dtype=torch.bool
        ),
        "boundary_tip_active_pair": tensor(
            "prelift_lift_boundary_tip_active_pair", dtype=torch.bool
        ),
        "boundary_tip_qualified": tensor(
            "prelift_lift_boundary_tip_contact_qualified", dtype=torch.bool
        ),
        "boundary_force": tensor(
            "prelift_lift_boundary_contact_force_n"
        ),
        "lift_permitted": tensor(
            "prelift_lift_permitted", dtype=torch.bool
        ),
        "reference_valid": tensor(
            "prelift_lift_reference_valid", dtype=torch.bool
        ),
        "start_body_position": tensor(
            "prelift_lift_start_body_position"
        ),
        "start_quaternion": tensor(
            "prelift_lift_start_quaternion"
        ),
    }

    for name in (
        "required_ticks",
        "stage_index",
        "stable_count",
        "max_stable_count",
        "gate_latched",
        "first_latch_step",
        "current_contact_count",
        "boundary_valid",
        "lift_permitted",
        "reference_valid",
    ):
        if tuple(values[name].shape) != world_shape:
            raise ValueError(
                f"certification funnel pre-lift {name} has wrong shape"
            )
    stage_shape = (*world_shape, len(stage_names))
    for name in (
        "stage_observations",
        "stage_qualified",
        "stage_failures",
        "stage_max_contacts",
        "stage_first_failure",
        "stage_last_failure",
    ):
        if tuple(values[name].shape) != stage_shape:
            raise ValueError(
                f"certification funnel pre-lift {name} has wrong shape"
            )
    tip_shape = (*world_shape, len(contract.TIP_ORDER))
    for name in (
        "current_tip_raw_pair",
        "current_tip_active_pair",
        "current_tip_qualified",
        "boundary_tip_raw_pair",
        "boundary_tip_active_pair",
        "boundary_tip_qualified",
        "boundary_force",
    ):
        if tuple(values[name].shape) != tip_shape:
            raise ValueError(
                f"certification funnel pre-lift {name} has wrong shape"
            )
    stage_tip_shape = (
        *world_shape,
        len(stage_names),
        len(contract.TIP_ORDER),
    )
    for name in (
        "tip_pair_missing",
        "tip_raw_pair_missing",
        "tip_efc_inactive",
        "tip_force_failures",
    ):
        if tuple(values[name].shape) != stage_tip_shape:
            raise ValueError(
                f"certification funnel pre-lift {name} has wrong shape"
            )
    if tuple(values["start_body_position"].shape) != (*world_shape, 3):
        raise ValueError(
            "certification funnel pre-lift start body position has wrong shape"
        )
    if tuple(values["start_quaternion"].shape) != (*world_shape, 4):
        raise ValueError(
            "certification funnel pre-lift start quaternion has wrong shape"
        )

    required_ticks = values["required_ticks"]
    required_unique = torch.unique(required_ticks)
    if required_unique.numel() != 1 or int(required_unique[0].item()) <= 0:
        raise ValueError(
            "certification funnel pre-lift stable-tick contract is invalid"
        )
    integer_counts = (
        values["stable_count"],
        values["max_stable_count"],
        values["current_contact_count"],
        values["stage_observations"],
        values["stage_qualified"],
        values["stage_failures"],
        values["stage_max_contacts"],
        values["tip_pair_missing"],
        values["tip_raw_pair_missing"],
        values["tip_efc_inactive"],
        values["tip_force_failures"],
    )
    if not all(bool((item >= 0).all()) for item in integer_counts):
        raise ValueError("certification funnel pre-lift counts are invalid")
    if not bool(
        ((values["stage_index"] >= -1) & (values["stage_index"] < 2)).all()
        and (values["stable_count"] <= required_ticks).all()
        and (values["max_stable_count"] <= required_ticks).all()
        and (
            values["max_stable_count"] >= values["stable_count"]
        ).all()
        and (values["current_contact_count"] <= len(contract.TIP_ORDER)).all()
        and (
            values["stage_max_contacts"] <= len(contract.TIP_ORDER)
        ).all()
    ):
        raise ValueError(
            "certification funnel pre-lift scalar state is inconsistent"
        )
    if not bool(
        (
            (~values["current_tip_active_pair"])
            | values["current_tip_raw_pair"]
        ).all()
        and (
            (~values["current_tip_qualified"])
            | values["current_tip_active_pair"]
        ).all()
        and (
            values["current_tip_qualified"].to(torch.long).sum(dim=1)
            == values["current_contact_count"]
        ).all()
        and (
            values["stage_qualified"] + values["stage_failures"]
            == values["stage_observations"]
        ).all()
        and (
            values["tip_pair_missing"]
            == values["tip_raw_pair_missing"]
            + values["tip_efc_inactive"]
        ).all()
        and (
            values["tip_pair_missing"] + values["tip_force_failures"]
            <= values["stage_observations"][:, :, None]
        ).all()
    ):
        raise ValueError(
            "certification funnel pre-lift contact accounting is inconsistent"
        )
    failed = values["stage_failures"] > 0
    failure_sentinels_valid = torch.where(
        failed,
        (values["stage_first_failure"] >= 0)
        & (
            values["stage_last_failure"]
            >= values["stage_first_failure"]
        ),
        (values["stage_first_failure"] == -1)
        & (values["stage_last_failure"] == -1),
    )
    latch_sentinel_valid = torch.where(
        values["gate_latched"],
        values["first_latch_step"] >= 0,
        values["first_latch_step"] == -1,
    )
    boundary_qualified_count = values["boundary_tip_qualified"].to(
        torch.long
    ).sum(dim=1)
    if not bool(
        failure_sentinels_valid.all()
        and latch_sentinel_valid.all()
        and (
            (~values["boundary_tip_active_pair"])
            | values["boundary_tip_raw_pair"]
        ).all()
        and (
            (~values["boundary_tip_qualified"])
            | values["boundary_tip_active_pair"]
        ).all()
        and (
            ~values["boundary_valid"]
            | (
                values["gate_latched"]
                & (boundary_qualified_count >= contract.EXPERT_MIN_CONTACTS)
            )
        ).all()
        and (
            values["lift_permitted"]
            == (values["boundary_valid"] & values["reference_valid"])
        ).all()
    ):
        raise ValueError(
            "certification funnel pre-lift gate state is inconsistent"
        )
    if not bool(
        torch.isfinite(values["boundary_force"]).all()
        and (values["boundary_force"] >= 0.0).all()
        and torch.isfinite(values["start_body_position"]).all()
        and torch.isfinite(values["start_quaternion"]).all()
    ):
        raise ValueError("certification funnel pre-lift state is non-finite")

    values["stage_names"] = stage_names
    values["required_tick_count"] = int(required_unique[0].item())
    return values


def _optional_translation_entry_contact_layers(
    torch: Any,
    *,
    info: dict[str, Any],
    device: Any,
    world_shape: tuple[int, ...],
    legacy_active_pair: Any,
    legacy_force_valid: Any,
    legacy_pair_missing_count: Any,
    force_failure_count: Any,
) -> dict[str, Any] | None:
    """Decode optional raw-contact/EFC/force entry-failure telemetry."""

    present = tuple(
        name for name in _TRANSLATION_ENTRY_CONTACT_LAYER_KEYS if name in info
    )
    if not present:
        return None
    missing = tuple(
        name
        for name in _TRANSLATION_ENTRY_CONTACT_LAYER_KEYS
        if name not in info
    )
    if missing:
        raise ValueError(
            "certification funnel translation-entry contact-layer schema is "
            f"incomplete; missing={list(missing)!r}"
        )

    tip_shape = (*world_shape, len(contract.TIP_ORDER))

    def tensor(name: str, *, dtype: Any) -> Any:
        value = torch.as_tensor(info[name], device=device, dtype=dtype)
        if tuple(value.shape) != tip_shape:
            raise ValueError(
                "certification funnel translation-entry contact-layer "
                f"{name} has wrong shape"
            )
        return value

    values = {
        "raw_pair": tensor(
            "translation_entry_tip_object_raw_pair", dtype=torch.bool
        ),
        "active_pair": tensor(
            "translation_entry_tip_object_active_pair", dtype=torch.bool
        ),
        "qualified": tensor(
            "translation_entry_tip_contact_qualified", dtype=torch.bool
        ),
        "raw_pair_missing_count": tensor(
            "translation_entry_tip_raw_pair_missing_count", dtype=torch.long
        ),
        "efc_inactive_count": tensor(
            "translation_entry_tip_efc_inactive_count", dtype=torch.long
        ),
    }
    if not bool(
        (values["raw_pair_missing_count"] >= 0).all()
        and (values["efc_inactive_count"] >= 0).all()
        and (
            (~values["active_pair"]) | values["raw_pair"]
        ).all()
        and (
            (~values["qualified"]) | values["active_pair"]
        ).all()
        and (values["active_pair"] == legacy_active_pair).all()
        and (
            values["qualified"]
            == (values["active_pair"] & legacy_force_valid)
        ).all()
        and (
            legacy_pair_missing_count
            == values["raw_pair_missing_count"]
            + values["efc_inactive_count"]
        ).all()
        and (force_failure_count >= 0).all()
    ):
        raise ValueError(
            "certification funnel translation-entry contact-layer state is "
            "inconsistent"
        )
    return values


def _certification_funnel(
    torch: Any,
    *,
    config: Any,
    info: dict[str, Any],
    actual_compliance: Any,
    completed_episode_controller_counters: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe endpoint/route/plane/drop diagnostic funnel."""

    endpoint = info["success"].to(torch.bool)
    route = info["path_tracking_certified"].to(torch.bool)
    plane = torch.as_tensor(
        actual_compliance, device=endpoint.device
    ).to(torch.bool)
    not_dropped = ~info["object_dropped"].to(torch.bool)
    object_index = info["object_index"].to(torch.long)
    path_index = info["path_shape_index"].to(torch.long)
    position_axis_abs = info["final_position_error_xyz_m"].abs()
    euler_axis_abs = info["final_euler_error_xyz_deg"].abs()
    endpoint_object_velocity = torch.as_tensor(
        info["velocity"], device=endpoint.device
    )
    tracking_axis_abs_max = info["path_tracking_axis_abs_max_m"].abs()
    tracking_rmse = info["path_tracking_rmse_m"].abs()
    tracking_sample_count = torch.as_tensor(
        info["path_tracking_sample_count"],
        device=endpoint.device,
        dtype=torch.long,
    )
    schedule_progress = torch.as_tensor(
        info["path_schedule_progress"], device=endpoint.device
    )
    schedule_limit_valid = torch.as_tensor(
        info["path_schedule_limit_valid"],
        device=endpoint.device,
        dtype=torch.bool,
    )
    schedule_complete = torch.as_tensor(
        info["path_schedule_complete"],
        device=endpoint.device,
        dtype=torch.bool,
    )
    schedule_max_speed = torch.as_tensor(
        info["path_schedule_max_speed_m_s"], device=endpoint.device
    )
    schedule_max_acceleration = torch.as_tensor(
        info["path_schedule_max_acceleration_m_s2"], device=endpoint.device
    )
    first_violation_sample_tick = torch.as_tensor(
        info["path_tracking_first_violation_sample_tick"],
        device=endpoint.device,
        dtype=torch.long,
    )
    first_violation_control_tick = torch.as_tensor(
        info["path_tracking_first_violation_control_tick"],
        device=endpoint.device,
        dtype=torch.long,
    )
    first_violation_axis_index = torch.as_tensor(
        info["path_tracking_first_violation_axis_index"],
        device=endpoint.device,
        dtype=torch.long,
    )
    first_violation_error_xyz_abs_m = torch.as_tensor(
        info["path_tracking_first_violation_error_xyz_abs_m"],
        device=endpoint.device,
    )
    entry_predicate_names = tuple(
        str(name) for name in info["translation_entry_predicate_names"]
    )
    expected_entry_predicates = (
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
    if entry_predicate_names != expected_entry_predicates:
        raise ValueError(
            "certification funnel translation-entry predicate schema "
            "does not match the runtime contract"
        )
    entry_latched = torch.as_tensor(
        info["translation_entry_certificate_latched"],
        device=endpoint.device,
        dtype=torch.bool,
    )
    entry_first_latch_step = torch.as_tensor(
        info["translation_entry_certificate_first_control_step"],
        device=endpoint.device,
        dtype=torch.long,
    )
    entry_max_stable_count = torch.as_tensor(
        info["translation_entry_max_stable_count"],
        device=endpoint.device,
        dtype=torch.long,
    )
    entry_route_admitted = torch.as_tensor(
        info["translation_route_admitted"],
        device=endpoint.device,
        dtype=torch.bool,
    )
    entry_route_start_safe = torch.as_tensor(
        info["translation_entry_route_start_safety_valid"],
        device=endpoint.device,
        dtype=torch.bool,
    )
    entry_predicate_current = torch.as_tensor(
        info["translation_entry_predicate_current"],
        device=endpoint.device,
        dtype=torch.bool,
    )
    entry_failure_counts = torch.as_tensor(
        info["translation_entry_failure_count_by_predicate"],
        device=endpoint.device,
        dtype=torch.long,
    )
    entry_first_failure_step = torch.as_tensor(
        info["translation_entry_first_failure_control_step"],
        device=endpoint.device,
        dtype=torch.long,
    )
    entry_last_failure_step = torch.as_tensor(
        info["translation_entry_last_failure_control_step"],
        device=endpoint.device,
        dtype=torch.long,
    )
    entry_tip_height_fraction = torch.as_tensor(
        info["translation_entry_tip_height_fraction"],
        device=endpoint.device,
    )
    entry_tip_pair = torch.as_tensor(
        info["translation_entry_tip_object_pair"],
        device=endpoint.device,
        dtype=torch.bool,
    )
    entry_tip_force = torch.as_tensor(
        info["translation_entry_tip_contact_force_n"],
        device=endpoint.device,
    )
    entry_tip_force_valid = torch.as_tensor(
        info["translation_entry_tip_force_valid"],
        device=endpoint.device,
        dtype=torch.bool,
    )
    entry_tip_height_failures = torch.as_tensor(
        info["translation_entry_tip_height_failure_count"],
        device=endpoint.device,
        dtype=torch.long,
    )
    entry_tip_pair_failures = torch.as_tensor(
        info["translation_entry_tip_pair_missing_count"],
        device=endpoint.device,
        dtype=torch.long,
    )
    entry_tip_force_failures = torch.as_tensor(
        info["translation_entry_tip_force_failure_count"],
        device=endpoint.device,
        dtype=torch.long,
    )
    entry_position_error = torch.as_tensor(
        info["translation_entry_position_error_xyz_abs_m"],
        device=endpoint.device,
    )
    entry_euler_error = torch.as_tensor(
        info["translation_entry_euler_error_xyz_abs_deg"],
        device=endpoint.device,
    )
    entry_bottom_z = torch.as_tensor(
        info["translation_entry_estimated_bottom_z_m"],
        device=endpoint.device,
    )
    entry_linear_speed = torch.as_tensor(
        info["translation_entry_object_linear_speed_m_s"],
        device=endpoint.device,
    )
    entry_angular_speed = torch.as_tensor(
        info["translation_entry_object_angular_speed_rad_s"],
        device=endpoint.device,
    )
    ik_stage_names = tuple(str(name) for name in info["ik_trust_stage_names"])
    if ik_stage_names != ("grasp", "lift", "path_start", "route"):
        raise ValueError("certification funnel IK-stage schema is invalid")
    ik_trust_by_stage = torch.as_tensor(
        info["ik_trust_region_projection_count_by_stage"],
        device=endpoint.device,
        dtype=torch.long,
    )
    ik_cage_by_stage = torch.as_tensor(
        info["ik_cage_projection_count_by_stage"],
        device=endpoint.device,
        dtype=torch.long,
    )
    ik_cage_scale_min = torch.as_tensor(
        info["ik_cage_projection_scale_min"], device=endpoint.device
    )
    ik_cage_scale_mean = torch.as_tensor(
        info["ik_cage_projection_scale_mean"], device=endpoint.device
    )
    expected = tuple(endpoint.shape)
    if expected != (int(config.num_envs),):
        raise ValueError("certification funnel endpoint mask has wrong shape")
    prelift = _optional_prelift_diagnostics(
        torch,
        info=info,
        device=endpoint.device,
        world_shape=expected,
    )
    entry_contact_layers = _optional_translation_entry_contact_layers(
        torch,
        info=info,
        device=endpoint.device,
        world_shape=expected,
        legacy_active_pair=entry_tip_pair,
        legacy_force_valid=entry_tip_force_valid,
        legacy_pair_missing_count=entry_tip_pair_failures,
        force_failure_count=entry_tip_force_failures,
    )
    for name, value in (
        ("route", route),
        ("plane", plane),
        ("not_dropped", not_dropped),
        ("object_index", object_index),
        ("path_shape_index", path_index),
        ("path_tracking_rmse_m", tracking_rmse),
        ("path_tracking_sample_count", tracking_sample_count),
        ("path_schedule_progress", schedule_progress),
        ("path_schedule_limit_valid", schedule_limit_valid),
        ("path_schedule_complete", schedule_complete),
        ("path_schedule_max_speed_m_s", schedule_max_speed),
        (
            "path_schedule_max_acceleration_m_s2",
            schedule_max_acceleration,
        ),
        (
            "path_tracking_first_violation_sample_tick",
            first_violation_sample_tick,
        ),
        (
            "path_tracking_first_violation_control_tick",
            first_violation_control_tick,
        ),
        (
            "path_tracking_first_violation_axis_index",
            first_violation_axis_index,
        ),
        ("translation_entry_latched", entry_latched),
        ("translation_entry_first_latch_step", entry_first_latch_step),
        ("translation_entry_max_stable_count", entry_max_stable_count),
        ("translation_route_admitted", entry_route_admitted),
        ("translation_entry_route_start_safe", entry_route_start_safe),
        ("translation_entry_bottom_z_m", entry_bottom_z),
        ("translation_entry_linear_speed_m_s", entry_linear_speed),
        ("translation_entry_angular_speed_rad_s", entry_angular_speed),
        ("ik_cage_projection_scale_min", ik_cage_scale_min),
        ("ik_cage_projection_scale_mean", ik_cage_scale_mean),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(f"certification funnel {name} has wrong shape")
    for name, value in (
        ("endpoint_position_axis_abs_m", position_axis_abs),
        ("endpoint_euler_axis_abs_deg", euler_axis_abs),
        ("path_tracking_axis_abs_max_m", tracking_axis_abs_max),
        (
            "path_tracking_first_violation_error_xyz_abs_m",
            first_violation_error_xyz_abs_m,
        ),
        ("translation_entry_position_error_xyz_abs_m", entry_position_error),
        ("translation_entry_euler_error_xyz_abs_deg", entry_euler_error),
    ):
        if tuple(value.shape) != (*expected, 3):
            raise ValueError(f"certification funnel {name} has wrong shape")
    for name, value, width in (
        (
            "translation_entry_predicate_current",
            entry_predicate_current,
            len(entry_predicate_names),
        ),
        (
            "translation_entry_failure_count_by_predicate",
            entry_failure_counts,
            len(entry_predicate_names),
        ),
        (
            "translation_entry_first_failure_control_step",
            entry_first_failure_step,
            len(entry_predicate_names),
        ),
        (
            "translation_entry_last_failure_control_step",
            entry_last_failure_step,
            len(entry_predicate_names),
        ),
        ("translation_entry_tip_height_fraction", entry_tip_height_fraction, 4),
        ("translation_entry_tip_object_pair", entry_tip_pair, 4),
        ("translation_entry_tip_contact_force_n", entry_tip_force, 4),
        ("translation_entry_tip_force_valid", entry_tip_force_valid, 4),
        (
            "translation_entry_tip_height_failure_count",
            entry_tip_height_failures,
            4,
        ),
        (
            "translation_entry_tip_pair_missing_count",
            entry_tip_pair_failures,
            4,
        ),
        (
            "translation_entry_tip_force_failure_count",
            entry_tip_force_failures,
            4,
        ),
        ("ik_trust_region_projection_count_by_stage", ik_trust_by_stage, 4),
        ("ik_cage_projection_count_by_stage", ik_cage_by_stage, 4),
    ):
        if tuple(value.shape) != (*expected, width):
            raise ValueError(f"certification funnel {name} has wrong shape")
    if tuple(endpoint_object_velocity.shape) != (*expected, 6):
        raise ValueError(
            "certification funnel endpoint object velocity has wrong shape"
        )
    if not bool(torch.isfinite(endpoint_object_velocity).all()):
        raise ValueError(
            "certification funnel endpoint object velocity is non-finite"
        )
    endpoint_object_linear_speed = endpoint_object_velocity[:, :3].norm(
        dim=1
    )
    endpoint_object_angular_speed = endpoint_object_velocity[:, 3:].norm(
        dim=1
    )

    task = contract.validate_task(config.task)
    translating = task == contract.TASK_TRANSLATE
    route_samples = contract.TRANSLATION_PATH_ROUTE_SAMPLES
    schedule_epsilon = 1.0e-6
    exact_sample_count = tracking_sample_count == route_samples
    schedule_state_finite = (
        torch.isfinite(schedule_progress)
        & torch.isfinite(schedule_max_speed)
        & torch.isfinite(schedule_max_acceleration)
    )
    schedule_state_nonnegative = (
        (tracking_sample_count >= 0)
        & (schedule_progress >= 0.0)
        & (schedule_max_speed >= 0.0)
        & (schedule_max_acceleration >= 0.0)
    )
    if translating:
        expected_schedule_complete = (
            exact_sample_count
            & schedule_limit_valid
            & ((schedule_progress - 1.0).abs() <= schedule_epsilon)
        )
        schedule_state_valid = (
            schedule_state_finite
            & schedule_state_nonnegative
            & (tracking_sample_count <= route_samples)
            & (schedule_progress <= 1.0 + schedule_epsilon)
            & (schedule_complete == expected_schedule_complete)
            & (
                ~schedule_limit_valid
                | (
                    schedule_max_speed
                    <= contract.TRANSLATION_ROUTE_HARD_MAX_SPEED_M_S
                    + schedule_epsilon
                )
            )
            & (
                ~schedule_limit_valid
                | (
                    schedule_max_acceleration
                    <= contract.TRANSLATION_ROUTE_HARD_MAX_ACCELERATION_M_S2
                    + schedule_epsilon
                )
            )
        )
        expected_route = (
            exact_sample_count
            & schedule_complete
            & schedule_limit_valid
            & torch.isfinite(tracking_axis_abs_max).all(dim=1)
            & (
                tracking_axis_abs_max
                <= contract.TRANSLATION_PATH_TRACKING_AXIS_ABS_TOLERANCE_M
            ).all(dim=1)
        )
    else:
        schedule_state_valid = (
            schedule_state_finite
            & (tracking_sample_count == 0)
            & (schedule_progress == 0.0)
            & schedule_limit_valid
            & (~schedule_complete)
            & (schedule_max_speed == 0.0)
            & (schedule_max_acceleration == 0.0)
        )
        expected_route = torch.ones_like(route)
    if not bool(schedule_state_valid.all()):
        raise ValueError(
            "certification funnel path-schedule state is invalid"
        )
    if not bool((route == expected_route).all()):
        raise ValueError(
            "certification funnel route certificate disagrees with its "
            "sample/schedule/tracking evidence"
        )

    route_control_tick_offset = (
        int(config.plan_ticks)
        + int(config.close_ticks)
        + int(config.settle_ticks)
        + int(config.lift_ticks)
        + int(config.hold_ticks)
        + int(config.translation_path_lead_in_ticks)
        + contract.TRANSLATION_PATH_START_SETTLE_TICKS
    )
    first_violation = first_violation_sample_tick >= 1
    violation_state_valid = (
        torch.isfinite(first_violation_error_xyz_abs_m).all(dim=1)
        & (first_violation_error_xyz_abs_m >= 0.0).all(dim=1)
        & torch.where(
            first_violation,
            translating
            & (first_violation_sample_tick <= route_samples)
            & (
                first_violation_control_tick
                == route_control_tick_offset + first_violation_sample_tick
            )
            & (first_violation_axis_index >= 0)
            & (first_violation_axis_index < 3)
            & (
                first_violation_error_xyz_abs_m.amax(dim=1)
                > contract.TRANSLATION_PATH_TRACKING_AXIS_ABS_TOLERANCE_M
            )
            & (
                first_violation_axis_index
                == first_violation_error_xyz_abs_m.argmax(dim=1)
            ),
            (first_violation_sample_tick == -1)
            & (first_violation_control_tick == -1)
            & (first_violation_axis_index == -1)
            & (first_violation_error_xyz_abs_m == 0.0).all(dim=1),
        )
    )
    if not bool(violation_state_valid.all()):
        raise ValueError(
            "certification funnel first-route-violation sentinel is invalid"
        )
    valid = endpoint & route & plane & not_dropped

    def quantiles(values: Any, selected: Any) -> dict[str, float | None]:
        chosen = values[selected].to(torch.float64)
        if chosen.numel() == 0:
            return {"p50": None, "p90": None, "p95": None, "max": None}
        levels = torch.tensor(
            (0.50, 0.90, 0.95),
            device=chosen.device,
            dtype=chosen.dtype,
        )
        measured = torch.quantile(chosen, levels).detach().cpu().tolist()
        return {
            "p50": float(measured[0]),
            "p90": float(measured[1]),
            "p95": float(measured[2]),
            "max": float(chosen.max().detach().cpu()),
        }

    def axis_summary(values: Any, selected: Any) -> dict[str, Any]:
        labels = ("x", "y", "z")
        result = {
            label: quantiles(values[:, axis], selected)
            for axis, label in enumerate(labels)
        }
        result["linf"] = quantiles(values.max(dim=-1).values, selected)
        return result

    def first_violation_summary(selected: Any) -> dict[str, Any]:
        selected_violation = selected & first_violation
        return {
            "count": int(selected_violation.sum().detach().cpu()),
            "sample_tick": quantiles(
                first_violation_sample_tick, selected_violation
            ),
            "control_tick": quantiles(
                first_violation_control_tick, selected_violation
            ),
            "axis_counts": {
                label: int(
                    (
                        selected_violation
                        & (first_violation_axis_index == axis)
                    )
                    .sum()
                    .detach()
                    .cpu()
                )
                for axis, label in enumerate(("x", "y", "z"))
            },
            "error_xyz_abs_m": axis_summary(
                first_violation_error_xyz_abs_m, selected_violation
            ),
        }

    def schedule_summary(selected: Any) -> dict[str, Any]:
        return {
            "expected_sample_count": route_samples,
            "exact_sample_count_worlds": int(
                (selected & exact_sample_count).sum().detach().cpu()
            ),
            "complete_worlds": int(
                (selected & schedule_complete).sum().detach().cpu()
            ),
            "limit_valid_worlds": int(
                (selected & schedule_limit_valid).sum().detach().cpu()
            ),
            "progress": quantiles(schedule_progress, selected),
            "peak_speed_m_s": quantiles(schedule_max_speed, selected),
            "peak_acceleration_m_s2": quantiles(
                schedule_max_acceleration, selected
            ),
        }

    def translation_entry_summary(selected: Any) -> dict[str, Any]:
        qualified_tip = entry_tip_pair & entry_tip_force_valid
        qualified_count = qualified_tip.to(torch.long).sum(dim=1)
        if entry_contact_layers is None:
            contact_layers: dict[str, Any] = {
                "available": False,
                "reason": "contact_layer_diagnostics_not_emitted",
            }
        else:
            raw_pair = entry_contact_layers["raw_pair"]
            active_pair = entry_contact_layers["active_pair"]
            decoded_qualified = entry_contact_layers["qualified"]

            def count_distribution(mask: Any) -> dict[str, int]:
                counts = mask.to(torch.long).sum(dim=1)
                return {
                    str(count): int(
                        (selected & (counts == count)).sum().detach().cpu()
                    )
                    for count in range(len(contract.TIP_ORDER) + 1)
                }

            contact_layers = {
                "available": True,
                "route_start_raw_pair_count_distribution": (
                    count_distribution(raw_pair)
                ),
                "route_start_active_efc_pair_count_distribution": (
                    count_distribution(active_pair)
                ),
                "route_start_force_qualified_count_distribution": (
                    count_distribution(decoded_qualified)
                ),
                "tips": {
                    tip_name: {
                        "route_start_raw_pair_present_count": int(
                            (selected & raw_pair[:, tip_index])
                            .sum().detach().cpu()
                        ),
                        "route_start_active_efc_pair_present_count": int(
                            (selected & active_pair[:, tip_index])
                            .sum().detach().cpu()
                        ),
                        "route_start_force_qualified_count": int(
                            (selected & decoded_qualified[:, tip_index])
                            .sum().detach().cpu()
                        ),
                        "raw_pair_missing_tick_total": int(
                            entry_contact_layers["raw_pair_missing_count"][
                                selected, tip_index
                            ].sum().detach().cpu()
                        ),
                        "raw_pair_missing_ticks": quantiles(
                            entry_contact_layers["raw_pair_missing_count"][
                                :, tip_index
                            ],
                            selected,
                        ),
                        "efc_inactive_tick_total": int(
                            entry_contact_layers["efc_inactive_count"][
                                selected, tip_index
                            ].sum().detach().cpu()
                        ),
                        "efc_inactive_ticks": quantiles(
                            entry_contact_layers["efc_inactive_count"][
                                :, tip_index
                            ],
                            selected,
                        ),
                        "force_failure_tick_total": int(
                            entry_tip_force_failures[selected, tip_index]
                            .sum().detach().cpu()
                        ),
                        "force_failure_ticks": quantiles(
                            entry_tip_force_failures[:, tip_index], selected
                        ),
                    }
                    for tip_index, tip_name in enumerate(contract.TIP_ORDER)
                },
            }
        predicate_summary = {}
        for axis, name in enumerate(entry_predicate_names):
            failed = selected & (entry_failure_counts[:, axis] > 0)
            first_failed = selected & (entry_first_failure_step[:, axis] >= 0)
            predicate_summary[name] = {
                "route_start_pass_count": int(
                    (selected & entry_predicate_current[:, axis])
                    .sum().detach().cpu()
                ),
                "failed_world_count": int(failed.sum().detach().cpu()),
                "failure_tick_total": int(
                    entry_failure_counts[selected, axis].sum().detach().cpu()
                ),
                "failure_ticks": quantiles(
                    entry_failure_counts[:, axis], selected
                ),
                "first_failure_control_step": quantiles(
                    entry_first_failure_step[:, axis], first_failed
                ),
                "last_failure_control_step": quantiles(
                    entry_last_failure_step[:, axis], first_failed
                ),
            }
        tip_summary = {}
        for axis, name in enumerate(contract.TIP_ORDER):
            tip_summary[name] = {
                "route_start_height_fraction": quantiles(
                    entry_tip_height_fraction[:, axis], selected
                ),
                "route_start_pair_present_count": int(
                    (selected & entry_tip_pair[:, axis]).sum().detach().cpu()
                ),
                "route_start_force_valid_count": int(
                    (selected & entry_tip_force_valid[:, axis])
                    .sum().detach().cpu()
                ),
                "route_start_qualified_contact_count": int(
                    (selected & qualified_tip[:, axis]).sum().detach().cpu()
                ),
                "route_start_force_n": quantiles(
                    entry_tip_force[:, axis], selected
                ),
                "height_failure_ticks": quantiles(
                    entry_tip_height_failures[:, axis], selected
                ),
                "pair_missing_ticks": quantiles(
                    entry_tip_pair_failures[:, axis], selected
                ),
                "force_failure_ticks": quantiles(
                    entry_tip_force_failures[:, axis], selected
                ),
            }
        return {
            "certificate_latched_count": int(
                (selected & entry_latched).sum().detach().cpu()
            ),
            "route_start_safety_valid_count": int(
                (selected & entry_route_start_safe).sum().detach().cpu()
            ),
            "route_admitted_count": int(
                (selected & entry_route_admitted).sum().detach().cpu()
            ),
            "maximum_consecutive_stable_ticks": quantiles(
                entry_max_stable_count, selected
            ),
            "first_certificate_control_step": quantiles(
                entry_first_latch_step,
                selected & (entry_first_latch_step >= 0),
            ),
            "qualified_contact_count_distribution": {
                str(count): int(
                    (selected & (qualified_count == count)).sum().detach().cpu()
                )
                for count in range(5)
            },
            "position_error_xyz_abs_m": axis_summary(
                entry_position_error, selected
            ),
            "euler_error_xyz_abs_deg": axis_summary(
                entry_euler_error, selected
            ),
            "estimated_bottom_z_m": quantiles(entry_bottom_z, selected),
            "table_clearance_m": quantiles(
                entry_bottom_z - contract.TABLE_TOP_Z_M, selected
            ),
            "object_linear_speed_m_s": quantiles(
                entry_linear_speed, selected
            ),
            "object_angular_speed_rad_s": quantiles(
                entry_angular_speed, selected
            ),
            "contact_layers": contact_layers,
            "predicates": predicate_summary,
            "tips": tip_summary,
        }

    def prelift_summary(selected: Any) -> dict[str, Any]:
        if prelift is None:
            return {
                "available": False,
                "reason": "prelift_diagnostics_not_emitted",
            }

        stage_names = prelift["stage_names"]
        stage_observations = prelift["stage_observations"]
        stage_qualified = prelift["stage_qualified"]
        stage_failures = prelift["stage_failures"]
        stage_max_contacts = prelift["stage_max_contacts"]
        stage_first_failure = prelift["stage_first_failure"]
        stage_last_failure = prelift["stage_last_failure"]
        tip_pair_missing = prelift["tip_pair_missing"]
        tip_raw_pair_missing = prelift["tip_raw_pair_missing"]
        tip_efc_inactive = prelift["tip_efc_inactive"]
        tip_force_failures = prelift["tip_force_failures"]
        boundary_tip_raw_pair = prelift["boundary_tip_raw_pair"]
        boundary_tip_active_pair = prelift["boundary_tip_active_pair"]
        boundary_tip_qualified = prelift["boundary_tip_qualified"]
        boundary_force = prelift["boundary_force"]
        stage_summary: dict[str, Any] = {}
        for stage_index, stage_name in enumerate(stage_names):
            failed = selected & (stage_failures[:, stage_index] > 0)
            first_failed = selected & (
                stage_first_failure[:, stage_index] >= 0
            )
            stage_summary[stage_name] = {
                "observation_tick_total": int(
                    stage_observations[selected, stage_index]
                    .sum().detach().cpu()
                ),
                "observation_ticks": quantiles(
                    stage_observations[:, stage_index], selected
                ),
                "qualified_tick_total": int(
                    stage_qualified[selected, stage_index]
                    .sum().detach().cpu()
                ),
                "qualified_ticks": quantiles(
                    stage_qualified[:, stage_index], selected
                ),
                "failure_tick_total": int(
                    stage_failures[selected, stage_index]
                    .sum().detach().cpu()
                ),
                "failure_ticks": quantiles(
                    stage_failures[:, stage_index], selected
                ),
                "failed_world_count": int(failed.sum().detach().cpu()),
                "maximum_qualified_contact_count": quantiles(
                    stage_max_contacts[:, stage_index], selected
                ),
                "first_failure_control_step": quantiles(
                    stage_first_failure[:, stage_index], first_failed
                ),
                "last_failure_control_step": quantiles(
                    stage_last_failure[:, stage_index], first_failed
                ),
                "tips": {
                    tip_name: {
                        "pair_missing_tick_total": int(
                            tip_pair_missing[
                                selected, stage_index, tip_index
                            ].sum().detach().cpu()
                        ),
                        "pair_missing_ticks": quantiles(
                            tip_pair_missing[:, stage_index, tip_index],
                            selected,
                        ),
                        "raw_pair_missing_tick_total": int(
                            tip_raw_pair_missing[
                                selected, stage_index, tip_index
                            ].sum().detach().cpu()
                        ),
                        "raw_pair_missing_ticks": quantiles(
                            tip_raw_pair_missing[:, stage_index, tip_index],
                            selected,
                        ),
                        "efc_inactive_tick_total": int(
                            tip_efc_inactive[
                                selected, stage_index, tip_index
                            ].sum().detach().cpu()
                        ),
                        "efc_inactive_ticks": quantiles(
                            tip_efc_inactive[:, stage_index, tip_index],
                            selected,
                        ),
                        "force_failure_tick_total": int(
                            tip_force_failures[
                                selected, stage_index, tip_index
                            ].sum().detach().cpu()
                        ),
                        "force_failure_ticks": quantiles(
                            tip_force_failures[:, stage_index, tip_index],
                            selected,
                        ),
                    }
                    for tip_index, tip_name in enumerate(contract.TIP_ORDER)
                },
            }
        boundary_raw_count = boundary_tip_raw_pair.to(torch.long).sum(dim=1)
        boundary_active_count = boundary_tip_active_pair.to(torch.long).sum(
            dim=1
        )
        boundary_count = boundary_tip_qualified.to(torch.long).sum(dim=1)
        return {
            "available": True,
            "required_consecutive_ticks_at_lift_boundary": prelift[
                "required_tick_count"
            ],
            "gate_latched_count": int(
                (selected & prelift["gate_latched"]).sum().detach().cpu()
            ),
            "lift_boundary_contact_valid_count": int(
                (selected & prelift["boundary_valid"]).sum().detach().cpu()
            ),
            "lift_permitted_count": int(
                (selected & prelift["lift_permitted"]).sum().detach().cpu()
            ),
            "lift_reference_valid_count": int(
                (selected & prelift["reference_valid"]).sum().detach().cpu()
            ),
            "current_consecutive_contact_ticks": quantiles(
                prelift["stable_count"], selected
            ),
            "maximum_consecutive_contact_ticks": quantiles(
                prelift["max_stable_count"], selected
            ),
            "first_gate_control_step": quantiles(
                prelift["first_latch_step"],
                selected & (prelift["first_latch_step"] >= 0),
            ),
            "last_observed_stage_count": {
                "not_observed": int(
                    (selected & (prelift["stage_index"] == -1))
                    .sum().detach().cpu()
                ),
                **{
                    stage_name: int(
                        (selected & (prelift["stage_index"] == stage_index))
                        .sum().detach().cpu()
                    )
                    for stage_index, stage_name in enumerate(stage_names)
                },
            },
            "current_contact_count_distribution": {
                str(count): int(
                    (
                        selected
                        & (prelift["current_contact_count"] == count)
                    ).sum().detach().cpu()
                )
                for count in range(len(contract.TIP_ORDER) + 1)
            },
            "current_tips": {
                tip_name: {
                    "raw_pair_present_count": int(
                        (
                            selected
                            & prelift["current_tip_raw_pair"][:, tip_index]
                        ).sum().detach().cpu()
                    ),
                    "active_efc_pair_present_count": int(
                        (
                            selected
                            & prelift["current_tip_active_pair"][:, tip_index]
                        ).sum().detach().cpu()
                    ),
                    "qualified_count": int(
                        (
                            selected
                            & prelift["current_tip_qualified"][:, tip_index]
                        ).sum().detach().cpu()
                    )
                }
                for tip_index, tip_name in enumerate(contract.TIP_ORDER)
            },
            "boundary_contact_count_distribution": {
                str(count): int(
                    (selected & (boundary_count == count)).sum().detach().cpu()
                )
                for count in range(len(contract.TIP_ORDER) + 1)
            },
            "boundary_raw_pair_count_distribution": {
                str(count): int(
                    (selected & (boundary_raw_count == count))
                    .sum().detach().cpu()
                )
                for count in range(len(contract.TIP_ORDER) + 1)
            },
            "boundary_active_efc_pair_count_distribution": {
                str(count): int(
                    (selected & (boundary_active_count == count))
                    .sum().detach().cpu()
                )
                for count in range(len(contract.TIP_ORDER) + 1)
            },
            "boundary_tips": {
                tip_name: {
                    "raw_pair_present_count": int(
                        (selected & boundary_tip_raw_pair[:, tip_index])
                        .sum().detach().cpu()
                    ),
                    "active_efc_pair_present_count": int(
                        (selected & boundary_tip_active_pair[:, tip_index])
                        .sum().detach().cpu()
                    ),
                    "qualified_count": int(
                        (selected & boundary_tip_qualified[:, tip_index])
                        .sum().detach().cpu()
                    ),
                    "force_n": quantiles(
                        boundary_force[:, tip_index], selected
                    ),
                    "qualified_force_n": quantiles(
                        boundary_force[:, tip_index],
                        selected & boundary_tip_qualified[:, tip_index],
                    ),
                }
                for tip_index, tip_name in enumerate(contract.TIP_ORDER)
            },
            "lift_start_body_position_m": axis_summary(
                prelift["start_body_position"], selected
            ),
            "lift_start_quaternion": {
                name: quantiles(prelift["start_quaternion"][:, axis], selected)
                for axis, name in enumerate(("w", "x", "y", "z"))
            },
            "stages": stage_summary,
        }

    def ik_projection_summary(selected: Any) -> dict[str, Any]:
        def stages(values: Any) -> dict[str, Any]:
            return {
                name: {
                    "events": int(values[selected, axis].sum().detach().cpu()),
                    "affected_worlds": int(
                        (selected & (values[:, axis] > 0)).sum().detach().cpu()
                    ),
                }
                for axis, name in enumerate(ik_stage_names)
            }

        return {
            "joint_trust_by_stage": stages(ik_trust_by_stage),
            "cage_workspace_by_stage": stages(ik_cage_by_stage),
            "cage_scale_min": quantiles(ik_cage_scale_min, selected),
            "cage_scale_mean": quantiles(ik_cage_scale_mean, selected),
        }

    def entry(selected: Any) -> dict[str, Any]:
        total = int(selected.sum().detach().cpu())
        return {
            "world_count": total,
            "endpoint_success_count": int(
                (selected & endpoint).sum().detach().cpu()
            ),
            "route_certified_count": int(
                (selected & route).sum().detach().cpu()
            ),
            "plane_compliant_count": int(
                (selected & plane).sum().detach().cpu()
            ),
            "not_dropped_count": int(
                (selected & not_dropped).sum().detach().cpu()
            ),
            "valid_count": int((selected & valid).sum().detach().cpu()),
            "endpoint_position_axis_abs_m": axis_summary(
                position_axis_abs, selected
            ),
            "endpoint_euler_axis_abs_deg": axis_summary(
                euler_axis_abs, selected
            ),
            "endpoint_object_linear_speed_m_s": quantiles(
                endpoint_object_linear_speed, selected
            ),
            "endpoint_object_angular_speed_rad_s": quantiles(
                endpoint_object_angular_speed, selected
            ),
            "path_tracking_axis_abs_max_m": axis_summary(
                tracking_axis_abs_max, selected
            ),
            "path_tracking_rmse_m": quantiles(tracking_rmse, selected),
            "path_schedule": schedule_summary(selected),
            "first_route_violation": first_violation_summary(selected),
            "prelift_contact_gate": prelift_summary(selected),
            "translation_entry": translation_entry_summary(selected),
            "ik_projection": ik_projection_summary(selected),
        }

    all_worlds = torch.ones_like(endpoint, dtype=torch.bool)
    result = {
        "criterion": _CERTIFIED_VALID_DEFINITION,
        "task": task,
        "path_tracking_axis_abs_tolerance_m": (
            contract.TRANSLATION_PATH_TRACKING_AXIS_ABS_TOLERANCE_M
        ),
        "path_schedule_route_first_control_tick": (
            route_control_tick_offset + 1
        ),
        "overall": entry(all_worlds),
        "by_object": {
            name: entry(object_index == index)
            for index, name in enumerate(contract.OBJECTS)
        },
        "by_path": (
            {
                name: entry(path_index == index)
                for index, name in enumerate(contract.TRANSLATION_PATH_SHAPES)
            }
            if task == contract.TASK_TRANSLATE
            else {}
        ),
    }
    if completed_episode_controller_counters is not None:
        counters = dict(completed_episode_controller_counters)
        if set(counters) != set(_CONTROLLER_COUNTER_FIELDS) or any(
            isinstance(counters[name], bool)
            or not isinstance(counters[name], int)
            or counters[name] < 0
            for name in _CONTROLLER_COUNTER_FIELDS
        ):
            raise ValueError(
                "completed episode controller counters have an invalid schema"
            )
        result["completed_episode_controller_counters"] = {
            name: counters[name] for name in _CONTROLLER_COUNTER_FIELDS
        }
    return result


def _certified_candidate_coverage(
    torch: Any,
    *,
    config: Any,
    goals: dict[str, Any],
    phase: str,
    valid_world: Any,
    raise_on_failure: bool,
) -> dict[str, Any]:
    """Certify first-valid, reward-independent IK expert demonstrations.

    Formal capacity admission is fail-closed.  A bounded diagnostic may ask
    for the failed certificate to be returned so its complete certification
    funnel is preserved in the report.
    """

    task = contract.validate_task(config.task)
    population = int(config.elite.population)
    groups = int(config.groups_per_vector_batch)
    valid = torch.as_tensor(valid_world, device=valid_world.device).to(
        torch.bool
    )
    if tuple(valid.shape) != (int(config.num_envs),):
        raise ValueError("certified-candidate mask has the wrong shape")
    grouped_valid = valid.reshape(groups, population)
    valid_group = grouped_valid.any(dim=1)
    first = torch.arange(groups, device=valid.device) * population
    group_fields = {
        name: torch.as_tensor(goals[name], device=valid.device).index_select(
            0, first
        )
        for name in (
            "object_index",
            "template_index",
            "task_index",
            "path_shape_index",
            "rotation_angle_deg",
        )
    }
    expected_task_index = contract.TASKS.index(task)
    if not bool((group_fields["task_index"] == expected_task_index).all()):
        raise RuntimeError("capacity expert emitted a cross-task group")
    formal = int(config.num_envs) == contract.FORMAL_GLOBAL_WORLDS
    expected_templates = list(range(contract.RESET_TEMPLATES_PER_OBJECT))
    expected_shapes = list(range(len(contract.TRANSLATION_PATH_SHAPES)))
    rotation_maximum = contract.ROTATION_ANGLE_MAX_DEG
    result: dict[str, Any] = {
        "passed": True,
        "phase": phase,
        "task": task,
        "task_index": expected_task_index,
        "population": population,
        "groups": groups,
        "formal_coverage_required": formal,
        "selection_rule": "first_valid_replica_no_reward",
        "valid_definition": _CERTIFIED_VALID_DEFINITION,
        "valid_world_count": int(valid.sum().detach().cpu()),
        "valid_group_count": int(valid_group.sum().detach().cpu()),
        "by_object": {},
    }
    for object_index, object_name in enumerate(contract.OBJECTS):
        selected = group_fields["object_index"] == object_index
        selected_valid = selected & valid_group
        group_count = int(selected.sum().detach().cpu())
        valid_group_count = int(selected_valid.sum().detach().cpu())
        template_values = sorted(
            int(value)
            for value in torch.unique(
                group_fields["template_index"][selected_valid], sorted=True
            )
            .detach()
            .cpu()
        )
        shape_values = sorted(
            int(value)
            for value in torch.unique(
                group_fields["path_shape_index"][selected_valid], sorted=True
            )
            .detach()
            .cpu()
        )
        maximum_angle_count = int(
            (
                group_fields["rotation_angle_deg"][selected_valid]
                == rotation_maximum
            )
            .sum()
            .detach()
            .cpu()
        )
        template_complete = template_values == expected_templates
        shapes_complete = shape_values == expected_shapes
        translation_shapes = (
            {
                "active": True,
                "valid_shape_indices": shape_values,
                "all_shapes_covered": shapes_complete,
            }
            if task == contract.TASK_TRANSLATE
            else {
                "active": False,
                "reason": _INACTIVE_COVERAGE_REASON,
                "valid_shape_indices": [],
                "all_shapes_covered": False,
            }
        )
        rotation_maximum_coverage = (
            {
                "active": True,
                "maximum_angle_deg": rotation_maximum,
                "valid_group_count": maximum_angle_count,
                "covered": maximum_angle_count > 0,
            }
            if task == contract.TASK_ROTATE
            else {
                "active": False,
                "reason": _INACTIVE_COVERAGE_REASON,
                "maximum_angle_deg": None,
                "valid_group_count": 0,
                "covered": False,
            }
        )
        entry = {
            "group_count": group_count,
            "valid_group_count": valid_group_count,
            "valid_template_indices": template_values,
            "valid_template_count": len(template_values),
            "all_reset_templates_covered": template_complete,
            "translation_path_shapes": translation_shapes,
            "rotation_maximum_angle": rotation_maximum_coverage,
        }
        result["by_object"][object_name] = entry
        if valid_group_count <= 0:
            result["passed"] = False
        if formal and not template_complete:
            result["passed"] = False
        if formal and task == contract.TASK_TRANSLATE and not shapes_complete:
            result["passed"] = False
        if (
            formal
            and task == contract.TASK_ROTATE
            and maximum_angle_count <= 0
        ):
            result["passed"] = False
    if not result["passed"] and raise_on_failure:
        raise RuntimeError(
            "capacity IK expert produced insufficient certified candidates: "
            + json.dumps(result, sort_keys=True)
        )
    return result


def _certified_candidate_coverage_is_valid(
    value: Any,
    *,
    task: str,
    phase: str,
    expected_world_counts: tuple[int, ...],
    population: int,
) -> bool:
    """Validate the no-reward demonstration admission certificate."""

    try:
        selected_task = contract.validate_task(task)
    except ValueError:
        return False
    formal = sum(expected_world_counts) == contract.FORMAL_GLOBAL_WORLDS
    if any(worlds % population for worlds in expected_world_counts):
        return False
    expected_group_counts = tuple(
        worlds // population for worlds in expected_world_counts
    )
    expected_templates = list(range(contract.RESET_TEMPLATES_PER_OBJECT))
    expected_shapes = list(range(len(contract.TRANSLATION_PATH_SHAPES)))
    rotation_maximum = contract.ROTATION_ANGLE_MAX_DEG
    by_object = value.get("by_object") if isinstance(value, dict) else None
    if not (
        isinstance(value, dict)
        and value.get("passed") is True
        and value.get("phase") == phase
        and value.get("task") == selected_task
        and value.get("task_index") == contract.TASKS.index(selected_task)
        and value.get("population") == population
        and value.get("groups") == sum(expected_group_counts)
        and value.get("formal_coverage_required") is formal
        and value.get("selection_rule") == "first_valid_replica_no_reward"
        and value.get("valid_definition") == _CERTIFIED_VALID_DEFINITION
        and isinstance(value.get("valid_world_count"), int)
        and not isinstance(value.get("valid_world_count"), bool)
        and value["valid_world_count"] > 0
        and isinstance(value.get("valid_group_count"), int)
        and not isinstance(value.get("valid_group_count"), bool)
        and value["valid_group_count"] > 0
        and isinstance(by_object, dict)
        and set(by_object) == set(contract.OBJECTS)
    ):
        return False
    valid_group_total = 0
    for index, object_name in enumerate(contract.OBJECTS):
        entry = by_object.get(object_name)
        if not isinstance(entry, dict):
            return False
        templates = entry.get("valid_template_indices")
        translation = entry.get("translation_path_shapes")
        rotation = entry.get("rotation_maximum_angle")
        if not (
            entry.get("group_count") == expected_group_counts[index]
            and isinstance(entry.get("valid_group_count"), int)
            and not isinstance(entry.get("valid_group_count"), bool)
            and 0 < entry["valid_group_count"] <= expected_group_counts[index]
            and isinstance(templates, list)
            and templates == sorted(set(templates))
            and all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and 0 <= item < contract.RESET_TEMPLATES_PER_OBJECT
                for item in templates
            )
            and entry.get("valid_template_count") == len(templates)
            and entry.get("all_reset_templates_covered")
            is (templates == expected_templates)
            and isinstance(translation, dict)
            and isinstance(rotation, dict)
        ):
            return False
        if formal and templates != expected_templates:
            return False
        if selected_task == contract.TASK_TRANSLATE:
            if not (
                translation.get("active") is True
                and isinstance(translation.get("valid_shape_indices"), list)
                and translation["valid_shape_indices"]
                == sorted(set(translation["valid_shape_indices"]))
                and all(
                    isinstance(item, int)
                    and not isinstance(item, bool)
                    and 0 <= item < len(contract.TRANSLATION_PATH_SHAPES)
                    for item in translation["valid_shape_indices"]
                )
                and translation.get("all_shapes_covered")
                is (translation["valid_shape_indices"] == expected_shapes)
                and (
                    not formal
                    or translation["valid_shape_indices"] == expected_shapes
                )
                and rotation
                == {
                    "active": False,
                    "reason": _INACTIVE_COVERAGE_REASON,
                    "maximum_angle_deg": None,
                    "valid_group_count": 0,
                    "covered": False,
                }
            ):
                return False
        else:
            if not (
                translation
                == {
                    "active": False,
                    "reason": _INACTIVE_COVERAGE_REASON,
                    "valid_shape_indices": [],
                    "all_shapes_covered": False,
                }
                and rotation.get("active") is True
                and rotation.get("maximum_angle_deg") == rotation_maximum
                and isinstance(rotation.get("valid_group_count"), int)
                and not isinstance(rotation.get("valid_group_count"), bool)
                and rotation["valid_group_count"] >= 0
                and rotation.get("covered")
                is (rotation["valid_group_count"] > 0)
                and (not formal or rotation["valid_group_count"] > 0)
            ):
                return False
        valid_group_total += entry["valid_group_count"]
    return valid_group_total == value["valid_group_count"]


def _finite_state_summary(torch: Any, tensor: Any) -> dict[str, Any]:
    """Summarize one batched state tensor without retaining a full clone."""

    finite_elements = torch.isfinite(tensor)
    return {
        "finite": finite_elements.all(dim=1),
        # A non-finite row's maximum is deliberately ignored when the report
        # is materialized; avoiding ``where`` here prevents another full-size
        # state temporary on every control tick.
        "max_abs": tensor.abs().amax(dim=1),
    }


def _capture_native_pre_step_diagnostic(
    *,
    torch: Any,
    environment: Any,
    requested_action: Any,
) -> dict[str, Any]:
    """Keep only compact GPU-resident state needed to explain a bad step.

    Full qpos/qvel/qacc clones would add tens of megabytes and extra memory
    traffic at every control tick for the formal global batch. Per-world
    finite/max-abs
    summaries preserve the required pre-step evidence.  Contact force and
    requested action are small and are cloned because the environment and
    controller may reuse their backing storage after ``step``.
    """

    qacc_warmstart = (
        environment.backend.tensor("qacc_warmstart")
        if environment.backend.has_field("qacc_warmstart")
        else None
    )
    return {
        "qpos": _finite_state_summary(
            torch, environment.backend.tensor("qpos")
        ),
        "qvel": _finite_state_summary(
            torch, environment.backend.tensor("qvel")
        ),
        "qacc_warmstart": (
            _finite_state_summary(torch, qacc_warmstart)
            if qacc_warmstart is not None
            else None
        ),
        "contact_force_n": environment._last_contact_force.detach().clone(),
        "force_decode_finite": (
            environment._last_force_decode_finite.detach().clone()
        ),
        "requested_action": requested_action.detach().clone(),
    }


def _first_failure_transition_diagnostic(
    *,
    torch: Any,
    environment: Any,
    pre_step: dict[str, Any],
    first_world: int,
) -> dict[str, Any]:
    """Materialize detailed diagnostics for only the first failed world."""

    def previous_state(name: str) -> tuple[bool | None, float | None]:
        summary = pre_step.get(name)
        if summary is None:
            return None, None
        finite = bool(summary["finite"][first_world].detach().cpu())
        maximum = (
            float(summary["max_abs"][first_world].detach().cpu())
            if finite
            else None
        )
        return finite, maximum

    previous_qpos_finite, previous_qpos_max_abs = previous_state("qpos")
    previous_qvel_finite, previous_qvel_max_abs = previous_state("qvel")
    previous_qacc_finite, previous_qacc_max_abs = previous_state(
        "qacc_warmstart"
    )
    qpos = environment.backend.tensor("qpos")[first_world]
    qvel = environment.backend.tensor("qvel")[first_world]
    return {
        "first_done_qpos_nonfinite_indices": _jsonable(
            torch.nonzero(~torch.isfinite(qpos), as_tuple=False).flatten()
        ),
        "first_done_qvel_nonfinite_indices": _jsonable(
            torch.nonzero(~torch.isfinite(qvel), as_tuple=False).flatten()
        ),
        "first_done_previous_qpos_finite": previous_qpos_finite,
        "first_done_previous_qpos_max_abs": previous_qpos_max_abs,
        "first_done_previous_qvel_finite": previous_qvel_finite,
        "first_done_previous_qvel_max_abs": previous_qvel_max_abs,
        "first_done_previous_qacc_warmstart_finite": previous_qacc_finite,
        "first_done_previous_qacc_warmstart_max_abs": previous_qacc_max_abs,
        "first_done_previous_contact_force_n": _jsonable(
            pre_step["contact_force_n"][first_world]
        ),
        "first_done_previous_force_decode_finite": bool(
            pre_step["force_decode_finite"][first_world].detach().cpu()
        ),
        "first_done_force_decode_finite": bool(
            environment._last_force_decode_finite[first_world]
            .detach()
            .cpu()
        ),
        "first_done_requested_action": _jsonable(
            pre_step["requested_action"][first_world]
        ),
        "first_done_filtered_action": _jsonable(
            environment.filtered_action[first_world]
        ),
        "first_done_used_action": _jsonable(
            environment.previous_action[first_world]
        ),
    }


def _health_is_valid(
    health: dict[str, Any],
    *,
    expected_world_counts: tuple[int, ...],
) -> bool:
    return formal_sharded_health_is_valid(
        health,
        expected_world_counts=expected_world_counts,
    ) and all(
        _is_sha256(health.get(name))
        for name in _PLATFORM_FINGERPRINT_FIELDS
    )



def _native_stage_uses_expected_jacobian(result: dict[str, Any]) -> bool:
    """Validate the v19 single-run native capacity certificate."""

    try:
        selected_task = contract.validate_task(str(result.get("task", "")))
    except ValueError:
        return False
    config = replace(DEFAULT_CONFIG, task=selected_task)
    stages = result.get("stages")
    native = stages.get("native_search") if isinstance(stages, dict) else None
    gpu = result.get("gpu")
    total_bytes = gpu.get("memory_bytes") if isinstance(gpu, dict) else None
    if not (
        isinstance(native, dict)
        and isinstance(total_bytes, int)
        and total_bytes > 0
        and stage_has_actual_memory_certificate(
            native,
            expected_stage="native_search",
            expected_total_bytes=total_bytes,
        )
    ):
        return False
    full = native.get("full_episode")
    remainder = native.get("remainder")
    certificates = native.get("post_release_baseline_certificates")
    actual_memory = native.get("actual_device_memory")
    segments = (
        actual_memory.get("segments")
        if isinstance(actual_memory, dict)
        else None
    )
    samples = (
        actual_memory.get("checkpoint_samples")
        if isinstance(actual_memory, dict)
        else None
    )
    health = native.get("backend_health")
    if not (
        remainder is None
        and config.remainder_worlds == 0
        and isinstance(full, dict)
        and isinstance(segments, dict)
        and isinstance(samples, list)
    ):
        return False
    control_ticks, complete_count, final_tail = _run_lifecycle_counts(config)
    lifecycle = full.get("post_extraction_lifecycle")
    completed_counters = full.get(
        "completed_episode_controller_safety_counters"
    )
    continuation_counters = (
        lifecycle.get("controller_safety_counters")
        if isinstance(lifecycle, dict)
        else None
    )
    segment = full.get("actual_device_memory_segment")
    segment_fraction = (
        segment.get("actual_peak_device_memory_fraction")
        if isinstance(segment, dict)
        else None
    )
    full_valid = (
        full.get("passed") is True
        and full.get("phase") == PHASE
        and full.get("worlds") == contract.FORMAL_GLOBAL_WORLDS
        and full.get("object_world_counts")
        == list(contract.FORMAL_OBJECT_WORLD_COUNTS)
        and full.get("episode_control_ticks") == config.episode_length
        and full.get("run_control_ticks") == control_ticks
        and full.get("configured_run_control_ticks") == control_ticks
        and full.get("complete_episode_count") == complete_count
        and full.get("final_tail_ticks") == final_tail
        and _rotation_goal_coverage_is_valid(
            full.get("rotation_goal_coverage"),
            PHASE,
            task=selected_task,
            expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
        )
        and _translation_path_goal_coverage_is_valid(
            full.get("translation_path_goal_coverage"),
            PHASE,
            task=selected_task,
            expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
        )
        and _certified_candidate_coverage_is_valid(
            full.get("certified_candidate_coverage"),
            task=selected_task,
            phase=PHASE,
            expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
            population=config.elite.population,
        )
        and _controller_counters_are_formally_valid(
            full.get("controller_safety_counters")
        )
        and _controller_counters_are_formally_valid(completed_counters)
        and isinstance(lifecycle, dict)
        and lifecycle.get("passed") is True
        and _controller_counters_are_formally_valid(continuation_counters)
        and isinstance(segment_fraction, (int, float))
        and not isinstance(segment_fraction, bool)
        and 0.0 <= float(segment_fraction) <= 0.90
        and segment == segments.get("formal_global_batch")
    )
    certificates_valid = (
        isinstance(certificates, dict)
        and set(certificates) == {"formal_global_batch"}
        and post_release_certificate_is_valid(
            certificates["formal_global_batch"],
            label="formal_global_batch",
            expected_total_bytes=total_bytes,
            checkpoint_samples=samples,
        )
    )
    return bool(
        full_valid
        and certificates_valid
        and native.get("protocol_order")
        == ["formal_global_batch", "formal_global_batch_release"]
        and _controller_counters_are_formally_valid(
            native.get("controller_safety_counters_total")
        )
        and isinstance(health, dict)
        and formal_sharded_health_is_valid(
            health,
            expected_world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
        )
        and all(
            _is_sha256(result.get(name))
            and health.get(name) == result.get(name)
            for name in _PLATFORM_FINGERPRINT_FIELDS
        )
    )


def _run_remainder_transition(
    *,
    torch: Any,
    config: Any,
    platform_spec: Any,
    device: Any,
    memory: _DeviceMemoryTracker,
    phase: str,
    validate_official_config: bool,
) -> dict[str, Any]:
    """Certify an optional four-shard remainder before a full batch."""

    if phase != PHASE:
        raise ValueError(f"unsupported remainder phase: {phase!r}")
    worlds = int(config.remainder_worlds)
    allocation = tuple(config.remainder_object_world_counts)
    if worlds <= 0:
        raise ValueError("remainder transition requires positive worlds")
    environment = make_env(
        phase,
        config=config,
        platform_spec=platform_spec,
        worlds=worlds,
        object_world_counts=allocation,
        device=str(device),
        require_linux=True,
        validate_official_config=validate_official_config,
    )
    try:
        memory.sample(f"{phase}_remainder_four_shard_environment_initialized")
        observation = environment.reset()
        translation_path_goal_coverage = _translation_path_goal_coverage(
            torch, environment, phase, task=config.task
        )
        action = torch.zeros(
            (worlds, contract.ACTION_DIM),
            device=device,
            dtype=observation.dtype,
        )
        transition = environment.step(action)
        controller_counters = _controller_counter_totals(environment)
        _assert_hard_controller_counters_zero(
            controller_counters,
            context=f"{phase} remainder",
        )
        if bool(_unexpected_early_done(transition, torch).any()):
            raise RuntimeError("remainder transition terminated prematurely")
        if not bool(transition.info["transition_finite"].all()):
            raise FloatingPointError("remainder transition is non-finite")
        health = environment.backend.health()
        if not _health_is_valid(
            health, expected_world_counts=allocation
        ):
            raise RuntimeError(f"remainder MJWarp health failed: {health}")
        memory.sample(f"{phase}_remainder_four_shard_transition_complete")
        return {
            "passed": True,
            "phase": phase,
            "worlds": worlds,
            "object_world_counts": list(allocation),
            "translation_path_goal_coverage": (
                translation_path_goal_coverage
            ),
            "backend_health": _jsonable(health),
            "controller_safety_counters": controller_counters,
            "reward_mean": float(
                transition.reward.mean().detach().cpu()
            ),
            "object_drop_count": int(
                torch.as_tensor(
                    transition.info.get(
                        "object_dropped", torch.zeros_like(transition.done)
                    ),
                    dtype=torch.bool,
                )
                .sum()
                .detach()
                .cpu()
            ),
        }
    finally:
        environment.close()


def _run_full_search_episode_peak(
    *,
    torch: Any,
    config: Any,
    platform_spec: Any,
    device: Any,
    memory: _DeviceMemoryTracker,
    teacher: MJWarpIKExpert,
    phase: str,
    validate_official_config: bool,
    diagnostic_only: bool,
) -> dict[str, Any]:
    """Exercise one full batch in the same allocation order as training."""

    if phase != PHASE:
        raise ValueError(f"unsupported full-search phase: {phase!r}")
    if bool(validate_official_config) == bool(diagnostic_only):
        raise ValueError(
            "capacity diagnostic and official-validation modes disagree"
        )
    environment = make_env(
        phase,
        config=config,
        platform_spec=platform_spec,
        worlds=config.num_envs,
        object_world_counts=tuple(config.object_world_counts),
        device=str(device),
        require_linux=True,
        validate_official_config=validate_official_config,
    )
    try:
        memory.sample(f"{phase}_four_shard_environment_initialized")
        # This order is contractual: env -> trajectory/counter buffers ->
        # teacher episode -> reset/CUDA graph.  It mirrors _run_search_phase
        # and captures the real transient peak instead of a cheaper ordering.
        trajectory_observation = torch.empty(
            (
                config.num_envs,
                config.episode_length + 1,
                OBSERVATION_DIM,
            ),
            device=device,
            dtype=torch.float16,
        )
        trajectory_action = torch.empty(
            (
                config.num_envs,
                config.episode_length,
                contract.ACTION_DIM,
            ),
            device=device,
            dtype=torch.float16,
        )
        episode_return = torch.zeros(
            config.num_envs,
            device=device,
            dtype=torch.float32,
        )
        actual_compliant_count = torch.zeros(
            (), device=device, dtype=torch.int64
        )
        commanded_compliant_count = torch.zeros(
            (), device=device, dtype=torch.int64
        )
        actual_relevant_count = torch.zeros(
            (), device=device, dtype=torch.int64
        )
        actual_fraction_finite = torch.ones(
            (), device=device, dtype=torch.bool
        )
        grasp_plane_sample_count = 0
        memory.sample(f"{phase}_formal_trajectory_buffers_allocated")
        goals = teacher.begin_episode(phase, 0)
        rotation_goal_coverage = _rotation_goal_coverage(
            torch,
            goals,
            phase,
            task=config.task,
            expected_world_counts=tuple(config.object_world_counts),
        )
        observation = environment.reset(goals=goals)
        translation_path_goal_coverage = _translation_path_goal_coverage(
            torch,
            environment,
            phase,
            task=config.task,
            require_complete_shape_coverage=not diagnostic_only,
        )
        environment.backend.prepare_cuda_graph()
        memory.sample(f"{phase}_four_cuda_graphs_captured")
        observation = environment.reset(goals=goals)
        trajectory_observation[:, 0].copy_(observation.to(torch.float16))
        transition = None
        for tick in range(config.episode_length):
            with torch.no_grad():
                requested_action = teacher.action(tick, observation)
                pre_step_diagnostic = _capture_native_pre_step_diagnostic(
                    torch=torch,
                    environment=environment,
                    requested_action=requested_action,
                )
                transition = environment.step(requested_action)
            commanded_compliance = transition.info[
                "commanded_grasp_plane_compliant"
            ].to(torch.bool)
            actual_compliance = transition.info[
                "actual_grasp_plane_compliant"
            ].to(torch.bool)
            actual_relevant = transition.info[
                "actual_grasp_plane_relevant"
            ].to(torch.bool)
            actual_compliant_count.add_(
                (actual_compliance & actual_relevant).sum()
            )
            commanded_compliant_count.add_(
                (commanded_compliance & actual_relevant).sum()
            )
            actual_relevant_count.add_(actual_relevant.sum())
            grasp_plane_sample_count += int(actual_compliance.numel())
            actual_fraction_finite.logical_and_(
                transition.info["actual_grasp_plane_finite"]
                .to(torch.bool)
                .all()
            )
            trajectory_action[:, tick].copy_(
                requested_action.to(torch.float16)
            )
            trajectory_observation[:, tick + 1].copy_(
                transition.observation.to(torch.float16)
            )
            episode_return.add_(transition.reward)
            observation = transition.observation
            expected_terminal = tick + 1 == config.episode_length
            done = transition.done.to(torch.bool)
            unexplained_done = _unexpected_early_done(transition, torch)
            transition_finite = transition.info["transition_finite"].to(
                torch.bool
            )
            if (not bool(transition_finite.all())) or (
                bool(unexplained_done.any()) and not expected_terminal
            ):
                qpos_finite = torch.isfinite(
                    environment.backend.tensor("qpos")
                ).all(dim=1)
                qvel_finite = torch.isfinite(
                    environment.backend.tensor("qvel")
                ).all(dim=1)
                position_finite = torch.isfinite(
                    environment.position_error_m
                )
                euler_finite = torch.isfinite(
                    environment.euler_l2_error_deg
                )
                observation_finite = torch.isfinite(
                    transition.observation
                ).all(dim=1)
                bad_ids = torch.nonzero(
                    unexplained_done, as_tuple=False
                ).flatten()
                first_world = int(bad_ids[0].detach().cpu())
                object_index = int(
                    environment.object_index[first_world].detach().cpu()
                )
                task_index = int(
                    environment.task_index[first_world].detach().cpu()
                )
                diagnostic = {
                    "done_worlds": int(
                        unexplained_done.sum().detach().cpu()
                    ),
                    "finite_object_drop_worlds": int(
                        transition.info["object_dropped"]
                        .to(torch.bool)
                        .sum()
                        .detach()
                        .cpu()
                    ),
                    "qpos_nonfinite_worlds": int(
                        (~qpos_finite).sum().detach().cpu()
                    ),
                    "qvel_nonfinite_worlds": int(
                        (~qvel_finite).sum().detach().cpu()
                    ),
                    "position_error_nonfinite_worlds": int(
                        (~position_finite).sum().detach().cpu()
                    ),
                    "euler_error_nonfinite_worlds": int(
                        (~euler_finite).sum().detach().cpu()
                    ),
                    "observation_nonfinite_worlds": int(
                        (~observation_finite).sum().detach().cpu()
                    ),
                    "first_done_world": first_world,
                    "first_done_object_index": object_index,
                    "first_done_object": contract.OBJECTS[object_index],
                    "first_done_task_index": task_index,
                    "first_done_task": contract.TASKS[task_index],
                    "first_done_template_index": int(
                        environment.template_index[first_world]
                        .detach()
                        .cpu()
                    ),
                    "first_done_source_region": int(
                        environment.source_region[first_world]
                        .detach()
                        .cpu()
                    ),
                    "first_done_path_seed": int(
                        environment.path_seed[first_world].detach().cpu()
                    ),
                    "first_done_path_shape_index": int(
                        environment.path_shape_index[first_world]
                        .detach()
                        .cpu()
                    ),
                    "first_done_path_shape": contract.TRANSLATION_PATH_SHAPES[
                        int(
                            environment.path_shape_index[first_world]
                            .detach()
                            .cpu()
                        )
                    ],
                    "first_done_path_step_m": float(
                        environment.path_step_m[first_world].detach().cpu()
                    ),
                    "first_done_path_length_m": float(
                        environment.path_length_m[first_world]
                        .detach()
                        .cpu()
                    ),
                    "first_done_path_waypoints_xy_m": _jsonable(
                        environment.path_waypoints_xy_m[first_world]
                    ),
                    "first_done_rotation_angle_deg": int(
                        environment.rotation_angle_deg[first_world]
                        .detach()
                        .cpu()
                    ),
                    "first_done_ik_solve_feasible": bool(
                        environment._last_ik_solve_feasible[first_world]
                        .detach()
                        .cpu()
                    ),
                    "first_done_ik_solve_fallback_count": int(
                        environment.ik_solve_fallback_count[first_world]
                        .detach()
                        .cpu()
                    ),
                    "first_done_ik_trust_region_projection_count": int(
                        environment.ik_trust_region_projection_count[
                            first_world
                        ]
                        .detach()
                        .cpu()
                    ),
                    "first_done_position_feedback_m": _jsonable(
                        environment.position_feedback_m[first_world]
                    ),
                    "first_done_preload_trim_m": _jsonable(
                        environment.preload_trim_m[first_world]
                    ),
                    "first_done_contact_force_n": _jsonable(
                        environment._last_contact_force[first_world]
                    ),
                    "first_done_ctrl_velocity_max_abs": float(
                        environment.ctrl_velocity[first_world]
                        .abs()
                        .amax()
                        .detach()
                        .cpu()
                    ),
                    "first_done_ctrl_acceleration_max_abs": float(
                        environment.ctrl_acceleration[first_world]
                        .abs()
                        .amax()
                        .detach()
                        .cpu()
                    ),
                    "first_done_ctrl_certificate_fallback_count": int(
                        environment.ctrl_certificate_fallback_count[
                            first_world
                        ]
                        .detach()
                        .cpu()
                    ),
                    "backend_health": _jsonable(
                        environment.backend.health()
                    ),
                }
                diagnostic.update(
                    _first_failure_transition_diagnostic(
                        torch=torch,
                        environment=environment,
                        pre_step=pre_step_diagnostic,
                        first_world=first_world,
                    )
                )
                raise RuntimeError(
                    "native search produced an invalid terminal at control tick "
                    f"{tick + 1}: "
                    + json.dumps(diagnostic, sort_keys=True)
                )
            if expected_terminal and not bool(done.all()):
                raise RuntimeError(
                    "native search did not terminate every formal world"
                )
            if (
                tick % 32 == 0
                and not bool(torch.isfinite(observation).all())
            ):
                raise FloatingPointError(
                    f"non-finite observation at control tick {tick}"
                )
            if tick % 32 == 0:
                memory.sample(
                    f"control_tick_{tick + 1}", synchronize=True
                )
        if transition is None:
            raise RuntimeError("formal episode contained no control ticks")
        if not bool(actual_fraction_finite):
            raise FloatingPointError(
                "physical hard-tip grasp-plane fraction became non-finite"
            )
        relevant_sample_count = int(actual_relevant_count.detach().cpu())
        if relevant_sample_count <= 0:
            raise RuntimeError(
                "formal episode never activated the physical grasp-plane gate"
            )
        # Sharded env info tensors are persistent buffers and are reused by
        # the steady-state step below. Materialize endpoint scalars now.
        endpoint_success_rate = float(
            transition.info["success"]
            .to(torch.float32)
            .mean()
            .detach()
            .cpu()
        )
        endpoint_object_drop_count = int(
            transition.info["object_dropped"]
            .to(torch.bool)
            .sum()
            .detach()
            .cpu()
        )

        def drop_breakdown(
            names: tuple[str, ...], indices: Any
        ) -> dict[str, dict[str, int | float]]:
            result: dict[str, dict[str, int | float]] = {}
            dropped = transition.info["object_dropped"].to(torch.bool)
            for index, name in enumerate(names):
                selected = indices == index
                episodes = int(selected.sum().detach().cpu())
                drops = int(dropped[selected].sum().detach().cpu())
                result[name] = {
                    "drop_count": drops,
                    "episode_count": episodes,
                    "drop_rate": drops / episodes if episodes else 0.0,
                }
            return result

        endpoint_drop_by_object = drop_breakdown(
            contract.OBJECTS, transition.info["object_index"]
        )
        endpoint_drop_by_task = drop_breakdown(
            contract.TASKS, transition.info["task_index"]
        )
        endpoint_actual_plane_rate = float(
            actual_compliance.to(torch.float32).mean().detach().cpu()
        )
        completed_episode_controller_counters = (
            _controller_counter_totals(environment)
        )
        _assert_hard_controller_counters_zero(
            completed_episode_controller_counters,
            context=f"{phase} completed formal episode",
        )
        endpoint_success = transition.info["success"].to(torch.bool)
        endpoint_dropped = transition.info["object_dropped"].to(torch.bool)
        endpoint_route_certified = transition.info[
            "path_tracking_certified"
        ].to(torch.bool)
        # Embed soft saturation/projection evidence in the funnel so the
        # fail-closed coverage exception below cannot discard it.
        certification_funnel = _certification_funnel(
            torch,
            config=config,
            info=transition.info,
            actual_compliance=actual_compliance,
            completed_episode_controller_counters=(
                completed_episode_controller_counters
            ),
        )
        try:
            certified_candidate_coverage = _certified_candidate_coverage(
                torch,
                config=config,
                goals=goals,
                phase=phase,
                valid_world=(
                    endpoint_success
                    & endpoint_route_certified
                    & actual_compliance
                    & (~endpoint_dropped)
                ),
                raise_on_failure=not diagnostic_only,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}; certification_funnel="
                + json.dumps(certification_funnel, sort_keys=True)
            ) from exc
        certified_candidate_coverage["certification_funnel"] = (
            certification_funnel
        )
        winner = teacher.select_winners(
            episode_return,
            endpoint_success
            & endpoint_route_certified
            & (~endpoint_dropped),
            actual_compliance,
        )
        coverage_passed = bool(certified_candidate_coverage["passed"])
        if coverage_passed:
            teacher.update_centroids(winner)
        teacher.clear_episode_state()
        # A completed formal batch can retain one winner from every
        # population group. Exercise the two GPU index-select temporaries
        # sequentially, as the real store append path does.
        elite_observation = trajectory_observation.index_select(0, winner)
        elite_observation_shape = list(elite_observation.shape)
        torch.cuda.synchronize(device)
        del elite_observation
        elite_action = trajectory_action.index_select(0, winner)
        elite_action_shape = list(elite_action.shape)
        torch.cuda.synchronize(device)
        del elite_action
        memory.sample(f"{phase}_maximum_elite_extraction_completed")
        # Reproduce the exact multi-episode lifecycle used by formal training
        # while persistent trajectory buffers remain live.  Every complete
        # episode is begun/reset explicitly; only the divmod remainder is a
        # final tail.  The formal 20k profile has exactly one 3000-tick
        # episode per 60M-step run, while diagnostics use the same lifecycle.
        (
            configured_run_control_ticks,
            complete_episode_count,
            final_tail_ticks,
        ) = _run_lifecycle_counts(config)
        certified_run_control_ticks = (
            complete_episode_count * config.episode_length + final_tail_ticks
        )
        continuation_controller_counters = {
            name: 0 for name in _CONTROLLER_COUNTER_FIELDS
        }
        additional_complete_episodes: list[dict[str, Any]] = []

        def run_continuation_segment(
            *, episode_index: int, ticks: int, complete: bool
        ) -> dict[str, Any]:
            goals = teacher.begin_episode(phase, episode_index)
            segment_observation = environment.reset(goals=goals)
            segment_transition = None
            for segment_tick in range(ticks):
                with torch.no_grad():
                    segment_action = teacher.action(
                        segment_tick, segment_observation
                    )
                    segment_transition = environment.step(segment_action)
                terminal_tick = complete and segment_tick + 1 == ticks
                unexplained_done = _unexpected_early_done(
                    segment_transition, torch
                )
                if bool(unexplained_done.any()) and not terminal_tick:
                    raise RuntimeError(
                        f"{phase} lifecycle episode {episode_index} had an "
                        f"unexplained terminal at tick {segment_tick + 1}"
                    )
                if terminal_tick and not bool(
                    segment_transition.done.to(torch.bool).all()
                ):
                    raise RuntimeError(
                        f"{phase} lifecycle episode {episode_index} did not "
                        "terminate every formal world"
                    )
                if not bool(
                    segment_transition.info["transition_finite"].all()
                ):
                    raise FloatingPointError(
                        f"{phase} lifecycle episode {episode_index} is "
                        f"non-finite at tick {segment_tick + 1}"
                    )
                tick_counters = _controller_counter_totals(environment)
                _assert_hard_controller_counters_zero(
                    tick_counters,
                    context=(
                        f"{phase} lifecycle episode {episode_index} tick "
                        f"{segment_tick + 1}"
                    ),
                )
                memory.sample(
                    f"{phase}_lifecycle_episode_{episode_index}_tick_"
                    f"{segment_tick + 1}"
                )
                segment_observation = segment_transition.observation
            if segment_transition is None:
                raise RuntimeError("lifecycle segment contained no control ticks")
            counters = _controller_counter_totals(environment)
            _assert_hard_controller_counters_zero(
                counters,
                context=f"{phase} lifecycle episode {episode_index}",
            )
            for name in _CONTROLLER_COUNTER_FIELDS:
                continuation_controller_counters[name] += counters[name]
            object_drop_count = int(
                segment_transition.info["object_dropped"]
                .to(torch.bool)
                .sum()
                .detach()
                .cpu()
            )
            return {
                "episode_index": episode_index,
                "validated_ticks": ticks,
                "complete": complete,
                "transition_finite": True,
                "terminated_worlds": int(
                    segment_transition.done.to(torch.bool).sum().detach().cpu()
                ),
                "object_drop_count": object_drop_count,
                "controller_safety_counters": counters,
            }

        for episode_index in range(1, complete_episode_count):
            additional_complete_episodes.append(
                run_continuation_segment(
                    episode_index=episode_index,
                    ticks=config.episode_length,
                    complete=True,
                )
            )
        if final_tail_ticks:
            final_tail = run_continuation_segment(
                episode_index=complete_episode_count,
                ticks=final_tail_ticks,
                complete=False,
            )
        else:
            final_tail = {
                "episode_index": None,
                "validated_ticks": 0,
                "complete": False,
                "transition_finite": True,
                "terminated_worlds": 0,
                "object_drop_count": 0,
                "controller_safety_counters": {
                    name: 0 for name in _CONTROLLER_COUNTER_FIELDS
                },
            }
        teacher.clear_episode_state()
        memory.sample(f"{phase}_post_extraction_lifecycle_complete")
        if not bool(torch.isfinite(episode_return).all()):
            raise FloatingPointError("non-finite elite-search return")
        health = environment.backend.health()
        if not _health_is_valid(
            health,
            expected_world_counts=tuple(environment.world_counts),
        ):
            raise RuntimeError(f"MJWarp health failed: {health}")
        controller_counters = {
            name: completed_episode_controller_counters[name]
            + continuation_controller_counters[name]
            for name in _CONTROLLER_COUNTER_FIELDS
        }
        _assert_hard_controller_counters_zero(
            controller_counters,
            context=f"{phase} formal native search",
        )
        torch.cuda.synchronize(device)
        memory.sample(f"{phase}_formal_episode_complete", synchronize=False)
        peak_memory_gb, peak_reserved_gb = _memory_gb(torch, device)
        total_memory_gb = (
            torch.cuda.get_device_properties(device).total_memory
            / (1024.0**3)
        )
        if peak_reserved_gb > total_memory_gb:
            raise RuntimeError("native peak reservation exceeds GPU memory")
        return {
            "passed": coverage_passed,
            "failure_stage": (
                None
                if coverage_passed
                else "certified_candidate_coverage"
            ),
            "phase": phase,
            "worlds": int(config.num_envs),
            "object_world_counts": list(environment.world_counts),
            "episode_control_ticks": config.episode_length,
            "run_control_ticks": certified_run_control_ticks,
            "configured_run_control_ticks": (
                configured_run_control_ticks
            ),
            "complete_episode_count": complete_episode_count,
            "final_tail_ticks": final_tail_ticks,
            "rotation_goal_coverage": rotation_goal_coverage,
            "translation_path_goal_coverage": (
                translation_path_goal_coverage
            ),
            "certified_candidate_coverage": (
                certified_candidate_coverage
            ),
            "certification_funnel": certification_funnel,
            "trajectory_observation_shape": list(
                trajectory_observation.shape
            ),
            "trajectory_action_shape": list(trajectory_action.shape),
            "maximum_elite_extraction_count": int(winner.numel()),
            "certified_winner_count": int(
                certified_candidate_coverage["valid_group_count"]
            ),
            "centroid_update_applied": coverage_passed,
            "elite_extraction_mode": (
                "certified_winners"
                if coverage_passed
                else "diagnostic_memory_probe_only"
            ),
            "elite_observation_extraction_shape": elite_observation_shape,
            "elite_action_extraction_shape": elite_action_shape,
            "trajectory_buffer_bytes": (
                trajectory_observation.numel()
                * trajectory_observation.element_size()
                + trajectory_action.numel()
                * trajectory_action.element_size()
            ),
            "diffusion_model_present": False,
            "controller_safety_counters": controller_counters,
            "completed_episode_controller_safety_counters": (
                completed_episode_controller_counters
            ),
            "post_extraction_lifecycle": {
                "passed": True,
                "additional_complete_episode_count": (
                    complete_episode_count - 1
                ),
                "additional_complete_episodes": (
                    additional_complete_episodes
                ),
                "final_tail": final_tail,
                "controller_safety_counters": (
                    continuation_controller_counters
                ),
            },
            **controller_counters,
            "actual_grasp_plane_finite_validated_every_control_tick": True,
            "endpoint_success_rate": endpoint_success_rate,
            "endpoint_object_drop_count": endpoint_object_drop_count,
            "endpoint_object_drop_rate": (
                endpoint_object_drop_count / int(config.num_envs)
            ),
            "endpoint_drop_by_object": endpoint_drop_by_object,
            "endpoint_drop_by_task": endpoint_drop_by_task,
            "object_drop_contract": {
                "table_top_z_m": contract.TABLE_TOP_Z_M,
                "margin_m": contract.DROP_MARGIN_M,
                "threshold_z_m": contract.OBJECT_DROP_THRESHOLD_Z_M,
                "quarantine_position_m": list(
                    contract.OBJECT_DROP_QUARANTINE_POSITION_M
                ),
            },
            "grasp_plane_compliance_rate": float(
                actual_compliant_count.detach().cpu()
            )
            / relevant_sample_count,
            "actual_grasp_plane_compliance_rate": float(
                actual_compliant_count.detach().cpu()
            )
            / relevant_sample_count,
            "actual_grasp_plane_sample_count": relevant_sample_count,
            "actual_grasp_plane_total_sample_count": (
                grasp_plane_sample_count
            ),
            "endpoint_actual_grasp_plane_compliance_rate": (
                endpoint_actual_plane_rate
            ),
            "commanded_grasp_plane_compliance_rate": float(
                commanded_compliant_count.detach().cpu()
            )
            / relevant_sample_count,
            "peak_memory_gb": peak_memory_gb,
            "peak_reserved_memory_gb": peak_reserved_gb,
            "total_gpu_memory_gb": total_memory_gb,
            "reserved_memory_headroom_gb": (
                total_memory_gb - peak_reserved_gb
            ),
            "backend_health": _jsonable(health),
        }
    finally:
        teacher.clear_episode_state()
        environment.close()


def _run_native_search_peak(
    *,
    torch: Any,
    config: Any,
    platform_spec: Any,
    device: Any,
    diagnostic_only: bool,
) -> dict[str, Any]:
    """Replay the full task run, including every allocation boundary."""

    validate_official_config = not diagnostic_only
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    # Training constructs one persistent teacher before the full task run.
    teacher = MJWarpIKExpert(
        config,
        platform_spec=platform_spec,
        device=str(device),
    )
    teacher.clear_episode_state()
    memory = _DeviceMemoryTracker(
        torch, device, stage="native_search"
    )
    remainder_result: dict[str, Any] | None = None
    post_release_baseline_certificates: dict[str, Any] = {}
    protocol_order: list[str] = []

    phase = PHASE
    teacher.clear_episode_state()
    remainder_worlds = config.remainder_worlds
    if int(remainder_worlds) > 0:
            remainder_label = "remainder"
            protocol_order.append(remainder_label)
            remainder_baseline = memory.sample(
                f"pre_allocation:{remainder_label}"
            )
            memory.begin_segment(remainder_label)
            remainder_result = _run_remainder_transition(
                torch=torch,
                config=config,
                platform_spec=platform_spec,
                device=device,
                memory=memory,
                phase=phase,
                validate_official_config=validate_official_config,
            )
            teacher.clear_episode_state()
            _release_cuda_stage_memory(torch, device)
            post_release_baseline_certificates[remainder_label] = (
                _post_release_baseline_certificate(
                    memory=memory,
                    label=remainder_label,
                    baseline=remainder_baseline,
                )
            )
            protocol_order.append(f"{remainder_label}_release")

    full_label = "formal_global_batch"
    protocol_order.append(full_label)
    full_baseline = memory.sample(f"pre_allocation:{full_label}")
    memory.begin_segment(full_label)
    full_episode = _run_full_search_episode_peak(
            torch=torch,
            config=config,
            platform_spec=platform_spec,
            device=device,
            memory=memory,
            teacher=teacher,
            phase=phase,
            validate_official_config=validate_official_config,
            diagnostic_only=diagnostic_only,
        )
    teacher.clear_episode_state()
    _release_cuda_stage_memory(torch, device)
    post_release_baseline_certificates[full_label] = (
        _post_release_baseline_certificate(
            memory=memory,
            label=full_label,
            baseline=full_baseline,
        )
    )
    protocol_order.append(f"{full_label}_release")

    teacher.clear_episode_state()
    memory.sample("native_stage_complete", synchronize=False)
    actual_device_memory = memory.report()
    if remainder_result is not None:
        remainder_result["actual_device_memory_segment"] = (
            actual_device_memory["segments"].get("remainder")
        )
    full_episode["actual_device_memory_segment"] = (
        actual_device_memory["segments"].get("formal_global_batch")
    )

    controller_totals = {
        name: sum(
            int(result["controller_safety_counters"][name])
            for result in ((remainder_result,) if remainder_result else ())
            + (full_episode,)
        )
        for name in _CONTROLLER_COUNTER_FIELDS
    }
    _assert_hard_controller_counters_zero(
        controller_totals, context="capacity native protocol"
    )
    peak_memory_gb, peak_reserved_gb = _memory_gb(torch, device)
    total_memory_gb = (
        torch.cuda.get_device_properties(device).total_memory / (1024.0**3)
    )
    if peak_reserved_gb > total_memory_gb:
        raise RuntimeError("native peak reservation exceeds GPU memory")
    return {
        "passed": bool(full_episode["passed"]),
        "failed_runs": [] if full_episode["passed"] else [PHASE],
        "protocol_order": protocol_order,
        "episode_control_ticks": config.episode_length,
        "full_episode": full_episode,
        "remainder": remainder_result,
        "post_release_baseline_certificates": (
            post_release_baseline_certificates
        ),
        "controller_safety_counters_total": controller_totals,
        **controller_totals,
        "trajectory_observation_shape": full_episode[
            "trajectory_observation_shape"
        ],
        "trajectory_action_shape": full_episode["trajectory_action_shape"],
        "trajectory_buffer_bytes": full_episode["trajectory_buffer_bytes"],
        "diffusion_model_present": False,
        "backend_health": full_episode["backend_health"],
        "formal_global_batch_device_memory": (
            full_episode["actual_device_memory_segment"]
        ),
        "peak_memory_gb": peak_memory_gb,
        "peak_reserved_memory_gb": peak_reserved_gb,
        "actual_device_memory": actual_device_memory,
        "actual_peak_device_used_bytes": actual_device_memory[
            "actual_peak_device_used_bytes"
        ],
        "actual_peak_device_used_gb": actual_device_memory[
            "actual_peak_device_used_gb"
        ],
        "actual_peak_device_memory_fraction": actual_device_memory[
            "actual_peak_device_memory_fraction"
        ],
        "total_gpu_memory_gb": total_memory_gb,
        "reserved_memory_headroom_gb": total_memory_gb - peak_reserved_gb,
    }


def _optimizer_is_finite(torch: Any, optimizer: Any) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                return False
    return True


def _run_diffusion_peak(
    *,
    torch: Any,
    config: Any,
    device: Any,
) -> dict[str, Any]:
    """Exercise the maximum formal CUDA demo/training/deployment allocation."""

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    memory = _DeviceMemoryTracker(torch, device, stage="diffusion")
    memory.begin_segment("diffusion_training")
    settings = config.diffusion
    demo_capacity = int(config.maximum_demo_episodes)
    if demo_capacity <= 0:
        raise RuntimeError("formal configuration has no elite demo capacity")
    demo_observations = torch.empty(
        (
            demo_capacity,
            config.episode_length + 1,
            OBSERVATION_DIM,
        ),
        device=device,
        dtype=torch.float16,
    )
    demo_actions = torch.empty(
        (
            demo_capacity,
            config.episode_length,
            contract.ACTION_DIM,
        ),
        device=device,
        dtype=torch.float16,
    )
    # The real cache is copied from mmap and therefore committed.  Zeroing
    # makes this exact-size equivalent deterministic and safe to sample.
    demo_observations.zero_()
    demo_actions.zero_()
    memory.sample("maximum_demo_cache_committed")
    cache_bytes = (
        demo_observations.numel() * demo_observations.element_size()
        + demo_actions.numel() * demo_actions.element_size()
    )
    policy = DiffusionPolicy(
        obs_dim=OBSERVATION_DIM,
        action_dim=contract.ACTION_DIM,
        obs_horizon=settings.obs_horizon,
        horizon=settings.action_horizon,
        num_diffusion_steps=settings.diffusion_steps,
        hidden_dim=settings.hidden_dim,
        depth=settings.depth,
        dropout=settings.dropout,
    ).to(device)
    ema = ExponentialMovingAverage(policy, decay=settings.ema_decay)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    memory.sample("policy_ema_optimizer_initialized")

    # This mirrors DemoWindowSampler.sample while always retaining the
    # configured batch of 4096 for the fixed formal global candidate.
    batch_size = int(settings.batch_size)
    if batch_size != 4_096:
        raise RuntimeError("formal capacity protocol requires batch_size=4096")
    episode_index = torch.arange(
        batch_size, device=device, dtype=torch.long
    ) % demo_capacity
    timestep_index = torch.arange(
        batch_size, device=device, dtype=torch.long
    ) % (config.episode_length - settings.action_horizon + 1)
    previous_index = torch.clamp(timestep_index - 1, min=0)
    training_history = torch.stack(
        (
            demo_observations[episode_index, previous_index],
            demo_observations[episode_index, timestep_index],
        ),
        dim=1,
    ).to(torch.float32)
    offsets = torch.arange(
        settings.action_horizon, device=device, dtype=torch.long
    )[None, :]
    training_action = demo_actions[
        episode_index[:, None], timestep_index[:, None] + offsets
    ].to(torch.float32)
    optimizer.zero_grad(set_to_none=True)
    use_bfloat16 = (
        settings.use_bfloat16
        and torch.cuda.get_device_properties(device).major >= 8
    )
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=use_bfloat16,
    ):
        loss = policy.compute_loss(training_history, training_action)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Diffusion capacity loss is non-finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), settings.max_grad_norm
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("Diffusion capacity gradient is non-finite")
    optimizer.step()
    ema.update(policy)
    if not _optimizer_is_finite(torch, optimizer):
        raise FloatingPointError("Diffusion optimizer state is non-finite")
    loss_value = float(loss.detach().cpu())
    gradient_norm_value = float(gradient_norm.detach().cpu())
    torch.cuda.synchronize(device)
    memory.sample("maximum_diffusion_training_step", synchronize=False)
    training_peak_gb, training_reserved_gb = _memory_gb(torch, device)

    # Training temporaries/gradients do not coexist with runtime inference,
    # while the formal cache, policy, EMA and initialized AdamW state remain.
    optimizer.zero_grad(set_to_none=True)
    del training_history, training_action, loss, offsets
    del episode_index, timestep_index, previous_index
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    runtime_baseline_gb, runtime_baseline_reserved_gb = _allocated_gb(
        torch, device
    )
    memory.begin_segment("diffusion_runtime")
    history = BatchedObservationHistory(
        worlds=config.num_envs,
        obs_horizon=settings.obs_horizon,
        obs_dim=OBSERVATION_DIM,
        device=device,
    )
    current_observation = torch.zeros(
        (config.num_envs, OBSERVATION_DIM),
        device=device,
        dtype=torch.float32,
    )
    history.append(current_observation)
    controller = BatchedDiffusionController(
        policy,
        worlds=config.num_envs,
        inference_steps=settings.inference_steps,
        replan_interval=settings.replan_interval,
        seed=config.seed,
    )
    action = controller.act(history.values)
    if tuple(action.shape) != (config.num_envs, contract.ACTION_DIM):
        raise RuntimeError("formal DDIM runtime returned the wrong action shape")
    if not bool(torch.isfinite(action).all()):
        raise FloatingPointError("formal DDIM runtime action is non-finite")
    torch.cuda.synchronize(device)
    memory.sample("formal_global_batch_ddim_runtime", synchronize=False)
    actual_device_memory = memory.report()
    runtime_peak_gb, runtime_reserved_gb = _memory_gb(torch, device)
    peak_reserved_gb = max(training_reserved_gb, runtime_reserved_gb)
    total_memory_gb = (
        torch.cuda.get_device_properties(device).total_memory / (1024.0**3)
    )
    if peak_reserved_gb > total_memory_gb:
        raise RuntimeError("Diffusion peak reservation exceeds GPU memory")
    return {
        "passed": True,
        "maximum_official_elite_demos": demo_capacity,
        "demo_cache_equivalence": (
            "exact float16 DemoWindowSampler CUDA observation/action cache"
        ),
        "demo_observation_cache_shape": list(demo_observations.shape),
        "demo_action_cache_shape": list(demo_actions.shape),
        "demo_cache_bytes": cache_bytes,
        "policy_dtype": str(next(policy.parameters()).dtype),
        "ema_present": True,
        "adamw_state_initialized": bool(optimizer.state),
        "training_batch_size": batch_size,
        "training_autocast_dtype": (
            "torch.bfloat16" if use_bfloat16 else "disabled"
        ),
        "diffusion_loss": loss_value,
        "gradient_norm": gradient_norm_value,
        "training_peak_memory_gb": training_peak_gb,
        "training_peak_reserved_memory_gb": training_reserved_gb,
        "runtime_worlds": config.num_envs,
        "runtime_ddim_inference_steps": settings.inference_steps,
        "runtime_replan_interval": settings.replan_interval,
        "runtime_baseline_memory_gb": runtime_baseline_gb,
        "runtime_baseline_reserved_memory_gb": (
            runtime_baseline_reserved_gb
        ),
        "runtime_peak_memory_gb": runtime_peak_gb,
        "runtime_peak_reserved_memory_gb": runtime_reserved_gb,
        "peak_memory_gb": max(training_peak_gb, runtime_peak_gb),
        "peak_reserved_memory_gb": peak_reserved_gb,
        "actual_device_memory": actual_device_memory,
        "actual_peak_device_used_bytes": actual_device_memory[
            "actual_peak_device_used_bytes"
        ],
        "actual_peak_device_used_gb": actual_device_memory[
            "actual_peak_device_used_gb"
        ],
        "actual_peak_device_memory_fraction": actual_device_memory[
            "actual_peak_device_memory_fraction"
        ],
        "total_gpu_memory_gb": total_memory_gb,
        "reserved_memory_headroom_gb": total_memory_gb - peak_reserved_gb,
    }


def _worker(
    num_envs: int,
    device: str,
    output: Path,
    *,
    task: str,
    diagnostic_only: bool = False,
) -> int:
    selected_task = contract.validate_task(task)
    task_index = contract.TASKS.index(selected_task)
    root_height_m = float(TASK_ROOT_HEIGHT_M[selected_task])
    started = time.perf_counter()
    result: dict[str, Any] = {
        "num_envs": int(num_envs),
        "passed": False,
        "error": None,
        "platform": PLATFORM,
        "task": selected_task,
        "task_index": task_index,
        "root_height_m": root_height_m,
        "algorithm_signature": contract.ALGORITHM_SIGNATURE,
        "runtime_code_fingerprint_sha256": (
            runtime_code_fingerprint_sha256()
        ),
        "physics_layout": contract.PHYSICS_LAYOUT,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": contract.ACTION_DIM,
        "episode_control_ticks": DEFAULT_CONFIG.episode_length,
        "diagnostic_only": bool(diagnostic_only),
        "formal_training_eligible": not diagnostic_only,
        "formal_training_ineligibility_reason": None,
        "stages": {},
    }
    try:
        configure_cublas_workspace(deterministic=True)
        import torch

        selected_device = torch.device(device)
        torch.cuda.set_device(selected_device)
        torch.cuda.empty_cache()
        config = _worker_config(
            num_envs=int(num_envs),
            device=device,
            task=selected_task,
            diagnostic_only=diagnostic_only,
        )
        result.update(
            _worker_lifecycle_metadata(
                config, diagnostic_only=diagnostic_only
            )
        )
        if diagnostic_only:
            _validate_diagnostic_config(config)
            try:
                config.validate()
            except ValueError as exc:
                result["formal_training_ineligibility_reason"] = str(exc)
            else:
                result["formal_training_ineligibility_reason"] = (
                    "diagnostic-only reports never authorize formal training"
                )
        else:
            config.validate()
        result["object_world_counts"] = list(config.object_world_counts)
        result["remainder_worlds"] = config.remainder_worlds
        result["remainder_object_world_counts"] = list(
            config.remainder_object_world_counts
        )
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        result["candidate_config_canonical_sha256"] = _config_sha256(config)
        runtime = inspect_runtime(
            strict=True,
            task=selected_task,
            device=device,
            deterministic_torch=True,
        )
        result["gpu"] = _gpu_binding(runtime["torch"])
        result["runtime"] = {
            "torch": runtime["torch"],
            "warp": runtime["warp"],
        }
        result["runtime_errors"] = list(runtime.get("errors", ()))
        if not runtime["ok"]:
            raise RuntimeError("; ".join(runtime["errors"]))
        platform_spec = build_platform_spec(selected_task)
        if (
            platform_spec.task != selected_task
            or platform_spec.task_index != task_index
            or not math.isclose(
                float(platform_spec.root_height_m),
                root_height_m,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise RuntimeError("capacity platform task/root binding mismatch")
        result["model_shards"] = ordered_shard_manifest(
            platform_spec,
            tuple(config.object_world_counts),
        )
        for name in _PLATFORM_FINGERPRINT_FIELDS:
            result[name] = str(getattr(platform_spec, name))
        result[_RL_MODEL_FINGERPRINT_FIELD] = str(
            platform_spec.rl_model_fingerprint_sha256
        )
        try:
            result["stages"]["native_search"] = _run_native_search_peak(
                torch=torch,
                config=config,
                platform_spec=platform_spec,
                device=selected_device,
                diagnostic_only=diagnostic_only,
            )
            if (
                not diagnostic_only
                and not result["stages"]["native_search"].get("passed")
            ):
                raise RuntimeError(
                    "formal native capacity search returned a failed stage"
                )
        except BaseException as exc:
            peak, reserved = _memory_gb(torch, selected_device)
            tracker = _ACTIVE_DEVICE_MEMORY_TRACKER
            actual_device_memory = (
                tracker.failure_report()
                if tracker is not None
                and tracker.stage == "native_search"
                else None
            )
            result["stages"]["native_search"] = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "peak_memory_gb": peak,
                "peak_reserved_memory_gb": reserved,
                "actual_device_memory": actual_device_memory,
                "actual_peak_device_used_gb": (
                    actual_device_memory.get("actual_peak_device_used_gb")
                    if actual_device_memory is not None
                    else None
                ),
            }
            raise
        finally:
            # Match the real physics-to-Diffusion boundary: Warp owns a
            # separate CUDA mempool that Torch cannot release.
            _release_cuda_stage_memory(torch, selected_device)
            allocated, reserved = _allocated_gb(torch, selected_device)
            result["stages"]["native_search"][
                "post_cleanup_memory_gb"
            ] = allocated
            result["stages"]["native_search"][
                "post_cleanup_reserved_memory_gb"
            ] = reserved

        try:
            result["stages"]["diffusion"] = _run_diffusion_peak(
                torch=torch,
                config=config,
                device=selected_device,
            )
        except BaseException as exc:
            peak, reserved = _memory_gb(torch, selected_device)
            tracker = _ACTIVE_DEVICE_MEMORY_TRACKER
            actual_device_memory = (
                tracker.failure_report()
                if tracker is not None and tracker.stage == "diffusion"
                else None
            )
            result["stages"]["diffusion"] = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "peak_memory_gb": peak,
                "peak_reserved_memory_gb": reserved,
                "actual_device_memory": actual_device_memory,
                "actual_peak_device_used_gb": (
                    actual_device_memory.get("actual_peak_device_used_gb")
                    if actual_device_memory is not None
                    else None
                ),
            }
            raise
        native_peak = float(
            result["stages"]["native_search"]
            ["actual_peak_device_used_gb"]
        )
        diffusion_peak = float(
            result["stages"]["diffusion"]
            ["actual_peak_device_used_gb"]
        )
        result["peak_memory_gb"] = max(native_peak, diffusion_peak)
        result["peak_memory_measurement"] = (
            "background device-wide cudaMemGetInfo polling; includes Warp "
            "and Torch"
        )
        result["peak_stage"] = (
            "native_search"
            if native_peak >= diffusion_peak
            else "diffusion"
        )
        result["passed"] = bool(
            result["stages"]["native_search"].get("passed")
            and result["stages"]["diffusion"].get("passed")
        )
        if not result["passed"]:
            result["error"] = (
                "diagnostic completed but one or more certification stages "
                "did not pass"
            )
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        actual_peaks = [
            stage.get("actual_peak_device_used_gb")
            for stage in result.get("stages", {}).values()
            if isinstance(stage, dict)
            and isinstance(
                stage.get("actual_peak_device_used_gb"), (int, float)
            )
        ]
        if actual_peaks:
            result["peak_memory_gb"] = max(float(value) for value in actual_peaks)
            result["peak_memory_measurement"] = (
                "background device-wide cudaMemGetInfo polling; includes "
                "Warp and Torch"
            )
        if "torch" in sys.modules:
            try:
                import torch

                if torch.cuda.is_available() and not actual_peaks:
                    result["peak_memory_gb"] = (
                        torch.cuda.max_memory_allocated(torch.device(device))
                        / (1024.0**3)
                    )
            except BaseException:
                pass
    finally:
        if "torch" in sys.modules:
            try:
                import torch

                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
            except BaseException:
                pass
    result["elapsed_s"] = time.perf_counter() - started
    _atomic_json(result, output)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["passed"] else 1


def _validate_worker_result(
    result: dict[str, Any],
    *,
    candidate: int,
    returncode: int,
    device: str,
    gpu: dict[str, Any],
    task: str,
) -> None:
    """Make a parent-side pass conditional on exit status and all bindings."""

    reported_passed = bool(result.get("passed"))
    errors: list[str] = []
    selected_task = contract.validate_task(task)
    expected_root_height_m = float(TASK_ROOT_HEIGHT_M[selected_task])
    expected_config = replace(
        DEFAULT_CONFIG,
        task=selected_task,
        num_envs=int(candidate),
        device=str(device),
    )
    expected_world_counts = tuple(expected_config.object_world_counts)
    expected = {
        "num_envs": int(candidate),
        "platform": PLATFORM,
        "task": selected_task,
        "task_index": contract.TASKS.index(selected_task),
        "root_height_m": expected_root_height_m,
        "algorithm_signature": contract.ALGORITHM_SIGNATURE,
        "runtime_code_fingerprint_sha256": (
            runtime_code_fingerprint_sha256()
        ),
        "physics_layout": contract.PHYSICS_LAYOUT,
        "object_world_counts": list(expected_world_counts),
        "remainder_worlds": expected_config.remainder_worlds,
        "remainder_object_world_counts": list(
            expected_config.remainder_object_world_counts
        ),
        "observation_dim": OBSERVATION_DIM,
        "action_dim": contract.ACTION_DIM,
        "episode_control_ticks": DEFAULT_CONFIG.episode_length,
        "candidate_config_canonical_sha256": _config_sha256(expected_config),
        "gpu": gpu,
    }
    for name, value in expected.items():
        if result.get(name) != value:
            errors.append(f"worker binding mismatch: {name}")
    for name in _PLATFORM_FINGERPRINT_FIELDS:
        if not _is_sha256(result.get(name)):
            errors.append(f"worker omitted valid {name}")
    if not _is_sha256(result.get(_RL_MODEL_FINGERPRINT_FIELD)):
        errors.append(
            f"worker omitted valid {_RL_MODEL_FINGERPRINT_FIELD}"
        )
    if not _valid_shard_manifest(
        result.get("model_shards"),
        expected_world_counts=expected_world_counts,
        expected_task=selected_task,
        expected_root_height_m=expected_root_height_m,
    ):
        errors.append("worker omitted a valid ordered four-shard manifest")
    stages = result.get("stages")
    if not isinstance(stages, dict):
        errors.append("worker omitted stage reports")
    else:
        for stage_name in ("native_search", "diffusion"):
            stage = stages.get(stage_name)
            if not isinstance(stage, dict) or not bool(stage.get("passed")):
                errors.append(f"worker stage did not pass: {stage_name}")
            elif not stage_has_actual_memory_certificate(
                stage,
                expected_stage=stage_name,
                expected_total_bytes=int(gpu["memory_bytes"]),
            ):
                errors.append(
                    f"worker stage lacks a valid device-wide 90% memory "
                    f"certificate: {stage_name}"
                )
        if not _native_stage_uses_expected_jacobian(result):
            errors.append(
                "worker native search did not certify dense Newton with "
                "model nv <= 60"
            )
    if int(returncode) != 0:
        errors.append(f"worker returncode was {returncode}, not zero")
    result["worker_reported_passed"] = reported_passed
    result["returncode"] = int(returncode)
    result["validation_errors"] = errors
    result["passed"] = reported_passed and not errors
    if errors and not result.get("error"):
        result["error"] = "; ".join(errors)


def _validate_common_report(
    other: dict[str, Any],
    *,
    device: str,
    gpu: dict[str, Any],
    task: str,
    config_canonical_sha256: str,
    candidate_config_set_sha256: str,
) -> tuple[str, float, str]:
    """Validate a single-platform report from the opposite gripper."""

    if other.get("schema") != REPORT_SCHEMA:
        raise ValueError("common capacity report has an incompatible schema")
    if other.get("is_common_report") is not False:
        raise ValueError("--common-with must reference a single-platform report")
    selected_task = contract.validate_task(task)
    other_root_height_m = other.get("root_height_m")
    if not (
        isinstance(other_root_height_m, (int, float))
        and not isinstance(other_root_height_m, bool)
        and math.isfinite(float(other_root_height_m))
    ):
        raise ValueError("common capacity report omitted a finite root height")
    for name in (*_PLATFORM_FINGERPRINT_FIELDS, _RL_MODEL_FINGERPRINT_FIELD):
        if not _is_sha256(other.get(name)):
            raise ValueError(
                f"common capacity report omitted valid {name}"
            )
    expected = {
        "task": selected_task,
        "task_index": contract.TASKS.index(selected_task),
        "algorithm_signature": contract.ALGORITHM_SIGNATURE,
        "runtime_code_fingerprint_sha256": (
            runtime_code_fingerprint_sha256()
        ),
        "physics_layout": contract.PHYSICS_LAYOUT,
        "object_world_counts": list(
            replace(
                DEFAULT_CONFIG,
                task=selected_task,
                num_envs=contract.FORMAL_GLOBAL_WORLDS,
                device=str(device),
            ).object_world_counts
        ),
        "remainder_worlds": replace(
            DEFAULT_CONFIG, task=selected_task
        ).remainder_worlds,
        "remainder_object_world_counts": list(
            replace(
                DEFAULT_CONFIG, task=selected_task
            ).remainder_object_world_counts
        ),
        "observation_dim": OBSERVATION_DIM,
        "action_dim": contract.ACTION_DIM,
        "episode_control_ticks": DEFAULT_CONFIG.episode_length,
        "device": str(device),
        "candidates": list(CANDIDATES),
        "gpu": gpu,
        "config_canonical_sha256": config_canonical_sha256,
        "candidate_config_set_sha256": candidate_config_set_sha256,
        "candidate_config_sha256": {
            str(candidate): _config_sha256(
                replace(
                    DEFAULT_CONFIG,
                    task=selected_task,
                    num_envs=int(candidate),
                    device=str(device),
                )
            )
            for candidate in CANDIDATES
        },
    }
    for name, value in expected.items():
        if other.get(name) != value:
            raise ValueError(
                f"common capacity report binding mismatch: {name}"
            )
    formal_world_counts = tuple(other["object_world_counts"])
    if not _valid_shard_manifest(
        other.get("model_shards"),
        expected_world_counts=formal_world_counts,
        expected_task=selected_task,
        expected_root_height_m=float(other_root_height_m),
    ):
        raise ValueError("common capacity report has an invalid shard manifest")
    other_platform = str(other.get("platform", ""))
    if {PLATFORM, other_platform} != {"gripper3", "gripper6"}:
        raise ValueError(
            "common capacity report must come from the opposite gripper"
        )
    if other.get("common_platforms") != [other_platform]:
        raise ValueError("single-platform report has invalid common_platforms")
    if other.get("platform_root_heights_m") != {
        other_platform: float(other_root_height_m)
    }:
        raise ValueError("single-platform report has invalid root-height pins")
    if other.get("platform_rl_model_fingerprints_sha256") != {
        other_platform: other.get(_RL_MODEL_FINGERPRINT_FIELD)
    }:
        raise ValueError("single-platform report has invalid RL model pins")
    if (
        other.get("common_with") is not None
        or other.get("common_with_sha256") is not None
    ):
        raise ValueError("single-platform report unexpectedly cites a peer")
    passing_values = other.get("passing_candidates")
    if not isinstance(passing_values, list) or any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in passing_values
    ):
        raise ValueError("common capacity report has invalid passing candidates")
    passing = set(passing_values)
    if len(passing) != len(passing_values):
        raise ValueError("common capacity report repeats a passing candidate")
    if not passing.issubset(set(CANDIDATES)):
        raise ValueError("common capacity report has non-formal candidates")
    selected = other.get("selected_num_envs")
    if passing:
        if int(selected) != max(passing):
            raise ValueError(
                "common capacity report did not select its largest pass"
            )
    elif selected is not None:
        raise ValueError("empty common capacity report selected a candidate")
    raw_results = other.get("results")
    if (
        not isinstance(raw_results, list)
        or len(raw_results) != len(CANDIDATES)
        or any(not isinstance(result, dict) for result in raw_results)
    ):
        raise ValueError("common capacity report has incomplete worker results")
    result_ids = [result.get("num_envs") for result in raw_results]
    if len(set(result_ids)) != len(result_ids) or set(result_ids) != set(
        CANDIDATES
    ):
        raise ValueError("common capacity report worker candidates are invalid")
    if other.get("runtime_errors") != []:
        raise ValueError("common capacity report records runtime errors")
    results = {int(result["num_envs"]): result for result in raw_results}
    for candidate in passing:
        result = results.get(candidate)
        expected_config = replace(
            DEFAULT_CONFIG,
            task=selected_task,
            num_envs=int(candidate),
            device=str(device),
        )
        stages = result.get("stages") if isinstance(result, dict) else None
        native_stage = (
            stages.get("native_search") if isinstance(stages, dict) else None
        )
        diffusion_stage = (
            stages.get("diffusion") if isinstance(stages, dict) else None
        )
        if (
            not isinstance(result, dict)
            or result.get("passed") is not True
            or result.get("worker_reported_passed") is not True
            or int(result.get("returncode", -1)) != 0
            or result.get("validation_errors") != []
            or result.get("error") is not None
            or result.get("runtime_errors") != []
            or result.get("diagnostic_only") is not False
            or result.get("formal_training_eligible") is not True
            or result.get("formal_training_ineligibility_reason") is not None
            or result.get("platform") != other_platform
            or result.get("gpu") != gpu
            or result.get("algorithm_signature")
            != contract.ALGORITHM_SIGNATURE
            or result.get("runtime_code_fingerprint_sha256")
            != other.get("runtime_code_fingerprint_sha256")
            or result.get("physics_layout") != contract.PHYSICS_LAYOUT
            or result.get("object_world_counts")
            != list(
                replace(
                    DEFAULT_CONFIG,
                    task=selected_task,
                    num_envs=int(candidate),
                    device=str(device),
                ).object_world_counts
            )
            or not _valid_shard_manifest(
                result.get("model_shards"),
                expected_world_counts=tuple(result["object_world_counts"]),
                expected_task=selected_task,
                expected_root_height_m=float(other_root_height_m),
                expected_manifest=other.get("model_shards"),
            )
            or result.get("remainder_worlds")
            != expected_config.remainder_worlds
            or result.get("remainder_object_world_counts")
            != list(expected_config.remainder_object_world_counts)
            or int(result.get("observation_dim", -1)) != OBSERVATION_DIM
            or int(result.get("action_dim", -1)) != contract.ACTION_DIM
            or int(result.get("episode_control_ticks", -1))
            != DEFAULT_CONFIG.episode_length
            or result.get("candidate_config_canonical_sha256")
            != _config_sha256(
                replace(
                    DEFAULT_CONFIG,
                    task=selected_task,
                    num_envs=int(candidate),
                    device=str(device),
                )
            )
            or not isinstance(result.get("runtime"), dict)
            or set(result["runtime"]) != {"torch", "warp"}
            or not all(
                isinstance(result["runtime"].get(name), dict)
                for name in ("torch", "warp")
            )
            or not isinstance(stages, dict)
            or set(stages) != {"native_search", "diffusion"}
            or not isinstance(native_stage, dict)
            or native_stage.get("passed") is not True
            or not isinstance(diffusion_stage, dict)
            or diffusion_stage.get("passed") is not True
            or not stage_has_actual_memory_certificate(
                native_stage,
                expected_stage="native_search",
                expected_total_bytes=int(gpu["memory_bytes"]),
            )
            or not stage_has_actual_memory_certificate(
                diffusion_stage,
                expected_stage="diffusion",
                expected_total_bytes=int(gpu["memory_bytes"]),
            )
            or not _native_stage_uses_expected_jacobian(result)
            or any(
                result.get(name) != other.get(name)
                for name in _PLATFORM_FINGERPRINT_FIELDS
            )
            or result.get("task") != selected_task
            or result.get("task_index") != contract.TASKS.index(selected_task)
            or result.get("root_height_m") != float(other_root_height_m)
            or result.get(_RL_MODEL_FINGERPRINT_FIELD)
            != other.get(_RL_MODEL_FINGERPRINT_FIELD)
        ):
            raise ValueError(
                f"common capacity candidate {candidate} lacks a valid worker"
            )
    return (
        other_platform,
        float(other_root_height_m),
        str(other[_RL_MODEL_FINGERPRINT_FIELD]),
    )


def _scan(
    *,
    output: Path,
    device: str,
    task: str,
) -> int:
    selected_task = contract.validate_task(task)
    task_index = contract.TASKS.index(selected_task)
    root_height_m = float(TASK_ROOT_HEIGHT_M[selected_task])
    base_config = replace(DEFAULT_CONFIG, task=selected_task)
    configure_cublas_workspace(deterministic=True)
    config_canonical_sha256 = _config_sha256(base_config)
    candidate_config_set_sha256 = _scan_config_sha256(
        selected_task, device
    )
    gpu: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="grab-capacity-scan-") as directory:
        root = Path(directory)
        for candidate in CANDIDATES:
            print(
                f"[capacity] testing {candidate} worlds through one full "
                "episode...",
                flush=True,
            )
            worker_output = root / f"{candidate}.json"
            command = [
                sys.executable,
                "-m",
                "MJWarp_RL.capacity_scan",
                "--worker",
                "--task",
                selected_task,
                "--num-envs",
                str(candidate),
                "--device",
                device,
                "--output",
                str(worker_output),
            ]
            completed = subprocess.run(command, check=False)
            if worker_output.is_file():
                result = json.loads(worker_output.read_text(encoding="utf-8"))
            else:
                result = {
                    "num_envs": candidate,
                    "passed": False,
                    "error": (
                        "capacity worker exited without a report "
                        f"(returncode={completed.returncode})"
                    ),
                }
            candidate_gpu = result.get("gpu")
            if (
                gpu is None
                and isinstance(candidate_gpu, dict)
                and candidate_gpu.get("name") is not None
                and candidate_gpu.get("memory_bytes") is not None
                and candidate_gpu.get("compute_capability") is not None
            ):
                gpu = dict(candidate_gpu)
            _validate_worker_result(
                result,
                candidate=candidate,
                returncode=completed.returncode,
                device=device,
                gpu=(
                    gpu
                    if gpu is not None
                    else _gpu_binding({})
                ),
                task=selected_task,
            )
            results.append(result)
            print(
                f"[capacity] {candidate}: "
                f"{'PASS' if result.get('passed') else 'FAIL'}",
                flush=True,
            )
    passing = sorted(
        int(result["num_envs"]) for result in results if result.get("passed")
    )
    platform_fingerprints: dict[str, str | None] = {}
    for name in (*_PLATFORM_FINGERPRINT_FIELDS, _RL_MODEL_FINGERPRINT_FIELD):
        values = {
            str(result[name])
            for result in results
            if _is_sha256(result.get(name))
        }
        platform_fingerprints[name] = (
            next(iter(values)) if len(values) == 1 else None
        )
    if gpu is None:
        gpu = _gpu_binding({})
    platform_root_heights_m = {PLATFORM: root_height_m}
    platform_rl_model_fingerprints_sha256 = {
        PLATFORM: platform_fingerprints[_RL_MODEL_FINGERPRINT_FIELD]
    }
    formal_world_counts = tuple(base_config.object_world_counts)
    shard_manifests = [
        result.get("model_shards")
        for result in results
        if _valid_shard_manifest(
            result.get("model_shards"),
            expected_world_counts=tuple(
                result.get("object_world_counts", ())
            ),
            expected_task=selected_task,
            expected_root_height_m=root_height_m,
        )
    ]
    model_shards = shard_manifests[0] if shard_manifests else None
    payload = {
        "schema": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": PLATFORM,
        "platforms": [PLATFORM],
        "task": selected_task,
        "task_index": task_index,
        "root_height_m": root_height_m,
        "platform_root_heights_m": platform_root_heights_m,
        "platform_rl_model_fingerprints_sha256": (
            platform_rl_model_fingerprints_sha256
        ),
        "algorithm_signature": contract.ALGORITHM_SIGNATURE,
        "runtime_code_fingerprint_sha256": (
            runtime_code_fingerprint_sha256()
        ),
        "physics_layout": contract.PHYSICS_LAYOUT,
        "object_world_counts": list(formal_world_counts),
        "remainder_worlds": base_config.remainder_worlds,
        "remainder_object_world_counts": list(
            base_config.remainder_object_world_counts
        ),
        "model_shards": model_shards,
        **platform_fingerprints,
        "config_canonical_sha256": config_canonical_sha256,
        "candidate_config_set_sha256": candidate_config_set_sha256,
        "candidate_config_sha256": {
            str(candidate): _config_sha256(
                replace(
                    DEFAULT_CONFIG,
                    task=selected_task,
                    num_envs=int(candidate),
                    device=str(device),
                )
            )
            for candidate in CANDIDATES
        },
        "observation_dim": OBSERVATION_DIM,
        "action_dim": contract.ACTION_DIM,
        "episode_control_ticks": base_config.episode_length,
        "device": device,
        "gpu": gpu,
        "runtime_errors": sorted(
            {
                str(error)
                for result in results
                for error in result.get("runtime_errors", ())
            }
        ),
        "candidates": list(CANDIDATES),
        "results": results,
        "passing_candidates": passing,
        "selected_num_envs": max(passing) if passing else None,
        "selection_rule": (
            "largest candidate whose zero-exit worker passes separate native "
            "search and maximum formal Diffusion peaks on G6"
        ),
    }
    _atomic_json(payload, output.expanduser().resolve())
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if passing else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=contract.TASKS,
        required=True,
        help="certify exactly one task-specific model/root bundle",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument(
        "--diagnostic-worlds",
        type=int,
        help=(
            "run the native full-task episode and steady-state boundaries, "
            "plus the Diffusion peak, as a "
            "capacity-only diagnostic; this cannot authorize training"
        ),
    )
    parser.add_argument("--device", default=DEFAULT_CONFIG.device)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "artifacts"
            / "capacity_scan.json"
        ),
    )
    args = parser.parse_args(argv)
    if args.worker:
        if args.diagnostic_worlds is not None:
            parser.error("--worker and --diagnostic-worlds are mutually exclusive")
        if args.num_envs not in CANDIDATES:
            parser.error("worker --num-envs must be a formal candidate")
        return _worker(
            args.num_envs,
            args.device,
            args.output.resolve(),
            task=args.task,
        )
    if args.diagnostic_worlds is not None:
        if args.num_envs is not None:
            parser.error(
                "--diagnostic-worlds cannot be combined with --num-envs"
            )
        if args.diagnostic_worlds <= 0:
            parser.error("--diagnostic-worlds must be positive")
        if (
            args.task == contract.TASK_ROTATE
            and args.diagnostic_worlds < ROTATION_DIAGNOSTIC_MIN_WORLDS
        ):
            parser.error(
                "rotation --diagnostic-worlds must be at least "
                f"{ROTATION_DIAGNOSTIC_MIN_WORLDS} to cover all 36 integer "
                "angles for every object with population=10"
            )
        return _worker(
            args.diagnostic_worlds,
            args.device,
            args.output.resolve(),
            task=args.task,
            diagnostic_only=True,
        )
    if args.num_envs is not None:
        parser.error("--num-envs is reserved for the internal worker")
    return _scan(
        output=args.output,
        device=args.device,
        task=args.task,
    )


if __name__ == "__main__":
    raise SystemExit(main())
