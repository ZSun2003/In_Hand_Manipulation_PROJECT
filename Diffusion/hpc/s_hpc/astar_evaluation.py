"""First-episode evaluation for the one-shot global A*--IK stack.

This module deliberately has no learned-checkpoint dependency.  One vector
environment is created with a classical nominal planner, every initial state
is committed to a SHA-256-bound manifest before the first physics step, and
only the first terminal outcome from each world is retained.  While any target
episode remains active, every terminal physics row is reset immediately and
an accounting mask prevents later episodes from being retained.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import is_dataclass, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from .contracts import (
    RESIDUAL_ACTION_DIM,
    SCHEMA_HASH,
    TASK_ASSET_HASHES,
    TASK_GEOMETRY_HASH,
)
from .monitoring import (
    append_history,
    atomic_write_json,
    jsonable,
    utc_now,
)


EVALUATION_SCHEMA = "screw-astar-spline-ik-evaluation-v1"
EPISODE_SCHEMA = "screw-astar-spline-ik-episode-v1"
INITIAL_CONDITIONS_SCHEMA = (
    "screw-astar-spline-ik-initial-conditions-v1"
)
SUMMARY_SCHEMA = "screw-astar-spline-ik-summary-v1"
EVALUATION_SEED_OFFSET = 100_000

_FAILURE_KEYS = (
    "nonfinite",
    "fell_below_table",
    "lateral_escape",
    "safety_stop",
    "contact_constraint_violation",
    "contact_violation_before_capture",
    "contact_violation_before_thread",
    "contact_violation_premature_regrasp",
    "contact_violation_material_drift",
    "planner_ik_failure",
)


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_with_evaluation_seed(config: Any) -> tuple[Any, int, int]:
    """Return a detached config rooted at the standard evaluation seed.

    ``HPCConfig`` and its sections are frozen dataclasses.  The fallback keeps
    the evaluator usable with lightweight test doubles while preserving the
    same important property: the caller's object and its on-disk source are
    never mutated.
    """

    base_seed = int(config.runtime.seed)
    evaluation_seed = base_seed + EVALUATION_SEED_OFFSET
    if is_dataclass(config) and is_dataclass(config.runtime):
        evaluation_runtime = replace(
            config.runtime, seed=evaluation_seed
        )
        evaluation_config = replace(
            config, runtime=evaluation_runtime
        )
    else:
        evaluation_config = copy.deepcopy(config)
        setattr(evaluation_config.runtime, "seed", evaluation_seed)
    if int(config.runtime.seed) != base_seed:
        raise RuntimeError("constructing the evaluation config mutated input")
    return evaluation_config, base_seed, evaluation_seed


def _torch():
    import torch

    return torch


def _as_cpu(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value


def _selected_scalar(
    mapping: Mapping[str, Any],
    key: str,
    index: int,
    default: Any,
) -> Any:
    value = mapping.get(key)
    if value is None:
        return default
    try:
        value = value[index]
    except (IndexError, KeyError, TypeError):
        pass
    value = _as_cpu(value)
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            return default
        return value.reshape(()).item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return default
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return default


def _planner_parameters(planner: Any) -> dict[str, Any]:
    """Return stable, JSON-safe planner parameters without path tensors."""

    for name in ("provenance", "parameters", "config_dict"):
        value = getattr(planner, name, None)
        if callable(value):
            value = value()
        if isinstance(value, Mapping):
            return dict(jsonable(value))
    result: dict[str, Any] = {}
    for name in (
        "workspace_radius_m",
        "grid_resolution_m",
        "staging_standoff_m",
        "_obstacle_center_x",
        "_obstacle_half_x",
        "_obstacle_half_y",
    ):
        value = getattr(planner, name, None)
        if isinstance(value, (bool, int, float, str)):
            result[name.lstrip("_")] = value
    neighbours = getattr(planner, "_NEIGHBOURS", None)
    if isinstance(neighbours, (tuple, list)):
        result["connectivity"] = len(neighbours)
    module = inspect.getmodule(type(planner))
    if module is not None:
        for name in (
            "SPLINE_VALIDATION_SPACING_M",
            "TRANSPORT_STEPS",
            "ROTATE_RAISE_STEPS",
            "DOCK_STEPS",
        ):
            value = getattr(module, name, None)
            if isinstance(value, (bool, int, float, str)):
                result[name.lower()] = value
    diagnostics = _planner_diagnostics(planner)
    for name in (
        "grid_resolution_m",
        "staging_standoff_m",
        "transport_steps",
        "rotate_raise_steps",
        "dock_steps",
    ):
        value = diagnostics.get(name)
        if isinstance(value, (bool, int, float, str)):
            result[name] = value
    if result:
        return dict(jsonable(result))
    config = getattr(planner, "config", None)
    if config is not None:
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        elif hasattr(config, "__dict__"):
            config = vars(config)
        if isinstance(config, Mapping):
            return dict(jsonable(config))
    return {}


def _planner_diagnostics(planner: Any) -> Mapping[str, Any]:
    for name in ("diagnostics", "diagnostic_state", "metrics"):
        value = getattr(planner, name, None)
        if callable(value):
            value = value()
        if isinstance(value, Mapping):
            return value
    return {}


def _diagnostic_row(
    planner: Any,
    info: Mapping[str, Any],
    world_id: int,
    worlds: int,
    *,
    planner_diagnostics: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Collect only diagnostics with an explicit per-world leading axis.

    Planner-wide counters (cache hits, total plans, mean latency, and similar
    accumulated values) are intentionally excluded here.  They describe the
    whole vector run, not the terminal episode, and are captured once in the
    final summary instead.
    """

    combined: dict[str, Any] = dict(
        _planner_diagnostics(planner)
        if planner_diagnostics is None
        else planner_diagnostics
    )
    combined.update(
        {
            str(key): value
            for key, value in info.items()
            if str(key).startswith(("astar_", "planner_"))
        }
    )
    result: dict[str, Any] = {}
    for key, raw in sorted(combined.items()):
        shape = tuple(getattr(raw, "shape", ()))
        if not shape or shape[0] != worlds:
            continue
        try:
            value = raw[world_id]
        except (IndexError, TypeError):
            continue
        value = _as_cpu(value)
        if hasattr(value, "numel"):
            if int(value.numel()) != 1:
                continue
            value = value.reshape(()).item()
        elif hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                continue
        if isinstance(value, (bool, int, float, str)) or value is None:
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            result[str(key)] = value
    return result


def _global_planner_diagnostics(
    planner: Any, *, worlds: int
) -> dict[str, Any]:
    """Return final scalar planner-wide diagnostics exactly once."""

    result: dict[str, Any] = {}
    for key, raw in sorted(_planner_diagnostics(planner).items()):
        shape = tuple(getattr(raw, "shape", ()))
        if shape and shape[0] == worlds:
            continue
        value = _as_cpu(raw)
        if hasattr(value, "numel"):
            if int(value.numel()) != 1:
                continue
            value = value.reshape(()).item()
        elif hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                continue
        if isinstance(value, (bool, int, float, str)) or value is None:
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            result[str(key)] = value
    return result


def _source_record(value: Any) -> Optional[dict[str, str]]:
    try:
        source = inspect.getsourcefile(value)
    except (OSError, TypeError):
        source = None
    if not source:
        return None
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        return None
    return {"path": str(path), "sha256": sha256_file(path)}


def _implementation_source_records() -> dict[str, dict[str, str]]:
    """Hash every executable layer of the classical controller stack."""

    package = Path(__file__).resolve().parent
    candidates = {
        "planner": package / "astar_spline_exact_collision.py",
        "planner_grid_base": package / "astar_spline.py",
        "env": package / "env.py",
        "supervisor": package / "supervisor.py",
        "ik": package / "analytic_ik.py",
        "backend": package / "mjwarp_backend.py",
        "task": package / "task.py",
        "evaluator": Path(__file__).resolve(),
        "cli": package.parent / "evaluate_astar_spline_gpu.py",
    }
    missing = [str(path) for path in candidates.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "evaluation provenance source is missing: " + ", ".join(missing)
        )
    return {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in candidates.items()
    }


def _default_health_check(env: Any, *, context: str) -> dict[str, Any]:
    """Run the production MJWarp health gate, with a test-double escape."""

    if not hasattr(env, "backend"):
        return {
            "check": "skipped_no_backend",
            "context": str(context),
        }
    # Import lazily to avoid the heavyweight workflow dependency during
    # provenance-only unit tests.  A real MJWarp environment always takes this
    # path, so non-finite or capacity-overflow state cannot be hidden by later
    # tensor sanitisation.
    from .workflows import _assert_backend_health

    return dict(_assert_backend_health(env, context=context))


def _initial_conditions(env: Any, *, episodes: int, root_seed: int) -> list[dict[str, Any]]:
    spawn = _as_cpu(env.state.spawn_xy[:episodes])
    socket = _as_cpu(env.state.socket_mouth[:episodes, :2])
    angle = _as_cpu(env.state.socket_angle_deg[:episodes])
    rows: list[dict[str, Any]] = []
    for world_id in range(episodes):
        rows.append(
            {
                "episode_index": world_id,
                "world_id": world_id,
                "root_seed": int(root_seed),
                # This is a stable row identity, not a claim that the vector
                # reset sampler creates one independent generator per world.
                "row_seed": int(root_seed) + world_id,
                "spawn_x_m": float(spawn[world_id, 0]),
                "spawn_y_m": float(spawn[world_id, 1]),
                "socket_mouth_x_m": float(socket[world_id, 0]),
                "socket_mouth_y_m": float(socket[world_id, 1]),
                "socket_angle_deg": float(angle[world_id]),
            }
        )
    return rows


def write_initial_conditions_manifest(
    path: str | Path,
    *,
    env: Any,
    episodes: int,
    base_seed: int,
    evaluation_seed: int,
    base_config_hash: str,
    evaluation_config_hash: str,
    planner_hash: str,
) -> tuple[Path, str, dict[str, Any]]:
    """Commit exact reset states before any call to ``env.step``."""

    payload = {
        "schema": INITIAL_CONDITIONS_SCHEMA,
        "created_utc": utc_now(),
        "control_stack": "global_astar_safety_analytic_ik",
        "residual": "disabled_zero_16d",
        "base_seed": int(base_seed),
        "evaluation_seed": int(evaluation_seed),
        "root_seed": int(evaluation_seed),
        "evaluation_seed_offset": EVALUATION_SEED_OFFSET,
        "seed_semantics": (
            "one vector-reset generator rooted at evaluation_seed; "
            "evaluation_seed=base_seed+100000; row_seed is a stable episode "
            "identity"
        ),
        "episodes": int(episodes),
        "worlds": int(env.worlds),
        "config_canonical_sha256": evaluation_config_hash,
        "base_config_canonical_sha256": base_config_hash,
        "evaluation_config_canonical_sha256": evaluation_config_hash,
        "planner_parameters_canonical_sha256": planner_hash,
        "rows": _initial_conditions(
            env, episodes=episodes, root_seed=evaluation_seed
        ),
    }
    target = atomic_write_json(path, payload)
    return target, sha256_file(target), payload


def _terminal_phase(info: Mapping[str, Any], world_id: int) -> Any:
    return _selected_scalar(
        info,
        "task_phase",
        world_id,
        _selected_scalar(info, "terminal_phase", world_id, -1),
    )


def _failure_reasons(
    info: Mapping[str, Any], world_id: int, *, forced: bool
) -> list[str]:
    reasons = [
        key
        for key in _FAILURE_KEYS
        if bool(_selected_scalar(info, key, world_id, False))
    ]
    if forced:
        reasons.append("evaluation_step_limit")
    return list(dict.fromkeys(reasons))


def _episode_record(
    *,
    info: Mapping[str, Any],
    world_id: int,
    episode_index: int,
    initial: Mapping[str, Any],
    planner: Any,
    worlds: int,
    forced_truncation: bool,
    max_steps: int,
    planner_diagnostics: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if forced_truncation:
        success = False
        failure = False
        truncated = True
        steps = int(max_steps)
    else:
        success = bool(_selected_scalar(info, "success", world_id, False))
        failure = bool(_selected_scalar(info, "failure", world_id, False))
        truncated = bool(
            _selected_scalar(info, "truncated", world_id, False)
        )
        steps = int(
            _selected_scalar(info, "elapsed_steps", world_id, max_steps)
        )
    if sum((success, failure, truncated)) != 1:
        raise RuntimeError(
            "terminal outcome must contain exactly one of success, failure, "
            "or truncated"
        )
    twist = float(_selected_scalar(info, "twist", world_id, 0.0))
    depth = float(_selected_scalar(info, "depth", world_id, 0.0))
    lateral = float(
        _selected_scalar(info, "lateral_error", world_id, 0.0)
    )
    axis = float(_selected_scalar(info, "axis_error", world_id, 0.0))
    speed = float(_selected_scalar(info, "speed", world_id, 0.0))
    return {
        "schema": EPISODE_SCHEMA,
        "episode_index": int(episode_index),
        "world_id": int(world_id),
        "initial_condition": dict(initial),
        "outcome": (
            "success" if success else "failure" if failure else "truncated"
        ),
        "success": success,
        "failure": failure,
        "truncated": truncated,
        "forced_evaluation_step_limit": bool(forced_truncation),
        "terminal_steps": steps,
        "terminal_phase": _terminal_phase(info, world_id),
        "terminal_depth_m": depth,
        "terminal_depth_mm": depth * 1000.0,
        "terminal_turns": twist / (2.0 * math.pi),
        "terminal_lateral_error_m": lateral,
        "terminal_lateral_error_mm": lateral * 1000.0,
        "terminal_axis_error_rad": axis,
        "terminal_axis_error_deg": math.degrees(axis),
        "terminal_speed_m_s": speed,
        "ik_failures": int(
            _selected_scalar(info, "ik_failures", world_id, 0)
        ),
        "ik_anchor_fallbacks": int(
            _selected_scalar(info, "ik_anchor_fallbacks", world_id, 0)
        ),
        "ik_safety_stops": int(
            _selected_scalar(info, "ik_safety_stops", world_id, 0)
        ),
        "planner_ik_failure": bool(
            _selected_scalar(info, "planner_ik_failure", world_id, False)
        ),
        "failure_reasons": _failure_reasons(
            info, world_id, forced=forced_truncation
        ),
        "planner_diagnostics": _diagnostic_row(
            planner,
            info,
            world_id,
            worlds,
            planner_diagnostics=planner_diagnostics,
        ),
    }


def _wilson(successes: int, count: int) -> Optional[dict[str, float]]:
    if count <= 0:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count**2))
        / denominator
    )
    return {
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sample")
    position = (len(ordered) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _aggregate_planner_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate immutable terminal-plan rows into paper-ready metrics."""

    diagnostic_rows = [
        row.get("planner_diagnostics", {}) for row in records
    ]
    result: dict[str, Any] = {}

    def present(key: str) -> list[Any]:
        return [
            row[key]
            for row in diagnostic_rows
            if isinstance(row, Mapping) and row.get(key) is not None
        ]

    plan_found = present("plan_found")
    if plan_found:
        count = sum(bool(value) for value in plan_found)
        result["plan_found"] = {
            "count": int(count),
            "rate": count / len(plan_found),
            "episodes_with_metric": len(plan_found),
        }
    spline_fallback = present("spline_fallback")
    if spline_fallback:
        count = sum(bool(value) for value in spline_fallback)
        result["spline_fallback"] = {
            "count": int(count),
            "rate": count / len(spline_fallback),
            "episodes_with_metric": len(spline_fallback),
        }

    def finite_values(key: str) -> list[float]:
        values: list[float] = []
        for value in present(key):
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values.append(number)
        return values

    for key in ("path_length_m", "planning_latency_ms"):
        values = finite_values(key)
        if values:
            result[key] = {
                "mean": statistics.fmean(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "max": max(values),
                "episodes_with_metric": len(values),
            }
    clearance = finite_values("minimum_obstacle_clearance_m")
    if clearance:
        result["minimum_obstacle_clearance_m"] = {
            "min": min(clearance),
            "p50": _percentile(clearance, 0.50),
            "episodes_with_metric": len(clearance),
        }
    expanded = finite_values("astar_expanded")
    if expanded:
        result["astar_expanded"] = {
            "mean": statistics.fmean(expanded),
            "max": max(expanded),
            "episodes_with_metric": len(expanded),
        }
    return result


def build_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    log_path: Path,
    execution: Optional[Mapping[str, Any]] = None,
    planner_diagnostics: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    count = len(records)
    outcomes = Counter(str(row["outcome"]) for row in records)

    def mean(key: str) -> Optional[float]:
        if not records:
            return None
        return statistics.fmean(float(row[key]) for row in records)

    successes = outcomes["success"]
    planner_failures = sum(
        bool(row.get("planner_ik_failure")) for row in records
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "created_utc": utc_now(),
        "control_stack": "global_astar_safety_analytic_ik",
        "residual": "disabled_zero_16d",
        "episodes": count,
        "successes": successes,
        "failures": outcomes["failure"],
        "truncations": outcomes["truncated"],
        "success_rate": successes / count if count else None,
        "success_rate_ci95": _wilson(successes, count),
        "mean_terminal_steps": mean("terminal_steps"),
        "mean_terminal_depth_m": mean("terminal_depth_m"),
        "mean_terminal_turns": mean("terminal_turns"),
        "mean_terminal_lateral_error_m": mean(
            "terminal_lateral_error_m"
        ),
        "mean_terminal_axis_error_rad": mean(
            "terminal_axis_error_rad"
        ),
        "mean_terminal_speed_m_s": mean("terminal_speed_m_s"),
        "ik_failures": sum(int(row["ik_failures"]) for row in records),
        "ik_anchor_fallbacks": sum(
            int(row["ik_anchor_fallbacks"]) for row in records
        ),
        "ik_safety_stops": sum(
            int(row["ik_safety_stops"]) for row in records
        ),
        "planner_ik_failure_episodes": planner_failures,
        "failure_reasons": dict(
            sorted(
                Counter(
                    reason
                    for row in records
                    for reason in row.get("failure_reasons", [])
                ).items()
            )
        ),
        "initial_conditions_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "episode_log": {"path": str(log_path)},
        "execution": dict(execution or {}),
        "health": dict((execution or {}).get("last_health", {})),
        "planner_diagnostics": {
            "scope": "final_global_scalar_diagnostics",
            "final_global": dict(planner_diagnostics or {}),
        },
        "planner_metrics": _aggregate_planner_metrics(records),
        "provenance": dict(provenance),
    }


def _default_planner_factory(config: Any, *, worlds: int, device: str):
    # Imported lazily so provenance/unit tests do not require planner kernels.
    from .astar_spline_exact_collision import (
        ExactCollisionBatchedAStarSplinePlanner,
    )

    return ExactCollisionBatchedAStarSplinePlanner(
        config,
        worlds=worlds,
        device=device,
    )


def _default_env_factory(config: Any, **kwargs):
    from .env import MJWarpScrewEnv

    return MJWarpScrewEnv(config, **kwargs)


def evaluate_astar_spline(
    config: Any,
    *,
    episodes: int,
    run_dir: str | Path,
    log_path: str | Path,
    worlds: Optional[int] = None,
    max_steps: Optional[int] = None,
    config_path: Optional[str | Path] = None,
    monitor: Any = None,
    planner_factory: Optional[Callable[..., Any]] = None,
    env_factory: Optional[Callable[..., Any]] = None,
    health_check: Optional[Callable[..., Mapping[str, Any]]] = None,
    cli_arguments: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Evaluate exactly one initial episode in every requested world."""

    torch = _torch()
    evaluation_started = time.perf_counter()
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("evaluation episodes must be positive")
    evaluation_config, base_seed, evaluation_seed = (
        _config_with_evaluation_seed(config)
    )
    capacity = int(
        evaluation_config.physics.worlds if worlds is None else worlds
    )
    if capacity < episodes:
        raise ValueError(
            "--worlds/config physics.worlds must be at least --episodes so "
            "all initial conditions can be committed before the first step"
        )
    configured_steps = int(evaluation_config.physics.max_episode_steps)
    step_limit = configured_steps if max_steps is None else int(max_steps)
    if step_limit <= 0 or step_limit > configured_steps:
        raise ValueError(
            "max_steps must lie in [1, config.physics.max_episode_steps]"
        )
    output_root = Path(run_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log = Path(log_path).expanduser().resolve()
    manifest_path = output_root / "initial_conditions_manifest.json"
    summary_path = output_root / "summary.json"
    summary_hash_path = output_root / "summary.sha256.json"
    log_hash_path = log.with_name(log.name + ".sha256.json")
    base_config_payload = config.to_dict()
    evaluation_config_payload = evaluation_config.to_dict()
    base_config_hash = canonical_sha256(base_config_payload)
    evaluation_config_hash = canonical_sha256(evaluation_config_payload)
    planner_builder = planner_factory or _default_planner_factory
    environment_builder = env_factory or _default_env_factory
    backend_health = health_check or _default_health_check
    # Use one world per requested episode.  ``worlds`` is a declared capacity
    # gate, not permission to add unreported background episodes.
    live_worlds = episodes
    planner = planner_builder(
        evaluation_config,
        worlds=live_worlds,
        device=str(evaluation_config.runtime.device),
    )
    planner_parameters = _planner_parameters(planner)
    planner_hash = canonical_sha256(planner_parameters)
    provenance: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "config": {
            "name": str(evaluation_config.name),
            "canonical_sha256": evaluation_config_hash,
            "base_canonical_sha256": base_config_hash,
            "evaluation_canonical_sha256": evaluation_config_hash,
        },
        "contracts": {
            "schema_hash": SCHEMA_HASH,
            "task_geometry_hash": TASK_GEOMETRY_HASH,
        },
        "assets": dict(TASK_ASSET_HASHES),
        "planner": {
            "class": (
                f"{type(planner).__module__}.{type(planner).__qualname__}"
            ),
            "provenance": planner_parameters,
            "provenance_canonical_sha256": planner_hash,
            "parameters": planner_parameters,
            "parameters_canonical_sha256": planner_hash,
        },
        "source": _implementation_source_records(),
        "checkpoints": {},
        "residual": {
            "mode": "disabled_zero",
            "dimensions": RESIDUAL_ACTION_DIM,
        },
        "base_seed": base_seed,
        "evaluation_seed": evaluation_seed,
        "evaluation_seed_offset": EVALUATION_SEED_OFFSET,
        "root_seed": evaluation_seed,
        "first_episode_only": True,
    }
    if cli_arguments is not None:
        provenance["invocation"] = {
            "argv": [str(value) for value in cli_arguments]
        }
    if config_path is not None:
        resolved_config = Path(config_path).expanduser().resolve()
        if not resolved_config.is_file():
            raise FileNotFoundError(f"config source is missing: {resolved_config}")
        provenance["config"]["path"] = str(resolved_config)
        provenance["config"]["sha256"] = sha256_file(resolved_config)
    env = None
    records: list[dict[str, Any]] = []
    zero_residual = None
    last_info: Mapping[str, Any] | None = None
    active = None
    vector_steps = 0
    last_health: dict[str, Any] = {}
    rollout_started = 0.0
    try:
        env = environment_builder(
            evaluation_config,
            worlds=live_worlds,
            baseline="astar_spline",
            nominal_planner=planner,
            curriculum_progress=1.0,
        )
        if int(env.worlds) != live_worlds:
            raise RuntimeError("environment world count differs from episodes")
        environment_source = _source_record(type(env))
        if environment_source is not None:
            provenance["source"]["runtime_environment"] = (
                environment_source
            )
        manifest_path, manifest_sha256, manifest = (
            write_initial_conditions_manifest(
                manifest_path,
                env=env,
                episodes=episodes,
                base_seed=base_seed,
                evaluation_seed=evaluation_seed,
                base_config_hash=base_config_hash,
                evaluation_config_hash=evaluation_config_hash,
                planner_hash=planner_hash,
            )
        )
        provenance["initial_conditions_manifest"] = {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "rows_canonical_sha256": canonical_sha256(manifest["rows"]),
        }
        append_history(
            log,
            {
                "schema": EVALUATION_SCHEMA,
                "event": "evaluation_start",
                "created_utc": utc_now(),
                "episodes": episodes,
                "worlds": live_worlds,
                "max_steps": step_limit,
                "base_seed": base_seed,
                "evaluation_seed": evaluation_seed,
                "initial_conditions_manifest": (
                    provenance["initial_conditions_manifest"]
                ),
                "provenance": provenance,
            },
        )
        if monitor is not None:
            monitor.set_stage(
                "evaluation",
                current=0,
                total=episodes,
                unit="episode",
                message="Global A*--IK first-episode rollouts are running.",
                metrics={
                    "completed": 0,
                    "successes": 0,
                    "failures": 0,
                    "truncations": 0,
                },
                artifacts={
                    "initial_conditions_manifest": {
                        "path": str(manifest_path),
                        "sha256": manifest_sha256,
                    }
                },
                stale_after_seconds=900.0,
            )
        zero_residual = torch.zeros(
            (live_worlds, RESIDUAL_ACTION_DIM),
            device=env.device,
            dtype=torch.float32,
        )
        active = torch.ones(
            live_worlds, device=env.device, dtype=torch.bool
        )
        initial_rows = manifest["rows"]
        log_every = int(
            getattr(
                evaluation_config.runtime,
                "log_every_vector_steps",
                50,
            )
        )
        if log_every <= 0:
            raise ValueError("runtime.log_every_vector_steps must be positive")
        last_health = dict(
            backend_health(
                env, context="A* evaluation environment initialization"
            )
        )
        if monitor is not None:
            monitor.heartbeat(
                current=0,
                message="Global A*--IK environment passed health checks.",
                metrics={
                    "completed": 0,
                    "vector_steps": 0,
                },
                health=last_health,
                force=True,
            )
        rollout_started = time.perf_counter()
        for vector_step in range(1, step_limit + 1):
            result = env.step(residual_action=zero_residual)
            vector_steps = vector_step
            last_info = result.info
            first_done = result.done & active
            if bool(first_done.any()):
                planner_diagnostics_snapshot = dict(
                    _planner_diagnostics(planner)
                )
                world_ids = (
                    torch.nonzero(first_done, as_tuple=False)
                    .flatten()
                    .detach()
                    .cpu()
                    .tolist()
                )
                for world_id in world_ids:
                    row = _episode_record(
                        info=result.info,
                        world_id=int(world_id),
                        episode_index=int(world_id),
                        initial=initial_rows[int(world_id)],
                        planner=planner,
                        worlds=live_worlds,
                        forced_truncation=False,
                        max_steps=step_limit,
                        planner_diagnostics=(
                            planner_diagnostics_snapshot
                        ),
                    )
                    records.append(row)
                    append_history(
                        log,
                        {
                            "schema": EPISODE_SCHEMA,
                            "event": "episode",
                            "updated_utc": utc_now(),
                            "episode": row,
                        },
                    )
                active[first_done] = False
            monitor_due = monitor is not None and monitor.due
            periodic_health = vector_step % log_every == 0
            health = None
            if periodic_health or monitor_due:
                health = dict(
                    backend_health(
                        env,
                        context=(
                            f"A* evaluation vector step {vector_step}"
                        ),
                    )
                )
                last_health = health
            # While evaluation continues, every terminal physics row is reset,
            # including rows whose first episode was already retained.
            # ``active`` controls accounting only; it must never strand an
            # inactive terminal body in the shared simulator batch.  Once the
            # final target episode completes, skip reset because the env closes
            # immediately and replanning would pollute final diagnostics.
            done_any = bool(result.done.any())
            if done_any and bool(active.any()):
                env.reset(result.done)
            emit_monitor = monitor is not None and (
                monitor_due or not bool(active.any())
            )
            if emit_monitor:
                if health is None:
                    health = dict(
                        backend_health(
                            env,
                            context=(
                                "A* evaluation heartbeat vector step "
                                f"{vector_step}"
                            ),
                        )
                    )
                    last_health = health
                outcomes = Counter(row["outcome"] for row in records)
                elapsed = max(
                    time.perf_counter() - rollout_started, 1.0e-12
                )
                monitor.heartbeat(
                    current=len(records),
                    message=(
                        f"Global A*--IK completed {len(records)}/"
                        f"{episodes} first episodes."
                    ),
                    metrics={
                        "completed": len(records),
                        "successes": outcomes["success"],
                        "failures": outcomes["failure"],
                        "truncations": outcomes["truncated"],
                        "vector_steps": vector_step,
                        "wall_duration_seconds": elapsed,
                        "throughput_episodes_per_second": (
                            len(records) / elapsed
                        ),
                    },
                    health=last_health,
                    force=not bool(active.any()),
                )
            if not bool(active.any()):
                break
        if active is not None and bool(active.any()):
            if last_info is None:
                raise RuntimeError("evaluation produced no physics transition")
            world_ids = (
                torch.nonzero(active, as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            )
            planner_diagnostics_snapshot = dict(
                _planner_diagnostics(planner)
            )
            for world_id in world_ids:
                row = _episode_record(
                    info=last_info,
                    world_id=int(world_id),
                    episode_index=int(world_id),
                    initial=initial_rows[int(world_id)],
                    planner=planner,
                    worlds=live_worlds,
                    forced_truncation=True,
                    max_steps=step_limit,
                    planner_diagnostics=planner_diagnostics_snapshot,
                )
                records.append(row)
                append_history(
                    log,
                    {
                        "schema": EPISODE_SCHEMA,
                        "event": "episode",
                        "updated_utc": utc_now(),
                        "episode": row,
                    },
                )
            active.zero_()
        if vector_steps == 0:
            raise RuntimeError("evaluation produced no physics transition")
        if vector_steps % log_every != 0:
            last_health = dict(
                backend_health(
                    env,
                    context=f"A* evaluation final step {vector_steps}",
                )
            )
        records.sort(key=lambda row: int(row["episode_index"]))
        if len(records) != episodes:
            raise RuntimeError(
                f"retained {len(records)} first episodes, expected {episodes}"
            )
        rollout_wall_duration = max(
            time.perf_counter() - rollout_started, 1.0e-12
        )
        wall_duration = max(
            time.perf_counter() - evaluation_started, 1.0e-12
        )
        execution = {
            "vector_steps": int(vector_steps),
            "wall_duration_seconds": float(wall_duration),
            "rollout_wall_duration_seconds": float(
                rollout_wall_duration
            ),
            "throughput_episodes_per_second": float(
                episodes / wall_duration
            ),
            "throughput_vector_steps_per_second": float(
                vector_steps / rollout_wall_duration
            ),
            "last_health": dict(last_health),
        }
        provenance["execution"] = dict(execution)
        final_planner_diagnostics = _global_planner_diagnostics(
            planner, worlds=live_worlds
        )
        summary = build_summary(
            records,
            provenance=provenance,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            log_path=log,
            execution=execution,
            planner_diagnostics=final_planner_diagnostics,
        )
        summary["outputs"] = {
            "summary": {
                "path": str(summary_path),
                "sha256_sidecar": str(summary_hash_path),
            },
            "episode_log": {
                "path": str(log),
                "sha256_sidecar": str(log_hash_path),
            },
        }
        atomic_write_json(summary_path, summary)
        summary_file_sha256 = sha256_file(summary_path)
        atomic_write_json(
            summary_hash_path,
            {
                "algorithm": "sha256",
                "path": str(summary_path),
                "sha256": summary_file_sha256,
            },
        )
        append_history(
            log,
            {
                "schema": SUMMARY_SCHEMA,
                "event": "evaluation_summary",
                "updated_utc": utc_now(),
                "summary_file": {
                    "path": str(summary_path),
                    "sha256": summary_file_sha256,
                },
                "summary": summary,
            },
        )
        atomic_write_json(
            log_hash_path,
            {
                "algorithm": "sha256",
                "path": str(log),
                "sha256": sha256_file(log),
            },
        )
        if monitor is not None:
            monitor.heartbeat(
                current=episodes,
                message="Global A*--IK first-episode evaluation completed.",
                metrics={
                    key: summary[key]
                    for key in (
                        "episodes",
                        "successes",
                        "failures",
                        "truncations",
                        "success_rate",
                    )
                },
                artifacts=summary["outputs"],
                force=True,
            )
        return summary
    finally:
        if env is not None:
            env.close()
        close_planner = getattr(planner, "close", None)
        if callable(close_planner):
            close_planner()


__all__ = [
    "EVALUATION_SCHEMA",
    "EPISODE_SCHEMA",
    "INITIAL_CONDITIONS_SCHEMA",
    "SUMMARY_SCHEMA",
    "build_summary",
    "canonical_sha256",
    "evaluate_astar_spline",
    "sha256_file",
    "write_initial_conditions_manifest",
]
