"""Low-overhead, crash-tolerant live monitoring for hpc runs.

The training process is the only writer of ``status/live.json``.  A separate
stdlib-only sidecar writes ``status/system.json``.  Readers therefore never
need to load a multi-gigabyte checkpoint or attach to the CUDA process.
"""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Optional, Sequence
import uuid


LIVE_SCHEMA = "screw-hpc-live-v1"
SYSTEM_SCHEMA = "screw-hpc-system-v1"
STOP_SCHEMA = "screw-hpc-stop-v1"
RUN_MANIFEST_SCHEMA = "screw-hpc-run-manifest-v1"
EPISODE_LOG_SCHEMA = "screw-hpc-episodes-v1"
DEFAULT_RECENT_EPISODES = 2048
DEFAULT_RECENT_WINDOWS = (128, 512, 2048)
TERMINAL_STATES = frozenset({"completed", "interrupted", "failed"})
TASK_PHASE_NAMES = (
    "pregrasp",
    "close",
    "lift",
    "align",
    "insert",
    "engage",
    "power",
    "lr_open",
    "lr_reset",
    "lr_close",
    "ud_open",
    "ud_reset",
    "ud_close",
    "final_hold",
    "complete",
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_SUFFIXES = frozenset(
    {".py", ".json", ".xml", ".stl", ".sh", ".pbs", ".slurm", ".txt"}
)
_SOURCE_EXCLUDED_PARTS = frozenset(
    {
        ".cache",
        ".git",
        ".pytest_cache",
        "RL_env",
        "__pycache__",
        "checkpoints",
        "datasets",
        "logs",
        "outputs",
        "runs",
        "status",
    }
)
_SOURCE_EXCLUDED_NAMES = frozenset(
    {
        # Recreated by setup_hpc.sh and intentionally contains a timestamp,
        # hostname, and pip snapshot.  It is environment provenance, not
        # training source, so refreshing it must not invalidate a checkpoint.
        "environment-manifest.txt",
    }
)
_TRAINING_CONTRACT_PATHS = (
    "diffusion_policy.py",
    "kinematics.py",
    "requirements-hpc.txt",
    "screw_config.py",
    "screw_control.py",
    "assets/Gripper.xml",
    "assets/Project_hpc.xml",
    "assets/Share_hpc.xml",
    "assets/mesh/backbased_mujoco.STL",
    "assets/mesh/base_mujoco.STL",
    "assets/mesh/frontbased_mujoco.STL",
    "assets/mesh/handbase_module_mujoco.STL",
    "assets/mesh/linear_module_mujoco.STL",
    "assets/mesh/long.STL",
    "assets/mesh/short.STL",
    "s_hpc/analytic_ik.py",
    "s_hpc/contracts.py",
    "s_hpc/diffusion_runtime.py",
    "s_hpc/env.py",
    "s_hpc/mjwarp_backend.py",
    "s_hpc/mjwarp_compat.py",
    "s_hpc/replay.py",
    "s_hpc/rollout_audit.py",
    "s_hpc/supervisor.py",
    "s_hpc/task.py",
    "s_hpc/torch_compat.py",
    "s_hpc/tqc.py",
    "s_hpc/workflows.py",
    "s_hpc/wrench_kernel.py",
)


def utc_now() -> str:
    """Return an unambiguous, machine-readable UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_utc(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def heartbeat_age_seconds(
    snapshot: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
) -> Optional[float]:
    stamp = parse_utc(snapshot.get("updated_utc"))
    if stamp is None:
        return None
    reference = now or datetime.now(timezone.utc)
    return max(0.0, (reference - stamp).total_seconds())


def _finite_number(value: object) -> object:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def jsonable(value: Any) -> Any:
    """Convert common tensor/scalar/path values into strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int, float)) or value is None:
        return _finite_number(value)
    if hasattr(value, "item"):
        try:
            return _finite_number(value.item())
        except (RuntimeError, TypeError, ValueError):
            pass
    return str(value)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace a small JSON status file on a shared filesystem."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                jsonable(dict(payload)),
                stream,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def read_json(path: str | Path) -> Optional[dict[str, Any]]:
    """Read an atomic JSON file, returning ``None`` if it is absent/invalid."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def append_history(path: str | Path, event: Mapping[str, Any]) -> None:
    """Append one compact snapshot; readers deliberately ignore bad tail lines."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                jsonable(dict(event)),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()


def tail_jsonl(path: str | Path, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    source = Path(path).expanduser().resolve()
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    parsed: list[dict[str, Any]] = []
    for line in lines[-max(count * 2, count) :]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            parsed.append(item)
    return parsed[-count:]


def describe_paths(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    """Return metadata only; checkpoint contents are never loaded."""

    result: dict[str, Any] = {}
    for name, raw_path in paths.items():
        path = Path(raw_path).expanduser().resolve()
        entry: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
        }
        if entry["exists"]:
            try:
                stat = path.stat()
                entry.update(
                    {
                        "size_bytes": int(stat.st_size),
                        "modified_utc": datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc
                        ).isoformat(timespec="seconds"),
                    }
                )
            except OSError as exc:
                entry["stat_error"] = str(exc)
        result[str(name)] = entry
    return result


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with deterministic strict encoding."""

    encoded = json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_path_set(paths: Sequence[Path]) -> tuple[str, dict[str, str]]:
    entries: dict[str, str] = {}
    digest = hashlib.sha256()
    for source in sorted(paths, key=lambda item: item.as_posix()):
        relative = source.relative_to(_PACKAGE_ROOT).as_posix()
        file_hash = _sha256_file(source)
        entries[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), entries


def _source_fingerprints() -> dict[str, Any]:
    contract_paths = [
        _PACKAGE_ROOT / relative
        for relative in _TRAINING_CONTRACT_PATHS
        if (_PACKAGE_ROOT / relative).is_file()
    ]
    missing = [
        relative
        for relative in _TRAINING_CONTRACT_PATHS
        if not (_PACKAGE_ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"training-contract files are missing: {missing}"
        )
    contract_hash, contract_files = _hash_path_set(contract_paths)
    source_paths = [
        path
        for path in _PACKAGE_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in _SOURCE_SUFFIXES
            and path.name != "run_manifest.json"
            and path.name not in _SOURCE_EXCLUDED_NAMES
            and not path.name.startswith("._")
            and not _SOURCE_EXCLUDED_PARTS.intersection(
                path.relative_to(_PACKAGE_ROOT).parts
            )
        )
    ]
    source_hash, source_files = _hash_path_set(source_paths)
    return {
        "training_contract_hash": contract_hash,
        "training_contract_files": contract_files,
        "source_tree_sha256": source_hash,
        "source_file_count": len(source_paths),
        "source_files": source_files,
    }


def _dependency_versions() -> dict[str, Optional[str]]:
    result: dict[str, Optional[str]] = {}
    for name in (
        "mujoco",
        "mujoco-warp",
        "numpy",
        "torch",
        "warp-lang",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _runtime_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "dependencies": _dependency_versions(),
    }
    try:
        import torch

        cuda: dict[str, Any] = {
            "build": torch.version.cuda,
            "available": bool(torch.cuda.is_available()),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            cuda.update(
                {
                    "device_index": int(device),
                    "device_name": torch.cuda.get_device_name(device),
                    "device_count": int(torch.cuda.device_count()),
                    "total_memory_bytes": int(properties.total_memory),
                    "capability": [
                        int(properties.major),
                        int(properties.minor),
                    ],
                }
            )
        result["torch_cuda"] = cuda
    except (ImportError, RuntimeError, ValueError) as exc:
        result["torch_cuda"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return result


def _config_payload(
    config_name: str,
    config: Optional[Mapping[str, Any] | object],
) -> dict[str, Any]:
    if config is not None:
        if isinstance(config, Mapping):
            return dict(config)
        to_dict = getattr(config, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
            if isinstance(value, Mapping):
                return dict(value)
        raise TypeError("monitor config must be a mapping or expose to_dict()")
    candidate = _PACKAGE_ROOT / "configs" / f"{config_name}.json"
    if candidate.is_file():
        value = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            return dict(value)
    # Unit tests and small external callers may use a synthetic config name.
    # Production entrypoints always pass a full config or resolve this file.
    return {"name": str(config_name)}


def build_run_manifest(
    *,
    run_id: str,
    run_created_utc: str,
    config_name: str,
    config: Optional[Mapping[str, Any] | object] = None,
) -> dict[str, Any]:
    """Build the immutable provenance contract for one run directory."""

    payload = _config_payload(config_name, config)
    try:
        from .contracts import (
            BACKEND,
            CHECKPOINT_FORMAT,
            SCHEMA_HASH,
            TASK_GEOMETRY_HASH,
        )
    except ImportError:  # pragma: no cover - direct module fallback
        from s_hpc.contracts import (  # type: ignore
            BACKEND,
            CHECKPOINT_FORMAT,
            SCHEMA_HASH,
            TASK_GEOMETRY_HASH,
        )
    runtime = payload.get("runtime", {})
    demos = payload.get("demos", {})
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "run": {
            "id": str(run_id),
            "created_utc": str(run_created_utc),
        },
        "config": {
            "name": str(config_name),
            "sha256": canonical_sha256(payload),
            "payload": payload,
        },
        "seeds": {
            "runtime": (
                runtime.get("seed") if isinstance(runtime, Mapping) else None
            ),
            "demonstrations": (
                demos.get("seed") if isinstance(demos, Mapping) else None
            ),
            "evaluation": (
                int(runtime["seed"]) + 100_000
                if isinstance(runtime, Mapping)
                and isinstance(runtime.get("seed"), int)
                else None
            ),
            "deterministic": (
                runtime.get("deterministic")
                if isinstance(runtime, Mapping)
                else None
            ),
        },
        "contracts": {
            "backend": BACKEND,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "schema_hash": SCHEMA_HASH,
            "task_geometry_hash": TASK_GEOMETRY_HASH,
            **_source_fingerprints(),
        },
        "environment": _runtime_environment(),
        "invocation": {
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        },
    }


def ensure_run_manifest(
    run_dir: str | Path,
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Create one immutable manifest or reject an incompatible resume."""

    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_manifest.json"
    existing = read_json(path)
    if existing is None and path.exists():
        raise ValueError(f"run manifest is invalid JSON: {path}")
    if existing is None:
        material = [
            item.name
            for item in root.iterdir()
            if item.name not in {"status", "run_manifest.json"}
        ]
        if material:
            raise ValueError(
                "existing run artifacts have no immutable run manifest; "
                f"move them aside or start a fresh run: {sorted(material)}"
            )
        atomic_write_json(path, candidate)
        existing = read_json(path)
        if existing is None:
            raise OSError(f"failed to read newly written run manifest: {path}")
    if existing.get("schema") != RUN_MANIFEST_SCHEMA:
        raise ValueError("run manifest schema is incompatible")
    existing_config = existing.get("config", {})
    if (
        not isinstance(existing_config, Mapping)
        or canonical_sha256(existing_config.get("payload"))
        != existing_config.get("sha256")
    ):
        raise ValueError(
            "run manifest config payload/hash integrity check failed"
        )

    def selected(value: Mapping[str, Any]) -> tuple[Any, ...]:
        config_value = value.get("config", {})
        contracts = value.get("contracts", {})
        return (
            config_value.get("sha256"),
            contracts.get("backend"),
            contracts.get("checkpoint_format"),
            contracts.get("schema_hash"),
            contracts.get("task_geometry_hash"),
            contracts.get("training_contract_hash"),
            contracts.get("source_tree_sha256"),
        )

    if selected(existing) != selected(candidate):
        raise ValueError(
            "run manifest differs from the selected config/code/contracts; "
            "resume is refused, use a fresh run directory"
        )
    return existing, _sha256_file(path)


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class EpisodeTracker:
    """Exact terminal metrics, multi-window trends, and diagnostics."""

    def __init__(
        self,
        recent_episodes: Optional[int] = None,
        *,
        scope: str = "stochastic_training_rollouts",
        recent_windows: Optional[Sequence[int]] = None,
        curriculum_warmup_fraction: float = 0.05,
        curriculum_full_fraction: float = 0.70,
        event_path: Optional[str | Path] = None,
    ) -> None:
        if recent_windows is None:
            windows = (
                (int(recent_episodes),)
                if recent_episodes is not None
                else DEFAULT_RECENT_WINDOWS
            )
        else:
            windows = tuple(int(value) for value in recent_windows)
            if recent_episodes is not None:
                windows = (*windows, int(recent_episodes))
        if not windows or any(value <= 0 for value in windows):
            raise ValueError("recent episode windows must be positive")
        self.recent_windows = tuple(sorted(set(windows)))
        self.recent_episodes = max(self.recent_windows)
        if not (
            0.0
            <= float(curriculum_warmup_fraction)
            < float(curriculum_full_fraction)
            <= 1.0
        ):
            raise ValueError("episode curriculum boundaries are invalid")
        self.curriculum_warmup_fraction = float(
            curriculum_warmup_fraction
        )
        self.curriculum_full_fraction = float(curriculum_full_fraction)
        self.scope = str(scope)
        self.event_path = (
            None
            if event_path is None
            else Path(event_path).expanduser().resolve()
        )
        self.event_log_error: Optional[str] = None
        self.recent: deque[dict[str, Any]] = deque(
            maxlen=self.recent_episodes
        )
        self.completed = 0
        self.successes = 0
        self.failures = 0
        self.truncations = 0
        self.return_sum = 0.0
        self.depth_sum_m = 0.0
        self.turns_sum = 0.0
        self.lateral_sum_m = 0.0
        self.axis_sum_rad = 0.0
        self.speed_sum = 0.0
        self.length_sum = 0.0
        self.ik_failure_sum = 0
        self.ik_anchor_fallback_sum = 0
        self.ik_safety_stop_sum = 0
        self.failure_reasons: Counter[str] = Counter()
        self.curriculum_groups: dict[str, dict[str, float]] = {
            name: self._empty_group()
            for name in ("warmup", "ramp", "full")
        }
        self.terminal_phase_groups: dict[str, dict[str, float]] = {}

    @staticmethod
    def _empty_group() -> dict[str, float]:
        return {
            "completed": 0,
            "successes": 0,
            "failures": 0,
            "truncations": 0,
            "return_sum": 0.0,
            "depth_sum_m": 0.0,
            "turns_sum": 0.0,
            "steps_sum": 0.0,
        }

    @staticmethod
    def _wilson_interval(
        successes: int, count: int
    ) -> Optional[dict[str, float]]:
        if count <= 0:
            return None
        z = 1.959963984540054
        proportion = successes / count
        denominator = 1.0 + z * z / count
        center = (
            proportion + z * z / (2.0 * count)
        ) / denominator
        radius = (
            z
            * math.sqrt(
                proportion * (1.0 - proportion) / count
                + z * z / (4.0 * count * count)
            )
            / denominator
        )
        return {
            "lower": max(0.0, center - radius),
            "upper": min(1.0, center + radius),
        }

    @staticmethod
    def _percentile(
        records: Sequence[Mapping[str, Any]],
        key: str,
        fraction: float,
    ) -> Optional[float]:
        if not records:
            return None
        values = sorted(float(item.get(key, 0.0)) for item in records)
        position = fraction * (len(values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        blend = position - lower
        return values[lower] * (1.0 - blend) + values[upper] * blend

    def _curriculum_segment(self, progress: float) -> str:
        if progress < self.curriculum_warmup_fraction:
            return "warmup"
        if progress < self.curriculum_full_fraction:
            return "ramp"
        return "full"

    @staticmethod
    def _phase_name(value: Any) -> str:
        try:
            index = int(value)
        except (TypeError, ValueError):
            return "unknown"
        if 0 <= index < len(TASK_PHASE_NAMES):
            return TASK_PHASE_NAMES[index]
        return f"unknown_{index}"

    @staticmethod
    def _update_group(
        group: dict[str, float],
        record: Mapping[str, Any],
        *,
        succeeded: bool,
        failed: bool,
        truncated: bool,
    ) -> None:
        group["completed"] += 1
        group["successes"] += int(succeeded)
        group["failures"] += int(failed)
        group["truncations"] += int(truncated)
        group["return_sum"] += float(record["return"])
        group["depth_sum_m"] += float(record["depth_m"])
        group["turns_sum"] += float(record["turns"])
        group["steps_sum"] += float(record["length"])

    @classmethod
    def _group_snapshot(
        cls, groups: Mapping[str, Mapping[str, float]]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, raw in sorted(groups.items()):
            count = int(raw.get("completed", 0))
            successes = int(raw.get("successes", 0))
            result[str(name)] = {
                "completed": count,
                "successes": successes,
                "failures": int(raw.get("failures", 0)),
                "truncations": int(raw.get("truncations", 0)),
                "success_rate": successes / count if count else None,
                "success_rate_ci95": cls._wilson_interval(
                    successes, count
                ),
                "mean_return": (
                    float(raw.get("return_sum", 0.0)) / count
                    if count
                    else None
                ),
                "mean_depth_m": (
                    float(raw.get("depth_sum_m", 0.0)) / count
                    if count
                    else None
                ),
                "mean_turns": (
                    float(raw.get("turns_sum", 0.0)) / count
                    if count
                    else None
                ),
                "mean_episode_steps": (
                    float(raw.get("steps_sum", 0.0)) / count
                    if count
                    else None
                ),
            }
        return result

    @staticmethod
    def _selected(info: Mapping[str, Any], key: str, done_ids: Any) -> list[Any]:
        value = info.get(key)
        if value is None:
            return []
        try:
            value = value[done_ids]
        except (IndexError, KeyError, TypeError):
            pass
        return _sequence(value)

    def add_from_info(
        self,
        info: Mapping[str, Any],
        done_ids: Any,
        *,
        episode_lengths: Any,
        event_context: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Consume only terminal worlds from one vector-environment step."""

        ids = _sequence(done_ids)
        count = len(ids)
        if count == 0:
            return 0

        def selected(key: str, default: Any) -> list[Any]:
            values = self._selected(info, key, done_ids)
            if not values:
                values = [default] * count
            if len(values) != count:
                raise ValueError(
                    f"terminal metric {key!r} has {len(values)} values, "
                    f"expected {count}"
                )
            return values

        success = selected("success", False)
        failure = selected("failure", False)
        truncated = selected("truncated", False)
        returns = selected("episode_return", 0.0)
        depths = selected("depth", 0.0)
        twists = selected("twist", 0.0)
        lateral_errors = selected("lateral_error", 0.0)
        axis_errors = selected("axis_error", 0.0)
        speeds = selected("speed", 0.0)
        lengths = _sequence(episode_lengths)
        if len(lengths) != count:
            raise ValueError(
                f"episode_lengths has {len(lengths)} values, expected {count}"
            )
        ik_failures = selected("ik_failures", 0)
        ik_fallbacks = selected("ik_anchor_fallbacks", 0)
        ik_stops = selected("ik_safety_stops", 0)
        terminal_phases = selected("terminal_phase", -1)
        curriculum_progress = selected("curriculum_progress", 1.0)
        spawn_radius = selected("spawn_radius_m", 0.0)
        thread_engaged = selected("thread_engaged", False)
        holder_active = selected("holder_active", False)
        grasp_latched = selected("grasp_latched", False)
        handoff_ready = selected("diffusion_handoff_ready", False)
        handoff_hold = selected("diffusion_handoff_hold_steps", 0)
        expert_step = selected("expert_step", -1)
        preinsert_ready = selected("preinsert_alignment_ready", False)
        preinsert_hold = selected("preinsert_alignment_hold", False)
        preinsert_stable = selected(
            "preinsert_alignment_stable_steps", 0
        )
        stable_grasp_support = selected(
            "stable_grasp_support", False
        )
        all_four_contact = selected("all_four_contact", False)
        primary_pair_contact = selected(
            "primary_pair_contact", False
        )
        two_finger_contact = selected(
            "two_finger_contact", False
        )
        capture_ready = selected("capture_alignment_ready", False)
        capture_hold = selected("capture_alignment_hold", False)
        capture_stable = selected(
            "capture_alignment_stable_steps", 0
        )
        capture_head_standoff = selected(
            "capture_head_bottom_standoff", 0.0
        )
        shaft_captured = selected("shaft_captured", False)
        secondary_released = selected("secondary_released", False)
        thread_ready = selected("thread_engagement_ready", False)
        thread_hold = selected("thread_engagement_hold", False)
        thread_regrasp_hold = selected("thread_regrasp_hold", False)
        thread_stable = selected("thread_ready_stable_steps", 0)
        world_ids = selected("world_id", -1)
        spawn_x = selected("spawn_x_m", 0.0)
        spawn_y = selected("spawn_y_m", 0.0)
        socket_x = selected("socket_mouth_x_m", 0.0)
        socket_y = selected("socket_mouth_y_m", 0.0)
        socket_angle = selected("socket_angle_deg", 45.0)
        reason_values = {
            "nonfinite": selected("nonfinite", False),
            "fell_below_table": selected("fell_below_table", False),
            "lateral_escape": selected("lateral_escape", False),
            "safety_stop": selected("safety_stop", False),
            "contact_constraint_violation": selected(
                "contact_constraint_violation", False
            ),
        }

        outcomes = [
            (
                bool(success[index]),
                bool(failure[index]),
                bool(truncated[index]),
            )
            for index in range(count)
        ]
        for succeeded, failed, was_truncated in outcomes:
            if sum((succeeded, failed, was_truncated)) > 1:
                raise ValueError(
                    "terminal success/failure/truncated flags must be "
                    "mutually exclusive"
                )

        batch_records: list[dict[str, Any]] = []
        for index in range(count):
            succeeded, failed, was_truncated = outcomes[index]
            if succeeded:
                outcome = "success"
            elif failed:
                outcome = "failure"
            elif was_truncated:
                outcome = "truncated"
            else:
                outcome = "unknown"
            reasons = [
                name
                for name, flags in reason_values.items()
                if bool(flags[index])
            ]
            phase = self._phase_name(terminal_phases[index])
            if (failed or was_truncated) and (
                not bool(grasp_latched[index])
                and not bool(holder_active[index])
            ):
                reasons.append(
                    "grasp_not_acquired"
                    if phase in {"pregrasp", "close"}
                    else "grasp_lost"
                )
            if bool(reason_values["safety_stop"][index]):
                reasons.append("ik_exhausted")
            if was_truncated:
                if (
                    int(handoff_hold[index]) > 0
                    and not bool(handoff_ready[index])
                ):
                    reasons.append("handoff_timeout")
                elif bool(preinsert_hold[index]):
                    reasons.append("preinsert_gate_timeout")
                elif bool(capture_hold[index]):
                    reasons.append("capture_gate_timeout")
                elif bool(thread_hold[index]):
                    reasons.append("thread_gate_timeout")
                elif bool(thread_regrasp_hold[index]):
                    reasons.append("thread_regrasp_gate_timeout")
                elif phase == "insert" and not bool(
                    shaft_captured[index]
                ):
                    reasons.append("capture_gate_timeout")
                elif phase == "insert":
                    reasons.append("thread_gate_timeout")
                elif phase == "engage" and not bool(thread_engaged[index]):
                    reasons.append("thread_not_engaged")
                elif phase in {
                    "power",
                    "lr_open",
                    "lr_reset",
                    "lr_close",
                    "ud_open",
                    "ud_reset",
                    "ud_close",
                }:
                    reasons.append("rotation_or_regrasp_stall")
                elif phase in {"final_hold", "complete"}:
                    reasons.append("final_tolerance_timeout")
                else:
                    reasons.append(f"timeout_{phase}")
            if failed and not reasons:
                reasons = ["unknown_failure"]
            if was_truncated and not reasons:
                reasons = ["unknown_timeout"]
            reasons = list(dict.fromkeys(reasons))
            for reason in reasons:
                self.failure_reasons[reason] += 1
            progress = min(
                max(float(curriculum_progress[index]), 0.0), 1.0
            )
            curriculum_segment = self._curriculum_segment(progress)
            record = {
                "episode_index": self.completed,
                "world_id": int(world_ids[index]),
                "outcome": outcome,
                "return": float(returns[index]),
                "depth_m": float(depths[index]),
                "turns": float(twists[index]) / (2.0 * math.pi),
                "lateral_error_m": float(lateral_errors[index]),
                "axis_error_rad": float(axis_errors[index]),
                "speed": float(speeds[index]),
                "length": int(lengths[index]),
                "ik_failures": int(ik_failures[index]),
                "ik_anchor_fallbacks": int(ik_fallbacks[index]),
                "ik_safety_stops": int(ik_stops[index]),
                "terminal_phase": phase,
                "curriculum_progress": progress,
                "curriculum_segment": curriculum_segment,
                "spawn_radius_m": float(spawn_radius[index]),
                "thread_engaged": bool(thread_engaged[index]),
                "holder_active": bool(holder_active[index]),
                "grasp_latched": bool(grasp_latched[index]),
                "diffusion_handoff_ready": bool(handoff_ready[index]),
                "diffusion_handoff_hold_steps": int(handoff_hold[index]),
                "expert_step": int(expert_step[index]),
                "preinsert_alignment_ready": bool(
                    preinsert_ready[index]
                ),
                "preinsert_alignment_hold": bool(
                    preinsert_hold[index]
                ),
                "preinsert_alignment_stable_steps": int(
                    preinsert_stable[index]
                ),
                "stable_grasp_support": bool(
                    stable_grasp_support[index]
                ),
                "all_four_contact": bool(all_four_contact[index]),
                "primary_pair_contact": bool(
                    primary_pair_contact[index]
                ),
                "two_finger_contact": bool(
                    two_finger_contact[index]
                ),
                "capture_alignment_ready": bool(
                    capture_ready[index]
                ),
                "capture_alignment_hold": bool(
                    capture_hold[index]
                ),
                "capture_alignment_stable_steps": int(
                    capture_stable[index]
                ),
                "capture_head_bottom_standoff_m": float(
                    capture_head_standoff[index]
                ),
                "shaft_captured": bool(shaft_captured[index]),
                "secondary_released": bool(
                    secondary_released[index]
                ),
                "thread_engagement_ready": bool(thread_ready[index]),
                "thread_engagement_hold": bool(thread_hold[index]),
                "thread_regrasp_hold": bool(
                    thread_regrasp_hold[index]
                ),
                "thread_ready_stable_steps": int(
                    thread_stable[index]
                ),
                "spawn_xy_m": [
                    float(spawn_x[index]),
                    float(spawn_y[index]),
                ],
                "socket_mouth_xy_m": [
                    float(socket_x[index]),
                    float(socket_y[index]),
                ],
                "socket_angle_deg": float(socket_angle[index]),
                "failure_reasons": reasons,
            }
            batch_records.append(record)
            self.recent.append(record)
            self.completed += 1
            self.successes += int(succeeded)
            self.failures += int(failed)
            self.truncations += int(was_truncated)
            self.return_sum += record["return"]
            self.depth_sum_m += record["depth_m"]
            self.turns_sum += record["turns"]
            self.lateral_sum_m += record["lateral_error_m"]
            self.axis_sum_rad += record["axis_error_rad"]
            self.speed_sum += record["speed"]
            self.length_sum += record["length"]
            self.ik_failure_sum += record["ik_failures"]
            self.ik_anchor_fallback_sum += record["ik_anchor_fallbacks"]
            self.ik_safety_stop_sum += record["ik_safety_stops"]
            self._update_group(
                self.curriculum_groups[curriculum_segment],
                record,
                succeeded=succeeded,
                failed=failed,
                truncated=was_truncated,
            )
            phase_group = self.terminal_phase_groups.setdefault(
                phase, self._empty_group()
            )
            self._update_group(
                phase_group,
                record,
                succeeded=succeeded,
                failed=failed,
                truncated=was_truncated,
            )
        if self.event_path is not None:
            event = {
                "schema": EPISODE_LOG_SCHEMA,
                "event": "terminal_batch",
                "updated_utc": utc_now(),
                "scope": self.scope,
                "episode_index_start": self.completed - count,
                "episode_index_stop": self.completed,
                "terminal_count": count,
                "context": dict(event_context or {}),
                "episodes": batch_records,
            }
            try:
                append_history(self.event_path, event)
                self.event_log_error = None
            except OSError as exc:
                self.event_log_error = f"{type(exc).__name__}: {exc}"
        return count

    @staticmethod
    def _summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        materialized = list(records)
        count = len(materialized)
        outcomes = Counter(str(item.get("outcome", "unknown")) for item in materialized)
        reasons: Counter[str] = Counter()
        for item in materialized:
            reasons.update(str(value) for value in item.get("failure_reasons", []))

        def mean(key: str) -> Optional[float]:
            if not count:
                return None
            return sum(float(item.get(key, 0.0)) for item in materialized) / count

        def std(key: str) -> Optional[float]:
            if not count:
                return None
            average = mean(key)
            assert average is not None
            return math.sqrt(
                sum(
                    (float(item.get(key, 0.0)) - average) ** 2
                    for item in materialized
                )
                / count
            )

        return {
            "window": count,
            "successes": outcomes["success"],
            "failures": outcomes["failure"],
            "truncations": outcomes["truncated"],
            "unknown": outcomes["unknown"],
            "success_rate": outcomes["success"] / count if count else None,
            "success_rate_ci95": EpisodeTracker._wilson_interval(
                outcomes["success"], count
            ),
            "failure_rate": outcomes["failure"] / count if count else None,
            "truncation_rate": outcomes["truncated"] / count if count else None,
            "mean_return": mean("return"),
            "std_return": std("return"),
            "p50_return": EpisodeTracker._percentile(
                materialized, "return", 0.50
            ),
            "p90_return": EpisodeTracker._percentile(
                materialized, "return", 0.90
            ),
            "mean_depth_m": mean("depth_m"),
            "p50_depth_m": EpisodeTracker._percentile(
                materialized, "depth_m", 0.50
            ),
            "p90_depth_m": EpisodeTracker._percentile(
                materialized, "depth_m", 0.90
            ),
            "mean_turns": mean("turns"),
            "p50_turns": EpisodeTracker._percentile(
                materialized, "turns", 0.50
            ),
            "p90_turns": EpisodeTracker._percentile(
                materialized, "turns", 0.90
            ),
            "mean_lateral_error_m": mean("lateral_error_m"),
            "mean_axis_error_rad": mean("axis_error_rad"),
            "mean_speed": mean("speed"),
            "mean_episode_steps": mean("length"),
            "mean_ik_failures": mean("ik_failures"),
            "mean_ik_anchor_fallbacks": mean("ik_anchor_fallbacks"),
            "mean_ik_safety_stops": mean("ik_safety_stops"),
            "failure_reasons": dict(sorted(reasons.items())),
        }

    def snapshot(self) -> dict[str, Any]:
        count = self.completed
        recent_values = list(self.recent)
        windows = {
            str(window): self._summary(recent_values[-window:])
            for window in self.recent_windows
        }
        return {
            "completed": count,
            "successes": self.successes,
            "failures": self.failures,
            "truncations": self.truncations,
            "success_rate": self.successes / count if count else None,
            "success_rate_ci95": self._wilson_interval(
                self.successes, count
            ),
            "failure_rate": self.failures / count if count else None,
            "truncation_rate": self.truncations / count if count else None,
            "mean_return": self.return_sum / count if count else None,
            "mean_depth_m": self.depth_sum_m / count if count else None,
            "mean_turns": self.turns_sum / count if count else None,
            "mean_lateral_error_m": (
                self.lateral_sum_m / count if count else None
            ),
            "mean_axis_error_rad": (
                self.axis_sum_rad / count if count else None
            ),
            "mean_speed": self.speed_sum / count if count else None,
            "mean_episode_steps": self.length_sum / count if count else None,
            "mean_ik_failures": self.ik_failure_sum / count if count else None,
            "mean_ik_anchor_fallbacks": (
                self.ik_anchor_fallback_sum / count if count else None
            ),
            "mean_ik_safety_stops": (
                self.ik_safety_stop_sum / count if count else None
            ),
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
            "recent": self._summary(self.recent),
            "recent_capacity": self.recent_episodes,
            "windows": windows,
            "curriculum": self._group_snapshot(self.curriculum_groups),
            "terminal_phases": self._group_snapshot(
                self.terminal_phase_groups
            ),
            "event_log": (
                {
                    "path": str(self.event_path),
                    "error": self.event_log_error,
                }
                if self.event_path is not None
                else None
            ),
            "scope": self.scope,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "recent_episodes": self.recent_episodes,
            "recent_windows": list(self.recent_windows),
            "scope": self.scope,
            "curriculum_warmup_fraction": (
                self.curriculum_warmup_fraction
            ),
            "curriculum_full_fraction": self.curriculum_full_fraction,
            "recent": list(self.recent),
            "completed": self.completed,
            "successes": self.successes,
            "failures": self.failures,
            "truncations": self.truncations,
            "return_sum": self.return_sum,
            "depth_sum_m": self.depth_sum_m,
            "turns_sum": self.turns_sum,
            "lateral_sum_m": self.lateral_sum_m,
            "axis_sum_rad": self.axis_sum_rad,
            "speed_sum": self.speed_sum,
            "length_sum": self.length_sum,
            "ik_failure_sum": self.ik_failure_sum,
            "ik_anchor_fallback_sum": self.ik_anchor_fallback_sum,
            "ik_safety_stop_sum": self.ik_safety_stop_sum,
            "failure_reasons": dict(self.failure_reasons),
            "curriculum_groups": self.curriculum_groups,
            "terminal_phase_groups": self.terminal_phase_groups,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        version = int(state.get("version", -1))
        if version not in {1, 2}:
            raise ValueError("unsupported episode-monitor checkpoint version")
        stored_capacity = int(state.get("recent_episodes", -1))
        if stored_capacity != self.recent_episodes:
            raise ValueError(
                "episode-monitor recent-window capacity differs from checkpoint"
            )
        if version >= 2:
            stored_windows = tuple(
                int(value) for value in state.get("recent_windows", ())
            )
            if stored_windows != self.recent_windows:
                raise ValueError(
                    "episode-monitor recent windows differ from checkpoint"
                )
            stored_warmup = float(
                state.get(
                    "curriculum_warmup_fraction",
                    self.curriculum_warmup_fraction,
                )
            )
            stored_full = float(
                state.get(
                    "curriculum_full_fraction",
                    self.curriculum_full_fraction,
                )
            )
            if not math.isclose(
                stored_warmup, self.curriculum_warmup_fraction
            ) or not math.isclose(
                stored_full, self.curriculum_full_fraction
            ):
                raise ValueError(
                    "episode-monitor curriculum boundaries differ from checkpoint"
                )
        stored_scope = str(state.get("scope", self.scope))
        if stored_scope != self.scope:
            raise ValueError(
                "episode-monitor metric scope differs from checkpoint"
            )
        self.completed = int(state.get("completed", 0))
        self.successes = int(state.get("successes", 0))
        self.failures = int(state.get("failures", 0))
        self.truncations = int(state.get("truncations", 0))
        self.return_sum = float(state.get("return_sum", 0.0))
        self.depth_sum_m = float(state.get("depth_sum_m", 0.0))
        self.turns_sum = float(state.get("turns_sum", 0.0))
        self.lateral_sum_m = float(state.get("lateral_sum_m", 0.0))
        self.axis_sum_rad = float(state.get("axis_sum_rad", 0.0))
        self.speed_sum = float(state.get("speed_sum", 0.0))
        self.length_sum = float(state.get("length_sum", 0.0))
        self.ik_failure_sum = int(state.get("ik_failure_sum", 0))
        self.ik_anchor_fallback_sum = int(
            state.get("ik_anchor_fallback_sum", 0)
        )
        self.ik_safety_stop_sum = int(state.get("ik_safety_stop_sum", 0))
        self.failure_reasons = Counter(
            {
                str(key): int(value)
                for key, value in dict(
                    state.get("failure_reasons", {})
                ).items()
            }
        )
        recent = state.get("recent", [])
        if not isinstance(recent, list):
            raise ValueError("episode-monitor recent payload must be a list")
        self.recent = deque(
            (dict(item) for item in recent[-self.recent_episodes :]),
            maxlen=self.recent_episodes,
        )
        if version >= 2:
            self.curriculum_groups = {
                str(name): {
                    key: float(value)
                    for key, value in dict(raw).items()
                }
                for name, raw in dict(
                    state.get("curriculum_groups", {})
                ).items()
            }
            for name in ("warmup", "ramp", "full"):
                self.curriculum_groups.setdefault(
                    name, self._empty_group()
                )
            self.terminal_phase_groups = {
                str(name): {
                    key: float(value)
                    for key, value in dict(raw).items()
                }
                for name, raw in dict(
                    state.get("terminal_phase_groups", {})
                ).items()
            }
        else:
            self.curriculum_groups = {
                name: self._empty_group()
                for name in ("warmup", "ramp", "full")
            }
            self.terminal_phase_groups = {}
        if not (
            0 <= self.successes <= self.completed
            and 0 <= self.failures <= self.completed
            and 0 <= self.truncations <= self.completed
            and (
                self.successes + self.failures + self.truncations
                <= self.completed
            )
        ):
            raise ValueError("episode-monitor checkpoint counts are invalid")


class RunMonitor:
    """Single-writer application heartbeat and graceful-stop protocol."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        pipeline_mode: str,
        config_name: str,
        config: Optional[Mapping[str, Any] | object] = None,
        interval_seconds: float = 10.0,
    ) -> None:
        if float(interval_seconds) <= 0.0:
            raise ValueError("monitor interval_seconds must be positive")
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.status_dir = self.run_dir / "status"
        self.live_path = self.status_dir / "live.json"
        self.history_path = self.status_dir / "history.jsonl"
        self.episodes_path = self.run_dir / "logs" / "episodes.jsonl"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.stop_path = self.status_dir / "stop-request.json"
        self.stop_ack_path = self.status_dir / "stop-ack.json"
        self.interval_seconds = float(interval_seconds)
        self.pipeline_mode = str(pipeline_mode)
        self.config_name = str(config_name)
        self.session_id = uuid.uuid4().hex
        prior = read_json(self.live_path) or {}
        prior_run = prior.get("run", {})
        prior_manifest = read_json(self.manifest_path) or {}
        manifest_run = prior_manifest.get("run", {})
        wrapper = read_json(self.status_dir / "job.json") or {}
        wrapper_run = wrapper.get("run", {})
        proposed_run_id = str(
            prior_run.get("id")
            or (
                manifest_run.get("id")
                if isinstance(manifest_run, Mapping)
                else None
            )
            or (
                wrapper_run.get("id")
                if isinstance(wrapper_run, Mapping)
                else None
            )
            or uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"screw-hpc-run:{self.run_dir}",
            ).hex
        )
        proposed_created_utc = str(
            prior_run.get("created_utc")
            or (
                manifest_run.get("created_utc")
                if isinstance(manifest_run, Mapping)
                else None
            )
            or wrapper.get("started_utc")
            or utc_now()
        )
        candidate_manifest = build_run_manifest(
            run_id=proposed_run_id,
            run_created_utc=proposed_created_utc,
            config_name=self.config_name,
            config=config,
        )
        self.manifest, self.manifest_sha256 = ensure_run_manifest(
            self.run_dir, candidate_manifest
        )
        prior_manifest_sha = (
            prior.get("manifest", {}).get("sha256")
            if isinstance(prior.get("manifest"), Mapping)
            else None
        )
        if (
            isinstance(prior_manifest_sha, str)
            and prior_manifest_sha
            and prior_manifest_sha != self.manifest_sha256
        ):
            raise ValueError(
                "run manifest changed since the previous heartbeat; "
                "resume is refused"
            )
        manifest_run = self.manifest.get("run", {})
        self.run_id = str(manifest_run.get("id") or proposed_run_id)
        self.run_created_utc = str(
            manifest_run.get("created_utc") or proposed_created_utc
        )
        self.session_created_utc = utc_now()
        self.sequence = int(prior.get("sequence", 0))
        self._created_monotonic = time.monotonic()
        self._stage_started_monotonic = self._created_monotonic
        self._stage_start_current = 0.0
        self._last_write_monotonic = -math.inf
        self._last_stop_poll_monotonic = -math.inf
        self._stop_requested = False
        self._stop_acknowledged = False
        self._last_write_error: Optional[str] = None
        self._last_error_report_monotonic = -math.inf
        self.stage_name = "initializing"
        self.stage_current = 0.0
        self.stage_total = 1.0
        self.stage_unit = "step"
        self.stage_message = "Starting pipeline"
        self.last_metrics: dict[str, Any] = {}
        self.last_health: dict[str, Any] = {}
        self.last_artifacts: dict[str, Any] = {}
        self.pipeline_order = {
            "full": ("demonstrations", "diffusion", "residual_tqc", "evaluation"),
            "smoke": ("demonstrations", "diffusion", "residual_tqc", "evaluation"),
            "diffusion-full": (
                "demonstrations",
                "diffusion",
                "evaluation",
            ),
            "diffusion-smoke": (
                "demonstrations",
                "diffusion",
                "evaluation",
            ),
            "diffusion-eval": ("evaluation",),
            "eval": ("evaluation",),
            "resume-probe": ("resume_probe",),
            "diffusion": ("diffusion",),
            "residual": ("residual_tqc",),
            "evaluation": ("evaluation",),
        }.get(self.pipeline_mode, (self.stage_name,))
        self.heartbeat(state="starting", force=True)

    def event_context(self, **cursor: Any) -> dict[str, Any]:
        """Return stable lineage fields for append-only diagnostic events."""

        result = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "manifest_sha256": self.manifest_sha256,
            "pipeline_mode": self.pipeline_mode,
            "stage": self.stage_name,
        }
        result.update(cursor)
        return result

    @property
    def due(self) -> bool:
        return (
            time.monotonic() - self._last_write_monotonic
            >= self.interval_seconds
        )

    def set_stage(
        self,
        name: str,
        *,
        current: float,
        total: float,
        unit: str,
        message: str = "",
        metrics: Optional[Mapping[str, Any]] = None,
        artifacts: Optional[Mapping[str, Any]] = None,
        stale_after_seconds: Optional[float] = None,
    ) -> None:
        self.stage_name = str(name)
        self.stage_current = float(current)
        self.stage_total = max(float(total), 0.0)
        self.stage_unit = str(unit)
        self.stage_message = str(message)
        self._stage_started_monotonic = time.monotonic()
        self._stage_start_current = self.stage_current
        self.last_metrics = dict(metrics or {})
        if artifacts is not None:
            self.last_artifacts = dict(artifacts)
        self.heartbeat(
            state="running",
            force=True,
            stale_after_seconds=stale_after_seconds,
        )

    def _job_info(self) -> dict[str, Any]:
        environment_names = (
            "PBS_JOBID",
            "PBS_JOBNAME",
            "PBS_QUEUE",
            "PBS_O_WORKDIR",
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_PARTITION",
            "SLURM_SUBMIT_DIR",
            "SCREW_ALLOCATION_ID",
            "CUDA_VISIBLE_DEVICES",
        )
        environment = {
            name: os.environ[name]
            for name in environment_names
            if os.environ.get(name)
        }
        scheduler = "pbs" if "PBS_JOBID" in environment else (
            "slurm" if "SLURM_JOB_ID" in environment else "local"
        )
        scheduler_job_id = environment.get(
            "PBS_JOBID", environment.get("SLURM_JOB_ID")
        )
        return {
            "scheduler": scheduler,
            "scheduler_job_id": scheduler_job_id,
            "allocation_id": environment.get("SCREW_ALLOCATION_ID"),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "environment": environment,
        }

    def _process_info(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "pid": os.getpid(),
            "cpu_time_seconds": sum(os.times()[:2]),
        }
        status_path = Path(f"/proc/{os.getpid()}/status")
        try:
            fields = {}
            for line in status_path.read_text(encoding="utf-8").splitlines():
                key, separator, raw = line.partition(":")
                if separator:
                    fields[key] = raw.strip()
            result.update(
                {
                    "rss": fields.get("VmRSS"),
                    "peak_rss": fields.get("VmHWM"),
                    "threads": (
                        int(fields["Threads"])
                        if fields.get("Threads", "").isdigit()
                        else None
                    ),
                }
            )
            result["open_fds"] = len(list(Path(f"/proc/{os.getpid()}/fd").iterdir()))
        except OSError:
            pass

        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_available():
                device = torch.cuda.current_device()
                result["torch_cuda"] = {
                    "device_index": int(device),
                    "device_name": torch.cuda.get_device_name(device),
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                    "peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(device)
                    ),
                }
        except (AttributeError, RuntimeError, ValueError):
            pass
        return result

    def _progress(self) -> dict[str, Any]:
        elapsed = max(
            0.0, time.monotonic() - self._stage_started_monotonic
        )
        advanced = max(0.0, self.stage_current - self._stage_start_current)
        rate = advanced / elapsed if elapsed > 0.0 and advanced > 0.0 else None
        remaining = max(0.0, self.stage_total - self.stage_current)
        eta = remaining / rate if rate and rate > 0.0 else None
        fraction = (
            min(max(self.stage_current / self.stage_total, 0.0), 1.0)
            if self.stage_total > 0.0
            else None
        )
        return {
            "current": self.stage_current,
            "total": self.stage_total,
            "unit": self.stage_unit,
            "fraction": fraction,
            "percent": 100.0 * fraction if fraction is not None else None,
            "rate_per_second": rate,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
        }

    def _pipeline_info(self) -> dict[str, Any]:
        try:
            index = self.pipeline_order.index(self.stage_name) + 1
        except ValueError:
            index = None
        return {
            "mode": self.pipeline_mode,
            "stage_index": index,
            "stage_count": len(self.pipeline_order),
            "stage_order": list(self.pipeline_order),
        }

    @staticmethod
    def _health_alerts(health: Mapping[str, Any]) -> list[dict[str, str]]:
        alerts: list[dict[str, str]] = []

        def add(code: str, severity: str, message: str) -> None:
            alerts.append(
                {"code": code, "severity": severity, "message": message}
            )

        if health and not bool(health.get("finite", True)):
            add("nonfinite", "critical", "Physics/task state is non-finite.")
        for key in (
            "sticky_nonfinite",
            "sticky_external_nonfinite",
            "overflow_worlds",
            "sticky_overflow",
            "sticky_contacts_full",
            "sticky_constraints_full",
        ):
            if int(health.get(key, 0) or 0) != 0:
                add(key, "critical", f"Backend health counter {key} is nonzero.")
        contacts = float(health.get("contacts_total", 0) or 0)
        contact_capacity = float(
            health.get("contact_capacity_total", 0) or 0
        )
        if contact_capacity > 0 and contacts / contact_capacity >= 0.80:
            add(
                "contact_capacity_high",
                "warning",
                f"Contact capacity is {100.0 * contacts / contact_capacity:.1f}% full.",
            )
        max_nefc = float(health.get("max_nefc", 0) or 0)
        njmax = float(
            health.get(
                "njmax", health.get("constraint_capacity", 0)
            )
            or 0
        )
        if njmax > 0 and max_nefc / njmax >= 0.80:
            add(
                "constraint_capacity_high",
                "warning",
                f"Constraint capacity is {100.0 * max_nefc / njmax:.1f}% full.",
            )
        return alerts

    def heartbeat(
        self,
        *,
        current: Optional[float] = None,
        total: Optional[float] = None,
        state: str = "running",
        message: Optional[str] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        health: Optional[Mapping[str, Any]] = None,
        artifacts: Optional[Mapping[str, Any]] = None,
        alerts: Optional[Iterable[Mapping[str, Any]]] = None,
        error: Optional[Mapping[str, Any]] = None,
        force: bool = False,
        stale_after_seconds: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        now_monotonic = time.monotonic()
        if not force and not self.due:
            return None
        if current is not None:
            self.stage_current = float(current)
        if total is not None:
            self.stage_total = max(float(total), 0.0)
        if message is not None:
            self.stage_message = str(message)
        if metrics is not None:
            self.last_metrics = dict(metrics)
        if health is not None:
            self.last_health = dict(health)
        if artifacts is not None:
            self.last_artifacts = dict(artifacts)
        self.sequence += 1
        combined_alerts = self._health_alerts(self.last_health)
        combined_alerts.extend(dict(item) for item in (alerts or ()))
        updated = utc_now()
        snapshot = {
            "schema": LIVE_SCHEMA,
            "sequence": self.sequence,
            "state": str(state),
            "updated_utc": updated,
            "monitor": {
                "interval_seconds": self.interval_seconds,
                "stale_after_seconds": max(
                    60.0,
                    3.0 * self.interval_seconds,
                    float(stale_after_seconds or 0.0),
                ),
            },
            "run": {
                "id": self.run_id,
                "created_utc": self.run_created_utc,
                "directory": str(self.run_dir),
                "config": self.config_name,
            },
            "manifest": {
                "path": str(self.manifest_path),
                "sha256": self.manifest_sha256,
                "config_sha256": self.manifest.get(
                    "config", {}
                ).get("sha256"),
                "training_contract_hash": self.manifest.get(
                    "contracts", {}
                ).get("training_contract_hash"),
                "source_tree_sha256": self.manifest.get(
                    "contracts", {}
                ).get("source_tree_sha256"),
                "schema_hash": self.manifest.get(
                    "contracts", {}
                ).get("schema_hash"),
                "task_geometry_hash": self.manifest.get(
                    "contracts", {}
                ).get("task_geometry_hash"),
                "seeds": self.manifest.get("seeds", {}),
            },
            "session": {
                "id": self.session_id,
                "created_utc": self.session_created_utc,
                "elapsed_seconds": max(
                    0.0, now_monotonic - self._created_monotonic
                ),
            },
            "pipeline": self._pipeline_info(),
            "stage": {
                "name": self.stage_name,
                "message": self.stage_message,
                "progress": self._progress(),
            },
            "metrics": self.last_metrics,
            "health": self.last_health,
            "process": self._process_info(),
            "job": self._job_info(),
            "artifacts": self.last_artifacts,
            "alerts": combined_alerts,
            "stop": {
                "requested": self._stop_requested,
                "acknowledged": self._stop_acknowledged,
                "request_path": str(self.stop_path),
            },
            "error": dict(error) if error is not None else None,
            "monitor_write_error": self._last_write_error,
        }
        try:
            atomic_write_json(self.live_path, snapshot)
            self._last_write_monotonic = now_monotonic
            self._last_write_error = None
        except Exception as exc:
            # Monitoring must never destroy an otherwise valid formal run.
            self._last_write_error = f"{type(exc).__name__}: {exc}"
            # A broken/full shared filesystem must not turn every environment
            # step into another status write plus GPU-to-CPU metric transfer.
            self._last_write_monotonic = now_monotonic
            if now_monotonic - self._last_error_report_monotonic >= 60.0:
                print(
                    f"WARNING: live monitor write failed: {self._last_write_error}",
                    file=sys.stderr,
                    flush=True,
                )
                self._last_error_report_monotonic = now_monotonic
            return None
        try:
            append_history(
                self.history_path,
                {
                    "sequence": self.sequence,
                    "state": state,
                    "updated_utc": updated,
                    "session_id": self.session_id,
                    "manifest_sha256": self.manifest_sha256,
                    "stage": snapshot["stage"],
                    "metrics": self.last_metrics,
                    "health": self.last_health,
                    "alerts": combined_alerts,
                    "error": snapshot["error"],
                },
            )
        except Exception as exc:
            # A history failure must not make the hot loop retry on every step;
            # the authoritative atomic live snapshot was already committed.
            self._last_write_error = (
                f"history {type(exc).__name__}: {exc}"
            )
            if now_monotonic - self._last_error_report_monotonic >= 60.0:
                print(
                    f"WARNING: monitor history write failed: "
                    f"{self._last_write_error}",
                    file=sys.stderr,
                    flush=True,
                )
                self._last_error_report_monotonic = now_monotonic
        return snapshot

    def finish(
        self,
        state: str,
        *,
        message: str,
        error: Optional[Mapping[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        if state not in TERMINAL_STATES:
            raise ValueError(f"invalid terminal monitor state {state!r}")
        return self.heartbeat(
            state=state,
            message=message,
            error=error,
            force=True,
        )

    def stop_requested(self, *, poll_seconds: float = 1.0) -> bool:
        """Acknowledge a stop request bound to this exact process session."""

        if self._stop_requested:
            return True
        now_monotonic = time.monotonic()
        if (
            now_monotonic - self._last_stop_poll_monotonic
            < max(float(poll_seconds), 0.1)
        ):
            return False
        self._last_stop_poll_monotonic = now_monotonic
        request = read_json(self.stop_path)
        if not request or request.get("schema") != STOP_SCHEMA:
            return False
        if request.get("target_session_id") != self.session_id:
            return False
        self._stop_requested = True
        acknowledgement = {
            "schema": STOP_SCHEMA,
            "request_id": request.get("request_id"),
            "target_session_id": self.session_id,
            "acknowledged_utc": utc_now(),
            "pid": os.getpid(),
            "status": "accepted_at_safe_cursor",
        }
        try:
            atomic_write_json(self.stop_ack_path, acknowledgement)
            self._stop_acknowledged = True
            self.heartbeat(
                message="Graceful stop accepted; saving at a consistent cursor.",
                force=True,
            )
        except OSError:
            # The request was read successfully.  Stop even if its ack cannot
            # be written; the final live status may still explain the exit.
            pass
        return True


def request_graceful_stop(
    run_dir: str | Path,
    *,
    reason: str = "",
) -> dict[str, Any]:
    """Write a stop request targeting only the currently visible session."""

    root = Path(run_dir).expanduser().resolve()
    status_dir = root / "status"
    live = read_json(status_dir / "live.json")
    if not live:
        raise FileNotFoundError(
            f"no live training heartbeat found at {status_dir / 'live.json'}"
        )
    wrapper = read_json(status_dir / "job.json") or {}
    wrapper_state = wrapper.get("state")
    wrapper_allocation = wrapper.get("allocation", {})
    wrapper_job_id = wrapper_allocation.get("scheduler_job_id")
    live_job = live.get("job", {})
    live_job_id = live_job.get("scheduler_job_id")
    if (
        wrapper_allocation.get("identity_source")
        == "SCREW_ALLOCATION_ID"
    ):
        wrapper_identity = wrapper_allocation.get("id")
        live_identity = live_job.get("allocation_id")
    else:
        wrapper_identity = wrapper_job_id
        live_identity = live_job_id
    if (
        wrapper_state in {"starting", "preflight", "running"}
        and wrapper_identity
        and wrapper_identity != live_identity
    ):
        raise RuntimeError(
            "the current allocation has not started pipeline.py; the visible "
            "heartbeat belongs to a previous allocation"
        )
    if live.get("state") in TERMINAL_STATES:
        raise RuntimeError(
            f"run is already {live.get('state')}; no stop request was written"
        )
    session = live.get("session", {})
    session_id = session.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("live heartbeat has no valid session id")
    request = {
        "schema": STOP_SCHEMA,
        "request_id": uuid.uuid4().hex,
        "target_session_id": session_id,
        "requested_utc": utc_now(),
        "requested_by": os.environ.get("USER") or os.environ.get("USERNAME"),
        "requested_from_host": socket.gethostname(),
        "reason": str(reason),
    }
    atomic_write_json(status_dir / "stop-request.json", request)
    return request


__all__ = [
    "DEFAULT_RECENT_EPISODES",
    "DEFAULT_RECENT_WINDOWS",
    "EPISODE_LOG_SCHEMA",
    "EpisodeTracker",
    "LIVE_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "RunMonitor",
    "STOP_SCHEMA",
    "SYSTEM_SCHEMA",
    "TERMINAL_STATES",
    "append_history",
    "atomic_write_json",
    "build_run_manifest",
    "canonical_sha256",
    "describe_paths",
    "ensure_run_manifest",
    "heartbeat_age_seconds",
    "jsonable",
    "parse_utc",
    "read_json",
    "request_graceful_stop",
    "tail_jsonl",
    "utc_now",
]
