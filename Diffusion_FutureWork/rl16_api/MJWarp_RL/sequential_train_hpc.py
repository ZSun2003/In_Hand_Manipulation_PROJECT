#!/usr/bin/env python3
"""Fail-closed G6-only formal HPC training sequencer.

Deploy this standalone file outside the immutable G6 release, then run one
command; the default invocation launches one detached ``screen`` session::

    python -m MJWarp_RL.sequential_train_hpc

No training output directory is created until the corresponding
``MJWarp_RL.train`` process has passed its own admission checks.  Runs execute
strictly as G6 translation followed by G6 rotation.  A
failure or persisted-artifact mismatch prevents every later run from starting.
This module deliberately is not part of the formal runtime fingerprint: it
only sequences two independently certified formal training programs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
from typing import Any, Iterator, Mapping, Sequence


SCHEMA = "grab-g6-task-split-sequential-training-v9"
CAPACITY_REPORT_SCHEMA = "grab-g6-mjwarp-capacity-scan-v14"
ADMISSION_MARKER = "GRAB_G6_ADMISSION_JSON="
DEFAULT_RELEASE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSLATION_CAPACITY_REPORT = Path(__file__).resolve().parents[2] / "capacity_reports" / "translation" / "g6_20000.json"
DEFAULT_ROTATION_CAPACITY_REPORT = Path(__file__).resolve().parents[2] / "capacity_reports" / "rotation" / "g6_20000.json"
DEFAULT_TRANSLATION_CAPACITY_REPORT_SHA256 = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)
DEFAULT_ROTATION_CAPACITY_REPORT_SHA256 = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)
DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256 = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)
DEFAULT_OUTPUT_BASE = Path(__file__).resolve().parents[3] / "RL_env" / "runs" / "futurework"
DEFAULT_PYTHON = Path(__file__).resolve().parents[3] / "RL_env" / "profiles" / "core" / "RL_env" / "bin" / "python"
DEFAULT_LOCK_FILE = Path(__file__).resolve().parents[3] / "RL_env" / ".locks" / "grab_formal_gpu0.lock"
FIXED_SEED = 7
FIXED_DEVICE = "cuda:0"
EXPECTED_WORLDS = 20_000
EXPECTED_OBJECT_WORLD_COUNTS = [5_000] * 4
EXPECTED_TOTAL_STEPS = 90_000_000
TRAINING_MANIFEST_SCHEMA = (
    "native-mjwarp-sharded-certified-ik-diffusion-training-v6"
)
EXPECTED_ALGORITHM = "ActionChunkDiffusionIKImitation"
EXPECTED_ALGORITHM_SIGNATURE = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)
EXPECTED_PHYSICS_LAYOUT = "task_specific_four_single_object_xml_shards"
EXPECTED_EXPERT = "deterministic-certified-closed-loop-ik-v1"
EXPECTED_TRAJECTORY_CERTIFICATION = (
    "endpoint_success_and_route_tracking_per_axis_10mm_and_plane_"
    "compliance_and_not_dropped_and_frozen_path_plane_four_anchors"
)
EXPECTED_ACTION_SUPERVISION = (
    "requested normalized closed-loop IK-expert command before "
    "environment low-pass filtering and episode-channel freezing"
)
EXPECTED_OBJECTS = ("orange", "banana", "pear", "strawberry")
EXPECTED_OBSERVATION_DIM = 607
EXPECTED_ACTION_DIM = 25
EXPECTED_REMAINDER_WORLDS = 0
EXPECTED_REMAINDER_OBJECT_WORLD_COUNTS = [0] * 4
INITIAL_DISK_GIB = 70.0
BETWEEN_RUNS_DISK_GIB = 10.0
MAX_IDLE_MEMORY_MIB = 2048
MAX_IDLE_UTILIZATION_PERCENT = 5
HARD_COUNTERS = (
    "ik_solve_fallback_count",
    "ctrl_certificate_fallback_count",
)
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
TASK_TRANSLATE = "translate"
TASK_ROTATE = "rotate"
TASKS = (TASK_TRANSLATE, TASK_ROTATE)
EXPECTED_TASK_INDEX = {
    TASK_TRANSLATE: 0,
    TASK_ROTATE: 1,
}
EXPECTED_ROOT_HEIGHT_M = {
    ("gripper6", TASK_TRANSLATE): 0.56,
    ("gripper6", TASK_ROTATE): 0.56,
}
RUN_SPECS = (
    ("g6_translation", "gripper6", TASK_TRANSLATE, "g6"),
    ("g6_rotation", "gripper6", TASK_ROTATE, "g6"),
)


class SequenceFailure(RuntimeError):
    """A fail-closed sequencing or persisted-artifact failure."""


class SequenceInterrupted(SequenceFailure):
    """The sequencer received SIGINT or SIGTERM."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        super().__init__(f"received signal {self.signum}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_id() -> str:
    return "g6t_g6r_90m_seed7_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


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


def _legacy_phase_key_paths(value: Any, *, path: str = "manifest") -> list[str]:
    """Locate every stale phase1/phase2 key in a persisted artifact."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            lowered = name.lower()
            child_path = f"{path}.{name}"
            if "phase1" in lowered or "phase2" in lowered:
                found.append(child_path)
            found.extend(_legacy_phase_key_paths(item, path=child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(
                _legacy_phase_key_paths(item, path=f"{path}[{index}]")
            )
    return found


def _strict_json_bytes(payload: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        return json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SequenceFailure(f"invalid {label}: {exc}") from exc


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SequenceFailure(f"{label} is missing or not a regular file: {path}")
    value = _strict_json_bytes(path.read_bytes(), label=label)
    if not isinstance(value, dict):
        raise SequenceFailure(f"{label} is not a JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _existing_disk_anchor(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise SequenceFailure(f"no existing filesystem ancestor for {path}")
        candidate = candidate.parent
    return candidate


def _disk_gate(path: Path, *, required_gib: float) -> dict[str, Any]:
    anchor = _existing_disk_anchor(path)
    usage = shutil.disk_usage(anchor)
    free_gib = usage.free / (1024**3)
    result = {
        "path": str(path),
        "filesystem_anchor": str(anchor),
        "required_free_gib": float(required_gib),
        "actual_free_gib": free_gib,
    }
    if free_gib < required_gib:
        raise SequenceFailure(
            f"disk gate failed for {path}: {free_gib:.2f} GiB free, "
            f"{required_gib:.2f} GiB required"
        )
    return result


def _gpu_idle_snapshot() -> dict[str, Any]:
    applications = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if applications.returncode != 0:
        raise SequenceFailure(
            "cannot query GPU 0 compute processes: "
            + applications.stderr.strip()
        )
    process_rows = [
        line.strip()
        for line in applications.stdout.splitlines()
        if line.strip() and "No running processes" not in line
    ]
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if gpu.returncode != 0:
        raise SequenceFailure("cannot query GPU 0 state: " + gpu.stderr.strip())
    rows = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise SequenceFailure(f"expected exactly one GPU 0 row, found {rows!r}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise SequenceFailure(f"malformed GPU 0 state: {rows[0]!r}")
    try:
        index, used_mib, total_mib, utilization = (int(value) for value in fields)
    except ValueError as exc:
        raise SequenceFailure(f"non-integer GPU 0 state: {rows[0]!r}") from exc
    result = {
        "index": index,
        "memory_used_mib": used_mib,
        "memory_total_mib": total_mib,
        "utilization_percent": utilization,
        "compute_processes": process_rows,
        "idle_memory_limit_mib": MAX_IDLE_MEMORY_MIB,
        "idle_utilization_limit_percent": MAX_IDLE_UTILIZATION_PERCENT,
    }
    if (
        index != 0
        or process_rows
        or used_mib > MAX_IDLE_MEMORY_MIB
        or utilization > MAX_IDLE_UTILIZATION_PERCENT
    ):
        raise SequenceFailure("GPU 0 is not idle: " + json.dumps(result))
    return result


def _verify_g6_capacity_report(
    *,
    capacity_report: Path,
    capacity_sha256: str,
    expected_task: str,
) -> dict[str, Any]:
    """Verify one immutable, task-specific G6 capacity certificate."""

    if expected_task not in TASKS:
        raise SequenceFailure(f"unknown capacity task: {expected_task!r}")
    if not _is_sha256(capacity_sha256):
        raise SequenceFailure(
            "capacity report pin must be a lowercase SHA256 value"
        )
    if not capacity_report.is_file() or capacity_report.is_symlink():
        raise SequenceFailure(
            "G6 capacity report is missing, a symlink, or not a regular "
            f"file: {capacity_report}"
        )
    actual_sha = _sha256(capacity_report)
    if actual_sha != capacity_sha256:
        raise SequenceFailure(
            "G6 capacity report SHA256 mismatch: "
            f"expected {capacity_sha256}, found {actual_sha}"
        )
    report = _read_json_object(
        capacity_report, label="G6 capacity report"
    )
    expected_scalars = {
        "schema": CAPACITY_REPORT_SCHEMA,
        "platform": "gripper6",
        "platforms": ["gripper6"],
        "task": expected_task,
        "task_index": EXPECTED_TASK_INDEX[expected_task],
        "root_height_m": EXPECTED_ROOT_HEIGHT_M[
            ("gripper6", expected_task)
        ],
        "platform_root_heights_m": {
            "gripper6": EXPECTED_ROOT_HEIGHT_M[
                ("gripper6", expected_task)
            ]
        },
        "selected_num_envs": EXPECTED_WORLDS,
        "passing_candidates": [EXPECTED_WORLDS],
        "candidates": [EXPECTED_WORLDS],
        "object_world_counts": EXPECTED_OBJECT_WORLD_COUNTS,
        "remainder_worlds": EXPECTED_REMAINDER_WORLDS,
        "remainder_object_world_counts": (
            EXPECTED_REMAINDER_OBJECT_WORLD_COUNTS
        ),
        "runtime_code_fingerprint_sha256": (
            DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256
        ),
        "algorithm_signature": EXPECTED_ALGORITHM_SIGNATURE,
        "physics_layout": EXPECTED_PHYSICS_LAYOUT,
        "observation_dim": EXPECTED_OBSERVATION_DIM,
        "action_dim": EXPECTED_ACTION_DIM,
        "episode_control_ticks": 4_500,
        "device": FIXED_DEVICE,
        "runtime_errors": [],
    }
    mismatches = {
        name: {"expected": wanted, "found": report.get(name)}
        for name, wanted in expected_scalars.items()
        if report.get(name) != wanted
    }
    if mismatches:
        raise SequenceFailure(
            "G6 capacity report binding mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    fingerprints = report.get(
        "platform_rl_model_fingerprints_sha256"
    )
    if (
        not isinstance(fingerprints, dict)
        or set(fingerprints) != {"gripper6"}
        or not _is_sha256(fingerprints.get("gripper6"))
    ):
        raise SequenceFailure(
            "G6 capacity report lacks its sole platform model fingerprint"
        )
    results = report.get("results")
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
        or results[0].get("num_envs") != EXPECTED_WORLDS
        or results[0].get("passed") is not True
        or results[0].get("worker_reported_passed") is not True
        or results[0].get("returncode") != 0
        or results[0].get("validation_errors") != []
        or results[0].get("runtime_errors") != []
    ):
        raise SequenceFailure(
            "G6 capacity report does not contain one clean 20k-world pass"
        )
    for name in (
        "config_canonical_sha256",
        "candidate_config_set_sha256",
        "rl_model_fingerprint_sha256",
        "parent_model_fingerprint_sha256",
        "parent_controller_fingerprint_sha256",
        "platform_adapter_fingerprint_sha256",
    ):
        if not _is_sha256(report.get(name)):
            raise SequenceFailure(
                f"G6 capacity report {name} is not a SHA256"
            )
    return {
        "report_path": str(capacity_report),
        "report_sha256": actual_sha,
        "task": expected_task,
        "platform": "gripper6",
        "platforms": ["gripper6"],
        "root_height_m": EXPECTED_ROOT_HEIGHT_M[
            ("gripper6", expected_task)
        ],
        "platform_root_heights_m": report[
            "platform_root_heights_m"
        ],
        "platform_rl_model_fingerprints_sha256": fingerprints,
        "selected_num_envs": EXPECTED_WORLDS,
        "runtime_code_fingerprint_sha256": (
            DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256
        ),
    }


_G6_ADMISSION_PROGRAM = r"""
from dataclasses import replace
import json
from pathlib import Path
from MJWarp_RL.hpc_config import DEFAULT_CONFIG
from MJWarp_RL.platform_adapter import build_platform_spec
from MJWarp_RL.preflight import inspect_runtime
from MJWarp_RL.runtime_fingerprint import runtime_code_fingerprint_sha256
from MJWarp_RL.train import _validate_capacity_selection

report = Path(__import__('sys').argv[1]).resolve()
task = __import__('sys').argv[2]
runtime = inspect_runtime(
    strict=True,
    device='cuda:0',
    deterministic_torch=DEFAULT_CONFIG.deterministic_torch,
    task=task,
)
if not runtime.get('ok'):
    raise RuntimeError('strict runtime inspection failed: ' + repr(runtime.get('errors')))
payload = json.loads(report.read_text(encoding='utf-8'))
worlds = int(payload['selected_num_envs'])
config = replace(
    DEFAULT_CONFIG,
    seed=7,
    num_envs=worlds,
    device='cuda:0',
    task=task,
)
config.validate()
platform = build_platform_spec(task)
selection = _validate_capacity_selection(
    report_path=report,
    config=config,
    runtime=runtime,
    platform_spec=platform,
)
result = {
    'platform': platform.platform,
    'task': task,
    'task_index': selection['task_index'],
    'root_height_m': selection['root_height_m'],
    'platform_root_heights_m': selection['platform_root_heights_m'],
    'platform_rl_model_fingerprints_sha256': selection[
        'platform_rl_model_fingerprints_sha256'
    ],
    'selected_num_envs': worlds,
    'runtime_code_fingerprint_sha256': runtime_code_fingerprint_sha256(),
    'report_sha256': selection['report_sha256'],
    'object_world_counts': selection['object_world_counts'],
    'config_canonical_sha256': selection['selected_config_sha256'],
}
print('GRAB_G6_ADMISSION_JSON=' + json.dumps(result, sort_keys=True), flush=True)
"""


def _project_env(project_root: Path, *, python: Path | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(project_root),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MUJOCO_GL": "egl",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "IRP_HPC_PROFILE": "core",
        }
    )
    if python is not None:
        environment["IRP_HPC_CORE_PYTHON"] = str(python)
        environment["IRP_HPC_CORE_ENV"] = str(python.parent.parent)
    return environment


class ChildOwner:
    """Own at most one process group and terminate only that group."""

    def __init__(self) -> None:
        self.active: subprocess.Popen[str] | None = None
        self.received_signal: int | None = None

    def install_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.received_signal = int(signum)
        self.terminate_active()
        raise SequenceInterrupted(signum)

    def terminate_active(self) -> None:
        child = self.active
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)

    def run_capture(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        child = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.active = child
        try:
            stdout, _ = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self.terminate_active()
            raise SequenceFailure(f"child timed out: {command[0]}") from exc
        except BaseException:
            # A decode error, broken pipe, signal, or other relay failure must
            # never orphan a CUDA-owning child after the global flock exits.
            self.terminate_active()
            raise
        finally:
            self.active = None
        return subprocess.CompletedProcess(command, child.returncode, stdout, "")

    def run_log(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
    ) -> int:
        with log_path.open("x", encoding="utf-8") as log:
            child = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            self.active = child
            try:
                if child.stdout is None:
                    raise SequenceFailure("training child has no stdout pipe")
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                return child.wait()
            except BaseException:
                # This includes a full disk, a detached/broken screen stdout,
                # decode/pipe errors, and signals.  Kill and reap our process
                # group before clearing ownership or releasing the GPU lock.
                self.terminate_active()
                raise
            finally:
                if child.stdout is not None:
                    child.stdout.close()
                self.active = None


def _g6_admission(
    *,
    owner: ChildOwner,
    python: Path,
    project_root: Path,
    capacity_report: Path,
    expected_platform: str,
    expected_task: str,
    expected_root_height_m: float,
    report_pins: Mapping[str, Any],
) -> dict[str, Any]:
    result = owner.run_capture(
        [
            str(project_root / "MJWarp_RL" / "run_hpc.sh"),
            "-c",
            _G6_ADMISSION_PROGRAM,
            str(capacity_report),
            expected_task,
        ],
        cwd=project_root,
        env=_project_env(project_root, python=python),
        timeout=1800,
    )
    lines = [
        line[len(ADMISSION_MARKER) :]
        for line in result.stdout.splitlines()
        if line.startswith(ADMISSION_MARKER)
    ]
    if result.returncode != 0 or len(lines) != 1:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        raise SequenceFailure(
            f"{expected_platform} admission failed with rc={result.returncode}:\n"
            + tail
        )
    value = _strict_json_bytes(lines[0].encode(), label="G6 admission result")
    if not isinstance(value, dict):
        raise SequenceFailure("G6 admission result is not an object")
    expected = {
        "platform": expected_platform,
        "task": expected_task,
        "task_index": EXPECTED_TASK_INDEX[expected_task],
        "root_height_m": expected_root_height_m,
        "platform_root_heights_m": report_pins[
            "platform_root_heights_m"
        ],
        "platform_rl_model_fingerprints_sha256": report_pins[
            "platform_rl_model_fingerprints_sha256"
        ],
        "selected_num_envs": EXPECTED_WORLDS,
        "runtime_code_fingerprint_sha256": (
            DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256
        ),
        "report_sha256": report_pins["report_sha256"],
        "object_world_counts": EXPECTED_OBJECT_WORLD_COUNTS,
    }
    mismatches = {
        name: {"expected": wanted, "found": value.get(name)}
        for name, wanted in expected.items()
        if value.get(name) != wanted
    }
    if mismatches:
        raise SequenceFailure(
            f"{expected_platform} admission binding mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if not _is_sha256(value.get("config_canonical_sha256")):
        raise SequenceFailure(
            f"{expected_platform} admission config hash is invalid"
        )
    return value


def _validate_memory(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SequenceFailure(f"{label} memory report is not an object")
    expected = {
        "valid": True,
        "closed": True,
        "limit_violated": False,
        "total_memory_changed": False,
    }
    for name, wanted in expected.items():
        if value.get(name) is not wanted:
            raise SequenceFailure(
                f"{label} memory report {name}={value.get(name)!r}"
            )
    if value.get("errors") != []:
        raise SequenceFailure(f"{label} memory report contains errors")
    fraction = value.get("peak_device_memory_fraction")
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not 0.0 <= float(fraction) <= 0.90
    ):
        raise SequenceFailure(f"{label} invalid peak memory fraction: {fraction!r}")
    return value


def _validate_hard_counters(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise SequenceFailure(f"{label} controller counters are not an object")
    for name in HARD_COUNTERS:
        count = value.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count != 0:
            raise SequenceFailure(f"{label} hard counter {name}={count!r}")
    return {str(name): int(count) for name, count in value.items()}


def _validate_training_jsonl(path: Path) -> tuple[int, dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise SequenceFailure(f"training JSONL is missing: {path}")
    last: dict[str, Any] | None = None
    count = 0
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise SequenceFailure(f"blank training JSONL line {line_number}")
            value = _strict_json_bytes(
                line, label=f"training JSONL line {line_number}"
            )
            if not isinstance(value, dict):
                raise SequenceFailure(
                    f"training JSONL line {line_number} is not an object"
                )
            count += 1
            if value.get("sequence") != count:
                raise SequenceFailure(
                    f"training JSONL sequence is not contiguous at line {line_number}"
                )
            if value.get("state") == "failed" or value.get("stage") == "failed":
                raise SequenceFailure("training JSONL contains a failure record")
            if value.get("phase") in {"phase1", "phase2"} or value.get(
                "stage"
            ) in {"phase1", "phase2"}:
                raise SequenceFailure(
                    "training JSONL contains a legacy two-phase record"
                )
            last = value
    if last is None:
        raise SequenceFailure("training JSONL is empty")
    if not (
        last.get("state") == "complete"
        and last.get("stage") == "complete"
        and last.get("phase") == "complete"
        and last.get("target_steps") == EXPECTED_TOTAL_STEPS
        and last.get("total_steps") == EXPECTED_TOTAL_STEPS
    ):
        raise SequenceFailure("training JSONL does not end in formal completion")
    return count, last


def _validate_single_phase_demo_coverage(
    manifest: Mapping[str, Any], *, expected_task: str
) -> None:
    """Bind saved demonstrations to the one full-task ``all`` run."""

    count = manifest.get("certified_demo_count")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or not 24 <= count <= 2_000
    ):
        raise SequenceFailure(
            f"training manifest certified demo count is invalid: {count!r}"
        )
    required_distribution = {
        f"all/{object_kind}/{expected_task}" for object_kind in EXPECTED_OBJECTS
    }
    distribution = manifest.get("certified_demo_distribution")
    if not isinstance(distribution, dict) or set(distribution) != (
        required_distribution
    ):
        raise SequenceFailure(
            "certified demo distribution is not the four full-task strata"
        )
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        for value in distribution.values()
    ) or sum(distribution.values()) != count:
        raise SequenceFailure("certified demo distribution counts are invalid")

    coverage = manifest.get("certified_demo_coverage")
    if not isinstance(coverage, dict):
        raise SequenceFailure("certified demo coverage is missing")
    if coverage.get("run_counts") != {"all": count}:
        raise SequenceFailure("certified demo coverage is not phase='all'")
    if coverage.get("object_task_counts") != distribution:
        raise SequenceFailure(
            "certified demo coverage and distribution do not agree"
        )
    expected_task_counts = {
        task: count if task == expected_task else 0 for task in TASKS
    }
    if coverage.get("task_counts") != expected_task_counts:
        raise SequenceFailure("certified demo task coverage is inconsistent")
    object_counts = coverage.get("object_counts")
    if not isinstance(object_counts, dict) or set(object_counts) != set(
        EXPECTED_OBJECTS
    ) or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        for value in object_counts.values()
    ) or sum(object_counts.values()) != count:
        raise SequenceFailure("certified demo object coverage is inconsistent")
    if coverage.get("source_region_counts") != {"all": count}:
        raise SequenceFailure("certified demos do not all use source_region=all")

    template_counts = coverage.get("template_counts")
    expected_templates = {str(index) for index in range(32)}
    if not isinstance(template_counts, dict) or set(template_counts) != set(
        EXPECTED_OBJECTS
    ):
        raise SequenceFailure("certified reset-template coverage is missing")
    for object_kind in EXPECTED_OBJECTS:
        values = template_counts.get(object_kind)
        if not isinstance(values, dict) or set(values) != expected_templates or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in values.values()
        ):
            raise SequenceFailure(
                f"certified reset-template coverage failed for {object_kind}"
            )

    if expected_task == TASK_TRANSLATE:
        paths = coverage.get("translation_path_by_object")
        if not isinstance(paths, dict) or set(paths) != set(EXPECTED_OBJECTS):
            raise SequenceFailure("translation path coverage is missing")
        for object_kind in EXPECTED_OBJECTS:
            values = paths.get(object_kind)
            successes = (
                values.get("shape_success_counts")
                if isinstance(values, dict)
                else None
            )
            if not isinstance(successes, dict) or len(successes) != 5 or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                for value in successes.values()
            ):
                raise SequenceFailure(
                    f"translation path-shape coverage failed for {object_kind}"
                )
    else:
        angles = coverage.get("rotation_angle_by_object")
        if not isinstance(angles, dict) or set(angles) != set(EXPECTED_OBJECTS):
            raise SequenceFailure("rotation-angle coverage is missing")
        for object_kind in EXPECTED_OBJECTS:
            values = angles.get(object_kind)
            if not isinstance(values, dict) or (
                values.get("maximum_target_deg") != 45
                or not isinstance(
                    values.get("maximum_target_success_count"), int
                )
                or isinstance(
                    values.get("maximum_target_success_count"), bool
                )
                or values["maximum_target_success_count"] < 1
            ):
                raise SequenceFailure(
                    f"rotation maximum-angle coverage failed for {object_kind}"
                )


def _expected_training_config(task: str) -> dict[str, Any]:
    if task not in TASKS:
        raise SequenceFailure(f"unknown training task: {task!r}")
    return {
        "seed": FIXED_SEED,
        "task": task,
        "total_steps": EXPECTED_TOTAL_STEPS,
        "num_envs": EXPECTED_WORLDS,
        "num_eval_envs": 96,
        "device": FIXED_DEVICE,
        "cuda_graph": True,
        "deterministic_torch": True,
        "log_every_control_ticks": 25,
        "status_every_seconds": 15.0,
        "elite": {
            "population": 10,
            "elite_per_group": 1,
            "minimum_successful_demos": 24,
            "minimum_demos_per_object_task": 1,
            "require_all_reset_templates": True,
            "require_all_translation_path_shapes": True,
            "episode_action_sigma": 0.18,
            "dynamic_action_sigma": 0.12,
            "dynamic_knot_ticks": 16,
            "centroid_momentum": 0.85,
            "success_bonus": 1_000.0,
            "storage_dtype": "float16",
        },
        "diffusion": {
            "obs_horizon": 2,
            "action_horizon": 16,
            "diffusion_steps": 100,
            "inference_steps": 16,
            "replan_interval": 4,
            "hidden_dim": 256,
            "depth": 6,
            "dropout": 0.0,
            "epochs": 60,
            "batch_size": 4_096,
            "updates_per_epoch": 1_100,
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-6,
            "ema_decay": 0.999,
            "max_grad_norm": 1.0,
            "checkpoint_every_epochs": 1,
            "cache_demos_on_cuda": True,
            "use_bfloat16": True,
            "stratified_object_task_sampling": True,
        },
        "physics_substeps_per_control_tick": 40,
        "plan_ticks": 8,
        "close_ticks": 90,
        "settle_ticks": 60,
        "lift_ticks": 70,
        "hold_ticks": 70,
        "translation_action_ticks": 4_202,
        "rotation_action_ticks": 4_202,
        "action_ticks": 4_202,
        "translation_path_lead_in_ticks": 300,
        "translation_segment_ticks": 90,
        "rotation_motion_ticks": 2_400,
        "contacts_per_world": 128,
        "constraints_per_world": 640,
        "episode_length": 4_500,
        "task_index": EXPECTED_TASK_INDEX[task],
        "control_ticks": 4_500,
        "remainder_worlds": EXPECTED_REMAINDER_WORLDS,
        "groups_per_vector_batch": 2_000,
        "object_world_counts": EXPECTED_OBJECT_WORLD_COUNTS,
        "object_group_counts": [500] * 4,
        "remainder_object_world_counts": (
            EXPECTED_REMAINDER_OBJECT_WORLD_COUNTS
        ),
        "maximum_demo_episodes": 2_000,
    }


def _validate_model_shards(
    value: Any,
    *,
    platform: str,
    task: str,
    root_height_m: float,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(EXPECTED_OBJECTS):
        raise SequenceFailure("training manifest must contain four model shards")
    if platform != "gripper6":
        raise SequenceFailure(f"unsupported formal platform: {platform!r}")
    prefix = "_g6_project_rl"
    validated: list[dict[str, Any]] = []
    for index, (object_kind, shard) in enumerate(zip(EXPECTED_OBJECTS, value)):
        if not isinstance(shard, dict):
            raise SequenceFailure(f"model shard {index} is not an object")
        expected = {
            "object_index": index,
            "object": object_kind,
            "task": task,
            "task_index": EXPECTED_TASK_INDEX[task],
            "root_height_m": root_height_m,
            "worlds": EXPECTED_OBJECT_WORLD_COUNTS[index],
        }
        for name, wanted in expected.items():
            if shard.get(name) != wanted:
                raise SequenceFailure(
                    f"model shard {index} {name}: expected {wanted!r}, "
                    f"found {shard.get(name)!r}"
                )
        expected_xml = f"{prefix}_{task}_{object_kind}.xml"
        if Path(str(shard.get("model_xml", ""))).name != expected_xml:
            raise SequenceFailure(
                f"model shard {index} XML is not {expected_xml}"
            )
        if not _is_sha256(shard.get("model_xml_sha256")):
            raise SequenceFailure(f"model shard {index} lacks an XML SHA256")
        for name in ("nq", "nv", "nu"):
            item = shard.get(name)
            if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
                raise SequenceFailure(f"model shard {index} has invalid {name}")
        if int(shard["nv"]) > 60:
            raise SequenceFailure(f"model shard {index} exceeds dense-Newton nv=60")
        validated.append(shard)
    return validated


def validate_training_artifacts(
    *,
    output_dir: Path,
    expected_platform: str,
    expected_task: str,
    expected_root_height_m: float,
    report_pins: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate a completed run without importing Torch."""

    if not output_dir.is_dir() or output_dir.is_symlink():
        raise SequenceFailure(f"training output is not a real directory: {output_dir}")
    manifest_path = output_dir / "training_manifest.json"
    status_path = output_dir / "run_status.json"
    jsonl_path = output_dir / "training.jsonl"
    manifest = _read_json_object(manifest_path, label="training manifest")
    status = _read_json_object(status_path, label="run status")
    line_count, last = _validate_training_jsonl(jsonl_path)
    if status != last:
        raise SequenceFailure("run_status.json differs from the final JSONL record")
    if manifest.get("schema") != TRAINING_MANIFEST_SCHEMA:
        raise SequenceFailure("training manifest schema is incompatible")
    legacy_phase_keys = _legacy_phase_key_paths(manifest)
    if legacy_phase_keys:
        raise SequenceFailure(
            "training manifest contains legacy phase1/phase2 keys: "
            + ", ".join(legacy_phase_keys)
        )
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise SequenceFailure("training manifest config is missing")
    expected_config = _expected_training_config(expected_task)
    if config != expected_config:
        drift = {
            name: {"expected": expected_config.get(name), "found": config.get(name)}
            for name in sorted(set(expected_config) | set(config))
            if expected_config.get(name) != config.get(name)
            or (name in expected_config) != (name in config)
        }
        raise SequenceFailure(
            "training manifest config differs from the complete formal CLI/default "
            "contract: " + json.dumps(drift, sort_keys=True)
        )
    expected_manifest = {
        "algorithm": EXPECTED_ALGORITHM,
        "algorithm_signature": EXPECTED_ALGORITHM_SIGNATURE,
        "control_hierarchy": (
            "Diffusion imitation -> independent hard-tip IK commands -> "
            "per-tip constrained IK/KKT -> actuators"
        ),
        "physics_backend": "native-mujoco-warp",
        "torch_warp_exchange": "persistent-zero-copy-alias",
        "platform": expected_platform,
        "trained_task": expected_task,
        "task": expected_task,
        "task_index": EXPECTED_TASK_INDEX[expected_task],
        "root_height_m": expected_root_height_m,
        "expert": EXPECTED_EXPERT,
        "total_mjwarp_expert_steps": EXPECTED_TOTAL_STEPS,
        "expert_steps": EXPECTED_TOTAL_STEPS,
        "diffusion_epochs": 60,
        "success_gate": True,
        "trajectory_certification": EXPECTED_TRAJECTORY_CERTIFICATION,
        "anti_support_gate_used": False,
        "physical_hard_tip_plane_success_invariant": True,
        "grasp_plane_fraction_bounds": [0.4, 0.6],
        "learned_grasp_plane_objects": list(EXPECTED_OBJECTS),
        "fixed_grasp_plane_fractions": {},
        "fixed_grasp_plane_yaw_rad": 0.0,
        "fixed_grasp_plane_physical_tolerance_m": 0.002,
        "independent_hard_tip_control": True,
        "diffusion_action_supervision": EXPECTED_ACTION_SUPERVISION,
        "plate_trained": False,
        "reset_templates_per_object": 32,
        "physics_layout": EXPECTED_PHYSICS_LAYOUT,
        "global_worlds": EXPECTED_WORLDS,
        "object_world_counts": EXPECTED_OBJECT_WORLD_COUNTS,
        "remainder_worlds": EXPECTED_REMAINDER_WORLDS,
        "remainder_object_world_counts": (
            EXPECTED_REMAINDER_OBJECT_WORLD_COUNTS
        ),
        "observation_dim": EXPECTED_OBSERVATION_DIM,
        "action_dim": EXPECTED_ACTION_DIM,
        "runtime_code_fingerprint_sha256": (
            DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256
        ),
    }
    for name, wanted in expected_manifest.items():
        if manifest.get(name) != wanted:
            raise SequenceFailure(
                f"training manifest {name}: expected {wanted!r}, "
                f"found {manifest.get(name)!r}"
            )
    if not isinstance(manifest.get("created_utc"), str) or not manifest[
        "created_utc"
    ]:
        raise SequenceFailure("training manifest created_utc is missing")
    if not isinstance(manifest.get("translation_path_contract"), dict):
        raise SequenceFailure("training manifest translation path contract is missing")
    expected_rotation_distribution = {
        "bounds": [10, 45],
        "integer_closed_intervals": True,
        "maximum_successful_demo_required_per_object": True,
    }
    if manifest.get("rotation_angle_distribution_deg") != (
        expected_rotation_distribution
    ):
        raise SequenceFailure("training manifest rotation distribution is stale")
    expert_final = manifest.get("expert_final")
    if not isinstance(expert_final, dict):
        raise SequenceFailure("training manifest full-task expert summary is missing")
    expected_expert_final = {
        "phase": "all",
        "steps": EXPECTED_TOTAL_STEPS,
        "control_ticks": 4_500,
        "remainder_worlds": EXPECTED_REMAINDER_WORLDS,
        "remainder_object_world_counts": (
            EXPECTED_REMAINDER_OBJECT_WORLD_COUNTS
        ),
        "remainder_backend_health": None,
        "remainder_controller_safety_counters": None,
    }
    for name, wanted in expected_expert_final.items():
        if expert_final.get(name) != wanted:
            raise SequenceFailure(
                f"training expert_final {name}: expected {wanted!r}, "
                f"found {expert_final.get(name)!r}"
            )
    if expected_task == TASK_ROTATE:
        if expert_final.get("rotation_target_angle_bounds_deg") != [10, 45]:
            raise SequenceFailure(
                "rotation expert does not cover the complete 10-45 degree task"
            )
    elif "rotation_target_angle_bounds_deg" in expert_final:
        raise SequenceFailure("translation expert contains rotation curriculum data")
    _validate_single_phase_demo_coverage(
        manifest, expected_task=expected_task
    )
    model_shards = _validate_model_shards(
        manifest.get("model_shards"),
        platform=expected_platform,
        task=expected_task,
        root_height_m=expected_root_height_m,
    )
    maximum_nv = max(int(shard["nv"]) for shard in model_shards)
    if manifest.get("maximum_shard_nv") != maximum_nv:
        raise SequenceFailure("training manifest maximum_shard_nv is inconsistent")
    for name in (
        "parent_model_fingerprint_sha256",
        "parent_controller_fingerprint_sha256",
        "platform_adapter_fingerprint_sha256",
        "fruit_manifest_sha256",
    ):
        if not _is_sha256(manifest.get(name)):
            raise SequenceFailure(f"training manifest {name} is not a SHA256")
    capacity = manifest.get("capacity_selection")
    if not isinstance(capacity, dict):
        raise SequenceFailure("training manifest capacity selection is missing")
    expected_capacity = {
        "report_path": report_pins["report_path"],
        "report_sha256": report_pins["report_sha256"],
        "runtime_code_fingerprint_sha256": report_pins[
            "runtime_code_fingerprint_sha256"
        ],
        "platforms": ["gripper6"],
        "selected_num_envs": EXPECTED_WORLDS,
        "task": expected_task,
        "task_index": EXPECTED_TASK_INDEX[expected_task],
        "root_height_m": expected_root_height_m,
        "platform_root_heights_m": report_pins[
            "platform_root_heights_m"
        ],
        "platform_rl_model_fingerprints_sha256": report_pins[
            "platform_rl_model_fingerprints_sha256"
        ],
        "object_world_counts": EXPECTED_OBJECT_WORLD_COUNTS,
        "remainder_worlds": EXPECTED_REMAINDER_WORLDS,
        "remainder_object_world_counts": (
            EXPECTED_REMAINDER_OBJECT_WORLD_COUNTS
        ),
    }
    for name, wanted in expected_capacity.items():
        if capacity.get(name) != wanted:
            raise SequenceFailure(
                f"training capacity selection mismatch for {name}"
            )
    if capacity.get("model_shards") != model_shards:
        raise SequenceFailure("capacity and manifest model-shard bindings differ")
    if capacity.get("passing_candidates") != [EXPECTED_WORLDS]:
        raise SequenceFailure("capacity selection does not bind the sole formal worlds")
    for name in (
        "config_canonical_sha256",
        "candidate_config_set_sha256",
        "selected_config_sha256",
    ):
        if not _is_sha256(capacity.get(name)):
            raise SequenceFailure(f"capacity selection {name} is not a SHA256")
    for name in (
        "parent_model_fingerprint_sha256",
        "parent_controller_fingerprint_sha256",
        "platform_adapter_fingerprint_sha256",
    ):
        if capacity.get(name) != manifest.get(name):
            raise SequenceFailure(
                f"capacity and manifest platform fingerprint differ for {name}"
            )
    increments = capacity.get("certified_stage_peak_increment_by_stage")
    if not isinstance(increments, dict) or set(increments) != {
        "native_search",
        "diffusion",
    } or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in increments.values()
    ):
        raise SequenceFailure("capacity selection stage increments are invalid")
    admissions = capacity.get("training_memory_admissions")
    if not isinstance(admissions, dict) or set(admissions) != {
        "native_search",
        "diffusion",
    }:
        raise SequenceFailure("training memory admissions are incomplete")
    memory = _validate_memory(
        manifest.get("training_device_memory"), label=expected_platform
    )
    controller = manifest.get("controller_safety")
    if not isinstance(controller, dict):
        raise SequenceFailure("training manifest controller safety is missing")
    if tuple(controller.get("hard_failure_fields", ())) != HARD_COUNTERS:
        raise SequenceFailure("training hard-failure field contract changed")
    if controller.get("fail_closed") is not True:
        raise SequenceFailure("training controller safety is not fail closed")
    for label in ("all", "combined_totals"):
        _validate_hard_counters(
            controller.get(label), label=f"{expected_platform}/{label}"
        )
    _validate_hard_counters(
        expert_final.get("controller_safety_counters_total"),
        label=f"{expected_platform}/expert_final",
    )
    _validate_hard_counters(
        status.get("controller_safety_counters"),
        label=f"{expected_platform}/run_status",
    )
    _validate_memory(
        status.get("training_device_memory"),
        label=f"{expected_platform}/run_status",
    )
    model_name = manifest.get("final_model")
    if model_name != "final_diffusion_model.pt":
        raise SequenceFailure(
            "training manifest final model must be final_diffusion_model.pt"
        )
    if status.get("final_model") != model_name:
        raise SequenceFailure("run status final model differs from manifest")
    model = output_dir / model_name
    if not model.is_file() or model.is_symlink() or model.stat().st_size <= 0:
        raise SequenceFailure(f"final model is missing or empty: {model}")
    model_sha = _sha256(model)
    if not _is_sha256(manifest.get("final_model_sha256")) or (
        manifest.get("final_model_sha256") != model_sha
    ):
        raise SequenceFailure("final model SHA256 mismatch")
    return {
        "platform": expected_platform,
        "task": expected_task,
        "root_height_m": expected_root_height_m,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "status_sha256": _sha256(status_path),
        "training_jsonl_sha256": _sha256(jsonl_path),
        "training_jsonl_records": line_count,
        "final_model": model_name,
        "final_model_sha256": model_sha,
        "final_model_bytes": model.stat().st_size,
        "peak_device_used_gb": memory.get("peak_device_used_gb"),
        "peak_device_memory_fraction": memory.get(
            "peak_device_memory_fraction"
        ),
        "algorithm_signature": manifest.get("algorithm_signature"),
        "physics_layout": manifest.get("physics_layout"),
        "runtime_code_fingerprint_sha256": manifest.get(
            "runtime_code_fingerprint_sha256"
        ),
        "config": config,
        "capacity_selection": capacity,
    }


def create_models_archive(
    *, output_dir: Path, packages_dir: Path, label: str, run_id: str
) -> dict[str, Any]:
    packages_dir.mkdir(parents=True, exist_ok=True)
    archive = packages_dir / f"{label}_models_only_{run_id}.tar.gz"
    if archive.exists() or archive.with_suffix(archive.suffix + ".sha256").exists():
        raise SequenceFailure(f"models archive already exists: {archive}")
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    included: list[str] = []
    try:
        with tarfile.open(temporary, mode="w:gz", compresslevel=6) as stream:
            for path in sorted(output_dir.rglob("*")):
                relative = path.relative_to(output_dir)
                if relative.parts and relative.parts[0] in {
                    "certified_ik_demos",
                    "elite_demos",
                }:
                    continue
                if path.is_symlink():
                    raise SequenceFailure(f"refusing to archive symlink: {path}")
                if path.is_file():
                    arcname = Path(label) / relative
                    stream.add(path, arcname=str(arcname), recursive=False)
                    included.append(str(arcname))
        if not included or any(
            {"certified_ik_demos", "elite_demos"}.intersection(Path(name).parts)
            for name in included
        ):
            raise SequenceFailure("models-only archive member selection failed")
        os.replace(temporary, archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = _sha256(archive)
    sidecar = archive.with_name(archive.name + ".sha256")
    _atomic_text(sidecar, f"{digest}  {archive.name}\n")
    return {
        "path": str(archive),
        "sha256": digest,
        "sha256_path": str(sidecar),
        "bytes": archive.stat().st_size,
        "members": included,
        "certified_ik_demos_excluded": True,
    }


@contextmanager
def _exclusive_gpu_lock(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_absolute() or path == Path("/") or path.suffix != ".lock":
        raise SequenceFailure("GPU lock path must be an absolute .lock file")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve() != path.parent:
        raise SequenceFailure("GPU lock parent must be a real canonical directory")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SequenceFailure(f"cannot safely open GPU lock {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SequenceFailure(f"GPU lock is not a regular file: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SequenceFailure(f"GPU 0 formal-training lock is held: {path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()} started={_utc_now()}\n".encode())
        yield {"path": str(path), "pid": os.getpid()}
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _training_command(
    *,
    project_root: Path,
    capacity_report: Path,
    output_dir: Path,
    task: str,
) -> list[str]:
    if task not in TASKS:
        raise SequenceFailure(f"unknown training task: {task!r}")
    return [
        str(project_root / "MJWarp_RL" / "run_hpc.sh"),
        "-m",
        "MJWarp_RL.train",
        "--seed",
        str(FIXED_SEED),
        "--device",
        FIXED_DEVICE,
        "--task",
        task,
        "--num-envs",
        "auto",
        "--capacity-report",
        str(capacity_report),
        "--output",
        str(output_dir),
    ]


class StatusWriter:
    def __init__(self, path: Path, initial: Mapping[str, Any]) -> None:
        self.path = path
        self.value = dict(initial)
        self.write()

    def update(self, **changes: Any) -> None:
        self.value.update(changes)
        self.write()

    def run(self, name: str, **changes: Any) -> None:
        runs = dict(self.value.get("runs", {}))
        current = dict(runs.get(name, {}))
        current.update(changes)
        runs[name] = current
        self.update(runs=runs)

    def write(self) -> None:
        self.value["updated_utc"] = _utc_now()
        _atomic_json(self.path, self.value)
        print(
            json.dumps(
                {
                    "state": self.value.get("state"),
                    "stage": self.value.get("stage"),
                    "status": str(self.path),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _validate_g6_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve only the G6 release and its two task-specific reports."""

    release = args.release_root.expanduser().resolve()
    g6 = (args.g6_root or release / "rl16_api").expanduser().resolve()
    python = args.python.expanduser().resolve()
    capacity_reports = {
        TASK_TRANSLATE: (
            args.translation_capacity_report.expanduser().resolve()
        ),
        TASK_ROTATE: args.rotation_capacity_report.expanduser().resolve(),
    }
    output_base = args.output_base.expanduser().resolve(strict=False)
    lock_file = DEFAULT_LOCK_FILE
    for label, root in (("release", release), ("G6", g6)):
        if not root.is_dir() or root.is_symlink():
            raise SequenceFailure(
                f"{label} root is missing or a symlink: {root}"
            )
    if not _relative_to(g6, release):
        raise SequenceFailure("G6 root must be inside the selected release")
    for relative in ("MJWarp_RL/train.py", "MJWarp_RL/run_hpc.sh"):
        required = g6 / relative
        if not required.is_file() or required.is_symlink():
            raise SequenceFailure(
                f"G6 release file is missing or a symlink: {required}"
            )
    launcher = g6 / "MJWarp_RL" / "run_hpc.sh"
    if not os.access(launcher, os.X_OK):
        raise SequenceFailure(f"G6 HPC launcher is not executable: {launcher}")
    if (
        not python.is_file()
        or python.is_symlink()
        or not os.access(python, os.X_OK)
    ):
        raise SequenceFailure(
            f"HPC Python is missing, a symlink, or not executable: {python}"
        )
    if _relative_to(output_base.resolve(strict=False), release):
        raise SequenceFailure("formal output base must be outside the release")
    report_pins = {
        TASK_TRANSLATE: _verify_g6_capacity_report(
            capacity_report=capacity_reports[TASK_TRANSLATE],
            capacity_sha256=args.translation_capacity_report_sha256,
            expected_task=TASK_TRANSLATE,
        ),
        TASK_ROTATE: _verify_g6_capacity_report(
            capacity_report=capacity_reports[TASK_ROTATE],
            capacity_sha256=args.rotation_capacity_report_sha256,
            expected_task=TASK_ROTATE,
        ),
    }
    return {
        "release": release,
        "g6": g6,
        "python": python,
        "capacity_reports": capacity_reports,
        "output_base": output_base,
        "lock_file": lock_file,
        "report_pins": report_pins,
    }


def _validate_prepared_pair_root(
    *,
    pair_root: Path,
    run_id: str,
    release: Path,
    report_pins: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not pair_root.is_dir()
        or pair_root.is_symlink()
        or pair_root.resolve() != pair_root
    ):
        raise SequenceFailure(
            f"prepared pair root is missing, non-canonical, or a symlink: {pair_root}"
        )
    allowed = {"sequence_status.json", "screen.log"}
    unexpected = sorted(path.name for path in pair_root.iterdir() if path.name not in allowed)
    if unexpected:
        raise SequenceFailure(
            "prepared pair root contains unexpected entries: " + ", ".join(unexpected)
        )
    launch = _read_json_object(
        pair_root / "sequence_status.json", label="prepared launch status"
    )
    expected = {
        "schema": SCHEMA,
        "run_id": run_id,
        "state": "launching",
        "stage": "screen_launch",
        "release_root": str(release),
        "pair_root": str(pair_root),
        "capacity": dict(report_pins),
    }
    for name, wanted in expected.items():
        if launch.get(name) != wanted:
            raise SequenceFailure(
                f"prepared launch status mismatch for {name}: "
                f"expected {wanted!r}, found {launch.get(name)!r}"
            )
    return launch


def _foreground(args: argparse.Namespace) -> int:
    resolved = _validate_g6_inputs(args)
    run_id = args.run_id
    pair_root = resolved["output_base"] / run_id
    launch: dict[str, Any] | None = None
    if args.prepared_root:
        launch = _validate_prepared_pair_root(
            pair_root=pair_root,
            run_id=run_id,
            release=resolved["release"],
            report_pins=resolved["report_pins"],
        )
    else:
        pair_root.mkdir(parents=True, exist_ok=False)
    outputs = {label: pair_root / label for label, _, _, _ in RUN_SPECS}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise SequenceFailure(
            "formal output already exists; refusing resume/retry: "
            + ", ".join(existing)
        )
    initial = {
        "schema": SCHEMA,
        "run_id": run_id,
        "state": "checking",
        "stage": "foreground_start",
        "created_utc": launch.get("created_utc", _utc_now()) if launch else _utc_now(),
        "pid": os.getpid(),
        "fixed_training": {
            "seed": FIXED_SEED,
            "device": FIXED_DEVICE,
            "num_envs": "auto",
            "expected_selected_num_envs": EXPECTED_WORLDS,
            "total_steps_each": EXPECTED_TOTAL_STEPS,
            "aggregate_total_steps": EXPECTED_TOTAL_STEPS * len(RUN_SPECS),
            "order": [label for label, _, _, _ in RUN_SPECS],
        },
        "release_root": str(resolved["release"]),
        "capacity": resolved["report_pins"],
        "pair_root": str(pair_root),
        "runs": {
            label: {
                "state": "pending",
                "platform": platform,
                "task": task,
                "root_height_m": EXPECTED_ROOT_HEIGHT_M[(platform, task)],
                "output_dir": str(outputs[label]),
                "log_path": str(pair_root / f"{label}_training.log"),
            }
            for label, platform, task, _ in RUN_SPECS
        },
    }
    status = StatusWriter(pair_root / "sequence_status.json", initial)
    owner = ChildOwner()
    owner.install_handlers()
    try:
        with _exclusive_gpu_lock(resolved["lock_file"]) as lock:
            status.update(stage="initial_gates", gpu_lock=lock)
            initial_gpu = _gpu_idle_snapshot()
            initial_disk = _disk_gate(pair_root, required_gib=INITIAL_DISK_GIB)
            status.update(
                stage="g6_task_admission",
                initial_gpu_gate=initial_gpu,
                initial_disk_gate=initial_disk,
            )
            admissions: dict[str, dict[str, Any]] = {}
            for label, platform, task, root_key in RUN_SPECS:
                status.run(label, state="admitting", admitted_utc=None)
                admission = _g6_admission(
                    owner=owner,
                    python=resolved["python"],
                    project_root=resolved[root_key],
                    capacity_report=resolved["capacity_reports"][task],
                    expected_platform=platform,
                    expected_task=task,
                    expected_root_height_m=EXPECTED_ROOT_HEIGHT_M[
                        (platform, task)
                    ],
                    report_pins=resolved["report_pins"][task],
                )
                admissions[task] = admission
                status.run(
                    label,
                    state="admitted",
                    admission=admission,
                    admitted_utc=_utc_now(),
                )
            post_admission_gpu = _gpu_idle_snapshot()
            status.update(
                state="running",
                stage="training_start",
                admissions=admissions,
                post_admission_gpu_gate=post_admission_gpu,
            )
            validations: dict[str, dict[str, Any]] = {}
            packages: dict[str, dict[str, Any]] = {}
            inter_run_gates: dict[str, Any] = {}
            for index, (label, platform, task, root_key) in enumerate(RUN_SPECS):
                if index:
                    status.update(stage=f"before_{label}_gates")
                    inter_run_gates[label] = {
                        "gpu": _gpu_idle_snapshot(),
                        "disk": _disk_gate(
                            pair_root, required_gib=BETWEEN_RUNS_DISK_GIB
                        ),
                    }
                    status.update(inter_run_gates=inter_run_gates)
                status.update(stage=f"{label}_training", next_run=label)
                status.run(label, state="running", started_utc=_utc_now())
                root = resolved[root_key]
                returncode = owner.run_log(
                    _training_command(
                        project_root=root,
                        capacity_report=resolved["capacity_reports"][task],
                        output_dir=outputs[label],
                        task=task,
                    ),
                    cwd=root,
                    env=_project_env(root, python=resolved["python"]),
                    log_path=pair_root / f"{label}_training.log",
                )
                if returncode != 0:
                    status.run(label, state="failed", returncode=returncode)
                    raise SequenceFailure(
                        f"{label} training exited with status {returncode}"
                    )
                validation = validate_training_artifacts(
                    output_dir=outputs[label],
                    expected_platform=platform,
                    expected_task=task,
                    expected_root_height_m=EXPECTED_ROOT_HEIGHT_M[
                        (platform, task)
                    ],
                    report_pins=resolved["report_pins"][task],
                )
                archive = create_models_archive(
                    output_dir=outputs[label],
                    packages_dir=pair_root / "packages",
                    label=label,
                    run_id=run_id,
                )
                validations[label] = validation
                packages[label] = archive
                status.run(
                    label,
                    state="complete",
                    returncode=0,
                    completed_utc=_utc_now(),
                    validation=validation,
                    package=archive,
                )
            status.update(
                state="complete",
                stage="complete",
                completed_utc=_utc_now(),
                validations=validations,
                packages=packages,
                next_run=None,
            )
        return 0
    except BaseException as exc:
        owner.terminate_active()
        try:
            current = status.value.get("runs", {})
            running = next(
                (
                    name
                    for name, value in current.items()
                    if isinstance(value, Mapping) and value.get("state") == "running"
                ),
                None,
            )
            if running:
                status.run(
                    running,
                    state="interrupted"
                    if isinstance(exc, SequenceInterrupted)
                    else "failed",
                )
            status.update(
                state="interrupted"
                if isinstance(exc, SequenceInterrupted)
                else "failed",
                stage="interrupted"
                if isinstance(exc, SequenceInterrupted)
                else "failed",
                failed_utc=_utc_now(),
                error_type=type(exc).__name__,
                error=str(exc),
                received_signal=owner.received_signal,
                retry_attempted=False,
                outputs_deleted=False,
            )
        finally:
            if isinstance(exc, SequenceInterrupted):
                return 128 + exc.signum
            if isinstance(exc, KeyboardInterrupt):
                return 130
        raise


def _check_only(args: argparse.Namespace) -> int:
    resolved = _validate_g6_inputs(args)
    owner = ChildOwner()
    owner.install_handlers()
    with _exclusive_gpu_lock(resolved["lock_file"]) as lock:
        gpu = _gpu_idle_snapshot()
        disk = _disk_gate(
            resolved["output_base"] / args.run_id,
            required_gib=INITIAL_DISK_GIB,
        )
        admissions: dict[str, dict[str, Any]] = {}
        for _label, platform, task, root_key in RUN_SPECS:
            admissions[task] = _g6_admission(
                owner=owner,
                python=resolved["python"],
                project_root=resolved[root_key],
                capacity_report=resolved["capacity_reports"][task],
                expected_platform=platform,
                expected_task=task,
                expected_root_height_m=EXPECTED_ROOT_HEIGHT_M[
                    (platform, task)
                ],
                report_pins=resolved["report_pins"][task],
            )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "check_only": True,
                "gpu_lock": lock,
                "gpu_gate": gpu,
                "disk_gate": disk,
                "capacity": resolved["report_pins"],
                "admissions": admissions,
                "training_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _background(args: argparse.Namespace) -> int:
    resolved = _validate_g6_inputs(args)
    pair_root = resolved["output_base"] / args.run_id
    pair_root.mkdir(parents=True, exist_ok=False)
    initial = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "state": "launching",
        "stage": "screen_launch",
        "created_utc": _utc_now(),
        "release_root": str(resolved["release"]),
        "pair_root": str(pair_root),
        "capacity": resolved["report_pins"],
        "runs": {
            label: {
                "state": "pending",
                "platform": platform,
                "task": task,
                "root_height_m": EXPECTED_ROOT_HEIGHT_M[(platform, task)],
                "output_dir": str(pair_root / label),
                "log_path": str(pair_root / f"{label}_training.log"),
            }
            for label, platform, task, _ in RUN_SPECS
        },
    }
    _atomic_json(pair_root / "sequence_status.json", initial)
    session = ("grab_g6_" + args.run_id)[:80]
    command = [
        "screen",
        "-Logfile",
        str(pair_root / "screen.log"),
        "-L",
        "-dmS",
        session,
        str(resolved["python"]),
        str(Path(__file__).resolve()),
        "--foreground",
        "--prepared-root",
        "--run-id",
        args.run_id,
        "--release-root",
        str(resolved["release"]),
        "--g6-root",
        str(resolved["g6"]),
        "--translation-capacity-report",
        str(resolved["capacity_reports"][TASK_TRANSLATE]),
        "--translation-capacity-report-sha256",
        args.translation_capacity_report_sha256,
        "--rotation-capacity-report",
        str(resolved["capacity_reports"][TASK_ROTATE]),
        "--rotation-capacity-report-sha256",
        args.rotation_capacity_report_sha256,
        "--output-base",
        str(resolved["output_base"]),
        "--python",
        str(resolved["python"]),
    ]
    result = subprocess.run(
        command,
        cwd=resolved["g6"],
        env=_project_env(resolved["g6"], python=resolved["python"]),
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        initial.update(
            state="failed",
            stage="screen_launch_failed",
            error=result.stderr.strip() or result.stdout.strip(),
            updated_utc=_utc_now(),
        )
        _atomic_json(pair_root / "sequence_status.json", initial)
        raise SequenceFailure(
            f"screen launch failed with status {result.returncode}"
        )
    print(f"started screen session: {session}")
    print(f"G6 two-run root: {pair_root}")
    print(f"status: {pair_root / 'sequence_status.json'}")
    print(f"screen log: {pair_root / 'screen.log'}")
    print(f"attach: screen -r {session}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--foreground",
        action="store_true",
        help="run gates and G6 translation -> G6 rotation in this terminal",
    )
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="run both read-only G6 admissions and gates; do not train",
    )
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--g6-root", type=Path)
    parser.add_argument(
        "--translation-capacity-report",
        type=Path,
        default=DEFAULT_TRANSLATION_CAPACITY_REPORT,
    )
    parser.add_argument(
        "--translation-capacity-report-sha256",
        default=DEFAULT_TRANSLATION_CAPACITY_REPORT_SHA256,
    )
    parser.add_argument(
        "--rotation-capacity-report",
        type=Path,
        default=DEFAULT_ROTATION_CAPACITY_REPORT,
    )
    parser.add_argument(
        "--rotation-capacity-report-sha256",
        default=DEFAULT_ROTATION_CAPACITY_REPORT_SHA256,
    )
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--prepared-root", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not RUN_ID_RE.fullmatch(args.run_id):
        parser.error(
            "--run-id must be 1-80 safe characters: letters, digits, dot, dash, underscore"
        )
    if args.prepared_root and not args.foreground:
        parser.error("--prepared-root is internal and requires --foreground")
    try:
        if args.check_only:
            return _check_only(args)
        if args.foreground:
            return _foreground(args)
        return _background(args)
    except SequenceInterrupted as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 128 + exc.signum
    except SequenceFailure as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
