#!/usr/bin/env python3
"""Write GPU/node/process telemetry beside an hpc run.

This sidecar intentionally uses only the Python standard library plus the
``nvidia-smi`` executable supplied by the scheduled GPU node.  It does not
import Torch, MuJoCo, Warp, or load a checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Optional
import uuid


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.monitoring import (
    SYSTEM_SCHEMA,
    atomic_write_json,
    jsonable,
    read_json,
    utc_now,
)
from s_hpc.paths import add_external_path_flag, resolve_package_path


GPU_FIELDS = (
    "index",
    "name",
    "uuid",
    "driver_version",
    "pstate",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
)


def append_strict_jsonl(
    path: Path,
    payload: dict[str, Any],
) -> Path:
    """Durably append one strict-JSON telemetry event.

    The history file is deliberately never replaced or truncated, so a stable
    run directory retains telemetry from every scheduler allocation.
    """

    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with target.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return target


def persist_system_sample(
    target: Path,
    history_target: Path,
    payload: dict[str, Any],
    *,
    append_history: bool,
) -> None:
    """Best-effort telemetry persistence that can never stop training."""

    try:
        atomic_write_json(target, payload)
    except (OSError, TypeError, ValueError) as exc:
        print(
            "WARNING: system monitor snapshot write failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
    if append_history:
        try:
            append_strict_jsonl(history_target, payload)
        except (OSError, TypeError, ValueError) as exc:
            print(
                "WARNING: system monitor history append failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def _visible_gpu_selection() -> tuple[Optional[str], list[str]]:
    """Return the scheduler/container GPU visibility constraint, if known."""

    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        raw = os.environ.get(name)
        if raw is not None:
            return name, [
                token.strip()
                for token in raw.split(",")
                if token.strip()
            ]
    return None, []


def _gpu_matches_token(device: dict[str, Any], token: str) -> bool:
    normalized = token.strip()
    if normalized.startswith("MIG-"):
        normalized = normalized.removeprefix("MIG-").split("/", 1)[0]
    if normalized == str(device.get("index")):
        return True
    uuid_value = str(device.get("uuid") or "")
    return bool(
        uuid_value
        and (
            normalized == uuid_value
            or uuid_value.startswith(normalized)
            or normalized.startswith(uuid_value)
        )
    )


def _number(value: str) -> Any:
    stripped = value.strip()
    if stripped in {"", "N/A", "[N/A]"}:
        return None
    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return stripped


def query_nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if result.returncode != 0:
        return {
            "available": False,
            "returncode": result.returncode,
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    all_devices = []
    for line in result.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != len(GPU_FIELDS):
            continue
        raw = dict(zip(GPU_FIELDS, values))
        all_devices.append(
            {
                "index": _number(raw["index"]),
                "name": raw["name"],
                "uuid": raw["uuid"],
                "driver_version": raw["driver_version"],
                "performance_state": raw["pstate"],
                "gpu_utilization_percent": _number(
                    raw["utilization.gpu"]
                ),
                "memory_utilization_percent": _number(
                    raw["utilization.memory"]
                ),
                "memory_used_mib": _number(raw["memory.used"]),
                "memory_total_mib": _number(raw["memory.total"]),
                "temperature_c": _number(raw["temperature.gpu"]),
                "power_draw_w": _number(raw["power.draw"]),
                "power_limit_w": _number(raw["power.limit"]),
                "sm_clock_mhz": _number(raw["clocks.sm"]),
            }
        )
    selection_source, tokens = _visible_gpu_selection()
    lowered = {token.lower() for token in tokens}
    no_device_tokens = {"", "-1", "none", "void", "nodevfiles"}
    unrestricted_tokens = {"all"}
    selection_known = selection_source is not None
    if selection_known and (
        not tokens or lowered.intersection(no_device_tokens)
    ):
        devices: list[dict[str, Any]] = []
        selection_mode = "no_visible_devices"
    elif selection_known and lowered.intersection(unrestricted_tokens):
        devices = [dict(device, allocated=True) for device in all_devices]
        selection_mode = "all_visible"
    elif selection_known:
        devices = [
            dict(device, allocated=True)
            for device in all_devices
            if any(
                _gpu_matches_token(device, token)
                for token in tokens
            )
        ]
        selection_mode = "visibility_filter"
    elif len(all_devices) == 1:
        devices = [dict(all_devices[0], allocated=True)]
        selection_mode = "single_device_fallback"
    else:
        # nvidia-smi can expose GPUs owned by other jobs even when CUDA
        # isolation metadata is unavailable. Keep them visible for diagnosis,
        # but never generate utilization/temperature alarms from them.
        devices = [
            dict(device, allocated=None) for device in all_devices
        ]
        selection_mode = "allocation_unknown"
    return {
        "available": bool(devices),
        "devices": devices,
        "physical_devices_seen": len(all_devices),
        "selection": {
            "known": selection_known or len(all_devices) == 1,
            "source": selection_source,
            "raw": (
                os.environ.get(selection_source)
                if selection_source is not None
                else None
            ),
            "tokens": tokens,
            "mode": selection_mode,
            "matched_devices": len(devices),
        },
    }


def _read_key_values(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    result = {}
    for line in lines:
        key, separator, value = line.partition(":")
        if separator:
            result[key] = value.strip()
    return result


def _proc_times(pid: int) -> Optional[tuple[int, int]]:
    try:
        process_fields = Path(f"/proc/{pid}/stat").read_text(
            encoding="utf-8"
        ).split()
        total_fields = Path("/proc/stat").read_text(
            encoding="utf-8"
        ).splitlines()[0].split()[1:]
        process_ticks = int(process_fields[13]) + int(process_fields[14])
        total_ticks = sum(int(value) for value in total_fields)
    except (OSError, IndexError, ValueError):
        return None
    return process_ticks, total_ticks


def process_snapshot(
    pid: Optional[int],
    *,
    previous_times: Optional[tuple[int, int]],
) -> tuple[dict[str, Any], Optional[tuple[int, int]]]:
    if pid is None or pid <= 0:
        return {"pid": pid, "visible": False}, None
    root = Path(f"/proc/{pid}")
    if not root.is_dir():
        return {"pid": pid, "visible": False, "alive": False}, None
    fields = _read_key_values(root / "status")
    io_fields = _read_key_values(root / "io")
    current_times = _proc_times(pid)
    cpu_percent = None
    if previous_times is not None and current_times is not None:
        process_delta = current_times[0] - previous_times[0]
        total_delta = current_times[1] - previous_times[1]
        if total_delta > 0:
            cpu_percent = (
                100.0
                * max(os.cpu_count() or 1, 1)
                * process_delta
                / total_delta
            )
    try:
        open_fds = len(list((root / "fd").iterdir()))
    except OSError:
        open_fds = None
    return (
        {
            "pid": pid,
            "visible": True,
            "alive": True,
            "state": fields.get("State"),
            "rss": fields.get("VmRSS"),
            "peak_rss": fields.get("VmHWM"),
            "virtual_memory": fields.get("VmSize"),
            "threads": _number(fields.get("Threads", "")),
            "open_fds": open_fds,
            "cpu_percent_all_cores": cpu_percent,
            "io": {
                key: _number(value)
                for key, value in io_fields.items()
                if key
                in {
                    "rchar",
                    "wchar",
                    "syscr",
                    "syscw",
                    "read_bytes",
                    "write_bytes",
                    "cancelled_write_bytes",
                }
            },
        },
        current_times,
    )


def node_snapshot(run_dir: Path) -> dict[str, Any]:
    memory = _read_key_values(Path("/proc/meminfo"))
    disk = shutil.disk_usage(run_dir)
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except OSError:
        load_1m = load_5m = load_15m = None
    try:
        uptime_seconds = float(
            Path("/proc/uptime").read_text(encoding="utf-8").split()[0]
        )
    except (OSError, IndexError, ValueError):
        uptime_seconds = None
    return {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "load_average": {
            "1m": load_1m,
            "5m": load_5m,
            "15m": load_15m,
        },
        "uptime_seconds": uptime_seconds,
        "memory": {
            key: memory.get(key)
            for key in (
                "MemTotal",
                "MemAvailable",
                "MemFree",
                "Buffers",
                "Cached",
                "SwapTotal",
                "SwapFree",
            )
        },
        "run_disk": {
            "path": str(run_dir),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "free_percent": 100.0 * disk.free / max(disk.total, 1),
        },
    }


def scheduler_snapshot() -> dict[str, Any]:
    names = (
        "PBS_JOBID",
        "PBS_JOBNAME",
        "PBS_QUEUE",
        "PBS_O_WORKDIR",
        "PBS_NODEFILE",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_SUBMIT_DIR",
        "SCREW_ALLOCATION_ID",
        "CUDA_VISIBLE_DEVICES",
    )
    environment = {
        name: os.environ[name] for name in names if os.environ.get(name)
    }
    return {
        "scheduler": (
            "pbs"
            if "PBS_JOBID" in environment
            else "slurm"
            if "SLURM_JOB_ID" in environment
            else "local"
        ),
        "scheduler_job_id": environment.get(
            "PBS_JOBID", environment.get("SLURM_JOB_ID")
        ),
        "allocation_id": environment.get("SCREW_ALLOCATION_ID"),
        "environment": environment,
    }


def current_app(run_dir: Path) -> tuple[Optional[int], Optional[str]]:
    live = read_json(run_dir / "status" / "live.json") or {}
    process = live.get("process", {})
    session = live.get("session", {})
    try:
        pid = int(process.get("pid"))
    except (TypeError, ValueError):
        pid = None
    session_id = session.get("id")
    return pid, session_id if isinstance(session_id, str) else None


def _run_identifier(run_dir: Path) -> str:
    """Return one stable identifier for all allocations of a run directory."""

    for filename in ("job.json", "live.json"):
        snapshot = read_json(run_dir / "status" / filename) or {}
        run = snapshot.get("run", {})
        run_id = run.get("id") if isinstance(run, dict) else None
        if isinstance(run_id, str) and run_id:
            return run_id
    return uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"screw-hpc-run:{run_dir.resolve()}",
    ).hex


def _allocation_identifier(
    run_dir: Path,
    scheduler: dict[str, Any],
) -> str:
    """Prefer the wrapper's allocation UUID over a scheduler-local job ID."""

    job = read_json(run_dir / "status" / "job.json") or {}
    allocation = job.get("allocation", {})
    if isinstance(allocation, dict):
        allocation_id = allocation.get("id")
        job_id = allocation.get("scheduler_job_id")
        scheduler_job_id = scheduler.get("scheduler_job_id")
        if (
            isinstance(allocation_id, str)
            and allocation_id
            and job_id == scheduler_job_id
        ):
            return allocation_id
    scheduler_job_id = scheduler.get("scheduler_job_id")
    if isinstance(scheduler_job_id, str) and scheduler_job_id:
        return scheduler_job_id
    return f"local-{uuid.uuid4().hex}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument(
        "--history-interval",
        type=float,
        default=15.0,
        help="Persistent JSONL sampling interval (15-30 seconds).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Write one sample and exit (used by diagnostics/tests).",
    )
    add_external_path_flag(parser)
    args = parser.parse_args()
    if not 1.0 <= args.interval <= 300.0:
        parser.error("--interval must be between 1 and 300 seconds")
    if not 15.0 <= args.history_interval <= 30.0:
        parser.error(
            "--history-interval must be between 15 and 30 seconds"
        )
    try:
        run_dir = resolve_package_path(
            args.run_dir,
            allow_external=args.allow_external_paths,
            label="run directory",
        )
    except ValueError as exc:
        parser.error(str(exc))
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "status" / "system.json"
    history_target = run_dir / "status" / "system-history.jsonl"
    scheduler = scheduler_snapshot()
    allocation_id = _allocation_identifier(run_dir, scheduler)
    run_id = _run_identifier(run_dir)
    sidecar_id = uuid.uuid4().hex
    stop_event = threading.Event()

    def stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    sequence = 0
    previous_times = None
    previous_app_pid = None
    state = "running"
    last_history_write = float("-inf")
    sample_period = min(args.interval, args.history_interval)
    while True:
        sequence += 1
        sample_monotonic = time.monotonic()
        app_pid, app_session = current_app(run_dir)
        if app_pid != previous_app_pid:
            previous_times = None
        process, previous_times = process_snapshot(
            app_pid, previous_times=previous_times
        )
        previous_app_pid = app_pid
        payload = {
            "schema": SYSTEM_SCHEMA,
            "sequence": sequence,
            "state": state,
            "updated_utc": utc_now(),
            "updated_unix": time.time(),
            "interval_seconds": args.interval,
            "history_interval_seconds": args.history_interval,
            "sampling_period_seconds": sample_period,
            "allocation_id": allocation_id,
            "identifiers": {
                "run_id": run_id,
                "allocation_id": allocation_id,
                "session_id": app_session or sidecar_id,
                "sidecar_session_id": sidecar_id,
                "application_session_id": app_session,
            },
            "run": {
                "id": run_id,
                "name": run_dir.name,
                "directory": str(run_dir),
            },
            "sidecar": {
                "id": sidecar_id,
                "pid": os.getpid(),
                "started_utc": datetime.fromtimestamp(
                    ps_start_time, timezone.utc
                ).isoformat(timespec="seconds"),
            },
            "scheduler": scheduler,
            "application": {
                "session_id": app_session,
                "process": process,
            },
            "node": node_snapshot(run_dir),
            "gpu": query_nvidia_smi(),
        }
        history_due = (
            args.once
            or sample_monotonic - last_history_write
            >= args.history_interval
        )
        persist_system_sample(
            target,
            history_target,
            payload,
            append_history=history_due,
        )
        if history_due:
            last_history_write = sample_monotonic
        if args.once:
            break
        if stop_event.wait(sample_period):
            state = "stopped"
            sequence += 1
            payload.update(
                {
                    "sequence": sequence,
                    "state": state,
                    "updated_utc": utc_now(),
                    "updated_unix": time.time(),
                }
            )
            persist_system_sample(
                target,
                history_target,
                payload,
                append_history=True,
            )
            break


ps_start_time = time.time()


if __name__ == "__main__":
    main()
