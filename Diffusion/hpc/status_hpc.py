#!/usr/bin/env python3
"""Inspect an hpc run without loading CUDA libraries or checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Optional


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.monitoring import (
    TERMINAL_STATES,
    heartbeat_age_seconds,
    read_json,
    request_graceful_stop,
    tail_jsonl,
)
from s_hpc.paths import add_external_path_flag, resolve_package_path


SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}
ACTIVE_STATES = frozenset({"starting", "running"})


def _format_duration(seconds: object) -> str:
    if seconds is None:
        return "N/A"
    try:
        value = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "N/A"
    days, remainder = divmod(value, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_rate(value: object, unit: str) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if number >= 1000.0:
        return f"{number:,.0f} {unit}/s"
    if number >= 1.0:
        return f"{number:,.2f} {unit}/s"
    if number > 0.0:
        return f"{number * 60.0:,.2f} {unit}/min"
    return "N/A"


def _format_percent(value: object, *, fraction: bool = True) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if fraction:
        number *= 100.0
    return f"{number:.2f}%"


def _format_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "N/A"
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def _format_bytes(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(number) < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return "N/A"


def parse_qstat(output: str) -> dict[str, str]:
    """Parse the key/value subset of PBS ``qstat -f`` output."""

    fields: dict[str, str] = {}
    current_key: Optional[str] = None
    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("Job Id:"):
            continue
        if " = " in line:
            key, value = line.strip().split(" = ", 1)
            fields[key] = value.strip()
            current_key = key
        elif current_key is not None and line[:1].isspace():
            fields[current_key] += line.strip()
    selected = (
        "Job_Name",
        "job_state",
        "queue",
        "server",
        "ctime",
        "qtime",
        "stime",
        "mtime",
        "comment",
        "exec_host",
        "exec_vnode",
        "Resource_List.walltime",
        "Resource_List.mem",
        "Resource_List.ncpus",
        "Resource_List.ngpus",
        "resources_used.walltime",
        "resources_used.cput",
        "resources_used.cpupercent",
        "resources_used.mem",
        "resources_used.vmem",
    )
    return {key: fields[key] for key in selected if key in fields}


def scheduler_status(app: Mapping[str, Any]) -> dict[str, Any]:
    job = app.get("job", {})
    scheduler = job.get("scheduler")
    job_id = job.get("scheduler_job_id")
    if not isinstance(job_id, str) or not job_id:
        return {"available": False, "reason": "no scheduler job id"}
    if scheduler == "pbs":
        command = ["qstat", "-f", job_id]
        parser = parse_qstat
    elif scheduler == "slurm":
        command = [
            "squeue",
            "-h",
            "-j",
            job_id,
            "-o",
            "%i|%T|%P|%M|%l|%R",
        ]

        def parser(output: str) -> dict[str, str]:
            line = output.strip().splitlines()
            if not line:
                return {}
            values = line[0].split("|")
            names = (
                "job_id",
                "state",
                "partition",
                "elapsed",
                "time_limit",
                "reason_or_node",
            )
            return dict(zip(names, values))

    else:
        return {"available": False, "reason": f"scheduler={scheduler!r}"}
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=4.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {
            "available": False,
            "job_id": job_id,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": result.returncode == 0,
        "job_id": job_id,
        "returncode": result.returncode,
        "fields": parser(result.stdout) if result.returncode == 0 else {},
        "error": (
            None
            if result.returncode == 0
            else result.stderr.strip() or result.stdout.strip()
        ),
    }


def _alert(
    code: str,
    severity: str,
    message: str,
    action: str,
) -> dict[str, str]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "recommended_action": action,
    }


def derive_alerts(
    app: Mapping[str, Any],
    system: Mapping[str, Any],
    *,
    stale_seconds: Optional[float] = None,
    job_status: Optional[Mapping[str, Any]] = None,
) -> tuple[list[dict[str, Any]], Optional[float], Optional[float]]:
    alerts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(item: Mapping[str, Any]) -> None:
        code = str(item.get("code", "unspecified"))
        severity = str(item.get("severity", "warning"))
        key = (code, severity)
        if key not in seen:
            alerts.append(dict(item))
            seen.add(key)

    for item in app.get("alerts", []) or []:
        if isinstance(item, Mapping):
            normalized = dict(item)
            normalized.setdefault(
                "recommended_action",
                "Inspect health metrics and the PBS output before continuing.",
            )
            add(normalized)
    if app.get("monitor_write_error"):
        add(
            _alert(
                "monitor_write_error",
                "warning",
                f"Training reported a monitor write error: "
                f"{app.get('monitor_write_error')}",
                "Check shared-filesystem health and available space.",
            )
        )

    state = str(app.get("state", "unknown"))
    app_age = heartbeat_age_seconds(app)
    configured_stale = app.get("monitor", {}).get("stale_after_seconds")
    threshold = (
        float(stale_seconds)
        if stale_seconds is not None
        else float(configured_stale or 60.0)
    )
    if state in ACTIVE_STATES and (app_age is None or app_age > threshold):
        add(
            _alert(
                "app_heartbeat_stale",
                "critical",
                (
                    "Training heartbeat is missing or "
                    f"{_format_duration(app_age)} old."
                ),
                "Inspect the PBS output/GPU; request a graceful stop or use qdel if hung.",
            )
        )
    if state == "failed":
        add(
            _alert(
                "pipeline_failed",
                "critical",
                "The pipeline recorded a failed terminal state.",
                "Read the recorded exception and preserve the last-good checkpoint.",
            )
        )
    elif state == "interrupted":
        add(
            _alert(
                "pipeline_interrupted",
                "warning",
                "The allocation stopped at a resumable cursor.",
                "Resubmit the same PBS job to continue.",
            )
        )
    wrapper_state = str((job_status or {}).get("state", ""))
    if wrapper_state == "failed":
        add(
            _alert(
                "scheduler_wrapper_failed",
                "critical",
                "The scheduler wrapper recorded a failed exit.",
                "Inspect raw exit code, PBS output, and the last-good checkpoint.",
            )
        )
    elif wrapper_state == "interrupted_resumable":
        add(
            _alert(
                "allocation_interrupted_resumable",
                "warning",
                "The allocation ended cleanly before the full pipeline completed.",
                "Resubmit the same job to resume.",
            )
        )
        if state in ACTIVE_STATES:
            add(
                _alert(
                    "interruption_not_acknowledged_by_pipeline",
                    "critical",
                    "The wrapper stopped, but pipeline did not record a resumable interrupted state.",
                    "Verify the latest checkpoint before resubmitting; the kill-after deadline may have fired.",
                )
            )

    system_age = heartbeat_age_seconds(system) if system else None
    system_interval = float(system.get("interval_seconds", 15.0) or 15.0)
    session_elapsed = float(
        app.get("session", {}).get("elapsed_seconds", 0.0) or 0.0
    )
    if not system and state in ACTIVE_STATES and session_elapsed >= 60.0:
        add(
            _alert(
                "system_telemetry_missing",
                "warning",
                "No node/GPU sidecar telemetry is available.",
                "Inspect the PBS output and system_monitor.py process.",
            )
        )
    elif (
        system
        and system.get("state") == "running"
        and (system_age is None or system_age > max(60.0, 3.0 * system_interval))
    ):
        add(
            _alert(
                "system_heartbeat_stale",
                "warning",
                f"Node/GPU telemetry is {_format_duration(system_age)} old.",
                "Check whether the sidecar or scheduled allocation exited.",
            )
        )

    app_job = app.get("job", {}).get("scheduler_job_id")
    system_job = system.get("scheduler", {}).get("scheduler_job_id")
    if app_job and system_job and app_job != system_job:
        add(
            _alert(
                "allocation_mismatch",
                "warning",
                "Application and system snapshots belong to different jobs.",
                "Wait one telemetry interval; otherwise inspect for a stale sidecar.",
            )
        )
    app_allocation = app.get("job", {}).get("allocation_id")
    system_allocation = system.get("identifiers", {}).get(
        "allocation_id"
    )
    if (
        app_allocation
        and system_allocation
        and app_allocation != system_allocation
    ):
        add(
            _alert(
                "allocation_mismatch",
                "warning",
                "Application and system snapshots belong to different allocations.",
                "Wait one telemetry interval; otherwise inspect for a stale sidecar.",
            )
        )
    app_session_id = app.get("session", {}).get("id")
    system_application = system.get("application", {})
    if (
        state in ACTIVE_STATES
        and app_session_id
        and system_application.get("session_id") == app_session_id
        and system_application.get("process", {}).get("alive") is False
    ):
        add(
            _alert(
                "training_process_not_alive",
                "critical",
                "The sidecar cannot find the process named by the live heartbeat.",
                "Inspect PBS output and scheduler state; preserve the last-good checkpoint.",
            )
        )

    gpu_snapshot = system.get("gpu", {})
    gpu_selection = gpu_snapshot.get("selection", {})
    if (
        state in ACTIVE_STATES
        and gpu_selection.get("mode") == "allocation_unknown"
    ):
        add(
            _alert(
                "gpu_allocation_unknown",
                "warning",
                "nvidia-smi exposes multiple GPUs but the job has no visibility metadata.",
                "Check CUDA_VISIBLE_DEVICES; GPU-specific alarms are suppressed to avoid blaming another job.",
            )
        )
    for device in gpu_snapshot.get("devices", []) or []:
        if device.get("allocated") is not True:
            continue
        used = device.get("memory_used_mib")
        total = device.get("memory_total_mib")
        if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total:
            fraction = used / total
            if fraction >= 0.97:
                add(
                    _alert(
                        "gpu_memory_critical",
                        "critical",
                        f"GPU memory is {100.0 * fraction:.1f}% full.",
                        "Stop before OOM; reduce worlds/batch size after preserving the run.",
                    )
                )
            elif fraction >= 0.90:
                add(
                    _alert(
                        "gpu_memory_high",
                        "warning",
                        f"GPU memory is {100.0 * fraction:.1f}% full.",
                        "Watch peak reserved memory and OOM messages closely.",
                    )
                )
        temperature = device.get("temperature_c")
        if isinstance(temperature, (int, float)) and temperature >= 85:
            add(
                _alert(
                    "gpu_temperature_high",
                    "warning",
                    f"GPU temperature is {temperature:.0f}°C.",
                    "Inspect cooling/throttling and GPU clocks.",
                )
            )
    if (
        system
        and state in ACTIVE_STATES
        and system.get("gpu", {}).get("available") is False
    ):
        add(
            _alert(
                "gpu_telemetry_unavailable",
                "warning",
                "The job-side monitor cannot read nvidia-smi telemetry.",
                "Inspect the sidecar error and confirm nvidia-smi remains available.",
            )
        )

    disk = system.get("node", {}).get("run_disk", {})
    free_bytes = disk.get("free_bytes")
    if isinstance(free_bytes, (int, float)):
        if free_bytes < 5 * 1024**3:
            add(
                _alert(
                    "disk_space_critical",
                    "critical",
                    f"Only {_format_bytes(free_bytes)} remains on the run filesystem.",
                    "Gracefully stop now and free/extend storage before resuming.",
                )
            )
        elif free_bytes < 20 * 1024**3:
            add(
                _alert(
                    "disk_space_low",
                    "warning",
                    f"Only {_format_bytes(free_bytes)} remains on the run filesystem.",
                    "Ensure one additional full atomic checkpoint will fit.",
                )
            )

    metrics = app.get("metrics", {})
    tqc_metrics = metrics.get("tqc", {})
    rollout_metrics = metrics.get("rollout", {})
    if isinstance(rollout_metrics, Mapping):
        constraint_violation = rollout_metrics.get(
            "contact_constraint_violation_fraction"
        )
        if (
            isinstance(constraint_violation, (int, float))
            and constraint_violation > 0.0
        ):
            add(
                _alert(
                    "contact_constraint_violation",
                    "warning",
                    f"{100.0 * constraint_violation:.2f}% of live worlds "
                    "have violated the continuous-contact contract.",
                    "Inspect the capture/thread gate and terminal failure "
                    "records before trusting this rollout.",
                )
            )
        mouth_radius_min = rollout_metrics.get(
            "socket_mouth_radius_min_m"
        )
        mouth_radius_max = rollout_metrics.get(
            "socket_mouth_radius_max_m"
        )
        if (
            isinstance(mouth_radius_min, (int, float))
            and isinstance(mouth_radius_max, (int, float))
            and (
                abs(float(mouth_radius_min) - 0.020) > 0.0002
                or abs(float(mouth_radius_max) - 0.020) > 0.0002
            )
        ):
            add(
                _alert(
                    "socket_mouth_orbit_invalid",
                    "critical",
                    "A live world has a socket mouth outside the configured "
                    "20 mm orbit.",
                    "Gracefully stop and inspect reset geometry before using "
                    "the generated data or checkpoint.",
                )
            )
    if (
        app.get("stage", {}).get("name") == "residual_tqc"
        and isinstance(tqc_metrics, Mapping)
        and tqc_metrics.get("mode") == "learning"
        and isinstance(rollout_metrics, Mapping)
    ):
        saturation = rollout_metrics.get("action_saturation_fraction")
        if isinstance(saturation, (int, float)) and saturation >= 0.20:
            add(
                _alert(
                    "residual_action_saturation_high",
                    "warning",
                    f"{100.0 * saturation:.1f}% of residual action components are saturated.",
                    "Inspect actor loss/alpha and task-space errors before continuing.",
                )
            )
        ik_feasible = rollout_metrics.get("ik_feasible_fraction")
        if isinstance(ik_feasible, (int, float)) and ik_feasible < 0.80:
            add(
                _alert(
                    "ik_feasible_fraction_low",
                    "warning",
                    f"Only {100.0 * ik_feasible:.1f}% of candidate IK commands are feasible.",
                    "Inspect residual scale, phase distribution, and IK fallback counters.",
                )
            )
    episodes = metrics.get("episodes", {})
    recent = episodes.get("recent", {}) if isinstance(episodes, Mapping) else {}
    recent_count = int(recent.get("window", 0) or 0)
    if app.get("stage", {}).get("name") == "residual_tqc" and recent_count >= 512:
        success_rate = recent.get("success_rate")
        failure_rate = recent.get("failure_rate")
        truncation_rate = recent.get("truncation_rate")
        if success_rate == 0:
            add(
                _alert(
                    "no_recent_training_success",
                    "warning",
                    f"No success in the latest {recent_count} completed training episodes.",
                    "Inspect phase/depth/turn metrics; run deterministic evaluation before cancelling.",
                )
            )
        if isinstance(failure_rate, (int, float)) and failure_rate >= 0.50:
            add(
                _alert(
                    "recent_failure_rate_high",
                    "warning",
                    f"Recent physical failure rate is {100.0 * failure_rate:.1f}%.",
                    "Inspect failure reasons, IK counters, and backend health.",
                )
            )
        if (
            isinstance(truncation_rate, (int, float))
            and truncation_rate >= 0.50
        ):
            add(
                _alert(
                    "recent_truncation_rate_high",
                    "warning",
                    f"Recent truncation rate is {100.0 * truncation_rate:.1f}%.",
                    "Inspect task phase histogram and whether depth/turns are progressing.",
                )
            )

    alerts.sort(
        key=lambda item: SEVERITY_ORDER.get(
            str(item.get("severity", "warning")), 1
        ),
        reverse=True,
    )
    return alerts, app_age, system_age


def build_report(
    run_dir: Path,
    *,
    query_scheduler: bool,
    stale_seconds: Optional[float],
    history_count: int,
) -> tuple[Optional[dict[str, Any]], int]:
    status_dir = run_dir / "status"
    manifest = read_json(run_dir / "run_manifest.json") or {}
    app = read_json(status_dir / "live.json")
    system = read_json(status_dir / "system.json") or {}
    wrapper = read_json(status_dir / "job.json") or {}
    previous_app = None
    wrapper_state = str(wrapper.get("state", "unknown"))
    wrapper_allocation = wrapper.get("allocation", {})
    wrapper_job_id = wrapper_allocation.get("scheduler_job_id")
    app_job = app.get("job", {}) if app is not None else {}
    app_job_id = app_job.get("scheduler_job_id")
    if (
        wrapper_allocation.get("identity_source")
        == "SCREW_ALLOCATION_ID"
    ):
        wrapper_identity = wrapper_allocation.get("id")
        app_identity = app_job.get("allocation_id")
    else:
        wrapper_identity = wrapper_job_id
        app_identity = app_job_id
    if (
        app is not None
        and wrapper_identity
        and wrapper_identity != app_identity
    ):
        # A newly resubmitted allocation has not entered pipeline.py yet.
        # Never present the previous allocation's terminal app snapshot as if
        # it belonged to the currently running PBS job.
        previous_app = app
        app = None
    if app is None:
        evidence = {}
        for label, path in (
            ("demo_metadata", run_dir / "datasets" / "demos_gpu" / "metadata.json"),
            ("diffusion", run_dir / "checkpoints" / "diffusion.pt"),
            ("tqc", run_dir / "checkpoints" / "residual_tqc.pt"),
            (
                "tqc_policy",
                run_dir / "checkpoints" / "residual_tqc_policy.pt",
            ),
        ):
            if path.is_file():
                stat = path.stat()
                evidence[label] = {
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "modified_utc": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat(timespec="seconds"),
                }
        active_wrapper = wrapper_state in {"starting", "preflight", "running"}
        failed_wrapper = wrapper_state == "failed"
        wrapper_scheduler = (
            scheduler_status(
                {
                    "job": {
                        "scheduler": wrapper_allocation.get("scheduler"),
                        "scheduler_job_id": wrapper_allocation.get(
                            "scheduler_job_id"
                        ),
                    }
                }
            )
            if query_scheduler
            else {"available": False, "reason": "disabled"}
        )
        return (
            {
                "run_dir": str(run_dir),
                "state": (
                    "awaiting_application"
                    if active_wrapper
                    else "wrapper_failed"
                    if failed_wrapper
                    else "no_live_status"
                ),
                "message": (
                    "The scheduled job is in setup/preflight; the application "
                    "heartbeat begins when pipeline.py starts."
                    if active_wrapper
                    else "The scheduler wrapper failed before pipeline.py started."
                    if failed_wrapper
                    else "No application heartbeat exists yet. The job may be "
                    "queued or from a version before live monitoring."
                ),
                "system": system,
                "job_status": wrapper,
                "scheduler": wrapper_scheduler,
                "previous_app": previous_app,
                "existing_artifacts": evidence,
                "manifest": manifest,
            },
            0 if active_wrapper else 2 if failed_wrapper else 3,
        )
    scheduler = scheduler_status(app) if query_scheduler else {
        "available": False,
        "reason": "disabled",
    }
    alerts, app_age, system_age = derive_alerts(
        app,
        system,
        stale_seconds=stale_seconds,
        job_status=wrapper,
    )
    app_manifest = app.get("manifest", {})
    if not isinstance(app_manifest, Mapping):
        app_manifest = {}
    expected_manifest_sha = app_manifest.get("sha256")
    manifest_path = run_dir / "run_manifest.json"
    if isinstance(expected_manifest_sha, str) and expected_manifest_sha:
        try:
            actual_manifest_sha = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
        except OSError:
            actual_manifest_sha = None
        if actual_manifest_sha != expected_manifest_sha:
            alerts.append(
                _alert(
                    "run_manifest_changed",
                    "critical",
                    "run_manifest.json is missing or differs from the active run.",
                    "Stop the job and restore the original run directory; do not resume with mixed provenance.",
                )
            )
            alerts.sort(
                key=lambda item: SEVERITY_ORDER.get(
                    str(item.get("severity", "warning")), 1
                ),
                reverse=True,
            )
    severity = max(
        (
            SEVERITY_ORDER.get(str(item.get("severity", "warning")), 1)
            for item in alerts
        ),
        default=0,
    )
    report = {
        "run_dir": str(run_dir),
        "app": app,
        "system": system,
        "job_status": wrapper,
        "scheduler": scheduler,
        "manifest": manifest,
        "derived": {
            "app_heartbeat_age_seconds": app_age,
            "system_heartbeat_age_seconds": system_age,
            "alerts": alerts,
            "severity": (
                "critical" if severity >= 2 else "warning" if severity == 1 else "ok"
            ),
        },
        "history": tail_jsonl(status_dir / "history.jsonl", history_count),
    }
    return report, 2 if severity >= 2 else 1 if severity == 1 else 0


def _line(label: str, value: str, *, indent: int = 0) -> str:
    return f"{' ' * indent}{label:<18} {value}"


def _phase_text(histogram: object) -> str:
    if not isinstance(histogram, Mapping):
        return "N/A"
    active = [
        f"{key}:{int(value)}"
        for key, value in histogram.items()
        if int(value or 0) > 0
    ]
    return " ".join(active) if active else "N/A"


def render_report(
    report: Mapping[str, Any],
    *,
    alerts_only: bool = False,
) -> str:
    if report.get("state") in {
        "no_live_status",
        "awaiting_application",
        "wrapper_failed",
    }:
        wrapper = report.get("job_status", {})
        title = (
            "等待 pipeline 心跳"
            if report.get("state") == "awaiting_application"
            else "调度器包装脚本失败"
            if report.get("state") == "wrapper_failed"
            else "尚无训练心跳"
        )
        lines = [
            f"SCREW HPC 实时状态：{title}",
            _line("运行目录", str(report.get("run_dir"))),
            _line("说明", str(report.get("message"))),
        ]
        if wrapper:
            allocation = wrapper.get("allocation", {})
            lines.extend(
                [
                    _line(
                        "作业阶段",
                        f"{wrapper.get('state', 'N/A')}  "
                        f"{wrapper.get('message', '')} "
                        f"(状态更新于 "
                        f"{_format_duration(heartbeat_age_seconds(wrapper))} 前)",
                    ),
                    _line(
                        "作业",
                        f"{allocation.get('scheduler', 'N/A')} "
                        f"{allocation.get('scheduler_job_id') or 'N/A'} @ "
                        f"{allocation.get('hostname', 'N/A')}",
                    ),
                ]
            )
            job_id = allocation.get("scheduler_job_id")
            if job_id:
                lines.extend(
                    [
                        "pipeline 尚未启动，当前只能通过调度器取消：",
                        (
                            f"  qdel {job_id}"
                            if allocation.get("scheduler") == "pbs"
                            else f"  scancel {job_id}"
                        ),
                    ]
                )
            scheduler = report.get("scheduler", {})
            if scheduler.get("available"):
                fields = scheduler.get("fields", {})
                lines.append(
                    _line(
                        "调度器状态",
                        f"{fields.get('job_state', fields.get('state', 'N/A'))} "
                        f"walltime={fields.get('resources_used.walltime', fields.get('elapsed', 'N/A'))}/"
                        f"{fields.get('Resource_List.walltime', fields.get('time_limit', 'N/A'))}",
                    )
                )
        system = report.get("system", {})
        if system:
            lines.append(
                _line(
                    "系统心跳",
                    f"{_format_duration(heartbeat_age_seconds(system))} 前",
                )
            )
            selection = system.get("gpu", {}).get("selection", {})
            if selection:
                lines.append(
                    _line(
                        "GPU 分配",
                        f"mode={selection.get('mode', 'N/A')} "
                        f"{selection.get('source') or 'source'}="
                        f"{selection.get('raw') or 'N/A'} "
                        f"matched={selection.get('matched_devices', 'N/A')}",
                    )
                )
            for device in system.get("gpu", {}).get("devices", []) or []:
                lines.append(
                    _line(
                        f"GPU {device.get('index', '?')}",
                        f"{device.get('name', 'N/A')} util="
                        f"{_format_percent(device.get('gpu_utilization_percent'), fraction=False)} "
                        f"VRAM={_format_number(device.get('memory_used_mib'), 0)}/"
                        f"{_format_number(device.get('memory_total_mib'), 0)}MiB "
                        f"temp={_format_number(device.get('temperature_c'), 0)}°C",
                    )
                )
            disk = system.get("node", {}).get("run_disk", {})
            if disk:
                lines.append(
                    _line(
                        "磁盘剩余",
                        f"{_format_bytes(disk.get('free_bytes'))} "
                        f"({_format_percent(disk.get('free_percent'), fraction=False)})",
                    )
                )
        artifacts = report.get("existing_artifacts", {})
        manifest = report.get("manifest", {})
        if isinstance(manifest, Mapping) and manifest:
            manifest_config = manifest.get("config", {})
            manifest_contracts = manifest.get("contracts", {})
            lines.extend(
                [
                    _line(
                        "运行清单",
                        f"config={manifest_config.get('name', 'N/A')} "
                        f"sha={str(manifest_config.get('sha256', 'N/A'))[:12]} "
                        f"source={str(manifest_contracts.get('source_tree_sha256', 'N/A'))[:12]}",
                    ),
                    _line(
                        "随机种子",
                        json.dumps(
                            manifest.get("seeds", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                ]
            )
        previous_app = report.get("previous_app")
        if previous_app:
            lines.append(
                _line(
                    "上一 allocation",
                    f"state={previous_app.get('state', 'N/A')} "
                    f"updated={previous_app.get('updated_utc', 'N/A')}",
                )
            )
        if artifacts:
            lines.append("已发现文件：")
            for name, item in artifacts.items():
                lines.append(
                    _line(
                        str(name),
                        f"{_format_bytes(item.get('size_bytes'))}  {item.get('path')}",
                        indent=2,
                    )
                )
        return "\n".join(lines)

    app = report.get("app", {})
    system = report.get("system", {})
    derived = report.get("derived", {})
    alerts = derived.get("alerts", [])
    severity = str(derived.get("severity", "ok")).upper()
    if alerts_only:
        if not alerts:
            return "当前没有 warning/critical 告警。"
        return "\n".join(
            f"[{str(item.get('severity')).upper()}] {item.get('code')}: "
            f"{item.get('message')} 建议：{item.get('recommended_action')}"
            for item in alerts
        )

    stage = app.get("stage", {})
    progress = stage.get("progress", {})
    pipeline = app.get("pipeline", {})
    job = app.get("job", {})
    stop_status = app.get("stop", {})
    scheduler = report.get("scheduler", {})
    scheduler_fields = scheduler.get("fields", {})
    wrapper = report.get("job_status", {})
    metrics = app.get("metrics", {})
    manifest = report.get("manifest", {})
    app_manifest = app.get("manifest", {})
    if not isinstance(app_manifest, Mapping):
        app_manifest = {}
    lines = [
        f"SCREW HPC 实时状态  [{str(app.get('state', 'unknown')).upper()} / {severity}]",
        _line("运行目录", str(report.get("run_dir"))),
        _line(
            "更新时间",
            f"{app.get('updated_utc', 'N/A')}  "
            f"(距今 {_format_duration(derived.get('app_heartbeat_age_seconds'))})",
        ),
        _line(
            "作业",
            f"{job.get('scheduler', 'local')} "
            f"{job.get('scheduler_job_id') or 'N/A'} @ {job.get('hostname', 'N/A')} "
            f"PID={job.get('pid', 'N/A')}",
        ),
    ]
    if isinstance(manifest, Mapping) and manifest:
        manifest_config = manifest.get("config", {})
        manifest_contracts = manifest.get("contracts", {})
        lines.extend(
            [
                _line(
                    "运行清单",
                    f"config_sha={str(manifest_config.get('sha256', 'N/A'))[:12]} "
                    f"manifest_sha={str(app_manifest.get('sha256', 'N/A'))[:12]} "
                    f"source_sha={str(manifest_contracts.get('source_tree_sha256', 'N/A'))[:12]}",
                ),
                _line(
                    "随机种子",
                    json.dumps(
                        manifest.get("seeds", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            ]
        )
    if scheduler.get("available"):
        scheduler_state = scheduler_fields.get(
            "job_state", scheduler_fields.get("state", "N/A")
        )
        walltime = scheduler_fields.get(
            "resources_used.walltime",
            scheduler_fields.get("elapsed", "N/A"),
        )
        limit = scheduler_fields.get(
            "Resource_List.walltime",
            scheduler_fields.get("time_limit", "N/A"),
        )
        lines.append(
            _line(
                "调度器状态",
                f"{scheduler_state}  walltime={walltime}/{limit}",
            )
        )
    if wrapper:
        lines.append(
            _line(
                "包装脚本",
                f"{wrapper.get('state', 'N/A')} "
                f"raw_exit={wrapper.get('raw_pipeline_exit_code')} "
                f"{wrapper.get('message', '')}",
            )
        )
    if stop_status.get("requested") or stop_status.get("acknowledged"):
        lines.append(
            _line(
                "停止请求",
                f"requested={stop_status.get('requested')} "
                f"acknowledged={stop_status.get('acknowledged')}",
            )
        )
    stage_index = pipeline.get("stage_index")
    stage_count = pipeline.get("stage_count")
    lines.extend(
        [
            _line(
                "当前阶段",
                f"{stage.get('name', 'N/A')} "
                f"({stage_index or '?'}/{stage_count or '?'})  "
                f"{stage.get('message', '')}",
            ),
            _line(
                "阶段进度",
                f"{_format_number(progress.get('current'))}/"
                f"{_format_number(progress.get('total'))} "
                f"{progress.get('unit', '')}  "
                f"{_format_percent(progress.get('fraction'))}",
            ),
            _line(
                "速度 / ETA",
                f"{_format_rate(progress.get('rate_per_second'), str(progress.get('unit', 'unit')))}"
                f"  /  {_format_duration(progress.get('eta_seconds'))}"
                f"  (本阶段已运行 {_format_duration(progress.get('elapsed_seconds'))})",
            ),
        ]
    )

    episodes = metrics.get("episodes", {})
    if isinstance(episodes, Mapping) and episodes:
        recent = episodes.get("recent", {})
        episode_scope = str(
            episodes.get("scope", "stochastic_training_rollouts")
        )
        scope_label = {
            "stochastic_training_rollouts": "随机训练策略 rollout",
            "deterministic_evaluation": "确定性正式评估",
            "expert_demonstrations": "专家演示",
            "expert_demonstrations_this_session": "本次 allocation 的专家演示",
            "stochastic_resume_probe": "随机续训探针",
        }.get(episode_scope, episode_scope)
        lines.extend(
            [
                "",
                f"Episode 统计（{scope_label}）：",
                _line(
                    "累计",
                    f"完成={_format_number(episodes.get('completed'))} "
                    f"成功={_format_number(episodes.get('successes'))} "
                    f"失败={_format_number(episodes.get('failures'))} "
                    f"截断={_format_number(episodes.get('truncations'))} "
                    f"成功率={_format_percent(episodes.get('success_rate'))}",
                    indent=2,
                ),
                _line(
                    "最近窗口",
                    f"N={_format_number(recent.get('window'))}/"
                    f"{_format_number(episodes.get('recent_capacity'))} "
                    f"成功率={_format_percent(recent.get('success_rate'))} "
                    f"失败率={_format_percent(recent.get('failure_rate'))} "
                    f"截断率={_format_percent(recent.get('truncation_rate'))}",
                    indent=2,
                ),
                _line(
                    "最近均值",
                    f"return={_format_number(recent.get('mean_return'))} "
                    f"depth={_format_number(recent.get('mean_depth_m') * 1000 if recent.get('mean_depth_m') is not None else None)}mm "
                    f"turns={_format_number(recent.get('mean_turns'))} "
                    f"steps={_format_number(recent.get('mean_episode_steps'))}",
                    indent=2,
                ),
                _line(
                    "最近终态",
                    f"lateral={_format_number(recent.get('mean_lateral_error_m') * 1000 if recent.get('mean_lateral_error_m') is not None else None)}mm "
                    f"axis={_format_number(math.degrees(float(recent.get('mean_axis_error_rad'))) if recent.get('mean_axis_error_rad') is not None else None)}deg "
                    f"speed={_format_number(recent.get('mean_speed'))} "
                    f"IK fail/fallback/stop="
                    f"{_format_number(recent.get('mean_ik_failures'))}/"
                    f"{_format_number(recent.get('mean_ik_anchor_fallbacks'))}/"
                    f"{_format_number(recent.get('mean_ik_safety_stops'))}",
                    indent=2,
                ),
                _line(
                    "失败原因",
                    json.dumps(
                        recent.get("failure_reasons", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    indent=2,
                ),
            ]
        )
        windows = episodes.get("windows", {})
        if isinstance(windows, Mapping):
            for window_name in ("128", "512", "2048"):
                window = windows.get(window_name)
                if not isinstance(window, Mapping):
                    continue
                interval = window.get("success_rate_ci95") or {}
                lines.append(
                    _line(
                        f"窗口 {window_name}",
                        f"N={_format_number(window.get('window'))} "
                        f"success={_format_percent(window.get('success_rate'))} "
                        f"CI95=[{_format_percent(interval.get('lower'))}, "
                        f"{_format_percent(interval.get('upper'))}] "
                        f"return={_format_number(window.get('mean_return'))}"
                        f"±{_format_number(window.get('std_return'))} "
                        f"depth p50/p90="
                        f"{_format_number(window.get('p50_depth_m') * 1000 if window.get('p50_depth_m') is not None else None)}/"
                        f"{_format_number(window.get('p90_depth_m') * 1000 if window.get('p90_depth_m') is not None else None)}mm",
                        indent=2,
                    )
                )
        curriculum_groups = episodes.get("curriculum", {})
        if isinstance(curriculum_groups, Mapping):
            text = " ".join(
                f"{name}:N={_format_number(group.get('completed'))},"
                f"S={_format_percent(group.get('success_rate'))}"
                for name in ("warmup", "ramp", "full")
                for group in [curriculum_groups.get(name, {})]
                if isinstance(group, Mapping)
            )
            if text:
                lines.append(_line("课程分段", text, indent=2))
        terminal_phases = episodes.get("terminal_phases", {})
        if isinstance(terminal_phases, Mapping) and terminal_phases:
            most_common = sorted(
                terminal_phases.items(),
                key=lambda item: int(
                    item[1].get("completed", 0)
                    if isinstance(item[1], Mapping)
                    else 0
                ),
                reverse=True,
            )[:6]
            lines.append(
                _line(
                    "终止相位",
                    " ".join(
                        f"{name}:N={_format_number(group.get('completed'))},"
                        f"S={_format_percent(group.get('success_rate'))}"
                        for name, group in most_common
                        if isinstance(group, Mapping)
                    ),
                    indent=2,
                )
            )

    diffusion = metrics.get("diffusion", {})
    demos = metrics.get("demos", {})
    if isinstance(demos, Mapping) and demos:
        committed = demos.get("committed_episodes")
        target_episodes = demos.get("target_episodes")
        demo_fraction = (
            float(committed) / float(target_episodes)
            if isinstance(committed, (int, float))
            and isinstance(target_episodes, (int, float))
            and target_episodes
            else None
        )
        lines.extend(
            [
                "",
                "专家演示数据：",
                _line(
                    "已提交",
                    f"{_format_number(committed)}/"
                    f"{_format_number(target_episodes)} episodes "
                    f"({_format_percent(demo_fraction)})",
                    indent=2,
                ),
                _line(
                    "当前 batch",
                    f"start={_format_number(demos.get('inflight_batch_start'))} "
                    f"worlds={_format_number(demos.get('inflight_batch_episodes', demos.get('active_worlds')))} "
                    f"step={_format_number(demos.get('vector_step'))}/"
                    f"{_format_number(demos.get('steps_per_episode'))} "
                    f"committed={demos.get('inflight_is_committed', False)}",
                    indent=2,
                ),
            ]
        )
    if isinstance(diffusion, Mapping) and diffusion:
        lines.extend(
            [
                "",
                "Diffusion：",
                _line(
                    "Epoch / batch",
                    f"{diffusion.get('epoch', 'N/A')}/"
                    f"{diffusion.get('epochs', 'N/A')}  "
                    f"batch={diffusion.get('batch_in_epoch', 'N/A')}/"
                    f"{diffusion.get('batches_per_epoch', 'N/A')} "
                    f"global_step={diffusion.get('global_step', 'N/A')}",
                    indent=2,
                ),
                _line(
                    "Loss",
                    f"current={_format_number(diffusion.get('loss'), 6)} "
                    f"epoch_mean={_format_number(diffusion.get('epoch_mean_loss'), 6)} "
                    f"best={_format_number(diffusion.get('best_loss'), 6)}",
                    indent=2,
                ),
                _line(
                    "优化",
                    f"lr={_format_number(diffusion.get('learning_rate'), 8)} "
                    f"grad_norm={_format_number(diffusion.get('gradient_norm'), 5)}",
                    indent=2,
                ),
            ]
        )

    tqc = metrics.get("tqc", {})
    if isinstance(tqc, Mapping) and tqc:
        lines.extend(
            [
                "",
                "TQC residual：",
                _line(
                    "训练游标",
                    f"transitions={_format_number(tqc.get('transitions'))}/"
                    f"{_format_number(tqc.get('total_transitions'))} "
                    f"updates={_format_number(tqc.get('updates'))} "
                    f"mode={tqc.get('mode', 'N/A')}",
                    indent=2,
                ),
                _line(
                    "Replay",
                    f"{_format_number(tqc.get('replay_size'))}/"
                    f"{_format_number(tqc.get('replay_capacity'))} "
                    f"({_format_percent(tqc.get('replay_fill_fraction'))}) "
                    f"credit={_format_number(tqc.get('update_credit'))}",
                    indent=2,
                ),
                _line(
                    "Loss",
                    f"critic={_format_number(tqc.get('critic_loss'), 6)} "
                    f"actor={_format_number(tqc.get('actor_loss'), 6)} "
                    f"alpha_loss={_format_number(tqc.get('alpha_loss'), 6)} "
                    f"alpha={_format_number(tqc.get('alpha'), 6)}",
                    indent=2,
                ),
                _line(
                    "梯度",
                    f"critic={_format_number(tqc.get('critic_grad_norm'), 5)} "
                    f"actor={_format_number(tqc.get('actor_grad_norm'), 5)}",
                    indent=2,
                ),
                _line(
                    "Q / target",
                    f"Q={_format_number(tqc.get('q_pred_mean'), 5)}"
                    f"±{_format_number(tqc.get('q_pred_std'), 5)} "
                    f"target={_format_number(tqc.get('q_target_mean'), 5)}"
                    f"±{_format_number(tqc.get('q_target_std'), 5)} "
                    f"TD mean/p95={_format_number(tqc.get('td_abs_mean'), 5)}/"
                    f"{_format_number(tqc.get('td_abs_p95'), 5)}",
                    indent=2,
                ),
                _line(
                    "熵 / LR",
                    f"entropy={_format_number(tqc.get('policy_entropy'), 5)} "
                    f"target={_format_number(tqc.get('target_entropy'), 5)} "
                    f"lr actor/critic/alpha="
                    f"{_format_number(tqc.get('actor_learning_rate'), 8)}/"
                    f"{_format_number(tqc.get('critic_learning_rate'), 8)}/"
                    f"{_format_number(tqc.get('alpha_learning_rate'), 8)}",
                    indent=2,
                ),
                _line(
                    "滚动 EMA",
                    f"updates={_format_number(tqc.get('metric_ema_updates'))} "
                    f"Q={_format_number(tqc.get('q_pred_mean_ema'), 5)} "
                    f"target={_format_number(tqc.get('q_target_mean_ema'), 5)} "
                    f"TD={_format_number(tqc.get('td_abs_mean_ema'), 5)} "
                    f"entropy={_format_number(tqc.get('policy_entropy_ema'), 5)}",
                    indent=2,
                ),
            ]
        )

    rollout = metrics.get("rollout", {})
    curriculum = metrics.get("curriculum", {})
    if isinstance(curriculum, Mapping) and curriculum:
        lines.extend(
            [
                "",
                "课程难度：",
                _line(
                    "进度 / 刷新半径",
                    f"{_format_percent(curriculum.get('progress'))} / "
                    f"{_format_number(curriculum.get('spawn_radius_m') * 1000 if curriculum.get('spawn_radius_m') is not None else None)}mm",
                    indent=2,
                ),
                _line(
                    "孔洞固定角度",
                    f"{_format_number(curriculum.get('fixed_socket_angle_deg'))}deg "
                    "(孔口位于 20mm 圆周且始终朝向中心)",
                    indent=2,
                ),
                _line(
                    "螺丝刷新区域",
                    str(curriculum.get("spawn_region", "N/A")),
                    indent=2,
                ),
            ]
        )
    if isinstance(rollout, Mapping) and rollout:
        lateral_mean = rollout.get(
            "lateral_mean_m", rollout.get("lateral_error_mean_m")
        )
        lateral_max = rollout.get(
            "lateral_max_m", rollout.get("lateral_error_max_m")
        )
        axis_mean_deg = rollout.get("axis_mean_deg")
        if axis_mean_deg is None and rollout.get("axis_error_mean_rad") is not None:
            axis_mean_deg = math.degrees(
                float(rollout["axis_error_mean_rad"])
            )
        axis_max_deg = rollout.get("axis_max_deg")
        if axis_max_deg is None and rollout.get("axis_error_max_rad") is not None:
            axis_max_deg = math.degrees(
                float(rollout["axis_error_max_rad"])
            )
        lines.extend(
            [
                "",
                "当前并行 world：",
                _line(
                    "Reward / action",
                    f"reward mean={_format_number(rollout.get('reward_mean'))} "
                    f"std={_format_number(rollout.get('reward_std'))} "
                    f"min/max={_format_number(rollout.get('reward_min'))}/"
                    f"{_format_number(rollout.get('reward_max'))} "
                    f"action |mean|={_format_number(rollout.get('action_abs_mean'))} "
                    f"|max|={_format_number(rollout.get('action_abs_max'))} "
                    f"sat={_format_percent(rollout.get('action_saturation_fraction'))}",
                    indent=2,
                ),
                _line(
                    "任务几何",
                    f"depth mean/max={_format_number(rollout.get('depth_mean_m') * 1000 if rollout.get('depth_mean_m') is not None else None)}/"
                    f"{_format_number(rollout.get('depth_max_m') * 1000 if rollout.get('depth_max_m') is not None else None)}mm "
                    f"turns mean/max={_format_number(rollout.get('turns_mean'))}/"
                    f"{_format_number(rollout.get('turns_max'))} "
                    f"socket angle observed min/mean/max="
                    f"{_format_number(rollout.get('socket_angle_min_deg'))}/"
                    f"{_format_number(rollout.get('socket_angle_mean_deg'))}/"
                    f"{_format_number(rollout.get('socket_angle_max_deg'))}deg",
                    indent=2,
                ),
                _line(
                    "控制交接",
                    f"scripted={_format_percent(rollout.get('scripted_prefix_fraction'))} "
                    f"ready={_format_percent(rollout.get('diffusion_handoff_ready_fraction'))} "
                    f"Diffusion active={_format_percent(rollout.get('diffusion_active_fraction'))} "
                    f"pos err mean/max="
                    f"{_format_number(rollout.get('diffusion_handoff_position_error_mean_m') * 1000 if rollout.get('diffusion_handoff_position_error_mean_m') is not None else None)}/"
                    f"{_format_number(rollout.get('diffusion_handoff_position_error_max_m') * 1000 if rollout.get('diffusion_handoff_position_error_max_m') is not None else None)}mm "
                    f"rot err mean/max="
                    f"{_format_number(rollout.get('diffusion_handoff_orientation_error_mean_deg'))}/"
                    f"{_format_number(rollout.get('diffusion_handoff_orientation_error_max_deg'))}deg "
                    f"hold steps mean/max="
                    f"{_format_number(rollout.get('diffusion_handoff_hold_steps_mean'))}/"
                    f"{_format_number(rollout.get('diffusion_handoff_hold_steps_max'))}",
                    indent=2,
                ),
                _line(
                    "5mm预插入门",
                    f"ready={_format_percent(rollout.get('preinsert_alignment_ready_fraction'))} "
                    f"holding={_format_percent(rollout.get('preinsert_alignment_hold_fraction'))} "
                    f"stable frames mean/max="
                    f"{_format_number(rollout.get('preinsert_alignment_stable_steps_mean'))}/"
                    f"{_format_number(rollout.get('preinsert_alignment_stable_steps_max'))} "
                    f"lateral mean/max="
                    f"{_format_number(rollout.get('preinsert_lateral_error_mean_m') * 1000 if rollout.get('preinsert_lateral_error_mean_m') is not None else None)}/"
                    f"{_format_number(rollout.get('preinsert_lateral_error_max_m') * 1000 if rollout.get('preinsert_lateral_error_max_m') is not None else None)}mm "
                    f"axis mean/max="
                    f"{_format_number(rollout.get('preinsert_axis_error_mean_deg'))}/"
                    f"{_format_number(rollout.get('preinsert_axis_error_max_deg'))}deg "
                    f"depth err mean/max="
                    f"{_format_number(rollout.get('preinsert_depth_error_mean_m') * 1000 if rollout.get('preinsert_depth_error_mean_m') is not None else None)}/"
                    f"{_format_number(rollout.get('preinsert_depth_error_max_m') * 1000 if rollout.get('preinsert_depth_error_max_m') is not None else None)}mm",
                    indent=2,
                ),
                _line(
                    "预插入动态",
                    f"support={_format_percent(rollout.get('stable_grasp_support_fraction'))} "
                    f"tip speed mean/max="
                    f"{_format_number(rollout.get('preinsert_tip_speed_mean_m_s') * 1000 if rollout.get('preinsert_tip_speed_mean_m_s') is not None else None)}/"
                    f"{_format_number(rollout.get('preinsert_tip_speed_max_m_s') * 1000 if rollout.get('preinsert_tip_speed_max_m_s') is not None else None)}mm/s "
                    f"lateral speed mean/max="
                    f"{_format_number(rollout.get('preinsert_lateral_speed_mean_m_s') * 1000 if rollout.get('preinsert_lateral_speed_mean_m_s') is not None else None)}/"
                    f"{_format_number(rollout.get('preinsert_lateral_speed_max_m_s') * 1000 if rollout.get('preinsert_lateral_speed_max_m_s') is not None else None)}mm/s "
                    f"omega mean/max="
                    f"{_format_number(rollout.get('preinsert_angular_speed_mean_deg_s'))}/"
                    f"{_format_number(rollout.get('preinsert_angular_speed_max_deg_s'))}deg/s "
                    f"feedback XY mean/max="
                    f"{_format_number(rollout.get('preinsert_feedback_lateral_mean_m') * 1000 if rollout.get('preinsert_feedback_lateral_mean_m') is not None else None)}/"
                    f"{_format_number(rollout.get('preinsert_feedback_lateral_max_m') * 1000 if rollout.get('preinsert_feedback_lateral_max_m') is not None else None)}mm",
                    indent=2,
                ),
                _line(
                    "30mm捕获状态（当前值）",
                    f"captured={_format_percent(rollout.get('shaft_captured_fraction'))} "
                    f"ready-now={_format_percent(rollout.get('capture_alignment_ready_fraction'))} "
                    f"holding={_format_percent(rollout.get('capture_alignment_hold_fraction'))} "
                    f"four-contact={_format_percent(rollout.get('all_four_contact_fraction'))} "
                    f"secondary released={_format_percent(rollout.get('secondary_released_fraction'))} "
                    f"stable frames mean/max="
                    f"{_format_number(rollout.get('capture_alignment_stable_steps_mean'))}/"
                    f"{_format_number(rollout.get('capture_alignment_stable_steps_max'))} "
                    f"current head clearance min/mean="
                    f"{_format_number(rollout.get('capture_head_bottom_standoff_min_m') * 1000 if rollout.get('capture_head_bottom_standoff_min_m') is not None else None)}/"
                    f"{_format_number(rollout.get('capture_head_bottom_standoff_mean_m') * 1000 if rollout.get('capture_head_bottom_standoff_mean_m') is not None else None)}mm",
                    indent=2,
                ),
                _line(
                    "当前距30mm捕获目标",
                    f"lateral mean/max="
                    f"{_format_number(rollout.get('capture_lateral_error_mean_m') * 1000 if rollout.get('capture_lateral_error_mean_m') is not None else None)}/"
                    f"{_format_number(rollout.get('capture_lateral_error_max_m') * 1000 if rollout.get('capture_lateral_error_max_m') is not None else None)}mm "
                    f"axis mean/max="
                    f"{_format_number(rollout.get('capture_axis_error_mean_deg'))}/"
                    f"{_format_number(rollout.get('capture_axis_error_max_deg'))}deg "
                    f"depth mean/max="
                    f"{_format_number(rollout.get('capture_depth_error_mean_m') * 1000 if rollout.get('capture_depth_error_mean_m') is not None else None)}/"
                    f"{_format_number(rollout.get('capture_depth_error_max_m') * 1000 if rollout.get('capture_depth_error_max_m') is not None else None)}mm",
                    indent=2,
                ),
                _line(
                    "70mm螺纹状态（当前值）",
                    f"min-two-contact={_format_percent(rollout.get('two_finger_contact_fraction'))} "
                    f"L/R-diagnostic={_format_percent(rollout.get('primary_pair_contact_fraction'))} "
                    f"ready-now={_format_percent(rollout.get('thread_engagement_ready_fraction'))} "
                    f"holding={_format_percent(rollout.get('thread_engagement_hold_fraction'))} "
                    f"regrasp hold={_format_percent(rollout.get('thread_regrasp_hold_fraction'))} "
                    f"stable frames mean/max="
                    f"{_format_number(rollout.get('thread_ready_stable_steps_mean'))}/"
                    f"{_format_number(rollout.get('thread_ready_stable_steps_max'))} "
                    f"constraint violation="
                    f"{_format_percent(rollout.get('contact_constraint_violation_fraction'))} "
                    f"safety guard={_format_percent(rollout.get('safety_gate_guard_fraction'))}",
                    indent=2,
                ),
                _line(
                    "当前距70mm螺纹目标",
                    f"lateral mean/max="
                    f"{_format_number(rollout.get('thread_lateral_error_mean_m') * 1000 if rollout.get('thread_lateral_error_mean_m') is not None else None)}/"
                    f"{_format_number(rollout.get('thread_lateral_error_max_m') * 1000 if rollout.get('thread_lateral_error_max_m') is not None else None)}mm "
                    f"axis mean/max="
                    f"{_format_number(rollout.get('thread_axis_error_mean_deg'))}/"
                    f"{_format_number(rollout.get('thread_axis_error_max_deg'))}deg "
                    f"depth mean/max="
                    f"{_format_number(rollout.get('thread_depth_error_mean_m') * 1000 if rollout.get('thread_depth_error_mean_m') is not None else None)}/"
                    f"{_format_number(rollout.get('thread_depth_error_max_m') * 1000 if rollout.get('thread_depth_error_max_m') is not None else None)}mm",
                    indent=2,
                ),
                _line(
                    "随机孔口",
                    f"radius min/mean/max="
                    f"{_format_number(rollout.get('socket_mouth_radius_min_m') * 1000 if rollout.get('socket_mouth_radius_min_m') is not None else None)}/"
                    f"{_format_number(rollout.get('socket_mouth_radius_mean_m') * 1000 if rollout.get('socket_mouth_radius_mean_m') is not None else None)}/"
                    f"{_format_number(rollout.get('socket_mouth_radius_max_m') * 1000 if rollout.get('socket_mouth_radius_max_m') is not None else None)}mm "
                    f"x=[{_format_number(rollout.get('socket_mouth_x_min_m') * 1000 if rollout.get('socket_mouth_x_min_m') is not None else None)}, "
                    f"{_format_number(rollout.get('socket_mouth_x_max_m') * 1000 if rollout.get('socket_mouth_x_max_m') is not None else None)}]mm "
                    f"y=[{_format_number(rollout.get('socket_mouth_y_min_m') * 1000 if rollout.get('socket_mouth_y_min_m') is not None else None)}, "
                    f"{_format_number(rollout.get('socket_mouth_y_max_m') * 1000 if rollout.get('socket_mouth_y_max_m') is not None else None)}]mm",
                    indent=2,
                ),
                _line(
                    "误差 / 速度",
                    f"lateral mean/max={_format_number(lateral_mean * 1000 if lateral_mean is not None else None)}/"
                    f"{_format_number(lateral_max * 1000 if lateral_max is not None else None)}mm "
                    f"axis mean/max={_format_number(axis_mean_deg)}/"
                    f"{_format_number(axis_max_deg)}deg "
                    f"speed mean/max={_format_number(rollout.get('speed_mean'))}/"
                    f"{_format_number(rollout.get('speed_max'))}",
                    indent=2,
                ),
                _line(
                    "抓取 / 螺纹",
                    f"engaged={_format_percent(rollout.get('thread_engaged_fraction'))} "
                    f"holder={_format_percent(rollout.get('holder_active_fraction'))} "
                    f"grasp={_format_percent(rollout.get('grasp_latched_fraction'))} "
                    f"IK feasible={_format_percent(rollout.get('ik_feasible_fraction'))} "
                    f"active residual dims="
                    f"{_format_number(rollout.get('residual_active_dimensions_mean'))}",
                    indent=2,
                ),
                _line(
                    "任务相位",
                    _phase_text(rollout.get("phase_histogram")),
                    indent=2,
                ),
                _line(
                    "当前 episode",
                    f"steps min/mean/max="
                    f"{_format_number(rollout.get('current_episode_steps_min'))}/"
                    f"{_format_number(rollout.get('current_episode_steps_mean'))}/"
                    f"{_format_number(rollout.get('current_episode_steps_max'))} "
                    f"done_worlds={_format_number(rollout.get('done_worlds'))}",
                    indent=2,
                ),
            ]
        )

    health = app.get("health", {})
    if isinstance(health, Mapping) and health:
        contacts = health.get("contacts_total")
        contact_capacity = health.get("contact_capacity_total")
        contact_fraction = (
            float(contacts) / float(contact_capacity)
            if isinstance(contacts, (int, float))
            and isinstance(contact_capacity, (int, float))
            and contact_capacity
            else None
        )
        max_nefc = health.get("max_nefc")
        njmax = health.get("njmax", health.get("constraint_capacity"))
        constraint_fraction = (
            float(max_nefc) / float(njmax)
            if isinstance(max_nefc, (int, float))
            and isinstance(njmax, (int, float))
            and njmax
            else None
        )
        lines.extend(
            [
                "",
                "MJWarp 健康：",
                _line(
                    "有限 / overflow",
                    f"finite={health.get('finite', 'N/A')} "
                    f"overflow_worlds={health.get('overflow_worlds', 'N/A')} "
                    f"sticky_nonfinite={health.get('sticky_nonfinite', 'N/A')}",
                    indent=2,
                ),
                _line(
                    "容量",
                    f"contacts={_format_number(contacts)}/"
                    f"{_format_number(contact_capacity)} "
                    f"({_format_percent(contact_fraction)}) "
                    f"constraints={_format_number(max_nefc)}/"
                    f"{_format_number(njmax)} "
                    f"({_format_percent(constraint_fraction)})",
                    indent=2,
                ),
            ]
        )

    gpu_devices = system.get("gpu", {}).get("devices", [])
    gpu_selection = system.get("gpu", {}).get("selection", {})
    node = system.get("node", {})
    torch_cuda = app.get("process", {}).get("torch_cuda", {})
    if gpu_devices or node or torch_cuda:
        lines.extend(["", "节点 / GPU："])
        if gpu_selection:
            lines.append(
                _line(
                    "GPU 分配",
                    f"mode={gpu_selection.get('mode', 'N/A')} "
                    f"{gpu_selection.get('source') or 'source'}="
                    f"{gpu_selection.get('raw') or 'N/A'} "
                    f"matched={gpu_selection.get('matched_devices', 'N/A')}/"
                    f"{system.get('gpu', {}).get('physical_devices_seen', 'N/A')}",
                    indent=2,
                )
            )
        for device in gpu_devices or []:
            used = device.get("memory_used_mib")
            total = device.get("memory_total_mib")
            memory_fraction = (
                float(used) / float(total)
                if isinstance(used, (int, float))
                and isinstance(total, (int, float))
                and total
                else None
            )
            lines.append(
                _line(
                    f"GPU {device.get('index', '?')}",
                    f"{device.get('name', 'N/A')} util="
                    f"{_format_percent(device.get('gpu_utilization_percent'), fraction=False)} "
                    f"VRAM={_format_number(used, 0)}/{_format_number(total, 0)}MiB "
                    f"({_format_percent(memory_fraction)}) "
                    f"temp={_format_number(device.get('temperature_c'), 0)}°C "
                    f"power={_format_number(device.get('power_draw_w'), 0)}/"
                    f"{_format_number(device.get('power_limit_w'), 0)}W "
                    f"clock={_format_number(device.get('sm_clock_mhz'), 0)}MHz",
                    indent=2,
                )
            )
        if torch_cuda:
            lines.append(
                _line(
                    "Torch CUDA",
                    f"allocated={_format_bytes(torch_cuda.get('allocated_bytes'))} "
                    f"reserved={_format_bytes(torch_cuda.get('reserved_bytes'))} "
                    f"peak_alloc={_format_bytes(torch_cuda.get('peak_allocated_bytes'))} "
                    f"peak_reserved={_format_bytes(torch_cuda.get('peak_reserved_bytes'))}",
                    indent=2,
                )
            )
        process = system.get("application", {}).get("process", {})
        process_io = process.get("io", {})
        lines.append(
            _line(
                "训练进程",
                f"alive={process.get('alive', 'N/A')} "
                f"RSS={process.get('rss', 'N/A')} "
                f"peak={process.get('peak_rss', 'N/A')} "
                f"CPU={_format_percent(process.get('cpu_percent_all_cores'), fraction=False)} "
                f"threads={process.get('threads', 'N/A')}",
                indent=2,
            )
        )
        lines.append(
            _line(
                "进程 I/O",
                f"read={_format_bytes(process_io.get('read_bytes'))} "
                f"write={_format_bytes(process_io.get('write_bytes'))} "
                f"fds={process.get('open_fds', 'N/A')}",
                indent=2,
            )
        )
        disk = node.get("run_disk", {})
        memory = node.get("memory", {})
        lines.append(
            _line(
                "节点 / 磁盘",
                f"load={node.get('load_average', {}).get('1m', 'N/A')}/"
                f"{node.get('load_average', {}).get('5m', 'N/A')}/"
                f"{node.get('load_average', {}).get('15m', 'N/A')} "
                f"disk_free={_format_bytes(disk.get('free_bytes'))} "
                f"({_format_percent(disk.get('free_percent'), fraction=False)})",
                indent=2,
            )
        )
        lines.append(
            _line(
                "节点内存",
                f"available={memory.get('MemAvailable', 'N/A')} / "
                f"total={memory.get('MemTotal', 'N/A')} "
                f"swap_free={memory.get('SwapFree', 'N/A')}",
                indent=2,
            )
        )

    artifacts = app.get("artifacts", {})
    if isinstance(artifacts, Mapping) and artifacts:
        lines.extend(["", "Checkpoint / 文件："])
        for name, item in artifacts.items():
            if isinstance(item, Mapping):
                lines.append(
                    _line(
                        str(name),
                        f"exists={item.get('exists')} "
                        f"size={_format_bytes(item.get('size_bytes'))} "
                        f"mtime={item.get('modified_utc', 'N/A')} "
                        f"path={item.get('path', 'N/A')}",
                        indent=2,
                    )
                )

    lines.append("")
    if alerts:
        lines.append("告警：")
        for item in alerts:
            lines.append(
                f"  [{str(item.get('severity', 'warning')).upper()}] "
                f"{item.get('code')}: {item.get('message')}"
            )
            lines.append(f"    建议：{item.get('recommended_action')}")
    else:
        lines.append("告警：无")

    run_dir = str(report.get("run_dir"))
    quoted_run_dir = shlex.quote(run_dir)
    lines.extend(
        [
            "",
            "安全停止（在一致游标保存 checkpoint 后退出）：",
            f"  ./hpc_python.sh status_hpc.py --run-dir "
            f"{quoted_run_dir} --request-stop",
        ]
    )
    job_id = job.get("scheduler_job_id")
    if job_id:
        lines.extend(
            [
                "紧急强制取消（可能丢失最近 checkpoint 后的进度）：",
                f"  qdel {job_id}"
                if job.get("scheduler") == "pbs"
                else f"  scancel {job_id}",
            ]
        )

    history = report.get("history", [])
    if history:
        lines.extend(["", f"最近 {len(history)} 条心跳历史："])
        for item in history:
            history_stage = item.get("stage", {})
            history_progress = history_stage.get("progress", {})
            lines.append(
                "  "
                f"{item.get('updated_utc', 'N/A')} "
                f"{item.get('state', 'N/A')} "
                f"{history_stage.get('name', 'N/A')} "
                f"{_format_percent(history_progress.get('fraction'))}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/cx3_l40s_full")
    )
    parser.add_argument(
        "--watch",
        nargs="?",
        const=10.0,
        type=float,
        default=None,
        metavar="SECONDS",
        help="Refresh continuously (default interval: 10 seconds).",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--alerts-only", action="store_true")
    parser.add_argument("--history", type=int, default=0, metavar="N")
    parser.add_argument("--stale-seconds", type=float, default=None)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--request-stop", action="store_true")
    parser.add_argument("--reason", default="")
    add_external_path_flag(parser)
    args = parser.parse_args()
    if args.watch is not None and args.watch < 1.0:
        parser.error("--watch interval must be at least 1 second")
    if args.history < 0:
        parser.error("--history must be non-negative")
    if args.stale_seconds is not None and args.stale_seconds <= 0:
        parser.error("--stale-seconds must be positive")
    if args.request_stop and args.watch is not None:
        parser.error("--request-stop and --watch cannot be combined")
    try:
        run_dir = resolve_package_path(
            args.run_dir,
            allow_external=args.allow_external_paths,
            label="run directory",
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.request_stop:
        try:
            request = request_graceful_stop(run_dir, reason=args.reason)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(request, indent=2, sort_keys=True))
        else:
            print(
                "已提交安全停止请求。训练会在下一个一致游标确认请求，"
                "保存可续训 checkpoint 后退出。"
            )
            print(f"request_id={request['request_id']}")
            print(f"target_session={request['target_session_id']}")
        return 0

    last_code = 3
    while True:
        report, last_code = build_report(
            run_dir,
            query_scheduler=not args.no_scheduler,
            stale_seconds=args.stale_seconds,
            history_count=args.history,
        )
        if args.watch is not None and sys.stdout.isatty() and not args.json:
            print("\033[2J\033[H", end="")
        if args.json:
            print(
                json.dumps(
                    report,
                    indent=None if args.watch is not None else 2,
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
        else:
            print(
                render_report(report, alerts_only=args.alerts_only),
                flush=True,
            )
        if args.watch is None:
            return last_code
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
