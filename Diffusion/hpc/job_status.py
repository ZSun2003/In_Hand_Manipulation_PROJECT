#!/usr/bin/env python3
"""Record scheduler-wrapper lifecycle without importing the training stack."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any
import uuid


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.monitoring import (
    atomic_write_json,
    jsonable,
    read_json,
    utc_now,
)
from s_hpc.paths import add_external_path_flag, resolve_package_path


SCHEMA = "screw-hpc-job-v1"


def append_strict_jsonl(
    path: Path,
    payload: dict[str, Any],
) -> Path:
    """Durably append one strict-JSON lifecycle event without truncation."""

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


def persist_job_status(
    target: Path,
    payload: dict[str, Any],
) -> None:
    """Write the authoritative snapshot, then best-effort append its history."""

    atomic_write_json(target, payload)
    try:
        append_strict_jsonl(
            target.with_name("job-history.jsonl"),
            payload,
        )
    except (OSError, TypeError, ValueError) as exc:
        # Scheduler lifecycle telemetry must never turn a healthy training run
        # into a failed one merely because its append-only audit file is down.
        print(
            "WARNING: job status history append failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--state",
        choices=(
            "starting",
            "preflight",
            "running",
            "completed",
            "interrupted_resumable",
            "failed",
        ),
        required=True,
    )
    parser.add_argument("--raw-exit-code", type=int, default=None)
    parser.add_argument("--message", default="")
    add_external_path_flag(parser)
    args = parser.parse_args()
    try:
        run_dir = resolve_package_path(
            args.run_dir,
            allow_external=args.allow_external_paths,
            label="run directory",
        )
    except ValueError as exc:
        parser.error(str(exc))
    target = run_dir / "status" / "job.json"
    previous = read_json(target) or {}
    now = utc_now()
    scheduler = (
        "pbs"
        if os.environ.get("PBS_JOBID")
        else "slurm"
        if os.environ.get("SLURM_JOB_ID")
        else "local"
    )
    job_id = os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID")
    prior_allocation = previous.get("allocation", {})
    explicit_allocation_id = os.environ.get("SCREW_ALLOCATION_ID")
    same_allocation = (
        (
            prior_allocation.get("id") == explicit_allocation_id
            if explicit_allocation_id
            else prior_allocation.get("scheduler_job_id") == job_id
        )
        and prior_allocation.get("scheduler") == scheduler
    )
    allocation_id = (
        explicit_allocation_id
        or prior_allocation.get("id")
        if same_allocation
        else explicit_allocation_id or uuid.uuid4().hex
    )
    prior_run = previous.get("run", {})
    manifest = read_json(run_dir / "run_manifest.json") or {}
    manifest_run = manifest.get("run", {})
    run_id = (
        manifest_run.get("id")
        if isinstance(manifest_run, dict)
        else None
    ) or (
        prior_run.get("id")
        if isinstance(prior_run, dict)
        else None
    )
    if not isinstance(run_id, str) or not run_id:
        run_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"screw-hpc-run:{run_dir.resolve()}",
        ).hex
    payload = {
        "schema": SCHEMA,
        "sequence": int(previous.get("sequence", 0)) + 1,
        "state": args.state,
        "updated_utc": now,
        "started_utc": (
            previous.get("started_utc")
            if same_allocation
            else now
        ),
        "ended_utc": (
            now
            if args.state in {"completed", "interrupted_resumable", "failed"}
            else None
        ),
        "message": args.message,
        "raw_pipeline_exit_code": args.raw_exit_code,
        "identifiers": {
            "run_id": run_id,
            "allocation_id": allocation_id,
            "session_id": allocation_id,
            "wrapper_session_id": allocation_id,
            "scheduler_job_id": job_id,
        },
        "run": {
            "id": run_id,
            "name": run_dir.name,
            "directory": str(run_dir),
        },
        "allocation": {
            "id": allocation_id,
            "identity_source": (
                "SCREW_ALLOCATION_ID"
                if explicit_allocation_id
                else "scheduler_job_id"
                if job_id
                else "generated"
            ),
            "scheduler": scheduler,
            "scheduler_job_id": job_id,
            "job_name": os.environ.get("PBS_JOBNAME")
            or os.environ.get("SLURM_JOB_NAME"),
            "queue_or_partition": os.environ.get("PBS_QUEUE")
            or os.environ.get("SLURM_JOB_PARTITION"),
            "hostname": socket.gethostname(),
            "wrapper_pid": os.getppid(),
            "submit_directory": os.environ.get("PBS_O_WORKDIR")
            or os.environ.get("SLURM_SUBMIT_DIR"),
            "maximum_runtime": os.environ.get("SCREW_MAX_RUNTIME"),
        },
    }
    persist_job_status(target, payload)


if __name__ == "__main__":
    main()
