#!/usr/bin/env python3
"""Create a small, self-contained diagnostic bundle for one HPC run.

The bundle deliberately excludes datasets and checkpoints.  It preserves the
exact text source/configuration, scheduler state, CUDA/node information, run
status, telemetry tails, and the allocation-log tail needed to diagnose a
failure without reopening a large training directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterable, Optional


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.monitoring import atomic_write_json
from s_hpc.paths import add_external_path_flag, resolve_package_path


SCHEMA = "screw-hpc-diagnostics-v1"
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_SOURCE_FILE_BYTES = 4 * 1024 * 1024
SOURCE_SUFFIXES = frozenset(
    {
        ".json",
        ".md",
        ".mjcf",
        ".mtl",
        ".obj",
        ".pbs",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".urdf",
        ".xml",
        ".yaml",
        ".yml",
    }
)
ARTIFACT_SUFFIXES = frozenset(
    {".csv", ".env", ".json", ".jsonl", ".log", ".md", ".txt"}
)
EXCLUDED_PARTS = frozenset(
    {
        ".cache",
        ".git",
        ".pytest_cache",
        "RL_env",
        "__pycache__",
        "checkpoints",
        "datasets",
        "diagnostics",
        "logs",
        "outputs",
        "runs",
        "wheelhouse",
    }
)
SAFE_ENV_NAMES = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "HOSTNAME",
        "LANG",
        "LC_ALL",
        "MKL_NUM_THREADS",
        "MUJOCO_GL",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "PBS_ENVIRONMENT",
        "PBS_JOBID",
        "PBS_JOBNAME",
        "PBS_NCPUS",
        "PBS_NODEFILE",
        "PBS_O_HOST",
        "PBS_O_WORKDIR",
        "PBS_QUEUE",
        "SCREW_HPC_ROOT",
        "SCREW_PYTHON_MODULE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_PARTITION",
        "SLURM_SUBMIT_DIR",
        "WARP_CACHE_PATH",
        "XDG_CACHE_HOME",
    }
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_component(value: str) -> str:
    cleaned = "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in value
    )
    return cleaned.strip("._-") or "manual"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _tail_bytes(source: Path, limit: int = MAX_TEXT_BYTES) -> bytes:
    size = source.stat().st_size
    with source.open("rb") as stream:
        if size > limit:
            stream.seek(-limit, os.SEEK_END)
        return stream.read(limit)


def _copy_text_or_tail(source: Path, target: Path) -> dict[str, Any]:
    size = source.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    if size <= MAX_TEXT_BYTES:
        shutil.copyfile(source, target)
        mode = "complete"
    else:
        target = target.with_name(target.name + ".tail")
        target.write_bytes(_tail_bytes(source))
        mode = "tail"
    return {
        "source": str(source),
        "bundle_path": str(target),
        "source_bytes": size,
        "copy_mode": mode,
    }


def _iter_project_files(source_root: Path) -> Iterable[Path]:
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source_root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.name.startswith("._"):
            continue
        yield path


def snapshot_source(source_root: Path, destination: Path) -> dict[str, Any]:
    inventory = []
    aggregate = hashlib.sha256()
    for source in _iter_project_files(source_root):
        relative = source.relative_to(source_root)
        digest = sha256_file(source)
        size = source.stat().st_size
        record = {
            "path": relative.as_posix(),
            "bytes": size,
            "sha256": digest,
            "copy_mode": "inventory_only",
        }
        if (
            source.suffix.lower() in SOURCE_SUFFIXES
            and size <= MAX_SOURCE_FILE_BYTES
        ):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            record["copy_mode"] = "complete"
        inventory.append(record)
        aggregate.update(relative.as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
    return {
        "files": inventory,
        "file_count": len(inventory),
        "snapshot_sha256": aggregate.hexdigest(),
    }


def snapshot_run_artifacts(run_dir: Path, destination: Path) -> dict[str, Any]:
    inventory = []
    for source in sorted(run_dir.rglob("*")):
        if not source.is_file() or source.is_symlink():
            continue
        relative = source.relative_to(run_dir)
        if "diagnostics" in relative.parts:
            continue
        size = source.stat().st_size
        record: dict[str, Any] = {
            "path": relative.as_posix(),
            "bytes": size,
            "mtime_ns": source.stat().st_mtime_ns,
        }
        if size <= MAX_SOURCE_FILE_BYTES:
            record["sha256"] = sha256_file(source)
        if source.suffix.lower() in ARTIFACT_SUFFIXES:
            copied = _copy_text_or_tail(
                source, destination / relative
            )
            record["copy_mode"] = copied["copy_mode"]
            record["bundle_path"] = str(
                Path(copied["bundle_path"]).relative_to(destination.parent)
            )
        inventory.append(record)
    return {"files": inventory, "file_count": len(inventory)}


def _run_command(
    name: str,
    command: list[str],
    destination: Path,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    record: dict[str, Any] = {"name": name, "command": command}
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        record.update(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout[-MAX_TEXT_BYTES:],
                "stderr": completed.stderr[-MAX_TEXT_BYTES:],
            }
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    _write_json(destination / f"{name}.json", record)
    return record


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[str(name)] = distribution.version
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


def environment_snapshot(source_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(source_root)
    limits = {}
    for name in (
        "RLIMIT_AS",
        "RLIMIT_CORE",
        "RLIMIT_DATA",
        "RLIMIT_MEMLOCK",
        "RLIMIT_NOFILE",
        "RLIMIT_NPROC",
        "RLIMIT_STACK",
    ):
        key = getattr(resource, name, None)
        if key is not None:
            limits[name] = list(resource.getrlimit(key))
    ctypes_result: dict[str, Any]
    try:
        import _ctypes
        import ctypes.util

        ctypes_result = {
            "status": "pass",
            "module": str(getattr(_ctypes, "__file__", "")),
            "libffi": ctypes.util.find_library("ffi"),
        }
    except Exception as exc:  # pragma: no cover - only broken HPC runtimes
        ctypes_result = {
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "ctypes_libffi_check": ctypes_result,
        "environment": {
            name: os.environ[name]
            for name in sorted(SAFE_ENV_NAMES)
            if name in os.environ
        },
        "resource_limits": limits,
        "disk": {
            "path": str(source_root),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "packages": _package_versions(),
    }


def collect_diagnostics(
    *,
    source_root: Path,
    run_dir: Path,
    exit_code: Optional[int] = None,
    job_id: Optional[str] = None,
    allocation_log: Optional[Path] = None,
) -> Path:
    """Collect diagnostics and return the completed ``.tar.gz`` path."""

    source_root = source_root.resolve()
    run_dir = run_dir.resolve()
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    effective_job_id = (
        job_id
        or os.environ.get("PBS_JOBID")
        or os.environ.get("SLURM_JOB_ID")
        or "manual"
    )
    stem = f"diagnostics_{_safe_component(effective_job_id)}_{utc_stamp()}"
    archive = diagnostics_dir / f"{stem}.tar.gz"

    with tempfile.TemporaryDirectory(
        prefix=f".{stem}.", dir=diagnostics_dir
    ) as temporary:
        staging = Path(temporary) / stem
        staging.mkdir()
        source_inventory = snapshot_source(
            source_root, staging / "source_snapshot"
        )
        artifact_inventory = snapshot_run_artifacts(
            run_dir, staging / "run_artifacts"
        )
        environment = environment_snapshot(source_root)
        _write_json(staging / "environment.json", environment)
        _write_json(staging / "source_inventory.json", source_inventory)
        _write_json(staging / "run_inventory.json", artifact_inventory)

        commands = []
        if shutil.which("nvidia-smi"):
            commands.append(
                _run_command(
                    "nvidia_smi",
                    ["nvidia-smi", "-q"],
                    staging / "commands",
                    timeout=15.0,
                )
            )
        if effective_job_id != "manual" and shutil.which("qstat"):
            commands.append(
                _run_command(
                    "qstat_job",
                    ["qstat", "-fx", effective_job_id],
                    staging / "commands",
                )
            )
            user = os.environ.get("USER")
            if user:
                commands.append(
                    _run_command(
                        "qstat_user",
                        ["qstat", "-u", user],
                        staging / "commands",
                    )
                )
        if shutil.which("df"):
            commands.append(
                _run_command(
                    "disk_free",
                    ["df", "-h", str(source_root)],
                    staging / "commands",
                )
            )
        if shutil.which("ldd"):
            commands.append(
                _run_command(
                    "python_ldd",
                    ["ldd", sys.executable],
                    staging / "commands",
                )
            )

        allocation_record = None
        if allocation_log is not None and allocation_log.is_file():
            copied = _copy_text_or_tail(
                allocation_log, staging / "allocation.log"
            )
            allocation_record = {
                **copied,
                "bundle_path": str(
                    Path(copied["bundle_path"]).relative_to(staging)
                ),
            }

        summary = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": effective_job_id,
            "exit_code": exit_code,
            "run_dir": str(run_dir),
            "source_root": str(source_root),
            "source_snapshot_sha256": source_inventory["snapshot_sha256"],
            "source_file_count": source_inventory["file_count"],
            "run_artifact_count": artifact_inventory["file_count"],
            "allocation_log": allocation_record,
            "commands": [
                {
                    "name": item["name"],
                    "returncode": item.get("returncode"),
                    "error": item.get("error"),
                }
                for item in commands
            ],
            "libffi_status": environment["ctypes_libffi_check"],
        }
        _write_json(staging / "SUMMARY.json", summary)

        temporary_archive = archive.with_suffix(archive.suffix + ".tmp")
        with tarfile.open(temporary_archive, "w:gz") as stream:
            stream.add(staging, arcname=stem, recursive=True)
        os.replace(temporary_archive, archive)

    pointer = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "job_id": effective_job_id,
        "exit_code": exit_code,
    }
    atomic_write_json(diagnostics_dir / "latest.json", pointer)
    print(
        json.dumps(
            {
                "event": "diagnostics_bundle_created",
                **pointer,
                "path": str(archive),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--allocation-log", type=Path, default=None)
    add_external_path_flag(parser)
    args = parser.parse_args()
    try:
        run_dir = resolve_package_path(
            args.run_dir,
            allow_external=args.allow_external_paths,
            label="diagnostic run directory",
        )
        allocation_log = (
            resolve_package_path(
                args.allocation_log,
                allow_external=args.allow_external_paths,
                label="allocation log",
            )
            if args.allocation_log is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))
    collect_diagnostics(
        source_root=ROOT,
        run_dir=run_dir,
        exit_code=args.exit_code,
        job_id=args.job_id,
        allocation_log=allocation_log,
    )


if __name__ == "__main__":
    main()
