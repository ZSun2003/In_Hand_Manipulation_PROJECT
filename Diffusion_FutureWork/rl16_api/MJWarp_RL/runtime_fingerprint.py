"""Deterministic fingerprint of the shared formal training runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


RUNTIME_CODE_SCHEMA = "grab-g6-only-runtime-code-v6"
RUNTIME_CODE_FILES = (
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
)


def runtime_code_manifest(
    package_dir: Path | None = None,
) -> dict[str, Any]:
    """Hash every shared module that can alter formal physics or learning."""

    root = (
        Path(package_dir).expanduser().resolve()
        if package_dir is not None
        else Path(__file__).resolve().parent
    )
    files: dict[str, str] = {}
    for relative in RUNTIME_CODE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"formal runtime module is missing: {path}"
            )
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema": RUNTIME_CODE_SCHEMA,
        "files": files,
    }


def runtime_code_fingerprint_sha256(
    package_dir: Path | None = None,
) -> str:
    manifest = runtime_code_manifest(package_dir)
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "RUNTIME_CODE_FILES",
    "RUNTIME_CODE_SCHEMA",
    "runtime_code_fingerprint_sha256",
    "runtime_code_manifest",
]
