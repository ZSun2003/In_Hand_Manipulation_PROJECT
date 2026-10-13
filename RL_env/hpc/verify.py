#!/usr/bin/env python3
"""Verify one pinned IRP HPC runtime profile without importing project code."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent


def _base_version(value: str) -> str:
    return value.split("+", 1)[0]


def _numeric_prefix(value: str) -> tuple[int, ...]:
    match = re.match(r"(\d+(?:\.\d+)*)", value)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _requirements(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        if "==" not in line:
            raise SystemExit(f"non-exact central direct requirement: {line}")
        name, version = line.split("==", 1)
        expected[name.strip().lower()] = version.strip()
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("core",))
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="also require one visible CUDA GPU",
    )
    args = parser.parse_args()

    spec = json.loads((ROOT / "profiles.json").read_text(encoding="utf-8"))
    profile = spec["profiles"][args.profile]
    errors: list[str] = []

    if platform.system() != "Linux":
        errors.append(f"platform must be Linux, found {platform.system()}")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        errors.append(f"architecture must be x86_64, found {platform.machine()}")
    if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
        errors.append(
            "Python must be 3.11 or 3.12, found "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )

    libc_name, libc_version = platform.libc_ver()
    if libc_name.lower() != "glibc":
        errors.append(
            "glibc >= 2.28 is required; detected "
            f"{libc_name or 'unknown'} {libc_version or 'unknown'}"
        )
    elif _numeric_prefix(libc_version) < (2, 28):
        errors.append(f"glibc must be >= 2.28, found {libc_version}")

    expected = _requirements(ROOT / profile["requirements"])
    expected["torch"] = profile["torch"]
    installed: dict[str, str] = {}
    for package, wanted in expected.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"{package} is not installed")
            continue
        installed[package] = actual
        if _base_version(actual) != wanted:
            errors.append(f"{package}: expected {wanted}, found {actual}")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on target runtime
        errors.append(f"torch import failed: {exc}")
    else:
        if torch.version.cuda != profile["torch_cuda"]:
            errors.append(
                "torch CUDA runtime: expected "
                f"{profile['torch_cuda']}, found {torch.version.cuda}"
            )
        if args.require_gpu and not torch.cuda.is_available():
            errors.append("no CUDA GPU is visible to PyTorch")
        elif args.require_gpu:
            capability = torch.cuda.get_device_capability(0)
            print(
                "gpu",
                torch.cuda.get_device_name(0),
                "capability",
                f"{capability[0]}.{capability[1]}",
            )

    for module_name in ("mujoco", "mujoco_warp", "warp"):
        try:
            __import__(module_name)
        except Exception as exc:  # pragma: no cover - target system specific
            errors.append(f"{module_name} import failed: {exc}")

    if errors:
        print(f"profile={args.profile} status=FAIL", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"profile={args.profile} status=PASS")
    print("python", platform.python_version())
    print("platform", platform.platform())
    print("prefix", sys.prefix)
    for package in sorted(installed):
        print(package, installed[package])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
