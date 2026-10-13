#!/usr/bin/env python3
"""Fail-closed qualification for an AutoDL RTX PRO 6000 Blackwell instance."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback
from typing import Any


GPU_TYPE = "RTX_PRO_6000_BLACKWELL"
MINIMUM_DRIVER = "575.51.03"
RECOMMENDED_DRIVER_MAJOR = 580
MINIMUM_MEMORY_BYTES = 90 * 1024**3
EXPECTED_COMPUTE_CAPABILITY = (12, 0)
EXPECTED_TORCH_CUDA = "12.8"
EXPECTED_VERSIONS = {
    "torch": "2.7.1",
    "mujoco": "3.10.0",
    "mujoco-warp": "3.10.0.3",
    "warp-lang": "1.14.0",
}


def normalize_gpu_name(name: str) -> str:
    return "".join(character for character in name.upper() if character.isalnum())


def is_target_gpu_name(name: str) -> bool:
    normalized = normalize_gpu_name(name)
    return (
        "RTXPRO6000" in normalized
        and "BLACKWELL" in normalized
        and "MAXQ" not in normalized
    )


def numeric_version(value: str) -> tuple[int, ...]:
    pieces = []
    for token in value.strip().split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        pieces.append(int(digits))
    if not pieces:
        raise ValueError(f"version has no numeric components: {value!r}")
    return tuple(pieces)


def version_at_least(actual: str, required: str) -> bool:
    left = numeric_version(actual)
    right = numeric_version(required)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


def _driver_version() -> str:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"nvidia-smi is unavailable: {exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "nvidia-smi driver query failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    versions = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if not versions:
        raise RuntimeError("nvidia-smi returned no driver version")
    return versions[0]


def _exact_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"required package is missing: {package}") from exc
        if actual.split("+", 1)[0] != expected:
            raise RuntimeError(
                f"{package} must be exactly {expected}, got {actual}"
            )
        result[package] = actual
    return result


def run_checks() -> dict[str, Any]:
    if platform.system() != "Linux":
        raise RuntimeError("the AutoDL package requires Linux")
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError(
            "Python 3.10, 3.11, or 3.12 is required; "
            f"got {platform.python_version()}"
        )
    libc_name, libc_version = platform.libc_ver()
    if libc_name.lower() != "glibc" or not version_at_least(libc_version, "2.28"):
        raise RuntimeError(
            f"Warp requires glibc >= 2.28, got {libc_name} {libc_version}"
        )

    versions = _exact_versions()
    import mujoco
    import mujoco_warp
    import torch
    import warp as wp

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to PyTorch")
    # The packaged configuration is intentionally single-GPU and targets
    # logical cuda:0.  CUDA_VISIBLE_DEVICES may remap a physical card, but
    # both PyTorch and Warp must still qualify the same logical device.
    device_index = 0
    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    name = str(properties.name)
    if not is_target_gpu_name(name):
        raise RuntimeError(
            "expected an NVIDIA RTX PRO 6000 Blackwell, got "
            f"{name!r}"
        )
    capability = (int(properties.major), int(properties.minor))
    if capability != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError(
            "RTX PRO 6000 Blackwell must expose compute capability 12.0, "
            f"got {capability[0]}.{capability[1]}"
        )
    total_memory = int(properties.total_memory)
    if total_memory < MINIMUM_MEMORY_BYTES:
        raise RuntimeError(
            "a full 96 GB card is required; PyTorch sees only "
            f"{total_memory / 1024**3:.2f} GiB"
        )
    if str(torch.version.cuda) != EXPECTED_TORCH_CUDA:
        raise RuntimeError(
            f"PyTorch must use CUDA {EXPECTED_TORCH_CUDA}, "
            f"got {torch.version.cuda}"
        )
    architectures = list(torch.cuda.get_arch_list())
    if "sm_120" not in architectures:
        raise RuntimeError(
            "the installed PyTorch wheel has no native sm_120 code"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA bfloat16 support is unavailable")

    driver = _driver_version()
    if not version_at_least(driver, MINIMUM_DRIVER):
        raise RuntimeError(
            f"this package's conservative RTX PRO 6000 deployment gate is "
            f"driver >= "
            f"{MINIMUM_DRIVER}, got {driver}"
        )

    device = torch.device(f"cuda:{device_index}")
    first = torch.randn(
        (512, 512), device=device, dtype=torch.bfloat16, requires_grad=True
    )
    second = torch.randn(
        (512, 512), device=device, dtype=torch.bfloat16, requires_grad=True
    )
    loss = (first @ second).to(torch.float32).square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Blackwell bfloat16 forward/backward smoke is non-finite")
    bf16_loss = float(loss.detach().cpu())
    del first, second, loss
    torch.cuda.empty_cache()

    wp.init()
    warp_device = wp.get_device("cuda:0")
    if not bool(getattr(warp_device, "is_cuda", False)):
        raise RuntimeError(f"Warp did not expose cuda:0: {warp_device}")
    warp_arch = int(getattr(warp_device, "arch", 0))
    if warp_arch != 120:
        raise RuntimeError(
            f"Warp cuda:0 must be sm_120, got sm_{warp_arch}"
        )
    warp_toolkit = wp.get_cuda_toolkit_version()
    if warp_toolkit != (12, 9):
        raise RuntimeError(
            "the official Warp 1.14 wheel must use CUDA 12.9, "
            f"got {warp_toolkit}"
        )
    supported_architectures = list(wp.get_cuda_supported_archs())
    if 120 not in supported_architectures:
        raise RuntimeError("Warp NVRTC cannot compile native sm_120 kernels")
    from autodl_warp_kernel_smoke import run_warp_kernel_smoke

    warp_kernel = run_warp_kernel_smoke(warp_device)

    return {
        "mode": "autodl_rtx_pro_6000_blackwell",
        "status": "pass",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "glibc": libc_version,
        "gpu": {
            "type": GPU_TYPE,
            "index": int(device_index),
            "name": name,
            "compute_capability": list(capability),
            "total_memory_bytes": total_memory,
            "total_memory_gib": total_memory / 1024**3,
            "driver_version": driver,
            "recommended_driver_branch_or_newer": (
                numeric_version(driver)[0] >= RECOMMENDED_DRIVER_MAJOR
            ),
            "torch_cuda_build": str(torch.version.cuda),
            "torch_architectures": architectures,
            "bf16_forward_backward_loss": bf16_loss,
        },
        "warp": {
            "device": str(warp_device),
            "name": str(getattr(warp_device, "name", "")),
            "architecture": warp_arch,
            "toolkit_version": list(warp_toolkit),
            "supported_architectures": supported_architectures,
            "jit_kernel": warp_kernel,
        },
        "versions": versions,
        "native_imports": {
            "mujoco": str(mujoco.__version__),
            "mujoco_warp": str(mujoco_warp.__name__),
        },
    }


def main() -> int:
    try:
        result = run_checks()
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "mode": "autodl_rtx_pro_6000_blackwell",
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        traceback.print_exc()
        return 1
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
