#!/usr/bin/env python3
"""Validate the finalized 100 mm successful-policy rollout corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.contracts import ACTION_DIM, OBS_DIM, SCHEMA_HASH, TASK_GEOMETRY_HASH
from s_hpc.demos import DemoStore, DemoWindowSampler


INCOMPLETE_EXIT = 3


class DatasetIncomplete(RuntimeError):
    """The scanner may still be writing the requested dataset."""


def _load_metadata(root: Path) -> dict:
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise DatasetIncomplete(f"metadata is not available: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetIncomplete(
            f"metadata is not finalized JSON yet: {metadata_path}"
        ) from exc
    if not isinstance(metadata, dict):
        raise ValueError("dataset metadata must be a JSON object")
    return metadata


def validate_ready(root: Path) -> dict:
    """Check finalization and array contracts without hashing multi-GB data."""

    metadata = _load_metadata(root)
    if metadata.get("format") != DemoStore.FORMAT:
        raise ValueError("dataset is not an hpc GPU demonstration store")
    if metadata.get("schema_hash") != SCHEMA_HASH:
        raise ValueError("dataset observation schema differs from this package")
    if metadata.get("task_geometry_hash") != TASK_GEOMETRY_HASH:
        raise ValueError("dataset task geometry differs from this package")

    episodes = int(metadata.get("episodes", 0))
    completed = int(metadata.get("completed", 0))
    steps = int(metadata.get("steps", 0))
    if episodes <= 0 or steps <= 0:
        raise DatasetIncomplete("dataset has no finalized episodes or steps")
    if completed != episodes:
        raise DatasetIncomplete(
            f"dataset is still incomplete: completed={completed}/{episodes}"
        )
    if not bool(metadata.get("variable_episode_lengths", False)):
        raise ValueError("successful-policy corpus must use variable lengths")
    if metadata.get("valid_steps_file") != "valid_steps.npy":
        raise ValueError("successful-policy corpus has no valid_steps contract")

    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("dataset provenance is missing")
    if provenance.get("dataset_role") != (
        "successful_policy_rollout_self_distillation"
    ):
        raise ValueError("dataset is not a successful-policy rollout corpus")
    if provenance.get("expert_demonstrations") is not False:
        raise ValueError("successful-policy corpus is incorrectly expert-labelled")
    if provenance.get("controller") != "Diffusion--Safety--IK":
        raise ValueError(
            "formal self-distillation requires the zero-residual "
            "Diffusion--Safety--IK scan"
        )
    grid = provenance.get("fixed_start_grid")
    if not isinstance(grid, dict):
        raise ValueError("dataset fixed-grid provenance is missing")
    expected_grid = {
        "angular_bins": 360,
        "radial_bins": 50,
        "trials_per_cell": 2,
        "spawn_radius_mm": 100.0,
    }
    for name, expected in expected_grid.items():
        actual = grid.get(name)
        if isinstance(expected, float):
            matches = isinstance(actual, (int, float)) and abs(
                float(actual) - expected
            ) <= 1.0e-9
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"dataset grid {name} must be {expected!r}, found {actual!r}"
            )
    if grid.get("socket_exclusion_semantics") != (
        "exact initial CPU-MuJoCo screw-vs-socket physical contact"
    ):
        raise ValueError("dataset does not use the exact socket-contact filter")

    expected_hashes = metadata.get("content_sha256")
    required_files = {
        "observations.npy": ((episodes, steps + 1, OBS_DIM), np.float16),
        "actions.npy": ((episodes, steps, ACTION_DIM), np.float16),
        "spawn_xy.npy": ((episodes, 2), np.float32),
        "socket_mouth_xy.npy": ((episodes, 2), np.float32),
        "socket_angle_deg.npy": ((episodes,), np.float32),
        "valid_steps.npy": ((episodes,), np.int32),
    }
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(
        required_files
    ):
        raise DatasetIncomplete("dataset content hashes are not finalized")
    for name, (shape, dtype) in required_files.items():
        path = root / name
        if not path.is_file():
            raise DatasetIncomplete(f"dataset array is not available: {path}")
        array = np.load(path, mmap_mode="r")
        if tuple(array.shape) != shape:
            raise ValueError(
                f"dataset {name} shape differs: {tuple(array.shape)} != {shape}"
            )
        if array.dtype != np.dtype(dtype):
            raise ValueError(
                f"dataset {name} dtype differs: {array.dtype} != {np.dtype(dtype)}"
            )

    valid_steps = np.asarray(
        np.load(root / "valid_steps.npy", mmap_mode="r"), dtype=np.int64
    )
    policy_start = int(provenance.get("policy_start_step", -1))
    action_horizon = int(provenance.get("action_horizon", -1))
    if policy_start < 0 or action_horizon <= 0:
        raise ValueError("dataset policy-window provenance is invalid")
    minimum = policy_start + action_horizon
    if np.any(valid_steps < minimum) or np.any(valid_steps > steps):
        raise ValueError("dataset valid_steps are outside trainable bounds")
    windows = int(
        np.maximum(valid_steps - policy_start - action_horizon + 1, 0).sum()
    )
    if windows <= 0:
        raise ValueError("dataset has no complete Diffusion training windows")
    return {
        "status": "ready",
        "path": str(root),
        "episodes": episodes,
        "steps": steps,
        "valid_steps_min": int(valid_steps.min()),
        "valid_steps_max": int(valid_steps.max()),
        "training_windows": windows,
        "content_hashes_present": True,
        "full_hash_validation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--full",
        action="store_true",
        help="also recompute every content SHA-256 through DemoWindowSampler",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    try:
        result = validate_ready(root)
        if args.full:
            sampler = DemoWindowSampler(root, seed=args.seed)
            provenance = sampler.metadata.get("provenance") or {}
            horizon = int(provenance["action_horizon"])
            result.update(
                {
                    "full_hash_validation": True,
                    "training_windows": sampler.window_count(horizon),
                }
            )
    except DatasetIncomplete as exc:
        print(
            json.dumps(
                {"status": "incomplete", "path": str(root), "message": str(exc)},
                sort_keys=True,
            ),
            flush=True,
        )
        return INCOMPLETE_EXIT
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "status": "invalid",
                    "path": str(root),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
