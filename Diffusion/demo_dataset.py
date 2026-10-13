#!/usr/bin/env python3
"""Portable, episode-based demonstration dataset for the screw task.

The on-disk contract is intentionally small and does not depend on any file
outside this directory::

    demos/
      metadata.json
      episodes/
        episode_000000.npz
        episode_000001.npz

Each episode contains at least:

``observations``
    ``float32[T, obs_dim]`` full-state observations.  Observation ``t`` is the
    state immediately before applying action ``t``.

``actions``
    ``float32[T, action_dim]`` expert actions.

The metadata file owns all dimensions used by the policy.  In particular,
``obs_horizon`` is the number of states supplied to the policy and ``horizon``
is the length of the predicted action chunk.  Short observation histories at
the beginning of an episode are padded by repeating the first observation;
future action chunks are never padded.

This module is both an importable API for the expert-data generator and a
small command-line validation/inspection utility.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from screw_config import OBSERVATION_SCHEMA, TASK_GEOMETRY_HASH

FORMAT_VERSION = 3
METADATA_FILENAME = "metadata.json"
EPISODE_DIRNAME = "episodes"
OBSERVATION_KEY = "observations"
ACTION_KEY = "actions"
MIN_STD = 1.0e-6


@dataclass(frozen=True)
class DatasetMetadata:
    """Shape and timing contract shared by data generation and training."""

    obs_dim: int
    action_dim: int
    obs_horizon: int
    horizon: int
    policy_start_step: int
    dt: float = 0.02
    format_version: int = FORMAT_VERSION
    task: str = "screw_two_turns"
    observation_schema: str = OBSERVATION_SCHEMA
    task_geometry_hash: str = TASK_GEOMETRY_HASH
    observation_description: str = "full simulator state"
    action_description: str = "bounded high-level residual/IK command"

    def __post_init__(self) -> None:
        integer_fields = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "obs_horizon": self.obs_horizon,
            "horizon": self.horizon,
        }
        for name, value in integer_fields.items():
            if int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            object.__setattr__(self, name, int(value))
        if int(self.policy_start_step) != self.policy_start_step:
            raise ValueError("policy_start_step must be an integer")
        object.__setattr__(
            self, "policy_start_step", int(self.policy_start_step)
        )
        if self.policy_start_step < 0:
            raise ValueError("policy_start_step cannot be negative")
        object.__setattr__(self, "dt", float(self.dt))
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {self.dt!r}")
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported dataset format {self.format_version}; "
                f"this code supports {FORMAT_VERSION}"
            )
        if self.observation_schema != OBSERVATION_SCHEMA:
            raise ValueError(
                "dataset observation schema is incompatible with this task"
            )
        if self.task_geometry_hash != TASK_GEOMETRY_HASH:
            raise ValueError(
                "dataset task geometry hash is incompatible with this task"
            )

    @property
    def action_horizon(self) -> int:
        """Alias retained for callers that use the longer field name."""

        return self.horizon

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Keeping this alias in JSON makes the contract self-documenting while
        # ``horizon`` remains the canonical field requested by the project.
        result["action_horizon"] = self.horizon
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DatasetMetadata":
        data = dict(raw)
        if "horizon" not in data and "action_horizon" in data:
            data["horizon"] = data["action_horizon"]
        if (
            "action_horizon" in data
            and "horizon" in data
            and int(data["action_horizon"]) != int(data["horizon"])
        ):
            raise ValueError("metadata horizon and action_horizon disagree")
        allowed = set(cls.__dataclass_fields__)
        clean = {key: value for key, value in data.items() if key in allowed}
        required = {
            "obs_dim",
            "action_dim",
            "obs_horizon",
            "horizon",
            "policy_start_step",
            "format_version",
            "observation_schema",
            "task_geometry_hash",
        }
        missing = required - clean.keys()
        if missing:
            raise ValueError(
                "metadata is missing required fields: "
                f"{sorted(missing)}; legacy datasets must be regenerated"
            )
        return cls(**clean)


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_metadata(
    dataset_dir: str | os.PathLike[str],
    metadata: DatasetMetadata | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Create the dataset contract and episode directory.

    Existing metadata is never silently changed.  Calling this repeatedly with
    identical metadata is safe and is useful for resumable data generation.
    """

    root = Path(dataset_dir).expanduser().resolve()
    parsed = (
        metadata
        if isinstance(metadata, DatasetMetadata)
        else DatasetMetadata.from_dict(metadata)
    )
    metadata_path = root / METADATA_FILENAME
    if metadata_path.exists() and not overwrite:
        current = load_metadata(root)
        if current != parsed:
            raise FileExistsError(
                f"{metadata_path} already exists with different metadata"
            )
    else:
        _atomic_json_write(metadata_path, parsed.to_dict())
    (root / EPISODE_DIRNAME).mkdir(parents=True, exist_ok=True)
    return metadata_path


def load_metadata(dataset_dir: str | os.PathLike[str]) -> DatasetMetadata:
    path = Path(dataset_dir).expanduser().resolve() / METADATA_FILENAME
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"dataset metadata not found at {path}; create it with write_metadata()"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return DatasetMetadata.from_dict(raw)


def discover_episode_paths(
    dataset_dir: str | os.PathLike[str],
) -> Tuple[Path, ...]:
    """Return episode files in stable lexical order."""

    root = Path(dataset_dir).expanduser().resolve()
    episode_dir = root / EPISODE_DIRNAME
    if episode_dir.is_dir():
        paths = tuple(sorted(episode_dir.glob("*.npz")))
    else:
        # Read compatibility for an early flat layout.  New data is always
        # written into ``episodes/``.
        paths = tuple(sorted(root.glob("episode_*.npz")))
    return paths


def _coerce_episode_arrays(
    observations: np.ndarray | Sequence[Sequence[float]],
    actions: np.ndarray | Sequence[Sequence[float]],
    metadata: DatasetMetadata,
) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observations, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != metadata.obs_dim:
        raise ValueError(
            f"observations must have shape [T, {metadata.obs_dim}], got {obs.shape}"
        )
    if action.ndim != 2 or action.shape[1] != metadata.action_dim:
        raise ValueError(
            f"actions must have shape [T, {metadata.action_dim}], got {action.shape}"
        )
    if obs.shape[0] != action.shape[0]:
        raise ValueError(
            f"observations/actions lengths differ: {obs.shape[0]} vs {action.shape[0]}"
        )
    if (
        obs.shape[0] - metadata.policy_start_step
        < metadata.horizon
    ):
        raise ValueError(
            f"episode has only {obs.shape[0] - metadata.policy_start_step} "
            "post-handoff steps, shorter than horizon "
            f"{metadata.horizon}"
        )
    if not np.all(np.isfinite(obs)):
        raise ValueError("observations contain NaN or infinity")
    if not np.all(np.isfinite(action)):
        raise ValueError("actions contain NaN or infinity")
    return np.ascontiguousarray(obs), np.ascontiguousarray(action)


def save_episode(
    dataset_dir: str | os.PathLike[str],
    episode_id: int | str,
    observations: np.ndarray | Sequence[Sequence[float]],
    actions: np.ndarray | Sequence[Sequence[float]],
    *,
    extras: Optional[Mapping[str, np.ndarray | Sequence[Any]]] = None,
    overwrite: bool = False,
    compressed: bool = True,
) -> Path:
    """Validate and atomically save one expert episode.

    ``extras`` can hold diagnostics such as phase IDs, success flags, contact
    modes, or rewards.  Extra keys cannot replace the two canonical arrays.
    """

    root = Path(dataset_dir).expanduser().resolve()
    metadata = load_metadata(root)
    obs, action = _coerce_episode_arrays(observations, actions, metadata)
    if isinstance(episode_id, int):
        if episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        filename = f"episode_{episode_id:06d}.npz"
    else:
        safe_id = str(episode_id).strip()
        if safe_id.endswith(".npz"):
            safe_id = safe_id[:-4]
        if not safe_id or any(char not in "-_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for char in safe_id):
            raise ValueError("string episode_id may contain only letters, digits, '-' and '_'")
        filename = (
            safe_id if safe_id.startswith("episode_") else f"episode_{safe_id}"
        ) + ".npz"

    target = root / EPISODE_DIRNAME / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"episode already exists: {target}")

    payload: Dict[str, np.ndarray] = {
        OBSERVATION_KEY: obs,
        ACTION_KEY: action,
    }
    if extras:
        for key, value in extras.items():
            if key in payload:
                raise ValueError(f"extra key {key!r} is reserved")
            array = np.asarray(value)
            if array.dtype == object:
                raise ValueError(
                    f"extra {key!r} has object dtype, which is intentionally unsupported"
                )
            if array.ndim > 0 and array.shape[0] not in (1, obs.shape[0]):
                raise ValueError(
                    f"extra {key!r} first dimension must be 1 or T={obs.shape[0]}, "
                    f"got {array.shape}"
                )
            payload[key] = array

    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".npz.tmp", dir=target.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            if compressed:
                np.savez_compressed(stream, **payload)
            else:
                np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


def load_episode(
    path: str | os.PathLike[str],
    metadata: Optional[DatasetMetadata] = None,
) -> Dict[str, np.ndarray]:
    """Load one episode without permitting pickled/object arrays."""

    episode_path = Path(path).expanduser().resolve()
    with np.load(episode_path, allow_pickle=False) as archive:
        if OBSERVATION_KEY not in archive or ACTION_KEY not in archive:
            raise ValueError(
                f"{episode_path} must contain {OBSERVATION_KEY!r} and "
                f"{ACTION_KEY!r}"
            )
        result = {key: np.asarray(archive[key]) for key in archive.files}
    if metadata is not None:
        obs, action = _coerce_episode_arrays(
            result[OBSERVATION_KEY], result[ACTION_KEY], metadata
        )
        result[OBSERVATION_KEY] = obs
        result[ACTION_KEY] = action
    return result


def validate_dataset(
    dataset_dir: str | os.PathLike[str],
) -> Dict[str, Any]:
    """Fully validate all episodes and return a compact summary."""

    root = Path(dataset_dir).expanduser().resolve()
    metadata = load_metadata(root)
    paths = discover_episode_paths(root)
    if not paths:
        raise ValueError(f"no episode .npz files found under {root}")
    lengths = []
    windows = 0
    for path in paths:
        episode = load_episode(path, metadata)
        length = int(episode[OBSERVATION_KEY].shape[0])
        lengths.append(length)
        windows += (
            length
            - metadata.policy_start_step
            - metadata.horizon
            + 1
        )
    return {
        **metadata.to_dict(),
        "dataset_dir": str(root),
        "episodes": len(paths),
        "steps": int(sum(lengths)),
        "training_windows": int(windows),
        "min_episode_steps": int(min(lengths)),
        "max_episode_steps": int(max(lengths)),
    }


class DiffusionEpisodeDataset(Dataset[Dict[str, torch.Tensor]]):
    """Window full-state histories and expert action chunks from episodes."""

    def __init__(
        self,
        dataset_dir: str | os.PathLike[str],
        *,
        stride: int = 1,
        cache_size: int = 4,
    ) -> None:
        super().__init__()
        if stride <= 0:
            raise ValueError("stride must be positive")
        if cache_size < 0:
            raise ValueError("cache_size cannot be negative")
        self.root = Path(dataset_dir).expanduser().resolve()
        self.metadata = load_metadata(self.root)
        self.episode_paths = discover_episode_paths(self.root)
        if not self.episode_paths:
            raise ValueError(f"no episode .npz files found under {self.root}")
        self.cache_size = cache_size
        self._cache: OrderedDict[int, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._indices: list[Tuple[int, int]] = []

        for episode_index, path in enumerate(self.episode_paths):
            episode = load_episode(path, self.metadata)
            length = int(episode[OBSERVATION_KEY].shape[0])
            self._indices.extend(
                (episode_index, step)
                for step in range(
                    self.metadata.policy_start_step,
                    length - self.metadata.horizon + 1,
                    stride,
                )
            )
            if self.cache_size > 0 and len(self._cache) < self.cache_size:
                self._cache[episode_index] = (
                    episode[OBSERVATION_KEY],
                    episode[ACTION_KEY],
                )
        if not self._indices:
            raise ValueError("dataset does not contain any complete action chunks")

    def __len__(self) -> int:
        return len(self._indices)

    def _episode(self, episode_index: int) -> Tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(episode_index)
        if cached is not None:
            self._cache.move_to_end(episode_index)
            return cached
        episode = load_episode(
            self.episode_paths[episode_index], self.metadata
        )
        arrays = (episode[OBSERVATION_KEY], episode[ACTION_KEY])
        if self.cache_size > 0:
            self._cache[episode_index] = arrays
            self._cache.move_to_end(episode_index)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_index, step = self._indices[index]
        observations, actions = self._episode(episode_index)
        history_start = max(0, step - self.metadata.obs_horizon + 1)
        history = observations[history_start : step + 1]
        if history.shape[0] < self.metadata.obs_horizon:
            padding = np.repeat(
                observations[0:1],
                self.metadata.obs_horizon - history.shape[0],
                axis=0,
            )
            history = np.concatenate((padding, history), axis=0)
        action_chunk = actions[step : step + self.metadata.horizon]
        # Copy because torch.from_numpy warns on read-only archive-backed arrays,
        # and contiguous samples make host-to-device transfer predictable.
        return {
            "obs": torch.from_numpy(np.ascontiguousarray(history, dtype=np.float32)),
            "action": torch.from_numpy(
                np.ascontiguousarray(action_chunk, dtype=np.float32)
            ),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "step": torch.tensor(step, dtype=torch.long),
        }


def compute_normalization_stats(
    dataset_dir: str | os.PathLike[str],
    *,
    min_std: float = MIN_STD,
) -> Dict[str, np.ndarray]:
    """Compute per-channel mean/std over raw episode timesteps.

    Accumulation uses float64 and is streamed episode-by-episode, so generating
    statistics does not require holding the entire dataset in memory.
    """

    if min_std <= 0.0:
        raise ValueError("min_std must be positive")
    root = Path(dataset_dir).expanduser().resolve()
    metadata = load_metadata(root)
    paths = discover_episode_paths(root)
    if not paths:
        raise ValueError(f"no episode .npz files found under {root}")

    obs_sum = np.zeros(metadata.obs_dim, dtype=np.float64)
    obs_sq_sum = np.zeros(metadata.obs_dim, dtype=np.float64)
    action_sum = np.zeros(metadata.action_dim, dtype=np.float64)
    action_sq_sum = np.zeros(metadata.action_dim, dtype=np.float64)
    obs_count = 0
    action_count = 0
    for path in paths:
        episode = load_episode(path, metadata)
        observation_start = max(
            0,
            metadata.policy_start_step
            - metadata.obs_horizon
            + 1,
        )
        obs = episode[OBSERVATION_KEY][
            observation_start:
        ].astype(np.float64, copy=False)
        action = episode[ACTION_KEY][
            metadata.policy_start_step:
        ].astype(np.float64, copy=False)
        obs_sum += obs.sum(axis=0)
        obs_sq_sum += np.square(obs).sum(axis=0)
        action_sum += action.sum(axis=0)
        action_sq_sum += np.square(action).sum(axis=0)
        obs_count += obs.shape[0]
        action_count += action.shape[0]
    if obs_count <= 0 or action_count <= 0:
        raise ValueError("dataset is empty")

    obs_mean = obs_sum / obs_count
    action_mean = action_sum / action_count
    obs_variance = np.maximum(
        obs_sq_sum / obs_count - np.square(obs_mean), 0.0
    )
    action_variance = np.maximum(
        action_sq_sum / action_count - np.square(action_mean), 0.0
    )
    return {
        "obs_mean": obs_mean.astype(np.float32),
        "obs_std": np.maximum(np.sqrt(obs_variance), min_std).astype(np.float32),
        "action_mean": action_mean.astype(np.float32),
        "action_std": np.maximum(np.sqrt(action_variance), min_std).astype(
            np.float32
        ),
        "count": np.asarray(action_count, dtype=np.int64),
        "observation_count": np.asarray(obs_count, dtype=np.int64),
    }


def _create_smoke_dataset(output: Path, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    metadata = DatasetMetadata(
        obs_dim=6,
        action_dim=3,
        obs_horizon=2,
        horizon=4,
        policy_start_step=0,
        dt=0.05,
        task="diffusion_stack_smoke_test",
    )
    write_metadata(output, metadata, overwrite=True)
    for episode_id in range(2):
        steps = 12
        time = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        observations = rng.normal(0.0, 0.1, (steps, metadata.obs_dim)).astype(
            np.float32
        )
        observations[:, 0] += time
        observations[:, 1] += np.sin(time * np.pi * 2.0)
        actions = np.stack(
            (
                0.5 * observations[:, 0],
                -0.25 * observations[:, 1],
                observations[:, 2] + 0.1,
            ),
            axis=-1,
        ).astype(np.float32)
        save_episode(
            output,
            episode_id,
            observations,
            actions,
            extras={
                "phase": np.full(steps, episode_id, dtype=np.int16),
                "success": np.asarray([True]),
            },
            overwrite=True,
        )
    return validate_dataset(output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="validate all episodes and print dataset metadata"
    )
    inspect_parser.add_argument("dataset", type=Path)
    inspect_parser.add_argument(
        "--stats", action="store_true", help="also compute normalization statistics"
    )

    smoke_parser = subparsers.add_parser(
        "create-smoke", help="create a tiny deterministic dataset for tests"
    )
    smoke_parser.add_argument("output", type=Path)
    smoke_parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "inspect":
        result = validate_dataset(args.dataset)
        if args.stats:
            stats = compute_normalization_stats(args.dataset)
            result["normalization"] = {
                key: value.tolist() for key, value in stats.items()
            }
    elif args.command == "create-smoke":
        result = _create_smoke_dataset(args.output, args.seed)
    else:  # pragma: no cover - argparse prevents this branch.
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
