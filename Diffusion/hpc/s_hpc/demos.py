"""Memory-mapped GPU demonstration store and random window sampler."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .config import canonicalize_config_payload
from .contracts import (
    ACTION_DIM,
    OBS_DIM,
    OBS_HORIZON,
    OBS_SCHEMA,
    SCHEMA_HASH,
    TASK_GEOMETRY_HASH,
)


def _atomic_json(payload: Dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _canonicalize_provenance(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    normalized = json.loads(json.dumps(dict(value), allow_nan=False))
    config = normalized.get("config")
    if isinstance(config, Mapping):
        normalized["config"] = canonicalize_config_payload(config)
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DemoStore:
    FORMAT = "screw-mjwarp-demo-v2"

    def __init__(
        self,
        root: str | Path,
        *,
        episodes: int,
        steps: int,
        create: bool,
        provenance: Dict[str, Any] | None = None,
        variable_episode_lengths: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root / "metadata.json"
        self.observation_path = self.root / "observations.npy"
        self.action_path = self.root / "actions.npy"
        self.spawn_path = self.root / "spawn_xy.npy"
        self.socket_mouth_path = self.root / "socket_mouth_xy.npy"
        self.socket_angle_path = self.root / "socket_angle_deg.npy"
        self.valid_steps_path = self.root / "valid_steps.npy"
        if create and not self.metadata_path.exists():
            metadata = {
                "format": self.FORMAT,
                "obs_schema": OBS_SCHEMA,
                "schema_hash": SCHEMA_HASH,
                "task_geometry_hash": TASK_GEOMETRY_HASH,
                "obs_dim": OBS_DIM,
                "obs_horizon": OBS_HORIZON,
                "action_dim": ACTION_DIM,
                "episodes": int(episodes),
                "steps": int(steps),
                "completed": 0,
                "storage_dtype": "float16",
                "provenance": provenance,
            }
            if variable_episode_lengths:
                metadata.update(
                    {
                        "variable_episode_lengths": True,
                        "valid_steps_file": self.valid_steps_path.name,
                    }
                )
            _atomic_json(metadata, self.metadata_path)
            np.lib.format.open_memmap(
                self.observation_path,
                mode="w+",
                dtype=np.float16,
                shape=(episodes, steps + 1, OBS_DIM),
            ).flush()
            np.lib.format.open_memmap(
                self.action_path,
                mode="w+",
                dtype=np.float16,
                shape=(episodes, steps, ACTION_DIM),
            ).flush()
            np.lib.format.open_memmap(
                self.spawn_path,
                mode="w+",
                dtype=np.float32,
                shape=(episodes, 2),
            ).flush()
            np.lib.format.open_memmap(
                self.socket_mouth_path,
                mode="w+",
                dtype=np.float32,
                shape=(episodes, 2),
            ).flush()
            np.lib.format.open_memmap(
                self.socket_angle_path,
                mode="w+",
                dtype=np.float32,
                shape=(episodes,),
            ).flush()
            if variable_episode_lengths:
                np.lib.format.open_memmap(
                    self.valid_steps_path,
                    mode="w+",
                    dtype=np.int32,
                    shape=(episodes,),
                ).flush()
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._validate(
            episodes,
            steps,
            provenance,
            variable_episode_lengths=variable_episode_lengths,
        )
        self.observations = np.load(self.observation_path, mmap_mode="r+")
        self.actions = np.load(self.action_path, mmap_mode="r+")
        self.spawn_xy = np.load(self.spawn_path, mmap_mode="r+")
        self.socket_mouth_xy = np.load(
            self.socket_mouth_path, mmap_mode="r+"
        )
        self.socket_angle_deg = np.load(
            self.socket_angle_path, mmap_mode="r+"
        )
        self.valid_steps = (
            np.load(self.valid_steps_path, mmap_mode="r+")
            if self.metadata.get("variable_episode_lengths", False)
            else None
        )

    def _validate(
        self,
        episodes: int,
        steps: int,
        provenance: Dict[str, Any] | None,
        *,
        variable_episode_lengths: bool,
    ) -> None:
        metadata = self.metadata
        if metadata.get("format") != self.FORMAT:
            raise ValueError("unsupported GPU demo store format")
        if metadata.get("schema_hash") != SCHEMA_HASH:
            raise ValueError("demo store schema hash mismatch")
        if metadata.get("task_geometry_hash") != TASK_GEOMETRY_HASH:
            raise ValueError("demo store task geometry hash mismatch")
        if int(metadata["episodes"]) != episodes or int(metadata["steps"]) != steps:
            raise ValueError(
                "existing demo store shape differs from requested episodes/steps"
            )
        stored_variable_lengths = bool(
            metadata.get("variable_episode_lengths", False)
        )
        if stored_variable_lengths != bool(variable_episode_lengths):
            raise ValueError(
                "existing demo store variable-length mode differs from request"
            )
        if stored_variable_lengths:
            if metadata.get("valid_steps_file") != self.valid_steps_path.name:
                raise ValueError("demo store has an invalid valid_steps file")
            if not self.valid_steps_path.is_file():
                raise ValueError("demo store valid_steps file is missing")
        if provenance is not None and _canonicalize_provenance(
            metadata.get("provenance")
        ) != _canonicalize_provenance(provenance):
            raise ValueError(
                "existing demo store was generated with a different "
                "configuration or seed"
            )

    @property
    def completed(self) -> int:
        return int(self.metadata["completed"])

    def write(
        self,
        start: int,
        observations,
        actions,
        spawn_xy,
        *,
        socket_mouth_xy=None,
        socket_angle_deg=None,
        valid_steps=None,
        commit: bool = True,
    ) -> None:
        count = int(observations.shape[0])
        stop = start + count
        if commit and start != self.completed:
            raise ValueError("demonstrations must be appended in episode order")
        if not commit and start < self.completed:
            raise ValueError("cannot stage demonstrations before completed data")
        if stop > int(self.metadata["episodes"]):
            raise ValueError("demonstration write exceeds dataset capacity")
        if observations.shape != (
            count,
            int(self.metadata["steps"]) + 1,
            OBS_DIM,
        ):
            raise ValueError("observation batch shape mismatch")
        if actions.shape != (
            count,
            int(self.metadata["steps"]),
            ACTION_DIM,
        ):
            raise ValueError("action batch shape mismatch")
        spawn_xy = np.asarray(spawn_xy, dtype=np.float32)
        if spawn_xy.shape != (count, 2):
            raise ValueError("spawn batch shape mismatch")
        if socket_mouth_xy is None:
            socket_mouth_xy = np.full((count, 2), np.nan, dtype=np.float32)
        else:
            socket_mouth_xy = np.asarray(socket_mouth_xy, dtype=np.float32)
        if socket_mouth_xy.shape != (count, 2):
            raise ValueError("socket mouth batch shape mismatch")
        if socket_angle_deg is None:
            socket_angle_deg = np.full((count,), np.nan, dtype=np.float32)
        else:
            socket_angle_deg = np.asarray(
                socket_angle_deg, dtype=np.float32
            ).reshape(-1)
        if socket_angle_deg.shape != (count,):
            raise ValueError("socket angle batch shape mismatch")
        if self.valid_steps is None:
            if valid_steps is not None:
                raise ValueError(
                    "valid_steps requires a variable-length demo store"
                )
        else:
            if valid_steps is None:
                raise ValueError(
                    "variable-length demo writes require valid_steps"
                )
            valid_steps = np.asarray(valid_steps, dtype=np.int32).reshape(-1)
            if valid_steps.shape != (count,):
                raise ValueError("valid_steps batch shape mismatch")
            if np.any(valid_steps <= 0) or np.any(
                valid_steps > int(self.metadata["steps"])
            ):
                raise ValueError("valid_steps values are outside the dataset")
        self.observations[start:stop] = observations
        self.actions[start:stop] = actions
        self.spawn_xy[start:stop] = spawn_xy
        self.socket_mouth_xy[start:stop] = socket_mouth_xy
        self.socket_angle_deg[start:stop] = socket_angle_deg
        if self.valid_steps is not None:
            self.valid_steps[start:stop] = valid_steps
        self.observations.flush()
        self.actions.flush()
        self.spawn_xy.flush()
        self.socket_mouth_xy.flush()
        self.socket_angle_deg.flush()
        if self.valid_steps is not None:
            self.valid_steps.flush()
        if commit:
            self.metadata["completed"] = stop
            _atomic_json(self.metadata, self.metadata_path)
            if stop == int(self.metadata["episodes"]):
                self.finalize()

    def commit(self, start: int, stop: int) -> None:
        """Atomically expose a fully staged vector batch to readers/resume."""

        if int(start) != self.completed:
            raise ValueError("demo commit does not begin at completed cursor")
        if not int(start) < int(stop) <= int(self.metadata["episodes"]):
            raise ValueError("invalid demo commit range")
        self.metadata["completed"] = int(stop)
        _atomic_json(self.metadata, self.metadata_path)
        if int(stop) == int(self.metadata["episodes"]):
            self.finalize()

    def finalize(self) -> None:
        """Bind final metadata to the exact bytes of every data file."""

        if self.completed != int(self.metadata["episodes"]):
            raise ValueError("cannot finalize an incomplete demonstration store")
        hashes = {
            "observations.npy": _sha256_file(self.observation_path),
            "actions.npy": _sha256_file(self.action_path),
            "spawn_xy.npy": _sha256_file(self.spawn_path),
            "socket_mouth_xy.npy": _sha256_file(self.socket_mouth_path),
            "socket_angle_deg.npy": _sha256_file(self.socket_angle_path),
        }
        if self.valid_steps is not None:
            hashes[self.valid_steps_path.name] = _sha256_file(
                self.valid_steps_path
            )
        existing = self.metadata.get("content_sha256")
        if existing is not None and existing != hashes:
            raise ValueError(
                "demonstration content checksum mismatch; dataset is corrupted"
            )
        self.metadata["content_sha256"] = hashes
        _atomic_json(self.metadata, self.metadata_path)


class DemoWindowSampler:
    def __init__(self, root: str | Path, *, seed: int) -> None:
        self.root = Path(root).expanduser().resolve()
        self.metadata = json.loads(
            (self.root / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("format") != DemoStore.FORMAT:
            raise ValueError("not an hpc GPU demo dataset")
        if self.metadata.get("schema_hash") != SCHEMA_HASH:
            raise ValueError("dataset schema mismatch")
        if self.metadata.get("task_geometry_hash") != TASK_GEOMETRY_HASH:
            raise ValueError("dataset task geometry mismatch")
        self.episodes = int(self.metadata["completed"])
        self.steps = int(self.metadata["steps"])
        provenance = self.metadata.get("provenance") or {}
        if "policy_start_step" not in provenance:
            raise ValueError(
                "dataset has no policy_start_step; regenerate legacy demos"
            )
        self.policy_start_step = int(provenance["policy_start_step"])
        if self.episodes <= 0:
            raise ValueError("dataset has no completed episodes")
        if not 0 <= self.policy_start_step < self.steps:
            raise ValueError("dataset policy_start_step is outside the episode")
        if self.episodes != int(self.metadata["episodes"]):
            raise ValueError(
                "formal Diffusion training requires a complete demo dataset"
            )
        expected_hashes = self.metadata.get("content_sha256")
        if not isinstance(expected_hashes, dict):
            raise ValueError(
                "dataset metadata has no finalized content checksums"
            )
        data_paths = {
            "observations.npy": self.root / "observations.npy",
            "actions.npy": self.root / "actions.npy",
            "spawn_xy.npy": self.root / "spawn_xy.npy",
            "socket_mouth_xy.npy": self.root / "socket_mouth_xy.npy",
            "socket_angle_deg.npy": self.root / "socket_angle_deg.npy",
        }
        variable_episode_lengths = bool(
            self.metadata.get("variable_episode_lengths", False)
        )
        valid_steps_file = self.metadata.get("valid_steps_file")
        if variable_episode_lengths:
            if valid_steps_file != "valid_steps.npy":
                raise ValueError("dataset has an invalid valid_steps file")
            data_paths[valid_steps_file] = self.root / valid_steps_file
        elif valid_steps_file is not None:
            raise ValueError(
                "fixed-length dataset unexpectedly declares valid_steps"
            )
        actual_hashes = {
            name: _sha256_file(path)
            for name, path in data_paths.items()
        }
        if actual_hashes != expected_hashes:
            raise ValueError(
                "dataset content checksum mismatch; refuse to train"
            )
        self.observations = np.load(
            self.root / "observations.npy", mmap_mode="r"
        )
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r")
        if variable_episode_lengths:
            raw_valid_steps = np.load(
                self.root / valid_steps_file, mmap_mode="r"
            )
            if raw_valid_steps.shape != (self.episodes,):
                raise ValueError("dataset valid_steps shape mismatch")
            if not np.issubdtype(raw_valid_steps.dtype, np.integer):
                raise ValueError("dataset valid_steps must be integral")
            self.valid_steps = np.asarray(raw_valid_steps, dtype=np.int64)
            if np.any(self.valid_steps <= self.policy_start_step) or np.any(
                self.valid_steps > self.steps
            ):
                raise ValueError(
                    "dataset valid_steps are outside the trainable episode"
                )
        else:
            self.valid_steps = np.full(
                self.episodes, self.steps, dtype=np.int64
            )
        self.rng = np.random.default_rng(seed)
        self.device_observations = None
        self.device_actions = None

    def cache_on_device(
        self,
        device: str,
        *,
        chunk_episodes: int = 32,
    ) -> None:
        """Stage the completed float16 dataset once for random CUDA gathers."""

        import torch

        if chunk_episodes <= 0:
            raise ValueError("chunk_episodes must be positive")
        selected = torch.device(device)
        self.device_observations = torch.empty(
            (self.episodes, self.steps + 1, OBS_DIM),
            device=selected,
            dtype=torch.float16,
        )
        self.device_actions = torch.empty(
            (self.episodes, self.steps, ACTION_DIM),
            device=selected,
            dtype=torch.float16,
        )
        for start in range(0, self.episodes, chunk_episodes):
            stop = min(self.episodes, start + chunk_episodes)
            # Copy each mmap slice into a small writable host buffer.  This
            # avoids retaining a multi-gigabyte host duplicate and avoids
            # PyTorch's non-writable NumPy warning.
            observations = np.array(
                self.observations[start:stop],
                dtype=np.float16,
                copy=True,
                order="C",
            )
            actions = np.array(
                self.actions[start:stop],
                dtype=np.float16,
                copy=True,
                order="C",
            )
            self.device_observations[start:stop].copy_(
                torch.from_numpy(observations).to(selected)
            )
            self.device_actions[start:stop].copy_(
                torch.from_numpy(actions).to(selected)
            )

    def normalization(self, *, chunk_episodes: int = 8) -> Dict[str, np.ndarray]:
        obs_sum = np.zeros(OBS_DIM, dtype=np.float64)
        obs_sq = np.zeros(OBS_DIM, dtype=np.float64)
        action_sum = np.zeros(ACTION_DIM, dtype=np.float64)
        action_sq = np.zeros(ACTION_DIM, dtype=np.float64)
        obs_count = 0
        action_count = 0
        if bool(self.metadata.get("variable_episode_lengths", False)):
            for episode, valid_steps in enumerate(self.valid_steps):
                stop = int(valid_steps)
                obs = np.asarray(
                    self.observations[
                        episode, self.policy_start_step : stop + 1
                    ],
                    dtype=np.float32,
                ).reshape(-1, OBS_DIM)
                action = np.asarray(
                    self.actions[
                        episode, self.policy_start_step : stop
                    ],
                    dtype=np.float32,
                ).reshape(-1, ACTION_DIM)
                obs_sum += obs.sum(axis=0, dtype=np.float64)
                obs_sq += np.square(obs, dtype=np.float64).sum(axis=0)
                action_sum += action.sum(axis=0, dtype=np.float64)
                action_sq += np.square(action, dtype=np.float64).sum(axis=0)
                obs_count += obs.shape[0]
                action_count += action.shape[0]
        else:
            for start in range(0, self.episodes, chunk_episodes):
                stop = min(self.episodes, start + chunk_episodes)
                obs = np.asarray(
                    self.observations[
                        start:stop, self.policy_start_step :
                    ],
                    dtype=np.float32,
                ).reshape(-1, OBS_DIM)
                action = np.asarray(
                    self.actions[
                        start:stop, self.policy_start_step :
                    ],
                    dtype=np.float32,
                ).reshape(-1, ACTION_DIM)
                obs_sum += obs.sum(axis=0, dtype=np.float64)
                obs_sq += np.square(obs, dtype=np.float64).sum(axis=0)
                action_sum += action.sum(axis=0, dtype=np.float64)
                action_sq += np.square(action, dtype=np.float64).sum(axis=0)
                obs_count += obs.shape[0]
                action_count += action.shape[0]
        obs_mean = obs_sum / obs_count
        action_mean = action_sum / action_count
        obs_var = np.maximum(obs_sq / obs_count - obs_mean**2, 1.0e-8)
        action_var = np.maximum(
            action_sq / action_count - action_mean**2, 1.0e-12
        )
        return {
            "obs_mean": obs_mean.astype(np.float32),
            "obs_std": np.sqrt(obs_var).astype(np.float32),
            "action_mean": action_mean.astype(np.float32),
            "action_std": np.sqrt(action_var).astype(np.float32),
        }

    def window_count(self, horizon: int) -> int:
        if horizon <= 0:
            raise ValueError("action horizon must be positive")
        counts = self.valid_steps - self.policy_start_step - horizon + 1
        return int(np.maximum(counts, 0).sum())

    def sample(self, batch_size: int, horizon: int, *, device: str):
        import torch

        if horizon > self.steps - self.policy_start_step:
            raise ValueError("action horizon exceeds episode length")
        window_counts = np.maximum(
            self.valid_steps - self.policy_start_step - horizon + 1, 0
        )
        total_windows = int(window_counts.sum())
        if total_windows <= 0:
            raise ValueError("dataset has no complete action windows")
        flat_window = self.rng.integers(
            0, total_windows, size=batch_size
        )
        cumulative = np.cumsum(window_counts)
        episode = np.searchsorted(cumulative, flat_window, side="right")
        previous_episode_end = np.where(
            episode == 0, 0, cumulative[episode - 1]
        )
        timestep = (
            self.policy_start_step + flat_window - previous_episode_end
        )
        previous = np.maximum(timestep - 1, 0)
        if self.device_observations is not None:
            selected = torch.device(device)
            if self.device_observations.device != selected:
                raise ValueError("demo CUDA cache is on a different device")
            episode_index = torch.from_numpy(episode).to(selected)
            timestep_index = torch.from_numpy(timestep).to(selected)
            previous_index = torch.from_numpy(previous).to(selected)
            observation = torch.stack(
                (
                    self.device_observations[
                        episode_index, previous_index
                    ],
                    self.device_observations[
                        episode_index, timestep_index
                    ],
                ),
                dim=1,
            ).to(torch.float32)
            action_indices = timestep_index[:, None] + torch.arange(
                horizon, device=selected
            )[None, :]
            action = self.device_actions[
                episode_index[:, None], action_indices
            ].to(torch.float32)
            return observation, action
        observation = np.stack(
            (
                self.observations[episode, previous],
                self.observations[episode, timestep],
            ),
            axis=1,
        ).astype(np.float32)
        action = np.stack(
            [
                self.actions[episode, timestep + offset]
                for offset in range(horizon)
            ],
            axis=1,
        ).astype(np.float32)
        return (
            torch.from_numpy(observation).to(device, non_blocking=True),
            torch.from_numpy(action).to(device, non_blocking=True),
        )


__all__ = ["DemoStore", "DemoWindowSampler"]
