"""Host-resident replay with bounded batches returned on the learner device.

The public ``CUDAReplayBuffer`` name is retained for checkpoint/package API
compatibility.  Replay capacity is deliberately stored in ordinary CPU RAM;
only a sampled minibatch is materialised on the configured training device.
This keeps the one-million-transition formal buffer from consuming nearly
2 GiB of scarce device memory without changing its tensor schema or dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass


FINITE_CHECK_CHUNK_ROWS = 65_536
REPLAY_FORMAT = "screw-actuator-residual-replay-v2"


def _torch():
    import torch

    return torch


@dataclass
class ReplayBatch:
    observation: object
    action: object
    reward: object
    next_observation: object
    action_mask: object
    next_action_mask: object
    terminated: object
    truncated: object

    @property
    def done(self):
        """Episode boundary (for logging/reset, not critic bootstrapping)."""

        return self.terminated | self.truncated


class CUDAReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str,
    ) -> None:
        torch = _torch()
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        # ``device`` is the learner device, not the replay storage device.
        # Keep the legacy attribute name because callers already use it as the
        # destination for sampled batches.  Pinning the complete ~2 GiB replay
        # would unnecessarily lock host RAM; bounded batch staging can be
        # added independently if transfer profiling ever justifies it.
        self.device = torch.device(device)
        self.storage_device = torch.device("cpu")
        self.observation = torch.empty(
            (capacity, obs_dim),
            device=self.storage_device,
            dtype=torch.float32,
        )
        self.next_observation = torch.empty_like(self.observation)
        self.action = torch.empty(
            (capacity, action_dim),
            device=self.storage_device,
            dtype=torch.float32,
        )
        self.reward = torch.empty(
            (capacity, 1),
            device=self.storage_device,
            dtype=torch.float32,
        )
        self.action_mask = torch.empty(
            (capacity, action_dim),
            device=self.storage_device,
            dtype=torch.bool,
        )
        self.next_action_mask = torch.empty_like(self.action_mask)
        self.terminated = torch.empty(
            (capacity, 1),
            device=self.storage_device,
            dtype=torch.bool,
        )
        self.truncated = torch.empty_like(self.terminated)
        self.position = 0
        self.size = 0

    @property
    def storage_allocated_bytes(self) -> int:
        """Exact persistent host allocation for the replay tensors."""

        return int(
            sum(
                getattr(self, name).numel()
                * getattr(self, name).element_size()
                for name in (
                    "observation",
                    "action",
                    "reward",
                    "next_observation",
                    "action_mask",
                    "next_action_mask",
                    "terminated",
                    "truncated",
                )
            )
        )

    def sample_allocated_bytes(self, batch_size: int) -> int:
        """Exact tensor payload returned for one learner minibatch."""

        count = int(batch_size)
        if count <= 0:
            raise ValueError("batch_size must be positive")
        return int(
            count
            * (
                2 * self.obs_dim * self.observation.element_size()
                + self.action_dim * self.action.element_size()
                + self.reward.element_size()
                + 2 * self.action_dim * self.action_mask.element_size()
                + 2 * self.terminated.element_size()
            )
        )

    def add(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminated,
        truncated,
        *,
        action_mask,
        next_action_mask,
    ) -> None:
        """Store transitions with the exact masks used by the controller.

        ``terminated`` denotes an absorbing task/physical terminal, while
        ``truncated`` records the explicit episode time limit.  Both end the
        replay sequence and disable bootstrapping into a reset world's state;
        they remain separate for diagnostics and evaluation.
        Masks must come from the executable post-supervisor actuator authority;
        reconstructing them later from the base observation is insufficient.
        """

        torch = _torch()
        count = int(observation.shape[0])
        expected_observation = (count, self.obs_dim)
        expected_action = (count, self.action_dim)
        if tuple(observation.shape) != expected_observation:
            raise ValueError(
                f"observation must have shape {expected_observation}"
            )
        if tuple(next_observation.shape) != expected_observation:
            raise ValueError(
                f"next_observation must have shape {expected_observation}"
            )
        if tuple(action.shape) != expected_action:
            raise ValueError(f"action must have shape {expected_action}")
        action_mask = torch.as_tensor(action_mask)
        next_action_mask = torch.as_tensor(next_action_mask)
        if action_mask.dtype != torch.bool or tuple(action_mask.shape) != expected_action:
            raise ValueError(
                f"action_mask must be boolean with shape {expected_action}"
            )
        if (
            next_action_mask.dtype != torch.bool
            or tuple(next_action_mask.shape) != expected_action
        ):
            raise ValueError(
                "next_action_mask must be boolean with shape "
                f"{expected_action}"
            )
        terminated = torch.as_tensor(terminated).reshape(-1)
        truncated = torch.as_tensor(truncated).reshape(-1)
        if terminated.shape != (count,) or truncated.shape != (count,):
            raise ValueError("termination flags must have one value per transition")
        indices = (
            torch.arange(count, device=self.storage_device) + self.position
        ) % self.capacity
        # Only the bounded insertion batch crosses the device boundary.  The
        # formal workflow inserts at most 256 rows per vector step.
        self.observation[indices] = torch.as_tensor(
            observation,
            device=self.storage_device,
            dtype=self.observation.dtype,
        )
        self.action[indices] = torch.as_tensor(
            action,
            device=self.storage_device,
            dtype=self.action.dtype,
        )
        self.reward[indices, 0] = torch.as_tensor(
            reward,
            device=self.storage_device,
            dtype=self.reward.dtype,
        )
        self.next_observation[indices] = torch.as_tensor(
            next_observation,
            device=self.storage_device,
            dtype=self.next_observation.dtype,
        )
        self.action_mask[indices] = action_mask.to(
            device=self.storage_device
        )
        self.next_action_mask[indices] = next_action_mask.to(
            device=self.storage_device
        )
        self.terminated[indices, 0] = terminated.to(
            device=self.storage_device, dtype=torch.bool
        )
        self.truncated[indices, 0] = truncated.to(
            device=self.storage_device, dtype=torch.bool
        )
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int) -> ReplayBatch:
        torch = _torch()
        if self.size < batch_size:
            raise ValueError("not enough replay entries")
        # Preserve the established CUDA RNG stream and checkpoint behaviour:
        # draw replay indices on the learner device, then transfer only those
        # few integer indices to the host-resident store.
        device_indices = torch.randint(
            self.size, (batch_size,), device=self.device
        )

        indices = device_indices.to(device=self.storage_device)

        def sampled(name: str):
            value = getattr(self, name)[indices]
            return value.to(device=self.device)

        return ReplayBatch(
            observation=sampled("observation"),
            action=sampled("action"),
            reward=sampled("reward"),
            next_observation=sampled("next_observation"),
            action_mask=sampled("action_mask"),
            next_action_mask=sampled("next_action_mask"),
            terminated=sampled("terminated"),
            truncated=sampled("truncated"),
        )

    def state_dict(self, *, include_data: bool = False):
        result = {
            "format": REPLAY_FORMAT,
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "position": self.position,
            "size": self.size,
        }
        if include_data:
            for name in (
                "observation",
                "action",
                "reward",
                "next_observation",
                "action_mask",
                "next_action_mask",
                "terminated",
                "truncated",
            ):
                result[name] = getattr(self, name)[: self.size].detach().cpu()
        return result

    def load_state_dict(self, state) -> None:
        if state.get("format") != REPLAY_FORMAT:
            raise ValueError("replay checkpoint schema is incompatible")
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity differs from checkpoint")
        if int(state.get("obs_dim", -1)) != self.obs_dim:
            raise ValueError("replay observation dimension differs from checkpoint")
        if int(state.get("action_dim", -1)) != self.action_dim:
            raise ValueError("replay action dimension differs from checkpoint")
        size = int(state["size"])
        position = int(state["position"])
        if not 0 <= size <= self.capacity:
            raise ValueError("replay size is outside buffer capacity")
        if not 0 <= position < self.capacity:
            raise ValueError("replay position is outside buffer capacity")
        names = (
            "observation",
            "action",
            "reward",
            "next_observation",
            "action_mask",
            "next_action_mask",
            "terminated",
            "truncated",
        )
        if size and any(name not in state for name in names):
            raise ValueError("full replay data is missing from checkpoint")
        for name in names:
            if name in state:
                target = getattr(self, name)[:size]
                source = state[name]
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        f"replay {name} shape differs from checkpoint: "
                        f"{tuple(source.shape)} != {tuple(target.shape)}"
                    )
                if source.dtype != target.dtype:
                    raise ValueError(
                        f"replay {name} dtype differs from checkpoint: "
                        f"{source.dtype} != {target.dtype}"
                    )
        self.size = size
        self.position = position
        for name in names:
            if name in state:
                target = getattr(self, name)[:size]
                source = state[name]
                # Checkpoints already store replay tensors on CPU.  Retaining
                # that representation makes legacy v2 checkpoints directly
                # loadable without a full-capacity device allocation.
                target.copy_(source)
        self.assert_finite()

    def assert_finite(self) -> None:
        torch = _torch()
        finite = torch.ones(
            (), device=self.storage_device, dtype=torch.bool
        )
        for name in (
            "observation", "action", "reward", "next_observation"
        ):
            tensor = getattr(self, name)
            for start in range(0, self.size, FINITE_CHECK_CHUNK_ROWS):
                stop = min(start + FINITE_CHECK_CHUNK_ROWS, self.size)
                finite &= torch.isfinite(tensor[start:stop]).all()
        if not bool(finite):
            raise FloatingPointError(
                "replay checkpoint contains NaN or infinity"
            )


__all__ = ["CUDAReplayBuffer", "REPLAY_FORMAT", "ReplayBatch"]
