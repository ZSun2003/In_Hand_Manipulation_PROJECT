"""Legacy TQC replay buffer retained for old-checkpoint reproducibility.

The formal GRAB Diffusion pipeline stores elite trajectories through
``demos.py`` and does not import this module.
"""

from __future__ import annotations

from dataclasses import dataclass


def _torch():
    import torch

    return torch


@dataclass
class ReplayBatch:
    observation: object
    action: object
    reward: object
    next_observation: object
    done: object


class CUDAReplayBuffer:
    """A fixed-capacity float32 replay buffer that never leaves ``device``."""

    def __init__(self, *, capacity: int, obs_dim: int, action_dim: int, device: str) -> None:
        if capacity <= 0 or obs_dim <= 0 or action_dim <= 0:
            raise ValueError("capacity, obs_dim, and action_dim must be positive")
        torch = _torch()
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.observation = torch.empty((capacity, obs_dim), device=self.device, dtype=torch.float32)
        self.next_observation = torch.empty_like(self.observation)
        self.action = torch.empty((capacity, action_dim), device=self.device, dtype=torch.float32)
        self.reward = torch.empty((capacity, 1), device=self.device, dtype=torch.float32)
        self.done = torch.empty_like(self.reward)
        self.position = 0
        self.size = 0

    def add(self, observation, action, reward, next_observation, done) -> None:
        torch = _torch()
        count = int(observation.shape[0])
        if count <= 0:
            return
        if count > self.capacity:
            observation, action, reward, next_observation, done = (
                value[-self.capacity :] for value in (observation, action, reward, next_observation, done)
            )
            count = self.capacity
        values = (
            (self.observation, observation),
            (self.action, action),
            (self.reward, reward.reshape(-1, 1)),
            (self.next_observation, next_observation),
            (self.done, done.reshape(-1, 1)),
        )
        # CUDA advanced-index assignment lowers to index_put, which is
        # intentionally rejected by deterministic-algorithm mode even when
        # these ring indices are unique.  At most two contiguous copies are
        # sufficient for a circular buffer and avoid both index_put and the
        # per-step index allocation.
        first = min(count, self.capacity - self.position)
        second = count - first
        with torch.no_grad():
            for target, source in values:
                detached = source.detach()
                target[self.position : self.position + first].copy_(
                    detached[:first]
                )
                if second:
                    target[:second].copy_(detached[first:])
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int) -> ReplayBatch:
        torch = _torch()
        if self.size < batch_size:
            raise ValueError("not enough replay entries")
        indices = torch.randint(self.size, (batch_size,), device=self.device)
        return ReplayBatch(self.observation[indices], self.action[indices], self.reward[indices], self.next_observation[indices], self.done[indices])

    def clear(self) -> None:
        """Forget phase-1 data without reallocating the CUDA storage."""

        self.position = 0
        self.size = 0

    def state_dict(self, *, include_data: bool = False):
        result = {"capacity": self.capacity, "position": self.position, "size": self.size}
        if include_data:
            for name in ("observation", "action", "reward", "next_observation", "done"):
                result[name] = getattr(self, name)[: self.size].detach().cpu()
        return result

    def load_state_dict(self, state) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity differs from checkpoint")
        size, position = int(state["size"]), int(state["position"])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("replay checkpoint has invalid size or position")
        names = ("observation", "action", "reward", "next_observation", "done")
        if size and any(name not in state for name in names):
            raise ValueError("full replay data is missing from checkpoint")
        self.size, self.position = size, position
        for name in names:
            if name in state:
                getattr(self, name)[:size].copy_(state[name].to(self.device))
        self.assert_finite()

    def assert_finite(self) -> None:
        torch = _torch()
        finite = torch.ones((), device=self.device, dtype=torch.bool)
        for name in ("observation", "action", "reward", "next_observation", "done"):
            finite &= torch.isfinite(getattr(self, name)[: self.size]).all()
        if not bool(finite):
            raise FloatingPointError("replay checkpoint contains NaN or infinity")


__all__ = ["CUDAReplayBuffer", "ReplayBatch"]
