"""Batched receding-horizon runtime for the GRAB Diffusion policy.

The controller performs one full-world DDIM inference every four control
ticks.  Partial MJWarp resets receive private override chunks, so asynchronous
episode boundaries do not perturb plans for unrelated worlds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .diffusion_policy import (
    DDIM_INFERENCE_STEPS,
    REPLAN_INTERVAL,
    DiffusionPolicy,
    load_policy_checkpoint,
)


class BatchedObservationHistory:
    """Device-resident fixed-width observation history with masked resets."""

    def __init__(
        self,
        *,
        worlds: int,
        obs_horizon: int,
        obs_dim: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if int(worlds) <= 0:
            raise ValueError("worlds must be positive")
        if int(obs_horizon) <= 0 or int(obs_dim) <= 0:
            raise ValueError("observation dimensions must be positive")
        self.worlds = int(worlds)
        self.obs_horizon = int(obs_horizon)
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        self.values = torch.zeros(
            (self.worlds, self.obs_horizon, self.obs_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.initialized = torch.zeros(
            self.worlds,
            device=self.device,
            dtype=torch.bool,
        )

    def _observation(self, observation: Any) -> torch.Tensor:
        value = torch.as_tensor(
            observation,
            device=self.device,
            dtype=self.dtype,
        )
        if tuple(value.shape) != (self.worlds, self.obs_dim):
            raise ValueError(
                "observation must have shape "
                f"[{self.worlds}, {self.obs_dim}]"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError("observation contains NaN or infinity")
        return value

    def _mask(self, mask: Any) -> torch.Tensor:
        value = torch.as_tensor(
            mask,
            device=self.device,
            dtype=torch.bool,
        )
        if tuple(value.shape) != (self.worlds,):
            raise ValueError(f"mask must have shape [{self.worlds}]")
        return value

    def reset(self, mask: Any, observation: Any) -> torch.Tensor:
        """Fill every history slot for reset worlds with their reset state."""

        selected = self._mask(mask)
        current = self._observation(observation)
        if bool(selected.any()):
            self.values[selected] = current[selected, None, :].expand(
                -1,
                self.obs_horizon,
                -1,
            )
            self.initialized[selected] = True
        return self.values

    def append(
        self,
        observation: Any,
        *,
        reset_mask: Optional[Any] = None,
    ) -> torch.Tensor:
        """Append one vector observation, optionally resetting selected rows."""

        current = self._observation(observation)
        if reset_mask is None:
            selected = ~self.initialized
        else:
            selected = self._mask(reset_mask) | (~self.initialized)
        keep = ~selected
        if bool(keep.any()):
            self.values[keep, :-1] = self.values[keep, 1:].clone()
            self.values[keep, -1] = current[keep]
        if bool(selected.any()):
            self.values[selected] = current[selected, None, :].expand(
                -1,
                self.obs_horizon,
                -1,
            )
        self.initialized.fill_(True)
        return self.values

    def state_dict(self) -> dict[str, Any]:
        return {
            "values": self.values.detach().clone(),
            "initialized": self.initialized.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name, target in (
            ("values", self.values),
            ("initialized", self.initialized),
        ):
            source = state.get(name)
            if source is None or tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"observation history state {name!r} shape mismatch"
                )
            target.copy_(
                source.to(device=self.device, dtype=target.dtype)
            )


class BatchedDiffusionController:
    """One Diffusion action-chunk controller shared by all MJWarp worlds."""

    def __init__(
        self,
        policy: DiffusionPolicy,
        *,
        worlds: int,
        inference_steps: int = DDIM_INFERENCE_STEPS,
        replan_interval: int = REPLAN_INTERVAL,
        seed: int = 7,
        action_clip: bool = True,
    ) -> None:
        if int(worlds) <= 0:
            raise ValueError("worlds must be positive")
        if not 1 <= int(inference_steps) <= policy.num_diffusion_steps:
            raise ValueError(
                "inference_steps must lie within the DDPM schedule"
            )
        if not 1 <= int(replan_interval) <= policy.horizon:
            raise ValueError(
                "replan_interval must lie within the policy horizon"
            )
        self.policy = policy.eval()
        self.device = next(policy.parameters()).device
        self.worlds = int(worlds)
        self.inference_steps = int(inference_steps)
        self.replan_interval = int(replan_interval)
        self.seed = int(seed)
        self.action_clip = bool(action_clip)
        self.plan = torch.zeros(
            (self.worlds, policy.horizon, policy.action_dim),
            device=self.device,
        )
        self.override_plan = torch.zeros_like(self.plan)
        self.override_cursor = torch.zeros(
            self.worlds,
            device=self.device,
            dtype=torch.long,
        )
        self.override_active = torch.zeros(
            self.worlds,
            device=self.device,
            dtype=torch.bool,
        )
        self.cursor = self.replan_interval
        self.sample_count = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        worlds: int,
        device: str | torch.device = "auto",
        inference_steps: int = DDIM_INFERENCE_STEPS,
        replan_interval: int = REPLAN_INTERVAL,
        seed: int = 7,
        use_ema: bool = True,
        action_clip: bool = True,
        require_environment_dimensions: bool = True,
    ) -> "BatchedDiffusionController":
        policy, _ = load_policy_checkpoint(
            checkpoint_path,
            device=device,
            use_ema=use_ema,
            require_environment_dimensions=require_environment_dimensions,
        )
        return cls(
            policy,
            worlds=worlds,
            inference_steps=inference_steps,
            replan_interval=replan_interval,
            seed=seed,
            action_clip=action_clip,
        )

    def _mask(self, mask: Any) -> torch.Tensor:
        value = torch.as_tensor(
            mask,
            device=self.device,
            dtype=torch.bool,
        )
        if tuple(value.shape) != (self.worlds,):
            raise ValueError(f"reset mask must have shape [{self.worlds}]")
        return value

    def _history(self, observation_history: Any) -> torch.Tensor:
        value = torch.as_tensor(
            observation_history,
            device=self.device,
            dtype=next(self.policy.parameters()).dtype,
        )
        expected = (
            self.worlds,
            self.policy.obs_horizon,
            self.policy.obs_dim,
        )
        if tuple(value.shape) != expected:
            raise ValueError(
                f"observation_history must have shape {expected}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError("observation history contains NaN or infinity")
        return value

    @torch.no_grad()
    def reset(
        self,
        mask: Any,
        observation_history: Optional[Any] = None,
    ) -> None:
        """Reset selected worlds without resampling unrelated world plans."""

        selected = self._mask(mask)
        if not bool(selected.any()):
            return
        if observation_history is None:
            sampled = torch.zeros_like(self.override_plan[selected])
        else:
            history = self._history(observation_history)
            sampled = self.policy.sample(
                history[selected],
                num_inference_steps=self.inference_steps,
                eta=0.0,
                seed=self.seed + self.sample_count,
            )
            self.sample_count += 1
        if bool(selected.all()):
            self.plan.copy_(sampled)
            self.cursor = 0
            self.override_active.zero_()
            self.override_cursor.zero_()
            return
        self.override_plan[selected] = sampled
        self.override_cursor[selected] = 0
        self.override_active[selected] = True

    def state_dict(self) -> dict[str, Any]:
        """Return all state required for exact receding-horizon continuation."""

        return {
            "plan": self.plan.detach().clone(),
            "override_plan": self.override_plan.detach().clone(),
            "override_cursor": self.override_cursor.detach().clone(),
            "override_active": self.override_active.detach().clone(),
            "cursor": int(self.cursor),
            "sample_count": int(self.sample_count),
            "seed": int(self.seed),
            "worlds": int(self.worlds),
            "inference_steps": int(self.inference_steps),
            "replan_interval": int(self.replan_interval),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name, expected in (
            ("worlds", self.worlds),
            ("inference_steps", self.inference_steps),
            ("replan_interval", self.replan_interval),
        ):
            if name in state and int(state[name]) != expected:
                raise ValueError(
                    f"Diffusion controller state {name!r} mismatch"
                )
        for name, target in (
            ("plan", self.plan),
            ("override_plan", self.override_plan),
            ("override_cursor", self.override_cursor),
            ("override_active", self.override_active),
        ):
            source = state.get(name)
            if source is None or tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"Diffusion controller state {name!r} shape mismatch"
                )
            target.copy_(
                source.to(device=self.device, dtype=target.dtype)
            )
        cursor = int(state.get("cursor", -1))
        if not 0 <= cursor <= self.replan_interval:
            raise ValueError("Diffusion controller cursor is invalid")
        sample_count = int(state.get("sample_count", -1))
        if sample_count < 0:
            raise ValueError("Diffusion controller sample_count is invalid")
        self.cursor = cursor
        self.sample_count = sample_count
        if "seed" in state:
            self.seed = int(state["seed"])

    @torch.no_grad()
    def act(self, observation_history: Any) -> torch.Tensor:
        history = self._history(observation_history)
        if self.cursor >= self.replan_interval:
            self.plan.copy_(
                self.policy.sample(
                    history,
                    num_inference_steps=self.inference_steps,
                    eta=0.0,
                    seed=self.seed + self.sample_count,
                )
            )
            self.cursor = 0
            self.sample_count += 1
            self.override_active.zero_()
            self.override_cursor.zero_()
        base_action = self.plan[:, self.cursor]
        rows = torch.arange(self.worlds, device=self.device)
        override_index = self.override_cursor.clamp(
            max=self.policy.horizon - 1
        )
        override_action = self.override_plan[rows, override_index]
        action = torch.where(
            self.override_active[:, None],
            override_action,
            base_action,
        )
        self.cursor += 1
        next_override = (
            self.override_cursor
            + self.override_active.to(torch.long)
        )
        self.override_cursor.copy_(next_override)
        self.override_active &= next_override < self.replan_interval
        if self.action_clip:
            action = action.clamp(-1.0, 1.0)
        return action


__all__ = [
    "BatchedDiffusionController",
    "BatchedObservationHistory",
]
