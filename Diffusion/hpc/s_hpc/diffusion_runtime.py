"""Batched CUDA inference for the frozen Diffusion baseline."""

from __future__ import annotations

from pathlib import Path

from diffusion_policy import DiffusionPolicy

from .contracts import validate_checkpoint


def _torch():
    import torch

    return torch


class BatchedDiffusionController:
    def __init__(
        self,
        policy: DiffusionPolicy,
        *,
        worlds: int,
        inference_steps: int,
        replan_interval: int,
        seed: int,
    ) -> None:
        torch = _torch()
        self.policy = policy.eval()
        self.device = next(policy.parameters()).device
        self.worlds = int(worlds)
        self.inference_steps = int(inference_steps)
        self.replan_interval = int(replan_interval)
        if not 1 <= self.replan_interval <= policy.horizon:
            raise ValueError("replan_interval must lie within the policy horizon")
        self.plan = torch.zeros(
            (worlds, policy.horizon, policy.action_dim), device=self.device
        )
        self.override_plan = torch.zeros_like(self.plan)
        self.override_cursor = torch.zeros(
            worlds, device=self.device, dtype=torch.long
        )
        self.override_active = torch.zeros(
            worlds, device=self.device, dtype=torch.bool
        )
        self.cursor = self.replan_interval
        self.seed = int(seed)
        self.sample_count = 0

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        worlds: int,
        inference_steps: int,
        replan_interval: int,
        device: str,
        seed: int,
    ) -> "BatchedDiffusionController":
        torch = _torch()
        checkpoint = torch.load(
            Path(path).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        validate_checkpoint(checkpoint, kind="diffusion")
        policy = DiffusionPolicy(**checkpoint["model_config"])
        state = checkpoint.get("ema_state") or checkpoint["model_state"]
        policy.load_state_dict(state, strict=True)
        policy.to(torch.device(device))
        return cls(
            policy,
            worlds=worlds,
            inference_steps=inference_steps,
            replan_interval=replan_interval,
            seed=seed,
        )

    def reset(self, mask, observation_history=None) -> None:
        torch = _torch()
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if not mask.any():
            return
        if observation_history is None:
            sampled = torch.zeros_like(self.override_plan[mask])
        else:
            # Asynchronous environment reset has already synchronized the
            # terminal mask.  Replan only those worlds here; regular policy
            # calls retain one static full-batch replan every N control steps.
            sampled = self.policy.sample(
                observation_history[mask],
                num_inference_steps=self.inference_steps,
                eta=0.0,
                seed=self.seed + self.sample_count,
            )
            self.sample_count += 1
        if mask.all():
            self.plan.copy_(sampled)
            self.cursor = 0
            self.override_active.zero_()
            return
        self.override_plan[mask] = sampled
        self.override_cursor[mask] = 0
        self.override_active[mask] = True

    def state_dict(self):
        return {
            "plan": self.plan.detach().clone(),
            "override_plan": self.override_plan.detach().clone(),
            "override_cursor": self.override_cursor.detach().clone(),
            "override_active": self.override_active.detach().clone(),
            "cursor": int(self.cursor),
            "sample_count": int(self.sample_count),
        }

    def provenance(self):
        """Return the planning contract used by the formal ablation."""

        return {
            "schema": "screw-receding-horizon-diffusion-planner-v1",
            "algorithm": "conditional diffusion policy",
            "action_horizon_ticks": int(self.policy.horizon),
            "inference_steps": int(self.inference_steps),
            "replan_interval_ticks": int(self.replan_interval),
            "replanning": True,
            "plan_scope": "ALIGN task-space increments",
        }

    def load_state_dict(self, state) -> None:
        tensors = (
            ("plan", self.plan),
            ("override_plan", self.override_plan),
            ("override_cursor", self.override_cursor),
            ("override_active", self.override_active),
        )
        for name, target in tensors:
            source = state.get(name)
            if source is None or tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"Diffusion controller state {name!r} shape mismatch"
                )
            target.copy_(source.to(device=self.device, dtype=target.dtype))
        cursor = int(state.get("cursor", -1))
        if not 0 <= cursor <= self.replan_interval:
            raise ValueError("Diffusion controller cursor is invalid")
        self.cursor = cursor
        self.sample_count = int(state.get("sample_count", 0))

    @_torch().no_grad()
    def act(self, observation_history):
        torch = _torch()
        if self.cursor >= self.replan_interval:
            self.plan.copy_(
                self.policy.sample(
                    observation_history,
                    num_inference_steps=self.inference_steps,
                    eta=0.0,
                    seed=self.seed + self.sample_count,
                )
            )
            self.cursor = 0
            self.sample_count += 1
            self.override_active.zero_()
        base_action = self.plan[:, self.cursor]
        row = torch.arange(self.worlds, device=self.device)
        override_index = self.override_cursor.clamp(
            max=self.policy.horizon - 1
        )
        override_action = self.override_plan[row, override_index]
        action = torch.where(
            self.override_active[:, None],
            override_action,
            base_action,
        )
        self.cursor += 1
        next_override = self.override_cursor + self.override_active.to(
            torch.long
        )
        self.override_cursor.copy_(next_override)
        self.override_active &= (
            next_override < self.replan_interval
        )
        return action


__all__ = ["BatchedDiffusionController"]
