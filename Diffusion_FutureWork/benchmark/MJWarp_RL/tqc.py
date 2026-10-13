"""Legacy TQC learner retained only for old-checkpoint reproducibility.

The formal GRAB pipeline uses ``diffusion_policy.py`` and does not import this
module.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict


def _torch():
    import torch

    return torch


def _cpu(value):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def _require_finite(label: str, *values) -> None:
    """Raise once for an entire tensor group, avoiding per-tensor syncs."""

    torch = _torch()
    finite = torch.ones((), device=values[0].device, dtype=torch.bool)
    for value in values:
        finite &= torch.isfinite(value).all()
    if not bool(finite):
        raise FloatingPointError(f"{label} became NaN or infinity")


@dataclass(frozen=True)
class TQCConfig:
    """TQC settings; defaults are two critics, 25 quantiles, drop two each."""

    learning_starts: int = 65_536
    batch_size: int = 1_024
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    gamma: float = 0.99
    tau: float = 0.005
    critics: int = 2
    quantiles: int = 25
    top_quantiles_to_drop: int = 2
    hidden_dim: int = 512

    def validate(self) -> None:
        if self.critics != 2 or self.quantiles != 25:
            raise ValueError("this TQC profile requires 2 critics x 25 quantiles")
        if self.top_quantiles_to_drop != 2:
            raise ValueError("this TQC profile requires dropping 2 quantiles/critic")
        if self.learning_starts < 0 or self.batch_size <= 0 or self.hidden_dim <= 0:
            raise ValueError("TQC sizes must be positive")
        if min(self.actor_lr, self.critic_lr, self.alpha_lr) <= 0.0:
            raise ValueError("TQC learning rates must be positive")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("gamma and tau must be in (0, 1]")


def _tqc_config(config: Any) -> Any:
    """Accept either a TQCConfig or a project config exposing ``.tqc``."""

    result = getattr(config, "tqc", config)
    required = (
        "learning_starts", "batch_size", "actor_lr", "critic_lr",
        "alpha_lr", "gamma", "tau", "critics", "quantiles",
        "top_quantiles_to_drop", "hidden_dim",
    )
    missing = [name for name in required if not hasattr(result, name)]
    if missing:
        raise TypeError("config lacks TQC fields: " + ", ".join(missing))
    validate = getattr(result, "validate", None)
    if validate is not None:
        validate()
    return result


class SquashedGaussianActor(_torch().nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        initial_log_std: float = -1.5,
    ) -> None:
        torch = _torch()
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim), torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(),
        )
        self.mean = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std = torch.nn.Linear(hidden_dim, action_dim)
        # IK is already the stable nominal controller.  Begin exactly at the
        # zero residual with modest exploration instead of a random mean and
        # unit Gaussian noise.
        torch.nn.init.zeros_(self.mean.weight)
        torch.nn.init.zeros_(self.mean.bias)
        torch.nn.init.zeros_(self.log_std.weight)
        torch.nn.init.constant_(self.log_std.bias, float(initial_log_std))

    def forward(self, observation):
        hidden = self.backbone(observation)
        return self.mean(hidden), self.log_std(hidden).clamp(-20.0, 2.0)

    def sample(
        self,
        observation,
        *,
        deterministic: bool = False,
        action_mask=None,
    ):
        torch = _torch()
        mean, log_std = self(observation)
        pre_tanh = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        raw_action = torch.tanh(pre_tanh)
        if action_mask is None:
            action_mask = torch.ones_like(raw_action, dtype=torch.bool)
        mask = action_mask.to(raw_action.dtype)
        action = raw_action * mask
        if deterministic:
            log_prob = torch.zeros((observation.shape[0], 1), device=observation.device, dtype=observation.dtype)
        else:
            gaussian = -0.5 * (((pre_tanh - mean) / log_std.exp()).square() + 2.0 * log_std + math.log(2.0 * math.pi))
            log_prob = (
                (
                    gaussian
                    - torch.log(1.0 - raw_action.square() + 1.0e-6)
                )
                * mask
            ).sum(dim=-1, keepdim=True)
        return action, log_prob


class QuantileCritic(_torch().nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, quantiles: int) -> None:
        torch = _torch()
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + action_dim, hidden_dim), torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, quantiles),
        )

    def forward(self, observation, action):
        return self.network(_torch().cat((observation, action), dim=-1))


class QuantileCritics(_torch().nn.Module):
    def __init__(self, *, count: int, obs_dim: int, action_dim: int, hidden_dim: int, quantiles: int) -> None:
        torch = _torch()
        super().__init__()
        self.networks = torch.nn.ModuleList(QuantileCritic(obs_dim, action_dim, hidden_dim, quantiles) for _ in range(count))

    def forward(self, observation, action):
        return _torch().stack([network(observation, action) for network in self.networks], dim=1)


def quantile_huber_loss(prediction, target):
    """Pairwise quantile Huber regression used by TQC."""

    torch = _torch()
    quantiles = prediction.shape[-1]
    delta = target[:, None, None, :] - prediction[:, :, :, None]
    absolute = delta.abs()
    huber = torch.where(absolute <= 1.0, 0.5 * delta.square(), absolute - 0.5)
    tau = (torch.arange(quantiles, device=prediction.device, dtype=prediction.dtype) + 0.5) / quantiles
    weight = torch.abs(tau[None, None, :, None] - (delta.detach() < 0.0).to(prediction.dtype))
    return (weight * huber).mean()


class TQCTrainer:
    """TQC state for caller-supplied ``obs_dim``, ``action_dim``, and config."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        config: Any,
        device: str,
        action_mask_fn=None,
    ) -> None:
        torch = _torch()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        self.config = config
        self.tqc = _tqc_config(config)
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.device = torch.device(device)
        self.action_mask_fn = action_mask_fn
        c = self.tqc
        self.actor = SquashedGaussianActor(
            self.obs_dim,
            self.action_dim,
            c.hidden_dim,
            getattr(c, "initial_log_std", -1.5),
        ).to(self.device)
        self.critics = QuantileCritics(count=c.critics, obs_dim=self.obs_dim, action_dim=self.action_dim, hidden_dim=c.hidden_dim, quantiles=c.quantiles).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).eval()
        for parameter in self.target_critics.parameters():
            parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=c.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=c.critic_lr)
        self.log_alpha = torch.zeros((), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=c.alpha_lr)
        self.updates = self.transitions = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _action_mask(self, observation):
        torch = _torch()
        if self.action_mask_fn is None:
            return torch.ones(
                (observation.shape[0], self.action_dim),
                device=observation.device,
                dtype=torch.bool,
            )
        mask = self.action_mask_fn(observation)
        if mask.shape != (observation.shape[0], self.action_dim):
            raise ValueError("policy action mask has the wrong shape")
        return mask.to(device=observation.device, dtype=torch.bool)

    @_torch().no_grad()
    def act(self, observation, *, deterministic: bool = False):
        return self.actor.sample(
            observation,
            deterministic=deterministic,
            action_mask=self._action_mask(observation),
        )[0]

    def update(self, batch) -> Dict[str, object]:
        torch = _torch()
        c = self.tqc
        _require_finite("TQC replay batch", batch.observation, batch.action, batch.reward, batch.next_observation, batch.done)
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(
                batch.next_observation,
                action_mask=self._action_mask(batch.next_observation),
            )
            target_quantiles = self.target_critics(batch.next_observation, next_action).flatten(start_dim=1)
            target_quantiles, _ = torch.sort(target_quantiles, dim=-1)
            total_drop = c.top_quantiles_to_drop * c.critics
            target_quantiles = target_quantiles[:, :-total_drop]
            target = batch.reward + c.gamma * (1.0 - batch.done) * (target_quantiles - self.alpha.detach() * next_log_prob)
            _require_finite("TQC target", target)
        prediction = self.critics(batch.observation, batch.action)
        critic_loss = quantile_huber_loss(prediction, target)
        _require_finite("TQC critic loss", critic_loss)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.critics.parameters(), getattr(c, "max_grad_norm", 10.0)
        )
        _require_finite("TQC critic gradient", critic_grad)
        self.critic_optimizer.step()
        for parameter in self.critics.parameters():
            parameter.requires_grad_(False)
        try:
            action_mask = self._action_mask(batch.observation)
            action, log_prob = self.actor.sample(
                batch.observation, action_mask=action_mask
            )
            actor_value = self.critics(batch.observation, action).mean(dim=(1, 2), keepdim=False)[:, None]
            actor_loss = (self.alpha.detach() * log_prob - actor_value).mean()
            _require_finite("TQC actor loss", actor_loss)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), getattr(c, "max_grad_norm", 10.0)
            )
            _require_finite("TQC actor gradient", actor_grad)
            self.actor_optimizer.step()
        finally:
            for parameter in self.critics.parameters():
                parameter.requires_grad_(True)
        target_entropy = -getattr(c, "target_entropy_scale", 1.0) * (
            action_mask.sum(dim=-1, keepdim=True).to(log_prob.dtype)
        )
        alpha_loss = -(self.log_alpha * (log_prob.detach() + target_entropy)).mean()
        _require_finite("TQC temperature loss", alpha_loss)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        _require_finite("TQC temperature gradient", self.log_alpha.grad)
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critics.parameters(), self.critics.parameters()):
                target_parameter.lerp_(parameter, c.tau)
            q_prediction = prediction.detach().flatten()
            q_target = target.detach().flatten()
            td_absolute = (prediction.detach().mean(dim=(1, 2)) - target.detach().mean(dim=1)).abs()
            diagnostics = {"q_pred_mean": q_prediction.mean(), "q_target_mean": q_target.mean(), "td_abs_mean": td_absolute.mean(), "td_abs_p95": torch.quantile(td_absolute, 0.95), "policy_log_prob_mean": log_prob.detach().mean(), "policy_entropy": -log_prob.detach().mean(), "batch_reward_mean": batch.reward.mean(), "batch_done_fraction": batch.done.mean()}
            _require_finite("TQC diagnostic metrics", *diagnostics.values())
        self.updates += 1
        return {"critic_loss": critic_loss.detach(), "actor_loss": actor_loss.detach(), "alpha_loss": alpha_loss.detach(), "alpha": self.alpha.detach(), "critic_grad_norm": critic_grad.detach(), "actor_grad_norm": actor_grad.detach(), "active_action_dimensions": action_mask.sum(dim=-1).to(torch.float32).mean(), **diagnostics}

    def assert_finite(self) -> None:
        tensors = [self.log_alpha, *self.actor.parameters(), *self.critics.parameters(), *self.target_critics.parameters()]
        _require_finite("TQC parameters", *tensors)

    def checkpoint(self, replay=None, *, include_replay: bool = False) -> Dict[str, object]:
        config = asdict(self.tqc) if is_dataclass(self.tqc) else dict(vars(self.tqc))
        payload = {"format": "mjwarp-tqc-v1", "obs_dim": self.obs_dim, "action_dim": self.action_dim, "config": config, "actor": _cpu(self.actor.state_dict()), "critics": _cpu(self.critics.state_dict()), "target_critics": _cpu(self.target_critics.state_dict()), "actor_optimizer": _cpu(self.actor_optimizer.state_dict()), "critic_optimizer": _cpu(self.critic_optimizer.state_dict()), "alpha_optimizer": _cpu(self.alpha_optimizer.state_dict()), "log_alpha": _cpu(self.log_alpha), "updates": self.updates, "transitions": self.transitions}
        if replay is not None:
            payload["replay"] = replay.state_dict(include_data=include_replay)
        return payload

    def load(self, payload, replay=None) -> None:
        if payload.get("format") != "mjwarp-tqc-v1" or (payload.get("obs_dim"), payload.get("action_dim")) != (self.obs_dim, self.action_dim):
            raise ValueError("incompatible TQC checkpoint")
        self.actor.load_state_dict(payload["actor"])
        self.critics.load_state_dict(payload["critics"])
        self.target_critics.load_state_dict(payload["target_critics"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        with _torch().no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        self.updates, self.transitions = int(payload.get("updates", 0)), int(payload.get("transitions", 0))
        if replay is not None and "replay" in payload:
            replay.load_state_dict(payload["replay"])
        self.assert_finite()


__all__ = ["QuantileCritics", "SquashedGaussianActor", "TQCConfig", "TQCTrainer", "quantile_huber_loss"]
