"""Self-contained Torch CUDA Truncated Quantile Critics implementation."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict

from .actuator_residual import ACTUATOR_RESIDUAL_DIM
from .contracts import (
    OBS_DIM,
    TQC_CONTROLLER_STATE_NAMES,
    TQC_OBS_DIM,
    TQC_OBS_SCHEMA,
    checkpoint_metadata,
    validate_checkpoint,
)


TQC_ACTION_DIM = ACTUATOR_RESIDUAL_DIM
TQC_CONTROLLER_STATE_DIM = len(TQC_CONTROLLER_STATE_NAMES)


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
    torch = _torch()
    finite = torch.ones(
        (), device=values[0].device, dtype=torch.bool
    )
    for value in values:
        finite &= torch.isfinite(value).all()
    # One host check for the complete group, not one synchronization per
    # parameter/tensor.
    if not bool(finite):
        raise FloatingPointError(f"{label} became NaN or infinity")


def build_tqc_observation(
    base_observation,
    normalised_nominal_ctrl,
    normalised_previous_residual,
    executable_mask,
    controller_state,
):
    """Build the post-IK TQC state without changing the 191-D base schema.

    The caller prepares this state only after the nominal IK command and the
    exact supervisor/safety authority for the current tick are known.  Nominal
    controls must be normalised to their actuator ctrlrange and realised
    residuals to their unit-aware authority limits.  ``controller_state`` is
    ordered as :data:`TQC_CONTROLLER_STATE_NAMES`; these three values expose
    Diffusion-plan and anchor-handover memory that is absent from the base
    Markov observation.
    """

    torch = _torch()
    base = torch.as_tensor(base_observation)
    if base.ndim != 2 or base.shape[1] != OBS_DIM:
        raise ValueError(f"base_observation must have shape (N, {OBS_DIM})")
    device = base.device
    dtype = base.dtype
    count = base.shape[0]

    def _float(value, width: int, label: str):
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.shape != (count, width):
            raise ValueError(
                f"{label} must have shape {(count, width)}, got "
                f"{tuple(tensor.shape)}"
            )
        return tensor

    nominal = _float(
        normalised_nominal_ctrl, TQC_ACTION_DIM, "normalised_nominal_ctrl"
    )
    previous = _float(
        normalised_previous_residual,
        TQC_ACTION_DIM,
        "normalised_previous_residual",
    )
    controller = _float(
        controller_state, TQC_CONTROLLER_STATE_DIM, "controller_state"
    )
    mask = torch.as_tensor(executable_mask, device=device)
    if mask.dtype != torch.bool or mask.shape != (count, TQC_ACTION_DIM):
        raise ValueError(
            "executable_mask must be boolean with shape "
            f"{(count, TQC_ACTION_DIM)}"
        )
    state = torch.cat(
        (base, nominal, previous, mask.to(dtype), controller), dim=-1
    )
    if state.shape != (count, TQC_OBS_DIM):
        raise RuntimeError("constructed TQC observation has an invalid shape")
    if not bool(torch.isfinite(state).all()):
        raise FloatingPointError("TQC observation contains NaN or infinity")
    return state


class SquashedGaussianActor(_torch().nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        log_std_init: float = -3.0,
    ) -> None:
        torch = _torch()
        super().__init__()
        self.action_dim = action_dim
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.mean = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std = torch.nn.Linear(hidden_dim, action_dim)
        # Start as the certified nominal Diffusion--IK controller.  Exploration
        # grows through learning rather than an initially biased correction.
        torch.nn.init.zeros_(self.mean.weight)
        torch.nn.init.zeros_(self.mean.bias)
        torch.nn.init.zeros_(self.log_std.weight)
        torch.nn.init.constant_(self.log_std.bias, float(log_std_init))

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
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        if action_mask is None:
            action_mask = torch.ones_like(action, dtype=torch.bool)
        mask = action_mask.to(action.dtype)
        action = action * mask
        if deterministic:
            log_prob = torch.zeros(
                (observation.shape[0], 1),
                device=observation.device,
                dtype=observation.dtype,
            )
        else:
            gaussian = -0.5 * (
                ((pre_tanh - mean) / log_std.exp()).square()
                + 2.0 * log_std
                + math.log(2.0 * math.pi)
            )
            log_prob = (
                gaussian
                - torch.log(1.0 - action.square() + 1.0e-6)
            )
            log_prob = (log_prob * mask).sum(dim=-1, keepdim=True)
        return action, log_prob


class QuantileCritic(_torch().nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int, quantiles: int
    ) -> None:
        torch = _torch()
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, quantiles),
        )

    def forward(self, observation, action):
        return self.network(_torch().cat((observation, action), dim=-1))


class QuantileCritics(_torch().nn.Module):
    def __init__(
        self,
        *,
        count: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        quantiles: int,
    ) -> None:
        torch = _torch()
        super().__init__()
        self.networks = torch.nn.ModuleList(
            QuantileCritic(obs_dim, action_dim, hidden_dim, quantiles)
            for _ in range(count)
        )

    def forward(self, observation, action):
        return _torch().stack(
            [network(observation, action) for network in self.networks], dim=1
        )


def quantile_huber_loss(prediction, target):
    """Pairwise quantile Huber regression from the TQC paper."""

    torch = _torch()
    quantiles = prediction.shape[-1]
    delta = target[:, None, None, :] - prediction[:, :, :, None]
    absolute = delta.abs()
    huber = torch.where(
        absolute <= 1.0, 0.5 * delta.square(), absolute - 0.5
    )
    tau = (
        torch.arange(quantiles, device=prediction.device, dtype=prediction.dtype)
        + 0.5
    ) / quantiles
    weight = torch.abs(
        tau[None, None, :, None] - (delta.detach() < 0.0).to(prediction.dtype)
    )
    return (weight * huber).mean()


class TQCTrainer:
    def __init__(self, config, *, device: str) -> None:
        torch = _torch()
        self.config = config
        self.device = torch.device(device)
        c = config.tqc
        self.actor = SquashedGaussianActor(
            TQC_OBS_DIM,
            TQC_ACTION_DIM,
            c.hidden_dim,
            log_std_init=c.log_std_init,
        ).to(self.device)
        self.critics = QuantileCritics(
            count=c.critics,
            obs_dim=TQC_OBS_DIM,
            action_dim=TQC_ACTION_DIM,
            hidden_dim=c.hidden_dim,
            quantiles=c.quantiles,
        ).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).eval()
        for parameter in self.target_critics.parameters():
            parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=c.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critics.parameters(), lr=c.critic_lr
        )
        self.log_alpha = torch.zeros(
            (), device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=c.alpha_lr
        )
        self.updates = 0
        self.transitions = 0
        self.update_credit = 0.0
        self.metric_ema_decay = 0.99
        self.metric_ema_updates = 0
        self.metric_ema: Dict[str, object] = {}

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @_torch().no_grad()
    def act(
        self,
        observation,
        *,
        action_mask,
        deterministic: bool = False,
    ):
        if observation.ndim != 2 or observation.shape[1] != TQC_OBS_DIM:
            raise ValueError(
                f"TQC observation must have shape (N, {TQC_OBS_DIM})"
            )
        if action_mask.dtype != _torch().bool or action_mask.shape != (
            observation.shape[0],
            TQC_ACTION_DIM,
        ):
            raise ValueError(
                "action_mask must be boolean with shape "
                f"{(observation.shape[0], TQC_ACTION_DIM)}"
            )
        return self.actor.sample(
            observation,
            deterministic=deterministic,
            action_mask=action_mask,
        )[0]

    def _with_metric_ema(self, metrics: Dict[str, object]) -> Dict[str, object]:
        """Update on-device EMAs without introducing a host synchronization."""

        torch = _torch()
        with torch.no_grad():
            for name, raw in metrics.items():
                value = (
                    raw.detach().to(self.device).reshape(())
                    if isinstance(raw, torch.Tensor)
                    else torch.as_tensor(
                        raw, device=self.device, dtype=torch.float32
                    ).reshape(())
                )
                prior = self.metric_ema.get(name)
                if prior is None:
                    self.metric_ema[name] = value.clone()
                else:
                    prior.mul_(self.metric_ema_decay).add_(
                        value, alpha=1.0 - self.metric_ema_decay
                    )
            self.metric_ema_updates += 1
        result = dict(metrics)
        result.update(
            {
                f"{name}_ema": value.detach()
                for name, value in self.metric_ema.items()
            }
        )
        result["metric_ema_decay"] = self.metric_ema_decay
        result["metric_ema_updates"] = self.metric_ema_updates
        return result

    def update(self, batch) -> Dict[str, object]:
        torch = _torch()
        c = self.config.tqc
        _require_finite(
            "TQC replay batch",
            batch.observation,
            batch.action,
            batch.reward,
            batch.next_observation,
        )
        if batch.action_mask.dtype != torch.bool:
            raise TypeError("replay action_mask must be boolean")
        if batch.next_action_mask.dtype != torch.bool:
            raise TypeError("replay next_action_mask must be boolean")
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(
                batch.next_observation,
                action_mask=batch.next_action_mask,
            )
            target_quantiles = self.target_critics(
                batch.next_observation, next_action
            ).flatten(start_dim=1)
            target_quantiles, _ = torch.sort(target_quantiles, dim=-1)
            total_drop = c.top_quantiles_to_drop * c.critics
            if total_drop:
                target_quantiles = target_quantiles[:, :-total_drop]
            # Both physical terminals and the explicit 4000-step episode
            # boundary end this replay sequence; neither may bootstrap into a
            # reset world's different screw/socket initial condition.
            bootstrap = (~(batch.terminated | batch.truncated)).to(
                batch.reward.dtype
            )
            target = batch.reward + c.gamma * bootstrap * (
                target_quantiles - self.alpha.detach() * next_log_prob
            )
            _require_finite("TQC target", target)

        replay_action = batch.action * batch.action_mask.to(batch.action.dtype)
        prediction = self.critics(batch.observation, replay_action)
        critic_loss = quantile_huber_loss(prediction, target)
        _require_finite("TQC critic loss", critic_loss)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.critics.parameters(), 10.0
        )
        _require_finite("TQC critic gradient", critic_grad)
        self.critic_optimizer.step()

        action_mask = batch.action_mask
        actor_learning = self.transitions >= c.actor_learning_starts
        if actor_learning:
            # The critics are used as a differentiable value function for the
            # action, but their weights must not accumulate actor gradients.
            for parameter in self.critics.parameters():
                parameter.requires_grad_(False)
            try:
                action, log_prob = self.actor.sample(
                    batch.observation, action_mask=action_mask
                )
                actor_quantiles = self.critics(batch.observation, action)
                actor_value = actor_quantiles.mean(dim=(1, 2))[:, None]
                actor_loss = (
                    self.alpha.detach() * log_prob - actor_value
                ).mean()
                _require_finite("TQC actor loss", actor_loss)
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 10.0
                )
                _require_finite("TQC actor gradient", actor_grad)
                self.actor_optimizer.step()
            finally:
                for parameter in self.critics.parameters():
                    parameter.requires_grad_(True)

            target_entropy = -action_mask.sum(
                dim=-1, keepdim=True
            ).to(log_prob.dtype)
            alpha_loss = -(
                self.log_alpha * (log_prob.detach() + target_entropy)
            ).mean()
            _require_finite("TQC temperature loss", alpha_loss)
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            _require_finite("TQC temperature gradient", self.log_alpha.grad)
            self.alpha_optimizer.step()
        else:
            with torch.no_grad():
                # Diagnostics remain well-defined while actor/temperature are
                # deliberately frozen during the critic-first safety phase.
                _action, log_prob = self.actor.sample(
                    batch.observation, action_mask=action_mask
                )
                target_entropy = -action_mask.sum(
                    dim=-1, keepdim=True
                ).to(log_prob.dtype)
                actor_loss = torch.zeros((), device=self.device)
                alpha_loss = torch.zeros((), device=self.device)
                actor_grad = torch.zeros((), device=self.device)

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critics.parameters(), self.critics.parameters()
            ):
                target_parameter.lerp_(parameter, c.tau)
            q_prediction = prediction.detach().flatten()
            q_target = target.detach().flatten()
            q_prediction_mean_per_sample = prediction.detach().mean(
                dim=(1, 2)
            )
            q_target_mean_per_sample = target.detach().mean(dim=1)
            td_absolute = (
                q_prediction_mean_per_sample - q_target_mean_per_sample
            ).abs()
            diagnostics = {
                "q_pred_mean": q_prediction.mean(),
                "q_pred_std": q_prediction.std(unbiased=False),
                "q_pred_min": q_prediction.min(),
                "q_pred_max": q_prediction.max(),
                "q_target_mean": q_target.mean(),
                "q_target_std": q_target.std(unbiased=False),
                "q_target_min": q_target.min(),
                "q_target_max": q_target.max(),
                "td_abs_mean": td_absolute.mean(),
                "td_abs_p95": torch.quantile(td_absolute, 0.95),
                "policy_log_prob_mean": log_prob.detach().mean(),
                "policy_entropy": -log_prob.detach().mean(),
                "target_entropy": target_entropy.detach().mean(),
                "batch_reward_mean": batch.reward.mean(),
                "batch_reward_std": batch.reward.std(unbiased=False),
                "batch_reward_min": batch.reward.min(),
                "batch_reward_max": batch.reward.max(),
                "batch_terminated_fraction": batch.terminated.to(
                    torch.float32
                ).mean(),
                "batch_truncated_fraction": batch.truncated.to(
                    torch.float32
                ).mean(),
            }
            _require_finite(
                "TQC diagnostic metrics", *diagnostics.values()
            )
        self.updates += 1
        # Keep metrics on-device.  Converting each update to Python floats
        # would force six CUDA synchronizations; the workflow materializes only
        # the metrics that accompany a checkpoint/log event.
        metrics = {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "alpha_loss": alpha_loss.detach(),
            "alpha": self.alpha.detach(),
            "critic_grad_norm": critic_grad.detach(),
            "actor_grad_norm": actor_grad.detach(),
            "active_action_dimensions": action_mask.sum(
                dim=-1
            ).to(torch.float32).mean(),
            "actor_learning_started": torch.as_tensor(
                float(actor_learning), device=self.device
            ),
            "actor_learning_rate": float(
                self.actor_optimizer.param_groups[0]["lr"]
            ),
            "critic_learning_rate": float(
                self.critic_optimizer.param_groups[0]["lr"]
            ),
            "alpha_learning_rate": float(
                self.alpha_optimizer.param_groups[0]["lr"]
            ),
            **diagnostics,
        }
        return self._with_metric_ema(metrics)

    def assert_finite(self) -> None:
        """Fail before a checkpoint can replace the last-known-good state."""

        tensors = [self.log_alpha]
        tensors.extend(self.actor.parameters())
        tensors.extend(self.critics.parameters())
        tensors.extend(self.target_critics.parameters())
        tensors.extend(self.metric_ema.values())
        _require_finite("TQC parameters", *tensors)

    def updates_for(self, new_transitions: int) -> int:
        previous = self.transitions
        self.transitions += int(new_transitions)
        # Do not train before learning_starts, and do not accrue a large
        # catch-up burst from the random warm-up transitions.
        eligible = max(
            0,
            self.transitions
            - max(previous, int(self.config.tqc.learning_starts)),
        )
        self.update_credit += (
            self.config.tqc.replay_ratio
            * float(eligible)
            / float(self.config.tqc.batch_size)
        )
        requested = int(self.update_credit)
        updates = min(
            requested,
            int(self.config.tqc.gradient_steps_per_vector_step),
        )
        # A 14k-world vector step can create far more transition credit than
        # is safe to consume as one blocking optimizer burst.  This is an
        # explicit per-vector-step update ratio, not deferred catch-up debt.
        self.update_credit -= requested
        return updates

    def checkpoint(self, replay=None, *, include_replay: bool = False):
        payload = checkpoint_metadata("tqc", self.config.to_dict())
        payload.update(
            {
                "actor": _cpu(self.actor.state_dict()),
                "critics": _cpu(self.critics.state_dict()),
                "target_critics": _cpu(self.target_critics.state_dict()),
                "actor_optimizer": _cpu(self.actor_optimizer.state_dict()),
                "critic_optimizer": _cpu(self.critic_optimizer.state_dict()),
                "alpha_optimizer": _cpu(self.alpha_optimizer.state_dict()),
                "log_alpha": _cpu(self.log_alpha),
                "updates": self.updates,
                "transitions": self.transitions,
                "update_credit": self.update_credit,
                "metric_ema_decay": self.metric_ema_decay,
                "metric_ema_updates": self.metric_ema_updates,
                "metric_ema": _cpu(self.metric_ema),
                "tqc_obs_schema": TQC_OBS_SCHEMA,
                "tqc_obs_dim": TQC_OBS_DIM,
                "tqc_action_dim": TQC_ACTION_DIM,
            }
        )
        if replay is not None:
            payload["replay"] = replay.state_dict(include_data=include_replay)
        return payload

    def load(self, payload, replay=None) -> None:
        validate_checkpoint(payload, kind="tqc")
        if payload.get("tqc_obs_schema", TQC_OBS_SCHEMA) != TQC_OBS_SCHEMA:
            raise ValueError("TQC observation schema differs from checkpoint")
        if int(payload.get("tqc_obs_dim", TQC_OBS_DIM)) != TQC_OBS_DIM:
            raise ValueError("TQC observation dimension differs from checkpoint")
        if int(payload.get("tqc_action_dim", TQC_ACTION_DIM)) != TQC_ACTION_DIM:
            raise ValueError("TQC action dimension differs from checkpoint")
        self.actor.load_state_dict(payload["actor"])
        self.critics.load_state_dict(payload["critics"])
        self.target_critics.load_state_dict(payload["target_critics"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        with _torch().no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        self.updates = int(payload.get("updates", 0))
        self.transitions = int(payload.get("transitions", 0))
        self.update_credit = float(payload.get("update_credit", 0.0))
        self.metric_ema_decay = float(
            payload.get("metric_ema_decay", self.metric_ema_decay)
        )
        self.metric_ema_updates = int(
            payload.get("metric_ema_updates", 0)
        )
        stored_metric_ema = payload.get("metric_ema", {})
        torch = _torch()
        self.metric_ema = {
            str(name): (
                value.to(self.device).reshape(())
                if isinstance(value, torch.Tensor)
                else torch.as_tensor(
                    value, device=self.device, dtype=torch.float32
                ).reshape(())
            )
            for name, value in dict(stored_metric_ema).items()
        }
        if replay is not None and "replay" in payload:
            replay.load_state_dict(payload["replay"])
        self.assert_finite()


__all__ = [
    "QuantileCritics",
    "SquashedGaussianActor",
    "TQC_ACTION_DIM",
    "TQC_CONTROLLER_STATE_DIM",
    "TQC_CONTROLLER_STATE_NAMES",
    "TQC_OBS_DIM",
    "TQC_OBS_SCHEMA",
    "TQCTrainer",
    "build_tqc_observation",
    "quantile_huber_loss",
]
