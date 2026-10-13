#!/usr/bin/env python3
"""State-conditioned action-chunk diffusion policy.

This implementation uses only PyTorch and NumPy.  It deliberately avoids
``diffusers`` and project-external modules so the complete ``SCREW`` directory
can be moved on the same Apple Silicon Mac.

The denoiser is a compact FiLM-conditioned residual MLP.  It receives:

* a complete state history ``[B, obs_horizon, obs_dim]``;
* a noised action chunk ``[B, horizon, action_dim]``;
* one diffusion timestep per batch element.

Training uses epsilon-prediction DDPM objectives.  Inference uses deterministic
few-step DDIM by default, which is much faster than traversing every training
noise step and is suitable for receding-horizon robot control.
"""

from __future__ import annotations

import math
import os
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


CHECKPOINT_FORMAT_VERSION = 1


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch without assuming a particular backend."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # warn_only avoids turning an otherwise valid MPS run into a hard
        # failure when PyTorch lacks a deterministic implementation.
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` in the project-requested MPS -> CUDA -> CPU order."""

    if isinstance(requested, torch.device):
        device = requested
    else:
        name = str(requested).lower()
        if name == "auto":
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
                and torch.backends.mps.is_available()
            ):
                name = "mps"
            elif torch.cuda.is_available():
                name = "cuda"
            else:
                name = "cpu"
        device = torch.device(name)

    if device.type == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is unavailable")
    elif device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _cosine_beta_schedule(
    num_steps: int, *, cosine_offset: float = 0.008
) -> torch.Tensor:
    """Nichol-Dhariwal cosine variance schedule in float32."""

    if num_steps < 2:
        raise ValueError("num_diffusion_steps must be at least 2")
    steps = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((steps / num_steps) + cosine_offset)
        / (1.0 + cosine_offset)
        * math.pi
        * 0.5
    ).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1.0e-5, 0.999).to(dtype=torch.float32)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *((1,) * (target.ndim - 1)))


def _generator(seed: int, device: torch.device) -> torch.Generator:
    """Create RNG state on the sampling device.

    The standalone MPS profile keeps its compatibility-oriented CPU RNG.  This
    CUDA package avoids copying every diffusion noise chunk through host RAM.
    """

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _randn(
    shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if generator is None:
        return torch.randn(tuple(shape), device=device, dtype=dtype)
    return torch.randn(
        tuple(shape), device=device, dtype=dtype, generator=generator
    )


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4:
            raise ValueError("timestep embedding dimension must be at least 4")
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        exponent = -math.log(10_000.0) * torch.arange(
            half, device=timesteps.device, dtype=torch.float32
        ) / max(half - 1, 1)
        angles = timesteps.to(torch.float32).unsqueeze(-1) * exponent.exp()
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[-1]))
        return embedding


class StableLayerNorm(nn.Module):
    """LayerNorm with an explicit MPS backward path.

    PyTorch 2.11's fused MPS LayerNorm kernel can return NaN affine gradients
    for a finite, globally-standardized observation batch (the forward output
    and DDPM loss remain finite).  Expressing the same operation with primitive
    reductions avoids that kernel while keeping every tensor and gradient on
    MPS.  CPU/CUDA retain PyTorch's optimized implementation.

    Parameter names and shapes intentionally match ``nn.LayerNorm``, preserving
    state-dict compatibility with checkpoints created before this workaround.
    """

    def __init__(self, dimension: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("LayerNorm dimension must be positive")
        self.dimension = int(dimension)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dimension))
        self.bias = nn.Parameter(torch.zeros(self.dimension))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.dimension:
            raise ValueError(
                f"LayerNorm expected trailing dimension {self.dimension}, "
                f"got {values.shape[-1]}"
            )
        if values.device.type != "mps":
            return F.layer_norm(
                values,
                (self.dimension,),
                self.weight,
                self.bias,
                self.eps,
            )
        mean = values.mean(dim=-1, keepdim=True)
        centered = values - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, condition_dim: int, dropout: float) -> None:
        super().__init__()
        self.normalization = StableLayerNorm(hidden_dim)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim * 2)
        self.network = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition_projection(condition).chunk(2, dim=-1)
        transformed = self.normalization(hidden) * (1.0 + scale) + shift
        return hidden + self.network(transformed)


class ActionChunkDenoiser(nn.Module):
    """Predict DDPM noise for a flattened action chunk."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int,
        horizon: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim < 16:
            raise ValueError("hidden_dim must be at least 16")
        if depth < 1:
            raise ValueError("depth must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.horizon = horizon
        self.action_size = action_dim * horizon
        condition_dim = hidden_dim

        self.action_projection = nn.Linear(self.action_size, hidden_dim)
        self.observation_encoder = nn.Sequential(
            StableLayerNorm(obs_dim * obs_horizon),
            nn.Linear(obs_dim * obs_horizon, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, condition_dim),
        )
        timestep_dim = max(32, hidden_dim // 2)
        self.timestep_encoder = nn.Sequential(
            SinusoidalTimestepEmbedding(timestep_dim),
            nn.Linear(timestep_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(hidden_dim, condition_dim, dropout)
            for _ in range(depth)
        )
        self.output = nn.Sequential(
            StableLayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_size),
        )
        # Epsilon initially near zero is a stable starting point.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        observation_history: torch.Tensor,
    ) -> torch.Tensor:
        expected_action = (self.horizon, self.action_dim)
        expected_obs = (self.obs_horizon, self.obs_dim)
        if noisy_actions.ndim != 3 or tuple(noisy_actions.shape[1:]) != expected_action:
            raise ValueError(
                f"noisy_actions must have shape [B, {self.horizon}, "
                f"{self.action_dim}], got {tuple(noisy_actions.shape)}"
            )
        if (
            observation_history.ndim != 3
            or tuple(observation_history.shape[1:]) != expected_obs
        ):
            raise ValueError(
                f"observation_history must have shape [B, {self.obs_horizon}, "
                f"{self.obs_dim}], got {tuple(observation_history.shape)}"
            )
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy_actions.shape[0]:
            raise ValueError("timesteps must have shape [B]")
        if observation_history.shape[0] != noisy_actions.shape[0]:
            raise ValueError("action and observation batch sizes differ")

        hidden = self.action_projection(noisy_actions.flatten(start_dim=1))
        condition = self.observation_encoder(
            observation_history.flatten(start_dim=1)
        ) + self.timestep_encoder(timesteps)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden).reshape_as(noisy_actions)


class DiffusionPolicy(nn.Module):
    """DDPM training objective plus few-step DDIM action sampling."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int,
        horizon: int,
        num_diffusion_steps: int = 100,
        hidden_dim: int = 256,
        depth: int = 6,
        dropout: float = 0.0,
        normalization_clip: float = 10.0,
        sample_clip: float = 5.0,
    ) -> None:
        super().__init__()
        dimensions = {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "obs_horizon": obs_horizon,
            "horizon": horizon,
        }
        for name, value in dimensions.items():
            if int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if normalization_clip <= 0.0 or sample_clip <= 0.0:
            raise ValueError("normalization_clip and sample_clip must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs_horizon = int(obs_horizon)
        self.horizon = int(horizon)
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.normalization_clip = float(normalization_clip)
        self.sample_clip = float(sample_clip)

        self.denoiser = ActionChunkDenoiser(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            obs_horizon=self.obs_horizon,
            horizon=self.horizon,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            dropout=self.dropout,
        )

        betas = _cosine_beta_schedule(self.num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alpha_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alpha_cumprod).sqrt()
        )

        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_std", torch.ones(self.obs_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))

    def model_config(self) -> Dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "obs_horizon": self.obs_horizon,
            "horizon": self.horizon,
            "num_diffusion_steps": self.num_diffusion_steps,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "normalization_clip": self.normalization_clip,
            "sample_clip": self.sample_clip,
        }

    @torch.no_grad()
    def set_normalization(self, stats: Mapping[str, Any]) -> None:
        required = {
            "obs_mean": (self.obs_dim,),
            "obs_std": (self.obs_dim,),
            "action_mean": (self.action_dim,),
            "action_std": (self.action_dim,),
        }
        for name, expected_shape in required.items():
            if name not in stats:
                raise ValueError(f"normalization stats missing {name!r}")
            value = torch.as_tensor(
                stats[name], dtype=self.obs_mean.dtype, device=self.obs_mean.device
            )
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
            if name.endswith("_std") and torch.any(value <= 0.0):
                raise ValueError(f"{name} must be strictly positive")
            getattr(self, name).copy_(value)

    def normalization_dict(self, *, numpy: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name in ("obs_mean", "obs_std", "action_mean", "action_std"):
            tensor = getattr(self, name).detach().cpu()
            result[name] = tensor.numpy().copy() if numpy else tensor.clone()
        return result

    def normalize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = (observations - self.obs_mean) / self.obs_std
        return normalized.clamp(
            -self.normalization_clip, self.normalization_clip
        )

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        normalized = (actions - self.action_mean) / self.action_std
        return normalized.clamp(
            -self.normalization_clip, self.normalization_clip
        )

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.action_std + self.action_mean

    def q_sample(
        self,
        normalized_actions: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, normalized_actions)
            * normalized_actions
            + _extract(
                self.sqrt_one_minus_alphas_cumprod,
                timesteps,
                normalized_actions,
            )
            * noise
        )

    def compute_loss(
        self,
        observation_history: torch.Tensor,
        actions: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return the epsilon-prediction mean squared DDPM loss."""

        if actions.ndim != 3 or tuple(actions.shape[1:]) != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"actions must have shape [B, {self.horizon}, {self.action_dim}]"
            )
        if observation_history.ndim != 3 or tuple(
            observation_history.shape[1:]
        ) != (self.obs_horizon, self.obs_dim):
            raise ValueError(
                f"observation_history must have shape [B, {self.obs_horizon}, "
                f"{self.obs_dim}]"
            )
        if actions.shape[0] != observation_history.shape[0]:
            raise ValueError("action and observation batch sizes differ")

        device = actions.device
        batch_size = actions.shape[0]
        if generator is None:
            timesteps = torch.randint(
                self.num_diffusion_steps, (batch_size,), device=device
            )
            noise = torch.randn_like(actions)
        else:
            timesteps = torch.randint(
                self.num_diffusion_steps,
                (batch_size,),
                device="cpu",
                generator=generator,
            ).to(device)
            noise = _randn(
                actions.shape,
                device=device,
                dtype=actions.dtype,
                generator=generator,
            )

        normalized_obs = self.normalize_observations(observation_history)
        normalized_actions = self.normalize_actions(actions)
        noisy_actions = self.q_sample(normalized_actions, timesteps, noise)
        predicted_noise = self.denoiser(
            noisy_actions, timesteps, normalized_obs
        )
        return F.mse_loss(predicted_noise, noise)

    def forward(
        self,
        noisy_normalized_actions: torch.Tensor,
        timesteps: torch.Tensor,
        observation_history: torch.Tensor,
    ) -> torch.Tensor:
        """Expose the normalized denoiser for advanced training/debugging."""

        return self.denoiser(
            noisy_normalized_actions,
            timesteps,
            self.normalize_observations(observation_history),
        )

    @torch.no_grad()
    def sample(
        self,
        observation_history: torch.Tensor | np.ndarray | Sequence[Any],
        *,
        num_inference_steps: int = 10,
        eta: float = 0.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample raw-unit action chunks with a strided DDIM trajectory.

        A 1-D or 2-D observation input is treated as one batch and is
        automatically left-padded/truncated to ``obs_horizon``.  A 3-D batched
        input must already have the exact history shape.
        """

        if not 1 <= num_inference_steps <= self.num_diffusion_steps:
            raise ValueError(
                "num_inference_steps must be in [1, num_diffusion_steps]"
            )
        if eta < 0.0:
            raise ValueError("eta cannot be negative")
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        obs, squeeze_batch = prepare_observation_history(
            observation_history,
            obs_horizon=self.obs_horizon,
            obs_dim=self.obs_dim,
            device=device,
            dtype=dtype,
        )
        normalized_obs = self.normalize_observations(obs)
        generator = _generator(seed, device) if seed is not None else None
        sample = _randn(
            (obs.shape[0], self.horizon, self.action_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Rounded linspace includes both endpoints.  ``unique_consecutive``
        # protects very small schedules from duplicate integer timesteps.
        timestep_values = torch.linspace(
            self.num_diffusion_steps - 1,
            0,
            num_inference_steps,
            device="cpu",
        ).round().to(torch.long).unique_consecutive().tolist()
        was_training = self.training
        self.eval()
        try:
            for index, t_int in enumerate(timestep_values):
                batch_t = torch.full(
                    (obs.shape[0],), t_int, device=device, dtype=torch.long
                )
                predicted_noise = self.denoiser(
                    sample, batch_t, normalized_obs
                )
                alpha_t = self.alphas_cumprod[t_int]
                if index + 1 < len(timestep_values):
                    previous_t = timestep_values[index + 1]
                    alpha_previous = self.alphas_cumprod[previous_t]
                else:
                    alpha_previous = torch.ones(
                        (), device=device, dtype=dtype
                    )

                predicted_clean = (
                    sample - (1.0 - alpha_t).sqrt() * predicted_noise
                ) / alpha_t.sqrt()
                predicted_clean = predicted_clean.clamp(
                    -self.sample_clip, self.sample_clip
                )
                sigma = eta * torch.sqrt(
                    torch.clamp(
                        (1.0 - alpha_previous)
                        / (1.0 - alpha_t)
                        * (1.0 - alpha_t / alpha_previous),
                        min=0.0,
                    )
                )
                direction = torch.sqrt(
                    torch.clamp(
                        1.0 - alpha_previous - sigma.square(), min=0.0
                    )
                ) * predicted_noise
                sample = alpha_previous.sqrt() * predicted_clean + direction
                if eta > 0.0 and index + 1 < len(timestep_values):
                    sample = sample + sigma * _randn(
                        sample.shape,
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
        finally:
            self.train(was_training)

        raw_actions = self.denormalize_actions(sample)
        return raw_actions[0] if squeeze_batch else raw_actions


def prepare_observation_history(
    observations: torch.Tensor | np.ndarray | Sequence[Any],
    *,
    obs_horizon: int,
    obs_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, bool]:
    """Convert observations to ``[B, obs_horizon, obs_dim]``.

    Returns ``(history, squeeze_batch)``.  Unbatched history is left-padded by
    repeating its oldest available observation.
    """

    tensor = torch.as_tensor(observations, dtype=dtype, device=device)
    squeeze_batch = tensor.ndim < 3
    if tensor.ndim == 1:
        if tensor.shape[0] != obs_dim:
            raise ValueError(
                f"single observation must have {obs_dim} values, got "
                f"{tensor.shape[0]}"
            )
        tensor = tensor.reshape(1, 1, obs_dim)
    elif tensor.ndim == 2:
        if tensor.shape[1] != obs_dim or tensor.shape[0] < 1:
            raise ValueError(
                f"observation history must have shape [T, {obs_dim}]"
            )
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tuple(tensor.shape[1:]) != (obs_horizon, obs_dim):
            raise ValueError(
                f"batched history must have shape [B, {obs_horizon}, {obs_dim}]"
            )
        return tensor, False
    else:
        raise ValueError("observations must be 1-D, 2-D, or 3-D")

    if tensor.shape[1] > obs_horizon:
        tensor = tensor[:, -obs_horizon:]
    elif tensor.shape[1] < obs_horizon:
        padding = tensor[:, :1].expand(
            -1, obs_horizon - tensor.shape[1], -1
        )
        tensor = torch.cat((padding, tensor), dim=1)
    return tensor, squeeze_batch


class ExponentialMovingAverage:
    """Portable EMA shadow weights for smoother diffusion sampling."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise ValueError("EMA/model state keys differ")
        for key, value in current.items():
            shadow = self.shadow[key]
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(
                    value.detach().to(shadow.device), alpha=1.0 - self.decay
                )
            else:
                shadow.copy_(value.detach().to(shadow.device))

    def state_dict(self, *, cpu: bool = False) -> Dict[str, torch.Tensor]:
        if cpu:
            return {key: value.detach().cpu() for key, value in self.shadow.items()}
        return {key: value.detach().clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA checkpoint/model state keys differ")
        for key, value in state.items():
            self.shadow[key] = value.detach().clone().to(self.shadow[key].device)


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint {path} does not contain a dictionary")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {checkpoint.get('format_version')!r}"
        )
    return checkpoint


def load_policy_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    device: str | torch.device = "auto",
    use_ema: bool = True,
    eval_mode: bool = True,
) -> Tuple[DiffusionPolicy, Dict[str, Any]]:
    """Reconstruct a policy entirely from a portable training checkpoint."""

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = _torch_load(path)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint is missing model_config")
    policy = DiffusionPolicy(**dict(model_config))
    state_key = (
        "ema_state"
        if use_ema and checkpoint.get("ema_state") is not None
        else "model_state"
    )
    if state_key not in checkpoint:
        raise ValueError(f"checkpoint is missing {state_key}")
    policy.load_state_dict(checkpoint[state_key], strict=True)
    policy.to(resolve_device(device))
    if eval_mode:
        policy.eval()
    return policy, checkpoint


class DiffusionController:
    """Small receding-horizon runtime wrapper around a trained policy."""

    def __init__(
        self,
        policy: DiffusionPolicy,
        *,
        num_inference_steps: int = 10,
        replan_interval: int = 1,
        seed: int = 42,
    ) -> None:
        if replan_interval < 1 or replan_interval > policy.horizon:
            raise ValueError("replan_interval must be in [1, policy.horizon]")
        self.policy = policy
        self.num_inference_steps = int(num_inference_steps)
        self.replan_interval = int(replan_interval)
        self.seed = int(seed)
        self._plan_index = 0
        self._sample_index = 0
        self._history: deque[np.ndarray] = deque(maxlen=policy.obs_horizon)
        self._action_queue: deque[np.ndarray] = deque()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | os.PathLike[str],
        *,
        device: str | torch.device = "auto",
        use_ema: bool = True,
        num_inference_steps: int = 10,
        replan_interval: int = 1,
        seed: int = 42,
    ) -> "DiffusionController":
        policy, _ = load_policy_checkpoint(
            checkpoint_path, device=device, use_ema=use_ema
        )
        return cls(
            policy,
            num_inference_steps=num_inference_steps,
            replan_interval=replan_interval,
            seed=seed,
        )

    def reset(self, initial_observation: Optional[Sequence[float]] = None) -> None:
        self._history.clear()
        self._action_queue.clear()
        self._plan_index = 0
        self._sample_index = 0
        if initial_observation is not None:
            self.observe(initial_observation)

    def observe(self, observation: Sequence[float] | np.ndarray) -> None:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.policy.obs_dim,):
            raise ValueError(
                f"observation must have shape ({self.policy.obs_dim},), "
                f"got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("observation contains NaN or infinity")
        self._history.append(value.copy())

    def predict_action_chunk(
        self,
        observation_history: Optional[np.ndarray | Sequence[Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        if observation_history is None:
            if not self._history:
                raise RuntimeError("no observations available")
            observation_history = np.stack(tuple(self._history), axis=0)
        sample_seed = (
            self.seed + self._sample_index if seed is None else int(seed)
        )
        self._sample_index += 1
        with torch.inference_mode():
            actions = self.policy.sample(
                observation_history,
                num_inference_steps=self.num_inference_steps,
                seed=sample_seed,
            )
        return actions.detach().cpu().numpy()

    def act(self, observation: Sequence[float] | np.ndarray) -> np.ndarray:
        """Observe one state and return one action in receding-horizon mode."""

        self.observe(observation)
        if not self._action_queue or self._plan_index >= self.replan_interval:
            chunk = self.predict_action_chunk()
            self._action_queue = deque(row.copy() for row in chunk)
            self._plan_index = 0
        self._plan_index += 1
        return self._action_queue.popleft()


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "DiffusionController",
    "DiffusionPolicy",
    "ExponentialMovingAverage",
    "load_policy_checkpoint",
    "prepare_observation_history",
    "resolve_device",
    "seed_everything",
]
