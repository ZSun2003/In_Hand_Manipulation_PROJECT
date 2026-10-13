"""State-conditioned action-chunk diffusion policy for MJWarp GRAB.

The implementation follows the verified PyTorch-only design in
``Diffusion/S_HPC``: epsilon-prediction DDPM training, deterministic
few-step DDIM inference, EMA weights, and portable checkpoints.  Model
dimensions are obtained from the live GRAB contract and environment rather
than duplicating either action or observation widths here.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


CHECKPOINT_FORMAT = "mjwarp-grab-diffusion"
CHECKPOINT_FORMAT_VERSION = 1

OBSERVATION_HORIZON = 2
ACTION_HORIZON = 16
DDPM_TRAIN_STEPS = 100
DDIM_INFERENCE_STEPS = 16
REPLAN_INTERVAL = 4
DEFAULT_HIDDEN_DIM = 512
DEFAULT_DEPTH = 6


def environment_dimensions() -> tuple[int, int]:
    """Return ``(observation_dim, action_dim)`` from the current contract."""

    from .contract import ACTION_DIM
    from .env import OBSERVATION_DIM

    return int(OBSERVATION_DIM), int(ACTION_DIM)


def default_model_config() -> dict[str, Any]:
    """Return the canonical GRAB Diffusion model configuration."""

    observation_dim, action_dim = environment_dimensions()
    return {
        "obs_dim": observation_dim,
        "action_dim": action_dim,
        "obs_horizon": OBSERVATION_HORIZON,
        "horizon": ACTION_HORIZON,
        "num_diffusion_steps": DDPM_TRAIN_STEPS,
        "hidden_dim": DEFAULT_HIDDEN_DIM,
        "depth": DEFAULT_DEPTH,
        "dropout": 0.0,
    }


def build_default_policy(**overrides: Any) -> "DiffusionPolicy":
    """Build a policy tied to the current GRAB action/observation contract."""

    configuration = default_model_config()
    configuration.update(overrides)
    return DiffusionPolicy(**configuration)


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without assuming a CUDA device."""

    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA, MPS, or CPU, in training-preferred order."""

    if isinstance(requested, torch.device):
        device = requested
    else:
        name = str(requested).lower()
        if name == "auto":
            if torch.cuda.is_available():
                name = "cuda"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
                and torch.backends.mps.is_available()
            ):
                name = "mps"
            else:
                name = "cpu"
        device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _cosine_beta_schedule(
    num_steps: int,
    *,
    cosine_offset: float = 0.008,
) -> torch.Tensor:
    if int(num_steps) < 2:
        raise ValueError("num_diffusion_steps must be at least 2")
    steps = torch.linspace(0, int(num_steps), int(num_steps) + 1, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((steps / int(num_steps)) + cosine_offset)
        / (1.0 + cosine_offset)
        * math.pi
        * 0.5
    ).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1.0e-5, 0.999).to(dtype=torch.float32)


def _extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *((1,) * (target.ndim - 1)))


def _generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _randn(
    shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    return torch.randn(
        tuple(shape),
        device=device,
        dtype=dtype,
        generator=generator,
    )


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if int(dimension) < 4:
            raise ValueError("timestep embedding dimension must be at least 4")
        self.dimension = int(dimension)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        exponent = -math.log(10_000.0) * torch.arange(
            half,
            device=timesteps.device,
            dtype=torch.float32,
        ) / max(half - 1, 1)
        angles = timesteps.to(torch.float32).unsqueeze(-1) * exponent.exp()
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = F.pad(
                embedding,
                (0, self.dimension - embedding.shape[-1]),
            )
        return embedding


class StableLayerNorm(nn.Module):
    """LayerNorm with a primitive-reduction MPS backward path."""

    def __init__(self, dimension: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        if int(dimension) <= 0:
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

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = self.condition_projection(condition).chunk(2, dim=-1)
        transformed = self.normalization(hidden) * (1.0 + scale) + shift
        return hidden + self.network(transformed)


class ActionChunkDenoiser(nn.Module):
    """FiLM-conditioned residual MLP predicting one action-chunk noise tensor."""

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
        if int(hidden_dim) < 16:
            raise ValueError("hidden_dim must be at least 16")
        if int(depth) < 1:
            raise ValueError("depth must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs_horizon = int(obs_horizon)
        self.horizon = int(horizon)
        self.action_size = self.action_dim * self.horizon

        self.action_projection = nn.Linear(self.action_size, hidden_dim)
        self.observation_encoder = nn.Sequential(
            StableLayerNorm(self.obs_dim * self.obs_horizon),
            nn.Linear(self.obs_dim * self.obs_horizon, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        timestep_dim = max(32, int(hidden_dim) // 2)
        self.timestep_encoder = nn.Sequential(
            SinusoidalTimestepEmbedding(timestep_dim),
            nn.Linear(timestep_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(hidden_dim, hidden_dim, dropout)
            for _ in range(int(depth))
        )
        self.output = nn.Sequential(
            StableLayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_size),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        observation_history: torch.Tensor,
    ) -> torch.Tensor:
        expected_action = (self.horizon, self.action_dim)
        expected_observation = (self.obs_horizon, self.obs_dim)
        if (
            noisy_actions.ndim != 3
            or tuple(noisy_actions.shape[1:]) != expected_action
        ):
            raise ValueError(
                "noisy_actions must have shape "
                f"[B, {self.horizon}, {self.action_dim}]"
            )
        if (
            observation_history.ndim != 3
            or tuple(observation_history.shape[1:]) != expected_observation
        ):
            raise ValueError(
                "observation_history must have shape "
                f"[B, {self.obs_horizon}, {self.obs_dim}]"
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
    """DDPM training objective plus deterministic few-step DDIM sampling."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int = OBSERVATION_HORIZON,
        horizon: int = ACTION_HORIZON,
        num_diffusion_steps: int = DDPM_TRAIN_STEPS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        depth: int = DEFAULT_DEPTH,
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
            "num_diffusion_steps": num_diffusion_steps,
        }
        for name, value in dimensions.items():
            if int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if int(num_diffusion_steps) < 2:
            raise ValueError("num_diffusion_steps must be at least 2")
        if float(normalization_clip) <= 0.0 or float(sample_clip) <= 0.0:
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
        alpha_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alpha_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            (1.0 - alpha_cumprod).sqrt(),
        )
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_std", torch.ones(self.obs_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))

    def model_config(self) -> dict[str, Any]:
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
    def set_normalization(self, statistics: Mapping[str, Any]) -> None:
        required = {
            "obs_mean": (self.obs_dim,),
            "obs_std": (self.obs_dim,),
            "action_mean": (self.action_dim,),
            "action_std": (self.action_dim,),
        }
        for name, expected_shape in required.items():
            if name not in statistics:
                raise ValueError(f"normalization statistics missing {name!r}")
            value = torch.as_tensor(
                statistics[name],
                dtype=self.obs_mean.dtype,
                device=self.obs_mean.device,
            )
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, "
                    f"got {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
            if name.endswith("_std") and torch.any(value <= 0.0):
                raise ValueError(f"{name} must be strictly positive")
            getattr(self, name).copy_(value)

    def normalization_dict(self, *, numpy: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in ("obs_mean", "obs_std", "action_mean", "action_std"):
            tensor = getattr(self, name).detach().cpu()
            result[name] = tensor.numpy().copy() if numpy else tensor.clone()
        return result

    def normalize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        return ((observations - self.obs_mean) / self.obs_std).clamp(
            -self.normalization_clip,
            self.normalization_clip,
        )

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return ((actions - self.action_mean) / self.action_std).clamp(
            -self.normalization_clip,
            self.normalization_clip,
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
            _extract(
                self.sqrt_alphas_cumprod,
                timesteps,
                normalized_actions,
            )
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
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                "actions must have shape "
                f"[B, {self.horizon}, {self.action_dim}]"
            )
        if observation_history.ndim != 3 or tuple(
            observation_history.shape[1:]
        ) != (self.obs_horizon, self.obs_dim):
            raise ValueError(
                "observation_history must have shape "
                f"[B, {self.obs_horizon}, {self.obs_dim}]"
            )
        if actions.shape[0] != observation_history.shape[0]:
            raise ValueError("action and observation batch sizes differ")

        batch_size = actions.shape[0]
        timesteps = torch.randint(
            self.num_diffusion_steps,
            (batch_size,),
            device=actions.device,
            generator=generator,
        )
        noise = _randn(
            actions.shape,
            device=actions.device,
            dtype=actions.dtype,
            generator=generator,
        )
        normalized_observation = self.normalize_observations(
            observation_history
        )
        normalized_actions = self.normalize_actions(actions)
        noisy_actions = self.q_sample(
            normalized_actions,
            timesteps,
            noise,
        )
        predicted_noise = self.denoiser(
            noisy_actions,
            timesteps,
            normalized_observation,
        )
        return F.mse_loss(predicted_noise, noise)

    def forward(
        self,
        noisy_normalized_actions: torch.Tensor,
        timesteps: torch.Tensor,
        observation_history: torch.Tensor,
    ) -> torch.Tensor:
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
        num_inference_steps: int = DDIM_INFERENCE_STEPS,
        eta: float = 0.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if not 1 <= int(num_inference_steps) <= self.num_diffusion_steps:
            raise ValueError(
                "num_inference_steps must be in [1, num_diffusion_steps]"
            )
        if float(eta) < 0.0:
            raise ValueError("eta cannot be negative")
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        observation, squeeze_batch = prepare_observation_history(
            observation_history,
            obs_horizon=self.obs_horizon,
            obs_dim=self.obs_dim,
            device=device,
            dtype=dtype,
        )
        normalized_observation = self.normalize_observations(observation)
        generator = _generator(seed, device) if seed is not None else None
        sample = _randn(
            (observation.shape[0], self.horizon, self.action_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        timestep_values = (
            torch.linspace(
                self.num_diffusion_steps - 1,
                0,
                int(num_inference_steps),
                device="cpu",
            )
            .round()
            .to(torch.long)
            .unique_consecutive()
            .tolist()
        )
        was_training = self.training
        self.eval()
        try:
            for index, timestep in enumerate(timestep_values):
                batch_timestep = torch.full(
                    (observation.shape[0],),
                    timestep,
                    device=device,
                    dtype=torch.long,
                )
                predicted_noise = self.denoiser(
                    sample,
                    batch_timestep,
                    normalized_observation,
                )
                alpha = self.alphas_cumprod[timestep]
                if index + 1 < len(timestep_values):
                    previous_alpha = self.alphas_cumprod[
                        timestep_values[index + 1]
                    ]
                else:
                    previous_alpha = torch.ones(
                        (),
                        device=device,
                        dtype=dtype,
                    )
                predicted_clean = (
                    sample - (1.0 - alpha).sqrt() * predicted_noise
                ) / alpha.sqrt()
                predicted_clean = predicted_clean.clamp(
                    -self.sample_clip,
                    self.sample_clip,
                )
                sigma = float(eta) * torch.sqrt(
                    torch.clamp(
                        (1.0 - previous_alpha)
                        / (1.0 - alpha)
                        * (1.0 - alpha / previous_alpha),
                        min=0.0,
                    )
                )
                direction = torch.sqrt(
                    torch.clamp(
                        1.0 - previous_alpha - sigma.square(),
                        min=0.0,
                    )
                ) * predicted_noise
                sample = (
                    previous_alpha.sqrt() * predicted_clean + direction
                )
                if float(eta) > 0.0 and index + 1 < len(timestep_values):
                    sample = sample + sigma * _randn(
                        sample.shape,
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
        finally:
            self.train(was_training)
        action = self.denormalize_actions(sample)
        return action[0] if squeeze_batch else action


def prepare_observation_history(
    observations: torch.Tensor | np.ndarray | Sequence[Any],
    *,
    obs_horizon: int,
    obs_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, bool]:
    """Convert one observation/history or a batch to ``[B, H, obs_dim]``."""

    tensor = torch.as_tensor(observations, dtype=dtype, device=device)
    squeeze_batch = tensor.ndim < 3
    if tensor.ndim == 1:
        if tensor.shape[0] != int(obs_dim):
            raise ValueError(
                f"single observation must have {obs_dim} values, "
                f"got {tensor.shape[0]}"
            )
        tensor = tensor.reshape(1, 1, int(obs_dim))
    elif tensor.ndim == 2:
        if tensor.shape[1] != int(obs_dim) or tensor.shape[0] < 1:
            raise ValueError(
                f"observation history must have shape [T, {obs_dim}]"
            )
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tuple(tensor.shape[1:]) != (int(obs_horizon), int(obs_dim)):
            raise ValueError(
                "batched history must have shape "
                f"[B, {obs_horizon}, {obs_dim}]"
            )
        return tensor, False
    else:
        raise ValueError("observations must be 1-D, 2-D, or 3-D")

    if tensor.shape[1] > int(obs_horizon):
        tensor = tensor[:, -int(obs_horizon) :]
    elif tensor.shape[1] < int(obs_horizon):
        padding = tensor[:, :1].expand(
            -1,
            int(obs_horizon) - tensor.shape[1],
            -1,
        )
        tensor = torch.cat((padding, tensor), dim=1)
    return tensor, squeeze_batch


class ExponentialMovingAverage:
    """EMA shadow weights used for stable Diffusion inference."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise ValueError("EMA/model state keys differ")
        for name, value in current.items():
            shadow = self.shadow[name]
            value = value.detach().to(
                device=shadow.device,
                dtype=shadow.dtype,
            )
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(
                    value,
                    alpha=1.0 - self.decay,
                )
            else:
                shadow.copy_(value)

    def state_dict(self, *, cpu: bool = False) -> dict[str, torch.Tensor]:
        if cpu:
            return {
                name: value.detach().cpu()
                for name, value in self.shadow.items()
            }
        return {
            name: value.detach().clone()
            for name, value in self.shadow.items()
        }

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA checkpoint/model state keys differ")
        for name, value in state.items():
            target = self.shadow[name]
            self.shadow[name] = value.detach().clone().to(
                device=target.device,
                dtype=target.dtype,
            )


def policy_checkpoint(
    policy: DiffusionPolicy,
    *,
    ema: Optional[ExponentialMovingAverage] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Create a portable CPU checkpoint for training or evaluation."""

    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": policy.model_config(),
        "model_state": {
            name: value.detach().cpu()
            for name, value in policy.state_dict().items()
        },
        "ema_state": None if ema is None else ema.state_dict(cpu=True),
        "normalization": policy.normalization_dict(),
    }
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    return payload


def save_policy_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    policy: DiffusionPolicy,
    *,
    ema: Optional[ExponentialMovingAverage] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically save a portable policy/EMA checkpoint."""

    path = Path(checkpoint_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        policy_checkpoint(policy, ema=ema, metadata=metadata),
        temporary,
    )
    os.replace(temporary, path)
    return path


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint {path} does not contain a dictionary")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not an MJWarp GRAB Diffusion checkpoint")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "unsupported checkpoint format version "
            f"{checkpoint.get('format_version')!r}"
        )
    return checkpoint


def load_policy_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    device: str | torch.device = "auto",
    use_ema: bool = True,
    eval_mode: bool = True,
    require_environment_dimensions: bool = True,
) -> tuple[DiffusionPolicy, dict[str, Any]]:
    """Rebuild a policy from a checkpoint and validate the live contract."""

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = _torch_load(path)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint is missing model_config")
    policy = DiffusionPolicy(**dict(model_config))
    if require_environment_dimensions:
        expected_observation, expected_action = environment_dimensions()
        if (
            policy.obs_dim != expected_observation
            or policy.action_dim != expected_action
        ):
            raise ValueError(
                "checkpoint dimensions do not match the current GRAB "
                f"environment: checkpoint=({policy.obs_dim}, "
                f"{policy.action_dim}), current=({expected_observation}, "
                f"{expected_action})"
            )
    state = (
        checkpoint.get("ema_state")
        if use_ema and checkpoint.get("ema_state") is not None
        else checkpoint.get("model_state")
    )
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint is missing policy weights")
    policy.load_state_dict(state, strict=True)
    policy.to(resolve_device(device))
    if eval_mode:
        policy.eval()
    return policy, checkpoint


__all__ = [
    "ACTION_HORIZON",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "DDIM_INFERENCE_STEPS",
    "DDPM_TRAIN_STEPS",
    "DEFAULT_DEPTH",
    "DEFAULT_HIDDEN_DIM",
    "DiffusionPolicy",
    "ExponentialMovingAverage",
    "OBSERVATION_HORIZON",
    "REPLAN_INTERVAL",
    "build_default_policy",
    "default_model_config",
    "environment_dimensions",
    "load_policy_checkpoint",
    "policy_checkpoint",
    "prepare_observation_history",
    "resolve_device",
    "save_policy_checkpoint",
    "seed_everything",
]
