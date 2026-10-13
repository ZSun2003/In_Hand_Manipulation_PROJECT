"""Validated configuration for single-GPU MJWarp training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# The fixed 45-degree geometry first has positive collision-free spawn support
# above 36.284 mm.  A 60 mm operational floor also leaves enough accepted area
# for the bounded 128-draw GPU sampler at the largest supported vector batches.
MIN_RELIABLE_CURRICULUM_SPAWN_RADIUS_M = 0.060


@dataclass(frozen=True)
class PhysicsConfig:
    worlds: int = 1024
    physics_substeps: int = 40
    nconmax: int = 300
    njmax: int = 2000
    max_episode_steps: int = 4000
    cuda_graph: bool = True
    contact_sensor_maxmatch: int = 64


@dataclass(frozen=True)
class IKConfig:
    linear_samples: int = 51


@dataclass(frozen=True)
class DemoConfig:
    episodes: int = 4096
    shard_episodes: int = 64
    obs_horizon: int = 2
    action_horizon: int = 16
    seed: int = 20260726


@dataclass(frozen=True)
class DiffusionConfig:
    epochs: int = 30
    batch_size: int = 2048
    learning_rate: float = 3.0e-4
    hidden_dim: int = 256
    depth: int = 6
    diffusion_steps: int = 100
    inference_steps: int = 16
    replan_interval: int = 4
    ema_decay: float = 0.999


@dataclass(frozen=True)
class TQCConfig:
    total_transitions: int = 5_000_000
    learning_starts: int = 20_000
    # Screw-specific safety delay: critics first see a broader portion of the
    # nominal rollout before the residual actor/temperature are updated.
    actor_learning_starts: int = 200_000
    replay_capacity: int = 1_000_000
    batch_size: int = 256
    # Gradient updates per newly inserted replay batch.  With the formal
    # 256-row admission cap and batch size 256, 8.0 reproduces the validated
    # G6 setting of eight gradient steps per vector-environment tick.
    replay_ratio: float = 8.0
    # Cap spatially correlated samples admitted from one 14k-world vector
    # tick.  ``total_transitions`` counts these stored replay rows.
    replay_insert_per_vector_step: int = 256
    gradient_steps_per_vector_step: int = 8
    actor_lr: float = 1.0e-4
    critic_lr: float = 1.0e-4
    alpha_lr: float = 1.0e-4
    gamma: float = 0.99
    tau: float = 0.005
    critics: int = 4
    quantiles: int = 32
    top_quantiles_to_drop: int = 2
    hidden_dim: int = 256
    log_std_init: float = -3.0
    checkpoint_every_transitions: int = 250_000


@dataclass(frozen=True)
class CurriculumConfig:
    """Spawn-radius difficulty schedule for the fixed-angle task."""

    enabled: bool = True
    # True preserves the original opening-facing half-disc distribution.
    # False samples the complete workspace disc before applying the unchanged
    # conservative socket-clearance and scripted-prefix IK rejection gates.
    restrict_screw_spawn_to_opening_half_disk: bool = True
    start_spawn_radius_m: float = MIN_RELIABLE_CURRICULUM_SPAWN_RADIUS_M
    final_spawn_radius_m: float = 0.100
    fixed_socket_angle_deg: float = 45.0
    warmup_fraction: float = 0.05
    full_difficulty_fraction: float = 0.70


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 20260726
    device: str = "cuda:0"
    dtype: str = "float32"
    log_every_vector_steps: int = 50
    status_interval_seconds: float = 10.0
    masked_reset: bool = True
    deterministic: bool = False


@dataclass(frozen=True)
class HPCConfig:
    name: str = "cx3_l40s"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    demos: DemoConfig = field(default_factory=DemoConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    tqc: TQCConfig = field(default_factory=TQCConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _section(cls: type, payload: Mapping[str, Any], name: str):
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be an object")
    unknown = set(value) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown keys in {name}: {sorted(unknown)}")
    return cls(**dict(value))


def canonicalize_config_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a detached config mapping with legacy-safe defaults applied.

    Checkpoints and demonstration stores created before the full-disc switch
    do not contain ``restrict_screw_spawn_to_opening_half_disk``.  Missing
    therefore means ``True`` (the only behaviour those artifacts could have
    used), while an explicit ``False`` remains a distinct training
    distribution and must not compare equal to legacy data.
    """

    normalized = json.loads(json.dumps(dict(payload), allow_nan=False))
    curriculum = normalized.get("curriculum")
    if isinstance(curriculum, dict):
        curriculum.setdefault(
            "restrict_screw_spawn_to_opening_half_disk", True
        )
    return normalized


def validate_config(config: HPCConfig) -> HPCConfig:
    p = config.physics
    if p.worlds <= 0 or p.physics_substeps <= 0:
        raise ValueError("worlds and physics_substeps must be positive")
    if p.nconmax <= 0 or p.njmax <= 0 or p.max_episode_steps <= 0:
        raise ValueError("MJWarp capacities and episode length must be positive")
    if p.contact_sensor_maxmatch <= 0:
        raise ValueError("contact_sensor_maxmatch must be positive")
    if config.ik.linear_samples < 3:
        raise ValueError("IK linear_samples must be at least 3")
    d = config.demos
    if d.episodes <= 0 or d.shard_episodes <= 0:
        raise ValueError("demo episodes and shard_episodes must be positive")
    if config.demos.obs_horizon != 2:
        raise ValueError("the v2 observation contract requires obs_horizon=2")
    if config.demos.action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    diffusion = config.diffusion
    if (
        diffusion.epochs <= 0
        or diffusion.batch_size <= 0
        or diffusion.hidden_dim <= 0
        or diffusion.depth <= 0
        or diffusion.diffusion_steps <= 0
        or diffusion.inference_steps <= 0
        or diffusion.replan_interval <= 0
    ):
        raise ValueError("diffusion sizes and step counts must be positive")
    if diffusion.learning_rate <= 0.0:
        raise ValueError("diffusion learning_rate must be positive")
    if not 0.0 <= diffusion.ema_decay < 1.0:
        raise ValueError("diffusion ema_decay must be in [0, 1)")
    t = config.tqc
    if t.critics <= 0 or t.quantiles <= 1:
        raise ValueError("TQC needs positive critics and at least two quantiles")
    if not 0 <= t.top_quantiles_to_drop < t.quantiles:
        raise ValueError("invalid TQC top-quantile drop")
    if (
        t.total_transitions <= 0
        or t.learning_starts < 0
        or t.actor_learning_starts < 0
        or t.replay_capacity <= 0
        or t.batch_size <= 0
        or t.hidden_dim <= 0
        or t.replay_insert_per_vector_step <= 0
        or t.gradient_steps_per_vector_step <= 0
        or t.checkpoint_every_transitions <= 0
    ):
        raise ValueError("TQC capacities and transition counts are invalid")
    if t.learning_starts >= t.total_transitions:
        raise ValueError("TQC learning_starts must precede total_transitions")
    if not t.learning_starts <= t.actor_learning_starts < t.total_transitions:
        raise ValueError(
            "TQC actor_learning_starts must follow critic learning_starts "
            "and precede total_transitions"
        )
    if t.batch_size > t.replay_capacity:
        raise ValueError("TQC batch_size cannot exceed replay_capacity")
    if t.replay_insert_per_vector_step > config.physics.worlds:
        raise ValueError(
            "TQC replay_insert_per_vector_step cannot exceed physics worlds"
        )
    if t.replay_ratio <= 0.0 or min(t.actor_lr, t.critic_lr, t.alpha_lr) <= 0.0:
        raise ValueError("TQC replay ratio and learning rates must be positive")
    if not -20.0 <= t.log_std_init <= 2.0:
        raise ValueError("TQC log_std_init must be in [-20, 2]")
    if not 0.0 < t.gamma <= 1.0 or not 0.0 < t.tau <= 1.0:
        raise ValueError("TQC gamma and tau must be in (0, 1]")
    curriculum = config.curriculum
    if not isinstance(
        curriculum.restrict_screw_spawn_to_opening_half_disk, bool
    ):
        raise ValueError(
            "curriculum.restrict_screw_spawn_to_opening_half_disk must be bool"
        )
    if not 0.0 < curriculum.start_spawn_radius_m <= curriculum.final_spawn_radius_m:
        raise ValueError("curriculum spawn radii must be positive and ordered")
    if (
        curriculum.enabled
        and curriculum.start_spawn_radius_m
        < MIN_RELIABLE_CURRICULUM_SPAWN_RADIUS_M - 1.0e-12
    ):
        raise ValueError(
            "enabled curriculum start_spawn_radius_m must be at least "
            f"{MIN_RELIABLE_CURRICULUM_SPAWN_RADIUS_M:.3f} m for the "
            "fixed socket geometry and bounded GPU rejection sampler"
        )
    if curriculum.final_spawn_radius_m > 0.100 + 1.0e-12:
        raise ValueError("curriculum cannot exceed the task's 100 mm spawn radius")
    if abs(curriculum.fixed_socket_angle_deg - 45.0) > 1.0e-9:
        raise ValueError("the formal hpc task requires a fixed 45 degree socket")
    if not (
        0.0
        <= curriculum.warmup_fraction
        < curriculum.full_difficulty_fraction
        <= 1.0
    ):
        raise ValueError("curriculum progress fractions are invalid")
    runtime = config.runtime
    if not runtime.device.startswith("cuda"):
        raise ValueError("hpc is CUDA-only; use the root MPS profile on macOS")
    if runtime.dtype != "float32":
        raise ValueError("hpc v2 uses float32 physics and control tensors")
    if runtime.log_every_vector_steps <= 0:
        raise ValueError("log_every_vector_steps must be positive")
    if runtime.status_interval_seconds <= 0.0:
        raise ValueError("status_interval_seconds must be positive")
    if not runtime.masked_reset:
        raise ValueError("hpc v2 requires masked per-world reset")
    return config


def load_config(path: str | Path) -> HPCConfig:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("top-level config must be a JSON object")
    allowed = set(HPCConfig.__dataclass_fields__)
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    config = HPCConfig(
        name=str(payload.get("name", source.stem)),
        physics=_section(PhysicsConfig, payload, "physics"),
        ik=_section(IKConfig, payload, "ik"),
        demos=_section(DemoConfig, payload, "demos"),
        diffusion=_section(DiffusionConfig, payload, "diffusion"),
        tqc=_section(TQCConfig, payload, "tqc"),
        curriculum=_section(CurriculumConfig, payload, "curriculum"),
        runtime=_section(RuntimeConfig, payload, "runtime"),
    )
    return validate_config(config)


__all__ = [
    "HPCConfig",
    "CurriculumConfig",
    "MIN_RELIABLE_CURRICULUM_SPAWN_RADIUS_M",
    "PACKAGE_ROOT",
    "canonicalize_config_payload",
    "load_config",
    "validate_config",
]
