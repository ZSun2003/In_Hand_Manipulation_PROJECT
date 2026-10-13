"""Train one TQC IK+residual policy and save only its final artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

from .runtime import ensure_runtime

ensure_runtime()

import gymnasium as gym

from . import config
from .env import G3IKResidualEnv


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _env_factory(rank: int, seed: int) -> Callable[[], gym.Env]:
    def initialize() -> gym.Env:
        environment = G3IKResidualEnv(control_mode="ik_residual")
        environment.action_space.seed(seed + rank)
        return environment

    return initialize


def build_vec_env(
    *,
    n_envs: int,
    seed: int,
    vec_env_type: str,
):
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
        VecNormalize,
    )

    factories = [_env_factory(rank, seed) for rank in range(n_envs)]
    if vec_env_type == "subproc":
        vector = SubprocVecEnv(factories, start_method="spawn")
    elif vec_env_type == "dummy":
        vector = DummyVecEnv(factories)
    else:
        raise ValueError("vec_env_type must be 'subproc' or 'dummy'")
    vector.seed(seed)
    monitored = VecMonitor(vector)
    return VecNormalize(
        monitored,
        training=True,
        norm_obs=config.NORM_OBS,
        norm_reward=config.NORM_REWARD,
        clip_obs=config.CLIP_OBS,
        clip_reward=config.CLIP_REWARD,
    )


def build_model(
    vector_env,
    *,
    seed: int,
    device: str,
    total_timesteps: int = config.TOTAL_TIMESTEPS,
):
    from sb3_contrib import TQC

    formal_run = int(total_timesteps) == config.TOTAL_TIMESTEPS
    return TQC(
        "MlpPolicy",
        vector_env,
        learning_rate=config.LEARNING_RATE,
        buffer_size=(
            config.BUFFER_SIZE
            if formal_run
            else max(256, int(total_timesteps) + 1)
        ),
        learning_starts=(
            config.LEARNING_STARTS
            if formal_run
            else max(100, int(total_timesteps) + 1)
        ),
        batch_size=config.BATCH_SIZE,
        tau=config.TAU,
        gamma=config.GAMMA,
        train_freq=config.TRAIN_FREQ,
        gradient_steps=config.GRADIENT_STEPS,
        top_quantiles_to_drop_per_net=(
            config.TQC_TOP_QUANTILES_TO_DROP_PER_NET
        ),
        use_sde=config.TQC_USE_SDE,
        sde_sample_freq=config.TQC_SDE_SAMPLE_FREQ,
        policy_kwargs={
            "net_arch": list(config.POLICY_NET_ARCH),
            "n_critics": config.TQC_N_CRITICS,
            "n_quantiles": config.TQC_N_QUANTILES,
        },
        verbose=1,
        seed=seed,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the G3 current analytic-IK controller plus a bounded "
            "14D policy (12 actuator residuals plus the G3 banana-only "
            "height/yaw reachability residual) with one uninterrupted TQC "
            "learn call."
        )
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=config.TOTAL_TIMESTEPS,
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-envs", type=int, default=config.N_ENVS)
    parser.add_argument(
        "--vec-env",
        choices=("subproc", "dummy"),
        default=config.VEC_ENV_TYPE,
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Artifact directory; default is IK_RL/models.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> tuple[Path, Path]:
    config.validate()
    total_timesteps = int(args.total_timesteps)
    n_envs = int(args.n_envs)
    if total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if total_timesteps % n_envs:
        raise ValueError(
            "--total-timesteps must be divisible by --n-envs so the single "
            "vectorized learn call ends at the exact requested count"
        )

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = config.DEFAULT_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    model_base = output_dir / "final_model"
    model_path = output_dir / "final_model.zip"
    normalize_path = output_dir / "vec_normalize.pkl"
    for path in (model_path, normalize_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite training artifact: {path}")

    vector_env = build_vec_env(
        n_envs=n_envs,
        seed=int(args.seed),
        vec_env_type=str(args.vec_env),
    )
    try:
        model = build_model(
            vector_env,
            seed=int(args.seed),
            device=str(args.device),
            total_timesteps=total_timesteps,
        )
        # Formal training is exactly this one uninterrupted call: no
        # curriculum, shell, success gate, early stop, or checkpoint callback.
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        model.save(str(model_base))
        vector_env.save(str(normalize_path))
    finally:
        vector_env.close()

    if not model_path.is_file() or not normalize_path.is_file():
        raise RuntimeError("final TQC artifacts were not written")
    return model_path, normalize_path


def main() -> None:
    model_path, normalize_path = train(parse_args())
    print(f"final model: {model_path}")
    print(f"VecNormalize: {normalize_path}")


if __name__ == "__main__":
    main()
