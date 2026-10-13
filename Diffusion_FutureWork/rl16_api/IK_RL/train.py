#!/usr/bin/env python3
"""Train one TQC policy for all six G6 objects and both manipulation tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import config
    from .runtime import IK_RL_DIR, prepare_runtime
except ImportError:
    import config  # type: ignore
    from runtime import IK_RL_DIR, prepare_runtime  # type: ignore


prepare_runtime()

from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

try:
    from .env import G6IKResidualEnv
except ImportError:
    from env import G6IKResidualEnv  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the single G6 IK+residual TQC policy for six objects and "
            "uniformly sampled translation/rotation tasks."
        )
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=config.TOTAL_TIMESTEPS,
        help="one uninterrupted learn call (default: 2,000,000)",
    )
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-envs", type=int, default=config.N_ENVS)
    parser.add_argument(
        "--vec-env",
        choices=("subproc", "dummy"),
        default="subproc",
        help="default: 8 spawned MuJoCo workers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=IK_RL_DIR / "models",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="show Stable-Baselines3 progress output",
    )
    return parser.parse_args()


def _make_monitored_env() -> Monitor:
    return Monitor(G6IKResidualEnv())


def main() -> None:
    args = parse_args()
    if args.total_timesteps <= 0:
        raise SystemExit("--total-timesteps must be positive")
    if args.n_envs <= 0:
        raise SystemExit("--n-envs must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    factories = [_make_monitored_env for _ in range(args.n_envs)]
    if args.vec_env == "subproc" and args.n_envs > 1:
        vec_env = SubprocVecEnv(factories, start_method="spawn")
    else:
        vec_env = DummyVecEnv(factories)
    vec_env.seed(args.seed)
    vec_env = VecNormalize(
        vec_env,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
    )
    model = TQC(
        "MlpPolicy",
        vec_env,
        policy_kwargs={
            "net_arch": [256, 256],
            "n_critics": 4,
            "n_quantiles": 32,
        },
        learning_rate=1e-4,
        buffer_size=1_000_000,
        learning_starts=20_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=8,
        top_quantiles_to_drop_per_net=2,
        use_sde=True,
        sde_sample_freq=4,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    try:
        # Intentionally one learn call: no phases, shell, success gate,
        # checkpoint callback, early stop, or curriculum.
        model.learn(
            total_timesteps=args.total_timesteps,
            progress_bar=args.progress_bar,
        )
        model.save(output_dir / "final_model")
        vec_env.save(output_dir / "vec_normalize.pkl")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
