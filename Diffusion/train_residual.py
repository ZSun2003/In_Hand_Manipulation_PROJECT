#!/usr/bin/env python3
"""Train full-state residual TQC with parallel CPU MuJoCo workers."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from central_diffusion import CentralDiffusionVecEnv
from residual_control import RESIDUAL_ACTION_DIM
from runtime import PROJECT_ROOT, ensure_runtime_dirs
from screw_config import OBSERVATION_SCHEMA, TASK_GEOMETRY_HASH
from screw_env import ScrewTaskEnv


def auto_workers() -> int:
    logical = os.cpu_count() or 8
    # Leave cores for the learner, macOS window server and data feeding.
    return max(2, min(12, logical - 4))


def select_device(requested: str) -> str:
    value = requested.lower()
    if value != "auto":
        if value == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return value
    # TQC performs several quantile-critic updates per environment step and is
    # substantially more matrix-heavy than PPO, making MPS worthwhile here.
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _inside_project(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must remain inside SCREW: {resolved}") from exc
    return resolved


def _entropy_coefficient(value: str) -> str | float:
    text = value.strip().lower()
    if text == "auto" or text.startswith("auto_"):
        return text
    try:
        coefficient = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "entropy coefficient must be a number, auto, or auto_<initial>"
        ) from exc
    if coefficient < 0.0:
        raise argparse.ArgumentTypeError(
            "entropy coefficient must be non-negative"
        )
    return coefficient


def _resume_sidecar_path(
    checkpoint: Path,
    *,
    kind: str,
    explicit: Optional[Path] = None,
) -> Path:
    """Resolve both final-output and CheckpointCallback sidecar names."""

    if explicit is not None:
        return _inside_project(explicit)
    standard = checkpoint.with_suffix(f".{kind}.pkl")
    if standard.is_file():
        return standard
    match = re.fullmatch(r"(.+)_(\d+)_steps", checkpoint.stem)
    if match is not None:
        callback_name = (
            f"{match.group(1)}_{kind}_{match.group(2)}_steps.pkl"
        )
        callback_path = checkpoint.with_name(callback_name)
        if callback_path.is_file():
            return callback_path
    return standard


def make_env_factory(
    *,
    rank: int,
    seed: int,
    baseline: str,
    diffusion_checkpoint: Optional[Path],
    guide_mode: str,
    guide_scale: float,
    random_spawn: bool,
) -> Callable[[], ScrewTaskEnv]:
    def factory() -> ScrewTaskEnv:
        env = ScrewTaskEnv(
            baseline=baseline,
            diffusion_checkpoint=diffusion_checkpoint,
            random_spawn=random_spawn,
            guide_mode=guide_mode,
            guide_scale=guide_scale,
            strict_ik=False,
            seed=seed + rank,
        )
        env.reset(seed=seed + rank)
        return env

    return factory


def make_vector_env(
    *,
    workers: int,
    seed: int,
    baseline: str,
    diffusion_checkpoint: Optional[Path],
    guide_mode: str,
    guide_scale: float,
    random_spawn: bool,
):
    factories = [
        make_env_factory(
            rank=rank,
            seed=seed,
            baseline=baseline,
            diffusion_checkpoint=diffusion_checkpoint,
            guide_mode=guide_mode,
            guide_scale=guide_scale,
            random_spawn=random_spawn,
        )
        for rank in range(workers)
    ]
    if workers == 1:
        vector = DummyVecEnv(factories)
    else:
        vector = SubprocVecEnv(factories, start_method="spawn")
    return VecMonitor(vector)


def build_parser() -> argparse.ArgumentParser:
    dirs = ensure_runtime_dirs()
    parser = argparse.ArgumentParser(
        description=(
            "Train off-policy residual TQC. MuJoCo/IK run on CPU workers; "
            "actor/quantile critics use Apple MPS by default."
        )
    )
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--workers", type=int, default=auto_workers())
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--baseline",
        choices=("expert", "diffusion"),
        default="diffusion",
    )
    parser.add_argument(
        "--diffusion-checkpoint",
        type=Path,
        default=dirs["checkpoints"] / "screw_diffusion.pt",
    )
    parser.add_argument(
        "--diffusion-device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="device for the single main-process batched Diffusion policy",
    )
    parser.add_argument(
        "--inference-steps",
        "--diffusion-inference-steps",
        dest="diffusion_inference_steps",
        type=int,
        default=8,
        help="DDIM denoising evaluations per central replan",
    )
    parser.add_argument(
        "--replan-interval",
        "--diffusion-replan-interval",
        dest="diffusion_replan_interval",
        type=int,
        default=4,
        help="open-loop action-chunk steps between batched replans",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=dirs["checkpoints"] / "screw_residual_tqc",
    )
    parser.add_argument("--tensorboard-dir", type=Path, default=dirs["runs"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=-1,
        help=(
            "critic/actor updates per vector step; -1 matches the number of "
            "new transitions across all workers"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--n-critics", type=int, default=2)
    parser.add_argument("--n-quantiles", type=int, default=25)
    parser.add_argument("--top-quantiles-to-drop", type=int, default=2)
    parser.add_argument(
        "--ent-coef",
        type=_entropy_coefficient,
        default="auto_0.1",
    )
    parser.add_argument(
        "--guide-mode",
        choices=("none", "wrench", "mocap"),
        default="wrench",
    )
    parser.add_argument("--guide-scale", type=float, default=0.35)
    parser.add_argument(
        "--fixed-spawn",
        action="store_true",
        help="Debug mode; training normally samples the reachable disk.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a TQC .zip checkpoint.",
    )
    parser.add_argument(
        "--resume-vecnormalize",
        type=Path,
        default=None,
        help="defaults to the .vecnormalize.pkl beside --resume",
    )
    parser.add_argument(
        "--resume-replay-buffer",
        type=Path,
        default=None,
        help="defaults to the .replay_buffer.pkl beside --resume",
    )
    parser.add_argument(
        "--save-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save replay state for an exact off-policy curriculum resume",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timesteps <= 0 or args.workers <= 0:
        raise SystemExit("--timesteps and --workers must be positive")
    positive_arguments = {
        "--batch-size": args.batch_size,
        "--buffer-size": args.buffer_size,
        "--train-freq": args.train_freq,
        "--n-critics": args.n_critics,
        "--n-quantiles": args.n_quantiles,
    }
    if any(value <= 0 for value in positive_arguments.values()):
        raise SystemExit(
            ", ".join(positive_arguments) + " must all be positive"
        )
    if args.learning_starts < 0 or not (
        args.gradient_steps == -1 or args.gradient_steps >= 1
    ):
        raise SystemExit(
            "--learning-starts must be non-negative and "
            "--gradient-steps must be positive or -1"
        )
    if (
        args.diffusion_inference_steps <= 0
        or args.diffusion_replan_interval <= 0
    ):
        raise SystemExit(
            "--inference-steps and --replan-interval must be positive"
        )
    if not 0.0 < args.tau <= 1.0 or not 0.0 < args.gamma <= 1.0:
        raise SystemExit("--tau and --gamma must lie in (0, 1]")
    if not 0 <= args.top_quantiles_to_drop < args.n_quantiles:
        raise SystemExit(
            "--top-quantiles-to-drop must lie in [0, --n-quantiles)"
        )
    output = _inside_project(args.output)
    tensorboard_dir = _inside_project(args.tensorboard_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _inside_project(args.diffusion_checkpoint)
    if args.baseline == "diffusion" and not checkpoint.is_file():
        raise SystemExit(
            f"diffusion checkpoint not found: {checkpoint}\n"
            "Run generate_demos.py then train_diffusion.py first."
        )
    device = select_device(args.device)
    print(
        json.dumps(
            {
                "device": device,
                "workers": args.workers,
                "cpu_count": os.cpu_count(),
                "baseline": args.baseline,
                "guide_mode": args.guide_mode,
                "guide_scale": args.guide_scale,
                "algorithm": "TQC",
                "diffusion_execution": (
                    "central_batched" if args.baseline == "diffusion" else "none"
                ),
                "diffusion_device": (
                    args.diffusion_device
                    if args.baseline == "diffusion"
                    else None
                ),
                "diffusion_inference_steps": (
                    args.diffusion_inference_steps
                    if args.baseline == "diffusion"
                    else None
                ),
                "diffusion_replan_interval": (
                    args.diffusion_replan_interval
                    if args.baseline == "diffusion"
                    else None
                ),
                "buffer_size": args.buffer_size,
                "batch_size": args.batch_size,
                "gradient_steps": args.gradient_steps,
                "unified_memory_gb": "system managed",
            },
            indent=2,
        ),
        flush=True,
    )

    env = make_vector_env(
        workers=args.workers,
        seed=args.seed,
        baseline=(
            "external_diffusion"
            if args.baseline == "diffusion"
            else args.baseline
        ),
        diffusion_checkpoint=None,
        guide_mode=args.guide_mode,
        guide_scale=args.guide_scale,
        random_spawn=not args.fixed_spawn,
    )
    if args.baseline == "diffusion":
        env = CentralDiffusionVecEnv(
            env,
            checkpoint=checkpoint,
            device=args.diffusion_device,
            inference_steps=args.diffusion_inference_steps,
            replan_interval=args.diffusion_replan_interval,
            residual_action_dim=RESIDUAL_ACTION_DIM,
            seed=args.seed,
        )
    resume: Optional[Path] = None
    if args.resume is not None:
        resume = _inside_project(args.resume)
        if not resume.is_file():
            raise SystemExit(f"resume checkpoint not found: {resume}")
        normalization_path = _resume_sidecar_path(
            resume,
            kind="vecnormalize",
            explicit=args.resume_vecnormalize,
        )
        if not normalization_path.is_file():
            raise SystemExit(
                f"VecNormalize state not found: {normalization_path}"
            )
        normalized = VecNormalize.load(normalization_path, env)
        normalized.training = True
        normalized.norm_reward = True
    else:
        normalized = VecNormalize(
            env,
            training=True,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=args.gamma,
        )
    try:
        if resume is not None:
            model = TQC.load(
                resume,
                env=normalized,
                device=device,
            )
            if (
                getattr(model, "screw_observation_schema", None)
                != OBSERVATION_SCHEMA
                or getattr(model, "screw_task_geometry_hash", None)
                != TASK_GEOMETRY_HASH
            ):
                raise ValueError(
                    "residual checkpoint belongs to an incompatible "
                    "observation schema or task geometry"
                )
            replay_path = _resume_sidecar_path(
                resume,
                kind="replay_buffer",
                explicit=args.resume_replay_buffer,
            )
            if replay_path.is_file():
                model.load_replay_buffer(replay_path)
                print(f"loaded replay buffer: {replay_path}", flush=True)
            else:
                print(
                    f"warning: replay buffer not found; resuming with an "
                    f"empty buffer: {replay_path}",
                    flush=True,
                )
        else:
            model = TQC(
                "MlpPolicy",
                normalized,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                tau=args.tau,
                gamma=args.gamma,
                train_freq=(args.train_freq, "step"),
                gradient_steps=args.gradient_steps,
                ent_coef=args.ent_coef,
                top_quantiles_to_drop_per_net=(
                    args.top_quantiles_to_drop
                ),
                policy_kwargs={
                    "net_arch": {
                        "pi": [256, 256],
                        "qf": [256, 256],
                    },
                    "n_critics": args.n_critics,
                    "n_quantiles": args.n_quantiles,
                },
                tensorboard_log=str(tensorboard_dir),
                device=device,
                seed=args.seed,
                verbose=1,
            )
        model.screw_observation_schema = OBSERVATION_SCHEMA
        model.screw_task_geometry_hash = TASK_GEOMETRY_HASH
        save_frequency = max(100_000 // args.workers, 1)
        callback = CheckpointCallback(
            save_freq=save_frequency,
            save_path=str(output.parent / "residual_snapshots"),
            name_prefix=output.stem,
            save_replay_buffer=args.save_replay_buffer,
            save_vecnormalize=True,
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=args.resume is None,
        )
        model.save(output)
        replay_output = output.with_suffix(".replay_buffer.pkl")
        if args.save_replay_buffer:
            model.save_replay_buffer(replay_output)
        normalized.save(output.with_suffix(".vecnormalize.pkl"))
        print(f"saved residual policy: {output.with_suffix('.zip')}")
        if args.save_replay_buffer:
            print(f"saved replay buffer: {replay_output}")
        print(
            "Next curriculum stage: rerun with --guide-mode none "
            f"--resume {output.with_suffix('.zip')}"
        )
    finally:
        normalized.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
