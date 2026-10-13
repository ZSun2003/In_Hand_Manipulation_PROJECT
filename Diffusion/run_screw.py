#!/usr/bin/env python3
"""Interactive/headless entry point for the complete SCREW stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Optional, Sequence

import mujoco
import numpy as np

from runtime import PROJECT_ROOT, ensure_runtime_dirs
from screw_config import OBSERVATION_SCHEMA, TASK_GEOMETRY_HASH
from screw_env import ScrewTaskEnv


def build_parser() -> argparse.ArgumentParser:
    dirs = ensure_runtime_dirs()
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed-45-degree, two-turn screw task. "
            "The default expert view "
            "uses the demonstration guide; trained hybrid evaluation should "
            "use --guide-mode none."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("expert", "diffusion", "hybrid"),
        default="expert",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fixed-spawn", action="store_true")
    parser.add_argument(
        "--fixed-socket",
        action="store_true",
        help="Use the canonical socket mouth and angle for debugging.",
    )
    parser.add_argument(
        "--guide-mode",
        choices=("auto", "mocap", "wrench", "none"),
        default="auto",
    )
    parser.add_argument("--guide-scale", type=float, default=0.35)
    parser.add_argument(
        "--diffusion-checkpoint",
        type=Path,
        default=dirs["checkpoints"] / "screw_diffusion.pt",
    )
    parser.add_argument(
        "--residual-checkpoint",
        type=Path,
        default=dirs["checkpoints"] / "screw_residual_tqc.zip",
    )
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        default=dirs["checkpoints"]
        / "screw_residual_tqc.vecnormalize.pkl",
    )
    parser.add_argument(
        "--no-real-time",
        action="store_true",
        help="Do not pace GUI playback at 50 Hz.",
    )
    return parser


def _project_file(path: Path, *, required: bool = True) -> Path:
    result = path.expanduser().resolve()
    try:
        result.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"checkpoint must remain inside SCREW: {result}") from exc
    if required and not result.is_file():
        raise FileNotFoundError(result)
    return result


def _guide(args: argparse.Namespace) -> tuple[str, float]:
    if args.guide_mode != "auto":
        return args.guide_mode, (
            args.guide_scale if args.guide_mode == "wrench" else 1.0
        )
    if args.mode == "expert":
        return "mocap", 1.0
    return "none", 0.0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    guide_mode, guide_scale = _guide(args)
    baseline = "expert" if args.mode == "expert" else "diffusion"
    diffusion = (
        None
        if baseline == "expert"
        else _project_file(args.diffusion_checkpoint)
    )
    env = ScrewTaskEnv(
        baseline=baseline,
        diffusion_checkpoint=diffusion,
        random_spawn=not args.fixed_spawn,
        random_socket_position=not args.fixed_socket,
        random_socket_angle=False,
        guide_mode=guide_mode,
        guide_scale=guide_scale,
        strict_ik=False,
        seed=args.seed,
    )
    vector_env = None
    model = None
    try:
        if args.mode == "hybrid":
            from sb3_contrib import TQC
            from diffusion_policy import resolve_device
            from stable_baselines3.common.vec_env import (
                DummyVecEnv,
                VecNormalize,
            )

            residual = _project_file(args.residual_checkpoint)
            normalization = _project_file(args.vecnormalize)
            vector_env = DummyVecEnv([lambda: env])
            vector_env = VecNormalize.load(normalization, vector_env)
            vector_env.training = False
            vector_env.norm_reward = False
            observation = vector_env.reset()
            model = TQC.load(
                residual,
                env=vector_env,
                device=resolve_device("auto"),
            )
            if (
                getattr(model, "screw_observation_schema", None)
                != OBSERVATION_SCHEMA
                or getattr(model, "screw_task_geometry_hash", None)
                != TASK_GEOMETRY_HASH
            ):
                raise ValueError(
                    "residual checkpoint belongs to an incompatible "
                    "task geometry"
                )
        else:
            observation, reset_info = env.reset(seed=args.seed)
            print(
                "spawn_xy="
                + np.asarray(reset_info["spawn_xy"]).round(4).__repr__()
                + " socket_mouth_xy="
                + np.asarray(reset_info["socket_mouth_xy"])
                .round(4)
                .__repr__()
                + f" socket_angle_deg={reset_info['socket_angle_deg']:.0f}"
            )

        viewer_context = None
        if not args.headless:
            import mujoco.viewer

            viewer_context = mujoco.viewer.launch_passive(
                env.sim.model,
                env.sim.data,
            )
        total_reward = 0.0
        final_info = {}
        start = time.monotonic()
        try:
            for step in range(args.steps):
                tick = time.monotonic()
                if args.mode == "hybrid":
                    assert vector_env is not None and model is not None
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )
                    observation, reward, done, infos = vector_env.step(action)
                    total_reward += float(reward[0])
                    final_info = infos[0]
                    finished = bool(done[0])
                else:
                    observation, reward, terminated, truncated, final_info = (
                        env.step(
                            np.zeros(env.action_space.shape, dtype=np.float32)
                        )
                    )
                    total_reward += reward
                    finished = bool(terminated or truncated)

                if viewer_context is not None:
                    if not viewer_context.is_running():
                        break
                    viewer_context.sync()
                    if not args.no_real_time:
                        remaining = env.sim.control_dt_s - (
                            time.monotonic() - tick
                        )
                        if remaining > 0.0:
                            time.sleep(remaining)
                if finished:
                    break
        finally:
            if viewer_context is not None:
                viewer_context.close()

        metrics = env.sim.success_metrics()
        summary = {
            "mode": args.mode,
            "guide_mode": guide_mode,
            "steps": env.sim.control_step,
            "wall_time_s": time.monotonic() - start,
            "reward": total_reward,
            **metrics,
            "phase": env.sim.current_command.phase.name,
            "thread_engaged": env.sim.thread_engaged,
            "holder_active": env.sim.holder_active,
            "grasp_latched": env.sim.grasp_latched,
            "failure_reason": final_info.get("failure_reason", ""),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if not bool(summary["failure_reason"]) else 2
    finally:
        if vector_env is not None:
            vector_env.close()
        else:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
