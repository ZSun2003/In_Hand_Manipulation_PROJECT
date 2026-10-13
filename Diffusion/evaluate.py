#!/usr/bin/env python3
"""Evaluate expert, Diffusion, or Diffusion+residual policies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from diffusion_policy import resolve_device
from runtime import PROJECT_ROOT, ensure_runtime_dirs
from screw_config import OBSERVATION_SCHEMA, TASK_GEOMETRY_HASH
from screw_env import ScrewTaskEnv


def _project_path(path: Path) -> Path:
    result = path.expanduser().resolve()
    try:
        result.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must remain inside SCREW: {result}") from exc
    return result


def build_parser() -> argparse.ArgumentParser:
    dirs = ensure_runtime_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("expert", "diffusion", "hybrid"),
        default="expert",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260725)
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
        "--guide-mode",
        choices=("none", "wrench", "mocap"),
        default="none",
    )
    parser.add_argument("--guide-scale", type=float, default=0.0)
    parser.add_argument("--fixed-spawn", action="store_true")
    parser.add_argument(
        "--fixed-socket",
        action="store_true",
        help="Use the canonical socket mouth and angle for debugging.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=dirs["outputs"] / "evaluation.json",
    )
    return parser


def _base_env(args: argparse.Namespace, rank_seed: int) -> ScrewTaskEnv:
    baseline = "expert" if args.mode == "expert" else "diffusion"
    checkpoint = (
        None
        if baseline == "expert"
        else _project_path(args.diffusion_checkpoint)
    )
    return ScrewTaskEnv(
        baseline=baseline,
        diffusion_checkpoint=checkpoint,
        random_spawn=not args.fixed_spawn,
        random_socket_position=not args.fixed_socket,
        random_socket_angle=False,
        guide_mode=args.guide_mode,
        guide_scale=args.guide_scale,
        strict_ik=False,
        seed=rank_seed,
    )


def evaluate_without_residual(args: argparse.Namespace) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    env = _base_env(args, args.seed)
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            final_info: Dict[str, Any] = {}
            for steps in range(env.max_steps):
                observation, reward, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                total_reward += reward
                final_info = info
                if terminated or truncated:
                    break
            results.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "steps": steps + 1,
                    "reward": total_reward,
                    "socket_angle_deg": float(
                        final_info.get("socket_angle_deg", np.nan)
                    ),
                    "socket_mouth_xy": np.asarray(
                        final_info.get("socket_mouth_xy", (np.nan, np.nan)),
                        dtype=np.float64,
                    ).tolist(),
                    "socket_opening_azimuth_xy": np.asarray(
                        final_info.get(
                            "socket_opening_azimuth_xy", (np.nan, np.nan)
                        ),
                        dtype=np.float64,
                    ).tolist(),
                    **{
                        key: (
                            bool(value)
                            if isinstance(value, (np.bool_, bool))
                            else float(value)
                            if isinstance(value, (np.floating, float))
                            else value
                        )
                        for key, value in final_info.items()
                        if key
                        in (
                            "success",
                            "depth_m",
                            "turns",
                            "lateral_error_m",
                            "axis_error_rad",
                            "ik_failures",
                            "ik_anchor_fallbacks",
                            "ik_safety_stops",
                            "failure_reason",
                        )
                    },
                }
            )
    finally:
        env.close()
    return results


def evaluate_hybrid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    residual_path = _project_path(args.residual_checkpoint)
    normalize_path = _project_path(args.vecnormalize)
    if not residual_path.is_file() or not normalize_path.is_file():
        raise FileNotFoundError(
            f"missing residual policy or VecNormalize state: "
            f"{residual_path}, {normalize_path}"
        )
    raw_env = DummyVecEnv([lambda: _base_env(args, args.seed)])
    env = VecNormalize.load(normalize_path, raw_env)
    env.training = False
    env.norm_reward = False
    model = TQC.load(
        residual_path,
        env=env,
        device=resolve_device("auto"),
    )
    if (
        getattr(model, "screw_observation_schema", None)
        != OBSERVATION_SCHEMA
        or getattr(model, "screw_task_geometry_hash", None)
        != TASK_GEOMETRY_HASH
    ):
        raise ValueError(
            "residual checkpoint belongs to an incompatible task geometry"
        )
    results: List[Dict[str, Any]] = []
    try:
        for episode in range(args.episodes):
            env.seed(args.seed + episode)
            observation = env.reset()
            total_reward = 0.0
            final_info: Dict[str, Any] = {}
            for steps in range(3_000):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                observation, reward, done, infos = env.step(action)
                total_reward += float(reward[0])
                final_info = infos[0]
                if bool(done[0]):
                    break
            results.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "steps": steps + 1,
                    "reward": total_reward,
                    "socket_angle_deg": float(
                        final_info.get("socket_angle_deg", np.nan)
                    ),
                    "socket_mouth_xy": np.asarray(
                        final_info.get("socket_mouth_xy", (np.nan, np.nan)),
                        dtype=np.float64,
                    ).tolist(),
                    "socket_opening_azimuth_xy": np.asarray(
                        final_info.get(
                            "socket_opening_azimuth_xy", (np.nan, np.nan)
                        ),
                        dtype=np.float64,
                    ).tolist(),
                    "success": bool(final_info.get("success", False)),
                    "depth_m": float(final_info.get("depth_m", 0.0)),
                    "turns": float(final_info.get("turns", 0.0)),
                    "lateral_error_m": float(
                        final_info.get("lateral_error_m", np.inf)
                    ),
                    "axis_error_rad": float(
                        final_info.get("axis_error_rad", np.inf)
                    ),
                    "ik_anchor_fallbacks": float(
                        final_info.get("ik_anchor_fallbacks", 0.0)
                    ),
                    "ik_safety_stops": float(
                        final_info.get("ik_safety_stops", 0.0)
                    ),
                    "failure_reason": final_info.get("failure_reason", ""),
                }
            )
    finally:
        env.close()
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if args.mode == "hybrid":
        results = evaluate_hybrid(args)
    else:
        results = evaluate_without_residual(args)
    successes = sum(bool(result.get("success", False)) for result in results)
    summary = {
        "mode": args.mode,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "mean_depth_m": float(
            np.mean([result.get("depth_m", 0.0) for result in results])
        ),
        "mean_turns": float(
            np.mean([result.get("turns", 0.0) for result in results])
        ),
        "results": results,
    }
    output = _project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
