#!/usr/bin/env python3
"""Strictly pair zero-residual Diffusion and A* on one initial manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.config import load_config
from s_hpc.paired_planner_evaluation import evaluate_paired_planners


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--initial-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="defaults to every row in the supplied manifest",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optional limit no larger than config.physics.max_episode_steps",
    )
    parser.add_argument(
        "--order",
        choices=("diffusion-first", "astar-first"),
        default="diffusion-first",
        help="persisted because sequential same-host timing can have order effects",
    )
    parser.add_argument(
        "--astar-collision-model",
        choices=("conservative_obb", "exact_mujoco"),
        default="exact_mujoco",
        help=(
            "A* contract. exact_mujoco is the formal one-shot global full-pose "
            "ablation; conservative_obb is retained only for legacy replay."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    diffusion_path = args.diffusion.expanduser().resolve()
    manifest_path = args.initial_manifest.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    for label, path in (
        ("config", config_path),
        ("Diffusion checkpoint", diffusion_path),
        ("initial manifest", manifest_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    order = (
        ("diffusion", "astar_spline")
        if args.order == "diffusion-first"
        else ("astar_spline", "diffusion")
    )
    summary = evaluate_paired_planners(
        load_config(config_path),
        config_path=config_path,
        diffusion_checkpoint=diffusion_path,
        initial_conditions_manifest=manifest_path,
        output_dir=run_dir,
        episodes=args.episodes,
        max_steps=args.max_steps,
        stack_order=order,
        astar_collision_model=args.astar_collision_model,
    )
    print(
        json.dumps(
            {
                "event": "paired_planner_evaluation",
                "episodes": summary["episodes"],
                "diffusion_success_rate": summary["diffusion"]["summary"][
                    "success_rate"
                ],
                "astar_success_rate": summary["astar_spline"]["summary"][
                    "success_rate"
                ],
                "astar_collision_model": summary["astar_collision_model"],
                "summary": summary["outputs"]["summary"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
