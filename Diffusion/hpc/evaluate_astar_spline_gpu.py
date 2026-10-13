#!/usr/bin/env python3
"""Evaluate one-shot global A*--safety--analytic-IK with zero residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.astar_evaluation import evaluate_astar_spline
from s_hpc.config import load_config
from s_hpc.monitoring import RunMonitor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL output; defaults to RUN_DIR/logs/astar_evaluation.jsonl",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        default=None,
        help=(
            "declared live-world capacity; must be at least --episodes. "
            "Exactly one world is instantiated per requested episode"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optional evaluation limit no larger than the configured limit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config does not exist: {config_path}")
    run_dir = args.run_dir.expanduser().resolve()
    log_path = (
        args.log.expanduser().resolve()
        if args.log is not None
        else run_dir / "logs" / "astar_evaluation.jsonl"
    )
    if log_path.exists() and log_path.stat().st_size:
        raise FileExistsError(
            "refusing to append a new evaluation to a non-empty log: "
            f"{log_path}"
        )
    config = load_config(config_path)
    monitor = RunMonitor(
        run_dir,
        pipeline_mode="evaluation",
        config_name=config.name,
        config=config.to_dict(),
        interval_seconds=config.runtime.status_interval_seconds,
    )
    try:
        summary = evaluate_astar_spline(
            config,
            episodes=args.episodes,
            worlds=args.worlds,
            max_steps=args.max_steps,
            run_dir=run_dir,
            log_path=log_path,
            config_path=config_path,
            monitor=monitor,
            cli_arguments=raw_argv,
        )
    except KeyboardInterrupt:
        monitor.finish(
            "interrupted",
            message="Global A*--IK evaluation was interrupted.",
        )
        raise
    except BaseException as exc:
        monitor.finish(
            "failed",
            message="Global A*--IK evaluation failed; outputs were retained.",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    monitor.finish(
        "completed",
        message="Global A*--IK first-episode evaluation completed.",
    )
    print(
        json.dumps(
            {
                "event": "global_astar_ik_evaluation",
                "episodes": summary["episodes"],
                "successes": summary["successes"],
                "failures": summary["failures"],
                "truncations": summary["truncations"],
                "success_rate": summary["success_rate"],
                "summary": summary["outputs"]["summary"]["path"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
