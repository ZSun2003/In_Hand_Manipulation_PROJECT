#!/usr/bin/env python3
"""Evaluate Diffusion→safety→analytic IK, with optional legacy TQC."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.config import load_config
from s_hpc.monitoring import RunMonitor, describe_paths
from s_hpc.paths import add_external_path_flag, resolve_package_path
from s_hpc.workflows import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument(
        "--tqc",
        type=Path,
        default=None,
        help="optional legacy TQC residual checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    add_external_path_flag(parser)
    args = parser.parse_args()
    try:
        paths = {
            "config": resolve_package_path(
                args.config,
                allow_external=args.allow_external_paths,
                label="config",
            ),
            "diffusion": resolve_package_path(
                args.diffusion,
                allow_external=args.allow_external_paths,
                label="diffusion checkpoint",
            ),
            "tqc": (
                resolve_package_path(
                    args.tqc,
                    allow_external=args.allow_external_paths,
                    label="TQC checkpoint",
                )
                if args.tqc is not None
                else None
            ),
            "run_dir": resolve_package_path(
                args.run_dir,
                allow_external=args.allow_external_paths,
                label="run directory",
            ),
            "log": (
                resolve_package_path(
                    args.log,
                    allow_external=args.allow_external_paths,
                    label="log",
                )
                if args.log is not None
                else None
            ),
        }
    except ValueError as exc:
        parser.error(str(exc))
    config = load_config(paths["config"])
    log_path = paths["log"] or (
        paths["run_dir"] / "logs" / "evaluation.jsonl"
    )
    monitor = RunMonitor(
        paths["run_dir"],
        pipeline_mode="evaluation",
        config_name=config.name,
        config=config.to_dict(),
        interval_seconds=config.runtime.status_interval_seconds,
    )
    try:
        evaluate(
            config,
            diffusion_checkpoint=paths["diffusion"],
            tqc_checkpoint=paths["tqc"],
            episodes=args.episodes,
            log_path=log_path,
            monitor=monitor,
        )
    except KeyboardInterrupt:
        monitor.finish(
            "interrupted",
            message="Evaluation stopped at the last completed episode.",
        )
        raise
    except BaseException as exc:
        monitor.finish(
            "failed",
            message="Evaluation failed; diagnostic logs were preserved.",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    monitor.heartbeat(
        artifacts=describe_paths(
            {
                name: path
                for name, path in {
                    "diffusion_checkpoint": paths["diffusion"],
                    "tqc_checkpoint": paths["tqc"],
                    "evaluation_log": log_path,
                }.items()
                if path is not None
            }
        ),
        force=True,
    )
    monitor.finish("completed", message="Evaluation completed.")


if __name__ == "__main__":
    main()
