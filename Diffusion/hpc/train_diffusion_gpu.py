#!/usr/bin/env python3
"""Train the CUDA Diffusion baseline from GPU demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.config import load_config
from s_hpc.monitoring import RunMonitor, describe_paths
from s_hpc.paths import add_external_path_flag, resolve_package_path
from s_hpc.workflows import train_diffusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    add_external_path_flag(parser)
    args = parser.parse_args()
    try:
        paths = {
            "config": resolve_package_path(
                args.config,
                allow_external=args.allow_external_paths,
                label="config",
            ),
            "dataset": resolve_package_path(
                args.dataset,
                allow_external=args.allow_external_paths,
                label="dataset",
            ),
            "output": resolve_package_path(
                args.output,
                allow_external=args.allow_external_paths,
                label="diffusion checkpoint",
            ),
            "log": resolve_package_path(
                args.log,
                allow_external=args.allow_external_paths,
                label="log",
            ),
            "run_dir": resolve_package_path(
                args.run_dir,
                allow_external=args.allow_external_paths,
                label="run directory",
            ),
        }
    except ValueError as exc:
        parser.error(str(exc))
    config = load_config(paths["config"])
    monitor = RunMonitor(
        paths["run_dir"],
        pipeline_mode="diffusion",
        config_name=config.name,
        config=config.to_dict(),
        interval_seconds=config.runtime.status_interval_seconds,
    )
    try:
        train_diffusion(
            config,
            dataset=paths["dataset"],
            output=paths["output"],
            log_path=paths["log"],
            resume=not args.no_resume,
            monitor=monitor,
        )
    except KeyboardInterrupt:
        monitor.finish(
            "interrupted",
            message="Diffusion training stopped at a resumable cursor.",
        )
        raise
    except BaseException as exc:
        monitor.finish(
            "failed",
            message="Diffusion training failed; last-good files were preserved.",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    monitor.heartbeat(
        artifacts=describe_paths(
            {
                "diffusion_checkpoint": paths["output"],
                "diffusion_log": paths["log"],
            }
        ),
        force=True,
    )
    monitor.finish("completed", message="Diffusion training completed.")


if __name__ == "__main__":
    main()
