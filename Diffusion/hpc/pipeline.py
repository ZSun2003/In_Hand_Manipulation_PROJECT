#!/usr/bin/env python3
"""Run the MJWarp training and deterministic evaluation pipelines."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.config import load_config
from s_hpc.monitoring import RunMonitor, describe_paths
from s_hpc.paths import add_external_path_flag, resolve_package_path
from s_hpc.workflows import (
    evaluate,
    generate_demos,
    probe_residual_resume,
    train_diffusion,
    train_residual,
)


def _release_cuda_stage_memory(config) -> None:
    """Return cached stage allocations before the next CUDA subsystem starts.

    MJWarp and PyTorch use different CUDA allocation paths.  In particular,
    retaining PyTorch's multi-gigabyte Diffusion dataset cache could prevent
    the following MJWarp rollout or evaluation environment from allocating
    memory even though the dataset tensors were already out of scope.
    """

    gc.collect()
    if not str(config.runtime.device).startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        return
    device = torch.device(config.runtime.device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def _local_path(
    value: str | Path,
    *,
    allow_external_paths: bool = False,
    label: str = "path",
) -> Path:
    return resolve_package_path(
        value,
        allow_external=allow_external_paths,
        label=label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "smoke",
            "full",
            "resume-probe",
            "eval",
            "diffusion-smoke",
            "diffusion-full",
            "diffusion-eval",
        ),
        required=True,
    )
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "checkpoint override: TQC for --stage eval, Diffusion for "
            "--stage diffusion-eval"
        ),
    )
    parser.add_argument("--episodes", type=int, default=100)
    add_external_path_flag(parser)
    args = parser.parse_args()
    try:
        config_path = _local_path(
            args.config,
            allow_external_paths=args.allow_external_paths,
            label="config",
        )
        run_dir = _local_path(
            args.run_dir,
            allow_external_paths=args.allow_external_paths,
            label="run directory",
        )
        checkpoint_override = (
            None
            if args.checkpoint is None
            else _local_path(
                args.checkpoint,
                allow_external_paths=args.allow_external_paths,
                label="checkpoint",
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    config = load_config(config_path)
    if (
        args.resume == "never"
        and run_dir.is_dir()
        and any(path.name != "status" for path in run_dir.iterdir())
    ):
        raise FileExistsError(
            f"--resume never requires an empty run directory: {run_dir}"
        )
    dataset = run_dir / "datasets" / "demos_gpu"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    diffusion = checkpoint_dir / "diffusion.pt"
    tqc = checkpoint_dir / "residual_tqc.pt"
    tqc_policy = checkpoint_dir / "residual_tqc_policy.pt"
    resume = args.resume == "auto"
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor = RunMonitor(
        run_dir,
        pipeline_mode=args.stage,
        config_name=config.name,
        config=config.to_dict(),
        interval_seconds=config.runtime.status_interval_seconds,
    )
    print(
        json.dumps(
            {
                "event": "pipeline_start",
                "stage": args.stage,
                "config": config.name,
                "run_dir": str(run_dir),
                "resume": resume,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        if args.stage in (
            "smoke",
            "full",
            "diffusion-smoke",
            "diffusion-full",
        ):
            generate_demos(
                config, output=dataset, resume=resume, monitor=monitor
            )
            _release_cuda_stage_memory(config)
            train_diffusion(
                config,
                dataset=dataset,
                output=diffusion,
                log_path=log_dir / "diffusion.jsonl",
                resume=resume,
                monitor=monitor,
            )
            _release_cuda_stage_memory(config)
            evaluation_tqc = None
            if args.stage in ("smoke", "full"):
                evaluation_tqc = train_residual(
                    config,
                    diffusion_checkpoint=diffusion,
                    output=tqc,
                    log_path=log_dir / "tqc.jsonl",
                    resume=resume,
                    monitor=monitor,
                )
                _release_cuda_stage_memory(config)
            evaluate(
                config,
                diffusion_checkpoint=diffusion,
                tqc_checkpoint=evaluation_tqc,
                episodes=min(args.episodes, config.physics.worlds),
                log_path=log_dir / "evaluation.jsonl",
                monitor=monitor,
            )
            _release_cuda_stage_memory(config)
        elif args.stage == "resume-probe":
            probe_residual_resume(
                config,
                diffusion_checkpoint=diffusion,
                tqc_checkpoint=tqc,
                monitor=monitor,
            )
        elif args.stage == "diffusion-eval":
            selected_diffusion = (
                diffusion
                if checkpoint_override is None
                else checkpoint_override
            )
            if not selected_diffusion.is_file():
                raise FileNotFoundError(
                    "Diffusion checkpoint not found: "
                    f"{selected_diffusion}"
                )
            evaluate(
                config,
                diffusion_checkpoint=selected_diffusion,
                tqc_checkpoint=None,
                episodes=args.episodes,
                log_path=log_dir / "evaluation.jsonl",
                monitor=monitor,
            )
        else:
            if checkpoint_override is not None:
                selected_tqc = checkpoint_override
            elif tqc_policy.is_file():
                selected_tqc = tqc_policy
            else:
                selected_tqc = tqc
            selected_diffusion = (
                diffusion
                if checkpoint_override is None
                else selected_tqc.parent / "diffusion.pt"
            )
            if not selected_diffusion.is_file():
                raise FileNotFoundError(
                    "matching diffusion checkpoint not found: "
                    f"{selected_diffusion}"
                )
            evaluate(
                config,
                diffusion_checkpoint=selected_diffusion,
                tqc_checkpoint=selected_tqc,
                episodes=args.episodes,
                log_path=log_dir / "evaluation.jsonl",
                monitor=monitor,
            )
    except KeyboardInterrupt:
        monitor.finish(
            "interrupted",
            message=(
                "Stopped at a consistent cursor. Resubmit the same job to resume."
            ),
        )
        raise
    except BaseException as exc:
        monitor.finish(
            "failed",
            message="Pipeline failed; the last-good checkpoint was preserved.",
            error={
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise
    monitor.heartbeat(
        artifacts=describe_paths(
            {
                name: path
                for name, path in {
                    "diffusion_checkpoint": diffusion,
                    "tqc_checkpoint": (
                        None
                        if args.stage.startswith("diffusion-")
                        else tqc
                    ),
                    "tqc_policy_checkpoint": (
                        None
                        if args.stage.startswith("diffusion-")
                        else tqc_policy
                    ),
                }.items()
                if path is not None
            }
        ),
        force=True,
    )
    monitor.finish("completed", message="Requested pipeline completed.")
    print(json.dumps({"event": "pipeline_complete"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
