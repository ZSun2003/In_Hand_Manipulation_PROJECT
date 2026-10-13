#!/usr/bin/env python3
"""Train the portable full-state screw-task diffusion policy.

Example::

    ../RL_env/bin/python train_diffusion.py \
        --dataset datasets/demos \
        --output checkpoints/screw_diffusion.pt \
        --epochs 200 --batch-size 256

All policy dimensions come from ``datasets/demos/metadata.json``.  A checkpoint
contains the architecture, model and EMA weights, normalization statistics,
optimizer/scheduler states, dataset contract, and RNG state needed to resume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from demo_dataset import (
    DiffusionEpisodeDataset,
    compute_normalization_stats,
)
from diffusion_policy import (
    CHECKPOINT_FORMAT_VERSION,
    DiffusionPolicy,
    ExponentialMovingAverage,
    _torch_load,
    resolve_device,
    seed_everything,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = MODULE_DIR / "datasets" / "demos"
DEFAULT_CHECKPOINT = MODULE_DIR / "checkpoints" / "screw_diffusion.pt"


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            resolved = value.expanduser().resolve()
            try:
                result[key] = str(resolved.relative_to(MODULE_DIR))
            except ValueError:
                # This is provenance only; loading never follows train_args.
                result[key] = str(resolved)
        else:
            result[key] = value
    return result


def _rng_state(dataloader_generator: torch.Generator) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "dataloader": dataloader_generator.get_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def _restore_rng_state(
    state: Optional[Mapping[str, Any]],
    dataloader_generator: torch.Generator,
) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "dataloader" in state:
        dataloader_generator.set_state(state["dataloader"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(handle)
    try:
        torch.save(dict(payload), temporary_name)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _checkpoint_payload(
    *,
    policy: DiffusionPolicy,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metadata: Mapping[str, Any],
    epoch: int,
    global_step: int,
    best_loss: float,
    args: argparse.Namespace,
    dataloader_generator: torch.Generator,
) -> Dict[str, Any]:
    normalization = policy.normalization_dict()
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": policy.model_config(),
        "model_state": _to_cpu(policy.state_dict()),
        "ema_state": ema.state_dict(cpu=True),
        "normalization": _to_cpu(normalization),
        "optimizer_state": _to_cpu(optimizer.state_dict()),
        "scheduler_state": _to_cpu(scheduler.state_dict()),
        "dataset_metadata": dict(metadata),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_loss": float(best_loss),
        "train_args": _serializable_args(args),
        "rng_state": _to_cpu(_rng_state(dataloader_generator)),
    }


def _validate_resume_metadata(
    checkpoint: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    saved = checkpoint.get("dataset_metadata")
    if not isinstance(saved, Mapping):
        raise ValueError("resume checkpoint is missing dataset_metadata")
    for field in (
        "obs_dim",
        "action_dim",
        "obs_horizon",
        "horizon",
        "observation_schema",
        "task_geometry_hash",
    ):
        if saved.get(field) != current[field]:
            raise ValueError(
                f"resume checkpoint {field}={saved.get(field)!r} does not match "
                f"dataset {field}={current[field]!r}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint to resume; output may be the same path",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="overwrite the resumable output every N completed epochs",
    )
    parser.add_argument(
        "--keep-epoch-checkpoints",
        action="store_true",
        help="also retain output-stem_epoch_N.pt at each save",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optional total optimizer-step cap, useful for smoke tests",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="optional torch CPU thread count",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "epochs",
        "batch_size",
        "hidden_dim",
        "depth",
        "diffusion_steps",
        "gradient_clip",
        "save_every",
        "stride",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay cannot be negative")
    if args.num_workers < 0 or args.cache_size < 0:
        raise ValueError("--num-workers and --cache-size cannot be negative")
    if args.log_every < 0:
        raise ValueError("--log-every cannot be negative")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.cpu_threads is not None and args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")


def train(args: argparse.Namespace) -> Path:
    _validate_args(args)
    seed_everything(args.seed)
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)

    dataset = DiffusionEpisodeDataset(
        args.dataset, stride=args.stride, cache_size=args.cache_size
    )
    metadata = dataset.metadata.to_dict()
    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "dataset": str(dataset.root),
                "windows": len(dataset),
                "episodes": len(dataset.episode_paths),
                "obs_dim": dataset.metadata.obs_dim,
                "action_dim": dataset.metadata.action_dim,
                "obs_horizon": dataset.metadata.obs_horizon,
                "horizon": dataset.metadata.horizon,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    resume_checkpoint: Optional[Dict[str, Any]] = None
    if args.resume is not None:
        resume_checkpoint = _torch_load(args.resume.expanduser().resolve())
        _validate_resume_metadata(resume_checkpoint, metadata)
        model_config = resume_checkpoint.get("model_config")
        if not isinstance(model_config, Mapping):
            raise ValueError("resume checkpoint is missing model_config")
        policy = DiffusionPolicy(**dict(model_config))
        policy.load_state_dict(resume_checkpoint["model_state"], strict=True)
    else:
        policy = DiffusionPolicy(
            obs_dim=dataset.metadata.obs_dim,
            action_dim=dataset.metadata.action_dim,
            obs_horizon=dataset.metadata.obs_horizon,
            horizon=dataset.metadata.horizon,
            num_diffusion_steps=args.diffusion_steps,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
        )
        normalization = compute_normalization_stats(args.dataset)
        policy.set_normalization(normalization)
    policy.to(device)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    ema = ExponentialMovingAverage(policy, decay=args.ema_decay)
    start_epoch = 0
    global_step = 0
    best_loss = math.inf

    dataloader_generator = torch.Generator(device="cpu")
    dataloader_generator.manual_seed(args.seed)
    if resume_checkpoint is not None:
        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        if resume_checkpoint.get("ema_state") is not None:
            ema.load_state_dict(resume_checkpoint["ema_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        global_step = int(resume_checkpoint.get("global_step", 0))
        best_loss = float(resume_checkpoint.get("best_loss", math.inf))
        _restore_rng_state(
            resume_checkpoint.get("rng_state"), dataloader_generator
        )
        print(
            json.dumps(
                {
                    "event": "resume",
                    "checkpoint": str(args.resume),
                    "epoch": start_epoch,
                    "global_step": global_step,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=dataloader_generator,
        worker_init_fn=_worker_seed,
        drop_last=False,
    )
    output = args.output.expanduser().resolve()
    last_completed_epoch = start_epoch
    stop_requested = args.max_steps is not None and global_step >= args.max_steps
    latest_loss = math.inf

    try:
        for epoch in range(start_epoch, args.epochs):
            if stop_requested:
                break
            policy.train()
            epoch_loss_sum = 0.0
            epoch_samples = 0
            epoch_start = time.monotonic()
            for batch in loader:
                observations = batch["obs"].to(device, non_blocking=True)
                actions = batch["action"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = policy.compute_loss(observations, actions)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at optimizer step {global_step}: "
                        f"{float(loss.detach().cpu())}"
                    )
                loss.backward()
                gradient_norm = clip_grad_norm_(
                    policy.parameters(), args.gradient_clip
                )
                if not torch.isfinite(torch.as_tensor(gradient_norm)):
                    raise FloatingPointError(
                        "gradient norm became non-finite during clipping at "
                        f"optimizer step {global_step}"
                    )
                optimizer.step()
                ema.update(policy)

                batch_size = int(observations.shape[0])
                loss_value = float(loss.detach().cpu())
                epoch_loss_sum += loss_value * batch_size
                epoch_samples += batch_size
                global_step += 1
                if args.log_every and global_step % args.log_every == 0:
                    print(
                        json.dumps(
                            {
                                "event": "step",
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "loss": loss_value,
                                "gradient_norm": float(
                                    torch.as_tensor(gradient_norm).detach().cpu()
                                ),
                                "learning_rate": optimizer.param_groups[0]["lr"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop_requested = True
                    break

            if epoch_samples == 0:
                raise RuntimeError("DataLoader produced no samples")
            latest_loss = epoch_loss_sum / epoch_samples
            best_loss = min(best_loss, latest_loss)
            scheduler.step()
            last_completed_epoch = epoch + 1
            print(
                json.dumps(
                    {
                        "event": "epoch",
                        "epoch": last_completed_epoch,
                        "global_step": global_step,
                        "loss": latest_loss,
                        "best_loss": best_loss,
                        "seconds": time.monotonic() - epoch_start,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

            if (
                last_completed_epoch % args.save_every == 0
                or stop_requested
                or last_completed_epoch == args.epochs
            ):
                payload = _checkpoint_payload(
                    policy=policy,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata=metadata,
                    epoch=last_completed_epoch,
                    global_step=global_step,
                    best_loss=best_loss,
                    args=args,
                    dataloader_generator=dataloader_generator,
                )
                _atomic_torch_save(payload, output)
                if args.keep_epoch_checkpoints:
                    epoch_path = output.with_name(
                        f"{output.stem}_epoch_{last_completed_epoch}{output.suffix}"
                    )
                    _atomic_torch_save(payload, epoch_path)
                print(
                    json.dumps(
                        {
                            "event": "checkpoint",
                            "path": str(output),
                            "epoch": last_completed_epoch,
                            "global_step": global_step,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    except KeyboardInterrupt:
        print("training interrupted; saving resumable checkpoint", flush=True)
        stop_requested = True

    # Covers immediate --max-steps on resume, KeyboardInterrupt before an epoch
    # save, and unusual save intervals.
    payload = _checkpoint_payload(
        policy=policy,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        epoch=last_completed_epoch,
        global_step=global_step,
        best_loss=best_loss,
        args=args,
        dataloader_generator=dataloader_generator,
    )
    _atomic_torch_save(payload, output)
    print(
        json.dumps(
            {
                "event": "complete",
                "path": str(output),
                "epoch": last_completed_epoch,
                "global_step": global_step,
                "latest_loss": latest_loss,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
