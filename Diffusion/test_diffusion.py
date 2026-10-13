#!/usr/bin/env python3
"""Validate, evaluate, or smoke-test a trained diffusion checkpoint.

Fast self-contained CPU verification (no MuJoCo required)::

    ../RL_env/bin/python test_diffusion.py --smoke-test

Dataset evaluation::

    ../RL_env/bin/python test_diffusion.py \
        --checkpoint checkpoints/screw_diffusion.pt \
        --dataset datasets/demos --device auto
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from demo_dataset import (
    ACTION_KEY,
    OBSERVATION_KEY,
    DiffusionEpisodeDataset,
    _create_smoke_dataset,
    compute_normalization_stats,
)
from diffusion_policy import (
    CHECKPOINT_FORMAT_VERSION,
    DiffusionPolicy,
    ExponentialMovingAverage,
    load_policy_checkpoint,
    resolve_device,
    seed_everything,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = MODULE_DIR / "datasets" / "demos"


def _compatible_metadata(checkpoint: Dict[str, Any], dataset: DiffusionEpisodeDataset) -> None:
    saved = checkpoint.get("dataset_metadata")
    if not isinstance(saved, dict):
        raise ValueError("checkpoint is missing dataset_metadata")
    current = dataset.metadata.to_dict()
    for name in (
        "obs_dim",
        "action_dim",
        "obs_horizon",
        "horizon",
        "observation_schema",
        "task_geometry_hash",
    ):
        if saved.get(name) != current[name]:
            raise ValueError(
                f"checkpoint {name}={saved.get(name)!r} does not match "
                f"dataset {name}={current[name]!r}"
            )


@torch.no_grad()
def evaluate(
    *,
    checkpoint_path: Path,
    dataset_path: Path,
    device_name: str,
    use_ema: bool,
    batches: int,
    sample_batches: int,
    batch_size: int,
    inference_steps: int,
    seed: int,
) -> Dict[str, Any]:
    seed_everything(seed)
    device = resolve_device(device_name)
    policy, checkpoint = load_policy_checkpoint(
        checkpoint_path, device=device, use_ema=use_ema
    )
    dataset = DiffusionEpisodeDataset(dataset_path)
    _compatible_metadata(checkpoint, dataset)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
    )

    denoising_loss_sum = 0.0
    denoising_count = 0
    action_mse_sum = 0.0
    action_count = 0
    sample_seconds = 0.0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for batch_index, batch in enumerate(loader):
        if batch_index >= batches:
            break
        observations = batch["obs"].to(device)
        actions = batch["action"].to(device)
        # compute_loss creates no parameter gradients in this no_grad context.
        loss = policy.compute_loss(
            observations, actions, generator=generator
        )
        count = int(observations.shape[0])
        denoising_loss_sum += float(loss.cpu()) * count
        denoising_count += count

        if batch_index < sample_batches:
            start = time.perf_counter()
            predictions = policy.sample(
                observations,
                num_inference_steps=inference_steps,
                seed=seed + batch_index,
            )
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            sample_seconds += time.perf_counter() - start
            squared_error = torch.square(predictions - actions)
            action_mse_sum += float(squared_error.sum().cpu())
            action_count += squared_error.numel()

    if denoising_count == 0:
        raise ValueError("no evaluation batches were processed")
    return {
        "checkpoint": str(checkpoint_path.expanduser().resolve()),
        "dataset": str(dataset.root),
        "device": str(device),
        "weights": "ema" if use_ema and checkpoint.get("ema_state") else "model",
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "denoising_loss": denoising_loss_sum / denoising_count,
        "sample_action_mse": (
            action_mse_sum / action_count if action_count else None
        ),
        "sample_seconds": sample_seconds,
        "evaluated_windows": denoising_count,
        "sampled_action_values": action_count,
        "inference_steps": inference_steps,
    }


def _checkpoint_for_smoke(
    policy: DiffusionPolicy,
    ema: ExponentialMovingAverage,
    metadata: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    steps: int,
) -> Dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": policy.model_config(),
        "model_state": {
            key: value.detach().cpu() for key, value in policy.state_dict().items()
        },
        "ema_state": ema.state_dict(cpu=True),
        "normalization": policy.normalization_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": {},
        "dataset_metadata": metadata,
        "epoch": 1,
        "global_step": steps,
        "best_loss": 0.0,
        "train_args": {"smoke_test": True},
        "rng_state": {},
    }


def smoke_test(seed: int = 7) -> Dict[str, Any]:
    """Exercise NPZ I/O, learning, DDIM, and checkpoint reload on CPU."""

    seed_everything(seed)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory(prefix="screw_diffusion_smoke_") as tmp:
            root = Path(tmp)
            summary = _create_smoke_dataset(root / "demos", seed)
            dataset = DiffusionEpisodeDataset(root / "demos", cache_size=2)
            stats = compute_normalization_stats(root / "demos")
            policy = DiffusionPolicy(
                obs_dim=dataset.metadata.obs_dim,
                action_dim=dataset.metadata.action_dim,
                obs_horizon=dataset.metadata.obs_horizon,
                horizon=dataset.metadata.horizon,
                num_diffusion_steps=8,
                hidden_dim=32,
                depth=2,
            )
            policy.set_normalization(stats)
            policy.to("cpu")
            optimizer = torch.optim.AdamW(policy.parameters(), lr=2.0e-3)
            ema = ExponentialMovingAverage(policy, decay=0.9)
            loader = DataLoader(dataset, batch_size=8, shuffle=False)

            losses = []
            for step, batch in enumerate(loader):
                if step >= 3:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss = policy.compute_loss(batch["obs"], batch["action"])
                loss.backward()
                optimizer.step()
                ema.update(policy)
                losses.append(float(loss.detach()))
            if not losses or not all(np.isfinite(losses)):
                raise AssertionError("smoke training produced invalid loss")

            first = dataset[0]
            policy.eval()
            sample_a = policy.sample(
                first["obs"], num_inference_steps=4, seed=seed
            )
            sample_b = policy.sample(
                first["obs"], num_inference_steps=4, seed=seed
            )
            expected_shape = (
                dataset.metadata.horizon,
                dataset.metadata.action_dim,
            )
            if tuple(sample_a.shape) != expected_shape:
                raise AssertionError(
                    f"sample shape {tuple(sample_a.shape)} != {expected_shape}"
                )
            if not torch.isfinite(sample_a).all():
                raise AssertionError("sample contains NaN or infinity")
            if not torch.equal(sample_a, sample_b):
                raise AssertionError("fixed-seed DDIM sampling is not deterministic")

            checkpoint_path = root / "smoke.pt"
            torch.save(
                _checkpoint_for_smoke(
                    policy,
                    ema,
                    dataset.metadata.to_dict(),
                    optimizer,
                    len(losses),
                ),
                checkpoint_path,
            )
            loaded, checkpoint = load_policy_checkpoint(
                checkpoint_path, device="cpu", use_ema=False
            )
            reloaded_sample = loaded.sample(
                first["obs"], num_inference_steps=4, seed=seed
            )
            if not torch.equal(sample_a, reloaded_sample):
                raise AssertionError("checkpoint round-trip changed model output")

            return {
                "status": "passed",
                "device": "cpu",
                "dataset_episodes": summary["episodes"],
                "dataset_windows": len(dataset),
                "training_steps": len(losses),
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "sample_shape": list(sample_a.shape),
                "sample_finite": True,
                "fixed_seed_deterministic": True,
                "checkpoint_round_trip": True,
                "checkpoint_format": checkpoint["format_version"],
            }
    finally:
        torch.set_num_threads(previous_threads)


def _load_observation_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if OBSERVATION_KEY in archive:
                array = archive[OBSERVATION_KEY]
            elif "obs" in archive:
                array = archive["obs"]
            else:
                raise ValueError(
                    f"{path} has neither {OBSERVATION_KEY!r} nor 'obs'"
                )
    else:
        raise ValueError("--observation-file must be .npy or .npz")
    return np.asarray(array, dtype=np.float32)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--sample-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--observation-file",
        type=Path,
        default=None,
        help="instead of dataset metrics, sample one chunk from .npy/.npz history",
    )
    parser.add_argument(
        "--output-actions",
        type=Path,
        default=None,
        help="optional .npy destination for --observation-file inference",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke_test:
        result = smoke_test(args.seed)
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless --smoke-test is used")
        if args.inference_steps <= 0:
            raise SystemExit("--inference-steps must be positive")
        if args.observation_file is not None:
            policy, checkpoint = load_policy_checkpoint(
                args.checkpoint,
                device=args.device,
                use_ema=not args.no_ema,
            )
            observations = _load_observation_file(args.observation_file)
            actions = policy.sample(
                observations,
                num_inference_steps=args.inference_steps,
                seed=args.seed,
            ).detach().cpu().numpy()
            if args.output_actions is not None:
                destination = args.output_actions.expanduser().resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.save(destination, actions)
            result = {
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "weights": (
                    "ema"
                    if not args.no_ema and checkpoint.get("ema_state")
                    else "model"
                ),
                "input_shape": list(observations.shape),
                "action_shape": list(actions.shape),
                "actions": (
                    None if args.output_actions is not None else actions.tolist()
                ),
                "output_actions": (
                    str(args.output_actions.expanduser().resolve())
                    if args.output_actions is not None
                    else None
                ),
            }
        else:
            if args.batches <= 0 or args.batch_size <= 0:
                raise SystemExit("--batches and --batch-size must be positive")
            if args.sample_batches < 0:
                raise SystemExit("--sample-batches cannot be negative")
            result = evaluate(
                checkpoint_path=args.checkpoint,
                dataset_path=args.dataset,
                device_name=args.device,
                use_ema=not args.no_ema,
                batches=args.batches,
                sample_batches=min(args.sample_batches, args.batches),
                batch_size=args.batch_size,
                inference_steps=args.inference_steps,
                seed=args.seed,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
