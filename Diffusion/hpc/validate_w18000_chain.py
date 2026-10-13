#!/usr/bin/env python3
"""Validate completed artifacts for the isolated 18,000-world chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.checkpointing import sha256_file
from s_hpc.config import canonicalize_config_payload, load_config
from s_hpc.contracts import validate_checkpoint


def _load_torch(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _config_sha(config) -> str:
    payload = json.dumps(
        canonicalize_config_payload(config.to_dict()),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_formal_config(config) -> None:
    if config.physics.worlds != 18_000:
        raise ValueError("formal config must use exactly 18,000 worlds")
    if config.physics.max_episode_steps != 4_000:
        raise ValueError("formal config must use a 4,000-step episode ceiling")
    if config.curriculum.final_spawn_radius_m != 0.1:
        raise ValueError("formal config must use the 100 mm workspace")
    if config.curriculum.restrict_screw_spawn_to_opening_half_disk:
        raise ValueError("formal config must enable the complete workspace disc")
    if config.tqc.total_transitions != 5_000_000:
        raise ValueError("formal TQC budget must remain 5,000,000 transitions")
    if config.tqc.replay_insert_per_vector_step != 256:
        raise ValueError("formal TQC replay admission must remain 256 per tick")


def _same_config(payload: dict, config) -> bool:
    stored = payload.get("config")
    return isinstance(stored, dict) and canonicalize_config_payload(
        stored
    ) == canonicalize_config_payload(config.to_dict())


def validate_diffusion(path: Path, dataset: Path, config) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _load_torch(path)
    validate_checkpoint(payload, kind="diffusion")
    if not _same_config(payload, config):
        raise ValueError("Diffusion checkpoint configuration differs")
    if int(payload.get("epoch", -1)) != config.diffusion.epochs:
        raise ValueError("Diffusion checkpoint has not completed all epochs")
    if int(payload.get("batch_in_epoch", -1)) != 0:
        raise ValueError("Diffusion checkpoint has an incomplete batch cursor")
    dataset_metadata = dataset / "metadata.json"
    if payload.get("dataset_metadata_sha256") != sha256_file(dataset_metadata):
        raise ValueError("Diffusion checkpoint is pinned to another dataset")
    contract = payload.get("dataset_contract")
    if not isinstance(contract, dict):
        raise ValueError("Diffusion checkpoint has no dataset contract")
    provenance = contract.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("dataset_role") != (
        "successful_policy_rollout_self_distillation"
    ):
        raise ValueError("Diffusion checkpoint was not trained on success rollouts")
    return {
        "stage": "diffusion",
        "status": "complete",
        "checkpoint": str(path),
        "sha256": sha256_file(path),
        "epochs": int(payload["epoch"]),
        "global_step": int(payload.get("global_step", 0)),
        "dataset_metadata_sha256": payload["dataset_metadata_sha256"],
    }


def validate_tqc(policy_path: Path, diffusion_path: Path, config) -> dict:
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    if not diffusion_path.is_file():
        raise FileNotFoundError(diffusion_path)
    payload = _load_torch(policy_path)
    validate_checkpoint(payload, kind="tqc")
    if payload.get("policy_only") is not True:
        raise ValueError("TQC artifact is not the policy-only checkpoint")
    if not _same_config(payload, config):
        raise ValueError("TQC checkpoint configuration differs")
    if int(payload.get("transitions", -1)) < config.tqc.total_transitions:
        raise ValueError("TQC checkpoint has not reached the transition budget")
    diffusion_sha = sha256_file(diffusion_path)
    if payload.get("diffusion_sha256") != diffusion_sha:
        raise ValueError("TQC checkpoint is pinned to another Diffusion model")
    actor = payload.get("actor")
    if not isinstance(actor, dict) or not actor:
        raise ValueError("TQC policy checkpoint has no actor weights")
    return {
        "stage": "tqc",
        "status": "complete",
        "checkpoint": str(policy_path),
        "sha256": sha256_file(policy_path),
        "transitions": int(payload["transitions"]),
        "updates": int(payload.get("updates", 0)),
        "diffusion_sha256": diffusion_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("config", "diffusion", "tqc"), required=True
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    try:
        config = load_config(config_path)
        _validate_formal_config(config)
        result = {
            "stage": "config",
            "status": "complete",
            "config": str(config_path),
            "config_sha256": _config_sha(config),
            "worlds": config.physics.worlds,
            "spawn_radius_mm": config.curriculum.final_spawn_radius_m * 1000.0,
            "spawn_region": "full_workspace_disc",
        }
        diffusion = run_dir / "checkpoints" / "diffusion.pt"
        if args.stage == "diffusion":
            result = validate_diffusion(diffusion, dataset, config)
        elif args.stage == "tqc":
            result = validate_tqc(
                run_dir / "checkpoints" / "residual_tqc_policy.pt",
                diffusion,
                config,
            )
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "status": "incomplete_or_invalid",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
