#!/usr/bin/env python3
"""Validate the isolated 5,000-world TQC continuation artifacts."""

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


def _canonical_config(config) -> dict:
    return canonicalize_config_payload(config.to_dict())


def _runtime_width_independent(payload: dict) -> dict:
    """Remove only fields that cannot change a frozen Diffusion policy."""

    result = canonicalize_config_payload(payload)
    result.pop("name", None)
    physics = result.get("physics")
    if isinstance(physics, dict):
        physics.pop("worlds", None)
    return result


def _config_sha(config) -> str:
    encoded = json.dumps(
        _canonical_config(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_config(config) -> dict:
    if config.physics.worlds != 5_000:
        raise ValueError("continuation config must use exactly 5,000 worlds")
    if config.physics.max_episode_steps != 4_000:
        raise ValueError("continuation config must retain 4,000-step episodes")
    if config.curriculum.final_spawn_radius_m != 0.1:
        raise ValueError("continuation config must retain the 100 mm workspace")
    if config.curriculum.restrict_screw_spawn_to_opening_half_disk:
        raise ValueError("continuation config must retain the complete workspace disc")
    if config.tqc.total_transitions != 5_000_000:
        raise ValueError("TQC budget must remain 5,000,000 transitions")
    if config.tqc.replay_capacity != 1_000_000:
        raise ValueError("TQC replay capacity must remain 1,000,000")
    if config.tqc.replay_insert_per_vector_step != 256:
        raise ValueError("TQC replay admission must remain 256 per vector step")
    return {
        "stage": "config",
        "status": "complete",
        "config_sha256": _config_sha(config),
        "worlds": config.physics.worlds,
        "spawn_radius_mm": config.curriculum.final_spawn_radius_m * 1_000.0,
        "spawn_region": "full_workspace_disc",
        "tqc_transitions": config.tqc.total_transitions,
    }


def validate_diffusion(path: Path, config) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _load_torch(path)
    validate_checkpoint(payload, kind="diffusion")
    stored = payload.get("config")
    if not isinstance(stored, dict):
        raise ValueError("Diffusion checkpoint has no valid configuration")
    if _runtime_width_independent(stored) != _runtime_width_independent(
        config.to_dict()
    ):
        raise ValueError(
            "Diffusion checkpoint differs beyond the permitted name/worlds fields"
        )
    if int(payload.get("epoch", -1)) != config.diffusion.epochs:
        raise ValueError("Diffusion checkpoint has not completed all epochs")
    if int(payload.get("batch_in_epoch", -1)) != 0:
        raise ValueError("Diffusion checkpoint has an incomplete batch cursor")
    contract = payload.get("dataset_contract")
    provenance = contract.get("provenance") if isinstance(contract, dict) else None
    if not isinstance(provenance, dict) or provenance.get("dataset_role") != (
        "successful_policy_rollout_self_distillation"
    ):
        raise ValueError("Diffusion checkpoint is not the certified self-distillation model")
    metadata_sha = payload.get("dataset_metadata_sha256")
    if not isinstance(metadata_sha, str) or len(metadata_sha) != 64:
        raise ValueError("Diffusion checkpoint has no valid dataset SHA-256")
    return {
        "stage": "diffusion",
        "status": "complete",
        "checkpoint": str(path),
        "sha256": sha256_file(path),
        "epochs": int(payload["epoch"]),
        "global_step": int(payload.get("global_step", 0)),
        "source_worlds": int(stored["physics"]["worlds"]),
        "runtime_worlds": config.physics.worlds,
        "dataset_metadata_sha256": metadata_sha,
    }


def validate_tqc(path: Path, diffusion: Path, config) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    diffusion_result = validate_diffusion(diffusion, config)
    payload = _load_torch(path)
    validate_checkpoint(payload, kind="tqc")
    if payload.get("policy_only") is not True:
        raise ValueError("TQC artifact is not the policy-only checkpoint")
    stored = payload.get("config")
    if not isinstance(stored, dict) or canonicalize_config_payload(
        stored
    ) != _canonical_config(config):
        raise ValueError("TQC checkpoint configuration differs")
    if int(payload.get("transitions", -1)) < config.tqc.total_transitions:
        raise ValueError("TQC checkpoint has not reached its transition budget")
    if payload.get("diffusion_sha256") != diffusion_result["sha256"]:
        raise ValueError("TQC policy is pinned to another Diffusion checkpoint")
    actor = payload.get("actor")
    if not isinstance(actor, dict) or not actor:
        raise ValueError("TQC policy checkpoint has no actor weights")
    return {
        "stage": "tqc",
        "status": "complete",
        "checkpoint": str(path),
        "sha256": sha256_file(path),
        "transitions": int(payload["transitions"]),
        "updates": int(payload.get("updates", 0)),
        "diffusion_sha256": diffusion_result["sha256"],
        "worlds": config.physics.worlds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diffusion", type=Path)
    parser.add_argument("--tqc", type=Path)
    parser.add_argument(
        "--stage", choices=("config", "diffusion", "tqc"), required=True
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config.expanduser().resolve())
        result = validate_config(config)
        if args.stage in {"diffusion", "tqc"} and args.diffusion is None:
            raise ValueError("--diffusion is required for this stage")
        if args.stage == "diffusion":
            result = validate_diffusion(
                args.diffusion.expanduser().resolve(), config
            )
        elif args.stage == "tqc":
            if args.tqc is None:
                raise ValueError("--tqc is required for the tqc stage")
            result = validate_tqc(
                args.tqc.expanduser().resolve(),
                args.diffusion.expanduser().resolve(),
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
