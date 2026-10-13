#!/usr/bin/env python3
"""Strict paired IK-only versus IK+residual endpoint evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

try:
    from . import config
    from .runtime import IK_RL_DIR, prepare_runtime
except ImportError:
    import config  # type: ignore
    from runtime import IK_RL_DIR, prepare_runtime  # type: ignore


current = prepare_runtime()

import numpy as np
from sb3_contrib import TQC

try:
    from .controller_bridge import OBSERVATION_DIM
    from .env import G6IKResidualEnv
except ImportError:
    from controller_bridge import OBSERVATION_DIM  # type: ignore
    from env import G6IKResidualEnv  # type: ignore


SCHEMA = "ik-residual-paired-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired same-object/task/seed G6 IK-only and learned "
            "IK+residual runs."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=IK_RL_DIR / "models" / "final_model.zip",
    )
    parser.add_argument(
        "--vec-normalize",
        type=Path,
        default=IK_RL_DIR / "models" / "vec_normalize.pkl",
    )
    parser.add_argument(
        "--episodes-per-pair",
        type=int,
        default=20,
        help="seeds for each object/task combination",
    )
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=IK_RL_DIR / "eval_output" / "paired_eval.json",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_normalizer(path: Path) -> Any:
    with path.open("rb") as stream:
        normalizer = pickle.load(stream)
    normalizer.training = False
    normalizer.norm_reward = False
    mean = np.asarray(normalizer.obs_rms.mean)
    expected_shape = (OBSERVATION_DIM,)
    if mean.shape != expected_shape:
        raise ValueError(
            "vec_normalize observation shape mismatch: "
            f"{mean.shape}, expected {expected_shape}"
        )
    return normalizer


def _run_episode(
    *,
    object_kind: str,
    task: str,
    episode_seed: int,
    controller: str,
    model: TQC,
    normalizer: Any,
    record_trajectory: bool,
) -> dict[str, Any]:
    env = G6IKResidualEnv()
    trajectory: list[dict[str, Any]] = []
    total_reward = 0.0
    try:
        observation, initial_info = env.reset(
            seed=episode_seed,
            options={
                "object": object_kind,
                "task": task,
                "episode_seed": episode_seed,
            },
        )
        initial_state_hash = str(initial_info["initial_state_hash"])
        final_info = initial_info
        while True:
            if controller == "ik_only":
                action = np.zeros(config.ACTION_DIM, dtype=np.float32)
            elif controller == "ik_residual":
                policy_observation = normalizer.normalize_obs(
                    observation.copy()
                )
                action, _state = model.predict(
                    policy_observation,
                    deterministic=True,
                )
                action = np.asarray(action, dtype=np.float32).reshape(
                    config.ACTION_DIM
                )
            else:
                raise ValueError(f"unknown controller: {controller}")

            (
                observation,
                reward,
                terminated,
                truncated,
                final_info,
            ) = env.step(action)
            total_reward += float(reward)
            sample = final_info.get("trajectory_sample")
            if record_trajectory and sample is not None:
                trajectory.append(_jsonable(sample))
            if terminated or truncated:
                break

        endpoint = dict(final_info.get("endpoint") or {})
        # Reclassification is intentionally independent of the current
        # controller's own legacy endpoint criterion.
        endpoint["success"] = config.endpoint_success(
            object_kind,
            float(endpoint.get("position_error_m", float("nan"))),
            float(endpoint.get("euler_l2_error_deg", float("nan"))),
        )
        endpoint["position_ok"] = bool(
            endpoint.get("position_error_m", float("inf"))
            <= config.POSITION_TOLERANCE_M
        )
        endpoint["orientation_ok"] = bool(
            endpoint.get("euler_l2_error_deg", float("inf"))
            <= config.rotation_tolerance_deg(object_kind)
        )
        return _jsonable(
            {
                "controller": controller,
                "initial_state_hash": initial_state_hash,
                "action_ready_state_hash": final_info.get(
                    "action_ready_state_hash", ""
                ),
                "parent_contract_sha256": final_info.get(
                    "parent_contract_sha256", ""
                ),
                "success": bool(endpoint["success"]),
                "endpoint": endpoint,
                "workflow_reason": final_info.get("workflow_reason", ""),
                "return": total_reward,
                "control_steps": final_info.get("control_step", 0),
                "action_control_steps": final_info.get(
                    "action_control_step", 0
                ),
                "ready_result": final_info.get("ready_result"),
                "demo_result": final_info.get("demo_result"),
                "trajectory": trajectory,
            }
        )
    finally:
        env.close()


def _paired_initial_target(run: dict[str, Any]) -> dict[str, Any]:
    overall = run["endpoint"].get("overall_grasp_action", {})
    return {
        "target_reference": overall.get(
            "target_reference", "paired_initial_absolute_target"
        ),
        "target_com_m": overall.get("target_com_m"),
        "target_quaternion_wxyz": overall.get(
            "target_quaternion_wxyz"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.episodes_per_pair <= 0:
        raise SystemExit("--episodes-per-pair must be positive")
    model_path = args.model.resolve()
    normalize_path = args.vec_normalize.resolve()
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    if not normalize_path.is_file():
        raise SystemExit(f"normalization statistics not found: {normalize_path}")

    model = TQC.load(model_path, device=args.device)
    normalizer = _load_normalizer(normalize_path)
    pairs: list[dict[str, Any]] = []
    pair_index = 0
    for object_kind in config.OBJECTS:
        for task in config.TASKS:
            for repeat in range(args.episodes_per_pair):
                episode_seed = int(args.seed + pair_index)
                ik_only = _run_episode(
                    object_kind=object_kind,
                    task=task,
                    episode_seed=episode_seed,
                    controller="ik_only",
                    model=model,
                    normalizer=normalizer,
                    record_trajectory=(repeat == 0),
                )
                ik_residual = _run_episode(
                    object_kind=object_kind,
                    task=task,
                    episode_seed=episode_seed,
                    controller="ik_residual",
                    model=model,
                    normalizer=normalizer,
                    record_trajectory=(repeat == 0),
                )
                if (
                    ik_only["initial_state_hash"]
                    != ik_residual["initial_state_hash"]
                ):
                    raise RuntimeError(
                        "paired runs did not start from the same physics and "
                        "controller state"
                    )
                if (
                    not ik_only["parent_contract_sha256"]
                    or ik_only["parent_contract_sha256"]
                    != ik_residual["parent_contract_sha256"]
                ):
                    raise RuntimeError(
                        "paired runs do not share the same parent model contract"
                    )
                baseline_target = _paired_initial_target(ik_only)
                residual_target = _paired_initial_target(ik_residual)
                if baseline_target != residual_target:
                    raise RuntimeError(
                        "paired runs do not share the same initial absolute "
                        "grasp+action target"
                    )
                pairs.append(
                    {
                        "pair_id": (
                            f"{object_kind}:{task}:{episode_seed}"
                        ),
                        "object": object_kind,
                        "task": task,
                        "seed": episode_seed,
                        "repeat": repeat,
                        "initial_state_hash": ik_only[
                            "initial_state_hash"
                        ],
                        "parent_contract_sha256": ik_only[
                            "parent_contract_sha256"
                        ],
                        "parent_contract_verified": True,
                        "target_reference": {
                            "success": (
                                "post_lift_relative_action_target"
                            ),
                            "overall": (
                                "paired_initial_absolute_target"
                            ),
                            "grasp_lift": "paired_initial_lift_target",
                        },
                        "paired_initial_absolute_target": baseline_target,
                        "ik_only": ik_only,
                        "ik_residual": ik_residual,
                    }
                )
                pair_index += 1

    payload = {
        "schema": SCHEMA,
        "platform": "g6",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "vec_normalize_path": str(normalize_path),
            "vec_normalize_sha256": _sha256(normalize_path),
            "current_model_fingerprint_sha256": (
                current.MODEL_FINGERPRINT_SHA256
            ),
            "current_controller_fingerprint_sha256": (
                current.CONTROLLER_FINGERPRINT_SHA256
            ),
            "pairing": "same object, task, seed, and initial state hash",
        },
        "config": {
            "objects": list(config.OBJECTS),
            "tasks": list(config.TASKS),
            "plate_trained": False,
            "position_tolerance_m": config.POSITION_TOLERANCE_M,
            "rotation_tolerance_deg": {
                object_kind: config.rotation_tolerance_deg(object_kind)
                for object_kind in config.OBJECTS
            },
            "success_definition": (
                "endpoint COM L2 AND wrapped fixed-XYZ Euler-vector L2"
            ),
            "contact_or_hold_success_gate": False,
            "trajectory_policy": (
                "full 50 Hz telemetry only for repeat 0 of each "
                "object/task/controller; all repeats retain endpoints"
            ),
            "contact_force_threshold_n": (
                current.CONTACT_FORCE_THRESHOLD_N
            ),
            "control_hz": config.CONTROL_HZ,
        },
        "pairs": pairs,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")
    print(f"wrote {len(pairs)} paired episodes to {output}")


if __name__ == "__main__":
    main()
