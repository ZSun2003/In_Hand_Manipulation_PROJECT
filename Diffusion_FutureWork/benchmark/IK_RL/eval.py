"""Paired-seed evaluation of current IK-only versus IK+residual control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .runtime import ensure_runtime

ensure_runtime()

import numpy as np

from . import config
from .env import G3IKResidualEnv


def json_safe(value: Any) -> Any:
    """Return a strict-JSON value, mapping non-finite diagnostics to null."""

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def _sha256_file(value: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(value).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def endpoint_success(
    endpoint: Mapping[str, Any],
    *,
    object_kind: str,
    target_ready: bool,
) -> bool:
    """Independently apply the formal endpoint conjunction."""

    try:
        position_error = float(endpoint["position_error_m"])
        rotation_error = float(endpoint["euler_l2_error_deg"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        target_ready
        and np.isfinite(position_error)
        and np.isfinite(rotation_error)
        and position_error <= config.POSITION_TOLERANCE_M + 1.0e-12
        and rotation_error
        <= config.rotation_tolerance_deg(object_kind) + 1.0e-10
    )


def _pair_seed(base_seed: int, object_index: int, task_index: int, episode: int) -> int:
    sequence = np.random.SeedSequence(
        [int(base_seed), int(object_index), int(task_index), int(episode)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def load_policy(
    model_path: str | Path,
    normalize_path: str | Path,
    *,
    device: str,
):
    from sb3_contrib import TQC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model_value = Path(model_path).expanduser().resolve()
    normalize_value = Path(normalize_path).expanduser().resolve()
    if not model_value.is_file():
        raise FileNotFoundError(model_value)
    if not normalize_value.is_file():
        raise FileNotFoundError(normalize_value)

    normalization_env = DummyVecEnv(
        [
            lambda: G3IKResidualEnv(
                fixed_object="cube",
                fixed_task="translate_xy45_100mm",
            )
        ]
    )
    normalizer = VecNormalize.load(str(normalize_value), normalization_env)
    normalizer.training = False
    normalizer.norm_reward = False
    model = TQC.load(str(model_value), device=device)
    return model, normalizer, normalization_env


def _normalized_observation(normalizer, observation: np.ndarray) -> np.ndarray:
    batch = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    return np.asarray(normalizer.normalize_obs(batch.copy())[0], dtype=np.float32)


def run_episode(
    *,
    control_mode: str,
    object_kind: str,
    task: str,
    pair_seed: int,
    model: Any = None,
    normalizer: Any = None,
    record_trajectory: bool,
) -> dict[str, Any]:
    environment = G3IKResidualEnv(
        control_mode=control_mode,
        record_trajectory=record_trajectory,
        fixed_object=object_kind,
        fixed_task=task,
    )
    total_reward = 0.0
    episode_length = 0
    handshake_action: list[float] = []
    last_info: dict[str, Any] = {}
    try:
        observation, last_info = environment.reset(
            seed=pair_seed,
            options={
                "object_kind": object_kind,
                "task": task,
                "pair_seed": pair_seed,
            },
        )
        terminated = False
        truncated = False
        while not (terminated or truncated):
            if model is None:
                action = np.zeros(config.ACTION_DIM, dtype=np.float32)
            else:
                policy_observation = _normalized_observation(
                    normalizer, observation
                )
                predicted, _state = model.predict(
                    policy_observation,
                    deterministic=True,
                )
                action = np.asarray(predicted, dtype=np.float32).reshape(
                    config.ACTION_DIM
                )
            if episode_length == 0:
                handshake_action = action.astype(float).tolist()
            observation, reward, terminated, truncated, last_info = (
                environment.step(action)
            )
            total_reward += float(reward)
            episode_length += 1
            if episode_length > 20_000:
                raise RuntimeError("evaluation episode exceeded 20,000 Gym steps")
    finally:
        environment.close()

    endpoint = last_info.get("endpoint") or {}
    position_error = endpoint.get("position_error_m")
    rotation_error = endpoint.get("euler_l2_error_deg")
    endpoint_evaluable = bool(
        last_info.get("action_target_ready", False)
        and position_error is not None
        and rotation_error is not None
        and np.isfinite(float(position_error))
        and np.isfinite(float(rotation_error))
    )
    controller_result = last_info.get("controller_result") or {}
    process_completed = bool(
        isinstance(controller_result, Mapping)
        and controller_result.get("demo") is not None
    )
    target_ready = bool(last_info.get("action_target_ready", False))
    reclassified_success = endpoint_success(
        endpoint,
        object_kind=object_kind,
        target_ready=target_ready,
    )
    return json_safe(
        {
            "success": reclassified_success,
            "env_reported_success": bool(last_info.get("is_success", False)),
            "success_reclassified_by_eval": True,
            "endpoint_evaluable": endpoint_evaluable,
            "process_completed": process_completed,
            "endpoint": endpoint,
            "overall_endpoint": last_info.get("overall_endpoint") or {},
            "trajectory": last_info.get("trajectory") or [],
            "precondition": last_info.get("precondition") or {},
            "plan_residual": last_info.get("plan_residual") or {},
            "plan_residual_ablation": bool(
                last_info.get("plan_residual_ablation", False)
            ),
            "target_reference": last_info.get("target_reference") or {},
            "targets": last_info.get("targets") or {},
            "initial_state_hash": last_info.get("initial_state_hash", ""),
            "integration_state_hash": last_info.get(
                "integration_state_hash", ""
            ),
            "controller_state_hash": last_info.get(
                "controller_state_hash", ""
            ),
            "parent_contract_sha256": last_info.get(
                "parent_contract_sha256", ""
            ),
            "failure_reason": last_info.get("failure_reason", ""),
            "controller_result": controller_result,
            "handshake_action": handshake_action,
            "episode_reward": total_reward,
            "episode_length": episode_length,
            "control_ticks": int(last_info.get("control_step", 0)),
        }
    )


def evaluate(args: argparse.Namespace) -> Path:
    config.validate()
    episodes_per_pair = int(args.episodes_per_pair)
    if episodes_per_pair <= 0:
        raise ValueError("--episodes-per-pair must be positive")
    model, normalizer, normalization_env = load_policy(
        args.model,
        args.vec_normalize,
        device=str(args.device),
    )
    pairs: list[dict[str, Any]] = []
    ablation_episodes: list[dict[str, Any]] = []
    try:
        for object_index, object_kind in enumerate(config.OBJECTS):
            for task_index, task in enumerate(config.TASKS):
                for episode in range(episodes_per_pair):
                    seed = _pair_seed(
                        int(args.seed), object_index, task_index, episode
                    )
                    record = episode == 0
                    ik_only = run_episode(
                        control_mode="ik_only",
                        object_kind=object_kind,
                        task=task,
                        pair_seed=seed,
                        record_trajectory=record,
                    )
                    ik_residual = run_episode(
                        control_mode="ik_residual",
                        object_kind=object_kind,
                        task=task,
                        pair_seed=seed,
                        model=model,
                        normalizer=normalizer,
                        record_trajectory=record,
                    )
                    hashes = {
                        str(ik_only.get("initial_state_hash", "")),
                        str(ik_residual.get("initial_state_hash", "")),
                    }
                    if "" in hashes or len(hashes) != 1:
                        raise RuntimeError(
                            "paired reset mismatch for "
                            f"{object_kind}/{task}/seed={seed}"
                        )
                    parent_contracts = {
                        str(ik_only.get("parent_contract_sha256", "")),
                        str(ik_residual.get("parent_contract_sha256", "")),
                    }
                    if "" in parent_contracts or len(parent_contracts) != 1:
                        raise RuntimeError(
                            "paired parent model contract mismatch for "
                            f"{object_kind}/{task}/seed={seed}"
                        )
                    ik_target = (
                        ik_only.get("targets", {}).get(
                            "paired_initial_absolute_lift_plus_action"
                        )
                    )
                    residual_target = (
                        ik_residual.get("targets", {}).get(
                            "paired_initial_absolute_lift_plus_action"
                        )
                    )
                    if ik_target is None or ik_target != residual_target:
                        raise RuntimeError(
                            "paired initial absolute target mismatch for "
                            f"{object_kind}/{task}/seed={seed}"
                        )
                    pairs.append(
                        {
                            "pair_id": (
                                f"{object_kind}:{task}:{episode:04d}"
                            ),
                            "object": object_kind,
                            "task": task,
                            "seed": seed,
                            "initial_state_hash": hashes.pop(),
                            "hash_verified": True,
                            "parent_contract_sha256": (
                                parent_contracts.pop()
                            ),
                            "parent_contract_verified": True,
                            "paired_initial_absolute_target": ik_target,
                            "paired_target_verified": True,
                            "ik_only": ik_only,
                            "ik_residual": ik_residual,
                        }
                    )
                    if (
                        object_kind == "banana"
                        and not bool(args.skip_plan_ablation)
                    ):
                        no_plan = run_episode(
                            control_mode="ik_residual_no_plan",
                            object_kind=object_kind,
                            task=task,
                            pair_seed=seed,
                            model=model,
                            normalizer=normalizer,
                            record_trajectory=record,
                        )
                        if (
                            no_plan.get("initial_state_hash")
                            != ik_only.get("initial_state_hash")
                        ):
                            raise RuntimeError(
                                "banana no-plan ablation reset hash mismatch"
                            )
                        ablation_episodes.append(
                            {
                                "object": object_kind,
                                "task": task,
                                "seed": seed,
                                "initial_state_hash": no_plan.get(
                                    "initial_state_hash"
                                ),
                                "ik_residual_no_plan": no_plan,
                            }
                        )
    finally:
        normalization_env.close()

    payload = json_safe(
        {
            "schema": config.EVAL_SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "model": str(Path(args.model).expanduser().resolve()),
            "vec_normalize": str(
                Path(args.vec_normalize).expanduser().resolve()
            ),
            "artifact_sha256": {
                "model": _sha256_file(args.model),
                "vec_normalize": _sha256_file(args.vec_normalize),
                "parent_model_xml": _sha256_file(config.MODEL_XML),
                "parent_controller": _sha256_file(
                    config.GRAB_DIR / "G3_grab.py"
                ),
            },
            "config": {
                "objects": list(config.OBJECTS),
                "tasks": list(config.TASKS),
                "episodes_per_pair": episodes_per_pair,
                "position_tolerance_m": config.POSITION_TOLERANCE_M,
                "rotation_tolerance_deg_by_object": {
                    name: config.rotation_tolerance_deg(name)
                    for name in config.OBJECTS
                },
                "endpoint_success_rule": (
                    "COM Euclidean L2 AND wrapped fixed-XYZ Euler-vector L2"
                ),
                "success_target": "post_lift_relative_action_target",
                "overall_diagnostic_target": (
                    "paired_initial_absolute_lift_plus_action_target"
                ),
                "plan_residual": {
                    "role": "g3_banana_reachability_residual",
                    "main_pair_arm": "ik_residual",
                    "ablation_arm": "ik_residual_no_plan",
                    "does_not_change_main_pair_statistics": True,
                },
            },
            "pairs": pairs,
            "g3_plan_residual_ablation": {
                "scope": "banana_only_diagnostic",
                "included_in_main_statistics": False,
                "episodes": ablation_episodes,
            },
        }
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired IK-only versus G3 IK+residual evaluation."
    )
    parser.add_argument(
        "--model",
        default=str(config.DEFAULT_MODEL_DIR / "final_model.zip"),
    )
    parser.add_argument(
        "--vec-normalize",
        default=str(config.DEFAULT_MODEL_DIR / "vec_normalize.pkl"),
    )
    parser.add_argument(
        "--episodes-per-pair",
        type=int,
        default=config.EVAL_EPISODES_PER_PAIR,
    )
    parser.add_argument("--seed", type=int, default=config.SEED + 1_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default=str(config.DEFAULT_EVAL_PATH),
    )
    parser.add_argument("--skip-plan-ablation", action="store_true")
    return parser.parse_args()


def main() -> None:
    output = evaluate(parse_args())
    print(f"paired evaluation: {output}")


if __name__ == "__main__":
    main()
