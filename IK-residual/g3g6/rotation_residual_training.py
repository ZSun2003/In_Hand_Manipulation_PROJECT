"""Formal three-phase TQC training for the G3/G6 IK-residual policies.

Each gripper uses one policy and one replay buffer across the fixed curriculum:
500k grasp/lift steps, 2M translation steps, then 2M weighted 0--90 degree
rotation steps.  TQC hyperparameters match the G3 Translation reference.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import resource
from typing import Any, Sequence

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from . import config
from .env import G3G6ManipulationEnv, OBSERVATION_DIM


TRAINING_SCHEMA = "g3g6-full-residual-tqc-training-v1"
PHASE_BUDGETS = {
    config.Phase.GRASP_LIFT: 500_000,
    config.Phase.TRANSLATE: 2_000_000,
    config.Phase.ROTATE: 2_000_000,
}
TOTAL_TIMESTEPS = sum(PHASE_BUDGETS.values())
N_ENVS = 72
SEED = 7
POLICY_NET_ARCH = (256, 256)
N_CRITICS = 4
N_QUANTILES = 32
TOP_QUANTILES_TO_DROP_PER_NET = 2
LEARNING_RATE = 1.0e-4
BUFFER_SIZE = 1_000_000
LEARNING_STARTS = 20_000
BATCH_SIZE = 256
TAU = 0.005
GAMMA = 0.99
TRAIN_FREQ = 1
GRADIENT_STEPS = 8
USE_SDE = True
SDE_SAMPLE_FREQ = 64
CLIP_OBS = 10.0
CLIP_REWARD = 10.0
ROTATION_SAMPLING_PROFILE = "weighted_0_90"
ROTATION_SAMPLING_BINS = (
    {"minimum_deg": 0, "maximum_deg": 30, "probability": 0.20},
    {"minimum_deg": 31, "maximum_deg": 60, "probability": 0.50},
    {"minimum_deg": 61, "maximum_deg": 90, "probability": 0.30},
)
EXPECTED_ACTION_DIMS = {"g3": 12, "g6": 16}
EXPECTED_OBSERVATION_DIM = 149
REQUESTED_NOFILE_SOFT_LIMIT = 65_536


def _round_up_to_vector_step(steps: int) -> int:
    """Return the number of transitions SB3 executes with ``N_ENVS`` workers."""

    return int(math.ceil(int(steps) / N_ENVS) * N_ENVS)


EXECUTED_PHASE_TIMESTEPS = {
    phase: _round_up_to_vector_step(budget)
    for phase, budget in PHASE_BUDGETS.items()
}
EXECUTED_TOTAL_TIMESTEPS = sum(EXECUTED_PHASE_TIMESTEPS.values())


def raise_file_descriptor_limit() -> tuple[int, int]:
    """Raise this training process' open-file soft limit for 72 workers."""

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = REQUESTED_NOFILE_SOFT_LIMIT
    if hard != resource.RLIM_INFINITY:
        target = min(target, int(hard))
    if int(soft) < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    updated_soft, updated_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if int(updated_soft) < target:
        raise RuntimeError(
            f"open-file soft limit is {updated_soft}; required at least {target}"
        )
    return int(updated_soft), int(updated_hard)


def _inside_project(path: Path) -> Path:
    value = (
        (config.PROJECT_ROOT / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )
    try:
        value.relative_to(config.PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("rotation residual training output must stay inside IK-residual") from exc
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_close(label: str, actual: float, expected: float) -> None:
    if not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise RuntimeError(f"{label}={actual!r}; expected {expected!r}")


def validate_training_contract() -> None:
    """Fail before creating outputs if a dependency changed the experiment."""

    if PHASE_BUDGETS != {
        config.Phase.GRASP_LIFT: 500_000,
        config.Phase.TRANSLATE: 2_000_000,
        config.Phase.ROTATE: 2_000_000,
    } or config.PHASE_TIMESTEPS != PHASE_BUDGETS:
        raise RuntimeError("formal three-phase budgets changed")
    if TOTAL_TIMESTEPS != 4_500_000:
        raise RuntimeError("formal per-gripper total must be 4,500,000 steps")
    if N_ENVS != 72:
        raise RuntimeError("formal training must use 72 vector environments")
    if EXECUTED_PHASE_TIMESTEPS != {
        config.Phase.GRASP_LIFT: 500_040,
        config.Phase.TRANSLATE: 2_000_016,
        config.Phase.ROTATE: 2_000_016,
    } or EXECUTED_TOTAL_TIMESTEPS != 4_500_072:
        raise RuntimeError("72-environment SB3 step-rounding contract changed")
    for gripper, expected in EXPECTED_ACTION_DIMS.items():
        actual = config.action_dim(gripper)
        if actual != expected:
            raise RuntimeError(
                f"{gripper} action dimension is {actual}; expected {expected}"
            )
    if OBSERVATION_DIM != EXPECTED_OBSERVATION_DIM:
        raise RuntimeError(
            f"observation dimension is {OBSERVATION_DIM}; "
            f"expected {EXPECTED_OBSERVATION_DIM}"
        )
    _require_close(
        "rotation target height",
        config.ROTATION_TARGET_GRASP_PLANE_Z_M,
        0.325,
    )
    _require_close(
        "rotation maximum",
        config.ROTATION_SWEEP_ANGLE_MAX_DEG,
        90.0,
    )
    _require_close("angular residual limit", config.ANGULAR_RESIDUAL_LIMIT_RAD, 0.005)
    _require_close(
        "angular residual rate",
        config.ANGULAR_RESIDUAL_RATE_RAD_PER_TICK,
        0.0001,
    )
    _require_close("angular residual blend", config.ANGULAR_RESIDUAL_BLEND, 0.35)
    _require_close("linear residual limit", config.LINEAR_RESIDUAL_LIMIT_M, 0.0005)
    _require_close(
        "linear residual rate",
        config.LINEAR_RESIDUAL_RATE_M_PER_TICK,
        0.00002,
    )
    _require_close("linear residual blend", config.LINEAR_RESIDUAL_BLEND, 0.35)
    if config.WEIGHTED_ROTATION_SAMPLING_PROFILE != ROTATION_SAMPLING_PROFILE:
        raise RuntimeError("weighted rotation sampling profile name changed")
    expected_bins = tuple(
        (int(item["minimum_deg"]), int(item["maximum_deg"]))
        for item in ROTATION_SAMPLING_BINS
    )
    if tuple(config.WEIGHTED_ROTATION_MAGNITUDE_BINS_DEG) != expected_bins:
        raise RuntimeError("weighted rotation magnitude bins changed")
    expected_probabilities = tuple(
        float(item["probability"]) for item in ROTATION_SAMPLING_BINS
    )
    if tuple(config.WEIGHTED_ROTATION_BIN_PROBABILITIES) != expected_probabilities:
        raise RuntimeError("weighted rotation bin probabilities changed")


def _phase_env(
    gripper: str,
    phase: config.Phase,
    rank: int,
    seed: int,
) -> G3G6ManipulationEnv:
    """Build one phase worker with the G3 Translation seeding contract."""

    selected_phase = config.Phase(phase)
    kwargs: dict[str, Any] = {
        "gripper": gripper,
        "phase": selected_phase,
        "residual_nominal_feedback": True,
        # Phase 1 exists specifically to learn grasp/lift residuals.  For the
        # movement phases, keep the prefix identical to the pure-IK baseline.
        "residual_after_preaction_only": (
            selected_phase != config.Phase.GRASP_LIFT
        ),
    }
    if selected_phase == config.Phase.ROTATE:
        kwargs.update(
            {
                "rotation_goal_max_deg": config.ROTATION_SWEEP_ANGLE_MAX_DEG,
                "rotation_sampling_profile": ROTATION_SAMPLING_PROFILE,
            }
        )
    environment = G3G6ManipulationEnv(**kwargs)
    environment.action_space.seed(int(seed) + int(rank))
    return environment


def _build_phase_vec_env(
    gripper: str,
    phase: config.Phase,
    normalization_state: tuple[Any, Any] | None,
) -> VecNormalize:
    phase_seed = SEED + int(phase) * 10_000
    factories = [
        partial(_phase_env, gripper, phase, rank, phase_seed)
        for rank in range(N_ENVS)
    ]
    raw = SubprocVecEnv(factories, start_method="spawn")
    raw.seed(phase_seed)
    normalized = VecNormalize(
        VecMonitor(raw),
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=CLIP_OBS,
        clip_reward=CLIP_REWARD,
    )
    if normalization_state is not None:
        normalized.obs_rms, normalized.ret_rms = normalization_state
    return normalized


def _terminate_replay_phase_boundary(model: TQC) -> None:
    """Make an abandoned final state terminal before switching environments."""

    replay = model.replay_buffer
    if replay is None or replay.size() <= 0:
        return
    final_index = (int(replay.pos) - 1) % int(replay.buffer_size)
    replay.dones[final_index, :] = 1.0
    if hasattr(replay, "timeouts"):
        replay.timeouts[final_index, :] = 0.0


def _residual_manifest(gripper: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action_dimension": config.action_dim(gripper),
        "meaning": "normalized_direct_actuator_residual_after_nominal_ik",
        "nominal_ik_pose_feedback": True,
        "activation_by_phase": {
            "1": "all_ticks",
            "2": "after_preaction_tick_360",
            "3": "after_preaction_tick_360",
        },
        "blend": float(config.ANGULAR_RESIDUAL_BLEND),
        "rotary": {
            "count": 12,
            "unit": "rad",
            "limit": float(config.ANGULAR_RESIDUAL_LIMIT_RAD),
            "rate_per_control_tick": float(
                config.ANGULAR_RESIDUAL_RATE_RAD_PER_TICK
            ),
            "actuator_indices": list(config.ANGULAR_ACTUATOR_INDICES[gripper]),
        },
    }
    if gripper == "g6":
        payload["linear"] = {
            "count": 4,
            "unit": "m",
            "limit": float(config.LINEAR_RESIDUAL_LIMIT_M),
            "rate_per_control_tick": float(
                config.LINEAR_RESIDUAL_RATE_M_PER_TICK
            ),
            "blend": float(config.LINEAR_RESIDUAL_BLEND),
            "actuator_indices": list(config.G6_LINEAR_ACTUATOR_INDICES),
        }
    else:
        payload["linear"] = {
            "count": 0,
            "actuator_indices": [],
        }
    penalty: Any | None = getattr(config, "RESIDUAL_PENALTY_WEIGHTS", None)
    if penalty is None:
        penalty = getattr(config, "RESIDUAL_PENALTY_WEIGHT", None)
    if callable(penalty):
        penalty = penalty(gripper)
    elif isinstance(penalty, Mapping):
        penalty = penalty[gripper]
    if penalty is not None:
        payload["reward_penalty_weight"] = float(penalty)
    return payload


def formal_config(gripper: str, *, device: str = "auto") -> dict[str, Any]:
    """Return the JSON-safe immutable training contract for one gripper."""

    selected = config.GripperKind(gripper).value
    return {
        "schema": TRAINING_SCHEMA,
        "gripper": selected,
        "algorithm": "TQC",
        # The final artifact is produced after Phase 3; retain this scalar for
        # strict rotation-sweep artifact validation.
        "phase": int(config.Phase.ROTATE),
        "phase_name": "full_curriculum_final_rotate",
        "phase_sequence": [
            {
                "phase": int(phase),
                "name": phase.name.lower(),
                "budget": PHASE_BUDGETS[phase],
                "expected_executed_steps": EXECUTED_PHASE_TIMESTEPS[phase],
                "environment_seed": SEED + int(phase) * 10_000,
            }
            for phase in config.Phase
        ],
        "phase_budgets": {
            str(int(phase)): PHASE_BUDGETS[phase] for phase in config.Phase
        },
        "total_timesteps": TOTAL_TIMESTEPS,
        "expected_executed_timesteps": EXECUTED_TOTAL_TIMESTEPS,
        "learn_calls": 3,
        "one_uninterrupted_learn_call_per_phase": True,
        "same_policy_and_replay_across_phases": True,
        "normalization_statistics_carried_forward": True,
        "success_rate_gate": False,
        "n_envs": N_ENVS,
        "runtime_requirements": {
            "requested_nofile_soft_limit": REQUESTED_NOFILE_SOFT_LIMIT,
            "close_previous_phase_workers_before_spawn": True,
        },
        "vec_env": {
            "class": "SubprocVecEnv",
            "start_method": "spawn",
        },
        "seed": SEED,
        "requested_device": str(device),
        "observation_dimension": OBSERVATION_DIM,
        "task": {
            "final_phase": int(config.Phase.ROTATE),
            "target_grasp_plane_z_m": float(
                config.ROTATION_TARGET_GRASP_PLANE_Z_M
            ),
            "objects_uniform": list(config.OBJECT_NAMES),
            "sampling_profile": ROTATION_SAMPLING_PROFILE,
            "orientation_convention": "ordered_local_xyz_euler_components",
            "component_angle_integer_deg": {
                "minimum": 0,
                "maximum": 90,
                "bins": [dict(item) for item in ROTATION_SAMPLING_BINS],
            },
            "axis_sets_uniform": list(config.ROTATION_AXIS_SETS),
            "active_axes_share_one_component_magnitude": True,
            "active_axis_signs": "independent_uniform_plus_or_minus",
            "episode_ticks": config.FORMAL_EPISODE_TICKS,
            "preaction_ticks": config.PRE_ACTION_TICKS,
            "rotation_motion_ticks": config.ACTION_MOTION_TICKS,
            "terminal_hold_ticks": config.POST_ACTION_HOLD_TICKS,
            "control_hz": config.CONTROL_HZ,
            "physics_substeps_per_control_tick": config.PHYSICS_SUBSTEPS,
        },
        "phase_tasks": {
            "1": {
                "name": "grasp_lift",
                "target_grasp_plane_z_m": config.TARGET_GRASP_PLANE_Z_M,
                "objects_uniform": list(config.OBJECT_NAMES),
                "residual_activation": "all_ticks",
            },
            "2": {
                "name": "translate",
                "target_grasp_plane_z_m": config.TARGET_GRASP_PLANE_Z_M,
                "objects_uniform": list(config.OBJECT_NAMES),
                "paths_uniform": list(("square", "circle", "S", "U", "N")),
                "residual_activation": "after_preaction_tick_360",
            },
            "3": {
                "name": "rotate",
                "target_grasp_plane_z_m": (
                    config.ROTATION_TARGET_GRASP_PLANE_Z_M
                ),
                "objects_uniform": list(config.OBJECT_NAMES),
                "rotation_sampling_profile": ROTATION_SAMPLING_PROFILE,
                "residual_activation": "after_preaction_tick_360",
            },
        },
        "residual": _residual_manifest(selected),
        "vec_normalize": {
            "norm_obs": True,
            "norm_reward": True,
            "clip_obs": CLIP_OBS,
            "clip_reward": CLIP_REWARD,
            "gamma": GAMMA,
        },
        "policy": {
            "name": "MlpPolicy",
            "net_arch": list(POLICY_NET_ARCH),
            "n_critics": N_CRITICS,
            "n_quantiles": N_QUANTILES,
            "activation": "ReLU",
            "log_std_init": -3.0,
            "optimizer": "Adam",
        },
        "optimizer": {
            "learning_rate": LEARNING_RATE,
            "buffer_size": BUFFER_SIZE,
            "learning_starts": LEARNING_STARTS,
            "batch_size": BATCH_SIZE,
            "tau": TAU,
            "gamma": GAMMA,
            "train_freq": TRAIN_FREQ,
            "gradient_steps": GRADIENT_STEPS,
            "top_quantiles_to_drop_per_net": TOP_QUANTILES_TO_DROP_PER_NET,
            "use_sde": USE_SDE,
            "sde_sample_freq": SDE_SAMPLE_FREQ,
            "ent_coef": "auto",
            "target_entropy": "auto",
            "target_update_interval": 1,
            "use_sde_at_warmup": False,
            "action_noise": None,
            "optimize_memory_usage": False,
            "progress_bar": False,
        },
        "reward": {
            "position_scale_m": config.POSITION_REWARD_SCALE_M,
            "orientation_scale_deg": config.ORIENTATION_REWARD_SCALE_DEG,
            "contact_weight": config.CONTACT_REWARD_WEIGHT,
            "height_weight": config.HEIGHT_REWARD_WEIGHT,
            "endpoint_success_bonus": config.ENDPOINT_SUCCESS_BONUS,
            "residual_penalty": (
                "0.01 * normalized_residual_rms"
                if selected == "g3"
                else "0.002 * normalized_residual_mean_square"
            ),
        },
    }


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _build_model(vec_env: VecNormalize, *, device: str) -> TQC:
    return TQC(
        "MlpPolicy",
        vec_env,
        policy_kwargs={
            "net_arch": list(POLICY_NET_ARCH),
            "n_critics": N_CRITICS,
            "n_quantiles": N_QUANTILES,
            "log_std_init": -3.0,
        },
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        top_quantiles_to_drop_per_net=TOP_QUANTILES_TO_DROP_PER_NET,
        use_sde=USE_SDE,
        sde_sample_freq=SDE_SAMPLE_FREQ,
        ent_coef="auto",
        target_entropy="auto",
        target_update_interval=1,
        use_sde_at_warmup=False,
        action_noise=None,
        optimize_memory_usage=False,
        seed=SEED,
        device=device,
        verbose=1,
    )


def train_one_gripper(
    *,
    gripper: str,
    output_dir: Path,
    device: str = "auto",
) -> Path:
    """Train one policy sequentially through all three fixed phases."""

    validate_training_contract()
    nofile_soft, nofile_hard = raise_file_descriptor_limit()
    selected = config.GripperKind(gripper).value
    destination = _inside_project(Path(output_dir))
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite training output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    manifest_path = destination / "manifest.json"
    manifest = formal_config(selected, device=device)
    manifest.update(
        {
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "active_phase": None,
            "completed_phases": [],
            "runtime_file_descriptor_limit": {
                "soft": nofile_soft,
                "hard": (
                    "unlimited"
                    if nofile_hard == resource.RLIM_INFINITY
                    else nofile_hard
                ),
            },
        }
    )
    _write_manifest(manifest_path, manifest)

    vec_env: VecNormalize | None = None
    model: TQC | None = None
    active_phase: config.Phase | None = None
    try:
        for phase in config.Phase:
            active_phase = phase
            manifest["active_phase"] = int(phase)
            _write_manifest(manifest_path, manifest)

            normalization_state = None
            if vec_env is not None:
                normalization_state = (
                    deepcopy(vec_env.obs_rms),
                    deepcopy(vec_env.ret_rms),
                )
                if model is not None:
                    _terminate_replay_phase_boundary(model)
                # Close all old worker pipes before opening the next set of
                # 72.  Holding both sets at once exceeds the common macOS
                # launchd soft limit of 256 descriptors.
                vec_env.close()
                vec_env = None
            vec_env = _build_phase_vec_env(
                selected, phase, normalization_state
            )
            if model is None:
                model = _build_model(vec_env, device=device)
            else:
                model.set_env(vec_env)

            budget = PHASE_BUDGETS[phase]
            phase_start = int(model.num_timesteps)
            model.learn(
                total_timesteps=budget,
                reset_num_timesteps=(phase == config.Phase.GRASP_LIFT),
                progress_bar=False,
            )
            actual_steps = int(model.num_timesteps) - phase_start
            expected_steps = EXECUTED_PHASE_TIMESTEPS[phase]
            if actual_steps != expected_steps:
                raise RuntimeError(
                    f"phase {int(phase)} executed {actual_steps} steps; "
                    f"expected {expected_steps} after {N_ENVS}-env rounding"
                )

            phase_model = destination / f"phase{int(phase)}_model.zip"
            phase_normalizer = (
                destination / f"phase{int(phase)}_vec_normalize.pkl"
            )
            model.save(phase_model)
            vec_env.save(str(phase_normalizer))
            if not phase_model.is_file() or not phase_normalizer.is_file():
                raise RuntimeError(f"phase {int(phase)} artifacts were not written")
            manifest["completed_phases"].append(
                {
                    "phase": int(phase),
                    "name": phase.name.lower(),
                    "requested_steps": budget,
                    "expected_executed_steps": expected_steps,
                    "actual_steps": actual_steps,
                    "cumulative_steps": int(model.num_timesteps),
                    "model": {
                        "path": phase_model.name,
                        "sha256": _sha256(phase_model),
                    },
                    "vec_normalize": {
                        "path": phase_normalizer.name,
                        "sha256": _sha256(phase_normalizer),
                    },
                }
            )
            _write_manifest(manifest_path, manifest)

        assert model is not None and vec_env is not None
        model_path = destination / "final_model.zip"
        normalizer_path = destination / "vec_normalize.pkl"
        model.save(model_path)
        vec_env.save(str(normalizer_path))
        if not model_path.is_file() or not normalizer_path.is_file():
            raise RuntimeError("final TQC artifacts were not written")
        manifest.update(
            {
                "status": "complete",
                "active_phase": None,
                "completed_at": datetime.now().astimezone().isoformat(),
                "artifacts": {
                    "model": {
                        "path": model_path.name,
                        "sha256": _sha256(model_path),
                    },
                    "vec_normalize": {
                        "path": normalizer_path.name,
                        "sha256": _sha256(normalizer_path),
                    },
                },
            }
        )
        _write_manifest(manifest_path, manifest)
        return destination
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "active_phase": (
                    None if active_phase is None else int(active_phase)
                ),
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        _write_manifest(manifest_path, manifest)
        raise
    finally:
        if vec_env is not None:
            vec_env.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the complete three-phase IK-residual TQC curriculum "
            "(500k grasp/lift, 2M translation, 2M rotation) for G3/G6."
        )
    )
    parser.add_argument("--gripper", choices=("g3", "g6", "both"), default="both")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs")
        / datetime.now().strftime("full_residual_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the exact JSON contract and exit without creating files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_training_contract()
    grippers = ("g3", "g6") if args.gripper == "both" else (args.gripper,)
    if args.print_config:
        payload = {
            "schema": TRAINING_SCHEMA,
            "mode": "print_config_only",
            "training_started": False,
            "output_created": False,
            "grippers": [
                formal_config(gripper, device=args.device) for gripper in grippers
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    output_root = _inside_project(args.output_dir)
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite training output: {output_root}")
    for gripper in grippers:
        destination = train_one_gripper(
            gripper=gripper,
            output_dir=output_root / gripper,
            device=args.device,
        )
        print(f"completed {gripper}: {destination}")


if __name__ == "__main__":
    main()
