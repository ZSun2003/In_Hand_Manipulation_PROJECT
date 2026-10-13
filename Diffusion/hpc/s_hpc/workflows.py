"""End-to-end GPU workflows used by the CLI and HX job scripts."""

from __future__ import annotations

from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import random
import signal
import time
from typing import Dict, Optional

import numpy as np

from diffusion_policy import DiffusionPolicy, ExponentialMovingAverage
from screw_control import TaskPhase

from .checkpointing import append_jsonl, atomic_torch_save, sha256_file
from .config import canonicalize_config_payload
from .contracts import (
    ACTION_DIM,
    BACKEND,
    OBS_DIM,
    OBS_HORIZON,
    RESIDUAL_ACTION_DIM,
    SCHEMA_HASH,
    TQC_OBS_DIM,
    TASK_GEOMETRY_HASH,
    checkpoint_metadata,
    validate_checkpoint,
)
from screw_control import SCRIPTED_PREFIX_STEPS
from .demos import DemoStore, DemoWindowSampler
from .diffusion_runtime import BatchedDiffusionController
from .env import MJWarpScrewEnv
from .monitoring import (
    EpisodeTracker,
    append_history,
    canonical_sha256,
    describe_paths,
    utc_now,
)
from .rollout_audit import (
    ExpertRolloutAudit,
    certify_expert_rollout,
)


def _torch():
    import torch

    return torch


def _load(path: Path):
    return _torch().load(path, map_location="cpu", weights_only=False)


def _cpu_tree(value):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item) for item in value)
    return value


def _metric_scalars(metrics) -> Dict[str, float]:
    torch = _torch()
    return {
        key: (
            float(value.detach().cpu())
            if isinstance(value, torch.Tensor)
            else float(value)
        )
        for key, value in metrics.items()
    }


def _safe_cursor_interrupt(interrupt, monitor) -> None:
    """Honor signals/stop files only after a complete resumable cursor."""

    if monitor is not None and monitor.stop_requested():
        raise KeyboardInterrupt
    if interrupt is not None:
        interrupt.checkpoint()


_PENDING_REPLAY_KEYS = (
    "world_ids",
    "observation",
    "action",
    "reward",
    "action_mask",
    "terminated",
    "truncated",
)


def _restore_pending_replay(value, *, device):
    """Restore the one-step delayed TQC transition carried by a checkpoint."""

    if value is None:
        return None
    torch = _torch()
    if not isinstance(value, dict):
        raise ValueError("pending TQC replay state must be a dictionary")
    missing = set(_PENDING_REPLAY_KEYS) - set(value)
    if missing:
        raise ValueError(
            f"pending TQC replay state is missing {sorted(missing)}"
        )
    pending = {
        name: torch.as_tensor(value[name], device=device)
        for name in _PENDING_REPLAY_KEYS
    }
    count = int(pending["world_ids"].numel())
    expected = {
        "world_ids": (count,),
        "observation": (count, TQC_OBS_DIM),
        "action": (count, RESIDUAL_ACTION_DIM),
        "reward": (count,),
        "action_mask": (count, RESIDUAL_ACTION_DIM),
        "terminated": (count,),
        "truncated": (count,),
    }
    for name, shape in expected.items():
        if tuple(pending[name].shape) != shape:
            raise ValueError(
                f"pending TQC {name} shape {tuple(pending[name].shape)} "
                f"!= {shape}"
            )
    if pending["world_ids"].dtype != torch.long:
        raise ValueError("pending TQC world_ids must use torch.long")
    for name in ("action_mask", "terminated", "truncated"):
        if pending[name].dtype != torch.bool:
            raise ValueError(f"pending TQC {name} must be boolean")
    for name in ("observation", "action", "reward"):
        if not bool(torch.isfinite(pending[name]).all()):
            raise FloatingPointError(f"pending TQC {name} is non-finite")
    return None if count == 0 else pending


def _terminal_tqc_observation(result):
    """Construct the recorded terminal state before masked world reset.

    The exact next nominal command is prepared only on the following call to
    ``env.step``.  Terminated transitions never bootstrap.  For a time-limit
    truncation, preserve the post-physics 191-D state and the newly realised
    residual while carrying the current certified nominal/mask/controller
    fields.  This is the terminal state, not the freshly reset episode state;
    both termination types disable critic bootstrapping.
    """

    state = result.info["tqc_observation"].clone()
    state[:, :OBS_DIM] = result.observation
    previous = slice(
        OBS_DIM + RESIDUAL_ACTION_DIM,
        OBS_DIM + 2 * RESIDUAL_ACTION_DIM,
    )
    state[:, previous] = result.info[
        "normalised_realised_actuator_residual"
    ]
    if tuple(state.shape) != (result.observation.shape[0], TQC_OBS_DIM):
        raise RuntimeError("terminal TQC observation has an invalid shape")
    return state


def _track_terminal_episodes(
    tracker: EpisodeTracker,
    info,
    done_ids,
    *,
    env,
    monitor=None,
    cursor: Optional[Dict[str, object]] = None,
) -> int:
    """Move all terminal statistics to CPU with one device synchronization."""

    torch = _torch()
    count = int(done_ids.numel())
    if count == 0:
        return 0
    values = dict(info)
    spawn_radius, _fixed_angle = (
        env.supervisor.curriculum_parameters(env.curriculum_progress)
    )
    reference = info["depth"]
    values.update(
        {
            "terminal_phase": env.state.desired_command.phase,
            "curriculum_progress": torch.full_like(
                reference, float(env.curriculum_progress)
            ),
            "spawn_radius_m": torch.full_like(
                reference, float(spawn_radius)
            ),
            "world_id": torch.arange(
                env.worlds, device=env.device, dtype=torch.long
            ),
            "spawn_x_m": env.state.spawn_xy[:, 0],
            "spawn_y_m": env.state.spawn_xy[:, 1],
            "socket_mouth_x_m": env.state.socket_mouth[:, 0],
            "socket_mouth_y_m": env.state.socket_mouth[:, 1],
        }
    )
    keys = (
        "success",
        "failure",
        "truncated",
        "episode_return",
        "depth",
        "twist",
        "lateral_error",
        "axis_error",
        "speed",
        "elapsed_steps",
        "ik_failures",
        "ik_anchor_fallbacks",
        "ik_safety_stops",
        "nonfinite",
        "fell_below_table",
        "lateral_escape",
        "safety_stop",
        "contact_constraint_violation",
        "contact_violation_before_capture",
        "contact_violation_before_thread",
        "contact_violation_premature_regrasp",
        "contact_violation_material_drift",
        "terminal_phase",
        "curriculum_progress",
        "spawn_radius_m",
        "thread_engaged",
        "holder_active",
        "grasp_latched",
        "diffusion_handoff_ready",
        "diffusion_handoff_hold_steps",
        "expert_step",
        "preinsert_alignment_ready",
        "preinsert_alignment_hold",
        "preinsert_alignment_stable_steps",
        "stable_grasp_support",
        "all_four_contact",
        "primary_pair_contact",
        "two_finger_contact",
        "capture_alignment_ready",
        "capture_alignment_hold",
        "capture_alignment_stable_steps",
        "capture_head_bottom_standoff",
        "shaft_captured",
        "secondary_released",
        "thread_engagement_ready",
        "thread_engagement_hold",
        "thread_regrasp_hold",
        "thread_ready_stable_steps",
        "world_id",
        "spawn_x_m",
        "spawn_y_m",
        "socket_mouth_x_m",
        "socket_mouth_y_m",
        "socket_angle_deg",
    )
    packed = torch.stack(
        [
            values[key].index_select(0, done_ids).to(torch.float64)
            for key in keys
        ],
        dim=1,
    ).detach().cpu()
    cpu_info = {
        key: packed[:, index]
        for index, key in enumerate(keys)
    }
    cpu_ids = torch.arange(count, dtype=torch.long)
    return tracker.add_from_info(
        cpu_info,
        cpu_ids,
        episode_lengths=packed[:, keys.index("elapsed_steps")],
        event_context=(
            monitor.event_context(**dict(cursor or {}))
            if monitor is not None
            else dict(cursor or {})
        ),
    )


def _materialize_rollout_metrics(
    env,
    result,
    *,
    action=None,
    learner_metrics=None,
):
    """Materialize rollout and optional learner scalars in one CPU transfer."""

    torch = _torch()
    info = result.info
    phase_count = len(TaskPhase)
    phase_histogram = torch.bincount(
        env.state.desired_command.phase,
        minlength=phase_count,
    ).to(torch.float32)
    action_value = (
        action
        if action is not None
        else torch.zeros((env.worlds, 1), device=env.device)
    )
    scalar_names = (
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
        "depth_mean_m",
        "depth_max_m",
        "turns_mean",
        "turns_max",
        "lateral_error_mean_m",
        "lateral_error_max_m",
        "axis_error_mean_rad",
        "axis_error_max_rad",
        "speed_mean",
        "speed_max",
        "ik_feasible_fraction",
        "thread_engaged_fraction",
        "holder_active_fraction",
        "grasp_latched_fraction",
        "done_worlds",
        "current_episode_steps_min",
        "current_episode_steps_mean",
        "current_episode_steps_max",
        "action_abs_mean",
        "action_abs_max",
        "action_saturation_fraction",
        "residual_active_dimensions_mean",
        "diffusion_active_fraction",
        "scripted_prefix_fraction",
        "diffusion_handoff_ready_fraction",
        "diffusion_handoff_position_error_mean_m",
        "diffusion_handoff_position_error_max_m",
        "diffusion_handoff_orientation_error_mean_deg",
        "diffusion_handoff_orientation_error_max_deg",
        "diffusion_handoff_hold_steps_mean",
        "diffusion_handoff_hold_steps_max",
        "preinsert_alignment_ready_fraction",
        "preinsert_alignment_hold_fraction",
        "preinsert_alignment_stable_steps_mean",
        "preinsert_alignment_stable_steps_max",
        "preinsert_lateral_error_mean_m",
        "preinsert_lateral_error_max_m",
        "preinsert_axis_error_mean_deg",
        "preinsert_axis_error_max_deg",
        "preinsert_depth_error_mean_m",
        "preinsert_depth_error_max_m",
        "preinsert_tip_speed_mean_m_s",
        "preinsert_tip_speed_max_m_s",
        "preinsert_lateral_speed_mean_m_s",
        "preinsert_lateral_speed_max_m_s",
        "preinsert_angular_speed_mean_deg_s",
        "preinsert_angular_speed_max_deg_s",
        "preinsert_feedback_lateral_mean_m",
        "preinsert_feedback_lateral_max_m",
        "preinsert_feedback_depth_abs_mean_m",
        "preinsert_feedback_depth_abs_max_m",
        "stable_grasp_support_fraction",
        "safety_gate_guard_fraction",
        "all_four_contact_fraction",
        "primary_pair_contact_fraction",
        "two_finger_contact_fraction",
        "contact_constraint_violation_fraction",
        "contact_violation_before_capture_fraction",
        "contact_violation_before_thread_fraction",
        "contact_violation_premature_regrasp_fraction",
        "contact_violation_material_drift_fraction",
        "contact_recovery_pending_fraction",
        "contact_recovery_hold_fraction",
        "before_capture_contact_loss_steps_max",
        "before_thread_two_finger_loss_steps_max",
        "fingertip_contact_force_min_n",
        "fingertip_contact_force_mean_n",
        "shaft_captured_fraction",
        "secondary_released_fraction",
        "capture_alignment_ready_fraction",
        "capture_alignment_hold_fraction",
        "capture_alignment_stable_steps_mean",
        "capture_alignment_stable_steps_max",
        "capture_head_bottom_standoff_min_m",
        "capture_head_bottom_standoff_mean_m",
        "capture_lateral_error_mean_m",
        "capture_lateral_error_max_m",
        "capture_axis_error_mean_deg",
        "capture_axis_error_max_deg",
        "capture_depth_error_mean_m",
        "capture_depth_error_max_m",
        "thread_engagement_ready_fraction",
        "thread_engagement_hold_fraction",
        "thread_regrasp_hold_fraction",
        "thread_ready_stable_steps_mean",
        "thread_ready_stable_steps_max",
        "thread_lateral_error_mean_m",
        "thread_lateral_error_max_m",
        "thread_axis_error_mean_deg",
        "thread_axis_error_max_deg",
        "thread_depth_error_mean_m",
        "thread_depth_error_max_m",
        "socket_angle_min_deg",
        "socket_angle_mean_deg",
        "socket_angle_max_deg",
        "socket_mouth_radius_min_m",
        "socket_mouth_radius_mean_m",
        "socket_mouth_radius_max_m",
        "socket_mouth_x_min_m",
        "socket_mouth_x_max_m",
        "socket_mouth_y_min_m",
        "socket_mouth_y_max_m",
    )
    turns = info["twist"] / (2.0 * math.pi)
    socket_mouth_xy = info["socket_mouth_xy"]
    socket_mouth_radius = torch.linalg.vector_norm(
        socket_mouth_xy, dim=-1
    )
    scalar_values = [
        result.reward.mean(),
        result.reward.std(unbiased=False),
        result.reward.min(),
        result.reward.max(),
        info["depth"].mean(),
        info["depth"].max(),
        turns.mean(),
        turns.max(),
        info["lateral_error"].mean(),
        info["lateral_error"].max(),
        info["axis_error"].mean(),
        info["axis_error"].max(),
        info["speed"].mean(),
        info["speed"].max(),
        result.ik_feasible_fraction.mean(),
        info["thread_engaged"].to(torch.float32).mean(),
        info["holder_active"].to(torch.float32).mean(),
        info["grasp_latched"].to(torch.float32).mean(),
        result.done.to(torch.float32).sum(),
        info["elapsed_steps"].to(torch.float32).min(),
        info["elapsed_steps"].to(torch.float32).mean(),
        info["elapsed_steps"].to(torch.float32).max(),
        action_value.abs().mean(),
        action_value.abs().max(),
        (action_value.abs() >= 0.98).to(torch.float32).mean(),
        info["residual_active_dimensions"].to(torch.float32).mean(),
        info["diffusion_active"].to(torch.float32).mean(),
        info["scripted_prefix_active"].to(torch.float32).mean(),
        info["diffusion_handoff_ready"].to(torch.float32).mean(),
        info["diffusion_handoff_position_error_m"].mean(),
        info["diffusion_handoff_position_error_m"].max(),
        info["diffusion_handoff_orientation_error_deg"].mean(),
        info["diffusion_handoff_orientation_error_deg"].max(),
        info["diffusion_handoff_hold_steps"].to(torch.float32).mean(),
        info["diffusion_handoff_hold_steps"].to(torch.float32).max(),
        info["preinsert_alignment_ready"].to(torch.float32).mean(),
        info["preinsert_alignment_hold"].to(torch.float32).mean(),
        info["preinsert_alignment_stable_steps"].to(torch.float32).mean(),
        info["preinsert_alignment_stable_steps"].to(torch.float32).max(),
        info["preinsert_lateral_error_m"].mean(),
        info["preinsert_lateral_error_m"].max(),
        info["preinsert_axis_error_deg"].mean(),
        info["preinsert_axis_error_deg"].max(),
        info["preinsert_depth_error_m"].mean(),
        info["preinsert_depth_error_m"].max(),
        info["preinsert_tip_speed_m_s"].mean(),
        info["preinsert_tip_speed_m_s"].max(),
        info["preinsert_lateral_speed_m_s"].mean(),
        info["preinsert_lateral_speed_m_s"].max(),
        info["preinsert_angular_speed_deg_s"].mean(),
        info["preinsert_angular_speed_deg_s"].max(),
        info["preinsert_feedback_lateral_m"].mean(),
        info["preinsert_feedback_lateral_m"].max(),
        info["preinsert_feedback_depth_m"].abs().mean(),
        info["preinsert_feedback_depth_m"].abs().max(),
        info["stable_grasp_support"].to(torch.float32).mean(),
        info["safety_gate_guard_active"].to(torch.float32).mean(),
        info["all_four_contact"].to(torch.float32).mean(),
        info["primary_pair_contact"].to(torch.float32).mean(),
        info["two_finger_contact"].to(torch.float32).mean(),
        info["contact_constraint_violation"].to(torch.float32).mean(),
        info["contact_violation_before_capture"].to(
            torch.float32
        ).mean(),
        info["contact_violation_before_thread"].to(
            torch.float32
        ).mean(),
        info["contact_violation_premature_regrasp"].to(
            torch.float32
        ).mean(),
        info["contact_violation_material_drift"].to(
            torch.float32
        ).mean(),
        info["contact_recovery_pending"].to(torch.float32).mean(),
        info["contact_recovery_hold"].to(torch.float32).mean(),
        info["before_capture_contact_loss_steps"].to(
            torch.float32
        ).max(),
        info["before_thread_two_finger_loss_steps"].to(
            torch.float32
        ).max(),
        info["fingertip_contact_force_n"].min(),
        info["fingertip_contact_force_n"].mean(),
        info["shaft_captured"].to(torch.float32).mean(),
        info["secondary_released"].to(torch.float32).mean(),
        info["capture_alignment_ready"].to(torch.float32).mean(),
        info["capture_alignment_hold"].to(torch.float32).mean(),
        info["capture_alignment_stable_steps"].to(
            torch.float32
        ).mean(),
        info["capture_alignment_stable_steps"].to(
            torch.float32
        ).max(),
        info["capture_head_bottom_standoff"].min(),
        info["capture_head_bottom_standoff"].mean(),
        info["capture_lateral_error_m"].mean(),
        info["capture_lateral_error_m"].max(),
        info["capture_axis_error_deg"].mean(),
        info["capture_axis_error_deg"].max(),
        info["capture_depth_error_m"].mean(),
        info["capture_depth_error_m"].max(),
        info["thread_engagement_ready"].to(torch.float32).mean(),
        info["thread_engagement_hold"].to(torch.float32).mean(),
        info["thread_regrasp_hold"].to(torch.float32).mean(),
        info["thread_ready_stable_steps"].to(torch.float32).mean(),
        info["thread_ready_stable_steps"].to(torch.float32).max(),
        info["thread_lateral_error_m"].mean(),
        info["thread_lateral_error_m"].max(),
        info["thread_axis_error_deg"].mean(),
        info["thread_axis_error_deg"].max(),
        info["thread_depth_error_m"].mean(),
        info["thread_depth_error_m"].max(),
        info["socket_angle_deg"].min(),
        info["socket_angle_deg"].mean(),
        info["socket_angle_deg"].max(),
        socket_mouth_radius.min(),
        socket_mouth_radius.mean(),
        socket_mouth_radius.max(),
        socket_mouth_xy[:, 0].min(),
        socket_mouth_xy[:, 0].max(),
        socket_mouth_xy[:, 1].min(),
        socket_mouth_xy[:, 1].max(),
    ]
    learner_names = tuple(sorted((learner_metrics or {}).keys()))
    for name in learner_names:
        value = learner_metrics[name]
        scalar_values.append(
            value.detach().reshape(())
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(
                value, device=env.device, dtype=torch.float32
            ).reshape(())
        )
    packed = torch.cat(
        (
            torch.stack(
                [value.to(torch.float32) for value in scalar_values]
            ),
            phase_histogram,
        )
    ).detach().cpu().tolist()
    scalar_count = len(scalar_names)
    learner_count = len(learner_names)
    rollout = {
        name: float(packed[index])
        for index, name in enumerate(scalar_names)
    }
    rollout["phase_histogram"] = {
        phase.name.lower(): int(round(packed[scalar_count + learner_count + phase]))
        for phase in TaskPhase
    }
    learner = {
        name: float(packed[scalar_count + index])
        for index, name in enumerate(learner_names)
    }
    return rollout, learner


def _monitor_artifacts(**paths) -> Dict[str, object]:
    return describe_paths(
        {
            name: path
            for name, path in paths.items()
            if path is not None
        }
    )


def _episode_tracker(
    config,
    *,
    scope: str,
    monitor=None,
    log_path: Optional[str | Path] = None,
) -> EpisodeTracker:
    event_path = (
        monitor.episodes_path
        if monitor is not None
        else (
            Path(log_path).expanduser().resolve().parent / "episodes.jsonl"
            if log_path is not None
            else None
        )
    )
    return EpisodeTracker(
        scope=scope,
        curriculum_warmup_fraction=(
            config.curriculum.warmup_fraction
        ),
        curriculum_full_fraction=(
            config.curriculum.full_difficulty_fraction
        ),
        event_path=event_path,
    )


def _curriculum_status(env) -> Dict[str, object]:
    spawn_radius, fixed_angle = (
        env.supervisor.curriculum_parameters(env.curriculum_progress)
    )
    return {
        "progress": float(env.curriculum_progress),
        "spawn_radius_m": float(spawn_radius),
        "fixed_socket_angle_deg": float(fixed_angle),
        "restrict_screw_spawn_to_opening_half_disk": bool(
            env.supervisor.restrict_screw_spawn_to_opening_half_disk
        ),
        "spawn_region": (
            "opening-facing half-disc"
            if env.supervisor.restrict_screw_spawn_to_opening_half_disk
            else "full workspace disc minus socket exclusion"
        ),
    }


def _commit_checkpoint_cursor(
    cursor,
    last_saved_cursor,
    *,
    write,
    validate=None,
):
    """Commit one exact continuation cursor at most once.

    The returned cursor is advanced only after validation and the complete
    checkpoint write both succeed.  This matters when SIGINT arrives while a
    periodic checkpoint is already being written: the deferred signal handler
    reaches the exception path after that write, where a second multi-GB write
    would otherwise start at the identical cursor.
    """

    normalized = tuple(cursor)
    if last_saved_cursor == normalized:
        return last_saved_cursor, False
    if validate is not None:
        validate()
    write()
    return normalized, True


class MJWarpHealthError(RuntimeError):
    """A capacity overflow or non-finite state invalidated a rollout."""


class _DeferredSIGINT:
    """Convert SIGINT into a flag checked only at consistent state cursors."""

    def __init__(self) -> None:
        self.requested = False
        self.previous = None
        self.enabled = False

    def _handler(self, _signum, _frame) -> None:
        self.requested = True

    def __enter__(self):
        try:
            self.previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handler)
            self.enabled = True
        except ValueError:
            # Signal handlers can only be installed by the main thread.
            self.enabled = False
        return self

    def checkpoint(self) -> None:
        if self.requested:
            raise KeyboardInterrupt

    def restore(self) -> None:
        if self.enabled:
            signal.signal(signal.SIGINT, self.previous)
            self.enabled = False

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        self.restore()
        if exc_type is None and self.requested:
            raise KeyboardInterrupt
        return False


def _seed_everything(seed: int, *, deterministic: bool) -> None:
    torch = _torch()
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not bool(deterministic)
        torch.backends.cudnn.deterministic = bool(deterministic)


def _capture_rng(*, sampler=None):
    torch = _torch()
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if sampler is not None:
        state["sampler"] = sampler.rng.bit_generator.state
    return _cpu_tree(state)


def _restore_rng(state, *, sampler=None) -> None:
    if not state:
        return
    torch = _torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["cuda"]]
        )
    if sampler is not None and "sampler" in state:
        sampler.rng.bit_generator.state = state["sampler"]


def _assert_backend_health(env, *, context: str) -> Dict[str, object]:
    health = dict(env.backend.health())
    # ``nefc`` is reported per-world while the capacity lives in config.
    # Include the denominator in every heartbeat so the reader can calculate
    # pressure and warn before the hard gate is reached.
    health["njmax"] = int(env.config.physics.njmax)
    if (
        not health.get("finite", False)
        or health.get("sticky_nonfinite", 0) != 0
        or health.get("sticky_external_nonfinite", 0) != 0
    ):
        raise MJWarpHealthError(
            f"{context}: non-finite physics/policy/task state: {health}"
        )
    if (
        health.get("overflow_worlds", 0) != 0
        or health.get("sticky_overflow", 0) != 0
    ):
        raise MJWarpHealthError(f"{context}: MJWarp overflow: {health}")
    if (
        health.get("contacts_total", 0)
        >= health.get("contact_capacity_total", 1)
        or health.get("sticky_contacts_full", 0) != 0
    ):
        raise MJWarpHealthError(
            f"{context}: contact capacity reached: {health}"
        )
    if (
        health.get("max_nefc", 0) >= env.config.physics.njmax
        or health.get("sticky_constraints_full", 0) != 0
    ):
        raise MJWarpHealthError(
            f"{context}: constraint capacity reached: {health}"
        )
    return health


def _assert_tqc_continuation_health(
    env,
    replay,
    *,
    context: str,
) -> Dict[str, object]:
    """Certify every mutable rollout component before an interrupt save."""

    health = _assert_backend_health(env, context=context)
    replay.assert_finite()
    return health


def _validate_checkpoint_config(payload, config, *, kind: str) -> None:
    expected = canonicalize_config_payload(config.to_dict())
    actual = payload.get("config")
    if not isinstance(actual, dict) or canonicalize_config_payload(actual) != expected:
        raise ValueError(
            f"{kind} checkpoint configuration differs from the selected "
            "configuration; use the original config or start a fresh run"
        )


def _validate_frozen_diffusion_config(payload, config) -> None:
    """Validate a frozen Diffusion dependency across vector-width changes.

    ``physics.worlds`` and the descriptive configuration name do not change
    the trained Diffusion network or task contract.  Residual training may
    therefore reuse a checkpoint at a smaller runtime vector width.  Every
    other field remains exact, while schema/task hashes and the TQC-pinned
    Diffusion SHA are validated separately by the checkpoint loaders.
    """

    actual = payload.get("config")
    if not isinstance(actual, dict):
        raise ValueError("Diffusion checkpoint has no valid configuration")
    expected = canonicalize_config_payload(config.to_dict())

    def comparable(value):
        normalized = canonicalize_config_payload(value)
        normalized.pop("name", None)
        physics = normalized.get("physics")
        if isinstance(physics, dict):
            physics.pop("worlds", None)
        return normalized

    if comparable(actual) != comparable(expected):
        raise ValueError(
            "Diffusion checkpoint configuration differs from the selected "
            "configuration beyond the allowed name/physics.worlds runtime "
            "width fields"
        )


def _validate_diffusion_dependency(
    payload,
    diffusion_checkpoint,
    *,
    config=None,
) -> str:
    validate_checkpoint(payload, kind="tqc")
    if config is not None:
        _validate_checkpoint_config(payload, config, kind="TQC")
    actual = sha256_file(diffusion_checkpoint)
    expected = payload.get("diffusion_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(
            "TQC checkpoint has no valid Diffusion SHA-256 dependency"
        )
    if expected != actual:
        raise ValueError(
            "TQC/Diffusion checkpoint mismatch: the residual policy was "
            "trained against a different Diffusion checkpoint"
        )
    return actual


def generate_demos(
    config,
    *,
    output: str | Path,
    resume: bool = True,
    monitor=None,
) -> Path:
    torch = _torch()
    target = Path(output).expanduser().resolve()
    episodes = config.demos.episodes
    steps = config.physics.max_episode_steps
    target_has_files = target.is_dir() and any(target.iterdir())
    if not resume and target_has_files:
        raise FileExistsError(
            f"demo output is not empty at {target}; use --resume or a fresh "
            "run directory"
        )
    if (
        resume
        and target_has_files
        and not (target / "metadata.json").is_file()
    ):
        raise FileExistsError(
            f"demo output contains unrecognized partial files at {target}; "
            "move it aside and start a fresh run"
        )
    provenance = {
        "backend": BACKEND,
        "task_geometry_hash": TASK_GEOMETRY_HASH,
        "demo_seed": int(config.demos.seed),
        "policy_start_step": SCRIPTED_PREFIX_STEPS,
        "config": config.to_dict(),
    }
    store = DemoStore(
        target,
        episodes=episodes,
        steps=steps,
        create=True,
        provenance=provenance,
    )
    start = store.completed
    tracker = _episode_tracker(
        config,
        scope="expert_demonstrations_this_session",
        monitor=monitor,
    )
    demo_artifacts = _monitor_artifacts(
        demo_metadata=target / "metadata.json",
    )
    if monitor is not None:
        monitor.set_stage(
            "demonstrations",
            current=start * steps,
            total=episodes * steps,
            unit="episode_step",
            message=(
                "Building the expert MJWarp CUDA graph, then generating "
                f"demonstrations ({start}/{episodes} episodes committed)."
            ),
            metrics={
                "episodes": tracker.snapshot(),
                "demos": {
                    "committed_episodes": start,
                    "target_episodes": episodes,
                    "steps_per_episode": steps,
                },
            },
            artifacts=demo_artifacts,
            stale_after_seconds=900.0,
        )
    if start >= episodes:
        store.finalize()
        if monitor is not None:
            monitor.heartbeat(
                current=episodes * steps,
                message="Expert demonstration dataset is complete.",
                metrics={
                    "episodes": tracker.snapshot(),
                    "demos": {
                        "committed_episodes": episodes,
                        "target_episodes": episodes,
                        "steps_per_episode": steps,
                    },
                },
                artifacts=_monitor_artifacts(
                    demo_metadata=target / "metadata.json",
                ),
                force=True,
            )
        return target
    _seed_everything(
        config.demos.seed, deterministic=config.runtime.deterministic
    )
    demo_config = replace(
        config,
        runtime=replace(config.runtime, seed=config.demos.seed),
    )
    env = None
    zero_residual = None
    interrupt = _DeferredSIGINT()
    interrupt.__enter__()
    try:
        while start < episodes:
            simulation_worlds = min(
                config.physics.worlds, episodes - start
            )
            if env is None or env.worlds != simulation_worlds:
                if env is not None:
                    env.close()
                    env = None
                    zero_residual = None
                    # Return the superseded full-size world and trajectory
                    # allocations before constructing a smaller final batch.
                    # Otherwise an 8000 + 192 demo split can briefly retain
                    # two MJWarp worlds and two 9+ GiB observation buffers.
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                env = MJWarpScrewEnv(
                    demo_config,
                    worlds=simulation_worlds,
                    baseline="expert",
                )
                zero_residual = torch.zeros(
                    (simulation_worlds, RESIDUAL_ACTION_DIM),
                    device=env.device,
                )
            # Each vector batch owns a deterministic seed derived from its
            # first episode index.  A resumed generator therefore produces
            # exactly the same next batch as an uninterrupted run instead of
            # repeating batch zero.
            env.generator.manual_seed(config.demos.seed + start)
            initial = env.reset()
            # The on-disk demonstration contract is float16.  Keeping 3001
            # float32 observations and 3000 float32 actions in Python lists
            # and then stacking them briefly doubled the trajectory footprint
            # (about 18.8 GiB at 4096 worlds).  Fill the final-layout float16
            # buffers directly instead; conversion happens at the same point
            # in the data path, without the duplicate stack allocation.
            observation_tensor = torch.empty(
                (
                    simulation_worlds,
                    steps + 1,
                    OBS_DIM,
                ),
                device=env.device,
                dtype=torch.float16,
            )
            action_tensor = torch.empty(
                (
                    simulation_worlds,
                    steps,
                    ACTION_DIM,
                ),
                device=env.device,
                dtype=torch.float16,
            )
            observation_tensor[:, 0].copy_(initial)
            rollout_audit = ExpertRolloutAudit(
                torch,
                worlds=simulation_worlds,
                device=env.device,
            )
            for step_index in range(steps):
                result = env.step(zero_residual)
                observation_tensor[:, step_index + 1].copy_(
                    result.observation
                )
                action_tensor[:, step_index].copy_(result.expert_action)
                rollout_audit.update(
                    result, step_number=step_index + 1
                )
                periodic_health = (
                    (step_index + 1)
                    % config.runtime.log_every_vector_steps
                    == 0
                    or step_index + 1 == steps
                )
                monitor_due = monitor is not None and monitor.due
                health = None
                if periodic_health or monitor_due:
                    health = _assert_backend_health(
                        env,
                        context=(
                            f"demonstrations {start}/{episodes}, "
                            f"step {step_index + 1}"
                        ),
                    )
                    rollout_audit.assert_clean(
                        context=(
                            "expert demonstration at control step "
                            f"{step_index + 1}"
                        ),
                    )
                if monitor_due:
                    rollout, _unused = _materialize_rollout_metrics(
                        env,
                        result,
                        action=result.expert_action,
                    )
                    take = min(simulation_worlds, episodes - start)
                    monitor.heartbeat(
                        current=start * steps + (step_index + 1) * take,
                        message=(
                            f"Expert rollout batch at step "
                            f"{step_index + 1}/{steps}"
                        ),
                        metrics={
                            "episodes": tracker.snapshot(),
                            "demos": {
                                "committed_episodes": start,
                                "target_episodes": episodes,
                                "active_worlds": take,
                                "steps_per_episode": steps,
                                "vector_step": step_index + 1,
                                "inflight_batch_start": start,
                                "inflight_batch_episodes": take,
                                "inflight_is_committed": False,
                            },
                            "rollout": rollout,
                        },
                        health=health,
                    )
                _safe_cursor_interrupt(interrupt, monitor)
            certify_expert_rollout(
                env,
                result,
                rollout_audit,
                context="expert demonstration",
                require_terminal=True,
            )
            take = min(simulation_worlds, episodes - start)
            batch_start = start
            for offset in range(
                0, take, config.demos.shard_episodes
            ):
                count = min(
                    config.demos.shard_episodes, take - offset
                )
                selection = slice(offset, offset + count)
                store.write(
                    batch_start + offset,
                    observation_tensor[selection].cpu().numpy(),
                    action_tensor[selection].cpu().numpy(),
                    env.state.spawn_xy[selection].cpu().numpy(),
                    socket_mouth_xy=env.state.socket_mouth[
                        selection, :2
                    ].cpu().numpy(),
                    socket_angle_deg=env.state.socket_angle_deg[
                        selection
                    ].cpu().numpy(),
                    commit=False,
                )
            store.commit(batch_start, batch_start + take)
            start = batch_start + take
            done_ids = torch.arange(take, device=env.device)
            _track_terminal_episodes(
                tracker,
                result.info,
                done_ids,
                env=env,
                monitor=monitor,
                cursor={
                    "demo_episodes_committed": start,
                    "demo_batch_start": batch_start,
                },
            )
            health = _assert_backend_health(
                env,
                context=f"demonstration commit {start}/{episodes}",
            )
            rollout, _unused = _materialize_rollout_metrics(
                env,
                result,
                action=result.expert_action,
            )
            if monitor is not None:
                monitor.heartbeat(
                    current=start * steps,
                    message=(
                        f"Committed {start}/{episodes} expert "
                        "demonstrations"
                    ),
                    metrics={
                        "episodes": tracker.snapshot(),
                        "demos": {
                            "committed_episodes": start,
                            "target_episodes": episodes,
                            "active_worlds": take,
                            "steps_per_episode": steps,
                            "vector_step": steps,
                            "inflight_batch_start": batch_start,
                            "inflight_batch_episodes": take,
                            "inflight_is_committed": True,
                        },
                        "rollout": rollout,
                    },
                    health=health,
                    artifacts=_monitor_artifacts(
                        demo_metadata=target / "metadata.json",
                    ),
                    force=True,
                )
            _safe_cursor_interrupt(interrupt, monitor)
            print(
                json.dumps(
                    {
                        "event": "demo_batch",
                        "completed": start,
                        "episodes": episodes,
                        "simulation_worlds": simulation_worlds,
                        "disk_shard_episodes": config.demos.shard_episodes,
                        "steps": steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            # No next allocation may overlap the just-committed trajectory
            # buffers.  This is especially important when the final batch
            # has fewer worlds than the configured vector width.
            del (
                observation_tensor,
                action_tensor,
                rollout_audit,
                result,
                initial,
                done_ids,
            )
    finally:
        interrupt.restore()
        if env is not None:
            env.close()
    return target


def train_diffusion(
    config,
    *,
    dataset: str | Path,
    output: str | Path,
    log_path: str | Path,
    resume: bool = True,
    monitor=None,
) -> Path:
    torch = _torch()
    device = torch.device(config.runtime.device)
    _seed_everything(
        config.runtime.seed, deterministic=config.runtime.deterministic
    )
    c = config.diffusion
    target = Path(output).expanduser().resolve()
    if not resume and target.exists():
        raise FileExistsError(
            f"Diffusion checkpoint already exists at {target}; use "
            "--resume or a fresh run directory"
        )
    dataset = Path(dataset).expanduser().resolve()
    if monitor is not None:
        monitor.set_stage(
            "diffusion",
            current=0,
            total=c.epochs,
            unit="epoch",
            message="Hashing and indexing the demonstration dataset.",
            artifacts=_monitor_artifacts(
                diffusion_checkpoint=target,
                diffusion_log=log_path,
                demo_metadata=dataset / "metadata.json",
            ),
            stale_after_seconds=900.0,
        )
    dataset_metadata_sha256 = sha256_file(
        dataset / "metadata.json"
    )
    sampler = DemoWindowSampler(dataset, seed=config.runtime.seed)
    # Retain the full, JSON-safe source-data lineage in every new checkpoint.
    # This is especially important for successful-policy self-distillation
    # corpora whose full-disc / radial-extension distribution may be OOD with
    # respect to the checkpoint configuration used to collect them.  The
    # existing metadata SHA gate below remains the authority for resume.
    dataset_provenance = json.loads(
        json.dumps(
            sampler.metadata.get("provenance"),
            sort_keys=True,
            allow_nan=False,
        )
    )
    windows = sampler.window_count(config.demos.action_horizon)
    batches_per_epoch = max(1, math.ceil(windows / c.batch_size))
    total_batches = c.epochs * batches_per_epoch
    resume_checkpoint = None
    if resume and target.exists():
        resume_checkpoint = _load(target)
        validate_checkpoint(resume_checkpoint, kind="diffusion")
        _validate_checkpoint_config(
            resume_checkpoint, config, kind="Diffusion"
        )
        expected_dataset = resume_checkpoint.get(
            "dataset_metadata_sha256"
        )
        if expected_dataset != dataset_metadata_sha256:
            raise ValueError(
                "Diffusion checkpoint/demo metadata mismatch; resume needs "
                "the exact dataset configuration and completed cursor"
            )
        checkpoint_epoch = int(resume_checkpoint.get("epoch", -1))
        checkpoint_batch = int(
            resume_checkpoint.get("batch_in_epoch", -1)
        )
        if checkpoint_epoch > c.epochs:
            raise ValueError(
                "Diffusion checkpoint is ahead of configured epochs"
            )
        if checkpoint_epoch == c.epochs:
            if checkpoint_batch != 0:
                raise ValueError(
                    "completed Diffusion checkpoint has a nonzero batch cursor"
                )
            if monitor is not None:
                monitor.set_stage(
                    "diffusion",
                    current=total_batches,
                    total=total_batches,
                    unit="minibatch",
                    message="Diffusion training checkpoint is complete.",
                    metrics={
                        "diffusion": {
                            "epoch": checkpoint_epoch,
                            "epochs": c.epochs,
                            "batch_in_epoch": 0,
                            "batches_per_epoch": batches_per_epoch,
                            "global_step": int(
                                resume_checkpoint.get("global_step", 0)
                            ),
                            "best_loss": float(
                                resume_checkpoint.get(
                                    "best_loss", math.inf
                                )
                            ),
                            "windows": windows,
                        },
                    },
                    artifacts=_monitor_artifacts(
                        diffusion_checkpoint=target,
                        diffusion_log=log_path,
                        demo_metadata=dataset / "metadata.json",
                    ),
                )
            return target
    policy = DiffusionPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        obs_horizon=OBS_HORIZON,
        horizon=config.demos.action_horizon,
        num_diffusion_steps=c.diffusion_steps,
        hidden_dim=c.hidden_dim,
        depth=c.depth,
    ).to(device)
    if resume_checkpoint is None:
        policy.set_normalization(sampler.normalization())
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=c.learning_rate, weight_decay=1.0e-6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(c.epochs, 1), eta_min=c.learning_rate * 0.05
    )
    ema = ExponentialMovingAverage(policy, decay=c.ema_decay)
    start_epoch = 0
    global_step = 0
    best_loss = math.inf
    resume_batch = 0
    resume_loss_sum = 0.0
    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        policy.load_state_dict(checkpoint["model_state"])
        ema.load_state_dict(checkpoint["ema_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint.get("best_loss", math.inf))
        resume_batch = int(checkpoint.get("batch_in_epoch", 0))
        resume_loss_sum = float(
            checkpoint.get("partial_loss_sum", 0.0)
        )
        _restore_rng(checkpoint.get("rng_state"), sampler=sampler)
    if device.type == "cuda":
        if monitor is not None:
            monitor.heartbeat(
                message=(
                    "Caching demonstration windows on CUDA; this may take "
                    "several minutes."
                ),
                force=True,
                stale_after_seconds=900.0,
            )
        sampler.cache_on_device(str(device))
    diffusion_bfloat16 = (
        device.type == "cuda"
        and torch.cuda.get_device_properties(device).major >= 8
    )

    if not 0 <= resume_batch <= batches_per_epoch:
        raise ValueError(
            "Diffusion checkpoint batch_in_epoch is incompatible with "
            "the current dataset/batch size"
        )
    completed_epoch = start_epoch
    batch_in_epoch = resume_batch
    loss_sum = torch.as_tensor(
        resume_loss_sum, device=device, dtype=torch.float32
    )
    gradient_norm = torch.zeros((), device=device)
    last_diffusion_status = {
        "epoch": (
            completed_epoch + 1
            if completed_epoch < c.epochs
            else completed_epoch
        ),
        "completed_epochs": completed_epoch,
        "epochs": c.epochs,
        "batch_in_epoch": batch_in_epoch,
        "batches_per_epoch": batches_per_epoch,
        "global_step": global_step,
        "loss": None,
        "running_epoch_loss": (
            resume_loss_sum / resume_batch if resume_batch else None
        ),
        "epoch_mean_loss": (
            resume_loss_sum / resume_batch if resume_batch else None
        ),
        "best_loss": best_loss,
        "gradient_norm": None,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "windows": windows,
        "batch_size": c.batch_size,
    }
    if monitor is not None:
        monitor.set_stage(
            "diffusion",
            current=start_epoch * batches_per_epoch + resume_batch,
            total=total_batches,
            unit="minibatch",
            message=(
                f"Training Diffusion epoch {start_epoch + 1}/{c.epochs}"
                if start_epoch < c.epochs
                else "Diffusion training is complete."
            ),
            metrics={"diffusion": last_diffusion_status},
            artifacts=_monitor_artifacts(
                diffusion_checkpoint=target,
                diffusion_log=log_path,
                demo_metadata=dataset / "metadata.json",
            ),
        )

    last_saved_checkpoint_cursor = None

    def checkpoint_cursor():
        return (
            completed_epoch,
            batch_in_epoch,
            global_step,
            int(scheduler.last_epoch),
        )

    def assert_finite_checkpoint() -> None:
        for group_name, tensors in (
            ("Diffusion model", policy.state_dict()),
            ("Diffusion EMA", ema.state_dict(cpu=False)),
        ):
            for name, value in tensors.items():
                if torch.is_floating_point(value) and not bool(
                    torch.isfinite(value).all()
                ):
                    raise FloatingPointError(
                        f"{group_name} tensor {name!r} is non-finite"
                    )
        if not bool(torch.isfinite(loss_sum)):
            raise FloatingPointError(
                "Diffusion accumulated epoch loss is non-finite"
            )

    def save_checkpoint(*, interrupted: bool = False) -> None:
        assert_finite_checkpoint()
        payload = checkpoint_metadata("diffusion", config.to_dict())
        payload.update(
            {
                "model_config": policy.model_config(),
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in policy.state_dict().items()
                },
                "ema_state": ema.state_dict(cpu=True),
                "normalization": policy.normalization_dict(),
                "optimizer_state": _cpu_tree(optimizer.state_dict()),
                "scheduler_state": _cpu_tree(scheduler.state_dict()),
                "epoch": completed_epoch,
                "batch_in_epoch": batch_in_epoch,
                "partial_loss_sum": float(
                    loss_sum.detach().to(torch.float64).cpu()
                ),
                "global_step": global_step,
                "best_loss": best_loss,
                "dataset": dataset.name,
                "dataset_metadata_sha256": dataset_metadata_sha256,
                "dataset_contract": {
                    "format": sampler.metadata.get("format"),
                    "episodes": sampler.episodes,
                    "steps": sampler.steps,
                    "policy_start_step": sampler.policy_start_step,
                    "action_horizon": config.demos.action_horizon,
                    "windows": windows,
                    "variable_episode_lengths": bool(
                        sampler.metadata.get(
                            "variable_episode_lengths", False
                        )
                    ),
                    "provenance": dataset_provenance,
                },
                "rng_state": _capture_rng(sampler=sampler),
                "interrupted": interrupted,
            }
        )
        atomic_torch_save(payload, target)

    interrupt = _DeferredSIGINT()
    interrupt.__enter__()
    try:
        for epoch in range(start_epoch, c.epochs):
            policy.train()
            if epoch == start_epoch:
                batch_start = resume_batch
                loss_sum = torch.as_tensor(
                    resume_loss_sum, device=device, dtype=torch.float32
                )
            else:
                batch_start = 0
                loss_sum = torch.zeros((), device=device)
            batch_in_epoch = batch_start
            started = time.monotonic()
            for batch_index in range(batch_start, batches_per_epoch):
                observation, action = sampler.sample(
                    c.batch_size,
                    config.demos.action_horizon,
                    device=str(device),
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    # CX3's RTX6000 is Turing (compute capability 7.5).
                    # Its formal fallback is float32; unscaled float16 would
                    # risk gradient underflow.  A100 and L40S use bfloat16.
                    enabled=diffusion_bfloat16,
                ):
                    loss = policy.compute_loss(observation, action)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        "Diffusion loss became NaN or infinity"
                    )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), 1.0
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError(
                        "Diffusion gradient norm became NaN or infinity"
                    )
                optimizer.step()
                ema.update(policy)
                global_step += 1
                loss_sum += loss.detach()
                batch_in_epoch = batch_index + 1
                if monitor is not None and monitor.due:
                    materialized = torch.stack(
                        (
                            loss.detach().to(torch.float32),
                            (
                                loss_sum
                                / max(batch_in_epoch, 1)
                            ).to(torch.float32),
                            gradient_norm.detach().to(torch.float32),
                        )
                    ).cpu().tolist()
                    last_diffusion_status = {
                        "epoch": epoch + 1,
                        "completed_epochs": epoch,
                        "epochs": c.epochs,
                        "batch_in_epoch": batch_in_epoch,
                        "batches_per_epoch": batches_per_epoch,
                        "global_step": global_step,
                        "loss": float(materialized[0]),
                        "running_epoch_loss": float(materialized[1]),
                        "epoch_mean_loss": float(materialized[1]),
                        "best_loss": best_loss,
                        "gradient_norm": float(materialized[2]),
                        "learning_rate": float(
                            optimizer.param_groups[0]["lr"]
                        ),
                        "windows": windows,
                        "batch_size": c.batch_size,
                    }
                    monitor.heartbeat(
                        current=epoch * batches_per_epoch + batch_in_epoch,
                        message=(
                            f"Diffusion epoch {epoch + 1}/{c.epochs}, "
                            f"batch {batch_in_epoch}/{batches_per_epoch}"
                        ),
                        metrics={"diffusion": last_diffusion_status},
                    )
                _safe_cursor_interrupt(interrupt, monitor)
            # On SIGINT the checkpoint may be taken immediately after the
            # scheduler advanced but before the epoch cursor was committed.
            # ``last_epoch`` disambiguates that narrow boundary on resume.
            scheduler_epoch = int(scheduler.last_epoch)
            if scheduler_epoch == completed_epoch:
                scheduler.step()
            elif not (
                batch_in_epoch == batches_per_epoch
                and scheduler_epoch == completed_epoch + 1
            ):
                raise ValueError(
                    "Diffusion scheduler/epoch checkpoint cursor mismatch"
                )
            epoch_loss = float((loss_sum / batches_per_epoch).cpu())
            best_loss = min(best_loss, epoch_loss)
            completed_epoch = epoch + 1
            batch_in_epoch = 0
            loss_sum = torch.zeros((), device=device)
            resume_batch = 0
            resume_loss_sum = 0.0
            if monitor is not None:
                monitor.heartbeat(
                    current=completed_epoch * batches_per_epoch,
                    message=(
                        "Saving the full Diffusion training checkpoint; "
                        "this may take several minutes."
                    ),
                    metrics={"diffusion": last_diffusion_status},
                    force=True,
                    stale_after_seconds=900.0,
                )
            last_saved_checkpoint_cursor, _saved = (
                _commit_checkpoint_cursor(
                    checkpoint_cursor(),
                    last_saved_checkpoint_cursor,
                    write=save_checkpoint,
                )
            )
            event = {
                "event": "diffusion_epoch",
                "epoch": completed_epoch,
                "completed_epochs": completed_epoch,
                "global_step": global_step,
                "loss": epoch_loss,
                "best_loss": best_loss,
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "seconds": time.monotonic() - started,
            }
            append_jsonl(log_path, event)
            print(json.dumps(event, sort_keys=True), flush=True)
            last_diffusion_status = {
                "epoch": completed_epoch,
                "completed_epochs": completed_epoch,
                "epochs": c.epochs,
                "batch_in_epoch": 0,
                "batches_per_epoch": batches_per_epoch,
                "global_step": global_step,
                "loss": epoch_loss,
                "running_epoch_loss": epoch_loss,
                "epoch_mean_loss": epoch_loss,
                "best_loss": best_loss,
                "gradient_norm": event["gradient_norm"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "windows": windows,
                "batch_size": c.batch_size,
                "epoch_seconds": event["seconds"],
            }
            if monitor is not None:
                monitor.heartbeat(
                    current=completed_epoch * batches_per_epoch,
                    message=(
                        f"Completed Diffusion epoch "
                        f"{completed_epoch}/{c.epochs}"
                    ),
                    metrics={"diffusion": last_diffusion_status},
                    artifacts=_monitor_artifacts(
                        diffusion_checkpoint=target,
                        diffusion_log=log_path,
                        demo_metadata=dataset / "metadata.json",
                    ),
                    force=True,
                )
            _safe_cursor_interrupt(interrupt, monitor)
    except KeyboardInterrupt:
        # The PBS timeout guard sends SIGINT before walltime.  The batch cursor,
        # accumulated loss, sampler state, and all RNG streams make the next
        # invocation continue at the next not-yet-consumed mini-batch.
        if monitor is not None:
            monitor.heartbeat(
                current=completed_epoch * batches_per_epoch + batch_in_epoch,
                message=(
                    "Saving interrupted Diffusion continuation checkpoint; "
                    "this may take several minutes."
                ),
                metrics={"diffusion": last_diffusion_status},
                force=True,
                stale_after_seconds=900.0,
            )
        last_saved_checkpoint_cursor, wrote_interrupt_checkpoint = (
            _commit_checkpoint_cursor(
                checkpoint_cursor(),
                last_saved_checkpoint_cursor,
                write=lambda: save_checkpoint(interrupted=True),
            )
        )
        event = {
            "event": "diffusion_interrupted",
            "completed_epoch": completed_epoch,
            "batch_in_epoch": batch_in_epoch,
            "batches_per_epoch": batches_per_epoch,
            "global_step": global_step,
            "checkpoint": str(target),
            "checkpoint_written_for_interrupt": wrote_interrupt_checkpoint,
            "checkpoint_reused_at_same_cursor": (
                not wrote_interrupt_checkpoint
            ),
        }
        append_jsonl(log_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        if monitor is not None:
            monitor.heartbeat(
                current=completed_epoch * batches_per_epoch + batch_in_epoch,
                message="Diffusion interrupted after safe checkpoint.",
                metrics={"diffusion": last_diffusion_status},
                artifacts=_monitor_artifacts(
                    diffusion_checkpoint=target,
                    diffusion_log=log_path,
                    demo_metadata=dataset / "metadata.json",
                ),
                force=True,
            )
        raise
    finally:
        interrupt.restore()
    return target


def train_residual(
    config,
    *,
    diffusion_checkpoint: str | Path,
    output: str | Path,
    log_path: str | Path,
    resume: bool = True,
    monitor=None,
) -> Path:
    # Keep every residual-only dependency off the Diffusion-only import and
    # execution path.  The legacy entry point remains available explicitly.
    from .replay import CUDAReplayBuffer
    from .tqc import TQCTrainer

    torch = _torch()
    output = Path(output).expanduser().resolve()
    policy_output = output.with_name(
        f"{output.stem}_policy{output.suffix}"
    )
    diffusion_checkpoint = Path(
        diffusion_checkpoint
    ).expanduser().resolve()
    if not resume and (output.exists() or policy_output.exists()):
        raise FileExistsError(
            f"TQC checkpoint already exists at {output}; use --resume "
            "or a fresh run directory"
        )
    if monitor is not None:
        monitor.set_stage(
            "residual_tqc",
            current=0,
            total=config.tqc.total_transitions,
            unit="transition",
            message=(
                "Hashing/loading checkpoints before MJWarp CUDA graph "
                "construction."
            ),
            artifacts=_monitor_artifacts(
                diffusion_checkpoint=diffusion_checkpoint,
                tqc_full_checkpoint=output,
                tqc_policy_checkpoint=policy_output,
                tqc_log=log_path,
            ),
            stale_after_seconds=900.0,
        )
    _seed_everything(
        config.runtime.seed, deterministic=config.runtime.deterministic
    )
    diffusion_sha256 = sha256_file(diffusion_checkpoint)
    resume_payload = None
    if resume and output.exists():
        resume_payload = _load(output)
        _validate_diffusion_dependency(
            resume_payload,
            diffusion_checkpoint,
            config=config,
        )
    tracker = _episode_tracker(
        config,
        scope="stochastic_training_rollouts",
        monitor=monitor,
        log_path=log_path,
    )
    if resume_payload is not None:
        monitor_state = resume_payload.get("training_monitor_state")
        if monitor_state is not None:
            tracker.load_state_dict(monitor_state)
    residual_artifacts = _monitor_artifacts(
        diffusion_checkpoint=diffusion_checkpoint,
        tqc_full_checkpoint=output,
        tqc_policy_checkpoint=policy_output,
        tqc_log=log_path,
    )
    if monitor is not None:
        monitor.set_stage(
            "residual_tqc",
            current=(
                int(resume_payload.get("transitions", 0))
                if resume_payload is not None
                else 0
            ),
            total=config.tqc.total_transitions,
            unit="transition",
            message=(
                "Loading Diffusion and building the MJWarp CUDA graph; "
                "the first heartbeat can take several minutes."
            ),
            metrics={
                "episodes": tracker.snapshot(),
                "tqc": {
                    "transitions": (
                        int(resume_payload.get("transitions", 0))
                        if resume_payload is not None
                        else 0
                    ),
                    "total_transitions": config.tqc.total_transitions,
                    "updates": (
                        int(resume_payload.get("updates", 0))
                        if resume_payload is not None
                        else 0
                    ),
                },
            },
            artifacts=residual_artifacts,
            stale_after_seconds=900.0,
        )
    controller = BatchedDiffusionController.from_checkpoint(
        diffusion_checkpoint,
        worlds=config.physics.worlds,
        inference_steps=config.diffusion.inference_steps,
        replan_interval=config.diffusion.replan_interval,
        device=config.runtime.device,
        seed=config.runtime.seed,
    )
    env = MJWarpScrewEnv(
        config,
        baseline="diffusion",
        diffusion_controller=controller,
        curriculum_progress=(
            int(resume_payload.get("transitions", 0))
            / max(config.tqc.total_transitions, 1)
            if resume_payload is not None
            else 0.0
        ),
    )
    replay = CUDAReplayBuffer(
        capacity=config.tqc.replay_capacity,
        obs_dim=TQC_OBS_DIM,
        action_dim=RESIDUAL_ACTION_DIM,
        device=config.runtime.device,
    )
    trainer = TQCTrainer(config, device=config.runtime.device)
    pending_replay = None
    if resume_payload is not None:
        trainer.load(resume_payload, replay=replay)
        env_state = resume_payload.get("env_state")
        if env_state is None:
            raise ValueError(
                "TQC checkpoint predates rollout-state continuation; "
                "start a fresh hpc run"
            )
        env.load_state_dict(env_state)
        pending_replay = _restore_pending_replay(
            resume_payload.get("pending_replay"), device=env.device
        )
        _restore_rng(resume_payload.get("rng_state"))
        del env_state
        resume_payload = None
        gc.collect()
    next_checkpoint = (
        (trainer.transitions // config.tqc.checkpoint_every_transitions) + 1
    ) * config.tqc.checkpoint_every_transitions
    last_metrics: Dict[str, object] = {}
    last_monitor_metrics: Dict[str, object] = {
        "episodes": tracker.snapshot(),
        "curriculum": _curriculum_status(env),
        "tqc": {
            "transitions": trainer.transitions,
            "total_transitions": config.tqc.total_transitions,
            "updates": trainer.updates,
            "update_credit": trainer.update_credit,
            "replay_size": replay.size,
            "replay_capacity": replay.capacity,
            "replay_fraction": replay.size / max(replay.capacity, 1),
            "replay_fill_fraction": replay.size / max(replay.capacity, 1),
            "learning_starts": config.tqc.learning_starts,
            "learning_started": (
                trainer.transitions >= config.tqc.learning_starts
            ),
            "mode": (
                "learning"
                if trainer.transitions >= config.tqc.learning_starts
                else "warmup"
            ),
            "vector_steps_this_session": 0,
        },
    }
    vector_steps = 0
    if monitor is not None:
        monitor.heartbeat(
            current=trainer.transitions,
            message=(
                "MJWarp environment ready; residual TQC rollout is running."
            ),
            metrics=last_monitor_metrics,
            health=_assert_backend_health(
                env, context="TQC environment initialization"
            ),
            artifacts=residual_artifacts,
            force=True,
        )

    def save_policy_checkpoint() -> None:
        policy_payload = checkpoint_metadata("tqc", config.to_dict())
        policy_payload.update(
            {
                "policy_only": True,
                "actor": _cpu_tree(trainer.actor.state_dict()),
                "updates": trainer.updates,
                "transitions": trainer.transitions,
                "diffusion_checkpoint": diffusion_checkpoint.name,
                "diffusion_sha256": diffusion_sha256,
            }
        )
        atomic_torch_save(policy_payload, policy_output)

    def save_checkpoint(*, interrupted: bool = False) -> None:
        trainer.assert_finite()
        _restore_pending_replay(pending_replay, device=env.device)
        payload = trainer.checkpoint(replay, include_replay=True)
        payload.update(
            {
                "diffusion_checkpoint": diffusion_checkpoint.name,
                "diffusion_sha256": diffusion_sha256,
                "env_state": _cpu_tree(env.state_dict()),
                "pending_replay": _cpu_tree(pending_replay),
                "rng_state": _capture_rng(),
                "training_monitor_state": tracker.state_dict(),
                "interrupted": interrupted,
            }
        )
        atomic_torch_save(payload, output)
        save_policy_checkpoint()

    last_saved_checkpoint_cursor = None

    def checkpoint_cursor():
        return (
            trainer.transitions,
            trainer.updates,
            replay.position,
            replay.size,
            vector_steps,
            (
                0
                if pending_replay is None
                else int(pending_replay["world_ids"].numel())
            ),
        )

    interrupt = _DeferredSIGINT()
    interrupt.__enter__()
    try:
        while trainer.transitions < config.tqc.total_transitions:
            env.set_curriculum_progress(
                trainer.transitions
                / max(config.tqc.total_transitions, 1)
            )
            inserted_this_step = 0

            def insert_pending(next_observation, next_action_mask, rows=None):
                """Commit selected delayed transitions with an exact next state."""

                nonlocal pending_replay, inserted_this_step
                if pending_replay is None:
                    return
                count = int(pending_replay["world_ids"].numel())
                if rows is None:
                    rows = torch.ones(
                        count, device=env.device, dtype=torch.bool
                    )
                else:
                    rows = torch.as_tensor(
                        rows, device=env.device, dtype=torch.bool
                    )
                if tuple(rows.shape) != (count,):
                    raise ValueError("pending replay selector shape is invalid")
                selected_rows = torch.nonzero(
                    rows, as_tuple=False
                ).flatten()
                remaining_budget = max(
                    0,
                    int(config.tqc.total_transitions)
                    - trainer.transitions
                    - inserted_this_step,
                )
                if selected_rows.numel() > remaining_budget:
                    selected_rows = selected_rows[:remaining_budget]
                if selected_rows.numel():
                    world_ids = pending_replay["world_ids"][selected_rows]
                    replay.add(
                        pending_replay["observation"][selected_rows],
                        pending_replay["action"][selected_rows],
                        pending_replay["reward"][selected_rows],
                        next_observation[world_ids],
                        pending_replay["terminated"][selected_rows],
                        pending_replay["truncated"][selected_rows],
                        action_mask=pending_replay["action_mask"][
                            selected_rows
                        ],
                        next_action_mask=next_action_mask[world_ids],
                    )
                    inserted_this_step += int(selected_rows.numel())
                # Selected rows belong to this cursor even if the final budget
                # clipped their replay insertion; never carry them twice.
                keep = ~rows
                if bool(keep.any()):
                    pending_replay = {
                        name: value[keep]
                        for name, value in pending_replay.items()
                    }
                else:
                    pending_replay = None

            def residual_policy(tqc_state, action_mask):
                # The callback is invoked after Diffusion and nominal IK have
                # prepared the exact state for this tick.  It therefore also
                # supplies the exact next state for the preceding nonterminal
                # delayed transition.
                if pending_replay is not None:
                    insert_pending(tqc_state, action_mask)
                stored_cursor = trainer.transitions + inserted_this_step
                if stored_cursor < config.tqc.learning_starts:
                    warmup = 0.05 * (
                        2.0
                        * torch.rand(
                            (env.worlds, RESIDUAL_ACTION_DIM),
                            device=env.device,
                        )
                        - 1.0
                    )
                    return warmup * action_mask.to(warmup.dtype)
                return trainer.act(
                    tqc_state, action_mask=action_mask
                )

            result = env.step(residual_policy=residual_policy)
            vector_steps += 1
            periodic_health = (
                vector_steps % config.runtime.log_every_vector_steps
                == 0
            )
            monitor_due = monitor is not None and monitor.due
            health = None
            if periodic_health or monitor_due:
                health = _assert_backend_health(
                    env,
                    context=(
                        f"TQC transition {trainer.transitions}"
                    ),
                )
            executed_action = result.info["executed_residual_action"]
            learning_ids = torch.nonzero(
                result.info["diffusion_active"], as_tuple=False
            ).flatten()
            remaining_budget = max(
                0,
                int(config.tqc.total_transitions)
                - trainer.transitions
                - inserted_this_step,
            )
            insert_limit = min(
                int(config.tqc.replay_insert_per_vector_step),
                int(learning_ids.numel()),
                remaining_budget,
            )
            if insert_limit:
                order = torch.randperm(
                    int(learning_ids.numel()), device=env.device
                )[:insert_limit]
                selected = learning_ids[order]
                terminated = (
                    result.info["success"][selected]
                    | result.info["failure"][selected]
                )
                truncated = result.info["truncated"][selected]
                pending_replay = {
                    "world_ids": selected.clone(),
                    "observation": result.info[
                        "tqc_observation"
                    ][selected].clone(),
                    "action": executed_action[selected].clone(),
                    "reward": result.reward[selected].clone(),
                    "action_mask": result.info[
                        "residual_action_mask"
                    ][selected].clone(),
                    "terminated": terminated.clone(),
                    "truncated": truncated.clone(),
                }
                terminal_rows = terminated | truncated
                if bool(terminal_rows.any()):
                    insert_pending(
                        _terminal_tqc_observation(result),
                        result.info["residual_action_mask"],
                        rows=terminal_rows,
                    )
            update_count = trainer.updates_for(inserted_this_step)
            if replay.size >= config.tqc.batch_size:
                for _ in range(update_count):
                    last_metrics = trainer.update(
                        replay.sample(config.tqc.batch_size)
                    )
            done_any = bool(result.done.any())
            if done_any:
                done_ids = torch.nonzero(
                    result.done, as_tuple=False
                ).flatten()
                _track_terminal_episodes(
                    tracker,
                    result.info,
                    done_ids,
                    env=env,
                    monitor=monitor,
                    cursor={
                        "transitions": trainer.transitions,
                        "updates": trainer.updates,
                        "vector_step": vector_steps,
                    },
                )
            checkpoint_due = (
                trainer.transitions >= next_checkpoint
                or trainer.transitions >= config.tqc.total_transitions
            )
            emit_monitor = monitor is not None and (
                monitor.due or checkpoint_due
            )
            if emit_monitor:
                if health is None:
                    health = _assert_backend_health(
                        env,
                        context=(
                            f"TQC monitor transition {trainer.transitions}"
                        ),
                    )
                rollout_status, learner_status = (
                    _materialize_rollout_metrics(
                        env,
                        result,
                        action=executed_action,
                        learner_metrics=last_metrics,
                    )
                )
                last_monitor_metrics = {
                    "episodes": tracker.snapshot(),
                    "curriculum": _curriculum_status(env),
                    "tqc": {
                        "transitions": trainer.transitions,
                        "total_transitions": config.tqc.total_transitions,
                        "updates": trainer.updates,
                        "updates_this_vector_step": update_count,
                        "update_credit": trainer.update_credit,
                        "replay_size": replay.size,
                        "replay_capacity": replay.capacity,
                        "replay_fraction": (
                            replay.size / max(replay.capacity, 1)
                        ),
                        "replay_fill_fraction": (
                            replay.size / max(replay.capacity, 1)
                        ),
                        "learning_starts": config.tqc.learning_starts,
                        "learning_started": (
                            trainer.transitions
                            >= config.tqc.learning_starts
                        ),
                        "mode": (
                            "learning"
                            if trainer.transitions
                            >= config.tqc.learning_starts
                            else "warmup"
                        ),
                        "vector_steps_this_session": vector_steps,
                        **learner_status,
                    },
                    "rollout": rollout_status,
                }
            if done_any:
                env.reset(result.done)
            if emit_monitor and not checkpoint_due:
                monitor.heartbeat(
                    current=trainer.transitions,
                    message=(
                        f"Residual TQC rollout at "
                        f"{trainer.transitions}/"
                        f"{config.tqc.total_transitions} transitions"
                    ),
                    metrics=last_monitor_metrics,
                    health=health,
                )
            if checkpoint_due:
                checkpoint_health: Dict[str, object] = {}

                def validate_periodic_checkpoint() -> None:
                    checkpoint_health.update(
                        _assert_tqc_continuation_health(
                            env,
                            replay,
                            context=(
                                "TQC periodic continuation checkpoint "
                                f"{trainer.transitions}"
                            ),
                        )
                    )

                def write_periodic_checkpoint() -> None:
                    if monitor is not None:
                        monitor.heartbeat(
                            current=trainer.transitions,
                            message=(
                                "Saving full TQC/replay/environment checkpoint; "
                                "this may take several minutes."
                            ),
                            metrics=last_monitor_metrics,
                            health=checkpoint_health,
                            artifacts=residual_artifacts,
                            force=True,
                            stale_after_seconds=900.0,
                        )
                    save_checkpoint()

                last_saved_checkpoint_cursor, _saved = (
                    _commit_checkpoint_cursor(
                        checkpoint_cursor(),
                        last_saved_checkpoint_cursor,
                        validate=validate_periodic_checkpoint,
                        write=write_periodic_checkpoint,
                    )
                )
                health = checkpoint_health
                episode_status = tracker.snapshot()
                event = {
                    "event": "tqc_checkpoint",
                    "transitions": trainer.transitions,
                    "updates": trainer.updates,
                    "replay_size": replay.size,
                    "episodes_completed": episode_status["completed"],
                    "episode_success_rate": episode_status["success_rate"],
                    "recent_episode_success_rate": (
                        episode_status["recent"]["success_rate"]
                    ),
                    **_metric_scalars(last_metrics),
                }
                append_jsonl(log_path, event)
                print(json.dumps(event, sort_keys=True), flush=True)
                residual_artifacts = _monitor_artifacts(
                    diffusion_checkpoint=diffusion_checkpoint,
                    tqc_full_checkpoint=output,
                    tqc_policy_checkpoint=policy_output,
                    tqc_log=log_path,
                )
                if monitor is not None:
                    monitor.heartbeat(
                        current=trainer.transitions,
                        message=(
                            f"Saved full TQC checkpoint at "
                            f"{trainer.transitions} transitions"
                        ),
                        metrics=last_monitor_metrics,
                        health=health,
                        artifacts=residual_artifacts,
                        force=True,
                    )
                next_checkpoint += config.tqc.checkpoint_every_transitions
            _safe_cursor_interrupt(interrupt, monitor)
        # Also regenerates a missing/stale lightweight actor if a previous
        # allocation committed the full training checkpoint immediately before
        # its kill-after deadline.
        trainer.assert_finite()
        save_policy_checkpoint()
    except (KeyboardInterrupt, MJWarpHealthError, FloatingPointError) as exc:
        terminal_exc = exc
        health_abort = not isinstance(exc, KeyboardInterrupt)
        checkpoint_written_for_interrupt = False
        checkpoint_reused_at_same_cursor = False
        # A transient overflow/non-finite flag may have contaminated the
        # current replay/model since the last host health poll.  Never replace
        # a known-good checkpoint with that state.  A deferred SIGINT is at a
        # complete vector-step cursor, but the backend and replay are certified
        # once more here because the last periodic health poll may be older.
        if not health_abort:
            interrupt_health: Dict[str, object] = {}

            def validate_interrupt_checkpoint() -> None:
                interrupt_health.update(
                    _assert_tqc_continuation_health(
                        env,
                        replay,
                        context=(
                            "TQC graceful-stop continuation checkpoint "
                            f"{trainer.transitions}"
                        ),
                    )
                )

            def write_interrupt_checkpoint() -> None:
                if monitor is not None:
                    monitor.heartbeat(
                        current=trainer.transitions,
                        message=(
                            "Graceful stop accepted; saving full TQC "
                            "continuation checkpoint."
                        ),
                        metrics=last_monitor_metrics,
                        health=interrupt_health,
                        artifacts=residual_artifacts,
                        force=True,
                        stale_after_seconds=900.0,
                    )
                save_checkpoint(interrupted=True)

            try:
                (
                    last_saved_checkpoint_cursor,
                    checkpoint_written_for_interrupt,
                ) = _commit_checkpoint_cursor(
                    checkpoint_cursor(),
                    last_saved_checkpoint_cursor,
                    validate=validate_interrupt_checkpoint,
                    write=write_interrupt_checkpoint,
                )
                checkpoint_reused_at_same_cursor = (
                    not checkpoint_written_for_interrupt
                )
                if checkpoint_reused_at_same_cursor and monitor is not None:
                    monitor.heartbeat(
                        current=trainer.transitions,
                        message=(
                            "Graceful stop accepted; reusing the full TQC "
                            "checkpoint already committed at this cursor."
                        ),
                        metrics=last_monitor_metrics,
                        artifacts=residual_artifacts,
                        force=True,
                    )
            except (MJWarpHealthError, FloatingPointError) as validation_exc:
                # The current in-memory cursor is unsafe.  Leave the existing
                # atomic checkpoint untouched and surface a failed health gate
                # instead of labelling bad state as resumable.
                terminal_exc = validation_exc
                health_abort = True
        event = {
            "event": (
                "tqc_health_abort"
                if health_abort
                else "tqc_interrupted"
            ),
            "transitions": trainer.transitions,
            "updates": trainer.updates,
            "replay_size": replay.size,
            "checkpoint": str(output) if output.exists() else None,
            "last_good_checkpoint_preserved": (
                health_abort and output.is_file()
            ),
            "checkpoint_written_for_interrupt": (
                checkpoint_written_for_interrupt
            ),
            "checkpoint_reused_at_same_cursor": (
                checkpoint_reused_at_same_cursor
            ),
            "reason": str(terminal_exc),
            **_metric_scalars(last_metrics),
        }
        append_jsonl(log_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        if monitor is not None:
            monitor.heartbeat(
                current=trainer.transitions,
                message=(
                    "TQC health gate aborted; last good checkpoint preserved."
                    if health_abort
                    else "TQC interrupted after safe continuation checkpoint."
                ),
                metrics=last_monitor_metrics,
                artifacts=_monitor_artifacts(
                    diffusion_checkpoint=diffusion_checkpoint,
                    tqc_full_checkpoint=output,
                    tqc_policy_checkpoint=policy_output,
                    tqc_log=log_path,
                ),
                force=True,
            )
        if terminal_exc is exc:
            raise
        raise terminal_exc from exc
    finally:
        interrupt.restore()
        env.close()
    return policy_output


def probe_residual_resume(
    config,
    *,
    diffusion_checkpoint: str | Path,
    tqc_checkpoint: str | Path,
    monitor=None,
) -> Dict[str, float]:
    """Load the full continuation cursor and execute one unsaved vector step."""

    from .replay import CUDAReplayBuffer
    from .tqc import TQCTrainer

    torch = _torch()
    diffusion_checkpoint = Path(
        diffusion_checkpoint
    ).expanduser().resolve()
    tqc_checkpoint = Path(tqc_checkpoint).expanduser().resolve()
    probe_artifacts = _monitor_artifacts(
        diffusion_checkpoint=diffusion_checkpoint,
        tqc_full_checkpoint=tqc_checkpoint,
    )
    tracker = _episode_tracker(
        config,
        scope="stochastic_resume_probe",
        monitor=monitor,
    )
    if monitor is not None:
        monitor.set_stage(
            "resume_probe",
            current=0,
            total=1,
            unit="vector_step",
            message=(
                "Loading the full continuation checkpoint and building the "
                "MJWarp CUDA graph."
            ),
            metrics={"episodes": tracker.snapshot()},
            artifacts=probe_artifacts,
            stale_after_seconds=900.0,
        )
    payload = _load(tqc_checkpoint)
    _validate_diffusion_dependency(
        payload, diffusion_checkpoint, config=config
    )
    env_state = payload.get("env_state")
    if env_state is None:
        raise ValueError("resume probe requires a full TQC training checkpoint")
    _seed_everything(
        config.runtime.seed, deterministic=config.runtime.deterministic
    )
    controller = BatchedDiffusionController.from_checkpoint(
        diffusion_checkpoint,
        worlds=config.physics.worlds,
        inference_steps=config.diffusion.inference_steps,
        replan_interval=config.diffusion.replan_interval,
        device=config.runtime.device,
        seed=config.runtime.seed,
    )
    env = MJWarpScrewEnv(
        config,
        baseline="diffusion",
        diffusion_controller=controller,
    )
    replay = CUDAReplayBuffer(
        capacity=config.tqc.replay_capacity,
        obs_dim=TQC_OBS_DIM,
        action_dim=RESIDUAL_ACTION_DIM,
        device=config.runtime.device,
    )
    trainer = TQCTrainer(config, device=config.runtime.device)
    try:
        trainer.load(payload, replay=replay)
        env.load_state_dict(env_state)
        pending_replay = _restore_pending_replay(
            payload.get("pending_replay"), device=env.device
        )
        _restore_rng(payload.get("rng_state"))

        def residual_policy(tqc_state, action_mask):
            return trainer.act(
                tqc_state, action_mask=action_mask
            )

        result = env.step(residual_policy=residual_policy)
        health = _assert_backend_health(env, context="TQC resume probe")
        done_ids = torch.nonzero(
            result.done, as_tuple=False
        ).flatten()
        _track_terminal_episodes(
            tracker,
            result.info,
            done_ids,
            env=env,
            monitor=monitor,
            cursor={
                "saved_transitions": trainer.transitions,
                "probe_vector_step": 1,
            },
        )
        rollout_status, _unused = _materialize_rollout_metrics(
            env,
            result,
            action=result.info["executed_residual_action"],
        )
        episode_status = tracker.snapshot()
        metrics = {
            "saved_transitions": float(trainer.transitions),
            "saved_replay_size": float(replay.size),
            "saved_pending_replay": float(
                0
                if pending_replay is None
                else pending_replay["world_ids"].numel()
            ),
            "probe_worlds": float(env.worlds),
            "probe_successes": float(episode_status["successes"]),
            "probe_mean_reward": rollout_status["reward_mean"],
        }
        if monitor is not None:
            monitor.heartbeat(
                current=1,
                message="TQC continuation probe completed one vector step.",
                metrics={
                    "episodes": episode_status,
                    "tqc": {
                        "transitions": trainer.transitions,
                        "updates": trainer.updates,
                        "replay_size": replay.size,
                    },
                    "rollout": rollout_status,
                },
                health=health,
                artifacts=probe_artifacts,
                force=True,
            )
        _safe_cursor_interrupt(None, monitor)
        print(
            json.dumps(
                {"event": "tqc_resume_probe", **metrics},
                sort_keys=True,
            ),
            flush=True,
        )
        return metrics
    finally:
        env.close()


def evaluate(
    config,
    *,
    diffusion_checkpoint: str | Path,
    tqc_checkpoint: str | Path | None = None,
    episodes: int,
    log_path: Optional[str | Path] = None,
    monitor=None,
) -> Dict[str, float]:
    torch = _torch()
    if int(episodes) <= 0:
        raise ValueError("evaluation episodes must be positive")
    evaluation_seed = config.runtime.seed + 100_000
    _seed_everything(
        evaluation_seed,
        deterministic=config.runtime.deterministic,
    )
    diffusion_checkpoint = Path(
        diffusion_checkpoint
    ).expanduser().resolve()
    tqc_checkpoint = (
        None
        if tqc_checkpoint is None
        else Path(tqc_checkpoint).expanduser().resolve()
    )
    evaluation_artifacts = _monitor_artifacts(
        diffusion_checkpoint=diffusion_checkpoint,
        tqc_checkpoint=tqc_checkpoint,
        evaluation_log=log_path,
    )
    tracker = _episode_tracker(
        config,
        scope="deterministic_evaluation",
        monitor=monitor,
        log_path=log_path,
    )
    if monitor is not None:
        monitor.set_stage(
            "evaluation",
            current=0,
            total=episodes,
            unit="episode",
            message=(
                "Loading policies and building the MJWarp evaluation CUDA "
                "graph."
            ),
            metrics={"episodes": tracker.snapshot()},
            artifacts=evaluation_artifacts,
            stale_after_seconds=900.0,
        )
    diffusion_payload = _load(diffusion_checkpoint)
    validate_checkpoint(diffusion_payload, kind="diffusion")
    _validate_frozen_diffusion_config(diffusion_payload, config)
    diffusion_sha256 = sha256_file(diffusion_checkpoint)
    payload = None
    tqc_sha256 = None
    if tqc_checkpoint is not None:
        payload = _load(tqc_checkpoint)
        diffusion_sha256 = _validate_diffusion_dependency(
            payload, diffusion_checkpoint, config=config
        )
        tqc_sha256 = sha256_file(tqc_checkpoint)
    worlds = min(config.physics.worlds, max(1, int(episodes)))
    controller = BatchedDiffusionController.from_checkpoint(
        diffusion_checkpoint,
        worlds=worlds,
        inference_steps=config.diffusion.inference_steps,
        replan_interval=config.diffusion.replan_interval,
        device=config.runtime.device,
        seed=evaluation_seed,
    )
    env = MJWarpScrewEnv(
        config,
        worlds=worlds,
        baseline="diffusion",
        diffusion_controller=controller,
    )
    trainer = None
    if payload is not None:
        from .tqc import TQCTrainer

        trainer = TQCTrainer(config, device=config.runtime.device)
        if payload.get("policy_only"):
            trainer.actor.load_state_dict(payload["actor"], strict=True)
        else:
            trainer.load(payload)
    zero_residual = torch.zeros(
        (worlds, RESIDUAL_ACTION_DIM),
        device=env.device,
        dtype=torch.float32,
    )
    completed = 0
    vector_steps = 0
    last_rollout_status: Dict[str, object] = {}
    last_health: Dict[str, object] = {}
    if monitor is not None:
        last_health = _assert_backend_health(
            env, context="evaluation environment initialization"
        )
        monitor.heartbeat(
            current=0,
            message="Evaluation environment ready; rollouts are running.",
            metrics={"episodes": tracker.snapshot()},
            health=last_health,
            artifacts=evaluation_artifacts,
            force=True,
        )
    try:
        while completed < episodes:
            if trainer is None:
                result = env.step(zero_residual)
            else:
                def residual_policy(tqc_state, action_mask):
                    return trainer.act(
                        tqc_state,
                        action_mask=action_mask,
                        deterministic=True,
                    )

                result = env.step(residual_policy=residual_policy)
            vector_steps += 1
            periodic_health = (
                vector_steps % config.runtime.log_every_vector_steps
                == 0
            )
            monitor_due = monitor is not None and monitor.due
            health = None
            if periodic_health or monitor_due:
                health = _assert_backend_health(
                    env,
                    context=f"evaluation vector step {vector_steps}",
                )
                last_health = health
            done_any = bool(result.done.any())
            if done_any:
                done_ids = torch.nonzero(
                    result.done, as_tuple=False
                ).flatten()
                remaining = episodes - completed
                done_ids = done_ids[:remaining]
                completed += _track_terminal_episodes(
                    tracker,
                    result.info,
                    done_ids,
                    env=env,
                    monitor=monitor,
                    cursor={
                        "evaluation_completed_before": completed,
                        "evaluation_vector_step": vector_steps,
                        "evaluation_seed": (
                            config.runtime.seed + 100_000
                        ),
                    },
                )
            emit_monitor = monitor is not None and (
                monitor.due or completed >= episodes
            )
            materialize_diagnostics = (
                completed >= episodes or emit_monitor
            )
            if materialize_diagnostics:
                if health is None:
                    health = _assert_backend_health(
                        env,
                        context=(
                            f"evaluation diagnostic step {vector_steps}"
                        ),
                    )
                    last_health = health
                last_rollout_status, _unused = (
                    _materialize_rollout_metrics(
                        env,
                        result,
                        action=(
                            result.info["executed_diffusion_action"]
                            if trainer is None
                            else result.info["executed_residual_action"]
                        ),
                    )
                )
            if done_any:
                env.reset(result.done)
            if emit_monitor:
                monitor.heartbeat(
                    current=completed,
                    message=(
                        f"Deterministic evaluation completed "
                        f"{completed}/{episodes} episodes"
                    ),
                    metrics={
                        "episodes": tracker.snapshot(),
                        "rollout": last_rollout_status,
                    },
                    health=health,
                    artifacts=evaluation_artifacts,
                    force=completed >= episodes,
                )
            _safe_cursor_interrupt(None, monitor)
        if vector_steps % config.runtime.log_every_vector_steps != 0:
            last_health = _assert_backend_health(
                env, context=f"evaluation final step {vector_steps}"
            )
    finally:
        env.close()
    episode_status = tracker.snapshot()
    metrics = {
        "episodes": float(completed),
        "successes": float(episode_status["successes"]),
        "failures": float(episode_status["failures"]),
        "truncations": float(episode_status["truncations"]),
        "success_rate": float(episode_status["success_rate"] or 0.0),
        "mean_return": float(episode_status["mean_return"] or 0.0),
        "mean_depth_m": float(episode_status["mean_depth_m"] or 0.0),
        "mean_turns": float(episode_status["mean_turns"] or 0.0),
    }
    if log_path is not None:
        evaluation_event = {
            "schema": "screw-hpc-evaluation-v2",
            "event": "evaluation",
            "updated_utc": utc_now(),
            "seed": evaluation_seed,
            "deterministic_policy": True,
            "control_stack": (
                "diffusion_safety_ik"
                if trainer is None
                else "diffusion_safety_ik_actuator_residual_tqc"
            ),
            "residual_mode": (
                "disabled_zero"
                if trainer is None
                else "tqc"
            ),
            "requested_episodes": int(episodes),
            "worlds": int(worlds),
            "vector_steps": int(vector_steps),
            "config": {
                "name": config.name,
                "sha256": canonical_sha256(config.to_dict()),
            },
            "contracts": {
                "schema_hash": SCHEMA_HASH,
                "task_geometry_hash": TASK_GEOMETRY_HASH,
            },
            "checkpoints": {
                "diffusion": {
                    "path": str(diffusion_checkpoint),
                    "sha256": diffusion_sha256,
                },
                **(
                    {}
                    if payload is None
                    else {
                        "tqc": {
                            "path": str(tqc_checkpoint),
                            "sha256": tqc_sha256,
                            "policy_only": bool(
                                payload.get("policy_only")
                            ),
                            "transitions": int(
                                payload.get("transitions", 0)
                            ),
                            "updates": int(
                                payload.get("updates", 0)
                            ),
                        },
                    }
                ),
            },
            "summary": metrics,
            "episodes": episode_status,
            "rollout": last_rollout_status,
            "health": last_health,
            "lineage": (
                monitor.event_context(
                    evaluation_seed=evaluation_seed,
                    evaluation_vector_steps=vector_steps,
                )
                if monitor is not None
                else {}
            ),
        }
        append_history(log_path, evaluation_event)
    print(json.dumps({"event": "evaluation", **metrics}, sort_keys=True), flush=True)
    if monitor is not None:
        monitor.heartbeat(
            current=completed,
            message="Deterministic evaluation is complete.",
            metrics={
                "episodes": episode_status,
                "rollout": last_rollout_status,
            },
            health=last_health,
            artifacts=_monitor_artifacts(
                diffusion_checkpoint=diffusion_checkpoint,
                tqc_checkpoint=tqc_checkpoint,
                evaluation_log=log_path,
            ),
            force=True,
        )
    return metrics


__all__ = [
    "evaluate",
    "generate_demos",
    "probe_residual_resume",
    "train_diffusion",
    "train_residual",
]
