#!/usr/bin/env python3
"""Measure the peak CUDA footprint of every formal training stage.

Unlike the lightweight MJWarp preflight, this probe exercises the allocations
that determine whether a formal run can survive:

* the optimized float16 demonstration trajectory buffer;
* the complete cached demonstration dataset plus a Diffusion backward/Adam
  step and EMA weights;
* either a pure-Diffusion full-world rollout, or the production
  Diffusion--IK--actuator-residual TQC stack.

Run each world count in a fresh process.  A CUDA OOM can leave the process
allocator unusable even after tensors are deleted.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from diffusion_policy import DiffusionPolicy, ExponentialMovingAverage
from s_hpc.config import load_config
from s_hpc.contracts import (
    ACTION_DIM,
    OBS_DIM,
    OBS_HORIZON,
    PHASES,
    RESIDUAL_ACTION_DIM,
    SCHEMA_HASH,
    TASK_GEOMETRY_HASH,
    TQC_CONTROLLER_STATE_NAMES,
    TQC_OBS_DIM,
)
from s_hpc.diffusion_runtime import BatchedDiffusionController
from s_hpc.env import MJWarpScrewEnv
from s_hpc.paths import add_external_path_flag, resolve_package_path
from s_hpc.rollout_audit import (
    ExpertRolloutAudit,
    certify_expert_rollout,
)


EXPECTED_VERSIONS = {
    "torch": "2.7.1",
    "mujoco": "3.10.0",
    "mujoco-warp": "3.10.0.3",
    "warp-lang": "1.14.0",
}
SINGLE_PHASE_CHOICES = (
    "expert_physical_canary",
    "demonstration_generation",
    "diffusion_training",
    "diffusion_only_rollout",
    "diffusion_tqc_residual",
)


def classify_gpu_type(name: str) -> str:
    """Map CUDA's product name onto the CX3 PBS ``gpu_type`` vocabulary."""

    normalized = "".join(character for character in name.upper() if character.isalnum())
    if "RTXPRO6000" in normalized and "BLACKWELL" in normalized:
        return "RTX_PRO_6000_BLACKWELL"
    if "L40S" in normalized:
        return "L40S"
    if "A100" in normalized:
        return "A100"
    if "RTX6000" in normalized:
        return "RTX6000"
    return "UNKNOWN"


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"required GPU package is missing: {package}") from exc
        if actual.split("+", 1)[0] != expected:
            raise RuntimeError(
                f"{package} version must be {expected}, got {actual}"
            )
        versions[package] = actual
    return versions


def _last_json_object(text: str) -> dict[str, Any] | None:
    starts = [
        index
        for index, character in enumerate(text)
        if character == "{"
    ]
    for index in reversed(starts):
        try:
            value = json.loads(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


class PhasePeak:
    """Combine whole-device samples with PyTorch's transient peak counters."""

    def __init__(self, torch, device, *, name: str, total_bytes: int) -> None:
        self.torch = torch
        self.device = device
        self.name = str(name)
        self.total_bytes = int(total_bytes)
        self.samples: list[dict[str, Any]] = []
        self.max_device_used = 0
        self.max_external_used = 0
        torch.cuda.reset_peak_memory_stats(device)
        self.sample("phase_start")

    def sample(self, label: str) -> None:
        self.torch.cuda.synchronize(self.device)
        free_bytes, observed_total = self.torch.cuda.mem_get_info(self.device)
        if int(observed_total) != self.total_bytes:
            raise RuntimeError("CUDA total memory changed during capacity probe")
        reserved = int(self.torch.cuda.memory_reserved(self.device))
        allocated = int(self.torch.cuda.memory_allocated(self.device))
        device_used = self.total_bytes - int(free_bytes)
        external_used = max(0, device_used - reserved)
        self.max_device_used = max(self.max_device_used, device_used)
        self.max_external_used = max(self.max_external_used, external_used)
        self.samples.append(
            {
                "label": str(label),
                "cuda_free_bytes": int(free_bytes),
                "cuda_device_used_bytes": device_used,
                "torch_allocated_bytes": allocated,
                "torch_reserved_bytes": reserved,
                "estimated_non_torch_used_bytes": external_used,
            }
        )

    def finish(self) -> dict[str, Any]:
        self.sample("phase_end")
        torch_peak_allocated = int(
            self.torch.cuda.max_memory_allocated(self.device)
        )
        torch_peak_reserved = int(
            self.torch.cuda.max_memory_reserved(self.device)
        )
        # Sampling total-free can miss a short-lived PyTorch temporary after
        # it has been released.  The second term combines the largest observed
        # external (MJWarp/driver) footprint with PyTorch's allocator peak.
        conservative_used = max(
            self.max_device_used,
            self.max_external_used + torch_peak_reserved,
        )
        return {
            "name": self.name,
            "cuda_total_memory_bytes": self.total_bytes,
            "sampled_peak_device_used_bytes": self.max_device_used,
            "estimated_peak_non_torch_used_bytes": self.max_external_used,
            "torch_peak_allocated_bytes": torch_peak_allocated,
            "torch_peak_reserved_bytes": torch_peak_reserved,
            "conservative_peak_used_bytes": conservative_used,
            "conservative_peak_used_fraction": (
                conservative_used / self.total_bytes
            ),
            "samples": self.samples,
        }


def _release_cuda(torch, device) -> None:
    # MJWarp frees its large per-world arrays asynchronously through Warp's
    # CUDA mempool. A Torch device synchronize does not guarantee that those
    # deferred frees have reached Warp's pool before the next phase allocates
    # another environment in the same process.
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    import warp as wp

    wp.set_mempool_release_threshold(str(device), 0)
    wp.synchronize_device(str(device))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def _assert_health(env, *, context: str) -> dict[str, Any]:
    sample = env.backend.health()
    if not sample.get("finite", False) or sample.get("sticky_nonfinite", 0):
        raise RuntimeError(f"{context}: non-finite MJWarp state: {sample}")
    if sample.get("sticky_external_nonfinite", 0):
        raise RuntimeError(f"{context}: non-finite external tensor: {sample}")
    if sample.get("overflow_worlds", 0) or sample.get("sticky_overflow", 0):
        raise RuntimeError(f"{context}: MJWarp overflow: {sample}")
    if (
        sample.get("contacts_total", 0)
        >= sample.get("contact_capacity_total", 1)
        or sample.get("sticky_contacts_full", 0)
    ):
        raise RuntimeError(f"{context}: contact capacity reached: {sample}")
    if (
        sample.get("max_nefc", 0) >= env.config.physics.njmax
        or sample.get("sticky_constraints_full", 0)
    ):
        raise RuntimeError(f"{context}: constraint capacity reached: {sample}")
    return sample


def _diffusion_policy(config, device):
    policy = DiffusionPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        obs_horizon=OBS_HORIZON,
        horizon=config.demos.action_horizon,
        num_diffusion_steps=config.diffusion.diffusion_steps,
        hidden_dim=config.diffusion.hidden_dim,
        depth=config.diffusion.depth,
    )
    return policy.to(device)


def _print_expert_truncation_context(step_result, *, max_records: int = 32) -> None:
    """Emit the final scheduler/gate state for worlds that timed out."""

    info = step_result.info
    truncated = info.get("truncated")
    if truncated is None or not bool(truncated.any().item()):
        return

    indices = truncated.nonzero(as_tuple=False).flatten()
    keys = (
        "expert_step",
        "task_phase",
        "holder_active",
        "grasp_latched",
        "all_four_contact",
        "two_finger_contact",
        "shaft_captured",
        "thread_engaged",
        "diffusion_handoff_ready",
        "preinsert_alignment_ready",
        "capture_alignment_ready",
        "thread_engagement_ready",
        "pickup_release_hold",
        "preinsert_alignment_hold",
        "capture_alignment_hold",
        "thread_engagement_hold",
        "thread_regrasp_hold",
        "contact_recovery_hold",
        "preinsert_alignment_stable_steps",
        "capture_alignment_stable_steps",
        "thread_ready_stable_steps",
        "diffusion_handoff_position_error_m",
        "diffusion_handoff_orientation_error_deg",
        "preinsert_lateral_error_m",
        "preinsert_axis_error_deg",
        "preinsert_depth_error_m",
        "preinsert_tip_speed_m_s",
        "preinsert_lateral_speed_m_s",
        "preinsert_angular_speed_deg_s",
        "capture_lateral_error_m",
        "capture_axis_error_deg",
        "capture_depth_error_m",
        "capture_tip_speed_m_s",
        "capture_lateral_speed_m_s",
        "capture_angular_speed_deg_s",
        "thread_lateral_error_m",
        "thread_axis_error_deg",
        "thread_depth_error_m",
        "thread_tip_speed_m_s",
        "thread_lateral_speed_m_s",
        "thread_angular_speed_deg_s",
        "fingertip_contact_force_n",
    )

    records: list[dict[str, Any]] = []
    for world_index_tensor in indices[:max_records]:
        world_index = int(world_index_tensor.item())
        record: dict[str, Any] = {"world": world_index}
        for key in keys:
            value = info.get(key)
            if value is None:
                continue
            item = value[world_index].detach().cpu()
            record[key] = item.item() if item.numel() == 1 else item.tolist()
        phase = record.get("task_phase")
        if isinstance(phase, int) and 0 <= phase < len(PHASES):
            record["phase_name"] = PHASES[phase]
        records.append(record)

    print(
        json.dumps(
            {
                "event": "expert_truncation_context",
                "truncated_worlds": int(indices.numel()),
                "records": records,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _expert_physical_phase(
    config,
    torch,
    device,
    total_bytes: int,
    physics_steps: int,
    max_expert_truncation_fraction: float = 0.0,
):
    """Certify the complete requested world population without fake buffers."""

    worlds = config.physics.worlds
    peak = PhasePeak(
        torch, device, name="expert_physical_canary", total_bytes=total_bytes
    )
    env = None
    started = time.monotonic()
    try:
        env = MJWarpScrewEnv(config, worlds=worlds, baseline="expert")
        zero = torch.zeros((worlds, RESIDUAL_ACTION_DIM), device=device)
        steps_run = min(physics_steps, config.physics.max_episode_steps)
        rollout_audit = ExpertRolloutAudit(
            torch, worlds=worlds, device=device
        )
        for step_index in range(steps_run):
            step_result = env.step(zero)
            rollout_audit.update(
                step_result, step_number=step_index + 1
            )
            if step_index + 1 == steps_run:
                _print_expert_truncation_context(step_result)
            if (
                (step_index + 1)
                % max(config.runtime.log_every_vector_steps, 1)
                == 0
            ):
                audit_context = (
                    "full-world expert canary at control step "
                    f"{step_index + 1}"
                )
                if max_expert_truncation_fraction == 0.0:
                    rollout_audit.assert_clean(context=audit_context)
                else:
                    dirty = torch.stack(
                        (
                            rollout_audit.any_failure.any(),
                            rollout_audit.any_truncation.any(),
                        )
                    )
                    if bool(dirty.any().item()):
                        certify_expert_rollout(
                            env,
                            step_result,
                            rollout_audit,
                            context=audit_context,
                            require_terminal=False,
                            max_truncation_fraction=(
                                max_expert_truncation_fraction
                            ),
                        )
        health = _assert_health(env, context="full-world expert canary")
        task_audit = certify_expert_rollout(
            env,
            step_result,
            rollout_audit,
            context="full-world expert canary",
            require_terminal=(
                steps_run == config.physics.max_episode_steps
            ),
            max_truncation_fraction=max_expert_truncation_fraction,
        )
        peak.sample("full_world_expert_physics")
        result = peak.finish()
        result.update(
            {
                "worlds": worlds,
                "physics_steps": steps_run,
                "max_expert_truncation_fraction": (
                    max_expert_truncation_fraction
                ),
                "elapsed_seconds": time.monotonic() - started,
                "health": health,
                "task_audit": task_audit,
            }
        )
        return result
    finally:
        if env is not None:
            env.close()


def _demo_phase(
    config,
    torch,
    device,
    total_bytes: int,
    physics_steps: int,
    max_expert_truncation_fraction: float = 0.0,
):
    worlds = min(config.physics.worlds, config.demos.episodes)
    peak = PhasePeak(
        torch, device, name="demonstration_generation", total_bytes=total_bytes
    )
    env = None
    started = time.monotonic()
    try:
        env = MJWarpScrewEnv(config, worlds=worlds, baseline="expert")
        zero = torch.zeros((worlds, RESIDUAL_ACTION_DIM), device=device)
        steps_run = min(physics_steps, config.physics.max_episode_steps)
        rollout_audit = ExpertRolloutAudit(
            torch, worlds=worlds, device=device
        )
        # Allocate the exact formal one-copy buffers before physics. They
        # remain resident while every env.step transient is exercised, which
        # catches generator peaks that an allocate-after-stepping probe misses.
        observations = torch.empty(
            (
                worlds,
                config.physics.max_episode_steps + 1,
                OBS_DIM,
            ),
            device=device,
            dtype=torch.float16,
        )
        actions = torch.empty(
            (
                worlds,
                config.physics.max_episode_steps,
                ACTION_DIM,
            ),
            device=device,
            dtype=torch.float16,
        )
        observations[:, 0].copy_(env.obs_history[:, -1])
        peak.sample("environment_and_trajectory_buffers")
        for step_index in range(steps_run):
            step_result = env.step(zero)
            observations[:, step_index + 1].copy_(
                step_result.observation
            )
            actions[:, step_index].copy_(step_result.expert_action)
            rollout_audit.update(
                step_result, step_number=step_index + 1
            )
            if (
                (step_index + 1)
                % max(config.runtime.log_every_vector_steps, 1)
                == 0
            ):
                audit_context = (
                    "demonstration probe at control step "
                    f"{step_index + 1}"
                )
                if max_expert_truncation_fraction == 0.0:
                    rollout_audit.assert_clean(context=audit_context)
                else:
                    dirty = torch.stack(
                        (
                            rollout_audit.any_failure.any(),
                            rollout_audit.any_truncation.any(),
                        )
                    )
                    if bool(dirty.any().item()):
                        certify_expert_rollout(
                            env,
                            step_result,
                            rollout_audit,
                            context=audit_context,
                            require_terminal=False,
                            max_truncation_fraction=(
                                max_expert_truncation_fraction
                            ),
                        )
        health = _assert_health(env, context="demonstration probe")
        task_audit = certify_expert_rollout(
            env,
            step_result,
            rollout_audit,
            context="demonstration probe",
            require_terminal=(
                steps_run == config.physics.max_episode_steps
            ),
            max_truncation_fraction=max_expert_truncation_fraction,
        )
        peak.sample("trajectory_buffers_and_physics")
        # Keep the tensors live through the final sample.
        trajectory_bytes = (
            observations.numel() * observations.element_size()
            + actions.numel() * actions.element_size()
        )
        result = peak.finish()
        result.update(
            {
                "worlds": worlds,
                "physics_steps": steps_run,
                "max_expert_truncation_fraction": (
                    max_expert_truncation_fraction
                ),
                "trajectory_buffer_bytes": int(trajectory_bytes),
                "elapsed_seconds": time.monotonic() - started,
                "health": health,
                "task_audit": task_audit,
            }
        )
        return result
    finally:
        if env is not None:
            env.close()


def _diffusion_training_phase(config, torch, device, total_bytes: int):
    peak = PhasePeak(
        torch, device, name="diffusion_training", total_bytes=total_bytes
    )
    started = time.monotonic()
    episodes = config.demos.episodes
    steps = config.physics.max_episode_steps
    # Exact shapes used by DemoWindowSampler.cache_on_device.
    cached_observations = torch.empty(
        (episodes, steps + 1, OBS_DIM),
        device=device,
        dtype=torch.float16,
    )
    cached_actions = torch.empty(
        (episodes, steps, ACTION_DIM),
        device=device,
        dtype=torch.float16,
    )
    peak.sample("dataset_cache")

    policy = _diffusion_policy(config, device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.diffusion.learning_rate,
        weight_decay=1.0e-6,
    )
    ema = ExponentialMovingAverage(
        policy, decay=config.diffusion.ema_decay
    )
    batch_size = config.diffusion.batch_size
    observation = torch.randn(
        (batch_size, OBS_HORIZON, OBS_DIM), device=device
    )
    action = torch.randn(
        (batch_size, config.demos.action_horizon, ACTION_DIM),
        device=device,
    ).clamp_(-1.0, 1.0)
    optimizer.zero_grad(set_to_none=True)
    diffusion_bfloat16 = (
        torch.cuda.get_device_properties(device).major >= 8
    )
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=diffusion_bfloat16,
    ):
        loss = policy.compute_loss(observation, action)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    ema.update(policy)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("Diffusion probe produced NaN or infinity")
    peak.sample("backward_adam_ema")
    cache_bytes = (
        cached_observations.numel() * cached_observations.element_size()
        + cached_actions.numel() * cached_actions.element_size()
    )
    result = peak.finish()
    result.update(
        {
            "dataset_cache_bytes": int(cache_bytes),
            "batch_size": batch_size,
            "autocast_dtype": (
                "bfloat16" if diffusion_bfloat16 else "float32"
            ),
            "loss": float(loss.detach().to(torch.float32).cpu()),
            "gradient_norm": float(
                gradient_norm.detach().to(torch.float32).cpu()
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return result


def _residual_phase(config, torch, device, total_bytes: int, physics_steps: int):
    from s_hpc.replay import CUDAReplayBuffer
    from s_hpc.tqc import TQCTrainer

    peak = PhasePeak(
        torch, device, name="diffusion_tqc_residual", total_bytes=total_bytes
    )
    started = time.monotonic()
    env = None
    try:
        policy = _diffusion_policy(config, device).eval()
        controller = BatchedDiffusionController(
            policy,
            worlds=config.physics.worlds,
            inference_steps=config.diffusion.inference_steps,
            replan_interval=config.diffusion.replan_interval,
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
            device=str(device),
        )
        trainer = TQCTrainer(config, device=str(device))
        # A formal continuation checkpoint validates the complete populated
        # replay.  Production keeps this capacity in ordinary host RAM and
        # transfers only sampled minibatches to the learner device.  Touch all
        # rows here so the preflight covers the exact full-capacity host path.
        for replay_tensor in (
            replay.observation,
            replay.action,
            replay.reward,
            replay.next_observation,
            replay.action_mask,
            replay.next_action_mask,
            replay.terminated,
            replay.truncated,
        ):
            replay_tensor.zero_()
        # Exercise actor/temperature learning with the maximum executable
        # residual width rather than an all-disabled synthetic batch.
        replay.action_mask.fill_(True)
        replay.next_action_mask.fill_(True)
        replay.size = replay.capacity
        replay.position = 0
        replay.assert_finite()
        if replay.storage_device.type != "cpu":
            raise RuntimeError("formal replay storage must remain on CPU")
        peak.sample("env_controller_host_replay_tqc_integrity_scan")

        # Environment reset deliberately avoids an early Diffusion sample.
        # Force the same full-world replan used after the scripted handoff.
        controller.cursor = controller.replan_interval
        diffusion_action = controller.act(env.obs_history)
        # The production workflow prepares this state after nominal IK.  The
        # capacity probe uses finite zero-normalised controller features and an
        # explicit all-enabled mask to exercise the exact 242-D actor shape.
        from s_hpc.tqc import build_tqc_observation

        actor_mask = torch.ones(
            (config.physics.worlds, RESIDUAL_ACTION_DIM),
            device=device,
            dtype=torch.bool,
        )
        actor_zeros = torch.zeros_like(actor_mask, dtype=torch.float32)
        actor_controller_state = torch.zeros(
            (
                config.physics.worlds,
                len(TQC_CONTROLLER_STATE_NAMES),
            ),
            device=device,
        )
        tqc_observation = build_tqc_observation(
            env.obs_history[:, -1],
            actor_zeros,
            actor_zeros,
            actor_mask,
            actor_controller_state,
        )
        residual_action = trainer.act(
            tqc_observation, action_mask=actor_mask
        )
        if not torch.isfinite(diffusion_action).all():
            raise FloatingPointError("Diffusion inference produced NaN/Inf")
        if not torch.isfinite(residual_action).all():
            raise FloatingPointError("TQC actor inference produced NaN/Inf")
        peak.sample("full_world_policy_inference")

        zero = torch.zeros(
            (config.physics.worlds, RESIDUAL_ACTION_DIM), device=device
        )
        steps_run = min(physics_steps, config.physics.max_episode_steps)
        for _ in range(steps_run):
            env.step(zero)
        health = _assert_health(env, context="residual probe")

        # Force all three optimiser paths and sample from the real populated
        # host replay.  Adam creates its state lazily on the first step, so a
        # critic-only warm-up probe would understate the production peak.
        trainer.transitions = int(config.tqc.actor_learning_starts)
        last_metrics = None
        for _ in range(2):
            batch = replay.sample(config.tqc.batch_size)
            batch_tensors = (
                batch.observation,
                batch.action,
                batch.reward,
                batch.next_observation,
                batch.action_mask,
                batch.next_action_mask,
                batch.terminated,
                batch.truncated,
            )
            if any(value.device != device for value in batch_tensors):
                raise RuntimeError(
                    "host replay sample was not transferred to the learner device"
                )
            last_metrics = trainer.update(batch)
        trainer.assert_finite()
        peak.sample("host_replay_tqc_actor_critic_alpha_backward_and_adam")

        # Formal resumable checkpoints clone every live world state before
        # transferring it to CPU.  Keep that clone live for the peak sample.
        env_snapshot = env.state_dict()
        peak.sample("environment_checkpoint_snapshot")
        del env_snapshot
        result = peak.finish()
        result.update(
            {
                "worlds": config.physics.worlds,
                "physics_steps": steps_run,
                "replay_capacity": config.tqc.replay_capacity,
                # Retain the original aggregate key for report consumers while
                # making host/device residency explicit.
                "replay_allocated_bytes": replay.storage_allocated_bytes,
                "replay_host_allocated_bytes": replay.storage_allocated_bytes,
                "replay_cuda_persistent_allocated_bytes": 0,
                "replay_cuda_sample_batch_bytes": (
                    replay.sample_allocated_bytes(config.tqc.batch_size)
                ),
                "replay_storage_device": str(replay.storage_device),
                "replay_training_device": str(replay.device),
                "tqc_updates": trainer.updates,
                "critic_loss": float(
                    last_metrics["critic_loss"].detach().cpu()
                ),
                "actor_loss": float(
                    last_metrics["actor_loss"].detach().cpu()
                ),
                "elapsed_seconds": time.monotonic() - started,
                "health": health,
            }
        )
        return result
    finally:
        if env is not None:
            env.close()


def _diffusion_rollout_phase(
    config, torch, device, total_bytes: int, physics_steps: int
):
    """Measure the production full-world Diffusion→safety→IK rollout."""

    peak = PhasePeak(
        torch,
        device,
        name="diffusion_only_rollout",
        total_bytes=total_bytes,
    )
    started = time.monotonic()
    env = None
    try:
        policy = _diffusion_policy(config, device).eval()
        controller = BatchedDiffusionController(
            policy,
            worlds=config.physics.worlds,
            inference_steps=config.diffusion.inference_steps,
            replan_interval=config.diffusion.replan_interval,
            seed=config.runtime.seed,
        )
        env = MJWarpScrewEnv(
            config,
            baseline="diffusion",
            diffusion_controller=controller,
        )
        peak.sample("environment_and_diffusion_controller")

        # Force the same full-world policy inference used after the scripted
        # pickup prefix.  The rollout itself always supplies a zero residual;
        # no replay buffer, actor/critic, or TQC optimizer is constructed.
        controller.cursor = controller.replan_interval
        diffusion_action = controller.act(env.obs_history)
        if not torch.isfinite(diffusion_action).all():
            raise FloatingPointError(
                "Diffusion full-world inference produced NaN/Inf"
            )
        peak.sample("full_world_diffusion_inference")

        zero_residual = torch.zeros(
            (config.physics.worlds, RESIDUAL_ACTION_DIM),
            device=device,
        )
        steps_run = min(
            physics_steps, config.physics.max_episode_steps
        )
        for _ in range(steps_run):
            env.step(zero_residual)
        health = _assert_health(
            env, context="Diffusion-only rollout probe"
        )
        peak.sample("diffusion_safety_ik_physics")
        result = peak.finish()
        result.update(
            {
                "worlds": config.physics.worlds,
                "physics_steps": steps_run,
                "residual_mode": "disabled_zero",
                "elapsed_seconds": time.monotonic() - started,
                "health": health,
            }
        )
        return result
    finally:
        if env is not None:
            env.close()


def _phase_specs(pipeline_mode: str):
    final_label, final_runner = (
        ("diffusion_only_rollout", _diffusion_rollout_phase)
        if pipeline_mode == "diffusion-only"
        else ("diffusion_tqc_residual", _residual_phase)
    )
    return (
        ("expert_physical_canary", _expert_physical_phase),
        ("demonstration_generation", _demo_phase),
        ("diffusion_training", _diffusion_training_phase),
        (final_label, final_runner),
    )


def _run_single_phase(
    config_path: Path,
    *,
    worlds: int | None,
    phase_label: str,
    physics_steps: int,
    pipeline_mode: str,
    max_expert_truncation_fraction: float = 0.0,
) -> dict[str, Any]:
    import torch

    config = load_config(config_path)
    if worlds is not None:
        config = replace(
            config, physics=replace(config.physics, worlds=int(worlds))
        )
    device = torch.device(config.runtime.device)
    _free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    runners = dict(_phase_specs(pipeline_mode))
    runner = runners.get(phase_label)
    if runner is None:
        raise ValueError(
            f"phase {phase_label!r} is invalid for {pipeline_mode}"
        )
    if runner is _diffusion_training_phase:
        return runner(config, torch, device, int(total_bytes))
    if runner in {_expert_physical_phase, _demo_phase}:
        return runner(
            config,
            torch,
            device,
            int(total_bytes),
            physics_steps,
            max_expert_truncation_fraction,
        )
    return runner(
        config,
        torch,
        device,
        int(total_bytes),
        physics_steps,
    )


def _run_phase_subprocess(
    config_path: Path,
    *,
    worlds: int,
    phase_label: str,
    max_memory_fraction: float,
    physics_steps: int,
    pipeline_mode: str,
    max_expert_truncation_fraction: float = 0.0,
) -> dict[str, Any]:
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--worlds",
        str(worlds),
        "--max-memory-fraction",
        str(max_memory_fraction),
        "--physics-steps",
        str(physics_steps),
        "--pipeline-mode",
        pipeline_mode,
        "--max-expert-truncation-fraction",
        str(max_expert_truncation_fraction),
        "--single-phase",
        phase_label,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    parsed = _last_json_object(completed.stdout)
    if (
        completed.returncode != 0
        or parsed is None
        or parsed.get("status") != "pass"
        or parsed.get("single_phase") != phase_label
        or not isinstance(parsed.get("phase"), dict)
    ):
        raise RuntimeError(
            f"formal phase {phase_label} failed in its isolated process; "
            f"returncode={completed.returncode}; "
            f"stdout_tail={completed.stdout[-12000:]!r}; "
            f"stderr_tail={completed.stderr[-12000:]!r}"
        )
    return parsed["phase"]


def run_probe(
    config_path: Path,
    *,
    worlds: int | None,
    max_memory_fraction: float,
    physics_steps: int,
    pipeline_mode: str = "diffusion-tqc",
    max_expert_truncation_fraction: float = 0.0,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to the training-stack probe")
    versions = _versions()
    config = load_config(config_path)
    if worlds is not None:
        config = replace(
            config, physics=replace(config.physics, worlds=int(worlds))
        )
    device = torch.device(config.runtime.device)
    properties = torch.cuda.get_device_properties(device)
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    gpu_type = classify_gpu_type(properties.name)
    if gpu_type == "UNKNOWN":
        raise RuntimeError(
            "allocated GPU is outside the supported training set "
            "(RTX PRO 6000 Blackwell/L40S/A100/RTX6000): "
            f"{properties.name}"
        )

    if pipeline_mode not in {"diffusion-only", "diffusion-tqc"}:
        raise ValueError(
            "pipeline_mode must be diffusion-only or diffusion-tqc"
        )
    phases = []
    for label, _runner in _phase_specs(pipeline_mode):
        print(
            f"training-stack preflight: {label}, worlds={config.physics.worlds}",
            file=sys.stderr,
            flush=True,
        )
        phase = _run_phase_subprocess(
            config_path,
            worlds=config.physics.worlds,
            phase_label=label,
            max_memory_fraction=max_memory_fraction,
            physics_steps=physics_steps,
            pipeline_mode=pipeline_mode,
            max_expert_truncation_fraction=(
                max_expert_truncation_fraction
            ),
        )
        phases.append(phase)

    peak = max(
        int(phase["conservative_peak_used_bytes"]) for phase in phases
    )
    peak_fraction = peak / int(total_bytes)
    status = (
        "pass"
        if peak_fraction <= max_memory_fraction
        else "over_memory_ceiling"
    )
    return {
        "mode": "formal_training_stack",
        "pipeline_mode": pipeline_mode,
        "status": status,
        "platform": platform.platform(),
        "config": config.name,
        "worlds": config.physics.worlds,
        "gpu": {
            "name": properties.name,
            "pbs_gpu_type": gpu_type,
            "total_memory_bytes": int(total_bytes),
            "free_before_probe_bytes": int(free_before),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
            "driver_version": _driver_version(),
            "torch_cuda_build": torch.version.cuda,
        },
        "versions": versions,
        "schema_hash": SCHEMA_HASH,
        "task_geometry_hash": TASK_GEOMETRY_HASH,
        "memory_ceiling_fraction": max_memory_fraction,
        "max_expert_truncation_fraction": (
            max_expert_truncation_fraction
        ),
        "conservative_peak_used_bytes": peak,
        "conservative_peak_used_fraction": peak_fraction,
        "limiting_phase": max(
            phases,
            key=lambda phase: int(phase["conservative_peak_used_bytes"]),
        )["name"],
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "cx3_l40s.json",
    )
    parser.add_argument("--worlds", type=int, default=None)
    parser.add_argument("--max-memory-fraction", type=float, default=0.90)
    parser.add_argument("--physics-steps", type=int, default=96)
    parser.add_argument(
        "--max-expert-truncation-fraction",
        type=float,
        default=0.0,
        help=(
            "Expert-rollout preflight timeout allowance for the physical "
            "canary and demonstration phases; physical failures remain "
            "strictly forbidden (default: 0)."
        ),
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=("diffusion-only", "diffusion-tqc"),
        default="diffusion-tqc",
    )
    parser.add_argument(
        "--single-phase",
        choices=SINGLE_PHASE_CHOICES,
        default=None,
        help=argparse.SUPPRESS,
    )
    add_external_path_flag(parser)
    args = parser.parse_args()
    if args.worlds is not None and args.worlds <= 0:
        parser.error("--worlds must be positive")
    if not 0.50 <= args.max_memory_fraction < 1.0:
        parser.error("--max-memory-fraction must be in [0.50, 1.0)")
    if args.physics_steps <= 0:
        parser.error("--physics-steps must be positive")
    if not 0.0 <= args.max_expert_truncation_fraction <= 1.0:
        parser.error(
            "--max-expert-truncation-fraction must be in [0, 1]"
        )
    try:
        config_path = resolve_package_path(
            args.config,
            allow_external=args.allow_external_paths,
            label="config",
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        if args.single_phase is None:
            result = run_probe(
                config_path,
                worlds=args.worlds,
                max_memory_fraction=args.max_memory_fraction,
                physics_steps=args.physics_steps,
                pipeline_mode=args.pipeline_mode,
                max_expert_truncation_fraction=(
                    args.max_expert_truncation_fraction
                ),
            )
        else:
            phase = _run_single_phase(
                config_path,
                worlds=args.worlds,
                phase_label=args.single_phase,
                physics_steps=args.physics_steps,
                pipeline_mode=args.pipeline_mode,
                max_expert_truncation_fraction=(
                    args.max_expert_truncation_fraction
                ),
            )
            result = {
                "status": "pass",
                "single_phase": args.single_phase,
                "phase": phase,
            }
    except BaseException as exc:
        failure = {
            "mode": "formal_training_stack",
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "worlds": args.worlds,
            "pipeline_mode": args.pipeline_mode,
        }
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        traceback.print_exc()
        raise SystemExit(1)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
