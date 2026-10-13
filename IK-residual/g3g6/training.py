"""Fixed-budget local TQC training for G3 and G6."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from functools import partial
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sb3_contrib import TQC
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch

from . import config
from .env import G3G6ManipulationEnv


class SafeWarmupTQC(TQC):
    """TQC constrained to learn locally around the analytic controller.

    The replay buffer first receives tiny bounded perturbations around zero,
    not destructive full-range random actions.  Between critic and actor start
    thresholds only the critic is trained against a zero-residual target.  Once
    actor updates begin, an explicit action L2 term prevents the policy from
    exploiting unsupported Q estimates far away from the replay data.
    """

    def __init__(
        self,
        *args: Any,
        actor_learning_starts: int | None = None,
        safe_action_limit: float = 1.0,
        action_trust_scale: float = 1.0,
        actor_residual_l2_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        learning_starts = int(kwargs.get("learning_starts", 0))
        self.actor_learning_starts = int(
            config.TQC_ACTOR_LEARNING_STARTS
            if actor_learning_starts is None
            else actor_learning_starts
        )
        self.safe_action_limit = float(safe_action_limit)
        self.action_trust_scale = float(action_trust_scale)
        self.actor_residual_l2_weight = float(actor_residual_l2_weight)
        self._actor_updates = 0
        if self.actor_learning_starts < learning_starts:
            raise ValueError("actor_learning_starts must not precede critic learning")
        if not 0.0 < self.safe_action_limit <= 1.0:
            raise ValueError("safe_action_limit must be inside (0, 1]")
        if not 0.0 < self.action_trust_scale <= 1.0:
            raise ValueError("action_trust_scale must be inside (0, 1]")
        if self.actor_residual_l2_weight < 0.0:
            raise ValueError("actor_residual_l2_weight must be non-negative")
        super().__init__(*args, **kwargs)

    def configure_phase(
        self,
        *,
        actor_learning_starts: int,
        action_trust_scale: float,
    ) -> None:
        """Set the fixed, success-independent trust region for a new phase."""

        value = float(action_trust_scale)
        if not 0.0 < value <= 1.0:
            raise ValueError("action_trust_scale must be inside (0, 1]")
        self.actor_learning_starts = int(actor_learning_starts)
        self.action_trust_scale = value

    def scale_policy_action(self, action: np.ndarray) -> np.ndarray:
        """Map a raw policy action into the action executed and replayed."""

        return np.asarray(action) * self.action_trust_scale

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Any | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps <= learning_starts:
            action = np.zeros(
                (n_envs, *self.action_space.shape), dtype=self.action_space.dtype
            )
            return action, self.policy.scale_action(action)
        return super()._sample_action(learning_starts, action_noise, n_envs)

    @staticmethod
    def _actor_residual_l2(actions: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.square(actions))

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Train the critic first, then a residual-regularized TQC actor."""

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        update_actor = self.num_timesteps >= self.actor_learning_starts
        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        actor_base_losses: list[float] = []
        actor_residual_l2_values: list[float] = []
        critic_losses: list[float] = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.item()))

            actions_pi_raw: torch.Tensor | None = None
            actions_pi: torch.Tensor | None = None
            log_prob: torch.Tensor | None = None
            if update_actor:
                if self.use_sde:
                    self.actor.reset_noise()
                actions_pi_raw, log_prob = self.actor.action_log_prob(
                    replay_data.observations
                )
                actions_pi = actions_pi_raw * self.action_trust_scale
                log_prob = log_prob.reshape(-1, 1)
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    ent_coef_loss = -(
                        self.log_ent_coef
                        * (log_prob + self.target_entropy).detach()
                    ).mean()
                    ent_coef_losses.append(float(ent_coef_loss.item()))
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

            with torch.no_grad():
                if update_actor:
                    next_actions_raw, next_log_prob = self.actor.action_log_prob(
                        replay_data.next_observations
                    )
                    next_actions = next_actions_raw * self.action_trust_scale
                    next_log_prob = next_log_prob.reshape(-1, 1)
                else:
                    next_actions = torch.zeros(
                        (batch_size, self.action_space.shape[0]),
                        dtype=replay_data.actions.dtype,
                        device=replay_data.actions.device,
                    )
                    next_log_prob = torch.zeros(
                        (batch_size, 1),
                        dtype=replay_data.rewards.dtype,
                        device=replay_data.rewards.device,
                    )
                next_quantiles = self.critic_target(
                    replay_data.next_observations, next_actions
                )
                n_target_quantiles = (
                    self.critic.quantiles_total
                    - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                )
                next_quantiles, _ = torch.sort(
                    next_quantiles.reshape(batch_size, -1)
                )
                next_quantiles = next_quantiles[:, :n_target_quantiles]
                target_quantiles = next_quantiles - ent_coef * next_log_prob
                target_quantiles = replay_data.rewards + (
                    1 - replay_data.dones
                ) * discounts * target_quantiles
                target_quantiles.unsqueeze_(dim=1)

            current_quantiles = self.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = quantile_huber_loss(
                current_quantiles,
                target_quantiles,
                sum_over_quantiles=False,
            )
            critic_losses.append(float(critic_loss.item()))
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if update_actor:
                assert (
                    actions_pi_raw is not None
                    and actions_pi is not None
                    and log_prob is not None
                )
                qf_pi = (
                    self.critic(replay_data.observations, actions_pi)
                    .mean(dim=2)
                    .mean(dim=1, keepdim=True)
                )
                actor_base_loss = (ent_coef * log_prob - qf_pi).mean()
                residual_l2 = self._actor_residual_l2(actions_pi_raw)
                actor_loss = (
                    actor_base_loss
                    + self.actor_residual_l2_weight * residual_l2
                )
                actor_base_losses.append(float(actor_base_loss.item()))
                actor_residual_l2_values.append(float(residual_l2.item()))
                actor_losses.append(float(actor_loss.item()))
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.batch_norm_stats, self.batch_norm_stats_target, 1.0
                )

        self._n_updates += gradient_steps
        if update_actor:
            self._actor_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record(
            "train/actor_updates", self._actor_updates, exclude="tensorboard"
        )
        self.logger.record("train/actor_updates_enabled", float(update_actor))
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        if actor_losses:
            self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
            self.logger.record(
                "train/actor_base_loss", float(np.mean(actor_base_losses))
            )
            self.logger.record(
                "train/actor_residual_l2",
                float(np.mean(actor_residual_l2_values)),
            )
        if ent_coef_losses:
            self.logger.record(
                "train/ent_coef_loss", float(np.mean(ent_coef_losses))
            )


def _zero_initialize_actor_mean(model: TQC) -> None:
    """Make the first learned residual policy continuous with zero warm-up."""

    linear_layers = [
        module
        for module in model.policy.actor.mu.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if not linear_layers:
        raise RuntimeError("TQC actor has no linear mean-output layer")
    output = linear_layers[-1]
    with torch.no_grad():
        output.weight.zero_()
        if output.bias is not None:
            output.bias.zero_()


def _monitored_env(gripper: str, phase: int) -> Monitor:
    return Monitor(G3G6ManipulationEnv(gripper=gripper, phase=phase))


def _local_output(path: Path) -> Path:
    value = (config.PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        value.relative_to(config.PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("training output must stay inside the IK-residual folder") from exc
    return value


def _make_vec_env(
    gripper: str,
    phase: config.Phase,
    n_envs: int,
    vec_env_type: str,
    seed: int,
    previous: VecNormalize | None = None,
) -> VecNormalize:
    factories = [partial(_monitored_env, gripper, int(phase)) for _ in range(n_envs)]
    if vec_env_type == "subproc" and n_envs > 1:
        raw = SubprocVecEnv(factories, start_method="spawn")
    else:
        raw = DummyVecEnv(factories)
    raw.seed(seed)
    normalized = VecNormalize(
        raw,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    if previous is not None:
        normalized.obs_rms = deepcopy(previous.obs_rms)
        normalized.ret_rms = deepcopy(previous.ret_rms)
    return normalized


def _terminate_replay_curriculum_boundary(model: TQC) -> None:
    """Prevent an optimize-memory replay transition crossing two phases."""

    replay = model.replay_buffer
    if replay is None or replay.size() <= 0:
        return
    last = (int(replay.pos) - 1) % int(replay.buffer_size)
    replay.dones[last, :] = 1.0
    if hasattr(replay, "timeouts"):
        replay.timeouts[last, :] = 0.0


class LocalCheckpointCallback(BaseCallback):
    """Save model, normalization state, and a tiny local resume record."""

    def __init__(
        self,
        *,
        output_dir: Path,
        vec_env: VecNormalize,
        phase: config.Phase,
        save_every_steps: int,
        n_envs: int,
    ) -> None:
        super().__init__(verbose=0)
        self.output_dir = output_dir
        self.vec_env = vec_env
        self.phase = phase
        self.save_frequency = max(1, int(save_every_steps) // max(1, int(n_envs)))

    def _on_step(self) -> bool:
        if self.n_calls % self.save_frequency != 0:
            return True
        stem = f"phase{int(self.phase)}_step{self.num_timesteps}"
        self.model.save(self.output_dir / f"{stem}_model")
        self.vec_env.save(self.output_dir / f"{stem}_vecnormalize.pkl")
        (self.output_dir / "latest.json").write_text(
            json.dumps(
                {
                    "phase": int(self.phase),
                    "num_timesteps": int(self.num_timesteps),
                    "model": f"{stem}_model.zip",
                    "vecnormalize": f"{stem}_vecnormalize.pkl",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return True


def train_one_gripper(
    *,
    gripper: str,
    phase_steps: Sequence[int],
    n_envs: int,
    vec_env_type: str,
    seed: int,
    device: str,
    output_dir: Path,
    progress_bar: bool,
    checkpoint_every: int,
) -> Path:
    gripper = config.GripperKind(gripper).value
    budgets = tuple(int(value) for value in phase_steps)
    if len(budgets) != 3 or any(value <= 0 for value in budgets):
        raise ValueError("phase_steps must contain three positive budgets")
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if any(budget % n_envs != 0 for budget in budgets):
        raise ValueError(
            "every phase budget must be divisible by n_envs so SB3 executes "
            "the exact requested number of environment steps"
        )
    formal = budgets == tuple(config.PHASE_TIMESTEPS[phase] for phase in config.Phase)
    learning_starts = min(
        config.TQC_CRITIC_LEARNING_STARTS,
        max(100, budgets[0] // 10),
    )
    actor_delay_steps = {
        config.Phase.GRASP_LIFT: min(
            config.TQC_PHASE_ACTOR_DELAY_STEPS[config.Phase.GRASP_LIFT],
            max(120, budgets[0] // 5),
        ),
        config.Phase.TRANSLATE: min(
            config.TQC_PHASE_ACTOR_DELAY_STEPS[config.Phase.TRANSLATE],
            max(16, budgets[1] // 10),
        ),
        config.Phase.ROTATE: min(
            config.TQC_PHASE_ACTOR_DELAY_STEPS[config.Phase.ROTATE],
            max(16, budgets[2] // 10),
        ),
    }
    run_dir = _local_output(output_dir) / gripper
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": "g3g6-local-tqc-v2-hold25-safe-residual",
        "created_at": datetime.now().astimezone().isoformat(),
        "gripper": gripper,
        "objects_random_uniform": list(config.OBJECT_NAMES),
        "phase_steps": list(budgets),
        "formal_fixed_budget": formal,
        "success_rate_gate": False,
        "replay_phase_boundaries_are_terminal": True,
        "replay_reset_at_phase_boundaries": True,
        "position_tolerance_m_per_axis": config.POSITION_TOLERANCE_M_PER_AXIS,
        "angle_tolerance_deg_per_axis": config.ANGLE_TOLERANCE_DEG_PER_AXIS,
        "root_height_m": config.ROOT_HEIGHT_M,
        # Retain the historical key for phase-1/2 manifest readers, and expose
        # the new phase-specific values explicitly.
        "target_grasp_plane_z_m": config.TARGET_GRASP_PLANE_Z_M,
        "target_grasp_plane_z_m_phase1_phase2": config.TARGET_GRASP_PLANE_Z_M,
        "target_grasp_plane_z_m_phase3": (
            config.ROTATION_TARGET_GRASP_PLANE_Z_M
        ),
        "episode_ticks": {
            "phase1": config.PHASE1_EPISODE_TICKS,
            "phase2": config.FORMAL_EPISODE_TICKS,
            "phase3": config.FORMAL_EPISODE_TICKS,
        },
        "timing_segments": {
            "grasp_lift_prefix": {
                "approach": config.APPROACH_TICKS,
                "contact": config.CONTACT_TICKS,
                "lift": config.LIFT_TICKS,
                "post_lift_wait_phase2_phase3": config.PRE_ACTION_HOLD_TICKS,
            },
            "phase1_terminal_hold": config.PHASE1_TERMINAL_HOLD_TICKS,
            "phase2_action": {
                "lead": 300,
                "settle": 300,
                "route": 1_800,
                "terminal_hold": config.POST_ACTION_HOLD_TICKS,
            },
            "phase3_action": {
                "motion": config.ACTION_MOTION_TICKS,
                "terminal_hold": config.POST_ACTION_HOLD_TICKS,
            },
        },
        "observation_includes_final_rotation_goal": True,
        "action": {
            "dimension": config.ACTION_DIM,
            "meaning": "normalized_direct_rotary_actuator_angle_residual",
            "limit_deg": float(np.degrees(config.ANGULAR_RESIDUAL_LIMIT_RAD)),
            "rate_deg_per_tick": float(
                np.degrees(config.ANGULAR_RESIDUAL_RATE_RAD_PER_TICK)
            ),
            "g6_linear_actuators": "nominal_ik_only",
        },
        "exploration": {
            "warmup_action": "exact_zero_residual",
            "learning_starts": learning_starts,
            "zero_warmup_includes_boundary_vector_batch": True,
            "safe_warmup_transitions": (
                (learning_starts // n_envs + 1) * n_envs
            ),
            "actor_mean_initialization": "zero",
            "log_std_init": -5.0,
            "ent_coef": "auto_0.01",
            "gSDE_noise_refresh": "once_per_one_tick_rollout",
        },
        "residual_penalty": {
            "normalization": "actual_residual_divided_by_limit",
            "weight": config.RESIDUAL_PENALTY_WEIGHT,
        },
        "device": device,
        "n_envs": n_envs,
        "seed": seed,
        "completed_phases": [],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    vec_env: VecNormalize | None = None
    model: TQC | None = None
    total_before = 0
    try:
        for phase, budget in zip(config.Phase, budgets):
            previous = vec_env
            vec_env = _make_vec_env(
                gripper,
                phase,
                n_envs,
                vec_env_type,
                seed + int(phase) * 10_000,
                previous=previous,
            )
            if previous is not None:
                previous.close()
            if model is None:
                model = SafeWarmupTQC(
                    "MlpPolicy",
                    vec_env,
                    policy_kwargs={
                        "net_arch": [256, 256],
                        "n_critics": 4,
                        "n_quantiles": 32,
                        "log_std_init": -5.0,
                    },
                    learning_rate=1.0e-4,
                    buffer_size=1_000_000,
                    learning_starts=learning_starts,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=(1, "step"),
                    gradient_steps=8,
                    top_quantiles_to_drop_per_net=2,
                    ent_coef="auto_0.01",
                    use_sde=True,
                    sde_sample_freq=-1,
                    use_sde_at_warmup=False,
                    optimize_memory_usage=True,
                    replay_buffer_kwargs={"handle_timeout_termination": False},
                    seed=seed,
                    device=device,
                    tensorboard_log=str(run_dir / "tensorboard"),
                    verbose=1,
                )
                _zero_initialize_actor_mean(model)
            else:
                _terminate_replay_curriculum_boundary(model)
                model.set_env(vec_env)
            callback = LocalCheckpointCallback(
                output_dir=checkpoints,
                vec_env=vec_env,
                phase=phase,
                save_every_steps=checkpoint_every,
                n_envs=n_envs,
            )
            phase_start_steps = int(model.num_timesteps)
            model.learn(
                total_timesteps=budget,
                reset_num_timesteps=(phase == config.Phase.GRASP_LIFT),
                tb_log_name=f"{gripper}_phase{int(phase)}",
                callback=callback,
                progress_bar=progress_bar,
            )
            actual_phase_steps = int(model.num_timesteps) - phase_start_steps
            if actual_phase_steps != budget:
                raise RuntimeError(
                    f"SB3 executed {actual_phase_steps} steps in phase {int(phase)}; "
                    f"expected exactly {budget}"
                )
            total_before += actual_phase_steps
            phase_stem = f"phase{int(phase)}_final"
            model.save(checkpoints / f"{phase_stem}_model")
            vec_env.save(checkpoints / f"{phase_stem}_vecnormalize.pkl")
            manifest["completed_phases"].append(
                {
                    "phase": int(phase),
                    "budget": budget,
                    "actual_steps": actual_phase_steps,
                    "cumulative_requested_steps": total_before,
                    "model": f"checkpoints/{phase_stem}_model.zip",
                    "vecnormalize": f"checkpoints/{phase_stem}_vecnormalize.pkl",
                }
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        assert model is not None and vec_env is not None
        model.save(run_dir / "final_model")
        vec_env.save(run_dir / "vecnormalize.pkl")
        return run_dir
    finally:
        if vec_env is not None:
            vec_env.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local fixed-budget TQC training for the G3/G6 manipulation task."
    )
    parser.add_argument("--gripper", choices=("g3", "g6", "both"), default="both")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--n-envs", type=int, default=config.DEFAULT_N_ENVS)
    parser.add_argument("--vec-env", choices=("subproc", "dummy"), default="subproc")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument(
        "--phase-steps",
        type=int,
        nargs=3,
        metavar=("PHASE1", "PHASE2", "PHASE3"),
        default=tuple(config.PHASE_TIMESTEPS[phase] for phase in config.Phase),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / datetime.now().strftime("local_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use 128/32/32 steps, one dummy env, and CPU; never a formal run",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_envs <= 0:
        raise SystemExit("--n-envs must be positive")
    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be positive")
    if args.smoke:
        # Phase 1 crosses the inclusive 100-step safe-warmup boundary so this
        # smoke run exercises at least one real TQC gradient update too.
        args.phase_steps = (128, 32, 32)
        args.n_envs = 1
        args.vec_env = "dummy"
        args.device = "cpu"
    grippers = ("g3", "g6") if args.gripper == "both" else (args.gripper,)
    root = _local_output(args.output_dir)
    for offset, gripper in enumerate(grippers):
        run_dir = train_one_gripper(
            gripper=gripper,
            phase_steps=args.phase_steps,
            n_envs=args.n_envs,
            vec_env_type=args.vec_env,
            seed=args.seed + offset * 1_000_000,
            device=args.device,
            output_dir=root,
            progress_bar=args.progress_bar,
            checkpoint_every=args.checkpoint_every,
        )
        print(f"completed {gripper}: {run_dir}")


if __name__ == "__main__":
    main()
