#!/usr/bin/env python3
"""Main-process batched Diffusion inference for vectorized screw workers.

Worker environments run MuJoCo, IK, and residual projection only.  This
``VecEnvWrapper`` owns one frozen Diffusion policy in the main process,
maintains raw observation histories for every worker, batches DDIM inference,
and appends the resulting 14-D task action to each worker's 6-D residual.

The required wrapper order is::

    VecNormalize(
        CentralDiffusionVecEnv(
            VecMonitor(SubprocVecEnv(external_diffusion workers))
        )
    )

Consequently Diffusion receives raw 191-D simulator observations while TQC
receives normalized observations and still exposes only a 6-D action space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class CentralDiffusionVecEnv(VecEnvWrapper):
    """Add one centrally sampled Diffusion action to every worker action."""

    def __init__(
        self,
        venv: VecEnv,
        *,
        checkpoint: Optional[str | Path] = None,
        policy: Optional[Any] = None,
        device: str = "auto",
        inference_steps: int = 8,
        replan_interval: int = 4,
        residual_action_dim: int = 6,
        seed: int = 42,
        use_ema: bool = True,
    ) -> None:
        if (checkpoint is None) == (policy is None):
            raise ValueError("provide exactly one of checkpoint or policy")
        if inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if replan_interval <= 0:
            raise ValueError("replan_interval must be positive")
        if residual_action_dim <= 0:
            raise ValueError("residual_action_dim must be positive")
        if policy is None:
            from diffusion_policy import load_policy_checkpoint

            policy, self.checkpoint_data = load_policy_checkpoint(
                checkpoint,
                device=device,
                use_ema=use_ema,
            )
        else:
            self.checkpoint_data = None
        self.policy = policy
        self.inference_steps = int(inference_steps)
        self.replan_interval = int(replan_interval)
        self.residual_action_dim = int(residual_action_dim)
        self.seed_value = int(seed)

        required_attributes = (
            "obs_dim",
            "obs_horizon",
            "action_dim",
            "horizon",
            "num_diffusion_steps",
            "sample",
        )
        missing = [name for name in required_attributes if not hasattr(policy, name)]
        if missing:
            raise TypeError(f"Diffusion policy is missing attributes: {missing}")
        self.obs_dim = int(policy.obs_dim)
        self.obs_horizon = int(policy.obs_horizon)
        self.diffusion_action_dim = int(policy.action_dim)
        self.plan_horizon = int(policy.horizon)
        if self.inference_steps > int(policy.num_diffusion_steps):
            raise ValueError(
                "inference_steps exceeds the checkpoint diffusion schedule"
            )
        if self.replan_interval > self.plan_horizon:
            raise ValueError(
                "replan_interval cannot exceed the predicted action horizon"
            )

        observation_space = venv.observation_space
        if not isinstance(observation_space, gym.spaces.Box) or (
            observation_space.shape != (self.obs_dim,)
        ):
            raise ValueError(
                "underlying observation space must be a flat Box matching "
                f"policy obs_dim={self.obs_dim}, got {observation_space}"
            )
        combined_dim = self.residual_action_dim + self.diffusion_action_dim
        worker_action_space = venv.action_space
        if not isinstance(worker_action_space, gym.spaces.Box) or (
            worker_action_space.shape != (combined_dim,)
        ):
            raise ValueError(
                "external-diffusion workers must accept "
                f"{self.residual_action_dim}D residual + "
                f"{self.diffusion_action_dim}D diffusion = {combined_dim}D, "
                f"got {worker_action_space}"
            )
        external_action_space = gym.spaces.Box(
            low=np.asarray(worker_action_space.low[: self.residual_action_dim]),
            high=np.asarray(worker_action_space.high[: self.residual_action_dim]),
            dtype=np.float32,
        )
        super().__init__(
            venv,
            observation_space=observation_space,
            action_space=external_action_space,
        )

        self._histories: Optional[np.ndarray] = None
        self._plan: Optional[np.ndarray] = None
        self._plan_offset = 0
        self._force_replan = True
        self._waiting = False
        self._last_combined_action: Optional[np.ndarray] = None
        self.plan_count = 0

    def _validate_observations(self, observations: Any) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float32)
        expected = (self.num_envs, self.obs_dim)
        if values.shape != expected:
            raise ValueError(
                f"vector observations must have shape {expected}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("raw worker observation contains NaN or infinity")
        return values

    def _initialize_histories(self, observations: np.ndarray) -> None:
        self._histories = np.repeat(
            observations[:, None, :],
            self.obs_horizon,
            axis=1,
        )

    def _append_histories(
        self,
        observations: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        assert self._histories is not None
        for index in range(self.num_envs):
            if bool(dones[index]):
                # SubprocVecEnv/DummyVecEnv already returned the automatic-reset
                # observation.  No terminal state may leak into the new history.
                self._histories[index] = observations[index][None, :]
            else:
                self._histories[index, :-1] = self._histories[index, 1:]
                self._histories[index, -1] = observations[index]

    def _sample_plan(self) -> None:
        if self._histories is None:
            raise RuntimeError("reset() must be called before stepping")
        sampled = self.policy.sample(
            self._histories,
            num_inference_steps=self.inference_steps,
            seed=self.seed_value + self.plan_count,
        )
        if hasattr(sampled, "detach"):
            sampled = sampled.detach().cpu().numpy()
        values = np.asarray(sampled, dtype=np.float32)
        expected = (
            self.num_envs,
            self.plan_horizon,
            self.diffusion_action_dim,
        )
        if values.shape != expected:
            raise ValueError(
                f"batched Diffusion sample must have shape {expected}, "
                f"got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(
                "batched Diffusion sample contains NaN or infinity"
            )
        self._plan = values
        self._plan_offset = 0
        self._force_replan = False
        self.plan_count += 1

    def reset(self) -> np.ndarray:
        observations = self._validate_observations(self.venv.reset())
        self._initialize_histories(observations)
        self._plan = None
        self._plan_offset = 0
        self._force_replan = True
        self._waiting = False
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        if self._waiting:
            raise RuntimeError("step_async called while another step is pending")
        residual = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, self.residual_action_dim)
        if residual.shape != expected:
            raise ValueError(
                f"residual action batch must have shape {expected}, "
                f"got {residual.shape}"
            )
        if not np.all(np.isfinite(residual)):
            raise ValueError("residual action contains NaN or infinity")
        if (
            self._force_replan
            or self._plan is None
            or self._plan_offset >= self.replan_interval
            or self._plan_offset >= self.plan_horizon
        ):
            self._sample_plan()
        assert self._plan is not None
        diffusion = self._plan[:, self._plan_offset, :]
        combined = np.concatenate((residual, diffusion), axis=1).astype(
            np.float32,
            copy=False,
        )
        if not np.all(np.isfinite(combined)):
            raise FloatingPointError("combined worker action is non-finite")
        self._last_combined_action = combined.copy()
        self._plan_offset += 1
        self._waiting = True
        self.venv.step_async(combined)

    def step_wait(self):
        if not self._waiting:
            raise RuntimeError("step_wait called without a pending step_async")
        observations, rewards, dones, infos = self.venv.step_wait()
        self._waiting = False
        observations = self._validate_observations(observations)
        done_values = np.asarray(dones, dtype=bool)
        if done_values.shape != (self.num_envs,):
            raise ValueError("worker done flags have an invalid shape")
        self._append_histories(observations, done_values)
        if np.any(done_values):
            # All plans are coupled to one batched sampling call.  Replanning
            # globally keeps batch shapes static and resets the finished
            # worker's history without leaving stale actions in another plan.
            self._plan = None
            self._plan_offset = 0
            self._force_replan = True
        return observations, rewards, dones, infos

    @property
    def last_combined_action(self) -> Optional[np.ndarray]:
        if self._last_combined_action is None:
            return None
        return self._last_combined_action.copy()


__all__ = ["CentralDiffusionVecEnv"]
