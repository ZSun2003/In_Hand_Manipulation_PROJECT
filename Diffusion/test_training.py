#!/usr/bin/env python3
"""Fast training-stack tests that do not run the full MuJoCo screw task."""

from __future__ import annotations

import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)
import torch

from central_diffusion import CentralDiffusionVecEnv
import evaluate
import run_screw
import train_residual
from runtime import PROJECT_ROOT, ensure_runtime_dirs


SCREW_OBSERVATION_DIM = 191


class TinySixDimensionalEnv(gym.Env[np.ndarray, np.ndarray]):
    """Deterministic continuous environment used only for TQC serialization."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(5,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )
        self._state = np.zeros(5, dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state.fill(0.0)
        self._step = 0
        return self._state.copy(), {}

    def step(self, action):
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("tiny environment requires one finite 6-D action")
        self._state[:4] = np.tanh(
            self._state[:4] + 0.05 * value[:4]
        )
        self._state[4] = self._step / 4.0
        self._step += 1
        reward = -float(np.square(value).mean())
        terminated = self._step >= 4
        return self._state.copy(), reward, terminated, False, {}


class TinyExternalDiffusionWorker(gym.Env[np.ndarray, np.ndarray]):
    """A subprocess worker that accepts residual+Diffusion, but owns no policy."""

    metadata = {"render_modes": []}

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = int(rank)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(SCREW_OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.concatenate(
                (
                    np.full(6, -1.0, dtype=np.float32),
                    np.full(14, -np.inf, dtype=np.float32),
                )
            ),
            high=np.concatenate(
                (
                    np.full(6, 1.0, dtype=np.float32),
                    np.full(14, np.inf, dtype=np.float32),
                )
            ),
            dtype=np.float32,
        )
        self._step = 0

    def _observation(self) -> np.ndarray:
        result = np.zeros(SCREW_OBSERVATION_DIM, dtype=np.float32)
        result[0] = self.rank
        result[1] = self._step
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self._observation(), {}

    def step(self, action):
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (20,) or not np.all(np.isfinite(value)):
            raise ValueError("worker requires finite 6D+14D action")
        self._step += 1
        terminated = self._step >= 2
        return (
            self._observation(),
            -float(np.square(value[:6]).mean()),
            terminated,
            False,
            {
                "received_action_shape": value.shape,
                "received_action_finite": bool(np.all(np.isfinite(value))),
                "diffusion_action_finite": bool(
                    np.all(np.isfinite(value[6:]))
                ),
            },
        )

    def worker_status(self):
        return {
            "diffusion_policy_loaded": False,
            "action_shape": self.action_space.shape,
            "rank": self.rank,
        }


class FakeBatchDiffusionPolicy:
    obs_dim = SCREW_OBSERVATION_DIM
    obs_horizon = 2
    action_dim = 14
    horizon = 8
    num_diffusion_steps = 8

    def __init__(self) -> None:
        self.sample_calls = 0
        self.last_history_shape = None

    def sample(self, histories, *, num_inference_steps, seed):
        values = np.asarray(histories, dtype=np.float32)
        self.last_history_shape = values.shape
        self.sample_calls += 1
        plan = np.zeros(
            (values.shape[0], self.horizon, self.action_dim),
            dtype=np.float32,
        )
        for step in range(self.horizon):
            plan[:, step, :] = (
                1.0e-4 * self.sample_calls + 1.0e-5 * step
            )
        return plan


class TrainingConfigurationTests(unittest.TestCase):
    def test_tqc_parser_defaults_and_entrypoint_names(self) -> None:
        args = train_residual.build_parser().parse_args([])
        self.assertEqual(args.output.name, "screw_residual_tqc")
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.baseline, "diffusion")
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.buffer_size, 1_000_000)
        self.assertEqual(args.learning_starts, 10_000)
        self.assertEqual(args.train_freq, 1)
        self.assertEqual(args.gradient_steps, -1)
        self.assertAlmostEqual(args.learning_rate, 3.0e-4)
        self.assertAlmostEqual(args.tau, 0.005)
        self.assertAlmostEqual(args.gamma, 0.995)
        self.assertEqual(args.n_critics, 2)
        self.assertEqual(args.n_quantiles, 25)
        self.assertEqual(args.top_quantiles_to_drop, 2)
        self.assertEqual(args.ent_coef, "auto_0.1")
        self.assertTrue(args.save_replay_buffer)
        self.assertEqual(args.diffusion_device, "auto")
        self.assertEqual(args.diffusion_inference_steps, 8)
        self.assertEqual(args.diffusion_replan_interval, 4)

        run_args = run_screw.build_parser().parse_args([])
        eval_args = evaluate.build_parser().parse_args([])
        for parsed in (run_args, eval_args):
            self.assertEqual(
                parsed.residual_checkpoint.name,
                "screw_residual_tqc.zip",
            )
            self.assertEqual(
                parsed.vecnormalize.name,
                "screw_residual_tqc.vecnormalize.pkl",
            )
        self.assertIs(train_residual.TQC, TQC)
        self.assertIs(evaluate.TQC, TQC)

    def test_select_device_auto_can_be_monkeypatched(self) -> None:
        with (
            mock.patch.object(
                torch.backends.mps, "is_available", return_value=True
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
        ):
            self.assertEqual(train_residual.select_device("auto"), "mps")
        with (
            mock.patch.object(
                torch.backends.mps, "is_available", return_value=False
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
        ):
            self.assertEqual(train_residual.select_device("auto"), "cuda")
        with (
            mock.patch.object(
                torch.backends.mps, "is_available", return_value=False
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
        ):
            self.assertEqual(train_residual.select_device("auto"), "cpu")
            self.assertEqual(train_residual.select_device("cpu"), "cpu")
            with self.assertRaisesRegex(RuntimeError, "MPS"):
                train_residual.select_device("mps")

    def test_periodic_snapshot_sidecars_resolve_for_resume(self) -> None:
        runtime_dirs = ensure_runtime_dirs()
        with tempfile.TemporaryDirectory(
            prefix="snapshot_names_",
            dir=runtime_dirs["tmp"],
        ) as temporary:
            root = Path(temporary)
            callback = CheckpointCallback(
                save_freq=1,
                save_path=str(root),
                name_prefix="screw_residual_tqc",
                save_replay_buffer=True,
                save_vecnormalize=True,
            )
            callback.num_timesteps = 100
            model_path = Path(callback._checkpoint_path(extension="zip"))
            callback_vec = Path(
                callback._checkpoint_path(
                    "vecnormalize_",
                    extension="pkl",
                )
            )
            callback_replay = Path(
                callback._checkpoint_path(
                    "replay_buffer_",
                    extension="pkl",
                )
            )
            model_path.touch()
            callback_vec.touch()
            callback_replay.touch()
            self.assertEqual(
                train_residual._resume_sidecar_path(
                    model_path,
                    kind="vecnormalize",
                ),
                callback_vec,
            )
            self.assertEqual(
                train_residual._resume_sidecar_path(
                    model_path,
                    kind="replay_buffer",
                ),
                callback_replay,
            )


class TinyTQCSerializationTests(unittest.TestCase):
    def test_six_dimensional_tqc_train_save_load_predict_and_resume_files(
        self,
    ) -> None:
        runtime_dirs = ensure_runtime_dirs()
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(
                prefix="tqc_training_test_",
                dir=runtime_dirs["tmp"],
            ) as temporary:
                root = Path(temporary)
                vector = VecNormalize(
                    DummyVecEnv([TinySixDimensionalEnv]),
                    training=True,
                    norm_obs=True,
                    norm_reward=True,
                    gamma=0.99,
                )
                model = TQC(
                    "MlpPolicy",
                    vector,
                    learning_rate=1.0e-3,
                    buffer_size=32,
                    learning_starts=0,
                    batch_size=2,
                    train_freq=(1, "step"),
                    gradient_steps=1,
                    ent_coef=0.01,
                    top_quantiles_to_drop_per_net=1,
                    policy_kwargs={
                        "net_arch": {"pi": [16], "qf": [16]},
                        "n_critics": 1,
                        "n_quantiles": 5,
                    },
                    device="cpu",
                    seed=17,
                    verbose=0,
                )
                model.learn(total_timesteps=4, progress_bar=False)
                self.assertGreater(model.replay_buffer.size(), 0)
                observation = vector.reset()
                action, _ = model.predict(observation, deterministic=True)
                self.assertEqual(action.shape, (1, 6))
                self.assertTrue(np.all(np.isfinite(action)))

                output = root / "tiny_residual_tqc"
                model_path = output.with_suffix(".zip")
                normalize_path = output.with_suffix(".vecnormalize.pkl")
                replay_path = output.with_suffix(".replay_buffer.pkl")
                model.save(output)
                model.save_replay_buffer(replay_path)
                vector.save(normalize_path)
                self.assertTrue(model_path.is_file())
                self.assertTrue(normalize_path.is_file())
                self.assertTrue(replay_path.is_file())
                vector.close()

                resumed_vector = VecNormalize.load(
                    normalize_path,
                    DummyVecEnv([TinySixDimensionalEnv]),
                )
                resumed_vector.training = True
                resumed_model = TQC.load(
                    model_path,
                    env=resumed_vector,
                    device="cpu",
                )
                resumed_model.load_replay_buffer(replay_path)
                self.assertGreater(resumed_model.replay_buffer.size(), 0)
                resumed_observation = resumed_vector.reset()
                resumed_action, _ = resumed_model.predict(
                    resumed_observation,
                    deterministic=True,
                )
                self.assertEqual(resumed_action.shape, (1, 6))
                self.assertTrue(np.all(np.isfinite(resumed_action)))
                resumed_vector.close()
        finally:
            torch.set_num_threads(old_threads)


class CentralDiffusionWorkerTests(unittest.TestCase):
    def test_two_real_external_workers_do_not_import_diffusion_policy(
        self,
    ) -> None:
        workers = train_residual.make_vector_env(
            workers=2,
            seed=31,
            baseline="external_diffusion",
            diffusion_checkpoint=None,
            guide_mode="mocap",
            guide_scale=1.0,
            random_spawn=False,
        )
        fake_policy = FakeBatchDiffusionPolicy()
        central = CentralDiffusionVecEnv(
            workers,
            policy=fake_policy,
            device="cpu",
            inference_steps=4,
            replan_interval=4,
            residual_action_dim=6,
            seed=31,
        )
        try:
            observations = central.reset()
            self.assertEqual(
                observations.shape, (2, SCREW_OBSERVATION_DIM)
            )
            statuses = central.env_method("diffusion_worker_status")
            self.assertTrue(
                all(item["external_diffusion"] for item in statuses)
            )
            self.assertTrue(
                all(not item["diffusion_policy_loaded"] for item in statuses)
            )
            self.assertTrue(
                all(
                    not item["diffusion_policy_module_loaded"]
                    for item in statuses
                )
            )
            self.assertTrue(
                all(tuple(item["action_shape"]) == (20,) for item in statuses)
            )
            next_observations, rewards, _, infos = central.step(
                np.zeros((2, 6), dtype=np.float32)
            )
            self.assertEqual(
                next_observations.shape, (2, SCREW_OBSERVATION_DIM)
            )
            self.assertTrue(np.all(np.isfinite(next_observations)))
            self.assertTrue(np.all(np.isfinite(rewards)))
            self.assertTrue(
                all(info["external_diffusion"] for info in infos)
            )
            self.assertTrue(
                all(not info["diffusion_policy_loaded"] for info in infos)
            )
            self.assertEqual(central.last_combined_action.shape, (2, 20))
            self.assertTrue(np.all(np.isfinite(central.last_combined_action)))
        finally:
            central.close()

    def test_two_worker_batching_has_no_worker_policy_and_replans_on_done(
        self,
    ) -> None:
        workers = SubprocVecEnv(
            [
                partial(TinyExternalDiffusionWorker, rank=0),
                partial(TinyExternalDiffusionWorker, rank=1),
            ],
            start_method="spawn",
        )
        fake_policy = FakeBatchDiffusionPolicy()
        central = CentralDiffusionVecEnv(
            workers,
            policy=fake_policy,
            device="cpu",
            inference_steps=4,
            replan_interval=4,
            residual_action_dim=6,
            seed=23,
        )
        try:
            self.assertEqual(central.action_space.shape, (6,))
            self.assertEqual(central.venv.action_space.shape, (20,))
            observations = central.reset()
            self.assertEqual(
                observations.shape, (2, SCREW_OBSERVATION_DIM)
            )
            statuses = central.env_method("worker_status")
            self.assertEqual(len(statuses), 2)
            self.assertTrue(
                all(not item["diffusion_policy_loaded"] for item in statuses)
            )
            self.assertTrue(
                all(tuple(item["action_shape"]) == (20,) for item in statuses)
            )

            residual = np.zeros((2, 6), dtype=np.float32)
            _, _, first_done, first_infos = central.step(residual)
            self.assertFalse(np.any(first_done))
            self.assertEqual(fake_policy.sample_calls, 1)
            self.assertEqual(
                fake_policy.last_history_shape,
                (2, 2, SCREW_OBSERVATION_DIM),
            )
            self.assertTrue(
                all(info["received_action_finite"] for info in first_infos)
            )
            self.assertTrue(
                all(info["diffusion_action_finite"] for info in first_infos)
            )
            self.assertEqual(central.last_combined_action.shape, (2, 20))
            self.assertTrue(np.all(np.isfinite(central.last_combined_action)))

            _, _, second_done, _ = central.step(residual)
            self.assertTrue(np.all(second_done))
            # Any worker reset invalidates the shared batch plan.  The next
            # step therefore makes one new global batch call, not two worker
            # policy calls.
            central.step(residual)
            self.assertEqual(fake_policy.sample_calls, 2)
            self.assertEqual(
                fake_policy.last_history_shape,
                (2, 2, SCREW_OBSERVATION_DIM),
            )
        finally:
            central.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
