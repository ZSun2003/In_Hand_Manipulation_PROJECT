from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from diffusion_policy import DiffusionPolicy
import s_hpc.replay as replay_module
import s_hpc.workflows as workflows_module
from s_hpc.checkpointing import sha256_file
from s_hpc.config import load_config
from s_hpc.contracts import (
    OBS_DIM,
    RESIDUAL_ACTION_DIM,
    TQC_CONTROLLER_STATE_NAMES,
    TQC_OBS_DIM,
    checkpoint_metadata,
)
from s_hpc.demos import DemoStore, DemoWindowSampler
from s_hpc.diffusion_runtime import BatchedDiffusionController
from s_hpc.replay import CUDAReplayBuffer, REPLAY_FORMAT
from s_hpc.tqc import TQCTrainer, build_tqc_observation
from s_hpc.workflows import (
    MJWarpHealthError,
    _assert_tqc_continuation_health,
    _capture_rng,
    _commit_checkpoint_cursor,
    _restore_rng,
    _restore_pending_replay,
    _seed_everything,
    _terminal_tqc_observation,
    _validate_checkpoint_config,
    _validate_diffusion_dependency,
    train_diffusion,
)


ROOT = Path(__file__).resolve().parents[1]


class LearningComponentsTest(unittest.TestCase):
    def test_pending_tqc_replay_checkpoint_contract(self):
        count = 5
        payload = {
            "world_ids": torch.arange(count, dtype=torch.long),
            "observation": torch.zeros((count, TQC_OBS_DIM)),
            "action": torch.zeros((count, RESIDUAL_ACTION_DIM)),
            "reward": torch.zeros(count),
            "action_mask": torch.ones(
                (count, RESIDUAL_ACTION_DIM), dtype=torch.bool
            ),
            "terminated": torch.zeros(count, dtype=torch.bool),
            "truncated": torch.zeros(count, dtype=torch.bool),
        }
        restored = _restore_pending_replay(payload, device="cpu")
        self.assertEqual(restored["observation"].shape, (count, TQC_OBS_DIM))
        self.assertEqual(restored["action"].shape, (count, RESIDUAL_ACTION_DIM))

    def test_terminal_tqc_state_keeps_terminal_base_and_realised_residual(self):
        count = 3
        current = torch.randn((count, TQC_OBS_DIM))
        terminal_base = torch.randn((count, OBS_DIM))
        realised = torch.randn((count, RESIDUAL_ACTION_DIM))
        result = type("Result", (), {})()
        result.observation = terminal_base
        result.info = {
            "tqc_observation": current,
            "normalised_realised_actuator_residual": realised,
        }
        terminal = _terminal_tqc_observation(result)
        torch.testing.assert_close(terminal[:, :OBS_DIM], terminal_base)
        previous = slice(
            OBS_DIM + RESIDUAL_ACTION_DIM,
            OBS_DIM + 2 * RESIDUAL_ACTION_DIM,
        )
        torch.testing.assert_close(terminal[:, previous], realised)

    def test_tqc_observation_augments_without_changing_base_schema(self):
        count = 3
        base = torch.randn((count, OBS_DIM))
        nominal = torch.randn((count, RESIDUAL_ACTION_DIM))
        previous = torch.randn((count, RESIDUAL_ACTION_DIM))
        mask = torch.randint(
            0, 2, (count, RESIDUAL_ACTION_DIM), dtype=torch.bool
        )
        controller = torch.rand((count, len(TQC_CONTROLLER_STATE_NAMES)))
        state = build_tqc_observation(
            base, nominal, previous, mask, controller
        )
        self.assertEqual(state.shape, (count, TQC_OBS_DIM))
        torch.testing.assert_close(state[:, :OBS_DIM], base)
        torch.testing.assert_close(
            state[:, OBS_DIM + 2 * RESIDUAL_ACTION_DIM:OBS_DIM + 3 * RESIDUAL_ACTION_DIM],
            mask.to(base.dtype),
        )

    def test_same_safe_cursor_is_checkpointed_only_once(self):
        events = []
        cursor = (250_000, 17, 42)
        saved_cursor, wrote = _commit_checkpoint_cursor(
            cursor,
            None,
            validate=lambda: events.append("validate"),
            write=lambda: events.append("write"),
        )
        self.assertTrue(wrote)
        self.assertEqual(saved_cursor, cursor)
        self.assertEqual(events, ["validate", "write"])

        saved_cursor, wrote = _commit_checkpoint_cursor(
            cursor,
            saved_cursor,
            validate=lambda: events.append("unexpected_validate"),
            write=lambda: events.append("unexpected_write"),
        )
        self.assertFalse(wrote)
        self.assertEqual(saved_cursor, cursor)
        self.assertEqual(events, ["validate", "write"])

    def test_tqc_interrupt_health_failure_cannot_commit_cursor(self):
        class Physics:
            njmax = 100

        class Config:
            physics = Physics()

        class Backend:
            def health(self):
                return {
                    "finite": False,
                    "sticky_nonfinite": 1,
                    "sticky_external_nonfinite": 0,
                    "overflow_worlds": 0,
                    "sticky_overflow": 0,
                    "contacts_total": 0,
                    "contact_capacity_total": 100,
                    "sticky_contacts_full": 0,
                    "max_nefc": 0,
                    "sticky_constraints_full": 0,
                }

        class Env:
            backend = Backend()
            config = Config()

        class Replay:
            finite_checks = 0

            def assert_finite(self):
                self.finite_checks += 1

        replay = Replay()
        writes = []
        saved_cursor = ("last-good",)
        with self.assertRaises(MJWarpHealthError):
            _commit_checkpoint_cursor(
                (250_000,),
                saved_cursor,
                validate=lambda: _assert_tqc_continuation_health(
                    Env(),
                    replay,
                    context="test interrupt",
                ),
                write=lambda: writes.append("unsafe overwrite"),
            )
        self.assertEqual(saved_cursor, ("last-good",))
        self.assertEqual(writes, [])
        self.assertEqual(replay.finite_checks, 0)

    def test_tqc_interrupt_rejects_nonfinite_replay_after_backend_gate(self):
        class Physics:
            njmax = 100

        class Config:
            physics = Physics()

        class Backend:
            def health(self):
                return {
                    "finite": True,
                    "sticky_nonfinite": 0,
                    "sticky_external_nonfinite": 0,
                    "overflow_worlds": 0,
                    "sticky_overflow": 0,
                    "contacts_total": 0,
                    "contact_capacity_total": 100,
                    "sticky_contacts_full": 0,
                    "max_nefc": 0,
                    "sticky_constraints_full": 0,
                }

        class Env:
            backend = Backend()
            config = Config()

        class Replay:
            def assert_finite(self):
                raise FloatingPointError("non-finite replay")

        writes = []
        with self.assertRaisesRegex(FloatingPointError, "non-finite replay"):
            _commit_checkpoint_cursor(
                (500_000,),
                None,
                validate=lambda: _assert_tqc_continuation_health(
                    Env(),
                    Replay(),
                    context="test interrupt",
                ),
                write=lambda: writes.append("unsafe overwrite"),
            )
        self.assertEqual(writes, [])

    def test_diffusion_sampling_stays_on_policy_device(self):
        policy = DiffusionPolicy(
            obs_dim=OBS_DIM,
            action_dim=14,
            obs_horizon=2,
            horizon=16,
            num_diffusion_steps=10,
            hidden_dim=32,
            depth=1,
        )
        sample = policy.sample(
            torch.zeros((3, 2, OBS_DIM)),
            num_inference_steps=4,
            seed=7,
        )
        self.assertEqual(sample.shape, (3, 16, 14))
        self.assertEqual(sample.device, next(policy.parameters()).device)

    def test_partial_reset_does_not_replan_unrelated_worlds(self):
        policy = DiffusionPolicy(
            obs_dim=OBS_DIM,
            action_dim=14,
            obs_horizon=2,
            horizon=8,
            num_diffusion_steps=10,
            hidden_dim=32,
            depth=1,
        )
        controller = BatchedDiffusionController(
            policy,
            worlds=3,
            inference_steps=2,
            replan_interval=4,
            seed=9,
        )
        history = torch.zeros((3, 2, OBS_DIM))
        controller.reset(torch.ones(3, dtype=torch.bool), history)
        controller.act(history)
        unaffected_plan = controller.plan[1:].clone()
        reset_mask = torch.tensor((True, False, False))
        changed_history = history.clone()
        changed_history[0] = 1.0
        controller.reset(reset_mask, changed_history)
        action = controller.act(history)
        torch.testing.assert_close(
            controller.plan[1:], unaffected_plan
        )
        torch.testing.assert_close(
            action[1:], unaffected_plan[:, 1]
        )
        torch.testing.assert_close(
            action[0], controller.override_plan[0, 0]
        )

    def test_diffusion_controller_checkpoint_continues_plan_cursor(self):
        policy = DiffusionPolicy(
            obs_dim=OBS_DIM,
            action_dim=14,
            obs_horizon=2,
            horizon=8,
            num_diffusion_steps=10,
            hidden_dim=32,
            depth=1,
        )
        history = torch.zeros((3, 2, OBS_DIM))
        original = BatchedDiffusionController(
            policy,
            worlds=3,
            inference_steps=2,
            replan_interval=4,
            seed=31,
        )
        original.reset(torch.ones(3, dtype=torch.bool), history)
        original.act(history)
        state = original.state_dict()
        expected = original.act(history)
        restored = BatchedDiffusionController(
            policy,
            worlds=3,
            inference_steps=2,
            replan_interval=4,
            seed=31,
        )
        restored.load_state_dict(state)
        actual = restored.act(history)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(restored.cursor, original.cursor)
        self.assertEqual(restored.sample_count, original.sample_count)

    def test_demo_store_resume_and_window_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DemoStore(
                directory,
                episodes=2,
                steps=20,
                create=True,
                provenance={"policy_start_step": 0},
            )
            observation = np.random.randn(2, 21, OBS_DIM).astype(np.float16)
            action = np.random.randn(2, 20, 14).astype(np.float16)
            spawn = np.zeros((2, 2), dtype=np.float32)
            socket_mouth = np.asarray(
                ((0.020, 0.0), (0.0, 0.020)), dtype=np.float32
            )
            socket_angle = np.full((2,), 45.0, dtype=np.float32)
            store.write(
                0,
                observation,
                action,
                spawn,
                socket_mouth_xy=socket_mouth,
                socket_angle_deg=socket_angle,
            )
            reopened = DemoStore(
                directory,
                episodes=2,
                steps=20,
                create=True,
                provenance={"policy_start_step": 0},
            )
            self.assertEqual(reopened.completed, 2)
            np.testing.assert_allclose(
                reopened.socket_mouth_xy, socket_mouth
            )
            np.testing.assert_allclose(
                reopened.socket_angle_deg, socket_angle
            )
            sampler = DemoWindowSampler(directory, seed=1)
            obs, actions = sampler.sample(4, 16, device="cpu")
            self.assertEqual(obs.shape, (4, 2, OBS_DIM))
            self.assertEqual(actions.shape, (4, 16, 14))
            stats = sampler.normalization()
            self.assertTrue(np.all(stats["obs_std"] > 0.0))
            sampler.cache_on_device("cpu", chunk_episodes=1)
            cached_obs, cached_actions = sampler.sample(
                4, 16, device="cpu"
            )
            self.assertEqual(cached_obs.dtype, torch.float32)
            self.assertEqual(cached_actions.shape, (4, 16, 14))

    def test_demo_sampler_excludes_scripted_pickup_prefix(self):
        policy_start_step = 215
        steps = 240
        with tempfile.TemporaryDirectory() as directory:
            store = DemoStore(
                directory,
                episodes=1,
                steps=steps,
                create=True,
                provenance={"policy_start_step": policy_start_step},
            )
            observation = np.zeros(
                (1, steps + 1, OBS_DIM), dtype=np.float16
            )
            action = np.zeros((1, steps, 14), dtype=np.float16)
            observation[0, :, 0] = np.arange(
                steps + 1, dtype=np.float16
            )
            action[0, :, 0] = np.arange(steps, dtype=np.float16)
            store.write(
                0,
                observation,
                action,
                np.zeros((1, 2), dtype=np.float32),
                socket_mouth_xy=np.asarray(
                    ((0.020, 0.0),), dtype=np.float32
                ),
                socket_angle_deg=np.asarray((45.0,), dtype=np.float32),
            )

            sampler = DemoWindowSampler(directory, seed=7)
            self.assertEqual(sampler.policy_start_step, policy_start_step)
            sampled_obs, sampled_action = sampler.sample(
                64, 16, device="cpu"
            )
            self.assertTrue(
                bool((sampled_obs[:, 1, 0] >= policy_start_step).all())
            )
            self.assertTrue(
                bool((sampled_action[:, 0, 0] >= policy_start_step).all())
            )
            stats = sampler.normalization()
            self.assertAlmostEqual(float(stats["action_mean"][0]), 227.0)
            self.assertAlmostEqual(float(stats["obs_mean"][0]), 227.5)

    def test_variable_length_policy_rollouts_exclude_padded_windows(self):
        policy_start_step = 5
        steps = 30
        valid_steps = np.asarray((20, 30), dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            store = DemoStore(
                directory,
                episodes=2,
                steps=steps,
                create=True,
                provenance={
                    "policy_start_step": policy_start_step,
                    "dataset_role": (
                        "successful_policy_rollout_self_distillation"
                    ),
                    "expert_demonstrations": False,
                },
                variable_episode_lengths=True,
            )
            observation = np.zeros(
                (2, steps + 1, OBS_DIM), dtype=np.float16
            )
            action = np.zeros((2, steps, 14), dtype=np.float16)
            observation[:, :, 0] = np.arange(
                steps + 1, dtype=np.float16
            )
            action[:, :, 0] = np.arange(steps, dtype=np.float16)
            # Poison the first episode's padded tail.  Neither sampling nor
            # normalization is allowed to observe these values.
            observation[0, 21:, 0] = np.float16(1000.0)
            action[0, 20:, 0] = np.float16(1000.0)
            store.write(
                0,
                observation,
                action,
                np.zeros((2, 2), dtype=np.float32),
                valid_steps=valid_steps,
            )
            self.assertIn(
                "valid_steps.npy", store.metadata["content_sha256"]
            )

            sampler = DemoWindowSampler(directory, seed=11)
            self.assertEqual(sampler.window_count(4), 34)
            sampled_obs, sampled_action = sampler.sample(
                2048, 4, device="cpu"
            )
            self.assertTrue(bool((sampled_obs[:, 1, 0] < 1000.0).all()))
            self.assertTrue(bool((sampled_action[:, :, 0] < 1000.0).all()))
            stats = sampler.normalization()
            expected_action = np.concatenate(
                (np.arange(5, 20), np.arange(5, 30))
            ).mean()
            expected_observation = np.concatenate(
                (np.arange(5, 21), np.arange(5, 31))
            ).mean()
            self.assertAlmostEqual(
                float(stats["action_mean"][0]), float(expected_action), places=6
            )
            self.assertAlmostEqual(
                float(stats["obs_mean"][0]),
                float(expected_observation),
                places=6,
            )
            sampler.cache_on_device("cpu", chunk_episodes=1)
            _cached_obs, cached_action = sampler.sample(
                128, 4, device="cpu"
            )
            self.assertTrue(bool((cached_action[:, :, 0] < 1000.0).all()))

    def test_demo_store_rejects_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            DemoStore(
                directory,
                episodes=1,
                steps=20,
                create=True,
                provenance={"seed": 1, "policy_start_step": 0},
            )
            with self.assertRaisesRegex(
                ValueError, "different configuration or seed"
            ):
                DemoStore(
                    directory,
                    episodes=1,
                    steps=20,
                    create=True,
                    provenance={"seed": 2, "policy_start_step": 0},
                )

    def test_demo_store_treats_missing_spawn_switch_as_legacy_true(self):
        config = load_config(ROOT / "configs" / "smoke.json")
        legacy_config = config.to_dict()
        legacy_config["curriculum"].pop(
            "restrict_screw_spawn_to_opening_half_disk"
        )
        legacy = {"seed": 1, "config": legacy_config}
        current = {"seed": 1, "config": config.to_dict()}
        with tempfile.TemporaryDirectory() as directory:
            DemoStore(
                directory,
                episodes=1,
                steps=20,
                create=True,
                provenance=legacy,
            )
            DemoStore(
                directory,
                episodes=1,
                steps=20,
                create=True,
                provenance=current,
            )

            full_disc = config.to_dict()
            full_disc["curriculum"][
                "restrict_screw_spawn_to_opening_half_disk"
            ] = False
            with self.assertRaisesRegex(
                ValueError, "different configuration or seed"
            ):
                DemoStore(
                    directory,
                    episodes=1,
                    steps=20,
                    create=True,
                    provenance={"seed": 1, "config": full_disc},
                )

    def test_demo_vector_batch_is_committed_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DemoStore(
                directory, episodes=2, steps=20, create=True
            )
            observation = np.zeros(
                (1, 21, OBS_DIM), dtype=np.float16
            )
            action = np.zeros((1, 20, 14), dtype=np.float16)
            spawn = np.zeros((1, 2), dtype=np.float32)
            store.write(
                0, observation, action, spawn, commit=False
            )
            self.assertEqual(store.completed, 0)
            store.write(
                1, observation, action, spawn, commit=False
            )
            self.assertEqual(store.completed, 0)
            store.commit(0, 2)
            self.assertEqual(store.completed, 2)

    def test_demo_checksum_detects_data_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DemoStore(
                directory,
                episodes=1,
                steps=20,
                create=True,
                provenance={"policy_start_step": 0},
            )
            store.write(
                0,
                np.zeros((1, 21, OBS_DIM), dtype=np.float16),
                np.zeros((1, 20, 14), dtype=np.float16),
                np.zeros((1, 2), dtype=np.float32),
            )
            store.observations[0, 0, 0] = np.float16(1.0)
            store.observations.flush()
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                DemoWindowSampler(directory, seed=1)

    def test_tqc_update_and_checkpoint_roundtrip(self):
        config = load_config(ROOT / "configs" / "smoke.json")
        replay = CUDAReplayBuffer(
            capacity=256,
            obs_dim=TQC_OBS_DIM,
            action_dim=RESIDUAL_ACTION_DIM,
            device="cpu",
        )
        observation = torch.randn((128, TQC_OBS_DIM))
        mask = torch.ones((128, RESIDUAL_ACTION_DIM), dtype=torch.bool)
        replay.add(
            observation,
            torch.tanh(torch.randn((128, RESIDUAL_ACTION_DIM))),
            torch.randn(128),
            torch.randn((128, TQC_OBS_DIM)),
            torch.zeros(128, dtype=torch.bool),
            torch.zeros(128, dtype=torch.bool),
            action_mask=mask,
            next_action_mask=mask,
        )
        trainer = TQCTrainer(config, device="cpu")
        metrics = trainer.update(replay.sample(64))
        diagnostic_names = (
            "critic_loss",
            "q_pred_mean",
            "q_target_mean",
            "td_abs_mean",
            "td_abs_p95",
            "policy_entropy",
            "actor_learning_rate",
            "critic_learning_rate",
            "alpha_learning_rate",
            "q_pred_mean_ema",
            "td_abs_mean_ema",
        )
        for name in diagnostic_names:
            self.assertIn(name, metrics)
            self.assertTrue(np.isfinite(float(metrics[name])), name)
        self.assertEqual(metrics["metric_ema_updates"], 1)
        payload = trainer.checkpoint(replay, include_replay=True)
        second_replay = CUDAReplayBuffer(
            capacity=256,
            obs_dim=TQC_OBS_DIM,
            action_dim=RESIDUAL_ACTION_DIM,
            device="cpu",
        )
        second = TQCTrainer(config, device="cpu")
        second.load(payload, replay=second_replay)
        self.assertEqual(second.updates, trainer.updates)
        self.assertEqual(second_replay.size, replay.size)
        self.assertEqual(second.metric_ema_updates, 1)
        torch.testing.assert_close(
            second.metric_ema["q_pred_mean"],
            trainer.metric_ema["q_pred_mean"],
        )

    def test_replay_is_host_resident_and_samples_on_training_device(self):
        capacity = 256
        replay = CUDAReplayBuffer(
            capacity=capacity,
            obs_dim=TQC_OBS_DIM,
            action_dim=RESIDUAL_ACTION_DIM,
            device="cpu",
        )
        tensors = (
            replay.observation,
            replay.action,
            replay.reward,
            replay.next_observation,
            replay.action_mask,
            replay.next_action_mask,
            replay.terminated,
            replay.truncated,
        )
        self.assertEqual(replay.storage_device, torch.device("cpu"))
        self.assertTrue(all(value.device.type == "cpu" for value in tensors))
        expected_storage_bytes = capacity * (
            2 * TQC_OBS_DIM * 4
            + RESIDUAL_ACTION_DIM * 4
            + 4
            + 2 * RESIDUAL_ACTION_DIM
            + 2
        )
        self.assertEqual(
            replay.storage_allocated_bytes, expected_storage_bytes
        )
        self.assertEqual(replay.sample_allocated_bytes(256), 521_728)

        observation = torch.arange(
            128 * TQC_OBS_DIM, dtype=torch.float32
        ).reshape(128, TQC_OBS_DIM)
        mask = torch.ones(
            (128, RESIDUAL_ACTION_DIM), dtype=torch.bool
        )
        replay.add(
            observation,
            torch.zeros((128, RESIDUAL_ACTION_DIM)),
            torch.zeros(128),
            observation + 1.0,
            torch.zeros(128, dtype=torch.bool),
            torch.zeros(128, dtype=torch.bool),
            action_mask=mask,
            next_action_mask=mask,
        )
        with mock.patch.object(
            torch, "randint", wraps=torch.randint
        ) as randint:
            batch = replay.sample(64)
        self.assertEqual(randint.call_args.kwargs["device"], replay.device)
        for value in (
            batch.observation,
            batch.action,
            batch.reward,
            batch.next_observation,
            batch.action_mask,
            batch.next_action_mask,
            batch.terminated,
            batch.truncated,
        ):
            self.assertEqual(value.device, replay.device)

        state = replay.state_dict(include_data=True)
        self.assertEqual(state["format"], REPLAY_FORMAT)
        self.assertEqual(REPLAY_FORMAT, "screw-actuator-residual-replay-v2")
        for name in (
            "observation",
            "action",
            "reward",
            "next_observation",
            "action_mask",
            "next_action_mask",
            "terminated",
            "truncated",
        ):
            self.assertEqual(state[name].device.type, "cpu")

    def test_tqc_act_after_learning_starts_uses_keyword_only_sampling(self):
        config = load_config(ROOT / "configs" / "smoke.json")
        trainer = TQCTrainer(config, device="cpu")
        trainer.transitions = config.tqc.learning_starts
        observation = torch.zeros((1, TQC_OBS_DIM))
        action_mask = torch.ones(
            (1, RESIDUAL_ACTION_DIM), dtype=torch.bool
        )
        with mock.patch.object(
            trainer.actor,
            "sample",
            wraps=trainer.actor.sample,
        ) as sample:
            action = trainer.act(observation, action_mask=action_mask)
        positional, keywords = sample.call_args
        self.assertEqual(len(positional), 1)
        self.assertIs(positional[0], observation)
        self.assertIn("deterministic", keywords)
        self.assertIn("action_mask", keywords)
        self.assertEqual(action.shape, (1, RESIDUAL_ACTION_DIM))
        self.assertTrue(bool(torch.isfinite(action).all()))

    def test_tqc_actor_starts_at_zero_deterministic_residual(self):
        config = load_config(ROOT / "configs" / "smoke.json")
        trainer = TQCTrainer(config, device="cpu")
        observation = torch.randn((5, TQC_OBS_DIM))
        mask = torch.ones((5, RESIDUAL_ACTION_DIM), dtype=torch.bool)
        action = trainer.act(
            observation, action_mask=mask, deterministic=True
        )
        torch.testing.assert_close(action, torch.zeros_like(action))
        _mean, log_std = trainer.actor(observation)
        torch.testing.assert_close(
            log_std,
            torch.full_like(log_std, config.tqc.log_std_init),
        )

    def test_tqc_rejects_nonfinite_replay_before_optimizer_step(self):
        config = load_config(ROOT / "configs" / "smoke.json")
        replay = CUDAReplayBuffer(
            capacity=128,
            obs_dim=TQC_OBS_DIM,
            action_dim=RESIDUAL_ACTION_DIM,
            device="cpu",
        )
        mask = torch.ones((64, RESIDUAL_ACTION_DIM), dtype=torch.bool)
        replay.add(
            torch.zeros((64, TQC_OBS_DIM)),
            torch.zeros((64, RESIDUAL_ACTION_DIM)),
            torch.zeros(64),
            torch.zeros((64, TQC_OBS_DIM)),
            torch.zeros(64, dtype=torch.bool),
            torch.zeros(64, dtype=torch.bool),
            action_mask=mask,
            next_action_mask=mask,
        )
        batch = replay.sample(64)
        batch.next_observation[0, 0] = float("nan")
        trainer = TQCTrainer(config, device="cpu")
        with self.assertRaisesRegex(FloatingPointError, "replay batch"):
            trainer.update(batch)
        self.assertEqual(trainer.updates, 0)

    def test_replay_finite_check_scans_every_bounded_chunk(self):
        replay = CUDAReplayBuffer(
            capacity=10, obs_dim=3, action_dim=2, device="cpu"
        )
        for tensor in (
            replay.observation,
            replay.action,
            replay.reward,
            replay.next_observation,
        ):
            tensor.zero_()
        replay.size = replay.capacity
        with mock.patch.object(
            replay_module, "FINITE_CHECK_CHUNK_ROWS", 3
        ):
            replay.assert_finite()
            replay.next_observation[8, 1] = float("nan")
            with self.assertRaisesRegex(
                FloatingPointError, "replay checkpoint"
            ):
                replay.assert_finite()

    def test_replay_restore_rejects_shape_and_dtype_changes(self):
        replay = CUDAReplayBuffer(
            capacity=4, obs_dim=3, action_dim=2, device="cpu"
        )
        replay.add(
            torch.zeros((2, 3)),
            torch.zeros((2, 2)),
            torch.zeros(2),
            torch.zeros((2, 3)),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            action_mask=torch.ones((2, 2), dtype=torch.bool),
            next_action_mask=torch.ones((2, 2), dtype=torch.bool),
        )
        state = replay.state_dict(include_data=True)

        bad_shape = dict(state)
        bad_shape["observation"] = torch.zeros((2, 4))
        restored = CUDAReplayBuffer(
            capacity=4, obs_dim=3, action_dim=2, device="cpu"
        )
        with self.assertRaisesRegex(ValueError, "shape differs"):
            restored.load_state_dict(bad_shape)

        bad_dtype = dict(state)
        bad_dtype["observation"] = state["observation"].to(torch.float64)
        with self.assertRaisesRegex(ValueError, "dtype differs"):
            restored.load_state_dict(bad_dtype)

    def test_tqc_learning_starts_has_no_warmup_catchup(self):
        config = load_config(ROOT / "configs" / "smoke.json")
        trainer = TQCTrainer(config, device="cpu")
        self.assertEqual(trainer.updates_for(config.tqc.learning_starts), 0)
        self.assertEqual(trainer.update_credit, 0.0)
        # With batch=64 and replay_ratio=.25, exactly 256 eligible
        # transitions produce one update.
        counts = [
            trainer.updates_for(64),
            trainer.updates_for(64),
            trainer.updates_for(64),
            trainer.updates_for(64),
        ]
        self.assertEqual(counts, [0, 0, 0, 1])

    def test_tqc_vector_step_update_cap_discards_burst_debt(self):
        config = load_config(ROOT / "configs" / "smoke.json")
        config = replace(
            config,
            tqc=replace(config.tqc, gradient_steps_per_vector_step=2),
        )
        trainer = TQCTrainer(config, device="cpu")
        trainer.transitions = config.tqc.learning_starts
        transitions_for_five_updates = int(
            5 * config.tqc.batch_size / config.tqc.replay_ratio
        )
        self.assertEqual(
            trainer.updates_for(transitions_for_five_updates), 2
        )
        self.assertEqual(trainer.update_credit, 0.0)

    def test_tqc_checkpoint_is_paired_to_diffusion_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            diffusion = Path(directory) / "diffusion.pt"
            diffusion.write_bytes(b"model-a")
            payload = checkpoint_metadata("tqc", {})
            payload["diffusion_sha256"] = sha256_file(diffusion)
            self.assertEqual(
                _validate_diffusion_dependency(payload, diffusion),
                payload["diffusion_sha256"],
            )
            diffusion.write_bytes(b"model-b")
            with self.assertRaisesRegex(
                ValueError, "checkpoint mismatch"
            ):
                _validate_diffusion_dependency(payload, diffusion)

    def test_resume_rejects_different_configuration(self):
        config = load_config(ROOT / "configs" / "smoke.json")
        payload = checkpoint_metadata("diffusion", config.to_dict())
        _validate_checkpoint_config(
            payload, config, kind="Diffusion"
        )
        changed = replace(
            config,
            runtime=replace(config.runtime, seed=config.runtime.seed + 1),
        )
        with self.assertRaisesRegex(
            ValueError, "configuration differs"
        ):
            _validate_checkpoint_config(
                payload, changed, kind="Diffusion"
            )

    def test_legacy_half_disc_checkpoint_config_remains_compatible(self):
        config = load_config(ROOT / "configs" / "smoke.json")
        legacy_config = config.to_dict()
        legacy_config["curriculum"].pop(
            "restrict_screw_spawn_to_opening_half_disk"
        )
        payload = checkpoint_metadata("diffusion", legacy_config)
        _validate_checkpoint_config(payload, config, kind="Diffusion")

        full_disc = replace(
            config,
            curriculum=replace(
                config.curriculum,
                restrict_screw_spawn_to_opening_half_disk=False,
            ),
        )
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            _validate_checkpoint_config(
                payload, full_disc, kind="Diffusion"
            )

    def test_rng_checkpoint_roundtrip(self):
        _seed_everything(1234, deterministic=False)
        state = _capture_rng()
        expected = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
        _restore_rng(state)
        actual = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
        self.assertEqual(actual, expected)

    def test_diffusion_resumes_at_exact_batch_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "demos"
            store = DemoStore(
                dataset,
                episodes=2,
                steps=20,
                create=True,
                provenance={"policy_start_step": 0},
            )
            rng = np.random.default_rng(5)
            store.write(
                0,
                rng.normal(size=(2, 21, OBS_DIM)).astype(np.float16),
                rng.normal(size=(2, 20, 14)).astype(np.float16),
                np.zeros((2, 2), dtype=np.float32),
            )
            base = load_config(ROOT / "configs" / "smoke.json")
            config = replace(
                base,
                demos=replace(
                    base.demos, episodes=2, action_horizon=16
                ),
                diffusion=replace(
                    base.diffusion,
                    epochs=1,
                    batch_size=4,
                    hidden_dim=32,
                    depth=1,
                ),
                runtime=replace(
                    base.runtime, device="cpu", deterministic=False
                ),
            )
            checkpoint = root / "diffusion.pt"
            log = root / "diffusion.jsonl"
            original_sample = DemoWindowSampler.sample
            calls = 0

            def interrupt_before_second_batch(instance, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt
                return original_sample(instance, *args, **kwargs)

            with mock.patch.object(
                DemoWindowSampler,
                "sample",
                new=interrupt_before_second_batch,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    train_diffusion(
                        config,
                        dataset=dataset,
                        output=checkpoint,
                        log_path=log,
                        resume=True,
                    )
            interrupted = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            self.assertEqual(interrupted["epoch"], 0)
            self.assertEqual(interrupted["batch_in_epoch"], 1)
            self.assertEqual(interrupted["global_step"], 1)
            self.assertTrue(interrupted["interrupted"])

            train_diffusion(
                config,
                dataset=dataset,
                output=checkpoint,
                log_path=log,
                resume=True,
            )
            completed = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            self.assertEqual(completed["epoch"], 1)
            self.assertEqual(completed["batch_in_epoch"], 0)
            self.assertEqual(completed["global_step"], 3)
            self.assertFalse(completed["interrupted"])

            reference_path = root / "diffusion_reference.pt"
            train_diffusion(
                config,
                dataset=dataset,
                output=reference_path,
                log_path=root / "reference.jsonl",
                resume=False,
            )
            reference = torch.load(
                reference_path, map_location="cpu", weights_only=False
            )
            for name, expected in reference["model_state"].items():
                torch.testing.assert_close(
                    completed["model_state"][name],
                    expected,
                    rtol=0.0,
                    atol=0.0,
                )
            self.assertEqual(
                completed["scheduler_state"],
                reference["scheduler_state"],
            )

    def test_diffusion_interrupt_after_epoch_save_reuses_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "demos"
            store = DemoStore(
                dataset,
                episodes=2,
                steps=20,
                create=True,
                provenance={"policy_start_step": 0},
            )
            rng = np.random.default_rng(17)
            store.write(
                0,
                rng.normal(size=(2, 21, OBS_DIM)).astype(np.float16),
                rng.normal(size=(2, 20, 14)).astype(np.float16),
                np.zeros((2, 2), dtype=np.float32),
            )
            base = load_config(ROOT / "configs" / "smoke.json")
            config = replace(
                base,
                demos=replace(
                    base.demos, episodes=2, action_horizon=16
                ),
                diffusion=replace(
                    base.diffusion,
                    epochs=1,
                    batch_size=4,
                    hidden_dim=32,
                    depth=1,
                ),
                runtime=replace(
                    base.runtime, device="cpu", deterministic=False
                ),
            )
            checkpoint = root / "diffusion.pt"
            safe_cursor_calls = 0

            def interrupt_after_epoch_checkpoint(_interrupt, _monitor):
                nonlocal safe_cursor_calls
                safe_cursor_calls += 1
                # Three minibatches, then the post-epoch checkpoint cursor.
                if safe_cursor_calls == 4:
                    raise KeyboardInterrupt

            with mock.patch(
                "s_hpc.workflows._safe_cursor_interrupt",
                new=interrupt_after_epoch_checkpoint,
            ), mock.patch(
                "s_hpc.workflows.atomic_torch_save",
                wraps=workflows_module.atomic_torch_save,
            ) as atomic_save:
                with self.assertRaises(KeyboardInterrupt):
                    train_diffusion(
                        config,
                        dataset=dataset,
                        output=checkpoint,
                        log_path=root / "diffusion.jsonl",
                        resume=False,
                    )

            self.assertEqual(atomic_save.call_count, 1)
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(payload["batch_in_epoch"], 0)
            # The already committed normal checkpoint is fully resumable; the
            # interrupt path deliberately does not rewrite it just to flip this
            # diagnostic flag.
            self.assertFalse(payload["interrupted"])


if __name__ == "__main__":
    unittest.main()
