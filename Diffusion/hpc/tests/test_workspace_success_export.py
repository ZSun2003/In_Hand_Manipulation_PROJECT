from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from s_hpc.config import load_config
from s_hpc.contracts import ACTION_DIM, OBS_DIM
from s_hpc.demos import DemoWindowSampler
from s_hpc.workflows import train_diffusion
from evaluation.workspace_policy_rollouts import (
    TimeMajorTrajectoryStager,
    export_successful_policy_rollouts,
    release_cuda_batch_memory,
    rollout_batch_slices,
    trajectory_storage_estimate,
)


ROOT = Path(__file__).resolve().parents[1]


class WorkspaceSuccessExportTest(unittest.TestCase):
    def test_rollout_batches_cover_global_order_without_loss(self):
        batches = rollout_batch_slices(13_976, 7_000)
        self.assertEqual(
            [(value.start, value.stop) for value in batches],
            [(0, 7_000), (7_000, 13_976)],
        )
        indices = np.concatenate(
            [np.arange(value.start, value.stop) for value in batches]
        )
        np.testing.assert_array_equal(indices, np.arange(13_976))
        with self.assertRaisesRegex(ValueError, "positive"):
            rollout_batch_slices(13_976, 0)

    def test_batch_release_flushes_warp_mempool_before_next_env(self):
        calls = []

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def synchronize(device):
                calls.append(("torch_sync", device))

            @staticmethod
            def empty_cache():
                calls.append(("torch_empty", None))

            @staticmethod
            def memory_allocated(device):
                return 123

            @staticmethod
            def memory_reserved(device):
                return 456

        fake_torch = SimpleNamespace(
            cuda=FakeCuda(), device=lambda value: value
        )
        fake_warp = SimpleNamespace(
            set_mempool_release_threshold=lambda device, value: calls.append(
                ("warp_threshold", device, value)
            ),
            synchronize_device=lambda device: calls.append(
                ("warp_sync", device)
            ),
        )
        with mock.patch.dict(sys.modules, {"warp": fake_warp}):
            result = release_cuda_batch_memory(fake_torch, "cuda:0")
        self.assertEqual(result["torch_allocated_bytes"], 123)
        self.assertEqual(result["torch_reserved_bytes"], 456)
        self.assertIn(("warp_threshold", "cuda:0", 0), calls)
        self.assertIn(("warp_sync", "cuda:0"), calls)
        self.assertGreaterEqual(
            sum(name == "torch_sync" for name, *_ in calls), 2
        )

    def test_storage_estimate_accounts_for_staging_and_final_store(self):
        estimate = trajectory_storage_estimate(
            worlds=10408,
            steps=4000,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
        )
        self.assertEqual(estimate["staging_bytes"], 17_073_095_856)
        self.assertGreater(
            estimate["worst_case_final_dataset_bytes"],
            estimate["staging_bytes"],
        )
        self.assertEqual(
            estimate["worst_case_peak_bytes_before_reserve"],
            estimate["staging_bytes"]
            + estimate["worst_case_final_dataset_bytes"],
        )

    def test_success_export_is_variable_length_and_training_compatible(self):
        worlds = 3
        steps = 10
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            stager = TimeMajorTrajectoryStager(
                staging,
                worlds=worlds,
                steps=steps,
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
            )
            initial = np.zeros((worlds, OBS_DIM), dtype=np.float16)
            initial[:, 0] = np.arange(worlds, dtype=np.float16) * 100
            stager.write_initial_batch(0, initial[:2])
            stager.write_initial_batch(2, initial[2:])
            for step in range(1, steps + 1):
                observation = np.zeros(
                    (worlds, OBS_DIM), dtype=np.float16
                )
                action = np.zeros((worlds, ACTION_DIM), dtype=np.float16)
                observation[:, 0] = (
                    np.arange(worlds, dtype=np.float16) * 100 + step
                )
                action[:, 0] = (
                    np.arange(worlds, dtype=np.float16) * 100 + step - 1
                )
                stager.write_step_batch(
                    step, 0, observation[:2], action[:2]
                )
                stager.write_step_batch(
                    step, 2, observation[2:], action[2:]
                )

            trial_csv = root / "trials.csv"
            trial_csv.write_text("world,success\n0,true\n1,false\n2,true\n")
            rows = (
                {"success": True, "terminal_steps": 7},
                {"success": False, "terminal_steps": 8},
                {"success": True, "terminal_steps": 9},
            )
            target = root / "successful_policy_rollouts"
            result = export_successful_policy_rollouts(
                stager=stager,
                target=target,
                trial_rows=rows,
                trial_csv=trial_csv,
                fixed_spawn=np.asarray(
                    ((0.0, 0.0), (0.01, 0.0), (0.02, 0.0)),
                    dtype=np.float32,
                ),
                fixed_mouths=np.zeros((worlds, 2), dtype=np.float32),
                socket_angle_deg=45.0,
                policy_start_step=2,
                action_horizon=3,
                shard_episodes=1,
                max_episodes=8192,
                selection_seed=17,
                provenance={
                    "policy_start_step": 2,
                    "distribution_shift": {
                        "training_spawn_radius_mm": 100.0,
                        "rollout_spawn_radius_mm": 140.0,
                        "radial_extension_is_ood": True,
                    },
                },
            )
            self.assertEqual(result["status"], "complete")
            self.assertFalse(staging.exists())
            metadata = json.loads(
                (target / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["episodes"], 2)
            self.assertEqual(metadata["steps"], 9)
            self.assertTrue(metadata["variable_episode_lengths"])
            self.assertFalse(metadata["provenance"]["expert_demonstrations"])
            self.assertTrue(
                metadata["provenance"]["distribution_shift"][
                    "radial_extension_is_ood"
                ]
            )
            np.testing.assert_array_equal(
                np.load(target / "valid_steps.npy"), (7, 9)
            )
            observations = np.load(target / "observations.npy")
            actions = np.load(target / "actions.npy")
            np.testing.assert_array_equal(
                observations[0, 8:, 0], observations[0, 7, 0]
            )
            np.testing.assert_array_equal(actions[0, 7:, 0], 0.0)

            sampler = DemoWindowSampler(target, seed=3)
            self.assertEqual(sampler.window_count(3), 8)
            sampled_observation, sampled_action = sampler.sample(
                512, 3, device="cpu"
            )
            sampled_world = (sampled_observation[:, 1, 0] >= 100).long()
            sampled_step = sampled_action[:, 0, 0] - sampled_world * 200
            self.assertTrue(bool((sampled_step >= 2).all()))
            self.assertTrue(bool((sampled_step <= 6).all()))

            # A fresh train_diffusion run accepts the finalized variable-length
            # corpus even though config.demos.episodes is not the retained
            # corpus size.  This is not checkpoint resume or fine-tuning.
            base = load_config(ROOT / "configs" / "smoke.json")
            config = replace(
                base,
                demos=replace(base.demos, action_horizon=3),
                diffusion=replace(
                    base.diffusion,
                    epochs=1,
                    batch_size=8,
                    hidden_dim=32,
                    depth=1,
                ),
                runtime=replace(
                    base.runtime, device="cpu", deterministic=False
                ),
            )
            checkpoint = root / "fresh_diffusion.pt"
            train_diffusion(
                config,
                dataset=target,
                output=checkpoint,
                log_path=root / "fresh_diffusion.jsonl",
                resume=False,
            )
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["dataset_contract"]["episodes"], 2)
            self.assertTrue(
                payload["dataset_contract"]["variable_episode_lengths"]
            )
            self.assertTrue(
                payload["dataset_contract"]["provenance"][
                    "distribution_shift"
                ]["radial_extension_is_ood"]
            )


if __name__ == "__main__":
    unittest.main()
