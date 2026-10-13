from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import torch

from MJWarp_RL import contract
from MJWarp_RL.replay import CUDAReplayBuffer, ReplayBatch
from MJWarp_RL.tqc import (
    QuantileCritics,
    SquashedGaussianActor,
    TQCConfig,
    TQCTrainer,
    quantile_huber_loss,
)


class LegacyTQCCoreTests(unittest.TestCase):
    """Keep old-checkpoint primitives healthy without binding formal training."""

    def test_quantile_pairing_and_truncation_dimensions(self) -> None:
        prediction = torch.zeros((3, 2, 25))
        retained_target = torch.ones((3, 46))
        self.assertAlmostEqual(
            float(quantile_huber_loss(prediction, retained_target)),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(quantile_huber_loss(prediction, -retained_target)),
            0.25,
            places=6,
        )
        critics = QuantileCritics(
            count=2,
            obs_dim=7,
            action_dim=4,
            hidden_dim=16,
            quantiles=25,
        )
        values = critics(torch.zeros((3, 7)), torch.zeros((3, 4)))
        self.assertEqual(values.shape, (3, 2, 25))
        flattened = values.flatten(start_dim=1)
        self.assertEqual(flattened.shape, (3, 50))
        self.assertEqual(flattened[:, :-4].shape, (3, 46))

    def test_action_mask_zeros_action_and_entropy_terms(self) -> None:
        actor = SquashedGaussianActor(7, 4, 16)
        observation = torch.zeros((3, 7))
        mask = torch.zeros((3, 4), dtype=torch.bool)
        action, log_probability = actor.sample(
            observation, action_mask=mask
        )
        self.assertTrue(torch.equal(action, torch.zeros_like(action)))
        self.assertTrue(
            torch.equal(
                log_probability, torch.zeros_like(log_probability)
            )
        )

    def test_replay_ring_uses_detached_contiguous_writes(self) -> None:
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            replay = CUDAReplayBuffer(
                capacity=5, obs_dim=2, action_dim=1, device="cpu"
            )

            def add(start: int, count: int) -> None:
                values = torch.arange(
                    start, start + count, dtype=torch.float32
                )
                observation = values[:, None].repeat(1, 2)
                observation.requires_grad_(True)
                replay.add(
                    observation,
                    values[:, None],
                    values,
                    observation + 1.0,
                    torch.zeros(count, dtype=torch.bool),
                )

            add(0, 3)
            add(3, 4)
            self.assertEqual((replay.position, replay.size), (2, 5))
            self.assertEqual(
                replay.observation[:, 0].tolist(),
                [5.0, 6.0, 2.0, 3.0, 4.0],
            )
            self.assertFalse(replay.observation.requires_grad)
            self.assertIsNone(replay.observation.grad_fn)
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_one_legacy_masked_tqc_update_is_finite(self) -> None:
        torch.manual_seed(20260729)
        settings = replace(
            TQCConfig(),
            learning_starts=8,
            batch_size=8,
            hidden_dim=32,
        )

        def mask(observation):
            result = torch.zeros(
                (observation.shape[0], contract.ACTION_DIM),
                dtype=torch.bool,
            )
            result[:, :4] = True
            return result

        trainer = TQCTrainer(
            obs_dim=10,
            action_dim=contract.ACTION_DIM,
            config=settings,
            device="cpu",
            action_mask_fn=mask,
        )
        batch = ReplayBatch(
            observation=torch.randn((8, 10)),
            action=torch.zeros((8, contract.ACTION_DIM)),
            reward=torch.randn((8, 1)),
            next_observation=torch.randn((8, 10)),
            done=torch.zeros((8, 1)),
        )
        diagnostics = trainer.update(batch)
        self.assertEqual(trainer.updates, 1)
        self.assertTrue(
            all(torch.isfinite(value) for value in diagnostics.values())
        )
        self.assertEqual(
            float(diagnostics["active_action_dimensions"]), 4.0
        )
        checkpoint = trainer.checkpoint()
        self.assertNotIn("update_credit", checkpoint)

    def test_formal_pipeline_is_isolated_from_legacy_tqc(self) -> None:
        import MJWarp_RL.evaluate as evaluation
        import MJWarp_RL.train as training

        self.assertEqual(
            contract.ALGORITHM, "ActionChunkDiffusionIKImitation"
        )
        self.assertNotIn("from .tqc import", inspect.getsource(training))
        self.assertNotIn("from .tqc import", inspect.getsource(evaluation))
        self.assertIn(
            "Legacy TQC learner",
            inspect.getdoc(__import__("MJWarp_RL.tqc", fromlist=["*"])) or "",
        )


if __name__ == "__main__":
    unittest.main()
