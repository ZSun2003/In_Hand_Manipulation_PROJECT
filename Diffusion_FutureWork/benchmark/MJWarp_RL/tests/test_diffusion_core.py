from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from MJWarp_RL import contract
from MJWarp_RL.diffusion_policy import (
    ACTION_HORIZON,
    DDIM_INFERENCE_STEPS,
    DDPM_TRAIN_STEPS,
    OBSERVATION_HORIZON,
    REPLAN_INTERVAL,
    DiffusionPolicy,
    ExponentialMovingAverage,
    build_default_policy,
    default_model_config,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from MJWarp_RL.diffusion_runtime import (
    BatchedDiffusionController,
    BatchedObservationHistory,
)
from MJWarp_RL.env import OBSERVATION_DIM


class DiffusionCoreTests(unittest.TestCase):
    def _small_policy(self) -> DiffusionPolicy:
        return build_default_policy(
            num_diffusion_steps=8,
            hidden_dim=32,
            depth=1,
        )

    def test_default_contract_uses_live_environment_dimensions(self) -> None:
        configuration = default_model_config()
        self.assertEqual(configuration["obs_dim"], OBSERVATION_DIM)
        self.assertEqual(configuration["action_dim"], contract.ACTION_DIM)
        self.assertEqual(configuration["obs_horizon"], OBSERVATION_HORIZON)
        self.assertEqual(configuration["horizon"], ACTION_HORIZON)
        self.assertEqual(
            configuration["num_diffusion_steps"],
            DDPM_TRAIN_STEPS,
        )
        self.assertEqual(
            (ACTION_HORIZON, OBSERVATION_HORIZON),
            (16, 2),
        )
        self.assertEqual(
            (DDPM_TRAIN_STEPS, DDIM_INFERENCE_STEPS, REPLAN_INTERVAL),
            (100, 16, 4),
        )

    def test_ddpm_loss_backward_and_ddim_sample_are_finite(self) -> None:
        torch.manual_seed(20260730)
        policy = self._small_policy()
        observation = torch.randn((3, 2, OBSERVATION_DIM))
        action = torch.tanh(
            torch.randn((3, ACTION_HORIZON, contract.ACTION_DIM))
        )
        loss = policy.compute_loss(observation, action)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in policy.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(
            all(bool(torch.isfinite(value).all()) for value in gradients)
        )
        sample = policy.sample(
            observation,
            num_inference_steps=2,
            seed=17,
        )
        self.assertEqual(
            sample.shape,
            (3, ACTION_HORIZON, contract.ACTION_DIM),
        )
        self.assertTrue(bool(torch.isfinite(sample).all()))

    def test_ema_checkpoint_roundtrip_uses_ema_weights(self) -> None:
        policy = self._small_policy()
        ema = ExponentialMovingAverage(policy, decay=0.5)
        with torch.no_grad():
            next(policy.parameters()).add_(1.0)
        ema.update(policy)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "diffusion.pt"
            save_policy_checkpoint(
                checkpoint_path,
                policy,
                ema=ema,
                metadata={"step": 123},
            )
            loaded, checkpoint = load_policy_checkpoint(
                checkpoint_path,
                device="cpu",
                use_ema=True,
            )
        self.assertEqual(checkpoint["metadata"]["step"], 123)
        self.assertEqual(loaded.model_config(), policy.model_config())
        for name, value in loaded.state_dict().items():
            torch.testing.assert_close(value, ema.shadow[name].cpu())

    def test_partial_reset_preserves_unrelated_world_plans(self) -> None:
        policy = self._small_policy()
        controller = BatchedDiffusionController(
            policy,
            worlds=3,
            inference_steps=2,
            replan_interval=4,
            seed=29,
            action_clip=False,
        )
        history = torch.zeros((3, 2, OBSERVATION_DIM))
        controller.reset(torch.ones(3, dtype=torch.bool), history)
        controller.act(history)
        unaffected_plan = controller.plan[1:].clone()
        changed = history.clone()
        changed[0] = 1.0
        controller.reset(
            torch.tensor((True, False, False)),
            changed,
        )
        action = controller.act(history)
        torch.testing.assert_close(
            controller.plan[1:],
            unaffected_plan,
        )
        torch.testing.assert_close(
            action[1:],
            unaffected_plan[:, 1],
        )
        torch.testing.assert_close(
            action[0],
            controller.override_plan[0, 0],
        )

    def test_controller_state_continues_exact_plan_cursor(self) -> None:
        policy = self._small_policy()
        history = torch.zeros((3, 2, OBSERVATION_DIM))
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
            seed=999,
        )
        restored.load_state_dict(state)
        actual = restored.act(history)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(restored.cursor, original.cursor)
        self.assertEqual(restored.sample_count, original.sample_count)
        self.assertEqual(restored.seed, 31)

    def test_observation_history_handles_asynchronous_reset(self) -> None:
        history = BatchedObservationHistory(
            worlds=3,
            obs_horizon=2,
            obs_dim=OBSERVATION_DIM,
            device="cpu",
        )
        first = torch.arange(
            3 * OBSERVATION_DIM,
            dtype=torch.float32,
        ).reshape(3, OBSERVATION_DIM)
        history.append(first)
        second = first + 10.0
        history.append(
            second,
            reset_mask=torch.tensor((False, True, False)),
        )
        torch.testing.assert_close(history.values[0, 0], first[0])
        torch.testing.assert_close(history.values[0, 1], second[0])
        torch.testing.assert_close(history.values[1, 0], second[1])
        torch.testing.assert_close(history.values[1, 1], second[1])

        state = history.state_dict()
        restored = BatchedObservationHistory(
            worlds=3,
            obs_horizon=2,
            obs_dim=OBSERVATION_DIM,
            device="cpu",
        )
        restored.load_state_dict(state)
        torch.testing.assert_close(restored.values, history.values)
        torch.testing.assert_close(
            restored.initialized,
            history.initialized,
        )


if __name__ == "__main__":
    unittest.main()
