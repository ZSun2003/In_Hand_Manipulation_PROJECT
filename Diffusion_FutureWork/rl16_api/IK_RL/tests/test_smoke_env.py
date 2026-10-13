from __future__ import annotations

import unittest

import numpy as np

from IK_RL import config
from IK_RL.controller_bridge import OBSERVATION_DIM
from IK_RL.env import G6IKResidualEnv


class EnvironmentSmokeTests(unittest.TestCase):
    def test_one_gym_step_is_one_zero_residual_control_tick(self) -> None:
        env = G6IKResidualEnv(wait_timeout_s=30.0)
        try:
            observation, info = env.reset(
                seed=0,
                options={
                    "object": "cube",
                    "task": config.TASK_TRANSLATE,
                    "episode_seed": 0,
                },
            )
            self.assertEqual(observation.shape, (OBSERVATION_DIM,))
            self.assertTrue(np.all(np.isfinite(observation)))
            self.assertNotEqual(info["initial_state_hash"], "")
            assert env._bridge is not None
            bridge = env._bridge
            request_before = bridge._request_id
            # The controller publishes the next pending request during step().
            # Zero residual must preserve this tick's latched nominal command.
            nominal_before = bridge.nominal_ctrl.copy()
            next_observation, _reward, terminated, truncated, next_info = (
                env.step(np.zeros(config.ACTION_DIM, dtype=np.float32))
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(bridge._request_id - request_before, 1)
            self.assertEqual(next_info["control_step"], 1)
            self.assertEqual(next_info["action_control_step"], 0)
            self.assertTrue(next_info["residual_enabled"])
            self.assertFalse(next_info["manipulation_started"])
            np.testing.assert_array_equal(
                bridge.last_residual, np.zeros(config.ACTION_DIM)
            )
            np.testing.assert_array_equal(
                bridge.applied_ctrl, nominal_before
            )
            self.assertEqual(next_observation.shape, (OBSERVATION_DIM,))
            self.assertTrue(np.all(np.isfinite(next_observation)))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
