from __future__ import annotations

from threading import Thread
import unittest

import numpy as np

from IK_RL import config
from IK_RL.controller_bridge import (
    BridgeAborted,
    ControlTickBridge,
    apply_residual_once,
)


class ControllerBridgeTests(unittest.TestCase):
    def test_zero_action_is_bitwise_current_ik(self) -> None:
        calls: list[np.ndarray] = []
        bridge = ControlTickBridge(
            ctrl_low=np.full(config.ACTUATOR_DIM, -2.0),
            ctrl_high=np.full(config.ACTUATOR_DIM, 2.0),
            snapshot=lambda *_args: {},
        )
        base = np.linspace(-0.5, 0.5, config.ACTUATOR_DIM)

        def worker() -> None:
            bridge.wrapped_physics_ctrl(
                lambda value: calls.append(np.asarray(value).copy()),
                base,
            )
            bridge.finish(result="done")

        thread = Thread(target=worker)
        thread.start()
        self.assertIsNone(bridge.wait_until_ready())
        tick, done = bridge.advance(np.zeros(config.ACTUATOR_DIM))
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(calls), 1)
        self.assertTrue(np.array_equal(calls[0], base))
        self.assertTrue(np.array_equal(tick.base_ctrl, tick.applied_ctrl))
        self.assertTrue(np.array_equal(tick.residual, np.zeros(12)))
        self.assertEqual(done.result, "done")

    def test_rate_and_amplitude_bounds(self) -> None:
        residual = np.zeros(config.ACTUATOR_DIM)
        for _ in range(100):
            following = apply_residual_once(residual, np.ones(12))
            self.assertLessEqual(
                float(np.max(np.abs(following - residual))),
                config.RESIDUAL_RATE_LIMIT_RAD_PER_STEP + 1.0e-15,
            )
            residual = following
        self.assertLessEqual(
            float(np.max(np.abs(residual))),
            config.RESIDUAL_MAX_RAD,
        )

    def test_nonfinite_action_rejected(self) -> None:
        action = np.zeros(config.ACTUATOR_DIM)
        action[3] = np.nan
        with self.assertRaises(ValueError):
            apply_residual_once(np.zeros(config.ACTUATOR_DIM), action)

    def test_abort_wakes_published_request_race(self) -> None:
        bridge = ControlTickBridge(
            ctrl_low=np.full(config.ACTUATOR_DIM, -2.0),
            ctrl_high=np.full(config.ACTUATOR_DIM, 2.0),
            snapshot=lambda *_args: {},
        )
        caught: list[type[BaseException]] = []

        def worker() -> None:
            try:
                bridge.wrapped_physics_ctrl(
                    lambda _value: None,
                    np.zeros(config.ACTUATOR_DIM),
                )
            except BaseException as exc:
                caught.append(type(exc))

        thread = Thread(target=worker)
        thread.start()
        bridge.abort()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(caught, [BridgeAborted])


if __name__ == "__main__":
    unittest.main()
