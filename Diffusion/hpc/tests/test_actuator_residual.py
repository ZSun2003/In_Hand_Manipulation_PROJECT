from __future__ import annotations

import unittest

import torch

from s_hpc.actuator_residual import (
    BatchedActuatorResidualMapper,
    LINEAR_ACTUATOR_INDICES,
    ROTARY_ACTUATOR_INDICES,
)


class BatchedActuatorResidualMapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.worlds = 3
        self.mapper = BatchedActuatorResidualMapper(
            self.worlds,
            torch.full((16,), -1.0),
            torch.full((16,), 1.0),
        )
        self.nominal = torch.zeros((self.worlds, 16))
        self.enabled = torch.ones((self.worlds, 16), dtype=torch.bool)

    def test_g6_unit_aware_limits_and_first_tick_slew(self) -> None:
        self.assertEqual(LINEAR_ACTUATOR_INDICES, (0, 3, 6, 9))
        self.assertEqual(len(ROTARY_ACTUATOR_INDICES), 12)
        linear = torch.tensor(LINEAR_ACTUATOR_INDICES)
        rotary = torch.tensor(ROTARY_ACTUATOR_INDICES)
        torch.testing.assert_close(
            self.mapper.limits[linear], torch.full((4,), 0.0005)
        )
        torch.testing.assert_close(
            self.mapper.limits[rotary], torch.full((12,), 0.005)
        )

        result = self.mapper.apply(
            self.nominal, torch.ones_like(self.nominal), self.enabled
        )
        torch.testing.assert_close(
            result.realised_residual[:, linear],
            torch.full((self.worlds, 4), 0.00002),
        )
        torch.testing.assert_close(
            result.realised_residual[:, rotary],
            torch.full((self.worlds, 12), 0.0001),
        )

    def test_residual_reaches_but_never_exceeds_limits(self) -> None:
        previous = self.mapper.residual.clone()
        for _ in range(80):
            result = self.mapper.apply(
                self.nominal, torch.ones_like(self.nominal), self.enabled
            )
            self.assertTrue(
                bool(
                    (
                        (result.realised_residual - previous).abs()
                        <= self.mapper.rates + 1.0e-7
                    ).all()
                )
            )
            self.assertTrue(
                bool(
                    (
                        result.realised_residual.abs()
                        <= self.mapper.limits + 1.0e-7
                    ).all()
                )
            )
            previous = result.realised_residual.clone()
        torch.testing.assert_close(
            result.realised_residual,
            self.mapper.limits.expand(self.worlds, -1),
        )

    def test_mask_removes_authority_and_clears_stale_state(self) -> None:
        self.mapper.apply(
            self.nominal, torch.ones_like(self.nominal), self.enabled
        )
        mask = self.enabled.clone()
        mask[0] = False
        mask[1, 3] = False
        result = self.mapper.apply(
            self.nominal, torch.ones_like(self.nominal), mask
        )
        torch.testing.assert_close(result.realised_residual[0], torch.zeros(16))
        self.assertEqual(float(result.realised_residual[1, 3]), 0.0)

        # Re-enabling starts at the first-tick slew, not the residual carried
        # before the gate was closed.
        result = self.mapper.apply(
            self.nominal, torch.ones_like(self.nominal), self.enabled
        )
        torch.testing.assert_close(
            result.realised_residual[0, torch.tensor(ROTARY_ACTUATOR_INDICES)],
            torch.full((12,), 0.0001),
        )
        self.assertAlmostEqual(float(result.realised_residual[1, 3]), 0.00002)

    def test_ctrlrange_clipping_is_anti_windup(self) -> None:
        nominal_at_high = torch.ones_like(self.nominal)
        for _ in range(20):
            result = self.mapper.apply(
                nominal_at_high,
                torch.ones_like(self.nominal),
                self.enabled,
            )
        torch.testing.assert_close(result.applied_ctrl, nominal_at_high)
        torch.testing.assert_close(
            result.realised_residual, torch.zeros_like(self.nominal)
        )
        self.assertTrue(bool(result.ctrl_saturation_mask.all()))

        # No hidden outward residual is released when the nominal command
        # returns inside its ctrlrange.
        result = self.mapper.apply(
            self.nominal, torch.zeros_like(self.nominal), self.enabled
        )
        torch.testing.assert_close(result.applied_ctrl, self.nominal)
        torch.testing.assert_close(
            result.realised_residual, torch.zeros_like(self.nominal)
        )

    def test_masked_world_reset_and_state_round_trip(self) -> None:
        self.mapper.apply(
            self.nominal, torch.ones_like(self.nominal), self.enabled
        )
        before = self.mapper.residual.clone()
        self.mapper.reset(torch.tensor([False, True, False]))
        torch.testing.assert_close(self.mapper.residual[0], before[0])
        torch.testing.assert_close(self.mapper.residual[1], torch.zeros(16))
        torch.testing.assert_close(self.mapper.residual[2], before[2])

        state = self.mapper.state_dict()
        restored = BatchedActuatorResidualMapper(
            self.worlds,
            torch.full((16,), -1.0),
            torch.full((16,), 1.0),
        )
        restored.load_state_dict(state)
        torch.testing.assert_close(restored.residual, self.mapper.residual)

    def test_shape_dtype_and_finite_validation(self) -> None:
        with self.assertRaisesRegex(TypeError, "floating-point"):
            BatchedActuatorResidualMapper(
                1,
                torch.full((16,), -1),
                torch.full((16,), 1),
                dtype=torch.int64,
            )
        with self.assertRaisesRegex(ValueError, "normalized_action must have shape"):
            self.mapper.apply(
                self.nominal, torch.zeros((self.worlds, 15)), self.enabled
            )
        with self.assertRaisesRegex(TypeError, "boolean dtype"):
            self.mapper.apply(
                self.nominal,
                torch.zeros_like(self.nominal),
                torch.ones_like(self.nominal),
            )
        action = torch.zeros_like(self.nominal)
        action[1, 2] = torch.nan
        with self.assertRaisesRegex(FloatingPointError, "NaN or infinity"):
            self.mapper.apply(self.nominal, action, self.enabled)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_execution_keeps_all_state_and_outputs_on_device(self) -> None:
        mapper = BatchedActuatorResidualMapper(
            14000,
            torch.full((16,), -1.0),
            torch.full((16,), 1.0),
            device="cuda",
        )
        nominal = torch.zeros((14000, 16), device="cuda")
        action = torch.ones_like(nominal)
        mask = torch.ones_like(nominal, dtype=torch.bool)
        result = mapper.apply(nominal, action, mask)
        self.assertEqual(result.applied_ctrl.device.type, "cuda")
        self.assertEqual(result.realised_residual.shape, (14000, 16))
        self.assertEqual(mapper.residual.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
