from __future__ import annotations

import unittest

import numpy as np

import G3_grab as grab
from IK_RL.env import G3IKResidualEnv, _PlanningHandshake


class EnvironmentContractTests(unittest.TestCase):
    def test_one_step_one_tick_and_ik_only_zero_forcing(self) -> None:
        environment = G3IKResidualEnv(
            control_mode="ik_only",
            fixed_object="cube",
            fixed_task="translate_xy45_100mm",
        )
        try:
            observation, _info = environment.reset(
                seed=3, options={"pair_seed": 91}
            )
            self.assertEqual(observation.shape, (89,))
            _obs, _reward, terminal, _truncated, handshake = environment.step(
                np.ones(14)
            )
            self.assertFalse(terminal)
            self.assertTrue(handshake["planning_handshake"])
            self.assertFalse(handshake["control_tick"])
            self.assertEqual(handshake["control_step"], 0)
            self.assertEqual(
                handshake["plan_residual"]["height_residual_m"], 0.0
            )
            _obs, _reward, terminal, _truncated, tick = environment.step(
                np.ones(14)
            )
            self.assertFalse(terminal)
            self.assertTrue(tick["control_tick"])
            self.assertEqual(tick["control_step"], 1)
            self.assertTrue(
                np.array_equal(
                    environment.last_tick.base_ctrl,
                    environment.last_tick.applied_ctrl,
                )
            )
        finally:
            environment.close()

    def test_banana_zero_residual_preserves_baseline_plan_and_initial_state(
        self,
    ) -> None:
        baseline = G3IKResidualEnv(
            control_mode="ik_only",
            fixed_object="banana",
            fixed_task="translate_xy45_100mm",
        )
        residual = G3IKResidualEnv(
            control_mode="ik_residual",
            fixed_object="banana",
            fixed_task="translate_xy45_100mm",
        )
        try:
            baseline.reset(seed=5, options={"pair_seed": 19})
            _obs, _reward, baseline_terminal, baseline_truncated, baseline_info = baseline.step(
                np.ones(14)
            )

            residual.reset(seed=5, options={"pair_seed": 19})
            _obs, _reward, residual_terminal, residual_truncated, residual_info = residual.step(
                np.zeros(14)
            )
            self.assertEqual(baseline_terminal, residual_terminal)
            self.assertEqual(baseline_truncated, residual_truncated)
            self.assertEqual(
                baseline_info["failure_reason"], residual_info["failure_reason"]
            )
            self.assertEqual(
                baseline_info["initial_state_hash"],
                residual_info["initial_state_hash"],
            )
            for info in (baseline_info, residual_info):
                self.assertEqual(info["plan_residual"]["height_residual_m"], 0.0)
                self.assertEqual(info["plan_residual"]["yaw_residual_rad"], 0.0)

            baseline_plan = baseline.grabber.selected_plan
            residual_plan = residual.grabber.selected_plan
            self.assertIsNotNone(baseline_plan)
            self.assertIsNotNone(residual_plan)
            self.assertEqual(baseline_plan.object_kind, residual_plan.object_kind)
            self.assertEqual(baseline_plan.valid, residual_plan.valid)
            np.testing.assert_array_equal(
                baseline_plan.plane.vector6(), residual_plan.plane.vector6()
            )
            for attribute in (
                "pregrasp_targets_local",
                "contact_targets_local",
                "surface_points_local",
                "contact_normals_local",
                "outward_directions_local",
            ):
                baseline_targets = getattr(baseline_plan, attribute)
                residual_targets = getattr(residual_plan, attribute)
                self.assertEqual(baseline_targets.keys(), residual_targets.keys())
                for finger in baseline_targets:
                    np.testing.assert_array_equal(
                        baseline_targets[finger], residual_targets[finger]
                    )
        finally:
            baseline.close()
            residual.close()

    def test_reachable_banana_trial_preserves_true_contact_onset_failure(
        self,
    ) -> None:
        grabber = grab.IKGrabber(render=False)
        try:
            grabber.planner = _PlanningHandshake(
                grabber.planner,
                height_residual_m=0.0,
                yaw_residual_rad=0.0,
                before_plan=lambda: None,
            )
            result = grabber.run_trial("banana", seed=0)
        finally:
            grabber.close()

        self.assertFalse(result.success)
        self.assertEqual(
            result.termination_reason,
            "tip_contact_onset_not_simultaneous",
        )
        self.assertGreater(
            result.contact_onset_span_steps,
            grab.MAX_CONTACT_ONSET_SPAN_STEPS,
        )
        self.assertEqual(result.contact_count, 4)
        self.assertNotIn("coplanarity", result.termination_reason)
        self.assertTrue(
            grab._tip_formation_within_tolerance(
                grab.ObjectKind.BANANA,
                10.0 * grab.TIP_COPLANARITY_TOLERANCE_M,
                0.0,
            )
        )

    def test_nonfinite_action_never_reaches_mujoco(self) -> None:
        environment = G3IKResidualEnv(
            fixed_object="cube",
            fixed_task="rotate_z_45deg",
        )
        try:
            environment.reset(seed=1)
            action = np.zeros(14)
            action[0] = np.nan
            with self.assertRaises(ValueError):
                environment.step(action)
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
