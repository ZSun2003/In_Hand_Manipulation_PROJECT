"""Contracts for the no-training analytic-IK runner and baseline evaluation helpers."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from g3g6 import config, trajectories
from g3g6.env import G3G6ManipulationEnv
from g3g6.ik_control import (
    ExperimentSpec,
    add_route_distances,
    build_experiment_specs,
    quaternion_geodesic_error_deg,
)
from g3g6.rotation_sweep import (
    RUN_SCHEMA,
    TRACE_SCHEMA,
    RotationSweepSpec,
    _record_tick,
    build_rotation_sweep_specs,
)
from g3g6.rotations import (
    ROTATION_MOTION_TICKS,
    build_coupled_rotation_euler_schedule,
    build_rotation_euler_schedule,
    validate_goal_euler_deg,
)


class CoupledScheduleTests(unittest.TestCase):
    def test_rotation_occurs_only_during_translation_route(self) -> None:
        goal = np.array((30.0, -20.0, 0.0))
        schedule = build_coupled_rotation_euler_schedule(goal)
        route_start, route_stop = trajectories.PHASE_BOUNDS["route"]
        self.assertEqual(schedule.shape, (trajectories.TOTAL_TICKS, 3))
        np.testing.assert_array_equal(schedule[:route_start], 0.0)
        self.assertGreater(float(np.linalg.norm(schedule[route_start])), 0.0)
        np.testing.assert_allclose(schedule[route_stop - 1], goal, atol=1.0e-12)
        np.testing.assert_array_equal(
            schedule[route_stop:],
            np.repeat(goal[None, :], trajectories.TERMINAL_HOLD_TICKS, axis=0),
        )

    def test_explicit_coupled_target_contains_translation_and_rotation(self) -> None:
        env = G3G6ManipulationEnv(
            gripper="g3",
            phase=config.Phase.TRANSLATE,
            object_name="cube",
            path_name="circle",
            coupled_rotation_axes="xy",
            coupled_rotation_euler_deg=(20.0, -20.0, 0.0),
        )
        try:
            env.reset(seed=3)
            route_start = trajectories.PHASE_BOUNDS["route"][0]
            env.step_count = config.PRE_ACTION_TICKS + route_start
            target = env._target_for_tick(env.step_count)
            self.assertTrue(env.coupled_rotation)
            self.assertGreater(float(np.linalg.norm(target.position[:2])), 0.0)
            self.assertGreater(float(np.linalg.norm(target.quaternion[1:])), 0.0)
        finally:
            env.close()

    def test_rotation_sweep_opt_in_accepts_90_without_changing_default_limit(self) -> None:
        self.assertEqual(config.ROTATION_ANGLE_MAX_DEG, 45)
        with self.assertRaises(ValueError):
            validate_goal_euler_deg("xy", (90.0, 90.0, 0.0))
        accepted = validate_goal_euler_deg(
            "xy",
            (90.0, 90.0, 0.0),
            max_magnitude_deg=90.0,
        )
        np.testing.assert_array_equal(accepted, (90.0, 90.0, 0.0))
        with self.assertRaises(ValueError):
            validate_goal_euler_deg(
                "x",
                (90.0, 0.0, 0.0),
                max_magnitude_deg=config.ROTATION_SWEEP_ANGLE_MAX_DEG + 1.0e-9,
            )
        schedule = build_rotation_euler_schedule(accepted)
        np.testing.assert_allclose(schedule[-1], accepted, atol=0.0, rtol=0.0)
        self.assertTrue(np.all(np.diff(schedule[:, 0]) >= 0.0))
        np.testing.assert_array_equal(schedule[:, 2], 0.0)

    def test_rotation_phase_requires_explicit_extended_goal_limit(self) -> None:
        with self.assertRaises(ValueError):
            G3G6ManipulationEnv(
                gripper="g3",
                phase=config.Phase.ROTATE,
                object_name="cube",
                rotation_axes="x",
                rotation_goal_max_deg=config.ROTATION_SWEEP_ANGLE_MAX_DEG + 1.0e-9,
            )
        default_env = G3G6ManipulationEnv(
            gripper="g3",
            phase=config.Phase.ROTATE,
            object_name="cube",
            rotation_axes="x",
        )
        try:
            with self.assertRaises(ValueError):
                default_env.reset(
                    seed=3,
                    options={"rotation_axes": "x", "rotation_euler_deg": (90, 0, 0)},
                )
        finally:
            default_env.close()

        sweep_env = G3G6ManipulationEnv(
            gripper="g3",
            phase=config.Phase.ROTATE,
            object_name="cube",
            rotation_axes="x",
            rotation_goal_max_deg=90.0,
        )
        try:
            _observation, info = sweep_env.reset(
                seed=3,
                options={"rotation_axes": "x", "rotation_euler_deg": (90, 0, 0)},
            )
            np.testing.assert_array_equal(info["rotation_goal_euler_deg"], (90, 0, 0))
            self.assertEqual(info["rotation_goal_max_deg"], 90.0)
            self.assertEqual(
                sweep_env.task_target_grasp_plane_z_m,
                config.ROTATION_TARGET_GRASP_PLANE_Z_M,
            )
            self.assertEqual(
                info["task_target_grasp_plane_z_m"],
                config.ROTATION_TARGET_GRASP_PLANE_Z_M,
            )
            self.assertNotEqual(
                info["task_target_grasp_plane_z_m"],
                config.TARGET_GRASP_PLANE_Z_M,
            )
        finally:
            sweep_env.close()


class RotationSweepStaticContractTests(unittest.TestCase):
    def test_v2_trace_baseline_records_rotation_task_height_from_environment(
        self,
    ) -> None:
        self.assertTrue(RUN_SCHEMA.endswith("-v2"))
        self.assertTrue(TRACE_SCHEMA.endswith("-v2"))
        self.assertEqual(config.ROTATION_TARGET_GRASP_PLANE_Z_M, 0.325)
        self.assertEqual(config.TARGET_GRASP_PLANE_Z_M, 0.360)

        spec = RotationSweepSpec(
            gripper="g3",
            object_name="cube",
            rotation_axes="x",
            max_angle_deg=90.0,
            seed=7,
        )
        env = mock.Mock()
        env.task_target_grasp_plane_z_m = (
            config.ROTATION_TARGET_GRASP_PLANE_Z_M
        )
        env.nominal_ctrl = np.zeros(2, dtype=np.float64)
        env.applied_ctrl = np.zeros(2, dtype=np.float64)
        info = {
            "task_target_grasp_plane_z_m": (
                config.ROTATION_TARGET_GRASP_PLANE_Z_M
            ),
            "step": config.PRE_ACTION_TICKS,
            "phase": int(config.Phase.ROTATE),
            "task": "rotate",
            "angle_error_deg_per_axis": np.array((2.0, -3.0, 4.0)),
            "object_grasp_frame_position_m": np.array(
                (0.012, -0.018, 0.324)
            ),
            "target_position_m": np.array(
                (0.0, 0.0, config.ROTATION_TARGET_GRASP_PLANE_Z_M)
            ),
            "tip_contact_count": 4,
            "ik_exact": True,
            "ik_projection_fraction": 1.0,
            "max_abs_actuator_residual": 0.0,
            "is_success": False,
            "termination_reason": "running",
        }
        schedule = build_rotation_euler_schedule(spec.goal_euler_deg)
        row = _record_tick(
            spec,
            env,
            info,
            schedule,
            terminated=False,
        )

        self.assertEqual(row["schema"], TRACE_SCHEMA)
        self.assertEqual(row["reference_baseline"], 1)
        self.assertEqual(row["target_active_component_angle_deg"], 0.0)
        self.assertEqual(row["rotation_error_l2_deg"], 0.0)
        self.assertEqual(
            row["rotation_target_grasp_plane_z_m"],
            config.ROTATION_TARGET_GRASP_PLANE_Z_M,
        )
        self.assertEqual(
            row["target_position_m_z"],
            config.ROTATION_TARGET_GRASP_PLANE_Z_M,
        )
        self.assertEqual(row["target_position_m_x"], 0.012)
        self.assertEqual(row["target_position_m_y"], -0.018)

    def test_two_grippers_four_objects_and_seven_axis_sets_make_56_unique_specs(
        self,
    ) -> None:
        specs = build_rotation_sweep_specs(
            grippers=("g3", "g6"),
            objects=config.OBJECT_NAMES,
            axes=config.ROTATION_AXIS_SETS,
            max_angle_deg=config.ROTATION_SWEEP_ANGLE_MAX_DEG,
            seed=7,
        )
        self.assertEqual(len(specs), 2 * 4 * 7)
        self.assertEqual(len({spec.episode_id for spec in specs}), len(specs))
        self.assertEqual(
            {(spec.gripper, spec.object_name, spec.rotation_axes) for spec in specs},
            {
                (gripper, object_name, axis_set)
                for gripper in ("g3", "g6")
                for object_name in config.OBJECT_NAMES
                for axis_set in config.ROTATION_AXIS_SETS
            },
        )

        seeds_by_case: dict[tuple[str, str], dict[str, int]] = {}
        for spec in specs:
            case = (spec.object_name, spec.rotation_axes)
            seeds_by_case.setdefault(case, {})[spec.gripper] = spec.seed
        for case, gripper_seeds in seeds_by_case.items():
            with self.subTest(object=case[0], rotation_axes=case[1]):
                self.assertEqual(set(gripper_seeds), {"g3", "g6"})
                self.assertEqual(gripper_seeds["g3"], gripper_seeds["g6"])

        with self.assertRaises(ValueError):
            build_rotation_sweep_specs(
                grippers=("g3",),
                objects=("cube",),
                axes=("x",),
                max_angle_deg=config.ROTATION_SWEEP_ANGLE_MAX_DEG + 1.0e-9,
                seed=7,
            )

    def test_common_active_component_is_the_zero_to_90_degree_horizontal_axis(
        self,
    ) -> None:
        specs = build_rotation_sweep_specs(
            grippers=("g3",),
            objects=("cube",),
            axes=config.ROTATION_AXIS_SETS,
            max_angle_deg=config.ROTATION_SWEEP_ANGLE_MAX_DEG,
            seed=7,
        )
        for spec in specs:
            with self.subTest(rotation_axes=spec.rotation_axes):
                expected_goal = np.array(
                    [
                        config.ROTATION_SWEEP_ANGLE_MAX_DEG
                        if axis in spec.rotation_axes
                        else 0.0
                        for axis in "xyz"
                    ],
                    dtype=np.float64,
                )
                np.testing.assert_array_equal(spec.goal_euler_deg, expected_goal)

                schedule = build_rotation_euler_schedule(expected_goal)
                motion = schedule[:ROTATION_MOTION_TICKS]
                first_active_index = "xyz".index(spec.rotation_axes[0])
                theta = motion[:, first_active_index]
                self.assertGreater(theta[0], 0.0)
                self.assertLess(theta[0], 1.0e-6)
                self.assertEqual(theta[-1], config.ROTATION_SWEEP_ANGLE_MAX_DEG)
                self.assertTrue(np.all(np.diff(theta) >= 0.0))

                for index, axis in enumerate("xyz"):
                    if axis in spec.rotation_axes:
                        np.testing.assert_array_equal(motion[:, index], theta)
                    else:
                        np.testing.assert_array_equal(motion[:, index], 0.0)
                np.testing.assert_array_equal(
                    schedule[ROTATION_MOTION_TICKS:],
                    np.repeat(
                        expected_goal[None, :],
                        schedule.shape[0] - ROTATION_MOTION_TICKS,
                        axis=0,
                    ),
                )


class PureIKStepTests(unittest.TestCase):
    def test_step_ik_bypasses_residual_action_for_both_grippers(self) -> None:
        for gripper in ("g3", "g6"):
            with self.subTest(gripper=gripper):
                env = G3G6ManipulationEnv(
                    gripper=gripper,
                    phase=config.Phase.GRASP_LIFT,
                    object_name="cube",
                )
                try:
                    env.reset(seed=5)
                    with mock.patch("g3g6.env.mujoco.mj_step"):
                        _observation, _reward, _terminated, _truncated, info = (
                            env.step_ik()
                        )
                    self.assertEqual(info["control_mode"], "pure_ik_feedback")
                    self.assertEqual(info["max_abs_actuator_residual"], 0.0)
                    np.testing.assert_array_equal(env.last_residual, 0.0)
                    np.testing.assert_array_equal(env.applied_ctrl, env.nominal_ctrl)
                finally:
                    env.close()


class TraceMetricTests(unittest.TestCase):
    @staticmethod
    def _row(action_tick: int, target_x: float, actual_x: float) -> dict[str, float | int]:
        return {
            "action_tick": action_tick,
            "target_position_m_x": target_x,
            "target_position_m_y": 0.0,
            "actual_position_m_x": actual_x,
            "actual_position_m_y": 0.0,
            "target_route_distance_m": 0.0,
            "actual_route_distance_m": 0.0,
        }

    def test_route_distance_is_cumulative_and_hold_does_not_add_distance(self) -> None:
        rows = [
            self._row(598, 8.0, 8.0),
            self._row(599, 0.0, 0.0),
            self._row(600, 0.1, 0.08),
            self._row(601, 0.1, 0.10),
            self._row(602, 0.2, 0.18),
            self._row(2400, 9.0, 9.0),
        ]
        add_route_distances(rows)
        self.assertAlmostEqual(float(rows[4]["target_route_distance_m"]), 0.2)
        self.assertAlmostEqual(float(rows[4]["actual_route_distance_m"]), 0.18)
        self.assertAlmostEqual(
            float(rows[5]["target_route_distance_m"]),
            float(rows[4]["target_route_distance_m"]),
        )

    def test_quaternion_geodesic_error_is_sign_invariant(self) -> None:
        quaternion = np.array((0.5, 0.5, 0.5, 0.5))
        self.assertAlmostEqual(
            quaternion_geodesic_error_deg(quaternion, -quaternion), 0.0
        )



    def test_full_default_style_suite_has_translation_and_seven_axis_sets(self) -> None:
        specs = build_experiment_specs(
            grippers=("g3",),
            objects=("cube",),
            paths=("circle",),
            axes=config.ROTATION_AXIS_SETS,
            suite="full",
            rotation_angle_deg=30.0,
            seed=7,
        )
        self.assertEqual(len(specs), 8)
        self.assertIsInstance(specs[0], ExperimentSpec)
        self.assertIsNone(specs[0].rotation_axes)
        self.assertEqual(
            tuple(spec.rotation_axes for spec in specs[1:]),
            config.ROTATION_AXIS_SETS,
        )

    def test_four_objects_all_paths_full_suite_has_160_unique_episodes(self) -> None:
        specs = build_experiment_specs(
            grippers=("g3",),
            objects=config.OBJECT_NAMES,
            paths=trajectories.PATH_NAMES,
            axes=config.ROTATION_AXIS_SETS,
            suite="full",
            rotation_angle_deg=30.0,
            seed=7,
        )
        self.assertEqual(len(specs), 4 * 5 * 8)
        self.assertEqual(len({spec.episode_id for spec in specs}), len(specs))


if __name__ == "__main__":
    unittest.main()
