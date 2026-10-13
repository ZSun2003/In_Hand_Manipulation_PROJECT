#!/usr/bin/env python3
"""Task-level contracts for the non-redundant six-dimensional residual stack."""

from __future__ import annotations

import math
import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from kinematics import ACTUATOR_COUNT, GripperIK
from residual_control import (
    RESIDUAL_ACTION_DIM,
    RESIDUAL_ACTION_NAMES,
    ResidualCommandProjector,
)
from screw_config import DEFAULT_CONFIG
from screw_control import (
    ACTION_DIM,
    BASE_FINGER_ANGLES_RAD,
    FINGERS,
    SCRIPTED_CENTER_TRANSFER_STEPS,
    SCRIPTED_CLOSE_STEPS,
    SCRIPTED_PICKUP_LIFT_STEPS,
    SCRIPTED_PREFIX_STEPS,
    SCRIPTED_PREGRASP_STEPS,
    SOCKET_ALIGN_STEPS,
    SOCKET_YAW_ROLL_GAIN,
    ScrewCommand,
    ScrewExpert,
    TaskPhase,
    align_local_z_to,
    finger_quadrant_permutation,
    minimum_jerk,
    permute_regrasp_phase,
    rot_z,
)
from screw_env import ScrewTaskEnv
from screw_sim import ScrewSimulator


def make_command(
    *,
    phase: TaskPhase = TaskPhase.POWER,
    engaged: bool = True,
    twist_rad: float = 1.0,
) -> ScrewCommand:
    return ScrewCommand(
        tip_world=np.array([0.021, -0.013, 0.307], dtype=np.float64),
        rotation_world=align_local_z_to(
            -np.asarray(DEFAULT_CONFIG.socket_axis_world, dtype=np.float64)
        ),
        twist_rad=float(twist_rad),
        material_angles_rad=dict(BASE_FINGER_ANGLES_RAD),
        tip_center_distances_m={finger: 0.0405 for finger in FINGERS},
        phase=phase,
        engaged=engaged,
    )


def assert_command_exact(
    testcase: unittest.TestCase,
    actual: ScrewCommand,
    expected: ScrewCommand,
) -> None:
    """Bitwise equality required by the zero-residual teacher contract."""
    testcase.assertTrue(np.array_equal(actual.tip_world, expected.tip_world))
    testcase.assertTrue(
        np.array_equal(actual.rotation_world, expected.rotation_world)
    )
    testcase.assertEqual(actual.twist_rad, expected.twist_rad)
    testcase.assertEqual(actual.material_angles_rad, expected.material_angles_rad)
    testcase.assertEqual(
        actual.tip_center_distances_m,
        expected.tip_center_distances_m,
    )
    testcase.assertEqual(actual.phase, expected.phase)
    testcase.assertEqual(actual.engaged, expected.engaged)


def make_expert(cfg=DEFAULT_CONFIG) -> ScrewExpert:
    return ScrewExpert(
        spawn_xy_m=cfg.default_screw_xy_m,
        table_top_z_m=cfg.table_top_z_m,
        socket_mouth_world_m=cfg.socket_mouth_position_m,
        socket_axis_inward_world=cfg.socket_axis_world,
        engage_depth_m=cfg.thread_engagement_depth_m,
        final_depth_m=cfg.target_insertion_depth_m,
        pitch_m=cfg.thread_pitch_m,
        target_turns=cfg.target_turns,
        released_distance_m=cfg.released_tip_center_distance_m,
        grasp_height_from_tip_m=cfg.screw_grasp_height_from_tip_m,
        scripted_grasp_plane_world_z_m=(
            cfg.scripted_grasp_plane_world_z_m
        ),
        scripted_center_xy_m=cfg.scripted_center_xy_m,
    )


class ScriptedPrefixTests(unittest.TestCase):
    def test_pickup_then_centre_trajectory_has_exact_handoff_boundary(self):
        cfg = DEFAULT_CONFIG
        expert = make_expert()
        pickup_start = SCRIPTED_PREGRASP_STEPS + SCRIPTED_CLOSE_STEPS
        pickup_end = pickup_start + SCRIPTED_PICKUP_LIFT_STEPS
        handoff = pickup_end + SCRIPTED_CENTER_TRANSFER_STEPS
        self.assertEqual(
            (
                SCRIPTED_PREGRASP_STEPS,
                SCRIPTED_CLOSE_STEPS,
                SCRIPTED_PICKUP_LIFT_STEPS,
                SCRIPTED_CENTER_TRANSFER_STEPS,
            ),
            (25, 40, 50, 100),
        )
        self.assertEqual(handoff, 215)
        self.assertEqual(SCRIPTED_PREFIX_STEPS, 215)
        self.assertEqual(expert.diffusion_handoff_step, 215)

        commands = [expert.current]
        commands.extend(
            expert.advance() for _ in range(SCRIPTED_PREFIX_STEPS)
        )
        spawn_xy = np.asarray(cfg.default_screw_xy_m, dtype=np.float64)
        target_tip = np.array(
            (
                cfg.scripted_center_xy_m[0],
                cfg.scripted_center_xy_m[1],
                0.320,
            ),
            dtype=np.float64,
        )

        # The screw remains upright and stationary while the fingers
        # pregrasp and close.
        for command in commands[: pickup_start + 1]:
            np.testing.assert_allclose(command.tip_world[:2], spawn_xy)
            self.assertAlmostEqual(command.tip_world[2], cfg.table_top_z_m)
            np.testing.assert_array_equal(
                command.rotation_world, np.eye(3, dtype=np.float64)
            )

        # Pickup changes only height: the 100 mm tip-to-grasp offset places
        # the grasp plane at z=0.420 exactly when the tip reaches z=0.320.
        lift = commands[pickup_start : pickup_end + 1]
        lift_z = np.asarray([command.tip_world[2] for command in lift])
        self.assertTrue(np.all(np.diff(lift_z) >= -1.0e-15))
        for command in lift:
            np.testing.assert_allclose(command.tip_world[:2], spawn_xy)
        self.assertAlmostEqual(lift[-1].tip_world[2], 0.320)
        self.assertAlmostEqual(
            lift[-1].tip_world[2] + cfg.screw_grasp_height_from_tip_m,
            0.420,
        )

        # The second move holds that height and transfers the upright screw
        # to the world centre.
        transfer = commands[pickup_end : handoff + 1]
        centre_distance = np.asarray(
            [
                np.linalg.norm(
                    command.tip_world[:2]
                    - np.asarray(cfg.scripted_center_xy_m)
                )
                for command in transfer
            ]
        )
        self.assertTrue(np.all(np.diff(centre_distance) <= 1.0e-15))
        for command in transfer:
            self.assertAlmostEqual(command.tip_world[2], 0.320)
            np.testing.assert_array_equal(
                command.rotation_world, np.eye(3, dtype=np.float64)
            )
        np.testing.assert_allclose(commands[handoff].tip_world, target_tip)

        # Step 215 is the measured handoff pose. Learned control starts only
        # after the gate advances to the first post-handoff command.
        self.assertEqual(expert.step_index, 215)
        self.assertFalse(expert.learned_control_active)
        expert.advance()
        self.assertEqual(expert.step_index, 216)
        self.assertTrue(expert.learned_control_active)


class SocketYawEquivarianceTests(unittest.TestCase):
    @staticmethod
    def config_at_azimuth(degrees: float):
        yaw = math.radians(float(degrees))
        radius = DEFAULT_CONFIG.socket_mouth_orbit_radius_m
        return DEFAULT_CONFIG.with_socket_pose(
            (radius * math.cos(yaw), radius * math.sin(yaw)),
            45.0,
        )

    @staticmethod
    def advance_to(expert: ScrewExpert, step: int) -> ScrewCommand:
        expert.reset()
        for _ in range(int(step)):
            expert.advance()
        return expert.current

    def test_cardinal_rotation_maps_pose_finger_roles_and_targets_exactly(self):
        canonical = make_expert(self.config_at_azimuth(0.0))
        rotated = make_expert(self.config_at_azimuth(90.0))
        Q = rot_z(0.5 * math.pi)
        permutation = finger_quadrant_permutation(1)

        self.assertEqual(rotated.socket_quadrant, 1)
        self.assertAlmostEqual(rotated.socket_residual_yaw_rad, 0.0)
        self.assertEqual(
            permutation,
            {"U": "L", "D": "R", "L": "D", "R": "U"},
        )

        for step in (0, 216, 226, 291, 365, 516, 703, 2515):
            with self.subTest(step=step):
                first = self.advance_to(canonical, step)
                second = self.advance_to(rotated, step)
                np.testing.assert_allclose(
                    second.tip_world, Q @ first.tip_world, atol=1.0e-14
                )
                np.testing.assert_allclose(
                    second.rotation_world,
                    Q @ first.rotation_world @ Q.T,
                    atol=1.0e-14,
                )
                self.assertEqual(
                    second.phase,
                    permute_regrasp_phase(first.phase, 1),
                )
                for logical, physical in permutation.items():
                    self.assertAlmostEqual(
                        second.material_angles_rad[physical],
                        first.material_angles_rad[logical] + 0.5 * math.pi,
                        delta=1.0e-14,
                    )
                    self.assertAlmostEqual(
                        second.tip_center_distances_m[physical],
                        first.tip_center_distances_m[logical],
                        delta=1.0e-14,
                    )

                first_targets = first.fingertip_targets(
                    grasp_height_from_tip_m=(
                        DEFAULT_CONFIG.screw_grasp_height_from_tip_m
                    )
                )
                second_targets = second.fingertip_targets(
                    grasp_height_from_tip_m=(
                        DEFAULT_CONFIG.screw_grasp_height_from_tip_m
                    )
                )
                for logical, physical in permutation.items():
                    np.testing.assert_allclose(
                        second_targets[physical],
                        Q @ first_targets[logical],
                        atol=2.0e-14,
                    )

    def test_action_finger_dimensions_follow_physical_role_permutation(self):
        canonical = make_expert(self.config_at_azimuth(0.0))
        rotated = make_expert(self.config_at_azimuth(90.0))
        permutation = finger_quadrant_permutation(1)

        previous_first = canonical.reset()
        previous_second = rotated.reset()
        checked_nonzero_material = False
        checked_nonzero_distance = False
        for _step in range(1, 330):
            following_first = canonical.advance()
            following_second = rotated.advance()
            action_first = canonical.command_action(
                previous_first, following_first
            )
            action_second = rotated.command_action(
                previous_second, following_second
            )
            for logical, physical in permutation.items():
                logical_index = FINGERS.index(logical)
                physical_index = FINGERS.index(physical)
                self.assertAlmostEqual(
                    float(action_second[6 + physical_index]),
                    float(action_first[6 + logical_index]),
                    delta=2.0e-7,
                )
                self.assertAlmostEqual(
                    float(action_second[10 + physical_index]),
                    float(action_first[10 + logical_index]),
                    delta=2.0e-9,
                )
            checked_nonzero_material |= bool(
                np.any(np.abs(action_first[6:10]) > 0.0)
            )
            checked_nonzero_distance |= bool(
                np.any(np.abs(action_first[10:14]) > 0.0)
            )
            previous_first = following_first
            previous_second = following_second

        self.assertTrue(checked_nonzero_material)
        self.assertTrue(checked_nonzero_distance)

    def test_89_9_degree_boundary_right_multiplies_roll_after_interpolation(self):
        expert = make_expert(self.config_at_azimuth(89.9))
        self.assertEqual(expert.socket_quadrant, 0)
        self.assertAlmostEqual(
            math.degrees(expert.socket_residual_yaw_rad),
            89.9,
            delta=1.0e-10,
        )
        self.assertAlmostEqual(
            math.degrees(expert.socket_roll_rad),
            SOCKET_YAW_ROLL_GAIN * 89.9,
            delta=1.0e-10,
        )
        self.assertEqual(expert._align_roll_boundary_step, 365)
        self.assertEqual(
            expert._align_roll_end_step,
            expert._align_roll_boundary_step + SOCKET_ALIGN_STEPS,
        )

        for step in (365, 366, 400, 515, 516):
            with self.subTest(step=step):
                command = self.advance_to(expert, step)
                fraction = (
                    step - expert._align_roll_boundary_step
                ) / float(SOCKET_ALIGN_STEPS)
                expected_roll = (
                    expert.socket_roll_rad * minimum_jerk(fraction)
                    if step > expert._align_roll_boundary_step
                    else 0.0
                )
                expected_rotation = (
                    expert.quadrant_rotation_world
                    @ expert._last_command.rotation_world
                    @ expert.quadrant_rotation_world.T
                    @ rot_z(expected_roll)
                )
                np.testing.assert_allclose(
                    command.rotation_world,
                    expected_rotation,
                    atol=2.0e-14,
                )

        self.assertAlmostEqual(
            expert._socket_roll_at_step(515),
            expert.socket_roll_rad,
            delta=1.0e-15,
        )
        self.assertAlmostEqual(
            expert._socket_roll_at_step(516),
            expert.socket_roll_rad,
            delta=1.0e-15,
        )

    @unittest.skipUnless(
        os.environ.get("SCREW_RUN_SLOW_IK") == "1",
        "set SCREW_RUN_SLOW_IK=1 for the full eight-azimuth IK sweep",
    )
    def test_full_trajectory_ik_for_eight_socket_azimuths(self):
        solver = GripperIK.from_xml(Path(__file__).with_name("Project.xml"))
        for degrees in range(0, 360, 45):
            with self.subTest(degrees=degrees):
                expert = make_expert(self.config_at_azimuth(float(degrees)))
                previous_ctrl = np.zeros(
                    ACTUATOR_COUNT, dtype=np.float64
                )
                command = expert.reset()
                for step in range(expert.total_steps + 1):
                    solution = solver.solve_tip_targets(
                        command.fingertip_targets(
                            grasp_height_from_tip_m=(
                                DEFAULT_CONFIG.screw_grasp_height_from_tip_m
                            )
                        ),
                        prev_ctrl=previous_ctrl,
                        strict=False,
                    )
                    self.assertTrue(
                        solution.success,
                        msg=f"azimuth={degrees} step={step}",
                    )
                    previous_ctrl = solution.ctrl
                    if step < expert.total_steps:
                        command = expert.advance()


class SpawnTrajectoryRejectionTests(unittest.TestCase):
    def test_reset_rejects_any_unreachable_step_in_continuous_prefix(self):
        class FakeSolver:
            def __init__(self):
                self.mode = "good"
                self.calls = 0
                self.previous_seed_checks = 0

            def solve_tip_targets(
                self, _targets, *, prev_ctrl=None, strict=False
            ):
                del strict
                self.calls += 1
                expected_seed = np.zeros(
                    ACTUATOR_COUNT, dtype=np.float64
                )
                if self.calls > 1:
                    expected_seed.fill(float(self.calls - 1))
                np.testing.assert_array_equal(prev_ctrl, expected_seed)
                self.previous_seed_checks += 1
                success = not (
                    self.mode == "bad" and self.calls == 61
                )
                return mock.Mock(
                    success=success,
                    ctrl=np.full(
                        ACTUATOR_COUNT,
                        float(self.calls),
                        dtype=np.float64,
                    ),
                )

        solver = FakeSolver()
        audit = {}

        def fake_sample(_rng, _cfg, *, ik_reachable, reject):
            del ik_reachable
            solver.mode = "bad"
            solver.calls = 0
            self.assertTrue(reject((-0.080, 0.0, 0.360)))
            audit["bad_calls"] = solver.calls

            solver.mode = "good"
            solver.calls = 0
            self.assertFalse(reject((-0.070, -0.020, 0.360)))
            audit["good_calls"] = solver.calls
            return (-0.070, -0.020)

        with mock.patch(
            "screw_sim.sample_spawn_xy", side_effect=fake_sample
        ):
            sampled = ScrewSimulator.sample_reachable_spawn(
                config=DEFAULT_CONFIG,
                seed=3,
                ik=solver,
            )

        self.assertEqual(sampled, (-0.070, -0.020))
        self.assertEqual(audit["bad_calls"], 61)
        self.assertEqual(
            audit["good_calls"], SCRIPTED_PREFIX_STEPS + 1
        )


class ResidualInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.socket_rotation = align_local_z_to(
            DEFAULT_CONFIG.socket_axis_world
        )

    def projector(self) -> ResidualCommandProjector:
        return ResidualCommandProjector(
            socket_rotation_world=self.socket_rotation,
            pitch_m=DEFAULT_CONFIG.thread_pitch_m,
        )

    def test_action_dimension_and_names(self):
        self.assertEqual(RESIDUAL_ACTION_DIM, 6)
        self.assertEqual(
            RESIDUAL_ACTION_NAMES,
            (
                "lateral_x",
                "lateral_y",
                "axial",
                "tilt_x",
                "tilt_y",
                "tighten",
            ),
        )

    def test_zero_action_is_bitwise_identical_to_teacher_from_zero_state(self):
        projector = self.projector()
        projector.reset()
        # The episode starts upright, so the real teacher's initial rotation
        # is exactly identity rather than the later socket-aligned frame.
        teacher = make_command(
            phase=TaskPhase.PREGRASP,
            engaged=False,
            twist_rad=0.0,
        )
        teacher.rotation_world = np.eye(3, dtype=np.float64)
        projected = projector.project(
            teacher,
            np.zeros(RESIDUAL_ACTION_DIM, dtype=np.float32),
        )
        assert_command_exact(self, projected, teacher)
        np.testing.assert_array_equal(
            projector.last_applied,
            np.zeros(RESIDUAL_ACTION_DIM, dtype=np.float32),
        )

    def test_tighten_dimension_is_inactive_before_thread_engagement(self):
        projector = self.projector()
        teacher = make_command(
            phase=TaskPhase.INSERT,
            engaged=False,
            twist_rad=0.0,
        )
        projected = projector.project(
            teacher,
            np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(projected.tip_world, teacher.tip_world, atol=0.0)
        np.testing.assert_allclose(
            projected.rotation_world,
            teacher.rotation_world,
            atol=1e-15,
        )
        self.assertEqual(projected.twist_rad, teacher.twist_rad)

    def test_unengaged_axial_is_gated_to_alignment_insertion_and_engage(self):
        active_phases = (
            TaskPhase.ALIGN,
            TaskPhase.INSERT,
            TaskPhase.ENGAGE,
        )
        inactive_phases = (
            TaskPhase.PREGRASP,
            TaskPhase.CLOSE,
            TaskPhase.LIFT,
            TaskPhase.POWER,
            TaskPhase.FINAL_HOLD,
        )
        axial_action = np.array(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        for phase in active_phases:
            with self.subTest(active=phase.name):
                projector = self.projector()
                teacher = make_command(phase=phase, engaged=False)
                projected = projector.project(teacher, axial_action)
                socket_delta = self.socket_rotation.T @ (
                    projected.tip_world - teacher.tip_world
                )
                self.assertAlmostEqual(socket_delta[0], 0.0, delta=1e-15)
                self.assertAlmostEqual(socket_delta[1], 0.0, delta=1e-15)
                self.assertAlmostEqual(
                    socket_delta[2],
                    projector.limits.axial_delta_m_per_step,
                    delta=1e-15,
                )
        for phase in inactive_phases:
            with self.subTest(inactive=phase.name):
                projector = self.projector()
                teacher = make_command(phase=phase, engaged=False)
                projected = projector.project(teacher, axial_action)
                np.testing.assert_allclose(
                    projected.tip_world,
                    teacher.tip_world,
                    atol=0.0,
                )

    def test_last_three_dimensions_map_to_socket_xyz_rotation(self):
        projector = self.projector()
        teacher = make_command(
            phase=TaskPhase.ALIGN,
            engaged=False,
        )
        action = np.array(
            [0.0, 0.0, 0.0, 1.0, -1.0, 1.0],
            dtype=np.float32,
        )
        projected = projector.project(teacher, action)
        # Before engagement, Z/tighten is suppressed while X/Y tilt remain.
        relative_rotation = (
            self.socket_rotation.T
            @ projected.rotation_world
            @ teacher.rotation_world.T
            @ self.socket_rotation
        )
        expected_tilt = projector.limits.tilt_delta_rad_per_step
        from screw_control import rotation_exp

        np.testing.assert_allclose(
            relative_rotation,
            rotation_exp((expected_tilt, -expected_tilt, 0.0)),
            atol=1e-14,
        )
        self.assertEqual(projected.twist_rad, teacher.twist_rad)

    def test_engaged_tighten_obeys_exact_pitch_coupling(self):
        projector = self.projector()
        teacher = make_command(engaged=True, twist_rad=1.25)
        # Deliberately request raw axial motion too: after engagement it must
        # be replaced by, not added to, the helical lead from tighten.
        action = np.array(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.5],
            dtype=np.float32,
        )
        projected = projector.project(teacher, action)

        twist_offset = (
            float(action[5]) * projector.limits.twist_rad
        )
        expected_advance = (
            DEFAULT_CONFIG.thread_pitch_m
            * twist_offset
            / (2.0 * math.pi)
        )
        world_delta = projected.tip_world - teacher.tip_world
        socket_delta = self.socket_rotation.T @ world_delta
        self.assertAlmostEqual(socket_delta[0], 0.0, delta=1e-15)
        self.assertAlmostEqual(socket_delta[1], 0.0, delta=1e-15)
        self.assertAlmostEqual(
            socket_delta[2],
            expected_advance,
            delta=1e-15,
        )
        self.assertAlmostEqual(
            projected.twist_rad - teacher.twist_rad,
            twist_offset,
            delta=1e-15,
        )

        relative_rotation = (
            self.socket_rotation.T
            @ projected.rotation_world
            @ teacher.rotation_world.T
            @ self.socket_rotation
        )
        measured_angle = math.atan2(
            float(relative_rotation[1, 0]),
            float(relative_rotation[0, 0]),
        )
        self.assertAlmostEqual(measured_angle, twist_offset, delta=1e-12)

    def test_regrasp_phases_freeze_all_pose_residuals(self):
        regrasp_phases = (
            TaskPhase.LR_OPEN,
            TaskPhase.LR_RESET,
            TaskPhase.LR_CLOSE,
            TaskPhase.UD_OPEN,
            TaskPhase.UD_RESET,
            TaskPhase.UD_CLOSE,
        )
        for phase in regrasp_phases:
            with self.subTest(phase=phase.name):
                projector = self.projector()
                teacher = make_command(phase=phase, engaged=True)
                projected = projector.project(
                    teacher,
                    np.ones(RESIDUAL_ACTION_DIM, dtype=np.float32),
                )
                np.testing.assert_allclose(
                    projected.tip_world,
                    teacher.tip_world,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    projected.rotation_world,
                    teacher.rotation_world,
                    atol=1e-15,
                )
                self.assertEqual(projected.twist_rad, teacher.twist_rad)
                np.testing.assert_array_equal(
                    projector.lateral_state_m,
                    np.zeros(2, dtype=np.float64),
                )
                np.testing.assert_array_equal(
                    projector.tilt_state_rad,
                    np.zeros(2, dtype=np.float64),
                )


class SimulatorProjectionTests(unittest.TestCase):
    def test_unreachable_candidate_projects_from_certified_safe_anchor(self):
        simulator = ScrewSimulator(seed=11)
        try:
            safe = simulator.current_command.copy()
            candidate = safe.copy()
            candidate.tip_world = (
                np.asarray(candidate.tip_world, dtype=np.float64)
                + np.array([2.0, 2.0, 2.0], dtype=np.float64)
            )

            candidate_solution = simulator.teacher.solve_tip_targets(
                candidate.fingertip_targets(
                    grasp_height_from_tip_m=(
                        simulator.config.screw_grasp_height_from_tip_m
                    )
                ),
                strict=False,
                update=False,
            )
            self.assertFalse(candidate_solution.success)

            projected, fraction = simulator.project_command_to_ik(
                safe,
                candidate,
            )
            self.assertGreaterEqual(fraction, 0.0)
            self.assertLess(fraction, 1.0)
            expected_tip = (
                (1.0 - fraction) * safe.tip_world
                + fraction * candidate.tip_world
            )
            np.testing.assert_allclose(projected.tip_world, expected_tip)

            projected_solution = simulator.teacher.solve_tip_targets(
                projected.fingertip_targets(
                    grasp_height_from_tip_m=(
                        simulator.config.screw_grasp_height_from_tip_m
                    )
                ),
                strict=False,
                update=False,
            )
            self.assertTrue(projected_solution.success)
        finally:
            simulator.close()

    def test_unreachable_supervisor_uses_last_certified_command(self):
        simulator = ScrewSimulator(seed=12)
        try:
            certified = simulator._last_certified_ik_command.copy()
            unreachable = certified.copy()
            unreachable.tip_world = (
                np.asarray(unreachable.tip_world, dtype=np.float64)
                + np.array([2.0, 2.0, 2.0], dtype=np.float64)
            )

            projected, fraction = simulator.project_command_to_ik(
                unreachable,
                unreachable,
            )

            self.assertEqual(fraction, 0.0)
            assert_command_exact(self, projected, certified)
            self.assertTrue(
                simulator.last_ik_projection_used_fallback
            )
            self.assertFalse(simulator.ik_safety_stop)
            self.assertEqual(
                simulator.ik_safety_reason,
                "supervisor_anchor_unreachable",
            )
            self.assertEqual(simulator.ik_anchor_fallback_count, 1)
            self.assertEqual(simulator.ik_safety_stop_count, 0)
            diagnostics = simulator.apply_command(projected)
            self.assertTrue(diagnostics.ik_success)
        finally:
            simulator.close()

    def test_lost_certified_anchor_holds_without_raising(self):
        simulator = ScrewSimulator(seed=13)
        try:
            unreachable = simulator.current_command.copy()
            unreachable.tip_world = (
                np.asarray(unreachable.tip_world, dtype=np.float64)
                + np.array([2.0, 2.0, 2.0], dtype=np.float64)
            )
            simulator._last_certified_ik_command = unreachable.copy()

            _projected, fraction = simulator.project_command_to_ik(
                unreachable,
                unreachable,
            )

            self.assertEqual(fraction, 0.0)
            self.assertTrue(simulator.ik_safety_stop)
            self.assertEqual(
                simulator.ik_safety_reason,
                "certified_ik_anchor_lost",
            )
            self.assertEqual(simulator.ik_safety_stop_count, 1)
            previous_step = simulator.control_step
            diagnostics = simulator.apply_ik_safety_hold()
            self.assertFalse(diagnostics.ik_success)
            self.assertEqual(simulator.control_step, previous_step + 1)
        finally:
            simulator.close()


class EnvironmentSmokeTests(unittest.TestCase):
    def test_default_environment_keeps_socket_inclination_fixed_at_45_degrees(
        self,
    ):
        environment = ScrewTaskEnv(
            baseline="expert",
            random_spawn=False,
            guide_mode="mocap",
            guide_scale=1.0,
            max_steps=8,
            seed=4,
        )
        try:
            self.assertFalse(environment.random_socket_angle)
            self.assertEqual(environment.socket_angle_min_deg, 45.0)
            self.assertEqual(environment.socket_angle_max_deg, 45.0)
            self.assertEqual(
                {environment._select_socket_angle(None) for _ in range(32)},
                {45.0},
            )
            with self.assertRaisesRegex(ValueError, "fixed at 45"):
                environment._select_socket_angle(
                    {"socket_angle_deg": 90.0}
                )
        finally:
            environment.close()

        with self.assertRaisesRegex(ValueError, "must be False"):
            ScrewTaskEnv(random_socket_angle=True)

    def test_pre_handoff_masks_external_diffusion_and_residual_actions(self):
        environment = ScrewTaskEnv(
            baseline="external_diffusion",
            random_spawn=False,
            random_socket_position=False,
            random_socket_angle=False,
            guide_mode="mocap",
            guide_scale=1.0,
            max_steps=8,
            seed=5,
        )
        try:
            environment.reset(seed=5)
            assert environment.diffusion_adapter is not None
            with mock.patch.object(
                environment.diffusion_adapter,
                "adapt",
                side_effect=AssertionError(
                    "Diffusion must not run during the scripted prefix"
                ),
            ) as adapt:
                _observation, _reward, _terminated, _truncated, info = (
                    environment.step(
                        np.ones(
                            RESIDUAL_ACTION_DIM + ACTION_DIM,
                            dtype=np.float32,
                        )
                    )
                )
            adapt.assert_not_called()
            self.assertFalse(info["diffusion_active"])
            self.assertTrue(info["scripted_prefix_active"])
            self.assertEqual(info["diffusion_handoff_step"], 215)
            self.assertEqual(info["expert_step"], 1)
            assert environment.residual_projector is not None
            np.testing.assert_array_equal(
                environment.residual_projector.last_applied,
                np.zeros(RESIDUAL_ACTION_DIM, dtype=np.float32),
            )
        finally:
            environment.close()

    def test_measured_centre_gate_holds_step_215_then_enables_diffusion(self):
        environment = ScrewTaskEnv(
            baseline="external_diffusion",
            random_spawn=False,
            random_socket_position=False,
            random_socket_angle=False,
            guide_mode="mocap",
            guide_scale=1.0,
            max_steps=8,
            seed=6,
        )
        try:
            expert = environment.sim.expert
            for _ in range(SCRIPTED_PREFIX_STEPS):
                expert.advance()
            self.assertEqual(expert.step_index, 215)
            self.assertFalse(expert.learned_control_active)
            observation = environment.sim.observation()
            policy_action = np.full(ACTION_DIM, 0.25, dtype=np.float32)
            assert environment.diffusion_adapter is not None

            with (
                mock.patch.object(
                    environment.sim,
                    "diffusion_handoff_ready",
                    return_value=False,
                ),
                mock.patch.object(
                    environment.diffusion_adapter,
                    "adapt",
                    side_effect=AssertionError(
                        "Diffusion must remain masked while handoff is unready"
                    ),
                ) as blocked_adapt,
            ):
                held, anchor = environment._baseline_command(
                    observation,
                    external_diffusion_action=policy_action,
                )
            blocked_adapt.assert_not_called()
            self.assertEqual(expert.step_index, 215)
            self.assertFalse(environment._diffusion_active)
            assert_command_exact(self, held, anchor)
            np.testing.assert_allclose(
                held.tip_world,
                np.array((0.0, 0.0, 0.320), dtype=np.float64),
            )

            def passthrough(
                action,
                *,
                previous_command,
                supervisor_command,
            ):
                np.testing.assert_array_equal(action, policy_action)
                self.assertIsNotNone(previous_command)
                return supervisor_command.copy()

            with (
                mock.patch.object(
                    environment.sim,
                    "diffusion_handoff_ready",
                    return_value=True,
                ),
                mock.patch.object(
                    environment.diffusion_adapter,
                    "adapt",
                    side_effect=passthrough,
                ) as active_adapt,
            ):
                baseline, safe_anchor = environment._baseline_command(
                    observation,
                    external_diffusion_action=policy_action,
                )
            active_adapt.assert_called_once()
            self.assertEqual(expert.step_index, 216)
            self.assertTrue(expert.learned_control_active)
            self.assertTrue(environment._diffusion_active)
            assert_command_exact(self, baseline, safe_anchor)
        finally:
            environment.close()

    def test_insert_schedule_waits_for_measured_thread_ready_pose(self):
        environment = ScrewTaskEnv(
            baseline="expert",
            random_spawn=False,
            guide_mode="none",
            max_steps=10,
            seed=7,
        )
        try:
            expert = environment.sim.expert
            insert_target = (
                environment.sim.socket_mouth_world
                + environment.config.thread_engagement_depth_m
                * environment.sim.socket_axis_world
            )
            held = expert.current
            held.phase = TaskPhase.INSERT
            held.engaged = False
            held.tip_world = insert_target.copy()
            expert._last_command = held.copy()

            # The reset screw is nowhere near the thread, so the expert must
            # not advance into its regrasp/engagement schedule.
            actual_depth = environment.sim.insertion_depth_m
            supervisor, anchor = environment._baseline_command(
                environment.sim.observation()
            )
            self.assertEqual(supervisor.phase, TaskPhase.INSERT)
            self.assertFalse(supervisor.engaged)
            expected_depth = (
                actual_depth
                + environment.sim.settings.insertion_command_lead_m
            )
            self.assertAlmostEqual(
                environment.sim.world_to_socket(supervisor.tip_world)[2],
                expected_depth,
                delta=1e-12,
            )
            assert_command_exact(self, anchor, supervisor)
        finally:
            environment.close()

    def test_one_environment_step_is_finite(self):
        environment = ScrewTaskEnv(
            baseline="expert",
            random_spawn=False,
            guide_mode="mocap",
            guide_scale=1.0,
            max_steps=8,
            seed=17,
        )
        try:
            observation, reset_info = environment.reset(seed=17)
            self.assertTrue(np.all(np.isfinite(observation)))
            self.assertIn("observation_schema", reset_info)
            socket_mouth_xy = np.asarray(
                reset_info["socket_mouth_xy"], dtype=np.float64
            )
            socket_opening_xy = np.asarray(
                reset_info["socket_opening_azimuth_xy"], dtype=np.float64
            )
            self.assertAlmostEqual(
                float(np.linalg.norm(socket_mouth_xy)), 0.020, places=7
            )
            np.testing.assert_allclose(
                socket_opening_xy,
                -socket_mouth_xy / np.linalg.norm(socket_mouth_xy),
                atol=1e-7,
            )
            self.assertAlmostEqual(
                float(environment.sim.socket_mouth_world[2]),
                0.330,
                places=7,
            )
            self.assertEqual(
                float(reset_info["socket_angle_deg"]), 45.0
            )
            self.assertEqual(
                reset_info["socket_yaw_equivariance_version"], 1
            )
            self.assertIn(reset_info["socket_yaw_quadrant"], range(4))
            self.assertGreaterEqual(
                reset_info["socket_yaw_residual_deg"], 0.0
            )
            self.assertLess(
                reset_info["socket_yaw_residual_deg"], 90.0
            )

            next_observation, reward, terminated, truncated, info = (
                environment.step(
                    np.zeros(RESIDUAL_ACTION_DIM, dtype=np.float32)
                )
            )
            self.assertEqual(next_observation.shape, environment.observation_space.shape)
            self.assertTrue(np.all(np.isfinite(next_observation)))
            self.assertTrue(math.isfinite(float(reward)))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertTrue(math.isfinite(float(info["depth_m"])))
            self.assertTrue(math.isfinite(float(info["twist_rad"])))
            self.assertEqual(
                info["socket_yaw_equivariance_version"], 1
            )
        finally:
            environment.close()

    def test_ik_safety_stop_returns_terminal_transition(self):
        environment = ScrewTaskEnv(
            baseline="expert",
            random_spawn=False,
            guide_mode="none",
            strict_ik=True,
            max_steps=8,
            seed=19,
        )
        try:
            environment.reset(seed=19)

            def force_safety_stop(_safe, _candidate):
                environment.sim.ik_safety_stop = True
                environment.sim.ik_safety_reason = (
                    "certified_ik_anchor_lost"
                )
                return environment.sim.current_command.copy(), 0.0

            environment.sim.project_command_to_ik = force_safety_stop
            _observation, reward, terminated, truncated, info = (
                environment.step(
                    np.zeros(RESIDUAL_ACTION_DIM, dtype=np.float32)
                )
            )
            self.assertTrue(math.isfinite(reward))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(
                info["failure_reason"],
                "certified_ik_anchor_lost",
            )
            self.assertTrue(info["ik_safety_stop"])
            self.assertFalse(info["ik_success"])
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
