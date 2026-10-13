from __future__ import annotations

import math
import unittest

import torch

from screw_config import DEFAULT_CONFIG, ScrewConfigError
from screw_control import (
    CAPTURE_GATE_STEP,
    PREINSERT_GATE_STEP,
    SCRIPTED_PREFIX_STEPS,
    THREAD_GATE_STEP,
    THREAD_REGRASP_STEP,
    TaskPhase,
    YAW_EQUIVARIANT_ALIGN_BOUNDARY_STEP,
    YAW_EQUIVARIANT_ALIGN_STEPS,
)
from s_hpc.supervisor import BatchedSupervisor
from s_hpc.task import BatchedTaskState


class PreinsertContractTest(unittest.TestCase):
    def test_default_measured_gate_and_h110_geometry(self):
        task = DEFAULT_CONFIG
        self.assertAlmostEqual(task.screw_grasp_height_from_tip_m, 0.110)
        self.assertAlmostEqual(task.scripted_grasp_plane_world_z_m, 0.420)
        self.assertAlmostEqual(task.scripted_center_tip_world_z_m, 0.310)
        self.assertGreater(
            task.screw_grasp_height_from_tip_m,
            task.screw_shaft_length_m,
        )
        self.assertLess(
            task.screw_grasp_height_from_tip_m,
            task.screw_total_height_m,
        )

        self.assertAlmostEqual(task.preinsert_standoff_m, 0.005)
        self.assertAlmostEqual(task.preinsert_lateral_tolerance_m, 0.0005)
        self.assertAlmostEqual(
            task.preinsert_orientation_tolerance_deg, 1.0
        )
        self.assertAlmostEqual(task.preinsert_depth_tolerance_m, 0.002)
        self.assertEqual(task.preinsert_stable_steps, 5)

        self.assertAlmostEqual(task.capture_depth_m, 0.030)
        self.assertAlmostEqual(task.capture_lateral_tolerance_m, 0.0005)
        self.assertAlmostEqual(
            task.capture_orientation_tolerance_deg, 1.0
        )
        worst_case_preinsert_offset = (
            task.preinsert_lateral_tolerance_m
            + task.screw_shaft_length_m
            * math.sin(
                math.radians(task.preinsert_orientation_tolerance_deg)
            )
        )
        self.assertLess(worst_case_preinsert_offset, task.radial_clearance_m)
        self.assertAlmostEqual(task.capture_depth_tolerance_m, 0.002)
        self.assertEqual(task.capture_stable_steps, 5)
        self.assertEqual(task.all_four_contact_stable_steps, 5)
        self.assertEqual(task.thread_stable_steps, 5)
        self.assertAlmostEqual(task.thread_lateral_tolerance_m, 0.0005)
        self.assertAlmostEqual(task.thread_orientation_tolerance_deg, 1.0)
        self.assertAlmostEqual(task.thread_depth_tolerance_m, 0.0015)
        self.assertAlmostEqual(task.contact_force_threshold_n, 0.10)
        self.assertAlmostEqual(
            task.contact_force_release_threshold_n, 0.04
        )
        self.assertEqual(task.contact_loss_grace_steps, 20)
        self.assertAlmostEqual(
            task.screw_shaft_length_m - task.capture_depth_m, 0.050
        )
        self.assertAlmostEqual(task.thread_engagement_depth_m, 0.070)

    def test_invalid_gate_geometry_and_stability_are_rejected(self):
        positive_fields = (
            "preinsert_standoff_m",
            "preinsert_lateral_tolerance_m",
            "preinsert_orientation_tolerance_deg",
            "preinsert_depth_tolerance_m",
            "capture_depth_m",
            "capture_lateral_tolerance_m",
            "capture_orientation_tolerance_deg",
            "capture_depth_tolerance_m",
            "capture_tip_speed_tolerance_m_s",
            "capture_lateral_speed_tolerance_m_s",
            "capture_angular_speed_tolerance_deg_s",
            "thread_lateral_tolerance_m",
            "thread_orientation_tolerance_deg",
            "contact_force_threshold_n",
            "contact_force_release_threshold_n",
        )
        for field in positive_fields:
            with self.subTest(field=field):
                with self.assertRaises(ScrewConfigError):
                    DEFAULT_CONFIG.with_overrides(**{field: 0.0})
                with self.assertRaises(ScrewConfigError):
                    DEFAULT_CONFIG.with_overrides(
                        **{field: float("nan")}
                    )
        for field in (
            "preinsert_stable_steps",
            "capture_stable_steps",
            "all_four_contact_stable_steps",
            "minimum_contact_count_after_capture",
            "contact_loss_grace_steps",
            "thread_stable_steps",
            "thread_depth_tolerance_m",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ScrewConfigError):
                    DEFAULT_CONFIG.with_overrides(**{field: 0})
                with self.assertRaises(ScrewConfigError):
                    DEFAULT_CONFIG.with_overrides(**{field: True})

        with self.assertRaises(ScrewConfigError):
            DEFAULT_CONFIG.with_overrides(
                minimum_contact_count_after_capture=1
            )
        with self.assertRaises(ScrewConfigError):
            DEFAULT_CONFIG.with_overrides(
                minimum_contact_count_after_capture=5
            )

        with self.assertRaises(ScrewConfigError):
            DEFAULT_CONFIG.with_overrides(
                contact_force_release_threshold_n=0.10
            )

        with self.assertRaises(ScrewConfigError):
            DEFAULT_CONFIG.with_overrides(capture_depth_m=0.004)
        with self.assertRaises(ScrewConfigError):
            DEFAULT_CONFIG.with_overrides(capture_depth_m=0.070)
        with self.assertRaises(ScrewConfigError):
            DEFAULT_CONFIG.with_overrides(
                capture_depth_tolerance_m=0.030
            )
        with self.assertRaises(ScrewConfigError):
            DEFAULT_CONFIG.with_overrides(
                screw_grasp_height_from_tip_m=0.080
            )
        with self.assertRaises(ScrewConfigError):
            DEFAULT_CONFIG.with_overrides(
                screw_grasp_height_from_tip_m=0.120
            )

    def test_preinsert_pose_is_collinear_parallel_and_five_mm_outside(self):
        supervisor = BatchedSupervisor(device="cpu")
        last_align_step = (
            YAW_EQUIVARIANT_ALIGN_BOUNDARY_STEP
            + YAW_EQUIVARIANT_ALIGN_STEPS
        )
        self.assertEqual(last_align_step, PREINSERT_GATE_STEP)

        yaw = torch.arange(0.0, 360.0, 45.0)
        yaw_rad = torch.deg2rad(yaw)
        mouths_xy = 0.020 * torch.stack(
            (torch.cos(yaw_rad), torch.sin(yaw_rad)), dim=-1
        )
        command = supervisor.at(
            torch.full((8,), last_align_step, dtype=torch.long),
            torch.zeros((8, 2)),
            torch.full((8,), 45.0),
            mouths_xy,
        )
        frame, inward_axis, mouth, _aligned = supervisor.socket_geometry(
            torch.full((8,), 45.0), mouths_xy
        )
        tip_socket = torch.einsum(
            "bji,bj->bi", frame, command.tip - mouth
        )
        expected_tip_socket = torch.zeros((8, 3))
        expected_tip_socket[:, 2] = -DEFAULT_CONFIG.preinsert_standoff_m
        torch.testing.assert_close(
            tip_socket, expected_tip_socket, atol=2.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            command.rotation[:, :, 2],
            -inward_axis,
            atol=2.0e-6,
            rtol=0.0,
        )
        self.assertTrue(
            bool((command.phase == int(TaskPhase.ALIGN)).all())
        )

    def test_schedule_commands_all_four_until_measured_thread_gate(self):
        supervisor = BatchedSupervisor(device="cpu")
        yaw = torch.arange(0.0, 360.0, 45.0)
        yaw_rad = torch.deg2rad(yaw)
        mouths_xy = 0.020 * torch.stack(
            (torch.cos(yaw_rad), torch.sin(yaw_rad)), dim=-1
        )
        spawn = torch.zeros((8, 2))
        angle = torch.full((8,), 45.0)

        pickup = supervisor.at(
            torch.full((8,), SCRIPTED_PREFIX_STEPS, dtype=torch.long),
            spawn,
            angle,
            mouths_xy,
        )
        capture = supervisor.at(
            torch.full((8,), CAPTURE_GATE_STEP, dtype=torch.long),
            spawn,
            angle,
            mouths_xy,
        )
        prethread_steps = torch.arange(
            CAPTURE_GATE_STEP,
            THREAD_GATE_STEP + 1,
            dtype=torch.long,
        )
        prethread = supervisor.at(
            prethread_steps,
            spawn[0].expand(prethread_steps.shape[0], 2),
            angle[0].expand(prethread_steps.shape[0]),
            mouths_xy[0].expand(prethread_steps.shape[0], 2),
        )
        before_regrasp = supervisor.at(
            torch.full((8,), THREAD_REGRASP_STEP, dtype=torch.long),
            spawn,
            angle,
            mouths_xy,
        )
        regrasp = supervisor.at(
            torch.full((8,), THREAD_REGRASP_STEP + 1, dtype=torch.long),
            spawn,
            angle,
            mouths_xy,
        )

        contact = torch.full(
            (8, 4), supervisor.contact_distance_m
        )
        torch.testing.assert_close(capture.distance, contact)
        torch.testing.assert_close(
            prethread.distance,
            torch.full_like(
                prethread.distance,
                supervisor.contact_distance_m,
            ),
        )
        torch.testing.assert_close(before_regrasp.distance, contact)
        torch.testing.assert_close(
            capture.material, pickup.material, atol=1.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            prethread.material,
            pickup.material[0].expand_as(prethread.material),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            before_regrasp.material,
            pickup.material,
            atol=1.0e-6,
            rtol=0.0,
        )
        self.assertTrue(
            bool((prethread.phase == int(TaskPhase.INSERT)).all())
        )
        self.assertTrue(
            bool((before_regrasp.phase == int(TaskPhase.ENGAGE)).all())
        )
        self.assertTrue(
            bool((regrasp.phase == int(TaskPhase.UD_CLOSE)).all())
        )

        forbidden = {
            int(TaskPhase.LR_OPEN),
            int(TaskPhase.LR_RESET),
            int(TaskPhase.LR_CLOSE),
            int(TaskPhase.UD_OPEN),
            int(TaskPhase.UD_RESET),
            int(TaskPhase.UD_CLOSE),
        }
        sampled = torch.arange(0, THREAD_GATE_STEP + 1)
        phases = supervisor.phase.index_select(0, sampled)
        self.assertTrue(
            all(int(value) not in forbidden for value in phases.tolist())
        )

    @staticmethod
    def _measured_state(worlds: int) -> BatchedTaskState:
        state = object.__new__(BatchedTaskState)
        state.task = DEFAULT_CONFIG
        state.device = torch.device("cpu")
        state.dtype = torch.float64
        state.grasp_latched = torch.ones(worlds, dtype=torch.bool)
        state.stable_grasp_support = torch.ones(
            worlds, dtype=torch.bool
        )
        state.all_four_contact = torch.ones(worlds, dtype=torch.bool)
        state.primary_pair_contact = torch.ones(
            worlds, dtype=torch.bool
        )
        state.two_finger_contact = torch.ones(
            worlds, dtype=torch.bool
        )
        state.holder_active = torch.zeros(worlds, dtype=torch.bool)
        state.shaft_captured = torch.ones(worlds, dtype=torch.bool)
        state.socket_frame = torch.eye(3, dtype=torch.float64).expand(
            worlds, 3, 3
        ).clone()
        state.socket_axis = torch.tensor(
            (0.0, 0.0, -1.0), dtype=torch.float64
        ).expand(worlds, 3).clone()
        return state

    @staticmethod
    def _rotation_about_x(angles: torch.Tensor) -> torch.Tensor:
        rotation = torch.eye(
            3, dtype=angles.dtype, device=angles.device
        ).expand(angles.shape[0], 3, 3).clone()
        cosine = torch.cos(angles)
        sine = torch.sin(angles)
        rotation[:, 1, 1] = cosine
        rotation[:, 1, 2] = -sine
        rotation[:, 2, 1] = sine
        rotation[:, 2, 2] = cosine
        return rotation

    def test_measured_preinsert_gate_boundaries_require_all_four(self):
        worlds = 5
        state = self._measured_state(worlds)
        state.preinsert_alignment_stable_steps = torch.full(
            (worlds,), DEFAULT_CONFIG.preinsert_stable_steps
        )
        tip_socket = torch.zeros((worlds, 3), dtype=torch.float64)
        tip_socket[:, 2] = -DEFAULT_CONFIG.preinsert_standoff_m
        tip_velocity = torch.zeros((worlds, 3), dtype=torch.float64)
        screw_velocity = torch.zeros((worlds, 6), dtype=torch.float64)
        angles = torch.zeros(worlds, dtype=torch.float64)

        tip_socket[1, 0] = (
            DEFAULT_CONFIG.preinsert_lateral_tolerance_m + 1.0e-6
        )
        angles[2] = math.radians(
            DEFAULT_CONFIG.preinsert_orientation_tolerance_deg + 1.0e-3
        )
        state.all_four_contact[3] = False
        state.preinsert_alignment_stable_steps[4] -= 1

        state.tip_socket = lambda: tip_socket
        state.screw_tip_velocity = lambda: tip_velocity
        state.screw_velocity = lambda: screw_velocity
        state.screw_pose = lambda: (
            torch.zeros((worlds, 3), dtype=torch.float64),
            torch.zeros((worlds, 4), dtype=torch.float64),
            self._rotation_about_x(angles),
        )

        torch.testing.assert_close(
            state.preinsert_alignment_in_tolerance(),
            torch.tensor((True, False, False, False, True)),
        )
        torch.testing.assert_close(
            state.preinsert_alignment_ready(),
            torch.tensor((True, False, False, False, False)),
        )

    def test_measured_capture_gate_boundaries_keep_head_clear(self):
        worlds = 7
        state = self._measured_state(worlds)
        state.capture_alignment_stable_steps = torch.full(
            (worlds,), DEFAULT_CONFIG.capture_stable_steps
        )
        tip_socket = torch.zeros((worlds, 3), dtype=torch.float64)
        tip_socket[:, 2] = DEFAULT_CONFIG.capture_depth_m
        tip_velocity = torch.zeros((worlds, 3), dtype=torch.float64)
        screw_velocity = torch.zeros((worlds, 6), dtype=torch.float64)
        angles = torch.zeros(worlds, dtype=torch.float64)

        # Row zero is nominal. Row one is exactly on every inclusive limit.
        tip_socket[1, 0] = (
            DEFAULT_CONFIG.capture_lateral_tolerance_m
            * (1.0 - 1.0e-12)
        )
        tip_socket[1, 2] += (
            DEFAULT_CONFIG.capture_depth_tolerance_m
            * (1.0 - 1.0e-12)
        )
        angles[1] = math.radians(
            DEFAULT_CONFIG.capture_orientation_tolerance_deg - 1.0e-9
        )
        tip_velocity[1, 0] = (
            DEFAULT_CONFIG.capture_lateral_speed_tolerance_m_s
            * (1.0 - 1.0e-12)
        )
        tip_velocity[1, 2] = math.sqrt(
            DEFAULT_CONFIG.capture_tip_speed_tolerance_m_s**2
            - tip_velocity[1, 0].item() ** 2
        ) * (1.0 - 1.0e-12)
        screw_velocity[1, 3] = math.radians(
            DEFAULT_CONFIG.capture_angular_speed_tolerance_deg_s
            - 1.0e-9
        )
        tip_socket[2, 2] += (
            DEFAULT_CONFIG.capture_depth_tolerance_m + 1.0e-6
        )
        tip_socket[3, 0] = (
            DEFAULT_CONFIG.capture_lateral_tolerance_m + 1.0e-6
        )
        angles[4] = math.radians(
            DEFAULT_CONFIG.capture_orientation_tolerance_deg + 1.0e-3
        )
        state.all_four_contact[5] = False
        state.capture_alignment_stable_steps[6] -= 1

        state.tip_socket = lambda: tip_socket
        state.insertion_depth = lambda: tip_socket[:, 2]
        state.screw_tip_velocity = lambda: tip_velocity
        state.screw_velocity = lambda: screw_velocity
        state.screw_pose = lambda: (
            torch.zeros((worlds, 3), dtype=torch.float64),
            torch.zeros((worlds, 4), dtype=torch.float64),
            self._rotation_about_x(angles),
        )

        torch.testing.assert_close(
            state.head_bottom_standoff(),
            DEFAULT_CONFIG.screw_shaft_length_m - tip_socket[:, 2],
        )
        self.assertAlmostEqual(
            float(state.head_bottom_standoff()[0]), 0.050
        )
        torch.testing.assert_close(
            state.capture_alignment_in_tolerance(),
            torch.tensor(
                (True, True, False, False, False, False, True)
            ),
        )
        torch.testing.assert_close(
            state.capture_alignment_ready(),
            torch.tensor(
                (True, True, False, False, False, False, False)
            ),
        )

    def test_measured_thread_gate_requires_capture_two_fingers_and_stability(self):
        worlds = 6
        state = self._measured_state(worlds)
        state.thread_ready_stable_steps = torch.full(
            (worlds,), DEFAULT_CONFIG.thread_stable_steps
        )
        tip_socket = torch.zeros((worlds, 3), dtype=torch.float64)
        tip_socket[:, 2] = (
            DEFAULT_CONFIG.thread_engagement_depth_m
            - DEFAULT_CONFIG.thread_depth_tolerance_m
            + 1.0e-9
        )
        tip_velocity = torch.zeros((worlds, 3), dtype=torch.float64)
        screw_velocity = torch.zeros((worlds, 6), dtype=torch.float64)
        angles = torch.zeros(worlds, dtype=torch.float64)

        state.shaft_captured[1] = False
        state.two_finger_contact[2] = False
        tip_socket[3, 2] -= 2.0e-6
        tip_velocity[4, 2] = (
            DEFAULT_CONFIG.capture_tip_speed_tolerance_m_s + 1.0e-6
        )
        state.thread_ready_stable_steps[5] -= 1

        state.tip_socket = lambda: tip_socket
        state.insertion_depth = lambda: tip_socket[:, 2]
        state.screw_tip_velocity = lambda: tip_velocity
        state.screw_velocity = lambda: screw_velocity
        state.screw_pose = lambda: (
            torch.zeros((worlds, 3), dtype=torch.float64),
            torch.zeros((worlds, 4), dtype=torch.float64),
            self._rotation_about_x(angles),
        )

        torch.testing.assert_close(
            state.thread_engagement_in_tolerance(),
            torch.tensor((True, False, False, False, False, True)),
        )
        torch.testing.assert_close(
            state.thread_engagement_ready(),
            torch.tensor((True, False, False, False, False, False)),
        )
        tip_socket[0, 2] = (
            DEFAULT_CONFIG.thread_engagement_depth_m
            + DEFAULT_CONFIG.thread_depth_tolerance_m
            + 1.0e-6
        )
        self.assertFalse(
            bool(state.thread_engagement_in_tolerance()[0])
        )


if __name__ == "__main__":
    unittest.main()
