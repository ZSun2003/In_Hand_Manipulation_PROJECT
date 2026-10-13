from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import torch

from screw_config import DEFAULT_CONFIG
from screw_control import TaskPhase
from s_hpc.supervisor import CommandBatch
from s_hpc.task import BatchedTaskState


def _identity_command(worlds: int, phase: TaskPhase) -> CommandBatch:
    return CommandBatch(
        tip=torch.zeros((worlds, 3)),
        rotation=torch.eye(3).expand(worlds, 3, 3).clone(),
        twist=torch.zeros(worlds),
        material=torch.tensor(
            ((0.5 * math.pi, -0.5 * math.pi, math.pi, 0.0),)
            * worlds
        ),
        distance=torch.full((worlds, 4), 0.041),
        phase=torch.full((worlds,), int(phase), dtype=torch.long),
        engaged=torch.zeros(worlds, dtype=torch.bool),
    )


class CaptureContactContractTest(unittest.TestCase):
    def test_primary_hold_mask_is_fixed_physical_lr(self):
        state = object.__new__(BatchedTaskState)
        state.device = torch.device("cpu")
        state.dtype = torch.float32
        calls = []

        def supplied(mouth_xy):
            calls.append(mouth_xy.clone())
            result = torch.zeros((mouth_xy.shape[0], 4), dtype=torch.bool)
            result[:, 2:] = True
            return result

        state.supervisor = SimpleNamespace(primary_hold_mask=supplied)
        mask = state._primary_hold_mask(3)
        torch.testing.assert_close(
            mask,
            torch.tensor(((False, False, True, True),) * 3),
        )
        self.assertEqual(tuple(calls[0].shape), (3, 2))

        state.supervisor = SimpleNamespace(
            primary_hold_mask=lambda mouth_xy: torch.tensor(
                ((True, True, False, False),) * mouth_xy.shape[0]
            )
        )
        with self.assertRaisesRegex(ValueError, "physical L/R"):
            state._primary_hold_mask(3)

    def test_diffusion_handoff_stays_over_each_spawn_xy(self):
        state = object.__new__(BatchedTaskState)
        state.task = DEFAULT_CONFIG
        state.backend = SimpleNamespace(worlds=2)
        state.device = torch.device("cpu")
        state.dtype = torch.float32
        state.spawn_xy = torch.tensor(((0.04, -0.03), (-0.07, 0.02)))
        target = torch.cat(
            (
                state.spawn_xy,
                torch.full(
                    (2, 1),
                    DEFAULT_CONFIG.scripted_center_tip_world_z_m,
                ),
            ),
            dim=-1,
        )
        tip = target.clone()
        state.screw_tip = lambda: tip
        state.screw_pose = lambda: (
            torch.zeros((2, 3)),
            torch.zeros((2, 4)),
            torch.eye(3).expand(2, 3, 3).clone(),
        )
        position_error, upright_error = state.diffusion_handoff_errors()
        torch.testing.assert_close(position_error, torch.zeros(2))
        torch.testing.assert_close(upright_error, torch.zeros(2))
        tip[1, 0] += 0.004
        position_error, _ = state.diffusion_handoff_errors()
        torch.testing.assert_close(
            position_error, torch.tensor((0.0, 0.004))
        )

    def test_capture_gate_rejects_each_measured_error_and_is_inclusive(self):
        state = object.__new__(BatchedTaskState)
        state.task = DEFAULT_CONFIG
        state.device = torch.device("cpu")
        state.dtype = torch.float64
        worlds = 11
        state.grasp_latched = torch.ones(worlds, dtype=torch.bool)
        state.all_four_contact = torch.ones(worlds, dtype=torch.bool)
        state.holder_active = torch.zeros(worlds, dtype=torch.bool)
        state.capture_alignment_stable_steps = torch.full(
            (worlds,), DEFAULT_CONFIG.capture_stable_steps, dtype=torch.long
        )
        state.socket_frame = torch.eye(3, dtype=torch.float64).expand(
            worlds, 3, 3
        ).clone()
        state.socket_axis = torch.tensor(
            (0.0, 0.0, -1.0), dtype=torch.float64
        ).expand(worlds, 3).clone()

        tip_socket = torch.zeros((worlds, 3), dtype=torch.float64)
        tip_socket[:, 2] = DEFAULT_CONFIG.capture_depth_m
        tip_velocity = torch.zeros((worlds, 3), dtype=torch.float64)
        screw_velocity = torch.zeros((worlds, 6), dtype=torch.float64)
        rotation = torch.eye(3, dtype=torch.float64).expand(
            worlds, 3, 3
        ).clone()

        state.grasp_latched[1] = False
        state.all_four_contact[2] = False
        state.holder_active[3] = True
        tip_socket[4, 0] = (
            DEFAULT_CONFIG.capture_lateral_tolerance_m + 1.0e-4
        )
        angle = math.radians(
            DEFAULT_CONFIG.capture_orientation_tolerance_deg + 0.1
        )
        rotation[5] = torch.tensor(
            (
                (1.0, 0.0, 0.0),
                (0.0, math.cos(angle), -math.sin(angle)),
                (0.0, math.sin(angle), math.cos(angle)),
            ),
            dtype=torch.float64,
        )
        tip_socket[6, 2] += (
            DEFAULT_CONFIG.capture_depth_tolerance_m + 1.0e-4
        )
        tip_velocity[7, 2] = (
            DEFAULT_CONFIG.capture_tip_speed_tolerance_m_s + 1.0e-4
        )
        tip_velocity[8, 0] = (
            DEFAULT_CONFIG.capture_lateral_speed_tolerance_m_s + 1.0e-4
        )
        screw_velocity[9, 3] = math.radians(
            DEFAULT_CONFIG.capture_angular_speed_tolerance_deg_s + 0.1
        )

        # The final world is exactly on every inclusive boundary.  At the
        # deepest accepted point its head bottom still has 48 mm clearance.
        epsilon = 1.0e-12
        tip_socket[10, 0] = (
            DEFAULT_CONFIG.capture_lateral_tolerance_m - epsilon
        )
        tip_socket[10, 2] += (
            DEFAULT_CONFIG.capture_depth_tolerance_m - epsilon
        )
        boundary_angle = math.radians(
            DEFAULT_CONFIG.capture_orientation_tolerance_deg - 1.0e-10
        )
        rotation[10] = torch.tensor(
            (
                (1.0, 0.0, 0.0),
                (
                    0.0,
                    math.cos(boundary_angle),
                    -math.sin(boundary_angle),
                ),
                (
                    0.0,
                    math.sin(boundary_angle),
                    math.cos(boundary_angle),
                ),
            ),
            dtype=torch.float64,
        )
        tip_velocity[10, 0] = (
            DEFAULT_CONFIG.capture_lateral_speed_tolerance_m_s - epsilon
        )
        tip_velocity[10, 2] = math.sqrt(
            (DEFAULT_CONFIG.capture_tip_speed_tolerance_m_s - epsilon) ** 2
            - (
                DEFAULT_CONFIG.capture_lateral_speed_tolerance_m_s
                - epsilon
            )
            ** 2
        )
        screw_velocity[10, 3] = math.radians(
            DEFAULT_CONFIG.capture_angular_speed_tolerance_deg_s - 1.0e-10
        )

        state.tip_socket = lambda: tip_socket
        state.insertion_depth = lambda: tip_socket[:, 2]
        state.screw_tip_velocity = lambda: tip_velocity
        state.screw_velocity = lambda: screw_velocity
        state.screw_pose = lambda: (
            torch.zeros((worlds, 3), dtype=torch.float64),
            torch.zeros((worlds, 4), dtype=torch.float64),
            rotation,
        )
        expected = torch.tensor(
            (
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            )
        )
        torch.testing.assert_close(
            state.capture_alignment_in_tolerance(), expected
        )
        torch.testing.assert_close(
            state.capture_alignment_ready(), expected
        )
        self.assertAlmostEqual(
            float(state.head_bottom_standoff()[0]),
            DEFAULT_CONFIG.screw_shaft_length_m
            - DEFAULT_CONFIG.capture_depth_m,
        )
        self.assertAlmostEqual(
            float(state.head_bottom_standoff()[10]),
            DEFAULT_CONFIG.capture_head_bottom_clearance_m,
        )

    def test_holder_release_requires_four_contacts_for_five_frames(self):
        state = self._post_state(worlds=2, phase=TaskPhase.LIFT)
        force = torch.tensor(
            ((0.10, 0.10, 0.10, 0.10), (0.10, 0.10, 0.10, 0.0))
        )
        state.fingertip_state = lambda: (
            torch.zeros((2, 4, 3)),
            torch.zeros((2, 4, 3)),
            force,
        )
        released = []
        state.backend.release_holder = lambda mask: released.append(
            mask.clone()
        )

        for _ in range(DEFAULT_CONFIG.all_four_contact_stable_steps - 1):
            state.post_physics()
            self.assertTrue(bool(state.holder_active.all()))
        state.post_physics()
        self.assertFalse(bool(state.holder_active[0]))
        self.assertTrue(bool(state.holder_active[1]))
        self.assertTrue(bool(released[-1][0]))
        self.assertFalse(bool(released[-1][1]))
        torch.testing.assert_close(
            state.pickup_material_ref[0],
            state.desired_command.material[0],
        )

    def test_capture_and_thread_counters_are_consecutive(self):
        state = self._post_state(worlds=2, phase=TaskPhase.INSERT)
        state.holder_active[:] = False
        state.grasp_latched[:] = True
        state.pickup_material_ref.copy_(
            state.desired_command.material
        )
        state.fingertip_state = lambda: (
            torch.zeros((2, 4, 3)),
            torch.zeros((2, 4, 3)),
            torch.full((2, 4), 0.2),
        )
        capture_ok = torch.tensor((True, False))
        state.capture_alignment_in_tolerance = lambda: capture_ok
        state.thread_engagement_in_tolerance = lambda: torch.zeros(
            2, dtype=torch.bool
        )

        for _ in range(DEFAULT_CONFIG.capture_stable_steps):
            state.post_physics()
        self.assertTrue(bool(state.shaft_captured[0]))
        self.assertFalse(bool(state.shaft_captured[1]))
        self.assertEqual(
            int(state.capture_alignment_stable_steps[0]),
            DEFAULT_CONFIG.capture_stable_steps,
        )
        capture_ok[:] = torch.tensor((False, True))
        state.post_physics()
        self.assertEqual(int(state.capture_alignment_stable_steps[0]), 0)
        self.assertEqual(int(state.capture_alignment_stable_steps[1]), 1)

        thread_ok = torch.tensor((True, False))
        state.thread_engagement_in_tolerance = lambda: thread_ok
        for _ in range(DEFAULT_CONFIG.thread_stable_steps):
            state.post_physics()
        self.assertTrue(bool(state.thread_engagement_ready()[0]))
        self.assertFalse(bool(state.thread_engagement_ready()[1]))
        thread_ok[:] = torch.tensor((False, True))
        state.post_physics()
        self.assertEqual(int(state.thread_ready_stable_steps[0]), 0)
        self.assertEqual(int(state.thread_ready_stable_steps[1]), 1)

    def test_contact_constraints_require_three_before_capture_and_any_two_after(self):
        state = self._post_state(worlds=8, phase=TaskPhase.INSERT)
        state.holder_active[:] = False
        state.grasp_latched[:] = True
        state.pickup_material_ref.copy_(
            state.desired_command.material
        )
        state.shaft_captured[:] = torch.tensor(
            (False, True, True, True, True, True, True, True)
        )
        # World zero loses two contacts before capture. Worlds one through
        # six exercise all six possible two-finger pairs after capture.
        # World seven has only one remaining contact after capture.
        force = torch.tensor(
            (
                (0.2, 0.0, 0.0, 0.2),
                (0.2, 0.2, 0.0, 0.0),
                (0.2, 0.0, 0.2, 0.0),
                (0.2, 0.0, 0.0, 0.2),
                (0.0, 0.2, 0.2, 0.0),
                (0.0, 0.2, 0.0, 0.2),
                (0.0, 0.0, 0.2, 0.2),
                (0.0, 0.0, 0.0, 0.2),
            )
        )
        state.fingertip_state = lambda: (
            torch.zeros((8, 4, 3)),
            torch.zeros((8, 4, 3)),
            force,
        )
        state.capture_alignment_in_tolerance = lambda: torch.zeros(
            8, dtype=torch.bool
        )
        state.thread_engagement_in_tolerance = lambda: torch.zeros(
            8, dtype=torch.bool
        )
        for _ in range(DEFAULT_CONFIG.contact_loss_grace_steps - 1):
            state.post_physics()
        torch.testing.assert_close(
            state.contact_constraint_violation,
            torch.zeros(8, dtype=torch.bool),
        )
        self.assertTrue(bool(state.contact_recovery_pending()[0]))
        self.assertTrue(bool(state.contact_recovery_pending()[7]))
        state.post_physics()
        torch.testing.assert_close(
            state.contact_constraint_violation,
            torch.tensor(
                (True, False, False, False, False, False, False, True)
            ),
        )
        self.assertTrue(bool(state.contact_violation_before_capture[0]))
        self.assertTrue(bool(state.contact_violation_before_thread[7]))
        self.assertTrue(bool(state.two_finger_contact[1:7].all()))
        self.assertFalse(bool(state.two_finger_contact[7]))
        self.assertFalse(bool(state.secondary_released.any()))

        # RESET/CLOSE remain illegal until measured thread engagement.
        state.contact_constraint_violation[1] = False
        state.desired_command.phase[1] = int(TaskPhase.UD_RESET)
        state.post_physics()
        self.assertTrue(bool(state.contact_constraint_violation[1]))

        # Even with uninterrupted force, sliding a material point before the
        # measured thread latch is a forbidden hidden regrasp.
        state.contact_constraint_violation[2] = False
        state.contact_violation_before_thread[2] = False
        state.before_thread_two_finger_loss_steps[2] = 0
        state.desired_command.phase[2] = int(TaskPhase.INSERT)
        state.desired_command.material[2, 2] += math.radians(0.1)
        force[2] = 0.2
        state.post_physics()
        self.assertTrue(bool(state.contact_constraint_violation[2]))
        self.assertTrue(bool(state.contact_violation_material_drift[2]))

    def test_thread_gate_requires_any_two_contacts_and_low_velocity(self):
        state = object.__new__(BatchedTaskState)
        state.task = DEFAULT_CONFIG
        state.socket_frame = torch.eye(3).expand(7, 3, 3).clone()
        state.socket_axis = torch.tensor((0.0, 0.0, -1.0)).expand(
            7, 3
        ).clone()
        state.shaft_captured = torch.ones(7, dtype=torch.bool)
        state.two_finger_contact = torch.ones(7, dtype=torch.bool)
        state.thread_ready_stable_steps = torch.full(
            (7,), DEFAULT_CONFIG.thread_stable_steps, dtype=torch.long
        )
        tip_socket = torch.zeros((7, 3))
        tip_socket[:, 2] = DEFAULT_CONFIG.thread_engagement_depth_m
        tip_velocity = torch.zeros((7, 3))
        screw_velocity = torch.zeros((7, 6))
        rotation = torch.eye(3).expand(7, 3, 3).clone()

        state.two_finger_contact[1] = False
        tip_socket[2, 2] = DEFAULT_CONFIG.thread_engagement_depth_m - 0.0016
        tip_socket[3, 0] = 0.0031
        tip_velocity[4, 2] = (
            DEFAULT_CONFIG.capture_tip_speed_tolerance_m_s + 1.0e-4
        )
        tip_velocity[5, 0] = (
            DEFAULT_CONFIG.capture_lateral_speed_tolerance_m_s + 1.0e-4
        )
        screw_velocity[6, 3] = math.radians(
            DEFAULT_CONFIG.capture_angular_speed_tolerance_deg_s + 0.1
        )
        state.tip_socket = lambda: tip_socket
        state.screw_tip_velocity = lambda: tip_velocity
        state.screw_velocity = lambda: screw_velocity
        state.screw_pose = lambda: (
            torch.zeros((7, 3)),
            torch.zeros((7, 4)),
            rotation,
        )
        expected = torch.tensor(
            (True, False, False, False, False, False, False)
        )
        torch.testing.assert_close(
            state.thread_engagement_in_tolerance(), expected
        )
        torch.testing.assert_close(
            state.thread_engagement_ready(), expected
        )
        tip_socket[0, 2] = (
            DEFAULT_CONFIG.thread_engagement_depth_m
            + DEFAULT_CONFIG.thread_depth_tolerance_m
            + 1.0e-4
        )
        self.assertFalse(
            bool(state.thread_engagement_in_tolerance()[0])
        )

    def test_insert_phase_latches_measured_thread_without_command_flag(self):
        state = object.__new__(BatchedTaskState)
        state.task = DEFAULT_CONFIG
        state.backend = SimpleNamespace(worlds=2)
        state.device = torch.device("cpu")
        state.dtype = torch.float32
        state.socket_axis = torch.tensor((0.0, 0.0, -1.0)).expand(
            2, 3
        ).clone()
        state.reference_marker = torch.tensor(
            (1.0, 0.0, 0.0)
        ).expand(2, 3).clone()
        state.previous_marker_mod = torch.zeros(2)
        state.marker_mod = torch.zeros(2)
        state.twist = torch.zeros(2)
        state.thread_resistance = torch.zeros(2)
        state.thread_engaged = torch.zeros(2, dtype=torch.bool)
        state.desired_command = _identity_command(2, TaskPhase.INSERT)
        self.assertFalse(bool(state.desired_command.engaged.any()))
        state.screw_pose = lambda: (
            torch.zeros((2, 3)),
            torch.zeros((2, 4)),
            torch.eye(3).expand(2, 3, 3).clone(),
        )
        state.thread_engagement_ready = lambda: torch.tensor(
            (True, False)
        )

        state.update_measured()

        torch.testing.assert_close(
            state.thread_engaged, torch.tensor((True, False))
        )

    def test_state_roundtrip_requires_capture_contact_fields(self):
        original = self._checkpoint_state(worlds=2)
        original.shaft_captured[:] = torch.tensor((True, False))
        original.two_finger_contact[:] = torch.tensor((True, False))
        original.secondary_released[:] = torch.tensor((True, False))
        original.contact_constraint_violation[:] = torch.tensor((False, True))
        original.pickup_material_ref[:] = torch.arange(8).reshape(2, 4)
        original.capture_alignment_stable_steps[:] = torch.tensor((5, 2))
        original.thread_ready_stable_steps[:] = torch.tensor((5, 1))
        original.before_thread_two_finger_loss_steps[:] = torch.tensor(
            (0, 4)
        )
        snapshot = original.state_dict()
        self.assertEqual(snapshot["version"], 5)

        restored = self._checkpoint_state(worlds=2)
        restored.load_state_dict(snapshot)
        for name in (
            "pickup_material_ref",
            "primary_hold_mask",
            "all_four_contact",
            "primary_pair_contact",
            "two_finger_contact",
            "all_four_contact_stable_steps",
            "primary_pair_contact_stable_steps",
            "capture_alignment_stable_steps",
            "thread_ready_stable_steps",
            "shaft_captured",
            "secondary_released",
            "contact_constraint_violation",
            "contact_violation_before_capture",
            "contact_violation_before_thread",
            "contact_violation_premature_regrasp",
            "contact_violation_material_drift",
            "before_capture_contact_loss_steps",
            "before_thread_two_finger_loss_steps",
        ):
            torch.testing.assert_close(
                getattr(restored, name), getattr(original, name)
            )

        missing = original.state_dict()
        del missing["tensors"]["shaft_captured"]
        with self.assertRaisesRegex(ValueError, "shaft_captured"):
            restored.load_state_dict(missing)

    @staticmethod
    def _post_state(worlds: int, phase: TaskPhase):
        state = object.__new__(BatchedTaskState)
        state.task = DEFAULT_CONFIG
        state.desired_command = _identity_command(worlds, phase)
        state.holder_active = torch.ones(worlds, dtype=torch.bool)
        state.grasp_latched = torch.zeros(worlds, dtype=torch.bool)
        state.stable_grasp_support = torch.zeros(worlds, dtype=torch.bool)
        state.backend = SimpleNamespace(release_holder=lambda _mask: None)
        state.preinsert_alignment_stable_steps = torch.zeros(
            worlds, dtype=torch.long
        )
        state.preinsert_alignment_in_tolerance = lambda: torch.zeros(
            worlds, dtype=torch.bool
        )
        state.capture_alignment_in_tolerance = lambda: torch.zeros(
            worlds, dtype=torch.bool
        )
        state.thread_engagement_in_tolerance = lambda: torch.zeros(
            worlds, dtype=torch.bool
        )
        state.thread_engaged = torch.zeros(worlds, dtype=torch.bool)
        state.update_measured = lambda: None
        state._ensure_contact_contract_state(
            torch.zeros((worlds, 4), dtype=torch.bool)
        )
        return state

    @staticmethod
    def _checkpoint_state(worlds: int):
        state = object.__new__(BatchedTaskState)
        state.backend = SimpleNamespace(worlds=worlds)
        state.supervisor = SimpleNamespace(fixed_socket_angle_deg=45.0)
        state.device = torch.device("cpu")
        state.dtype = torch.float32
        matrix = torch.eye(3).expand(worlds, 3, 3).clone()
        vectors = {
            "spawn_xy": torch.zeros((worlds, 2)),
            "socket_angle_deg": torch.full((worlds,), 45.0),
            "socket_frame": matrix.clone(),
            "socket_axis": torch.zeros((worlds, 3)),
            "socket_mouth": torch.zeros((worlds, 3)),
            "aligned_rotation": matrix.clone(),
            "reference_marker": torch.zeros((worlds, 3)),
            "step_count": torch.zeros(worlds, dtype=torch.long),
            "elapsed_steps": torch.zeros(worlds, dtype=torch.long),
            "episode_return": torch.zeros(worlds),
            "episode_id": torch.zeros(worlds, dtype=torch.long),
            "thread_engaged": torch.zeros(worlds, dtype=torch.bool),
            "holder_active": torch.ones(worlds, dtype=torch.bool),
            "grasp_latched": torch.zeros(worlds, dtype=torch.bool),
            "stable_grasp_support": torch.zeros(worlds, dtype=torch.bool),
            "pickup_material_ref": torch.zeros((worlds, 4)),
            "primary_hold_mask": torch.tensor(
                ((False, False, True, True),) * worlds
            ),
            "all_four_contact": torch.zeros(worlds, dtype=torch.bool),
            "primary_pair_contact": torch.zeros(worlds, dtype=torch.bool),
            "two_finger_contact": torch.zeros(worlds, dtype=torch.bool),
            "all_four_contact_stable_steps": torch.zeros(
                worlds, dtype=torch.long
            ),
            "primary_pair_contact_stable_steps": torch.zeros(
                worlds, dtype=torch.long
            ),
            "capture_alignment_stable_steps": torch.zeros(
                worlds, dtype=torch.long
            ),
            "thread_ready_stable_steps": torch.zeros(
                worlds, dtype=torch.long
            ),
            "shaft_captured": torch.zeros(worlds, dtype=torch.bool),
            "secondary_released": torch.zeros(worlds, dtype=torch.bool),
            "contact_constraint_violation": torch.zeros(
                worlds, dtype=torch.bool
            ),
            "contact_violation_before_capture": torch.zeros(
                worlds, dtype=torch.bool
            ),
            "contact_violation_before_thread": torch.zeros(
                worlds, dtype=torch.bool
            ),
            "contact_violation_premature_regrasp": torch.zeros(
                worlds, dtype=torch.bool
            ),
            "contact_violation_material_drift": torch.zeros(
                worlds, dtype=torch.bool
            ),
            "before_capture_contact_loss_steps": torch.zeros(
                worlds, dtype=torch.long
            ),
            "before_thread_two_finger_loss_steps": torch.zeros(
                worlds, dtype=torch.long
            ),
            "previous_marker_mod": torch.zeros(worlds),
            "marker_mod": torch.zeros(worlds),
            "twist": torch.zeros(worlds),
            "thread_resistance": torch.zeros(worlds),
            "axial_preload": torch.zeros(worlds),
            "previous_depth": torch.zeros(worlds),
            "previous_twist": torch.zeros(worlds),
            "ik_failures": torch.zeros(worlds, dtype=torch.long),
            "ik_anchor_fallbacks": torch.zeros(worlds, dtype=torch.long),
            "ik_safety_stops": torch.zeros(worlds, dtype=torch.long),
            "diffusion_handoff_hold_steps": torch.zeros(
                worlds, dtype=torch.long
            ),
            "preinsert_alignment_stable_steps": torch.zeros(
                worlds, dtype=torch.long
            ),
            "preinsert_feedback_offset_socket": torch.zeros((worlds, 3)),
        }
        for name, value in vectors.items():
            setattr(state, name, value)
        state.desired_command = _identity_command(worlds, TaskPhase.ALIGN)
        return state


if __name__ == "__main__":
    unittest.main()
