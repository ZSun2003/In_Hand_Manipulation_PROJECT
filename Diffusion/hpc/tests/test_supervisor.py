from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from screw_control import (
    BASE_FINGER_ANGLES_RAD,
    CAPTURE_GATE_STEP,
    CONTINUOUS_GRASP_STRATEGY_VERSION,
    PICKUP_RELEASE_GATE_STABLE_STEPS,
    PICKUP_RELEASE_GATE_STEP,
    PREINSERT_GATE_STEP,
    RELEASED_TIP_CENTER_DISTANCE_M,
    SCRIPTED_PREFIX_STEPS,
    TaskPhase,
    THREAD_GATE_STEP,
    THREAD_REGRASP_STEP,
)
from s_hpc.config import CurriculumConfig
from s_hpc.env import MJWarpScrewEnv, _blend_command, _select_command
from s_hpc.task import BatchedTaskState
from s_hpc.supervisor import (
    BatchedSupervisor,
    CommandBatch,
    TaskSpaceSafety,
    quaternion_to_matrix,
    residual_action_mask,
    rotation_exp,
    rotation_log,
)


class SupervisorTest(unittest.TestCase):
    def setUp(self):
        self.supervisor = BatchedSupervisor(device="cpu")

    def test_affine_spawn_and_action_contract(self):
        spawn = torch.tensor(((-0.075, -0.04), (0.0, 0.09)))
        step = torch.tensor((0, 309))
        command = self.supervisor.at(step, spawn)
        self.assertEqual(command.tip.shape, (2, 3))
        self.assertEqual(command.fingertip_targets().shape, (2, 4, 3))
        self.assertEqual(self.supervisor.action_at(step, spawn).shape, (2, 14))
        explicit_fixed = self.supervisor.at(
            step,
            spawn,
            self.supervisor.fixed_angle_batch(2),
        )
        torch.testing.assert_close(command.tip, explicit_fixed.tip)
        torch.testing.assert_close(
            command.rotation, explicit_fixed.rotation
        )

    def test_socket_opening_always_faces_fixed_centre(self):
        angles = torch.full((3,), 45.0)
        frame, axis, mouth, _aligned = self.supervisor.socket_geometry(angles)
        center, _quaternion = self.supervisor.socket_mocap_pose(angles)
        expected_mouth = torch.tensor(
            self.supervisor.config.socket_mouth_position_m
        ).expand(3, 3)
        torch.testing.assert_close(mouth, expected_mouth)
        torch.testing.assert_close(
            axis,
            torch.tensor(
                (
                    (math.sqrt(0.5), 0.0, -math.sqrt(0.5)),
                    (math.sqrt(0.5), 0.0, -math.sqrt(0.5)),
                    (math.sqrt(0.5), 0.0, -math.sqrt(0.5)),
                )
            ),
            atol=1.0e-6,
            rtol=0.0,
        )
        # The opening normal is exactly body-centre -> mouth centre.
        opening_normal = -axis
        body_to_mouth = mouth - center
        torch.testing.assert_close(
            body_to_mouth,
            0.5
            * self.supervisor.config.socket_outer_length_m
            * opening_normal,
            atol=1.0e-7,
            rtol=0.0,
        )
        self.assertTrue(frame.is_contiguous())
        self.assertTrue(axis.is_contiguous())
        self.assertTrue(mouth.is_contiguous())
        self.assertTrue(_aligned.is_contiguous())
        identity = frame.transpose(-1, -2) @ frame
        torch.testing.assert_close(
            identity,
            torch.eye(3).expand_as(identity),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_random_socket_orbit_and_yaw_geometry(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)
        mouths = self.supervisor.sample_socket_mouth(
            4096, generator=generator
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(mouths, dim=-1),
            torch.full((4096,), 0.020),
            atol=2.0e-7,
            rtol=0.0,
        )

        mouths = torch.tensor(
            ((0.020, 0.0), (0.0, 0.020), (-0.020, 0.0))
        )
        angles = torch.full((3,), 45.0)
        frame, axis, mouth, _aligned = self.supervisor.socket_geometry(
            angles, mouths
        )
        centre, quaternion = self.supervisor.socket_mocap_pose(
            angles, mouths
        )
        torch.testing.assert_close(
            quaternion_to_matrix(quaternion), frame, atol=2.0e-6, rtol=0.0
        )
        radial = mouths / 0.020
        horizontal = axis[:, :2]
        expected_horizontal = (
            torch.cos(torch.deg2rad(angles))[:, None] * radial
        )
        torch.testing.assert_close(
            horizontal, expected_horizontal, atol=1.0e-6, rtol=0.0
        )
        opening = mouth - centre
        self.assertTrue(bool(((opening[:, :2] * mouths).sum(-1) <= 1.0e-8).all()))

    def test_direct_alignment_uses_actual_yaw_frame_and_fixed_material(self):
        rotation = torch.tensor(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )
        mouth = torch.tensor(((0.020, 0.0), (0.0, 0.020)))
        spawn0 = torch.tensor((-0.070, 0.015))
        spawn = torch.stack((spawn0, rotation[:2, :2] @ spawn0))
        step = torch.full((2,), PREINSERT_GATE_STEP)
        angle = torch.tensor((45.0, 45.0))
        command = self.supervisor.at(step, spawn, angle, mouth)
        torch.testing.assert_close(
            command.tip[1], rotation @ command.tip[0], atol=2.0e-6, rtol=0.0
        )
        _frame, axis, _mouth, aligned = self.supervisor.socket_geometry(
            angle, mouth
        )
        torch.testing.assert_close(
            command.rotation,
            aligned,
            atol=2.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            command.rotation[:, :, 2],
            -axis,
            atol=2.0e-6,
            rtol=0.0,
        )
        expected_material = torch.tensor(
            tuple(BASE_FINGER_ANGLES_RAD[finger] for finger in ("U", "D", "L", "R"))
        ).expand(2, 4)
        torch.testing.assert_close(command.material, expected_material)

    def test_physical_lr_mask_is_diagnostic_while_all_four_stay_closed(self):
        self.assertAlmostEqual(
            self.supervisor.released_distance_m,
            RELEASED_TIP_CENTER_DISTANCE_M,
        )
        mouths = torch.tensor(
            (
                (0.020, 0.0),
                (0.0, 0.020),
                (-0.020, 0.0),
                (0.0, -0.020),
            )
        )
        _yaw, quadrant, delta, *_rest = self.supervisor.yaw_strategy(
            mouths
        )
        self.assertEqual(
            torch.remainder(quadrant, 4).tolist(), [0, 1, 2, 3]
        )
        torch.testing.assert_close(
            delta,
            torch.zeros(4),
            atol=3.0e-7,
            rtol=0.0,
        )
        mask = self.supervisor.primary_hold_mask(mouths)
        torch.testing.assert_close(
            mask,
            torch.tensor((False, False, True, True)).expand(4, 4),
        )
        capture = self.supervisor.at(
            torch.full((4,), CAPTURE_GATE_STEP),
            torch.zeros((4, 2)),
            torch.full((4,), 45.0),
            mouths,
        )
        torch.testing.assert_close(
            capture.distance,
            torch.full((4, 4), self.supervisor.contact_distance_m),
        )
        supported = self.supervisor.at(
            torch.full((4,), CAPTURE_GATE_STEP + 20),
            torch.zeros((4, 2)),
            torch.full((4,), 45.0),
            mouths,
        )
        torch.testing.assert_close(
            supported.distance,
            torch.full((4, 4), self.supervisor.contact_distance_m),
            atol=5.0e-6,
            rtol=0.0,
        )
        self.assertEqual(
            supported.phase.tolist(), [int(TaskPhase.INSERT)] * 4
        )
        expected_material = torch.tensor(
            tuple(BASE_FINGER_ANGLES_RAD[finger] for finger in ("U", "D", "L", "R"))
        ).expand(4, 4)
        torch.testing.assert_close(
            supported.material, expected_material
        )

    def test_direct_align_slerp_has_exact_boundaries_without_extra_roll(self):
        yaw_deg = 89.9
        yaw_rad = math.radians(yaw_deg)
        mouth = 0.020 * torch.tensor(
            ((math.cos(yaw_rad), math.sin(yaw_rad)),)
        )
        steps = torch.tensor(
            (
                SCRIPTED_PREFIX_STEPS,
                SCRIPTED_PREFIX_STEPS + 1,
                SCRIPTED_PREFIX_STEPS + 175,
                PREINSERT_GATE_STEP,
                PREINSERT_GATE_STEP + 1,
            )
        )
        repeated_mouth = mouth.expand(steps.shape[0], 2)
        spawn = torch.tensor((-0.070, 0.015)).expand(steps.shape[0], 2)
        command = self.supervisor.at(
            steps,
            spawn,
            torch.full((steps.shape[0],), 45.0),
            repeated_mouth,
        )
        _frame, axis, _mouth, aligned = self.supervisor.socket_geometry(
            torch.full((steps.shape[0],), 45.0), repeated_mouth
        )
        torch.testing.assert_close(
            command.rotation[0], torch.eye(3), atol=2.0e-6, rtol=0.0
        )
        self.assertFalse(
            torch.allclose(command.rotation[2], torch.eye(3))
        )
        torch.testing.assert_close(
            command.rotation[-2:],
            aligned[-2:],
            atol=2.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            command.rotation[-1, :, 2],
            -axis[-1],
            atol=2.0e-6,
            rtol=0.0,
        )
        self.assertEqual(
            self.supervisor.yaw_strategy_version,
            CONTINUOUS_GRASP_STRATEGY_VERSION,
        )

    def test_continuous_grasp_gate_indices_and_contact_schedule(self):
        self.assertEqual(self.supervisor.preinsert_gate_step, 525)
        self.assertEqual(self.supervisor.capture_gate_step, 635)
        self.assertEqual(self.supervisor.thread_gate_step, 775)
        self.assertEqual(self.supervisor.thread_regrasp_step, 776)
        grasp_heights = self.supervisor.grasp_height_at(
            torch.tensor(
                (
                    SCRIPTED_PREFIX_STEPS,
                    SCRIPTED_PREFIX_STEPS + 30,
                    SCRIPTED_PREFIX_STEPS + 60,
                )
            )
        )
        self.assertAlmostEqual(float(grasp_heights[0]), 0.110)
        self.assertAlmostEqual(float(grasp_heights[1]), 0.110)
        self.assertAlmostEqual(float(grasp_heights[2]), 0.110)
        steps = torch.tensor(
            (
                PREINSERT_GATE_STEP,
                CAPTURE_GATE_STEP,
                CAPTURE_GATE_STEP + 20,
                THREAD_GATE_STEP,
                THREAD_GATE_STEP + 1,
                THREAD_REGRASP_STEP,
                THREAD_REGRASP_STEP + 1,
            )
        )
        mouth_xy = torch.tensor((0.020, 0.0)).expand(
            steps.shape[0], 2
        )
        command = self.supervisor.at(
            steps,
            torch.tensor((-0.075, -0.040)).expand(
                steps.shape[0], 2
            ),
            torch.full((steps.shape[0],), 45.0),
            mouth_xy,
        )
        frame, axis, mouth, _aligned = self.supervisor.socket_geometry(
            torch.full((steps.shape[0],), 45.0), mouth_xy
        )
        depth = torch.einsum(
            "bi,bi->b", command.tip - mouth, axis
        )
        torch.testing.assert_close(
            depth,
            torch.tensor((-0.005, 0.030, 0.030, 0.070, 0.070, 0.070, 0.070)),
            atol=2.0e-6,
            rtol=0.0,
        )
        self.assertEqual(
            command.phase.tolist(),
            [
                int(TaskPhase.ALIGN),
                int(TaskPhase.INSERT),
                int(TaskPhase.INSERT),
                int(TaskPhase.INSERT),
                int(TaskPhase.ENGAGE),
                int(TaskPhase.ENGAGE),
                int(TaskPhase.UD_CLOSE),
            ],
        )
        self.assertEqual(
            command.engaged.tolist(),
            [False, False, False, False, True, True, True],
        )
        torch.testing.assert_close(
            command.distance,
            torch.full_like(
                command.distance, self.supervisor.contact_distance_m
            ),
            atol=5.0e-6,
            rtol=0.0,
        )
        expected_material = torch.tensor(
            tuple(BASE_FINGER_ANGLES_RAD[finger] for finger in ("U", "D", "L", "R"))
        ).expand(steps.shape[0], 4)
        torch.testing.assert_close(command.material, expected_material)
        self.assertAlmostEqual(
            self.supervisor.config.screw_shaft_length_m
            - self.supervisor.config.capture_depth_m
            - self.supervisor.config.capture_depth_tolerance_m,
            self.supervisor.config.capture_head_bottom_clearance_m,
            places=9,
        )

    def test_curriculum_changes_only_confirmed_task_difficulty(self):
        start = self.supervisor.curriculum_parameters(0.0)
        end = self.supervisor.curriculum_parameters(1.0)
        self.assertEqual(start, (0.1, 45.0))
        self.assertEqual(end, start)
        self.assertEqual(self.supervisor.angle_grid_deg.tolist(), [45.0])
        self.assertEqual(self.supervisor.angle_min_deg, 45.0)
        self.assertEqual(self.supervisor.angle_max_deg, 45.0)
        self.assertFalse(hasattr(self.supervisor, "sample_angle"))
        fixed = self.supervisor.fixed_angle_batch(4096)
        torch.testing.assert_close(fixed, torch.full((4096,), 45.0))
        # Without a curriculum, the full distribution is active immediately.
        self.assertAlmostEqual(
            self.supervisor.config.rotation_resistance_min_nm, 0.030
        )
        self.assertAlmostEqual(
            self.supervisor.config.rotation_resistance_max_nm, 0.120
        )
        self.assertAlmostEqual(self.supervisor.config.axial_preload_n, 2.0)

    def test_task_reset_rejects_non_fixed_socket_angle(self):
        state = object.__new__(BatchedTaskState)
        state.device = torch.device("cpu")
        state.dtype = torch.float32
        state.supervisor = SimpleNamespace(fixed_socket_angle_deg=45.0)
        with self.assertRaisesRegex(ValueError, "fixed 45 degree"):
            state.reset(
                torch.tensor((True, False)),
                spawn_xy=None,
                socket_angle_deg=torch.tensor((90.0, 45.0)),
                socket_mouth_xy=None,
                command=None,
            )

    def test_spawn_sampler_stays_in_opening_facing_half_disk(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260726)
        spawn = self.supervisor.sample_spawn(
            4096, radius_m=0.100, generator=generator
        )
        self.assertTrue(bool((spawn[:, 0] <= 1.0e-7).all()))
        self.assertTrue(
            bool(
                (
                    torch.linalg.vector_norm(spawn, dim=1)
                    <= 0.100 + 1.0e-7
                ).all()
            )
        )

        mouths = self.supervisor.sample_socket_mouth(
            4096, generator=generator
        )
        angles = self.supervisor.fixed_angle_batch(4096)
        spawn = self.supervisor.sample_spawn(
            4096,
            socket_mouth_xy=mouths,
            socket_angle_deg=angles,
            radius_m=0.100,
            generator=generator,
        )
        opening = -mouths / 0.020
        self.assertTrue(bool(((spawn * opening).sum(-1) >= -1.0e-7).all()))

    def test_start_curriculum_spawn_is_collision_free_at_18000_worlds(self):
        curriculum = CurriculumConfig()
        supervisor = BatchedSupervisor(
            device="cpu", curriculum=curriculum
        )
        self.assertEqual(
            supervisor.curriculum_parameters(0.0), (0.060, 45.0)
        )
        count = 18_000
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260730)
        mouths = supervisor.sample_socket_mouth(
            count, generator=generator
        )
        angles = supervisor.fixed_angle_batch(count)
        radius_limit = curriculum.start_spawn_radius_m
        spawn = supervisor.sample_spawn(
            count,
            socket_mouth_xy=mouths,
            socket_angle_deg=angles,
            radius_m=radius_limit,
            generator=generator,
        )

        self.assertTrue(bool(torch.isfinite(spawn).all()))
        self.assertTrue(
            bool(
                (
                    torch.linalg.vector_norm(spawn, dim=-1)
                    <= radius_limit + 1.0e-7
                ).all()
            )
        )
        inward = mouths / torch.linalg.vector_norm(
            mouths, dim=-1, keepdim=True
        )
        opening = -inward
        self.assertTrue(
            bool(((spawn * opening).sum(dim=-1) >= -1.0e-7).all())
        )
        tangent = torch.stack(
            (-inward[:, 1], inward[:, 0]), dim=-1
        )
        elevation = torch.deg2rad(angles)
        cosine = torch.cos(elevation).clamp_min(0.0)
        sine = torch.sin(elevation).clamp_min(0.0)
        centre = (
            mouths
            + 0.5
            * supervisor.config.socket_outer_length_m
            * cosine[:, None]
            * inward
        )
        relative = spawn - centre
        margin = (
            supervisor.config.screw_head_radius_m
            + supervisor.config.spawn_clearance_m
        )
        radial_extent = (
            0.5 * supervisor.config.socket_outer_length_m * cosine
            + 0.5 * supervisor.config.socket_outer_width_m * sine
            + margin
        )
        tangent_extent = (
            0.5 * supervisor.config.socket_outer_height_m + margin
        )
        invalid = (
            ((relative * inward).sum(dim=-1).abs() <= radial_extent)
            & (
                (relative * tangent).sum(dim=-1).abs()
                <= tangent_extent
            )
        )
        self.assertEqual(int(invalid.sum()), 0)

        # At 60 mm the tangent escape boundary remains outside the disk, so
        # the radial circular segment gives the exact per-draw acceptance.
        radial_escape = (
            0.5
            * supervisor.config.socket_outer_width_m
            * math.sin(math.radians(45.0))
            + margin
            - supervisor.config.socket_mouth_orbit_radius_m
        )
        self.assertLess(radius_limit, tangent_extent)
        ratio = radial_escape / radius_limit
        acceptance = (2.0 / math.pi) * (
            math.acos(ratio)
            - ratio * math.sqrt(1.0 - ratio * ratio)
        )
        exhaustion_union_bound = (
            count * (1.0 - acceptance) ** 128
        )
        self.assertGreater(acceptance, 0.27)
        self.assertLess(exhaustion_union_bound, 1.0e-12)

    def test_full_workspace_sampler_covers_both_sides_and_rejects_socket_obb(self):
        curriculum = CurriculumConfig(
            restrict_screw_spawn_to_opening_half_disk=False
        )
        supervisor = BatchedSupervisor(
            device="cpu", curriculum=curriculum
        )
        count = 8192
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260815)
        mouths = supervisor.sample_socket_mouth(
            count, generator=generator
        )
        angles = supervisor.fixed_angle_batch(count)
        spawn = supervisor.sample_spawn(
            count,
            socket_mouth_xy=mouths,
            socket_angle_deg=angles,
            radius_m=0.100,
            generator=generator,
        )

        self.assertTrue(bool(torch.isfinite(spawn).all()))
        self.assertTrue(
            bool(
                (
                    torch.linalg.vector_norm(spawn, dim=-1)
                    <= 0.100 + 1.0e-7
                ).all()
            )
        )
        inward = mouths / torch.linalg.vector_norm(
            mouths, dim=-1, keepdim=True
        )
        opening = -inward
        opening_coordinate = (spawn * opening).sum(dim=-1)
        self.assertTrue(bool((opening_coordinate > 1.0e-4).any()))
        self.assertTrue(bool((opening_coordinate < -1.0e-4).any()))

        tangent = torch.stack(
            (-inward[:, 1], inward[:, 0]), dim=-1
        )
        elevation = torch.deg2rad(angles)
        cosine = torch.cos(elevation).clamp_min(0.0)
        sine = torch.sin(elevation).clamp_min(0.0)
        centre = (
            mouths
            + 0.5
            * supervisor.config.socket_outer_length_m
            * cosine[:, None]
            * inward
        )
        relative = spawn - centre
        margin = (
            supervisor.config.screw_head_radius_m
            + supervisor.config.spawn_clearance_m
        )
        radial_extent = (
            0.5 * supervisor.config.socket_outer_length_m * cosine
            + 0.5 * supervisor.config.socket_outer_width_m * sine
            + margin
        )
        tangent_extent = (
            0.5 * supervisor.config.socket_outer_height_m + margin
        )
        invalid = (
            ((relative * inward).sum(dim=-1).abs() <= radial_extent)
            & (
                (relative * tangent).sum(dim=-1).abs()
                <= tangent_extent
            )
        )
        self.assertEqual(int(invalid.sum()), 0)

    def test_full_workspace_start_sampler_is_bounded_at_18000_worlds(self):
        curriculum = CurriculumConfig(
            restrict_screw_spawn_to_opening_half_disk=False
        )
        supervisor = BatchedSupervisor(
            device="cpu", curriculum=curriculum
        )
        count = 18_000
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260815)
        mouths = supervisor.sample_socket_mouth(
            count, generator=generator
        )
        angles = supervisor.fixed_angle_batch(count)
        radius_limit = curriculum.start_spawn_radius_m
        spawn = supervisor.sample_spawn(
            count,
            socket_mouth_xy=mouths,
            socket_angle_deg=angles,
            radius_m=radius_limit,
            generator=generator,
        )
        self.assertTrue(bool(torch.isfinite(spawn).all()))
        self.assertTrue(
            bool(
                (
                    torch.linalg.vector_norm(spawn, dim=-1)
                    <= radius_limit + 1.0e-7
                ).all()
            )
        )

        margin = (
            supervisor.config.screw_head_radius_m
            + supervisor.config.spawn_clearance_m
        )
        radial_escape = (
            0.5
            * supervisor.config.socket_outer_width_m
            * math.sin(math.radians(45.0))
            + margin
            - supervisor.config.socket_mouth_orbit_radius_m
        )
        ratio = radial_escape / radius_limit
        acceptance = (
            math.acos(ratio)
            - ratio * math.sqrt(1.0 - ratio * ratio)
        ) / math.pi
        exhaustion_union_bound = count * (1.0 - acceptance) ** 256
        self.assertGreater(acceptance, 0.13)
        self.assertLess(exhaustion_union_bound, 1.0e-10)

    def test_spawn_sampler_fails_fast_for_zero_support_25mm_radius(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260730)
        before = generator.get_state().clone()
        with self.assertRaisesRegex(
            ValueError,
            r"no positive-area collision-free support.*radius > 0\.03628",
        ):
            self.supervisor.sample_spawn(
                64, radius_m=0.025, generator=generator
            )
        torch.testing.assert_close(generator.get_state(), before)

    def test_scripted_pickup_lifts_at_spawn_then_aligns_directly(self):
        self.assertEqual(SCRIPTED_PREFIX_STEPS, 175)
        self.assertEqual(self.supervisor.diffusion_handoff_step, 175)
        spawn = torch.tensor((-0.075, -0.040)).expand(5, 2)
        mouth = torch.tensor((0.020, 0.0)).expand(5, 2)
        command = self.supervisor.at(
            torch.tensor((65, 115, 125, 175, 200)),
            spawn,
            torch.full((5,), 45.0),
            mouth,
        )
        torch.testing.assert_close(
            command.distance[0], torch.full((4,), 0.053)
        )
        torch.testing.assert_close(
            command.distance[1:4],
            torch.full((3, 4), self.supervisor.contact_distance_m),
            atol=2.0e-7,
            rtol=0.0,
        )
        torch.testing.assert_close(
            command.distance[4],
            torch.full((4,), self.supervisor.contact_distance_m),
            atol=2.0e-7,
            rtol=0.0,
        )
        # The explicit close hold ends at 125.  The lift endpoint keeps the
        # original spawn XY and puts the H=110 mm grasp plane at z=.420.
        torch.testing.assert_close(
            command.tip[2],
            torch.tensor((-0.075, -0.040, 0.260)),
            atol=2.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            command.tip[3],
            torch.tensor((-0.075, -0.040, 0.310)),
            atol=2.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            command.rotation[3],
            torch.eye(3),
            atol=2.0e-6,
            rtol=0.0,
        )
        self.assertEqual(int(command.phase[3]), int(TaskPhase.LIFT))
        self.assertEqual(int(command.phase[4]), int(TaskPhase.ALIGN))
        self.assertGreater(
            float(
                torch.linalg.vector_norm(
                    command.tip[4] - command.tip[3]
                )
            ),
            0.0,
        )
        # Step 175 is the measured feedback boundary; only step 176 belongs
        # to the Diffusion/residual-controlled suffix.
        self.assertFalse(bool((torch.tensor(175) > SCRIPTED_PREFIX_STEPS)))
        self.assertTrue(bool((torch.tensor(176) > SCRIPTED_PREFIX_STEPS)))

    def test_measured_handoff_gate_holds_only_unready_worlds(self):
        env = object.__new__(MJWarpScrewEnv)
        phase = torch.full((900,), int(TaskPhase.LIFT), dtype=torch.long)
        env.supervisor = SimpleNamespace(
            length=len(phase),
            phase=phase,
            diffusion_handoff_step=SCRIPTED_PREFIX_STEPS,
            preinsert_gate_step=PREINSERT_GATE_STEP,
            capture_gate_step=CAPTURE_GATE_STEP,
            thread_gate_step=THREAD_GATE_STEP,
            thread_regrasp_step=THREAD_REGRASP_STEP,
        )
        env.state = SimpleNamespace(
            step_count=torch.tensor((174, 175, 175), dtype=torch.long),
            holder_active=torch.zeros(3, dtype=torch.bool),
            grasp_latched=torch.ones(3, dtype=torch.bool),
            all_four_contact=torch.ones(3, dtype=torch.bool),
            all_four_contact_stable_steps=torch.full(
                (3,), PICKUP_RELEASE_GATE_STABLE_STEPS
            ),
            diffusion_handoff_ready=lambda: torch.tensor(
                (False, False, True)
            ),
            preinsert_alignment_ready=lambda: torch.ones(
                3, dtype=torch.bool
            ),
            shaft_captured=torch.ones(3, dtype=torch.bool),
            capture_alignment_ready=lambda: torch.ones(
                3, dtype=torch.bool
            ),
            thread_engagement_ready=lambda: torch.ones(
                3, dtype=torch.bool
            ),
            thread_engaged=torch.ones(3, dtype=torch.bool),
            contact_recovery_pending=lambda: torch.zeros(
                3, dtype=torch.bool
            ),
        )
        (
            next_step,
            pickup_release_hold,
            thread_hold,
            handoff_hold,
            preinsert_hold,
            capture_hold,
            regrasp_hold,
            contact_recovery_hold,
        ) = env._schedule_next_step()
        torch.testing.assert_close(next_step, torch.tensor((175, 175, 176)))
        torch.testing.assert_close(
            pickup_release_hold,
            torch.tensor((False, False, False)),
        )
        torch.testing.assert_close(
            thread_hold, torch.tensor((False, False, False))
        )
        torch.testing.assert_close(
            handoff_hold, torch.tensor((False, True, False))
        )
        torch.testing.assert_close(
            preinsert_hold, torch.tensor((False, False, False))
        )
        torch.testing.assert_close(
            capture_hold, torch.tensor((False, False, False))
        )
        torch.testing.assert_close(
            regrasp_hold, torch.tensor((False, False, False))
        )
        torch.testing.assert_close(
            contact_recovery_hold,
            torch.tensor((False, False, False)),
        )

    def test_pickup_release_gate_enters_lift_then_waits_per_world(self):
        env = object.__new__(MJWarpScrewEnv)
        phase = torch.full((900,), int(TaskPhase.LIFT), dtype=torch.long)
        env.supervisor = SimpleNamespace(
            length=len(phase),
            phase=phase,
            diffusion_handoff_step=SCRIPTED_PREFIX_STEPS,
            preinsert_gate_step=PREINSERT_GATE_STEP,
            capture_gate_step=CAPTURE_GATE_STEP,
            thread_gate_step=THREAD_GATE_STEP,
            thread_regrasp_step=THREAD_REGRASP_STEP,
        )
        env.state = SimpleNamespace(
            step_count=torch.tensor(
                (
                    PICKUP_RELEASE_GATE_STEP - 1,
                    PICKUP_RELEASE_GATE_STEP,
                    PICKUP_RELEASE_GATE_STEP,
                    PICKUP_RELEASE_GATE_STEP,
                ),
                dtype=torch.long,
            ),
            holder_active=torch.tensor(
                (True, True, False, False)
            ),
            grasp_latched=torch.tensor(
                (False, False, True, True)
            ),
            all_four_contact=torch.tensor(
                (False, True, True, True)
            ),
            all_four_contact_stable_steps=torch.tensor(
                (
                    0,
                    PICKUP_RELEASE_GATE_STABLE_STEPS,
                    PICKUP_RELEASE_GATE_STABLE_STEPS - 1,
                    PICKUP_RELEASE_GATE_STABLE_STEPS,
                )
            ),
            diffusion_handoff_ready=lambda: torch.ones(
                4, dtype=torch.bool
            ),
            preinsert_alignment_ready=lambda: torch.ones(
                4, dtype=torch.bool
            ),
            shaft_captured=torch.ones(4, dtype=torch.bool),
            capture_alignment_ready=lambda: torch.ones(
                4, dtype=torch.bool
            ),
            thread_engagement_ready=lambda: torch.ones(
                4, dtype=torch.bool
            ),
            thread_engaged=torch.ones(4, dtype=torch.bool),
            contact_recovery_pending=lambda: torch.zeros(
                4, dtype=torch.bool
            ),
        )
        values = env._schedule_next_step()
        torch.testing.assert_close(
            values[0],
            torch.tensor(
                (
                    PICKUP_RELEASE_GATE_STEP,
                    PICKUP_RELEASE_GATE_STEP,
                    PICKUP_RELEASE_GATE_STEP,
                    PICKUP_RELEASE_GATE_STEP + 1,
                )
            ),
        )
        torch.testing.assert_close(
            values[1],
            torch.tensor((False, True, True, False)),
        )

    def test_policy_mask_is_zero_through_handoff(self):
        steps = torch.tensor((174, 175, 176))
        phases = torch.full((3,), int(TaskPhase.ALIGN), dtype=torch.long)
        phase_mask = residual_action_mask(
            phases, torch.zeros(3, dtype=torch.bool)
        )
        learned_active = steps > SCRIPTED_PREFIX_STEPS
        effective = phase_mask & learned_active[:, None]
        torch.testing.assert_close(
            effective[:2], torch.zeros((2, 6), dtype=torch.bool)
        )
        torch.testing.assert_close(effective[2], phase_mask[2])

        anchor = self.supervisor.at(
            torch.tensor((176, 176, 176)), torch.zeros((3, 2))
        )
        candidate = anchor.clone()
        candidate.tip.add_(1.0)
        selected = _select_command(learned_active, candidate, anchor)
        torch.testing.assert_close(selected.tip[:2], anchor.tip[:2])
        torch.testing.assert_close(selected.tip[2], candidate.tip[2])

    def test_phase_conditioned_residual_mask(self):
        phases = torch.tensor(
            (
                int(TaskPhase.PREGRASP),
                int(TaskPhase.ALIGN),
                int(TaskPhase.INSERT),
                int(TaskPhase.ENGAGE),
                int(TaskPhase.POWER),
                int(TaskPhase.LR_RESET),
            )
        )
        engaged = torch.tensor((False, False, False, False, True, True))
        expected = torch.tensor(
            (
                (1, 1, 0, 1, 1, 0),
                (1, 1, 1, 1, 1, 0),
                (1, 1, 1, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
                (1, 1, 0, 1, 1, 1),
                (0, 0, 0, 0, 0, 0),
            ),
            dtype=torch.bool,
        )
        torch.testing.assert_close(
            residual_action_mask(phases, engaged), expected
        )

    def test_next_phase_mask_drops_prethread_tilt_and_regrasp_actions(self):
        insert = residual_action_mask(
            torch.tensor((int(TaskPhase.INSERT),)),
            torch.tensor((False,)),
        )
        torch.testing.assert_close(
            insert,
            torch.tensor(((True, True, True, False, False, False),)),
        )
        regrasp = residual_action_mask(
            torch.tensor((int(TaskPhase.LR_OPEN),)),
            torch.tensor((True,)),
        )
        torch.testing.assert_close(
            regrasp, torch.zeros((1, 6), dtype=torch.bool)
        )
        waiting_for_thread = residual_action_mask(
            torch.tensor((int(TaskPhase.ENGAGE),)),
            torch.tensor((False,)),
        )
        torch.testing.assert_close(
            waiting_for_thread, torch.zeros((1, 6), dtype=torch.bool)
        )

        closing = residual_action_mask(
            torch.tensor((int(TaskPhase.UD_CLOSE),)),
            torch.tensor((True,)),
        )
        power = residual_action_mask(
            torch.tensor((int(TaskPhase.POWER),)),
            torch.tensor((True,)),
        )
        torch.testing.assert_close(
            closing & power, torch.zeros((1, 6), dtype=torch.bool)
        )

    def test_command_checkpoint_restores_in_place(self):
        command = self.supervisor.at(
            torch.tensor((0, 100)), torch.zeros((2, 2))
        )
        state = command.state_dict()
        pointers = {
            name: getattr(command, name).data_ptr()
            for name in state
        }
        command.tip.add_(3.0)
        command.rotation.zero_()
        command.twist.add_(1.0)
        command.load_state_dict(state)
        for name, expected in state.items():
            self.assertEqual(getattr(command, name).data_ptr(), pointers[name])
            torch.testing.assert_close(getattr(command, name), expected)

    def test_full_prefix_gate_retries_and_has_finite_hard_failure(self):
        class FakeSupervisor:
            def __init__(self):
                self.config = SimpleNamespace(spawn_radius_m=0.100)
                self.fixed_socket_angle_deg = 45.0
                self.calls = 0
                self.radii = []

            def curriculum_parameters(self, progress):
                return (0.060 + 0.040 * float(progress), 45.0)

            def sample_socket_mouth(self, count, *, generator=None):
                return torch.tensor((0.020, 0.0)).expand(count, 2).clone()

            @staticmethod
            def fixed_angle_batch(count):
                return torch.full((count,), 45.0)

            def sample_spawn(
                self,
                count,
                *,
                socket_mouth_xy,
                socket_angle_deg,
                radius_m,
                generator=None,
            ):
                self.calls += 1
                self.radii.append(float(radius_m))
                return torch.full((count, 2), 0.010 * self.calls)

        def command(count):
            return CommandBatch(
                tip=torch.zeros((count, 3)),
                rotation=torch.eye(3).expand(count, 3, 3).clone(),
                twist=torch.zeros(count),
                material=torch.zeros((count, 4)),
                distance=torch.zeros((count, 4)),
                phase=torch.zeros(count, dtype=torch.long),
                engaged=torch.zeros(count, dtype=torch.bool),
            )

        env = object.__new__(MJWarpScrewEnv)
        env.worlds = 2
        env.device = torch.device("cpu")
        env.generator = torch.Generator(device="cpu")
        env.state = SimpleNamespace(
            spawn_xy=torch.zeros((2, 2)),
            socket_angle_deg=torch.full((2,), 45.0),
            socket_mouth=torch.tensor(
                ((0.020, 0.0, 0.345), (0.020, 0.0, 0.345))
            ),
        )
        env.current_command = command(2)
        env.supervisor = FakeSupervisor()
        env.curriculum_progress = 1.0
        certification_calls = []

        def retry_once(spawn, angle, mouth):
            certification_calls.append(int(spawn.shape[0]))
            accepted = torch.ones(spawn.shape[0], dtype=torch.bool)
            if len(certification_calls) == 1:
                accepted[0] = False
            return (
                accepted,
                command(int(spawn.shape[0])),
                torch.zeros((spawn.shape[0], 16)),
                torch.zeros((spawn.shape[0], 4)),
            )

        env._certify_scripted_prefix = retry_once
        sampled = env._sample_reachable(torch.ones(2, dtype=torch.bool))
        self.assertEqual(certification_calls, [2, 1])
        self.assertEqual(env.supervisor.radii, [0.100, 0.100])
        self.assertTrue(torch.isfinite(sampled[4]).all())
        self.assertTrue(torch.isfinite(sampled[5]).all())

        env.supervisor = FakeSupervisor()
        env.curriculum_progress = 0.0

        def never_reachable(spawn, angle, mouth):
            return (
                torch.zeros(spawn.shape[0], dtype=torch.bool),
                command(int(spawn.shape[0])),
                torch.zeros((spawn.shape[0], 16)),
                torch.zeros((spawn.shape[0], 4)),
            )

        env._certify_scripted_prefix = never_reachable
        with patch("s_hpc.env.RESET_IK_RESAMPLE_BATCHES", 2):
            with self.assertRaisesRegex(
                RuntimeError,
                "exhausted 2 batches.*2 world",
            ):
                env._sample_reachable(torch.ones(2, dtype=torch.bool))
        self.assertEqual(env.supervisor.radii, [0.060, 0.060])

    def test_so3_roundtrip(self):
        vector = torch.randn((100, 3)) * 0.25
        recovered = rotation_log(rotation_exp(vector))
        torch.testing.assert_close(recovered, vector, atol=2.0e-6, rtol=0.0)

    def test_diffusion_anchor_blend_uses_minimum_jerk_and_so3(self):
        worlds = 3
        identity = torch.eye(3).expand(worlds, 3, 3).clone()
        source = CommandBatch(
            tip=torch.zeros((worlds, 3)),
            rotation=identity,
            twist=torch.zeros(worlds),
            material=torch.full((worlds, 4), 3.10),
            distance=torch.full((worlds, 4), 0.039),
            phase=torch.zeros(worlds, dtype=torch.long),
            engaged=torch.zeros(worlds, dtype=torch.bool),
        )
        target_rotation = rotation_exp(
            torch.tensor(
                ((0.0, 0.0, math.pi / 2.0),) * worlds
            )
        )
        target = CommandBatch(
            tip=torch.full((worlds, 3), 0.020),
            rotation=target_rotation,
            twist=torch.full((worlds,), 2.0),
            material=torch.full((worlds, 4), -3.10),
            distance=torch.full((worlds, 4), 0.049),
            phase=torch.full((worlds,), int(TaskPhase.INSERT)),
            engaged=torch.ones(worlds, dtype=torch.bool),
        )

        blended = _blend_command(
            source, target, torch.tensor((0.0, 0.5, 1.0))
        )

        torch.testing.assert_close(blended.tip[0], source.tip[0])
        torch.testing.assert_close(blended.tip[1], torch.full((3,), 0.010))
        torch.testing.assert_close(blended.tip[2], target.tip[2])
        torch.testing.assert_close(
            rotation_log(blended.rotation)[1],
            torch.tensor((0.0, 0.0, math.pi / 4.0)),
        )
        orthogonal = (
            blended.rotation.transpose(-1, -2) @ blended.rotation
        )
        torch.testing.assert_close(
            orthogonal, identity, atol=2.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            torch.linalg.det(blended.rotation),
            torch.ones(worlds),
            atol=2.0e-6,
            rtol=0.0,
        )
        # Angular material coordinates cross +/-pi by the shortest path.
        self.assertGreater(abs(float(blended.material[1, 0])), 3.0)
        torch.testing.assert_close(blended.phase, target.phase)
        torch.testing.assert_close(blended.engaged, target.engaged)

    def test_diffusion_anchor_takeover_reaches_anchor_on_frame_twenty(self):
        worlds = 2

        def command(tip_value):
            return CommandBatch(
                tip=torch.full((worlds, 3), float(tip_value)),
                rotation=torch.eye(3).expand(worlds, 3, 3).clone(),
                twist=torch.zeros(worlds),
                material=torch.zeros((worlds, 4)),
                distance=torch.full((worlds, 4), 0.039),
                phase=torch.full(
                    (worlds,), int(TaskPhase.ALIGN), dtype=torch.long
                ),
                engaged=torch.zeros(worlds, dtype=torch.bool),
            )

        env = object.__new__(MJWarpScrewEnv)
        env.device = torch.device("cpu")
        env.current_command = command(0.0)
        env.diffusion_anchor_blend_source = command(0.0)
        env.diffusion_anchor_blend_step = torch.zeros(
            worlds, dtype=torch.long
        )
        env.diffusion_anchor_takeover_active = torch.zeros(
            worlds, dtype=torch.bool
        )
        diffusion = command(-1.0)
        anchor = command(1.0)
        takeover = torch.tensor((True, False))

        first, active, progress = env._smooth_diffusion_to_anchor(
            diffusion, anchor, takeover
        )
        self.assertGreater(float(first.tip[0, 0]), 0.0)
        self.assertLess(float(first.tip[0, 0]), 0.01)
        torch.testing.assert_close(first.tip[1], diffusion.tip[1])
        self.assertTrue(bool(active[0]))
        self.assertAlmostEqual(float(progress[0]), 1.0 / 20.0)

        result = first
        for _ in range(19):
            env.current_command = result
            result, active, progress = env._smooth_diffusion_to_anchor(
                diffusion, anchor, takeover
            )
        torch.testing.assert_close(result.tip[0], anchor.tip[0])
        self.assertFalse(bool(active[0]))
        self.assertAlmostEqual(float(progress[0]), 1.0)

        env.current_command = result
        released, active, _progress = env._smooth_diffusion_to_anchor(
            diffusion, anchor, torch.zeros(worlds, dtype=torch.bool)
        )
        torch.testing.assert_close(released.tip, diffusion.tip)
        self.assertFalse(bool(active.any()))
        self.assertFalse(bool(env.diffusion_anchor_takeover_active.any()))
        self.assertFalse(bool(env.diffusion_anchor_blend_step.any()))

    def test_nominal_rate_limiter_does_not_feed_back_applied_residual(self):
        """The next IK nominal must not absorb the prior TQC correction."""

        env = object.__new__(MJWarpScrewEnv)
        env.device = torch.device("cpu")
        env.ctrl_velocity = torch.zeros((1, 16))
        env.ctrl_acceleration = torch.zeros((1, 16))
        env.last_nominal_ctrl = torch.zeros((1, 16))
        # Deliberately provide no backend.tensor("ctrl") accessor: the nominal
        # rate limiter may use only its independent nominal cursor.
        env.backend = SimpleNamespace(
            control_dt=0.02,
            ctrl_range=torch.tensor(((-1.0, 1.0),) * 16),
        )
        desired = torch.full((1, 16), 0.1)

        limited = env._rate_limit(desired)

        self.assertEqual(limited.shape, (1, 16))
        self.assertTrue(bool(torch.isfinite(limited).all()))
        torch.testing.assert_close(env.last_nominal_ctrl, limited)

    def test_residual_freezes_regrasp_and_obeys_lead(self):
        safety = TaskSpaceSafety(self.supervisor)
        spawn = torch.zeros((2, 2))
        lr_index = int(
            torch.nonzero(
                self.supervisor.phase == int(TaskPhase.LR_RESET)
            )[0]
        )
        power_index = int(
            torch.nonzero(
                self.supervisor.phase == int(TaskPhase.POWER)
            )[0]
        )
        command = self.supervisor.at(
            torch.tensor((lr_index, power_index)), spawn
        )
        residual = torch.ones((2, 6))
        projected = safety.residual(command, residual)
        torch.testing.assert_close(projected.tip[0], command.tip[0])
        expected_advance = (
            self.supervisor.config.thread_pitch_m
            * math.radians(2.0)
            / (2.0 * math.pi)
        )
        actual_advance = torch.dot(
            projected.tip[1] - command.tip[1], self.supervisor.socket_axis
        )
        self.assertAlmostEqual(float(actual_advance), expected_advance, places=7)

    def test_residual_can_slow_but_never_reverse_certified_twist(self):
        safety = TaskSpaceSafety(self.supervisor)
        spawn = torch.zeros((1, 2))
        power_index = int(
            torch.nonzero(
                self.supervisor.phase == int(TaskPhase.POWER)
            )[10]
        )
        base = self.supervisor.at(torch.tensor((power_index,)), spawn)
        previous = base.clone()
        previous.twist -= math.radians(0.5)
        residual = torch.zeros((1, 6))
        residual[:, 5] = -1.0
        projected = safety.residual(
            base, residual, previous=previous
        )
        torch.testing.assert_close(projected.twist, previous.twist)
        self.assertLess(float(projected.twist), float(base.twist))
        axial = torch.dot(
            projected.tip[0] - base.tip[0],
            self.supervisor.socket_axis,
        )
        expected = (
            self.supervisor.config.thread_pitch_m
            * float(projected.twist[0] - base.twist[0])
            / (2.0 * math.pi)
        )
        self.assertAlmostEqual(float(axial), expected, places=7)

    def test_axial_residual_is_disabled_away_from_insertion(self):
        safety = TaskSpaceSafety(self.supervisor)
        spawn = torch.zeros((1, 2))
        pregrasp_index = int(
            torch.nonzero(
                self.supervisor.phase == int(TaskPhase.PREGRASP)
            )[0]
        )
        command = self.supervisor.at(
            torch.tensor((pregrasp_index,)), spawn
        )
        residual = torch.zeros((1, 6))
        residual[:, 2] = 1.0
        projected = safety.residual(command, residual)
        torch.testing.assert_close(projected.tip, command.tip)


if __name__ == "__main__":
    unittest.main()
