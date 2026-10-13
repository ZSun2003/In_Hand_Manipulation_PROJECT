"""Focused CPU regressions for expert gates and Cartesian tip limiting."""

from __future__ import annotations

import inspect
from types import MethodType
import unittest

from MJWarp_RL import contract
from MJWarp_RL.env import (
    EXPERT_OBJECT_ANGULAR_SPEED_MAX_RAD_S,
    EXPERT_OBJECT_LINEAR_SPEED_MAX_M_S,
    EXPERT_STABLE_TICKS,
    IK_TRUST_STAGES,
    MJWarpIKResidualEnv,
    PRELIFT_CONTACT_STABLE_TICKS,
    PRELIFT_CONTACT_STAGES,
    TIP_TARGET_ACCELERATION_MAX_M_S2,
    TIP_TARGET_JERK_MAX_M_S3,
    TIP_TARGET_SPEED_MAX_M_S,
)
from MJWarp_RL.hpc_config import DEFAULT_CONFIG


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ExpertGateTests(unittest.TestCase):
    def test_contact_cause_telemetry_is_exported_by_snapshot(self) -> None:
        source = inspect.getsource(MJWarpIKResidualEnv.snapshot)
        for name in (
            "prelift_tip_raw_pair_current",
            "prelift_tip_active_pair_current",
            "prelift_tip_contact_qualified_current",
            "prelift_tip_raw_pair_missing_count_by_stage",
            "prelift_tip_efc_inactive_count_by_stage",
            "prelift_tip_force_failure_count_by_stage",
            "prelift_lift_boundary_tip_raw_pair",
            "prelift_lift_boundary_tip_active_pair",
            "prelift_lift_boundary_tip_contact_qualified",
            "translation_entry_tip_object_raw_pair",
            "translation_entry_tip_object_active_pair",
            "translation_entry_tip_contact_qualified",
            "translation_entry_tip_raw_pair_missing_count",
            "translation_entry_tip_efc_inactive_count",
            "translation_entry_tip_force_failure_count",
        ):
            self.assertIn(f'"{name}"', source)

    def test_contact_decoder_ignores_stale_slots_after_nacon(self) -> None:
        class Backend:
            def __init__(self) -> None:
                self.views = {
                    "contact.geom": torch.tensor(
                        [
                            [10, 20],
                            [11, 20],
                            [12, 20],
                            [13, 20],
                            [10, 20],
                        ]
                    ),
                    "contact.worldid": torch.zeros(5, dtype=torch.long),
                    "contact.dim": torch.tensor([1, 3, 1, 1, 1]),
                    # Slot three is a live sensor-only contact; slot four is
                    # a stale constraint contact outside nacon.
                    "contact.type": torch.tensor([1, 1, 1, 2, 1]),
                    "contact.efc_address": torch.tensor(
                        [
                            [-1, -1, -1, -1],
                            # The normal/first cell may be unavailable while
                            # a later native matrix cell remains active.
                            [-1, 0, -1, -1],
                            [1, -1, -1, -1],
                            [2, -1, -1, -1],
                            [3, -1, -1, -1],
                        ]
                    ),
                    "efc.force": torch.tensor(
                        [[0.02, 0.04, 100.0, 200.0, 0.0]]
                    ),
                    "nacon": torch.tensor([4]),
                }

            def tensor(self, name: str):
                return self.views[name]

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.torch = torch
        environment.backend = Backend()
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = 1
        environment.object_index = torch.tensor([0])
        environment._object_geom_mask = torch.zeros((1, 32), dtype=torch.bool)
        environment._object_geom_mask[0, 20] = True
        environment._tip_geom_ids = torch.tensor([10, 11, 12, 13])
        environment._last_force_decode_finite = torch.ones(1, dtype=torch.bool)
        environment._last_tip_object_contact_raw_pair = torch.zeros(
            (1, 4), dtype=torch.bool
        )
        environment._last_tip_object_contact_pair = torch.zeros(
            (1, 4), dtype=torch.bool
        )
        environment._last_tip_object_contact_force_qualified_pair = (
            torch.zeros((1, 4), dtype=torch.bool)
        )

        force = environment._finger_contact_force()

        torch.testing.assert_close(
            force, torch.tensor([[0.0, 0.02, 0.04, 0.0]])
        )
        self.assertEqual(
            environment._last_tip_object_contact_raw_pair.tolist(),
            [[True, True, True, True]],
        )
        self.assertEqual(
            environment._last_tip_object_contact_pair.tolist(),
            [[False, True, True, False]],
        )
        self.assertEqual(
            environment._last_tip_object_contact_force_qualified_pair.tolist(),
            [[False, False, True, False]],
        )
        self.assertEqual(environment._last_force_decode_finite.tolist(), [True])

        # A non-pinned legacy backend lacking Contact.type is accepted for
        # diagnostics, but cannot prove an active EFC pair and therefore
        # fails closed without raising or inventing force.
        del environment.backend.views["contact.type"]
        force_without_type = environment._finger_contact_force()
        torch.testing.assert_close(force_without_type, torch.zeros((1, 4)))
        self.assertEqual(
            environment._last_tip_object_contact_raw_pair.tolist(),
            [[True, True, True, True]],
        )
        self.assertEqual(
            environment._last_tip_object_contact_pair.tolist(),
            [[False, False, False, False]],
        )

    def test_float32_early_one_progress_uses_integer_sample_certificate(
        self,
    ) -> None:
        worlds = len(contract.TRANSLATION_PATH_SHAPES)
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.torch = torch
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = worlds
        environment.path_shape_index = torch.arange(worlds)
        environment._translation_route_progress_table = torch.tensor(
            contract.TRANSLATION_ROUTE_PROGRESS_TABLE, dtype=torch.float32
        )
        environment._translation_route_targets_xy_table = torch.tensor(
            contract.TRANSLATION_ROUTE_TARGETS_XY_M, dtype=torch.float32
        )
        starts = environment._translation_route_targets_xy_table[:, 0].clone()
        environment.path_tracking_sample_count = torch.zeros(
            worlds, dtype=torch.long
        )
        environment.path_schedule_previous_offset_xy_m = starts
        environment.path_schedule_previous_velocity_xy_m_s = torch.zeros_like(
            starts
        )
        environment.path_schedule_progress = torch.zeros(worlds)
        environment.path_schedule_limit_valid = torch.ones(
            worlds, dtype=torch.bool
        )
        environment.path_schedule_complete = torch.zeros(
            worlds, dtype=torch.bool
        )
        environment.path_schedule_max_speed_m_s = torch.zeros(worlds)
        environment.path_schedule_max_acceleration_m_s2 = torch.zeros(worlds)
        self.assertTrue(
            bool(
                (
                    environment._translation_route_progress_table[:, 1:-1]
                    == 1.0
                ).any()
            )
        )
        sample = torch.ones(worlds, dtype=torch.bool)
        for index in range(1, contract.TRANSLATION_PATH_ROUTE_SAMPLES + 1):
            environment._update_path_schedule_certificate(
                sample,
                environment._translation_route_progress_table[:, index],
                environment._translation_route_targets_xy_table[:, index],
            )
            environment.path_tracking_sample_count.add_(1)
        self.assertTrue(bool(environment.path_schedule_limit_valid.all()))
        self.assertTrue(bool(environment.path_schedule_complete.all()))

    def test_unadmitted_route_never_jumps_to_endpoint(self) -> None:
        action_start = sum(
            (
                DEFAULT_CONFIG.plan_ticks,
                DEFAULT_CONFIG.close_ticks,
                DEFAULT_CONFIG.settle_ticks,
                DEFAULT_CONFIG.lift_ticks,
                DEFAULT_CONFIG.hold_ticks,
            )
        )
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.torch = torch
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = 2
        environment.config = DEFAULT_CONFIG
        environment.control_step = torch.tensor(
            (action_start + 600, action_start + 2_500)
        )
        environment.path_shape_index = torch.zeros(2, dtype=torch.long)
        environment.task_index = torch.full(
            (2,), contract.TASKS.index(contract.TASK_TRANSLATE)
        )
        environment.action_reference_valid = torch.ones(2, dtype=torch.bool)
        environment.translation_route_admitted = torch.zeros(
            2, dtype=torch.bool
        )
        environment.path_waypoints_xy_m = torch.tensor(
            contract.TRANSLATION_PATH_WAYPOINTS_XY_M[0], dtype=torch.float32
        )[None].expand(2, -1, -1)
        environment.action_start_com = torch.zeros((2, 3))
        environment._translation_route_progress_table = torch.tensor(
            contract.TRANSLATION_ROUTE_PROGRESS_TABLE, dtype=torch.float32
        )
        environment._translation_lead_progress_table = torch.tensor(
            contract.TRANSLATION_LEAD_PROGRESS_TABLE, dtype=torch.float32
        )
        environment._translation_route_targets_xy_table = torch.tensor(
            contract.TRANSLATION_ROUTE_TARGETS_XY_M, dtype=torch.float32
        )
        environment._translation_lead_offsets_xy_table = torch.tensor(
            contract.TRANSLATION_LEAD_OFFSETS_XY_M, dtype=torch.float32
        )
        offset, _, *_, following = environment._translation_path_state()
        torch.testing.assert_close(
            offset[:, :2], environment.path_waypoints_xy_m[:, 0]
        )
        self.assertEqual(following.tolist(), [False, False])

    def test_every_commanded_tip_is_projected_into_40_60_band(self) -> None:
        class Backend:
            views = {
                "xpos": torch.zeros((1, 1, 3)),
                "xquat": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            }

            def tensor(self, name: str):
                return self.views[name]

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = Backend()
        environment._rows = torch.tensor([0])
        environment._object_body_ids = torch.zeros(4, dtype=torch.long)
        environment.object_index = torch.tensor([0])
        environment._object_local_bottom_z_m = torch.zeros(4)
        environment._object_local_top_z_m = torch.ones(4)
        environment._grasp_plane_nominal_fraction_by_object = torch.tensor(
            [contract.grasp_plane_nominal_fraction(name) for name in contract.OBJECTS]
        )
        environment._grasp_plane_fraction_limits_by_object = torch.tensor(
            [contract.grasp_plane_fraction_limits(name) for name in contract.OBJECTS]
        )
        environment._grasp_plane_learned_by_object = torch.tensor(
            [contract.grasp_plane_is_learned(name) for name in contract.OBJECTS]
        )
        targets = torch.tensor(
            [[[0.0, 0.0, 0.10], [0.0, 0.0, 0.20],
              [0.0, 0.0, 0.61], [0.0, 0.0, 0.40]]]
        )
        projected = environment._project_grasp_plane(targets)
        fractions, _, compliant = environment._grasp_plane_metrics(projected)
        torch.testing.assert_close(
            fractions,
            torch.tensor([[0.40, 0.40, 0.60, 0.40]]),
        )
        self.assertTrue(compliant.item())
        low_action = torch.zeros((1, contract.ACTION_DIM))
        low_action[:, contract.TIP_HEIGHT_SLICE] = -1.0
        high_action = torch.zeros((1, contract.ACTION_DIM))
        high_action[:, contract.TIP_HEIGHT_SLICE] = 1.0
        torch.testing.assert_close(
            environment._requested_tip_height_fractions(low_action),
            torch.full((1, 4), contract.GRASP_PLANE_MIN_FRACTION),
        )
        torch.testing.assert_close(
            environment._requested_tip_height_fractions(high_action),
            torch.full((1, 4), contract.GRASP_PLANE_MAX_FRACTION),
        )

    def test_cartesian_tip_target_v_a_j_limits_and_nonfinite_fallback(
        self,
    ) -> None:
        class Backend:
            control_dt = 0.02

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = Backend()
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = 2
        shape = (2, 4, 3)
        environment.last_certified_tip_targets = torch.zeros(shape)
        environment.tip_target_velocity_world_m_s = torch.zeros(shape)
        environment.tip_target_acceleration_world_m_s2 = torch.zeros(shape)
        environment.tip_target_jerk_world_m_s3 = torch.zeros(shape)
        environment.tip_target_slew_clip_count = torch.zeros(
            2, dtype=torch.long
        )
        environment.tip_target_certificate_fallback_count = torch.zeros(
            2, dtype=torch.long
        )
        environment._project_grasp_plane = MethodType(
            lambda self, value: value, environment
        )
        environment._grasp_plane_metrics = MethodType(
            lambda self, value: (
                torch.zeros(self.worlds),
                torch.zeros((self.worlds, len(contract.TIP_ORDER))),
                torch.ones(self.worlds, dtype=torch.bool),
            ),
            environment,
        )
        desired = torch.ones(shape)
        desired[1, 0, 0] = torch.nan
        result = environment._rate_limit_tip_targets(
            desired, torch.ones(2, dtype=torch.bool)
        )
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(
            bool(
                (
                    environment.tip_target_velocity_world_m_s.norm(dim=-1)
                    <= TIP_TARGET_SPEED_MAX_M_S + 1.0e-6
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    environment.tip_target_acceleration_world_m_s2.norm(dim=-1)
                    <= TIP_TARGET_ACCELERATION_MAX_M_S2 + 1.0e-6
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    environment.tip_target_jerk_world_m_s3.norm(dim=-1)
                    <= TIP_TARGET_JERK_MAX_M_S3 + 1.0e-6
                ).all()
            )
        )
        self.assertEqual(
            environment.tip_target_certificate_fallback_count.tolist(),
            [0, 1],
        )

    def test_cartesian_tip_target_servo_never_crosses_stopped_target(self) -> None:
        class Backend:
            control_dt = 0.02

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = Backend()
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = 1
        shape = (1, len(contract.TIP_ORDER), 3)
        environment.last_certified_tip_targets = torch.zeros(shape)
        environment.tip_target_velocity_world_m_s = torch.zeros(shape)
        environment.tip_target_acceleration_world_m_s2 = torch.zeros(shape)
        environment.tip_target_jerk_world_m_s3 = torch.zeros(shape)
        environment.tip_target_slew_clip_count = torch.zeros(
            1, dtype=torch.long
        )
        environment.tip_target_certificate_fallback_count = torch.zeros(
            1, dtype=torch.long
        )
        environment._project_grasp_plane = MethodType(
            lambda self, value: value, environment
        )
        desired = torch.zeros(shape)
        desired[:, :, 0] = 0.012
        active = torch.ones(1, dtype=torch.bool)

        maximum = environment.last_certified_tip_targets.clone()
        for _ in range(500):
            previous_target = environment.last_certified_tip_targets.clone()
            previous_velocity = (
                environment.tip_target_velocity_world_m_s.clone()
            )
            previous_acceleration = (
                environment.tip_target_acceleration_world_m_s2.clone()
            )
            value = environment._rate_limit_tip_targets(desired, active)
            self.assertTrue(bool((value[:, :, 0] <= 0.012 + 1.0e-9).all()))
            self.assertTrue(
                bool(
                    (
                        value[:, :, 0]
                        >= previous_target[:, :, 0] - 1.0e-9
                    ).all()
                )
            )
            maximum = torch.maximum(maximum, value)
            realized_velocity = (
                value - previous_target
            ) / environment.backend.control_dt
            realized_acceleration = (
                realized_velocity - previous_velocity
            ) / environment.backend.control_dt
            realized_jerk = (
                realized_acceleration - previous_acceleration
            ) / environment.backend.control_dt
            torch.testing.assert_close(
                environment.tip_target_velocity_world_m_s,
                realized_velocity,
            )
            torch.testing.assert_close(
                environment.tip_target_acceleration_world_m_s2,
                realized_acceleration,
            )
            torch.testing.assert_close(
                environment.tip_target_jerk_world_m_s3,
                realized_jerk,
            )

        self.assertLess(float((desired - value).norm(dim=-1).max()), 1.0e-5)
        self.assertLessEqual(float(maximum[:, :, 0].max()), 0.012 + 1.0e-9)
        self.assertEqual(
            environment.tip_target_certificate_fallback_count.tolist(), [0]
        )

    def test_expert_contact_needs_exact_pair_and_force_not_direction(self) -> None:
        class Backend:
            views = {"xpos": torch.zeros((1, 4, 3))}

            def tensor(self, name: str):
                return self.views[name]

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = Backend()
        environment._tip_body_ids = torch.arange(4)
        environment._last_force_decode_finite = torch.tensor([True])
        environment._last_tip_object_contact_pair = torch.tensor(
            [[True, True, True, False]]
        )
        environment._grasp_plane_metrics = MethodType(
            lambda self, targets, physical=False: (
                torch.full((1, 4), 0.5),
                torch.tensor([0.5]),
                torch.tensor([True]),
            ),
            environment,
        )
        environment._grasp_plane_contract_rows = MethodType(
            lambda self: (
                torch.tensor([0.5]),
                torch.tensor([0.4]),
                torch.tensor([0.6]),
                torch.tensor([True]),
            ),
            environment,
        )
        environment._active_object_drop_measurement = MethodType(
            lambda self: (
                torch.zeros((1, 3)),
                torch.zeros((1, 3)),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.tensor([contract.TABLE_TOP_Z_M + 0.01]),
            ),
            environment,
        )
        environment._object_velocity = MethodType(
            lambda self: torch.zeros((1, 6)), environment
        )
        metrics = environment._expert_physical_metrics(
            torch.tensor([[0.10, 0.10, 0.10, 100.0]])
        )
        self.assertEqual(metrics[2].tolist(), [3])
        self.assertEqual(metrics[1].tolist(), [[True, True, True, False]])
        environment._last_tip_object_contact_pair[0, 3] = True
        metrics = environment._expert_physical_metrics(
            torch.tensor([[0.10, 0.10, 0.10, 0.10]])
        )
        self.assertEqual(metrics[2].tolist(), [4])
        environment._last_force_decode_finite[0] = False
        metrics = environment._expert_physical_metrics(
            torch.full((1, 4), 100.0)
        )
        self.assertEqual(metrics[2].tolist(), [0])

    def test_cage_workspace_projection_uses_one_scale_for_all_tips(self) -> None:
        class Backend:
            views = {"xpos": torch.zeros((2, 4, 3))}

            def tensor(self, name: str):
                return self.views[name]

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = Backend()
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = 2
        environment.config = DEFAULT_CONFIG
        environment._tip_body_ids = torch.arange(4)
        environment.control_step = torch.zeros(2, dtype=torch.long)
        environment.task_index = torch.full(
            (2,), contract.TASKS.index(contract.TASK_TRANSLATE)
        )
        environment.ik_cage_projection_count = torch.zeros(2, dtype=torch.long)
        environment.ik_cage_projection_count_by_stage = torch.zeros(
            (2, len(IK_TRUST_STAGES)), dtype=torch.long
        )
        environment.ik_cage_projection_scale_min = torch.ones(2)
        environment.ik_cage_projection_scale_sum = torch.zeros(2)
        environment.ik_cage_projection_sample_count = torch.zeros(
            2, dtype=torch.long
        )
        targets = torch.tensor(
            (
                ((0.006, 0.0, 0.0), (0.0, 0.003, 0.0),
                 (0.0, 0.0, 0.0015), (-0.004, 0.0, 0.0)),
                ((0.001, 0.0, 0.0), (0.0, 0.001, 0.0),
                 (0.0, 0.0, 0.001), (-0.001, 0.0, 0.0)),
            )
        )
        projected = environment._project_ik_cage_targets(
            targets, torch.tensor([True, True])
        )
        torch.testing.assert_close(projected[0], targets[0] * 0.5)
        torch.testing.assert_close(projected[1], targets[1])
        self.assertEqual(environment.ik_cage_projection_count.tolist(), [1, 0])
        self.assertEqual(environment.ik_cage_projection_sample_count.tolist(), [1, 1])

    def test_prelift_gate_requires_five_boundary_contacts_and_freezes_pose(
        self,
    ) -> None:
        class Backend:
            views = {
                "xpos": torch.tensor(
                    [[[0.10, 0.20, 0.30]], [[0.40, 0.50, 0.60]]]
                ),
                "xquat": torch.tensor(
                    [
                        [[1.0, 0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0, 2.0]],
                    ]
                ),
            }

            def tensor(self, name: str):
                return self.views[name]

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = Backend()
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = 2
        environment.config = DEFAULT_CONFIG
        environment._rows = torch.arange(2)
        environment._object_body_ids = torch.zeros(4, dtype=torch.long)
        # Different fruit indices prove that diagnostics remain per world and
        # can therefore be grouped per object by the capacity scanner.
        environment.object_index = torch.tensor([0, 2], dtype=torch.long)
        environment.object_dropped = torch.zeros(2, dtype=torch.bool)
        environment._last_force_decode_finite = torch.ones(
            2, dtype=torch.bool
        )
        environment._last_tip_object_contact_raw_pair = torch.tensor(
            [[True, True, True, False], [True, True, True, True]]
        )
        environment._last_tip_object_contact_pair = torch.tensor(
            [[True, True, True, False], [True, True, True, True]]
        )
        environment._last_tip_object_contact_force_qualified_pair = (
            environment._last_tip_object_contact_pair.clone()
        )
        environment._active_object_drop_measurement = MethodType(
            lambda self: (
                torch.zeros((2, 3)),
                torch.zeros((2, 3)),
                torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
                ),
                torch.full(
                    (2,), contract.OBJECT_DROP_THRESHOLD_Z_M + 0.01
                ),
            ),
            environment,
        )
        environment.prelift_contact_stable_count = torch.zeros(
            2, dtype=torch.long
        )
        environment.prelift_contact_max_stable_count = torch.zeros(
            2, dtype=torch.long
        )
        environment.prelift_contact_gate_latched = torch.zeros(
            2, dtype=torch.bool
        )
        environment.prelift_contact_gate_first_control_step = torch.full(
            (2,), -1, dtype=torch.long
        )
        environment.prelift_lift_boundary_contact_valid = torch.zeros(
            2, dtype=torch.bool
        )
        environment.prelift_lift_permitted = torch.zeros(2, dtype=torch.bool)
        environment.prelift_lift_reference_valid = torch.zeros(
            2, dtype=torch.bool
        )
        environment.prelift_lift_start_body_pos = torch.zeros((2, 3))
        environment.prelift_lift_start_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        )
        environment.prelift_contact_count_current = torch.zeros(
            2, dtype=torch.long
        )
        environment.prelift_tip_raw_pair_current = torch.zeros(
            (2, 4), dtype=torch.bool
        )
        environment.prelift_tip_active_pair_current = torch.zeros(
            (2, 4), dtype=torch.bool
        )
        environment.prelift_tip_contact_qualified_current = torch.zeros(
            (2, 4), dtype=torch.bool
        )
        environment.prelift_contact_stage_index = torch.full(
            (2,), -1, dtype=torch.long
        )
        stage_shape = (2, len(PRELIFT_CONTACT_STAGES))
        environment.prelift_stage_observation_count = torch.zeros(
            stage_shape, dtype=torch.long
        )
        environment.prelift_stage_qualified_count = torch.zeros(
            stage_shape, dtype=torch.long
        )
        environment.prelift_stage_failure_count = torch.zeros(
            stage_shape, dtype=torch.long
        )
        environment.prelift_stage_max_contact_count = torch.zeros(
            stage_shape, dtype=torch.long
        )
        environment.prelift_stage_first_failure_control_step = torch.full(
            stage_shape, -1, dtype=torch.long
        )
        environment.prelift_stage_last_failure_control_step = torch.full(
            stage_shape, -1, dtype=torch.long
        )
        environment.prelift_tip_pair_missing_count_by_stage = torch.zeros(
            (2, len(PRELIFT_CONTACT_STAGES), 4), dtype=torch.long
        )
        environment.prelift_tip_raw_pair_missing_count_by_stage = torch.zeros(
            (2, len(PRELIFT_CONTACT_STAGES), 4), dtype=torch.long
        )
        environment.prelift_tip_efc_inactive_count_by_stage = torch.zeros(
            (2, len(PRELIFT_CONTACT_STAGES), 4), dtype=torch.long
        )
        environment.prelift_tip_force_failure_count_by_stage = torch.zeros(
            (2, len(PRELIFT_CONTACT_STAGES), 4), dtype=torch.long
        )
        environment.prelift_lift_boundary_tip_raw_pair = torch.zeros(
            (2, 4), dtype=torch.bool
        )
        environment.prelift_lift_boundary_tip_active_pair = torch.zeros(
            (2, 4), dtype=torch.bool
        )
        environment.prelift_lift_boundary_tip_contact_qualified = torch.zeros(
            (2, 4), dtype=torch.bool
        )
        environment.prelift_lift_boundary_contact_force_n = torch.zeros(
            (2, 4)
        )

        close_first = DEFAULT_CONFIG.plan_ticks + 1
        force = torch.full((2, 4), 1.0)
        for tick in range(
            close_first, close_first + PRELIFT_CONTACT_STABLE_TICKS
        ):
            environment.control_step = torch.full((2,), tick)
            environment._update_prelift_contact_gate(force)
        self.assertEqual(
            environment.prelift_contact_stable_count.tolist(), [5, 5]
        )
        self.assertEqual(
            environment.prelift_contact_gate_latched.tolist(), [True, True]
        )
        self.assertEqual(
            environment.prelift_lift_permitted.tolist(), [False, False]
        )

        lift_start = (
            DEFAULT_CONFIG.plan_ticks
            + DEFAULT_CONFIG.close_ticks
            + DEFAULT_CONFIG.settle_ticks
        )
        # World 1 had a historical five-tick latch, then loses contact once
        # and recovers for only four ticks through the lift boundary.  The
        # historical latch must not authorize its lift.
        environment._last_tip_object_contact_raw_pair[0].fill_(True)
        environment._last_tip_object_contact_pair[0].fill_(True)
        environment._last_tip_object_contact_force_qualified_pair[0] = (
            torch.tensor([True, True, True, False])
        )
        environment._last_tip_object_contact_pair[1] = torch.tensor(
            [True, True, False, False]
        )
        environment._last_tip_object_contact_force_qualified_pair[1] = (
            environment._last_tip_object_contact_pair[1]
        )
        environment.control_step = torch.full((2,), lift_start - 4)
        environment._update_prelift_contact_gate(force)
        environment._last_tip_object_contact_pair[1] = torch.tensor(
            [True, True, True, True]
        )
        environment._last_tip_object_contact_force_qualified_pair[1] = (
            environment._last_tip_object_contact_pair[1]
        )
        for tick in range(lift_start - 3, lift_start + 1):
            environment.control_step = torch.full((2,), tick)
            environment._update_prelift_contact_gate(force)
        self.assertEqual(
            environment.prelift_lift_boundary_contact_valid.tolist(),
            [True, False],
        )
        self.assertEqual(
            environment.prelift_lift_permitted.tolist(), [True, False]
        )
        self.assertEqual(
            environment.prelift_lift_reference_valid.tolist(), [True, True]
        )
        torch.testing.assert_close(
            environment.prelift_lift_start_body_pos,
            Backend.views["xpos"][:, 0],
        )
        torch.testing.assert_close(
            environment.prelift_lift_start_quat[1],
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        self.assertEqual(
            environment.prelift_stage_observation_count.tolist(),
            [[5, 5], [5, 5]],
        )
        self.assertEqual(
            environment.prelift_stage_qualified_count.tolist(),
            [[5, 5], [5, 4]],
        )
        self.assertEqual(
            environment.prelift_stage_max_contact_count.tolist(),
            [[3, 3], [4, 4]],
        )
        self.assertEqual(
            environment.prelift_tip_raw_pair_missing_count_by_stage[
                0, :, 3
            ].tolist(),
            [5, 0],
        )
        self.assertEqual(
            environment.prelift_tip_efc_inactive_count_by_stage[
                1, :, 2:
            ].tolist(),
            [[0, 0], [1, 1]],
        )
        self.assertEqual(
            environment.prelift_tip_force_failure_count_by_stage[
                0, :, 3
            ].tolist(),
            [0, 5],
        )
        self.assertEqual(
            environment.prelift_lift_boundary_tip_raw_pair.tolist(),
            [[True, True, True, True], [True, True, True, True]],
        )
        self.assertEqual(
            environment.prelift_lift_boundary_tip_active_pair.tolist(),
            [[True, True, True, True], [True, True, True, True]],
        )
        self.assertEqual(
            environment.prelift_lift_boundary_tip_contact_qualified.tolist(),
            [[True, True, True, False], [True, True, True, True]],
        )

    def test_denied_prelift_gate_keeps_lift_origin_stationary(self) -> None:
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.dtype = torch.float32
        environment.device = torch.device("cpu")
        environment.config = DEFAULT_CONFIG
        lift_start = (
            DEFAULT_CONFIG.plan_ticks
            + DEFAULT_CONFIG.close_ticks
            + DEFAULT_CONFIG.settle_ticks
        )
        environment.control_step = torch.tensor(
            [lift_start, lift_start], dtype=torch.long
        )
        environment.prelift_lift_permitted = torch.tensor([True, False])
        environment.prelift_lift_reference_valid = torch.tensor([True, True])
        environment.prelift_lift_start_body_pos = torch.tensor(
            [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]]
        )
        environment.initial_body_pos = torch.zeros((2, 3))
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(2, 1)
        environment.prelift_lift_start_quat = identity.clone()
        environment.initial_quat = identity.clone()
        environment.action_start_com = torch.zeros((2, 3))
        environment.action_start_quat = identity.clone()
        environment.goal_quat = identity.clone()
        environment.object_index = torch.zeros(2, dtype=torch.long)
        environment.task_index = torch.full(
            (2,), contract.TASKS.index(contract.TASK_ROTATE), dtype=torch.long
        )
        environment.action_reference_valid = torch.zeros(2, dtype=torch.bool)
        zeros3 = torch.zeros((2, 3))
        zeros1 = torch.zeros(2)
        false = torch.zeros(2, dtype=torch.bool)
        environment._translation_path_state = MethodType(
            lambda self: (
                zeros3,
                zeros3,
                zeros1,
                zeros1,
                zeros1,
                zeros1,
                zeros1,
                false,
            ),
            environment,
        )

        position, _, lift_fraction, _, active = (
            environment._desired_object_pose()
        )
        self.assertGreater(lift_fraction[0].item(), 0.0)
        self.assertEqual(lift_fraction[1].item(), 0.0)
        torch.testing.assert_close(
            position[1], environment.prelift_lift_start_body_pos[1]
        )
        self.assertEqual(active.tolist(), [False, False])

    def test_entry_and_terminal_require_25_consecutive_ticks(self) -> None:
        class Backend:
            views = {
                "xipos": torch.zeros((1, 1, 3)),
                "xquat": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            }

            def tensor(self, name: str):
                return self.views[name]

        environment = object.__new__(MJWarpIKResidualEnv)
        environment.backend = Backend()
        environment.device = torch.device("cpu")
        environment.dtype = torch.float32
        environment.worlds = 1
        environment.config = DEFAULT_CONFIG
        environment._rows = torch.tensor([0])
        environment._object_body_ids = torch.zeros(4, dtype=torch.long)
        environment.object_index = torch.tensor([0])
        environment.task_index = torch.tensor(
            [contract.TASKS.index(contract.TASK_TRANSLATE)]
        )
        environment.action_reference_valid = torch.tensor([True])
        environment.action_start_com = torch.zeros((1, 3))
        environment.action_start_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]]
        )
        environment.path_waypoints_xy_m = torch.zeros((1, 21, 2))
        environment.object_dropped = torch.tensor([False])
        environment.translation_entry_stable_count = torch.zeros(
            1, dtype=torch.long
        )
        environment.translation_entry_max_stable_count = torch.zeros(
            1, dtype=torch.long
        )
        environment.translation_entry_certificate_latched = torch.tensor(
            [False]
        )
        environment.translation_entry_certificate_first_control_step = (
            torch.full((1,), -1, dtype=torch.long)
        )
        environment.translation_route_admitted = torch.tensor([False])
        environment.translation_entry_route_start_safety_valid = torch.tensor(
            [False]
        )
        environment.translation_entry_gate_current = torch.tensor([False])
        environment.translation_entry_tip_height_valid = torch.tensor([False])
        environment.translation_entry_lifted = torch.tensor([False])
        environment.translation_entry_pose_valid = torch.tensor([False])
        environment.translation_entry_start_position_valid = torch.tensor(
            [False]
        )
        environment.translation_entry_start_orientation_valid = torch.tensor(
            [False]
        )
        environment.translation_entry_linear_speed_valid = torch.tensor(
            [False]
        )
        environment.translation_entry_angular_speed_valid = torch.tensor(
            [False]
        )
        environment.translation_entry_predicate_current = torch.zeros(
            (1, 9), dtype=torch.bool
        )
        environment.translation_entry_failure_count_by_predicate = torch.zeros(
            (1, 9), dtype=torch.long
        )
        environment.translation_entry_first_failure_control_step = torch.full(
            (1, 9), -1, dtype=torch.long
        )
        environment.translation_entry_last_failure_control_step = torch.full(
            (1, 9), -1, dtype=torch.long
        )
        environment.translation_entry_tip_height_fraction = torch.zeros((1, 4))
        environment.translation_entry_tip_object_pair = torch.zeros(
            (1, 4), dtype=torch.bool
        )
        environment.translation_entry_tip_object_raw_pair = torch.zeros(
            (1, 4), dtype=torch.bool
        )
        environment.translation_entry_tip_object_active_pair = torch.zeros(
            (1, 4), dtype=torch.bool
        )
        environment.translation_entry_tip_contact_qualified = torch.zeros(
            (1, 4), dtype=torch.bool
        )
        environment.translation_entry_tip_contact_force_n = torch.zeros((1, 4))
        environment.translation_entry_tip_force_valid = torch.zeros(
            (1, 4), dtype=torch.bool
        )
        environment.translation_entry_tip_height_failure_count = torch.zeros(
            (1, 4), dtype=torch.long
        )
        environment.translation_entry_tip_pair_missing_count = torch.zeros(
            (1, 4), dtype=torch.long
        )
        environment.translation_entry_tip_raw_pair_missing_count = torch.zeros(
            (1, 4), dtype=torch.long
        )
        environment.translation_entry_tip_efc_inactive_count = torch.zeros(
            (1, 4), dtype=torch.long
        )
        environment.translation_entry_tip_force_failure_count = torch.zeros(
            (1, 4), dtype=torch.long
        )
        environment.translation_entry_position_error_xyz_abs_m = torch.zeros(
            (1, 3)
        )
        environment.translation_entry_euler_error_xyz_abs_deg = torch.zeros(
            (1, 3)
        )
        environment.translation_entry_estimated_bottom_z_m = torch.zeros(1)
        environment.translation_entry_object_linear_speed_m_s = torch.zeros(1)
        environment.translation_entry_object_angular_speed_rad_s = torch.zeros(1)
        environment._last_force_decode_finite = torch.tensor([True])
        environment._last_tip_object_contact_raw_pair = torch.tensor(
            [[True, True, False, True]], dtype=torch.bool
        )
        environment._last_tip_object_contact_pair = torch.tensor(
            [[True, False, False, True]], dtype=torch.bool
        )
        environment._last_tip_object_contact_force_qualified_pair = torch.tensor(
            [[True, False, False, False]], dtype=torch.bool
        )
        environment.expert_tip_object_contact = torch.zeros(
            (1, 4), dtype=torch.bool
        )
        environment.expert_contact_count = torch.zeros(1, dtype=torch.long)
        environment.expert_four_tip_contact = torch.tensor([False])
        environment.expert_object_linear_speed_m_s = torch.zeros(1)
        environment.expert_object_angular_speed_rad_s = torch.zeros(1)
        stable = [True]

        def metrics(self, force):
            passed = torch.tensor(stable)
            tip_contact = torch.tensor(
                [[True, True, True, stable[0]]], dtype=torch.bool
            )
            count = tip_contact.to(torch.long).sum(dim=1)
            return (
                passed,
                tip_contact,
                count,
                count == 4,
                passed,
                passed,
                passed,
                torch.tensor([0.5 * EXPERT_OBJECT_LINEAR_SPEED_MAX_M_S]),
                torch.tensor([0.5 * EXPERT_OBJECT_ANGULAR_SPEED_MAX_RAD_S]),
                torch.full((1, 4), 0.5),
                torch.tensor([contract.TABLE_TOP_Z_M + 0.010]),
            )

        environment._expert_physical_metrics = MethodType(metrics, environment)
        environment._grasp_plane_contract_rows = MethodType(
            lambda self: (
                torch.tensor([0.5]),
                torch.tensor([0.4]),
                torch.tensor([0.6]),
                torch.tensor([True]),
            ),
            environment,
        )
        action_start = sum(
            (
                DEFAULT_CONFIG.plan_ticks,
                DEFAULT_CONFIG.close_ticks,
                DEFAULT_CONFIG.settle_ticks,
                DEFAULT_CONFIG.lift_ticks,
                DEFAULT_CONFIG.hold_ticks,
            )
        )
        route_start = (
            action_start
            + contract.TRANSLATION_PATH_LEAD_IN_TICKS
            + contract.TRANSLATION_PATH_START_SETTLE_TICKS
        )
        for tick in range(
            route_start - EXPERT_STABLE_TICKS - 4, route_start - 4
        ):
            environment.control_step = torch.tensor([tick])
            environment._update_translation_entry_gate(torch.ones((1, 4)))
        self.assertTrue(environment.translation_entry_certificate_latched.item())
        self.assertFalse(environment.translation_route_admitted.item())
        stable[0] = False
        for tick in range(route_start - 4, route_start):
            environment.control_step = torch.tensor([tick])
            environment._update_translation_entry_gate(torch.ones((1, 4)))
        stable[0] = True
        environment.control_step = torch.tensor([route_start])
        environment._update_translation_entry_gate(torch.ones((1, 4)))
        self.assertEqual(
            environment.translation_entry_stable_count.item(),
            1,
        )
        self.assertEqual(
            environment.translation_entry_max_stable_count.item(),
            EXPERT_STABLE_TICKS,
        )
        self.assertFalse(environment.translation_route_admitted.item())

        # A fresh uninterrupted run of exactly 25 ticks ending at route_start
        # is admitted.  Historical latch state alone is never sufficient.
        environment.translation_entry_stable_count.zero_()
        environment.translation_entry_max_stable_count.zero_()
        environment.translation_entry_certificate_latched.fill_(False)
        environment.translation_entry_certificate_first_control_step.fill_(-1)
        environment.translation_route_admitted.fill_(False)
        for tick in range(route_start - EXPERT_STABLE_TICKS + 1, route_start + 1):
            environment.control_step = torch.tensor([tick])
            environment._update_translation_entry_gate(torch.ones((1, 4)))
        self.assertEqual(
            environment.translation_entry_stable_count.item(),
            EXPERT_STABLE_TICKS,
        )
        self.assertTrue(environment.translation_route_admitted.item())
        self.assertEqual(
            environment.translation_entry_tip_object_raw_pair.tolist(),
            [[True, True, False, True]],
        )
        self.assertEqual(
            environment.translation_entry_tip_object_active_pair.tolist(),
            [[True, False, False, True]],
        )
        self.assertEqual(
            environment.translation_entry_tip_contact_qualified.tolist(),
            [[True, False, False, False]],
        )
        self.assertGreater(
            environment.translation_entry_tip_raw_pair_missing_count[
                0, 2
            ].item(),
            0,
        )
        self.assertGreater(
            environment.translation_entry_tip_efc_inactive_count[0, 1].item(),
            0,
        )
        self.assertGreater(
            environment.translation_entry_tip_force_failure_count[0, 3].item(),
            0,
        )

        environment.task_index.fill_(contract.TASKS.index(contract.TASK_ROTATE))
        environment.path_tracking_sample_count = torch.zeros(1, dtype=torch.long)
        environment.path_tracking_axis_abs_max_m = torch.zeros((1, 3))
        environment.path_schedule_complete = torch.tensor([False])
        environment.path_schedule_limit_valid = torch.tensor([False])
        environment.terminal_stable_count = torch.zeros(1, dtype=torch.long)
        environment.terminal_stable_certified = torch.tensor([False])
        environment.terminal_stability_gate_current = torch.tensor([False])
        for _ in range(EXPERT_STABLE_TICKS):
            certified = environment._update_terminal_stability_gate(
                pose_success=torch.tensor([True]),
                actual_plane_compliant=torch.tensor([True]),
                finger_force=torch.ones((1, 4)),
            )
        self.assertTrue(certified.item())
        stable[0] = False
        certified = environment._update_terminal_stability_gate(
            pose_success=torch.tensor([True]),
            actual_plane_compliant=torch.tensor([True]),
            finger_force=torch.ones((1, 4)),
        )
        self.assertFalse(certified.item())
        self.assertEqual(environment.terminal_stable_count.item(), 0)

    def test_trust_stage_boundaries(self) -> None:
        environment = object.__new__(MJWarpIKResidualEnv)
        environment.config = DEFAULT_CONFIG
        lift_start = (
            DEFAULT_CONFIG.plan_ticks
            + DEFAULT_CONFIG.close_ticks
            + DEFAULT_CONFIG.settle_ticks
        )
        action_start = lift_start + DEFAULT_CONFIG.lift_ticks + DEFAULT_CONFIG.hold_ticks
        route_start = (
            action_start
            + contract.TRANSLATION_PATH_LEAD_IN_TICKS
            + contract.TRANSLATION_PATH_START_SETTLE_TICKS
        )
        environment.control_step = torch.tensor(
            [0, lift_start, action_start, route_start]
        )
        environment.task_index = torch.full(
            (4,), contract.TASKS.index(contract.TASK_TRANSLATE)
        )
        self.assertEqual(
            environment._ik_trust_stage_index().tolist(),
            [
                IK_TRUST_STAGES.index("grasp"),
                IK_TRUST_STAGES.index("lift"),
                IK_TRUST_STAGES.index("path_start"),
                IK_TRUST_STAGES.index("route"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
