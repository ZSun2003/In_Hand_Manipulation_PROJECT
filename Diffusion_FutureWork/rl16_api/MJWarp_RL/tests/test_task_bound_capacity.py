from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
import unittest

from MJWarp_RL import contract
import MJWarp_RL.capacity_scan as capacity
import MJWarp_RL.train as training
from MJWarp_RL.hpc_config import DEFAULT_CONFIG


def _certified_coverage(
    task: str,
    *,
    world_counts: tuple[int, ...],
    phase: str = "all",
) -> dict[str, object]:
    selected = contract.validate_task(task)
    population = DEFAULT_CONFIG.elite.population
    group_counts = tuple(count // population for count in world_counts)
    formal = sum(world_counts) == contract.FORMAL_GLOBAL_WORLDS
    templates = (
        list(range(contract.RESET_TEMPLATES_PER_OBJECT)) if formal else [0]
    )
    shapes = (
        list(range(len(contract.TRANSLATION_PATH_SHAPES))) if formal else [0]
    )
    rotation_maximum = contract.ROTATION_ANGLE_MAX_DEG
    return {
        "passed": True,
        "phase": phase,
        "task": selected,
        "task_index": contract.TASKS.index(selected),
        "population": population,
        "groups": sum(group_counts),
        "formal_coverage_required": formal,
        "selection_rule": "first_valid_replica_no_reward",
        "valid_definition": (
            "success & path_tracking_certified & "
            "actual_grasp_plane_compliant & !object_dropped"
        ),
        "valid_world_count": sum(group_counts),
        "valid_group_count": sum(group_counts),
        "by_object": {
            name: {
                "group_count": groups,
                "valid_group_count": groups,
                "valid_template_indices": templates,
                "valid_template_count": len(templates),
                "all_reset_templates_covered": formal,
                "translation_path_shapes": (
                    {
                        "active": True,
                        "valid_shape_indices": shapes,
                        "all_shapes_covered": formal,
                    }
                    if selected == contract.TASK_TRANSLATE
                    else {
                        "active": False,
                        "reason": "inactive_for_task",
                        "valid_shape_indices": [],
                        "all_shapes_covered": False,
                    }
                ),
                "rotation_maximum_angle": (
                    {
                        "active": True,
                        "maximum_angle_deg": rotation_maximum,
                        "valid_group_count": 1 if formal else 0,
                        "covered": formal,
                    }
                    if selected == contract.TASK_ROTATE
                    else {
                        "active": False,
                        "reason": "inactive_for_task",
                        "maximum_angle_deg": None,
                        "valid_group_count": 0,
                        "covered": False,
                    }
                ),
            }
            for name, groups in zip(contract.OBJECTS, group_counts)
        },
    }


def _prelift_diagnostics(torch: object) -> dict[str, object]:
    raw_pair_missing = torch.zeros((4, 2, 4), dtype=torch.long)
    raw_pair_missing[1, 0, 3] = 5
    efc_inactive = torch.zeros((4, 2, 4), dtype=torch.long)
    efc_inactive[1, 0, 3] = 7
    return {
        "prelift_contact_stage_names": ("close", "settle"),
        "prelift_contact_required_stable_ticks": torch.full(
            (4,), 5, dtype=torch.long
        ),
        "prelift_contact_stage_index": torch.ones(4, dtype=torch.long),
        "prelift_contact_stable_count": torch.tensor(
            (5, 0, 5, 0), dtype=torch.long
        ),
        "prelift_contact_max_stable_count": torch.tensor(
            (5, 0, 5, 0), dtype=torch.long
        ),
        "prelift_contact_gate_latched": torch.tensor(
            (True, False, True, False)
        ),
        "prelift_contact_gate_first_control_step": torch.tensor(
            (120, -1, 130, -1), dtype=torch.long
        ),
        "prelift_contact_count_current": torch.tensor(
            (4, 2, 4, 0), dtype=torch.long
        ),
        "prelift_tip_raw_pair_current": torch.tensor(
            (
                (True,) * 4,
                (True,) * 4,
                (True,) * 4,
                (True, False, False, False),
            )
        ),
        "prelift_tip_active_pair_current": torch.tensor(
            (
                (True,) * 4,
                (True, True, False, False),
                (True,) * 4,
                (False,) * 4,
            )
        ),
        "prelift_tip_contact_qualified_current": torch.tensor(
            (
                (True,) * 4,
                (True, True, False, False),
                (True,) * 4,
                (False,) * 4,
            )
        ),
        "prelift_stage_observation_count": torch.tensor(
            ((90, 60),) * 4, dtype=torch.long
        ),
        "prelift_stage_qualified_count": torch.tensor(
            ((20, 60), (0, 0), (30, 60), (0, 0)), dtype=torch.long
        ),
        "prelift_stage_failure_count": torch.tensor(
            ((70, 0), (90, 60), (60, 0), (90, 60)), dtype=torch.long
        ),
        "prelift_stage_max_contact_count": torch.tensor(
            ((4, 4), (2, 2), (4, 4), (0, 0)), dtype=torch.long
        ),
        "prelift_stage_first_failure_control_step": torch.tensor(
            ((9, -1), (9, 99), (9, -1), (9, 99)), dtype=torch.long
        ),
        "prelift_stage_last_failure_control_step": torch.tensor(
            ((78, -1), (98, 158), (68, -1), (98, 158)), dtype=torch.long
        ),
        "prelift_tip_pair_missing_count_by_stage": (
            raw_pair_missing + efc_inactive
        ),
        "prelift_tip_raw_pair_missing_count_by_stage": raw_pair_missing,
        "prelift_tip_efc_inactive_count_by_stage": efc_inactive,
        "prelift_tip_force_failure_count_by_stage": torch.zeros(
            (4, 2, 4), dtype=torch.long
        ),
        "prelift_lift_boundary_contact_valid": torch.tensor(
            (True, False, True, False)
        ),
        "prelift_lift_boundary_tip_raw_pair": torch.tensor(
            (
                (True,) * 4,
                (True,) * 4,
                (True,) * 4,
                (True, False, False, False),
            )
        ),
        "prelift_lift_boundary_tip_active_pair": torch.tensor(
            (
                (True,) * 4,
                (True, True, False, False),
                (True,) * 4,
                (False,) * 4,
            )
        ),
        "prelift_lift_boundary_tip_contact_qualified": torch.tensor(
            (
                (True,) * 4,
                (True, True, False, False),
                (True,) * 4,
                (False,) * 4,
            )
        ),
        "prelift_lift_boundary_contact_force_n": torch.tensor(
            (
                (0.1,) * 4,
                (0.1, 0.1, 0.0, 0.0),
                (0.2,) * 4,
                (0.0,) * 4,
            ),
            dtype=torch.float64,
        ),
        "prelift_lift_permitted": torch.tensor(
            (True, False, True, False)
        ),
        "prelift_lift_reference_valid": torch.tensor((True,) * 4),
        "prelift_lift_start_body_position": torch.zeros((4, 3)),
        "prelift_lift_start_quaternion": torch.tensor(
            ((1.0, 0.0, 0.0, 0.0),) * 4
        ),
    }


def _translation_entry_contact_layers(torch: object) -> dict[str, object]:
    return {
        "translation_entry_tip_object_raw_pair": torch.tensor(
            (
                (True,) * 4,
                (True,) * 4,
                (True,) * 4,
                (True, False, False, False),
            )
        ),
        "translation_entry_tip_object_active_pair": torch.tensor(
            (
                (True,) * 4,
                (True, True, False, False),
                (True,) * 4,
                (False,) * 4,
            )
        ),
        "translation_entry_tip_contact_qualified": torch.tensor(
            (
                (True,) * 4,
                (True, True, False, False),
                (True,) * 4,
                (False,) * 4,
            )
        ),
        "translation_entry_tip_raw_pair_missing_count": torch.tensor(
            ((0,) * 4, (0,) * 4, (0,) * 4, (0, 20, 20, 20)),
            dtype=torch.long,
        ),
        "translation_entry_tip_efc_inactive_count": torch.tensor(
            ((0,) * 4, (0, 0, 10, 10), (0,) * 4, (20, 0, 0, 0)),
            dtype=torch.long,
        ),
    }


class TaskBoundCapacityTests(unittest.TestCase):
    def test_certification_funnel_is_json_safe_and_exposes_failure_axes(
        self,
    ) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        config = replace(
            DEFAULT_CONFIG, num_envs=4, task=contract.TASK_TRANSLATE
        )
        info = {
            "success": torch.tensor((True, True, False, True)),
            "path_tracking_certified": torch.tensor(
                (True, False, True, True)
            ),
            "object_dropped": torch.tensor((False, False, False, True)),
            "object_index": torch.arange(4, dtype=torch.long),
            "path_shape_index": torch.tensor((0, 0, 1, 1)),
            "final_position_error_xyz_m": torch.tensor(
                (
                    (0.001, 0.002, 0.003),
                    (0.004, 0.005, 0.006),
                    (0.007, 0.008, 0.009),
                    (0.010, 0.011, 0.012),
                ),
                dtype=torch.float64,
            ),
            "final_euler_error_xyz_deg": torch.arange(
                12, dtype=torch.float64
            ).reshape(4, 3),
            "velocity": torch.tensor(
                (
                    (0.003, 0.004, 0.0, 0.0, 0.0, 0.2),
                    (0.0, 0.0, 0.0, 0.3, 0.4, 0.0),
                    (0.006, 0.008, 0.0, 0.0, 0.0, 0.1),
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                ),
                dtype=torch.float64,
            ),
            "path_tracking_axis_abs_max_m": torch.tensor(
                (
                    (0.001, 0.002, 0.003),
                    (0.020, 0.001, 0.001),
                    (0.002, 0.002, 0.002),
                    (0.003, 0.003, 0.003),
                ),
                dtype=torch.float64,
            ),
            "path_tracking_rmse_m": torch.tensor(
                (0.001, 0.015, 0.002, 0.003), dtype=torch.float64
            ),
            "path_tracking_sample_count": torch.full(
                (4,),
                contract.TRANSLATION_PATH_ROUTE_SAMPLES,
                dtype=torch.long,
            ),
            "path_schedule_progress": torch.ones(4, dtype=torch.float64),
            "path_schedule_limit_valid": torch.tensor(
                (True, False, True, True)
            ),
            "path_schedule_complete": torch.tensor(
                (True, False, True, True)
            ),
            "path_schedule_max_speed_m_s": torch.tensor(
                (0.030, 0.050, 0.025, 0.030), dtype=torch.float64
            ),
            "path_schedule_max_acceleration_m_s2": torch.tensor(
                (0.150, 0.250, 0.120, 0.150), dtype=torch.float64
            ),
            "path_tracking_first_violation_sample_tick": torch.tensor(
                (-1, 5, -1, -1), dtype=torch.long
            ),
            "path_tracking_first_violation_control_tick": torch.tensor(
                (-1, 903, -1, -1), dtype=torch.long
            ),
            "path_tracking_first_violation_axis_index": torch.tensor(
                (-1, 0, -1, -1), dtype=torch.long
            ),
            "path_tracking_first_violation_error_xyz_abs_m": torch.tensor(
                (
                    (0.0, 0.0, 0.0),
                    (0.020, 0.001, 0.001),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                ),
                dtype=torch.float64,
            ),
            "translation_entry_predicate_names": (
                "action_reference_valid",
                "object_not_dropped",
                "tip_height_valid",
                "minimum_tip_object_contacts",
                "linear_speed_valid",
                "angular_speed_valid",
                "object_lifted",
                "start_position_valid",
                "start_orientation_valid",
            ),
            "translation_entry_certificate_latched": torch.tensor(
                (True, False, True, False)
            ),
            "translation_entry_certificate_first_control_step": torch.tensor(
                (850, -1, 860, -1), dtype=torch.long
            ),
            "translation_entry_max_stable_count": torch.tensor(
                (25, 4, 25, 0), dtype=torch.long
            ),
            "translation_route_admitted": torch.tensor(
                (True, False, True, False)
            ),
            "translation_entry_route_start_safety_valid": torch.tensor(
                (True, False, True, False)
            ),
            "translation_entry_predicate_current": torch.tensor(
                (
                    (True,) * 9,
                    (True, True, True, False, True, True, True, True, True),
                    (True,) * 9,
                    (True, False, False, False, True, True, False, True, True),
                )
            ),
            "translation_entry_failure_count_by_predicate": torch.tensor(
                (
                    (0,) * 9,
                    (0, 0, 0, 10, 0, 0, 0, 0, 0),
                    (0,) * 9,
                    (0, 4, 3, 5, 0, 0, 2, 0, 0),
                ),
                dtype=torch.long,
            ),
            "translation_entry_first_failure_control_step": torch.tensor(
                ((-1,) * 9, (-1, -1, -1, 800, -1, -1, -1, -1, -1),
                 (-1,) * 9, (-1, 790, 791, 792, -1, -1, 793, -1, -1)),
                dtype=torch.long,
            ),
            "translation_entry_last_failure_control_step": torch.tensor(
                ((-1,) * 9, (-1, -1, -1, 809, -1, -1, -1, -1, -1),
                 (-1,) * 9, (-1, 793, 793, 796, -1, -1, 794, -1, -1)),
                dtype=torch.long,
            ),
            "translation_entry_tip_height_fraction": torch.full((4, 4), 0.5),
            "translation_entry_tip_object_pair": torch.tensor(
                ((True,) * 4, (True, True, False, False),
                 (True,) * 4, (False,) * 4)
            ),
            "translation_entry_tip_contact_force_n": torch.tensor(
                ((0.1,) * 4, (0.1, 0.1, 0.0, 0.0),
                 (0.2,) * 4, (0.0,) * 4), dtype=torch.float64
            ),
            "translation_entry_tip_force_valid": torch.tensor(
                ((True,) * 4, (True, True, False, False),
                 (True,) * 4, (False,) * 4)
            ),
            "translation_entry_tip_height_failure_count": torch.zeros(
                (4, 4), dtype=torch.long
            ),
            "translation_entry_tip_pair_missing_count": torch.tensor(
                ((0,) * 4, (0, 0, 10, 10), (0,) * 4, (20,) * 4),
                dtype=torch.long,
            ),
            "translation_entry_tip_force_failure_count": torch.zeros(
                (4, 4), dtype=torch.long
            ),
            "translation_entry_position_error_xyz_abs_m": torch.zeros((4, 3)),
            "translation_entry_euler_error_xyz_abs_deg": torch.zeros((4, 3)),
            "translation_entry_estimated_bottom_z_m": torch.full(
                (4,), contract.TABLE_TOP_Z_M + 0.01
            ),
            "translation_entry_object_linear_speed_m_s": torch.tensor(
                (0.001, 0.006, 0.002, 0.0)
            ),
            "translation_entry_object_angular_speed_rad_s": torch.tensor(
                (0.01, 0.06, 0.02, 0.0)
            ),
            "ik_trust_stage_names": ("grasp", "lift", "path_start", "route"),
            "ik_trust_region_projection_count_by_stage": torch.tensor(
                ((1, 2, 3, 4), (0, 0, 1, 1), (2, 2, 2, 2), (0, 0, 0, 0)),
                dtype=torch.long,
            ),
            "ik_cage_projection_count_by_stage": torch.tensor(
                ((1, 1, 1, 1), (1, 0, 0, 0), (2, 2, 2, 2), (0, 0, 0, 0)),
                dtype=torch.long,
            ),
            "ik_cage_projection_scale_min": torch.tensor((0.5, 0.8, 0.4, 1.0)),
            "ik_cage_projection_scale_mean": torch.tensor((0.9, 0.95, 0.8, 1.0)),
        }
        funnel = capacity._certification_funnel(
            torch,
            config=config,
            info=info,
            actual_compliance=torch.tensor((True, True, True, False)),
            completed_episode_controller_counters={
                "ik_solve_fallback_count": 0,
                "ik_trust_region_projection_count": 11,
                "position_feedback_slew_clip_count": 12,
                "preload_trim_slew_clip_count": 13,
                "ctrl_slew_clip_count": 14,
                "ctrl_certificate_fallback_count": 0,
            },
        )
        json.dumps(funnel, sort_keys=True, allow_nan=False)
        self.assertEqual(funnel["overall"]["world_count"], 4)
        self.assertEqual(funnel["overall"]["endpoint_success_count"], 3)
        self.assertEqual(funnel["overall"]["route_certified_count"], 3)
        self.assertEqual(funnel["overall"]["plane_compliant_count"], 3)
        self.assertEqual(funnel["overall"]["not_dropped_count"], 3)
        self.assertEqual(funnel["overall"]["valid_count"], 1)
        self.assertEqual(funnel["by_path"]["square_100mm"]["valid_count"], 1)
        self.assertEqual(funnel["by_path"]["circle_r50mm"]["valid_count"], 0)
        self.assertAlmostEqual(
            funnel["overall"]["path_tracking_axis_abs_max_m"]["linf"][
                "max"
            ],
            0.020,
        )
        self.assertAlmostEqual(
            funnel["overall"]["endpoint_object_linear_speed_m_s"]["max"],
            0.010,
        )
        self.assertAlmostEqual(
            funnel["overall"]["endpoint_object_angular_speed_rad_s"]["max"],
            0.500,
        )
        first = funnel["overall"]["first_route_violation"]
        self.assertEqual(first["count"], 1)
        self.assertEqual(first["sample_tick"]["p50"], 5.0)
        self.assertEqual(first["control_tick"]["p50"], 903.0)
        self.assertEqual(first["axis_counts"], {"x": 1, "y": 0, "z": 0})
        self.assertAlmostEqual(first["error_xyz_abs_m"]["x"]["max"], 0.020)
        schedule = funnel["overall"]["path_schedule"]
        self.assertEqual(schedule["expected_sample_count"], 1_800)
        self.assertEqual(schedule["exact_sample_count_worlds"], 4)
        self.assertEqual(schedule["complete_worlds"], 3)
        self.assertEqual(schedule["limit_valid_worlds"], 3)
        self.assertEqual(schedule["progress"]["p50"], 1.0)
        self.assertAlmostEqual(schedule["peak_speed_m_s"]["max"], 0.050)
        self.assertAlmostEqual(
            schedule["peak_acceleration_m_s2"]["max"], 0.250
        )
        self.assertEqual(
            funnel["path_schedule_route_first_control_tick"], 899
        )
        self.assertEqual(
            funnel["by_path"]["square_100mm"]["first_route_violation"][
                "count"
            ],
            1,
        )
        self.assertEqual(
            funnel["by_path"]["circle_r50mm"]["first_route_violation"][
                "count"
            ],
            0,
        )
        self.assertEqual(
            funnel["completed_episode_controller_counters"][
                "position_feedback_slew_clip_count"
            ],
            12,
        )
        entry = funnel["overall"]["translation_entry"]
        self.assertEqual(entry["certificate_latched_count"], 2)
        self.assertEqual(entry["route_admitted_count"], 2)
        self.assertEqual(
            entry["predicates"]["minimum_tip_object_contacts"][
                "failed_world_count"
            ],
            2,
        )
        self.assertEqual(
            entry["qualified_contact_count_distribution"],
            {"0": 1, "1": 0, "2": 1, "3": 0, "4": 2},
        )
        self.assertFalse(
            funnel["overall"]["prelift_contact_gate"]["available"]
        )
        self.assertFalse(
            funnel["by_object"]["orange"]["prelift_contact_gate"][
                "available"
            ]
        )
        self.assertFalse(entry["contact_layers"]["available"])

        bad_tick = dict(info)
        bad_tick["path_tracking_first_violation_control_tick"] = torch.tensor(
            (-1, 902, -1, -1), dtype=torch.long
        )
        with self.assertRaisesRegex(ValueError, "first-route-violation"):
            capacity._certification_funnel(
                torch,
                config=config,
                info=bad_tick,
                actual_compliance=torch.tensor((True, True, True, False)),
            )

        bad_sample = dict(info)
        bad_sample["path_tracking_first_violation_sample_tick"] = torch.tensor(
            (-1, 601, -1, -1), dtype=torch.long
        )
        bad_sample["path_tracking_first_violation_control_tick"] = torch.tensor(
            (-1, 1001, -1, -1), dtype=torch.long
        )
        with self.assertRaisesRegex(ValueError, "first-route-violation"):
            capacity._certification_funnel(
                torch,
                config=config,
                info=bad_sample,
                actual_compliance=torch.tensor((True, True, True, False)),
            )

        partial_prelift = dict(info)
        partial_prelift["prelift_contact_stage_names"] = ("close", "settle")
        with self.assertRaisesRegex(ValueError, "schema is incomplete"):
            capacity._certification_funnel(
                torch,
                config=config,
                info=partial_prelift,
                actual_compliance=torch.tensor((True, True, True, False)),
            )

        partial_layers = dict(info)
        partial_layers["translation_entry_tip_object_raw_pair"] = torch.ones(
            (4, 4), dtype=torch.bool
        )
        with self.assertRaisesRegex(ValueError, "contact-layer schema"):
            capacity._certification_funnel(
                torch,
                config=config,
                info=partial_layers,
                actual_compliance=torch.tensor((True, True, True, False)),
            )

        complete_prelift = dict(info)
        complete_prelift.update(_prelift_diagnostics(torch))
        complete_prelift.update(_translation_entry_contact_layers(torch))
        with_prelift = capacity._certification_funnel(
            torch,
            config=config,
            info=complete_prelift,
            actual_compliance=torch.tensor((True, True, True, False)),
        )
        json.dumps(with_prelift, sort_keys=True, allow_nan=False)
        prelift = with_prelift["overall"]["prelift_contact_gate"]
        self.assertTrue(prelift["available"])
        self.assertEqual(prelift["gate_latched_count"], 2)
        self.assertEqual(prelift["lift_boundary_contact_valid_count"], 2)
        self.assertEqual(prelift["lift_permitted_count"], 2)
        self.assertEqual(prelift["lift_reference_valid_count"], 4)
        self.assertEqual(
            prelift["boundary_contact_count_distribution"],
            {"0": 1, "1": 0, "2": 1, "3": 0, "4": 2},
        )
        self.assertEqual(
            prelift["stages"]["close"]["observation_tick_total"], 360
        )
        self.assertEqual(
            prelift["stages"]["settle"]["qualified_tick_total"], 120
        )
        self.assertEqual(prelift["boundary_tips"]["U"]["qualified_count"], 3)
        self.assertEqual(
            prelift["stages"]["close"]["tips"]["R"][
                "raw_pair_missing_tick_total"
            ],
            5,
        )
        self.assertEqual(
            prelift["stages"]["close"]["tips"]["R"][
                "efc_inactive_tick_total"
            ],
            7,
        )
        self.assertEqual(
            with_prelift["by_object"]["orange"]["prelift_contact_gate"][
                "lift_permitted_count"
            ],
            1,
        )
        layers = with_prelift["overall"]["translation_entry"][
            "contact_layers"
        ]
        self.assertTrue(layers["available"])
        self.assertEqual(
            layers["route_start_raw_pair_count_distribution"],
            {"0": 0, "1": 1, "2": 0, "3": 0, "4": 3},
        )
        self.assertEqual(
            layers["tips"]["R"]["efc_inactive_tick_total"], 10
        )

    def test_cli_requires_explicit_task(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                capacity.main([])
        self.assertEqual(raised.exception.code, 2)

    def test_task_specific_config_hashes_differ(self) -> None:
        device = DEFAULT_CONFIG.device
        self.assertNotEqual(
            capacity._scan_config_sha256(contract.TASK_TRANSLATE, device),
            capacity._scan_config_sha256(contract.TASK_ROTATE, device),
        )
        self.assertNotEqual(
            training._capacity_config_sha256(
                contract.TASK_TRANSLATE, device
            ),
            training._capacity_config_sha256(contract.TASK_ROTATE, device),
        )

    def test_inactive_task_coverage_is_structured_and_fail_closed(self) -> None:
        counts = contract.FORMAL_OBJECT_WORLD_COUNTS
        inactive_rotation = capacity._inactive_goal_coverage(
            coverage="rotation",
            task=contract.TASK_TRANSLATE,
            phase="all",
            world_counts=counts,
        )
        inactive_translation = capacity._inactive_goal_coverage(
            coverage="translation_path",
            task=contract.TASK_ROTATE,
            phase="all",
            world_counts=counts,
        )
        self.assertTrue(
            capacity._rotation_goal_coverage_is_valid(
                inactive_rotation,
                "all",
                task=contract.TASK_TRANSLATE,
                expected_world_counts=counts,
            )
        )
        self.assertTrue(
            capacity._translation_path_goal_coverage_is_valid(
                inactive_translation,
                "all",
                task=contract.TASK_ROTATE,
                expected_world_counts=counts,
            )
        )
        inactive_translation["reason"] = "tampered"
        self.assertFalse(
            capacity._translation_path_goal_coverage_is_valid(
                inactive_translation,
                "all",
                task=contract.TASK_ROTATE,
                expected_world_counts=counts,
            )
        )

    def test_certified_coverage_formal_and_diagnostic_semantics(self) -> None:
        for task in contract.TASKS:
            with self.subTest(task=task, profile="formal"):
                formal = _certified_coverage(
                    task,
                    world_counts=contract.FORMAL_OBJECT_WORLD_COUNTS,
                )
                self.assertTrue(
                    capacity._certified_candidate_coverage_is_valid(
                        formal,
                        task=task,
                        phase="all",
                        expected_world_counts=(
                            contract.FORMAL_OBJECT_WORLD_COUNTS
                        ),
                        population=DEFAULT_CONFIG.elite.population,
                    )
                )
                formal["by_object"][contract.OBJECTS[0]][
                    "valid_template_indices"
                ] = list(range(contract.RESET_TEMPLATES_PER_OBJECT - 1))
                self.assertFalse(
                    capacity._certified_candidate_coverage_is_valid(
                        formal,
                        task=task,
                        phase="all",
                        expected_world_counts=(
                            contract.FORMAL_OBJECT_WORLD_COUNTS
                        ),
                        population=DEFAULT_CONFIG.elite.population,
                    )
                )

            diagnostic_counts = (20, 20, 20, 20)
            with self.subTest(task=task, profile="diagnostic"):
                diagnostic = _certified_coverage(
                    task, world_counts=diagnostic_counts
                )
                self.assertTrue(
                    capacity._certified_candidate_coverage_is_valid(
                        diagnostic,
                        task=task,
                        phase="all",
                        expected_world_counts=diagnostic_counts,
                        population=DEFAULT_CONFIG.elite.population,
                    )
                )
                diagnostic["valid_group_count"] = 0
                self.assertFalse(
                    capacity._certified_candidate_coverage_is_valid(
                        diagnostic,
                        task=task,
                        phase="all",
                        expected_world_counts=diagnostic_counts,
                        population=DEFAULT_CONFIG.elite.population,
                    )
                )

    def test_cross_task_common_report_is_rejected(self) -> None:
        peer = {
            "schema": capacity.REPORT_SCHEMA,
            "is_common_report": False,
            "task": contract.TASK_ROTATE,
            "task_index": contract.TASKS.index(contract.TASK_ROTATE),
            "root_height_m": 0.525,
            "parent_model_fingerprint_sha256": "a" * 64,
            "parent_controller_fingerprint_sha256": "b" * 64,
            "platform_adapter_fingerprint_sha256": "c" * 64,
            "rl_model_fingerprint_sha256": "d" * 64,
        }
        with self.assertRaisesRegex(ValueError, "binding mismatch: task"):
            capacity._validate_common_report(
                peer,
                device=DEFAULT_CONFIG.device,
                gpu={},
                task=contract.TASK_TRANSLATE,
                config_canonical_sha256="e" * 64,
                candidate_config_set_sha256="f" * 64,
            )


if __name__ == "__main__":
    unittest.main()
