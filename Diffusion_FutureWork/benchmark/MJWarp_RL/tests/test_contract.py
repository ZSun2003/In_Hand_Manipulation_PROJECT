from __future__ import annotations

import math
import random
import unittest
from unittest import mock

from MJWarp_RL import contract, curriculum, success


class _BoundaryRandom:
    def __init__(
        self,
        *,
        high_radius: bool,
        high_theta: bool,
        mixture_value: float = 0.0,
    ) -> None:
        self.high_radius = high_radius
        self.high_theta = high_theta
        self.mixture_value = mixture_value

    def random(self) -> float:
        return self.mixture_value

    def getrandbits(self, k: int) -> int:
        if k != 53:
            raise AssertionError(k)
        return (1 << k) - 1 if self.high_radius else 0

    def randrange(self, start: int, stop: int | None = None) -> int:
        if stop is None:
            start, stop = 0, start
        assert stop is not None
        return stop - 1 if self.high_theta else start


class ContractTests(unittest.TestCase):
    def test_objects_tasks_and_task_geometry(self) -> None:
        self.assertEqual(
            contract.OBJECTS,
            ("orange", "banana", "pear", "strawberry"),
        )
        self.assertEqual(
            contract.INFEASIBLE_OBJECTS,
            frozenset(("cylinder", "cone")),
        )
        self.assertEqual(contract.TASKS, ("translate", "rotate"))
        self.assertNotIn("plate", contract.OBJECTS)
        self.assertEqual(contract.ROTATION_AXIS_LOCAL, (0.0, 0.0, 1.0))
        self.assertEqual(contract.ROTATION_ANGLE_DEG, 45.0)
        self.assertEqual(
            (
                contract.ROTATION_ANGLE_MIN_DEG,
                contract.ROTATION_ANGLE_MAX_DEG,
            ),
            (10, 45),
        )
        self.assertEqual(contract.TRANSLATION_DZ_M, 0.0)
        self.assertEqual(
            contract.ALGORITHM, "ActionChunkDiffusionIKImitation"
        )
        self.assertEqual(
            contract.CONTROLLERS,
            ("closed_loop_ik_expert", "ik_diffusion"),
        )
        self.assertEqual(contract.PHYSICS_BACKEND, "native-mujoco-warp")
        self.assertEqual(
            contract.ACTION_REFERENCE_CAPTURE,
            "post_lift_measured_com_and_commanded_cage",
        )
        self.assertTrue(contract.SUCCESS_GATE_ENABLED)
        self.assertEqual(contract.TOTAL_TRAINING_STEPS, 60_000_000)

    def test_reset_template_seed_bank_is_complete_and_collision_free(
        self,
    ) -> None:
        self.assertEqual(contract.RESET_TEMPLATES_PER_OBJECT, 32)
        seeds = [
            contract.reset_template_seed(object_index, template_index)
            for object_index in range(len(contract.OBJECTS))
            for template_index in range(
                contract.RESET_TEMPLATES_PER_OBJECT
            )
        ]
        self.assertEqual(len(seeds), 4 * 32)
        self.assertEqual(len(set(seeds)), len(seeds))
        payload = contract.algorithm_contract_payload()["reset_templates"]
        self.assertEqual(payload["count_per_object"], 32)
        self.assertEqual(payload["training_sampling"], "uniform")
        self.assertEqual(
            payload["paired_evaluation"],
            "unique_template_per_repeat",
        )
        with self.assertRaises(ValueError):
            contract.reset_template_seed(0, 32)

    def test_action_layout_is_exactly_25d_and_partitioned(self) -> None:
        self.assertEqual(contract.ACTION_DIM, 25)
        self.assertEqual(contract.TIP_HEIGHT_SLICE, slice(0, 4))
        self.assertEqual(contract.YAW_SLICE, slice(4, 5))
        self.assertEqual(contract.TIP_PRELOAD_SLICE, slice(5, 9))
        self.assertEqual(contract.TIP_XYZ_RESIDUAL_SLICE, slice(9, 21))
        self.assertEqual(contract.FORCE_SCALE_SLICE, slice(21, 25))
        self.assertEqual(
            contract.EPISODE_LEVEL_ACTION_INDICES,
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24),
        )
        self.assertEqual(contract.DYNAMIC_ACTION_INDICES, tuple(range(9, 21)))
        covered = [
            index
            for field in contract.ACTION_FIELDS
            for index in range(field.start, field.stop)
        ]
        self.assertEqual(covered, list(range(25)))
        self.assertEqual(
            [field.scope for field in contract.ACTION_FIELDS],
            ["episode", "episode", "episode", "control_tick", "episode"],
        )
        self.assertEqual(
            [field.applicability for field in contract.ACTION_FIELDS],
            ["fruit_only", "fruit_only", "all_objects", "all_objects", "all_objects"],
        )

    def test_action_limits_and_physical_decode(self) -> None:
        low = contract.decode_action(contract.ACTION_LOW, "banana")
        high = contract.decode_action(contract.ACTION_HIGH, "banana")
        self.assertEqual(
            low.tip_height_fraction,
            (contract.GRASP_PLANE_MIN_FRACTION,) * 4,
        )
        self.assertEqual(
            high.tip_height_fraction,
            (contract.GRASP_PLANE_MAX_FRACTION,) * 4,
        )
        self.assertAlmostEqual(low.yaw_rad, -math.pi / 2.0)
        self.assertAlmostEqual(high.yaw_rad, math.pi / 2.0)
        self.assertEqual(low.tip_preload_m, (-0.003,) * 4)
        self.assertEqual(high.tip_preload_m, (0.003,) * 4)
        self.assertTrue(
            all(value == -0.003 for tip in low.tip_xyz_residual_m for value in tip)
        )
        self.assertTrue(
            all(value == 0.003 for tip in high.tip_xyz_residual_m for value in tip)
        )
        self.assertEqual(low.force_scale, (0.5,) * 4)
        self.assertEqual(high.force_scale, (1.5,) * 4)

    def test_all_fruits_learn_plane_while_force_remains_per_tip(self) -> None:
        action = list(contract.ZERO_ACTION)
        action[contract.TIP_HEIGHT_SLICE] = (-1.0, -0.5, 0.5, 1.0)
        action[contract.FORCE_SCALE_SLICE] = (-1.0, -0.5, 0.5, 1.0)
        expected_heights = (0.40, 0.45, 0.55, 0.60)
        expected_force = (0.5, 0.75, 1.25, 1.5)
        for kind in contract.OBJECTS:
            with self.subTest(kind=kind):
                decoded = contract.decode_action(action, kind)
                for actual, expected in zip(
                    decoded.tip_height_fraction, expected_heights
                ):
                    self.assertAlmostEqual(actual, expected)
                self.assertEqual(decoded.yaw_rad, 0.0)
                self.assertEqual(decoded.force_scale, expected_force)
        self.assertFalse(
            contract.algorithm_contract_payload()["grasp_plane"][
                "hard_tip_xy_coplanarity_required"
            ]
        )

    def test_fixed_plane_mapping_helpers_and_mask_are_exact(self) -> None:
        self.assertEqual(
            contract.FIXED_GRASP_PLANE_FRACTIONS,
            {},
        )
        self.assertEqual(
            contract.LEARNED_GRASP_PLANE_OBJECTS,
            frozenset(contract.OBJECTS),
        )
        expected = (True, 0.5, (0.4, 0.6))
        for kind in contract.OBJECTS:
            with self.subTest(kind=kind):
                self.assertEqual(contract.grasp_plane_is_learned(kind), expected[0])
                self.assertEqual(
                    contract.grasp_plane_nominal_fraction(kind), expected[1]
                )
                self.assertEqual(
                    contract.grasp_plane_fraction_limits(kind), expected[2]
                )

        masked = contract.mask_action_for_object(contract.ACTION_HIGH, "orange")
        self.assertEqual(masked, contract.ACTION_HIGH)
        orange = contract.decode_action(contract.ACTION_HIGH, "orange")
        self.assertEqual(orange.tip_height_fraction, (0.6,) * 4)
        self.assertAlmostEqual(orange.yaw_rad, math.pi / 2.0)
        for infeasible in contract.INFEASIBLE_OBJECTS:
            with self.subTest(infeasible=infeasible):
                with self.assertRaises(ValueError):
                    contract.decode_action(contract.ZERO_ACTION, infeasible)
        with mock.patch.dict(
            contract.FIXED_GRASP_PLANE_FRACTIONS, {"orange": 0.40}
        ):
            with self.assertRaisesRegex(
                RuntimeError, "fixed and learned grasp-plane"
            ):
                contract.validate_contract()
        self.assertFalse(
            contract.algorithm_contract_payload()["grasp_plane"][
                "anti_support_gate"
            ]
        )

    def test_action_validation_and_episode_stability(self) -> None:
        self.assertEqual(contract.validate_action(contract.ZERO_ACTION), (0.0,) * 25)
        for invalid in (
            (0.0,) * 24,
            (0.0,) * 26,
            (0.0,) * 24 + (1.000001,),
            (0.0,) * 24 + (math.nan,),
        ):
            with self.assertRaises(ValueError):
                contract.validate_action(invalid)

        initial = list(contract.ZERO_ACTION)
        tick = list(contract.ZERO_ACTION)
        tick[9] = 1.0
        contract.validate_episode_action_stability(initial, tick, "banana")
        tick[4] = 0.1
        with self.assertRaises(ValueError):
            contract.validate_episode_action_stability(initial, tick, "banana")

        # Every active object is a fruit, so height, yaw, and preload are all
        # executable episode-scoped channels for every object.
        for kind in contract.OBJECTS:
            for index in (0, 4, 5):
                with self.subTest(kind=kind, index=index):
                    changed = list(contract.ZERO_ACTION)
                    changed[index] = 1.0
                    with self.assertRaises(ValueError):
                        contract.validate_episode_action_stability(
                            contract.ZERO_ACTION, changed, kind
                        )

    def test_algorithm_signature_is_stable_and_canonical(self) -> None:
        self.assertEqual(contract.ALGORITHM_SIGNATURE, contract.algorithm_signature())
        self.assertEqual(len(contract.ALGORITHM_SIGNATURE), 64)
        self.assertEqual(
            contract.ALGORITHM_SIGNATURE,
            "e0b847f36d6f7a60465eff66ef328d5c4834897b94a4ba8deb0fcf45e402d6f0",
        )
        payload = contract.algorithm_contract_payload()
        self.assertEqual(payload["schema"], "mjwarp-independent-hard-tip-contract-v21")
        bounded_schedule = payload["tasks"]["translation"]["bounded_schedule"]
        self.assertEqual(bounded_schedule["quantization_decimals"], 12)
        self.assertEqual(
            bounded_schedule["progress_table_hash_encoding"],
            "sha256_big_endian_signed_int64_grid_1e-12",
        )
        self.assertEqual(
            bounded_schedule["canonical_blob"],
            {
                "encoding": "base85_zlib_be_i64_delta_v1",
                "version": 1,
                "encoded_sha256": (
                    "c3de5f0eb864d63059b009313f56236ff720df9247b05d50391660f53f584ddb"
                ),
                "compressed_sha256": (
                    "e315bafef7df586aaba0a85fa6aac8e170f814628263629c92f80d3503bf475b"
                ),
                "decompressed_sha256": (
                    "eb7b18ffe69511b99a6de8d0faf854c2ecd7c3082639f3424e5c3d8955e1601e"
                ),
            },
        )
        self.assertEqual(
            payload["tasks"]["translation"]["waypoints_xy_sha256"],
            "a3573b55b5d8b233aaa746455f6099e3eb95b7e5c9aefc3427ded5638c24554c",
        )
        self.assertEqual(
            bounded_schedule["lead"]["offsets_xy_sha256"],
            "26efb26000be1dd9e786e05b2bb6ece3059b3c0e6a4511fad52a66b194fc1498",
        )
        self.assertEqual(
            bounded_schedule["route"]["targets_xy_sha256"],
            "30aae552dce3d3308a9fcfe572e8bd0e9592914abc8ed475f8482b1892999d29",
        )
        self.assertEqual(
            bounded_schedule["lead"]["kinematic_peaks_sha256"],
            "ea5216203faeb632e8661ad40819b787c1e847faeb55e92d1e6e3a905d6839c9",
        )
        self.assertEqual(
            bounded_schedule["route"]["kinematic_peaks_sha256"],
            "c872342019ee1a06a3bcd8cc9d0e44865c91f286d9b099ff448482d1836cc7f0",
        )
        for section in ("lead", "route"):
            for peaks in bounded_schedule[section]["kinematic_peaks_by_shape"]:
                for value in peaks:
                    self.assertEqual(
                        value,
                        float(round(value * 1_000_000_000_000) / 1_000_000_000_000),
                    )
        self.assertEqual(payload["training"]["algorithm"], contract.ALGORITHM)
        self.assertEqual(
            payload["training"]["diffusion_controls"],
            "independent_hard_tip_ik_commands",
        )
        self.assertEqual(payload["training"]["solver"], "Newton")
        self.assertEqual(
            payload["training"]["constraint_jacobian"], "dense"
        )
        self.assertEqual(
            payload["training"]["xml_solver_tolerance"], 1.0e-10
        )
        self.assertEqual(
            payload["training"]["effective_solver_tolerance"], 1.0e-6
        )
        self.assertEqual(payload["training"]["dense_newton_max_nv"], 60)
        self.assertEqual(
            payload["success"]["translation"],
            {
                "position_axis_abs_tolerance_m": 0.010,
                "euler_axis_abs_tolerance_deg": 20.0,
            },
        )
        self.assertEqual(
            payload["success"]["rotation"],
            {
                "position_axis_abs_tolerance_m": 0.010,
                "euler_axis_abs_tolerance_deg": 10.0,
            },
        )
        self.assertTrue(
            payload["success"]["l2_norms_are_diagnostic_only"]
        )
        self.assertEqual(
            payload["training"]["physics_layout"],
            "task_specific_four_single_object_xml_shards",
        )
        self.assertEqual(
            tuple(payload["training"]["shard_object_order"]),
            contract.OBJECTS,
        )
        self.assertEqual(payload["training"]["object_shard_count"], 4)
        self.assertEqual(
            payload["training"]["active_free_objects_per_shard"], 1
        )
        self.assertEqual(
            payload["training"]["cuda_graphs_per_global_batch"], 4
        )
        self.assertEqual(payload["training"]["formal_global_worlds"], 20_000)
        self.assertEqual(
            tuple(payload["training"]["formal_object_world_counts"]),
            (5_000,) * 4,
        )
        self.assertEqual(payload["training"]["formal_episode_ticks"], 3_000)
        self.assertEqual(payload["training"]["formal_pre_action_ticks"], 298)
        self.assertEqual(
            tuple(
                payload["training"][
                    "formal_remainder_object_world_counts"
                ]
            ),
            (0,) * 4,
        )
        self.assertEqual(
            payload["training"]["world_count_scope"],
            "global_sum_not_per_shard",
        )
        self.assertTrue(
            payload["training"]["rl_xml_excludes_untrained_plate"]
        )
        self.assertEqual(
            payload["objects"]["classified_infeasible"],
            ("cone", "cylinder"),
        )
        self.assertEqual(payload["action"]["dim"], 25)
        plane = payload["grasp_plane"]
        self.assertEqual(
            plane["learned_objects"], contract.OBJECTS
        )
        self.assertEqual(
            plane["fixed_nominal_square_height_fractions"],
            (),
        )
        self.assertEqual(plane["fixed_square_scope"], "nominal_contact_cage_only")
        self.assertTrue(plane["instantaneous_residual_deformation_allowed"])
        self.assertEqual(payload["task_distribution"]["phase_count"], 1)
        self.assertEqual(
            payload["task_distribution"]["rotation_angle_integer_deg"],
            (10, 45),
        )
        self.assertEqual(
            payload["training"]["expert_certification"],
            {
                "stable_ticks": 25,
                "prelift_contact_stable_ticks": 5,
                "prelift_boundary_policy": (
                    "consecutive_exact_force_qualified_contacts_through_"
                    "the_lift_boundary"
                ),
                "translation_entry_policy": (
                    "consecutive_certificate_must_end_at_route_start"
                ),
                "minimum_hard_tip_object_contacts": 3,
                "preferred_hard_tip_object_contacts": 4,
                "contact_identity": "native_contact_geom_pair_tip_and_active_object",
                "contact_direction_gate": False,
                "object_linear_speed_max_m_s": 0.005,
                "object_angular_speed_max_rad_s": 0.05,
                "object_table_clearance_m": 0.005,
                "tip_target_speed_max_m_s": 0.04,
                "tip_target_acceleration_max_m_s2": 0.15,
                "tip_target_jerk_max_m_s3": 1.0,
                "ik_cage_target_error_max_m": 0.003,
                "ik_cage_projection": "shared_scale_across_four_hard_tips",
                "trust_projection_stages": (
                    "grasp",
                    "lift",
                    "path_start",
                    "route",
                ),
            },
        )
        translation_path = payload["tasks"]["translation"]
        self.assertTrue(translation_path["phase_invariant"])
        self.assertEqual(
            tuple(translation_path["shape_names"]),
            contract.TRANSLATION_PATH_SHAPES,
        )
        self.assertEqual(translation_path["transforms"], "none")


class CurriculumTests(unittest.TestCase):
    def test_full_task_closed_seed_and_shape_boundaries(self) -> None:
        low = curriculum.sample_translation(
            curriculum.PHASE,
            _BoundaryRandom(high_radius=False, high_theta=False),
        )
        high = curriculum.sample_translation(
            curriculum.PHASE,
            _BoundaryRandom(high_radius=True, high_theta=True),
        )
        self.assertEqual(low.path_seed, contract.TRANSLATION_PATH_SEED_MIN)
        self.assertEqual(low.path_shape_index, 0)
        self.assertEqual(high.path_seed, contract.TRANSLATION_PATH_SEED_MAX)
        self.assertEqual(
            high.path_shape_index,
            len(contract.TRANSLATION_PATH_SHAPES) - 1,
        )
        self.assertEqual(low.path_step_m, 0.020)
        self.assertEqual(high.path_length_m, 0.200)
        self.assertEqual(low.delta_m[2], 0.0)
        self.assertEqual(high.delta_m[2], 0.0)

    def test_full_task_has_one_region_and_exact_path_length(self) -> None:
        main_low = curriculum.sample_translation(
            curriculum.PHASE,
            _BoundaryRandom(
                high_radius=False,
                high_theta=False,
                mixture_value=0.0,
            ),
        )
        main_high = curriculum.sample_translation(
            curriculum.PHASE,
            _BoundaryRandom(
                high_radius=True,
                high_theta=True,
                mixture_value=0.0,
            ),
        )
        self.assertEqual(main_low.source_region, curriculum.REGION_ALL)
        self.assertEqual(main_high.source_region, curriculum.REGION_ALL)
        self.assertEqual(main_low.path_step_m, 0.020)
        self.assertEqual(main_high.path_length_m, 0.200)
        self.assertEqual(len(main_low.waypoints_xy_m), 21)

    def test_full_task_sampling_statistics(self) -> None:
        rng = random.Random(20260729)
        samples = [
            curriculum.sample_translation(curriculum.PHASE, rng)
            for _ in range(20_000)
        ]
        summary = curriculum.summarize_samples(samples)
        self.assertEqual(summary["count"], 20_000)
        self.assertEqual(
            summary["source_region_counts"],
            {"all": 20_000},
        )
        self.assertEqual(float(summary["path_length_min_m"]), 0.200)
        self.assertEqual(float(summary["path_length_max_m"]), 0.400)
        self.assertLessEqual(
            float(summary["maximum_waypoint_radius_m"]),
            contract.TRANSLATION_PATH_GENERATION_RADIUS_M + 1.0e-12,
        )
        self.assertTrue(all(count > 0 for count in summary["shape_counts"].values()))

    def test_rotation_curriculum_closed_integer_boundaries(self) -> None:
        full_low = curriculum.sample_rotation(
            curriculum.PHASE,
            _BoundaryRandom(high_radius=False, high_theta=False),
        )
        full_high = curriculum.sample_rotation(
            curriculum.PHASE,
            _BoundaryRandom(high_radius=False, high_theta=True),
        )
        self.assertEqual(
            (full_low.rotation_angle_deg, full_high.rotation_angle_deg),
            (10, 45),
        )
        self.assertEqual(full_low.source_region, curriculum.REGION_ALL)
        self.assertEqual(
            curriculum.summarize_rotation_samples(
                (full_low, full_high)
            )["angle_unique_count"],
            2,
        )

    def test_curriculum_rejects_out_of_range_boundaries(self) -> None:
        for seed in (0, 1.5, contract.TRANSLATION_PATH_SEED_MAX + 1):
            with self.assertRaises(ValueError):
                curriculum.translation_path_waypoints_m(
                    curriculum.PHASE, 0, seed
                )
        for shape in (-1, len(contract.TRANSLATION_PATH_SHAPES), 1.5):
            with self.assertRaises(ValueError):
                curriculum.translation_path_waypoints_m(
                    curriculum.PHASE, shape, 1
                )
        for angle in (9, 46, 10.5):
            with self.assertRaisesRegex(ValueError, "rotation_angle_deg"):
                curriculum.validate_rotation_angle_deg(curriculum.PHASE, angle)
        for legacy in ("phase1", "phase2"):
            with self.assertRaisesRegex(ValueError, "unsupported curriculum"):
                curriculum.validate_phase(legacy)


class SuccessTests(unittest.TestCase):
    def test_all_fruit_per_axis_boundaries_are_inclusive(self) -> None:
        for object_kind in contract.OBJECTS:
            self.assertTrue(
                success.endpoint_success(
                    object_kind,
                    (0.010, -0.010, 0.010),
                    (10.0, -10.0, 10.0),
                )
            )
            self.assertFalse(
                success.endpoint_success(
                    object_kind,
                    (0.010001, 0.0, 0.0),
                    (10.0, 0.0, 0.0),
                )
            )
            self.assertFalse(
                success.endpoint_success(
                    object_kind,
                    (0.010, 0.0, 0.0),
                    (10.001, 0.0, 0.0),
                )
            )
        with self.assertRaises(ValueError):
            success.endpoint_success("plate", (0.0,) * 3, (0.0,) * 3)
        for infeasible in contract.INFEASIBLE_OBJECTS:
            with self.assertRaises(ValueError):
                success.endpoint_success(
                    infeasible, (0.0,) * 3, (0.0,) * 3
                )

    def test_l2_metrics_are_diagnostic_but_success_is_an_axis_box(self) -> None:
        self.assertAlmostEqual(
            success.euclidean_position_error_m((0.003, 0.004, 0.0), (0, 0, 0)),
            0.005,
        )
        self.assertEqual(
            success.wrapped_euler_delta_deg((359, 1, 180), (1, 359, 0)),
            (-2.0, 2.0, -180.0),
        )
        self.assertAlmostEqual(
            success.euler_l2_error_deg((3, 4, 0), (0, 0, 0)),
            5.0,
        )
        self.assertTrue(
            success.pose_success(
                "orange",
                (0.003, 0.004, 0.0),
                (0.0, 0.0, 0.0),
                (3.0, 4.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        )
        self.assertTrue(
            success.endpoint_success(
                "orange",
                (0.010, -0.010, 0.010),
                (10.0, -10.0, 10.0),
            )
        )
        self.assertGreater(
            success.euclidean_position_error_m(
                (0.010, -0.010, 0.010), (0.0, 0.0, 0.0)
            ),
            contract.ROTATION_POSITION_AXIS_ABS_TOLERANCE_M,
        )
        self.assertGreater(
            success.euler_l2_error_deg(
                (10.0, -10.0, 10.0), (0.0, 0.0, 0.0)
            ),
            contract.ROTATION_EULER_AXIS_ABS_TOLERANCE_DEG,
        )

    def test_nonfinite_or_malformed_axis_errors_fail(self) -> None:
        for position, euler in (
            ((math.nan, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, math.inf, 0.0)),
            ((0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), "not-a-vector"),
        ):
            self.assertFalse(success.endpoint_success("orange", position, euler))


if __name__ == "__main__":
    unittest.main()
