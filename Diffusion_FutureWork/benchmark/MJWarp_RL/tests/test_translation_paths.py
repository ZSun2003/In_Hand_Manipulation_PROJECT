"""Regression tests for the fixed five-path translation contract."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import math
import random
import struct
from types import MethodType
import unittest
from unittest import mock

from MJWarp_RL import contract, curriculum, env, success
from MJWarp_RL.hpc_config import DEFAULT_CONFIG


_PHASES = (curriculum.PHASE,)
_ALL_SHAPES = range(len(contract.TRANSLATION_PATH_SHAPES))


class TranslationPathCpuTests(unittest.TestCase):
    def test_action_start_boundary_is_defined_and_applied(self) -> None:
        torch = env.torch
        if torch is None:
            self.skipTest("PyTorch is unavailable")
        instance = object.__new__(env.MJWarpIKResidualEnv)
        instance.dtype = torch.float32
        instance.device = torch.device("cpu")
        instance.config = replace(DEFAULT_CONFIG, task=contract.TASK_ROTATE)
        action_start = sum(
            (
                instance.config.plan_ticks,
                instance.config.close_ticks,
                instance.config.settle_ticks,
                instance.config.lift_ticks,
                instance.config.hold_ticks,
            )
        )
        instance.control_step = torch.tensor(
            (action_start - 1, action_start), dtype=torch.long
        )
        instance.initial_body_pos = torch.zeros((2, 3), dtype=torch.float32)
        instance.action_start_com = torch.zeros((2, 3), dtype=torch.float32)
        identity = torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(2, 1)
        instance.initial_quat = identity.clone()
        instance.action_start_quat = identity.clone()
        instance.goal_quat = identity.clone()
        instance.object_index = torch.zeros(2, dtype=torch.long)
        instance.task_index = torch.full(
            (2,), contract.TASKS.index(contract.TASK_ROTATE), dtype=torch.long
        )
        instance.action_reference_valid = torch.ones(2, dtype=torch.bool)
        zeros3 = torch.zeros((2, 3), dtype=torch.float32)
        zeros1 = torch.zeros(2, dtype=torch.float32)
        false = torch.zeros(2, dtype=torch.bool)
        instance._translation_path_state = MethodType(
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
            instance,
        )

        position, quaternion, _, _, active = instance._desired_object_pose()

        self.assertEqual(active.tolist(), [False, True])
        self.assertTrue(bool(torch.isfinite(position).all()))
        self.assertTrue(bool(torch.isfinite(quaternion).all()))

    def test_fixed_path_contract_is_exact_and_phase_invariant(self) -> None:
        self.assertEqual(
            contract.TRANSLATION_PATH_SHAPES,
            (
                "square_100mm",
                "circle_r50mm",
                "letter_s_200mm",
                "letter_u_200mm",
                "letter_n_200mm",
            ),
        )
        self.assertEqual(contract.TRANSLATION_PATH_SEGMENTS, 20)
        self.assertEqual(contract.TRANSLATION_PATH_WAYPOINTS, 21)
        self.assertEqual(contract.TRANSLATION_PATH_REGION_RADIUS_M, 0.100)
        self.assertEqual(
            contract.TRANSLATION_PATH_LENGTH_M_BY_SHAPE,
            (0.400, 0.314159265359, 0.200, 0.200, 0.200),
        )
        for shape_index in _ALL_SHAPES:
            full = curriculum.translation_path_waypoints_m(
                curriculum.PHASE, shape_index, 1
            )
            self.assertEqual(
                full, contract.TRANSLATION_PATH_WAYPOINTS_XY_M[shape_index]
            )
            self.assertEqual(len(full), 21)
            self.assertLessEqual(
                max(math.hypot(*point) for point in full),
                contract.TRANSLATION_PATH_REGION_RADIUS_M + 1.0e-12,
            )
            for phase in _PHASES:
                step = curriculum.translation_path_step_m(phase, shape_index)
                length = curriculum.translation_path_length_m(
                    phase, shape_index
                )
                self.assertAlmostEqual(step * 20, length, delta=1.0e-12)

    def test_square_and_circle_are_centered_closed_paths(self) -> None:
        square = contract.TRANSLATION_PATH_WAYPOINTS_XY_M[0]
        self.assertEqual(square[0], (0.05, 0.05))
        for waypoint_index, expected in (
            (0, (0.05, 0.05)),
            (5, (-0.05, 0.05)),
            (10, (-0.05, -0.05)),
            (15, (0.05, -0.05)),
        ):
            for actual, target in zip(square[waypoint_index], expected):
                self.assertAlmostEqual(actual, target, delta=1.0e-12)
        for actual, target in zip(square[-1], square[0]):
            self.assertAlmostEqual(actual, target, delta=1.0e-12)

        circle = contract.TRANSLATION_PATH_WAYPOINTS_XY_M[1]
        self.assertEqual(circle[0], (0.05, 0.0))
        self.assertEqual(circle[-1], circle[0])
        for point in circle:
            self.assertAlmostEqual(math.hypot(*point), 0.05, delta=1.0e-12)

    def test_letters_are_centered_and_have_exact_200mm_arc_length(self) -> None:
        for shape_index in (2, 3, 4):
            self.assertAlmostEqual(
                contract.TRANSLATION_PATH_LENGTH_M_BY_SHAPE[shape_index],
                0.200,
                delta=1.0e-12,
            )
            samples = [
                contract.translation_path_point_xy_m(shape_index, index / 2000)
                for index in range(2001)
            ]
            xs = [point[0] for point in samples]
            ys = [point[1] for point in samples]
            self.assertAlmostEqual(min(xs) + max(xs), 0.0, delta=1.0e-9)
            self.assertAlmostEqual(min(ys) + max(ys), 0.0, delta=1.0e-9)
            self.assertLessEqual(
                max(math.hypot(*point) for point in samples),
                contract.TRANSLATION_PATH_REGION_RADIUS_M + 1.0e-12,
            )

        s_radius = 0.100 / math.pi
        self.assertEqual(
            contract.translation_path_point_xy_m(2, 0.0),
            (0.0, 2.0 * s_radius),
        )
        self.assertAlmostEqual(
            math.hypot(*contract.translation_path_point_xy_m(2, 0.5)),
            0.0,
            delta=1.0e-12,
        )
        self.assertEqual(
            contract.translation_path_point_xy_m(2, 1.0),
            (0.0, -2.0 * s_radius),
        )

    def test_path_seed_is_a_fixed_provenance_sentinel(self) -> None:
        self.assertEqual(contract.TRANSLATION_PATH_SEED_MIN, 1)
        self.assertEqual(contract.TRANSLATION_PATH_SEED_MAX, 1)
        with self.assertRaises(ValueError):
            curriculum.translation_path_waypoints_m(
                curriculum.PHASE, 0, 2
            )

    def test_bounded_schedule_tables_and_discrete_vector_peaks(self) -> None:
        self.assertEqual(contract.TRANSLATION_CONTROL_DT_S, 0.020)
        self.assertEqual(contract.TRANSLATION_SCHEDULE_QUANTIZATION_DECIMALS, 12)
        self.assertEqual(
            contract.TRANSLATION_SCHEDULE_QUANTIZATION_SCALE,
            1_000_000_000_000,
        )
        self.assertEqual(
            contract.TRANSLATION_SCHEDULE_TABLE_HASH_ENCODING,
            "sha256_big_endian_signed_int64_grid_1e-12",
        )
        self.assertEqual(contract.TRANSLATION_PATH_ROUTE_SAMPLES, 1_800)
        self.assertEqual(contract.TRANSLATION_PATH_START_SETTLE_TICKS, 300)
        self.assertEqual(contract.TRANSLATION_PATH_TERMINAL_HOLD_TICKS, 302)
        self.assertEqual(contract.TRANSLATION_LEAD_HARD_MAX_SPEED_M_S, 0.040)
        self.assertEqual(
            contract.TRANSLATION_LEAD_HARD_MAX_ACCELERATION_M_S2, 0.150
        )
        self.assertEqual(
            contract.TRANSLATION_LEAD_PROFILE_ACCELERATION_M_S2, 0.149
        )
        self.assertEqual(contract.TRANSLATION_ROUTE_HARD_MAX_SPEED_M_S, 0.040)
        self.assertEqual(
            contract.TRANSLATION_ROUTE_HARD_MAX_ACCELERATION_M_S2, 0.150
        )
        self.assertEqual(
            contract.TRANSLATION_ROUTE_PROFILE_ACCELERATION_M_S2, 0.149
        )
        self.assertEqual(
            tuple(map(sum, contract.TRANSLATION_ROUTE_STROKE_TICKS)),
            (1_800,) * 5,
        )
        self.assertEqual(contract.TRANSLATION_ROUTE_STROKE_TICKS[0], (450,) * 4)
        self.assertEqual(contract.TRANSLATION_ROUTE_STROKE_TICKS[4], (555, 690, 555))
        self.assertEqual(
            contract.TRANSLATION_ROUTE_PROGRESS_TABLE_SHA256,
            "6865f79543c5a42ab9019965538a8a4f0aa0d35f2286c378621ab48d41e09784",
        )
        self.assertEqual(
            contract.TRANSLATION_LEAD_PROGRESS_TABLE_SHA256,
            "43ecd7d772e4db8dd1fb05c47ba5e5d9c659310a705dab6b6beeaac0880664dd",
        )

        def discrete_vector_peaks(
            points: list[tuple[float, float]],
        ) -> tuple[float, float]:
            dt = contract.TRANSLATION_CONTROL_DT_S
            velocity = [(0.0, 0.0)]
            velocity.extend(
                (
                    (right[0] - left[0]) / dt,
                    (right[1] - left[1]) / dt,
                )
                for left, right in zip(points, points[1:])
            )
            acceleration = [
                (
                    (right[0] - left[0]) / dt,
                    (right[1] - left[1]) / dt,
                )
                for left, right in zip(velocity, velocity[1:])
            ]
            return (
                max(math.hypot(*value) for value in velocity),
                max(math.hypot(*value) for value in acceleration),
            )

        for shape_index in _ALL_SHAPES:
            route = contract.TRANSLATION_ROUTE_PROGRESS_TABLE[shape_index]
            lead = contract.TRANSLATION_LEAD_PROGRESS_TABLE[shape_index]
            self.assertEqual(len(route), 1_801)
            self.assertEqual(len(lead), 301)
            self.assertEqual((route[0], route[-1]), (0.0, 1.0))
            self.assertEqual((lead[0], lead[-1]), (0.0, 1.0))
            self.assertTrue(all(a <= b for a, b in zip(route, route[1:])))
            self.assertTrue(all(a <= b for a, b in zip(lead, lead[1:])))
            for value in (*route, *lead):
                self.assertEqual(
                    value,
                    float(
                        round(
                            value
                            * contract.TRANSLATION_SCHEDULE_QUANTIZATION_SCALE
                        )
                        / contract.TRANSLATION_SCHEDULE_QUANTIZATION_SCALE
                    ),
                )

            route_points = list(
                contract.TRANSLATION_ROUTE_TARGETS_XY_M[shape_index]
            )
            lead_points = list(
                contract.TRANSLATION_LEAD_OFFSETS_XY_M[shape_index]
            )
            route_peaks = discrete_vector_peaks(route_points)
            lead_peaks = discrete_vector_peaks(lead_points)
            for actual, recorded in zip(
                route_peaks,
                contract.TRANSLATION_ROUTE_KINEMATIC_PEAKS[shape_index],
            ):
                self.assertAlmostEqual(actual, recorded, delta=4.0e-9)
            for actual, recorded in zip(
                lead_peaks,
                contract.TRANSLATION_LEAD_KINEMATIC_PEAKS[shape_index],
            ):
                self.assertAlmostEqual(actual, recorded, delta=4.0e-9)
            for value in (*route_peaks, *lead_peaks):
                recorded = float(
                    round(
                        value
                        * contract.TRANSLATION_SCHEDULE_QUANTIZATION_SCALE
                    )
                    / contract.TRANSLATION_SCHEDULE_QUANTIZATION_SCALE
                )
                self.assertAlmostEqual(value, recorded, delta=1.0e-12)
            for value in (
                *contract.TRANSLATION_ROUTE_KINEMATIC_PEAKS[shape_index],
                *contract.TRANSLATION_LEAD_KINEMATIC_PEAKS[shape_index],
            ):
                self.assertEqual(
                    value,
                    float(
                        round(
                            value
                            * contract.TRANSLATION_SCHEDULE_QUANTIZATION_SCALE
                        )
                        / contract.TRANSLATION_SCHEDULE_QUANTIZATION_SCALE
                    ),
                )
            self.assertLessEqual(
                route_peaks[0], contract.TRANSLATION_ROUTE_HARD_MAX_SPEED_M_S
            )
            self.assertLessEqual(
                route_peaks[1],
                contract.TRANSLATION_ROUTE_HARD_MAX_ACCELERATION_M_S2,
            )
            self.assertLessEqual(
                lead_peaks[0], contract.TRANSLATION_LEAD_HARD_MAX_SPEED_M_S
            )
            self.assertLessEqual(
                lead_peaks[1],
                contract.TRANSLATION_LEAD_HARD_MAX_ACCELERATION_M_S2,
            )

        square = contract.TRANSLATION_ROUTE_PROGRESS_TABLE[0]
        for actual, expected in zip(
            (square[450], square[900], square[1350], square[1800]),
            (0.25, 0.50, 0.75, 1.0),
        ):
            self.assertAlmostEqual(actual, expected, delta=1.0e-12)
        letter_n = contract.TRANSLATION_ROUTE_PROGRESS_TABLE[4]
        self.assertAlmostEqual(letter_n[555], 4.0 / 13.0, delta=1.0e-12)
        self.assertAlmostEqual(letter_n[1245], 9.0 / 13.0, delta=1.0e-12)
        self.assertEqual(letter_n[1800], 1.0)
        # The worst lead is the square diagonal and retains explicit headroom.
        self.assertAlmostEqual(
            contract.TRANSLATION_LEAD_KINEMATIC_PEAKS[0][0],
            0.022096432202,
            delta=0.0,
        )
        self.assertLess(
            contract.TRANSLATION_LEAD_KINEMATIC_PEAKS[0][0],
            contract.TRANSLATION_LEAD_HARD_MAX_SPEED_M_S,
        )

    def test_canonical_blob_and_fixed_xy_tables_are_golden(self) -> None:
        self.assertEqual(
            contract.TRANSLATION_CANONICAL_BLOB_ENCODING,
            "base85_zlib_be_i64_delta_v1",
        )
        self.assertEqual(contract.TRANSLATION_CANONICAL_BLOB_VERSION, 1)
        self.assertEqual(
            contract.TRANSLATION_CANONICAL_BLOB_SHA256,
            "c3de5f0eb864d63059b009313f56236ff720df9247b05d50391660f53f584ddb",
        )
        self.assertEqual(
            contract.TRANSLATION_CANONICAL_COMPRESSED_SHA256,
            "e315bafef7df586aaba0a85fa6aac8e170f814628263629c92f80d3503bf475b",
        )
        self.assertEqual(
            contract.TRANSLATION_CANONICAL_RAW_SHA256,
            "eb7b18ffe69511b99a6de8d0faf854c2ecd7c3082639f3424e5c3d8955e1601e",
        )
        self.assertEqual(len(contract._TRANSLATION_CANONICAL_BLOB_B85), 87_783)
        self.assertEqual(contract._TRANSLATION_CANONICAL_COMPRESSED_BYTES, 70_226)
        self.assertEqual(contract._TRANSLATION_CANONICAL_RAW_BYTES, 254_132)
        self.assertEqual(
            contract._TRANSLATION_CANONICAL_SECTION_COUNTS,
            (9005, 1505, 210, 18010, 3010, 10, 10),
        )

        def int64_hash(values: list[float]) -> str:
            integers = [
                round(value * contract.TRANSLATION_SCHEDULE_QUANTIZATION_SCALE)
                for value in values
            ]
            return hashlib.sha256(
                struct.pack(f">{len(integers)}q", *integers)
            ).hexdigest()

        sections = (
            (
                [value for row in contract.TRANSLATION_ROUTE_PROGRESS_TABLE for value in row],
                contract.TRANSLATION_ROUTE_PROGRESS_TABLE_SHA256,
            ),
            (
                [value for row in contract.TRANSLATION_LEAD_PROGRESS_TABLE for value in row],
                contract.TRANSLATION_LEAD_PROGRESS_TABLE_SHA256,
            ),
            (
                [
                    value
                    for row in contract.TRANSLATION_PATH_WAYPOINTS_XY_M
                    for point in row
                    for value in point
                ],
                contract.TRANSLATION_PATH_WAYPOINTS_XY_SHA256,
            ),
            (
                [
                    value
                    for row in contract.TRANSLATION_ROUTE_TARGETS_XY_M
                    for point in row
                    for value in point
                ],
                contract.TRANSLATION_ROUTE_TARGETS_XY_SHA256,
            ),
            (
                [
                    value
                    for row in contract.TRANSLATION_LEAD_OFFSETS_XY_M
                    for point in row
                    for value in point
                ],
                contract.TRANSLATION_LEAD_OFFSETS_XY_SHA256,
            ),
            (
                [value for row in contract.TRANSLATION_ROUTE_KINEMATIC_PEAKS for value in row],
                contract.TRANSLATION_ROUTE_KINEMATIC_PEAKS_SHA256,
            ),
            (
                [value for row in contract.TRANSLATION_LEAD_KINEMATIC_PEAKS for value in row],
                contract.TRANSLATION_LEAD_KINEMATIC_PEAKS_SHA256,
            ),
        )
        for values, expected in sections:
            self.assertEqual(int64_hash(values), expected)

        self.assertEqual(
            tuple(len(row) for row in contract.TRANSLATION_ROUTE_TARGETS_XY_M),
            (1_801,) * 5,
        )
        self.assertEqual(
            tuple(len(row) for row in contract.TRANSLATION_LEAD_OFFSETS_XY_M),
            (301,) * 5,
        )
        for shape_index in _ALL_SHAPES:
            anchors = contract.TRANSLATION_PATH_WAYPOINTS_XY_M[shape_index]
            route = contract.TRANSLATION_ROUTE_TARGETS_XY_M[shape_index]
            lead = contract.TRANSLATION_LEAD_OFFSETS_XY_M[shape_index]
            self.assertEqual(route[0], anchors[0])
            self.assertEqual(route[-1], anchors[-1])
            self.assertEqual(lead[0], (0.0, 0.0))
            self.assertEqual(lead[-1], anchors[0])
            for progress, actual in zip(
                contract.TRANSLATION_ROUTE_PROGRESS_TABLE[shape_index], route
            ):
                expected = contract.translation_path_point_xy_m(
                    shape_index, progress
                )
                for value, reference in zip(actual, expected):
                    self.assertAlmostEqual(value, reference, delta=1.1e-12)
            start = anchors[0]
            for progress, actual in zip(
                contract.TRANSLATION_LEAD_PROGRESS_TABLE[shape_index], lead
            ):
                for value, reference in zip(
                    actual, (progress * start[0], progress * start[1])
                ):
                    self.assertAlmostEqual(value, reference, delta=1.1e-12)

    def test_production_schedule_uses_only_fixed_decoded_tables(self) -> None:
        contract_source = inspect.getsource(contract)
        self.assertNotIn("_build_translation_route_progress_table", contract_source)
        self.assertNotIn("_build_translation_lead_progress_table", contract_source)
        self.assertNotIn("_schedule_xy_peaks", contract_source)
        decoder_source = inspect.getsource(
            contract._decode_translation_canonical_blob
        )
        self.assertNotIn("math.", decoder_source)
        self.assertNotIn("translation_path_point_xy_m", decoder_source)
        self.assertIn("_TRANSLATION_CANONICAL_RAW_BYTES + 1", decoder_source)
        state_source = inspect.getsource(
            env.MJWarpIKResidualEnv._translation_path_state
        )
        certificate_source = inspect.getsource(
            env.MJWarpIKResidualEnv._update_path_schedule_certificate
        )
        self.assertNotIn("_translation_path_point_torch", state_source)
        self.assertNotIn("_translation_path_point_torch", certificate_source)
        self.assertIn("_translation_route_targets_xy_table", state_source)
        self.assertIn("_translation_lead_offsets_xy_table", state_source)
        self.assertIn("_translation_route_targets_xy_table", certificate_source)

    def test_canonical_blob_tamper_fails_closed(self) -> None:
        blob = contract._TRANSLATION_CANONICAL_BLOB_B85
        replacement = "0" if blob[0] != "0" else "1"
        with mock.patch.object(
            contract,
            "_TRANSLATION_CANONICAL_BLOB_B85",
            replacement + blob[1:],
        ):
            with self.assertRaisesRegex(RuntimeError, "encoded SHA-256"):
                contract._decode_translation_canonical_blob()
        with mock.patch.object(
            contract, "_TRANSLATION_CANONICAL_BLOB_B85", blob[:-1]
        ):
            with self.assertRaisesRegex(RuntimeError, "encoded byte count"):
                contract._decode_translation_canonical_blob()

    def test_sampling_is_uniform_over_only_the_five_fixed_paths(self) -> None:
        rng = random.Random(20_260_803)
        samples = tuple(
            curriculum.sample_translation(curriculum.PHASE, rng)
            for _ in range(10_000)
        )
        counts = [0] * len(contract.TRANSLATION_PATH_SHAPES)
        for sample in samples:
            counts[sample.path_shape_index] += 1
            self.assertEqual(sample.path_seed, 1)
            self.assertEqual(
                sample.waypoints_xy_m,
                contract.TRANSLATION_PATH_WAYPOINTS_XY_M[
                    sample.path_shape_index
                ],
            )
        self.assertTrue(all(1800 <= count <= 2200 for count in counts))


class TranslationPathContractIntegrationTests(unittest.TestCase):
    def test_runtime_config_rejects_zero_or_drifted_task_timing(self) -> None:
        from MJWarp_RL.env import _validate_runtime_config

        for field in (
            "translation_path_lead_in_ticks",
            "translation_segment_ticks",
            "rotation_motion_ticks",
        ):
            with self.subTest(field=field, value=0):
                with self.assertRaises(ValueError):
                    _validate_runtime_config(
                        replace(DEFAULT_CONFIG, **{field: 0})
                    )
            with self.subTest(field=field, value="drifted"):
                with self.assertRaises(ValueError):
                    _validate_runtime_config(
                        replace(
                            DEFAULT_CONFIG,
                            **{field: int(getattr(DEFAULT_CONFIG, field)) + 1},
                        )
                    )

    def test_formal_timing_is_single_run_for_translation(self) -> None:
        config = DEFAULT_CONFIG
        config.validate()
        self.assertEqual(config.episode_length, 3_000)
        self.assertEqual(config.control_ticks, 3_000)
        self.assertEqual(config.translation_motion_ticks, 2_100)
        self.assertEqual(config.action_ticks, 2_702)
        self.assertEqual(
            config.action_ticks - config.translation_motion_ticks, 602
        )
        self.assertEqual(
            config.action_ticks - config.translation_motion_ticks,
            contract.TRANSLATION_PATH_START_SETTLE_TICKS
            + contract.TRANSLATION_PATH_TERMINAL_HOLD_TICKS,
        )
        self.assertEqual(
            contract.TRANSLATION_PATH_LEAD_IN_TICKS
            + contract.TRANSLATION_PATH_START_SETTLE_TICKS
            + contract.TRANSLATION_PATH_ROUTE_SAMPLES
            + contract.TRANSLATION_PATH_TERMINAL_HOLD_TICKS,
            config.action_ticks,
        )
        rotation = replace(config, task=contract.TASK_ROTATE)
        rotation.validate()
        self.assertEqual(rotation.action_ticks, 2_702)
        self.assertEqual(rotation.episode_length, 3_000)

    def test_task_specific_success_thresholds_are_exact(self) -> None:
        self.assertEqual(
            contract.TRANSLATION_POSITION_AXIS_ABS_TOLERANCE_M, 0.010
        )
        self.assertEqual(
            contract.TRANSLATION_EULER_AXIS_ABS_TOLERANCE_DEG, 20.0
        )
        self.assertEqual(
            contract.TRANSLATION_PATH_TRACKING_AXIS_ABS_TOLERANCE_M, 0.010
        )
        self.assertEqual(
            contract.ROTATION_POSITION_AXIS_ABS_TOLERANCE_M, 0.010
        )
        self.assertEqual(
            contract.ROTATION_EULER_AXIS_ABS_TOLERANCE_DEG, 10.0
        )
        for object_kind in contract.OBJECTS:
            translation = success.threshold_for_object(
                object_kind, contract.TASK_TRANSLATE
            )
            rotation = success.threshold_for_object(
                object_kind, contract.TASK_ROTATE
            )
            self.assertEqual(translation.position_axis_abs_m, 0.010)
            self.assertEqual(translation.euler_axis_abs_deg, 20.0)
            self.assertEqual(rotation.position_axis_abs_m, 0.010)
            self.assertEqual(rotation.euler_axis_abs_deg, 10.0)


class TranslationPathTorchTests(unittest.TestCase):
    def test_route_certificate_accepts_exact_tracking_and_rejects_closed_hold(
        self,
    ) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        translate = contract.TASKS.index(contract.TASK_TRANSLATE)
        rotation = contract.TASKS.index(contract.TASK_ROTATE)
        task_index = torch.tensor(
            (translate, translate, translate, rotation), dtype=torch.long
        )
        axis_maxima = []
        # Row 0 follows every square target exactly. Rows 1 and 2 hold the
        # starting point of the closed square/circle, which shares the final
        # endpoint but necessarily misses the intervening route by >10 mm.
        square_targets = torch.as_tensor(
            contract.TRANSLATION_ROUTE_TARGETS_XY_M[0],
            dtype=torch.float64,
        )
        exact_error_xy = (square_targets - square_targets.clone()).abs().max(
            dim=0
        ).values
        axis_maxima.append(
            torch.cat(
                (exact_error_xy, torch.zeros(1, dtype=torch.float64))
            )
        )
        for shape_index in (0, 1):
            target_xy = torch.as_tensor(
                contract.TRANSLATION_ROUTE_TARGETS_XY_M[shape_index],
                dtype=torch.float64,
            )
            held_xy = target_xy[0].expand_as(target_xy)
            error_xy = (target_xy - held_xy).abs().max(dim=0).values
            axis_maxima.append(
                torch.cat((error_xy, torch.zeros(1, dtype=torch.float64)))
            )
        axis_maxima.append(torch.full((3,), 1.0, dtype=torch.float64))
        certified = env._path_tracking_certified(
            task_index,
            torch.tensor((1800, 1800, 1800, 0), dtype=torch.long),
            torch.stack(axis_maxima),
            torch.tensor((True, True, True, False)),
            torch.tensor((True, True, True, False)),
        )
        self.assertEqual(certified.tolist(), [True, False, False, True])
        self.assertGreater(float(axis_maxima[1].max()), 0.010)
        self.assertGreater(float(axis_maxima[2].max()), 0.010)

        fail_closed = env._path_tracking_certified(
            torch.full((3,), translate, dtype=torch.long),
            torch.tensor((1799, 1800, 1800), dtype=torch.long),
            torch.zeros((3, 3), dtype=torch.float64),
            torch.tensor((True, False, True)),
            torch.tensor((True, True, False)),
        )
        self.assertEqual(fail_closed.tolist(), [False, False, False])

    def test_route_starts_after_settle_and_has_exactly_1800_samples(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        action_start = sum(
            (
                DEFAULT_CONFIG.plan_ticks,
                DEFAULT_CONFIG.close_ticks,
                DEFAULT_CONFIG.settle_ticks,
                DEFAULT_CONFIG.lift_ticks,
                DEFAULT_CONFIG.hold_ticks,
            )
        )
        steps = torch.arange(
            action_start - 1,
            action_start + DEFAULT_CONFIG.action_ticks + 1,
            dtype=torch.long,
        )
        worlds = int(steps.numel())
        instance = object.__new__(env.MJWarpIKResidualEnv)
        instance.torch = torch
        instance.device = torch.device("cpu")
        instance.dtype = torch.float64
        instance.worlds = worlds
        instance.config = DEFAULT_CONFIG
        instance.control_step = steps
        instance.path_shape_index = torch.zeros(worlds, dtype=torch.long)
        instance.task_index = torch.full(
            (worlds,),
            contract.TASKS.index(contract.TASK_TRANSLATE),
            dtype=torch.long,
        )
        instance.action_reference_valid = torch.ones(worlds, dtype=torch.bool)
        instance.path_waypoints_xy_m = torch.as_tensor(
            contract.TRANSLATION_PATH_WAYPOINTS_XY_M[0],
            dtype=torch.float64,
        )[None, :, :].expand(worlds, -1, -1)
        instance.action_start_com = torch.zeros((worlds, 3), dtype=torch.float64)
        instance._translation_route_progress_table = torch.as_tensor(
            contract.TRANSLATION_ROUTE_PROGRESS_TABLE,
            dtype=torch.float64,
        )
        instance._translation_lead_progress_table = torch.as_tensor(
            contract.TRANSLATION_LEAD_PROGRESS_TABLE,
            dtype=torch.float64,
        )
        instance._translation_route_targets_xy_table = torch.as_tensor(
            contract.TRANSLATION_ROUTE_TARGETS_XY_M,
            dtype=torch.float64,
        )
        instance._translation_lead_offsets_xy_table = torch.as_tensor(
            contract.TRANSLATION_LEAD_OFFSETS_XY_M,
            dtype=torch.float64,
        )

        (
            offset,
            target,
            lead_progress,
            _,
            _,
            _,
            route_progress,
            following,
        ) = instance._translation_path_state(steps)

        following_steps = steps[following]
        self.assertEqual(int(following.sum()), 1_800)
        self.assertEqual(int(following_steps[0]), action_start + 600)
        self.assertEqual(int(following_steps[-1]), action_start + 2_399)
        self.assertTrue(torch.equal(offset, target))

        def row(step: int) -> int:
            return int(torch.nonzero(steps == step, as_tuple=False).item())

        start_xy = torch.tensor((0.05, 0.05), dtype=torch.float64)
        self.assertTrue(torch.equal(offset[row(action_start - 1), :2], torch.zeros(2)))
        self.assertEqual(float(lead_progress[row(action_start + 299)]), 1.0)
        for control_tick in (action_start + 299, action_start + 300, action_start + 599):
            torch.testing.assert_close(
                offset[row(control_tick), :2], start_xy, rtol=0.0, atol=1.0e-12
            )
        self.assertEqual(float(route_progress[row(action_start + 599)]), 0.0)
        self.assertAlmostEqual(
            float(route_progress[row(action_start + 600)]),
            contract.TRANSLATION_ROUTE_PROGRESS_TABLE[0][1],
            delta=1.0e-12,
        )
        self.assertEqual(float(route_progress[row(action_start + 2_399)]), 1.0)
        self.assertEqual(float(route_progress[row(action_start + 2_400)]), 1.0)
        self.assertEqual(float(route_progress[row(action_start + 2_701)]), 1.0)

    def test_path_state_reads_fixed_xy_tables(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        action_start = sum(
            (
                DEFAULT_CONFIG.plan_ticks,
                DEFAULT_CONFIG.close_ticks,
                DEFAULT_CONFIG.settle_ticks,
                DEFAULT_CONFIG.lift_ticks,
                DEFAULT_CONFIG.hold_ticks,
            )
        )
        instance = object.__new__(env.MJWarpIKResidualEnv)
        instance.torch = torch
        instance.device = torch.device("cpu")
        instance.dtype = torch.float64
        instance.worlds = 2
        instance.config = DEFAULT_CONFIG
        instance.control_step = torch.tensor(
            (action_start, action_start + 600), dtype=torch.long
        )
        instance.path_shape_index = torch.zeros(2, dtype=torch.long)
        instance.task_index = torch.full(
            (2,), contract.TASKS.index(contract.TASK_TRANSLATE), dtype=torch.long
        )
        instance.action_reference_valid = torch.ones(2, dtype=torch.bool)
        instance.path_waypoints_xy_m = torch.as_tensor(
            contract.TRANSLATION_PATH_WAYPOINTS_XY_M[0], dtype=torch.float64
        )[None, :, :].expand(2, -1, -1)
        instance.action_start_com = torch.zeros((2, 3), dtype=torch.float64)
        instance._translation_route_progress_table = torch.as_tensor(
            contract.TRANSLATION_ROUTE_PROGRESS_TABLE, dtype=torch.float64
        )
        instance._translation_lead_progress_table = torch.as_tensor(
            contract.TRANSLATION_LEAD_PROGRESS_TABLE, dtype=torch.float64
        )
        instance._translation_route_targets_xy_table = torch.as_tensor(
            contract.TRANSLATION_ROUTE_TARGETS_XY_M, dtype=torch.float64
        ).clone()
        instance._translation_lead_offsets_xy_table = torch.as_tensor(
            contract.TRANSLATION_LEAD_OFFSETS_XY_M, dtype=torch.float64
        ).clone()
        lead_sentinel = torch.tensor(
            (0.001234, -0.002345), dtype=torch.float64
        )
        route_sentinel = torch.tensor(
            (0.012345, -0.023456), dtype=torch.float64
        )
        instance._translation_lead_offsets_xy_table[0, 1] = lead_sentinel
        instance._translation_route_targets_xy_table[0, 1] = route_sentinel

        offset = instance._translation_path_state()[0]

        torch.testing.assert_close(
            offset[0, :2], lead_sentinel, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            offset[1, :2], route_sentinel, rtol=0.0, atol=0.0
        )

    def test_unadmitted_route_holds_start_without_sampling_or_endpoint_jump(
        self,
    ) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        action_start = sum(
            (
                DEFAULT_CONFIG.plan_ticks,
                DEFAULT_CONFIG.close_ticks,
                DEFAULT_CONFIG.settle_ticks,
                DEFAULT_CONFIG.lift_ticks,
                DEFAULT_CONFIG.hold_ticks,
            )
        )
        instance = object.__new__(env.MJWarpIKResidualEnv)
        instance.torch = torch
        instance.device = torch.device("cpu")
        instance.dtype = torch.float32
        instance.worlds = 2
        instance.config = DEFAULT_CONFIG
        instance.control_step = torch.tensor(
            (action_start + 600, action_start + 2_500), dtype=torch.long
        )
        instance.path_shape_index = torch.zeros(2, dtype=torch.long)
        instance.task_index = torch.full(
            (2,), contract.TASKS.index(contract.TASK_TRANSLATE), dtype=torch.long
        )
        instance.action_reference_valid = torch.ones(2, dtype=torch.bool)
        instance.translation_route_admitted = torch.zeros(2, dtype=torch.bool)
        instance.path_waypoints_xy_m = torch.as_tensor(
            contract.TRANSLATION_PATH_WAYPOINTS_XY_M[0], dtype=torch.float32
        )[None, :, :].expand(2, -1, -1)
        instance.action_start_com = torch.zeros((2, 3), dtype=torch.float32)
        instance._translation_route_progress_table = torch.as_tensor(
            contract.TRANSLATION_ROUTE_PROGRESS_TABLE, dtype=torch.float32
        )
        instance._translation_lead_progress_table = torch.as_tensor(
            contract.TRANSLATION_LEAD_PROGRESS_TABLE, dtype=torch.float32
        )
        instance._translation_route_targets_xy_table = torch.as_tensor(
            contract.TRANSLATION_ROUTE_TARGETS_XY_M, dtype=torch.float32
        )
        instance._translation_lead_offsets_xy_table = torch.as_tensor(
            contract.TRANSLATION_LEAD_OFFSETS_XY_M, dtype=torch.float32
        )

        offset, _, *_, following = instance._translation_path_state()
        expected_start = instance.path_waypoints_xy_m[:, 0]
        torch.testing.assert_close(offset[:, :2], expected_start)
        self.assertEqual(following.tolist(), [False, False])

    def test_runtime_schedule_certificate_matches_static_tables(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        worlds = len(contract.TRANSLATION_PATH_SHAPES)
        instance = object.__new__(env.MJWarpIKResidualEnv)
        instance.torch = torch
        instance.device = torch.device("cpu")
        instance.dtype = torch.float64
        instance.worlds = worlds
        instance.path_shape_index = torch.arange(worlds, dtype=torch.long)
        instance._translation_route_progress_table = torch.as_tensor(
            contract.TRANSLATION_ROUTE_PROGRESS_TABLE, dtype=torch.float64
        )
        instance._translation_route_targets_xy_table = torch.as_tensor(
            contract.TRANSLATION_ROUTE_TARGETS_XY_M, dtype=torch.float64
        )
        starts = torch.as_tensor(
            [points[0] for points in contract.TRANSLATION_PATH_WAYPOINTS_XY_M],
            dtype=torch.float64,
        )
        instance.path_tracking_sample_count = torch.zeros(worlds, dtype=torch.long)
        instance.path_schedule_previous_offset_xy_m = starts.clone()
        instance.path_schedule_previous_velocity_xy_m_s = torch.zeros_like(starts)
        instance.path_schedule_progress = torch.zeros(worlds, dtype=torch.float64)
        instance.path_schedule_limit_valid = torch.ones(worlds, dtype=torch.bool)
        instance.path_schedule_complete = torch.zeros(worlds, dtype=torch.bool)
        instance.path_schedule_max_speed_m_s = torch.zeros(worlds, dtype=torch.float64)
        instance.path_schedule_max_acceleration_m_s2 = torch.zeros(
            worlds, dtype=torch.float64
        )
        shapes = instance.path_shape_index
        translating = torch.ones(worlds, dtype=torch.bool)
        for sample_index in range(1, contract.TRANSLATION_PATH_ROUTE_SAMPLES + 1):
            progress = instance._translation_route_progress_table[:, sample_index]
            offset = instance._translation_route_targets_xy_table[
                :, sample_index
            ]
            instance._update_path_schedule_certificate(
                translating, progress, offset
            )
            instance.path_tracking_sample_count.add_(1)

        self.assertTrue(bool(instance.path_schedule_limit_valid.all()))
        self.assertTrue(bool(instance.path_schedule_complete.all()))
        torch.testing.assert_close(
            instance.path_schedule_progress,
            torch.ones(worlds, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-12,
        )
        expected_peaks = torch.as_tensor(
            contract.TRANSLATION_ROUTE_KINEMATIC_PEAKS, dtype=torch.float64
        )
        torch.testing.assert_close(
            instance.path_schedule_max_speed_m_s,
            expected_peaks[:, 0],
            rtol=0.0,
            atol=4.0e-9,
        )
        torch.testing.assert_close(
            instance.path_schedule_max_acceleration_m_s2,
            expected_peaks[:, 1],
            rtol=0.0,
            atol=4.0e-9,
        )
        certified = env._path_tracking_certified(
            torch.full(
                (worlds,),
                contract.TASKS.index(contract.TASK_TRANSLATE),
                dtype=torch.long,
            ),
            instance.path_tracking_sample_count,
            torch.zeros((worlds, 3), dtype=torch.float64),
            instance.path_schedule_complete,
            instance.path_schedule_limit_valid,
        )
        self.assertTrue(bool(certified.all()))

        # A target perturbation invalidates the sticky schedule certificate.
        instance.path_tracking_sample_count.zero_()
        instance.path_schedule_previous_offset_xy_m.copy_(starts)
        instance.path_schedule_previous_velocity_xy_m_s.zero_()
        instance.path_schedule_progress.zero_()
        instance.path_schedule_limit_valid.fill_(True)
        instance.path_schedule_complete.zero_()
        first_progress = instance._translation_route_progress_table[:, 1]
        first_offset = instance._translation_route_targets_xy_table[:, 1].clone()
        first_offset[0, 0] += 0.002
        instance._update_path_schedule_certificate(
            translating, first_progress, first_offset
        )
        self.assertFalse(bool(instance.path_schedule_limit_valid[0]))
        self.assertTrue(bool(instance.path_schedule_limit_valid[1:].all()))

        snapshot_source = inspect.getsource(env.MJWarpIKResidualEnv.snapshot)
        for name in (
            '"path_schedule_limit_valid"',
            '"path_schedule_complete"',
            '"path_schedule_max_speed_m_s"',
            '"path_schedule_max_acceleration_m_s2"',
        ):
            self.assertIn(name, snapshot_source)

    def test_float32_endpoint_rounding_is_certified_by_integer_sample(self) -> None:
        """Preterminal float32 progress==1 must not invalidate the route."""

        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        worlds = len(contract.TRANSLATION_PATH_SHAPES)
        instance = object.__new__(env.MJWarpIKResidualEnv)
        instance.torch = torch
        instance.device = torch.device("cpu")
        instance.dtype = torch.float32
        instance.worlds = worlds
        instance.path_shape_index = torch.arange(worlds, dtype=torch.long)
        instance._translation_route_progress_table = torch.as_tensor(
            contract.TRANSLATION_ROUTE_PROGRESS_TABLE, dtype=torch.float32
        )
        instance._translation_route_targets_xy_table = torch.as_tensor(
            contract.TRANSLATION_ROUTE_TARGETS_XY_M, dtype=torch.float32
        )
        starts = instance._translation_route_targets_xy_table[:, 0].clone()
        instance.path_tracking_sample_count = torch.zeros(worlds, dtype=torch.long)
        instance.path_schedule_previous_offset_xy_m = starts
        instance.path_schedule_previous_velocity_xy_m_s = torch.zeros_like(starts)
        instance.path_schedule_progress = torch.zeros(worlds, dtype=torch.float32)
        instance.path_schedule_limit_valid = torch.ones(worlds, dtype=torch.bool)
        instance.path_schedule_complete = torch.zeros(worlds, dtype=torch.bool)
        instance.path_schedule_max_speed_m_s = torch.zeros(worlds)
        instance.path_schedule_max_acceleration_m_s2 = torch.zeros(worlds)

        preterminal_is_one = (
            instance._translation_route_progress_table[:, 1:-1] == 1.0
        ).any(dim=1)
        self.assertTrue(bool(preterminal_is_one.any()))
        sample = torch.ones(worlds, dtype=torch.bool)
        for sample_index in range(1, contract.TRANSLATION_PATH_ROUTE_SAMPLES + 1):
            instance._update_path_schedule_certificate(
                sample,
                instance._translation_route_progress_table[:, sample_index],
                instance._translation_route_targets_xy_table[:, sample_index],
            )
            instance.path_tracking_sample_count.add_(1)

        self.assertTrue(bool(instance.path_schedule_limit_valid.all()))
        self.assertTrue(bool(instance.path_schedule_complete.all()))

    def test_torch_anchor_generator_matches_canonical_table(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        shapes = torch.tensor(tuple(_ALL_SHAPES) + (0,), dtype=torch.long)
        translating = torch.tensor((True,) * 5 + (False,), dtype=torch.bool)
        for dtype, tolerance in (
            (torch.float64, 1.0e-12),
            (torch.float32, 2.0e-7),
        ):
            actual = env._translation_path_waypoints_torch(
                shapes, translating, dtype=dtype
            )
            expected = torch.as_tensor(
                contract.TRANSLATION_PATH_WAYPOINTS_XY_M,
                dtype=dtype,
            )
            torch.testing.assert_close(
                actual[:5], expected, rtol=0.0, atol=tolerance
            )
            self.assertTrue(
                torch.equal(actual[5], torch.zeros_like(actual[5]))
            )

    def test_torch_analytic_targets_match_cpu_for_dense_progress(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        progresses = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
        shapes = torch.arange(5, dtype=torch.long).repeat_interleave(101)
        batched_progress = progresses.repeat(5)
        translating = torch.ones_like(shapes, dtype=torch.bool)
        actual = env._translation_path_point_torch(
            shapes, batched_progress, translating
        ).reshape(5, 101, 2)
        expected = torch.tensor(
            [
                [
                    contract.translation_path_point_xy_m(
                        shape_index, progress_index / 100
                    )
                    for progress_index in range(101)
                ]
                for shape_index in range(5)
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-12)

    def test_rotation_rows_zero_translation_path_targets(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        shapes = torch.arange(5, dtype=torch.long)
        progress = torch.linspace(0.0, 1.0, 5)
        translating = torch.zeros(5, dtype=torch.bool)
        actual = env._translation_path_point_torch(
            shapes, progress, translating
        )
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))


if __name__ == "__main__":
    unittest.main()
