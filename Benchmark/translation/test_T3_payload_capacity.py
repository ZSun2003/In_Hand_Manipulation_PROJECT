#!/usr/bin/env python3
"""Focused regression tests for the analytic payload calculator."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from T3_kinematics import FINGERS, TIP_OFFSETS_UNIT, GripperIK
from T3_payload_capacity import (
    MOTOR_SPECS,
    build_pick_move_place_path,
    build_vertical_path,
    calculate_capacity_rows,
    calculate_spatial_capacity_rows,
    estimate_circular_contact_patch,
    maximise_spatial_payload,
    maximise_symmetric_contact_force,
    planar_active_jacobian,
)


class PayloadCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ik = GripperIK.from_xml(MODULE_DIR / "Project.xml")
        result = cls.ik.solve_cube_pose((0.0, 0.0, 0.285), strict=True)
        cls.solution = result.finger_solutions["R"]
        cls.geometry = cls.ik.fingers["R"]
        cls.contact_patch = estimate_circular_contact_patch(
            cls.ik.model_info.hard_tip_radius,
            0.020,
        )
        cls.contact_distance = (
            cls.ik.model_info.cube_half_size
            + cls.contact_patch.center_to_contact_plane_m
        )
        patch_result = cls.ik.solve_cube_pose(
            (0.0, 0.0, 0.285),
            distance=cls.contact_distance,
            strict=True,
        )
        cls.patch_solution = patch_result.finger_solutions["R"]

    def test_twenty_mm_contact_patch_geometry(self) -> None:
        self.assertAlmostEqual(self.contact_patch.area_m2, math.pi * 1e-4)
        self.assertAlmostEqual(
            self.contact_patch.center_to_contact_plane_m,
            0.012489995996796796,
        )
        self.assertAlmostEqual(
            self.contact_patch.estimated_indentation_m,
            0.003510004003203204,
        )

    def test_analytic_jacobian_matches_central_difference_of_ik(self) -> None:
        actual = planar_active_jacobian(
            self.geometry,
            self.solution,
        ).matrix_m_per_rad
        tip = np.array([0.041, 0.0, 0.285], dtype=float)
        step = 1e-6
        inverse_jacobian = np.zeros((2, 2), dtype=float)
        perturbations = (
            np.array([step, 0.0, 0.0]),
            np.array([0.0, 0.0, step]),
        )
        for column, perturbation in enumerate(perturbations):
            plus = self.ik.solve_finger_target(
                "R",
                tip + perturbation,
                previous_solution=self.solution,
            )
            minus = self.ik.solve_finger_target(
                "R",
                tip - perturbation,
                previous_solution=self.solution,
            )
            self.assertIsNotNone(plus)
            self.assertIsNotNone(minus)
            inverse_jacobian[:, column] = (
                np.array([plus.q_back, plus.q_front])
                - np.array([minus.q_back, minus.q_front])
            ) / (2.0 * step)
        expected = np.linalg.inv(inverse_jacobian)
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-8)

    def test_nominal_mu_one_point_two_payload_regression(self) -> None:
        jacobian = planar_active_jacobian(
            self.geometry,
            self.patch_solution,
            contact_offset_planar_m=(
                -self.contact_patch.center_to_contact_plane_m,
                0.0,
            ),
        ).matrix_m_per_rad
        capacity = maximise_symmetric_contact_force(
            jacobian,
            torque_limit_nm=0.28,
            friction_coefficient=1.2,
        )
        self.assertAlmostEqual(capacity.normal_force_n, 0.3567186477, places=8)
        self.assertAlmostEqual(capacity.upward_force_n, 0.4280623773, places=8)
        self.assertAlmostEqual(capacity.payload_mass_kg, 0.1746008585, places=8)
        self.assertEqual(capacity.limiting_motor, "back")

    def test_capacity_scales_linearly_with_motor_torque(self) -> None:
        jacobian = planar_active_jacobian(
            self.geometry,
            self.patch_solution,
            contact_offset_planar_m=(
                -self.contact_patch.center_to_contact_plane_m,
                0.0,
            ),
        ).matrix_m_per_rad
        low = maximise_symmetric_contact_force(
            jacobian,
            torque_limit_nm=0.28,
            friction_coefficient=0.8,
        )
        high = maximise_symmetric_contact_force(
            jacobian,
            torque_limit_nm=1.40,
            friction_coefficient=0.8,
        )
        self.assertAlmostEqual(
            high.payload_mass_kg / low.payload_mass_kg,
            5.0,
            places=10,
        )

    def test_default_motor_comparison_and_vertical_path_bottleneck(self) -> None:
        poses = build_vertical_path(
            self.ik,
            start_z_m=0.285,
            end_z_m=0.400,
            samples=24,
            contact_distance_m=self.contact_distance,
            contact_plane_offset_m=self.contact_patch.center_to_contact_plane_m,
            condition_limit=1e4,
        )
        rows = calculate_capacity_rows(
            poses,
            motors=(MOTOR_SPECS["xl"], MOTOR_SPECS["xc"]),
            requested_voltage_v=None,
            torque_mode="both",
            friction_coefficient=1.2,
            safety_factor=1.0,
            gravity_m_s2=9.80665,
            max_normal_force_n=None,
        )
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertAlmostEqual(row.bottleneck_pose.object_center_z_m, 0.285)
        payloads = {
            (row.motor.key, row.torque_mode): row.contact.payload_mass_kg
            for row in rows
        }
        self.assertAlmostEqual(payloads[("xl", "continuous")], 0.1746008585, places=8)
        self.assertAlmostEqual(payloads[("xc", "continuous")], 0.2369583080, places=8)
        self.assertAlmostEqual(payloads[("xl", "stall")], 0.8730042925, places=8)
        self.assertAlmostEqual(payloads[("xc", "stall")], 1.1847915398, places=8)

    def test_spatial_lp_reduces_exactly_to_centered_static_result(self) -> None:
        poses = build_pick_move_place_path(
            self.ik,
            start_world_m=(0.0, 0.0, 0.285),
            target_world_m=(0.0, 0.0, 0.285),
            transit_z_m=0.285,
            spacing_m=0.005,
            contact_distance_m=self.contact_distance,
            contact_plane_offset_m=self.contact_patch.center_to_contact_plane_m,
            condition_limit=1e4,
        )
        self.assertEqual(len(poses), 1)
        capacity = maximise_spatial_payload(
            poses[0],
            torque_limit_nm=0.28,
            friction_coefficient=1.2,
            object_half_width_m=self.ik.model_info.cube_half_size,
        )
        self.assertAlmostEqual(capacity.payload_mass_kg, 0.1746008585, places=8)

        total_force = sum(
            capacity.contact_forces_world_n[finger]
            for finger in FINGERS
        )
        np.testing.assert_allclose(
            total_force,
            (0.0, 0.0, capacity.payload_mass_kg * 9.80665),
            atol=2e-9,
            rtol=0.0,
        )
        total_moment = sum(
            np.cross(
                TIP_OFFSETS_UNIT[finger] * self.ik.model_info.cube_half_size,
                capacity.contact_forces_world_n[finger],
            )
            for finger in FINGERS
        )
        np.testing.assert_allclose(total_moment, np.zeros(3), atol=2e-10)
        self.assertLessEqual(capacity.maximum_friction_utilisation, 1.0 + 1e-9)
        for torques in capacity.motor_torques_nm.values():
            self.assertLessEqual(max(abs(value) for value in torques), 0.28 + 1e-9)

    def test_pick_move_place_uses_off_center_route_bottleneck(self) -> None:
        poses = build_pick_move_place_path(
            self.ik,
            start_world_m=(0.0, 0.0, 0.285),
            target_world_m=(0.020, 0.0, 0.285),
            transit_z_m=0.310,
            spacing_m=0.001,
            contact_distance_m=self.contact_distance,
            contact_plane_offset_m=self.contact_patch.center_to_contact_plane_m,
            condition_limit=1e4,
        )
        rows = calculate_spatial_capacity_rows(
            poses,
            motors=(MOTOR_SPECS["xl"],),
            requested_voltage_v=None,
            torque_mode="continuous",
            friction_coefficient=1.2,
            safety_factor=1.0,
            gravity_m_s2=9.80665,
            object_half_width_m=self.ik.model_info.cube_half_size,
            max_normal_force_n=None,
            friction_facets=16,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row.contact.payload_mass_kg, 0.1489607919, places=8)
        np.testing.assert_allclose(
            row.bottleneck_pose.object_center_world_m,
            (0.020, 0.0, 0.285),
            atol=1e-12,
            rtol=0.0,
        )
        self.assertLess(
            row.contact.payload_mass_kg,
            0.1746008585,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
