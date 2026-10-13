#!/usr/bin/env python3
"""Small deterministic tests for the R6 workspace scanner helpers."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import R6_rotation_translation_workspace as workspace
from R6_kinematics import FINGERS, GripperIK


class R6RotationWorkspaceHelpersTest(unittest.TestCase):
    def test_gripper6_model_contract_and_linear_stages(self) -> None:
        ik = GripperIK.from_xml()
        self.assertEqual(len(ik.ctrl_ranges), 16)
        self.assertEqual(tuple(ik.fingers), FINGERS)
        for finger in FINGERS:
            self.assertEqual(
                ik.fingers[finger].active_ranges["linear"],
                (-0.05, 0.0),
            )

    def test_workspace_scanner_applies_root_height_shift(self) -> None:
        cfg = workspace.build_config(
            workspace.parse_args(
                [
                    "--root-height",
                    "0.600",
                    "--z-min",
                    "0.40",
                    "--z-max",
                    "0.40",
                ]
            )
        )
        original = GripperIK.from_xml()
        scanner = workspace.WorkspaceScanner(cfg)
        for finger in FINGERS:
            delta = (
                scanner.ik.fingers[finger].base_world
                - original.fingers[finger].base_world
            )
            np.testing.assert_allclose(
                delta,
                (0.0, 0.0, 0.075),
                atol=1e-12,
            )

    def test_rotation_vector_single_x_90(self) -> None:
        matrix = workspace.rotation_vector_matrix((90.0, 0.0, 0.0))
        expected = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=float,
        )
        np.testing.assert_allclose(matrix, expected, atol=1e-12)

    def test_signed_case_counts_and_zero_deduplication(self) -> None:
        self.assertEqual(
            len(workspace.orientation_matrices((0,), 20.0, "all")),
            2,
        )
        self.assertEqual(
            len(workspace.orientation_matrices((0, 1), 20.0, "all")),
            4,
        )
        self.assertEqual(
            len(workspace.orientation_matrices((0, 1, 2), 20.0, "all")),
            8,
        )
        self.assertEqual(
            len(workspace.orientation_matrices((0, 1, 2), 0.0, "all")),
            1,
        )

    def test_theta_grid_requested_step(self) -> None:
        values = workspace.theta_values(9.0)
        self.assertEqual(len(values), 40)
        self.assertEqual(float(values[0]), 0.0)
        self.assertEqual(float(values[-1]), 351.0)

    def test_pointwise_axis_aggregation(self) -> None:
        rows = []
        radii = {
            "X": 0.12,
            "Y": 0.10,
            "Z": 0.11,
            "XY": 0.08,
            "XZ": 0.09,
            "YZ": 0.07,
            "XYZ": 0.06,
        }
        for label, radius in radii.items():
            rows.append(
                workspace.SurfacePoint(
                    axis_count=len(workspace.COMBO_TO_AXES[label]),
                    axis_combination=label,
                    angle_deg=30.0,
                    z_m=0.4,
                    max_inner_circle_r_m=radius,
                    limiting_theta_deg=0.0,
                    center_pose_feasible=True,
                    search_limit_reached=False,
                    pose_checks=1,
                )
            )
        aggregate = workspace.aggregate_points(rows)
        by_count = {row.axis_count: row for row in aggregate}
        self.assertTrue(
            math.isclose(by_count[1].max_inner_circle_r_m, 0.10)
        )
        self.assertEqual(by_count[1].limiting_axis_combinations, "Y")
        self.assertTrue(
            math.isclose(by_count[2].max_inner_circle_r_m, 0.07)
        )
        self.assertEqual(by_count[2].limiting_axis_combinations, "YZ")
        self.assertTrue(
            math.isclose(by_count[3].max_inner_circle_r_m, 0.06)
        )

    def test_incomplete_checkpoint_tail_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_bytes(b'{"complete":true}\n{"partial":')
            self.assertTrue(
                workspace.repair_incomplete_checkpoint_tail(path)
            )
            self.assertEqual(path.read_bytes(), b'{"complete":true}\n')
            self.assertFalse(
                workspace.repair_incomplete_checkpoint_tail(path)
            )

    def test_incomplete_checkpoint_task_is_rejected(self) -> None:
        args = workspace.parse_args(
            [
                "--z-min",
                "0.42",
                "--z-max",
                "0.42",
                "--angle-min-deg",
                "0",
                "--angle-max-deg",
                "1",
            ]
        )
        cfg = workspace.build_config(args)
        incomplete = workspace.ComboLayerResult(
            axis_combination="X",
            z_m=0.42,
            points=(),
            target_cache_hits=0,
            target_cache_misses=0,
        )
        self.assertFalse(
            workspace.checkpoint_result_is_valid(
                incomplete,
                cfg,
                (0.0, 1.0),
                {("X", 0.42)},
            )
        )


if __name__ == "__main__":
    unittest.main()
