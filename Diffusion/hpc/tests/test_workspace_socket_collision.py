from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for location in (PROJECT_ROOT, PROJECT_ROOT / "hpc"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from s_hpc.supervisor import BatchedSupervisor
from evaluation.workspace_policy_rollouts import (
    HPC_ROOT,
    build_polar_grid,
    classify_initial_screw_socket_contacts,
    socket_outer_obb_center_mask,
)


class WorkspaceSocketCollisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.supervisor = BatchedSupervisor(device="cpu", curriculum=None)
        cls.mouth = torch.tensor([[0.020, 0.0]], dtype=torch.float32)
        cls.angle = torch.tensor([45.0], dtype=torch.float32)
        frame, _, _, _ = cls.supervisor.socket_geometry(
            cls.angle, cls.mouth
        )
        position, quaternion = cls.supervisor.socket_mocap_pose(
            cls.angle, cls.mouth
        )
        cls.frame = frame[0].numpy()
        cls.position = position[0].numpy()
        cls.quaternion = quaternion[0].numpy()

    def classify(self, xy):
        return classify_initial_screw_socket_contacts(
            np.asarray(xy, dtype=np.float32),
            screw_center_z_m=(
                self.supervisor.config.screw_initial_center_z_m
            ),
            socket_position=self.position,
            socket_quaternion_wxyz=self.quaternion,
            xml_path=HPC_ROOT / "assets" / "Project_hpc.xml",
        )

    def test_exact_classifier_ignores_table_and_gripper_contacts(self):
        mask, provenance = self.classify(
            [[0.0, 0.0], [-0.075, -0.040], [-0.090, 0.0]]
        )
        self.assertEqual(mask.tolist(), [True, False, False])
        self.assertEqual(provenance["clearance_inflation_m"], 0.0)
        self.assertIn("table", provenance["ignored_contact_classes"])
        self.assertEqual(
            provenance["included_screw_geoms"],
            ["screw_head_collision", "screw_shaft_collision"],
        )
        self.assertEqual(len(provenance["included_socket_geoms"]), 25)

    def test_uninflated_obb_is_only_a_centre_envelope(self):
        grid = build_polar_grid(
            angular_bins=360, radial_bins=50, radius_m=0.100
        )
        xy = np.asarray(grid["xy"], dtype=np.float32).reshape(-1, 2)
        dimensions = (
            self.supervisor.config.socket_outer_width_m,
            self.supervisor.config.socket_outer_height_m,
            self.supervisor.config.socket_outer_length_m,
        )
        obb = socket_outer_obb_center_mask(
            xy,
            screw_center_z_m=(
                self.supervisor.config.screw_initial_center_z_m
            ),
            socket_center=self.position,
            socket_frame=self.frame,
            socket_outer_dimensions_m=dimensions,
        )
        exact, _ = self.classify(xy)
        self.assertEqual(int(obb.sum()), 7252)
        self.assertEqual(int(exact.sum()), 11012)
        self.assertEqual(int(np.logical_and(exact, ~obb).sum()), 3760)

    def test_exact_mask_is_tighter_than_legacy_inflated_reset_obb(self):
        grid = build_polar_grid(
            angular_bins=360, radial_bins=50, radius_m=0.100
        )
        xy = np.asarray(grid["xy"], dtype=np.float32).reshape(-1, 2)
        exact, _ = self.classify(xy)
        repeated_mouth = self.mouth.expand(len(xy), 2)
        legacy = self.supervisor.spawn_overlaps_socket(
            torch.from_numpy(xy),
            socket_mouth_xy=repeated_mouth,
            socket_angle_deg=self.angle.expand(len(xy)),
        ).numpy()
        self.assertEqual(int(legacy.sum()), 12796)
        self.assertEqual(int(exact.sum()), 11012)
        self.assertGreater(int((legacy & ~exact).sum()), 0)


if __name__ == "__main__":
    unittest.main()
