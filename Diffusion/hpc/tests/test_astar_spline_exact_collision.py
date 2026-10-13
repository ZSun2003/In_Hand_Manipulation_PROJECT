from __future__ import annotations

import unittest

import numpy as np
import torch

from screw_control import (
    DIRECT_ALIGN_STEPS,
    PREINSERT_GATE_STEP,
    SCRIPTED_PREFIX_STEPS,
)
from s_hpc.astar_spline_exact_collision import (
    COLLISION_MODEL,
    GLOBAL_PLANNER_CONTRACT,
    GLOBAL_POSE_LAYERS,
    ExactCollisionBatchedAStarSplinePlanner,
)
from s_hpc.config import HPCConfig
from s_hpc.supervisor import BatchedSupervisor


class ExactCollisionGlobalAStarPlannerTest(unittest.TestCase):
    def test_one_shot_global_full_align_paths_are_exactly_certified(self):
        supervisor = BatchedSupervisor(device="cpu")
        # The opening-facing start has direct line of sight.  The rear-side
        # start is separated from the goal by the socket and must detour.
        yaw_deg = np.asarray((0.0, 45.0, 90.0, 137.0))
        yaw = np.deg2rad(yaw_deg)
        inward = np.column_stack((np.cos(yaw), np.sin(yaw)))
        tangent = np.column_stack((-np.sin(yaw), np.cos(yaw)))
        canonical_spawn = np.asarray(
            (
                (-0.07071068, -0.07071068),
                (-0.07071068, -0.07071068),
                (0.07313537, -0.06819984),
                (0.07313537, -0.06819984),
            )
        )
        spawn_xy = (
            canonical_spawn[:, :1] * inward
            + canonical_spawn[:, 1:] * tangent
        )
        mouth_points = 0.020 * inward
        # Formal paired-evaluation world 30 is a regression for a LOS segment
        # that is collision-free in float64 but penetrates socket wall 06 by
        # 0.129 micrometres after conversion to the executed float32 world
        # commands.  It must use the same certified A* lattice polyline rather
        # than relaxing the exact-contact oracle.
        spawn_xy = np.concatenate(
            (spawn_xy, [[0.03908604755997658, -0.0670032799243927]])
        )
        mouth_points = np.concatenate(
            (mouth_points, [[-0.016690384596586227, -0.011019576340913773]])
        )
        inward = mouth_points / np.linalg.norm(
            mouth_points, axis=1, keepdims=True
        )
        tangent = np.column_stack((-inward[:, 1], inward[:, 0]))
        spawn = torch.tensor(spawn_xy, dtype=torch.float32)
        mouth_xy = torch.tensor(mouth_points, dtype=torch.float32)
        angle = torch.full((5,), 45.0)
        start = supervisor.at(
            torch.full((5,), SCRIPTED_PREFIX_STEPS),
            spawn,
            angle,
            mouth_xy,
        )
        goal = supervisor.at(
            torch.full((5,), PREINSERT_GATE_STEP),
            spawn,
            angle,
            mouth_xy,
        )
        _, axis, mouth, _ = supervisor.socket_geometry(angle, mouth_xy)
        planner = ExactCollisionBatchedAStarSplinePlanner(
            HPCConfig(), worlds=5, device="cpu"
        )
        try:
            planner.reset(
                torch.ones(5, dtype=torch.bool),
                start_command=start,
                goal_command=goal,
                socket_mouth=mouth,
                socket_axis=axis,
            )
            diagnostics = planner.diagnostics()
            self.assertEqual(diagnostics["collision_model"], COLLISION_MODEL)
            self.assertEqual(
                diagnostics["planner_contract"], GLOBAL_PLANNER_CONTRACT
            )
            self.assertEqual(diagnostics["pose_layers"], GLOBAL_POSE_LAYERS)
            self.assertEqual(diagnostics["align_steps"], DIRECT_ALIGN_STEPS)
            self.assertFalse(diagnostics["staging_waypoint"])
            self.assertGreater(diagnostics["exact_colliding_pose_nodes"], 0)
            self.assertGreater(diagnostics["exact_free_pose_nodes"], 0)
            self.assertEqual(diagnostics["plan_found"].tolist(), [True] * 5)
            self.assertEqual(diagnostics["float32_shortcut_fallbacks"], 1)

            provenance = planner.provenance()
            self.assertEqual(
                provenance["schema"],
                "screw-global-full-align-astar-planner-v1",
            )
            self.assertEqual(
                provenance["planner_contract"], GLOBAL_PLANNER_CONTRACT
            )
            self.assertEqual(provenance["obstacle"]["model"], COLLISION_MODEL)
            self.assertEqual(
                provenance["obstacle"]["clearance_inflation_m"], 0.0
            )
            self.assertEqual(provenance["search"]["plans_per_episode"], 1)
            self.assertFalse(provenance["search"]["replanning"])
            self.assertFalse(provenance["postprocess"]["cubic_spline"])
            self.assertTrue(
                provenance["postprocess"][
                    "float32_shortcut_fallback_enabled"
                ]
            )
            self.assertFalse(
                provenance["postprocess"]["fallback_planner"]
            )
            self.assertFalse(provenance["schedule"]["staging_waypoint"])
            self.assertFalse(
                provenance["schedule"][
                    "fixed_transport_rotate_dock_segments"
                ]
            )

            plan_tip = planner._plan_tip.detach().cpu().numpy()
            plan_rotation = planner._plan_rotation.detach().cpu().numpy()
            np.testing.assert_allclose(
                plan_tip[:, -1], goal.tip.numpy(), atol=1.0e-7
            )
            np.testing.assert_allclose(
                plan_rotation[:, -1], goal.rotation.numpy(), atol=2.0e-6
            )
            self.assertTrue(
                np.all(np.diff(plan_tip[:, :, 2], axis=1) >= -1.0e-8)
            )

            # Every final 350-tick command chord must remain inside the 100 mm
            # workspace and pass the full-pose MuJoCo swept-contact oracle.
            # Canonical screw axes must also match the lattice schedule at
            # several nonzero socket yaws (the old per-world SLERP did not).
            z0 = float(start.tip[0, 2])
            z1 = float(goal.tip[0, 2])
            for world in range(5):
                yaw_frame = planner._yaw_frame(inward[world], tangent[world])
                actual_tips = np.concatenate(
                    (start.tip[world].numpy()[None, :], plan_tip[world]), axis=0
                )
                actual_rotations = np.concatenate(
                    (
                        start.rotation[world].numpy()[None, :, :],
                        plan_rotation[world],
                    ),
                    axis=0,
                )
                canonical_tips = actual_tips.copy()
                canonical_tips[:, 0] = actual_tips[:, :2] @ inward[world]
                canonical_tips[:, 1] = actual_tips[:, :2] @ tangent[world]
                canonical_rotations = np.einsum(
                    "ji,njk->nik", yaw_frame, actual_rotations
                )
                self.assertTrue(
                    planner._actual_command_sweep_free(
                        canonical_tips, canonical_rotations
                    )
                )
                progress = (plan_tip[world, :, 2] - z0) / (z1 - z0)
                certified_axis = planner._rotations_at(progress)[:, :, 2]
                np.testing.assert_allclose(
                    canonical_rotations[1:, :, 2],
                    certified_axis,
                    atol=4.0e-6,
                )

            path_length = diagnostics["path_length_m"].numpy()
            chord_length = np.linalg.norm(
                goal.tip[:, :2].numpy() - spawn.numpy(), axis=1
            )
            self.assertTrue(np.all(path_length[:2] / chord_length[:2] < 1.01))
            self.assertTrue(
                np.all(path_length[2:4] / chord_length[2:4] > 1.15)
            )
            self.assertTrue(
                bool((diagnostics["task_translation_limit_fraction"] <= 1.0).all())
            )
            self.assertTrue(
                bool((diagnostics["task_rotation_limit_fraction"] <= 1.0).all())
            )

            # Guard against collapsing full-pose collision checking back to a
            # fixed upright XY mask.
            exact = planner._exact_collision
            self.assertIsNotNone(exact)
            pose_dependent = np.argwhere(
                planner._global_free[0] != planner._global_free[-1]
            )
            self.assertGreater(len(pose_dependent), 0)
            i, j = pose_dependent[0]
            point_xy = planner._grid_xy(int(i), int(j))
            endpoint_tips = np.asarray(
                (
                    (point_xy[0], point_xy[1], z0),
                    (point_xy[0], point_xy[1], z1),
                ),
                dtype=np.float64,
            )
            endpoint_rotations = np.stack(
                (planner._pose_rotations[0], planner._pose_rotations[-1])
            )
            endpoint_collision = exact.collides_poses(
                endpoint_tips, endpoint_rotations
            )
            np.testing.assert_array_equal(
                endpoint_collision,
                np.asarray(
                    (
                        not planner._global_free[0, i, j],
                        not planner._global_free[-1, i, j],
                    )
                ),
            )
        finally:
            planner.close()


if __name__ == "__main__":
    unittest.main()
