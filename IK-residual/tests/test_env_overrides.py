"""Focused contracts for evaluation-only XML and object-spec overrides."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from g3g6 import config
from g3g6.env import G3G6ManipulationEnv
from g3g6.objects import OBJECTS


class EvaluationOverrideTests(unittest.TestCase):
    def _copy_g3_xml_bundle(self, destination: Path) -> Path:
        source = config.MODEL_XML["g3"].parent
        destination.mkdir(parents=True)
        for name in ("Project.xml", "Share.xml", "Gripper.xml"):
            shutil.copy2(source / name, destination / name)
        (destination / "mesh").symlink_to(source / "mesh", target_is_directory=True)
        return destination / "Project.xml"

    def test_custom_xml_and_fixed_cube_spec_overlay_drive_reset(self) -> None:
        cube = OBJECTS["cube"]
        override = replace(
            cube,
            dimensions_m=(0.080, 0.080, 0.100),
            rest_grasp_plane_z_m=cube.rest_grasp_plane_z_m + 0.001,
        )
        with tempfile.TemporaryDirectory() as temporary:
            custom_xml = self._copy_g3_xml_bundle(Path(temporary) / "G3")
            env = G3G6ManipulationEnv(
                gripper="g3",
                object_name="cube",
                model_xml_path=custom_xml,
                object_spec_override=override,
            )
            try:
                self.assertEqual(env.model_xml_path, custom_xml.resolve())
                self.assertIs(env._objects["cube"], override)
                self.assertIs(env._objects["cone"], OBJECTS["cone"])

                # The dynamics model uses the caller's XML, while analytic IK
                # remains tied to the authoritative robot XML in config.
                self.assertEqual(
                    env.kinematics.ik.model_info.project_xml,
                    config.MODEL_XML["g3"].resolve(),
                )

                _observation, _info = env.reset(seed=7)
                self.assertIs(env.object_spec, override)
                joint_id = env._object_joint_ids["cube"]
                qpos_address = int(env.model.jnt_qposadr[joint_id])
                self.assertAlmostEqual(
                    float(env.data.qpos[qpos_address + 2]),
                    override.rest_grasp_plane_z_m,
                    delta=1.0e-12,
                )
            finally:
                env.close()

    def test_default_construction_uses_registered_xml_and_specs(self) -> None:
        env = G3G6ManipulationEnv(gripper="g3", object_name="cube")
        try:
            self.assertEqual(env.model_xml_path, config.MODEL_XML["g3"].resolve())
            self.assertIs(env._objects["cube"], OBJECTS["cube"])
        finally:
            env.close()

    def test_override_requires_matching_fixed_registered_object(self) -> None:
        cube = OBJECTS["cube"]
        with self.assertRaisesRegex(ValueError, "requires a fixed object_name"):
            G3G6ManipulationEnv(object_spec_override=cube)
        with self.assertRaisesRegex(ValueError, "must match the fixed object_name"):
            G3G6ManipulationEnv(
                object_name="cube",
                object_spec_override=replace(cube, name="cone"),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
