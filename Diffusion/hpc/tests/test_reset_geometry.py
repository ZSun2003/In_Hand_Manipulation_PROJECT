from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

import preflight
from screw_config import DEFAULT_CONFIG
from s_hpc.mjwarp_backend import _asset_reset_center_z


ROOT = Path(__file__).resolve().parents[1]


class ResetGeometryTest(unittest.TestCase):
    def test_gpu_preflight_has_global_task_geometry_contract(self):
        self.assertIs(preflight.DEFAULT_CONFIG, DEFAULT_CONFIG)

    def test_asset_qpos_holder_and_task_contract_share_one_height(self):
        model = mujoco.MjModel.from_xml_path(
            str(ROOT / "assets" / "Project_hpc.xml")
        )
        screw_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "screw_free"
        )
        holder_body = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "screw_start_holder_body",
        )
        index = SimpleNamespace(
            screw_qpos=int(model.jnt_qposadr[screw_joint]),
            holder_body=int(holder_body),
        )
        reset_z = _asset_reset_center_z(model, index)
        self.assertAlmostEqual(reset_z, 0.300)
        self.assertAlmostEqual(
            reset_z, DEFAULT_CONFIG.screw_initial_center_z_m
        )
        self.assertAlmostEqual(
            reset_z + DEFAULT_CONFIG.screw_tip_local_z_m,
            DEFAULT_CONFIG.table_top_z_m,
        )

    def test_reset_geometry_rejects_asset_drift(self):
        model = SimpleNamespace(
            qpos0=np.asarray((0.0, 0.0, 0.320, 1.0)),
            body_pos=np.asarray(((0.0, 0.0, 0.300),)),
        )
        index = SimpleNamespace(screw_qpos=0, holder_body=0)
        with self.assertRaisesRegex(ValueError, "reset geometry"):
            _asset_reset_center_z(model, index)

    def test_backend_no_longer_contains_the_wrong_height_literal(self):
        source = (
            ROOT / "s_hpc" / "mjwarp_backend.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("0.320", source)


if __name__ == "__main__":
    unittest.main()
