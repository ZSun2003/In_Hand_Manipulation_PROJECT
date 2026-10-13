from __future__ import annotations

from types import ModuleType
import unittest

from s_hpc.mjwarp_compat import (
    PINNED_MJWRAP_VERSION,
    RTX_PRO_6000_EPA_HORIZON_EDGES,
    UPSTREAM_EPA_HORIZON_EDGES,
    _apply_epa_horizon,
)


class MJWarpCompatibilityTest(unittest.TestCase):
    @staticmethod
    def _module(name: str, value: int) -> ModuleType:
        module = ModuleType(name)
        module.MJ_MAX_EPAHORIZON = value
        return module

    def test_expands_all_runtime_module_globals(self):
        modules = [
            self._module("types", UPSTREAM_EPA_HORIZON_EDGES),
            self._module("collision_convex", UPSTREAM_EPA_HORIZON_EDGES),
            self._module("collision_flex", UPSTREAM_EPA_HORIZON_EDGES),
        ]
        result = _apply_epa_horizon(
            modules,
            installed_version=PINNED_MJWRAP_VERSION,
        )
        self.assertEqual(
            result["epa_horizon_edges"],
            RTX_PRO_6000_EPA_HORIZON_EDGES,
        )
        self.assertTrue(
            all(
                module.MJ_MAX_EPAHORIZON
                == RTX_PRO_6000_EPA_HORIZON_EDGES
                for module in modules
            )
        )

    def test_is_idempotent_in_one_process(self):
        module = self._module(
            "collision_convex",
            RTX_PRO_6000_EPA_HORIZON_EDGES,
        )
        _apply_epa_horizon(
            (module,),
            installed_version=PINNED_MJWRAP_VERSION,
        )
        self.assertEqual(
            module.MJ_MAX_EPAHORIZON,
            RTX_PRO_6000_EPA_HORIZON_EDGES,
        )

    def test_rejects_unvalidated_version(self):
        module = self._module("types", UPSTREAM_EPA_HORIZON_EDGES)
        with self.assertRaisesRegex(RuntimeError, "validated only"):
            _apply_epa_horizon(
                (module,),
                installed_version="3.10.0.4",
            )
        self.assertEqual(
            module.MJ_MAX_EPAHORIZON,
            UPSTREAM_EPA_HORIZON_EDGES,
        )

    def test_rejects_unknown_upstream_constant(self):
        module = self._module("types", 48)
        with self.assertRaisesRegex(RuntimeError, "unexpected value"):
            _apply_epa_horizon(
                (module,),
                installed_version=PINNED_MJWRAP_VERSION,
            )
