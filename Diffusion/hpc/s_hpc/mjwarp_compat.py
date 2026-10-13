"""Fail-closed compatibility settings for the pinned MuJoCo Warp runtime."""

from __future__ import annotations

from importlib import metadata
from types import ModuleType
from typing import Dict, Iterable


PINNED_MJWRAP_VERSION = "3.10.0.3"
UPSTREAM_EPA_HORIZON_EDGES = 24
RTX_PRO_6000_EPA_HORIZON_EDGES = 128


def _apply_epa_horizon(
    modules: Iterable[ModuleType],
    *,
    installed_version: str,
) -> Dict[str, object]:
    """Raise the EPA edge workspace without accepting an unknown runtime.

    MuJoCo Warp 3.10.0.3 allocates the EPA horizon from module-level Python
    constants when ``convex_narrowphase``/``flex_collision`` are called.  The
    Warp kernels receive the resulting array and use its runtime length, so
    changing these globals before the first simulation step expands only the
    workspace; it does not bypass overflow reporting or alter collision
    geometry.
    """

    if installed_version != PINNED_MJWRAP_VERSION:
        raise RuntimeError(
            "EPA compatibility is validated only for mujoco-warp "
            f"{PINNED_MJWRAP_VERSION}; found {installed_version}"
        )

    module_list = tuple(modules)
    if not module_list:
        raise RuntimeError("no MuJoCo Warp modules supplied for EPA setup")
    for module in module_list:
        value = getattr(module, "MJ_MAX_EPAHORIZON", None)
        if value not in {
            UPSTREAM_EPA_HORIZON_EDGES,
            RTX_PRO_6000_EPA_HORIZON_EDGES,
        }:
            raise RuntimeError(
                f"{module.__name__}.MJ_MAX_EPAHORIZON has unexpected "
                f"value {value!r}; refusing an unvalidated runtime"
            )

    for module in module_list:
        module.MJ_MAX_EPAHORIZON = RTX_PRO_6000_EPA_HORIZON_EDGES

    return {
        "mujoco_warp_version": installed_version,
        "epa_horizon_upstream_edges": UPSTREAM_EPA_HORIZON_EDGES,
        "epa_horizon_edges": RTX_PRO_6000_EPA_HORIZON_EDGES,
    }


def configure_mjwarp_epa_horizon() -> Dict[str, object]:
    """Apply and report the RTX PRO 6000 EPA workspace setting."""

    from mujoco_warp._src import collision_convex
    from mujoco_warp._src import collision_flex
    from mujoco_warp._src import types

    installed_version = metadata.version("mujoco-warp")
    return _apply_epa_horizon(
        (types, collision_convex, collision_flex),
        installed_version=installed_version,
    )
