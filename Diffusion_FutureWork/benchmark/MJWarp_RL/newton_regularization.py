"""Pinned local-tile stabilization for formal MJWarp dense Newton solves.

This module deliberately imports neither Warp nor MuJoCo-Warp at module import
time.  The formal Linux/CUDA runtime installs the patch lazily after the exact
pinned solver source has been verified; CPU-only development and unit tests can
therefore inspect the contract without those optional packages.
"""

import hashlib
import inspect
from pathlib import Path
from typing import Any, Mapping


NEWTON_REGULARIZATION_SCHEMA = (
    "grab-mjwarp-dense-newton-tile-regularization-v1"
)
PINNED_SOLVER_SOURCE_SHA256 = (
    "f07a7c33e038a312ea5c8cab4ec718b24d5e7d163dc8435a592118c9fc0e07fb"
)
PINNED_BLOCK_CHOLESKY_DIM = 32
NEWTON_RELATIVE_DIAGONAL_SHIFT = 1.0e-6
NEWTON_ABSOLUTE_DIAGONAL_SHIFT = 1.0e-8
PATCH_TARGET = "mujoco_warp._src.solver._update_gradient_cholesky"

# Bound only by ``install_formal_newton_regularization``.  Keeping the names
# module-global matches MJWarp's own cached-kernel source/codegen model without
# importing the optional formal runtime on macOS or other development hosts.
wp: Any = None
solve_search_sums: Any = None
cache_kernel: Any = None


_FORMAL_METADATA = {
    "newton_regularization_schema": NEWTON_REGULARIZATION_SCHEMA,
    "newton_regularization_applied": True,
    "newton_regularization_target": PATCH_TARGET,
    "newton_regularization_solver_source_sha256": (
        PINNED_SOLVER_SOURCE_SHA256
    ),
    "newton_regularization_block_cholesky_dim": (
        PINNED_BLOCK_CHOLESKY_DIM
    ),
    "newton_regularization_max_nv": PINNED_BLOCK_CHOLESKY_DIM,
    "newton_regularization_relative_diagonal_shift": (
        NEWTON_RELATIVE_DIAGONAL_SHIFT
    ),
    "newton_regularization_absolute_diagonal_shift": (
        NEWTON_ABSOLUTE_DIAGONAL_SHIFT
    ),
    "newton_regularization_scale": "max-absolute-local-tile",
    "newton_regularization_storage": "register-local-tile-only",
}


class NewtonRegularizationError(RuntimeError):
    """The pinned formal solver cannot be patched exactly and safely."""


def formal_newton_regularization_metadata() -> dict[str, object]:
    """Return a fresh canonical health/compatibility metadata mapping."""

    return dict(_FORMAL_METADATA)


def formal_newton_regularization_metadata_matches(
    value: Mapping[str, Any],
) -> bool:
    """Require every canonical field with its exact value and Python type."""

    if not isinstance(value, Mapping):
        return False
    return all(
        key in value
        and type(value[key]) is type(expected)
        and value[key] == expected
        for key, expected in _FORMAL_METADATA.items()
    )


def _verify_solver_source(solver_module: Any) -> None:
    source = getattr(solver_module, "__file__", None)
    if not source:
        raise NewtonRegularizationError(
            "formal MJWarp solver module has no source path"
        )
    digest = hashlib.sha256(Path(source).resolve().read_bytes()).hexdigest()
    if digest != PINNED_SOLVER_SOURCE_SHA256:
        raise NewtonRegularizationError(
            "formal MJWarp solver.py SHA-256 mismatch: "
            f"expected {PINNED_SOLVER_SOURCE_SHA256}, found {digest}"
        )
    block_dim = getattr(solver_module, "_BLOCK_CHOLESKY_DIM", None)
    if type(block_dim) is not int or block_dim != PINNED_BLOCK_CHOLESKY_DIM:
        raise NewtonRegularizationError(
            "formal MJWarp _BLOCK_CHOLESKY_DIM mismatch: "
            f"expected {PINNED_BLOCK_CHOLESKY_DIM}, found {block_dim!r}"
        )


def _verify_factory_signature(factory: Any) -> None:
    try:
        parameters = tuple(inspect.signature(factory).parameters.values())
    except (TypeError, ValueError) as exc:
        raise NewtonRegularizationError(
            "cannot inspect MJWarp _update_gradient_cholesky signature"
        ) from exc
    if not (
        len(parameters) == 2
        and parameters[0].name == "tile_size"
        and parameters[0].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameters[1].name == "skip_noflip"
        and parameters[1].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameters[1].default is False
    ):
        raise NewtonRegularizationError(
            "MJWarp _update_gradient_cholesky signature changed"
        )


def _define_regularized_factory() -> Any:
    """Define the cached factory after formal runtime globals are bound."""

    @cache_kernel
    def regularized_update_gradient_cholesky(
        tile_size: int, skip_noflip: bool = False
    ):
        skip_noflip_static = skip_noflip

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            # In:
            ctx_grad_in: wp.array2d[float],
            h_in: wp.array3d[float],
            state_changed_count_in: wp.array[int],
            ctx_done_in: wp.array[bool],
            # Out:
            ctx_search_out: wp.array2d[float],
            ctx_search_dot_out: wp.array[float],
            ctx_newton_decrement_out: wp.array[float],
        ):
            worldid = wp.tid()
            tile_size_static = wp.static(tile_size)

            if ctx_done_in[worldid]:
                return
            if wp.static(skip_noflip_static):
                if state_changed_count_in[worldid] == 0:
                    return

            mat_tile = wp.tile_load(
                h_in[worldid],
                shape=(tile_size_static, tile_size_static),
            )
            abs_tile = wp.tile_map(wp.abs, mat_tile)
            scale = wp.tile_max(abs_tile)[0]
            shift = wp.max(
                wp.float32(1.0e-8),
                wp.float32(1.0e-6) * scale,
            )
            diagonal = wp.tile_full(
                tile_size_static,
                value=shift,
                dtype=wp.float32,
                storage="register",
            )
            mat_tile = wp.tile_diag_add(mat_tile, diagonal)
            wp.tile_cholesky_inplace(mat_tile, fill_mode="upper")
            input_tile = wp.tile_load(
                ctx_grad_in[worldid], shape=tile_size_static
            )
            output_tile = wp.tile_cholesky_solve(
                mat_tile, input_tile, fill_mode="upper"
            )
            sums = wp.tile_reduce(
                wp.add,
                wp.tile_map(solve_search_sums, input_tile, output_tile),
            )[0]
            ctx_search_dot_out[worldid] = sums[0]
            ctx_newton_decrement_out[worldid] = sums[1]
            wp.tile_store(
                ctx_search_out[worldid],
                wp.tile_map(wp.mul, output_tile, -1.0),
            )

        return kernel

    return regularized_update_gradient_cholesky


def install_formal_newton_regularization(
    solver_module: Any | None = None,
) -> dict[str, object]:
    """Install the pinned cached factory, idempotently, for formal CUDA use.

    Only the tile Cholesky factory is replaced.  The Hessian remains untouched
    in global memory: each kernel loads its local tile, computes a scale-aware
    diagonal shift in registers, factorizes that shifted tile, and discards it.
    """

    if solver_module is None:
        from mujoco_warp._src import solver as solver_module

    _verify_solver_source(solver_module)
    current = getattr(solver_module, "_update_gradient_cholesky", None)
    if current is None:
        raise NewtonRegularizationError(
            "formal MJWarp solver lacks _update_gradient_cholesky"
        )
    marker = getattr(
        current, "_grab_newton_regularization_metadata", None
    )
    if marker is not None:
        if not formal_newton_regularization_metadata_matches(marker):
            raise NewtonRegularizationError(
                "an incompatible Newton regularization is already installed"
            )
        return formal_newton_regularization_metadata()

    _verify_factory_signature(current)
    try:
        import warp as wp_runtime
        from mujoco_warp._src.solver import (
            solve_search_sums as solve_search_sums_runtime,
        )
        from mujoco_warp._src.warp_util import (
            cache_kernel as cache_kernel_runtime,
        )
    except ImportError as exc:
        raise NewtonRegularizationError(
            "formal Newton regularization needs Warp and MuJoCo-Warp"
        ) from exc

    global wp, solve_search_sums, cache_kernel
    wp = wp_runtime
    solve_search_sums = solve_search_sums_runtime
    cache_kernel = cache_kernel_runtime
    regularized_update_gradient_cholesky = _define_regularized_factory()

    regularized_update_gradient_cholesky._grab_newton_regularization_metadata = (
        formal_newton_regularization_metadata()
    )
    solver_module._update_gradient_cholesky = (
        regularized_update_gradient_cholesky
    )
    if (
        solver_module._update_gradient_cholesky
        is not regularized_update_gradient_cholesky
    ):
        raise NewtonRegularizationError(
            "formal MJWarp Newton factory replacement did not stick"
        )
    return formal_newton_regularization_metadata()


__all__ = [
    "NEWTON_ABSOLUTE_DIAGONAL_SHIFT",
    "NEWTON_REGULARIZATION_SCHEMA",
    "NEWTON_RELATIVE_DIAGONAL_SHIFT",
    "NewtonRegularizationError",
    "PATCH_TARGET",
    "PINNED_BLOCK_CHOLESKY_DIM",
    "PINNED_SOLVER_SOURCE_SHA256",
    "formal_newton_regularization_metadata",
    "formal_newton_regularization_metadata_matches",
    "install_formal_newton_regularization",
]
