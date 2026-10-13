"""Native MuJoCo-Warp/Torch backend shared by Gripper3 and Gripper6.

The backend owns physics memory.  Torch receives zero-copy views of Warp
arrays; no JAX runtime, DLPack hand-off, or host rollout copy is involved.
All model-specific names and arrays arrive through ``PlatformSpec``.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import platform
from typing import Any, Callable, Iterator

import numpy as np

from . import contract


PINNED_RUNTIME = {
    "torch": "2.7.1",
    "mujoco": "3.10.0",
    "mujoco-warp": "3.10.0.3",
    "warp-lang": "1.14.0",
}
EMBEDDED_TEST_RUNTIME = {
    "mujoco": "3.11.0",
    "mujoco-mjx": "3.11.0",
    "warp-lang": "1.14.0",
}
UPSTREAM_EPA_HORIZON_EDGES = 24
RTX_PRO_6000_EPA_HORIZON_EDGES = 128


class MJWarpUnavailable(RuntimeError):
    """The native Linux/NVIDIA training runtime is unavailable."""


def _exact_version(distribution: str) -> str:
    try:
        value = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise MJWarpUnavailable(
            f"required package is missing: {distribution}"
        ) from exc
    return value


def _matches_runtime(contract: dict[str, str]) -> tuple[bool, dict[str, str]]:
    versions: dict[str, str] = {}
    for distribution, expected in contract.items():
        try:
            actual = _exact_version(distribution)
        except MJWarpUnavailable:
            return False, versions
        versions[distribution] = actual
        if actual.split("+", 1)[0] != expected:
            return False, versions
    return True, versions


def _configure_runtime(
    *, allow_embedded_fallback: bool
) -> tuple[Any, Any, Any, Any, dict[str, object]]:
    """Import production top-level MJWarp, or the explicit CPU-test fallback."""

    formal, versions = _matches_runtime(PINNED_RUNTIME)
    embedded = False
    newton_regularization: dict[str, object] = {}
    try:
        import torch
        import warp as wp
        if formal:
            import mujoco
            import mujoco_warp as mjw
            from mujoco_warp._src import collision_convex
            from mujoco_warp._src import collision_flex
            from mujoco_warp._src import solver as mjw_solver
            from mujoco_warp._src import types as mjw_types

            from .newton_regularization import (
                install_formal_newton_regularization,
            )

            newton_regularization = (
                install_formal_newton_regularization(mjw_solver)
            )
        elif allow_embedded_fallback:
            fallback, versions = _matches_runtime(EMBEDDED_TEST_RUNTIME)
            if not fallback:
                raise MJWarpUnavailable(
                    "CPU test fallback requires MuJoCo/MJX 3.11.0 and Warp 1.14.0"
                )
            import mujoco
            from mujoco.mjx import warp as mjx_warp
            from mujoco.mjx.third_party.mujoco_warp._src import (
                collision_convex,
            )
            from mujoco.mjx.third_party.mujoco_warp._src import collision_flex
            from mujoco.mjx.third_party.mujoco_warp._src import (
                types as mjw_types,
            )

            mjw = mjx_warp.mujoco_warp
            versions["torch"] = _exact_version("torch")
            embedded = True
        else:
            actual: dict[str, str] = {}
            for distribution in PINNED_RUNTIME:
                try:
                    actual[distribution] = importlib.metadata.version(
                        distribution
                    )
                except importlib.metadata.PackageNotFoundError:
                    actual[distribution] = "missing"
            raise MJWarpUnavailable(
                f"formal runtime must match {PINNED_RUNTIME}, found {actual}"
            )
    except ImportError as exc:
        raise MJWarpUnavailable(
            "native MJWarp needs Torch, MuJoCo and Warp"
        ) from exc

    modules = (mjw_types, collision_convex, collision_flex)
    for module in modules:
        value = getattr(module, "MJ_MAX_EPAHORIZON", None)
        if value not in {
            UPSTREAM_EPA_HORIZON_EDGES,
            RTX_PRO_6000_EPA_HORIZON_EDGES,
        }:
            raise MJWarpUnavailable(
                f"{module.__name__}.MJ_MAX_EPAHORIZON={value!r} is unvalidated"
            )
    for module in modules:
        module.MJ_MAX_EPAHORIZON = RTX_PRO_6000_EPA_HORIZON_EDGES
    compatibility: dict[str, object] = {
        "versions": versions,
        "epa_horizon_edges": RTX_PRO_6000_EPA_HORIZON_EDGES,
        "runtime_mode": (
            "embedded-mujoco-mjx-cpu-test"
            if embedded
            else "top-level-mujoco-warp-production"
        ),
        "embedded_runtime_fallback": embedded,
        **newton_regularization,
    }
    return torch, wp, mujoco, mjw, compatibility


@contextmanager
def _unique_cuda_index_writes(torch: Any, device_type: str) -> Iterator[None]:
    """Scope the PyTorch 2.7.1 Blackwell deterministic-index workaround."""

    enabled = bool(torch.are_deterministic_algorithms_enabled())
    if device_type != "cuda" or not enabled:
        yield
        return
    getter = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    warn_only = bool(getter()) if getter is not None else False
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


class MJWarpBackend:
    """Own one native MJWarp model and a CUDA-resident vector of worlds."""

    def __init__(
        self,
        config: Any,
        platform_spec: Any,
        *,
        worlds: int | None = None,
        device: str | None = None,
        require_linux: bool = True,
        require_cuda: bool = True,
    ) -> None:
        # Establish an idempotent cleanup surface before the first runtime or
        # CUDA allocation so fail-closed construction can release a partially
        # initialized backend.
        self.data: Any | None = None
        self.model: Any | None = None
        self.torch_stream: Any | None = None
        self.wp_stream: Any | None = None
        self._pre_step: Callable[[], None] | None = None
        self._graph: Any | None = None
        self._graph_ready = False
        self._jacobian_buffers: dict[str, dict[str, Any]] = {}
        self._views: dict[str, Any] = {}
        self._torch_to_warp_aliases: list[Any] = []
        self._reset_mask_wp: Any | None = None
        self.is_cuda = False
        torch, wp, mujoco, mjw, compatibility = _configure_runtime(
            allow_embedded_fallback=not require_cuda
        )
        if require_linux and platform.system() != "Linux":
            raise MJWarpUnavailable("native MJWarp training requires Linux")

        self.torch = torch
        self.wp = wp
        self.mujoco = mujoco
        self.mjw = mjw
        self.compatibility = compatibility
        self.config = config
        self.platform = platform_spec
        self.host_model = platform_spec.host_model
        self.compatibility.update(
            {
                "parent_model_fingerprint_sha256": str(
                    platform_spec.parent_model_fingerprint_sha256
                ),
                "parent_controller_fingerprint_sha256": str(
                    platform_spec.parent_controller_fingerprint_sha256
                ),
            }
        )
        self.worlds = int(worlds or config.num_envs)
        self.physics_substeps = int(
            config.physics_substeps_per_control_tick
        )
        self.contacts_per_world = int(config.contacts_per_world)
        self.constraints_per_world = int(config.constraints_per_world)
        self.cuda_graph_enabled = bool(
            getattr(config, "cuda_graph", True)
        )
        if min(
            self.worlds,
            self.physics_substeps,
            self.contacts_per_world,
            self.constraints_per_world,
        ) <= 0:
            raise ValueError("world and native MJWarp capacities must be positive")

        self.device = torch.device(
            device or getattr(config, "device", "cuda:0")
        )
        self.is_cuda = self.device.type == "cuda"
        if require_cuda and not self.is_cuda:
            raise MJWarpUnavailable("formal native MJWarp training is CUDA-only")
        if self.is_cuda and not torch.cuda.is_available():
            raise MJWarpUnavailable("CUDA is not visible to PyTorch")
        if self.is_cuda:
            if self.device.index is None:
                self.device = torch.device(
                    "cuda", torch.cuda.current_device()
                )
            torch.cuda.set_device(self.device)
        wp.init()
        self.wp_device = wp.get_device(str(self.device))
        if self.is_cuda and not bool(getattr(self.wp_device, "is_cuda", False)):
            raise MJWarpUnavailable(
                f"Warp device {self.wp_device!s} is not CUDA"
            )

        # CUDA Graph capture cannot use CUDA's legacy default stream.  The
        # same non-default stream is visible to both runtimes.
        self.torch_stream = (
            torch.cuda.Stream(device=self.device) if self.is_cuda else None
        )
        self.wp_stream = (
            wp.stream_from_torch(self.torch_stream)
            if self.torch_stream is not None
            else None
        )
        self.physics_dt = float(self.host_model.opt.timestep)
        self.control_dt = self.physics_dt * self.physics_substeps
        if abs(self.control_dt - 0.020) > 1.0e-9:
            raise ValueError(
                f"expected a 20 ms control period, got {self.control_dt:.9f}"
            )
        self._check_host_model()
        self._configure_nativeccd()

        with self._warp_context(sync_enter=True):
            self.model = mjw.put_model(self.host_model)
            self.model.opt.warn_overflow = True
            self.data = mjw.make_data(
                self.host_model,
                nworld=self.worlds,
                nconmax=self.contacts_per_world,
                njmax=self.constraints_per_world,
            )

        self._views: dict[str, Any] = {}
        self._torch_to_warp_aliases: list[Any] = []
        self._pre_step: Callable[[], None] | None = None
        self._graph: Any = None
        self._graph_ready = False
        self._jacobian_buffers: dict[str, dict[str, Any]] = {}
        self._sticky_health = torch.zeros(
            5, device=self.device, dtype=torch.bool
        )
        self._reset_mask = torch.zeros(
            self.worlds, device=self.device, dtype=torch.bool
        )
        self._reset_mask_wp = self.alias_from_torch(self._reset_mask)
        self.qpos_templates = torch.as_tensor(
            np.asarray(platform_spec.qpos_templates),
            device=self.device,
            dtype=torch.float32,
        )
        self.ctrl_templates = torch.as_tensor(
            np.asarray(platform_spec.ctrl_templates),
            device=self.device,
            dtype=torch.float32,
        )

    def _check_host_model(self) -> None:
        model = self.host_model
        model_nv = int(model.nv)
        if int(model.nu) <= 0 or int(model.nq) <= 0 or model_nv <= 0:
            raise ValueError("parent model has invalid nq/nv/nu")
        if model_nv > contract.FORMAL_DENSE_NEWTON_MAX_NV:
            raise ValueError(
                "formal GRAB dense Newton requires model nv <= "
                f"{contract.FORMAL_DENSE_NEWTON_MAX_NV}, found {model_nv}"
            )
        if self.compatibility.get("runtime_mode") == (
            "top-level-mujoco-warp-production"
        ):
            from .newton_regularization import (
                formal_newton_regularization_metadata_matches,
            )

            if not formal_newton_regularization_metadata_matches(
                self.compatibility
            ):
                raise MJWarpUnavailable(
                    "formal MJWarp Newton regularization metadata is invalid"
                )
            regularized_max_nv = self.compatibility.get(
                "newton_regularization_max_nv"
            )
            if model_nv > regularized_max_nv:
                raise ValueError(
                    "formal GRAB regularized tile Newton requires model nv <= "
                    f"{regularized_max_nv}, found {model_nv}"
                )
        if self.qpos_template_shape != (
            len(contract.OBJECTS),
            contract.RESET_TEMPLATES_PER_OBJECT,
            int(model.nq),
        ):
            raise ValueError("PlatformSpec qpos templates disagree with model")
        if tuple(np.asarray(self.platform.ctrl_templates).shape) != (
            len(contract.OBJECTS),
            contract.RESET_TEMPLATES_PER_OBJECT,
            int(model.nu),
        ):
            raise ValueError("PlatformSpec ctrl templates disagree with model")
        solver_option = int(model.opt.solver)
        expected_solver = int(self.mujoco.mjtSolver.mjSOL_NEWTON)
        requested_solver_tolerance = float(model.opt.tolerance)
        effective_solver_tolerance = max(
            requested_solver_tolerance,
            contract.FORMAL_EFFECTIVE_SOLVER_TOLERANCE,
        )
        jacobian_option = int(model.opt.jacobian)
        resolved_sparse_jacobian = bool(self.mujoco.mj_isSparse(model))
        self.compatibility.update(
            {
                "model_nv": model_nv,
                "dense_newton_max_nv": (
                    contract.FORMAL_DENSE_NEWTON_MAX_NV
                ),
                "solver_option": solver_option,
                "solver_name": (
                    contract.FORMAL_SOLVER
                    if solver_option == expected_solver
                    else "unsupported"
                ),
                "requested_solver_tolerance": requested_solver_tolerance,
                "effective_solver_tolerance": effective_solver_tolerance,
                "jacobian_option": jacobian_option,
                "resolved_sparse_jacobian": resolved_sparse_jacobian,
            }
        )
        if solver_option != expected_solver:
            raise ValueError("formal GRAB MJWarp requires the Newton solver")
        if (
            requested_solver_tolerance
            != contract.FORMAL_XML_SOLVER_TOLERANCE
        ):
            raise ValueError(
                "formal GRAB XML must request solver tolerance "
                f"{contract.FORMAL_XML_SOLVER_TOLERANCE:g}"
            )
        if (
            effective_solver_tolerance
            != contract.FORMAL_EFFECTIVE_SOLVER_TOLERANCE
        ):
            raise ValueError(
                "formal GRAB MJWarp requires an effective solver tolerance "
                f"of {contract.FORMAL_EFFECTIVE_SOLVER_TOLERANCE:g}"
            )
        if (
            jacobian_option
            != int(self.mujoco.mjtJacobian.mjJAC_DENSE)
            or resolved_sparse_jacobian
        ):
            raise ValueError(
                "formal GRAB MJWarp requires an explicitly dense "
                "constraint Jacobian with model nv <= 60"
            )

    def _configure_nativeccd(self) -> None:
        """Disable unsupported native CCD in memory when margins require it."""

        bit = int(self.mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        before = int(self.host_model.opt.disableflags)
        has_margin = bool(
            np.any(np.asarray(self.host_model.geom_margin) > 0.0)
        )
        changed = has_margin and not bool(before & bit)
        if changed:
            self.host_model.opt.disableflags = before | bit
        after = int(self.host_model.opt.disableflags)
        self.compatibility.update(
            {
                "positive_geom_margin": has_margin,
                "nativeccd_disable_bit": bit,
                "nativeccd_disableflags_before": before,
                "nativeccd_disableflags_after": after,
                "nativeccd_disabled_by_backend": changed,
                "nativeccd_disabled": bool(after & bit),
            }
        )
        if has_margin and not bool(after & bit):
            raise MJWarpUnavailable(
                "positive geom margins require native CCD to be disabled"
            )

    @property
    def qpos_template_shape(self) -> tuple[int, ...]:
        return tuple(np.asarray(self.platform.qpos_templates).shape)

    @contextmanager
    def _warp_context(self, *, sync_enter: bool = False) -> Iterator[None]:
        with self.wp.ScopedDevice(self.wp_device):
            if self.wp_stream is None:
                yield
            else:
                with self.wp.ScopedStream(
                    self.wp_stream, sync_enter=sync_enter
                ):
                    yield

    def _resolve_data_path(self, name: str) -> Any:
        value = self.data
        for component in name.split("."):
            value = getattr(value, component, None)
            if value is None:
                raise AttributeError(
                    f"MJWarp Data has no field path {name!r}"
                )
        return value

    def has_field(self, name: str) -> bool:
        if self.data is None:
            return False
        try:
            self._resolve_data_path(name)
        except AttributeError:
            return False
        return True

    def tensor(self, name: str) -> Any:
        """Return a cached zero-copy view; dotted nested paths are supported."""

        if name not in self._views:
            array = self._resolve_data_path(name)
            self._views[name] = self.wp.to_torch(array)
        return self._views[name]

    def alias_from_torch(self, tensor: Any, *, dtype: Any = None) -> Any:
        """Return and retain a zero-copy Warp alias of contiguous Torch storage."""

        if tensor.device != self.device:
            raise ValueError("Torch/Warp aliases must use the backend CUDA device")
        if not tensor.is_contiguous():
            raise ValueError("Torch/Warp aliases must be contiguous")
        kwargs = {"requires_grad": False}
        if dtype is not None:
            kwargs["dtype"] = dtype
        alias = self.wp.from_torch(tensor, **kwargs)
        self._torch_to_warp_aliases.append(alias)
        return alias

    def register_pre_step(self, callback: Callable[[], None]) -> None:
        if self._graph_ready:
            raise RuntimeError("cannot register pre-step work after graph capture")
        if self._pre_step is not None:
            raise RuntimeError("native pre-step work is already registered")
        self._pre_step = callback

    def _begin_physics_stream(self) -> Any:
        if not self.is_cuda:
            return None
        caller = self.torch.cuda.current_stream(device=self.device)
        self.torch_stream.wait_stream(caller)
        return caller

    def _end_physics_stream(self, caller: Any) -> None:
        if caller is not None:
            caller.wait_stream(self.torch_stream)

    def _physics_tick(self) -> None:
        # Canonical MJWarp stage order: current kinematics/velocity, optional
        # native external work, constraint/ACC stage, then integration.
        self.mjw.step1(self.model, self.data)
        if self._pre_step is not None:
            self._pre_step()
        self.mjw.step2(self.model, self.data)

    def _finish_control_observation(self) -> None:
        self.mjw.forward(self.model, self.data)

    def forward(self) -> None:
        caller = self._begin_physics_stream()
        try:
            with self._warp_context():
                self.mjw.forward(self.model, self.data)
        finally:
            self._end_physics_stream(caller)

    def prepare_cuda_graph(self) -> None:
        """Warm, capture and retain one complete 50 Hz physics period."""

        if (
            self._graph_ready
            or not self.is_cuda
            or not self.cuda_graph_enabled
        ):
            return
        caller = self._begin_physics_stream()
        try:
            with self._warp_context():
                self._physics_tick()
                self._finish_control_observation()
                self.wp.synchronize_stream(self.wp_stream)
                with self.wp.ScopedCapture(
                    stream=self.wp_stream
                ) as capture:
                    for _ in range(self.physics_substeps):
                        self._physics_tick()
                    self._finish_control_observation()
        finally:
            self._end_physics_stream(caller)
        self._graph = capture.graph
        self._graph_ready = True

    def step_control(self) -> None:
        caller = self._begin_physics_stream()
        try:
            with self._warp_context():
                if self._graph_ready:
                    self.wp.capture_launch(
                        self._graph, stream=self.wp_stream
                    )
                else:
                    for _ in range(self.physics_substeps):
                        self._physics_tick()
                    self._finish_control_observation()
        finally:
            self._end_physics_stream(caller)
        self._accumulate_health()

    def _new_jacobian_buffer(self) -> dict[str, Any]:
        """Allocate one persistent input/output set for a named Jacobian."""

        torch = self.torch
        points = torch.empty(
            (self.worlds, 3), device=self.device, dtype=torch.float32
        )
        bodies = torch.empty(
            self.worlds, device=self.device, dtype=torch.int32
        )
        point_scalar = self.alias_from_torch(points, dtype=self.wp.float32)
        point_wp = point_scalar.view(self.wp.vec3)
        body_wp = self.alias_from_torch(bodies, dtype=self.wp.int32)
        with self._warp_context():
            output_wp = self.wp.empty(
                (self.worlds, 3, int(self.host_model.nv)),
                dtype=self.wp.float32,
                device=self.wp_device,
            )
        output = self.wp.to_torch(output_wp)
        return {
            "points": points,
            "bodies": bodies,
            "point_wp": point_wp,
            "body_wp": body_wp,
            "output_wp": output_wp,
            "output": output,
        }

    def jacobian(
        self,
        name: str,
        point_torch: Any,
        body_id_torch: Any,
    ) -> Any:
        """Return cached ``[world, 3, nv]`` point Jacobians.

        ``name`` identifies independent persistent buffers, so callers may
        retain several Jacobian results until the next call with the same name.
        Point coordinates are world-frame positions and body IDs are int32.
        """

        key = str(name)
        if not key:
            raise ValueError("Jacobian cache name must be non-empty")
        buffer = self._jacobian_buffers.get(key)
        if buffer is None:
            buffer = self._new_jacobian_buffer()
            self._jacobian_buffers[key] = buffer
        points = self.torch.as_tensor(
            point_torch, device=self.device, dtype=self.torch.float32
        )
        if tuple(points.shape) != (self.worlds, 3):
            raise ValueError(
                f"Jacobian points must have shape {(self.worlds, 3)}"
            )
        bodies = self.torch.as_tensor(
            body_id_torch, device=self.device, dtype=self.torch.int32
        )
        if bodies.ndim == 0:
            bodies = bodies.expand(self.worlds)
        if tuple(bodies.shape) != (self.worlds,):
            raise ValueError(
                f"Jacobian body IDs must have shape {(self.worlds,)}"
            )
        buffer["points"].copy_(points)
        buffer["bodies"].copy_(bodies)
        caller = self._begin_physics_stream()
        try:
            with self._warp_context():
                self.mjw.jac(
                    self.model,
                    self.data,
                    buffer["output_wp"],
                    None,
                    buffer["point_wp"],
                    buffer["body_wp"],
                )
        finally:
            self._end_physics_stream(caller)
        return buffer["output"]

    def _selected_rows(
        self, value: Any, ids: Any, target: Any, label: str
    ) -> Any:
        source = self.torch.as_tensor(
            value, device=self.device, dtype=target.dtype
        )
        if tuple(source.shape) == tuple(target.shape):
            source = source.index_select(0, ids)
        expected_tail = tuple(target.shape[1:])
        if source.ndim != target.ndim or tuple(source.shape[1:]) != expected_tail:
            raise ValueError(f"{label} has incompatible shape {tuple(source.shape)}")
        if int(source.shape[0]) != int(ids.numel()):
            raise ValueError(f"{label} must contain all worlds or selected worlds")
        return source

    def _copy_selected(self, name: str, ids: Any, value: Any) -> None:
        target = self.tensor(name)
        source = self._selected_rows(value, ids, target, name)
        target.index_copy_(0, ids, source)

    def reset_worlds(
        self,
        mask: Any,
        *,
        qpos: Any,
        ctrl: Any,
        qvel: Any | None = None,
        act: Any | None = None,
        mocap_pos: Any | None = None,
        mocap_quat: Any | None = None,
        eq_active: Any | None = None,
    ) -> None:
        """Reset selected worlds, write base state and rebuild derived fields."""

        torch = self.torch
        selected = torch.as_tensor(
            mask, device=self.device, dtype=torch.bool
        ).reshape(self.worlds)
        ids = torch.nonzero(selected, as_tuple=False).flatten()
        if int(ids.numel()) == 0:
            return
        self._reset_mask.copy_(selected)
        caller = self._begin_physics_stream()
        try:
            with self._warp_context():
                self.mjw.reset_data(
                    self.model, self.data, reset=self._reset_mask_wp
                )
        finally:
            self._end_physics_stream(caller)

        with _unique_cuda_index_writes(torch, self.device.type):
            self._copy_selected("qpos", ids, qpos)
            self._copy_selected("ctrl", ids, ctrl)
            if qvel is None:
                self.tensor("qvel").index_fill_(0, ids, 0.0)
            else:
                self._copy_selected("qvel", ids, qvel)
            for name, value in (
                ("act", act),
                ("mocap_pos", mocap_pos),
                ("mocap_quat", mocap_quat),
                ("eq_active", eq_active),
            ):
                if value is not None and self.has_field(name):
                    self._copy_selected(name, ids, value)
        self.forward()

    def reset_from_templates(
        self,
        mask: Any,
        object_index: Any,
        template_index: Any | None = None,
    ) -> None:
        """Reset worlds from object/template rows in the parent reset bank."""

        object_indices = self.torch.as_tensor(
            object_index, device=self.device, dtype=self.torch.long
        ).reshape(self.worlds)
        if template_index is None:
            template_indices = self.torch.zeros_like(object_indices)
        else:
            template_indices = self.torch.as_tensor(
                template_index,
                device=self.device,
                dtype=self.torch.long,
            ).reshape(self.worlds)
        indices = (
            object_indices * contract.RESET_TEMPLATES_PER_OBJECT
            + template_indices
        )
        qpos = self.qpos_templates.reshape(
            -1, int(self.host_model.nq)
        ).index_select(0, indices)
        ctrl = self.ctrl_templates.reshape(
            -1, int(self.host_model.nu)
        ).index_select(0, indices)
        self.reset_worlds(mask, qpos=qpos, ctrl=ctrl)

    def accumulate_external_health(self, *values: Any) -> None:
        nonfinite = self.torch.zeros(
            (), device=self.device, dtype=self.torch.bool
        )
        for value in values:
            if value is not None:
                nonfinite |= ~self.torch.isfinite(value).all()
        self._sticky_health[4] |= nonfinite

    def _accumulate_health(self) -> None:
        torch = self.torch
        nonfinite = ~(
            torch.isfinite(self.tensor("qpos")).all()
            & torch.isfinite(self.tensor("qvel")).all()
        )
        zero = torch.zeros((), device=self.device, dtype=torch.bool)
        overflow = (
            (self.tensor("overflow") != 0).any()
            if self.has_field("overflow")
            else zero
        )
        contacts_full = (
            self.tensor("nacon")[0] >= int(self.data.naconmax)
            if self.has_field("nacon")
            else zero
        )
        constraints_full = (
            self.tensor("nefc").max() >= self.constraints_per_world
            if self.has_field("nefc")
            else zero
        )
        self._sticky_health[:4] |= torch.stack(
            (nonfinite, overflow, contacts_full, constraints_full)
        )

    def health(self) -> dict[str, object]:
        result: dict[str, object] = {
            "finite": bool(
                self.torch.isfinite(self.tensor("qpos")).all()
                & self.torch.isfinite(self.tensor("qvel")).all()
            ),
            "sticky_nonfinite": int(self._sticky_health[0]),
            "sticky_overflow": int(self._sticky_health[1]),
            "sticky_contacts_full": int(self._sticky_health[2]),
            "sticky_constraints_full": int(self._sticky_health[3]),
            "sticky_external_nonfinite": int(self._sticky_health[4]),
            **self.compatibility,
        }
        if self.has_field("overflow"):
            overflow = self.tensor("overflow")
            result["overflow_worlds"] = int((overflow != 0).sum())
            result["max_overflow_mask"] = int(overflow.max())
        if self.has_field("nacon"):
            result["contacts_total"] = int(self.tensor("nacon")[0])
            result["contact_capacity_total"] = int(self.data.naconmax)
        if self.has_field("nefc"):
            result["max_nefc"] = int(self.tensor("nefc").max())
            result["constraint_capacity"] = self.constraints_per_world
        return result

    def state_dict(self) -> dict[str, Any]:
        fields = (
            "time",
            "qpos",
            "qvel",
            "ctrl",
            "act",
            "qacc_warmstart",
            "eq_active",
            "mocap_pos",
            "mocap_quat",
        )
        return {
            name: self.tensor(name).detach().clone()
            for name in fields
            if self.has_field(name)
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for required in ("qpos", "qvel", "ctrl"):
            if required not in state:
                raise ValueError(f"backend state is missing {required!r}")
        for name, source in state.items():
            if not self.has_field(name):
                continue
            target = self.tensor(name)
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(f"backend state {name!r} has the wrong shape")
            target.copy_(source.to(device=self.device, dtype=target.dtype))
        self._sticky_health.zero_()
        self.forward()

    def close(self) -> None:
        torch = getattr(self, "torch", None)
        device = getattr(self, "device", None)
        synchronization_error: BaseException | None = None
        try:
            if bool(getattr(self, "is_cuda", False)) and torch is not None:
                torch.cuda.synchronize(device)
        except BaseException as exc:
            synchronization_error = exc
        finally:
            self._pre_step = None
            self._graph = None
            self._graph_ready = False
            getattr(self, "_jacobian_buffers", {}).clear()
            getattr(self, "_views", {}).clear()
            getattr(self, "_torch_to_warp_aliases", []).clear()
            self._reset_mask_wp = None
            self.data = None
            self.model = None
            self.wp_stream = None
            self.torch_stream = None
        if synchronization_error is not None:
            raise RuntimeError(
                "CUDA synchronization failed while closing MJWarp backend"
            ) from synchronization_error


__all__ = [
    "MJWarpBackend",
    "MJWarpUnavailable",
    "PINNED_RUNTIME",
    "RTX_PRO_6000_EPA_HORIZON_EDGES",
]
