"""Thin zero-copy bridge between MJWarp arrays and PyTorch CUDA."""

from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from screw_config import DEFAULT_CONFIG

from .config import HPCConfig, PACKAGE_ROOT
from .contracts import FINGERS
from .mjwarp_compat import configure_mjwarp_epa_horizon


class MJWarpUnavailable(RuntimeError):
    """The process is not a supported Linux/NVIDIA MJWarp runtime."""


def _asset_reset_center_z(host_model, index) -> float:
    """Return the common asset reset height or reject geometry drift."""

    screw_z = float(
        np.asarray(host_model.qpos0)[int(index.screw_qpos) + 2]
    )
    holder_z = float(
        np.asarray(host_model.body_pos)[int(index.holder_body), 2]
    )
    expected_z = float(DEFAULT_CONFIG.screw_initial_center_z_m)
    values = np.asarray((screw_z, holder_z, expected_z))
    if not np.isfinite(values).all() or not np.allclose(
        values, expected_z, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError(
            "screw reset geometry disagrees between qpos0, holder, and "
            f"task contract: {values.tolist()}"
        )
    return screw_z


@dataclass(frozen=True)
class ModelIndex:
    actuator_qpos: Tuple[int, ...]
    actuator_dof: Tuple[int, ...]
    passive_qpos: Tuple[Tuple[int, int], ...]
    passive_dof: Tuple[Tuple[int, int], ...]
    tip_bodies: Tuple[int, ...]
    tip_body_roots: Tuple[int, ...]
    screw_body: int
    screw_root_body: int
    screw_qpos: int
    screw_dof: int
    holder_body: int
    holder_mocap: int
    holder_equality: int
    socket_body: int
    socket_mocap: int
    touch_sensor_adr: Tuple[int, ...]


def _runtime_imports():
    try:
        import torch
        import warp as wp
        import mujoco
        import mujoco_warp as mjw
        compatibility = configure_mjwarp_epa_horizon()
    except ImportError as exc:
        raise MJWarpUnavailable(
            "hpc needs torch CUDA, warp-lang, mujoco, and mujoco-warp; "
            "run ./setup_hpc.sh on an HX Linux GPU node"
        ) from exc
    return torch, wp, mujoco, mjw, compatibility


class MJWarpBackend:
    """Own one MJWarp model and a batch of independent worlds."""

    def __init__(
        self,
        config: HPCConfig,
        *,
        worlds: Optional[int] = None,
        require_linux: bool = True,
    ) -> None:
        torch, wp, mujoco, mjw, compatibility = _runtime_imports()
        if require_linux and platform.system() != "Linux":
            raise MJWarpUnavailable(
                "native MJWarp training is Linux/NVIDIA only; use the root "
                "MPS tuning profile on macOS"
            )
        if not torch.cuda.is_available():
            raise MJWarpUnavailable("torch.cuda.is_available() is false")
        self.torch = torch
        self.wp = wp
        self.mujoco = mujoco
        self.mjw = mjw
        self.mjwarp_compatibility = compatibility
        self.config = config
        self.worlds = int(worlds or config.physics.worlds)
        self.physics_substeps = int(config.physics.physics_substeps)
        self.device = torch.device(config.runtime.device)
        torch.cuda.set_device(self.device)
        wp.init()
        self.wp_device = wp.get_device(str(self.device))
        if not self.wp_device.is_cuda:
            raise MJWarpUnavailable(f"Warp device {self.wp_device} is not CUDA")
        # CUDA graph capture cannot use CUDA's legacy default stream.  Keep a
        # dedicated non-default Torch stream and expose that exact stream to
        # Warp; event-based waits below order zero-copy Torch command updates
        # against physics without a device-to-host synchronization.
        self.torch_stream = torch.cuda.Stream(device=self.device)
        self.wp_stream = wp.stream_from_torch(
            self.torch_stream
        )

        self.xml_path = PACKAGE_ROOT / "assets" / "Project_hpc.xml"
        self.host_model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.physics_dt = float(self.host_model.opt.timestep)
        self.control_dt = self.physics_dt * self.physics_substeps
        if abs(self.control_dt - 0.020) > 1.0e-9:
            raise ValueError(
                "expert schedule and rate limiter require a 20 ms control "
                f"period, got {self.control_dt:.9f} s"
            )
        self._check_host_model()
        self.index = self._index_model()
        self.screw_reset_center_z = _asset_reset_center_z(
            self.host_model, self.index
        )
        with wp.ScopedDevice(self.wp_device):
            with wp.ScopedStream(self.wp_stream):
                self.model = mjw.put_model(self.host_model)
                # These are MJWarp-only options and therefore must be set on
                # the device model before CUDA graph capture.
                self.model.opt.contact_sensor_maxmatch = int(
                    config.physics.contact_sensor_maxmatch
                )
                self.model.opt.warn_overflow = True
                self.data = mjw.make_data(
                    self.host_model,
                    nworld=self.worlds,
                    nconmax=config.physics.nconmax,
                    njmax=config.physics.njmax,
                )
        self._views: Dict[str, object] = {}
        self._pre_step: Optional[Callable[[], None]] = None
        self._graph = None
        self._graph_ready = False
        self._sticky_health = torch.zeros(
            5, device=self.device, dtype=torch.bool
        )
        self.ctrl_range = torch.as_tensor(
            np.asarray(self.host_model.actuator_ctrlrange),
            device=self.device,
            dtype=torch.float32,
        )
        self.qpos0 = torch.as_tensor(
            np.asarray(self.host_model.qpos0),
            device=self.device,
            dtype=torch.float32,
        )
        self.eq_active0 = torch.as_tensor(
            np.asarray(self.host_model.eq_active0, dtype=np.bool_),
            device=self.device,
            dtype=torch.bool,
        )

    def _check_host_model(self) -> None:
        m = self.host_model
        unsupported_user = {
            name: int(getattr(m, name))
            for name in ("nuser_jnt", "nuser_site", "nuser_sensor", "nuser_actuator")
            if int(getattr(m, name)) != 0
        }
        if unsupported_user:
            raise ValueError(f"MJWarp-incompatible MuJoCo user fields: {unsupported_user}")
        if m.nu != 16 or m.nq != 31 or m.nv != 30:
            raise ValueError(
                f"unexpected model dimensions nq/nv/nu={m.nq}/{m.nv}/{m.nu}"
            )
        if m.nsensor != 4 or m.nsensordata != 4:
            raise ValueError("HPC XML must expose four scalar fingertip touch sensors")
        if int(m.opt.solver) != int(self.mujoco.mjtSolver.mjSOL_NEWTON):
            raise ValueError("HPC model must use the Newton solver")
        if int(m.opt.jacobian) == int(self.mujoco.mjtJacobian.mjJAC_SPARSE):
            raise ValueError("MJWarp does not support sparse constraint Jacobians")

    def _id(self, object_type, name: str) -> int:
        value = int(self.mujoco.mj_name2id(self.host_model, object_type, name))
        if value < 0:
            raise ValueError(f"model object not found: {name}")
        return value

    def _index_model(self) -> ModelIndex:
        mj = self.mujoco
        m = self.host_model
        actuator_qpos = []
        actuator_dof = []
        for actuator in range(m.nu):
            joint = int(m.actuator_trnid[actuator, 0])
            actuator_qpos.append(int(m.jnt_qposadr[joint]))
            actuator_dof.append(int(m.jnt_dofadr[joint]))
        passive_qpos = []
        passive_dof = []
        for finger in FINGERS:
            qpair = []
            dpair = []
            for kind in ("short", "elbow"):
                joint = self._id(
                    mj.mjtObj.mjOBJ_JOINT, f"robot0:{kind}_joint_{finger}"
                )
                qpair.append(int(m.jnt_qposadr[joint]))
                dpair.append(int(m.jnt_dofadr[joint]))
            passive_qpos.append(tuple(qpair))
            passive_dof.append(tuple(dpair))
        screw_joint = self._id(mj.mjtObj.mjOBJ_JOINT, "screw_free")
        holder_body = self._id(
            mj.mjtObj.mjOBJ_BODY, "screw_start_holder_body"
        )
        socket_body = self._id(mj.mjtObj.mjOBJ_BODY, "socket")
        sensors = tuple(
            self._id(mj.mjtObj.mjOBJ_SENSOR, f"tip_force_{finger}")
            for finger in FINGERS
        )
        tip_bodies = tuple(
            self._id(mj.mjtObj.mjOBJ_BODY, f"hard_tip_{finger}")
            for finger in FINGERS
        )
        screw_body = self._id(mj.mjtObj.mjOBJ_BODY, "screw")
        return ModelIndex(
            actuator_qpos=tuple(actuator_qpos),
            actuator_dof=tuple(actuator_dof),
            passive_qpos=tuple(passive_qpos),
            passive_dof=tuple(passive_dof),
            tip_bodies=tip_bodies,
            tip_body_roots=tuple(
                int(m.body_rootid[body]) for body in tip_bodies
            ),
            screw_body=screw_body,
            screw_root_body=int(m.body_rootid[screw_body]),
            screw_qpos=int(m.jnt_qposadr[screw_joint]),
            screw_dof=int(m.jnt_dofadr[screw_joint]),
            holder_body=holder_body,
            holder_mocap=int(m.body_mocapid[holder_body]),
            holder_equality=self._id(
                mj.mjtObj.mjOBJ_EQUALITY, "screw_start_holder"
            ),
            socket_body=socket_body,
            socket_mocap=int(m.body_mocapid[socket_body]),
            touch_sensor_adr=tuple(int(m.sensor_adr[sensor]) for sensor in sensors),
        )

    def tensor(self, name: str):
        """Return a cached zero-copy Torch view of one MJWarp data field."""

        if name not in self._views:
            array = getattr(self.data, name, None)
            if array is None:
                raise AttributeError(f"MJWarp Data has no field {name!r}")
            self._views[name] = self.wp.to_torch(array)
        return self._views[name]

    def has_field(self, name: str) -> bool:
        return getattr(self.data, name, None) is not None

    def register_pre_step(self, callback: Callable[[], None]) -> None:
        """Register the graph-captured native operation run before every step."""

        if self._graph_ready:
            raise RuntimeError("cannot register pre-step work after graph capture")
        if self._pre_step is not None:
            raise RuntimeError("MJWarp pre-step work is already registered")
        self._pre_step = callback

    def _begin_physics_stream(self):
        caller = self.torch.cuda.current_stream(device=self.device)
        self.torch_stream.wait_stream(caller)
        return caller

    def _end_physics_stream(self, caller) -> None:
        caller.wait_stream(self.torch_stream)

    def _run_pre_step(self) -> None:
        if self._pre_step is not None:
            self._pre_step()

    def _run_physics_tick(self) -> None:
        """Run the public MJWarp input boundary in its canonical order."""

        # step1 refreshes position/velocity-stage state before external input.
        # step2 computes actuation/constraints/ACC-stage touch sensors and then
        # integrates.  Do not move step1 after step2: it clears sensordata and
        # this model's four touch sensors are ACC-stage.
        self.mjw.step1(self.model, self.data)
        self._run_pre_step()
        self.mjw.step2(self.model, self.data)

    def _finish_control_observation(self) -> None:
        # The final step2 has integrated qpos/qvel.  A full forward refreshes
        # cvel/xpos/subtree_com and all sensor stages at that new state, so the
        # 50 Hz task observation is internally current.  This costs one extra
        # non-integrating forward per 40 physics ticks.
        self.mjw.forward(self.model, self.data)

    def forward(self) -> None:
        caller = self._begin_physics_stream()
        try:
            with self.wp.ScopedDevice(self.wp_device):
                with self.wp.ScopedStream(self.wp_stream, sync_enter=False):
                    self.mjw.forward(self.model, self.data)
        finally:
            self._end_physics_stream(caller)

    def prepare_cuda_graph(self) -> None:
        if not self.config.physics.cuda_graph or self._graph_ready:
            return
        # Compilation happens outside capture.  The caller resets afterwards,
        # so this warm-up integration is never part of an episode.
        caller = self._begin_physics_stream()
        try:
            with self.wp.ScopedDevice(self.wp_device):
                with self.wp.ScopedStream(
                    self.wp_stream, sync_enter=False
                ):
                    self._run_physics_tick()
                    self._finish_control_observation()
                    # Kernel/module compilation and the warm-up integration
                    # must finish before beginning capture on the dedicated
                    # stream.  This is a one-time setup synchronization, never
                    # part of an episode or physics substep.
                    self.wp.synchronize_stream(self.wp_stream)
                    with self.wp.ScopedCapture(
                        stream=self.wp_stream
                    ) as capture:
                        # Unroll one complete control period.  Replaying this
                        # graph is one host launch while every physics substep
                        # recomputes its wrench from the state produced by the
                        # preceding MJWarp step.
                        for _ in range(self.physics_substeps):
                            self._run_physics_tick()
                        self._finish_control_observation()
        finally:
            self._end_physics_stream(caller)
        self._graph = capture.graph
        self._graph_ready = True

    def step_control(self) -> None:
        """Advance one control period with one graph launch when enabled."""

        caller = self._begin_physics_stream()
        try:
            with self.wp.ScopedDevice(self.wp_device):
                with self.wp.ScopedStream(
                    self.wp_stream, sync_enter=False
                ):
                    if self._graph_ready:
                        self.wp.capture_launch(
                            self._graph, stream=self.wp_stream
                        )
                    else:
                        for _ in range(self.physics_substeps):
                            self._run_physics_tick()
                        self._finish_control_observation()
        finally:
            self._end_physics_stream(caller)
        self._accumulate_health()

    def _accumulate_health(self) -> None:
        """Record transient failures on-device without synchronizing the host."""

        torch = self.torch
        qpos = self.tensor("qpos")
        qvel = self.tensor("qvel")
        nonfinite = ~(
            torch.isfinite(qpos).all() & torch.isfinite(qvel).all()
        )
        overflow = (
            (self.tensor("overflow") != 0).any()
            if self.has_field("overflow")
            else torch.zeros((), device=self.device, dtype=torch.bool)
        )
        contacts_full = (
            self.tensor("nacon")[0] >= int(self.data.naconmax)
            if self.has_field("nacon")
            else torch.zeros((), device=self.device, dtype=torch.bool)
        )
        constraints_full = (
            self.tensor("nefc").max() >= self.config.physics.njmax
            if self.has_field("nefc")
            else torch.zeros((), device=self.device, dtype=torch.bool)
        )
        self._sticky_health |= torch.stack(
            (
                nonfinite,
                overflow,
                contacts_full,
                constraints_full,
                torch.zeros((), device=self.device, dtype=torch.bool),
            )
        )

    def accumulate_external_health(self, *tensors) -> None:
        """Latch non-finite policy/task tensors without a host sync."""

        torch = self.torch
        nonfinite = torch.zeros(
            (), device=self.device, dtype=torch.bool
        )
        for value in tensors:
            if value is not None:
                nonfinite |= ~torch.isfinite(value).all()
        self._sticky_health[4] |= nonfinite

    def reset_worlds(
        self,
        mask,
        *,
        spawn_xy,
        socket_position,
        socket_quaternion,
        ctrl,
        passive_q,
    ) -> None:
        """Reset selected worlds and recompute MJWarp derived state."""

        torch = self.torch
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        # Clear every dynamic/solver/contact field for only the selected
        # worlds.  Manually zeroing a small list is not sufficient: stale
        # contact and warm-start state can otherwise leak across episodes.
        reset_tensor = mask.contiguous()
        reset_mask = self.wp.from_torch(
            reset_tensor, requires_grad=False
        )
        caller = self._begin_physics_stream()
        try:
            with self.wp.ScopedDevice(self.wp_device):
                with self.wp.ScopedStream(
                    self.wp_stream, sync_enter=False
                ):
                    self.mjw.reset_data(
                        self.model, self.data, reset=reset_mask
                    )
        finally:
            self._end_physics_stream(caller)
        spawn_xy = torch.as_tensor(
            spawn_xy, device=self.device, dtype=torch.float32
        )
        ctrl = torch.as_tensor(ctrl, device=self.device, dtype=torch.float32)
        passive_q = torch.as_tensor(
            passive_q, device=self.device, dtype=torch.float32
        )
        socket_position = torch.as_tensor(
            socket_position, device=self.device, dtype=torch.float32
        )
        socket_quaternion = torch.as_tensor(
            socket_quaternion, device=self.device, dtype=torch.float32
        )
        qpos = self.tensor("qpos")
        qvel = self.tensor("qvel")
        controls = self.tensor("ctrl")
        qpos[ids] = self.qpos0
        qvel[ids] = 0.0
        controls[ids] = ctrl[ids]
        qpos_indices = torch.as_tensor(
            self.index.actuator_qpos, device=self.device, dtype=torch.long
        )
        qpos[ids[:, None], qpos_indices[None, :]] = ctrl[ids]
        for finger_index, qpair in enumerate(self.index.passive_qpos):
            indices = torch.as_tensor(qpair, device=self.device, dtype=torch.long)
            qpos[ids[:, None], indices[None, :]] = passive_q[ids, finger_index]

        qadr = self.index.screw_qpos
        qpos[ids, qadr] = spawn_xy[ids, 0]
        qpos[ids, qadr + 1] = spawn_xy[ids, 1]
        qpos[ids, qadr + 2] = self.screw_reset_center_z
        qpos[ids, qadr + 3 : qadr + 7] = torch.as_tensor(
            (1.0, 0.0, 0.0, 0.0), device=self.device
        )
        if self.has_field("eq_active"):
            eq_active = self.tensor("eq_active")
            eq_active[ids] = self.eq_active0
            eq_active[ids, self.index.holder_equality] = True
        if self.index.holder_mocap >= 0 and self.has_field("mocap_pos"):
            self.tensor("mocap_pos")[ids, self.index.holder_mocap, :2] = spawn_xy[ids]
            self.tensor("mocap_pos")[
                ids, self.index.holder_mocap, 2
            ] = self.screw_reset_center_z
            self.tensor("mocap_quat")[ids, self.index.holder_mocap] = torch.as_tensor(
                (1.0, 0.0, 0.0, 0.0), device=self.device
            )
        if self.index.socket_mocap < 0:
            raise RuntimeError(
                "socket body must be mocap-enabled for randomized orbit resets"
            )
        if self.has_field("mocap_pos"):
            self.tensor("mocap_pos")[
                ids, self.index.socket_mocap
            ] = socket_position[ids]
            self.tensor("mocap_quat")[
                ids, self.index.socket_mocap
            ] = socket_quaternion[ids]
        self.forward()

    def release_holder(self, mask) -> None:
        if not self.has_field("eq_active"):
            return
        mask = self.torch.as_tensor(mask, device=self.device, dtype=self.torch.bool)
        self.tensor("eq_active")[mask, self.index.holder_equality] = False

    def state_dict(self) -> Dict[str, object]:
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

    def load_state_dict(self, state: Dict[str, object]) -> None:
        for required in ("qpos", "qvel", "ctrl"):
            if required not in state:
                raise ValueError(
                    f"MJWarp checkpoint is missing {required!r}"
                )
        for name, source in state.items():
            if not self.has_field(name):
                continue
            target = self.tensor(name)
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"MJWarp state {name!r} shape mismatch: "
                    f"{tuple(source.shape)} != {tuple(target.shape)}"
                )
            target.copy_(
                source.to(device=self.device, dtype=target.dtype)
            )
        self._sticky_health.zero_()
        # Contacts, kinematics, sensors, and c-frame velocities are derived
        # afresh.  qacc_warmstart is retained when the installed MJWarp exposes
        # it, so a resumed allocation starts close to the saved solver state.
        self.forward()

    def health(self) -> Dict[str, object]:
        torch = self.torch
        qpos = self.tensor("qpos")
        qvel = self.tensor("qvel")
        result: Dict[str, object] = {
            "finite": bool(torch.isfinite(qpos).all() and torch.isfinite(qvel).all()),
            "max_abs_qpos": float(qpos.abs().max()),
            "max_abs_qvel": float(qvel.abs().max()),
            **self.mjwarp_compatibility,
            "sticky_nonfinite": int(self._sticky_health[0]),
            "sticky_overflow": int(self._sticky_health[1]),
            "sticky_contacts_full": int(self._sticky_health[2]),
            "sticky_constraints_full": int(self._sticky_health[3]),
            "sticky_external_nonfinite": int(self._sticky_health[4]),
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
            result["constraint_capacity"] = self.config.physics.njmax
        return result

    def close(self) -> None:
        """Release graph/model/data ownership for repeated in-process runs."""

        if self.data is None:
            return
        self.torch.cuda.synchronize(self.device)
        self._pre_step = None
        self._graph = None
        self._graph_ready = False
        self._views.clear()
        self.data = None
        self.model = None


__all__ = [
    "MJWarpBackend",
    "MJWarpUnavailable",
    "ModelIndex",
]
