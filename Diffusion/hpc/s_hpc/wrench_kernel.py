"""Native Warp virtual-grasp and screw-thread wrench kernel.

This module deliberately does not import :mod:`warp` at module-import time.
The MPS/static toolchain can therefore inspect the hpc package on macOS,
while the kernel is constructed lazily from the Warp module owned by the
Linux/CUDA backend.
"""

import math

from screw_config import DEFAULT_CONFIG


_KERNEL_CACHE = {}


def _build_wrench_kernel(wp):
    """Build and cache the version-local Warp kernel."""

    cache_key = id(wp)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    @wp.func
    def _clip_vec3(vector: wp.vec3, maximum: float) -> wp.vec3:
        length = wp.length(vector)
        if length > maximum:
            return vector * (maximum / length)
        return vector

    @wp.func
    def _quat_wxyz_to_matrix(
        qw_in: float,
        qx_in: float,
        qy_in: float,
        qz_in: float,
    ) -> wp.mat33:
        norm = wp.sqrt(
            qw_in * qw_in
            + qx_in * qx_in
            + qy_in * qy_in
            + qz_in * qz_in
        )
        if norm < 1.0e-9:
            norm = 1.0e-9
        inverse = 1.0 / norm
        qw = qw_in * inverse
        qx = qx_in * inverse
        qy = qy_in * inverse
        qz = qz_in * inverse
        return wp.mat33(
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )

    @wp.func
    def _rotation_log(rotation: wp.mat33) -> wp.vec3:
        cosine = wp.clamp(
            0.5
            * (
                rotation[0, 0]
                + rotation[1, 1]
                + rotation[2, 2]
                - 1.0
            ),
            -1.0,
            1.0,
        )
        theta = wp.acos(cosine)
        vector = wp.vec3(
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
        scale = 0.5 + theta * theta / 12.0
        if theta >= 1.0e-5:
            denominator = 2.0 * wp.sin(theta)
            if denominator < 1.0e-7:
                denominator = 1.0e-7
            scale = theta / denominator
        return vector * scale

    @wp.kernel
    def _screw_wrench(
        qpos_in: wp.array2d(dtype=float),
        cvel_in: wp.array2d(dtype=wp.spatial_vector),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        desired_tip_in: wp.array(dtype=wp.vec3),
        desired_rotation_in: wp.array(dtype=wp.mat33),
        grasp_latched_in: wp.array(dtype=bool),
        holder_active_in: wp.array(dtype=bool),
        thread_engaged_in: wp.array(dtype=bool),
        twist_in: wp.array(dtype=float),
        thread_resistance_in: wp.array(dtype=float),
        axial_preload_in: wp.array(dtype=float),
        screw_body: int,
        screw_root_body: int,
        screw_qpos: int,
        screw_tip_local_z: float,
        socket_frame_in: wp.array(dtype=wp.mat33),
        socket_axis_in: wp.array(dtype=wp.vec3),
        socket_mouth_in: wp.array(dtype=wp.vec3),
        aligned_rotation_in: wp.array(dtype=wp.mat33),
        thread_engagement_depth: float,
        thread_lead: float,
        xfrc_applied_out: wp.array2d(dtype=wp.spatial_vector),
    ):
        world = wp.tid()
        socket_frame = socket_frame_in[world]
        socket_axis = socket_axis_in[world]
        socket_mouth = socket_mouth_in[world]
        aligned_rotation = aligned_rotation_in[world]

        actual_position = wp.vec3(
            qpos_in[world, screw_qpos],
            qpos_in[world, screw_qpos + 1],
            qpos_in[world, screw_qpos + 2],
        )
        actual_rotation = _quat_wxyz_to_matrix(
            qpos_in[world, screw_qpos + 3],
            qpos_in[world, screw_qpos + 4],
            qpos_in[world, screw_qpos + 5],
            qpos_in[world, screw_qpos + 6],
        )
        spatial_velocity = cvel_in[world, screw_body]
        # MJWarp cvel is [angular, linear] at the root subtree COM.
        angular_velocity = wp.spatial_top(spatial_velocity)
        linear_velocity_com = wp.spatial_bottom(spatial_velocity)
        subtree_center = subtree_com_in[world, screw_root_body]

        local_tip = wp.vec3(0.0, 0.0, screw_tip_local_z)
        actual_tip = actual_position + actual_rotation @ local_tip
        linear_velocity_body = (
            linear_velocity_com
            + wp.cross(
                angular_velocity, actual_position - subtree_center
            )
        )
        linear_velocity_tip = (
            linear_velocity_com
            + wp.cross(
                angular_velocity, actual_tip - subtree_center
            )
        )
        desired_rotation = desired_rotation_in[world]
        desired_position = desired_tip_in[world] - desired_rotation @ local_tip

        total_force = wp.vec3(0.0, 0.0, 0.0)
        total_torque = wp.vec3(0.0, 0.0, 0.0)
        if grasp_latched_in[world] and not holder_active_in[world]:
            grasp_force = (
                500.0 * (desired_position - actual_position)
                - 20.0 * linear_velocity_body
            )
            grasp_error = _rotation_log(
                desired_rotation @ wp.transpose(actual_rotation)
            )
            grasp_torque = (
                4.0 * grasp_error - 0.15 * angular_velocity
            )
            total_force = total_force + _clip_vec3(grasp_force, 40.0)
            total_torque = total_torque + _clip_vec3(grasp_torque, 1.5)

        if thread_engaged_in[world]:
            frame_transpose = wp.transpose(socket_frame)
            tip_socket = frame_transpose @ (actual_tip - socket_mouth)
            linear_socket = frame_transpose @ linear_velocity_tip
            lateral_socket = wp.vec3(
                -1000.0 * tip_socket[0] - 18.0 * linear_socket[0],
                -1000.0 * tip_socket[1] - 18.0 * linear_socket[1],
                0.0,
            )
            lateral_world = socket_frame @ lateral_socket

            angular_socket = frame_transpose @ angular_velocity
            alignment_world_error = _rotation_log(
                aligned_rotation @ wp.transpose(actual_rotation)
            )
            alignment_socket_error = (
                frame_transpose @ alignment_world_error
            )
            alignment_socket = wp.vec3(
                alignment_socket_error[0] - 0.05 * angular_socket[0],
                alignment_socket_error[1] - 0.05 * angular_socket[1],
                0.0,
            )
            alignment_world = socket_frame @ alignment_socket

            helical_error = (
                tip_socket[2]
                - thread_engagement_depth
                - thread_lead * twist_in[world]
            )
            helical_velocity_error = (
                wp.dot(linear_velocity_tip, socket_axis)
                - thread_lead * wp.dot(angular_velocity, socket_axis)
            )
            constraint_force = wp.clamp(
                1800.0 * helical_error
                + 28.0 * helical_velocity_error,
                -25.0,
                25.0,
            )
            angular_speed = wp.dot(angular_velocity, socket_axis)
            resistance = (
                -thread_resistance_in[world]
                * wp.tanh(angular_speed / 0.10)
            )
            thread_force = (
                _clip_vec3(lateral_world, 15.0)
                - constraint_force * socket_axis
                + axial_preload_in[world] * socket_axis
            )
            thread_torque = (
                _clip_vec3(alignment_world, 0.30)
                + thread_lead * constraint_force * socket_axis
                + resistance * socket_axis
            )
            total_force = total_force + thread_force
            total_torque = total_torque + thread_torque

        # xfrc_applied follows MuJoCo's [force, torque] convention.  This is
        # intentionally different from cvel's [angular, linear] convention.
        xfrc_applied_out[world, screw_body] = wp.spatial_vector(
            total_force, total_torque
        )

    _KERNEL_CACHE[cache_key] = _screw_wrench
    return _screw_wrench


class NativeWrenchKernel:
    """Persistent zero-copy Torch/Warp bindings for the pre-step kernel."""

    def __init__(self, backend, task_state) -> None:
        wp = backend.wp
        self.backend = backend
        self.wp = wp
        self.worlds = backend.worlds
        self.kernel = _build_wrench_kernel(wp)
        self._aliases = []

        self.desired_tip = self._composite_alias(
            task_state.desired_command.tip, wp.vec3, (3,), "desired tip"
        )
        self.desired_rotation = self._composite_alias(
            task_state.desired_command.rotation,
            wp.mat33,
            (3, 3),
            "desired rotation",
        )
        self.grasp_latched = self._scalar_alias(task_state.grasp_latched, wp.bool)
        self.holder_active = self._scalar_alias(task_state.holder_active, wp.bool)
        self.thread_engaged = self._scalar_alias(task_state.thread_engaged, wp.bool)
        self.twist = self._scalar_alias(task_state.twist, wp.float32)
        self.thread_resistance = self._scalar_alias(
            task_state.thread_resistance, wp.float32
        )
        self.axial_preload = self._scalar_alias(
            task_state.axial_preload, wp.float32
        )

        self.socket_frame = self._composite_alias(
            task_state.socket_frame,
            wp.mat33,
            (3, 3),
            "socket frame",
        )
        self.socket_axis = self._composite_alias(
            task_state.socket_axis, wp.vec3, (3,), "socket axis"
        )
        self.socket_mouth = self._composite_alias(
            task_state.socket_mouth, wp.vec3, (3,), "socket mouth"
        )
        self.aligned_rotation = self._composite_alias(
            task_state.aligned_rotation,
            wp.mat33,
            (3, 3),
            "aligned rotation",
        )
        self.thread_lead = (
            DEFAULT_CONFIG.thread_pitch_m / (2.0 * math.pi)
        )

    def _scalar_alias(self, tensor, dtype):
        if not tensor.is_contiguous():
            raise ValueError("persistent wrench inputs must be contiguous")
        alias = self.wp.from_torch(
            tensor, dtype=dtype, requires_grad=False
        )
        self._aliases.append(alias)
        return alias

    def _composite_alias(self, tensor, dtype, tail_shape, label: str):
        if tuple(tensor.shape[1:]) != tuple(tail_shape):
            raise ValueError(
                f"{label} must have shape [world,{','.join(map(str, tail_shape))}]"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{label} must be contiguous for zero-copy Warp view")
        scalar = self.wp.from_torch(
            tensor, dtype=self.wp.float32, requires_grad=False
        )
        composite = scalar.view(dtype)
        if tuple(composite.shape) != (self.worlds,):
            raise ValueError(
                f"{label} Warp view has shape {composite.shape}, "
                f"expected {(self.worlds,)}"
            )
        self._aliases.extend((scalar, composite))
        return composite

    def launch(self) -> None:
        """Enqueue one wrench update without a device-to-host synchronization."""

        task = DEFAULT_CONFIG
        self.wp.launch(
            kernel=self.kernel,
            dim=self.worlds,
            inputs=[
                self.backend.data.qpos,
                self.backend.data.cvel,
                self.backend.data.subtree_com,
                self.desired_tip,
                self.desired_rotation,
                self.grasp_latched,
                self.holder_active,
                self.thread_engaged,
                self.twist,
                self.thread_resistance,
                self.axial_preload,
                int(self.backend.index.screw_body),
                int(self.backend.index.screw_root_body),
                int(self.backend.index.screw_qpos),
                float(task.screw_tip_local_z_m),
                self.socket_frame,
                self.socket_axis,
                self.socket_mouth,
                self.aligned_rotation,
                float(task.thread_engagement_depth_m),
                float(self.thread_lead),
            ],
            outputs=[self.backend.data.xfrc_applied],
            device=self.backend.wp_device,
        )


__all__ = ["NativeWrenchKernel"]
