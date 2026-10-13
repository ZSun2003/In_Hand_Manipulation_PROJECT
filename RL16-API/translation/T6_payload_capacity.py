#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用当前 TRAnslation 机械手几何计算四指夹持时的理论搬运质量。

这不是 MuJoCo 仿真。程序只读取 Project.xml / Gripper.xml / Share.xml，
复用 T6_kinematics.py 的解析式闭链 IK，并用虚功关系

    effort = J.T @ fingertip_force

计算每根手指的 linear / back / front 执行器负载。armbase 的姿态仍用于
坐标变换，但其电机力矩约束按用户要求完全忽略。

默认工况对应当前控制器：

* 50 mm 立方体、四个水平侧面接触；
* 半径 16 mm 的橡胶球形 hard tip 与物体形成直径 20 mm 的圆形接触斑；
* 从 Project.xml 的取物位置向上抬 25 mm；
* 可用 ``--planar-offset-mm DX DY`` 指定水平搬运距离；程序依次计算
  抬升、恒定高度直线平移、下降放置三段路径，并取整条路径的最小能力；
* 橡胶 hard tip 的静摩擦系数取 1.2；
* 自动搜索 -50...0 mm 的固定 linear 位置，最大化抬升路径采样点的最小载荷；
* DYNAMIXEL 同时给出堵转瞬时数学上限和 ROBOTIS 建议的 20% 堵转扭矩
  一般使用估算，再给出默认 1.5 安全系数后的设计起点。

重要限制：

* DYNAMIXEL 的 stall torque 不是连续额定扭矩，绝不能用来设计持续负载。
* XL430-W250-T 没有电流控制模式；20% 只是 ROBOTIS 选型建议，不是认证的
  连续扭矩额定值。
* IAI EC-SL3L 的 16 N 是押付模式推力，不是垂直搬运额定能力；本程序只用
  0.7 kg 垂直搬运额定值。
* 当前完整型号没有 ``-B``，所以断电 / Servo OFF 时没有厂家保证的制动保持。
* 按本任务要求，默认完全不计机械手自身重量：不扣除每个 linear 滑台上的
  DYNAMIXEL、连杆、支架或线缆质量，也不计它们对 back/front 的自重矩。
  ``--extra-slider-load-kg`` 和 ``--rotary-reserve-nm`` 只保留为可选的后续
  工程校核参数，默认都为 0。
* 结果只适用于准静态、刚性结构、均匀四点接触、不滑动且物体重心位于中心
  的工况；水平搬运加速度按接近 0 处理，不包含冲击、速度-扭矩下降、结构
  强度、轴承寿命、齿隙、温升、电源压降和接触材料失效。

示例：

    ./RL_env/bin/python theoretical_payload.py
    ./RL_env/bin/python theoretical_payload.py --friction 0.5
    ./RL_env/bin/python theoretical_payload.py --contact-patch-diameter-mm 16
    ./RL_env/bin/python theoretical_payload.py --planar-offset-mm 50 0
    ./RL_env/bin/python theoretical_payload.py --linear-position-mm 0
    ./RL_env/bin/python theoretical_payload.py --extra-slider-load-kg 0.12
    ./RL_env/bin/python theoretical_payload.py --lift-accel-g 0.3
    ./RL_env/bin/python theoretical_payload.py --self-test
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MODULE_DIR = Path(__file__).resolve().parent
LOCAL_ENV_DIR = (Path(os.environ["IRP_PYTHON_ENV"]) if os.environ.get("IRP_PYTHON_ENV") else next((parent / "RL_env" for parent in Path(__file__).resolve().parents if (parent / "RL_env" / "build.sh").is_file()), Path(sys.prefix)))
DEFAULT_XML = MODULE_DIR / "Project.xml"

# Match the other executable programs in this folder.  Importing this module does
# not re-exec the caller; direct invocation uses the checked local environment.
if __name__ == "__main__" and Path(sys.prefix).resolve() != LOCAL_ENV_DIR.resolve():
    local_python = LOCAL_ENV_DIR / "bin" / "python"
    if not local_python.is_file():
        raise SystemExit(f"找不到本目录虚拟环境: {local_python}")
    os.execv(
        str(local_python),
        [str(local_python), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

import numpy as np
from scipy.optimize import linprog

from T6_kinematics import (
    FINGERS,
    FingerGeometry,
    FingerSolution,
    GripperIK,
    IKUnreachableError,
    normalize,
    quat_to_rot,
    rot_to_quat,
    rot_axis,
)


G = 9.80665
LOAD_BEARING_ROTARY_MOTORS = 8
TOTAL_ROTARY_MOTORS = 12


@dataclass(frozen=True)
class DynamixelSpec:
    key: str
    name: str
    recommended_voltage_v: float
    maximum_voltage_v: float
    mass_kg: float
    gear_ratio: float
    stall_torque_nm: Mapping[float, float]
    stall_current_a: Mapping[float, float]

    def torque_at(self, voltage_v: float) -> float:
        return _exact_lookup(self.stall_torque_nm, voltage_v, self.name, "扭矩")

    def current_at(self, voltage_v: float) -> float:
        return _exact_lookup(self.stall_current_a, voltage_v, self.name, "电流")


MOTOR_SPECS: Dict[str, DynamixelSpec] = {
    "xl430": DynamixelSpec(
        key="xl430",
        name="DYNAMIXEL XL430-W250-T",
        recommended_voltage_v=11.1,
        maximum_voltage_v=12.0,
        mass_kg=0.0572,
        gear_ratio=258.5,
        stall_torque_nm={9.0: 1.0, 11.1: 1.4, 12.0: 1.5},
        stall_current_a={9.0: 1.0, 11.1: 1.3, 12.0: 1.4},
    ),
    "xc430": DynamixelSpec(
        key="xc430",
        name="DYNAMIXEL XC430-W240-T",
        recommended_voltage_v=12.0,
        maximum_voltage_v=14.8,
        mass_kg=0.0650,
        gear_ratio=245.22,
        stall_torque_nm={9.0: 1.4, 11.1: 1.7, 12.0: 1.9},
        stall_current_a={9.0: 1.1, 11.1: 1.3, 12.0: 1.4},
    ),
}


@dataclass(frozen=True)
class IAISpec:
    name: str = "IAI EC-SL3L-50-1-0"
    stroke_m: float = 0.050
    rated_vertical_payload_kg: float = 0.7
    rated_horizontal_payload_kg: float = 2.0
    maximum_pressing_force_n: float = 16.0
    rated_max_acceleration_g: float = 0.3
    body_mass_kg_without_brake: float = 0.25
    dynamic_moment_nm: Tuple[float, float, float] = (4.71, 4.71, 8.84)
    static_moment_nm: Tuple[float, float, float] = (11.7, 11.7, 22.0)
    has_brake: bool = False


IAI = IAISpec()


@dataclass(frozen=True)
class ContactPatchGeometry:
    sphere_radius_m: float
    patch_diameter_m: float
    patch_area_m2: float
    center_to_contact_plane_m: float
    equivalent_compression_m: float


def contact_patch_from_diameter(
    sphere_radius_m: float,
    patch_diameter_m: float,
) -> ContactPatchGeometry:
    """Equivalent plane/sphere geometry for a circular contact patch."""
    radius = float(sphere_radius_m)
    patch_radius = 0.5 * float(patch_diameter_m)
    if radius <= 0.0:
        raise ValueError("球形 hard tip 半径必须大于 0")
    if not 0.0 < patch_radius < radius:
        raise ValueError(
            "接触斑直径必须大于 0 且严格小于 hard-tip 球直径"
        )
    center_to_plane = math.sqrt(
        max(0.0, radius * radius - patch_radius * patch_radius)
    )
    return ContactPatchGeometry(
        sphere_radius_m=radius,
        patch_diameter_m=2.0 * patch_radius,
        patch_area_m2=math.pi * patch_radius * patch_radius,
        center_to_contact_plane_m=center_to_plane,
        equivalent_compression_m=radius - center_to_plane,
    )


def contact_patch_from_compression(
    sphere_radius_m: float,
    compression_m: float,
) -> ContactPatchGeometry:
    """Backward-compatible conversion from virtual indentation to patch size."""
    radius = float(sphere_radius_m)
    compression = float(compression_m)
    if radius <= 0.0:
        raise ValueError("球形 hard tip 半径必须大于 0")
    if not 0.0 <= compression < radius:
        raise ValueError("压入量必须在 [0, hard-tip 球半径) 内")
    center_to_plane = radius - compression
    patch_radius = math.sqrt(
        max(0.0, radius * radius - center_to_plane * center_to_plane)
    )
    return ContactPatchGeometry(
        sphere_radius_m=radius,
        patch_diameter_m=2.0 * patch_radius,
        patch_area_m2=math.pi * patch_radius * patch_radius,
        center_to_contact_plane_m=center_to_plane,
        equivalent_compression_m=compression,
    )


class PayloadGripperIK(GripperIK):
    """Analytic IK variant that can evaluate one exact fixed linear position."""

    forced_linear_m: Optional[float] = None

    def _linear_samples(
        self,
        finger: str,
        prev_linear: Optional[float] = None,
    ) -> np.ndarray:
        if self.forced_linear_m is None:
            return super()._linear_samples(finger, prev_linear)
        low, high = self.fingers[finger].active_ranges["linear"]
        value = float(self.forced_linear_m)
        if not low - 1.0e-12 <= value <= high + 1.0e-12:
            return np.empty(0, dtype=float)
        return np.array([float(np.clip(value, low, high))], dtype=float)


@dataclass(frozen=True)
class FingerStatics:
    finger: str
    solution: FingerSolution
    jacobian_world: np.ndarray
    angular_jacobian_world: np.ndarray
    linkage_constraint_condition: float
    normal_world: np.ndarray
    surface_offset_world: np.ndarray
    contact_arm_h: np.ndarray
    rotation_world_from_h: np.ndarray


@dataclass(frozen=True)
class PoseStatics:
    center_world: np.ndarray
    linear_reference_m: float
    fingers: Dict[str, FingerStatics]


@dataclass(frozen=True)
class EffortLimits:
    rotary_torque_nm: float
    linear_axial_force_n: float
    slider_vertical_payload_force_n: float
    slider_transverse_force_n: Optional[float]
    slider_moment_nm: Optional[Tuple[float, float, float]]


@dataclass(frozen=True)
class CapacityResult:
    lift_force_n: float
    payload_kg: float
    normal_forces_n: Dict[str, float]
    vertical_forces_n: Dict[str, float]
    utilization: Dict[str, float]
    limiting_constraints: Tuple[str, ...]


@dataclass(frozen=True)
class PathResult:
    motor: DynamixelSpec
    voltage_v: float
    label: str
    rotary_torque_nm: float
    linear_reference_m: float
    path_capacity_kg: float
    bottleneck_center_world: np.ndarray
    bottleneck_result: CapacityResult
    maximum_linkage_condition: float


def _exact_lookup(
    table: Mapping[float, float],
    voltage_v: float,
    model_name: str,
    quantity: str,
) -> float:
    for key, value in table.items():
        if math.isclose(float(key), float(voltage_v), rel_tol=0.0, abs_tol=1e-9):
            return float(value)
    supported = ", ".join(f"{value:g}" for value in sorted(table))
    raise ValueError(
        f"{model_name} 没有 {voltage_v:g} V 的官方{quantity}点；"
        f"可用电压为 {supported} V，本程序不外推。"
    )


def _append_abs_constraint(
    rows: List[np.ndarray],
    bounds: List[float],
    coefficients: np.ndarray,
    limit: float,
) -> None:
    rows.append(np.asarray(coefficients, dtype=float))
    bounds.append(float(limit))
    rows.append(-np.asarray(coefficients, dtype=float))
    bounds.append(float(limit))


def _planar_closed_chain_jacobian(
    geometry: FingerGeometry,
    solution: FingerSolution,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return translational/angular planar Jacobians and loop condition."""
    theta_back = (
        geometry.theta_back0
        + geometry.sigma_back * float(solution.q_back)
    )
    theta_front = (
        geometry.theta_front0
        + geometry.sigma_front * float(solution.q_front)
    )
    theta_ab = (
        theta_back
        + geometry.rel_short0
        + geometry.sigma_short * float(solution.q_short)
    )

    point_a = geometry.O2 + geometry.l1 * np.array(
        [math.cos(theta_back), math.sin(theta_back)],
        dtype=float,
    )
    point_b = point_a + geometry.l2 * np.array(
        [math.cos(theta_ab), math.sin(theta_ab)],
        dtype=float,
    )
    point_c = geometry.M2 + geometry.l3 * np.array(
        [math.cos(theta_front), math.sin(theta_front)],
        dtype=float,
    )
    link_ab = point_b - point_a
    link_bc = point_c - point_b
    loop_matrix = np.vstack((link_ab, link_bc))
    condition = float(np.linalg.cond(loop_matrix))
    if not np.isfinite(condition) or condition > 1.0e10:
        raise ValueError(
            f"{geometry.finger} 指闭链接近/处于奇异位形，"
            f"constraint condition={condition:.3e}"
        )

    columns: List[np.ndarray] = []
    angular_rates: List[float] = []
    for which in ("back", "front"):
        if which == "back":
            d_a = (
                geometry.l1
                * geometry.sigma_back
                * np.array(
                    [-math.sin(theta_back), math.cos(theta_back)],
                    dtype=float,
                )
            )
            d_c = np.zeros(2, dtype=float)
        else:
            d_a = np.zeros(2, dtype=float)
            d_c = (
                geometry.l3
                * geometry.sigma_front
                * np.array(
                    [-math.sin(theta_front), math.cos(theta_front)],
                    dtype=float,
                )
            )

        rhs = np.array(
            [float(np.dot(link_ab, d_a)), float(np.dot(link_bc, d_c))],
            dtype=float,
        )
        d_b = np.linalg.solve(loop_matrix, rhs)
        d_bc = d_c - d_b
        d_theta_bc = (
            link_bc[0] * d_bc[1] - link_bc[1] * d_bc[0]
        ) / float(np.dot(link_bc, link_bc))
        theta_bc = math.atan2(float(link_bc[1]), float(link_bc[0]))
        tip_angle = theta_bc - geometry.kappa
        d_tip = d_c + geometry.r_tip * np.array(
            [-math.sin(tip_angle), math.cos(tip_angle)],
            dtype=float,
        ) * d_theta_bc
        columns.append(d_tip)
        angular_rates.append(float(d_theta_bc))

    return (
        np.column_stack(columns),
        np.asarray(angular_rates, dtype=float),
        condition,
    )


def finger_tip_jacobian_world(
    geometry: FingerGeometry,
    solution: FingerSolution,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return translational/angular 3x3 J, columns [linear, back, front]."""
    planar_jacobian, angular_rates, condition = _planar_closed_chain_jacobian(
        geometry,
        solution,
    )
    arm_rotation = rot_axis(
        geometry.armbase_axis_A,
        float(solution.q_armbase),
    )
    rotation_world_from_h = (
        geometry.R_A_world @ arm_rotation @ geometry.R_AH0
    )
    linear_column = rotation_world_from_h @ geometry.linear_axis_H
    rotary_columns: List[np.ndarray] = []
    for column in planar_jacobian.T:
        derivative_h = (
            geometry.eX_H * float(column[0])
            + geometry.eY_H * float(column[1])
        )
        rotary_columns.append(rotation_world_from_h @ derivative_h)
    jacobian = np.column_stack(
        (linear_column, rotary_columns[0], rotary_columns[1])
    )
    plane_axis_world = (
        rotation_world_from_h @ geometry.plane_normal_H
    )
    angular_jacobian = np.column_stack(
        (
            np.zeros(3, dtype=float),
            plane_axis_world * float(angular_rates[0]),
            plane_axis_world * float(angular_rates[1]),
        )
    )
    return (
        jacobian.astype(float),
        angular_jacobian.astype(float),
        condition,
        rotation_world_from_h.astype(float),
    )


def solve_pose_statics(
    ik: GripperIK,
    center_world: Sequence[float],
    object_rotation: np.ndarray,
    contact_distance_m: float,
    object_half_size_m: float,
    *,
    previous_ctrl: Optional[np.ndarray] = None,
) -> Tuple[PoseStatics, np.ndarray]:
    center = np.asarray(center_world, dtype=float).reshape(3)
    solution = ik.solve_cube_pose(
        center,
        quat_world_object=rot_to_quat(object_rotation),
        distances=float(contact_distance_m),
        previous_ctrl=previous_ctrl,
        strict=True,
    )
    finger_statics: Dict[str, FingerStatics] = {}
    for finger in FINGERS:
        finger_solution = solution.finger_solutions[finger]
        geometry = ik.fingers[finger]
        (
            jacobian,
            angular_jacobian,
            condition,
            rotation_world_from_h,
        ) = (
            finger_tip_jacobian_world(geometry, finger_solution)
        )
        normal_world = normalize(
            object_rotation @ ik.contact_directions_world[finger],
            fallback=ik.contact_directions_world[finger],
        )
        if abs(float(normal_world[2])) > 1.0e-7:
            raise ValueError(
                "当前静力模型要求四个物体侧面法向保持水平；"
                "Project.xml 中的物体姿态发生了倾斜。"
            )
        fk = ik.finger_forward(finger, finger_solution)
        # With a finite circular contact patch, the nominal cube surface is
        # contact_distance-half_size from the sphere center (12.49 mm for the
        # default 20 mm patch), rather than the undeformed 16 mm sphere radius.
        contact_radius_m = contact_distance_m - object_half_size_m
        surface_offset_world = -normal_world * contact_radius_m
        # Contact is on the inner side of the spherical fingertip, not at its
        # center.  fk.tip_local is expressed in the moving H/linear-body frame.
        contact_arm_h = (
            fk.tip_local
            - rotation_world_from_h.T
            @ (normal_world * contact_radius_m)
        )
        finger_statics[finger] = FingerStatics(
            finger=finger,
            solution=finger_solution,
            jacobian_world=jacobian,
            angular_jacobian_world=angular_jacobian,
            linkage_constraint_condition=condition,
            normal_world=normal_world,
            surface_offset_world=surface_offset_world.astype(float),
            contact_arm_h=contact_arm_h.astype(float),
            rotation_world_from_h=rotation_world_from_h,
        )

    q_linear_values = np.array(
        [finger.solution.q_linear for finger in finger_statics.values()],
        dtype=float,
    )
    if float(np.ptp(q_linear_values)) > 1.0e-8:
        raise ValueError(
            "四指 linear 解不对称，当前固定-linear 路径模型不能使用。"
        )
    pose = PoseStatics(
        center_world=center,
        linear_reference_m=float(np.mean(q_linear_values)),
        fingers=finger_statics,
    )
    return pose, solution.ctrl.copy()


def effort_limits(
    motor: DynamixelSpec,
    voltage_v: float,
    rotary_fraction: float,
    rotary_reserve_nm: float,
    extra_slider_load_kg: float,
    effective_gravity: float,
    guide_limit: str,
    use_horizontal_payload_proxy: bool,
) -> EffortLimits:
    # Per user requirement, the gripper's own motor/link/bracket/cable masses
    # are not deducted.  Only an explicitly supplied optional extra load is
    # considered.
    carried_known_mass = extra_slider_load_kg
    remaining_vertical_mass = IAI.rated_vertical_payload_kg - carried_known_mass
    remaining_horizontal_mass = (
        IAI.rated_horizontal_payload_kg - carried_known_mass
    )
    if remaining_vertical_mass <= 0.0:
        raise ValueError(
            f"{motor.name}: 每个滑台已知随动质量 {carried_known_mass:.4f} kg "
            f"不小于 IAI 垂直额定负载 {IAI.rated_vertical_payload_kg:.3f} kg。"
        )
    if remaining_horizontal_mass <= 0.0:
        raise ValueError(
            f"{motor.name}: 每个滑台已知随动质量超过 IAI 水平额定负载。"
        )

    rotary_limit = (
        motor.torque_at(voltage_v) * rotary_fraction - rotary_reserve_nm
    )
    if rotary_limit <= 0.0:
        raise ValueError(
            f"{motor.name}: 扭矩利用率后的可用扭矩不大于零；"
            "请减小 --rotary-reserve-nm 或增大利用率。"
        )
    if guide_limit == "dynamic":
        moments: Optional[Tuple[float, float, float]] = IAI.dynamic_moment_nm
    elif guide_limit == "static":
        moments = IAI.static_moment_nm
    elif guide_limit == "ignore":
        moments = None
    else:
        raise ValueError(f"未知 guide_limit: {guide_limit}")
    return EffortLimits(
        rotary_torque_nm=float(rotary_limit),
        linear_axial_force_n=float(
            remaining_vertical_mass * effective_gravity
        ),
        # Also cap each slider's assigned vertical object-force share directly
        # by the catalog's 0.7 kg transported-mass rating.  This avoids treating
        # the catalog number only as one component of an arbitrary 3-D force
        # envelope.
        slider_vertical_payload_force_n=float(
            remaining_vertical_mass * effective_gravity
        ),
        # The catalog's 2 kg horizontal transported mass is not a published
        # arbitrary-direction guide-force rating.  It is therefore disabled by
        # default and available only as an explicitly requested engineering
        # proxy.
        slider_transverse_force_n=(
            float(remaining_horizontal_mass * G)
            if use_horizontal_payload_proxy
            else None
        ),
        slider_moment_nm=moments,
    )


def maximize_payload_at_pose(
    pose: PoseStatics,
    object_half_size_m: float,
    friction: float,
    effective_gravity: float,
    limits: EffortLimits,
) -> CapacityResult:
    """
    Maximize sum(T_i) for four side contacts.

    Variables are [N_U,T_U,N_D,T_D,N_L,T_L,N_R,T_R], where N is inward
    compression on the object and T is upward friction on the object.
    The equalities enforce horizontal force and object-COM moment balance.
    """
    if friction <= 0.0:
        raise ValueError("摩擦系数必须大于 0")
    variable_count = 2 * len(FINGERS)
    index = {
        finger: (2 * i, 2 * i + 1)
        for i, finger in enumerate(FINGERS)
    }
    objective = np.zeros(variable_count, dtype=float)
    for finger in FINGERS:
        objective[index[finger][1]] = -1.0

    rows: List[np.ndarray] = []
    upper_bounds: List[float] = []
    equality_rows: List[np.ndarray] = []
    equality_values: List[float] = []

    # Coulomb friction: 0 <= T_i <= mu*N_i.
    for finger in FINGERS:
        i_n, i_t = index[finger]
        row = np.zeros(variable_count, dtype=float)
        row[i_t] = 1.0
        row[i_n] = -friction
        rows.append(row)
        upper_bounds.append(0.0)

        # Manufacturer vertical transported-mass rating, assigned to the
        # vertical object-force share carried by this slider.
        payload_row = np.zeros(variable_count, dtype=float)
        payload_row[i_t] = 1.0
        rows.append(payload_row)
        upper_bounds.append(limits.slider_vertical_payload_force_n)

    # Horizontal normal-force equilibrium on the object.
    for axis in (0, 1):
        row = np.zeros(variable_count, dtype=float)
        for finger in FINGERS:
            i_n, _i_t = index[finger]
            row[i_n] = -pose.fingers[finger].normal_world[axis]
        if float(np.linalg.norm(row)) > 1.0e-12:
            equality_rows.append(row)
            equality_values.append(0.0)

    # Moment equilibrium about the object COM.  The contact point on the cube is
    # r_i = half_size*n_i; normal forces pass through the COM and add no moment.
    vertical = np.array([0.0, 0.0, 1.0], dtype=float)
    for axis in range(3):
        row = np.zeros(variable_count, dtype=float)
        for finger in FINGERS:
            _i_n, i_t = index[finger]
            radius_world = (
                object_half_size_m * pose.fingers[finger].normal_world
            )
            row[i_t] = np.cross(radius_world, vertical)[axis]
        if float(np.linalg.norm(row)) > 1.0e-12:
            equality_rows.append(row)
            equality_values.append(0.0)

    for finger in FINGERS:
        finger_data = pose.fingers[finger]
        i_n, i_t = index[finger]
        normal = finger_data.normal_world
        # The object pushes the fingertip outward and downward.
        reaction_n_world = normal
        reaction_t_world = -vertical
        contact_moment_n_world = np.cross(
            finger_data.surface_offset_world,
            reaction_n_world,
        )
        contact_moment_t_world = np.cross(
            finger_data.surface_offset_world,
            reaction_t_world,
        )

        # Virtual work at the three load-bearing coordinates.  armbase is
        # intentionally absent.  The angular-Jacobian term includes the moment
        # generated by tangential friction acting on the fixed spherical tip
        # surface rather than at the sphere center.
        for column, limit in (
            (0, limits.linear_axial_force_n),
            (1, limits.rotary_torque_nm),
            (2, limits.rotary_torque_nm),
        ):
            coefficients = np.zeros(variable_count, dtype=float)
            jacobian_column = finger_data.jacobian_world[:, column]
            angular_column = finger_data.angular_jacobian_world[:, column]
            coefficients[i_n] = float(
                np.dot(jacobian_column, reaction_n_world)
                + np.dot(angular_column, contact_moment_n_world)
            )
            coefficients[i_t] = float(
                np.dot(jacobian_column, reaction_t_world)
                + np.dot(angular_column, contact_moment_t_world)
            )
            _append_abs_constraint(
                rows,
                upper_bounds,
                coefficients,
                limit,
            )

        # IAI horizontal-payload rating is used conservatively as a transverse
        # guide-force proxy.  A 16-sided polygon approximates the circular norm
        # in the H-frame x/y plane.
        reaction_n_h = (
            finger_data.rotation_world_from_h.T @ reaction_n_world
        )
        reaction_t_h = (
            finger_data.rotation_world_from_h.T @ reaction_t_world
        )
        if limits.slider_transverse_force_n is not None:
            polygon_scale = math.cos(math.pi / 16.0)
            for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
                direction = np.array(
                    [math.cos(float(angle)), math.sin(float(angle))],
                    dtype=float,
                )
                coefficients = np.zeros(variable_count, dtype=float)
                coefficients[i_n] = float(
                    np.dot(direction, reaction_n_h[:2])
                )
                coefficients[i_t] = float(
                    np.dot(direction, reaction_t_h[:2])
                )
                rows.append(coefficients)
                upper_bounds.append(
                    limits.slider_transverse_force_n * polygon_scale
                )

        # Wrench on the moving IAI carriage.  The normalized L1 interaction is
        # conservative when several Ma/Mb/Mc components occur together.
        if limits.slider_moment_nm is not None:
            moment_n = np.cross(
                finger_data.contact_arm_h,
                reaction_n_h,
            )
            moment_t = np.cross(
                finger_data.contact_arm_h,
                reaction_t_h,
            )
            moment_limits = np.asarray(
                limits.slider_moment_nm,
                dtype=float,
            )
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        signs = np.array([sx, sy, sz], dtype=float)
                        coefficients = np.zeros(variable_count, dtype=float)
                        coefficients[i_n] = float(
                            np.dot(signs, moment_n / moment_limits)
                        )
                        coefficients[i_t] = float(
                            np.dot(signs, moment_t / moment_limits)
                        )
                        rows.append(coefficients)
                        upper_bounds.append(1.0)

    result = linprog(
        objective,
        A_ub=np.vstack(rows),
        b_ub=np.asarray(upper_bounds, dtype=float),
        A_eq=(
            np.vstack(equality_rows)
            if equality_rows
            else None
        ),
        b_eq=(
            np.asarray(equality_values, dtype=float)
            if equality_rows
            else None
        ),
        bounds=[(0.0, None)] * variable_count,
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            "静力优化失败: "
            f"status={result.status}, message={result.message}"
        )

    normal_forces = {
        finger: float(result.x[index[finger][0]])
        for finger in FINGERS
    }
    vertical_forces = {
        finger: float(result.x[index[finger][1]])
        for finger in FINGERS
    }
    lift_force = float(sum(vertical_forces.values()))

    utilization: Dict[str, float] = {}
    for finger in FINGERS:
        finger_data = pose.fingers[finger]
        normal_force = normal_forces[finger]
        vertical_force = vertical_forces[finger]
        reaction_world = (
            normal_force * finger_data.normal_world
            - vertical_force * vertical
        )
        contact_moment_world = np.cross(
            finger_data.surface_offset_world,
            reaction_world,
        )
        efforts = (
            finger_data.jacobian_world.T @ reaction_world
            + finger_data.angular_jacobian_world.T @ contact_moment_world
        )
        utilization[f"{finger}.linear"] = abs(float(efforts[0])) / max(
            limits.linear_axial_force_n,
            1.0e-12,
        )
        utilization[f"{finger}.back"] = abs(float(efforts[1])) / max(
            limits.rotary_torque_nm,
            1.0e-12,
        )
        utilization[f"{finger}.front"] = abs(float(efforts[2])) / max(
            limits.rotary_torque_nm,
            1.0e-12,
        )
        utilization[f"{finger}.friction"] = vertical_force / max(
            friction * normal_force,
            1.0e-12,
        )
        utilization[f"{finger}.iai_payload"] = vertical_force / max(
            limits.slider_vertical_payload_force_n,
            1.0e-12,
        )

        reaction_h = (
            finger_data.rotation_world_from_h.T @ reaction_world
        )
        if limits.slider_transverse_force_n is not None:
            utilization[f"{finger}.guide_force_proxy"] = (
                float(np.linalg.norm(reaction_h[:2]))
                / max(limits.slider_transverse_force_n, 1.0e-12)
            )
        if limits.slider_moment_nm is not None:
            moment_h = np.cross(
                finger_data.contact_arm_h,
                reaction_h,
            )
            utilization[f"{finger}.guide_moment"] = float(
                np.sum(
                    np.abs(
                        moment_h
                        / np.asarray(limits.slider_moment_nm, dtype=float)
                    )
                )
            )

    max_utilization = max(utilization.values(), default=0.0)
    limiting = tuple(
        sorted(
            name
            for name, value in utilization.items()
            if value >= max(0.995, max_utilization - 1.0e-5)
        )
    )
    return CapacityResult(
        lift_force_n=lift_force,
        payload_kg=lift_force / effective_gravity,
        normal_forces_n=normal_forces,
        vertical_forces_n=vertical_forces,
        utilization=utilization,
        limiting_constraints=limiting,
    )


def _inclusive_grid(low: float, high: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("linear 扫描步长必须大于 0")
    count = max(1, int(math.floor((high - low) / step + 1.0e-10)))
    values = low + step * np.arange(count + 1, dtype=float)
    values = values[values <= high + 1.0e-10]
    if values.size == 0 or not math.isclose(
        float(values[-1]),
        high,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        values = np.concatenate((values, np.array([high], dtype=float)))
    return np.unique(np.clip(values, low, high))


def _append_linear_segment(
    points: List[np.ndarray],
    start: np.ndarray,
    end: np.ndarray,
    samples: int,
) -> None:
    """Append a Cartesian line segment without duplicating shared endpoints."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if samples < 1:
        raise ValueError("路径段采样数必须不小于 1")
    if np.allclose(start, end, rtol=0.0, atol=1.0e-14):
        candidates = (start,)
    else:
        if samples < 2:
            raise ValueError("非零路径段的采样数必须不小于 2")
        candidates = tuple(
            start + float(alpha) * (end - start)
            for alpha in np.linspace(0.0, 1.0, samples, dtype=float)
        )
    for candidate in candidates:
        if not points or not np.allclose(
            points[-1],
            candidate,
            rtol=0.0,
            atol=1.0e-14,
        ):
            points.append(np.asarray(candidate, dtype=float))


def build_pick_transfer_place_centers(
    pickup_center: np.ndarray,
    lift_height_m: float,
    planar_offset_xy_m: Sequence[float],
    lift_samples: int,
    transfer_samples: int,
) -> List[np.ndarray]:
    """Build pickup -> vertical lift -> horizontal transfer -> place samples.

    With zero planar offset, retain the historical one-way lift path because
    the descent visits the same quasi-static poses in reverse and cannot
    change the path bottleneck.
    """
    pickup = np.asarray(pickup_center, dtype=float)
    offset_xy = np.asarray(planar_offset_xy_m, dtype=float)
    if pickup.shape != (3,) or offset_xy.shape != (2,):
        raise ValueError("取物中心必须为 3 维，平面偏移必须为 2 维")
    raised_pickup = pickup + np.array([0.0, 0.0, lift_height_m])
    points: List[np.ndarray] = []
    _append_linear_segment(points, pickup, raised_pickup, lift_samples)

    if np.linalg.norm(offset_xy) <= 1.0e-14:
        return points

    transfer_offset = np.array([offset_xy[0], offset_xy[1], 0.0])
    raised_place = raised_pickup + transfer_offset
    place = pickup + transfer_offset
    _append_linear_segment(
        points,
        raised_pickup,
        raised_place,
        transfer_samples,
    )
    _append_linear_segment(points, raised_place, place, lift_samples)
    return points


def build_candidate_paths(
    ik: PayloadGripperIK,
    centers_world: Sequence[np.ndarray],
    object_rotation: np.ndarray,
    contact_distance_m: float,
    object_half_size_m: float,
    linear_references_m: Iterable[float],
    linear_match_tolerance_m: float,
) -> Dict[float, List[PoseStatics]]:
    candidates: Dict[float, List[PoseStatics]] = {}
    for reference in linear_references_m:
        reference = float(reference)
        ik.linear_ref_m = reference
        ik.forced_linear_m = reference
        previous_ctrl: Optional[np.ndarray] = None
        path: List[PoseStatics] = []
        feasible = True
        for center in centers_world:
            try:
                pose, previous_ctrl = solve_pose_statics(
                    ik,
                    center,
                    object_rotation,
                    contact_distance_m,
                    object_half_size_m,
                    previous_ctrl=previous_ctrl,
                )
            except (IKUnreachableError, ValueError, np.linalg.LinAlgError):
                feasible = False
                break
            if abs(pose.linear_reference_m - reference) > linear_match_tolerance_m:
                feasible = False
                break
            path.append(pose)
        if feasible and len(path) == len(centers_world):
            candidates[reference] = path
    return candidates


def optimize_path_for_profile(
    candidate_paths: Mapping[float, Sequence[PoseStatics]],
    motor: DynamixelSpec,
    voltage_v: float,
    label: str,
    rotary_fraction: float,
    rotary_reserve_nm: float,
    extra_slider_load_kg: float,
    object_half_size_m: float,
    friction: float,
    effective_gravity: float,
    guide_limit: str,
    use_horizontal_payload_proxy: bool,
) -> PathResult:
    limits = effort_limits(
        motor,
        voltage_v,
        rotary_fraction,
        rotary_reserve_nm,
        extra_slider_load_kg,
        effective_gravity,
        guide_limit,
        use_horizontal_payload_proxy,
    )
    best: Optional[PathResult] = None
    for reference, poses in candidate_paths.items():
        pose_results = [
            maximize_payload_at_pose(
                pose,
                object_half_size_m,
                friction,
                effective_gravity,
                limits,
            )
            for pose in poses
        ]
        bottleneck_index = min(
            range(len(pose_results)),
            key=lambda i: pose_results[i].payload_kg,
        )
        bottleneck = pose_results[bottleneck_index]
        maximum_condition = max(
            finger.linkage_constraint_condition
            for pose in poses
            for finger in pose.fingers.values()
        )
        candidate = PathResult(
            motor=motor,
            voltage_v=voltage_v,
            label=label,
            rotary_torque_nm=limits.rotary_torque_nm,
            linear_reference_m=float(reference),
            path_capacity_kg=bottleneck.payload_kg,
            bottleneck_center_world=poses[bottleneck_index].center_world.copy(),
            bottleneck_result=bottleneck,
            maximum_linkage_condition=float(maximum_condition),
        )
        if (
            best is None
            or candidate.path_capacity_kg > best.path_capacity_kg + 1.0e-12
        ):
            best = candidate
    if best is None:
        raise RuntimeError("没有可评估的 fixed-linear 取放路径")
    return best


def resolve_voltage(spec: DynamixelSpec, voltage_argument: str) -> float:
    if voltage_argument == "auto":
        return spec.recommended_voltage_v
    voltage = float(voltage_argument)
    # torque_at performs the exact official-point check.
    spec.torque_at(voltage)
    return voltage


def _format_limit_names(names: Sequence[str]) -> str:
    if not names:
        return "未识别"
    simplified = sorted(
        {
            name.split(".", 1)[1] if "." in name else name
            for name in names
        }
    )
    translation = {
        "linear": "IAI 轴向力",
        "back": "back 电机扭矩",
        "front": "front 电机扭矩",
        "friction": "接触摩擦",
        "iai_payload": "IAI 0.7 kg 垂直搬运额定",
        "guide_force_proxy": "IAI 横向载荷代理",
        "guide_moment": "IAI 导轨力矩",
    }
    return " + ".join(translation.get(name, name) for name in simplified)


def print_report(
    args: argparse.Namespace,
    ik: GripperIK,
    centers_world: Sequence[np.ndarray],
    contact_distance_m: float,
    contact_patch: ContactPatchGeometry,
    candidate_paths: Mapping[float, Sequence[PoseStatics]],
    results: Mapping[str, Mapping[str, PathResult]],
) -> None:
    start = np.asarray(centers_world[0], dtype=float)
    end = np.asarray(centers_world[-1], dtype=float)
    print("=" * 78)
    print("当前四指机械手理论搬运能力（解析静力计算；不是 MuJoCo 仿真）")
    print("=" * 78)
    print(f"几何来源       : {ik.model_info.project_xml}")
    print(
        "物体           : "
        f"边长 {args.object_size_mm:.2f} mm, "
        f"取物中心 {start.tolist()} m"
    )
    planar_offset = np.asarray(args.planar_offset_mm, dtype=float)
    if np.linalg.norm(planar_offset) > 1.0e-12:
        print(
            "取放路径       : "
            f"抬升 {args.lift_height_mm:.2f} mm -> "
            f"XY 平移 [{planar_offset[0]:+.2f}, "
            f"{planar_offset[1]:+.2f}] mm -> 下降放置；"
            f"终点 {end.tolist()} m，共 {len(centers_world)} 个解析采样点"
        )
    else:
        print(
            "取放路径       : "
            f"同一 XY 位置竖直抬升 {args.lift_height_mm:.2f} mm；"
            "下降段与抬升段位姿相同，未重复采样；"
            f"共 {len(centers_world)} 个解析采样点"
        )
    print(
        "接触假设       : "
        f"橡胶球面, mu={args.friction:.3f}, "
        f"圆形接触斑直径={contact_patch.patch_diameter_m * 1000.0:.2f} mm, "
        f"面积={contact_patch.patch_area_m2 * 1.0e6:.2f} mm²"
    )
    print(
        "等效接触几何   : "
        f"球心到接触平面={contact_patch.center_to_contact_plane_m * 1000.0:.3f} mm, "
        f"压入={contact_patch.equivalent_compression_m * 1000.0:.3f} mm, "
        f"物心到球心={contact_distance_m * 1000.0:.3f} mm"
    )
    print(
        "运动假设       : "
        f"向上加速度={args.lift_accel_g:.3f} g, "
        "水平加速度≈0（准静态）, "
        f"安全系数={args.safety_factor:.3f}, "
        f"IAI 导轨限制={args.guide_limit}"
    )
    if args.linear_position_mm == "auto":
        refs = sorted(candidate_paths)
        print(
            "linear 搜索    : "
            f"{refs[0] * 1000.0:+.1f}...{refs[-1] * 1000.0:+.1f} mm, "
            f"步长 {args.linear_scan_step_mm:.2f} mm；"
            "对整条取放路径最大化最小载荷"
        )
    else:
        only_ref = next(iter(candidate_paths))
        print(f"linear 固定位置: {only_ref * 1000.0:+.2f} mm")
    print(
        "机械手自身重量 : 按要求完全忽略；"
        f"用户额外指定载荷={args.extra_slider_load_kg:.4f} kg/滑台"
    )
    print(
        "旋转扭矩预留   : "
        f"{args.rotary_reserve_nm:.4f} N·m/电机；armbase 力矩约束已忽略"
    )
    print(
        "IAI 横向 2 kg  : "
        + (
            "已按用户要求启用为横向导轨力工程代理"
            if args.use_iai_horizontal_payload_proxy
            else "未当作任意方向导轨力额定值（默认关闭代理）"
        )
    )
    print()

    for motor_key, profiles in results.items():
        spec = MOTOR_SPECS[motor_key]
        general = profiles["general"]
        peak = profiles["peak"]
        voltage = general.voltage_v
        stall_torque = spec.torque_at(voltage)
        stall_current = spec.current_at(voltage)
        residual_vertical = (
            IAI.rated_vertical_payload_kg
            - args.extra_slider_load_kg
        )

        print(f"{spec.name} @ {voltage:g} V")
        print(
            f"  官方堵转点   : {stall_torque:.3f} N·m @ "
            f"{stall_current:.3f} A/电机；质量 {spec.mass_kg * 1000.0:.1f} g"
        )
        print(
            f"  IAI 物体额定 : {residual_vertical:.4f} kg/滑台 "
            f"（按要求不扣机械手自身重量）"
        )
        print(
            f"  一般使用估算 : {general.path_capacity_kg:.4f} kg "
            f"({general.path_capacity_kg * 1000.0:.0f} g), "
            f"扭矩 {general.rotary_torque_nm:.3f} N·m, "
            f"linear={general.linear_reference_m * 1000.0:+.1f} mm"
        )
        print(
            "                  路径瓶颈 xyz="
            f"{general.bottleneck_center_world.tolist()} m, "
            f"约束={_format_limit_names(general.bottleneck_result.limiting_constraints)}, "
            f"每指 N={np.mean(list(general.bottleneck_result.normal_forces_n.values())):.3f} N, "
            f"T={np.mean(list(general.bottleneck_result.vertical_forces_n.values())):.3f} N, "
            f"平均压强="
            f"{np.mean(list(general.bottleneck_result.normal_forces_n.values())) / contact_patch.patch_area_m2 / 1000.0:.2f} kPa"
        )
        design_payload = general.path_capacity_kg / args.safety_factor
        print(
            f"  设计起点     : {design_payload:.4f} kg "
            f"({design_payload * 1000.0:.0f} g), "
            f"即上述一般使用估算再除安全系数 {args.safety_factor:g}"
        )
        print(
            f"  堵转瞬时上限 : {peak.path_capacity_kg:.4f} kg "
            f"({peak.path_capacity_kg * 1000.0:.0f} g), "
            f"linear={peak.linear_reference_m * 1000.0:+.1f} mm, "
            f"约束={_format_limit_names(peak.bottleneck_result.limiting_constraints)}；"
            "不可持续、不可作为选型载荷"
        )
        print(
            "  峰值供电提示 : "
            f"8 个承载旋转电机同时堵转约 {LOAD_BEARING_ROTARY_MOTORS * stall_current:.1f} A；"
            f"12 个旋转电机合计约 {TOTAL_ROTARY_MOTORS * stall_current:.1f} A"
        )
        print()

    print("结论边界：")
    print(
        "  1) 库仑摩擦上限是 mu*N；接触面积不再重复乘入摩擦力。面积用于"
        "等效球面几何和平均接触压强。"
    )
    print(
        "  2) 按要求没有计入机械手自身重量与自重矩，因此结果是物体外载荷的"
        "理想理论能力。"
    )
    print(
        "  3) IAI 16 N 押付力没有用于搬运；当前型号无 -B，断电不能保证保持。"
    )
    print(
        "  4) 还必须单独校核指尖、连杆、销轴、安装螺钉、轴承和被抓物表面强度。"
    )
    print(
        "  5) 路径结论只覆盖上面列出的离散笛卡尔路径采样点；IAI 导轨力矩以 XML 的 "
        "H 原点近似，实物必须按厂家基准与悬伸长度复核。"
    )
    print(
        "  6) 由 20 mm 圆斑得到的 3.51 mm 是球-平面截交的等效几何压入，"
        "不是橡胶材料模型；最好用实测球心到物面的距离替换。"
    )
    print(
        "  7) 平面搬运按低加速度准静态处理；若存在快速启停、急停或转弯，"
        "必须另加水平惯性力并重新计算二维切向摩擦。"
    )
    print()
    print("官方规格来源：")
    print(
        "  IAI EC-SL3: "
        "https://www.iai-robot.co.jp/product/details/EC-SL3/index.html"
    )
    print(
        "  IAI 规格 PDF: "
        "https://www.iai-robot.co.jp/product/details/EC-SL3/EC-SL3.pdf"
    )
    print(
        "  XL430: "
        "https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/"
    )
    print(
        "  XC430: "
        "https://emanual.robotis.com/docs/en/dxl/x/xc430-w240/"
    )
    print(
        "  ROBOTIS 20% 选型指南: "
        "https://emanual.robotis.com/docs/en/reference/dxl-selection-guide/"
    )


def run_self_test(xml_path: Path) -> None:
    ik = PayloadGripperIK.from_xml(xml_path, linear_ref_m=0.0)
    ik.forced_linear_m = 0.0
    pickup = ik.model_info.cube_pos.copy()
    pick_place_path = build_pick_transfer_place_centers(
        pickup,
        0.025,
        (0.020, -0.010),
        3,
        4,
    )
    if len(pick_place_path) != 8:
        raise AssertionError(
            f"取放路径采样去重检查失败: {len(pick_place_path)} != 8"
        )
    expected_waypoints = (
        pickup,
        pickup + np.array([0.0, 0.0, 0.025]),
        pickup + np.array([0.020, -0.010, 0.025]),
        pickup + np.array([0.020, -0.010, 0.0]),
    )
    sampled_waypoint_indices = (0, 2, 5, 7)
    for index, expected_point in zip(
        sampled_waypoint_indices,
        expected_waypoints,
    ):
        if not np.allclose(
            pick_place_path[index],
            expected_point,
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise AssertionError(
                "取放路径 waypoint 检查失败: "
                f"index={index}, actual={pick_place_path[index]}, "
                f"expected={expected_point}"
            )
    contact_patch = contact_patch_from_diameter(
        ik.model_info.hard_tip_radius,
        0.020,
    )
    contact_distance = (
        ik.model_info.cube_half_size
        + contact_patch.center_to_contact_plane_m
    )
    pose, _ctrl = solve_pose_statics(
        ik,
        ik.model_info.cube_pos,
        quat_to_rot(ik.model_info.cube_quat),
        contact_distance,
        ik.model_info.cube_half_size,
    )
    right = pose.fingers["R"]
    radial = right.normal_world
    expected = np.array(
        [
            [-0.36848945, 0.17838075, 0.30070580],
            [0.34681490, -0.28238049, 0.95371695],
        ],
        dtype=float,
    )
    actual = np.array(
        [
            [
                float(np.dot(radial, right.jacobian_world[:, 1])),
                float(np.dot(radial, right.jacobian_world[:, 2])),
                float(np.dot(radial, right.jacobian_world[:, 0])),
            ],
            [
                float(right.jacobian_world[2, 1]),
                float(right.jacobian_world[2, 2]),
                float(right.jacobian_world[2, 0]),
            ],
        ],
        dtype=float,
    )
    if not np.allclose(actual, expected, rtol=0.0, atol=2.0e-7):
        raise AssertionError(
            "当前 XML 的 Jacobian 回归检查失败:\n"
            f"actual=\n{actual}\nexpected=\n{expected}"
        )

    limits_xl = effort_limits(
        MOTOR_SPECS["xl430"],
        11.1,
        0.20,
        0.0,
        0.0,
        G,
        "dynamic",
        False,
    )
    capacity_xl = maximize_payload_at_pose(
        pose,
        ik.model_info.cube_half_size,
        1.2,
        G,
        limits_xl,
    )
    limits_xc = effort_limits(
        MOTOR_SPECS["xc430"],
        12.0,
        0.20,
        0.0,
        0.0,
        G,
        "dynamic",
        False,
    )
    capacity_xc = maximize_payload_at_pose(
        pose,
        ik.model_info.cube_half_size,
        1.2,
        G,
        limits_xc,
    )
    if not (0.0 < capacity_xl.payload_kg < capacity_xc.payload_kg):
        raise AssertionError(
            "电机载荷单调性检查失败: "
            f"XL={capacity_xl.payload_kg}, XC={capacity_xc.payload_kg}"
        )
    print("self-test: PASS")
    print("R finger J rows [radial,z], cols [back,front,linear]:")
    print(actual)
    print(
        "fixed linear=0 mm, patch=20 mm, mu=1.2, general-use payload: "
        f"XL={capacity_xl.payload_kg:.6f} kg, "
        f"XC={capacity_xc.payload_kg:.6f} kg"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "不用 MuJoCo 仿真，按当前 XML 解析几何、厂家执行器规格与四点"
            "摩擦静力学计算理论搬运质量。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="本地 Project.xml",
    )
    parser.add_argument(
        "--motor",
        choices=("both", "xl430", "xc430"),
        default="both",
        help="计算哪种旋转电机方案",
    )
    parser.add_argument(
        "--voltage",
        choices=("auto", "9", "11.1", "12"),
        default="auto",
        help="auto 使用 XL=11.1 V、XC=12 V；只允许厂家公布扭矩的电压点",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=1.2,
        help="实物指尖-物体静摩擦系数（不是 MuJoCo 参数）",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=1.5,
        help="设计起点对一般使用估算采用的载荷安全系数",
    )
    parser.add_argument(
        "--dxl-general-fraction",
        type=float,
        default=0.20,
        help="一般使用估算占官方堵转扭矩的比例",
    )
    parser.add_argument(
        "--rotary-reserve-nm",
        type=float,
        default=0.0,
        help="每个 back/front 电机为连杆自重、摩擦和线缆预留的扭矩",
    )
    parser.add_argument(
        "--extra-slider-load-kg",
        type=float,
        default=0.0,
        help="可选：每个 IAI 滑台额外计入的质量；默认 0，即忽略机械手自身重量",
    )
    parser.add_argument(
        "--object-size-mm",
        type=float,
        default=None,
        help="立方体边长；默认从 Project.xml 读取",
    )
    parser.add_argument(
        "--center-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="抬升起点的物体中心世界坐标；默认从 Project.xml 读取",
    )
    contact_group = parser.add_mutually_exclusive_group()
    contact_group.add_argument(
        "--contact-patch-diameter-mm",
        type=float,
        default=None,
        help="圆形接触斑直径；若未指定接触参数，默认使用 20 mm",
    )
    contact_group.add_argument(
        "--contact-compression-mm",
        type=float,
        default=None,
        help="可选旧接口：直接指定球形硬指尖压入量，并反算圆形接触斑",
    )
    parser.add_argument(
        "--lift-height-mm",
        type=float,
        default=25.0,
        help="取物和放物阶段的竖直抬升高度；0 表示不抬升",
    )
    parser.add_argument(
        "--path-samples",
        type=int,
        default=26,
        help="每个竖直路径段包含首尾的采样数",
    )
    parser.add_argument(
        "--planar-offset-mm",
        type=float,
        nargs=2,
        metavar=("DX", "DY"),
        default=(0.0, 0.0),
        help=(
            "从取物点到放物点的水平 XY 偏移；非零时计算"
            "抬升、恒高直线平移、下降三段路径"
        ),
    )
    parser.add_argument(
        "--transfer-samples",
        type=int,
        default=26,
        help="水平直线搬运段包含首尾的采样数",
    )
    parser.add_argument(
        "--lift-accel-g",
        type=float,
        default=0.0,
        help="向上准静态等效加速度，以 g 为单位；IAI 额定范围最大 0.3 g",
    )
    parser.add_argument(
        "--linear-position-mm",
        default="auto",
        help="固定 linear 位置（-50...0 mm）或 auto 搜索路径采样点最优值",
    )
    parser.add_argument(
        "--linear-scan-step-mm",
        type=float,
        default=1.0,
        help="linear-position=auto 时的扫描步长",
    )
    parser.add_argument(
        "--guide-limit",
        choices=("dynamic", "static", "ignore"),
        default="dynamic",
        help="IAI Ma/Mb/Mc 使用动态、静态或忽略；搬运默认 dynamic",
    )
    parser.add_argument(
        "--use-iai-horizontal-payload-proxy",
        action="store_true",
        help=(
            "可选：把 IAI 水平可搬质量 2 kg 换算为横向导轨力代理；"
            "该换算不是厂家公布的任意方向力额定值，默认关闭"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="运行当前 XML 的 Jacobian 与容量回归检查后退出",
    )
    args = parser.parse_args(argv)

    finite_scalars = {
        "--friction": args.friction,
        "--safety-factor": args.safety_factor,
        "--dxl-general-fraction": args.dxl_general_fraction,
        "--rotary-reserve-nm": args.rotary_reserve_nm,
        "--extra-slider-load-kg": args.extra_slider_load_kg,
        "--lift-height-mm": args.lift_height_mm,
        "--lift-accel-g": args.lift_accel_g,
        "--linear-scan-step-mm": args.linear_scan_step_mm,
    }
    if args.object_size_mm is not None:
        finite_scalars["--object-size-mm"] = args.object_size_mm
    if args.contact_patch_diameter_mm is not None:
        finite_scalars["--contact-patch-diameter-mm"] = (
            args.contact_patch_diameter_mm
        )
    if args.contact_compression_mm is not None:
        finite_scalars["--contact-compression-mm"] = (
            args.contact_compression_mm
        )
    for option, value in finite_scalars.items():
        if not math.isfinite(float(value)):
            parser.error(f"{option} 必须是有限数值")
    if args.center_m is not None and not all(
        math.isfinite(float(value)) for value in args.center_m
    ):
        parser.error("--center-m 的三个坐标都必须是有限数值")
    if not all(
        math.isfinite(float(value)) for value in args.planar_offset_mm
    ):
        parser.error("--planar-offset-mm 的两个分量都必须是有限数值")

    if args.friction <= 0.0:
        parser.error("--friction 必须大于 0")
    if args.safety_factor < 1.0:
        parser.error("--safety-factor 必须不小于 1")
    if not 0.0 < args.dxl_general_fraction <= 1.0:
        parser.error("--dxl-general-fraction 必须在 (0,1] 内")
    if args.rotary_reserve_nm < 0.0:
        parser.error("--rotary-reserve-nm 不能为负")
    if args.extra_slider_load_kg < 0.0:
        parser.error("--extra-slider-load-kg 不能为负")
    if args.object_size_mm is not None and args.object_size_mm <= 0.0:
        parser.error("--object-size-mm 必须大于 0")
    if (
        args.contact_patch_diameter_mm is not None
        and args.contact_patch_diameter_mm <= 0.0
    ):
        parser.error("--contact-patch-diameter-mm 必须大于 0")
    if (
        args.contact_compression_mm is not None
        and args.contact_compression_mm < 0.0
    ):
        parser.error("--contact-compression-mm 不能为负")
    if args.lift_height_mm < 0.0:
        parser.error("--lift-height-mm 不能为负")
    if args.path_samples < 1:
        parser.error("--path-samples 必须不小于 1")
    if args.lift_height_mm > 0.0 and args.path_samples < 2:
        parser.error("非零抬升路径的 --path-samples 必须不小于 2")
    if args.transfer_samples < 1:
        parser.error("--transfer-samples 必须不小于 1")
    if (
        np.linalg.norm(np.asarray(args.planar_offset_mm, dtype=float)) > 0.0
        and args.transfer_samples < 2
    ):
        parser.error("非零平面搬运路径的 --transfer-samples 必须不小于 2")
    if not 0.0 <= args.lift_accel_g <= IAI.rated_max_acceleration_g:
        parser.error(
            f"--lift-accel-g 必须在 0...{IAI.rated_max_acceleration_g:g} 内"
        )
    if args.linear_scan_step_mm <= 0.0:
        parser.error("--linear-scan-step-mm 必须大于 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    xml_path = args.xml.expanduser()
    if not xml_path.is_absolute():
        xml_path = (MODULE_DIR / xml_path).resolve()
    if args.self_test:
        run_self_test(xml_path)
        return 0

    ik = PayloadGripperIK.from_xml(xml_path)
    object_half_size_m = (
        ik.model_info.cube_half_size
        if args.object_size_mm is None
        else args.object_size_mm / 2000.0
    )
    args.object_size_mm = 2000.0 * object_half_size_m
    start_center = (
        ik.model_info.cube_pos.copy()
        if args.center_m is None
        else np.asarray(args.center_m, dtype=float)
    )
    object_rotation = quat_to_rot(ik.model_info.cube_quat)
    try:
        if args.contact_compression_mm is not None:
            contact_patch = contact_patch_from_compression(
                ik.model_info.hard_tip_radius,
                args.contact_compression_mm / 1000.0,
            )
        else:
            patch_diameter_mm = (
                20.0
                if args.contact_patch_diameter_mm is None
                else args.contact_patch_diameter_mm
            )
            contact_patch = contact_patch_from_diameter(
                ik.model_info.hard_tip_radius,
                patch_diameter_mm / 1000.0,
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.contact_patch_diameter_mm = (
        contact_patch.patch_diameter_m * 1000.0
    )
    args.contact_compression_mm = (
        contact_patch.equivalent_compression_m * 1000.0
    )
    if contact_patch.patch_diameter_m > 2.0 * object_half_size_m + 1.0e-12:
        raise SystemExit(
            "圆形接触斑直径不能大于物体侧面边长；"
            f"当前为 {contact_patch.patch_diameter_m * 1000.0:.3f} mm > "
            f"{2.0 * object_half_size_m * 1000.0:.3f} mm"
        )
    contact_distance_m = (
        object_half_size_m + contact_patch.center_to_contact_plane_m
    )
    if contact_distance_m <= object_half_size_m:
        raise SystemExit(
            "接触平面必须位于 hard-tip 球心外侧；"
            f"当前 tip radius={ik.model_info.hard_tip_radius * 1000.0:.3f} mm"
        )

    centers = build_pick_transfer_place_centers(
        start_center,
        args.lift_height_mm / 1000.0,
        np.asarray(args.planar_offset_mm, dtype=float) / 1000.0,
        args.path_samples,
        args.transfer_samples,
    )
    linear_low, linear_high = ik.fingers[FINGERS[0]].active_ranges["linear"]
    if args.linear_position_mm == "auto":
        references = _inclusive_grid(
            linear_low,
            linear_high,
            args.linear_scan_step_mm / 1000.0,
        )
        match_tolerance = max(
            5.0e-7,
            0.25 * args.linear_scan_step_mm / 1000.0,
        )
    else:
        try:
            fixed_reference = float(args.linear_position_mm) / 1000.0
        except ValueError as exc:
            raise SystemExit(
                "--linear-position-mm 必须是 auto 或数字"
            ) from exc
        if not linear_low - 1.0e-12 <= fixed_reference <= linear_high + 1.0e-12:
            raise SystemExit(
                f"linear 位置超出当前 XML 范围 "
                f"[{linear_low * 1000.0:g}, {linear_high * 1000.0:g}] mm"
            )
        references = np.array([fixed_reference], dtype=float)
        match_tolerance = 5.0e-7

    candidate_paths = build_candidate_paths(
        ik,
        centers,
        object_rotation,
        contact_distance_m,
        object_half_size_m,
        references,
        match_tolerance,
    )
    if not candidate_paths:
        raise SystemExit(
            "给定物体、完整取放路径与 fixed-linear 候选没有完整可达解；"
            "请减小抬升高度/平面偏移、改变物体中心/尺寸或使用 auto linear。"
        )

    motor_keys = (
        tuple(MOTOR_SPECS)
        if args.motor == "both"
        else (args.motor,)
    )
    effective_gravity = G * (1.0 + args.lift_accel_g)
    all_results: Dict[str, Dict[str, PathResult]] = {}
    for motor_key in motor_keys:
        motor = MOTOR_SPECS[motor_key]
        voltage = resolve_voltage(motor, args.voltage)
        all_results[motor_key] = {
            "general": optimize_path_for_profile(
                candidate_paths,
                motor,
                voltage,
                "一般使用估算",
                args.dxl_general_fraction,
                args.rotary_reserve_nm,
                args.extra_slider_load_kg,
                object_half_size_m,
                args.friction,
                effective_gravity,
                args.guide_limit,
                args.use_iai_horizontal_payload_proxy,
            ),
            "peak": optimize_path_for_profile(
                candidate_paths,
                motor,
                voltage,
                "堵转瞬时数学上限",
                1.0,
                args.rotary_reserve_nm,
                args.extra_slider_load_kg,
                object_half_size_m,
                args.friction,
                effective_gravity,
                args.guide_limit,
                args.use_iai_horizontal_payload_proxy,
            ),
        }

    print_report(
        args,
        ik,
        centers,
        contact_distance_m,
        contact_patch,
        candidate_paths,
        all_results,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
