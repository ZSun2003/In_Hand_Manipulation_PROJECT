#!/usr/bin/env python3
"""Calculate the quasi-static payload capacity of the current four-finger gripper.

This is an analytic rigid-body calculation, not a MuJoCo simulation.  The
program reads the current linkage geometry and joint limits through
``T3_kinematics.py` and then:

1. solves the closed-chain pose of the four fingers;
2. derives the exact reduced 2 x 2 fingertip Jacobian for the active ``back``
   and ``front`` joints;
3. uses virtual work, ``tau = J.T @ force``;
4. maximises upward contact force subject to both motor torque limits and the
   Coulomb constraint ``tangential_force <= mu * normal_force``.

The four ``armbase`` motor torque limits are intentionally omitted, as requested.
For a centred object all four fingers are symmetric, their horizontal normal
forces cancel, and their upward tangential forces add.  For an off-centre
pick--lift--XY-transport--place route, the program instead solves a four-contact
3-D rigid-body equilibrium linear program at every sampled pose and reports the
weakest point on the complete unsupported route.
The gripper's own weight is also intentionally excluded, as requested.

Each rubber hard tip is modelled as a sphere flattened into a centred circular
contact patch.  The default 20 mm patch diameter is supplied by the user.  A
spherical-cap estimate locates the contact plane, and the resultant normal and
friction forces act at the patch centre.  Patch area is reported as average
pressure; under a Coulomb model, area does not by itself multiply friction.

Official motor data used here:

* https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/
* https://emanual.robotis.com/docs/en/dxl/x/xc430-w240/
* https://www.robotis.us/dynamixel-xl430-w250-t/
* https://www.robotis.us/dynamixel-xc430-w240-t/

ROBOTIS describes stall torque as momentary.  The program therefore reports
both the instantaneous stall-based theoretical limit and ROBOTIS's estimated
continuous torque (20 percent of stall torque).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


MODULE_DIR = Path(__file__).resolve().parent
LOCAL_ENV_DIR = (Path(os.environ["IRP_PYTHON_ENV"]) if os.environ.get("IRP_PYTHON_ENV") else next((parent / "RL_env" for parent in Path(__file__).resolve().parents if (parent / "RL_env" / "build.sh").is_file()), Path(sys.prefix)))

# A direct launch uses this project's environment so the program works even
# when the user's system Python does not contain NumPy.  Importing the module
# from a test or another program does not replace that caller's interpreter.
if __name__ == "__main__" and Path(sys.prefix).resolve() != LOCAL_ENV_DIR.resolve():
    local_python = LOCAL_ENV_DIR / "bin" / "python"
    if local_python.exists():
        os.execv(
            str(local_python),
            [str(local_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

import numpy as np

from T3_kinematics import (  # noqa: E402  (local-environment bootstrap above)
    FINGERS,
    TIP_OFFSETS_UNIT,
    FingerGeometry,
    FingerSolution,
    GripperIK,
    IKSolution,
    rot2,
    rot_axis,
)


STANDARD_GRAVITY_M_S2 = 9.80665
PLANAR_SKEW = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)


@dataclass(frozen=True)
class MotorSpec:
    """Official output-shaft data at the published voltage test points."""

    key: str
    name: str
    recommended_voltage_v: float
    stall_torque_nm: Mapping[float, float]
    stall_current_a: Mapping[float, float]

    def values_at(self, requested_voltage_v: Optional[float]) -> Tuple[float, float, float]:
        voltage = (
            self.recommended_voltage_v
            if requested_voltage_v is None
            else float(requested_voltage_v)
        )
        for published_voltage, torque in self.stall_torque_nm.items():
            if math.isclose(voltage, published_voltage, rel_tol=0.0, abs_tol=1e-9):
                return (
                    float(published_voltage),
                    float(torque),
                    float(self.stall_current_a[published_voltage]),
                )
        available = ", ".join(f"{value:g}" for value in self.stall_torque_nm)
        raise ValueError(
            f"{self.name} 没有 {voltage:g} V 的官方扭矩数据；"
            f"可选电压为 {available} V（程序不做电压外插）。"
        )


MOTOR_SPECS: Mapping[str, MotorSpec] = {
    "xl": MotorSpec(
        key="xl",
        name="XL430-W250-T",
        recommended_voltage_v=11.1,
        stall_torque_nm={9.0: 1.0, 11.1: 1.4, 12.0: 1.5},
        stall_current_a={9.0: 1.0, 11.1: 1.3, 12.0: 1.4},
    ),
    "xc": MotorSpec(
        key="xc",
        name="XC430-W240-T",
        recommended_voltage_v=12.0,
        stall_torque_nm={9.0: 1.4, 11.1: 1.7, 12.0: 1.9},
        stall_current_a={9.0: 1.1, 11.1: 1.3, 12.0: 1.4},
    ),
}


@dataclass(frozen=True)
class JacobianResult:
    """Reduced planar fingertip Jacobian and its singularity diagnostics."""

    matrix_m_per_rad: np.ndarray
    passive_closure_condition: float
    active_jacobian_condition: float


@dataclass(frozen=True)
class ContactPatchGeometry:
    """Centred circular patch made by flattening a spherical rubber tip."""

    sphere_radius_m: float
    diameter_m: float
    radius_m: float
    area_m2: float
    center_to_contact_plane_m: float
    estimated_indentation_m: float


@dataclass(frozen=True)
class PoseMechanics:
    """Kinematic/static information at one centred object height."""

    object_center_z_m: float
    solution: FingerSolution
    jacobian: JacobianResult


@dataclass(frozen=True)
class ContactCapacity:
    """Best symmetric per-finger contact force for one torque limit."""

    normal_force_n: float
    upward_force_n: float
    back_torque_nm: float
    front_torque_nm: float
    payload_mass_kg: float
    limiting_motor: str
    friction_is_active: bool
    normal_limit_is_active: bool


@dataclass(frozen=True)
class CapacityRow:
    """One motor/torque-mode result, including a route bottleneck."""

    motor: MotorSpec
    voltage_v: float
    stall_current_a: float
    torque_mode: str
    torque_mode_label: str
    published_torque_nm: float
    effective_torque_nm: float
    bottleneck_pose: PoseMechanics
    contact: ContactCapacity


@dataclass(frozen=True)
class SpatialPoseMechanics:
    """IK and world contact Jacobians at one point on a transport route."""

    object_center_world_m: np.ndarray
    segment_label: str
    solutions: Mapping[str, FingerSolution]
    contact_jacobians_world_m_per_rad: Mapping[str, np.ndarray]
    worst_condition: float


@dataclass(frozen=True)
class SpatialContactCapacity:
    """Maximum payload from a four-contact 3-D quasi-static equilibrium LP."""

    payload_mass_kg: float
    contact_forces_world_n: Mapping[str, np.ndarray]
    normal_forces_n: Mapping[str, float]
    tangential_forces_n: Mapping[str, float]
    motor_torques_nm: Mapping[str, Tuple[float, float]]
    limiting_actuator: str
    maximum_friction_utilisation: float
    maximum_normal_force_n: float


@dataclass(frozen=True)
class SpatialCapacityRow:
    """One motor/torque-mode result at the weakest point of a 3-D route."""

    motor: MotorSpec
    voltage_v: float
    stall_current_a: float
    torque_mode: str
    torque_mode_label: str
    published_torque_nm: float
    effective_torque_nm: float
    bottleneck_pose: SpatialPoseMechanics
    contact: SpatialContactCapacity


@dataclass(frozen=True)
class HalfPlane:
    """A two-variable inequality ``a @ [normal, upward] <= b``."""

    a: np.ndarray
    b: float
    label: str


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Keep examples on separate lines while still displaying defaults."""


def _rotated_derivative(angle_rad: float, vector: Sequence[float]) -> np.ndarray:
    """Derivative of ``R(angle(q)) vector`` when ``d angle / d q = -1``."""

    return -PLANAR_SKEW @ rot2(angle_rad) @ np.asarray(vector, dtype=float)


def estimate_circular_contact_patch(
    sphere_radius_m: float,
    contact_diameter_m: float,
) -> ContactPatchGeometry:
    """Estimate a flat circular patch as a spherical cap.

    If the circular patch radius is ``a`` on a sphere of radius ``R``, the
    centre-to-contact-plane distance is ``sqrt(R**2 - a**2)``.  This is a
    geometric approximation for a deformed rubber tip, not a Hertz-material
    solution.
    """

    sphere_radius = float(sphere_radius_m)
    diameter = float(contact_diameter_m)
    if sphere_radius <= 0.0:
        raise ValueError("hard tip 球半径必须大于 0。")
    if diameter <= 0.0:
        raise ValueError("接触圆直径必须大于 0。")
    if diameter >= 2.0 * sphere_radius:
        raise ValueError(
            f"接触圆直径 {1000.0 * diameter:.3f} mm 必须小于 "
            f"hard tip 球直径 {2000.0 * sphere_radius:.3f} mm。"
        )

    patch_radius = 0.5 * diameter
    center_to_plane = math.sqrt(
        max(0.0, sphere_radius * sphere_radius - patch_radius * patch_radius)
    )
    return ContactPatchGeometry(
        sphere_radius_m=sphere_radius,
        diameter_m=diameter,
        radius_m=patch_radius,
        area_m2=math.pi * patch_radius * patch_radius,
        center_to_contact_plane_m=center_to_plane,
        estimated_indentation_m=sphere_radius - center_to_plane,
    )


def planar_active_jacobian(
    geometry: FingerGeometry,
    solution: FingerSolution,
    *,
    contact_offset_planar_m: Sequence[float] = (0.0, 0.0),
    singular_tolerance: float = 1e-12,
) -> JacobianResult:
    """Return ``d [contact_x, contact_z] / d [q_back, q_front]``.

    The passive ``short`` and ``elbow`` rates are eliminated by differentiating
    the two closed-loop position constraints.  No finite differencing and no
    physics engine are used.  ``contact_offset_planar_m`` is the instantaneous
    vector from the spherical tip centre to the material contact point,
    expressed in the finger's fixed radial/vertical plane.  For contact with a
    vertical object face it is ``[-tip_radius, 0]``; this correctly includes the
    moment made by tangential friction at the sphere surface.
    """

    q = solution.joint_map
    theta_back = -(geometry.alpha + q["back"])
    theta_short = -(geometry.alpha + q["back"] + q["short"])
    gamma = -(
        geometry.alpha
        + q["back"]
        + q["short"]
        + geometry.eta
        + q["elbow"]
    )
    theta_front = -(geometry.alpha + q["front"])

    link_1 = np.array([geometry.l1, 0.0], dtype=float)
    link_2 = np.array([geometry.l2, 0.0], dtype=float)
    link_3 = np.array([geometry.l3, 0.0], dtype=float)

    d_back = _rotated_derivative(theta_back, link_1)
    d_short_link = _rotated_derivative(theta_short, link_2)
    d_closure_long = _rotated_derivative(gamma, geometry.c0)
    contact_offset = np.asarray(contact_offset_planar_m, dtype=float).reshape(2)
    # Convert the current plane-frame surface offset to a vector fixed in the
    # long link.  Its instantaneous derivative then gives the correct rigid-body
    # contact-point velocity and therefore the correct friction-force moment.
    contact_point_long = geometry.t0 + rot2(-gamma) @ contact_offset
    d_contact_long = _rotated_derivative(gamma, contact_point_long)
    d_front = _rotated_derivative(theta_front, link_3)

    # C_long derivatives when qb, qs and qe are treated as independent.
    closure_qb = d_back + d_short_link + d_closure_long
    closure_qs = d_short_link + d_closure_long
    closure_qe = d_closure_long
    passive_matrix = np.column_stack((closure_qs, closure_qe))
    passive_condition = float(np.linalg.cond(passive_matrix))
    if (
        not np.isfinite(passive_condition)
        or abs(float(np.linalg.det(passive_matrix))) <= singular_tolerance
    ):
        raise ValueError("闭链被动关节处于奇异位形，无法得到有限的解析雅可比。")

    # From d(C_long - C_front) = 0:
    #   [C_qs C_qe] [dqs/dqb, dqe/dqb] = -C_qb
    #   [C_qs C_qe] [dqs/dqf, dqe/dqf] =  Cfront_qf
    passive_per_back = np.linalg.solve(passive_matrix, -closure_qb)
    passive_per_front = np.linalg.solve(passive_matrix, d_front)

    tip_qb = d_back + d_short_link + d_contact_long
    tip_qs = d_short_link + d_contact_long
    tip_qe = d_contact_long
    jacobian = np.column_stack(
        (
            tip_qb
            + tip_qs * passive_per_back[0]
            + tip_qe * passive_per_back[1],
            tip_qs * passive_per_front[0]
            + tip_qe * passive_per_front[1],
        )
    )
    active_condition = float(np.linalg.cond(jacobian))
    if (
        not np.isfinite(active_condition)
        or abs(float(np.linalg.det(jacobian))) <= singular_tolerance
    ):
        raise ValueError("主动关节雅可比处于奇异位形，理想刚体载荷没有有限解。")

    return JacobianResult(
        matrix_m_per_rad=jacobian,
        passive_closure_condition=passive_condition,
        active_jacobian_condition=active_condition,
    )


def _is_feasible(point: np.ndarray, constraints: Iterable[HalfPlane]) -> bool:
    for constraint in constraints:
        scale = max(1.0, abs(constraint.b), float(np.linalg.norm(constraint.a)))
        if float(constraint.a @ point) > constraint.b + 1e-9 * scale:
            return False
    return True


def maximise_symmetric_contact_force(
    jacobian_m_per_rad: np.ndarray,
    *,
    torque_limit_nm: float,
    friction_coefficient: float,
    gravity_m_s2: float = STANDARD_GRAVITY_M_S2,
    finger_count: int = 4,
    max_normal_force_n: Optional[float] = None,
) -> ContactCapacity:
    """Maximise lift force for one symmetric finger using a 2-D linear program.

    The force exerted *by the finger on the object*, expressed in the finger's
    local radial/vertical plane, is ``[-normal, upward]``.  Virtual work gives
    ``[tau_back, tau_front] = J.T @ [-normal, upward]``.
    """

    jacobian = np.asarray(jacobian_m_per_rad, dtype=float).reshape(2, 2)
    if torque_limit_nm <= 0.0:
        raise ValueError("电机扭矩上限必须大于 0。")
    if friction_coefficient <= 0.0:
        raise ValueError("摩擦系数必须大于 0。")
    if gravity_m_s2 <= 0.0:
        raise ValueError("重力加速度必须大于 0。")
    if finger_count <= 0:
        raise ValueError("手指数必须大于 0。")
    if max_normal_force_n is not None and max_normal_force_n <= 0.0:
        raise ValueError("法向力上限必须大于 0。")

    constraints = [
        HalfPlane(np.array([-1.0, 0.0]), 0.0, "normal >= 0"),
        HalfPlane(np.array([0.0, -1.0]), 0.0, "upward >= 0"),
        HalfPlane(
            np.array([-friction_coefficient, 1.0]),
            0.0,
            "Coulomb friction",
        ),
    ]
    if max_normal_force_n is not None:
        constraints.append(
            HalfPlane(
                np.array([1.0, 0.0]),
                float(max_normal_force_n),
                "normal-force limit",
            )
        )

    # tau_i = -J[x, i] * normal + J[z, i] * upward.
    for index, motor_name in enumerate(("back", "front")):
        torque_coefficients = np.array(
            [-jacobian[0, index], jacobian[1, index]],
            dtype=float,
        )
        constraints.extend(
            (
                HalfPlane(
                    torque_coefficients,
                    float(torque_limit_nm),
                    f"{motor_name} +torque",
                ),
                HalfPlane(
                    -torque_coefficients,
                    float(torque_limit_nm),
                    f"{motor_name} -torque",
                ),
            )
        )

    candidates = []
    for first, second in combinations(constraints, 2):
        boundary_matrix = np.vstack((first.a, second.a))
        determinant = float(np.linalg.det(boundary_matrix))
        if abs(determinant) <= 1e-14:
            continue
        point = np.linalg.solve(
            boundary_matrix,
            np.array([first.b, second.b], dtype=float),
        )
        if _is_feasible(point, constraints):
            candidates.append(point)

    if not candidates:
        raise ValueError("当前姿态、摩擦和法向力约束没有可行的夹持力解。")

    optimum = max(candidates, key=lambda point: (float(point[1]), -float(point[0])))
    normal_force = max(0.0, float(optimum[0]))
    upward_force = max(0.0, float(optimum[1]))
    contact_force = np.array([-normal_force, upward_force], dtype=float)
    torques = jacobian.T @ contact_force
    utilisation = np.abs(torques) / float(torque_limit_nm)
    limiting_index = int(np.argmax(utilisation))
    friction_slack = friction_coefficient * normal_force - upward_force
    normal_slack = (
        math.inf
        if max_normal_force_n is None
        else float(max_normal_force_n) - normal_force
    )

    return ContactCapacity(
        normal_force_n=normal_force,
        upward_force_n=upward_force,
        back_torque_nm=float(torques[0]),
        front_torque_nm=float(torques[1]),
        payload_mass_kg=float(finger_count * upward_force / gravity_m_s2),
        limiting_motor=("back" if limiting_index == 0 else "front"),
        friction_is_active=abs(friction_slack) <= 1e-7 * max(1.0, upward_force),
        normal_limit_is_active=(
            max_normal_force_n is not None
            and abs(normal_slack) <= 1e-7 * max(1.0, float(max_normal_force_n))
        ),
    )


def solve_centered_pose(
    ik: GripperIK,
    *,
    object_center_z_m: float,
    contact_distance_m: float,
    contact_plane_offset_m: float,
    previous_solutions: Optional[Mapping[str, FingerSolution]] = None,
    condition_limit: float = 1e4,
) -> Tuple[PoseMechanics, Mapping[str, FingerSolution]]:
    """Solve one centred four-finger grasp and verify four-fold symmetry."""

    result: IKSolution = ik.solve_cube_pose(
        (0.0, 0.0, float(object_center_z_m)),
        distance=float(contact_distance_m),
        previous_solutions=previous_solutions,
        strict=False,
    )
    if not result.success:
        failed = ", ".join(result.failed_fingers)
        raise ValueError(
            f"物体中心 z={object_center_z_m:.6f} m 时指尖不可达（失败手指: {failed}）。"
        )

    reference = result.finger_solutions["R"]
    reference_active = np.array(
        [reference.q_armbase, reference.q_back, reference.q_front],
        dtype=float,
    )
    if abs(reference.q_armbase) > 1e-8:
        raise ValueError(
            "本程序的对称载荷模型要求物体位于中心轴，但求得非零 armbase 角。"
        )
    for finger in FINGERS:
        solution = result.finger_solutions[finger]
        active = np.array(
            [solution.q_armbase, solution.q_back, solution.q_front],
            dtype=float,
        )
        if not np.allclose(active, reference_active, atol=2e-8, rtol=0.0):
            raise ValueError("四指没有落在对称的闭链分支，无法采用等载荷静力模型。")

    jacobian = planar_active_jacobian(
        ik.fingers["R"],
        reference,
        contact_offset_planar_m=(-float(contact_plane_offset_m), 0.0),
    )
    worst_condition = max(
        jacobian.passive_closure_condition,
        jacobian.active_jacobian_condition,
    )
    if condition_limit <= 1.0:
        raise ValueError("雅可比条件数上限必须大于 1。")
    if worst_condition > condition_limit:
        raise ValueError(
            f"z={object_center_z_m:.6f} m 的条件数为 {worst_condition:.3g}，"
            f"超过 --condition-limit={condition_limit:g}；该位形过于接近奇异点。"
        )

    return (
        PoseMechanics(
            object_center_z_m=float(object_center_z_m),
            solution=reference,
            jacobian=jacobian,
        ),
        result.finger_solutions,
    )


def build_vertical_path(
    ik: GripperIK,
    *,
    start_z_m: float,
    end_z_m: Optional[float],
    samples: int,
    contact_distance_m: float,
    contact_plane_offset_m: float,
    condition_limit: float,
) -> Tuple[PoseMechanics, ...]:
    """Build a branch-continuous centred vertical lift path."""

    if end_z_m is None or math.isclose(start_z_m, end_z_m, abs_tol=1e-12):
        heights = np.array([float(start_z_m)], dtype=float)
    else:
        if samples < 2:
            raise ValueError("分析一段抬升路径时 --samples 至少为 2。")
        heights = np.linspace(float(start_z_m), float(end_z_m), int(samples))

    poses = []
    previous: Optional[Mapping[str, FingerSolution]] = None
    for height in heights:
        pose, previous = solve_centered_pose(
            ik,
            object_center_z_m=float(height),
            contact_distance_m=contact_distance_m,
            contact_plane_offset_m=contact_plane_offset_m,
            previous_solutions=previous,
            condition_limit=condition_limit,
        )
        poses.append(pose)
    return tuple(poses)


def spatial_contact_jacobian(
    geometry: FingerGeometry,
    solution: FingerSolution,
    *,
    contact_plane_offset_m: float,
) -> JacobianResult:
    """Return the 3 x 2 world Jacobian of a face-centred contact point.

    ``planar_active_jacobian`` eliminates the two passive closed-chain joints.
    The armbase angle is retained in the pose transformation but its torque is
    intentionally not constrained.  The contact resultant is displaced from
    the rubber sphere centre toward the object by ``contact_plane_offset_m``.
    """

    planar = planar_active_jacobian(
        geometry,
        solution,
        contact_offset_planar_m=(-float(contact_plane_offset_m), 0.0),
    )
    arm_rotation = rot_axis(geometry.armbase_axis, solution.q_armbase)
    local_to_world = geometry.rotation_world @ arm_rotation
    world = np.empty((3, 2), dtype=float)
    for column in range(2):
        planar_velocity = np.array(
            [
                planar.matrix_m_per_rad[0, column],
                0.0,
                planar.matrix_m_per_rad[1, column],
            ],
            dtype=float,
        )
        world[:, column] = local_to_world @ planar_velocity
    return JacobianResult(
        matrix_m_per_rad=world,
        passive_closure_condition=planar.passive_closure_condition,
        active_jacobian_condition=planar.active_jacobian_condition,
    )


def _sample_polyline(
    labelled_waypoints: Sequence[Tuple[str, np.ndarray]],
    *,
    spacing_m: float,
) -> Tuple[Tuple[str, np.ndarray], ...]:
    """Sample every segment of a labelled polyline, without duplicate joins."""

    if spacing_m <= 0.0:
        raise ValueError("路径采样间距必须大于 0。")
    if not labelled_waypoints:
        raise ValueError("搬运路径至少需要一个路径点。")
    if len(labelled_waypoints) == 1:
        label, point = labelled_waypoints[0]
        return ((label, np.asarray(point, dtype=float).reshape(3)),)

    samples = []
    for segment_index, ((_, start), (label, end)) in enumerate(
        zip(labelled_waypoints[:-1], labelled_waypoints[1:])
    ):
        start_point = np.asarray(start, dtype=float).reshape(3)
        end_point = np.asarray(end, dtype=float).reshape(3)
        distance = float(np.linalg.norm(end_point - start_point))
        count = max(2, int(math.ceil(distance / spacing_m)) + 1)
        for sample_index, fraction in enumerate(np.linspace(0.0, 1.0, count)):
            if segment_index > 0 and sample_index == 0:
                continue
            point = (1.0 - float(fraction)) * start_point + float(fraction) * end_point
            samples.append((label, point))
    return tuple(samples)


def build_pick_move_place_path(
    ik: GripperIK,
    *,
    start_world_m: Sequence[float],
    target_world_m: Sequence[float],
    transit_z_m: float,
    spacing_m: float,
    contact_distance_m: float,
    contact_plane_offset_m: float,
    condition_limit: float,
) -> Tuple[SpatialPoseMechanics, ...]:
    """Build and solve the controller's lift--XY transport--lower route."""

    if condition_limit <= 1.0:
        raise ValueError("雅可比条件数上限必须大于 1。")
    start = np.asarray(start_world_m, dtype=float).reshape(3)
    target = np.asarray(target_world_m, dtype=float).reshape(3)
    if transit_z_m < max(float(start[2]), float(target[2])) - 1e-12:
        raise ValueError("运输平面 z 不能低于抓取点或放置点。")

    raw_waypoints = (
        ("lift", start),
        ("lift", np.array([start[0], start[1], transit_z_m], dtype=float)),
        ("plane", np.array([target[0], target[1], transit_z_m], dtype=float)),
        ("lower", target),
    )
    compact = [raw_waypoints[0]]
    for label, point in raw_waypoints[1:]:
        if float(np.linalg.norm(point - compact[-1][1])) > 1e-12:
            compact.append((label, point))
    labelled_samples = _sample_polyline(compact, spacing_m=spacing_m)

    poses = []
    previous: Optional[Mapping[str, FingerSolution]] = None
    for label, center in labelled_samples:
        result = ik.solve_cube_pose(
            center,
            distance=float(contact_distance_m),
            previous_solutions=previous,
            strict=False,
        )
        if not result.success:
            failed = ", ".join(result.failed_fingers)
            raise ValueError(
                "搬运路径不可达："
                f"segment={label}, xyz={center.tolist()}，失败手指: {failed}。"
            )

        jacobians: Dict[str, np.ndarray] = {}
        worst_condition = 0.0
        for finger in FINGERS:
            jacobian = spatial_contact_jacobian(
                ik.fingers[finger],
                result.finger_solutions[finger],
                contact_plane_offset_m=contact_plane_offset_m,
            )
            worst_condition = max(
                worst_condition,
                jacobian.passive_closure_condition,
                jacobian.active_jacobian_condition,
            )
            jacobians[finger] = jacobian.matrix_m_per_rad
        if worst_condition > condition_limit:
            raise ValueError(
                f"搬运路径 xyz={center.tolist()} 的最大条件数为 "
                f"{worst_condition:.3g}，超过 --condition-limit={condition_limit:g}。"
            )

        poses.append(
            SpatialPoseMechanics(
                object_center_world_m=center.copy(),
                segment_label=label,
                solutions=dict(result.finger_solutions),
                contact_jacobians_world_m_per_rad=jacobians,
                worst_condition=float(worst_condition),
            )
        )
        previous = result.finger_solutions
    return tuple(poses)


def _cross_product_matrix(vector: Sequence[float]) -> np.ndarray:
    """Return a matrix C such that ``C @ force == vector x force``."""

    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def maximise_spatial_payload(
    pose: SpatialPoseMechanics,
    *,
    torque_limit_nm: float,
    friction_coefficient: float,
    object_half_width_m: float,
    gravity_m_s2: float = STANDARD_GRAVITY_M_S2,
    max_normal_force_n: Optional[float] = None,
    friction_facets: int = 16,
    linear_acceleration_world_m_s2: Sequence[float] = (0.0, 0.0, 0.0),
) -> SpatialContactCapacity:
    """Maximise payload with four 3-D contact forces and rigid-body balance.

    The Coulomb circle is conservatively approximated by a regular polygon
    inscribed in it.  The polygon includes the world-up direction as a vertex,
    so a centred zero-acceleration lift exactly reproduces the earlier 2-D
    vertical-friction result.
    """

    if torque_limit_nm <= 0.0:
        raise ValueError("电机扭矩上限必须大于 0。")
    if friction_coefficient <= 0.0:
        raise ValueError("摩擦系数必须大于 0。")
    if object_half_width_m <= 0.0:
        raise ValueError("物体半宽必须大于 0。")
    if gravity_m_s2 <= 0.0:
        raise ValueError("重力加速度必须大于 0。")
    if friction_facets < 4 or friction_facets % 4 != 0:
        raise ValueError("摩擦锥面数必须是大于等于 4 的 4 的倍数。")
    if max_normal_force_n is not None and max_normal_force_n <= 0.0:
        raise ValueError("法向力上限必须大于 0。")

    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - local RL_env includes SciPy
        raise ValueError("完整 3-D 路径计算需要 scipy.optimize.linprog。") from exc

    force_slices = {
        finger: slice(3 * index, 3 * index + 3)
        for index, finger in enumerate(FINGERS)
    }
    mass_index = 3 * len(FINGERS)
    variable_count = mass_index + 1
    inequalities = []
    bounds = []
    polygon_apothem_factor = math.cos(math.pi / friction_facets)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)

    for finger in FINGERS:
        force_slice = force_slices[finger]
        outward = np.asarray(TIP_OFFSETS_UNIT[finger], dtype=float)
        inward_normal = -outward
        horizontal_tangent = np.cross(inward_normal, world_up)
        horizontal_tangent /= float(np.linalg.norm(horizontal_tangent))

        # n dot f >= 0.
        row = np.zeros(variable_count, dtype=float)
        row[force_slice] = -inward_normal
        inequalities.append(row)
        bounds.append(0.0)

        # An inscribed regular polygon in the two-dimensional friction plane.
        for facet in range(friction_facets):
            angle = (2 * facet + 1) * math.pi / friction_facets
            tangent_direction = (
                math.cos(angle) * world_up
                + math.sin(angle) * horizontal_tangent
            )
            row = np.zeros(variable_count, dtype=float)
            row[force_slice] = (
                tangent_direction
                - friction_coefficient
                * polygon_apothem_factor
                * inward_normal
            )
            inequalities.append(row)
            bounds.append(0.0)

        if max_normal_force_n is not None:
            row = np.zeros(variable_count, dtype=float)
            row[force_slice] = inward_normal
            inequalities.append(row)
            bounds.append(float(max_normal_force_n))

        jacobian = np.asarray(
            pose.contact_jacobians_world_m_per_rad[finger],
            dtype=float,
        ).reshape(3, 2)
        for motor_index in range(2):
            coefficient = jacobian[:, motor_index]
            for sign in (-1.0, 1.0):
                row = np.zeros(variable_count, dtype=float)
                row[force_slice] = sign * coefficient
                inequalities.append(row)
                bounds.append(float(torque_limit_nm))

    equalities = np.zeros((6, variable_count), dtype=float)
    equality_rhs = np.zeros(6, dtype=float)
    for finger in FINGERS:
        force_slice = force_slices[finger]
        equalities[:3, force_slice] = np.eye(3)
        contact_arm = (
            np.asarray(TIP_OFFSETS_UNIT[finger], dtype=float)
            * float(object_half_width_m)
        )
        equalities[3:, force_slice] = _cross_product_matrix(contact_arm)

    acceleration = np.asarray(
        linear_acceleration_world_m_s2,
        dtype=float,
    ).reshape(3)
    gravity_vector = np.array([0.0, 0.0, -float(gravity_m_s2)])
    equalities[:3, mass_index] = -(acceleration - gravity_vector)

    objective = np.zeros(variable_count, dtype=float)
    objective[mass_index] = -1.0
    result = linprog(
        objective,
        A_ub=np.asarray(inequalities, dtype=float),
        b_ub=np.asarray(bounds, dtype=float),
        A_eq=equalities,
        b_eq=equality_rhs,
        bounds=[(None, None)] * mass_index + [(0.0, None)],
        method="highs",
    )
    if result.status == 3:
        raise ValueError(
            "载荷线性规划无界；请设置 --max-normal-force，或不要完全豁免 "
            "armbase 绕轴扭矩。"
        )
    if not result.success:
        raise ValueError(f"载荷线性规划失败: {result.message}")

    forces: Dict[str, np.ndarray] = {}
    normal_forces: Dict[str, float] = {}
    tangential_forces: Dict[str, float] = {}
    motor_torques: Dict[str, Tuple[float, float]] = {}
    friction_utilisations = []
    actuator_utilisations = []
    for finger in FINGERS:
        force = np.asarray(result.x[force_slices[finger]], dtype=float)
        inward_normal = -np.asarray(TIP_OFFSETS_UNIT[finger], dtype=float)
        normal = max(0.0, float(inward_normal @ force))
        tangent = float(np.linalg.norm(force - normal * inward_normal))
        torques = (
            np.asarray(
                pose.contact_jacobians_world_m_per_rad[finger],
                dtype=float,
            ).T
            @ force
        )
        forces[finger] = force
        normal_forces[finger] = normal
        tangential_forces[finger] = tangent
        motor_torques[finger] = (float(torques[0]), float(torques[1]))
        friction_utilisations.append(
            tangent / (friction_coefficient * normal)
            if normal > 1e-12
            else (0.0 if tangent <= 1e-12 else math.inf)
        )
        actuator_utilisations.extend(
            (
                (abs(float(torques[0])) / torque_limit_nm, f"{finger}/back"),
                (abs(float(torques[1])) / torque_limit_nm, f"{finger}/front"),
            )
        )

    limiting_actuator = max(actuator_utilisations, key=lambda item: item[0])[1]
    return SpatialContactCapacity(
        payload_mass_kg=max(0.0, float(result.x[mass_index])),
        contact_forces_world_n=forces,
        normal_forces_n=normal_forces,
        tangential_forces_n=tangential_forces,
        motor_torques_nm=motor_torques,
        limiting_actuator=limiting_actuator,
        maximum_friction_utilisation=max(friction_utilisations),
        maximum_normal_force_n=max(normal_forces.values()),
    )


def calculate_spatial_capacity_rows(
    poses: Sequence[SpatialPoseMechanics],
    *,
    motors: Sequence[MotorSpec],
    requested_voltage_v: Optional[float],
    torque_mode: str,
    friction_coefficient: float,
    safety_factor: float,
    gravity_m_s2: float,
    object_half_width_m: float,
    max_normal_force_n: Optional[float],
    friction_facets: int,
) -> Tuple[SpatialCapacityRow, ...]:
    """Evaluate a complete route and retain each motor/mode's weakest pose."""

    if not poses:
        raise ValueError("至少需要一个搬运姿态。")
    if safety_factor < 1.0:
        raise ValueError("安全系数必须大于或等于 1。")

    rows = []
    for motor in motors:
        voltage, stall_torque, stall_current = motor.values_at(requested_voltage_v)
        for mode_key, mode_label, published_torque in _torque_modes(
            torque_mode,
            stall_torque,
        ):
            effective_torque = published_torque / safety_factor
            capacities = [
                (
                    pose,
                    maximise_spatial_payload(
                        pose,
                        torque_limit_nm=effective_torque,
                        friction_coefficient=friction_coefficient,
                        object_half_width_m=object_half_width_m,
                        gravity_m_s2=gravity_m_s2,
                        max_normal_force_n=max_normal_force_n,
                        friction_facets=friction_facets,
                    ),
                )
                for pose in poses
            ]
            bottleneck_pose, bottleneck_contact = min(
                capacities,
                key=lambda item: item[1].payload_mass_kg,
            )
            rows.append(
                SpatialCapacityRow(
                    motor=motor,
                    voltage_v=voltage,
                    stall_current_a=stall_current,
                    torque_mode=mode_key,
                    torque_mode_label=mode_label,
                    published_torque_nm=published_torque,
                    effective_torque_nm=effective_torque,
                    bottleneck_pose=bottleneck_pose,
                    contact=bottleneck_contact,
                )
            )
    return tuple(rows)


def _selected_motors(selection: str) -> Tuple[MotorSpec, ...]:
    if selection == "both":
        return (MOTOR_SPECS["xl"], MOTOR_SPECS["xc"])
    return (MOTOR_SPECS[selection],)


def _torque_modes(selection: str, stall_torque_nm: float) -> Tuple[Tuple[str, str, float], ...]:
    modes = {
        "continuous": (
            "continuous",
            "连续估算(20%)",
            0.20 * float(stall_torque_nm),
        ),
        "stall": ("stall", "堵转瞬时", float(stall_torque_nm)),
    }
    if selection == "both":
        return (modes["continuous"], modes["stall"])
    return (modes[selection],)


def calculate_capacity_rows(
    poses: Sequence[PoseMechanics],
    *,
    motors: Sequence[MotorSpec],
    requested_voltage_v: Optional[float],
    torque_mode: str,
    friction_coefficient: float,
    safety_factor: float,
    gravity_m_s2: float,
    max_normal_force_n: Optional[float],
) -> Tuple[CapacityRow, ...]:
    """Evaluate every motor/mode and retain the weakest pose on the path."""

    if not poses:
        raise ValueError("至少需要一个姿态。")
    if safety_factor < 1.0:
        raise ValueError("安全系数必须大于或等于 1。")

    rows = []
    for motor in motors:
        voltage, stall_torque, stall_current = motor.values_at(requested_voltage_v)
        for mode_key, mode_label, published_torque in _torque_modes(
            torque_mode,
            stall_torque,
        ):
            effective_torque = published_torque / safety_factor
            pose_capacities = [
                (
                    pose,
                    maximise_symmetric_contact_force(
                        pose.jacobian.matrix_m_per_rad,
                        torque_limit_nm=effective_torque,
                        friction_coefficient=friction_coefficient,
                        gravity_m_s2=gravity_m_s2,
                        finger_count=4,
                        max_normal_force_n=max_normal_force_n,
                    ),
                )
                for pose in poses
            ]
            bottleneck_pose, bottleneck_contact = min(
                pose_capacities,
                key=lambda item: item[1].payload_mass_kg,
            )
            rows.append(
                CapacityRow(
                    motor=motor,
                    voltage_v=voltage,
                    stall_current_a=stall_current,
                    torque_mode=mode_key,
                    torque_mode_label=mode_label,
                    published_torque_nm=published_torque,
                    effective_torque_nm=effective_torque,
                    bottleneck_pose=bottleneck_pose,
                    contact=bottleneck_contact,
                )
            )
    return tuple(rows)


def _row_to_json(
    row: CapacityRow,
    *,
    contact_patch_area_m2: float,
) -> Dict[str, object]:
    pose = row.bottleneck_pose
    contact = row.contact
    return {
        "motor": row.motor.name,
        "voltage_v": row.voltage_v,
        "torque_mode": row.torque_mode,
        "published_torque_nm": row.published_torque_nm,
        "effective_torque_after_safety_factor_nm": row.effective_torque_nm,
        "stall_current_per_motor_a": row.stall_current_a,
        "bottleneck_object_center_z_m": pose.object_center_z_m,
        "payload_mass_kg": contact.payload_mass_kg,
        "per_finger_normal_force_n": contact.normal_force_n,
        "per_finger_upward_force_n": contact.upward_force_n,
        "per_finger_average_contact_pressure_pa": (
            contact.normal_force_n / contact_patch_area_m2
        ),
        "back_torque_nm": contact.back_torque_nm,
        "front_torque_nm": contact.front_torque_nm,
        "limiting_motor": contact.limiting_motor,
        "friction_constraint_active": contact.friction_is_active,
        "normal_force_constraint_active": contact.normal_limit_is_active,
        "q_armbase_deg": math.degrees(pose.solution.q_armbase),
        "q_back_deg": math.degrees(pose.solution.q_back),
        "q_front_deg": math.degrees(pose.solution.q_front),
        "jacobian_m_per_rad": pose.jacobian.matrix_m_per_rad.tolist(),
        "passive_closure_condition": pose.jacobian.passive_closure_condition,
        "active_jacobian_condition": pose.jacobian.active_jacobian_condition,
    }


def _spatial_row_to_json(
    row: SpatialCapacityRow,
    *,
    contact_patch_area_m2: float,
) -> Dict[str, object]:
    pose = row.bottleneck_pose
    contact = row.contact
    return {
        "motor": row.motor.name,
        "voltage_v": row.voltage_v,
        "torque_mode": row.torque_mode,
        "published_torque_nm": row.published_torque_nm,
        "effective_torque_after_safety_factor_nm": row.effective_torque_nm,
        "stall_current_per_motor_a": row.stall_current_a,
        "payload_mass_kg": contact.payload_mass_kg,
        "bottleneck_segment": pose.segment_label,
        "bottleneck_object_center_world_m": (
            pose.object_center_world_m.tolist()
        ),
        "limiting_actuator": contact.limiting_actuator,
        "maximum_friction_utilisation": contact.maximum_friction_utilisation,
        "maximum_normal_force_n": contact.maximum_normal_force_n,
        "maximum_average_contact_pressure_pa": (
            contact.maximum_normal_force_n / contact_patch_area_m2
        ),
        "worst_jacobian_or_closure_condition": pose.worst_condition,
        "per_finger": {
            finger: {
                "contact_force_world_n": (
                    contact.contact_forces_world_n[finger].tolist()
                ),
                "normal_force_n": contact.normal_forces_n[finger],
                "tangential_force_n": contact.tangential_forces_n[finger],
                "average_contact_pressure_pa": (
                    contact.normal_forces_n[finger] / contact_patch_area_m2
                ),
                "back_torque_nm": contact.motor_torques_nm[finger][0],
                "front_torque_nm": contact.motor_torques_nm[finger][1],
                "q_armbase_deg": math.degrees(
                    pose.solutions[finger].q_armbase
                ),
                "q_back_deg": math.degrees(pose.solutions[finger].q_back),
                "q_front_deg": math.degrees(pose.solutions[finger].q_front),
                "contact_jacobian_world_m_per_rad": (
                    pose.contact_jacobians_world_m_per_rad[finger].tolist()
                ),
            }
            for finger in FINGERS
        },
    }


def print_human_report(
    *,
    xml_path: Path,
    start_z_m: float,
    end_z_m: Optional[float],
    path_samples: int,
    object_half_width_m: float,
    contact_patch: ContactPatchGeometry,
    contact_distance_m: float,
    friction_coefficient: float,
    safety_factor: float,
    gravity_m_s2: float,
    max_normal_force_n: Optional[float],
    poses: Sequence[PoseMechanics],
    rows: Sequence[CapacityRow],
) -> None:
    print("Gripper3 理论载荷计算（解析静力；未使用 MuJoCo 仿真）")
    print(f"几何来源: {xml_path}")
    if end_z_m is None:
        print(f"物体中心: x=y=0, z={start_z_m:.4f} m")
    else:
        print(
            f"抬升路径: x=y=0, z={start_z_m:.4f} -> {end_z_m:.4f} m, "
            f"{path_samples} 个采样点"
        )
    print(
        f"接触几何: 物体半宽={1000.0 * object_half_width_m:.1f} mm, "
        f"橡胶球半径={1000.0 * contact_patch.sphere_radius_m:.1f} mm, "
        f"接触圆直径={1000.0 * contact_patch.diameter_m:.1f} mm"
    )
    print(
        f"接触圆面积={1e6 * contact_patch.area_m2:.2f} mm^2, "
        f"球心到接触面={1000.0 * contact_patch.center_to_contact_plane_m:.2f} mm, "
        f"几何压入量≈{1000.0 * contact_patch.estimated_indentation_m:.2f} mm, "
        f"指尖中心半径={1000.0 * contact_distance_m:.2f} mm"
    )
    normal_limit_text = (
        "未设置"
        if max_normal_force_n is None
        else f"{max_normal_force_n:.3f} N/指"
    )
    print(
        f"接触/折减: mu={friction_coefficient:g}, 安全系数={safety_factor:g}, "
        f"法向力上限={normal_limit_text}, g={gravity_m_s2:g} m/s^2"
    )
    print(
        "armbase: 当前中心对称姿态的理想接触力不需要其绕轴驱动扭矩；"
        "轴承载荷/弯矩按用户要求不设上限。"
    )
    print()

    header = (
        f"{'电机':<17} {'电压':>6} {'扭矩口径':<14} {'单电机上限':>12} "
        f"{'瓶颈z':>8} {'最大物体质量':>14} {'N/指':>10} {'T↑/指':>10} "
        f"{'平均压强':>10} {'限制':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        contact = row.contact
        print(
            f"{row.motor.name:<17} {row.voltage_v:>5.1f}V "
            f"{row.torque_mode_label:<14} {row.effective_torque_nm:>9.3f} Nm "
            f"{row.bottleneck_pose.object_center_z_m:>7.4f}m "
            f"{contact.payload_mass_kg:>8.3f} kg"
            f" ({1000.0 * contact.payload_mass_kg:>5.0f}g) "
            f"{contact.normal_force_n:>8.3f}N "
            f"{contact.upward_force_n:>8.3f}N "
            f"{contact.normal_force_n / contact_patch.area_m2 / 1000.0:>8.3f}kPa "
            f"{contact.limiting_motor:>7}"
        )

    print()
    worst_condition = max(
        max(
            pose.jacobian.passive_closure_condition,
            pose.jacobian.active_jacobian_condition,
        )
        for pose in poses
    )
    print(f"路径最大雅可比/闭链条件数: {worst_condition:.2f}")
    if worst_condition > 50.0:
        print("警告: 路径靠近奇异位形；理想刚体模型可能明显夸大实际能力。")
    print(
        "说明: N 是单指夹紧法向力，T↑ 是单指向上摩擦力；四指的 T↑ 相加。"
    )
    print(
        "20 mm 接触圆采用球冠几何近似且合力位于圆心；库仑模型中接触面积"
        "不直接放大 mu*N，只用于合力位置与平均压强。"
    )
    print("按用户要求，机械手自身重量（连杆、电机和紧固件）完全不计入。")
    print("此外未计入加速度冲击、结构强度、间隙/柔性、温升及电源压降。")
    print(
        "堵转值只代表瞬时数学上限，不能持续持重；连续值是 ROBOTIS 披露的"
        "“堵转扭矩 20%”估算值，仍应通过实物称重和温升试验校准。"
    )
    print(
        "轴承只能卸载径向/轴向载荷与弯矩；若改成非中心姿态而需要 armbase"
        " 绕自身轴输出扭矩，该驱动扭矩仍需另行校核。"
    )


def print_spatial_route_report(
    *,
    xml_path: Path,
    start_world_m: np.ndarray,
    target_world_m: np.ndarray,
    transit_z_m: float,
    path_spacing_m: float,
    object_half_width_m: float,
    contact_patch: ContactPatchGeometry,
    contact_distance_m: float,
    friction_coefficient: float,
    friction_facets: int,
    safety_factor: float,
    gravity_m_s2: float,
    max_normal_force_n: Optional[float],
    poses: Sequence[SpatialPoseMechanics],
    rows: Sequence[SpatialCapacityRow],
) -> None:
    print("Gripper3 全路径理论载荷（解析准静态；未使用 MuJoCo 仿真）")
    print(f"几何来源: {xml_path}")
    print(
        "任务路径: 抓取点 "
        f"{np.array2string(start_world_m, precision=4)} -> "
        f"竖直抬升 -> 世界 XY 平面 z={transit_z_m:.4f} m -> "
        f"放置点 {np.array2string(target_world_m, precision=4)}"
    )
    print(
        f"路径采样: {len(poses)} 点，最大间距≈{1000.0 * path_spacing_m:g} mm；"
        "从刚离桌到刚触桌均按四指独立承受物体重量计算"
    )
    print(
        f"接触几何: 物体半宽={1000.0 * object_half_width_m:.1f} mm, "
        f"橡胶球半径={1000.0 * contact_patch.sphere_radius_m:.1f} mm, "
        f"接触圆直径={1000.0 * contact_patch.diameter_m:.1f} mm, "
        f"指尖中心半径={1000.0 * contact_distance_m:.2f} mm"
    )
    normal_limit_text = (
        "未设置"
        if max_normal_force_n is None
        else f"{max_normal_force_n:.3f} N/指"
    )
    print(
        f"接触/折减: mu={friction_coefficient:g}, "
        f"{friction_facets} 面内接摩擦锥, 安全系数={safety_factor:g}, "
        f"法向力上限={normal_limit_text}, g={gravity_m_s2:g} m/s^2"
    )
    print(
        "运动口径: 速度与加速度取趋近于 0 的准静态极限；"
        "平稳匀速段等价，但启动、刹车、转弯冲击未计。"
    )
    print(
        "armbase 参与完整 3-D IK/Jacobian，但其驱动扭矩上限按用户要求豁免；"
        "back/front 的 8 个电机均受扭矩限制。"
    )
    print(
        "注意: 轴承通常只能承受径向/轴向载荷和弯矩，并不会替代绕 armbase"
        " 电机轴的驱动扭矩；因此该豁免会使非中心位姿结果偏乐观。"
    )
    print()

    header = (
        f"{'电机':<17} {'电压':>6} {'扭矩口径':<14} {'单电机上限':>12} "
        f"{'最大物体质量':>14} {'最弱段':>8} {'最弱点 xyz (m)':>27} "
        f"{'最大N/指':>10} {'限制电机':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        pose = row.bottleneck_pose
        contact = row.contact
        xyz = ",".join(f"{value:+.3f}" for value in pose.object_center_world_m)
        print(
            f"{row.motor.name:<17} {row.voltage_v:>5.1f}V "
            f"{row.torque_mode_label:<14} {row.effective_torque_nm:>9.3f} Nm "
            f"{contact.payload_mass_kg:>8.3f} kg"
            f" ({1000.0 * contact.payload_mass_kg:>5.0f}g) "
            f"{pose.segment_label:>8} {xyz:>27} "
            f"{contact.maximum_normal_force_n:>8.3f}N "
            f"{contact.limiting_actuator:>10}"
        )

    print()
    worst_condition = max(pose.worst_condition for pose in poses)
    print(f"路径最大雅可比/闭链条件数: {worst_condition:.2f}")
    if worst_condition > 50.0:
        print("警告: 路径靠近奇异位形；理想刚体模型可能明显夸大实际能力。")
    print(
        "每个路径点均由 12 个接触力分量满足三维合力与合力矩平衡，"
        "程序不强制把重量平均分给四指。"
    )
    print(
        "20 mm 接触圆面积为 "
        f"{1e6 * contact_patch.area_m2:.2f} mm^2；库仑模型中面积不直接"
        "放大 mu*N，只用于接触位置与平均压强。"
    )
    print("按用户要求，机械手自身重量（连杆、电机和紧固件）完全不计入。")
    print(
        "连续值适用于低速搬运和中途停住持物；堵转值只代表瞬时数学上限，"
        "不能用于搬运或停住持重。"
    )
    print(
        "结果还未计入启动/刹车加速度、物体质心偏差、结构柔性、碰撞、"
        "电源压降和电机热累积；实际额定载荷应再留安全余量并实测温升。"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "用当前四指闭链几何、Dynamixel 官方扭矩和库仑摩擦，"
            "计算中心抓取或完整抬升—XY搬运—放置路径的理论载荷。"
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "示例:\n"
            "  python T3_payload_capacity.py\n"
            "  python T3_payload_capacity.py --friction 0.6 --lift-to-z 0.40\n"
            "  python T3_payload_capacity.py --target-x 0.02 --target-y 0\n"
            "  python T3_payload_capacity.py --voltage 12 --mode stall\n"
            "  python T3_payload_capacity.py --safety-factor 2 --json"
        ),
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=MODULE_DIR / "Project.xml",
        help="当前模型 Project XML；这里只读取几何/关节限制，不运行仿真",
    )
    parser.add_argument(
        "--motor",
        choices=("both", "xl", "xc"),
        default="both",
        help="比较两款电机，或只计算指定机型",
    )
    parser.add_argument(
        "--voltage",
        type=float,
        default=None,
        help="统一指定官方测试电压；省略时各机型采用官方推荐电压",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "continuous", "stall"),
        default="both",
        help="输出连续估算、堵转瞬时上限，或两者",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=1.2,
        help="实物指尖与物体之间的静摩擦系数（应以实测值替换）",
    )
    parser.add_argument(
        "--object-center-z",
        type=float,
        default=None,
        help="起始物体中心世界 z；省略时读取 Project.xml 中的 cube 初始位置",
    )
    parser.add_argument(
        "--object-half-width",
        type=float,
        default=None,
        help="中心到垂直接触面的距离，单位 m；省略时读取 XML 的 cube 半宽",
    )
    parser.add_argument(
        "--contact-diameter",
        type=float,
        default=0.020,
        help="每个橡胶 hard tip 的圆形接触斑直径，单位 m",
    )
    parser.add_argument(
        "--lift-to-z",
        type=float,
        default=None,
        help="若给定，则沿中心轴分析到此世界 z 的整段竖直抬升路径",
    )
    parser.add_argument(
        "--target-x",
        type=float,
        default=None,
        help="放置点世界 x，单位 m；给定任一 target 参数即启用完整搬运路径",
    )
    parser.add_argument(
        "--target-y",
        type=float,
        default=None,
        help="放置点世界 y，单位 m；未给定的坐标沿用抓取点",
    )
    parser.add_argument(
        "--target-z",
        type=float,
        default=None,
        help="放置点物体中心世界 z，单位 m；默认与抓取点相同",
    )
    parser.add_argument(
        "--transit-z",
        type=float,
        default=None,
        help=(
            "水平搬运平面的世界 z；默认采用控制器规则 "
            "max(起点z, 终点z, 桌面最低中心z+25mm)"
        ),
    )
    parser.add_argument(
        "--path-spacing",
        type=float,
        default=0.005,
        help="完整搬运路径的最大采样间距，单位 m",
    )
    parser.add_argument(
        "--friction-facets",
        type=int,
        default=16,
        help="3-D 路径计算使用的内接多面摩擦锥面数（4 的倍数）",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=51,
        help="抬升路径采样点数",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=1.0,
        help="额外安全系数；有效扭矩 = 口径扭矩 / 安全系数",
    )
    parser.add_argument(
        "--max-normal-force",
        type=float,
        default=None,
        help="可选的单指结构/物体允许法向力上限，单位 N",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=STANDARD_GRAVITY_M_S2,
        help="重力加速度，单位 m/s^2",
    )
    parser.add_argument(
        "--condition-limit",
        type=float,
        default=1e4,
        help="超过此雅可比/闭链条件数时拒绝报告理想奇异增益",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出适合其它程序读取的 JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        xml_path = args.xml.expanduser().resolve()
        ik = GripperIK.from_xml(xml_path)
        start_z = (
            float(ik.model_info.cube_initial_pos[2])
            if args.object_center_z is None
            else float(args.object_center_z)
        )
        object_half_width = (
            float(ik.model_info.cube_half_size)
            if args.object_half_width is None
            else float(args.object_half_width)
        )
        if object_half_width <= 0.0:
            raise ValueError("物体半宽必须大于 0。")
        contact_patch = estimate_circular_contact_patch(
            float(ik.model_info.hard_tip_radius),
            float(args.contact_diameter),
        )
        contact_distance = (
            object_half_width + contact_patch.center_to_contact_plane_m
        )
        route_requested = any(
            value is not None
            for value in (
                args.target_x,
                args.target_y,
                args.target_z,
                args.transit_z,
            )
        )
        if route_requested and args.lift_to_z is not None:
            raise ValueError(
                "--lift-to-z 只用于旧的中心竖直检查，不能与完整 target 路径同时使用。"
            )

        start_world = np.asarray(ik.model_info.cube_initial_pos, dtype=float).copy()
        start_world[2] = start_z
        target_world = start_world.copy()
        if args.target_x is not None:
            target_world[0] = float(args.target_x)
        if args.target_y is not None:
            target_world[1] = float(args.target_y)
        if args.target_z is not None:
            target_world[2] = float(args.target_z)

        if route_requested:
            if target_world[2] < ik.model_info.minimum_cube_center_z - 1e-10:
                raise ValueError(
                    f"放置点 z={target_world[2]:.6f} m 低于桌面允许的物体中心 "
                    f"z={ik.model_info.minimum_cube_center_z:.6f} m。"
                )
            transit_z = (
                max(
                    float(start_world[2]),
                    float(target_world[2]),
                    float(ik.model_info.minimum_cube_center_z) + 0.025,
                )
                if args.transit_z is None
                else float(args.transit_z)
            )
            spatial_poses = build_pick_move_place_path(
                ik,
                start_world_m=start_world,
                target_world_m=target_world,
                transit_z_m=transit_z,
                spacing_m=args.path_spacing,
                contact_distance_m=contact_distance,
                contact_plane_offset_m=contact_patch.center_to_contact_plane_m,
                condition_limit=args.condition_limit,
            )
            spatial_rows = calculate_spatial_capacity_rows(
                spatial_poses,
                motors=_selected_motors(args.motor),
                requested_voltage_v=args.voltage,
                torque_mode=args.mode,
                friction_coefficient=args.friction,
                safety_factor=args.safety_factor,
                gravity_m_s2=args.gravity,
                object_half_width_m=object_half_width,
                max_normal_force_n=args.max_normal_force,
                friction_facets=args.friction_facets,
            )
        else:
            poses = build_vertical_path(
                ik,
                start_z_m=start_z,
                end_z_m=args.lift_to_z,
                samples=args.samples,
                contact_distance_m=contact_distance,
                contact_plane_offset_m=contact_patch.center_to_contact_plane_m,
                condition_limit=args.condition_limit,
            )
            rows = calculate_capacity_rows(
                poses,
                motors=_selected_motors(args.motor),
                requested_voltage_v=args.voltage,
                torque_mode=args.mode,
                friction_coefficient=args.friction,
                safety_factor=args.safety_factor,
                gravity_m_s2=args.gravity,
                max_normal_force_n=args.max_normal_force,
            )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        common_assumptions = {
            "object_half_width_m": object_half_width,
            "hard_tip_radius_m": float(ik.model_info.hard_tip_radius),
            "contact_patch_diameter_m": contact_patch.diameter_m,
            "contact_patch_area_m2": contact_patch.area_m2,
            "contact_plane_offset_from_tip_center_m": (
                contact_patch.center_to_contact_plane_m
            ),
            "estimated_rubber_indentation_m": (
                contact_patch.estimated_indentation_m
            ),
            "contact_patch_model": "geometric_flat_spherical_cap_estimate",
            "contact_distance_m": contact_distance,
            "friction_coefficient": args.friction,
            "safety_factor": args.safety_factor,
            "gravity_m_s2": args.gravity,
            "max_normal_force_per_finger_n": args.max_normal_force,
            "armbase_motor_torque_constrained": False,
            "armbase_retained_in_kinematics": True,
            "link_self_weight_included": False,
            "link_self_weight_ignored_by_user_request": True,
            "dynamic_acceleration_included": False,
        }
        payload: Dict[str, object] = {
            "calculation": (
                "analytic_quasi_static_pick_move_place_not_mujoco_simulation"
                if route_requested
                else "analytic_static_closed_chain_not_mujoco_simulation"
            ),
            "xml": str(xml_path),
            "official_sources": [
                "https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/",
                "https://emanual.robotis.com/docs/en/dxl/x/xc430-w240/",
                "https://www.robotis.us/dynamixel-xl430-w250-t/",
                "https://www.robotis.us/dynamixel-xc430-w240-t/",
            ],
        }
        if route_requested:
            payload["assumptions"] = {
                **common_assumptions,
                "start_object_center_world_m": start_world.tolist(),
                "target_object_center_world_m": target_world.tolist(),
                "transport_plane_z_m": transit_z,
                "path_spacing_m": args.path_spacing,
                "path_samples": len(spatial_poses),
                "friction_cone_facets": args.friction_facets,
                "object_center_of_mass_at_geometric_center": True,
                "object_rotation_and_angular_acceleration": False,
                "linear_acceleration_world_m_s2": [0.0, 0.0, 0.0],
            }
            payload["results"] = [
                _spatial_row_to_json(
                    row,
                    contact_patch_area_m2=contact_patch.area_m2,
                )
                for row in spatial_rows
            ]
        else:
            payload["assumptions"] = {
                **common_assumptions,
                "object_center_x_m": 0.0,
                "object_center_y_m": 0.0,
                "start_z_m": start_z,
                "end_z_m": args.lift_to_z,
                "path_samples": len(poses),
            }
            payload["results"] = [
                _row_to_json(
                    row,
                    contact_patch_area_m2=contact_patch.area_m2,
                )
                for row in rows
            ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if route_requested:
            print_spatial_route_report(
                xml_path=xml_path,
                start_world_m=start_world,
                target_world_m=target_world,
                transit_z_m=transit_z,
                path_spacing_m=args.path_spacing,
                object_half_width_m=object_half_width,
                contact_patch=contact_patch,
                contact_distance_m=contact_distance,
                friction_coefficient=args.friction,
                friction_facets=args.friction_facets,
                safety_factor=args.safety_factor,
                gravity_m_s2=args.gravity,
                max_normal_force_n=args.max_normal_force,
                poses=spatial_poses,
                rows=spatial_rows,
            )
        else:
            print_human_report(
                xml_path=xml_path,
                start_z_m=start_z,
                end_z_m=args.lift_to_z,
                path_samples=len(poses),
                object_half_width_m=object_half_width,
                contact_patch=contact_patch,
                contact_distance_m=contact_distance,
                friction_coefficient=args.friction,
                safety_factor=args.safety_factor,
                gravity_m_s2=args.gravity,
                max_normal_force_n=args.max_normal_force,
                poses=poses,
                rows=rows,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
