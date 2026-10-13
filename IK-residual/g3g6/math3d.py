"""Small wxyz-quaternion helpers with no external geometry dependency."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def normalize_quat(quaternion: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(quaternion), dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("quaternion must be finite and nonzero")
    q = q / norm
    return -q if q[0] < 0.0 else q


def quat_conjugate(quaternion: Iterable[float]) -> np.ndarray:
    q = normalize_quat(quaternion)
    return np.array((q[0], -q[1], -q[2], -q[3]), dtype=np.float64)


def quat_multiply(left: Iterable[float], right: Iterable[float]) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quat(left)
    w2, x2, y2, z2 = normalize_quat(right)
    return normalize_quat(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def quat_to_matrix(quaternion: Iterable[float]) -> np.ndarray:
    w, x, y, z = normalize_quat(quaternion)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def euler_xyz_deg_to_quat(euler_deg: Iterable[float]) -> np.ndarray:
    """Return q for ordered local X, then Y, then Z component rotations."""

    rx, ry, rz = np.radians(np.asarray(tuple(euler_deg), dtype=np.float64))
    qx = np.array((math.cos(rx / 2), math.sin(rx / 2), 0, 0))
    qy = np.array((math.cos(ry / 2), 0, math.sin(ry / 2), 0))
    qz = np.array((math.cos(rz / 2), 0, 0, math.sin(rz / 2)))
    return quat_multiply(quat_multiply(qx, qy), qz)


def quat_to_euler_xyz_deg(quaternion: Iterable[float]) -> np.ndarray:
    """Inverse of the ordered XYZ convention, returned in [-180, 180]."""

    matrix = quat_to_matrix(quaternion)
    sy = float(np.clip(matrix[0, 2], -1.0, 1.0))
    ry = math.asin(sy)
    if abs(math.cos(ry)) > 1.0e-8:
        rx = math.atan2(-matrix[1, 2], matrix[2, 2])
        rz = math.atan2(-matrix[0, 1], matrix[0, 0])
    else:
        rx = math.atan2(matrix[2, 1], matrix[1, 1])
        rz = 0.0
    return wrap_degrees(np.degrees((rx, ry, rz)))


def orientation_error_xyz_deg(
    actual_quaternion: Iterable[float],
    target_quaternion: Iterable[float],
) -> np.ndarray:
    relative = quat_multiply(quat_conjugate(target_quaternion), actual_quaternion)
    return wrap_degrees(quat_to_euler_xyz_deg(relative))


def rotate_vectors(quaternion: Iterable[float], vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    return values @ quat_to_matrix(quaternion).T


def slerp(left: Iterable[float], right: Iterable[float], fraction: float) -> np.ndarray:
    q0 = normalize_quat(left)
    q1 = normalize_quat(right)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    value = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        return normalize_quat(q0 + value * (q1 - q0))
    angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
    return normalize_quat(
        math.sin((1.0 - value) * angle) / math.sin(angle) * q0
        + math.sin(value * angle) / math.sin(angle) * q1
    )


def wrap_degrees(value: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return (array + 180.0) % 360.0 - 180.0


def minimum_jerk(value: float) -> float:
    t = float(np.clip(value, 0.0, 1.0))
    return t**3 * (10.0 - 15.0 * t + 6.0 * t**2)
