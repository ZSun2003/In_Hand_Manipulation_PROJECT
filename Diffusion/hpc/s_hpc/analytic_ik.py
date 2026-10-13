"""Batched closed-chain analytic IK implemented with PyTorch CUDA tensors.

The equations are the same as the portable CPU oracle in ``kinematics.py``.
No MuJoCo Jacobian is used: the four passive closed-chain joints make a naive
active-column DLS Jacobian incorrect.  Constants are extracted once on the
host, then every solve remains on the selected Torch device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from .config import PACKAGE_ROOT
from .contracts import FINGERS


CTRL_INDEX = {
    "U": {"linear": 9, "back": 11, "front": 10, "armbase": 12},
    "D": {"linear": 6, "back": 7, "front": 8, "armbase": 13},
    "L": {"linear": 3, "back": 5, "front": 4, "armbase": 14},
    "R": {"linear": 0, "back": 1, "front": 2, "armbase": 15},
}


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by preflight
        raise RuntimeError("PyTorch is required for batched IK") from exc
    return torch


def _wrap_pi(value):
    torch = _torch()
    return torch.remainder(value + torch.pi, 2.0 * torch.pi) - torch.pi


def _rotate_axis(axis, angle, vector):
    """Rodrigues rotation with broadcast shapes ending in xyz."""

    torch = _torch()
    while axis.ndim < vector.ndim:
        axis = axis.unsqueeze(0)
    while angle.ndim < vector.ndim - 1:
        angle = angle.unsqueeze(-1)
    axis = torch.broadcast_to(axis, vector.shape)
    angle = torch.broadcast_to(angle, vector.shape[:-1])
    cosine = torch.cos(angle).unsqueeze(-1)
    sine = torch.sin(angle).unsqueeze(-1)
    dot = (axis * vector).sum(dim=-1, keepdim=True)
    return (
        vector * cosine
        + torch.linalg.cross(axis, vector, dim=-1) * sine
        + axis * dot * (1.0 - cosine)
    )


@dataclass
class IKResult:
    ctrl: object
    passive_q: object
    success: object
    finger_success: object
    score: object


class BatchedAnalyticIK:
    """Solve ``[world, finger, xyz]`` targets without a CPU hot path."""

    def __init__(
        self,
        *,
        device: str,
        linear_samples: int = 51,
        xml_path: str | Path | None = None,
    ) -> None:
        torch = _torch()
        from kinematics import GripperGeometryModel

        source = (
            PACKAGE_ROOT / "assets" / "Project_hpc.xml"
            if xml_path is None
            else Path(xml_path).expanduser().resolve()
        )
        geometry = GripperGeometryModel.from_xml(source)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.linear_samples = int(linear_samples)
        if self.linear_samples < 3:
            raise ValueError("linear_samples must be at least three")
        self.ctrl_ranges = torch.as_tensor(
            geometry.ctrl_ranges, device=self.device, dtype=self.dtype
        )

        def stack(name: str):
            return torch.as_tensor(
                np.stack(
                    [getattr(geometry.fingers[finger], name) for finger in FINGERS]
                ),
                device=self.device,
                dtype=self.dtype,
            )

        def scalars(name: str):
            return torch.as_tensor(
                [float(getattr(geometry.fingers[finger], name)) for finger in FINGERS],
                device=self.device,
                dtype=self.dtype,
            )

        self.p_A_world = stack("p_A_world")
        self.R_A_world = stack("R_A_world")
        self.p_AH0 = stack("p_AH0")
        self.R_AH0 = stack("R_AH0")
        self.arm_axis = stack("armbase_axis_A")
        self.linear_axis = stack("linear_axis_H")
        self.plane_normal = stack("plane_normal_H")
        self.eX = stack("eX_H")
        self.eY = stack("eY_H")
        self.O_H = stack("O_H")
        self.M2 = stack("M2")
        self.plane_offset = scalars("plane_offset")
        self.l1 = scalars("l1")
        self.l2 = scalars("l2")
        self.l3 = scalars("l3")
        self.dl = scalars("dl")
        self.r_tip = scalars("r_tip")
        self.kappa = scalars("kappa")
        self.theta_back0 = scalars("theta_back0")
        self.theta_front0 = scalars("theta_front0")
        self.rel_short0 = scalars("rel_short0")
        self.rel_elbow0 = scalars("rel_elbow0")
        self.sigma_back = scalars("sigma_back")
        self.sigma_front = scalars("sigma_front")
        self.sigma_short = scalars("sigma_short")
        self.sigma_elbow = scalars("sigma_elbow")

        active = []
        passive = []
        for finger in FINGERS:
            item = geometry.fingers[finger]
            active.append(
                [
                    item.active_ranges["linear"],
                    item.active_ranges["armbase"],
                    item.active_ranges["back"],
                    item.active_ranges["front"],
                ]
            )
            passive.append(
                [
                    item.passive_ranges["short"],
                    item.passive_ranges["elbow"],
                ]
            )
        self.active_ranges = torch.as_tensor(
            active, device=self.device, dtype=self.dtype
        )
        self.passive_ranges = torch.as_tensor(
            passive, device=self.device, dtype=self.dtype
        )
        self._ctrl_gather = {
            name: torch.as_tensor(
                [CTRL_INDEX[finger][name] for finger in FINGERS],
                device=self.device,
                dtype=torch.long,
            )
            for name in ("linear", "armbase", "back", "front")
        }

    @staticmethod
    def _in_range(value, limits, tolerance: float = 1.0e-6):
        return (value >= limits[..., 0] - tolerance) & (
            value <= limits[..., 1] + tolerance
        )

    def _previous(self, prev_ctrl, name: str):
        return prev_ctrl.index_select(1, self._ctrl_gather[name])

    def solve(self, targets, prev_ctrl) -> IKResult:
        """Return a safe control for every world.

        Failed fingers retain their previous actuator controls.  ``success``
        is per-world and callers can terminate/reset only those worlds.
        """

        torch = _torch()
        targets = torch.as_tensor(
            targets, device=self.device, dtype=self.dtype
        )
        prev_ctrl = torch.as_tensor(
            prev_ctrl, device=self.device, dtype=self.dtype
        )
        if targets.ndim != 3 or targets.shape[1:] != (4, 3):
            raise ValueError(f"targets must be [world,4,3], got {tuple(targets.shape)}")
        if prev_ctrl.shape != (targets.shape[0], 16):
            raise ValueError("prev_ctrl must be [world,16]")
        batch = targets.shape[0]
        eps = torch.tensor(1.0e-9, device=self.device, dtype=self.dtype)

        # World target expressed in each armbase zero frame.
        delta_world = targets - self.p_A_world.unsqueeze(0)
        p_A = torch.einsum(
            "fji,bfj->bfi", self.R_A_world, delta_world
        )

        # Linear slide candidates, exact inclusive range.
        lo = self.active_ranges[:, 0, 0]
        hi = self.active_ranges[:, 0, 1]
        fraction = torch.linspace(
            0.0, 1.0, self.linear_samples,
            device=self.device, dtype=self.dtype,
        )
        slide = lo[:, None] + (hi - lo)[:, None] * fraction[None, :]
        slide = slide.unsqueeze(0).expand(batch, -1, -1)  # B,F,S

        # Analytic armbase candidates from the moving linkage-plane equation.
        axis = self.arm_axis.unsqueeze(0)
        w = torch.einsum(
            "fij,fj->fi", self.R_AH0, self.plane_normal
        ).unsqueeze(0)
        parallel = axis * (axis * p_A).sum(dim=-1, keepdim=True)
        perpendicular = p_A - parallel
        cross = torch.linalg.cross(axis.expand_as(p_A), p_A, dim=-1)
        acoef = (w * perpendicular).sum(dim=-1)
        bcoef = (w * (-cross)).sum(dim=-1)
        base_c = (
            (w * (parallel - self.p_AH0.unsqueeze(0))).sum(dim=-1)
            - self.plane_offset.unsqueeze(0)
        )
        slide_plane = (
            self.plane_normal * self.linear_axis
        ).sum(dim=-1)
        ccoef = base_c.unsqueeze(-1) - slide * slide_plane[None, :, None]
        amplitude = torch.sqrt(acoef.square() + bcoef.square()).clamp_min(eps)
        rhs_raw = -ccoef / amplitude.unsqueeze(-1)
        arm_equation_valid = (
            (rhs_raw >= -1.0 - 1.0e-6)
            & (rhs_raw <= 1.0 + 1.0e-6)
            & (amplitude.unsqueeze(-1) > 1.0e-8)
        )
        alpha = torch.acos(rhs_raw.clamp(-1.0, 1.0))
        phi = torch.atan2(bcoef, acoef).unsqueeze(-1)
        arm = torch.stack((phi + alpha, phi - alpha), dim=-1)
        arm = _wrap_pi(arm)  # B,F,S,A
        arm_limits = self.active_ranges[:, 1, :][None, :, None, None, :]
        arm_valid = arm_equation_valid.unsqueeze(-1) & self._in_range(
            arm, arm_limits
        )

        # Transform target through inverse armbase rotation into handbase.
        p_A_expanded = p_A[:, :, None, None, :].expand(
            -1, -1, self.linear_samples, 2, -1
        )
        axis_expanded = self.arm_axis[None, :, None, None, :]
        p_after_arm = _rotate_axis(axis_expanded, -arm, p_A_expanded)
        handbase_delta = p_after_arm - self.p_AH0[None, :, None, None, :]
        p_H = torch.einsum(
            "fji,bfsaj->bfsai", self.R_AH0, handbase_delta
        )
        p_H = (
            p_H
            - slide[:, :, :, None, None]
            * self.linear_axis[None, :, None, None, :]
        )
        local_delta = p_H - self.O_H[None, :, None, None, :]
        d_point = torch.stack(
            (
                (local_delta * self.eX[None, :, None, None, :]).sum(-1),
                (local_delta * self.eY[None, :, None, None, :]).sum(-1),
            ),
            dim=-1,
        )  # B,F,S,A,2

        # Two circle intersections: centre M/r=l3 with centre D/r=r_tip.
        m_point = self.M2[None, :, None, None, :]
        md = d_point - m_point
        distance = torch.linalg.vector_norm(md, dim=-1).clamp_min(eps)
        l3 = self.l3[None, :, None, None]
        rtip = self.r_tip[None, :, None, None]
        circle_valid = (
            (distance <= l3 + rtip + 1.0e-6)
            & (distance >= torch.abs(l3 - rtip) - 1.0e-6)
            & (distance > 1.0e-8)
        )
        unit = md / distance.unsqueeze(-1)
        along = (
            l3.square() - rtip.square() + distance.square()
        ) / (2.0 * distance)
        height_sq = l3.square() - along.square()
        circle_valid &= height_sq >= -1.0e-7
        height = torch.sqrt(height_sq.clamp_min(0.0))
        middle = m_point + along.unsqueeze(-1) * unit
        perpendicular_2d = torch.stack((-unit[..., 1], unit[..., 0]), dim=-1)
        c_point = torch.stack(
            (
                middle + height.unsqueeze(-1) * perpendicular_2d,
                middle - height.unsqueeze(-1) * perpendicular_2d,
            ),
            dim=-2,
        )  # B,F,S,A,C,2

        d_expanded = d_point.unsqueeze(-2)
        dc_unit = (d_expanded - c_point) / rtip.unsqueeze(-1).unsqueeze(-1)
        cos_k = torch.cos(self.kappa)[None, :, None, None, None]
        sin_k = torch.sin(self.kappa)[None, :, None, None, None]
        u_bc = torch.stack(
            (
                cos_k * dc_unit[..., 0] - sin_k * dc_unit[..., 1],
                sin_k * dc_unit[..., 0] + cos_k * dc_unit[..., 1],
            ),
            dim=-1,
        )
        b_point = c_point - self.dl[None, :, None, None, None, None] * u_bc
        rho2 = b_point.square().sum(dim=-1)
        l1 = self.l1[None, :, None, None, None]
        l2 = self.l2[None, :, None, None, None]
        elbow_cos_raw = (
            rho2 - l1.square() - l2.square()
        ) / (2.0 * l1 * l2)
        elbow_valid = (
            (elbow_cos_raw >= -1.0 - 1.0e-6)
            & (elbow_cos_raw <= 1.0 + 1.0e-6)
        )
        elbow_cos = elbow_cos_raw.clamp(-1.0, 1.0)
        elbow_sin_abs = torch.sqrt((1.0 - elbow_cos.square()).clamp_min(0.0))
        elbow_sin = torch.stack((elbow_sin_abs, -elbow_sin_abs), dim=-1)
        elbow_cos = elbow_cos.unsqueeze(-1).expand_as(elbow_sin)

        b_angle = torch.atan2(b_point[..., 1], b_point[..., 0]).unsqueeze(-1)
        theta_back = b_angle - torch.atan2(
            l2.unsqueeze(-1) * elbow_sin,
            l1.unsqueeze(-1) + l2.unsqueeze(-1) * elbow_cos,
        )
        a_point = self.l1[None, :, None, None, None, None, None] * torch.stack(
            (torch.cos(theta_back), torch.sin(theta_back)), dim=-1
        )
        b_for_elbow = b_point.unsqueeze(-2)
        c_for_elbow = c_point.unsqueeze(-2)
        theta_ab = torch.atan2(
            b_for_elbow[..., 1] - a_point[..., 1],
            b_for_elbow[..., 0] - a_point[..., 0],
        )
        theta_front = torch.atan2(
            c_for_elbow[..., 1]
            - self.M2[None, :, None, None, None, None, 1],
            c_for_elbow[..., 0]
            - self.M2[None, :, None, None, None, None, 0],
        ).expand_as(theta_back)
        theta_bc = torch.atan2(
            c_for_elbow[..., 1] - b_for_elbow[..., 1],
            c_for_elbow[..., 0] - b_for_elbow[..., 0],
        ).expand_as(theta_back)

        q_back = self.sigma_back[None, :, None, None, None, None] * _wrap_pi(
            theta_back
            - self.theta_back0[None, :, None, None, None, None]
        )
        q_front = self.sigma_front[None, :, None, None, None, None] * _wrap_pi(
            theta_front
            - self.theta_front0[None, :, None, None, None, None]
        )
        q_short = self.sigma_short[None, :, None, None, None, None] * _wrap_pi(
            (theta_ab - theta_back)
            - self.rel_short0[None, :, None, None, None, None]
        )
        q_elbow = self.sigma_elbow[None, :, None, None, None, None] * _wrap_pi(
            (theta_bc - theta_ab)
            - self.rel_elbow0[None, :, None, None, None, None]
        )

        active_back = self.active_ranges[:, 2, :][None, :, None, None, None, None, :]
        active_front = self.active_ranges[:, 3, :][None, :, None, None, None, None, :]
        passive_short = self.passive_ranges[:, 0, :][None, :, None, None, None, None, :]
        passive_elbow = self.passive_ranges[:, 1, :][None, :, None, None, None, None, :]
        valid = (
            arm_valid.unsqueeze(-1).unsqueeze(-1)
            & circle_valid.unsqueeze(-1).unsqueeze(-1)
            & elbow_valid.unsqueeze(-1)
            & self._in_range(q_back, active_back)
            & self._in_range(q_front, active_front)
            & self._in_range(q_short, passive_short)
            & self._in_range(q_elbow, passive_elbow)
        )

        prev_back = self._previous(prev_ctrl, "back")[:, :, None, None, None, None]
        prev_front = self._previous(prev_ctrl, "front")[:, :, None, None, None, None]
        prev_arm = self._previous(prev_ctrl, "armbase")[:, :, None, None, None, None]
        prev_linear = self._previous(prev_ctrl, "linear")[:, :, None, None, None, None]
        arm_all = arm.unsqueeze(-1).unsqueeze(-1).expand_as(q_back)
        slide_all = slide.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(q_back)
        score = (
            torch.abs(_wrap_pi(q_back - prev_back))
            + torch.abs(_wrap_pi(q_front - prev_front))
            + 0.25 * torch.abs(_wrap_pi(arm_all - prev_arm))
            + 10.0 * torch.abs(slide_all - prev_linear)
            + torch.abs(slide_all)
        )
        score = torch.where(valid, score, torch.full_like(score, torch.inf))
        candidate_shape = score.shape[2:]
        score_flat = score.reshape(batch, 4, -1)
        best_score, best_index = score_flat.min(dim=-1)
        finger_success = torch.isfinite(best_score)

        def choose(value):
            flat = value.reshape(batch, 4, -1)
            return torch.gather(flat, 2, best_index.unsqueeze(-1)).squeeze(-1)

        selected = {
            "linear": choose(slide_all),
            "armbase": choose(arm_all),
            "back": choose(q_back),
            "front": choose(q_front),
            "short": choose(q_short),
            "elbow": choose(q_elbow),
        }
        ctrl = prev_ctrl.clone()
        for name in ("linear", "armbase", "back", "front"):
            indices = self._ctrl_gather[name]
            prior = ctrl.index_select(1, indices)
            values = torch.where(finger_success, selected[name], prior)
            ctrl[:, indices] = values
        ctrl = torch.maximum(
            torch.minimum(ctrl, self.ctrl_ranges[:, 1]),
            self.ctrl_ranges[:, 0],
        )
        passive_q = torch.stack(
            (selected["short"], selected["elbow"]), dim=-1
        )
        passive_q = torch.where(
            finger_success.unsqueeze(-1), passive_q, torch.zeros_like(passive_q)
        )
        return IKResult(
            ctrl=ctrl,
            passive_q=passive_q,
            success=finger_success.all(dim=-1),
            finger_success=finger_success,
            score=best_score,
        )


__all__ = ["BatchedAnalyticIK", "CTRL_INDEX", "IKResult"]
