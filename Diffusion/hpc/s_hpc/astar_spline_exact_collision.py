"""One-shot global A* using the task's exact MuJoCo contact geometry.

The publication ablation compares this single global full-ALIGN plan with the
Diffusion controller's receding-horizon plan refreshed every four ticks.  The
A* state combines tabletop ``(x, y)`` with monotonic pose progress from the
nominal supervisor handoff to the exact pre-insertion pose.  Node, edge,
shortcut, and final command chords use the same zero-inflation filtered MuJoCo
screw--socket contact model; there is no fixed staging point or scripted
transport/rotate/dock decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

import numpy as np

from screw_control import (
    DIRECT_ALIGN_STEPS,
    PREINSERT_GATE_STEP,
    SCRIPTED_PREFIX_STEPS,
)
from screw_config import (
    SCREW_FREE_JOINT_NAME,
    SCREW_HEAD_GEOM_NAME,
    SCREW_SHAFT_GEOM_NAME,
    SOCKET_BODY_NAME,
)

from .astar_spline import (
    SPLINE_VALIDATION_SPACING_M,
    BatchedAStarSplinePlanner,
    AStarPlanError,
    _minimum_jerk,
)
from .supervisor import (
    BatchedSupervisor,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_log,
)


CONTACT_TOLERANCE_M = 1.0e-9
SOCKET_COLLISION_GEOM_PREFIX = "socket_wall_"
SOCKET_COLLISION_GEOM_NAMES = frozenset({"socket_bottom_collision"})
COLLISION_MODEL = "exact_mujoco_filtered_screw_socket_contacts"
GLOBAL_POSE_LAYERS = 33
GLOBAL_POSE_EDGE_COST_M = 1.0e-6
GLOBAL_ROTATION_VALIDATION_RAD = math.radians(0.25)
GLOBAL_ROTATION_PATH_WEIGHT_M = 0.040
GLOBAL_PLANNER_CONTRACT = "single_global_full_align_pose_lattice_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ExactMuJoCoTransportCollision:
    """Reusable CPU MuJoCo contact classifier for one transport pose."""

    def __init__(
        self,
        *,
        xml_path: Path,
        screw_tip_z_m: float,
        screw_rotation: np.ndarray,
        socket_position: np.ndarray,
        socket_quaternion_wxyz: np.ndarray,
        contact_tolerance_m: float = CONTACT_TOLERANCE_M,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.xml_path = Path(xml_path).expanduser().resolve()
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"MuJoCo XML is missing: {self.xml_path}")
        if contact_tolerance_m < 0.0:
            raise ValueError("contact tolerance must be non-negative")
        self.contact_tolerance_m = float(contact_tolerance_m)
        self.screw_tip_z_m = float(screw_tip_z_m)
        self.screw_rotation = np.asarray(screw_rotation, dtype=np.float64)
        if self.screw_rotation.shape != (3, 3):
            raise ValueError("screw rotation must have shape [3, 3]")

        model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        data = mujoco.MjData(model)
        self.model = model
        self.data = data

        def object_id(object_type, name: str) -> int:
            identifier = int(mujoco.mj_name2id(model, object_type, name))
            if identifier < 0:
                raise ValueError(f"MuJoCo object is missing: {name}")
            return identifier

        screw_joint = object_id(
            mujoco.mjtObj.mjOBJ_JOINT, SCREW_FREE_JOINT_NAME
        )
        self.screw_qpos = int(model.jnt_qposadr[screw_joint])
        socket_body = object_id(mujoco.mjtObj.mjOBJ_BODY, SOCKET_BODY_NAME)
        socket_mocap = int(model.body_mocapid[socket_body])
        if socket_mocap < 0:
            raise ValueError("socket body is not mocap-controlled")
        self.socket_mocap = socket_mocap

        self.geom_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in range(model.ngeom)
        ]
        screw_names = frozenset(
            {SCREW_SHAFT_GEOM_NAME, SCREW_HEAD_GEOM_NAME}
        )
        self.screw_geom_ids = {
            geom_id
            for geom_id, name in enumerate(self.geom_names)
            if name in screw_names
        }
        self.socket_geom_ids = {
            geom_id
            for geom_id, name in enumerate(self.geom_names)
            if name in SOCKET_COLLISION_GEOM_NAMES
            or (
                name is not None
                and name.startswith(SOCKET_COLLISION_GEOM_PREFIX)
            )
        }
        if len(self.screw_geom_ids) != len(screw_names):
            raise ValueError("MuJoCo XML does not contain both screw geoms")
        if not self.socket_geom_ids:
            raise ValueError("MuJoCo XML contains no socket collision geoms")

        socket_position = np.asarray(socket_position, dtype=np.float64)
        socket_quaternion = np.asarray(
            socket_quaternion_wxyz, dtype=np.float64
        )
        if socket_position.shape != (3,) or socket_quaternion.shape != (4,):
            raise ValueError("socket pose must have shapes [3] and [4]")
        self.socket_position = socket_position.copy()
        socket_norm = float(np.linalg.norm(socket_quaternion))
        if not math.isfinite(socket_norm) or socket_norm <= 1.0e-12:
            raise ValueError("socket quaternion must have finite non-zero norm")
        self.socket_quaternion = socket_quaternion / socket_norm

        # The grid coordinates describe the screw tip.  Convert them to the
        # free-joint body centre for the fixed transport orientation.
        self.local_tip = np.asarray((0.0, 0.0, -0.040), dtype=np.float64)
        self.tip_to_center = -(self.screw_rotation @ self.local_tip)
        rotation_tensor = __import__("torch").as_tensor(
            self.screw_rotation, dtype=__import__("torch").float64
        ).reshape(1, 3, 3)
        quaternion = matrix_to_quaternion(rotation_tensor)[0].numpy()
        screw_norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(screw_norm) or screw_norm <= 1.0e-12:
            raise ValueError("screw quaternion must have finite non-zero norm")
        self.screw_quaternion = quaternion / screw_norm
        self._reset_data_state()

    def _reset_data_state(self) -> None:
        """Clear derived MuJoCo state before each independent pose batch.

        A global lattice and its shortcut sweeps issue hundreds of thousands
        of ``mj_forward`` calls.  MuJoCo 3.10 is not robust to carrying the
        same derived solver/contact state across that entire sequence.  A
        batch-boundary ``mj_resetData`` keeps the exact same model and contact
        predicate while isolating independent vector queries.
        """

        model = self.model
        data = self.data
        self.mujoco.mj_resetData(model, data)
        if model.neq:
            data.eq_active[:] = 0
        data.mocap_pos[self.socket_mocap] = self.socket_position
        data.mocap_quat[self.socket_mocap] = self.socket_quaternion
        data.qpos[
            self.screw_qpos + 3 : self.screw_qpos + 7
        ] = self.screw_quaternion

    def _filtered_contact_present(self) -> bool:
        data = self.data
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            screw_socket = (
                geom1 in self.screw_geom_ids
                and geom2 in self.socket_geom_ids
            ) or (
                geom2 in self.screw_geom_ids
                and geom1 in self.socket_geom_ids
            )
            if (
                screw_socket
                and float(contact.dist) <= self.contact_tolerance_m
            ):
                return True
        return False

    def collides_poses(
        self,
        tips_xyz: np.ndarray,
        rotations: np.ndarray,
    ) -> np.ndarray:
        """Classify arbitrary full screw poses against the socket.

        The positions describe the screw tip, not the free-joint body centre.
        This is the collision oracle used by the global pose-lattice planner;
        it therefore covers translation, raising and rotation with the same
        filtered MuJoCo contacts instead of validating only an upright XY map.
        """

        tips = np.asarray(tips_xyz, dtype=np.float64).reshape(-1, 3)
        rotation = np.asarray(rotations, dtype=np.float64)
        if rotation.shape == (3, 3):
            # ``broadcast_to`` returns a read-only view, while Torch warns that
            # a non-writable NumPy input has undefined mutation semantics.
            rotation = np.broadcast_to(rotation, (len(tips), 3, 3)).copy()
        if rotation.shape != (len(tips), 3, 3):
            raise ValueError(
                "rotations must have shape [sample, 3, 3] or [3, 3]"
            )
        if not np.isfinite(tips).all() or not np.isfinite(rotation).all():
            raise FloatingPointError("full-pose collision query is non-finite")

        torch = __import__("torch")
        quaternion = matrix_to_quaternion(
            torch.as_tensor(rotation, dtype=torch.float64)
        ).numpy()
        quaternion_norm = np.linalg.norm(quaternion, axis=1)
        if (
            not np.isfinite(quaternion_norm).all()
            or bool((quaternion_norm <= 1.0e-12).any())
        ):
            raise FloatingPointError(
                "full-pose collision query produced an invalid quaternion"
            )
        quaternion = quaternion / quaternion_norm[:, None]
        centres = tips - np.einsum(
            "bij,j->bi", rotation, self.local_tip
        )
        collision = np.zeros(len(tips), dtype=np.bool_)
        data = self.data
        qpos = self.screw_qpos
        mujoco = self.mujoco
        self._reset_data_state()
        for index in range(len(tips)):
            data.qpos[qpos : qpos + 3] = centres[index]
            data.qpos[qpos + 3 : qpos + 7] = quaternion[index]
            mujoco.mj_forward(self.model, data)
            collision[index] = self._filtered_contact_present()
        return collision

    def collides(self, points_xy: np.ndarray) -> np.ndarray:
        """Compatibility wrapper for the legacy fixed transport pose."""

        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        tips = np.column_stack(
            (
                points,
                np.full(len(points), self.screw_tip_z_m, dtype=np.float64),
            )
        )
        return self.collides_poses(tips, self.screw_rotation)

    def provenance(self) -> Dict[str, Any]:
        return {
            "model": COLLISION_MODEL,
            "method": "cpu_mujoco_mj_forward_filtered_contacts",
            "mujoco_version": str(self.mujoco.__version__),
            "xml_path": str(self.xml_path),
            "xml_sha256": _sha256_file(self.xml_path),
            "contact_tolerance_m": self.contact_tolerance_m,
            "clearance_inflation_m": 0.0,
            "screw_tip_z_m": self.screw_tip_z_m,
            "reference_screw_pose": "nominal supervisor handoff pose",
            "supports_arbitrary_full_pose_queries": True,
            "mjdata_reset_before_each_pose_batch": True,
            "query_quaternions_normalised": True,
            "included_screw_geoms": sorted(
                self.geom_names[index] for index in self.screw_geom_ids
            ),
            "included_socket_geoms": sorted(
                self.geom_names[index] for index in self.socket_geom_ids
            ),
            "ignored_contact_classes": [
                "table",
                "gripper/fingertips",
                "spawn holder",
                "visual-only geoms",
            ],
        }


@dataclass(frozen=True)
class _GlobalCachedPlan:
    states: Tuple[int, ...]
    expanded: int


class ExactCollisionBatchedAStarSplinePlanner(BatchedAStarSplinePlanner):
    """One-shot global full-ALIGN A* on a reduced pose lattice.

    The state is ``(pose_progress, x, y)``.  Pose progress monotonically
    interpolates the exact supervisor handoff height/orientation to the exact
    pre-insertion height/orientation.  A* is therefore free to translate and
    rotate concurrently anywhere along its route; no spatial staging point or
    scripted transport/rotate/dock decomposition remains.

    The historical class name is retained for checkpoint/tool compatibility.
    Its provenance explicitly identifies the new global pose-lattice contract
    and the absence of a cubic-spline or expert fallback.
    """

    def __init__(self, config, worlds: int, device) -> None:
        self._exact_collision: _ExactMuJoCoTransportCollision | None = None
        self._exact_collision_provenance: Dict[str, Any] | None = None
        self._global_free: np.ndarray | None = None
        self._global_reverse_distance: np.ndarray | None = None
        self._global_successor: np.ndarray | None = None
        self._global_goal_ij: Tuple[int, int] | None = None
        self._global_goal_flat = -1
        self._global_plan_cache: Dict[Tuple[int, int], _GlobalCachedPlan] = {}
        self._global_edge_cache: Dict[Tuple[int, int], bool] = {}
        self._pose_fractions = np.linspace(
            0.0, 1.0, GLOBAL_POSE_LAYERS, dtype=np.float64
        )
        self._pose_tip_z = np.zeros(GLOBAL_POSE_LAYERS, dtype=np.float64)
        self._pose_rotations = np.zeros(
            (GLOBAL_POSE_LAYERS, 3, 3), dtype=np.float64
        )
        self._start_quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
        self._goal_quaternion = self._start_quaternion.copy()
        self._rotation_angle_rad = 0.0
        self._canonical_goal_xy = np.zeros(2, dtype=np.float64)
        self._exact_colliding_pose_nodes = 0
        self._exact_free_pose_nodes = 0
        self._direct_shortcuts = 0
        self._float32_shortcut_fallbacks = 0
        self._lazy_search_restarts = 0
        self._lazy_edge_evaluations = 0
        self._collision_signature: Tuple[float, ...] | None = None
        super().__init__(config, worlds, device)
        torch = __import__("torch")
        self._task_translation_limit_fraction = torch.zeros(
            self.worlds, device=self.device, dtype=self.dtype
        )
        self._task_rotation_limit_fraction = torch.zeros_like(
            self._task_translation_limit_fraction
        )

    def _canonical_basis(
        self, mouth_xy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        radial = np.asarray(mouth_xy, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(radial))
        if (
            not math.isfinite(norm)
            or abs(norm - self.task.socket_mouth_orbit_radius_m) > 2.0e-5
        ):
            raise AStarPlanError(
                "socket mouth must lie on the configured 20 mm orbit"
            )
        inward = radial / norm
        tangent = np.asarray((-inward[1], inward[0]), dtype=np.float64)
        return inward, tangent

    @staticmethod
    def _yaw_frame(inward: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        return np.asarray(
            (
                (inward[0], tangent[0], 0.0),
                (inward[1], tangent[1], 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _canonical_rotation(
        rotation: np.ndarray,
        inward: np.ndarray,
        tangent: np.ndarray,
    ) -> np.ndarray:
        frame = ExactCollisionBatchedAStarSplinePlanner._yaw_frame(
            inward, tangent
        )
        return frame.T @ np.asarray(rotation, dtype=np.float64).reshape(3, 3)

    @staticmethod
    def _quaternion_slerp_numpy(
        first: np.ndarray, second: np.ndarray, fraction: np.ndarray
    ) -> np.ndarray:
        q0 = np.asarray(first, dtype=np.float64).reshape(4)
        q1 = np.asarray(second, dtype=np.float64).reshape(4)
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = float(np.clip(dot, 0.0, 1.0))
        value = np.asarray(fraction, dtype=np.float64).reshape(-1, 1)
        if dot > 0.9995:
            result = (1.0 - value) * q0 + value * q1
        else:
            theta = math.acos(dot)
            sine = math.sin(theta)
            result = (
                np.sin((1.0 - value) * theta) / sine * q0
                + np.sin(value * theta) / sine * q1
            )
        return result / np.linalg.norm(result, axis=1, keepdims=True)

    def _rotations_at(self, fractions: np.ndarray) -> np.ndarray:
        torch = __import__("torch")
        quaternion = self._quaternion_slerp_numpy(
            self._start_quaternion, self._goal_quaternion, fractions
        )
        return quaternion_to_matrix(
            torch.as_tensor(quaternion, dtype=torch.float64)
        ).numpy()

    def _tips_and_rotations(
        self, points_xy: np.ndarray, fractions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        progress = np.asarray(fractions, dtype=np.float64).reshape(-1)
        if len(points) != len(progress):
            raise ValueError("point and pose-progress sample counts differ")
        z = self._pose_tip_z[0] + progress * (
            self._pose_tip_z[-1] - self._pose_tip_z[0]
        )
        return np.column_stack((points, z)), self._rotations_at(progress)

    @staticmethod
    def _rotation_z_numpy(angles: np.ndarray) -> np.ndarray:
        value = np.asarray(angles, dtype=np.float64).reshape(-1)
        cosine, sine = np.cos(value), np.sin(value)
        result = np.zeros((len(value), 3, 3), dtype=np.float64)
        result[:, 0, 0] = cosine
        result[:, 0, 1] = -sine
        result[:, 1, 0] = sine
        result[:, 1, 1] = cosine
        result[:, 2, 2] = 1.0
        return result

    @staticmethod
    def _local_z_roll_angle(base: np.ndarray, target: np.ndarray) -> float:
        relative = np.asarray(base, dtype=np.float64).T @ np.asarray(
            target, dtype=np.float64
        )
        if (
            float(np.max(np.abs(relative[:2, 2]))) > 3.0e-5
            or float(np.max(np.abs(relative[2, :2]))) > 3.0e-5
            or abs(float(relative[2, 2]) - 1.0) > 3.0e-5
        ):
            raise AStarPlanError(
                "world endpoint differs from canonical pose by more than axial roll"
            )
        return math.atan2(float(relative[1, 0]), float(relative[0, 0]))

    def _world_rotations_at(
        self,
        fractions: np.ndarray,
        *,
        inward: np.ndarray,
        tangent: np.ndarray,
        start_rotation: np.ndarray,
        goal_rotation: np.ndarray,
    ) -> np.ndarray:
        """Lift the yaw-equivariant collision schedule to an exact world frame.

        Screw collision geoms are cylinders, so right-multiplication by a local
        Z roll leaves their physical pose unchanged.  We use that free roll to
        match each world's exact supervisor endpoint frames without changing
        the canonical screw-axis schedule certified by the lattice.
        """

        progress = np.asarray(fractions, dtype=np.float64).reshape(-1)
        frame = self._yaw_frame(inward, tangent)
        canonical = self._rotations_at(progress)
        base = np.einsum("ij,njk->nik", frame, canonical)
        base_start = frame @ self._pose_rotations[0]
        base_goal = frame @ self._pose_rotations[-1]
        roll_start = self._local_z_roll_angle(base_start, start_rotation)
        roll_goal = self._local_z_roll_angle(base_goal, goal_rotation)
        roll_delta = (roll_goal - roll_start + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        roll = roll_start + progress * roll_delta
        result = np.einsum(
            "nij,njk->nik", base, self._rotation_z_numpy(roll)
        )
        if progress.size and abs(float(progress[-1]) - 1.0) <= 1.0e-7:
            result[-1] = np.asarray(goal_rotation, dtype=np.float64)
        return result

    def _actual_command_sweep_free(
        self, canonical_tips: np.ndarray, canonical_rotations: np.ndarray
    ) -> bool:
        """Certify the actual float32 command chords in canonical coordinates."""

        if self._exact_collision is None:
            raise RuntimeError("global exact collision model is not configured")
        tips = np.asarray(canonical_tips, dtype=np.float64).reshape(-1, 3)
        rotations = np.asarray(canonical_rotations, dtype=np.float64).reshape(
            -1, 3, 3
        )
        if len(tips) != len(rotations) or len(tips) < 2:
            raise ValueError("actual command sweep needs matching pose endpoints")
        torch = __import__("torch")
        quaternions = matrix_to_quaternion(
            torch.as_tensor(rotations.copy(), dtype=torch.float64)
        ).numpy()
        sampled_tips: list[np.ndarray] = []
        sampled_quaternions: list[np.ndarray] = []
        for index in range(len(tips) - 1):
            translation = float(np.linalg.norm(tips[index + 1] - tips[index]))
            dot = abs(float(np.dot(quaternions[index], quaternions[index + 1])))
            angle = 2.0 * math.acos(float(np.clip(dot, 0.0, 1.0)))
            count = max(
                2,
                int(math.ceil(translation / SPLINE_VALIDATION_SPACING_M)) + 1,
                int(math.ceil(angle / GLOBAL_ROTATION_VALIDATION_RAD)) + 1,
            )
            local = np.linspace(0.0, 1.0, count, dtype=np.float64)
            sampled_tips.append(
                tips[index][None, :]
                + local[:, None] * (tips[index + 1] - tips[index])[None, :]
            )
            sampled_quaternions.append(
                self._quaternion_slerp_numpy(
                    quaternions[index], quaternions[index + 1], local
                )
            )
        dense_tips = np.concatenate(sampled_tips, axis=0)
        if bool(
            (
                np.sum(dense_tips[:, :2] ** 2, axis=1)
                > (self.workspace_radius_m + 1.0e-6) ** 2
            ).any()
        ):
            return False
        dense_rotations = quaternion_to_matrix(
            torch.as_tensor(
                np.concatenate(sampled_quaternions, axis=0),
                dtype=torch.float64,
            )
        ).numpy()
        return not bool(
            self._exact_collision.collides_poses(
                dense_tips, dense_rotations
            ).any()
        )

    @staticmethod
    def _task_increment_fractions(
        tips: np.ndarray, rotations: np.ndarray, socket_frame: np.ndarray
    ) -> Tuple[float, float]:
        """Return maximum use of the Diffusion per-tick task-space limits."""

        position_limits = np.asarray((0.002, 0.002, 0.003), dtype=np.float64)
        rotation_limits = np.deg2rad((2.0, 2.0, 4.0))
        delta_position_world = np.diff(np.asarray(tips, dtype=np.float64), axis=0)
        delta_position_socket = np.einsum(
            "ji,nj->ni", socket_frame, delta_position_world
        )
        rotation = np.asarray(rotations, dtype=np.float64)
        delta_world = np.einsum(
            "nij,nkj->nik", rotation[1:], rotation[:-1]
        )
        delta_socket = np.einsum(
            "ji,njk,kl->nil", socket_frame, delta_world, socket_frame
        )
        torch = __import__("torch")
        rotvec = rotation_log(
            torch.as_tensor(delta_socket, dtype=torch.float64)
        ).numpy()
        return (
            float(np.max(np.abs(delta_position_socket) / position_limits)),
            float(np.max(np.abs(rotvec) / rotation_limits)),
        )

    def _sample_joint_segment(
        self,
        first_xy: np.ndarray,
        first_fraction: float,
        second_xy: np.ndarray,
        second_fraction: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        first = np.asarray(first_xy, dtype=np.float64).reshape(2)
        second = np.asarray(second_xy, dtype=np.float64).reshape(2)
        df = float(second_fraction) - float(first_fraction)
        if df < -1.0e-12:
            raise ValueError("global pose progress may not move backwards")
        dz = df * (self._pose_tip_z[-1] - self._pose_tip_z[0])
        translation = math.sqrt(float(np.sum((second - first) ** 2)) + dz * dz)
        rotation = abs(df) * self._rotation_angle_rad
        count = max(
            2,
            int(math.ceil(translation / SPLINE_VALIDATION_SPACING_M)) + 1,
            int(math.ceil(rotation / GLOBAL_ROTATION_VALIDATION_RAD)) + 1,
        )
        local = np.linspace(0.0, 1.0, count, dtype=np.float64)
        points = first[None, :] + local[:, None] * (second - first)[None, :]
        progress = float(first_fraction) + local * df
        return points, progress

    def _joint_segment_free(
        self,
        first_xy: np.ndarray,
        first_fraction: float,
        second_xy: np.ndarray,
        second_fraction: float,
    ) -> bool:
        if self._exact_collision is None:
            raise RuntimeError("global exact collision model is not configured")
        points, progress = self._sample_joint_segment(
            first_xy, first_fraction, second_xy, second_fraction
        )
        radius = self.workspace_radius_m + 1.0e-6
        if bool((np.sum(points * points, axis=1) > radius * radius).any()):
            return False
        tips, rotations = self._tips_and_rotations(points, progress)
        return not bool(
            self._exact_collision.collides_poses(tips, rotations).any()
        )

    def _points_free(self, points_xy: np.ndarray) -> np.ndarray:
        """Exact layer-zero compatibility query (never the inherited OBB)."""

        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        radius = self.workspace_radius_m + 1.0e-6
        inside = np.sum(points * points, axis=1) <= radius * radius
        if self._exact_collision is None:
            return inside
        tips, rotations = self._tips_and_rotations(
            points, np.zeros(len(points), dtype=np.float64)
        )
        return inside & ~self._exact_collision.collides_poses(tips, rotations)

    def _execution_chords_free(
        self, continuous_start_xy: np.ndarray, sampled_path_xy: np.ndarray
    ) -> bool:
        """Exact layer-zero compatibility query for legacy analysis callers."""

        previous = np.asarray(continuous_start_xy, dtype=np.float64).reshape(2)
        for point in np.asarray(sampled_path_xy, dtype=np.float64).reshape(-1, 2):
            if not self._joint_segment_free(previous, 0.0, point, 0.0):
                return False
            previous = point
        return True

    def _global_flat(self, layer: int, i: int, j: int) -> int:
        rows, columns = self._grid_shape
        return (int(layer) * rows + int(i)) * columns + int(j)

    def _global_unflatten(self, flat: int) -> Tuple[int, int, int]:
        rows, columns = self._grid_shape
        layer_row, j = divmod(int(flat), columns)
        layer, i = divmod(layer_row, rows)
        return layer, i, j

    def _transition_allowed(
        self,
        first: Tuple[int, int, int],
        second: Tuple[int, int, int],
    ) -> bool:
        if self._global_free is None:
            return False
        layer, i, j = first
        following_layer, ni, nj = second
        dk, di, dj = following_layer - layer, ni - i, nj - j
        if dk not in (0, 1) or abs(di) > 1 or abs(dj) > 1:
            return False
        if dk == 0 and di == 0 and dj == 0:
            return False
        if not self._global_free[layer, i, j] or not self._global_free[
            following_layer, ni, nj
        ]:
            return False
        if di and dj:
            for checked_layer in {layer, following_layer}:
                if not (
                    self._global_free[checked_layer, i + di, j]
                    and self._global_free[checked_layer, i, j + dj]
                ):
                    return False
        return True

    def _node_edge_free(self, first_flat: int, second_flat: int) -> bool:
        key = (int(first_flat), int(second_flat))
        cached = self._global_edge_cache.get(key)
        if cached is not None:
            return cached
        self._lazy_edge_evaluations += 1
        first = self._global_unflatten(first_flat)
        second = self._global_unflatten(second_flat)
        first_xy = self._grid_xy(first[1], first[2])
        second_xy = self._grid_xy(second[1], second[2])
        result = self._joint_segment_free(
            first_xy,
            self._pose_fractions[first[0]],
            second_xy,
            self._pose_fractions[second[0]],
        )
        self._global_edge_cache[key] = result
        if first[0] == second[0]:
            self._global_edge_cache[(int(second_flat), int(first_flat))] = result
        return result

    def _iter_global_neighbours(
        self, layer: int, i: int, j: int, *, validate_edges: bool
    ) -> Iterator[Tuple[int, int, int, float]]:
        for dk in (0, 1):
            following_layer = layer + dk
            if following_layer >= GLOBAL_POSE_LAYERS:
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if not (
                        0 <= ni < self._grid_shape[0]
                        and 0 <= nj < self._grid_shape[1]
                    ):
                        continue
                    following = (following_layer, ni, nj)
                    if not self._transition_allowed((layer, i, j), following):
                        continue
                    first_flat = self._global_flat(layer, i, j)
                    second_flat = self._global_flat(*following)
                    if validate_edges and not self._node_edge_free(
                        first_flat, second_flat
                    ):
                        continue
                    edge = math.hypot(di, dj) * self.grid_resolution_m
                    edge += dk * GLOBAL_POSE_EDGE_COST_M
                    yield following_layer, ni, nj, edge

    def _iter_global_predecessors(
        self, layer: int, i: int, j: int
    ) -> Iterator[Tuple[int, int, int, float]]:
        for previous_layer in (layer, layer - 1):
            if previous_layer < 0:
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    pi, pj = i + di, j + dj
                    if not (
                        0 <= pi < self._grid_shape[0]
                        and 0 <= pj < self._grid_shape[1]
                    ):
                        continue
                    previous = (previous_layer, pi, pj)
                    if not self._transition_allowed(previous, (layer, i, j)):
                        continue
                    edge = math.hypot(i - pi, j - pj) * self.grid_resolution_m
                    edge += (layer - previous_layer) * GLOBAL_POSE_EDGE_COST_M
                    yield previous_layer, pi, pj, edge

    def _build_global_reverse_shortest_path_tree(self) -> None:
        node_count = GLOBAL_POSE_LAYERS * self._grid_shape[0] * self._grid_shape[1]
        distance = np.full(node_count, np.inf, dtype=np.float64)
        successor = np.full(node_count, -1, dtype=np.int32)
        distance[self._global_goal_flat] = 0.0
        queue: list[Tuple[float, int]] = [(0.0, self._global_goal_flat)]
        while queue:
            cost, flat = heapq.heappop(queue)
            if cost != distance[flat]:
                continue
            layer, i, j = self._global_unflatten(flat)
            for pl, pi, pj, edge in self._iter_global_predecessors(layer, i, j):
                previous = self._global_flat(pl, pi, pj)
                candidate = cost + edge
                if candidate + 1.0e-15 < distance[previous]:
                    distance[previous] = candidate
                    successor[previous] = flat
                    heapq.heappush(queue, (candidate, previous))
        successor[self._global_goal_flat] = self._global_goal_flat
        self._global_reverse_distance = distance
        self._global_successor = successor

    def _nearest_global_cell(
        self,
        point_xy: np.ndarray,
        *,
        layer: int,
        require_visible: bool,
    ) -> Tuple[int, int] | None:
        if self._global_free is None:
            return None
        point = np.asarray(point_xy, dtype=np.float64).reshape(2)
        base_i = int(
            round((point[0] + self.workspace_radius_m) / self.grid_resolution_m)
        )
        base_j = int(
            round((point[1] + self.workspace_radius_m) / self.grid_resolution_m)
        )
        fraction = self._pose_fractions[layer]
        for radius in range(0, 9):
            candidates: list[Tuple[float, int, int]] = []
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if radius and max(abs(di), abs(dj)) != radius:
                        continue
                    i, j = base_i + di, base_j + dj
                    if not (
                        0 <= i < self._grid_shape[0]
                        and 0 <= j < self._grid_shape[1]
                        and self._global_free[layer, i, j]
                    ):
                        continue
                    grid = self._grid_xy(i, j)
                    if require_visible and not self._joint_segment_free(
                        point, fraction, grid, fraction
                    ):
                        continue
                    candidates.append((float(np.linalg.norm(grid - point)), i, j))
            if candidates:
                candidates.sort()
                return candidates[0][1], candidates[0][2]
        return None

    def _global_astar(self, start_ij: Tuple[int, int]) -> _GlobalCachedPlan:
        cached = self._global_plan_cache.get(start_ij)
        if cached is not None:
            return cached
        if self._global_reverse_distance is None:
            raise RuntimeError("global A* reverse heuristic is not configured")
        start = self._global_flat(0, *start_ij)
        heuristic = float(self._global_reverse_distance[start])
        if not math.isfinite(heuristic):
            raise AStarPlanError(
                f"global A* start grid cell {start_ij} is disconnected"
            )
        # Lazy A*: search the node-valid graph first, then certify only the
        # edges on the candidate shortest path.  A rejected swept edge is
        # retained in the global cache and the search restarts without it.
        # This is equivalent to ordinary A* on the exact edge graph, but avoids
        # thousands of MuJoCo calls for outgoing edges that no solution uses.
        expanded_total = 0
        restarts = 0
        while True:
            queue: list[Tuple[float, float, int, int]] = [
                (heuristic, heuristic, 0, start)
            ]
            g_score: Dict[int, float] = {start: 0.0}
            parent: Dict[int, int] = {}
            closed: set[int] = set()
            counter = 0
            while queue:
                _f, _h, _order, flat = heapq.heappop(queue)
                if flat in closed:
                    continue
                closed.add(flat)
                expanded_total += 1
                if flat == self._global_goal_flat:
                    break
                layer, i, j = self._global_unflatten(flat)
                for nl, ni, nj, edge in self._iter_global_neighbours(
                    layer, i, j, validate_edges=False
                ):
                    following = self._global_flat(nl, ni, nj)
                    if self._global_edge_cache.get((flat, following)) is False:
                        continue
                    if following in closed:
                        continue
                    candidate = g_score[flat] + edge
                    if candidate + 1.0e-15 >= g_score.get(following, math.inf):
                        continue
                    h = float(self._global_reverse_distance[following])
                    if not math.isfinite(h):
                        continue
                    g_score[following] = candidate
                    parent[following] = flat
                    counter += 1
                    heapq.heappush(
                        queue, (candidate + h, h, counter, following)
                    )
            if self._global_goal_flat not in closed:
                raise AStarPlanError(
                    f"global full-pose A* found no route from {start_ij}"
                )
            reverse = [self._global_goal_flat]
            while reverse[-1] != start:
                previous = parent.get(reverse[-1])
                if previous is None:
                    raise AStarPlanError("global A* parent chain is incomplete")
                reverse.append(previous)
            states = tuple(reversed(reverse))
            rejected = False
            for first, second in zip(states[:-1], states[1:]):
                if not self._node_edge_free(first, second):
                    rejected = True
                    break
            if not rejected:
                plan = _GlobalCachedPlan(states, expanded_total)
                self._global_plan_cache[start_ij] = plan
                self._lazy_search_restarts += restarts
                return plan
            restarts += 1

    def _shortcut_joint_path(
        self, points_xy: np.ndarray, progress: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        fractions = np.asarray(progress, dtype=np.float64).reshape(-1)
        keep = np.ones(len(points), dtype=np.bool_)
        if len(points) > 1:
            keep[1:] = (
                np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.0e-10
            ) | (np.abs(np.diff(fractions)) > 1.0e-10)
        points, fractions = points[keep], fractions[keep]
        selected = [0]
        index = 0
        while index < len(points) - 1:
            following = len(points) - 1
            while following > index + 1 and not self._joint_segment_free(
                points[index],
                fractions[index],
                points[following],
                fractions[following],
            ):
                following -= 1
            selected.append(following)
            index = following
        if len(selected) == 2:
            self._direct_shortcuts += 1
        ids = np.asarray(selected, dtype=np.int64)
        return points[ids], fractions[ids]

    def _dense_joint_polyline(
        self, points_xy: np.ndarray, progress: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dense_points: list[np.ndarray] = []
        dense_progress: list[np.ndarray] = []
        vertex_indices = [0]
        samples = 0
        for index in range(len(points_xy) - 1):
            points, fractions = self._sample_joint_segment(
                points_xy[index],
                float(progress[index]),
                points_xy[index + 1],
                float(progress[index + 1]),
            )
            if index:
                points, fractions = points[1:], fractions[1:]
            dense_points.append(points)
            dense_progress.append(fractions)
            samples += len(points)
            vertex_indices.append(samples - 1)
        return (
            np.concatenate(dense_points),
            np.concatenate(dense_progress),
            np.asarray(vertex_indices, dtype=np.int64),
        )

    def _resample_global_plan(
        self,
        continuous_start_xy: np.ndarray,
        continuous_goal_xy: np.ndarray,
        states: Sequence[int],
        *,
        line_of_sight_shortcut: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        grid_points = []
        grid_progress = []
        for flat in states:
            layer, i, j = self._global_unflatten(flat)
            grid_points.append(self._grid_xy(i, j))
            grid_progress.append(self._pose_fractions[layer])
        raw_points = np.concatenate(
            (
                np.asarray(continuous_start_xy, dtype=np.float64).reshape(1, 2),
                np.asarray(grid_points, dtype=np.float64).reshape(-1, 2),
                np.asarray(continuous_goal_xy, dtype=np.float64).reshape(1, 2),
            ),
            axis=0,
        )
        raw_progress = np.concatenate(
            ((0.0,), np.asarray(grid_progress, dtype=np.float64), (1.0,))
        )
        if line_of_sight_shortcut:
            points, progress = self._shortcut_joint_path(
                raw_points, raw_progress
            )
        else:
            # This path is used only when the shortcut, after conversion to
            # the actual float32 world-frame commands, fails the same exact
            # swept-contact oracle.  Retaining the original A* vertices is a
            # quantisation-safe post-processing fallback, not a different
            # planner or an expert/straight-line trajectory.
            points, progress = raw_points, raw_progress
        dense_xy, dense_progress, vertex_indices = self._dense_joint_polyline(
            points, progress
        )
        dense_tips, _dense_rotation = self._tips_and_rotations(
            dense_xy, dense_progress
        )
        translation = np.linalg.norm(np.diff(dense_tips, axis=0), axis=1)
        rotation = (
            np.abs(np.diff(dense_progress))
            * self._rotation_angle_rad
            * GLOBAL_ROTATION_PATH_WEIGHT_M
        )
        metric = translation + rotation
        cumulative = np.concatenate(((0.0,), np.cumsum(metric)))
        total = float(cumulative[-1])
        if not math.isfinite(total) or total <= 1.0e-10:
            raise AStarPlanError("global A* path has zero combined length")
        target = _minimum_jerk(
            np.arange(1, DIRECT_ALIGN_STEPS + 1, dtype=np.float64)
            / float(DIRECT_ALIGN_STEPS)
        ) * total
        # Retain every certified SE(3) line-of-sight waypoint.  Replacing the
        # nearest non-reserved minimum-jerk sample prevents a time-resampled
        # chord from cutting across a safe polyline corner.
        mandatory = cumulative[vertex_indices[1:-1]]
        if len(mandatory) >= DIRECT_ALIGN_STEPS:
            raise AStarPlanError(
                "global A* shortcut has too many corners for ALIGN schedule"
            )
        reserved: set[int] = set()
        for distance in mandatory:
            candidates = [
                index
                for index in range(DIRECT_ALIGN_STEPS - 1)
                if index not in reserved
            ]
            selected = min(
                candidates, key=lambda index: abs(float(target[index] - distance))
            )
            target[selected] = distance
            reserved.add(selected)
        target.sort()
        target[-1] = total
        sampled_xy = np.column_stack(
            (
                np.interp(target, cumulative, dense_xy[:, 0]),
                np.interp(target, cumulative, dense_xy[:, 1]),
            )
        )
        sampled_progress = np.interp(target, cumulative, dense_progress)
        sampled_xy[-1] = continuous_goal_xy
        sampled_progress[-1] = 1.0

        path_xy = np.concatenate(
            (
                np.asarray(continuous_start_xy, dtype=np.float64).reshape(1, 2),
                sampled_xy,
            ),
            axis=0,
        )
        path_length = float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum())
        return sampled_xy, sampled_progress, path_length

    def _configure_global_collision(
        self,
        mask,
        start_command,
        goal_command,
        socket_mouth,
    ) -> None:
        torch = __import__("torch")
        mask_value = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        ids = torch.nonzero(mask_value, as_tuple=False).flatten()
        count = int(ids.numel())
        if not count:
            return
        start_tip = self._masked_rows(
            start_command.tip, ids, count, "start tip"
        ).detach().to("cpu", torch.float64).numpy()
        goal_tip = self._masked_rows(
            goal_command.tip, ids, count, "goal tip"
        ).detach().to("cpu", torch.float64).numpy()
        start_rotation = self._masked_rows(
            start_command.rotation, ids, count, "start rotation"
        ).detach().to("cpu", torch.float64).numpy()
        goal_rotation = self._masked_rows(
            goal_command.rotation, ids, count, "goal rotation"
        ).detach().to("cpu", torch.float64).numpy()
        mouth = self._masked_rows(
            socket_mouth, ids, count, "socket mouth"
        ).detach().to("cpu", torch.float64).numpy()

        canonical_goal = []
        canonical_start_rotation = []
        canonical_goal_rotation = []
        for local in range(count):
            inward, tangent = self._canonical_basis(mouth[local, :2])
            canonical_goal.append(
                (
                    float(np.dot(goal_tip[local, :2], inward)),
                    float(np.dot(goal_tip[local, :2], tangent)),
                )
            )
            canonical_start_rotation.append(
                self._canonical_rotation(start_rotation[local], inward, tangent)
            )
            canonical_goal_rotation.append(
                self._canonical_rotation(goal_rotation[local], inward, tangent)
            )
        canonical_goal = np.asarray(canonical_goal, dtype=np.float64)
        start_rotation_c = np.asarray(canonical_start_rotation, dtype=np.float64)
        goal_rotation_c = np.asarray(canonical_goal_rotation, dtype=np.float64)
        if float(np.max(np.abs(canonical_goal - canonical_goal[0]))) > 2.0e-5:
            raise AStarPlanError("global A* requires one canonical goal XY pose")
        if float(np.max(np.abs(start_tip[:, 2] - start_tip[0, 2]))) > 2.0e-5:
            raise AStarPlanError("global A* requires one handoff tip height")
        if float(np.max(np.abs(goal_tip[:, 2] - goal_tip[0, 2]))) > 2.0e-5:
            raise AStarPlanError("global A* requires one preinsert tip height")
        if float(
            np.max(np.abs(start_rotation_c[:, :, 2] - start_rotation_c[0, :, 2]))
        ) > 2.0e-5 or float(
            np.max(np.abs(goal_rotation_c[:, :, 2] - goal_rotation_c[0, :, 2]))
        ) > 2.0e-5:
            raise AStarPlanError(
                "global A* requires shared canonical screw-axis endpoints"
            )

        supervisor = BatchedSupervisor(device="cpu", curriculum=None)
        angle = torch.tensor([45.0], dtype=torch.float32)
        canonical_mouth = torch.tensor(
            [[self.task.socket_mouth_orbit_radius_m, 0.0]], dtype=torch.float32
        )
        canonical_spawn = torch.tensor(
            [[-self.task.spawn_radius_m, 0.0]], dtype=torch.float32
        )
        representative_start = supervisor.at(
            torch.tensor([SCRIPTED_PREFIX_STEPS]),
            canonical_spawn,
            angle,
            canonical_mouth,
        ).rotation[0].to(torch.float64).numpy()
        representative_goal = supervisor.at(
            torch.tensor([PREINSERT_GATE_STEP]),
            canonical_spawn,
            angle,
            canonical_mouth,
        ).rotation[0].to(torch.float64).numpy()
        if float(
            np.max(
                np.abs(
                    start_rotation_c[:, :, 2]
                    - representative_start[None, :, 2]
                )
            )
        ) > 3.0e-5 or float(
            np.max(
                np.abs(
                    goal_rotation_c[:, :, 2]
                    - representative_goal[None, :, 2]
                )
            )
        ) > 3.0e-5:
            raise AStarPlanError(
                "world endpoint screw axes differ from the canonical contract"
            )
        signature = (
            float(start_tip[0, 2]),
            float(goal_tip[0, 2]),
            float(canonical_goal[0, 0]),
            float(canonical_goal[0, 1]),
            *representative_start[:, 2].tolist(),
            *representative_goal[:, 2].tolist(),
        )
        if self._exact_collision is not None:
            if self._collision_signature is None or not np.allclose(
                signature,
                self._collision_signature,
                atol=2.0e-6,
                rtol=0.0,
            ):
                raise AStarPlanError(
                    "reset pose contract differs from the cached collision lattice"
                )
            return
        socket_position, socket_quaternion = supervisor.socket_mocap_pose(
            angle, canonical_mouth
        )
        xml_path = Path(__file__).resolve().parents[1] / "assets" / "Project_hpc.xml"
        exact = _ExactMuJoCoTransportCollision(
            xml_path=xml_path,
            screw_tip_z_m=float(start_tip[0, 2]),
            screw_rotation=representative_start,
            socket_position=socket_position[0].numpy(),
            socket_quaternion_wxyz=socket_quaternion[0].numpy(),
        )
        self._exact_collision = exact
        self._collision_signature = signature
        self._exact_collision_provenance = exact.provenance()
        self._canonical_goal_xy = canonical_goal[0]
        self._pose_tip_z = np.linspace(
            float(start_tip[0, 2]),
            float(goal_tip[0, 2]),
            GLOBAL_POSE_LAYERS,
            dtype=np.float64,
        )
        q0 = matrix_to_quaternion(
            torch.as_tensor(representative_start, dtype=torch.float64).reshape(1, 3, 3)
        )[0]
        q1 = matrix_to_quaternion(
            torch.as_tensor(representative_goal, dtype=torch.float64).reshape(1, 3, 3)
        )[0]
        self._start_quaternion = q0.numpy()
        self._goal_quaternion = q1.numpy()
        dot = abs(float(np.dot(self._start_quaternion, self._goal_quaternion)))
        self._rotation_angle_rad = 2.0 * math.acos(float(np.clip(dot, 0.0, 1.0)))
        self._pose_rotations = self._rotations_at(self._pose_fractions)

        grid_x, grid_y = np.meshgrid(
            self._grid_values, self._grid_values, indexing="ij"
        )
        in_disc = grid_x * grid_x + grid_y * grid_y <= (
            self.workspace_radius_m * self.workspace_radius_m + 1.0e-12
        )
        grid_points = np.column_stack((grid_x[in_disc], grid_y[in_disc]))
        free = np.zeros(
            (GLOBAL_POSE_LAYERS, *in_disc.shape), dtype=np.bool_
        )
        colliding = 0
        for layer in range(GLOBAL_POSE_LAYERS):
            tips = np.column_stack(
                (
                    grid_points,
                    np.full(len(grid_points), self._pose_tip_z[layer]),
                )
            )
            collision = exact.collides_poses(tips, self._pose_rotations[layer])
            free[layer, in_disc] = ~collision
            colliding += int(collision.sum())
        self._global_free = free
        self._free = free[0]
        self._grid_shape = in_disc.shape
        self._exact_colliding_pose_nodes = colliding
        self._exact_free_pose_nodes = int(free.sum())
        goal = self._nearest_global_cell(
            self._canonical_goal_xy,
            layer=GLOBAL_POSE_LAYERS - 1,
            require_visible=True,
        )
        if goal is None:
            raise AStarPlanError("exact pre-insertion goal has no visible free cell")
        self._global_goal_ij = goal
        self._global_goal_flat = self._global_flat(
            GLOBAL_POSE_LAYERS - 1, *goal
        )
        self._build_global_reverse_shortest_path_tree()
        self._global_plan_cache.clear()
        self._global_edge_cache.clear()
        # Base diagnostics count this cache; make it the authoritative cache.
        self._grid_plan_cache = self._global_plan_cache

    def reset(
        self,
        mask,
        *,
        start_command,
        goal_command,
        socket_mouth,
        socket_axis,
    ) -> None:
        del socket_axis
        torch = __import__("torch")
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        count = int(ids.numel())
        if not count:
            return
        self._configure_global_collision(
            mask, start_command, goal_command, socket_mouth
        )
        start_tip = self._masked_rows(
            start_command.tip, ids, count, "start tip"
        ).detach().to("cpu", torch.float64).numpy()
        goal_tip = self._masked_rows(
            goal_command.tip, ids, count, "goal tip"
        ).detach().to("cpu", torch.float64).numpy()
        start_rotation = self._masked_rows(
            start_command.rotation, ids, count, "start rotation"
        )
        goal_rotation = self._masked_rows(
            goal_command.rotation, ids, count, "goal rotation"
        )
        start_rotation_cpu = start_rotation.detach().to(
            "cpu", torch.float64
        ).numpy()
        goal_rotation_cpu = goal_rotation.detach().to(
            "cpu", torch.float64
        ).numpy()
        mouth = self._masked_rows(
            socket_mouth, ids, count, "socket mouth"
        ).detach().to("cpu", torch.float64).numpy()

        plan_tip = np.empty((count, DIRECT_ALIGN_STEPS, 3), dtype=np.float32)
        plan_rotation = np.empty(
            (count, DIRECT_ALIGN_STEPS, 3, 3), dtype=np.float32
        )
        plan_progress = np.empty((count, DIRECT_ALIGN_STEPS), dtype=np.float32)
        path_length = np.empty(count, dtype=np.float32)
        latency = np.empty(count, dtype=np.float32)
        expanded = np.empty(count, dtype=np.int64)
        start_flat = np.empty(count, dtype=np.int64)
        cache_hit = np.zeros(count, dtype=np.bool_)
        translation_limit_fraction = np.empty(count, dtype=np.float32)
        rotation_limit_fraction = np.empty(count, dtype=np.float32)
        try:
            for local in range(count):
                began = time.perf_counter()
                inward, tangent = self._canonical_basis(mouth[local, :2])
                canonical_start = np.asarray(
                    (
                        float(np.dot(start_tip[local, :2], inward)),
                        float(np.dot(start_tip[local, :2], tangent)),
                    ),
                    dtype=np.float64,
                )
                canonical_goal = np.asarray(
                    (
                        float(np.dot(goal_tip[local, :2], inward)),
                        float(np.dot(goal_tip[local, :2], tangent)),
                    ),
                    dtype=np.float64,
                )
                start_ij = self._nearest_global_cell(
                    canonical_start, layer=0, require_visible=True
                )
                if start_ij is None:
                    raise AStarPlanError(
                        f"continuous global A* start {canonical_start.tolist()} "
                        "has no visible free cell"
                    )
                cache_hit[local] = start_ij in self._global_plan_cache
                discrete = self._global_astar(start_ij)
                direct_shortcuts_before = self._direct_shortcuts
                for use_shortcut in (True, False):
                    sampled_xy, sampled_progress, length = (
                        self._resample_global_plan(
                            canonical_start,
                            canonical_goal,
                            discrete.states,
                            line_of_sight_shortcut=use_shortcut,
                        )
                    )
                    world_xy = (
                        sampled_xy[:, :1] * inward[None, :]
                        + sampled_xy[:, 1:] * tangent[None, :]
                    )
                    z = start_tip[local, 2] + sampled_progress * (
                        goal_tip[local, 2] - start_tip[local, 2]
                    )
                    candidate_tip = np.empty(
                        (DIRECT_ALIGN_STEPS, 3), dtype=np.float32
                    )
                    candidate_tip[:, :2] = world_xy.astype(np.float32)
                    candidate_tip[:, 2] = z.astype(np.float32)
                    candidate_rotation = self._world_rotations_at(
                        sampled_progress,
                        inward=inward,
                        tangent=tangent,
                        start_rotation=start_rotation_cpu[local],
                        goal_rotation=goal_rotation_cpu[local],
                    ).astype(np.float32)
                    yaw_frame = self._yaw_frame(inward, tangent)
                    actual_tips = np.concatenate(
                        (
                            start_tip[local]
                            .astype(np.float32)
                            .reshape(1, 3),
                            candidate_tip,
                        ),
                        axis=0,
                    ).astype(np.float64)
                    actual_rotations = np.concatenate(
                        (
                            start_rotation_cpu[local]
                            .astype(np.float32)
                            .reshape(1, 3, 3),
                            candidate_rotation,
                        ),
                        axis=0,
                    ).astype(np.float64)
                    canonical_tips = actual_tips.copy()
                    canonical_tips[:, 0] = actual_tips[:, :2] @ inward
                    canonical_tips[:, 1] = actual_tips[:, :2] @ tangent
                    canonical_rotations = np.einsum(
                        "ji,njk->nik", yaw_frame, actual_rotations
                    )
                    if self._actual_command_sweep_free(
                        canonical_tips, canonical_rotations
                    ):
                        plan_tip[local] = candidate_tip
                        plan_rotation[local] = candidate_rotation
                        plan_progress[local] = sampled_progress.astype(
                            np.float32
                        )
                        break
                    if use_shortcut:
                        # Do not report a direct shortcut that was rejected by
                        # the certification applied to the commands that will
                        # actually be executed.
                        self._direct_shortcuts = direct_shortcuts_before
                        self._float32_shortcut_fallbacks += 1
                        continue
                    raise AStarPlanError(
                        "actual float32 world-frame A* lattice-polyline "
                        "command sweep collides"
                    )
                elevation = math.radians(45.0)
                canonical_socket_frame = np.asarray(
                    (
                        (-math.sin(elevation), 0.0, math.cos(elevation)),
                        (0.0, 1.0, 0.0),
                        (-math.cos(elevation), 0.0, -math.sin(elevation)),
                    ),
                    dtype=np.float64,
                )
                socket_frame = yaw_frame @ canonical_socket_frame
                translation_fraction, rotation_fraction = (
                    self._task_increment_fractions(
                        actual_tips, actual_rotations, socket_frame
                    )
                )
                if translation_fraction > 1.0002 or rotation_fraction > 1.0002:
                    raise AStarPlanError(
                        "global A* command exceeds the shared task-space "
                        f"increment limits: translation={translation_fraction:.6f}, "
                        f"rotation={rotation_fraction:.6f}"
                    )
                translation_limit_fraction[local] = translation_fraction
                rotation_limit_fraction[local] = rotation_fraction
                path_length[local] = length
                expanded[local] = discrete.expanded
                start_flat[local] = self._flatten(*start_ij)
                latency[local] = float(time.perf_counter() - began)
        except BaseException as exc:
            self._plan_found.index_fill_(0, ids, False)
            if isinstance(exc, AStarPlanError):
                raise
            raise AStarPlanError(f"global full-ALIGN A* failed: {exc}") from exc

        tip_tensor = torch.as_tensor(
            plan_tip, device=self.device, dtype=self.dtype
        )
        self._plan_tip.index_copy_(0, ids, tip_tensor)
        self._plan_rotation.index_copy_(
            0,
            ids,
            torch.as_tensor(
                plan_rotation, device=self.device, dtype=self.dtype
            ),
        )
        self._plan_found.index_fill_(0, ids, True)
        self._spline_fallback.index_fill_(0, ids, False)
        self._cache_hit.index_copy_(
            0, ids, torch.as_tensor(cache_hit, device=self.device)
        )
        self._planning_latency_s.index_copy_(
            0, ids, torch.as_tensor(latency, device=self.device)
        )
        self._path_length_m.index_copy_(
            0, ids, torch.as_tensor(path_length, device=self.device)
        )
        self._minimum_obstacle_clearance_m.index_fill_(0, ids, 0.0)
        self._astar_expanded.index_copy_(
            0, ids, torch.as_tensor(expanded, device=self.device)
        )
        self._start_grid_flat.index_copy_(
            0, ids, torch.as_tensor(start_flat, device=self.device)
        )
        self._task_translation_limit_fraction.index_copy_(
            0,
            ids,
            torch.as_tensor(
                translation_limit_fraction,
                device=self.device,
                dtype=self.dtype,
            ),
        )
        self._task_rotation_limit_fraction.index_copy_(
            0,
            ids,
            torch.as_tensor(
                rotation_limit_fraction,
                device=self.device,
                dtype=self.dtype,
            ),
        )
        self._cache_hits += int(cache_hit.sum())
        self._cache_misses += int((~cache_hit).sum())
        self._total_reset_plans += count
        self._total_planning_latency_s += float(np.sum(latency, dtype=np.float64))

    def diagnostics(self) -> Dict[str, object]:
        result = super().diagnostics()
        for obsolete in (
            "staging_standoff_m",
            "transport_steps",
            "rotate_raise_steps",
            "dock_steps",
        ):
            result.pop(obsolete, None)
        result.update(
            {
                "planner_contract": GLOBAL_PLANNER_CONTRACT,
                "collision_model": COLLISION_MODEL,
                "pose_layers": GLOBAL_POSE_LAYERS,
                "align_steps": DIRECT_ALIGN_STEPS,
                "staging_waypoint": False,
                "exact_colliding_pose_nodes": self._exact_colliding_pose_nodes,
                "exact_free_pose_nodes": self._exact_free_pose_nodes,
                "direct_shortcuts": self._direct_shortcuts,
                "float32_shortcut_fallbacks": (
                    self._float32_shortcut_fallbacks
                ),
                "lazy_search_restarts": self._lazy_search_restarts,
                "lazy_edge_evaluations": self._lazy_edge_evaluations,
                "full_pose_swept_collision": True,
                "task_translation_limit_fraction": (
                    self._task_translation_limit_fraction.detach().clone()
                ),
                "task_rotation_limit_fraction": (
                    self._task_rotation_limit_fraction.detach().clone()
                ),
            }
        )
        return result

    def provenance(self) -> Dict[str, object]:
        obstacle = (
            {
                "model": COLLISION_MODEL,
                "status": "configured_on_first_reset",
                "clearance_inflation_m": 0.0,
            }
            if self._exact_collision_provenance is None
            else {
                **self._exact_collision_provenance,
                "colliding_pose_nodes": self._exact_colliding_pose_nodes,
                "free_pose_nodes": self._exact_free_pose_nodes,
            }
        )
        return {
            "schema": "screw-global-full-align-astar-planner-v1",
            "planner_contract": GLOBAL_PLANNER_CONTRACT,
            "algorithm": "global-reduced-pose-lattice-astar",
            "workspace_radius_m": float(self.workspace_radius_m),
            "socket": {
                "table_angle_deg": 45.0,
                "mouth_orbit_radius_m": float(
                    self.task.socket_mouth_orbit_radius_m
                ),
            },
            "grid": {
                "resolution_m": float(self.grid_resolution_m),
                "connectivity": 8,
                "pose_layers": GLOBAL_POSE_LAYERS,
                "state": "canonical_xy_plus_monotonic_pose_progress",
                "diagonal_corner_cutting": False,
            },
            "obstacle": obstacle,
            "search": {
                "algorithm": "Lazy A* (exact A* edge graph)",
                "heuristic": "node-valid reverse shortest-path lower bound",
                "edge_cost": (
                    "XY translation metres + 1e-6 metres per monotonic "
                    "pose-progress layer"
                ),
                "pose_layer_cost_m": GLOBAL_POSE_EDGE_COST_M,
                "edge_validation": (
                    "lazy full-pose filtered MuJoCo sweep at <=0.25 mm and "
                    "<=0.25 degree spacing"
                ),
                "cache_key": "canonical start grid cell",
                "plans_per_episode": 1,
                "replanning": False,
            },
            "postprocess": {
                "se3_line_of_sight_shortcut": True,
                "cubic_spline": False,
                "fallback": (
                    "only when the LOS result fails actual-float32 full-pose "
                    "sweep certification, resample the same certified A* "
                    "lattice polyline; no fallback planner"
                ),
                "float32_shortcut_fallback_enabled": True,
                "fallback_planner": False,
                "time_scaling": (
                    "minimum-jerk targets over combined translation/rotation "
                    "arc with certified waypoint retention"
                ),
                "rotation_arc_weight_m_per_rad": (
                    GLOBAL_ROTATION_PATH_WEIGHT_M
                ),
                "waypoint_preserving_resample": True,
                "final_command_chord_validation": True,
            },
            "command_limits": {
                "translation_socket_frame_m": [0.002, 0.002, 0.003],
                "rotation_socket_frame_rad": [
                    math.radians(2.0),
                    math.radians(2.0),
                    math.radians(4.0),
                ],
                "validated_on_actual_float32_commands": True,
            },
            "schedule": {
                "align_steps": DIRECT_ALIGN_STEPS,
                "global_full_align": True,
                "staging_waypoint": False,
                "fixed_transport_rotate_dock_segments": False,
                "start": "nominal supervisor handoff pose",
                "goal": "exact supervisor pre-insertion pose",
            },
        }

    def close(self) -> None:
        super().close()
        self._global_plan_cache.clear()
        self._global_edge_cache.clear()
        self._exact_collision = None


__all__ = [
    "COLLISION_MODEL",
    "CONTACT_TOLERANCE_M",
    "ExactCollisionBatchedAStarSplinePlanner",
    "GLOBAL_PLANNER_CONTRACT",
    "GLOBAL_POSE_LAYERS",
]
