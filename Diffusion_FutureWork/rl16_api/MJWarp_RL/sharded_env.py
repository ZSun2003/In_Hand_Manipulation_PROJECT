"""Object-major facade over four independent single-object MJWarp worlds."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

from . import contract
from .env import (
    MJWarpIKResidualEnv,
    OBSERVATION_DIM,
    StepBatch,
    _construct_fail_closed,
)
from .hpc_config import (
    DEFAULT_CONFIG,
    FORMAL_OBJECT_WORLD_COUNTS,
    HPCConfig,
    balanced_object_world_counts,
    object_world_slices,
)
from .newton_regularization import (
    formal_newton_regularization_metadata,
    formal_newton_regularization_metadata_matches,
)


def _sha256_like(value: Any) -> str | None:
    text = str(value) if value is not None else ""
    if len(text) == 64 and all(c in "0123456789abcdef" for c in text):
        return text
    return None


def ordered_shard_manifest(
    platform_spec: Any,
    world_counts: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """Return a path-stable ordered audit description of all active shards."""

    shards = tuple(getattr(platform_spec, "shards", ()))
    if len(shards) != contract.OBJECT_SHARD_COUNT:
        raise ValueError("platform aggregate must contain four object shards")
    if world_counts is not None and len(world_counts) != len(shards):
        raise ValueError("world allocation and shard count disagree")
    result: list[dict[str, Any]] = []
    for expected_index, shard in enumerate(shards):
        actual_index = int(getattr(shard, "fixed_object_index", -1))
        if actual_index != expected_index:
            raise ValueError(
                "platform shards must follow contract.OBJECTS order"
            )
        model = shard.host_model
        xml = getattr(shard, "model_xml", None)
        xml_sha = _sha256_like(
            getattr(shard, "model_xml_sha256", None)
        ) or _sha256_like(getattr(shard, "rl_model_fingerprint_sha256", None))
        if xml_sha is None:
            raise ValueError(
                f"{contract.OBJECTS[expected_index]} shard lacks an XML hash"
            )
        task = contract.validate_task(str(getattr(shard, "task", "")))
        task_index = int(getattr(shard, "task_index", -1))
        root_height_m = float(getattr(shard, "root_height_m", float("nan")))
        if task_index != contract.TASKS.index(task):
            raise ValueError("platform shard task index is inconsistent")
        if not math.isfinite(root_height_m):
            raise ValueError("platform shard root height is not finite")
        if task != getattr(platform_spec, "task", None) or not math.isclose(
            root_height_m,
            float(getattr(platform_spec, "root_height_m", float("nan"))),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("platform shard task/root bundle is inconsistent")
        entry: dict[str, Any] = {
            "object_index": expected_index,
            "object": contract.OBJECTS[expected_index],
            "task": task,
            "task_index": task_index,
            "root_height_m": root_height_m,
            "model_xml": str(Path(xml).resolve()) if xml is not None else None,
            "model_xml_sha256": xml_sha,
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
        }
        if world_counts is not None:
            entry["worlds"] = int(world_counts[expected_index])
        result.append(entry)
    return result


def formal_sharded_health_is_valid(
    health: Mapping[str, Any],
    *,
    expected_world_counts: tuple[int, ...] | None = None,
) -> bool:
    """Validate every object-local Newton solve, never only the aggregate."""

    by_object = health.get("by_object")
    if not isinstance(by_object, Mapping) or set(by_object) != set(
        contract.OBJECTS
    ):
        return False
    if int(health.get("shard_count", -1)) != contract.OBJECT_SHARD_COUNT:
        return False
    if expected_world_counts is not None:
        expected = {
            name: int(count)
            for name, count in zip(contract.OBJECTS, expected_world_counts)
        }
        if health.get("world_allocation") != expected:
            return False
        if health.get("worlds") != sum(expected_world_counts):
            return False
    sticky = (
        "sticky_nonfinite",
        "sticky_overflow",
        "sticky_contacts_full",
        "sticky_constraints_full",
        "sticky_external_nonfinite",
    )
    regularization_metadata = formal_newton_regularization_metadata()
    if not formal_newton_regularization_metadata_matches(health):
        return False
    aggregate_nv = health.get("model_nv")
    if not (
        health.get("solver_option") == 2
        and health.get("solver_name") == contract.FORMAL_SOLVER
        and health.get("requested_solver_tolerance")
        == contract.FORMAL_XML_SOLVER_TOLERANCE
        and health.get("effective_solver_tolerance")
        == contract.FORMAL_EFFECTIVE_SOLVER_TOLERANCE
        and health.get("jacobian_option") == 0
        and health.get("resolved_sparse_jacobian") is False
        and isinstance(aggregate_nv, int)
        and not isinstance(aggregate_nv, bool)
        and 0 < aggregate_nv <= contract.FORMAL_DENSE_NEWTON_MAX_NV
        and aggregate_nv
        <= regularization_metadata["newton_regularization_max_nv"]
    ):
        return False
    for object_name in contract.OBJECTS:
        shard = by_object.get(object_name)
        if not isinstance(shard, Mapping):
            return False
        if not formal_newton_regularization_metadata_matches(shard):
            return False
        if any(
            type(shard.get(name)) is not type(health.get(name))
            or shard.get(name) != health.get(name)
            for name in regularization_metadata
        ):
            return False
        nv = shard.get("model_nv")
        if not (
            bool(shard.get("finite", False))
            and shard.get("solver_option") == 2
            and shard.get("solver_name") == contract.FORMAL_SOLVER
            and shard.get("requested_solver_tolerance")
            == contract.FORMAL_XML_SOLVER_TOLERANCE
            and shard.get("effective_solver_tolerance")
            == contract.FORMAL_EFFECTIVE_SOLVER_TOLERANCE
            and shard.get("jacobian_option") == 0
            and shard.get("resolved_sparse_jacobian") is False
            and isinstance(nv, int)
            and not isinstance(nv, bool)
            and 0 < nv <= contract.FORMAL_DENSE_NEWTON_MAX_NV
            and nv
            <= regularization_metadata["newton_regularization_max_nv"]
            and not any(bool(shard.get(name, False)) for name in sticky)
        ):
            return False
    return bool(health.get("finite", False)) and not any(
        bool(health.get(name, False)) for name in sticky
    )


class ShardedBackendFacade:
    """Read-only global facade for four independently owned MJWarp backends."""

    def __init__(self, owner: "ShardedMJWarpEnv") -> None:
        self.owner = owner
        self.worlds = owner.worlds
        self.device = owner.device
        self.torch = owner.torch
        self.control_dt = float(owner.shards[0].backend.control_dt)
        if any(
            float(shard.backend.control_dt) != self.control_dt
            for shard in owner.shards[1:]
        ):
            raise ValueError("object shards disagree on the control period")
        self._tensor_buffers: dict[str, Any] = {}

    @property
    def _graph_ready(self) -> bool:
        return all(bool(shard.backend._graph_ready) for shard in self.owner.shards)

    def prepare_cuda_graph(self) -> None:
        # Fixed-order launches are intentionally serialized by each backend's
        # existing caller-stream fence.  This is deterministic and minimizes
        # transient overlap; concurrent graph launch requires a new capacity
        # certificate and is outside the formal v5 contract.
        for shard in self.owner.shards:
            shard.backend.prepare_cuda_graph()
        if not self._graph_ready:
            raise RuntimeError("not every object shard captured its CUDA graph")

    def has_field(self, name: str) -> bool:
        return all(shard.backend.has_field(name) for shard in self.owner.shards)

    def _merge_tensors(self, name: str, values: list[Any]) -> Any:
        if len(values) != len(self.owner.world_slices):
            raise RuntimeError("object-shard tensor count changed")
        first = values[0]
        if first.ndim < 1:
            if not all(bool((value == first).all()) for value in values[1:]):
                raise ValueError(f"scalar backend field {name!r} differs by shard")
            return first
        if any(
            getattr(value, "ndim", 0) < 1
            or int(value.shape[0]) != shard.worlds
            for shard, value in zip(self.owner.shards, values)
        ):
            raise ValueError(
                f"backend field {name!r} is not world-major in every shard"
            )
        tail = tuple(first.shape[1:])
        if any(tuple(value.shape[1:]) != tail for value in values):
            raise ValueError(f"backend field {name!r} has heterogeneous widths")
        shape = (self.worlds, *tail)
        buffer = self._tensor_buffers.get(name)
        if (
            buffer is None
            or tuple(buffer.shape) != shape
            or buffer.dtype != first.dtype
            or buffer.device != first.device
        ):
            buffer = self.torch.empty(
                shape, device=first.device, dtype=first.dtype
            )
            self._tensor_buffers[name] = buffer
        for world_slice, value in zip(self.owner.world_slices, values):
            buffer[world_slice].copy_(value)
        return buffer

    def tensor(self, name: str) -> Any:
        if not self.has_field(name):
            raise AttributeError(f"not every shard exposes backend field {name!r}")
        return self._merge_tensors(
            f"backend.{name}",
            [shard.backend.tensor(name) for shard in self.owner.shards],
        )

    def health(self) -> dict[str, Any]:
        by_object = {
            object_name: shard.backend.health()
            for object_name, shard in zip(contract.OBJECTS, self.owner.shards)
        }
        values = list(by_object.values())

        def uniform(name: str, invalid: Any = None) -> Any:
            observed = [value.get(name) for value in values]
            return observed[0] if all(v == observed[0] for v in observed) else invalid

        sticky_names = (
            "sticky_nonfinite",
            "sticky_overflow",
            "sticky_contacts_full",
            "sticky_constraints_full",
            "sticky_external_nonfinite",
        )
        regularization_names = tuple(
            formal_newton_regularization_metadata()
        )
        result: dict[str, Any] = {
            "finite": all(bool(value.get("finite", False)) for value in values),
            "shard_count": len(values),
            "worlds": self.worlds,
            "world_allocation": {
                name: count
                for name, count in zip(contract.OBJECTS, self.owner.world_counts)
            },
            "solver_option": uniform("solver_option", -1),
            "solver_name": uniform("solver_name", "mixed"),
            "requested_solver_tolerance": uniform(
                "requested_solver_tolerance"
            ),
            "effective_solver_tolerance": uniform(
                "effective_solver_tolerance"
            ),
            "jacobian_option": uniform("jacobian_option", -1),
            "resolved_sparse_jacobian": any(
                value.get("resolved_sparse_jacobian") is not False
                for value in values
            ),
            "model_nv": max(int(value.get("model_nv", 0)) for value in values),
            "model_nv_by_object": {
                name: int(value.get("model_nv", 0))
                for name, value in by_object.items()
            },
            "dense_newton_max_nv": contract.FORMAL_DENSE_NEWTON_MAX_NV,
            "parent_model_fingerprint_sha256": str(
                self.owner.platform.parent_model_fingerprint_sha256
            ),
            "parent_controller_fingerprint_sha256": str(
                self.owner.platform.parent_controller_fingerprint_sha256
            ),
            "platform_adapter_fingerprint_sha256": str(
                self.owner.platform.platform_adapter_fingerprint_sha256
            ),
            **{
                name: uniform(name)
                for name in regularization_names
            },
            "by_object": by_object,
        }
        for name in sticky_names:
            result[name] = int(any(bool(value.get(name, False)) for value in values))
        for name in ("contacts_total", "contact_capacity_total"):
            if all(name in value for value in values):
                result[name] = sum(int(value[name]) for value in values)
        for name in ("max_nefc", "constraint_capacity"):
            if all(name in value for value in values):
                result[name] = max(int(value[name]) for value in values)
        return result

    def state_dict(self) -> dict[str, Any]:
        states = [shard.backend.state_dict() for shard in self.owner.shards]
        common = set(states[0])
        for state in states[1:]:
            common &= set(state)
        return {
            name: self._merge_tensors(
                f"state.{name}", [state[name] for state in states]
            ).detach().clone()
            for name in sorted(common)
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for shard, world_slice in zip(self.owner.shards, self.owner.world_slices):
            shard.backend.load_state_dict(
                {
                    name: value[world_slice]
                    for name, value in state.items()
                }
            )

    def close(self) -> None:
        self.owner.close()


class ShardedMJWarpEnv:
    """One logical vector batch backed by four object-local models."""

    def __init__(
        self,
        *,
        curriculum_phase: str,
        config: HPCConfig = DEFAULT_CONFIG,
        residual_enabled: bool = True,
        platform_spec: Any,
        worlds: int | None = None,
        capacity_num_envs: int | None = None,
        object_world_counts: tuple[int, ...] | None = None,
        device: str = "cuda:0",
        require_linux: bool = True,
        require_cuda: bool = True,
        validate_official_config: bool = True,
        child_env_type: Any = MJWarpIKResidualEnv,
        backend: Any | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        if backend is not None:
            raise ValueError("a sharded environment cannot accept one backend")
        if kwargs.get("fixed_object") is not None:
            raise ValueError("global sharded environment owns all four objects")
        kwargs.pop("fixed_object", None)
        requested_fixed_task = kwargs.pop("fixed_task", None)
        if (
            requested_fixed_task is not None
            and int(requested_fixed_task) != config.task_index
        ):
            raise ValueError("fixed_task disagrees with config.task")
        shards = tuple(getattr(platform_spec, "shards", ()))
        if len(shards) != contract.OBJECT_SHARD_COUNT:
            raise ValueError("platform aggregate must contain four object shards")
        for index, shard in enumerate(shards):
            if int(getattr(shard, "fixed_object_index", -1)) != index:
                raise ValueError("object shards are not in contract order")
        if getattr(platform_spec, "task", None) != config.task:
            raise ValueError("platform task bundle disagrees with config.task")
        if any(getattr(shard, "task", None) != config.task for shard in shards):
            raise ValueError("every object shard must own config.task")
        expected_root_height_m = float(
            getattr(platform_spec, "root_height_m", float("nan"))
        )
        if not math.isfinite(expected_root_height_m) or any(
            not math.isclose(
                float(getattr(shard, "root_height_m", float("nan"))),
                expected_root_height_m,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for shard in shards
        ):
            raise ValueError("every object shard must own one finite root height")
        requested_worlds = int(worlds or capacity_num_envs or config.num_envs)
        population = int(config.elite.population)
        group_aligned = (
            requested_worlds % population == 0
            and requested_worlds // population >= contract.OBJECT_SHARD_COUNT
        )
        allocation_population = population if group_aligned else 1
        if object_world_counts is None:
            self.world_counts = balanced_object_world_counts(
                requested_worlds,
                population=allocation_population,
                object_count=contract.OBJECT_SHARD_COUNT,
            )
        else:
            self.world_counts = tuple(
                int(value) for value in object_world_counts
            )
            if (
                len(self.world_counts) != contract.OBJECT_SHARD_COUNT
                or sum(self.world_counts) != requested_worlds
                or any(value <= 0 for value in self.world_counts)
            ):
                raise ValueError(
                    "explicit object_world_counts must be four positive "
                    "counts summing to global worlds"
                )
            if group_aligned and any(
                value % population for value in self.world_counts
            ):
                raise ValueError(
                    "explicit object shard allocation splits an elite group"
                )
        if requested_worlds == contract.FORMAL_GLOBAL_WORLDS and (
            self.world_counts != FORMAL_OBJECT_WORLD_COUNTS
        ):
            raise RuntimeError("formal global object-world allocation changed")
        self.world_slices = object_world_slices(self.world_counts)
        self.platform = platform_spec
        self.config = config
        self.curriculum_phase = curriculum_phase
        self.residual_enabled = bool(residual_enabled)
        self.worlds = requested_worlds
        self._closed = False
        base_seed = int(config.seed if seed is None else seed)
        child_envs: list[Any] = []
        try:
            for object_index, (child_spec, count) in enumerate(
                zip(shards, self.world_counts)
            ):
                child_envs.append(
                    _construct_fail_closed(
                        child_env_type,
                        curriculum_phase=curriculum_phase,
                        config=config,
                        residual_enabled=residual_enabled,
                        platform_spec=child_spec,
                        worlds=count,
                        device=device,
                        require_linux=require_linux,
                        require_cuda=require_cuda,
                        fixed_object=object_index,
                        fixed_task=config.task_index,
                        seed=base_seed + object_index * 100_003,
                        validate_official_config=validate_official_config,
                        **kwargs,
                    )
                )
        except BaseException:
            for child in reversed(child_envs):
                try:
                    child.close()
                except BaseException:
                    # Preserve the constructor/OOM error and continue
                    # releasing every successfully constructed sibling.
                    pass
            raise
        self.shards = tuple(child_envs)
        try:
            first = self.shards[0]
            self.torch = first.torch
            self.device = first.device
            self.dtype = first.dtype
            if any(shard.device != self.device for shard in self.shards):
                raise ValueError("all object shards must use one Torch device")
            self._buffers: dict[str, Any] = {}
            self._observation_tensor = self.torch.empty(
                (self.worlds, OBSERVATION_DIM),
                device=self.device,
                dtype=self.dtype,
            )
            self._reward_tensor = self.torch.empty(
                self.worlds, device=self.device, dtype=self.dtype
            )
            self._done_tensor = self.torch.empty(
                self.worlds, device=self.device, dtype=self.torch.bool
            )
            self.backend = ShardedBackendFacade(self)
        except BaseException:
            try:
                self.close()
            except BaseException:
                # Preserve the aggregate allocation/facade constructor error.
                pass
            raise

    @property
    def observation_size(self) -> int:
        return OBSERVATION_DIM

    @property
    def action_size(self) -> int:
        return contract.ACTION_DIM

    @property
    def episode_length(self) -> int:
        return self.config.episode_length

    def _global_tensor(self, name: str, values: list[Any]) -> Any:
        first = values[0]
        if first.ndim < 1:
            return first
        tail = tuple(first.shape[1:])
        if any(tuple(value.shape[1:]) != tail for value in values):
            raise ValueError(f"shard tensor {name!r} has heterogeneous widths")
        shape = (self.worlds, *tail)
        buffer = self._buffers.get(name)
        if (
            buffer is None
            or tuple(buffer.shape) != shape
            or buffer.dtype != first.dtype
            or buffer.device != first.device
        ):
            buffer = self.torch.empty(
                shape, device=first.device, dtype=first.dtype
            )
            self._buffers[name] = buffer
        for world_slice, value in zip(self.world_slices, values):
            buffer[world_slice].copy_(value)
        return buffer

    def __getattr__(self, name: str) -> Any:
        shards = self.__dict__.get("shards")
        if not shards:
            raise AttributeError(name)
        try:
            values = [getattr(shard, name) for shard in shards]
        except AttributeError as exc:
            raise AttributeError(name) from exc
        if all(
            hasattr(value, "shape")
            and getattr(value, "ndim", 0) >= 1
            and int(value.shape[0]) == shard.worlds
            for shard, value in zip(shards, values)
        ):
            return self._global_tensor(f"attribute.{name}", values)
        first = values[0]
        if all(value == first for value in values[1:]):
            return first
        raise AttributeError(f"shard attribute {name!r} is not globally uniform")

    def _mask(self, mask: Any | None) -> Any:
        if mask is None:
            return self.torch.ones(
                self.worlds, device=self.device, dtype=self.torch.bool
            )
        return self.torch.as_tensor(
            mask, device=self.device, dtype=self.torch.bool
        ).reshape(self.worlds)

    def _split_goal(
        self,
        value: Any,
        *,
        world_slice: slice,
        global_mask: Any,
    ) -> Any:
        tensor = self.torch.as_tensor(value, device=self.device).reshape(-1)
        if int(tensor.numel()) == 1:
            return tensor
        if int(tensor.numel()) == self.worlds:
            return tensor[world_slice]
        selected = self.torch.nonzero(global_mask, as_tuple=False).flatten()
        if int(tensor.numel()) == int(selected.numel()):
            belongs = (selected >= int(world_slice.start)) & (
                selected < int(world_slice.stop)
            )
            return tensor[belongs]
        raise ValueError(
            "global goals must be scalar, full-world, or selected-world length"
        )

    def _child_goals(
        self,
        goals: Mapping[str, Any] | None,
        *,
        object_index: int,
        world_slice: slice,
        global_mask: Any,
        local_mask: Any,
    ) -> dict[str, Any] | None:
        if goals is None:
            return None
        result = {
            name: self._split_goal(
                value,
                world_slice=world_slice,
                global_mask=global_mask,
            )
            for name, value in goals.items()
        }
        if "object_index" in result:
            values = result["object_index"]
            if int(values.numel()) == 1:
                selected_values = values
            elif int(values.numel()) == int(local_mask.numel()):
                selected_values = values[local_mask]
            else:
                selected_values = values
            if bool(local_mask.any()) and selected_values.numel() and not bool(
                (selected_values == object_index).all()
            ):
                raise ValueError(
                    f"goals route a non-{contract.OBJECTS[object_index]} "
                    "object into its fixed MJWarp shard"
                )
        if "task_index" in result:
            values = result["task_index"]
            if int(values.numel()) == 1:
                selected_values = values
            elif int(values.numel()) == int(local_mask.numel()):
                selected_values = values[local_mask]
            else:
                selected_values = values
            if bool(local_mask.any()) and selected_values.numel() and not bool(
                (selected_values == self.config.task_index).all()
            ):
                raise ValueError("goals attempted to cross the fixed task bundle")
        return result

    def reset(
        self,
        mask: Any | None = None,
        goals: Mapping[str, Any] | None = None,
    ) -> Any:
        global_mask = self._mask(mask)
        for object_index, (shard, world_slice) in enumerate(
            zip(self.shards, self.world_slices)
        ):
            local_mask = global_mask[world_slice]
            local_goals = self._child_goals(
                goals,
                object_index=object_index,
                world_slice=world_slice,
                global_mask=global_mask,
                local_mask=local_mask,
            )
            observation = shard.reset(mask=local_mask, goals=local_goals)
            self._observation_tensor[world_slice].copy_(observation)
        return self._observation_tensor

    def reset_with_goals(
        self,
        object_index: Any,
        template_index: Any,
        task_index: Any,
        path_seed: Any,
        path_shape_index: Any,
        rotation_angle_deg: Any,
        source_region: Any = 0,
    ) -> Any:
        return self.reset(
            goals={
                "object_index": object_index,
                "template_index": template_index,
                "task_index": task_index,
                "path_seed": path_seed,
                "path_shape_index": path_shape_index,
                "rotation_angle_deg": rotation_angle_deg,
                "source_region": source_region,
            }
        )

    def _merge_value(self, name: str, values: list[Any]) -> Any:
        if all(
            hasattr(value, "shape")
            and getattr(value, "ndim", 0) >= 1
            and int(value.shape[0]) == shard.worlds
            for shard, value in zip(self.shards, values)
        ):
            return self._global_tensor(name, values)
        if all(isinstance(value, Mapping) for value in values):
            keys = set(values[0])
            if any(set(value) != keys for value in values[1:]):
                raise ValueError(f"shard mapping {name!r} has inconsistent keys")
            return {
                key: self._merge_value(
                    f"{name}.{key}", [value[key] for value in values]
                )
                for key in sorted(keys)
            }
        first = values[0]
        if all(value == first for value in values[1:]):
            return first
        raise ValueError(f"non-world shard value {name!r} differs")

    def step(self, action: Any) -> StepBatch:
        action = self.torch.as_tensor(
            action, device=self.device, dtype=self.dtype
        )
        if tuple(action.shape) != (self.worlds, contract.ACTION_DIM):
            raise ValueError(
                f"action must be [{self.worlds},{contract.ACTION_DIM}]"
            )
        transitions: list[StepBatch] = []
        for shard, world_slice in zip(self.shards, self.world_slices):
            # ``narrow``/slice is a view: no global action gather or host copy.
            transitions.append(shard.step(action[world_slice]))
        for world_slice, transition in zip(self.world_slices, transitions):
            self._observation_tensor[world_slice].copy_(transition.observation)
            self._reward_tensor[world_slice].copy_(transition.reward)
            self._done_tensor[world_slice].copy_(transition.done)
        keys = set(transitions[0].info)
        if any(set(transition.info) != keys for transition in transitions[1:]):
            raise RuntimeError("object-shard StepBatch info schemas disagree")
        info = {
            key: self._merge_value(
                f"info.{key}",
                [transition.info[key] for transition in transitions],
            )
            for key in sorted(keys)
        }
        return StepBatch(
            observation=self._observation_tensor,
            reward=self._reward_tensor,
            done=self._done_tensor,
            info=info,
        )

    def snapshot(self) -> dict[str, Any]:
        snapshots = [shard.snapshot() for shard in self.shards]
        keys = set(snapshots[0])
        if any(set(snapshot) != keys for snapshot in snapshots[1:]):
            raise RuntimeError("object-shard snapshot schemas disagree")
        return {
            key: self._merge_value(
                f"snapshot.{key}", [snapshot[key] for snapshot in snapshots]
            )
            for key in sorted(keys)
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        for object_name, shard in reversed(
            tuple(zip(contract.OBJECTS, getattr(self, "shards", ())))
        ):
            try:
                shard.close()
            except BaseException as exc:
                errors.append(
                    f"{object_name}: {type(exc).__name__}: {exc}"
                )
        if errors:
            raise RuntimeError(
                "one or more object shards failed to close: "
                + "; ".join(errors)
            )


__all__ = [
    "formal_sharded_health_is_valid",
    "ShardedBackendFacade",
    "ShardedMJWarpEnv",
    "ordered_shard_manifest",
]
