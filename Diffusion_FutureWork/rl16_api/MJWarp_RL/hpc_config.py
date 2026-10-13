"""Shared RTX PRO 6000 configuration for Diffusion plus native MJWarp."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from .contract import (
    FORMAL_ACTION_TICKS,
    FORMAL_EPISODE_TICKS,
    FORMAL_GLOBAL_WORLDS,
    FORMAL_OBJECT_WORLD_COUNTS as CONTRACT_FORMAL_OBJECT_WORLD_COUNTS,
    FORMAL_REMAINDER_OBJECT_WORLD_COUNTS,
    FORMAL_REMAINDER_WORLDS,
    OBJECT_SHARD_COUNT,
    TRANSLATION_PATH_SEGMENT_TICKS,
    ROTATION_MOTION_TICKS,
    ROTATION_FORMAL_ACTION_TICKS,
    TOTAL_TRAINING_STEPS,
    TRANSLATION_PATH_LEAD_IN_TICKS,
    TRANSLATION_PATH_SEGMENTS,
    TASKS,
    TASK_TRANSLATE,
)


FORMAL_WORLD_CANDIDATES = (FORMAL_GLOBAL_WORLDS,)
CAPACITY_REPORT_SCHEMA = "grab-g6-mjwarp-capacity-scan-v14"


def balanced_object_world_counts(
    total_worlds: int,
    *,
    population: int = 1,
    object_count: int = OBJECT_SHARD_COUNT,
    extra_group_order: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Split one global batch into stable object-major population groups."""

    total = int(total_worlds)
    group_size = int(population)
    count = int(object_count)
    if total <= 0 or group_size <= 0 or count <= 0:
        raise ValueError("world, population, and object counts must be positive")
    if total % group_size:
        raise ValueError("global worlds must be divisible by population")
    groups = total // group_size
    if groups < count:
        raise ValueError("global batch must provide at least one group per object")
    base, extra = divmod(groups, count)
    order = (
        tuple(range(count))
        if extra_group_order is None
        else tuple(int(index) for index in extra_group_order)
    )
    if len(order) != count or set(order) != set(range(count)):
        raise ValueError("extra_group_order must be an object-index permutation")
    extra_objects = frozenset(order[:extra])
    allocation = tuple(
        (base + (index in extra_objects)) * group_size
        for index in range(count)
    )
    if sum(allocation) != total or any(value <= 0 for value in allocation):
        raise RuntimeError("balanced object-world allocation is inconsistent")
    return allocation


def balanced_object_world_slices(
    total_worlds: int,
    *,
    population: int = 1,
    object_count: int = OBJECT_SHARD_COUNT,
    extra_group_order: tuple[int, ...] | None = None,
) -> tuple[slice, ...]:
    """Return contiguous object-major slices for a balanced allocation."""

    counts = balanced_object_world_counts(
        total_worlds,
        population=population,
        object_count=object_count,
        extra_group_order=extra_group_order,
    )
    start = 0
    result: list[slice] = []
    for count in counts:
        result.append(slice(start, start + count))
        start += count
    return tuple(result)


def object_world_slices(world_counts: tuple[int, ...]) -> tuple[slice, ...]:
    """Return contiguous object-major slices for one explicit allocation."""

    if not world_counts or any(int(value) <= 0 for value in world_counts):
        raise ValueError("every object shard must own at least one world")
    start = 0
    result: list[slice] = []
    for value in world_counts:
        count = int(value)
        result.append(slice(start, start + count))
        start += count
    return tuple(result)


FORMAL_OBJECT_WORLD_COUNTS = CONTRACT_FORMAL_OBJECT_WORLD_COUNTS
if FORMAL_OBJECT_WORLD_COUNTS != balanced_object_world_counts(
    FORMAL_WORLD_CANDIDATES[0],
    population=10,
    object_count=OBJECT_SHARD_COUNT,
):
    raise RuntimeError("contract and runtime object-world allocations disagree")


@dataclass(frozen=True)
class EliteSearchSettings:
    """Parallel allocation used to certify closed-loop IK demonstrations.

    The public name is retained for artifact-reader compatibility.  The
    formal teacher no longer perturbs actions or ranks rollouts by reward;
    ``population`` now means identical-size expert certification groups.
    """

    population: int = 10
    # One winner per population group and completed episode is retained.
    elite_per_group: int = 1
    minimum_successful_demos: int = 24
    minimum_demos_per_object_task: int = 1
    require_all_reset_templates: bool = True
    require_all_translation_path_shapes: bool = True
    episode_action_sigma: float = 0.18
    dynamic_action_sigma: float = 0.12
    dynamic_knot_ticks: int = 16
    centroid_momentum: float = 0.85
    success_bonus: float = 1_000.0
    storage_dtype: str = "float16"

    def validate(self) -> None:
        if self.population < 2:
            raise ValueError("elite-search population must be at least 2")
        if self.elite_per_group != 1:
            raise ValueError("the controlled experiment retains one group winner")
        if self.minimum_successful_demos <= 0:
            raise ValueError("minimum_successful_demos must be positive")
        if self.minimum_demos_per_object_task <= 0:
            raise ValueError(
                "minimum_demos_per_object_task must be positive"
            )
        if self.dynamic_knot_ticks <= 0:
            raise ValueError("dynamic_knot_ticks must be positive")
        if not 0.0 < self.centroid_momentum < 1.0:
            raise ValueError("centroid_momentum must be in (0, 1)")
        if min(self.episode_action_sigma, self.dynamic_action_sigma) <= 0.0:
            raise ValueError("elite-search sigmas must be positive")
        if self.success_bonus <= 0.0:
            raise ValueError("success_bonus must be positive")
        if self.storage_dtype != "float16":
            raise ValueError("formal demonstrations use float16 storage")


@dataclass(frozen=True)
class DiffusionSettings:
    """SCREW-compatible state-conditioned action-chunk Diffusion profile."""

    obs_horizon: int = 2
    action_horizon: int = 16
    diffusion_steps: int = 100
    inference_steps: int = 16
    replan_interval: int = 4
    hidden_dim: int = 256
    depth: int = 6
    dropout: float = 0.0
    epochs: int = 60
    batch_size: int = 4_096
    # Fixed across both G6 tasks even when expert search yields different
    # numbers of successful elite trajectories.
    updates_per_epoch: int = 1_100
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    ema_decay: float = 0.999
    max_grad_norm: float = 1.0
    checkpoint_every_epochs: int = 1
    cache_demos_on_cuda: bool = True
    use_bfloat16: bool = True
    stratified_object_task_sampling: bool = True

    def validate(self) -> None:
        for name in (
            "obs_horizon",
            "action_horizon",
            "diffusion_steps",
            "inference_steps",
            "replan_interval",
            "hidden_dim",
            "depth",
            "epochs",
            "batch_size",
            "updates_per_epoch",
            "checkpoint_every_epochs",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.obs_horizon != 2:
            raise ValueError("the controlled Diffusion contract uses obs_horizon=2")
        if self.inference_steps > self.diffusion_steps:
            raise ValueError("inference_steps cannot exceed diffusion_steps")
        if self.replan_interval > self.action_horizon:
            raise ValueError("replan_interval cannot exceed action_horizon")
        if min(self.learning_rate, self.weight_decay, self.max_grad_norm) <= 0.0:
            raise ValueError("Diffusion optimizer values must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")


@dataclass(frozen=True)
class HPCConfig:
    """One G6 task-specific 90M IK-expert run plus Diffusion imitation."""

    seed: int = 7
    task: str = TASK_TRANSLATE
    total_steps: int = TOTAL_TRAINING_STEPS
    num_envs: int = FORMAL_GLOBAL_WORLDS
    num_eval_envs: int = 96
    device: str = "cuda:0"
    cuda_graph: bool = True
    deterministic_torch: bool = True
    log_every_control_ticks: int = 25
    status_every_seconds: float = 15.0
    elite: EliteSearchSettings = EliteSearchSettings()
    diffusion: DiffusionSettings = DiffusionSettings()

    # Parent XML runs at 0.5 ms; one learned/control decision is 20 ms.
    physics_substeps_per_control_tick: int = 40
    plan_ticks: int = 8
    close_ticks: int = 90
    settle_ticks: int = 60
    lift_ticks: int = 70
    hold_ticks: int = 70
    translation_action_ticks: int = FORMAL_ACTION_TICKS
    rotation_action_ticks: int = ROTATION_FORMAL_ACTION_TICKS
    translation_path_lead_in_ticks: int = TRANSLATION_PATH_LEAD_IN_TICKS
    translation_segment_ticks: int = TRANSLATION_PATH_SEGMENT_TICKS
    rotation_motion_ticks: int = ROTATION_MOTION_TICKS

    # Correct active-set IK can sustain all four hard-tip contacts at once.
    # The previous 96-slot cap overflowed at 97 contacts in otherwise finite
    # worlds, so retain power-of-two headroom for the full trajectory.
    contacts_per_world: int = 128
    constraints_per_world: int = 640

    def validate(self) -> None:
        self.elite.validate()
        self.diffusion.validate()
        if self.task not in TASKS:
            raise ValueError(
                f"task must be one of {TASKS!r}, got {self.task!r}"
            )
        if self.total_steps != TOTAL_TRAINING_STEPS:
            raise ValueError("official search must contain exactly 90M steps")
        if self.num_envs <= 0 or self.num_eval_envs <= 0:
            raise ValueError("environment counts must be positive")
        if self.num_envs % self.elite.population:
            raise ValueError("num_envs must be divisible by elite population")
        allocation = balanced_object_world_counts(
            self.num_envs,
            population=self.elite.population,
        )
        if self.num_envs == FORMAL_WORLD_CANDIDATES[0] and (
            allocation != FORMAL_OBJECT_WORLD_COUNTS
        ):
            raise ValueError("formal object-shard world allocation changed")
        if self.num_envs == FORMAL_GLOBAL_WORLDS and (
            self.remainder_worlds != FORMAL_REMAINDER_WORLDS
            or self.remainder_object_world_counts
            != FORMAL_REMAINDER_OBJECT_WORLD_COUNTS
        ):
            raise ValueError("formal remainder allocation changed")
        if self.control_ticks < self.episode_length:
            raise ValueError("formal run must include a complete endpoint episode")
        if self.maximum_demo_episodes < self.elite.minimum_successful_demos:
            raise ValueError("90M budget cannot satisfy the minimum demo count")
        for name in (
            "physics_substeps_per_control_tick",
            "plan_ticks",
            "close_ticks",
            "settle_ticks",
            "lift_ticks",
            "hold_ticks",
            "translation_action_ticks",
            "rotation_action_ticks",
            "translation_path_lead_in_ticks",
            "translation_segment_ticks",
            "rotation_motion_ticks",
            "contacts_per_world",
            "constraints_per_world",
            "log_every_control_ticks",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            self.translation_action_ticks != FORMAL_ACTION_TICKS
            or self.rotation_action_ticks != ROTATION_FORMAL_ACTION_TICKS
        ):
            raise ValueError("formal task-specific action horizons changed")
        if (
            self.translation_path_lead_in_ticks
            != TRANSLATION_PATH_LEAD_IN_TICKS
            or self.translation_segment_ticks
            != TRANSLATION_PATH_SEGMENT_TICKS
            or self.rotation_motion_ticks != ROTATION_MOTION_TICKS
        ):
            raise ValueError("formal task-specific motion timing changed")
        if self.translation_motion_ticks > self.translation_action_ticks:
            raise ValueError("translation path exceeds action horizon")
        if self.rotation_motion_ticks > self.rotation_action_ticks:
            raise ValueError("rotation motion exceeds action horizon")
        if self.episode_length != FORMAL_EPISODE_TICKS:
            raise ValueError("formal episode must contain exactly 4500 ticks")
        if self.status_every_seconds <= 0.0:
            raise ValueError("status_every_seconds must be positive")

    @property
    def action_ticks(self) -> int:
        """Task-specific action horizon without stale ``replace`` state."""

        if self.task == TASK_TRANSLATE:
            return self.translation_action_ticks
        return self.rotation_action_ticks

    @property
    def episode_length(self) -> int:
        return (
            self.plan_ticks
            + self.close_ticks
            + self.settle_ticks
            + self.lift_ticks
            + self.hold_ticks
            + self.action_ticks
        )

    @property
    def task_index(self) -> int:
        """Index of the one task owned by this formal training process."""

        return TASKS.index(self.task)

    @property
    def control_ticks(self) -> int:
        return self.total_steps // self.num_envs

    @property
    def translation_motion_ticks(self) -> int:
        return self.translation_path_lead_in_ticks + (
            TRANSLATION_PATH_SEGMENTS * self.translation_segment_ticks
        )

    @property
    def remainder_worlds(self) -> int:
        return self.total_steps % self.num_envs

    @property
    def groups_per_vector_batch(self) -> int:
        return self.num_envs // self.elite.population

    @property
    def object_world_counts(self) -> tuple[int, ...]:
        return balanced_object_world_counts(
            self.num_envs,
            population=self.elite.population,
        )

    @property
    def object_group_counts(self) -> tuple[int, ...]:
        return tuple(
            worlds // self.elite.population
            for worlds in self.object_world_counts
        )

    @property
    def remainder_object_world_counts(self) -> tuple[int, ...]:
        if self.remainder_worlds == 0:
            return (0,) * OBJECT_SHARD_COUNT
        if self.remainder_worlds == FORMAL_REMAINDER_WORLDS:
            return FORMAL_REMAINDER_OBJECT_WORLD_COUNTS
        return balanced_object_world_counts(
            self.remainder_worlds,
            population=self.elite.population,
        )

    @property
    def maximum_demo_episodes(self) -> int:
        complete_batches = self.control_ticks // self.episode_length
        return complete_batches * self.groups_per_vector_batch

    def diffusion_batches_per_epoch(self, completed_demos: int) -> int:
        if int(completed_demos) <= 0:
            raise ValueError("Diffusion training requires at least one demo")
        return self.diffusion.updates_per_epoch

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "action_ticks": self.action_ticks,
                "episode_length": self.episode_length,
                "task_index": self.task_index,
                "control_ticks": self.control_ticks,
                "remainder_worlds": self.remainder_worlds,
                "groups_per_vector_batch": self.groups_per_vector_batch,
                "object_world_counts": self.object_world_counts,
                "object_group_counts": self.object_group_counts,
                "remainder_object_world_counts": (
                    self.remainder_object_world_counts
                ),
                "maximum_demo_episodes": self.maximum_demo_episodes,
            }
        )
        return result

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


DEFAULT_CONFIG = HPCConfig()
DEFAULT_CONFIG.validate()


__all__ = [
    "CAPACITY_REPORT_SCHEMA",
    "DEFAULT_CONFIG",
    "DiffusionSettings",
    "EliteSearchSettings",
    "FORMAL_WORLD_CANDIDATES",
    "FORMAL_OBJECT_WORLD_COUNTS",
    "HPCConfig",
    "balanced_object_world_counts",
    "balanced_object_world_slices",
    "object_world_slices",
]
