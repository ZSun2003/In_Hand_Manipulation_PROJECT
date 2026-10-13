"""Gymnasium environment for the G6 current-IK + 16-D residual controller."""

from __future__ import annotations

from typing import Any

try:
    from . import config
    from .controller_bridge import OBSERVATION_DIM, ResidualControlBridge
    from .runtime import append_rl_site_packages
except ImportError:
    import config  # type: ignore
    from controller_bridge import (  # type: ignore
        OBSERVATION_DIM,
        ResidualControlBridge,
    )
    from runtime import append_rl_site_packages  # type: ignore


append_rl_site_packages()

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class G6IKResidualEnv(gym.Env[np.ndarray, np.ndarray]):
    """Uniform six-object/two-task environment with endpoint-only success."""

    metadata = {"render_modes": []}

    def __init__(self, *, wait_timeout_s: float = 120.0) -> None:
        super().__init__()
        self.wait_timeout_s = float(wait_timeout_s)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self._bridge: ResidualControlBridge | None = None
        self._fixed_episode: tuple[str, str, int] | None = None

    def set_fixed_episode(
        self,
        object_kind: str,
        task: str,
        seed: int,
    ) -> None:
        """Set the next reset for strict paired evaluation."""
        if object_kind not in config.OBJECTS:
            raise ValueError(f"unknown object: {object_kind!r}")
        if task not in config.TASKS:
            raise ValueError(f"unknown task: {task!r}")
        self._fixed_episode = (object_kind, task, int(seed))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._close_bridge()

        fixed = self._fixed_episode
        self._fixed_episode = None
        if options is not None and any(
            key in options for key in ("object", "task", "episode_seed")
        ):
            object_kind = str(options["object"])
            task = str(options["task"])
            episode_seed = int(options.get("episode_seed", seed or 0))
        elif fixed is not None:
            object_kind, task, episode_seed = fixed
        else:
            object_kind = str(self.np_random.choice(config.OBJECTS))
            task = str(self.np_random.choice(config.TASKS))
            episode_seed = int(
                self.np_random.integers(0, np.iinfo(np.int32).max)
            )

        self._bridge = ResidualControlBridge(
            object_kind,
            task,
            episode_seed,
            wait_timeout_s=self.wait_timeout_s,
        )
        observation, info = self._bridge.start()
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._bridge is None:
            raise RuntimeError("reset() must be called before step()")
        return self._bridge.advance(action)

    def close(self) -> None:
        self._close_bridge()
        super().close()

    def _close_bridge(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None


def make_env() -> G6IKResidualEnv:
    """Factory used by Stable-Baselines3 and smoke tests."""
    return G6IKResidualEnv()
