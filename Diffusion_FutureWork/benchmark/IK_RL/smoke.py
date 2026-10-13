"""Run Gym contract checks and a two-step lightweight TQC save smoke."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile

from .runtime import ensure_runtime

ensure_runtime()

from gymnasium.utils.env_checker import check_env
import numpy as np

from .env import G3IKResidualEnv
from .train import train


def main() -> None:
    environment = G3IKResidualEnv(
        control_mode="ik_only",
        fixed_object="cube",
        fixed_task="translate_xy45_100mm",
    )
    try:
        check_env(environment, skip_render_check=True)
        environment.reset(seed=17, options={"pair_seed": 17})
        environment.step(np.zeros(14))
        _obs, _reward, terminated, _truncated, info = environment.step(
            np.zeros(14)
        )
        if terminated or info["control_step"] != 1:
            raise RuntimeError("one-step/one-control-tick smoke failed")
    finally:
        environment.close()

    with tempfile.TemporaryDirectory(prefix="g3_ik_rl_smoke_") as temporary:
        model, normalizer = train(
            Namespace(
                total_timesteps=2,
                seed=17,
                device="cpu",
                n_envs=1,
                vec_env="dummy",
                output_dir=temporary,
            )
        )
        if not model.is_file() or not normalizer.is_file():
            raise RuntimeError("two-step TQC artifacts are missing")
        artifact_names = sorted(
            path.name for path in Path(temporary).iterdir() if path.is_file()
        )
        if artifact_names != ["final_model.zip", "vec_normalize.pkl"]:
            raise RuntimeError(
                f"unexpected training artifacts: {artifact_names}"
            )
        print(f"two-step TQC model: {Path(model).name}")
        print(f"two-step VecNormalize: {Path(normalizer).name}")
    print("G3 IK_RL smoke passed")


if __name__ == "__main__":
    main()
