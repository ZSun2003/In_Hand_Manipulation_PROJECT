"""Contracts for the independent 28-sample rotation endpoint grid."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from g3g6 import config
from g3g6.rotation_residual_endpoint_eval import (
    BIN_QUOTAS,
    EVAL_AXIS_ORDER,
    OBJECT_ORDER,
    SAMPLES_PER_CONDITION,
    build_endpoint_specs,
    magnitude_bin,
    paired_sampling_contract,
    run_endpoint_episode,
    run_endpoint_shard,
    sample_bin_for_index,
    _validate_resume_artifacts,
    _validate_row_against_spec,
)


class _FakeNormalizer:
    def normalize_obs(self, observation: np.ndarray) -> np.ndarray:
        return observation


class _FakeModel:
    def predict(self, _observation: np.ndarray, deterministic: bool = True):
        assert deterministic
        return np.zeros(12, dtype=np.float64), None


class _FakeEndpointEnv:
    def __init__(self, spec, *, endpoint_angle_error: float = 0.0) -> None:
        self.spec = spec
        self.endpoint_angle_error = endpoint_angle_error
        self.step_count = 0

    def reset(self, *, seed: int, options):
        assert seed == self.spec.seed
        assert options == {"rotation_axes": self.spec.rotation_axes}
        return np.zeros(149, dtype=np.float32), {
            "rotation_goal_euler_deg": np.asarray(self.spec.goal_euler_deg),
            "rotation_sampling_profile": config.WEIGHTED_ROTATION_SAMPLING_PROFILE,
        }

    def step(self, _action):
        self.step_count += 1
        terminal = self.step_count == config.FORMAL_EPISODE_TICKS
        angle = self.endpoint_angle_error if terminal else 0.0
        success = terminal and angle <= config.ANGLE_TOLERANCE_DEG_PER_AXIS
        info = {
            "control_mode": "residual_action",
            "step": self.step_count,
            "position_error_m_per_axis": np.zeros(3),
            "angle_error_deg_per_axis": np.asarray((angle, 0.0, 0.0)),
            "object_grasp_frame_position_m": np.asarray((0.0, 0.0, 0.325)),
            "ik_projection_fraction": 1.0,
            "tip_contact_count": 4,
            "termination_reason": "horizon" if terminal else "running",
            "is_success": success,
        }
        return np.zeros(149, dtype=np.float32), 0.0, terminal, False, info

    def close(self) -> None:
        pass


def _fake_artifacts(run_dir: Path = Path("/project/run")):
    return SimpleNamespace(
        run_dir=run_dir,
        gripper="g3",
        rotation_sampling_profile=config.WEIGHTED_ROTATION_SAMPLING_PROFILE,
        residual_nominal_feedback=True,
        residual_after_preaction_only=True,
        action_dim=12,
        model_sha256="model-a",
        vec_normalize_sha256="vec-a",
    )


class RotationResidualEndpointEvalTests(unittest.TestCase):
    def test_full_grid_has_784_unique_cases_per_gripper(self) -> None:
        specs = build_endpoint_specs("g3")
        self.assertEqual(
            len(specs), len(OBJECT_ORDER) * len(EVAL_AXIS_ORDER) * 28
        )
        self.assertEqual(len({spec.episode_id for spec in specs}), len(specs))
        counts = Counter((spec.object_name, spec.rotation_axes) for spec in specs)
        self.assertEqual(set(counts.values()), {SAMPLES_PER_CONDITION})

    def test_each_object_axis_has_exact_6_14_8_bins(self) -> None:
        specs = build_endpoint_specs("g6")
        for object_name in OBJECT_ORDER:
            for axes in EVAL_AXIS_ORDER:
                selected = [
                    spec
                    for spec in specs
                    if spec.object_name == object_name and spec.rotation_axes == axes
                ]
                counts = Counter(spec.magnitude_bin for spec in selected)
                self.assertEqual(tuple(counts[index] for index in range(3)), BIN_QUOTAS)
                for spec in selected:
                    self.assertEqual(
                        magnitude_bin(spec.goal_magnitude_deg), spec.magnitude_bin
                    )

    def test_g3_g6_goals_and_seeds_are_exactly_paired(self) -> None:
        paired_sampling_contract()
        g3 = build_endpoint_specs("g3")
        g6 = build_endpoint_specs("g6")
        for first, second in zip(g3, g6):
            self.assertEqual(first.pair_id, second.pair_id)
            self.assertEqual(first.seed, second.seed)
            self.assertEqual(first.goal_euler_deg, second.goal_euler_deg)

    def test_sample_indices_encode_the_stratified_quota(self) -> None:
        counts = Counter(sample_bin_for_index(index) for index in range(28))
        self.assertEqual(tuple(counts[index] for index in range(3)), BIN_QUOTAS)

    def test_requested_axis_order_is_preserved(self) -> None:
        self.assertEqual(
            EVAL_AXIS_ORDER, ("x", "y", "z", "xz", "xy", "yz", "xyz")
        )

    def test_resume_rejects_a_different_model_or_normalizer(self) -> None:
        artifacts = _fake_artifacts()
        valid = {
            "training_run": str(artifacts.run_dir),
            "model_sha256": artifacts.model_sha256,
            "vec_normalize_sha256": artifacts.vec_normalize_sha256,
        }
        _validate_resume_artifacts(valid, [], artifacts)
        tampered = dict(valid, model_sha256="model-b")
        with self.assertRaisesRegex(ValueError, "model_sha256 mismatch"):
            _validate_resume_artifacts(tampered, [], artifacts)
        missing = {"training_run": str(artifacts.run_dir)}
        with self.assertRaisesRegex(ValueError, "missing model_sha256"):
            _validate_resume_artifacts(missing, [{}], artifacts)

    def test_rebuilt_spec_detects_a_tampered_seed(self) -> None:
        spec = build_endpoint_specs("g3")[0]
        row = {
            "episode_id": spec.episode_id,
            "pair_id": spec.pair_id,
            "gripper": spec.gripper,
            "object": spec.object_name,
            "rotation_axes": spec.rotation_axes,
            "sample_index": spec.sample_index,
            "magnitude_bin": spec.magnitude_bin,
            "seed": spec.seed,
            "goal_euler_deg_x": spec.goal_euler_deg[0],
            "goal_euler_deg_y": spec.goal_euler_deg[1],
            "goal_euler_deg_z": spec.goal_euler_deg[2],
            "goal_magnitude_deg": spec.goal_magnitude_deg,
        }
        _validate_row_against_spec(row, spec)
        row["seed"] += 1
        with self.assertRaisesRegex(ValueError, "does not match rebuilt spec"):
            _validate_row_against_spec(row, spec)

    def test_fake_endpoint_env_audits_strict_success_and_endpoint_failure(self) -> None:
        spec = build_endpoint_specs("g3")[0]
        artifacts = _fake_artifacts()
        controller = SimpleNamespace(model=_FakeModel(), normalizer=_FakeNormalizer())
        success = run_endpoint_episode(
            spec,
            artifacts=artifacts,
            controller=controller,
            env_factory=lambda **_kwargs: _FakeEndpointEnv(spec),
        )
        self.assertEqual(success["is_success"], 1)
        self.assertEqual(success["ticks"], config.FORMAL_EPISODE_TICKS)
        failure = run_endpoint_episode(
            spec,
            artifacts=artifacts,
            controller=controller,
            env_factory=lambda **_kwargs: _FakeEndpointEnv(
                spec, endpoint_angle_error=5.001
            ),
        )
        self.assertEqual(failure["is_success"], 0)
        self.assertGreater(failure["endpoint_euler_linf_deg"], 5.0)

    def test_controller_load_failure_is_checkpointed_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as temporary:
            root = Path(temporary)
            training_run = root / "training"
            artifacts = _fake_artifacts(training_run.resolve())
            with self.assertRaisesRegex(RuntimeError, "controller load failed"):
                run_endpoint_shard(
                    gripper="g3",
                    training_run=training_run,
                    output_dir=root / "output",
                    shard_index=0,
                    num_shards=1,
                    limit=1,
                    artifact_loader=lambda _path: artifacts,
                    controller_loader=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        RuntimeError("controller load failed")
                    ),
                )
            manifest = json.loads(
                (root / "output" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failure"]["type"], "RuntimeError")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
