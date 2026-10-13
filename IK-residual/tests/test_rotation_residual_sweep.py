"""No-training contracts for deterministic trained-residual evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from g3g6 import config
from g3g6 import rotation_residual_sweep as sweep
from g3g6 import rotation_residual_training as training


class RotationResidualSweepContractTests(unittest.TestCase):
    @staticmethod
    def _artifact_run(root: Path, gripper: str = "g3") -> Path:
        run = root / gripper
        run.mkdir(parents=True)
        model = run / sweep.MODEL_FILENAME
        normalizer = run / sweep.VEC_NORMALIZE_FILENAME
        model.write_bytes(b"model-contract")
        normalizer.write_bytes(b"normalizer-contract")
        manifest = training.formal_config(gripper, device="cpu")
        manifest.update(
            {
                "status": "complete",
                "artifacts": {
                    "model": {
                        "path": model.name,
                        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    },
                    "vec_normalize": {
                        "path": normalizer.name,
                        "sha256": hashlib.sha256(
                            normalizer.read_bytes()
                        ).hexdigest(),
                    },
                },
            }
        )
        (run / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return run

    def test_loader_accepts_exact_trainer_manifest_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as temporary:
            run = self._artifact_run(Path(temporary), "g6")
            artifacts = sweep.load_training_artifacts(run)
            self.assertEqual(artifacts.gripper, "g6")
            self.assertEqual(artifacts.action_dim, 16)
            self.assertEqual(artifacts.observation_dim, 149)
            self.assertTrue(artifacts.residual_nominal_feedback)
            self.assertTrue(artifacts.residual_after_preaction_only)

            (run / sweep.MODEL_FILENAME).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                sweep.load_training_artifacts(run)

    def test_loader_rejects_contract_drift_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as temporary:
            run = self._artifact_run(Path(temporary), "g3")
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["observation_dimension"] = 145
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "OBSERVATION_DIM=149"):
                sweep.load_training_artifacts(run)

    def test_policy_uses_frozen_normalized_observation_deterministically(self) -> None:
        class Normalizer:
            def __init__(self) -> None:
                self.seen: np.ndarray | None = None

            def normalize_obs(self, observation: np.ndarray) -> np.ndarray:
                self.seen = observation.copy()
                return observation + 2.0

        class Model:
            def __init__(self) -> None:
                self.deterministic: bool | None = None
                self.seen: np.ndarray | None = None

            def predict(
                self, observation: np.ndarray, *, deterministic: bool
            ) -> tuple[np.ndarray, None]:
                self.seen = observation.copy()
                self.deterministic = deterministic
                return np.linspace(-1.0, 1.0, 12)[None, :], None

        raw = np.zeros(149, dtype=np.float32)
        model = Model()
        normalizer = Normalizer()
        action = sweep._normalized_policy_action(
            model, normalizer, raw, action_dim=12
        )
        self.assertTrue(model.deterministic)
        self.assertEqual(normalizer.seen.shape, (1, 149))
        np.testing.assert_array_equal(model.seen, np.full((1, 149), 2.0))
        np.testing.assert_allclose(action, np.linspace(-1.0, 1.0, 12))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
