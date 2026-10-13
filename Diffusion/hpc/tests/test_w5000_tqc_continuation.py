import copy
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]

from s_hpc.config import load_config
from s_hpc.workflows import _validate_frozen_diffusion_config
from validate_w5000_tqc_continuation import (
    _runtime_width_independent,
    validate_config,
)


class W5000TQCContinuationTests(unittest.TestCase):
    def setUp(self):
        self.source = load_config(
            ROOT
            / "configs"
            / "autodl_rtx_pro_6000_r100_success_selfdistill_w18000_tqc16.json"
        )
        self.continuation = load_config(
            ROOT
            / "configs"
            / "autodl_rtx_pro_6000_r100_w5000_tqc16_continuation.json"
        )

    def test_config_changes_only_name_and_runtime_width(self):
        result = validate_config(self.continuation)
        self.assertEqual(result["worlds"], 5_000)
        self.assertEqual(self.source.physics.worlds, 18_000)
        self.assertEqual(
            _runtime_width_independent(self.source.to_dict()),
            _runtime_width_independent(self.continuation.to_dict()),
        )

    def test_production_loader_accepts_width_only_change(self):
        _validate_frozen_diffusion_config(
            {"config": self.source.to_dict()}, self.continuation
        )

    def test_production_loader_rejects_learning_contract_change(self):
        stored = copy.deepcopy(self.source.to_dict())
        stored["diffusion"]["inference_steps"] += 1
        with self.assertRaisesRegex(ValueError, "beyond the allowed"):
            _validate_frozen_diffusion_config(
                {"config": stored}, self.continuation
            )

    def test_launcher_is_isolated_and_uses_formal_timeout_gate(self):
        launcher = (
            ROOT.parent
            / "run_autodl_r100_w5000_tqc16_continuation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--worlds 5000", launcher)
        self.assertIn("--max-expert-truncation-fraction", launcher)
        self.assertIn(
            'MAX_EXPERT_TRUNCATION_FRACTION="${SCREW_MAX_EXPERT_TRUNCATION_FRACTION:-0.005}"',
            launcher,
        )
        self.assertIn("r100_w5000_tqc16_continuation", launcher)
        self.assertIn("--no-resume", launcher)
        self.assertNotIn("train_diffusion_gpu.py", launcher)
        self.assertNotIn("--worlds 18000", launcher)

    def test_launcher_pins_preflight_marker_to_source_model(self):
        launcher = (
            ROOT.parent
            / "run_autodl_r100_w5000_tqc16_continuation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('p.get("diffusion_sha256")==diffusion_sha', launcher)
        self.assertIn(
            'str(p.get("max_expert_truncation_fraction"))==allowance',
            launcher,
        )
        self.assertIn(
            'payload.get("max_expert_truncation_fraction")', launcher
        )


if __name__ == "__main__":
    unittest.main()
