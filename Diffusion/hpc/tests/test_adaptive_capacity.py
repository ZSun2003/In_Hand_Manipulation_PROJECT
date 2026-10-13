from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adaptive_capacity import (
    SELECTION_FORMAT,
    _sha256,
    _tail_text,
    derive_config,
    gate,
    probe,
    resolve_qualified_config,
    summarize,
)
from s_hpc.contracts import SCHEMA_HASH, TASK_GEOMETRY_HASH
from s_hpc.config import PACKAGE_ROOT
from training_stack_preflight import classify_gpu_type


class AdaptiveCapacityTest(unittest.TestCase):
    @staticmethod
    def _record(worlds: int, status: str = "safe_pass") -> dict:
        return {
            "worlds": worlds,
            "status": status,
            "returncode": 0 if status != "fail" else 1,
            "elapsed_seconds": 0.1,
            "preflight": (
                {
                    "status": (
                        "pass"
                        if status == "safe_pass"
                        else "over_memory_ceiling"
                    ),
                    "gpu": {
                        "name": (
                            "NVIDIA RTX PRO 6000 Blackwell "
                            "Workstation Edition"
                        ),
                        "pbs_gpu_type": "RTX_PRO_6000_BLACKWELL",
                        "compute_capability": [12, 0],
                        "total_memory_bytes": 96 * 1024**3,
                    },
                    "versions": {
                        "torch": "2.7.1",
                        "mujoco": "3.10.0",
                        "mujoco-warp": "3.10.0.3",
                        "warp-lang": "1.14.0",
                    },
                }
                if status != "fail"
                else None
            ),
        }

    def test_cx3_gpu_names_map_to_scheduler_types(self):
        self.assertEqual(classify_gpu_type("NVIDIA L40S"), "L40S")
        self.assertEqual(
            classify_gpu_type("NVIDIA A100-PCIE-40GB"), "A100"
        )
        self.assertEqual(
            classify_gpu_type("Quadro RTX 6000"), "RTX6000"
        )
        for name in (
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    classify_gpu_type(name),
                    "RTX_PRO_6000_BLACKWELL",
                )
        self.assertEqual(classify_gpu_type("NVIDIA H100"), "UNKNOWN")

    def test_derived_config_changes_only_name_and_world_count(self):
        base = {
            "name": "cx3_l40s",
            "physics": {"worlds": 1024, "physics_substeps": 40},
            "diffusion": {"hidden_dim": 256},
        }
        original = copy.deepcopy(base)
        result = derive_config(
            base, worlds=4096, gpu_name="NVIDIA A100-PCIE-40GB"
        )
        self.assertEqual(base, original)
        self.assertEqual(result["physics"]["worlds"], 4096)
        self.assertEqual(
            result["name"],
            "cx3_adaptive_nvidia-a100-pcie-40gb_w4096",
        )
        self.assertEqual(result["physics"]["physics_substeps"], 40)
        self.assertEqual(result["diffusion"], base["diffusion"])

    def test_aggressive_budget_scaling_preserves_vector_step_ratios(self):
        base = json.loads(
            (
                PACKAGE_ROOT
                / "configs"
                / "autodl_rtx_pro_6000_blackwell.json"
            ).read_text(encoding="utf-8")
        )
        expected = {
            2048: (10_000_000, 131_072, 2_000_000, 2048, 500_000),
            4096: (20_000_000, 262_144, 4_000_000, 4096, 1_000_000),
            8192: (40_000_000, 524_288, 8_000_000, 8192, 2_000_000),
            16384: (
                80_000_000,
                1_048_576,
                16_000_000,
                16384,
                4_000_000,
            ),
        }
        for worlds, values in expected.items():
            with self.subTest(worlds=worlds):
                derived = derive_config(
                    base,
                    worlds=worlds,
                    gpu_name="capacity-candidate",
                    scale_world_budgets=True,
                )
                tqc = derived["tqc"]
                self.assertEqual(
                    (
                        tqc["total_transitions"],
                        tqc["learning_starts"],
                        tqc["replay_capacity"],
                        tqc["batch_size"],
                        tqc["checkpoint_every_transitions"],
                    ),
                    values,
                )
                self.assertAlmostEqual(
                    tqc["total_transitions"] / worlds,
                    5_000_000 / 1024,
                )
                self.assertEqual(derived["demos"], base["demos"])
                self.assertEqual(derived["diffusion"], base["diffusion"])

    def test_extreme_budget_scaling_covers_all_new_candidates(self):
        base = json.loads(
            (
                PACKAGE_ROOT
                / "configs"
                / "autodl_rtx_pro_6000_extreme.json"
            ).read_text(encoding="utf-8")
        )
        expected = {
            8000: (40_000_000, 512_000, 8_000_000, 8000, 2_000_000),
            10000: (
                50_000_000,
                640_000,
                10_000_000,
                10000,
                2_500_000,
            ),
            12000: (
                60_000_000,
                768_000,
                12_000_000,
                12000,
                3_000_000,
            ),
            14000: (
                70_000_000,
                896_000,
                14_000_000,
                14000,
                3_500_000,
            ),
        }
        for worlds, values in expected.items():
            with self.subTest(worlds=worlds):
                derived = derive_config(
                    base,
                    worlds=worlds,
                    gpu_name="capacity-candidate",
                    scale_world_budgets=True,
                )
                tqc = derived["tqc"]
                self.assertEqual(
                    (
                        tqc["total_transitions"],
                        tqc["learning_starts"],
                        tqc["replay_capacity"],
                        tqc["batch_size"],
                        tqc["checkpoint_every_transitions"],
                    ),
                    values,
                )
                self.assertEqual(derived["demos"]["episodes"], 8192)
                self.assertEqual(derived["diffusion"]["epochs"], 60)
                self.assertEqual(
                    derived["diffusion"]["batch_size"], 4096
                )

    def test_diffusion_only_capacity_stops_at_first_unsafe_candidate(
        self,
    ):
        candidates = [12000, 14000, 16000, 18000, 20000]
        base_path = (
            PACKAGE_ROOT
            / "configs"
            / "autodl_rtx_pro_6000_extreme.json"
        )
        base = json.loads(base_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "adaptive_capacity._run_child",
                side_effect=lambda **kwargs: self._record(
                    kwargs["worlds"],
                    (
                        "raw_pass_over_ceiling"
                        if kwargs["worlds"] == 18000
                        else "safe_pass"
                    ),
                ),
            ) as runner:
                result = probe(
                    base_config=base_path,
                    run_dir=root,
                    worlds=candidates,
                    memory_ceiling=0.90,
                    physics_steps=256,
                    timeout_seconds=10,
                    require_worlds=12000,
                    profile_id=(
                        "autodl_rtx_pro_6000_extreme_v8_"
                        "diffusion_only_spawn60_c12k_step2k_first_unsafe"
                    ),
                    pipeline_mode="diffusion-only",
                    stop_after_first_unsafe=True,
                )
            selection = result["selection"]
            self.assertEqual(runner.call_count, 4)
            self.assertEqual(
                [call.kwargs["pipeline_mode"] for call in runner.call_args_list],
                ["diffusion-only"] * 4,
            )
            self.assertEqual(
                selection["safe_qualified_worlds"], candidates[:3]
            )
            self.assertEqual(
                selection["attempted_worlds"], candidates[:4]
            )
            self.assertEqual(selection["first_unsafe_worlds"], 18000)
            self.assertTrue(selection["stopped_after_first_unsafe"])
            self.assertFalse(selection["candidate_cap_reached"])
            self.assertEqual(
                selection["pipeline_mode"], "diffusion-only"
            )
            self.assertEqual(
                selection["budget_policy"],
                "diffusion_only_worlds_v1",
            )
            _selection, candidate_path, selected = (
                resolve_qualified_config(
                    root,
                    16000,
                    expected_pipeline_mode="diffusion-only",
                    expected_candidates=candidates,
                    expected_budget_policy="diffusion_only_worlds_v1",
                )
            )
            candidate = json.loads(
                candidate_path.read_text(encoding="utf-8")
            )
            self.assertEqual(selected, 16000)
            self.assertEqual(candidate["physics"]["worlds"], 16000)
            self.assertEqual(candidate["tqc"], base["tqc"])

    def test_first_unsafe_scan_refuses_to_treat_safe_cap_as_maximum(self):
        candidates = [12000, 14000]
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "adaptive_capacity._run_child",
                side_effect=lambda **kwargs: self._record(
                    kwargs["worlds"]
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "final safety candidate"
                ):
                    probe(
                        base_config=(
                            PACKAGE_ROOT
                            / "configs"
                            / "autodl_rtx_pro_6000_extreme.json"
                        ),
                        run_dir=Path(directory),
                        worlds=candidates,
                        memory_ceiling=0.90,
                        physics_steps=256,
                        timeout_seconds=10,
                        require_worlds=12000,
                        pipeline_mode="diffusion-only",
                        stop_after_first_unsafe=True,
                    )

    def test_summary_keeps_raw_max_separate_from_safe_selection(self):
        records = [
            {"worlds": 512, "status": "safe_pass"},
            {"worlds": 1024, "status": "safe_pass"},
            {"worlds": 2048, "status": "raw_pass_over_ceiling"},
            {"worlds": 4096, "status": "fail"},
        ]
        result = summarize(
            records, [512, 1024, 2048, 4096, 8192]
        )
        self.assertEqual(result["selected_training_worlds"], 1024)
        self.assertEqual(result["raw_maximum_completed_worlds"], 2048)
        self.assertEqual(result["first_failed_worlds"], 4096)
        self.assertEqual(result["first_unsafe_worlds"], 2048)
        self.assertEqual(
            result["attempted_worlds"], [512, 1024, 2048, 4096]
        )
        self.assertFalse(result["all_requested_completed"])

    def test_gate_rejects_a_different_gpu_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "adaptive_capacity._run_child",
                side_effect=lambda **kwargs: self._record(
                    kwargs["worlds"]
                ),
            ):
                probe(
                    base_config=(
                        PACKAGE_ROOT
                        / "configs"
                        / "autodl_rtx_pro_6000_blackwell.json"
                    ),
                    run_dir=root,
                    worlds=[2048],
                    memory_ceiling=0.88,
                    physics_steps=256,
                    timeout_seconds=10,
                    require_worlds=2048,
                    scale_world_budgets=True,
                    profile_id="autodl_rtx_pro_6000_aggressive_v1",
                )
            current = {
                "status": "safe_pass",
                "preflight": {
                    "gpu": {
                        "pbs_gpu_type": "A100",
                        "compute_capability": [8, 0],
                        "total_memory_bytes": 80 * 1024**3,
                    }
                },
            }
            with patch("adaptive_capacity._run_child", return_value=current):
                with self.assertRaisesRegex(
                    RuntimeError, "GPU fingerprint"
                ):
                    gate(
                        run_dir=root,
                        timeout_seconds=10,
                        requested_worlds=2048,
                    )

    def test_probe_signs_all_safe_candidates_and_resolves_without_resweep(self):
        candidates = [2048, 4096, 8192, 16384]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "adaptive_capacity._run_child",
                side_effect=lambda **kwargs: self._record(
                    kwargs["worlds"]
                ),
            ) as runner:
                result = probe(
                    base_config=(
                        PACKAGE_ROOT
                        / "configs"
                        / "autodl_rtx_pro_6000_blackwell.json"
                    ),
                    run_dir=root,
                    worlds=candidates,
                    memory_ceiling=0.88,
                    physics_steps=256,
                    timeout_seconds=10,
                    require_worlds=2048,
                    scale_world_budgets=True,
                    profile_id="autodl_rtx_pro_6000_aggressive_v1",
                )
            selection = result["selection"]
            self.assertEqual(
                selection["safe_qualified_worlds"], candidates
            )
            self.assertEqual(
                selection["maximum_qualified_candidate"], 16384
            )
            self.assertTrue(selection["candidate_cap_reached"])
            self.assertEqual(runner.call_count, 4)
            _selection, config_path, worlds = resolve_qualified_config(
                root,
                8192,
                expected_profile_id=(
                    "autodl_rtx_pro_6000_aggressive_v1"
                ),
                expected_candidates=candidates,
                expected_budget_policy="scale_tqc_with_worlds_v1",
                minimum_sweep_physics_steps=216,
                maximum_memory_ceiling=0.88,
            )
            self.assertEqual(worlds, 8192)
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["physics"]["worlds"], 8192)
            self.assertEqual(payload["tqc"]["replay_capacity"], 8_000_000)
            self.assertIn(
                "s_hpc/env.py",
                selection["training_contract_files"],
            )
            self.assertIn(
                "s_hpc/mjwarp_compat.py",
                selection["training_contract_files"],
            )
            self.assertIn(
                "s_hpc/torch_compat.py",
                selection["training_contract_files"],
            )
            self.assertIn(
                "s_hpc/replay.py",
                selection["training_contract_files"],
            )
            self.assertIn(
                "s_hpc/tqc.py",
                selection["training_contract_files"],
            )
            self.assertEqual(
                {
                    name
                    for name in selection["training_contract_files"]
                    if name.startswith("assets/mesh/")
                },
                {
                    "assets/mesh/backbased_mujoco.STL",
                    "assets/mesh/base_mujoco.STL",
                    "assets/mesh/frontbased_mujoco.STL",
                    "assets/mesh/handbase_module_mujoco.STL",
                    "assets/mesh/linear_module_mujoco.STL",
                    "assets/mesh/long.STL",
                    "assets/mesh/short.STL",
                },
            )
            with self.assertRaisesRegex(ValueError, "budget policy mismatch"):
                resolve_qualified_config(
                    root,
                    8192,
                    expected_budget_policy="worlds_only_v1",
                )
            with self.assertRaisesRegex(ValueError, "candidates mismatch"):
                resolve_qualified_config(
                    root,
                    8192,
                    expected_candidates=[2048, 4096, 8192],
                )
            changed_fingerprints = {
                "training_contract_hash": "changed",
                "training_contract_files": selection[
                    "training_contract_files"
                ],
                "source_tree_sha256": selection["source_tree_sha256"],
                "source_file_count": selection["source_file_count"],
            }
            with patch(
                "adaptive_capacity._source_fingerprints",
                return_value=changed_fingerprints,
            ):
                with self.assertRaisesRegex(
                    ValueError, "training contract changed"
                ):
                    resolve_qualified_config(root, 8192)

    def test_probe_scans_every_isolated_candidate_after_hard_failure(self):
        statuses = {
            2048: "safe_pass",
            4096: "raw_pass_over_ceiling",
            8192: "fail",
            16384: "fail",
        }
        calls = []

        def child(**kwargs):
            worlds = kwargs["worlds"]
            calls.append(worlds)
            return self._record(worlds, statuses[worlds])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("adaptive_capacity._run_child", side_effect=child):
                result = probe(
                    base_config=(
                        PACKAGE_ROOT
                        / "configs"
                        / "autodl_rtx_pro_6000_blackwell.json"
                    ),
                    run_dir=root,
                    worlds=[2048, 4096, 8192, 16384],
                    memory_ceiling=0.88,
                    physics_steps=256,
                    timeout_seconds=10,
                    require_worlds=2048,
                    scale_world_budgets=True,
                    profile_id="autodl_rtx_pro_6000_aggressive_v1",
                )
            self.assertEqual(calls, [2048, 4096, 8192, 16384])
            self.assertEqual(
                result["selection"]["safe_qualified_worlds"], [2048]
            )
            with self.assertRaisesRegex(ValueError, "not safely qualified"):
                resolve_qualified_config(root, 4096)

    def test_timeout_tail_decodes_bytes_for_json_records(self):
        self.assertEqual(_tail_text(b"abcdef", 3), "def")


class FormalMemoryCoverageTest(unittest.TestCase):
    def test_demo_generation_uses_one_float16_trajectory_buffer(self):
        from s_hpc.config import PACKAGE_ROOT

        source = (PACKAGE_ROOT / "s_hpc" / "workflows.py").read_text(
            encoding="utf-8"
        )
        start = source.index("def generate_demos(")
        stop = source.index("\ndef train_diffusion(", start)
        body = source[start:stop]
        self.assertIn("dtype=torch.float16", body)
        self.assertIn("observation_tensor[:, step_index + 1].copy_", body)
        self.assertIn("action_tensor[:, step_index].copy_", body)
        self.assertNotIn("observations.append", body)
        self.assertNotIn("torch.stack(observations", body)
        self.assertIn(
            "config.physics.worlds, episodes - start", body
        )
        self.assertIn(
            "env.worlds != simulation_worlds", body
        )

    def test_formal_probe_covers_diffusion_only_peak_phases(self):
        from s_hpc.config import PACKAGE_ROOT

        source = (
            PACKAGE_ROOT / "training_stack_preflight.py"
        ).read_text(encoding="utf-8")
        self.assertIn("demonstration_generation", source)
        self.assertIn("expert_physical_canary", source)
        self.assertIn("diffusion_training", source)
        self.assertIn("diffusion_only_rollout", source)
        self.assertIn("ExpertRolloutAudit", source)
        self.assertIn("certify_expert_rollout", source)
        self.assertIn("worlds = config.physics.worlds", source)
        self.assertIn(
            "worlds = min(config.physics.worlds, config.demos.episodes)",
            source,
        )
        start = source.index("def _diffusion_rollout_phase(")
        stop = source.index("\ndef run_probe(", start)
        diffusion_only = source[start:stop]
        self.assertIn("env.step(zero_residual)", diffusion_only)
        self.assertNotIn("CUDAReplayBuffer", diffusion_only)
        self.assertNotIn("TQCTrainer", diffusion_only)
        self.assertNotIn("trainer.update", diffusion_only)

    def test_formal_probe_synchronizes_warp_between_phases(self):
        source = (
            PACKAGE_ROOT / "training_stack_preflight.py"
        ).read_text(encoding="utf-8")
        start = source.index("def _release_cuda(")
        stop = source.index("\ndef _assert_health(", start)
        release = source[start:stop]
        self.assertIn("wp.set_mempool_release_threshold", release)
        self.assertIn("wp.synchronize_device", release)
        self.assertLess(
            release.index("gc.collect()"),
            release.index("wp.synchronize_device"),
        )

    def test_residual_probe_uses_full_host_replay_and_real_learner_batch(self):
        source = (
            PACKAGE_ROOT / "training_stack_preflight.py"
        ).read_text(encoding="utf-8")
        start = source.index("def _residual_phase(")
        stop = source.index("\ndef _diffusion_rollout_phase(", start)
        body = source[start:stop]
        self.assertIn("replay.size = replay.capacity", body)
        self.assertIn('replay.storage_device.type != "cpu"', body)
        self.assertIn("replay.sample(config.tqc.batch_size)", body)
        self.assertIn(
            "trainer.transitions = int(config.tqc.actor_learning_starts)",
            body,
        )
        self.assertIn('"replay_host_allocated_bytes"', body)
        self.assertIn('"replay_cuda_persistent_allocated_bytes": 0', body)
        self.assertIn('"replay_cuda_sample_batch_bytes"', body)
        self.assertNotIn("_engaged_batch", body)

    def test_formal_probe_isolates_each_peak_phase_in_a_process(self):
        source = (
            PACKAGE_ROOT / "training_stack_preflight.py"
        ).read_text(encoding="utf-8")
        run_start = source.index("def run_probe(")
        run_stop = source.index("\ndef main(", run_start)
        run_body = source[run_start:run_stop]
        self.assertIn("_run_phase_subprocess(", run_body)
        self.assertNotIn("phase = runner(", run_body)
        self.assertIn('"--single-phase"', source)
        self.assertIn("subprocess.run(", source)

    def test_trajectory_buffers_belong_only_to_demo_probe(self):
        from s_hpc.config import PACKAGE_ROOT

        source = (
            PACKAGE_ROOT / "training_stack_preflight.py"
        ).read_text(encoding="utf-8")
        expert_start = source.index("def _expert_physical_phase(")
        demo_start = source.index("\ndef _demo_phase(", expert_start)
        diffusion_start = source.index(
            "\ndef _diffusion_training_phase(", demo_start
        )
        expert_body = source[expert_start:demo_start]
        demo_body = source[demo_start:diffusion_start]

        self.assertNotIn("observations = torch.empty", expert_body)
        self.assertNotIn("actions = torch.empty", expert_body)
        self.assertIn("observations = torch.empty", demo_body)
        self.assertIn("actions = torch.empty", demo_body)
        self.assertIn(
            "observations[:, step_index + 1].copy_", demo_body
        )
        self.assertIn("actions[:, step_index].copy_", demo_body)
        self.assertLess(
            demo_body.index("observations = torch.empty"),
            demo_body.index("for step_index in range(steps_run)"),
        )


if __name__ == "__main__":
    unittest.main()
