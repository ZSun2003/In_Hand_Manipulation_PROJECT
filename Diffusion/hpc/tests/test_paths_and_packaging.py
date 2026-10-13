from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

from s_hpc.config import PACKAGE_ROOT, load_config
from s_hpc.paths import resolve_package_path
from preflight import _time_advance_tolerance, static_checks


class PackagePathPolicyTest(unittest.TestCase):
    def test_gpu_time_tolerance_allows_float32_drift_not_a_lost_tick(self):
        tolerance = _time_advance_tolerance(0.0005)
        self.assertGreater(tolerance, 4.8e-5)
        self.assertLess(tolerance, 0.0005)
        with self.assertRaises(ValueError):
            _time_advance_tolerance(0.0)

    def test_relative_paths_are_anchored_inside_package(self):
        selected = resolve_package_path("runs/example")
        self.assertEqual(selected, (PACKAGE_ROOT / "runs" / "example").resolve())

    def test_external_path_requires_explicit_opt_in(self):
        outside = (PACKAGE_ROOT.parent / "outside-s-hpc").resolve()
        with self.assertRaisesRegex(ValueError, "--allow-external-paths"):
            resolve_package_path(outside, label="test output")
        self.assertEqual(
            resolve_package_path(outside, allow_external=True),
            outside,
        )

    def test_all_path_entrypoints_expose_escape_hatch(self):
        entrypoints = (
            "pipeline.py",
            "preflight.py",
            "capacity_probe.py",
            "adaptive_capacity.py",
            "training_stack_preflight.py",
            "status_hpc.py",
            "system_monitor.py",
            "job_status.py",
            "collect_diagnostics.py",
            "generate_demos_gpu.py",
            "train_diffusion_gpu.py",
            "train_residual_gpu.py",
            "evaluate_gpu.py",
        )
        for filename in entrypoints:
            with self.subTest(filename=filename):
                source = (PACKAGE_ROOT / filename).read_text(encoding="utf-8")
                self.assertIn("add_external_path_flag", source)

    def test_static_preflight_ignores_generated_cache_sources(self):
        cache = PACKAGE_ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        probe = cache / "generated-null-probe.py"
        apple_double = PACKAGE_ROOT / "._generated-null-probe.py"
        probe.write_bytes(b"generated\x00binary")
        apple_double.write_bytes(b"apple-double\x00binary")
        try:
            result = static_checks(
                PACKAGE_ROOT / "configs" / "smoke.json"
            )
        finally:
            probe.unlink(missing_ok=True)
            apple_double.unlink(missing_ok=True)
        self.assertEqual(result["status"], "pass")


class SetupScriptPolicyTest(unittest.TestCase):
    def test_setup_only_validates_external_shared_environment(self):
        script = PACKAGE_ROOT / "setup_hpc.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        source = script.read_text(encoding="utf-8")
        self.assertIn("screw_use_core_env", source)
        self.assertIn('"${PYTHON}" -m pip check', source)
        self.assertIn("environment_mode=shared-read-only", source)
        self.assertNotIn('"${PYTHON}" -m pip install', source)
        self.assertNotIn("-m venv", source)

        helper = (PACKAGE_ROOT / "hpc_env.sh").read_text(encoding="utf-8")
        self.assertIn(
            "${SCREW_HPC_PACKAGE_DIR}/../../RL_env/profiles/core/RL_env",
            helper,
        )
        self.assertIn("refusing a project-local HPC environment", helper)

    def test_native_stack_is_exactly_pinned(self):
        requirements = (
            PACKAGE_ROOT / "requirements-hpc.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("mujoco-warp==3.10.0.3", requirements)
        self.assertIn("mujoco==3.10.0", requirements)
        self.assertIn("warp-lang==1.14.0", requirements)
        self.assertIn("torch==2.7.1", requirements)

    def test_autodl_scripts_are_valid_and_scheduler_independent(self):
        setup = PACKAGE_ROOT / "setup_autodl_rtx_pro_6000.sh"
        runner = PACKAGE_ROOT / "autodl_rtx_pro_6000.sh"
        builder = PACKAGE_ROOT / "build_autodl_package.sh"
        generic_runner = PACKAGE_ROOT / "run_hpc.sh"
        generic_builder = PACKAGE_ROOT / "build_hpc_package.sh"
        for script in (
            setup,
            runner,
            builder,
            generic_runner,
            generic_builder,
        ):
            with self.subTest(script=script.name):
                self.assertTrue(script.stat().st_mode & 0o111)
                subprocess.run(["bash", "-n", str(script)], check=True)

        setup_source = setup.read_text(encoding="utf-8")
        self.assertIn("/root/autodl-tmp", setup_source)
        self.assertIn("SCREW_AUTODL_MIN_FREE_GIB:-110", setup_source)
        self.assertIn('TMPDIR="${SCRIPT_DIR}/.cache/tmp"', setup_source)
        self.assertIn("screw_use_core_env", setup_source)
        self.assertNotIn("pip install", setup_source)
        self.assertIn("libEGL.so.1", setup_source)
        self.assertIn("apt-get install -y libegl1", setup_source)
        self.assertIn("autodl_rtx_pro_6000_preflight.py", setup_source)
        self.assertIn('"${NORMALIZED_NAME}" == *MAXQ*', setup_source)
        self.assertIn("conservative deployment gate", setup_source)
        hardware_source = (
            PACKAGE_ROOT / "autodl_rtx_pro_6000_preflight.py"
        ).read_text(encoding="utf-8")
        self.assertIn("warp_arch != 120", hardware_source)
        self.assertIn("warp_toolkit != (12, 9)", hardware_source)
        self.assertIn("run_warp_kernel_smoke", hardware_source)

        runner_source = runner.read_text(encoding="utf-8")
        for forbidden in ("qsub", "module load", "PBS_", "SLURM_"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, runner_source)
        self.assertIn("nohup", runner_source)
        self.assertIn("flock -n", runner_source)
        self.assertIn("--request-stop", runner_source)
        self.assertIn("--resume auto", runner_source)
        self.assertIn(
            'candidate_worlds <= 100000; candidate_worlds += 2000',
            runner_source,
        )
        self.assertIn('--worlds "${CAPACITY_CANDIDATES[@]}"', runner_source)
        self.assertIn("--stop-after-first-unsafe", runner_source)
        self.assertIn("--require-worlds 12000", runner_source)
        self.assertIn(
            "autodl_rtx_pro_6000_extreme_v8_diffusion_only_spawn60_c12k_step2k_first_unsafe",
            runner_source,
        )
        self.assertNotIn("--scale-world-budgets", runner_source)
        self.assertIn("--pipeline-mode diffusion-only", runner_source)
        self.assertIn("--require-pipeline-mode diffusion-only", runner_source)
        self.assertIn(
            "--require-budget-policy diffusion_only_worlds_v1",
            runner_source,
        )
        self.assertIn(
            '--require-candidates "${CAPACITY_CANDIDATES[@]}"',
            runner_source,
        )
        self.assertIn(
            'SCREW_AUTODL_SWEEP_PHYSICS_STEPS:-256',
            runner_source,
        )
        self.assertIn(
            'SCREW_AUTODL_GATE_PHYSICS_STEPS:-3000',
            runner_source,
        )
        self.assertIn(
            'SCREW_AUTODL_MEMORY_CEILING:-0.90',
            runner_source,
        )
        self.assertIn(
            'MAXIMUM_MEMORY_CEILING="0.90"',
            runner_source,
        )
        self.assertIn(
            "FORMAL_GATE_PHYSICS_STEPS < 3000",
            runner_source,
        )
        self.assertEqual(
            runner_source.count(
                '--maximum-memory-ceiling "${MAXIMUM_MEMORY_CEILING}"'
            ),
            2,
        )
        self.assertIn(
            'CUBLAS_WORKSPACE_CONFIG:-:4096:8',
            runner_source,
        )
        self.assertIn(
            'SCREW_AUTODL_RUN_MIN_FREE_GIB:-80',
            runner_source,
        )
        self.assertIn(
            'SCREW_EVAL_EPISODES:-2000',
            runner_source,
        )
        self.assertIn('SCREW_AUTODL_WORLDS:-', runner_source)
        self.assertIn('WORLD_REQUEST="${WORLD_REQUEST:-12000}"', runner_source)
        self.assertIn(
            "autodl_rtx_pro_6000_extreme_v8_diffusion_only_control",
            runner_source,
        )
        self.assertIn(
            "autodl_rtx_pro_6000_extreme_v8_diffusion_only_capacity",
            runner_source,
        )
        self.assertIn(
            "autodl_rtx_pro_6000_extreme_v8_diffusion_only_${RUN_VARIANT}_smoke",
            runner_source,
        )
        self.assertIn(
            "autodl_rtx_pro_6000_extreme_v8_diffusion_only_${RUN_VARIANT}_full",
            runner_source,
        )
        self.assertIn(
            "configs/autodl_rtx_pro_6000_extreme.json",
            runner_source,
        )
        self.assertIn(
            'SCREW_AUTODL_RUN_MIN_FREE_GIB must be an integer >= 50',
            runner_source,
        )
        self.assertIn(
            'Extreme V8 Diffusion-only requires at least ${MINIMUM_FREE_GIB} GiB',
            runner_source,
        )
        self.assertIn('--worlds "${WORLD_REQUEST}"', runner_source)
        self.assertIn('--config "${SELECTED_CONFIG}"', runner_source)
        self.assertLess(
            runner_source.index(
                "run_platform_qualification\n"
                "  run_smoke_chain\n"
                "  run_capacity_qualification"
            ),
            runner_source.index("--stage diffusion-full"),
        )
        smoke_start = runner_source.index("run_smoke_chain() {")
        smoke_end = runner_source.index("\n}\n\nforeground_run()", smoke_start)
        smoke_body = runner_source[smoke_start:smoke_end]
        self.assertEqual(smoke_body.count('configs/smoke.json"'), 2)
        self.assertEqual(smoke_body.count("--stage diffusion-smoke"), 2)
        self.assertNotIn("resume-probe", smoke_body)
        self.assertNotIn("SELECTED_CONFIG", smoke_body)

        builder_source = builder.read_text(encoding="utf-8")
        for excluded in (
            "RL_env",
            ".venv",
            "pyvenv.cfg",
            "site-packages",
            "conda-meta",
            ".python",
            "_g6_python_runtime",
            ".wheelhouse",
            ".cache",
            ".pytest_cache",
            "runs",
            "datasets",
            "checkpoints",
            "environment-manifest.txt",
            "__pycache__",
            ".DS_Store",
        ):
            with self.subTest(excluded=excluded):
                self.assertIn(excluded, builder_source)
        self.assertIn("sha256sum", builder_source)
        self.assertIn("COPYFILE_DISABLE=1", builder_source)
        self.assertIn("--no-mac-metadata", builder_source)
        self.assertIn("--no-xattrs", builder_source)
        self.assertIn("must be outside", builder_source)
        self.assertIn("mktemp", builder_source)
        self.assertIn(
            "AUTODL_H110_MOUTH330_W18000_EP4000_BLEND100_"
            "HX1_COMPAT",
            builder_source,
        )
        self.assertIn('mv -f -- "${TEMP_ARCHIVE}" "${ARCHIVE}"', builder_source)
        self.assertIn("-name pyvenv.cfg", builder_source)
        self.assertIn("-name conda-meta", builder_source)
        self.assertIn("discovered environment leaked", builder_source)
        self.assertIn("SCREW_PACKAGE_REQUIRE_MODELS", builder_source)
        self.assertIn("model_artifact_members=", builder_source)
        self.assertIn("required model artifacts are missing", builder_source)
        self.assertNotIn("Model artifacts included:", builder_source)
        generic_builder_source = generic_builder.read_text(encoding="utf-8")
        self.assertIn(
            'SCREW_PACKAGE_INCLUDE_MODELS="${SCREW_PACKAGE_INCLUDE_MODELS:-0}"',
            generic_builder_source,
        )
        readme_source = (
            PACKAGE_ROOT / "README_AUTODL_RTX_PRO_6000.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "hpc_AUTODL_H110_MOUTH330_W18000_EP4000_"
            "BLEND100_HX1_COMPAT_20260809.tar.gz",
            readme_source,
        )

    def test_cx3_jobs_leave_queue_selection_to_pbs_routing(self):
        for filename in (
            "cx3_capacity.pbs",
            "cx3_smoke.pbs",
            "cx3_full.pbs",
            "cx3_eval.pbs",
            "cx3_capacity_8192.pbs",
            "cx3_smoke_8192.pbs",
            "cx3_full_8192.pbs",
        ):
            with self.subTest(filename=filename):
                script = PACKAGE_ROOT / "jobs" / filename
                subprocess.run(["bash", "-n", str(script)], check=True)
                source = script.read_text(encoding="utf-8")
                self.assertNotIn("#PBS -q", source)
                self.assertIn("gpu_type=L40S", source)

    def test_hx1_w14000_package_is_isolated_and_fail_closed(self):
        config = load_config(
            PACKAGE_ROOT
            / "configs"
            / "hx1_a100_w14000_diffusion_only.json"
        )
        autodl = load_config(
            PACKAGE_ROOT
            / "configs"
            / "autodl_rtx_pro_6000_h105_w18000.json"
        )
        self.assertEqual(config.physics.worlds, 14000)
        self.assertEqual(autodl.physics.worlds, 18000)
        self.assertEqual(config.physics.max_episode_steps, 4000)
        self.assertEqual(config.demos, autodl.demos)
        self.assertEqual(config.diffusion, autodl.diffusion)

        job = PACKAGE_ROOT / "jobs" / "hx1_w14000_diffusion_only_full.pbs"
        subprocess.run(["bash", "-n", str(job)], check=True)
        job_source = job.read_text(encoding="utf-8")
        self.assertIn("gpu_type=A100", job_source)
        self.assertIn("--worlds 14000", job_source)
        self.assertIn("--max-memory-fraction", job_source)
        self.assertIn("--physics-steps", job_source)
        self.assertIn("--pipeline-mode diffusion-only", job_source)
        self.assertIn("--stage diffusion-full", job_source)
        self.assertNotIn("--stage full", job_source)
        self.assertIn("--resume auto", job_source)

        builder = PACKAGE_ROOT / "build_hx1_a100_w14000_package.sh"
        self.assertTrue(builder.stat().st_mode & 0o111)
        subprocess.run(["bash", "-n", str(builder)], check=True)
        builder_source = builder.read_text(encoding="utf-8")
        self.assertIn("SCREW_PACKAGE_INCLUDE_MODELS=0", builder_source)
        self.assertIn("HX1_A100_W14000_DIFFUSION_ONLY", builder_source)

        common = (PACKAGE_ROOT / "jobs" / "_common.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("module is-avail tools/prod", common)

    def test_unattended_cx3_chain_is_valid_and_fail_closed(self):
        script = PACKAGE_ROOT / "submit_cx3_chain.sh"
        self.assertTrue(script.stat().st_mode & 0o111)
        subprocess.run(["bash", "-n", str(script)], check=True)
        source = script.read_text(encoding="utf-8")
        self.assertIn('depend=afterok:${CAP_JOB}', source)
        self.assertIn('depend=afterok:${SMOKE_JOB}', source)
        self.assertIn("runs/submissions/latest.env", source)
        self.assertIn("write_record", source)
        self.assertNotIn("gpu72", source)

    def test_aggressive_8192_chain_is_isolated_and_fail_closed(self):
        script = PACKAGE_ROOT / "submit_cx3_8192_chain.sh"
        self.assertTrue(script.stat().st_mode & 0o111)
        subprocess.run(["bash", "-n", str(script)], check=True)
        source = script.read_text(encoding="utf-8")
        self.assertIn("cx3_capacity_8192.pbs", source)
        self.assertIn("cx3_smoke_8192.pbs", source)
        self.assertIn("cx3_full_8192.pbs", source)
        self.assertIn('depend=afterok:${CAP_JOB}', source)
        self.assertIn('depend=afterok:${SMOKE_JOB}', source)
        self.assertIn("runs/submissions/latest_8192.env", source)
        self.assertNotIn("runs/submissions/latest.env", source)

        capacity = (
            PACKAGE_ROOT / "jobs" / "cx3_capacity_8192.pbs"
        ).read_text(encoding="utf-8")
        self.assertIn("--worlds 1024 2048 4096 8192", capacity)
        self.assertIn("--require-worlds 8192", capacity)
        self.assertIn(".cache/warp_8192", capacity)
        smoke = (
            PACKAGE_ROOT / "jobs" / "cx3_smoke_8192.pbs"
        ).read_text(encoding="utf-8")
        self.assertIn("--worlds 8192", smoke)
        self.assertIn(".cache/warp_8192", smoke)
        full = (
            PACKAGE_ROOT / "jobs" / "cx3_full_8192.pbs"
        ).read_text(encoding="utf-8")
        self.assertIn("configs/cx3_l40s_8192.json", full)
        self.assertIn("runs/cx3_l40s_8192_full", full)
        self.assertIn("runs/.cx3_l40s_8192_full.lock", full)
        self.assertIn(".cache/warp_8192", full)

    def test_adaptive_any_gpu_job_is_one_allocation_and_not_type_pinned(self):
        submit = PACKAGE_ROOT / "submit_cx3_adaptive_any.sh"
        job = PACKAGE_ROOT / "jobs" / "cx3_adaptive_any.pbs"
        self.assertTrue(submit.stat().st_mode & 0o111)
        subprocess.run(["bash", "-n", str(submit)], check=True)
        subprocess.run(["bash", "-n", str(job)], check=True)
        submit_source = submit.read_text(encoding="utf-8")
        job_source = job.read_text(encoding="utf-8")
        self.assertIn("SCREW_ADAPTIVE_ID", submit_source)
        self.assertIn("latest_adaptive_any.env", submit_source)
        self.assertIn("ngpus=1", job_source)
        self.assertNotIn("gpu_type=", job_source)
        self.assertIn("--worlds 512 1024 2048 4096 8192", job_source)
        self.assertIn('"${SCREW_HPC_ROOT}/adaptive_capacity.py" gate', job_source)
        self.assertIn("--stage smoke", job_source)
        self.assertIn("--stage full", job_source)
        self.assertLess(
            job_source.index("--stage smoke"),
            job_source.index("--stage full"),
        )
        self.assertIn("selected_config.json", job_source)

    def test_allocation_log_is_append_only_and_still_streamed(self):
        common = PACKAGE_ROOT / "jobs" / "_common.sh"
        subprocess.run(["bash", "-n", str(common)], check=True)
        source = common.read_text(encoding="utf-8")
        self.assertIn(
            'SCREW_ALLOCATION_LOG="${SCREW_HPC_ROOT}/logs/'
            'allocation-${job_tag}.log"',
            source,
        )
        self.assertIn(': >>"${SCREW_ALLOCATION_LOG}"', source)
        self.assertIn(
            'exec > >(tee -a -- "${SCREW_ALLOCATION_LOG}") 2>&1',
            source,
        )
        self.assertIn("screw_start_allocation_log", source)

    def test_exit_trap_creates_a_small_diagnostic_bundle(self):
        common = PACKAGE_ROOT / "jobs" / "_common.sh"
        subprocess.run(["bash", "-n", str(common)], check=True)
        source = common.read_text(encoding="utf-8")
        self.assertIn("screw_collect_diagnostics", source)
        self.assertIn(
            '"${SCREW_HPC_ROOT}/collect_diagnostics.py"',
            source,
        )
        trap_start = source.index("screw_hpc_exit_trap()")
        self.assertIn(
            'screw_collect_diagnostics "${exit_code}"',
            source[trap_start:],
        )

    def test_jobs_reload_the_python_module_used_by_the_venv(self):
        common = PACKAGE_ROOT / "jobs" / "_common.sh"
        subprocess.run(["bash", "-n", str(common)], check=True)
        source = common.read_text(encoding="utf-8")
        self.assertIn("screw_load_python_module", source)
        self.assertIn("module purge", source)
        self.assertIn("module load tools/prod", source)
        self.assertIn(
            "Python/3.11.5-GCCcore-13.2.0",
            source,
        )
        self.assertLess(
            source.index("screw_load_python_module", source.index("screw_hpc_init()")),
            source.index("screw_use_core_env", source.index("screw_hpc_init()")),
        )

    def test_every_pbs_job_initializes_shared_allocation_logging(self):
        jobs = PACKAGE_ROOT / "jobs"
        for script in sorted(jobs.glob("*.pbs")):
            with self.subTest(filename=script.name):
                subprocess.run(["bash", "-n", str(script)], check=True)
                source = script.read_text(encoding="utf-8")
                self.assertLess(
                    source.index('source "${SUBMIT_DIR}/jobs/_common.sh"'),
                    source.index("screw_hpc_init"),
                )

    def test_cx3_full_requires_1000_deterministic_eval_episodes(self):
        source = (
            PACKAGE_ROOT / "jobs" / "cx3_full.pbs"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'EVAL_EPISODES="${SCREW_EVAL_EPISODES:-1000}"',
            source,
        )
        self.assertIn('--episodes "${EVAL_EPISODES}"', source)
        self.assertIn(
            '${EVAL_EPISODES}-episode deterministic evaluation',
            source,
        )


class PackageBuilderArtifactAuditTest(unittest.TestCase):
    def test_archive_counts_only_actual_s_hpc_model_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "hpc"
            output = root / "packages"
            package.mkdir()
            for filename in (
                "build_autodl_package.sh",
                "build_hpc_package.sh",
            ):
                shutil.copy2(PACKAGE_ROOT / filename, package / filename)

            # These mimic the sibling macOS/MPS smoke artifacts in SCREW.
            # They are outside hpc and must never affect the archive count.
            sibling_checkpoints = root / "checkpoints"
            sibling_checkpoints.mkdir()
            (sibling_checkpoints / "_smoke_diffusion.pt").write_bytes(b"mac")
            (
                sibling_checkpoints
                / "_smoke_residual_tqc_mps.replay_buffer.pkl"
            ).write_bytes(b"mps")

            archive_prefix = (
                "hpc_AUTODL_H110_MOUTH330_W18000_EP4000_"
                "BLEND100_HX1_COMPAT"
            )

            def build(
                stamp: str,
                *,
                include_models: str = "1",
                require_models: str = "0",
                generic: bool = True,
            ):
                environment = os.environ.copy()
                environment.update(
                    {
                        "PYTHON_BOOTSTRAP": sys.executable,
                        "SCREW_PACKAGE_INCLUDE_MODELS": include_models,
                        "SCREW_PACKAGE_OUTPUT_DIR": str(output),
                        "SCREW_PACKAGE_REQUIRE_MODELS": require_models,
                        "SCREW_PACKAGE_STAMP": stamp,
                    }
                )
                script = (
                    package / "build_hpc_package.sh"
                    if generic
                    else package / "build_autodl_package.sh"
                )
                return subprocess.run(
                    [str(script)],
                    cwd=package,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            def archive_path(stamp: str) -> Path:
                return output / f"{archive_prefix}_{stamp}.tar.gz"

            source_only = build("source-only")
            self.assertEqual(
                source_only.returncode,
                0,
                msg=source_only.stdout + source_only.stderr,
            )
            self.assertIn(
                "model_artifact_members=0",
                source_only.stdout,
            )
            source_archive = archive_path("source-only")
            self.assertTrue(source_archive.is_file())
            with tarfile.open(source_archive, "r:gz") as stream:
                source_names = set(stream.getnames())
            self.assertFalse(
                any(name.startswith("checkpoints/") for name in source_names)
            )

            required_empty = build(
                "required-empty",
                require_models="1",
            )
            self.assertNotEqual(required_empty.returncode, 0)
            self.assertIn(
                "model_artifact_members=0",
                required_empty.stdout,
            )
            self.assertIn(
                "required model artifacts are missing from archive",
                required_empty.stderr,
            )
            self.assertFalse(archive_path("required-empty").exists())
            self.assertFalse(
                Path(
                    f"{archive_path('required-empty')}.sha256"
                ).exists()
            )

            run_checkpoints = (
                package / "runs" / "cx3_l40s_full" / "checkpoints"
            )
            run_checkpoints.mkdir(parents=True)
            (run_checkpoints / "diffusion.pt").write_bytes(b"diffusion")
            (run_checkpoints / "residual_tqc.pt").write_bytes(b"tqc")
            exported = package / "checkpoints"
            exported.mkdir()
            (exported / "policy.onnx").write_bytes(b"onnx")
            (exported / "replay.pkl").write_bytes(b"not-a-model")
            tests_dir = package / "tests"
            tests_dir.mkdir()
            (tests_dir / "decoy.pt").write_bytes(b"not-a-checkpoint")
            outputs_dir = package / "outputs"
            outputs_dir.mkdir()
            (outputs_dir / "plot.pt").write_bytes(b"not-a-model")
            (
                package
                / "runs"
                / "cx3_l40s_full"
                / "run_manifest.json"
            ).write_text("{}", encoding="utf-8")

            with_models = build(
                "with-models",
                require_models="1",
            )
            self.assertEqual(
                with_models.returncode,
                0,
                msg=with_models.stdout + with_models.stderr,
            )
            self.assertIn(
                "model_artifact_members=3",
                with_models.stdout,
            )
            model_archive = archive_path("with-models")
            with tarfile.open(model_archive, "r:gz") as stream:
                model_names = set(stream.getnames())
            self.assertIn(
                "hpc/runs/cx3_l40s_full/checkpoints/diffusion.pt",
                model_names,
            )
            self.assertNotIn(
                "hpc/checkpoints/_smoke_diffusion.pt",
                model_names,
            )

            explicitly_light = build(
                "explicitly-light",
                include_models="0",
                generic=False,
            )
            self.assertEqual(
                explicitly_light.returncode,
                0,
                msg=explicitly_light.stdout + explicitly_light.stderr,
            )
            self.assertIn(
                "model_artifact_members=0",
                explicitly_light.stdout,
            )

            required_but_excluded = build(
                "required-but-excluded",
                include_models="0",
                require_models="1",
                generic=False,
            )
            self.assertNotEqual(required_but_excluded.returncode, 0)
            self.assertIn(
                "model_artifact_members=0",
                required_but_excluded.stdout,
            )
            self.assertFalse(
                archive_path("required-but-excluded").exists()
            )

            invalid = build("invalid-require", require_models="2")
            self.assertEqual(invalid.returncode, 64)
            self.assertIn(
                "SCREW_PACKAGE_REQUIRE_MODELS must be 0 or 1",
                invalid.stderr,
            )


if __name__ == "__main__":
    unittest.main()
