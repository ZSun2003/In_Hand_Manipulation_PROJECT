"""Shared-environment relocation and launcher boundary regressions."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import runtime


class SharedRuntimeTests(unittest.TestCase):
    def test_launcher_uses_sibling_environment_from_any_working_directory(self):
        with tempfile.TemporaryDirectory(prefix="diffusion runtime ") as directory:
            workspace = Path(directory).resolve()
            project = workspace / "Diffusion"
            project.mkdir()
            bin_dir = workspace / "RL_env" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "python").write_text("fixture")
            entry = project / "train_diffusion.py"
            entry.write_text("fixture")
            command = runtime.build_launch_command(
                "train_diffusion.py", ["--help"], root=project, mode="python"
            )
            self.assertEqual(command, [str(bin_dir / "python"), str(entry), "--help"])
            with self.assertRaises(runtime.RuntimeErrorWithHint):
                runtime.build_launch_command("../outside.py", root=project)

    def test_environment_has_no_removed_embedded_python_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve() / "Diffusion"
            env = runtime.configured_environment(
                project,
                {"PATH": "/stale/venv/bin", "PYTHONHOME": "/stale/python",
                 "DYLD_FRAMEWORK_PATH": "/removed/.python"},
            )
            self.assertEqual(env["VIRTUAL_ENV"], str(project.parent / "RL_env"))
            self.assertEqual(env["PATH"].split(os.pathsep)[0], str(project.parent / "RL_env/bin"))
            self.assertNotIn("PYTHONHOME", env)
            self.assertNotIn("DYLD_FRAMEWORK_PATH", env)
            self.assertNotIn("SCREW_EMBEDDED_PYTHON", env)
            self.assertNotIn(".python", env["PATH"])
            Path(env["SCREW_DATASET_DIR"]).relative_to(project)


if __name__ == "__main__":
    unittest.main()
