"""Launch Diffusion with the shared sibling RL_env.

RL_env/build.sh manages the virtual environment using the host's Python.
Rebuild the environment on a different host. Training outputs and caches
remain inside Diffusion; no additional interpreter is bundled here.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT.parent / "RL_env"
RUNTIME_DIRS: Mapping[str, Path] = {
    name: PROJECT_ROOT / dirname
    for name, dirname in (
        ("cache", ".cache"), ("tmp", "tmp"), ("outputs", "outputs"),
        ("runs", "runs"), ("checkpoints", "checkpoints"), ("datasets", "datasets"),
    )
}


class RuntimeErrorWithHint(RuntimeError):
    """Runtime failure with a recovery hint."""


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def project_path(*parts: str | os.PathLike[str], must_exist: bool = False) -> Path:
    result = PROJECT_ROOT.joinpath(*parts).resolve(strict=False)
    if not _inside(result, PROJECT_ROOT):
        raise ValueError(f"Path escapes Diffusion: {result}")
    if must_exist and not result.exists():
        raise FileNotFoundError(result)
    return result


def ensure_runtime_dirs(root: Path = PROJECT_ROOT) -> dict[str, Path]:
    root = root.resolve()
    dirs = {name: root / path.name for name, path in RUNTIME_DIRS.items()}
    cache_names = ("matplotlib", "pip", "torch", "huggingface", "xdg", "pycache", "numba", "wandb")
    for path in (*dirs.values(), *(dirs["cache"] / name for name in cache_names), dirs["tmp"] / "joblib"):
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def configured_environment(root: Path = PROJECT_ROOT, base: Mapping[str, str] | None = None) -> dict[str, str]:
    root = root.resolve()
    dirs = ensure_runtime_dirs(root)
    cache = dirs["cache"]
    venv = root.parent / "RL_env"
    env = dict(os.environ if base is None else base)
    for inherited in (
        "PYTHONHOME", "PYTHONUSERBASE", "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "SCREW_EMBEDDED_PYTHON",
    ):
        env.pop(inherited, None)
    env.update({
        "SCREW_ROOT": str(root),
        "SCREW_CACHE_DIR": str(cache),
        "SCREW_TMP_DIR": str(dirs["tmp"]),
        "SCREW_OUTPUT_DIR": str(dirs["outputs"]),
        "SCREW_RUNS_DIR": str(dirs["runs"]),
        "SCREW_CHECKPOINT_DIR": str(dirs["checkpoints"]),
        "SCREW_DATASET_DIR": str(dirs["datasets"]),
        "PYTHONPATH": str(root),
        "PYTHONNOUSERSITE": "1",
        "VIRTUAL_ENV": str(venv),
        "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "TORCH_HOME": str(cache / "torch"),
        "HF_HOME": str(cache / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "XDG_CONFIG_HOME": str(cache / "xdg" / "config"),
        "XDG_DATA_HOME": str(cache / "xdg" / "data"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "WANDB_CACHE_DIR": str(cache / "wandb"),
        "WANDB_CONFIG_DIR": str(cache / "wandb" / "config"),
        "WANDB_DIR": str(dirs["runs"] / "wandb"),
        "JOBLIB_TEMP_FOLDER": str(dirs["tmp"] / "joblib"),
        "TMPDIR": str(dirs["tmp"]), "TEMP": str(dirs["tmp"]), "TMP": str(dirs["tmp"]),
        "PATH": os.pathsep.join((str(venv / "bin"), "/usr/bin", "/bin", "/usr/sbin", "/sbin")),
    })
    return env


def configure_current_process(root: Path = PROJECT_ROOT) -> dict[str, str]:
    root = root.resolve()
    env = configured_environment(root)
    os.environ.clear()
    os.environ.update(env)
    os.chdir(root)
    if not sys.path or sys.path[0] != str(root):
        sys.path.insert(0, str(root))
    return env


def local_python(root: Path = PROJECT_ROOT) -> Path:
    venv = root.resolve().parent / "RL_env"
    interpreter = venv / "bin" / "python"
    if not interpreter.exists():
        raise RuntimeErrorWithHint(
            f"Missing shared interpreter: {interpreter}\n"
            f"Rebuild it with {venv / 'build.sh'} --profile local."
        )
    return interpreter


def mjpython_command(entry_args: Iterable[str | os.PathLike[str]], root: Path = PROJECT_ROOT) -> list[str]:
    """Bypass mjpython's generated shebang, including paths containing spaces."""
    root = root.resolve()
    trampoline = root.parent / "RL_env" / "bin" / "mjpython"
    if not trampoline.is_file():
        raise RuntimeErrorWithHint(
            f"Missing MuJoCo trampoline: {trampoline}; rebuild ../RL_env with build.sh."
        )
    return [str(local_python(root)), str(trampoline), *(os.fspath(value) for value in entry_args)]


def python_command(entry_args: Iterable[str | os.PathLike[str]], root: Path = PROJECT_ROOT) -> list[str]:
    return [str(local_python(root)), *(os.fspath(value) for value in entry_args)]


def _entry_within_root(entry: str, root: Path) -> Path:
    path = Path(entry)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    if not _inside(path, root):
        raise RuntimeErrorWithHint(f"Entry point must be inside Diffusion: {path}")
    if not path.is_file():
        raise RuntimeErrorWithHint(f"Entry point does not exist: {path}")
    return path


def _auto_mode(entry: Path | None, module: str | None) -> str:
    override = os.environ.get("SCREW_INTERPRETER", "").strip().lower()
    if override in {"python", "mjpython"}:
        return override
    name = (module or (entry.name if entry else "")).lower()
    gui = any(word in name for word in ("viewer", "visual", "interactive"))
    default_app = entry is not None and entry.name == "run_screw.py"
    return "mjpython" if platform.system() == "Darwin" and (gui or default_app) else "python"


def build_launch_command(
    entry: str | None = None, entry_args: Sequence[str] = (), *,
    module: str | None = None, mode: str = "auto", root: Path = PROJECT_ROOT,
) -> list[str]:
    root = root.resolve()
    if module and entry:
        raise ValueError("Use either an entry script or --module, not both.")
    resolved_entry = None
    if module is None:
        resolved_entry = _entry_within_root(entry or "run_screw.py", root)
        target = [str(resolved_entry), *entry_args]
    else:
        target = ["-m", module, *entry_args]
    selected = _auto_mode(resolved_entry, module) if mode == "auto" else mode
    if selected == "mjpython":
        return mjpython_command(target, root)
    if selected == "python":
        return python_command(target, root)
    raise ValueError(f"Unknown interpreter mode: {selected}")


def dependency_report(root: Path = PROJECT_ROOT) -> dict[str, object]:
    interpreter = local_python(root)
    imports = {}
    for name in ("numpy", "torch", "mujoco", "gymnasium", "ssl", "sqlite3", "lzma", "decimal"):
        try:
            module = importlib.import_module(name)
            imports[name] = str(getattr(module, "__version__", "present"))
        except Exception as exc:
            imports[name] = f"ERROR: {type(exc).__name__}: {exc}"
    return {
        "root": str(root.resolve()), "python": str(interpreter),
        "python_resolved": str(interpreter.resolve()),
        "python_version": platform.python_version(), "machine": platform.machine(),
        "environment": str(root.resolve().parent / "RL_env"), "imports": imports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Diffusion using the shared sibling RL_env.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--python", dest="mode", action="store_const", const="python")
    modes.add_argument("--mjpython", dest="mode", action="store_const", const="mjpython")
    parser.set_defaults(mode="auto")
    parser.add_argument("-m", "--module", help="run a project-local Python module")
    parser.add_argument("--check", action="store_true", help="print shared-environment health JSON")
    parser.add_argument("entry", nargs="?", help="script below Diffusion (default: run_screw.py)")
    parser.add_argument("entry_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    env = configure_current_process(PROJECT_ROOT)
    if args.check:
        print(json.dumps(dependency_report(), indent=2, sort_keys=True))
        return 0
    command = build_launch_command(args.entry, args.entry_args, module=args.module, mode=args.mode)
    try:
        return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False).returncode
    except OSError as exc:
        raise RuntimeErrorWithHint(f"Could not launch {command[0]}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
