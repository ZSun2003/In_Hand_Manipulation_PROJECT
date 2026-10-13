"""Shared-runtime bootstrap and source-asset contract verification.

Uses the central sibling RL_env. Controller, planner and source meshes remain
inside this gripper archive; model fingerprints are computed from source bytes.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import site
import sys
from types import ModuleType
from typing import Any
from xml.etree import ElementTree

from . import config


PACKAGE_DIR = Path(__file__).resolve().parent
GRAB_DIR = PACKAGE_DIR.parent
BASE_ENV_DIR = Path(os.environ.get("IRP_PYTHON_ENV", str(Path(__file__).resolve().parents[3] / "RL_env")))
RL_ENV_DIR = Path(os.environ.get("IRP_PYTHON_ENV", str(Path(__file__).resolve().parents[3] / "RL_env")))
_PINNED_PARENT_CONTRACT_SHA256: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_parent_artifact(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(GRAB_DIR.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escaped the parent GRAB directory") from exc
    try:
        resolved.relative_to(PACKAGE_DIR.resolve())
    except ValueError:
        return resolved
    raise RuntimeError(f"{label} must not be loaded from IK_RL: {resolved}")


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RuntimeError(f"{label} is not a lowercase SHA-256 digest")
    return digest


def _validate_generated_collision_xml(
    *,
    fruit: str,
    path: Path,
    capsules: list[dict[str, Any]],
) -> None:
    geoms = list(ElementTree.parse(path).getroot().iter("geom"))
    if len(geoms) != len(capsules):
        raise RuntimeError(
            f"{fruit} XML/manifest capsule count mismatch: "
            f"{len(geoms)} != {len(capsules)}"
        )
    for index, (geom, capsule) in enumerate(zip(geoms, capsules, strict=True)):
        expected_name = f"object_{fruit}_collision_{index:03d}"
        if geom.get("name") != expected_name or geom.get("type") != "capsule":
            raise RuntimeError(
                f"{fruit} collision XML drifted at capsule {index}"
            )
        expected_fromto = tuple(
            float(value)
            for value in (*capsule["from_m"], *capsule["to_m"])
        )
        actual_fromto = tuple(
            float(value) for value in str(geom.get("fromto", "")).split()
        )
        if len(actual_fromto) != 6 or any(
            abs(actual - expected) > 1.0e-8
            for actual, expected in zip(
                actual_fromto, expected_fromto, strict=True
            )
        ):
            raise RuntimeError(
                f"{fruit} collision XML fromto drifted at capsule {index}"
            )
        if abs(
            float(str(geom.get("size", "nan")))
            - float(capsule["radius_m"])
        ) > 1.0e-8:
            raise RuntimeError(
                f"{fruit} collision XML radius drifted at capsule {index}"
            )


def validate_parent_contract(current: ModuleType) -> dict[str, str]:
    """Verify that controller, planner and MuJoCo use one parent fruit model."""
    config.validate()
    controller_path = _require_parent_artifact(
        Path(str(current.__file__)),
        "controller module",
    )
    if controller_path != config.PARENT_CONTROLLER.resolve():
        raise RuntimeError(
            f"loaded controller is not the parent controller: {controller_path}"
        )
    if Path(current.ROOT_DIR).resolve() != GRAB_DIR.resolve():
        raise RuntimeError("controller ROOT_DIR is not the parent GRAB directory")
    model_xml = _require_parent_artifact(
        Path(current.MODEL_XML),
        "model XML",
    )
    if model_xml != config.PARENT_MODEL_XML.resolve():
        raise RuntimeError(f"controller model XML drifted: {model_xml}")

    dependency_paths = tuple(
        _require_parent_artifact(Path(path), "model dependency")
        for path in current.MODEL_DEPENDENCY_PATHS
    )
    expected_model_dependencies = {
        model_xml,
        *(
            _require_parent_artifact(path, "fruit collision XML")
            for path in config.PARENT_FRUIT_COLLISION_XMLS
        ),
    }
    missing_dependencies = expected_model_dependencies.difference(
        dependency_paths
    )
    if missing_dependencies:
        raise RuntimeError(
            "parent model fingerprint omits shared model artifacts: "
            f"{sorted(str(path) for path in missing_dependencies)}"
        )

    planner_name = current.AdaptiveGraspPlanePlanner.__module__
    planner_module = importlib.import_module(planner_name)
    planner_path = _require_parent_artifact(
        Path(str(planner_module.__file__)),
        "planner module",
    )
    if planner_path != config.PARENT_PLANNER.resolve():
        raise RuntimeError(f"loaded planner is not the parent planner: {planner_path}")

    manifest_path = _require_parent_artifact(
        config.PARENT_FRUIT_MANIFEST,
        "fruit capsule manifest",
    )
    planner_manifest_path = _require_parent_artifact(
        Path(planner_module.FRUIT_CAPSULE_MANIFEST_PATH),
        "planner fruit capsule manifest",
    )
    if planner_manifest_path != manifest_path:
        raise RuntimeError(
            "planner and IK_RL reference different fruit capsule manifests"
        )
    manifest_sha256 = _sha256_file(manifest_path)
    if (
        _require_sha256(
            planner_module.FRUIT_CAPSULE_MANIFEST_SHA256,
            "planner fruit manifest fingerprint",
        )
        != manifest_sha256
    ):
        raise RuntimeError("planner fruit capsule manifest SHA-256 is stale")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != config.PARENT_FRUIT_MANIFEST_SCHEMA:
        raise RuntimeError("unsupported parent fruit capsule manifest schema")
    if float(payload["maximum_capsule_spacing_m"]) > 0.005 + 1.0e-12:
        raise RuntimeError("parent fruit capsule spacing exceeds 5 mm")

    model_includes = {
        Path(element.attrib["file"]).name
        for element in ElementTree.parse(model_xml).getroot().iter("include")
        if "file" in element.attrib
    }
    source_paths: list[Path] = []
    objects = payload["objects"]
    for fruit, collision_xml in zip(
        ("banana", "pear", "strawberry"),
        config.PARENT_FRUIT_COLLISION_XMLS,
        strict=True,
    ):
        record = objects[fruit]
        capsules = list(record["capsules"])
        if int(record["capsule_count"]) != len(capsules) or not capsules:
            raise RuntimeError(f"invalid {fruit} capsule count in parent manifest")
        xml_path = _require_parent_artifact(
            collision_xml,
            f"{fruit} collision XML",
        )
        if xml_path.name not in model_includes:
            raise RuntimeError(f"MODEL_XML does not include {xml_path.name}")
        _validate_generated_collision_xml(
            fruit=fruit,
            path=xml_path,
            capsules=capsules,
        )
        source_path = _require_parent_artifact(
            GRAB_DIR / str(record["source_path"]),
            f"{fruit} source mesh",
        )
        if _sha256_file(source_path) != _require_sha256(
            record["source_sha256"],
            f"{fruit} source mesh fingerprint",
        ):
            raise RuntimeError(f"{fruit} source mesh SHA-256 is stale")
        source_paths.append(source_path)
        planner_capsules = planner_module.FRUIT_CAPSULES_BY_NAME[fruit]
        if len(planner_capsules) != len(capsules):
            raise RuntimeError(f"planner {fruit} capsule count drifted")

    model_fingerprint = _require_sha256(
        current.MODEL_FINGERPRINT_SHA256,
        "parent model fingerprint",
    )
    controller_fingerprint = _require_sha256(
        current.CONTROLLER_FINGERPRINT_SHA256,
        "parent controller fingerprint",
    )
    contract_digest = hashlib.sha256()
    contract_paths = (
        model_xml,
        controller_path,
        planner_path,
        manifest_path,
        *tuple(path.resolve() for path in config.PARENT_FRUIT_COLLISION_XMLS),
        *source_paths,
    )
    for path in sorted(contract_paths, key=lambda value: str(value)):
        contract_digest.update(str(path.relative_to(GRAB_DIR.resolve())).encode())
        contract_digest.update(b"\0")
        contract_digest.update(bytes.fromhex(_sha256_file(path)))
    contract_sha256 = contract_digest.hexdigest()
    global _PINNED_PARENT_CONTRACT_SHA256
    if _PINNED_PARENT_CONTRACT_SHA256 is None:
        _PINNED_PARENT_CONTRACT_SHA256 = contract_sha256
    elif _PINNED_PARENT_CONTRACT_SHA256 != contract_sha256:
        raise RuntimeError(
            "parent model contract changed during this IK_RL process"
        )
    return {
        "model_xml": str(model_xml),
        "controller": str(controller_path),
        "planner": str(planner_path),
        "fruit_manifest": str(manifest_path),
        "fruit_manifest_sha256": manifest_sha256,
        "model_fingerprint_sha256": model_fingerprint,
        "controller_fingerprint_sha256": controller_fingerprint,
        "contract_sha256": contract_sha256,
    }


def _rl_site_packages() -> Path:
    candidates = sorted((RL_ENV_DIR / "lib").glob("python*/site-packages"))
    if len(candidates) != 1 or not candidates[0].is_dir():
        raise RuntimeError(
            "expected exactly one RL_env site-packages directory, got "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0].resolve()


def ensure_runtime() -> None:
    """Use the parent simulation venv and expose the sibling RL packages."""

    interpreter = BASE_ENV_DIR / "bin" / "python"
    if not interpreter.is_file():
        raise RuntimeError(f"parent simulation interpreter not found: {interpreter}")

    try:
        in_base_env = Path(sys.prefix).resolve() == BASE_ENV_DIR.resolve()
    except OSError:
        in_base_env = False
    if not in_base_env:
        main_module = getattr(
            getattr(sys.modules.get("__main__"), "__spec__", None),
            "name",
            None,
        )
        if main_module:
            arguments = [
                str(interpreter),
                "-m",
                str(main_module),
                *sys.argv[1:],
            ]
        else:
            arguments = [
                str(interpreter),
                str(Path(sys.argv[0]).resolve()),
                *sys.argv[1:],
            ]
        os.execv(
            str(interpreter),
            arguments,
        )

    cache_dir = PACKAGE_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(cache_dir / "pycache"))
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    rl_site = str(_rl_site_packages())
    if rl_site not in sys.path:
        site.addsitedir(rl_site)
    grab_path = str(GRAB_DIR.resolve())
    if grab_path not in sys.path:
        sys.path.insert(0, grab_path)


def import_current_controller() -> ModuleType:
    """Import and validate the authoritative parent G3 controller."""
    ensure_runtime()
    current = importlib.import_module("G3_grab")
    validate_parent_contract(current)
    return current


__all__ = [
    "BASE_ENV_DIR",
    "GRAB_DIR",
    "PACKAGE_DIR",
    "RL_ENV_DIR",
    "ensure_runtime",
    "import_current_controller",
    "validate_parent_contract",
]
