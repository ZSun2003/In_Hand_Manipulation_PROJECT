"""Movable-package path policy shared by every hpc command-line tool."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PACKAGE_ROOT


def add_external_path_flag(parser: argparse.ArgumentParser) -> None:
    """Add the explicit escape hatch for paths outside the movable package."""

    parser.add_argument(
        "--allow-external-paths",
        action="store_true",
        help=(
            "allow input/output paths outside hpc; by default every path is "
            "resolved inside the movable hpc directory"
        ),
    )


def resolve_package_path(
    value: str | Path,
    *,
    allow_external: bool = False,
    label: str = "path",
) -> Path:
    """Resolve *value* and reject package escapes unless explicitly allowed.

    Relative paths are always anchored at ``hpc`` instead of the caller's
    current working directory.  ``Path.resolve`` also closes symlink and
    ``..`` escapes before the containment check.
    """

    candidate = Path(value).expanduser()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (PACKAGE_ROOT / candidate).resolve()
    )
    if not allow_external:
        try:
            resolved.relative_to(PACKAGE_ROOT)
        except ValueError as exc:
            raise ValueError(
                f"{label} must remain inside hpc ({PACKAGE_ROOT}); got "
                f"{resolved}. Pass --allow-external-paths to opt in."
            ) from exc
    return resolved


__all__ = [
    "add_external_path_flag",
    "resolve_package_path",
]
