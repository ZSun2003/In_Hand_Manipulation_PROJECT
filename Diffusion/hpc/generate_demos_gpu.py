#!/usr/bin/env python3
"""Generate MJWarp expert demonstrations into a local memory-mapped store."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.config import load_config
from s_hpc.paths import add_external_path_flag, resolve_package_path
from s_hpc.workflows import generate_demos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    add_external_path_flag(parser)
    args = parser.parse_args()
    try:
        config_path = resolve_package_path(
            args.config,
            allow_external=args.allow_external_paths,
            label="config",
        )
        output_path = resolve_package_path(
            args.output,
            allow_external=args.allow_external_paths,
            label="demo output",
        )
    except ValueError as exc:
        parser.error(str(exc))
    generate_demos(
        load_config(config_path),
        output=output_path,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
