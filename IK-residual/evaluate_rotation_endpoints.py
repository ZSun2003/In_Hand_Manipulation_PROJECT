#!/usr/bin/env python3
"""Evaluate residual-policy endpoint accuracy with resumable shards."""

from __future__ import annotations

import os
from pathlib import Path
import sys


RUNTIME = Path(__file__).resolve().parent.parent / "RL_env"
if __name__ == "__main__" and not Path(sys.prefix).resolve().is_relative_to(RUNTIME.resolve()):
    python = RUNTIME / "bin" / "python"
    if not python.is_file():
        raise SystemExit(f"Shared Python not found: {python}. See ../RL_env/README.md.")
    os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])

from g3g6.rotation_residual_endpoint_eval import main


if __name__ == "__main__":
    main()
