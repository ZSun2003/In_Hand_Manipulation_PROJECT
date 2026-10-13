"""Tiny Warp JIT kernel used by the RTX PRO 6000 hardware gate."""

from typing import Any

import warp as wp


@wp.kernel
def _fill_probe(values: wp.array(dtype=wp.float32)):
    values[wp.tid()] = 3.25


def run_warp_kernel_smoke(device: Any) -> dict[str, Any]:
    values = wp.zeros(256, dtype=wp.float32, device=device)
    wp.launch(_fill_probe, dim=values.size, inputs=[values], device=device)
    wp.synchronize_device(device)
    host = values.numpy()
    minimum = float(host.min())
    maximum = float(host.max())
    if minimum != 3.25 or maximum != 3.25:
        raise RuntimeError(
            f"Warp JIT kernel returned [{minimum}, {maximum}], expected 3.25"
        )
    return {
        "elements": int(values.size),
        "minimum": minimum,
        "maximum": maximum,
    }
