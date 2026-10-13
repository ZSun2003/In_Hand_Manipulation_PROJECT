"""Check the active local training environment; never write project outputs."""
from __future__ import annotations
import argparse
import importlib.metadata
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-mps', action='store_true')
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    import gymnasium
    import mujoco
    import numpy as np
    import torch
    import stable_baselines3
    import sb3_contrib
    print(f'Python: {sys.version.split()[0]} | {platform.system()} {platform.machine()}')
    print(f'Environment: {sys.prefix}')
    for name in ('torch', 'mujoco', 'numpy', 'gymnasium', 'stable-baselines3', 'sb3-contrib'):
        print(f'{name}: {importlib.metadata.version(name)}')
    print(f'MPS available: {torch.backends.mps.is_available()}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.01"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    for device, required in [('mps', args.require_mps), ('cuda', args.require_cuda)]:
        if required:
            value = torch.tensor([1., 2.], device=device).square().sum().cpu().item()
            assert value == 5
    print('Local imports and MuJoCo stepping: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
