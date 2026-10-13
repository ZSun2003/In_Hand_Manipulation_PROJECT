# HX1 A100: 12,000-world formal Diffusion + IK + residual training

This is an isolated Imperial College HX1 profile. It does not change the
AutoDL 14,000-world package or any previous HX1 configuration/run directory.

## Contract

- one HX1 A100 GPU, 18 CPUs and 250 GB host memory;
- 12,000 MJWarp worlds and 4,000 control steps per episode;
- 8,192 demonstrations, 60 Diffusion epochs and 5,000,000 stored TQC
  transitions;
- controller order: Diffusion task-space command, certified analytical IK,
  then bounded 16-dimensional actuator-residual TQC;
- unchanged 191-dimensional Diffusion observation; 242-dimensional TQC state;
- 30 mm screw shaft, 34 mm bore and 20-step Diffusion-to-anchor blend;
- `--resume auto`, atomic checkpoints and a 4,290-minute runtime guard inside
  the 71:50 PBS allocation.

The archive is source-only: environments, datasets, runs and checkpoints are
excluded. GPU capacity and the complete training stack are checked inside the
allocated A100 job before formal training starts.

## Upload from the Mac

```bash
cd <LOCAL_DIFFUSION_DIRECTORY>

scp \
  hpc_HX1_A100_W12000_DIFFUSION_IK_RESIDUAL_TQC16_EP4000_BLEND20_CLEAR4_20260813.tar.gz \
  hpc_HX1_A100_W12000_DIFFUSION_IK_RESIDUAL_TQC16_EP4000_BLEND20_CLEAR4_20260813.tar.gz.sha256 \
  sz1725@login.hx1.hpc.ic.ac.uk:~/
```

## Verify and extract on HX1

Use a new directory so existing `~/hpc` results are untouched:

```bash
cd ~

sha256sum -c \
  hpc_HX1_A100_W12000_DIFFUSION_IK_RESIDUAL_TQC16_EP4000_BLEND20_CLEAR4_20260813.tar.gz.sha256

mkdir hpc_W12000_TQC16_20260813
tar -xzf \
  hpc_HX1_A100_W12000_DIFFUSION_IK_RESIDUAL_TQC16_EP4000_BLEND20_CLEAR4_20260813.tar.gz \
  --strip-components=1 \
  -C hpc_W12000_TQC16_20260813

cd ~/hpc_W12000_TQC16_20260813
export IRP_HPC_CORE_ENV="$HOME/IRP/RL_env/profiles/core/RL_env"
./setup_hpc.sh
```

## Submit formal training

```bash
cd ~/hpc_W12000_TQC16_20260813
export IRP_HPC_CORE_ENV="$HOME/IRP/RL_env/profiles/core/RL_env"

qsub \
  -v IRP_HPC_CORE_ENV="$IRP_HPC_CORE_ENV" \
  jobs/hx1_w12000_diffusion_ik_residual_full.pbs
```

The stable run directory is:

```text
runs/hx1_a100_w12000_diffusion_ik_residual_full
```

If the walltime guard stops the allocation, submit the identical PBS command
again. The job resumes from the latest consistent checkpoint.

## Status

```bash
qstat -u sz1725

cd ~/hpc_W12000_TQC16_20260813
./hpc_python.sh status_hpc.py \
  --run-dir runs/hx1_a100_w12000_diffusion_ik_residual_full
```

To view the PBS output:

```bash
tail -n 80 screw_hx1_w12k_tqc.o<JOB_NUMBER>
```

Expected final checkpoints are `checkpoints/diffusion.pt`,
`checkpoints/residual_tqc.pt`, and
`checkpoints/residual_tqc_policy.pt`.
