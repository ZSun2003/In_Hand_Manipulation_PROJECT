# HX1 A100 14,000-world Diffusion + IK + residual TQC package

This is an isolated full-pipeline profile. It does not change the existing
AutoDL or HX1 Diffusion-only configurations, jobs, run directories or package
names.

## Frozen task and model contract

- simulator width: 14,000 MJWarp worlds;
- episode ceiling: 4,000 control steps;
- base SCREW observation: 191 dimensions (unchanged);
- residual-TQC state: the 191-dimensional base state augmented by the current
  16-dimensional nominal IK command, previous realised 16-dimensional residual
  and 16-dimensional executable-channel mask, plus Diffusion-plan cursor,
  anchor-handover state and progress (242 dimensions total);
- demonstrations: 8,192 trajectories;
- Diffusion: 2-frame observation history, 16-command horizon, 4-command
  execution interval, 100 training noise steps and 16 DDIM inference steps;
- Diffusion-to-anchor handover: 20-step minimum-jerk blend;
- geometry: 30 mm screw shaft in a 34 mm bore (4 mm diametral clearance);
- controller order: Diffusion task-space command, analytical IK nominal
  actuator command, then bounded 16-dimensional actuator-residual TQC;
- residual mapper: blend 0.35; rotary limit/slew 0.005/0.0001 rad; linear
  limit/slew 0.5/0.02 mm; linear actuator indices `(0,3,6,9)`;
- TQC: two 256-unit hidden layers, four critics with 32 quantiles each, and
  two upper quantiles dropped per critic; actor log standard deviation starts
  at -3.0 and training uses eight gradient steps per eligible vector tick,
  capped at eight.

At most 256 active worlds are inserted into replay at each vector tick. The
five-million stored-transition budget therefore spans approximately 19,532
vector ticks (nearly five 4,000-step episode lengths), rather than ending after
only a few 14,000-world batches. The run is resumable across HX1 allocations.

## Build on the local machine

```bash
cd <LOCAL_DIFFUSION_DIRECTORY>/hpc
./build_hx1_a100_w14000_diffusion_ik_residual_package.sh
```

The builder creates a source-only archive and adjacent `.sha256` file in the
parent `Diffusion` directory. It deliberately excludes virtual environments, caches,
datasets, runs and checkpoints.

## Upload and extract on HX1

From the local machine, replace `<STAMP>` with the archive date printed by the
builder:

```bash
cd <LOCAL_DIFFUSION_DIRECTORY>
scp \
  hpc_HX1_A100_W14000_DIFFUSION_IK_RESIDUAL_TQC16_EP4000_BLEND20_CLEAR4_<STAMP>.tar.gz{,.sha256} \
  sz1725@login.hx1.hpc.ic.ac.uk:~/
```

On HX1:

```bash
cd ~
sha256sum -c \
  hpc_HX1_A100_W14000_DIFFUSION_IK_RESIDUAL_TQC16_EP4000_BLEND20_CLEAR4_<STAMP>.tar.gz.sha256
tar -xzf \
  hpc_HX1_A100_W14000_DIFFUSION_IK_RESIDUAL_TQC16_EP4000_BLEND20_CLEAR4_<STAMP>.tar.gz

cd ~/hpc
export IRP_HPC_CORE_ENV="$HOME/IRP/RL_env/profiles/core/RL_env"
./setup_hpc.sh
```

## Submit or resume the full pipeline

```bash
cd ~/hpc
export IRP_HPC_CORE_ENV="$HOME/IRP/RL_env/profiles/core/RL_env"
qsub -v IRP_HPC_CORE_ENV="$IRP_HPC_CORE_ENV" \
  jobs/hx1_w14000_diffusion_ik_residual_full.pbs
```

The job runs the GPU/static preflight and the complete Diffusion-TQC training
stack gate before starting `pipeline.py --stage full`. The full stage executes
demonstration generation, Diffusion training, actuator-residual TQC training and
evaluation in this order. Its stable run directory is:

```text
runs/hx1_a100_w14000_diffusion_ik_residual_full
```

If the 71-hour-30-minute guard ends an allocation, submit the identical PBS job
again. `--resume auto` continues from the newest atomic checkpoint. The guard
uses the valid GNU `timeout` duration `4290m`, rather than `71h30m`.

## Monitor

```bash
qstat -u "$USER"

cd ~/hpc
./hpc_python.sh status_hpc.py \
  --run-dir runs/hx1_a100_w14000_diffusion_ik_residual_full

tail -n 5 \
  runs/hx1_a100_w14000_diffusion_ik_residual_full/logs/tqc.jsonl
```

Expected checkpoints are `checkpoints/diffusion.pt`,
`checkpoints/residual_tqc.pt` and
`checkpoints/residual_tqc_policy.pt`.
