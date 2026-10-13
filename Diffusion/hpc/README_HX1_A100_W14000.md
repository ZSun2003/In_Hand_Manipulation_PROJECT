# HX1 A100 14,000-world Diffusion-only package

This profile is isolated from the AutoDL launchers and configurations.  It
uses 14,000 MJWarp worlds, a 4,000-step episode ceiling, the shared 100-step
Diffusion-to-anchor minimum-jerk transition, 8,192 demonstrations, 60
Diffusion epochs, batch size 4,096, and no TQC training.

The PBS job first runs the complete Diffusion-only training-stack memory and
physics gate at a 90% A100 memory ceiling.  Formal training starts only if
that gate passes.  The stable run directory is resumable across HX1's 72-hour
allocations.

After extracting over `~/hpc` and validating the external environment:

```bash
cd ~/hpc
export IRP_HPC_CORE_ENV="$HOME/IRP/RL_env/profiles/core/RL_env"
qsub -v IRP_HPC_CORE_ENV="$IRP_HPC_CORE_ENV" \
  jobs/hx1_w14000_diffusion_only_full.pbs
```

Monitor with `qstat -u "$USER"`.  If the walltime guard ends an allocation,
resubmit the identical command; `--resume auto` reuses the last atomic
checkpoint in `runs/hx1_a100_w14000_diffusion_only_full`.

The HX1 walltime guard uses `4290m` (71 hours 30 minutes), because the GNU
`timeout` version installed on HX1 does not accept the compound form
`71h30m`.
