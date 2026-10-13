# AutoDL RTX PRO 6000: 14,000-world Diffusion + IK + residual TQC

This is an isolated full-training profile for the rented AutoDL RTX PRO 6000
Blackwell GPU. It adds new configuration, launcher, run directories, logs and
archive names; it does not alter the parameters or outputs of the existing
AutoDL Diffusion-only package.

## Frozen controller contract

The default formal pipeline is:

```text
8,192 expert demonstrations
→ Diffusion training
→ analytical IK nominal control
→ bounded 16-actuator residual TQC training
→ 2,000-episode evaluation
```

The task/model profile uses:

- 14,000 MJWarp worlds and a 4,000-control-step episode ceiling;
- the unchanged 191-dimensional Diffusion observation and two-frame history;
- a 242-dimensional TQC input: 191 base fields, 16 nominal IK commands,
  16 previous realised residuals, 16 executable-channel mask values, and three
  Diffusion/anchor controller-state values;
- a 14-command task-space Diffusion action, 16-command plan horizon, four-command
  execution interval, 100 DDPM training steps and 16 DDIM inference steps;
- a 20-control-step minimum-jerk Diffusion-to-anchor transition;
- a bore diameter exactly 4 mm larger than the 30 mm screw-shaft diameter;
- four 256-unit, 32-quantile critics with two upper quantiles discarded per
  critic; actor/critics learning rate `1e-4`, batch 256, replay capacity one
  million and eight gradient updates per eligible vector tick, hard-capped at
  eight;
- at most 256 uniformly selected active worlds are inserted into replay per
  vector tick; the five-million-transition budget therefore spans about 19,532
  vector ticks rather than ending after only a few full-width batches;
- post-IK residual blend 0.35; rotary limit/slew 0.005/0.0001 rad and linear
  limit/slew 0.5/0.02 mm per control tick.

## Build locally

The final builder validates the observation/action dimensions, geometry, blend,
training budget and both launcher modes before creating the source-only archive:

```bash
cd <LOCAL_DIFFUSION_DIRECTORY>/hpc
./build_autodl_rtx_pro_6000_w14000_diffusion_ik_residual_package.sh
```

It writes the archive and adjacent checksum to the parent `Diffusion` directory:

```text
hpc_AUTODL_RTX_PRO_6000_W14000_DIFFUSION_IK_TQC16_OBS191_EP4000_BLEND20_CLEAR4_<STAMP>.tar.gz
hpc_AUTODL_RTX_PRO_6000_W14000_DIFFUSION_IK_TQC16_OBS191_EP4000_BLEND20_CLEAR4_<STAMP>.tar.gz.sha256
```

Virtual environments, caches, datasets, runs and checkpoints are excluded.

## Upload without overwriting the current AutoDL package

From the local Mac, replace `<STAMP>`, `<PORT>` and `<HOST>` with the values for
the current AutoDL instance:

```bash
cd <LOCAL_DIFFUSION_DIRECTORY>

scp -P <PORT> \
  hpc_AUTODL_RTX_PRO_6000_W14000_DIFFUSION_IK_TQC16_OBS191_EP4000_BLEND20_CLEAR4_<STAMP>.tar.gz \
  hpc_AUTODL_RTX_PRO_6000_W14000_DIFFUSION_IK_TQC16_OBS191_EP4000_BLEND20_CLEAR4_<STAMP>.tar.gz.sha256 \
  root@<HOST>:/root/autodl-tmp/
```

On AutoDL, extract into a new directory so the previous `/root/autodl-tmp/hpc`
tree remains untouched:

```bash
cd /root/autodl-tmp

sha256sum -c \
  hpc_AUTODL_RTX_PRO_6000_W14000_DIFFUSION_IK_TQC16_OBS191_EP4000_BLEND20_CLEAR4_<STAMP>.tar.gz.sha256

mkdir -p hpc_W14000_TQC16
tar -xzf \
  hpc_AUTODL_RTX_PRO_6000_W14000_DIFFUSION_IK_TQC16_OBS191_EP4000_BLEND20_CLEAR4_<STAMP>.tar.gz \
  --strip-components=1 -C hpc_W14000_TQC16
```

Do not extract over a partially used `hpc_W14000_TQC16` directory. Rename the
old directory first if a clean restart is required.

## Setup and preflight

```bash
cd /root/autodl-tmp/hpc_W14000_TQC16
export IRP_HPC_CORE_ENV=/path/to/IRP/RL_env/profiles/core/RL_env

./autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh setup
./autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh preflight
```

`start` repeats the static, GPU and complete 14,000-world training-stack gate,
so a separate `preflight` is optional. The gate uses at most 90% of GPU memory
and executes 4,000 control steps, covering the complete episode contract.

## Start the complete controller

```bash
cd /root/autodl-tmp/hpc_W14000_TQC16
export IRP_HPC_CORE_ENV=/path/to/IRP/RL_env/profiles/core/RL_env

./autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh start
```

The launcher uses `nohup` and `setsid`; training continues after SSH or
JupyterLab disconnects. Repeating `start` or `resume` continues atomically from:

```text
runs/autodl_rtx_pro_6000_w14000_diffusion_ik_residual_full
```

## Explicit TQC rollback

To bypass residual training and execution, launch the Diffusion--IK path with a
zero residual:

```bash
./autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh start-diffusion-only
```

Equivalent environment form:

```bash
SCREW_ENABLE_TQC=0 \
  ./autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh start
```

The rollback uses `pipeline.py --stage diffusion-full`, so TQC actor, critics,
replay, training and inference are not constructed. Its independent output is:

```text
runs/autodl_rtx_pro_6000_w14000_diffusion_only_full
```

This is an architecture rollback only. It retains the new 20-step blend and
4-mm diametral bore clearance, so it is not bitwise compatible with the old
task/checkpoint hashes and must not resume an old Diffusion-only run.

## Status and safe stop

The following commands automatically target the active or most recently selected
mode:

```bash
./autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh status
./autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh watch
./autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh logs
./autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh stop
```

`Ctrl+C` while using `watch` or `logs` only closes the viewer. `stop` asks the
training workflow to save at a consistent resumable cursor before exiting.

Full-mode checkpoints are:

```text
checkpoints/diffusion.pt
checkpoints/residual_tqc.pt
checkpoints/residual_tqc_policy.pt
```
