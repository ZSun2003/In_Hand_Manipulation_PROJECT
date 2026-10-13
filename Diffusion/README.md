# Diffusion：RL16-API 螺旋装配训练

这是已完成 IRP 项目的源码归档。本目录包含 RL16-API 四指机械手的专家示范生成、Diffusion Policy、解析 IK、TQC 残差强化学习和任务评估代码。它由原 `Gripper6/SCREW` 整理而来，保留 Apple Silicon MPS/CPU 开发链路与 Linux CUDA + MuJoCo Warp（MJWarp）HPC 训练链路。

仓库只包含源代码、训练配置、仿真 XML 和 STL 网格。历史权重、示范数据、运行日志、图片、表格、论文、LaTeX、视频及发布图的轨迹录制脚本已经删除。训练与评估运行时仍会按需生成数据和检查点；这些文件被 `.gitignore` 排除。

## 目录与职责

| 路径 | 职责 |
| --- | --- |
| `generate_demos.py`、`demo_dataset.py` | 专家轨迹采集、示范数据读写与训练窗口采样 |
| `diffusion_policy.py`、`train_diffusion.py` | MPS/CPU Diffusion 模型、训练与 EMA 权重 |
| `train_residual.py`、`central_diffusion.py` | SB3 TQC 残差训练、多环境共享 Diffusion 推理 |
| `policy_runtime.py`、`residual_control.py` | Diffusion 动作安全处理、残差动作映射 |
| `screw_env.py`、`screw_sim.py` | Gymnasium 环境、MuJoCo 仿真与任务状态 |
| `screw_config.py`、`screw_observation.py` | 任务参数、观测定义和契约校验 |
| `screw_scene.py`、`screw_control.py`、`grasp_plane.py` | 场景构建、专家控制与抓取平面规划 |
| `kinematics.py` | 此装配任务使用的 RL16-API 解析运动学与 IK |
| `Project.xml`、`Gripper.xml`、`Share.xml`、`mesh/` | MPS/CPU 原始场景与 STL 资产 |
| `evaluate.py`、`test_diffusion.py` | 控制策略评估、Diffusion 损失评估与微型训练验证 |
| `runtime.py`、`run_screw.command` | 共享 Python 启动、工作目录和缓存路径管理 |
| `run_mps_tuning.command` | 小预算 demos → diffusion → residual → eval 联调 |
| `hpc/` | CUDA/MJWarp 批量仿真、Diffusion/TQC 训练、调度与预检 |
| `evaluation/workspace_policy_rollouts.py` | 固定工作空间策略评估、成功轨迹自蒸馏数据导出 |
| `evaluation/evaluate_tqc_azimuth.py` | 螺孔开口方位角策略评估 |
| `test_geometry.py`、`test_task.py`、`test_training.py` | 几何、控制、环境与训练回归测试 |

MPS 与 HPC 的环境、控制器及模型文件具有各自的实现和运行契约，因此分别维护。不要交换两条链路的检查点；数据或检查点的结构、几何哈希、观测和动作契约须通过各自入口校验。代码中的 `SCREW_*` 环境变量与 `s_hpc` Python 包名保留为稳定运行接口。

## 统一环境与仓库布局

推荐将八个目录作为同一个 GitHub 仓库的同级目录发布，至少保留：

```text
IRP/
├── Diffusion/
└── RL_env/
    ├── bin/python                    # 本机 MPS/CPU Python
    ├── hpc/                          # Linux 环境构建与检查脚本
    └── profiles/core/RL_env/bin/python # Linux CUDA/MJWarp core 环境
```

所有环境统一由同级 [`RL_env`](../RL_env/README.md) 管理。本目录的 `setup_env.sh` 和 `hpc/setup_hpc.sh` 仅验证共享环境，不会另建或更新项目内虚拟环境。MPS 的版本参考见 `requirements-lock.txt`；HPC 的严格约束见 `hpc/requirements-hpc.txt`。这些文本是依赖清单，不是重复虚拟环境。

HPC 启动器默认寻找 `../../RL_env/profiles/core/RL_env`（相对于 `hpc/`）。如果 HPC 环境部署到其他位置，可显式设置：

```bash
export IRP_HPC_CORE_ENV=/absolute/path/to/RL_env/profiles/core/RL_env
```

共享环境使用目标主机的 Python，迁移到其他主机后应通过 `RL_env/build.sh` 重建；不捆绑额外解释器。macOS 本机环境不能直接复制到 Linux 使用。请在目标 Linux GPU 节点按 `RL_env/README.md` 构建 HPC profile。GPU 训练需要兼容的 NVIDIA GPU、驱动与 CUDA PyTorch；MPS 链路的物理仿真仍使用 CPU MuJoCo，PyTorch 学习/推理可使用 MPS。

## Apple Silicon：验证与小规模训练

以下命令均从 `Diffusion/` 执行。`.command` 启动器会定位同级环境并设置工作目录，因此也支持从其他目录用绝对路径调用。

```bash
bash setup_env.sh
./run_screw.command --check
./run_screw.command train_diffusion.py --help
./run_mps_tuning.command all
```

`all` 默认只做小预算链路联调（8 条示范、5 个 Diffusion epoch、两段各 8,000 步残差训练、3 次评估），不是收敛训练配置。参数可通过 `SCREW_MPS_DEMO_EPISODES`、`SCREW_MPS_DIFFUSION_EPOCHS`、`SCREW_MPS_WRENCH_STEPS`、`SCREW_MPS_NONE_STEPS`、`SCREW_MPS_EVAL_EPISODES` 调整；也可分别执行 `demos`、`diffusion`、`wrench`、`none`、`eval`。

需要逐步控制训练时：

```bash
./run_screw.command generate_demos.py --episodes 100 --output datasets/demos
./run_screw.command train_diffusion.py \
  --dataset datasets/demos --output checkpoints/diffusion.pt \
  --epochs 200 --batch-size 256 --device mps
./run_screw.command train_residual.py \
  --baseline diffusion --diffusion-checkpoint checkpoints/diffusion.pt \
  --diffusion-device mps --device mps --timesteps 5000000 \
  --output checkpoints/residual_tqc
```

生成示范和启动训练前需确认输出路径及预算。默认全量训练预算较大；参数详情由各入口 `--help` 提供。无图形界面的任务应调用训练/评估入口；`run_screw.py` 是可选 MuJoCo 交互调试入口，macOS 上启动器会为它选择 `mjpython`。

## HPC：预检、训练与恢复

从 `Diffusion/hpc/` 执行：

```bash
bash setup_hpc.sh
./run_hpc.sh preflight.py --mode static --config configs/smoke.json
./run_hpc.sh preflight.py --mode gpu --config configs/smoke.json
./run_hpc.sh pipeline.py --config configs/smoke.json \
  --stage smoke --run-dir runs/smoke --resume never
```

`pipeline.py` 会按配置串联示范生成、Diffusion 训练、TQC 训练和评估。训练启动后的输出位于所选 `runs/` 子目录。已有合法检查点的恢复使用 `--resume auto`；新任务使用新的 `--run-dir` 和 `--resume never`。

```bash
./run_hpc.sh pipeline.py --config configs/hx1_a100.json \
  --stage full --run-dir runs/hx1_full --resume auto
./run_hpc.sh status_hpc.py --run-dir runs/hx1_full
```

`--stage diffusion-smoke`、`diffusion-full`、`diffusion-eval` 可运行不训练 TQC 的 Diffusion 链路。`generate_demos_gpu.py`、`train_diffusion_gpu.py`、`train_residual_gpu.py`、`evaluate_gpu.py` 是分阶段入口。

| 配置/入口 | 用途 |
| --- | --- |
| `configs/smoke.json` | 8-world 小规模完整训练测试 |
| `configs/hx1_a100*.json`、`jobs/hx1_*.pbs` | HX1 / A100 预设与 PBS 作业 |
| `configs/cx3_l40s*.json`、`jobs/cx3_*.pbs` | CX3 / L40S 预设与 PBS 作业 |
| `jobs/hx2_TEMPLATE.slurm` | 需按实际集群填写的 Slurm 模板 |
| `configs/autodl_rtx_pro_6000*.json` | AutoDL RTX PRO 6000 预设 |
| `capacity_probe.py`、`adaptive_capacity.py` | world 数/显存容量探测与自适应选择 |
| `training_stack_preflight.py` | 实际训练栈、重放缓冲区和长 episode 压力验证 |
| `collect_diagnostics.py`、`system_monitor.py` | 故障诊断与训练遥测 |

集群参数、容量和固定 geometry 契约由预检验证，不应仅凭显存大小增大 world 数。详见 [`hpc/README_HPC.md`](hpc/README_HPC.md) 及同目录各硬件说明。

## 工作空间评估与成功轨迹自训练

`evaluation/workspace_policy_rollouts.py run` 在固定极坐标网格上执行策略，保留数值结果及可选成功轨迹数据，不绘制图片。`--export-success-demos` 将成功的真实策略 rollout 导出为有长度信息的 DemoStore，供 Diffusion 自蒸馏。它不把策略 rollout 标记为专家示范。

在 Linux HPC 上从 `Diffusion/` 调用示例：

```bash
../RL_env/profiles/core/RL_env/bin/python \
  evaluation/workspace_policy_rollouts.py run \
  --diffusion hpc/runs/fresh/checkpoints/diffusion.pt \
  --config hpc/configs/autodl_rtx_pro_6000_r100_success_selfdistill_w18000_tqc16.json \
  --output-dir results/workspace --export-success-demos
```

检查点需先通过新训练产生。高级入口 `run_autodl_r100_success_selftrain_w18000_tqc16.sh` 串联已有策略扫描、成功数据自蒸馏、18,000-world TQC 及评估；`run_autodl_r100_w5000_tqc16_continuation.sh` 用于合法已有检查点的 5,000-world 续训。它们不是空仓库首次训练入口。

`run_autodl_full_workspace_diffusion_reachability.sh` 可通过 `SCREW_SCAN_DIFFUSION`、`SCREW_SCAN_TQC`、`SCREW_SCAN_CONFIG` 指定输入。默认无需残差检查点；只有 `CONTROLLER_MODE=diffusion-ik-tqc` 才使用 TQC。成功扫描数据默认写入 `results/workspace_policy_rollouts.successful_policy_rollouts`。

A* 对照评估由 `hpc/evaluate_astar_spline_gpu.py`、`hpc/evaluate_paired_planners_gpu.py` 提供，用于训练效果比较和规划契约回归；论文专用绘图和精选轨迹录制不在源树中。

## 测试与验证范围

MPS/CPU 测试（从 `Diffusion/`）：

```bash
PYTHONDONTWRITEBYTECODE=1 ../RL_env/bin/python -m unittest \
  test_geometry test_task test_training test_runtime -q
PYTHONDONTWRITEBYTECODE=1 ../RL_env/bin/python test_diffusion.py --smoke-test
```

HPC 的 CPU 可执行回归测试与静态预检（在 Mac 上使用本机共享 Python）：

```bash
cd hpc
PYTHONDONTWRITEBYTECODE=1 ../../RL_env/bin/python -m unittest discover -s tests -q
PYTHONDONTWRITEBYTECODE=1 ../../RL_env/bin/python preflight.py \
  --mode static --config configs/smoke.json
```

回归测试覆盖几何、IK、动作/观测契约、TQC 更新、重放缓冲区、微型 Diffusion 训练、检查点恢复、工作空间成功数据导出、容量选择和打包排除规则。GPU 可用性、CUDA kernel 与大规模训练吞吐必须在目标 Linux GPU 节点运行 `preflight.py --mode gpu` 及训练栈预检；Mac 的 CPU 测试不能代替它们。

## STL 资产与 GitHub 发布

只保留仿真所需的 7 种 STL 网格，不包含 SolidWorks 文件。`mesh/` 服务根级 MPS XML，`hpc/assets/mesh/` 服务 HPC 的自包含资产包。两份网格随各自 XML 一并保留，HPC 的 geometry 哈希与源包审计依赖后者。XML 是训练输入定义，不是训练权重，应提交到 Git。

不要提交 `RL_env` 的已安装二进制、缓存、数据集、运行目录和检查点。环境构建脚本和依赖清单在同级 `RL_env` 中管理。`hpc/build_hpc_package.sh` 可生成不含环境和训练结果的 HPC 源码归档，默认包含 HPC XML/STL；独立部署该归档时请设置 `IRP_HPC_CORE_ENV`，并连同本目录的 `evaluation/` 使用需要跨目录评估的测试。GitHub 推荐直接提交完整目录结构，以保留全部训练和评估入口。
