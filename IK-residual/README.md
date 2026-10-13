# IK-residual

这是已完成 IRP 项目的源码归档。Benchmark（原 Gripper3）与 RL16-API（原 Gripper6）的解析 IK + TQC 残差强化学习。此目录由原 `G3G6` 项目整理而来，包含训练、仿真环境、策略评估和必要的解析控制基线。两个机械手分别训练策略，不共享动作输出层。

Python 包名 `g3g6` 与命令参数 `g3` / `g6` 保留，方便已有算法接口与 checkpoint 元信息继续识别机械手；目录和入口名称已经按职责整理。

| 命令参数 | 机械手名称 | 动作维度 | 控制通道 |
| --- | --- | --- | --- |
| `g3` | Benchmark | 12 | 12 个旋转关节残差 |
| `g6` | RL16-API | 16 | 12 个旋转关节 + 4 个直线关节残差 |

## 目录结构

```text
IK-residual/
├── train_curriculum.py                 # 固定三阶段正式训练
├── train_safe_residual.py              # 可配置安全预热训练、快速 smoke
├── evaluate_policy.py                  # checkpoint 定量评估
├── evaluate_translation_residual.py    # 平移/平移旋转耦合评估
├── evaluate_rotation_residual.py       # 原地旋转扫参评估
├── evaluate_rotation_endpoints.py      # 可分片、可续跑的端点评估
├── evaluate_ik_baseline.py              # 解析 IK 对照基线
├── evaluate_ik_rotation_baseline.py     # 解析 IK 原地旋转对照基线
├── ablate_api_linear_residual.py        # RL16-API 直线通道消融
├── preflight.py                        # 依赖、场景、IK 和控制契约检查
├── g3g6/
│   ├── config.py                      # 动作/观测、时序、奖励和场景常量
│   ├── env.py                         # Gymnasium + MuJoCo 环境
│   ├── training.py                    # SafeWarmupTQC 与可配置训练
│   ├── rotation_residual_training.py   # 正式三阶段 TQC 课程
│   ├── residual.py                    # 残差限幅、限速和单位转换
│   ├── objects.py                     # 训练物体及接触目标
│   ├── trajectories.py                # 平移轨迹和阶段时序
│   ├── rotations.py / math3d.py        # 旋转目标、姿态和几何运算
│   ├── kinematics/g3.py               # Benchmark 解析运动学
│   ├── kinematics/g6.py               # RL16-API 解析运动学
│   └── ...                            # 与各评估入口对应的实现
├── model/
│   ├── g3/                            # Benchmark 训练专用场景
│   │   ├── Project.xml
│   │   ├── Gripper.xml / Share.xml
│   │   └── mesh/*.STL
│   └── g6/                            # RL16-API 训练专用场景，结构相同
└── tests/                             # 控制、环境、课程和评估契约测试
```

`model/` 内的 XML 是 MuJoCo 源配置，STL 是仿真几何源资产，均应随 GitHub 项目提交。它们不是训练生成的策略模型。场景以固定根高度、桌面、物体质量和闭链约束定义本项目的训练契约；不能直接用其他工作空间场景替换。这里不包含 SolidWorks 工程、论文、绘图工具、视频工具、历史训练结果或预训练权重。

## 环境与启动

八个顶层目录共享相邻的 [`../RL_env`](../RL_env/README.md)。此目录不创建或保留独立虚拟环境。完整依赖、Mac MPS/CPU 路径与 HPC/CUDA 配置以该目录的说明为准。

以下命令均从 `IK-residual/` 运行：

```bash
../RL_env/bin/python preflight.py
../RL_env/bin/python train_curriculum.py --help
../RL_env/bin/python train_safe_residual.py --help
```

顶层 Python 入口在直接执行时会使用 `../RL_env/bin/python`；若已经由共享 `RL_env` 内的 HPC profile 启动，则保持该解释器。内部包可在已激活的共享环境中通过 `python -m g3g6.<模块>` 运行。所有路径均由源码位置定位，无需修改用户主目录路径。

Mac 上显式传入 `--device mps`，无 GPU 时使用 `--device cpu`；HPC 的 CUDA 环境使用 `--device cuda`。`auto` 由 Stable-Baselines3 选择设备，不保证自动选中 MPS。本项目的 MuJoCo 物理步进在 CPU 上执行，策略网络可在 MPS/CUDA 上执行；MJWarp 大规模仿真训练位于相邻的 `Diffusion` 项目。

## 训练方法与入口

环境使用统一的 149 维观测，对 Benchmark 的残差状态进行零填充。解析 IK 产生名义动作，策略输出各自机械手原生顺序的有界残差。角度和直线通道采用不同的单位、幅值和速率限制。

### 正式三阶段课程

`train_curriculum.py` 执行既有的固定训练配置：

| 阶段 | 任务 | 请求环境步数 |
| --- | --- | --- |
| 1 | 抓取与抬升 | 500,000 |
| 2 | 平移 | 2,000,000 |
| 3 | 旋转 | 2,000,000 |

每个机械手使用 72 个并行环境，向量化会将实际步数向上补齐到 72 的倍数。TQC 策略网络为 `[256, 256]`，4 个 critic、每个 32 个 quantile，学习率 `1e-4`，batch size 256，replay buffer 1,000,000。旋转阶段使用 0–90° 的加权采样。可先打印完整配置，确认资源需求后运行：

```bash
../RL_env/bin/python train_curriculum.py --gripper both --device mps --print-config
../RL_env/bin/python train_curriculum.py --gripper g3 --device mps --output-dir runs/full_residual
../RL_env/bin/python train_curriculum.py --gripper g6 --device mps --output-dir runs/full_residual
```

HPC 中先按 `RL_env/README.md` 使用 `RL_env/build.sh --profile local` 在 Linux 重建本项目需要的共享运行时（MuJoCo 3.7 / PyTorch 2.11 / SB3），然后从八个目录的共同父目录运行 `RL_env/run.sh ik-residual train_curriculum.py --gripper g6 --device cuda --output-dir runs/full_residual`。Diffusion 的 MJWarp core profile 使用不同依赖组合，不作为本项目的默认运行时。72 个 CPU 仿真 worker 和 replay buffer 需要相应 CPU、内存和文件描述符资源；该入口有意保持原正式实验配置。

### 可配置训练和 smoke

`train_safe_residual.py` 提供独立的 `SafeWarmupTQC` 路径：以名义控制器采集初始经验，先更新 critic，再启用带残差正则的 actor；可修改环境数、各阶段步数、向量环境实现和 checkpoint 周期。

```bash
../RL_env/bin/python train_safe_residual.py --gripper both --smoke --output-dir runs/smoke
../RL_env/bin/python train_safe_residual.py --gripper g6 --device mps --n-envs 4 --vec-env subproc --phase-steps 500000 2000000 2000000 --output-dir runs/safe_residual
```

`--smoke` 固定使用 CPU、一个 dummy 环境和 128/32/32 步，包含实际 TQC 梯度更新。它用于确认训练链路，不代表策略收敛。

这两个训练器的学习规则和产物格式不同：正式课程写出 `vec_normalize.pkl` 及完整校验 manifest；安全预热训练写出 `vecnormalize.pkl`。平移/旋转 residual sweep 和端点评估读取正式课程产物，并验证动作维度、配置以及文件 hash。安全预热 checkpoint 可使用 `evaluate_policy.py` 进行基础评估。

## 策略与基线评估

评估前必须完成对应机械手的训练；项目中没有保留旧权重。以下以正式课程输出为例：

```bash
../RL_env/bin/python evaluate_policy.py --gripper g3 --phase 3 --model runs/full_residual/g3/final_model.zip --vecnormalize runs/full_residual/g3/vec_normalize.pkl --episodes 20
../RL_env/bin/python evaluate_translation_residual.py --gripper g3 --g3-training-run runs/full_residual/g3 --objects cube --paths circle --axes xyz --output-dir evaluation/g3_translation
../RL_env/bin/python evaluate_rotation_residual.py --gripper g6 --g6-training-run runs/full_residual/g6 --objects cube --axes xyz --output-dir evaluation/g6_rotation
../RL_env/bin/python evaluate_rotation_endpoints.py run --gripper g6 --training-run runs/full_residual/g6 --output-dir evaluation/g6_endpoint_part0 --shard-index 0 --num-shards 4
../RL_env/bin/python ablate_api_linear_residual.py --training-run runs/full_residual/g6 --output-csv evaluation/api_linear_ablation/cases.csv
```

`evaluate_ik_baseline.py` 和 `evaluate_ik_rotation_baseline.py` 使用同一仿真任务产生解析 IK 基线，用于衡量残差策略的增益。它们保留数值轨迹与评估统计功能，已移除所有绘图调用：

```bash
../RL_env/bin/python evaluate_ik_baseline.py --gripper g3 --objects cube --paths circle --axes xyz --suite full --output-dir evaluation/g3_ik_baseline
../RL_env/bin/python evaluate_ik_rotation_baseline.py --gripper g6 --objects cube --axes xyz --max-angle-deg 45 --output-dir evaluation/g6_ik_rotation
```

端点评估的分片恢复和合并选项、其他评估参数均可通过入口 `--help` 查看。

## 测试

```bash
../RL_env/bin/python -B -m unittest discover -s tests -v
../RL_env/bin/python preflight.py
```

本次整理后通过 102 个单元测试、39 项 preflight 检查，以及两机械手各自的 128/32/32 步三阶段 smoke（含实际梯度更新）；没有执行完整规模训练。

测试覆盖 XML/include/STL 的依赖闭包、关节与控制顺序、动作维度、149 维观测、IK/残差限速和限幅、轨迹时序、物体目标、三阶段配置、残差评估的配置与产物契约。它们不运行完整规模的训练。

## 输出与 GitHub 提交

训练运行后会新建 `runs/`，评估运行后会新建 `evaluation/`。checkpoint、归一化统计、日志、CSV 轨迹、JSON 报告均属于运行产物；当前整理后的目录没有保留这些旧输出，`.gitignore` 已排除标准产物目录及缓存。

发布代码时提交入口、`g3g6/`、`tests/`、`model/` 和本 README，同时保留相邻 `RL_env` 的可复建依赖配置。不要提交虚拟环境二进制目录或后续训练生成的权重。若选择自定义输出位置，应将它放到 `runs/` 或 `evaluation/` 下；输入输出路径按本项目根目录解析。
