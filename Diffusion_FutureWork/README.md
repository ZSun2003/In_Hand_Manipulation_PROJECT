# Diffusion_FutureWork — YCB 抓取研究源码归档

这是已完成 IRP 项目中 YCB GRAB 研究方向的源码归档。名称中的 `FutureWork` 表示当时保留的研究方向，不代表当前有待继续执行的训练任务。本次整理只保留机械手对应的 Diffusion/RL 模型定义、训练与评估代码，以及训练依赖的控制器、运动学、仿真 XML、机械手网格和 YCB 物体模型。

归档来源为原有两手机械手 GRAB 子项目；这里不包含另一个已排除的独立 `GRAB_G6_DIFFUSION` 目录，也不将两份历史实现拼接为一个新算法。没有保留预训练权重、数据集、经验轨迹、模型压缩包、图片报告、日志、论文、LaTeX、图形 viewer、视频与轨迹回放工具。

## 布局与版本边界

```text
Diffusion_FutureWork/
├── benchmark/                         # 原 Benchmark / Gripper3 GRAB
│   ├── G3_grab.py                      # 供训练调用的无图形解析控制器
│   ├── _g3_kinematics.py / _g3_objects.py
│   ├── G3_*.xml                       # 全物体及单物体训练场景
│   ├── fruit_capsules_5mm*.{json,xml}  # YCB 碰撞几何源定义
│   ├── IK_RL/                         # 本机 SB3 TQC 残差训练
│   ├── MJWarp_RL/                     # HPC Diffusion / IK / TQC 组件
│   └── models/
│       ├── mesh/                      # 机械手 STL 与仿真物体网格
│       └── objects/ycb/               # 4 个实际训练使用的 YCB 水果
└── rl16_api/                          # 原 RL16-API / Gripper6 GRAB
    ├── _g6_grab.py
    ├── _g6_kinematics.py / _g6_objects.py
    ├── _g6_*.xml
    ├── fruit_capsules_5mm*.{json,xml}
    ├── IK_RL/
    ├── MJWarp_RL/
    └── models/                        # 自含本手机械手与 YCB 源模型
```

两个平台各自保留 `mesh`、`ycb`、`ycb-tools` 的内部相对符号链接，用于兼容原 XML 和模型清单的资源路径；实际源模型集中在各平台的 `models/` 中。符号链接最终都指向同一平台内部。

| 项目 | `benchmark` | `rl16_api` |
| --- | --- | --- |
| 机械手物理执行器 | 12 个旋转 | 12 个旋转 + 4 个直线 |
| 本机 `IK_RL` 动作 | 12 执行器残差 + 2 抓取规划残差 | 16 执行器残差 |
| 本机 `IK_RL` 默认预算 | 2,000,000 步，8 环境 | 2,000,000 步，8 环境 |
| `MJWarp_RL` 历史版本 | r21，共享/配对协议 | r22，G6-only 协议 |
| HPC 单任务预算 | 60,000,000 步 | 90,000,000 步 |
| HPC world 数 | 20,000 | 20,000 |
| HPC episode ticks | 3,000 | 4,500 |
| HPC 根高度 | 0.525 m | 0.560 m |
| HPC 策略接口 | 607 维观测、25 维任务空间动作 | 607 维观测、25 维任务空间动作 |

预算、物理常量和算法均沿用各自现有源码。r21 与 r22 的 episode 时序、模型高度、容量证书 schema 和 hash 契约不同，不能交换 checkpoint 或把当前两份调度器当作同一套双手正式实验。原 r21 双手调度器按其原协议要求兼容的 G6 r21 配对输入；本归档的 API 分支为 r22。

## 代码职责

两个 `MJWarp_RL/` 均包含：

| 模块 | 职责 |
| --- | --- |
| `diffusion_policy.py` | 动作分块 Diffusion 模型与学习组件 |
| `diffusion_runtime.py` | 训练/推理运行接口 |
| `demos.py`、`ik_expert.py` | 经控制契约校验的 IK 专家示范与采样 |
| `train.py`、`curriculum.py` | 训练入口和课程/阶段逻辑 |
| `tqc.py`、`replay.py` | 原有 TQC 网络、更新和 replay buffer 定义 |
| `env.py`、`sharded_env.py` | 向量化任务与按物体分片环境 |
| `mjwarp_backend.py` | MuJoCo Warp / Torch 后端 |
| `platform_adapter.py` | 各机械手场景、执行器、IK 和数据布局适配 |
| `contract.py`、`hpc_config.py` | 任务、观测/动作、采样和训练常量 |
| `newton_regularization.py` | 原有物理求解器正则化约束 |
| `capacity_scan.py`、`preflight.py` | 硬件/容量与训练前契约验证 |
| `memory_certificate.py`、`runtime_memory_guard.py` | 容量证书和运行时内存边界 |
| `runtime_fingerprint.py` | 影响正式训练的源码 hash 清单 |
| `evaluate.py`、`success.py` | 定量策略评估和逐轴成功判断 |
| `live_metrics.py`、`status_hpc.py` | 数值训练指标与终端状态 |
| `sequential_train_hpc.py` | 对应历史版本的正式训练调度源码 |

`IK_RL/` 保存本机 Gymnasium + MuJoCo + Stable-Baselines3 TQC 残差训练，`controller_bridge.py` 将策略残差连接到保留的解析控制器；`runtime.py` 校验控制器、物体几何与源模型 hash。GRAB 控制器只提供无图形的抓取、平移和旋转控制，已移除交互菜单、能力图表、viewer 与图片保存。

## 共享运行时

本目录不包含虚拟环境。与其他归档同级的 [`RL_env`](../RL_env/README.md) 管理两条运行链路：

- 本机 `IK_RL`：`RL_env/bin/python`，MuJoCo 3.7 / PyTorch 2.11 / SB3，可选 MPS 或 CPU。
- Linux HPC `MJWarp_RL`：`RL_env/profiles/core/RL_env/bin/python`，MuJoCo 3.10 / CUDA PyTorch 2.7.1 / MJWarp。

macOS 的已安装环境不能直接复制到 Linux 运行；依赖重建步骤以 `RL_env/README.md` 为准。`IRP_PYTHON_ENV` 可显式指定共享 Python 环境，HPC 启动器可通过 `IRP_HPC_CORE_ENV` 指定集中管理的 core profile。

两个平台中都存在名为 `IK_RL` 和 `MJWarp_RL` 的包，因此请在独立 Python 进程中选择一个平台，不要把两个平台同时添加到同一 `PYTHONPATH`。根启动器会选择工作目录：

```bash
# 从包含这八个归档目录的共同父目录执行；以下只查看帮助，不训练。
RL_env/run.sh grab-g3 -m IK_RL.train --help
RL_env/run.sh grab-g6 -m IK_RL.train --help
RL_BACKEND=hpc RL_env/run.sh grab-g3 -m MJWarp_RL.train --help
RL_BACKEND=hpc RL_env/run.sh grab-g6 -m MJWarp_RL.train --help
```

也可从 `benchmark/` 或 `rl16_api/` 执行 `../../RL_env/bin/python -m IK_RL.train --help`。HPC 原入口保留为 `./MJWarp_RL/run_hpc.sh -m MJWarp_RL.train --help`，同样仅引用共享 core 环境。

归档没有原容量证书、示范数据或策略权重。正式 HPC 训练前的严格验证仍保留；历史调度器中的零值证书/算法 pin 代表未配置的输入，不能把它们替换成伪造证书，也不能用 smoke 结果代替完整容量认证。单元测试和入口帮助并不代表已有可复现的收敛结果。

## 模型资产与 YCB 来源

只保留本训练实际使用的四种 YCB 物体：`011_banana`、`012_strawberry`、`016_pear`、`017_orange`。cube、cylinder、cone 等基本物体由 XML 原语或本地网格定义。

`models/objects/ycb/<物体>/google_16k/` 包含原始 PLY/STL/OBJ/DAE、MTL 和 `texture_map.png`。PNG 是 XML/MTL/DAE 引用的源纹理，不是实验输出图片；必须与网格一起提交。API 的 `models/mesh/_g6_ycb/` 保留其 XML 使用的相应网格与材质副本。

YCB 数据来自 [官方 YCB 模型库](https://ycb-benchmarks.s3.amazonaws.com/index.html)，源模型采用 CC BY 4.0。四种纹理从对应官方 `data/google/<物体>_google_16k.tgz` 恢复，未编辑或重新生成；每个包中的 `textured.obj` 和 `nontextured.ply` 均与归档两平台已有源网格通过 SHA-256 逐字节核对，确认版本一致。

`fruit_capsules_5mm.json` 和配套 XML 是训练使用的碰撞几何源定义。清单记录原 PLY 的 hash、胶囊参数与 5 mm 表面逼近约束；它是模型组成部分，不是历史训练输出。加载时会检查源 PLY 和碰撞定义的一致性。STL/OBJ 几何、XML 刚体/惯量/碰撞配置均应随代码提交。

## 归档验证

从对应平台目录执行：

```bash
../../RL_env/bin/python -B -m unittest discover -s IK_RL/tests
../../RL_env/bin/python -B -m unittest discover -s MJWarp_RL/tests
```

整理时验证源码语法、必要模块导入、两平台的全部 34 个 Project XML 实际加载、模型/include/MTL/DAE/纹理引用闭包、YCB PLY hash 与控制器桥接契约。MJWarp_RL 的适用 CPU 测试：Benchmark 234 项、RL16-API 213 项通过，覆盖模型接口、TQC/Diffusion 数值组件、数据采样、动作限幅、阶段时序和配置绑定。本机 IK_RL 测试：Benchmark 14 项、RL16-API 11 项全部通过。没有重新运行长训练、CUDA 仿真或收敛实验。

根据独立的规划器与控制时序检查，校正了三项本机测试样例：Benchmark 的零残差与强制零动作基线比较同一初始状态、规划和终止状态，不再假定候选规划器固定选择某个 yaw；香蕉接触时序样例使用零规划残差，并保留四触点、触点到达不同步等原物理断言。API 零残差测试比较同一控制 tick 的已锁存名义动作与执行动作，继续使用逐位相等断言。这些校正没有改变规划/控制算法或放宽数值容差，也不据此断言迁移前已存在相同测试结果。

原 API 测试中只针对旧 r21 双手协议的项不适用于已有 r22 源码；归档保留有意义的算法/物理测试，并按已存在的 r22 配置核对版本差异，不通过修改算法去满足旧测试。

## GitHub 提交

提交本 README、两平台源码、测试、XML 和 `models/`，并保留内部相对符号链接。`.gitignore` 排除缓存、checkpoint、轨迹、数据集、日志、压缩包和运行结果，源模型目录及源纹理不在排除范围内。共同父目录中的 `RL_env` 只提交环境构建脚本与依赖清单，不提交已安装的 Python 二进制和包。
