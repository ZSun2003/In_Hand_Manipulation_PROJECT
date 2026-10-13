# RL_env：共享训练环境

本目录属于已完成 IRP 项目的源码归档。集中管理本项目的 Python 环境、依赖声明和启动方式。其他七个顶层目录不再保留虚拟环境副本。现有本机环境已经迁移到此目录；上传 GitHub 时只提交本目录的管理脚本与依赖文件，安装好的二进制环境由 `.gitignore` 排除。

## 项目布局

```text
IRP/
├── Benchmark/     # 原 Gripper3：源模型、运动学、workspace
├── RL16-PA/       # 原 Gripper4
├── RL16-AP/       # 原 Gripper5
├── RL16-API/      # 原 Gripper6
├── IK-residual/   # 原 G3G6：解析 IK + TQC 残差训练
├── Diffusion/     # 原 SCREW：本机与 HPC/MJWarp 训练
├── Diffusion_FutureWork/ # YCB GRAB 后续研究训练与源模型
└── RL_env/        # 本目录
```

四个机械手的 CAD/STL 源资产位于各自 `models/` 下。IK-residual 和 Diffusion 单独保留训练场景及其 STL；这些是仿真输入，不是训练权重。项目不提供旧 checkpoint、训练数据、输出图片、论文或 LaTeX 推导文件。

## 环境目录与版本

| 用途 | Python 路径 | 主要依赖 |
| --- | --- | --- |
| 本机 MPS/CPU，IK residual | `bin/python` | MuJoCo 3.7.0、Torch 2.11.0、NumPy 2.4.4、Gymnasium 1.2.3、SB3/TQC 2.8.0 |
| HPC/MJWarp core | `profiles/core/RL_env/bin/python` | MuJoCo 3.10.0、mujoco-warp 3.10.0.3、Warp 1.14.0、Torch 2.7.1 CUDA 12.8 |

Python 使用 3.11 或 3.12；当前保留的本机环境为 macOS arm64 / CPython 3.11.15。Mac 的 MuJoCo 物理仿真在 CPU 上运行，MPS 加速策略网络。大规模 CUDA 物理仿真使用 Diffusion 的 `hpc/` 链路。

两组环境有不同的 MuJoCo/Torch 数值与接口契约，不能将 HPC core 的包直接覆盖进本机环境。每个平台在本目录统一安装兼容配置，不向每个项目复制环境。

```text
RL_env/
├── README.md
├── build.sh                 # 本机/IK 环境与 HPC 构建入口
├── run.sh                   # 按项目选择解释器和工作目录
├── activate.sh              # bash/zsh 激活共享配置
├── doctor.py                # 导入、MuJoCo 步进与设备检查
├── requirements/
│   ├── local.txt            # 可用于 Mac 或 Linux 的直接依赖
│   └── local-macos-lock.txt # 整理时现有 Mac 环境完整版本快照
├── hpc/
│   ├── build.sh             # Linux CUDA 安装实现
│   ├── profiles.json        # HPC 版本契约
│   ├── requirements/       # 精确固定的 HPC 直接依赖
│   └── verify.py            # HPC 平台、版本和可选 GPU 验证
├── bin/ lib/ include/ share/ pyvenv.cfg  # 本机已安装环境，不提交 Git
└── profiles/                # 集中安装的其他平台配置，不提交 Git
```

## 本机重建与使用

从 IRP 根目录执行。安装依赖会联网；当前已安装的环境可直接使用，无需重建。

```bash
PYTHON_BIN=python3.11 bash RL_env/build.sh --profile local
RL_env/run.sh local RL_env/doctor.py
RL_env/run.sh local RL_env/doctor.py --require-mps
```

`requirements/local-macos-lock.txt` 是现有 Mac 环境的完整依赖版本快照，包含平台相关包；跨平台安装应使用 `local.txt`。其中 Matplotlib 是 Stable-Baselines3 的上游依赖，保留它是为了正常导入 RL 框架；本项目的绘图实现已移除。

可激活环境后使用普通 Python 命令：

```bash
source RL_env/activate.sh local
python IK-residual/preflight.py
```

激活脚本支持 bash/zsh。也可始终显式调用解释器，避免当前 shell 环境影响训练。

```bash
RL_env/run.sh ik-residual train_curriculum.py --help
RL_env/run.sh ik-residual train_safe_residual.py --gripper both --smoke --output-dir runs/smoke
RL_env/run.sh diffusion-mps train_diffusion.py --help
RL_env/run.sh diffusion-mps train_residual.py --help
```

`run.sh` 切换到所选项目目录，因此随后给 Python 的相对脚本路径以该项目为基准。`local` 不改变工作目录。`diffusion` 在 Mac 默认选择本机链路，在 Linux 默认选择 HPC；`diffusion-mps` 和 `diffusion-hpc` 可以明确指定。

## Linux / HPC 安装

上传源码与依赖文件，不上传 Mac 的 `bin/`、`lib/` 或其他环境二进制。Linux 需要 x86_64、glibc 2.28+、CPython 3.11/3.12；GPU 节点还需兼容的 NVIDIA 驱动和系统 EGL 库。

IK-residual 使用 MuJoCo CPU 仿真和 CUDA 策略网络，先在 Linux 创建主环境：

```bash
PYTHON_BIN=python3.11 bash RL_env/build.sh --profile local
RL_env/run.sh local RL_env/doctor.py --require-cuda
RL_env/run.sh ik-residual train_curriculum.py --gripper g3 --device cuda --print-config
```

`local` 表示主环境依赖合同，并不限制为 Mac。Linux Torch 的 CUDA 可用性由安装的 wheel 与机器驱动决定，必须以 `--require-cuda` 检查为准。不要为 IK-residual 改用 `core`，后者属于另一套 MuJoCo/MJWarp 合同。

Diffusion 的 MJWarp 训练使用共享 core：

```bash
PYTHON_BIN=python3.11 bash RL_env/build.sh --profile core
RL_env/profiles/core/RL_env/bin/python RL_env/hpc/verify.py core --require-gpu
RL_env/run.sh diffusion-hpc preflight.py --mode static --config configs/smoke.json
```

具体 GPU 型号、并行环境数、PBS/AutoDL 启动方式和训练顺序见 `../Diffusion/README.md`。普通静态检查不等于 GPU 实测，仍应在分配好的 GPU 节点运行项目的严格 preflight/smoke。

HPC 构建支持 `PYTHON_BIN`、`TORCH_INDEX_URL`、`PYPI_INDEX_URL` 和 Linux `WHEELHOUSE` 覆盖。首次构建后会生成安装版本与状态记录；它们不作为源码提交。

## 搬迁、设备与故障排查

- venv 含绝对 shebang 和平台相关二进制，不是跨操作系统的可移植压缩包。克隆到新机器后按对应平台重建，不依赖旧用户目录。
- 缺少 `bin/python`：先安装受支持的 Python，再运行 `build.sh --profile local`。
- MPS 不可用：先执行 `doctor.py --require-mps`，确认使用原生 Apple Silicon Python；需要 CPU 时给训练入口显式传 `--device cpu`。
- 缺少 core：在 Linux 执行 `build.sh --profile core`。不要在 Mac 安装 CUDA profile。
- `run.sh` 支持 `RL_BACKEND=local|mps|cpu|hpc` 选择项目链路；设备本身由训练入口的 `--device` 选择。`RL_PYTHON=/绝对路径/python` 可显式覆盖解释器，使用者须保证符合项目依赖契约。
- 不同训练场景的 XML/STL 保持各自数值合同，不应仅因为文件名相同就相互替换。

## GitHub 提交与验证

提交七个项目的源码、源模型、测试、README，以及本目录的管理脚本、依赖声明和 `.gitignore`。无需提交已安装环境、pip/torch 缓存或重新生成的训练输出。本次整理没有初始化 Git 仓库，也没有上传或发布到远端。

核心验证命令：

```bash
RL_env/run.sh local RL_env/doctor.py
RL_env/run.sh ik-residual -B -m unittest discover -s tests -v
RL_env/run.sh ik-residual preflight.py
```

各项目还保留了自己的训练与场景测试，见对应 README。完整训练收敛、Linux GPU 性能及 HPC 集群调度需要在目标硬件运行，不由本机冒烟检查代替。

## 本次归档校验记录

2026-09-11 完成目录迁移与清理后，验证范围如下：

| 归档部分 | 验证结果 |
| --- | --- |
| 四手机构 | 21 项现有测试通过；6 套 MuJoCo 主场景、4 套 IK 控制器及数值 workspace 检查通过；212 个 CAD/STL 源文件 SHA-256 校验通过 |
| IK-residual | 102 项测试、39 项预检通过；两个机械手的短训练链路通过，验证产物已删除 |
| Diffusion | 本机 52 项、HPC CPU 侧 244 项测试完成，各跳过 1 项；微型 Diffusion 优化、采样和保存重载通过 |
| Diffusion_FutureWork | 472 项适用测试通过；34 个训练场景实际加载、161 项资源引用、YCB 原始 PLY hash 校验通过 |
| 共享环境 | 依赖检查、本机 MPS 张量计算、MuJoCo 步进、bash/zsh 激活及全部项目启动路径检查通过 |

FutureWork 的三项测试样例经过独立核对后校正：零残差比较锁定同一控制步；香蕉计划测试使用规划器实际选择的名义角度。保留严格物理断言和零动作逐位相等检查，没有通过修改训练算法、控制参数或放宽数值容差使测试通过。

归档未重新运行长训练或 Linux CUDA 仿真。FutureWork 的生产容量证书及相关输出已按归档范围移除，原严格资格认证入口保留；正式 GPU 训练仍需在目标节点完成相应认证。各项目的标准输出目录、缓存、旧权重、轨迹、论文及压缩包均不在本归档中。YCB 必需源纹理随模型保留，其来源见 `../Diffusion_FutureWork/README.md`。
