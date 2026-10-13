# RL16-API

本目录是已完成 IRP 项目的归档，用于保存源模型、数值运动学和复现实验入口。

每指四路主动坐标；armbase → linear → fingers。闭链 IK 同时检查被动关节、冗余直线坐标与执行器限位。 本目录由原 `Gripper6` 整理而来，当前 MuJoCo 模型包含 **16 个主动执行器**。文件名前缀 `T6_` 表示该机械手的平移控制/计算实现，便于日志、导入与不同机械手代码区分。

## 目录与职责

| 路径 | 职责 |
| --- | --- |
| `translation/T6_kinematics.py` | 从本手 XML 读取几何、限位及执行器顺序，计算解析闭链运动学；提供 RL 可复用的 IK teacher |
| `translation/T6_IKcontrol.py` | MuJoCo 中的 IK 控制与无窗口运行入口 |
| `translation/T6_workspace.py` | 数值工作空间扫描；保留可达性、连续可达半径和数值结果导出 |
| `translation/Project.xml` | 本手平移仿真的主模型，包含 `Gripper.xml` 与 `Share.xml` |
| `models/cad/` | 原始 SolidWorks 装配、零件及原始根目录模型；保留原目录关系 |
| `models/simulation/` | 运行时仿真网格的统一实体存放位置 |
| `rotation/` | `R6_kinematics.py` 与组合旋转/平移的数值工作空间计算，以及独立 XML 模型 |
| `translation/T6_payload_capacity.py` | 解析准静态载荷计算；输出数值，不绘图 |

## 环境与运行

与仓库根目录的 `RL_env/` 配套使用。数值计算使用 NumPy，MuJoCo 控制使用 MuJoCo；载荷计算还使用 SciPy。Python 环境统一位于仓库根目录，目录内部不再创建 `RL_env`、`.venv` 或 `CX_env`。

以下命令在整个 IRP 仓库根目录执行：

```bash
bash RL_env/run.sh local RL16-API/translation/T6_kinematics.py --help
bash RL_env/run.sh local RL16-API/translation/T6_IKcontrol.py --help
bash RL_env/run.sh local RL16-API/translation/T6_IKcontrol.py --no-render
```

`--no-render` 用于终端和服务器；需要交互窗口时省略该选项。macOS 的 MuJoCo 交互窗口由共享环境中的 `mjpython` 启动。控制目标采用 `z r theta`，长度单位为米、角度默认为度；具体控制限制以 `--help` 为准。

在 Linux / HPC 上先按 `RL_env/README.md` 执行 `bash RL_env/build.sh --profile local`，然后仍使用上述 `local` 启动命令。这些数值和 MuJoCo CPU 控制程序使用归档的 NumPy / SciPy / MuJoCo 3.7 依赖组合，不需要 MJWarp 的 `core` 依赖组合。通过 `IRP_PYTHON_ENV` 可以显式指定共享环境位置，统一启动器会自动设置该变量。

## 数值工作空间

```bash
bash RL_env/run.sh local RL16-API/translation/T6_workspace.py \
  --z-min 0.4 --z-max 0.4 --theta-step-deg 90 \
  --r-step 0.02 --workers 1
```

此命令是小规模可运行示例；较粗采样只用于快速验证。正式扫描可省略参数使用原有细网格，或根据需要设置高度、角度、径向步长和并行进程数。扫描只改变计算中的 root 高度，不重写源 XML。

可达半径定义为每条径向射线从中心开始的**连续可达前缀**；遇到首个不可达点即停止，不把更远处的不连通可达区域计入内切圆。工作空间属于解析 IK 和限位可行性计算，不等同于完整接触动力学成功率。图像、论文排版和 LaTeX 推导代码已移除；执行数值分析时新生成的 CSV/检查点由运行参数决定，仓库不附带历史输出。

## 组合旋转工作空间

```bash
bash RL_env/run.sh local RL16-API/rotation/R6_rotation_translation_workspace.py --validate-only
```

该模块使用世界坐标轴旋转向量；例如 XY 各 30° 表示 `[30, 30, 0]` 度旋转向量，而非顺序欧拉角。对每个高度和旋转幅度扫描平移可达半径，再汇总单轴、双轴、三轴条件。保留断点续算、签名与完整性校验；不再生成热图。HPC 环境安装通过 `rotation/R6_setup_cx.sh` 转发给共享环境构建器。

## 载荷计算

```bash
bash RL_env/run.sh local RL16-API/translation/T6_payload_capacity.py --help
```

该模块使用运动学雅可比、虚功和接触/执行器约束估计准静态承载能力。具体被计入的执行器、摩擦系数、接触斑、路径与力矩假设以参数和源代码说明为准；结果是所设模型条件下的数值估计。

## 模型保留与 GitHub

`models/` 当前保留 28 个 SolidWorks 零件、11 个 SolidWorks 装配文件、21 个 STL。不同仿真任务中的同名网格可能对应不同版本，按原用途分目录保留。

代码目录中的 `mesh` 是指向 `models/simulation/` 的**相对符号链接**，没有第二份实体网格；XML 仍可使用原相对网格名称。复制或检出仓库时必须保留符号链接。SolidWorks 中装配引用依赖原零件关系，编辑时以 `models/cad/` 为根目录；重新导出网格前请确认目标仿真任务。

需要把本目录与共享 `RL_env/` 放在同一仓库层级。虚拟环境二进制、训练输出和缓存不应提交 GitHub；环境重建方式见根 `RL_env/README.md`。已移除历史输出、训练权重、打包产物、绘图代码及 LaTeX 文件。

## 验证

本次整理已验证平移主 XML 可加载并执行一个 MuJoCo 仿真步，并完成单高度粗网格扫描；IK 控制器可实例化并执行仿真步。数值行为仍使用原始运动学算法和本手源模型。

```bash
bash RL_env/run.sh local -m unittest discover -s RL16-API/rotation -p 'test_*.py'
```

现有 8 个旋转工作空间测试通过。

与 YCB 有关的 GRAB 源码已单独归档到同级 `Diffusion_FutureWork/`；FLUID、CHECK、MAZE 等其他训练实现及其专属资产已按归档范围删除。本目录不再包含训练子目录。
