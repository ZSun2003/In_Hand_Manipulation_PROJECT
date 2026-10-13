# Diffusion / hpc：MuJoCo Warp 正式训练

`hpc` 是独立的 Linux/NVIDIA 版本，用于 Imperial College CX3 GPU
集群上的正式训练。它不读取上级 `Diffusion` 目录，也不复用 macOS/MPS
虚拟环境、数据集或 checkpoint。整个目录可以单独打包、移动和恢复。
源码、配置、数据与模型可随目录移动；Python/CUDA 环境统一复用外部
IRP `core` profile，默认路径为：

```text
/path/to/IRP/RL_env/profiles/core/RL_env
```

可用 `IRP_HPC_CORE_ENV` 显式指定同版本的其他绝对路径。项目脚本只读取
中央环境；`setup_hpc.sh` 现在只是版本/prefix 验证器，不创建 venv，也
不执行安装或升级。标准 venv 仍与 Linux、CPython 基础运行时和绝对路径
有关，因此中央 profile 必须由 `IRP/RL_env/build.sh` 在目标平台最终
持久路径构建一次。

当前默认目标是 CX3 的单张 L40S 48 GB。MuJoCo Warp 的吞吐来自同一张
GPU 上并行大量 world；初次提交先跑 128→256→512→1024 world 容量探测，
再跑 smoke job，确认显存、contact 容量、数值稳定性和 checkpoint 恢复。
正式首跑固定为 1024 world，不直接使用 8192 world。后者会把当前
`nconmax=300` 的总 contact 槽位推到 2,457,600，并同时放大 MJWarp 状态、
约束与 CUDA Graph；在尚未同时装入演示窗口、Diffusion 和 TQC replay
进行实测前，48 GB 显存不能被视为足够。

正式 backend 不是把 CPU MuJoCo 环境套一层 GPU tensor：模型和 1024 个
world 均由 MJWarp 持有，Torch 与 Warp 通过零拷贝 CUDA view 交换状态。
一个 20 ms 控制周期由 40 个 0.5 ms tick 组成；每个 tick 使用公开
`step1/step2` 时序执行“`step1` 刷新当前状态 → native Warp wrench →
`step2` 求解并积分”，40 个 tick 后再做一次不积分的 `forward`，全部展开
到同一个 CUDA Graph。因此抓持阻尼、螺纹约束、点速度和 ACC-stage 触觉
观测均在正确的 MJWarp stage 计算。

策略分工固定为：

- Diffusion 和 TQC 读取 191 维全知状态，其中包含孔口 XY、朝向中心的
  开口方位、孔角度、接触力和螺纹物理状态；
- Diffusion 输出 14 维任务空间动作：位置 xyz、姿态 xyz、四指材料角、
  四指距离；
- 解析 IK 把任务空间目标批量映射到 16 个主动关节，并做闭链可行性
  认证；
- TQC 只学习 6 维 residual：位置 xyz 与姿态 xyz。螺纹接合后，轴向
  位移由 5 mm 螺距和绕孔轴旋转严格耦合，不能由 residual 独立破坏；
- supervisor 管理换抓、孔外 5 mm 实测同轴门、插入反馈门、两圈/10 mm
  终点、接触释放与失败停止。

任务几何为：机械手 root `z=0.620 m`；80 mm 长、直径 30 mm 的螺杆
带 40 mm 高、直径 50 mm 的螺帽，螺丝总高 120 mm，初始螺杆中心
`z=0.300 m`。固定 hard-tip 抓取平面距螺丝前端 100 mm（螺丝局部
`z=60 mm`，位于螺帽高度中心），所以螺丝立在桌面时该平面的世界高度
精确为 `z=0.360 m`。四个指尖中心距螺丝轴 39 mm，形成约 2 mm 的
柔顺预压。孔口高度固定为 `z=0.330 m`
（桌面上方 70 mm），XY
按方位角均匀随机生成在以世界中心为圆心、半径 20 mm 的圆周上；孔的
水平开口方位始终朝向中心，与桌面的夹角固定为 45°，不再随机采样倾角。
孔直径 40 mm、深度 80 mm；
插座外块为 80×80×100 mm，其中最后 20 mm 是盲孔背板。螺丝只在随孔口
方向旋转的 100 mm 半圆内刷新。

确定性 supervisor 先以固定材料接触点完成四指抓取；只有四指接触连续
5 帧，初始夹具才释放。随后将固定抓取平面抬到世界
`z=0.420 m`（螺丝前端 `z=0.310 m`）。不再经过世界中心，也不在途中
松开或换抓；Diffusion + IK + Residual 从各 world 的 lifted-spawn
交接点开始，使位置与姿态同时沿直接路径对齐孔。最终 ALIGN 命令使螺丝前端面
与孔口平面平行、两条中心轴重合，并让前端面停在孔口平面外 5 mm。
真实物理状态必须满足横向误差不超过 1 mm、轴角误差不超过 2°、轴向
5 mm 目标误差不超过 2 mm、低线／角速度，并连续 5 个控制帧合格；
同时保持四指实测接触。否则该 world 保持在
ALIGN，使用增益 2、径向命令上限 4 mm 且逐步限速的实测 XY/depth
反馈纠偏，禁止进入 INSERT。刚通过门的第一个 INSERT 控制帧禁用学习动作；
孔内候选还会在 IK 前再次投影到径向／轴向安全包络。

螺杆进入 30 mm 并连续 5 帧满足同轴、低速度、四指接触后，才允许把
U/D 两指从 40 mm 单调放松到 49 mm；L/R 两指继续保持 40 mm 接触。
名义捕获点的螺丝头下表面距孔口 50 mm，计入 ±2 mm 深度容差后的合同
下限为 48 mm，因此不会在刚过孔口平面时突然放松。到 70 mm 螺纹位置后，
还要连续 5 帧满足 ±1.5 mm 深度、L/R 接触、位置、姿态和低速度门，才锁存 thread
engagement。只有此后才允许松开、移动材料接触点或重抓。70 mm 深度开始螺纹啮合，
螺纹实测承力后直接开始重抓，并将抓取模式从 H110 切换到已通过完整换抓
IK 审计的 H112；因此初始抓取和整个预插入/插入阶段仍是用户指定的
`z=0.370 m` 平面，而两圈换抓不会越过
机械手的 50 mm 直线滑轨边界。
随后按 5 mm 螺距完成两圈并前进 10 mm，最终深度为 80 mm。
所有计划内的 Diffusion→anchor 安全门接管都使用 100 个控制帧
（2.0 s）的 minimum-jerk 任务空间混合；旋转沿 SO(3) 测地线插值。
IK 不可行或 anchor 也不可行时的紧急回退仍立即生效。
当前专家名义轨迹为 2665 个控制步，学习控制交接点为第 175 步；episode
正式 AutoDL/HX1 配置上限为 4000 步，用于容纳更长的安全门平滑过渡、
实测反馈门与接触延迟。

孔口完整方位角直接进入 socket frame 和最终姿态，不再做象限置换，也
不额外加入 early roll。物理 U/D/L/R 职责及四个材料接触角在 thread
engagement 前保持固定；30 mm 后的主持握对始终是物理 L/R。每次 reset
都用上一 IK 解连续认证完整 0–175 步抓取／抬升前缀，拒绝不可达样本，
但不永久缩小课程最终 100 mm 的刷新半径。

孔道按设计埋入桌面。为允许螺丝沿埋入式孔道运动，螺丝—桌面碰撞被
过滤；hard tip 和 long link 与插座、孔壁、孔底以及桌面的接触仍保留。
其他机械手连杆继续按 collision mask 过滤。

> 当前观测 schema 为 v7；H100 抓取/插入、螺纹承力后的 25 帧
> H100→H112 轴向换位、20 帧 Diffusion→anchor 混合、30 mm
> 捕获门、固定物理 L/R
> 主持、70 mm 螺纹门、固定 45° 倾角、孔口高度及 5 mm 实测预插入
> 门都进入任务几何 hash。此前任何 root、螺丝长度、抓取平面、孔口
> 高度／半径、孔倾角、预插入门或分阶段交接流程产生的 demo、
> Diffusion/TQC checkpoint、归一化统计、replay buffer 和 run 目录均
> 不能续用；正式训练必须使用新的空 run 目录重新生成示范。
>
> IK 与碰撞资格必须在当前 `root z=0.620 m`、抓取/插入平面偏移
> 100 mm、螺纹承力平面偏移 112 mm、
> 搬运交接高度 420 mm、孔口 `z=0.330 m`／半径 20 mm 和固定 45°
> 倾角重新审计；旧扫描结果无效。

### L40S 正式首跑参数

`configs/cx3_l40s.json` 使用以下保守基线。它的目标是先得到一条完整、
可诊断、可恢复的正式训练曲线；不要在首个无人值守任务中同时放大 world
数量和网络。

| 模块 | 正式值 | 说明 |
|---|---:|---|
| MJWarp worlds | 1024 | 先由 128→1024 容量门验证；首跑不使用 8192 |
| demonstrations | 4096 episodes | 每条 3000 步，策略从第 175 步开始 |
| Diffusion | hidden 256，depth 6 | 14 维动作、191 维观测的首跑容量 |
| Diffusion batch / epochs | 2048 / 30 | 约 1135 万窗口、每 epoch 约 5540 update |
| Diffusion optimizer | AdamW，LR `3e-4` | 训练中记录 loss、梯度、LR 与 best loss |
| Diffusion schedule | 100 train steps / 16 inference steps | 每 4 个控制步重新规划，action horizon 16 |
| Diffusion EMA | `0.999` | 比 `0.995` 更适合约 16.6 万次 update 的长训练 |
| TQC actor/critic | hidden 512 | residual 6 维：位置 xyz + 姿态 xyz |
| TQC critics / quantiles / drop | 2 / 25 / 2 | 先建立稳定基线，再按 Q/TD 遥测决定是否扩容 |
| TQC batch / replay | 1024 / 1,000,000 | replay ratio 1.0，避免 2048 batch 导致 update 太少 |
| TQC transitions / warm-up | 5,000,000 / 65,536 | 每 250,000 transitions 原子 checkpoint |
| TQC optimizer | actor/critic/alpha LR `3e-4` | `gamma=0.99`，`tau=0.005` |

Diffusion 当前没有独立 validation split，因此不能只根据训练 loss 自动
early-stop。首轮完整跑完后，应同时使用 deterministic evaluation 成功率、
最近 128/512/2048 个 episode 的 Wilson 置信区间、失败阶段、Q/target-Q、
TD-error、策略熵、显存和 overflow 记录再改参数。

首轮调参使用以下门槛：

- 任一 non-finite、contact/constraint overflow、sticky backend error 或
  显存达到 97% 时安全停止；显存达到 90% 或容量达到 80% 时禁止增加
  world；
- Diffusion 到 epoch 3 仍未从约 1.0 明显下降，先检查数据与 normalization；
  梯度频繁触及 1.0 clip 时，下一轮把 LR 降至 `1e-4`；
- TQC 至少达到 1,000,000 transitions 且累计 512 个终止 episode 后再判断。
  full-difficulty 组达到 128 个 episode 后仍为 0 成功、失败／截断超过
  50% 且深度和圈数没有改善，才停止排查流程；
- 5M 结束时，若 recent-128 成功率比 recent-512 高至少 5 个百分点且
  TD-error 仍下降，下一轮优先用 10M transitions 或 `replay_ratio=2`；
  若两窗口差小于 2 个百分点并连续稳定，且 1000-episode deterministic
  evaluation 达到 90%（理想 95%），可视为首个基线收敛；
- 只有 1024-world 全阶段峰值显存低于 70%、GPU 利用率低于 70%、
  contact/constraint 使用低于 50%，且完整 2048-world 专家轨迹测试带来
  至少 30% 吞吐提升，才考虑 2048；不要从 1024 直接跳到 8192。

## 目录与命令约定

运维脚本调用以下稳定接口：

```text
preflight.py --mode static|gpu --config CONFIG [--worlds N]
capacity_probe.py --config CONFIG --run-dir DIR
                  [--worlds 128 256 512 1024]
training_stack_preflight.py --config CONFIG --worlds N
                            [--max-memory-fraction 0.90]
adaptive_capacity.py probe --base-config CONFIG --run-dir DIR
                           --worlds 512 1024 2048 4096 8192
adaptive_capacity.py gate --run-dir DIR
pipeline.py  --config CONFIG --stage smoke|full|resume-probe|eval
             [--resume auto] --run-dir DIR
             [--checkpoint FILE --episodes N]
status_hpc.py [--run-dir DIR] [--watch [SECONDS]] [--json]
              [--alerts-only] [--history N] [--request-stop]
collect_diagnostics.py --run-dir DIR [--job-id ID]
                       [--allocation-log FILE] [--exit-code N]
```

所有 Python CLI 默认把相对路径锚定到 `hpc`，并拒绝解析后落到目录外的
绝对路径、`..` 或符号链接逃逸。确实需要使用外部 project/RDS 存储时，必须
在对应命令中显式加入 `--allow-external-paths`；这样生成的 run 就不再满足
“移动 `hpc` 后零外部路径依赖”的默认承诺。

所有生成内容都留在 `hpc` 内：

```text
.cache/       Warp、XDG 等缓存
datasets/     GPU 演示数据
checkpoints/  独立导出的 checkpoint
runs/         各次训练、评估、日志和原子 checkpoint
outputs/      可视化或汇总产物
```

正式任务固定写入 `runs/cx3_l40s_full`。重复提交会通过 `--resume auto` 从该
目录最新的完整 checkpoint 继续，避免每个 PBS job 形成互不相干的训练。

## 1. 把目录传到 Imperial

推荐使用带递归环境审计的源码构建器。默认排除 `runs`、checkpoint、
策略权重和训练输出，保留代码、配置以及 XML/STL 模型资产。
构建输出的 `model_artifact_members=0` 表示没有训练权重：

```bash
cd /path/to/IRP/Diffusion/hpc
./build_hpc_package.sh
scp ../hpc_AUTODL_*.tar.gz* <username>@login.cx3.hpc.imperial.ac.uk:~/
```

源码包没有模型时 `N=0` 是正常结果。若本次上传必须携带至少一个原生 HPC
模型，则使用：

```bash
SCREW_PACKAGE_INCLUDE_MODELS=1 SCREW_PACKAGE_REQUIRE_MODELS=1 ./build_hpc_package.sh
```

此时 `N=0` 会让审计失败关闭。只统计本归档内
`hpc/checkpoints/` 或 `hpc/runs/**/checkpoints/` 下扩展名为
`.pt`、`.pth`、`.ckpt`、`.safetensors` 或 `.onnx` 的普通文件；
`.pkl`、数据集、manifest 和 `outputs/` 普通文件不计。父目录
`Diffusion/checkpoints/` 中的 macOS/MPS smoke checkpoint 不会进入归档，也
不能满足该检查。

在 Imperial 的 Linux 登录节点解压后，先进入目录。后续所有 `qsub`
都必须从这里执行，job 脚本会拒绝从父目录提交：

```bash
tar -xzf hpc_source.tar.gz
cd hpc
```

## 2. 验证中央 IRP core 环境

先由 IRP 环境包在最终持久路径构建一次 `core/RL_env`。Diffusion 不再维护
自己的虚拟环境：

```bash
export IRP_HPC_CORE_ENV=/path/to/IRP/RL_env/profiles/core/RL_env
./setup_hpc.sh
```

该脚本只读验证以下固定合同：

- `mujoco-warp==3.10.0.3`
- `mujoco==3.10.0`
- `warp-lang==1.14.0`
- `torch==2.7.1` 官方 CUDA 12.8 wheel

`setup_hpc.sh` 会验证安装的是 CUDA build；登录节点没有可见 GPU 时，
`cuda_visible_now=False` 是正常的，真正的可见性由 PBS smoke job 验证。
它还运行只读的 `pip check`，并把冻结清单写入项目自己的
`environment-manifest.txt`；不会向中央环境写入任何包或 `.pth`。
所有 PBS 任务会在启动时重新执行 `module purge`、加载 `tools/prod` 和
`Python/3.11.5-GCCcore-13.2.0`，确保该 venv 所需的 `libffi` 等
EasyBuild 运行库在计算节点可见。若中央环境改用其他 Python 模块，
提交时需用 `SCREW_PYTHON_MODULE` 显式指定相同模块。

## 3. CX3：容量探测与 smoke test

CX3 使用 PBS Pro。脚本不写 `#PBS -q`，由集群按 L40S 资源请求自动路由，
避免再次触发 `Unknown queue`。默认资源请求为：

```text
select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=L40S
```

推荐直接从 `hpc` 目录提交无人值守依赖链：

```bash
./submit_cx3_chain.sh
```

脚本依次提交 `capacity → smoke → full`，后一个任务只会在前一个返回成功
后启动，并把三个 PBS job ID 保存到 `runs/submissions/latest.env`。因此
容量、CUDA/MJWarp、checkpoint 恢复或最小全流程任一 gate 失败时，正式
训练不会误启动。若需要逐项调试，也可以手动先提交容量探测：

```bash
mkdir -p logs
qsub jobs/cx3_capacity.pbs
qstat -u "$USER"
```

它在四个彼此隔离的 Python 进程中依次运行 128、256、512、1024 world，
每一级都执行真实 MJWarp 创建、CUDA Graph、96 个控制步、NaN/overflow
和显存峰值检查。结果原子写入
`runs/cx3_capacity_<job-id>/capacity.json`；1024 未通过时 job 以非零
状态退出，不能继续正式配置。

1024 world 通过后提交完整 smoke：

```bash
qsub jobs/cx3_smoke.pbs
qstat -u "$USER"
```

smoke job 会依次检查：

1. 当前解释器确实是 `IRP_HPC_CORE_ENV/bin/python`；
2. CUDA PyTorch、L40S 和 `nvidia-smi` 可见；
3. 使用正式 `cx3_l40s.json` 创建 **1024 worlds × 每控制周期 40 physics
   substeps** 的 production-shaped MJWarp CUDA Graph，完成编译、capture
   和并行 step，验证每周期恰好推进 0.020 s，记录 Torch 显存峰值，并
   检查 NaN、contact/constraint overflow；
4. 使用小型 smoke 配置完成 demo → Diffusion → TQC → evaluation；
5. 从完整 TQC checkpoint 恢复 world、replay、controller 和 RNG，实际
   执行一个不写回 checkpoint 的 vector step；
6. 在完全相同的 run 目录再次执行 `--resume auto`，确认 demo metadata、
   Diffusion checkpoint、TQC checkpoint、配置和 SHA 配对都能重新加载。

第三项是正式发布门，不是缩小 world 数的近似测试：必须由 1024-world
graph 在目标 L40S 上通过。smoke 脚本因为要执行 production graph
preflight、续训一步、完整最小流程和第二次 checkpoint reload，预留 2
小时 walltime。
只有整个 job 成功，并且日志没有 NaN、contact overflow、Warp kernel
错误或显存不足，才进入正式训练。

### 独立的 8192-world 极限链

`submit_cx3_8192_chain.sh` 不覆盖 1024-world 基线。它使用独立配置、run
目录、锁和 submission record，依次执行：

```text
capacity 1024 → 2048 → 4096 → 8192
→ 8192-world production preflight + 8-world smoke/resume
→ 8192-world formal Diffusion/TQC
```

提交：

```bash
./submit_cx3_8192_chain.sh
source runs/submissions/latest_8192.env
qstat -f "$CAP_JOB"
```

容量结果写入
`runs/cx3_capacity_8192_<job-id>/capacity.json`。除最大通过值外，还记录
整卡显存峰值，并分别给出不超过 70%、80%、90% 显存的最大**已实测通过**
worlds；下次可以直接选择相应档位。任何档位失败后都会停止继续增大，因为
显存需求应随 worlds 单调增加；capacity job 返回非零，后续 8192 smoke
和 formal 保持不启动。

8192 是有意的极限/OOM 探测值，不是 L40S 48GB 的预期默认值。即使
simulation-only preflight 通过，正式训练还需要 Diffusion、TQC、replay
和数据集显存；70% 档位最适合作为正式训练候选，90% 只应视为激进的物理
仿真上限。8192 配置保留与 1024 基线相同的 5M transitions，因此它是
并行度/容量实验，不保证与 1024 基线具有相同的每-world 轨迹长度或学习
曲线。

### 任意 GPU 的自适应极限链

若固定申请 L40S 排队过久，使用：

```bash
./submit_cx3_adaptive_any.sh
source runs/submissions/latest_adaptive_any.env
qstat -f "$ADAPTIVE_JOB"
```

该 PBS 作业只请求 `ngpus=1`，不指定 `gpu_type`，因此 CX3 可以分配
RTX6000、A100 或 L40S。Capacity、Smoke 容量门和正式训练在同一个
71:50 allocation 内顺序运行，三阶段一定使用同一张实际 GPU，不会把
48 GB 卡测出的 worlds 错用到 24 GB 卡。

Capacity 在互相隔离的子进程中依次测试
`512 → 1024 → 2048 → 4096 → 8192`。每档同时覆盖：

- MJWarp 环境和正式 float16 demo 轨迹缓冲；
- 完整 demo CUDA cache、Diffusion backward、AdamW 和 EMA；
- 全 worlds Diffusion 推理、1,000,000 replay、TQC actor/critic、
  两次 backward/Adam 更新和环境 checkpoint snapshot。

结果写入
`$FULL_RUN_DIR/capacity_selection/capacity.json`。它同时保存能完整
分配的 `raw_maximum_completed_worlds`，以及整卡保守峰值不超过 90% 的
`selected_training_worlds`；正式训练使用后者。不可变派生配置和哈希位于
同目录的 `selected_config.json` 与 `selection.json`。因此下一轮可以按
GPU 型号、显存、软件版本和每档耗时直接选 worlds，不必重复盲扫。

正式训练状态：

```bash
./hpc_python.sh status_hpc.py \
  --run-dir "$FULL_RUN_DIR" --watch 10
```

若 walltime guard 保存 checkpoint，可复用同一配置：

```bash
./submit_cx3_adaptive_any.sh --resume "$ADAPTIVE_ID"
```

续跑仍可随机获得任意 GPU，但作业会先用完整训练栈复验已选 worlds；
显存较小的新卡不能容纳时会在修改 checkpoint 前失败关闭，重新提交即可。
若希望每次都能在任意卡上续跑，应从 capacity 结果中人工选择所有目标
GPU 都能通过的较小 worlds，而不是使用某张大显存卡的极限值。

## 4. CX3：正式训练与恢复

提交单 L40S 正式任务：

```bash
qsub jobs/cx3_full.pbs
qstat -u "$USER"
```

PBS walltime 是 71 小时 50 分。脚本会在 71 小时 30 分向 Python 发送
`SIGINT`，给 pipeline 留出写原子 checkpoint 的时间。正常完成或触发
walltime guard 后，再次执行同一条 `qsub` 即可继续：

```bash
qsub jobs/cx3_full.pbs
```

训练与部署文件位于：

```text
runs/cx3_l40s_full/checkpoints/diffusion.pt
runs/cx3_l40s_full/checkpoints/residual_tqc.pt
runs/cx3_l40s_full/checkpoints/residual_tqc_policy.pt
```

`--resume auto` 会分别从这两个原子 checkpoint 恢复 diffusion 和
residual TQC 阶段。`residual_tqc.pt` 是包含 replay、optimizer 和 world
状态的续训文件；`residual_tqc_policy.pt` 是轻量 actor 推理文件，评估
不必读取约 GB 级 replay。每个文件都先写临时文件，再原子替换正式文件；
若 `SIGINT` 到达写入途中，恢复逻辑会忽略不完整临时文件并回退到最近
一个完整 checkpoint。不要同时提交两个 `cx3_full.pbs`，否则它们会争用
同一 run 目录；脚本还会用 `runs/.cx3_l40s_full.lock` 的 Linux `flock`
覆盖整个 preflight/pipeline，误提交的第二个任务会立即失败且不会写数据。
锁由打开的 job 文件描述符持有，job 退出后自动释放，残留的空 lockfile
无需删除。若训练异常退出，保留 PBS 输出、`runs/cx3_l40s_full` 和恢复
文件。TQC 的恢复边界如下：

TQC 续训会恢复 replay buffer、actor/critic/optimizer、各 world 的已保存
动力学与任务状态，以及 Diffusion controller 的 plan、cursor 和 RNG
进度。MJWarp 的 contacts、传感器结果、运动学缓存等 derived fields 不直接
序列化；载入基础 world 状态后会调用 `forward` 重建。因此恢复保证训练语义
和进度连续，但不承诺与未中断 CUDA 运行逐 bit 相同。并行 contact 排序、
GPU reduction/atomic 运算和重新生成的 derived contacts 都可能产生极小的
浮点差异；验收应比较成功率、深度、圈数和 reward，而不是要求 checkpoint
前后的每个浮点数 bitwise identical。

演示数据完成时会把 `observations.npy`、`actions.npy`、`spawn_xy.npy`、
`socket_mouth_xy.npy` 和 `socket_angle_deg.npy` 各自的 SHA-256 写入
metadata；metadata 还绑定任务几何 hash。Diffusion 开始前重新计算并
核对，checkpoint 又绑定 exact metadata SHA。正式 4096-episode float16
数据约 4.2 GiB，
会一次缓存到 L40S 后做随机 window gather。物理状态、策略动作、观测、
奖励、Diffusion/TQC loss、gradient 和模型参数均设有 NaN/Inf gate；
异常只保留上一份 last-good checkpoint，不覆盖它。

reset 是按 world mask 独立执行的，不会重置同 batch 中未完成的 world。
当前 mask 的 world-id 提取仍会发生一次 host 同步，所以本文称
“masked/per-world reset”，不宣称完全 device-asynchronous；是否值得
进一步 kernel 化应以 CX3 profile 为准。

重新提交前先跑静态检查：

```bash
./hpc_python.sh preflight.py \
  --mode static \
  --config configs/cx3_l40s.json
```

然后确认对应阶段的 `diffusion.pt` 或 `residual_tqc.pt` 可加载，再重新
提交。

可以在提交时缩短主动退出时间，以验证恢复链路：

```bash
qsub -v SCREW_MAX_RUNTIME=20m jobs/cx3_full.pbs
```

## 5. 实时监控、诊断与安全停止

正式 job 启动后，在另一个登录节点终端进入同一个 `hpc` 目录。查看
一次完整状态：

```bash
./hpc_python.sh status_hpc.py --run-dir runs/cx3_l40s_full
```

每 10 秒刷新：

```bash
./hpc_python.sh status_hpc.py \
  --run-dir runs/cx3_l40s_full \
  --watch 10
```

该查询器只读取几份很小的原子 JSON，不会 import Torch/MJWarp，也不会
加载 GB 级 replay checkpoint 或重新计算数据集 SHA，因此可以安全地从
登录节点频繁运行。正式 job 会打印相同的监控命令。运行目录移动后仍使用
相对于新 `hpc` 的路径即可。

状态页包括：

- pipeline 当前阶段和阶段序号；
- demo 已提交 episode、当前 vector step、速度和 ETA；
- Diffusion epoch/batch/global step、当前/epoch/best loss、学习率和梯度；
- TQC transitions、updates、warmup/learning 状态、replay 容量、loss、
  alpha、梯度、速度和 ETA；
- 累计 episode 与最近 2048 个已完成 episode 的成功、失败、截断、
  return、深度、圈数和失败原因；成功率同时显示最近 128、512、2048 个
  完成 episode 的 Wilson 置信区间；
- 当前 1024 个 world 的 reward/action、深度、圈数、横向/轴向误差、
  速度、螺纹接合、holder、grasp、IK 可行率、活动 residual 维数、
  脚本搬运／Diffusion 活动比例、交接门通过率、交接位置／姿态误差与
  等待步数、当前孔倾角 min/mean/max（应均为 45°）、孔口半径及 XY
  范围、固定 45° 配置和 15 相任务分布；
- MJWarp NaN/Inf、overflow、contact/constraint 容量；
- `nvidia-smi` 的利用率、显存、温度、功率、clock，以及训练进程 RSS、
  CPU、线程、I/O、节点 load 和 run 所在磁盘剩余空间；
- checkpoint 是否存在、大小、修改时间，PBS job 状态和 raw pipeline
  exit code；
- heartbeat 过期、孔口偏离 20 mm 圆周、容量逼近上限、GPU 显存/温度、
  磁盘不足、近期失败/截断率等自动告警和建议动作。

GPU 告警会按 `CUDA_VISIBLE_DEVICES`（或容器的
`NVIDIA_VISIBLE_DEVICES`）只检查分配给本 job 的设备。若多 GPU 节点
没有提供可归属信息，状态页会显示 `gpu_allocation_unknown` 并抑制针对
单张物理 GPU 的显存/温度告警，避免把其他用户作业的问题误判成本任务。

这里显示的训练成功率只统计真正出现 `done` 的 episode；尚未有 episode
结束时显示 `N/A`，不会错误显示成 0%。`recent` 窗口按“最近完成的
episode”而不是最近 step 计算，并随完整 TQC checkpoint 一起恢复。
训练时 actor 是 stochastic 的，所以该成功率用于尽早诊断训练趋势，不等同
于正式的 deterministic evaluation 成功率。

其他查询方式：

```bash
# 只看 warning / critical
./hpc_python.sh status_hpc.py \
  --run-dir runs/cx3_l40s_full \
  --alerts-only

# 机器可读 JSON，适合自己的监控脚本
./hpc_python.sh status_hpc.py \
  --run-dir runs/cx3_l40s_full \
  --json

# 同时显示最近 20 条 app 心跳
./hpc_python.sh status_hpc.py \
  --run-dir runs/cx3_l40s_full \
  --history 20
```

单次查询的退出码为 `0=健康/正常完成/仍在 preflight`、`1=warning`、
`2=critical/失败/运行中 app heartbeat 过期`、`3=找不到状态`。TQC loss
本来就不要求单调；“近期 0 成功”也只有累积了至少 512 个完成 episode
后才告警，避免刚开始尚未到达完整任务终点时误报。

状态分别写入：

```text
runs/cx3_l40s_full/status/live.json          pipeline 单写者
runs/cx3_l40s_full/status/system.json        job 内 system sidecar 单写者
runs/cx3_l40s_full/status/job.json           PBS wrapper 单写者
runs/cx3_l40s_full/status/history.jsonl      app 心跳历史
runs/cx3_l40s_full/status/system-history.jsonl  系统/GPU 采样历史
runs/cx3_l40s_full/status/job-history.jsonl     PBS allocation 历史
runs/cx3_l40s_full/logs/episodes.jsonl          每个终止 episode 诊断
runs/cx3_l40s_full/run_manifest.json            不可变配置、seed、依赖与源码 hash
runs/cx3_l40s_full/diagnostics/latest.json       最新诊断包文件名、大小和 SHA-256
runs/cx3_l40s_full/diagnostics/*.tar.gz          可直接下载的小型完整诊断包
```

训练进程卡住时 `live.json` 会变旧，而独立 `system.json` 通常仍继续更新；
状态页会把它判为 critical。若两者一起停止更新，则优先检查 PBS job、
计算节点和共享文件系统。

每个 PBS allocation 正常完成、失败或被 walltime guard 安全中断时，退出
trap 会自动生成一个诊断压缩包。压缩包包含精确的文本源码/配置快照及
SHA-256、Python 和已安装包版本、`_ctypes`/`libffi` 检查、PBS `qstat`
快照、`nvidia-smi -q`、GPU/系统历史、run 状态、episode/训练日志尾部和
allocation log 尾部。它只记录环境变量白名单，不收集 token；只记录
checkpoint/dataset 的文件元数据，不把大型模型、replay 或演示数据装入
压缩包。

需要在 job 仍运行时立即制作一次诊断包，也可以手动执行：

```bash
./hpc_python.sh collect_diagnostics.py \
  --run-dir runs/cx3_l40s_full \
  --job-id "$PBS_JOBID"
```

退出后快速查看和下载目标：

```bash
cat runs/cx3_l40s_full/diagnostics/latest.json
```

发现问题后，推荐先请求安全停止：

```bash
./hpc_python.sh status_hpc.py \
  --run-dir runs/cx3_l40s_full \
  --request-stop \
  --reason "recent failure rate is abnormal"
```

请求绑定当前 pipeline session；训练只在完整 vector step 或 mini-batch
游标读取它。Diffusion/TQC 随后写原子可续训 checkpoint，并以
`interrupted` 退出；demo 未提交的当前 vector batch 会在续训时确定性重跑。
旧 session 的停止文件不会让下一次 `qsub` 立即退出。若仍处于 GPU
preflight、CUDA 编译彻底卡死或共享盘不可写，安全请求可能无法被 Python
接收，此时才使用状态页显示的紧急命令：

```bash
qdel <PBS_JOBID>
```

`qdel` 可能丢失最近一次完整 checkpoint 之后的进度，但不会破坏之前已经
原子提交的 checkpoint。

## 6. 评估并使用正式模型

默认评估 100 个 episode：

```bash
qsub jobs/cx3_eval.pbs
```

修改 episode 数量：

```bash
qsub -v SCREW_EVAL_EPISODES=500 jobs/cx3_eval.pbs
```

评估脚本向 pipeline 传入
`runs/cx3_l40s_full/checkpoints/residual_tqc_policy.pt`；pipeline 会从同一目录推断
并加载 `diffusion.pt`。因此评估前两个 checkpoint 都必须存在且属于同一
次兼容训练。结果写入独立的 `runs/cx3_eval_<job-id>`。部署/回放工具应
同样使用 GPU 版本的 checkpoint schema，不能静默载入根目录 MPS 版本的
SB3 或 diffusion checkpoint。

## 7. 拷回 Mac 与备份

训练过程中定期把 `runs/cx3_l40s_full` 复制到有备份的 Imperial project/RDS
存储。CX scratch/home 不应被当成唯一副本。具体 project 路径因账户而
异，这里不硬编码。

从 Mac 拷回时，目标仍放在本地 `hpc` 内：

```bash
cd /path/to/local/hpc
rsync -av --partial \
  <username>@login.cx3.hpc.imperial.ac.uk:/path/to/remote/hpc/runs/cx3_l40s_full/ \
  runs/cx3_l40s_full/
```

若要搬迁整个项目，只复制源码、配置、mesh、数据和 runs；不要复制
Linux 中央 `RL_env` 到 Mac，也不要把 Mac 的环境传回 CX3。换平台或
Python module 后，在目标平台最终持久路径重新构建一次
`IRP/RL_env/profiles/core/RL_env`，再运行 `setup_hpc.sh` 验证。数据集、
checkpoint 和源码可以搬动，venv 本身不属于项目归档。

## 8. HX2 模板的限制

`jobs/hx2_TEMPLATE.slurm` 只是 Slurm 模板，不是可提交脚本。它有意省略
partition 和 GPU resource 指令，并带有强制停止 gate。HX2 pilot 的
实际分区、GPU 类型和资源语法必须以当时的 Imperial RCS 文档以及
`sinfo` / `scontrol show partition` 输出为准。

确认后需要同时完成三件事：

1. 添加经过确认的 `#SBATCH --partition` 和 GPU 指令；
2. 创建并选择对应配置；
3. 将模板内 `HX2_DIRECTIVES_CONFIRMED` 改为 `YES`。

在这三项完成前不要执行 `sbatch jobs/hx2_TEMPLATE.slurm`。

## 官方参考

- MuJoCo Warp: <https://github.com/google-deepmind/mujoco_warp>
- MuJoCo Warp API: <https://mujoco.readthedocs.io/en/latest/mjwarp/>
- Imperial cluster specification:
  <https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/cluster-specification/>
- Imperial CX3 SSH:
  <https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/getting-started/using-ssh/>
- Imperial GPU jobs and job sizing:
  <https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/queues/gpu-jobs/>
  and
  <https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/queues/job-sizing-guidance/>
