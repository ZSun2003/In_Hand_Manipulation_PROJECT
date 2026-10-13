# AutoDL RTX PRO 6000 Blackwell 96 GB 部署说明

这个目录是 Diffusion 的单卡 AutoDL 版本，不依赖 PBS、Slurm 或 HPC
environment modules。Extreme V7 使用纯 Diffusion 控制：Diffusion 在任务
空间生成动作，安全监督器完成阶段门控和约束投影，解析 IK 将目标转换为
16 个执行器命令。

正式启动顺序如下：

```text
RTX PRO 6000 硬件门禁
→ 静态代码、资产和共享环境检查
→ 8 worlds 完整专家轨迹、端到端 smoke 与恢复验证
→ 12000 → 14000 → 16000 → 18000 worlds 独立进程容量扫描
→ 对所选 worlds 执行 3000 控制步正式显存与任务语义门
→ 生成 8192 条 Demo
→ 训练 Diffusion
→ 默认执行 2000 episodes 确定性评估
```

## 硬件与磁盘

目标实例是完整的 RTX PRO 6000 Blackwell 96 GB Server/Workstation
版本，不支持 Max-Q。当前验证的软件与硬件合同为：

```text
Python          3.10–3.12
torch           2.7.1 + cu128
mujoco          3.10.0
mujoco-warp     3.10.0.3
warp-lang       1.14.0
EPA horizon     128 edges（3.10.0.3 上游默认 24）
GPU             RTX PRO 6000 Blackwell，整卡，至少 90 GiB 可见
compute         sm_120 / 12.0
驱动            至少 575.51.03；580/595 分支均可
```

进入实例后先检查数据盘：

```bash
df -h /root/autodl-tmp
```

安装检查默认要求至少 110 GiB 空闲，训练启动器要求至少 80 GiB 空闲。环境、
Warp 缓存、8192 条 Demo、Diffusion checkpoint、评估和日志都写入
`/root/autodl-tmp`，不会挤占 AutoDL 的系统盘。不要在容器内自行更换
NVIDIA 驱动或安装另一套系统 CUDA Toolkit。

V7 保留已经验证的 Blackwell 兼容措施：

- 在创建第一个 MJWarp 环境前，将锁定版本的 EPA horizon 工作区从 24
  扩展到 128 条边；
- 对 PyTorch 2.7.1 的 deterministic CUDA 高级索引问题，仅在唯一、
  互不重叠的 reset 写入范围内使用普通索引 kernel；
- 在创建 CUDA 上下文前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
- 从 XML `qpos0` 读取 0.300 m 螺丝与起始固定器高度，并与任务合同核对。

这些设置不修改共享虚拟环境，也不屏蔽 non-finite、接触容量或 Warp
overflow；任何门禁失败都会停止正式流程。

## V7 的 Spawn 修复

V6 的 curriculum 从 25 mm spawn 半径开始。在固定 45° 插座几何下，插座
旋转包围盒加螺丝头和安全间隙占满了该区域：半径不超过约
36.284271 mm 时不存在任何合法样本，因此 rejection sampler 必然耗尽。

V7 将 curriculum 起始半径改为 60 mm，最终难度仍是原来的 100 mm。采样
分布仍是在开口朝向一侧的半圆内按面积均匀采样，再以真实旋转包围盒做
碰撞拒绝；不会静默夹紧半径或改变最终任务。配置加载时会先做解析可行性
检查，测试也会覆盖 18000 个随机孔位的无碰撞采样。

这一改动会改变配置与源码指纹。V6 的 capacity 结果、Demo、checkpoint、
run manifest 和 run 目录都不能用于 V7 续训。V7 必须解压到干净目录并
重新运行 smoke、四档容量扫描和正式流程。

## 抓取与插入安全合同

- 四指在距螺丝轴 40 mm 的固定材料接触点夹持；连续 5 帧确认四指接触后
  才释放初始夹具。
- 抬起后从各自 spawn 直接对齐孔轴，不经过世界中心，也不在途中换手。
- 入孔前要求螺杆前端面与孔口平行、轴线重合，并在孔外保持 5 mm 间隙；
  位置、姿态、速度与四指接触门连续 5 帧通过后才允许插入。
- 实测进入 30 mm 且安全门连续通过后，最低接触要求才放宽为任意两指。
- 到 70 mm 螺纹位置后，还需连续 5 帧满足深度、位置、姿态、速度与接触门，
  才锁存 thread engagement；此前禁止松开、滑动材料接触点或重抓。
- 接触建立阈值为 0.10 N，保持阈值为 0.04 N。短暂力抖动会冻结轨迹并
  保持预压，持续失触才终止对应 world。

正式容量门除显存外，还检查 finite、overflow、接触/约束容量、任务失败、
截断、最终成功、螺纹锁存、深度和旋转。失败记录包含首个失败步、world、
phase、接触力、IK safety-stop 和各类安全合同计数。

## 上传

本次共享环境源码包的文件名是：

```text
hpc_AUTODL_H110_MOUTH330_W18000_EP4000_BLEND100_HX1_COMPAT_20260809.tar.gz
hpc_AUTODL_H110_MOUTH330_W18000_EP4000_BLEND100_HX1_COMPAT_20260809.tar.gz.sha256
```

以后重新构建时，`20260809` 会变成实际 UTC 构建日。可以使用 AutoDL
网页文件管理，也可以把控制台显示的 SSH 主机和端口代入：

```bash
scp -P <SSH端口> \
  hpc_AUTODL_H110_MOUTH330_W18000_EP4000_BLEND100_HX1_COMPAT_20260809.tar.gz \
  hpc_AUTODL_H110_MOUTH330_W18000_EP4000_BLEND100_HX1_COMPAT_20260809.tar.gz.sha256 \
  root@<SSH主机>:/root/autodl-tmp/
```

不要使用 `hpc_AUTODL_*.tar.gz*` 这类宽泛通配符；同一目录若保留 V6
归档，很容易把旧包一起上传。

## 安装并复用共享环境

如果 `/root/autodl-tmp/hpc` 仍是失败的 V6 目录，先停止其中的进程，再
把目录改名保留。不要把 V7 直接覆盖进去：

```bash
cd /root/autodl-tmp
test ! -e hpc || mv hpc hpc_V6_FAILED_20260730
```

校验并解压：

```bash
cd /root/autodl-tmp

sha256sum -c \
  hpc_AUTODL_H110_MOUTH330_W18000_EP4000_BLEND100_HX1_COMPAT_20260809.tar.gz.sha256

test ! -e /root/autodl-tmp/hpc

tar -xzf \
  hpc_AUTODL_H110_MOUTH330_W18000_EP4000_BLEND100_HX1_COMPAT_20260809.tar.gz
```

本项目复用中央 IRP `core/RL_env`，不会创建项目私有环境，也不会执行
`pip install`。若共享环境已经存在，直接验证：

```bash
export IRP_HPC_CORE_ENV=/path/to/IRP/RL_env/profiles/core/RL_env

test -x "$IRP_HPC_CORE_ENV/bin/python"

cd /root/autodl-tmp/IRP
RL_env/profiles/core/RL_env/bin/python RL_env/hpc/verify.py core --require-gpu

cd /root/autodl-tmp/hpc
./setup_autodl_rtx_pro_6000.sh
```

只有在共享环境尚未构建时才执行一次：

```bash
cd /root/autodl-tmp/IRP
bash RL_env/build.sh --profile core
RL_env/profiles/core/RL_env/bin/python RL_env/hpc/verify.py core --require-gpu
```

每次重新登录后，如中央环境不在默认位置，需先重新设置
`IRP_HPC_CORE_ENV`。安装脚本只读验证精确依赖版本，并运行 BF16、Warp 和
小型 MJWarp GPU 探针。

## 开始训练

后台启动，SSH 断开后仍会继续：

```bash
cd /root/autodl-tmp/hpc
export IRP_HPC_CORE_ENV=/path/to/IRP/RL_env/profiles/core/RL_env
./autodl_rtx_pro_6000.sh start
```

第一次启动先执行 8-world smoke，再按顺序扫描
12000、14000、16000、18000。每档都在独立进程中运行，因此某一档硬失败
或超时不会阻止其余候选留下各自的诊断结果。默认正式请求仍是 12000
worlds；只有该档在当前实例上通过完整 3000
控制步正式门，才会进入 Demo 和 Diffusion 训练。

也可以明确请求某个已经安全通过的候选，或选择最大安全候选：

```bash
SCREW_AUTODL_WORLDS=12000 ./autodl_rtx_pro_6000.sh start
SCREW_AUTODL_WORLDS=14000 ./autodl_rtx_pro_6000.sh start
SCREW_AUTODL_WORLDS=16000 ./autodl_rtx_pro_6000.sh start
SCREW_AUTODL_WORLDS=18000 ./autodl_rtx_pro_6000.sh start
SCREW_AUTODL_WORLDS=max   ./autodl_rtx_pro_6000.sh start
```

未通过、超过 90% 显存门限或未测试的候选会被拒绝。`max` 是显式实验
选项，不是默认值。不同选择使用独立 run 目录，不能互相续训；最后一次选择
会记录在 control 目录，之后的 `status`、`logs`、`stop` 和 `resume`
无需重复设置 `SCREW_AUTODL_WORLDS`。

查看已安全通过的档位：

```bash
./hpc_python.sh -c \
  'import json; p=json.load(open("runs/autodl_rtx_pro_6000_extreme_v7_diffusion_only_capacity/selection.json")); print(p["safe_qualified_worlds"])'
```

## Worlds、Demo 与评估的关系

`SCREW_AUTODL_WORLDS` 是 full-world 部署容量门的候选规模。它决定正式
门一次创建多少个 MJWarp world，并用于证明全批 Diffusion 推理、环境状态、
CUDA Graph 和任务语义在该 GPU 上安全；它不按比例放大训练预算。

四个候选都使用同一份正式学习合同：

```text
Demo episodes          8192
Diffusion batch        4096
Diffusion epochs       60
Diffusion train steps  100
Diffusion inference    16 steps，每 4 个控制步重新规划
Evaluation episodes    2000（默认）
```

因此 12000、14000、16000 或 18000 只改变 full-world 容量认证。Demo
阶段始终生成 8192 条轨迹，Diffusion minibatch 训练不依赖 world 数，
最终评估默认只创建 `min(worlds, 2000)=2000` 个评估 world。不要把
18000 worlds 误解为 18000 条 Demo、18000 batch 或 18000 个评估 episode。

快速容量扫描每档运行 256 个控制步，覆盖第 175 步学习交接点和直接对齐
初段；对正式所选档运行 3000 控制步，覆盖完整名义轨迹和所有实测等待门。
每档在独立 Python 进程中执行，避免一次 CUDA OOM 污染后续 allocator。

## 查看状态

一次性状态：

```bash
./autodl_rtx_pro_6000.sh status
```

每 10 秒刷新：

```bash
./autodl_rtx_pro_6000.sh watch
```

按 `Ctrl+C` 只退出状态监视，不会停止训练。跟随完整日志：

```bash
./autodl_rtx_pro_6000.sh logs
```

另一个终端可查看 GPU：

```bash
watch -n 2 nvidia-smi
```

主要目录：

```text
runs/autodl_rtx_pro_6000_extreme_v7_diffusion_only_control/
runs/autodl_rtx_pro_6000_extreme_v7_diffusion_only_capacity/
runs/autodl_rtx_pro_6000_extreme_v7_diffusion_only_w12000_smoke/
runs/autodl_rtx_pro_6000_extreme_v7_diffusion_only_w12000_full/
runs/autodl_rtx_pro_6000_extreme_v7_diffusion_only_max_*/
logs/autodl_rtx_pro_6000_extreme_v7_diffusion_only_w12000.log
```

## 安全停止与恢复

不要直接 `kill -9`，也不要在训练仍运行时关闭 AutoDL 实例。请求一致游标
停止：

```bash
cd /root/autodl-tmp/hpc
./autodl_rtx_pro_6000.sh stop
```

确认 `status` 显示 `interrupted`、`interrupted_resumable` 或 `completed`
后再关机。恢复时使用同一份源码、相同共享环境和相同 run 目录：

```bash
cd /root/autodl-tmp/hpc
export IRP_HPC_CORE_ENV=/path/to/IRP/RL_env/profiles/core/RL_env
./autodl_rtx_pro_6000.sh resume
```

`resume` 会校验不可变 manifest、配置、源码、Demo 元数据和 Diffusion
checkpoint，然后从一致游标继续；不会静默加载不同版本或覆盖已有结果。

## 数据安全

AutoDL 本地数据盘不是冗余存储。灾备前先安全停止，然后一起保存：

```text
本次实际使用的 .tar.gz 与 .tar.gz.sha256
整个 runs/autodl_rtx_pro_6000_extreme_v7_diffusion_only_capacity/
所选的整个 runs/autodl_rtx_pro_6000_extreme_v7_diffusion_only_<档位>_full/
```

只复制单个 checkpoint 不够；完整 `full` 目录还包含 Demo 元数据、日志、
评估结果和不可变 manifest。不要备份或恢复 control 目录中的旧 PID/lock。

新实例恢复时，用原始源码包解压到空目录，验证中央共享环境，再把 capacity
和所选 full 目录放回相同相对路径。因为不恢复 control 目录，第一次恢复
需显式指定原分支：

```bash
# 原来使用 max 分支
SCREW_AUTODL_WORLDS=max ./autodl_rtx_pro_6000.sh resume

# 原来使用默认或显式 12000
SCREW_AUTODL_WORLDS=12000 ./autodl_rtx_pro_6000.sh resume
```

正式运行期间不要修改源码或锁定配置。换实例后可以复用同一份完整 capacity
证明，但仍会在当前 GPU 上重新执行所选档的正式门，以验证空闲显存、驱动和
硬件指纹。

## 重新制作干净共享环境包

从源码目录运行轻量构建器；它默认排除 `runs`、checkpoint、Demo、日志、
缓存、虚拟环境、AppleDouble 和 Python bytecode：

```bash
cd /path/to/IRP/Diffusion/hpc
SCREW_PACKAGE_INCLUDE_MODELS=0 ./build_autodl_package.sh

cd ..
ls -lh \
  hpc_AUTODL_H110_MOUTH330_W18000_EP4000_BLEND100_HX1_COMPAT_*.tar.gz*
```

构建器先写临时文件，完整审计通过后才原子发布归档和 SHA-256。构建日期
默认使用 UTC 当天，也可通过 `SCREW_PACKAGE_STAMP=YYYYMMDD` 明确指定。
归档不包含中央 `IRP/RL_env`；`SHAREDENV` 表示部署时复用经过验证的外部
共享环境，而不是把虚拟环境打进压缩包。
