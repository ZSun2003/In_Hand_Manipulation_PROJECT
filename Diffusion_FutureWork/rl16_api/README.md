# RL16-API / G6：YCB GRAB 源码归档

这是已完成 IRP 的 RL16-API / G6 训练源码。完整目录结构、模型来源、共享环境、版本区别与归档验证见 [FutureWork 主说明](../README.md)。

- `IK_RL/`：本机 SB3 TQC 残差训练、控制器桥接和评估。
- `MJWarp_RL/`：r22 G6-only 的 Diffusion / IK / TQC 定义与 HPC 训练；单任务历史预算 90,000,000 步、每个 episode 4,500 ticks。
- 根级 GRAB 控制器、运动学和物体规划：训练所需的无图形依赖，保留原控制逻辑。
- 根级 XML：原始多物体场景及训练单物体分片场景。
- `models/mesh/`：机械手和仿真网格。
- `models/objects/ycb/`：banana、pear、strawberry、orange 的源模型及源纹理。
- `fruit_capsules_5mm*.json/xml`：碰撞模型源定义和源网格 hash。

从此目录可检查训练接口：

```bash
../../RL_env/bin/python -m IK_RL.train --help
./MJWarp_RL/run_hpc.sh -m MJWarp_RL.train --help
../../RL_env/bin/python -B -m unittest discover -s IK_RL/tests
../../RL_env/bin/python -B -m unittest discover -s MJWarp_RL/tests
```

不要在一个 Python 进程中混入另一机械手的 `IK_RL` / `MJWarp_RL` 同名包。没有保留数据、权重或证书；本目录用于源码归档，以上帮助与回归验证不启动正式训练。
