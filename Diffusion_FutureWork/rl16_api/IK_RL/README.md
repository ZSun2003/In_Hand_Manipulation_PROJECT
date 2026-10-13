# RL16-API / G6 — IK_RL

本目录是已完成项目的 Gymnasium / MuJoCo / SB3 TQC 残差训练 源码归档。算法、预算和控制常量以当前 `config.py` 或 `contract.py` / `hpc_config.py` 为准。

源码职责、环境、模型资产、版本边界与验证命令统一记录在 [FutureWork 主 README](../../README.md)。从平台父目录通过 `python -m IK_RL.train --help` 查看入口；共享环境位于 `../../RL_env`（相对于平台目录）。

`train.py` 是训练入口，`env.py` 定义环境，测试位于 `tests/`。评估、模型 hash 与容量/控制契约保留，图形分析和历史运行产物已删除。本源码不附带训练权重、示范数据或历史容量证书。
