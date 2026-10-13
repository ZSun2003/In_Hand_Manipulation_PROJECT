# RL16-API 旋转/平移工作空间

`R6_rotation_translation_workspace.py` 从本目录 XML 和 `R6_kinematics.py` 计算组合旋转条件下的数值平移工作空间。详见上级 [README](../README.md)。

- `--validate-only` 校验模型与扫描参数。
- `--help` 列出高度、旋转分量、方向、径向网格和并行参数。
- `--resume` 使用输出目录中的有效检查点继续计算。
- `--force` 允许重建指定运行的计算状态；使用前确认输出目录。

当前保留数值结果、检查点和聚合计算，不包含绘图函数。`R6_setup_cx.sh` 调用仓库共享 HPC 构建器，不创建本地环境。模型实体网格位于 `../models/simulation/rotation/mesh/`，本目录 `mesh` 为相对符号链接。
