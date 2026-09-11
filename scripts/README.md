# 执行入口

| 范围 | 目录 |
| --- | --- |
| Paper 1 当前入口及按职责组织的历史工具 | [dsol_paper1](dsol_paper1/README.md) |
| 共享相机、观测、策略服务 | [vla_shared](vla_shared/README.md) |
| 基准与旧路径兼容工具 | [cabi_vla](cabi_vla/README.md) |
| FRESH 的两个历史共享工具兼容入口 | [fresh_vla](fresh_vla/README.md) |
| 框架训练范式 | `run_base_vla`、`run_brain_inspired_scripts`、`run_continual_learning_scripts`、`run_rl_scripts`、`run_world_model` |
| 框架验证 | `verify_vla` |

根目录的 `parse_config.py`、`download_pretrained.py`、`ai2r_smoke.py` 和训练／评测 Shell 是框架公共入口，保留原命令接口。它们不应该移动到文档目录。

通用算法和数据实现进入 `AlphaBrain`；报告生成进入 `reports`；说明进入 `docs`；退出研究主线的代码进入 `archive`。禁止把新研究实现堆回根目录，禁止从核心包反向导入这里。
