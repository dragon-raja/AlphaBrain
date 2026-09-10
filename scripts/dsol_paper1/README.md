# Paper 1 脚本导航

当前研究定义、协议和报告统一从 [研究入口](../../docs/dsol_paper1/README.md)进入。

| 用途 | 入口 / 共享模块 |
| --- | --- |
| 当前调度 | `dynamic_initial_scheduler_v1.py` |
| 标准初态整任务 | `standard_initial_aa0_v1.py`、`standard_initialization_v1.py` |
| 双机运行与闭环 | `dualhost_aa0_v1.py`、`evaluate_dsol_libero_hdf5_views.py` |
| 噪声与渲染一致性 | `explicit_flow_noise.py`、`render_bridge_common_v1.py`、`trace_view_repeatability_v1.py` |
| 训练匹配核验 | `audit_dsol_training_match.py`、`finalize_canonical_match_v1.py` |
| 指标和空间分析 | `analyze_view_metric_rules_v1.py`、`analyze_matched_view_metric_rules_v1.py`、`compare_matched_view_landscapes_v1.py` |
| 汇报 | `build_unified_research_progress_v1.py`；双周报使用 `docs/dsol_paper1/reports/biweekly/20260909/build_report.py` |
| 主动获取草案，非本轮已执行方法 | `view_acquisition_*.py` |

上述分析工具有历史数据 schema 和输入来源约束，不能直接把旧快照统计当成标准初态结果。当前调度脚本绑定冻结运行目录，也不是通用的一键新实验 CLI。不要为查看帮助而启动它。

当前运行使用共享盘的冻结 `repo/` 与 `scheduling/dynamic-v1/` 副本；本目录不是对正在运行实验的热更新接口。

## 查找与检查

```bash
python tools/paper1/check_layout.py --list
python tools/paper1/test.py
```

第一个命令提供按用途分组的保守导航和结构检查，不执行研究脚本；第二个命令统一运行 CPU 测试。所有 Paper 1 测试已移到 `tests/dsol_paper1/`。

## 为什么保留旧模块

当前代码仍复用历史阶段中的数据恢复、观测处理、可见性、Accel 和统计函数。名称含 `constructed`、`expectation` 或 `bridge` 不等于无用。`scripts/cabi_vla` 中的服务和相机工具也仍是依赖。

没有移走有调用关系的公共模块；12 个独立的一次性操作脚本已转入 [历史操作归档](../../archive/paper1/README.md)。旧实验的正式入口与分析器保留供证据复现，不属于本轮默认运行流程。

后续新增实验不再用 `after_*` / `retry_*` 文件堆叠主线：先明确是否可复用现有入口；特定运行的操作记录进入该运行目录，不修改冻结 release。涉及算法、默认参数、模块拆分的变更需单独经过科学一致性验收。
