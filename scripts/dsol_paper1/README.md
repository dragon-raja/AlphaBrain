# Paper 1 命令与历史工具

研究问题、结果和唯一当前 PDF 从 [研究入口](../../docs/dsol_paper1/README.md)进入。算法实现与目录规则见 [CONTRIBUTING](../../CONTRIBUTING.md)。

## 日常使用

| 要做什么 | 位置 |
| --- | --- |
| 当前初态结果核查、统计和选择器 | [薄分析入口](analysis/analyze_standard_initial_results_v1.py)；实现见 [研究包](../../AlphaBrain/research/dsol/README.md) |
| 当前统一汇报 | [统一入口](../../reports/paper1/unified.py)；实现见 [报告模块](../../reports/paper1/README.md) |
| 仿真输入、恢复、闭环与评分适配 | [runtime](runtime/)；不承担通用指标算法 |
| 历史指标、Oracle、闭环统计和画图 | [analysis](analysis/) |
| 历史候选空间、协议、噪声预算构造 | [protocols](protocols/) |
| 数据生成、训练、训练匹配与来源核查 | [training](training/) |
| 输入、渲染、运行身份和资源诊断 | [diagnostics](diagnostics/) |
| 历史运行编排 | [Python 控制器](operations/controllers/)、[Shell 启动配方](operations/launchers/) |
| 历史 PDF 构建 | [历史构建器](../../reports/paper1/historical/)；报告收尾脚本在 [reporting](reporting/) |
| 尚未用于正式评测的相机获取原语 | [acquisition](../../AlphaBrain/research/dsol/acquisition/) |

目录分组不是协议统一：历史工具仍有各自的数据 schema、模型与初始化约束。旧快照统计不能直接混入标准初态结果。

## 根目录不再保留旧入口

原 186 个平铺脚本已分两次完成职责迁移（149 + 37），根目录只保留导航，不留批量转发壳。Accel、可见性和显式噪声实现进入研究包；运行适配、调度、分析、协议和报告各有归属。

用户已授权结束暂停的旧 full64 控制器与旧 PDF 监视器；操作前复核没有评测子进程，保存了身份记录，未启动或重跑任何实验。原源码和路径映射见 [本轮迁移记录](../../archive/repository/20260911/runtime-modules/manifest.json)。

源码布局迁移不意味着旧 release 可以换上新源码继续跑。历史源码身份属于冻结 checkout；新的运行链正式投入实验前仍需首帧和整轨迹等价验收，不能给新文件冒用旧哈希。

## 旧路径在哪里

[迁移清单](../../archive/paper1/maintenance/script-layout-20260911/manifest.json) 记录旧路径、新路径、原始源码备份和受保护文件。旧命令需要按清单换路径，Python 调用方改用完整模块名；不会通过修改 `PYTHONPATH` 搜索所有子目录来掩盖依赖关系。

历史冻结 release 按源码哈希识别原实现。重放已有 release 应使用它自己的冻结 `repo/`；不要把迁移后的源码强行配上旧哈希清单运行。历史文档和实验记录不改写。

## 维护检查

```bash
python tools/paper1/check_architecture.py
python tools/paper1/check_layout.py --list
python tools/paper1/test.py
```

结构检查不执行研究脚本。完整测试走 CPU；少数显式列出的 `--help` 测试只验证参数入口。不要用导入或 `--help` 试探未审查的历史控制器。

新增通用实现进入研究包；新增 CLI 进入相应职责目录。不得恢复已经迁出的平铺路径，也不以 `after_*`、`retry_*` 或复制版本来代替接口设计。
