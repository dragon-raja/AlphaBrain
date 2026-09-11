# Paper 1：VLA 的视角利用

本仓库当前主线：**比较规范视角与多视角后训练后，模型在固定候选相机空间中的闭环表现，以及指标能否利用其中的选择收益。**

当前实验从官方任务初态执行完整任务；外部相机每次执行中固定。不是中间状态恢复实验，也不是任务途中主动移动相机。候选图可查询的指标分析不能写成单实体相机无需移动就能获得所有视角。

## 从这里开始

| 要找什么 | 唯一入口 |
| --- | --- |
| 当前完整实验定义与执行协议 | [标准初态 AA0 记录](execution/standard_initial_aa0_execution_20260908_zh.md) |
| 当前汇报材料 | [研究进展 PDF](reports/current/vla_view_research_progress_zh.pdf) · [图表说明](reports/current/vla_view_research_progress_zh.md) |
| 双周汇报交付 | [2026-09-09 双周报源文件](reports/biweekly/20260909/build_report.py) |
| 代码用途与入口 | [脚本导航](../../scripts/dsol_paper1/README.md) |
| 开发规则与模块边界 | [CONTRIBUTING](../../CONTRIBUTING.md) · [可复用研究包](../../AlphaBrain/research/dsol/README.md) |
| ACD / basin / branch 等旁支 | [代码归档与恢复映射](../../archive/research/README.md) |
| 共享服务与相机工具 | [VLA 共享模块](../../scripts/vla_shared/README.md) |
| 训练匹配与证据边界 | [训练匹配审计](audits/training_match_anchor_audit_20260907_zh.md) |
| 过去的计划、结果、版本 | [历史项目索引](HISTORY.md) |
| 不属于当前论文的旁支文档 | [历史研究档案](../archive/research/README.md) |
| 已退休的操作脚本与路径映射 | [操作归档](../../archive/paper1/README.md) |

历史文档中的“当前”“运行中”和预计用时只代表文档当时的截点。最新实际进度读取运行目录的状态和完成收据；已有 PDF 不因代码整理自动成为新矩阵结果。

2026-09-11 更新：198,656 条正式记录已完成并完成本轮全量身份／覆盖核对；当前唯一 PDF 已加入新初态统计、空间图、预定规则与两类探索性离线选择器。选择器只用初态 0、1 开发，在 2、3 评价；部分测试汇总此前已查看，不声称完全盲测确认。本次新增闭环 0 次、VLA 训练 0 步。统计与选择器产物入口为 `/share/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908/analysis/research-v1/latest.json`。

## 当前实验边界

- 两个匹配后训练模型；8 个任务、每任务 4 个官方初态。
- 每初态 97 个固定候选位姿、32 套显式 Flow 噪声序列，共 198,656 次整任务闭环。
- 按重规划索引配对噪声，去噪网格、时限和 AA0 渲染协议固定。
- 同一矩阵比较规范视角、全局/每任务经验最佳固定视角、每初态经验 Oracle；先对噪声平均，再选视角。
- Oracle 是有限样本事后参照，不等于可部署选择器。选择规则使用初态留出，且要说明候选图查询权限。
- 当前不自动扩展数据或候选，不启动主动相机或 RoboCasa 实验，不将历史恢复快照结果混入此矩阵。

具体定义、划分和限制以当前执行协议及冻结 release 为准；本 README 不另立实验口径。

## 运行与源码分离

正式运行目录：

```text
/share/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908/
  release.json                         冻结科学协议
  repo/                                冻结执行代码，不修改
  scheduling/dynamic-v1/               单独验收过的调度扩展
  hosts/fresh/status.json               fresh 当前状态
  hosts/gnu/status.json                 gnu 当前状态
```

当前工作仓库 `/workspace/projects/alphabrain-dsol-paper1` 用于后续开发；整理它不会热更新上面的冻结副本。不要在本仓库直接重跑建库、迁移或调度命令来“检查状态”。

## 开发检查

在具备本项目依赖的 Python 环境中，从仓库根目录运行：

```bash
python tools/paper1/check_layout.py
python tools/paper1/check_architecture.py
python tools/paper1/check_layout.py --list
python tools/paper1/test.py
```

本机可使用 `/alphabrain/.venv/bin/python` 替代 `python`。测试入口会关闭 GPU 可见性并限制 CPU 数值库线程；部分测试只读已有冻结 release，不代替真实闭环验收。默认不会安装依赖、训练模型或启动正式评测。

## 保存约定

- 代码、测试、配置、文字说明和来源收据进入 Git。
- 新生成的报告 PDF、DOCX、预览图和字体副本留在本地/共享盘，由限定目录的忽略规则管理；原文件不删除，已跟踪历史证据不取消跟踪。
- 报告源文件仍在 Git，但 PDF 重建还依赖外部图表与数据，不声称仅凭克隆仓库即可完整重建。
- 冻结协议、权重、原始结果和判废记录保持原路径及来源链，不在本轮清理。
- 95 份旁支文档已转入 `docs/archive/research/`；论文引用的两份 KYC 证据和原图/历史配置保留原位。
- 当前共享服务、观测和相机实现位于 `scripts/vla_shared/`。旧 CABI 四个同名文件仅作兼容入口；绑定旧冻结 release 的驱动只在哈希核验后使用其中的原副本。
- 2026-09-11：新初态数据核查、指标规则、选择器与统计已抽入 `AlphaBrain/research/dsol/`；报告布局移到 `reports/paper1/`。旧入口保留兼容，科学结果不因目录调整重写。
- 2026-09-11 后续整理：主线脚本平铺区从 186 个文件减到 37 个，149 个按职责迁移；完整旧→新路径见 [迁移清单](../../archive/paper1/maintenance/script-layout-20260911/manifest.json)。活动运行链及其源码身份目标原字节不动；4 个旧导入即执行脚本已增加显式入口。
- 8 个退出主线的旁支共 208 个文件已原字节归档到 `archive/research/code/scripts/`；数据、权重、冻结运行目录不动。FRESH 中仍被历史 CABI 引用的两项工具保留共享实现。
- 21 个混放测试已统一到 `tests/dsol_paper1`；12 个已核对的一次性操作脚本转为历史文本归档。其余脚本不能据此视为已完成模块化重构。
