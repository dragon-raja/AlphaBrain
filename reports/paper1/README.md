# Paper 1 报告层

`search_tracking.py` 将已有搜索训练曲线、历史标签回放、真实仿真A/B和共同时间上限复核
分别整理为可审查的W&B导出包；默认离线，只上传指标、筛选配置和统计图。
登录、授权及下一轮训练接入边界见[实验看板协议](../../docs/dsol_paper1/protocols/experiment_tracking_zh.md)。

`biweekly_review.py` 根据冻结分析产物生成2026-10-15双周报（PDF、可编辑DOCX、Markdown共用内容块）。
日期、证据截点、输入与输出路径位于`configs/dsol_paper1/analysis/biweekly_20261015.json`；
这是独立汇报快照，不覆盖唯一研究进展PDF。沿用上一份3页A4格式，区分汇报日期与实际数据截止日期。

`initial_results.py` 保留 LIBERO 阶段报告构建；`legacy_style.py` 保留历史训练图与原版样式。
`robocasa_results.py` 构建 LIBERO/RoboCasa 完整矩阵统一报告；
`robocasa_diagnostic_update.py` 在该16页版上保留前14页，更新为18页的官方模型诊断版。
该更新器显式接收原发布目录、诊断与分析目录；先构建、检查预览，再用 `--publish-built` 发布。
完整400初态与40初态诊断分开标注，不运行实验，不改变原始统计。

`confirmation_update.py` 消费已验收的新初态独立确认统计，保留19页历史版的前18页正文，
用五窗口对照与任务差异两页替换末页展望，发布为同一个20页PDF。
它不从闭环成败重新选窗口；构建、逐页像素保留检查、预览与发布分开进行。

科学统计、选择器与候选绘图实现位于 `AlphaBrain/research/dsol/`。报告层读取已核查产物，不训练 VLA，不重新评测，不修改统计总体。

`window_synthesis.py` 消费三环境五窗口统一验收统计，保留前18页图表正文，将原末两页
融合为五页更新（23页同一PDF）。明确候选权限、执行噪声预算、初态拆分及RoboCasa未覆盖
的排除规范位终点；完整数值与不确定性放在同名配套说明中。先构建并预览，再显式发布。

统一构建入口为 `reports/paper1/unified.py`，当前报告薄入口为 `scripts/dsol_paper1/reporting/report_standard_initial_results_v1.py`；唯一成品仍在 `docs/dsol_paper1/reports/current/`。旧脚本根目录不保留兼容壳。

`historical/` 收纳旧阶段报告构建器；输入口径各不相同，不参与当前报告的默认生成。旧 PDF 监视器已获用户授权结束，报告代码不再因其固定调用路径滞留脚本根目录。
