# Paper 1 报告层

`initial_results.py` 负责当前 12 页报告的布局、文字说明和发布；`legacy_style.py` 保留历史训练图与原版样式。

科学统计、选择器与候选绘图实现位于 `AlphaBrain/research/dsol/`。报告层读取已核查产物，不训练 VLA，不重新评测，不修改统计总体。

统一构建入口为 `reports/paper1/unified.py`，当前报告薄入口为 `scripts/dsol_paper1/reporting/report_standard_initial_results_v1.py`；唯一成品仍在 `docs/dsol_paper1/reports/current/`。旧脚本根目录不保留兼容壳。

`historical/` 收纳旧阶段报告构建器；输入口径各不相同，不参与当前报告的默认生成。旧 PDF 监视器已获用户授权结束，报告代码不再因其固定调用路径滞留脚本根目录。
