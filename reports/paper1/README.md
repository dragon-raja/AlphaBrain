# Paper 1 报告层

`initial_results.py` 负责当前 12 页报告的布局、文字说明和发布；`legacy_style.py` 保留历史训练图与原版样式。

科学统计、选择器与候选绘图实现位于 `AlphaBrain/research/dsol/`。报告层读取已核查产物，不训练 VLA，不重新评测，不修改统计总体。

兼容入口仍为 `scripts/dsol_paper1/build_unified_research_progress_v1.py`，唯一成品仍在 `docs/dsol_paper1/reports/current/`。

`historical/` 收纳 9 个旧阶段报告构建器；输入口径各不相同，不参与当前报告的默认生成。仍被活动控制器直接调用的旧报告依赖暂时保留原路径。
