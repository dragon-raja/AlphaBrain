# 已交付报告

当前 full 版已按用户要求于 **2026-10-23原位更新为39页**：保留前33页正文与图表，追加34–39页搜索进展，统一总页码，不另存历史PDF。
新增内容为固定97候选真实结果、连续PPO接口、旧连续首版任务成绩、三组修正版训练曲线，以及待做评测和center接管计划。奖励与任务成功率分别报告。
本机统一入口在源码根目录 `WORKSPACE.md` 中的“已汇报材料”，以及 `artifacts/reports/presentation.pdf`。
报告文件及机器人资产不因源码发布而自动上传。

本目录保留的 [旧报告](current/vla_view_research_progress_zh.pdf) 和[说明](current/vla_view_research_progress_zh.md)
属于 2026-10-13 截点，其 `current` 是历史生成器使用的兼容目录名，**不代表最新研究进度或最新汇报版本**。
原 publication.json 和文件字节保留，避免破坏已有证据链。

最新 RL 状态和未解决问题以 [项目首页](../README.md) 为准；此前 PDF 不包含后续全部 RL 结果。
历史交付、清理和恢复记录见 [报告档案](archive/README.md)。后续只有用户要求更新汇报时才改交付材料。

当前续接构建器为 `reports/paper1/search_progress_appendix.py`，显式接收项目数据根目录、当前PDF、核验输出目录和字体目录。
同一构建器重跑只替换34–39页，不重复追加；检查前33页正文像素保持一致。数值与来源哈希在项目 `runs/report-search-progress-20261023`，当前回执从 `artifacts/reports/current.json`进入。
旧日期目录的 `build.py`仅为先前修订来源，不再用它覆盖当前39页版。
