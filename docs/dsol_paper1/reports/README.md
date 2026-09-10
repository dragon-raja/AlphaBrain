# 报告入口

日常汇报只看 [current/vla_view_research_progress_zh.pdf](current/vla_view_research_progress_zh.pdf)，对应[图表说明](current/vla_view_research_progress_zh.md)。

当前为 12 页修订版：主线是任务初态完整执行、噪声平均后按初态选最佳视角；中间状态恢复实验仅为辅助证据。这里不是评测调度入口。

## 目录约定

- `current/`：当前唯一 PDF 和说明，后续生成器更新同一文件。
- [archive/](archive/README.md)：旧报告与历次交付版本索引，不作为当前结论入口。
- [cleanup-receipt-20260908.json](cleanup-receipt-20260908.json)：本次移动、去重、压缩记录及恢复所需的文件哈希。

根目录的 `vla_view_research_focus_20260907_zh.pdf` 和 `vla_view_landscape_and_metrics_20260908_zh.pdf` 只是兼容链接，指向同一个当前文件，不再存两份 PDF。旧链接仍然可用。

## 保留规则

每次交付的 PDF、统计来源、版本收据和原始实验数据保留。旧构建的逐页预览与拼装中间 PDF 压缩为 `build-previews.tar.gz`；最新构建预览保持展开，便于审阅。报告归档中的相同 PDF 备份去重为指向不可变交付版的链接，文件内容和访问路径不变。

恢复旧预览：在相应构建目录中解开 `build-previews.tar.gz`，即可恢复原来的 `pdf-qa/` 和 `research-panels.pdf`。恢复重复 PDF 的独立副本：复制兼容链接指向的文件即可。没有永久删除唯一一份报告内容。

实验执行代码、训练数据、权重、闭环账本、冻结配置与其他分支文件不在此次清理范围内；不要仅凭文件名像临时脚本或 Git 标为 untracked 就删除。
