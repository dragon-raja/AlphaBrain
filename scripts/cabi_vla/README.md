# 当前仍被引用的 CABI 兼容与基准链

此目录从 205 个文件收敛到 12 个既有文件：共享服务与相机兼容入口、OpenPI 请求噪声适配、基准运行入口及其分析依赖。它们保留原字节，供当前 Paper 1 调用和冻结身份核查。

其余 193 个历史 CABI/KYC 文件（含原测试）在 `archive/research/code/scripts/cabi_vla`。原路径、SHA256 与恢复信息见 `archive/repository/20260911/cabi/manifest.json`。恢复历史研究使用独立 checkout，不在当前目录回填整个旧项目。

仍在使用的分析与服务测试复制到 `tests/cabi_vla`；共享兼容性另由 `tests/vla_shared` 覆盖。此目录不是新算法实现位置。
