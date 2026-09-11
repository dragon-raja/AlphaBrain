# 已退出主线的研究代码

ACD、basin、branch、FRESH、CORA、CCV、policy-response 和 continuation-distill 的代码已整体保存在 `code/scripts/`。这些是历史研究资产，不是垃圾，也不是当前 Paper 1 默认入口。

- [逐文件原路径、归档路径与 SHA256](code/manifest.json)
- [本轮维护验收记录](../paper1/maintenance/acceptance-20260911.json)
- 历史研究文档仍在 `docs/archive/research/`，本轮没有改写它们的研究结论。
- 数据、模型、运行结果和共享盘冻结源码保持原路径，未迁移、未删除。
- 归档源码保留原字节与旧 import，因此不承诺从归档新位置直接运行。重放时使用历史冻结 checkout，或在独立恢复目录按 manifest 恢复原布局和依赖；不要把归档覆盖回当前主线。
- 默认测试和安装均排除归档，归档不能成为新主线模块的依赖。

FRESH 的视频工具、旧 Pi05 服务仍被部分历史 CABI 工具调用：实现保留在 `scripts/vla_shared/historical_*`，旧 `scripts/fresh_vla/` 仅保留这两项兼容入口。它们不是本轮标准初态评测的服务实现。

归档可恢复。维护验收会校验全部归档文件的原字节，不通过改写旧文件掩盖路径变化。

CABI/KYC 历史代码另有 193 个文件归入 `code/scripts/cabi_vla/`，来源链见 [CABI 归档收据](../repository/20260911/cabi/manifest.json)。仍被当前基准调用的 12 个原文件保留于 `scripts/cabi_vla`，不作为新算法目录。
