# KYC × Pi0.5 视觉对齐执行状态

更新时间：2026-08-03

## 当前裁决

双相机最终机制筛选已完成，裁决为：

> **DO_NOT_ADVANCE_FROM_DUAL_CAMERA_SCREEN**

在完全匹配的训练和评测条件下，Control 成功率为 20.71%，External-KYC 为
20.00%，Wrist-KYC 为 18.57%，Dual-KYC 为 15.71%。Dual 相对 Control 为
`-5.00 pp`，95% CI `[-10.71,0.00] pp`；默认相机位姿为 `-15 pp`，95% CI
`[-25,-5] pp`。正确腕部 ray 也没有优于初始固定或 previous-call ray。

因此没有启动原协议中条件性的 33K updates、seeds 42/43 扩展。完整结果、图表和
解释见 `docs/cabi_vla/kyc_dual_camera_validation_result_zh.md`。

## 已完成实验

- 五组训练：RGB、Control、External、Wrist、Dual；
- 每组 2,000 updates、seed 41、完全相同 Pi0.5 初始化和 22,464 条训练记录；
- 每组 140 个闭环 episode，五组共 700 个；
- 正确、初始固定、previous-call 腕部 ray 干预共 280 个 episode；
- 总计 980 个闭环 episode；
- 36 个单方法和 4 个配对 AV1/WebM 视频；
- 组级配对 bootstrap、子目标、视野边界和 2×2 因子统计均完成。

## LIBERO-Plus

- 6.4GB assets 与 16.6GB camparam RLDS 均通过固定 SHA256；
- runtime 固定到官方 commit `4976dc30028e805ff8094b55501d532c48fec182`；
- 2,876 episodes、449,053 steps、40 tasks、256 shards 全量审计完成；
- 目标任务有 63 episodes 和 33 个外部相机位姿；
- 独立 Python overlay、MagickWand、非交互配置和真实双相机 MuJoCo smoke 通过；
- prepare launcher 已可幂等恢复，不会重复下载或覆盖完整 runtime。

## 尚未执行

LIBERO-Plus 上的 Pi0.5 闭环基线尚未执行，因为当前没有经过格式核验的标准
Pi0.5-LIBERO checkpoint。当前 LIBERO-Bind 单任务模型与官方 40-task RLDS 分布
不一致，强行评测不能产生可解释结论。

下一阶段应先核验 4 卡机已有约 9GB `pi05-libero`；兼容则迁移后跑 canonical 与
291 个目标任务官方变体，不兼容再决定转换或训练标准基线。

## 镜像注意

`/workspace/.downloads/libero-plus` 仍保留约 22GB 下载 staging；正式校验副本已经
位于 `/share`。本轮未清理 staging，也未执行任何 image clean/apply。共享镜像前
应先走既有 size-check、image-guard、clean dry-run 和 artifacts verify 流程，再由
用户决定是否删除。
