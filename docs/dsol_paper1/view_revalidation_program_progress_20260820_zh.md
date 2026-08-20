# 视角泛化与 Active-Ready 研究计划真实进度

日期：2026-08-20  
结论：`PAPER_LEVEL_PROGRESS_APPROX_20_TO_30_PERCENT`

## 总体判断

当前不是接近完成，而是完成了一个正式门，并积累了若干开发证据：

- 完整通过：G0 seed-41 被动视角校准；
- 局部完成：G1/G2 的 Broad32 seed-41 quick gate、G5 自然场景两任务子门、G7 Camera Full；
- 核心尚未完成：三 seed、exposure、Broad64 pairing、constructed Blind-Reveal、完整 M1、Accel、Plus Full、RoboCasa 实验。

此前的迁移恢复、数据校验、renderer、正式 benchmark runner 和统计工具属于必要工程基础，
但不能等价计入论文级科学进度。

## 完整主线状态

| 门 | 研究内容 | 真实状态 | 可计入正式结论 |
|---|---|---|---|
| G0 | Official/Broad64 seed-41 Camera Full 校准 | COMPLETE | 是，单 seed 证据 |
| G1 | Legacy8/Broad32/Broad64 与 1x/2x/4x exposure | PARTIAL | 否 |
| G2 | unpaired/state-matched/paired FM/consistency | PARTIAL | 否 |
| G3 | seeds 41/42/43 正式矩阵 | NOT STARTED | 否 |
| G4 | Constructed Blind-Reveal 与 M0 可见性门 | NOT IMPLEMENTED | 否 |
| G5 | Info-pose-support 与完整 M1 | NATURAL DEV SUBGATE ONLY | 否 |
| G6 | Accel2-10、dense/dynamic、oracle relation | CORE NOT IMPLEMENTED | 否 |
| G7 | Original Full、Camera Full、Plus Full | Camera Full only; Original running | 部分 |
| G8 | RoboCasa 跨 benchmark | Assets ready; experiments not started | 否 |

## 已完成实物

### 数据和训练

- LIBERO HDF5：40 tasks、2,000 demonstrations、338,575 transitions；
- Broad32 与 Broad64：各 8 tasks、400 episodes、38,193 records；
- Broad32 seed 41：canonical unique/repeat、ImageAug、practical、state-matched、paired FM、consistency 七个 2,000-step 模型；
- Broad64 seed 41：unpaired practical 一个 2,000-step 模型；
- Info-only 与 Info+Control：两个自然场景开发模型。

### 评测

- Broad32 七模型：1,176 条 exact-state 闭环；
- Official 与 Broad64：各 168 条 exact-state；
- M0：160 states、14,080 candidate renders、160 montages；
- 自然 M1：3 models × 210 episodes，共 630；
- Camera Full：Official 与 Broad64 各 1,599，全部状态 complete；
- Broad64 Camera Full：82.61%，Official 75.86%，base-task clustered delta +5.23pp，95% CI `[+0.12,+10.69]`。

## 关键缺口

### G1/G2

- Legacy-MV8 只有 catalog；
- Broad64 缺 state-matched、paired FM、paired consistency；
- 1x/2x/4x exposure 尚未冻结定义和 runner；
- seeds 42/43 尚未开始；
- Broad32 pairing 只有 24 paired groups，CI 仍宽。

按正式三 seed G1/G2 矩阵估计，当前约完成四分之一。

### G4/G5

- 没有正式 constructed Blind-Reveal 场景；
- 没有场景级 Matched-control、constructed snapshot manifest、expert/oracle headroom；
- 现有自然 LIBERO 严格门仅剩两个任务，且 Info-only 收益被位姿 support 不对称混杂；
- 完整 M1 缺 6-10 tasks、三 seed、progress、Rescue、Harm 和 stage completion。

因此自然 M1 只能算开发参考，不能替代正式信息利用结论。

### G6

- 模型 sampler 能接收显式噪声并计算逐步 velocity，但不返回 flow trajectory；
- 尚无 `accel_2...accel_10`、candidate-group 共享 `x0` RPC、dense ranking、dynamic shortlist 或 oracle relation；
- G6 主执行链尚未实现，预计工程工作量约 4.5-7 人日。

## 执行纠偏

此前工作流过于串行：等待一个 GPU 门结束后才考虑下一模块，导致 benchmark 工程推进，
但 Blind-Reveal 和 Accel 没有同步实现。现改为三条并行主线：

1. GPU 线：Original retention -> Broad64 pairing -> selected multiseed；
2. 数据线：constructed Blind-Reveal/Matched-control -> M0 -> Info-pose-support/M1；
3. 模型线：flow trace/Accel2-10 -> dense relation -> dynamic shortlist。

当前 Original retention v3 已通过 10-episode smoke，并开始 Official 2,000-episode 正式评测。
