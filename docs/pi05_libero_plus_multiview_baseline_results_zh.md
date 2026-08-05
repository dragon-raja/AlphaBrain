# Pi0.5 × LIBERO-Plus 强多视角基线结果

## 结论摘要

在 seed 41、固定 33,000 次更新的第一轮门控中，强多视角 RGB 训练基本消除了 Pi0.5 在 LIBERO-Plus 官方相机扰动上的成功率缺口。最佳模型为 `SigLIP Visual-LoRA + Action Expert`、25% 可采样数据池：

- canonical 成功率：97.5%；
- 官方相机扰动成功率：97.5%；
- 视角缺口：0.0 个百分点，paired 95% CI `[-3.8, +3.8]`；
- 相对官方 Pi0.5 的相机扰动成功率：+17.5 个百分点，paired 95% CI `[+7.5, +28.7]`。

本轮判定为：

`MULTIVIEW_DATA_SUFFICIENT_ON_LIBERO_PLUS_OFFICIAL_PERTURBATIONS`

该判定只适用于当前 LIBERO-Plus 官方相机扰动分布，不能外推为 RoboCasa、真实机器人、相机与场景组合泛化或极端目标出画条件已经解决。

## 四模型闭环结果

统计单位为 40 个 `suite × 基础任务`。每个任务先平均两个相同初始状态，再执行 paired group-level bootstrap。

| 模型 | Canonical | 官方相机扰动 | 视角缺口 | 相对官方模型的扰动增益 |
|---|---:|---:|---:|---:|
| 官方 Pi0.5 | 98.8% | 80.0% | 18.8% | 0.0pp |
| Frozen-VLM Action Expert，100% 池 | 90.0% | 88.8% | 1.2% | +8.8pp |
| Frozen-VLM Action Expert，25% 池 | 95.0% | 83.8% | 11.2% | +3.8pp |
| SigLIP Visual-LoRA + Action Expert，100% 池 | 96.2% | 91.2% | 5.0% | +11.2pp |
| SigLIP Visual-LoRA + Action Expert，25% 池 | 97.5% | 97.5% | 0.0% | +17.5pp |

官方模型到最佳模型的扰动成功率增益为 +17.5pp，paired 95% CI `[+7.5, +28.7]`，同时 canonical 只下降 1.3pp。

## 视觉适配对照

模型间直接配对结果：

| 比较 | 官方相机扰动差异 | Paired 95% CI |
|---|---:|---:|
| Visual-LoRA 25% - Frozen-VLM 25% | +13.75pp | `[+5.0, +23.75]` |
| Visual-LoRA 100% - Frozen-VLM 100% | +2.5pp | `[-3.75, +8.75]` |
| Visual-LoRA 25% - Visual-LoRA 100% | +6.25pp | `[-1.25, +15.0]` |
| Frozen-VLM 100% - Frozen-VLM 25% | +5.0pp | `[-2.5, +13.75]` |

因此当前可以认为视觉表征适配在 25% 数据池条件下具有明确增益，但不能认为 Visual-LoRA 在所有数据规模下都稳定优于冻结视觉，也不能认为 25% 数据天然优于 100%。

33,000 次更新、梯度累积 2 意味着每组约消费 66,000 个窗口：25% 池约覆盖其 90,474 个窗口的 73%，100% 池只覆盖 361,006 个窗口的 18%。25%/100% 比较测量的是固定训练算力下的数据池效应，不是完整 epoch 的数据规模曲线。

## 候选视角与主动感知

最佳模型额外完成 480 个候选视角 episode，与 gap 评测合并为 640 episode：

- holdout canonical：90.6%；
- 各固定候选视角：90.6%–93.8%；
- 动作不确定性选择：96.9%，相对 canonical +6.2pp，95% CI `[0.0, +15.6]`；
- 事后 oracle：96.9%，相对最佳全局固定视角 +6.2pp；
- 仅 12.5% 的任务会随候选视角改变成败。

主动选择与固定视角优化均未获得 paired CI 排除 0 的稳定优势。多视角训练后，简单开局选视角的剩余上界已经较小，不应作为当前主要算法方向。

## 解释边界与下一步

第一轮只有 seed 41。论文级结论需要至少复验最佳模型 seeds 42、43；若要确认视觉适配的因果增益，应同步复验 25% Frozen-VLM 对照。

RoboCasa 应独立检验：

- 外部相机、腕部相机和双相机扰动；
- held-out camera pose cells；
- held-out kitchen/background；
- camera × scene 组合泛化；
- 多视角训练后是否仍保留至少 5–10pp 的缺口。

只有这些条件仍有稳定残余，才需要继续 KYC 或新的显式几何模块。

## 产物

- 四模型 gap 评测：`/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/gate-v1/*-gap`。
- 最佳模型候选视角评测：`/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/gate-v1/visual_b025-candidates`。
- 合并报告与图：`/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/gate-v1/final-best`。
- 四个自包含 checkpoint：`/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/runs`。
