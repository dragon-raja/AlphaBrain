# VLA 视角泛化与 Active-Ready 感知重验证总协议 v2

状态：`ACTIVE_CORRECTED_PROTOCOL`  
日期：2026-08-19

## 1. 研究目标

本研究重做旧 Phase A/B/M0/M1，回答两个递进问题：

1. 足够宽、足够密的多视角训练是否能解决被动视角泛化；
2. 当候选视角确实增加当前任务实体的可见信息时，策略是否能利用该信息改善完整闭环行为。

Accel 只在上述问题成立后用于候选视角排序和关系归因，不替代可见性定义、训练对照或闭环结果。

## 2. 三条不可混用的证据轨道

| 轨道 | 数据与协议 | 作用 |
|---|---|---|
| O：官方基线 | 完整 40-task LIBERO-Plus 官方协议 | 确认 Official Pi0.5 的标准能力和官方相机扰动缺口 |
| E：Exact-state | 官方 LIBERO HDF5 状态，在同一 LIBERO-Plus runtime 中统一重渲染 | 严格比较相机覆盖、数据组织、模型和通道消融 |
| C：Constructed evidence | 在 LIBERO-Plus runtime 内加入可审计的 Blind-Reveal 遮挡或场景变体 | 制造强信息差异，检验信息视角是否能改变闭环行为 |

轨道 E 不是完整 LIBERO-Plus 官方 benchmark。只有轨道 O 的结果可以称为“官方 LIBERO-Plus 指标”。

轨道 O 不只评 Official checkpoint。通过 Exact-state 快速门的关键模型也必须回到
完整 40-task 官方协议，至少包括 Canonical-unique、Broad32、Broad64 和最佳 paired
模型。这样分别检验标准 benchmark 泛化与自定义强信息干预，避免只在自建子集上得出结论。

## 3. WP0：基线锚定

必须同时报告：

| 基线 | 评测 |
|---|---|
| Official Pi0.5 frozen | 官方 40-task LIBERO-Plus canonical 与官方 camera perturbation |
| Official Pi0.5 frozen | Exact-state 168-episode 协议，与所有 LoRA 模型同初态、同视角、同 flow-noise seed |
| Canonical-unique | Exact-state 协议 |
| Canonical-repeat | Exact-state 协议；仅作为 paired exposure 的预算匹配对照 |
| Legacy-MV8 | Exact-state 协议；旧窄视角锚点 |

已有官方轨道 O 结果为 canonical `98.75%`、官方相机扰动 `80.0%`。它不能与 8-task Exact-state LoRA 结果直接横向比较。

## 4. Phase A：被动视角鲁棒性

Phase A 同时包含两张不可互换的主表：

1. 官方 LIBERO-Plus 40-task canonical / camera perturbation 表；
2. Exact-state 自建相机与通道干预表。

所有策略使用相同测试初态和完整闭环 horizon，Exact-state 表报告：

- canonical；
- Legacy held-out；
- Broad held-out；
- Wide extrapolation；
- Reveal；
- Blind / Look-away；
- external-only、wrist-only、all-camera blackout；
- scene shift × camera shift。

Phase A 的百分数是从任务初始状态开始的完整闭环任务成功率。

## 5. Phase B：训练覆盖、样本量与配对

### 5.1 固定 exposure，改变 pose support

```text
Legacy-MV8 -> Broad32 -> Broad64
```

固定 source windows、总模型样本数、global batch 32、更新数、优化器和 seed。

### 5.2 固定 pose support，改变 exposure

```text
Broad64-1x -> Broad64-2x -> Broad64-4x
```

### 5.3 数据组织与算法

| 组别 | 独立状态 | 同状态配对 | 显式一致性 |
|---|---:|---:|---:|
| Canonical-unique | 高 | 否 | 否 |
| Canonical-repeat | 低，每状态重复 | 否 | 否 |
| Broad-unpaired practical | 高 | 否 | 否 |
| Broad-unpaired state-matched | 与 paired 相同 | 否 | 否 |
| Broad-paired FM | 与 state-matched 相同 | 是 | 否 |
| Broad-paired consistency | 与 paired FM 相同 | 是 | 是 |

Extreme、Look-away、blackout 不进入正常动作训练，只作为评测负对照和信息视角筛选候选。

## 6. M0：强信息干预门

第一轮保持旧可见性定义，不引入决策信息、oracle action 或 Accel：

\[
I_{task}(v)=\operatorname{mean}_{camera,entity}
\left[\text{visible-pixel-fraction}(entity, camera, v)\right]
\]

候选相对 canonical 的增量为：

\[
\Delta I(v)=I_{task}(v)-I_{task}(v_{canonical})
\]

超过冻结阈值的候选定义为 Strong-info。每个状态必须同时有：

- Strong-info / Reveal；
- Medium-info；
- 位移幅度匹配但 `Delta I` 接近零的 Matched-control；
- Blind / Look-away；
- black external、black wrist、all-camera blackout。

若自然 LIBERO 场景不足，直接在 LIBERO-Plus runtime 中构造非碰撞遮挡板、容器/抽屉内部遮挡或左右目标可见性任务。M0 只有在人工可见差异、像素可见性差异和 matched-control 同时通过后才准入 M1。

## 7. M1：信息利用的完整闭环验证

M1 从冻结的中间物理状态开始，持续正常 replanning 至任务成功或超时。其成功率是“从反馈状态继续完成任务”的闭环成功率，不等同于 Phase A 的初始状态端到端成功率。

主要比较：

\[
\Delta SR_{info}=SR_{info}-SR_{canonical}
\]

\[
Specificity=(SR_{info}-SR_{canonical})-(SR_{control}-SR_{canonical})
\]

完整矩阵至少包括 Official、Canonical-unique、Legacy-MV8、Broad32、Broad64、Info-pose-support、Broad-paired FM 与 Broad-paired consistency，并分别报告 wrist on/off。

主要指标为 full-task success、progress、Rescue、Harm、completion steps 和 task-stage completion。动作差异只作诊断。

## 8. Accel 分析

在 M0/M1 候选冻结后实现：

- 同一状态所有候选共享 flow noise `x0`；
- 记录 `accel_2 ... accel_10`，主指标预注册为 `accel_3`；
- 64-96 视角 dense ranking；
- 8-16 视角 dynamic shortlist；
- 与 canonical、最近 train pose、Strong-info、Reveal 和闭环 oracle@shortlist 的关系。

Accel 若总选择训练熟悉视角，只能解释为兼容性指标；只有它稳定接近 Reveal/闭环 oracle 并改善闭环时，才能视作有效 view-value proxy。

## 9. 统计与正式门

- task × source episode / physical-state group 为独立单位；
- 同组条件严格配对；
- quick gate 只使用 seed 41；
- 正式结论使用 seeds 41/42/43；
- paired group bootstrap 95% CI；
- 任何方法结论必须同时给出绝对百分点差与每 seed 结果。

## 10. 当前结果的解释边界

当前自然场景 M1 只能说明：在 2 个任务和较弱可见性差异下，Info 没有稳定优于同位移 Control。它尚未排除：

- 信息增量不够强；
- 多看到的像素并非完成任务所必需；
- candidate pose OOD；
- wrist 提供视觉捷径；
- 任务和统计功效不足。

因此目前不能下结论“Pi0.5 无法利用信息视角”。只有 constructed Blind-Reveal 通过 M0 后的 M1 才能回答该问题。

## 11. 冻结执行顺序

1. 补 Official frozen 的 Exact-state 同协议基线；
2. 完成 Broad64 训练和同协议 Phase A；
3. 补 Legacy-MV8、Broad64 exposure 与 pairing 矩阵；
4. 在 LIBERO-Plus runtime 构造并审核 6-10 个强 Blind-Reveal 任务；
5. 通过 M0 后运行完整 M1；
6. 实现 Accel 并做关系分析；
7. quick gate 成立后扩展 seeds 42/43；
8. 最后再用 RoboCasa 做跨 benchmark 验证。
