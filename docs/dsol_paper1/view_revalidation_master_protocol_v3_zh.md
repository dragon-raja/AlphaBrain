# VLA 视角泛化与 Active-Ready 感知重验证总协议 v3

状态：`ACTIVE_PREREGISTERED_PROTOCOL`
日期：2026-08-19

## 1. 研究问题

本研究重做旧 Phase A/B/M0/M1，回答三个递进问题：

1. 足够宽、足够密的多视角训练能否提高标准 benchmark 上的被动视角泛化；
2. same-state pairing 及显式一致性是否在相同视角覆盖和训练预算下提供额外收益；
3. 当候选视角确实增加当前任务实体的可见像素时，策略能否利用该信息改善完整闭环行为。

Accel 只用于冻结候选池后的排序和关系归因，不替代可见性定义、训练对照或闭环结果。

## 2. 四层评价体系

| 层级 | 协议 | 规模 | 作用 | 能否称为官方 benchmark 结果 |
|---|---|---:|---|---|
| L0 | 原始 LIBERO Full | 40 tasks × 50 trials = 2,000 episodes/checkpoint | 基础任务能力与 canonical retention | 是，原始 LIBERO |
| L1 | LIBERO-Plus Camera Full | 1,599 tasks × 1 trial | 视角泛化正式主指标 | 是，LIBERO-Plus Camera |
| L2 | LIBERO-Plus Full | 10,030 tasks × 1 trial | 七类扰动整体鲁棒性与副作用 | 是，LIBERO-Plus Total |
| L3 | Exact-state / Constructed Blind-Reveal | 自建严格配对协议 | 机制、因果和信息利用 | 否，只能称研究诊断 |

此前 `view-gap-v1` 的 40-task Camera 抽样必须改称 `Camera-Dev40`：它按
4 suites × 5 difficulty levels × 2 tasks 分层抽样，共 40 个 Camera 变体；其
canonical 79/80 和 camera 64/80 都是完整闭环结果，但不是完整 LIBERO-Plus 指标。

## 3. 官方外部数字的使用边界

外部论文和排行榜只作背景参照，不作本方法的配对显著性检验：

| 外部结果 | 数值 | 使用方式 |
|---|---:|---|
| OpenPI `pi05_libero`，原始 LIBERO | 96.85% | 检查本地原始 LIBERO 复现是否合理 |
| LIBERO-Plus π0，Camera / Total | 13.8% / 53.6% | 说明官方 benchmark 难度；不是同一 π0.5 checkpoint |
| LIBERO-Plus OpenVLA-OFT+，Camera / Total | 92.8% / 79.6% | 数据多样化路线的外部上界参考；不是直接同模型对照 |

论文中的主要方法差异必须来自本地同 runtime、同 task IDs、同执行参数下的
`Official Pi0.5 frozen` 与各训练方法之间的配对比较。

## 4. 数据划分与泄漏控制

### 4.1 Camera-Dev40

- 只用于 pipeline smoke、视角目录调试和 seed 41 方法筛选；
- 任务清单冻结，不得因结果更改；
- 所有用 Dev40 选择过的方法必须在 Camera Full 上重新确认；
- 正式报告同时给 Camera Full 1,599 和 `Camera-Heldout1559`，后者排除 Dev40。

### 4.2 Camera Full

- 使用官方 `task_classification.json` 中全部 1,599 个 `Camera Viewpoints` 任务；
- 保留官方每任务一次闭环的 leaderboard-compatible 口径；
- 所有方法使用同一 task order、环境 seed、horizon、replan K、动作后处理和 policy noise ledger；
- 结果按 suite、difficulty、相机扰动族和原始 base task 分解。

### 4.3 Plus Full

- 使用官方四套任务中的全部 10,030 个扰动任务；
- 逐类报告 Camera、Robot、Language、Light、Background、Noise、Layout 和 pooled Total；
- 不用 Total 掩盖 Camera 主结果；
- 训练数据中出现的 pose support 可以重合，但 episode、物理状态和评测图像不得进入训练。

## 5. 模型与评测责任矩阵

`Required` 表示论文正式结果必须完成；`Gate` 表示 seed 41 通过后才扩展；`Context` 只作机制背景。

| 模型 | Original Full | Camera-Dev40 | Camera Full | Plus Full | Exact-state | Blind-Reveal |
|---|---|---|---|---|---|---|
| Official Pi0.5 frozen | Required | 已完成 | Required | Required | Required | Required |
| Canonical-unique | Gate | Required | Required, 3 seeds | 最终若入选 | Required | Context |
| Canonical-repeat | 不要求 | Required | 仅 seed 41 诊断 | 不要求 | Required | 不要求 |
| Image augmentation unique | Gate | Required | Required, 3 seeds | 最终若入选 | Required | Context |
| Legacy-MV8 | 不要求 | Required | seed 41 anchor | 不要求 | Required | Context |
| Broad32 unpaired | Gate | Required | seed 41；若接近 Broad64 则扩展 | 不要求 | Required | Required |
| Broad64 unpaired | Required, 3 seeds | Required | Required, 3 seeds | Required, 3 seeds | Required | Required |
| Broad paired FM | Gate | Required | 通过 quick gate 后 3 seeds | 最终若入选 | Required | Required |
| Broad paired consistency | Gate | Required | 通过 quick gate 后 3 seeds | 最终若入选 | Required | Required |
| Info-pose-support | Gate | Required | 通过 M0 后 3 seeds | 最终若入选 | Required | Required |

Plus Full 最终至少运行本地 Official frozen 与最终入选模型的三个训练 seeds。若最终方法不是
Broad64，则 Broad64 作为最强普通 coverage 基线也必须运行 Plus Full。

## 6. Phase A：被动视角鲁棒性

Phase A 的正式主表为 Camera Full，而不是 Camera-Dev40 或 Exact-state。

主要指标：

1. Camera Full success rate；
2. 相对本地 Official frozen 的绝对百分点差；
3. Camera-Heldout1559 success rate；
4. 按 suite、difficulty、扰动族的成功率；
5. Original LIBERO retention；
6. canonical-to-camera robustness ratio。

Exact-state 继续报告 canonical、Legacy held-out、Broad held-out、Wide extrapolation、Reveal、
Blind、external-only、wrist-only、all-camera blackout 和 scene × camera，但只能解释退化来源。

## 7. Phase B：训练覆盖、样本量与 pairing

### 7.1 固定 exposure，改变 pose support

```text
Legacy-MV8 -> Broad32 -> Broad64
```

固定 source windows、总模型样本数、global batch 32、更新数、优化器和 seed。

### 7.2 固定 pose support，改变 exposure

```text
Broad64-1x -> Broad64-2x -> Broad64-4x
```

### 7.3 固定状态与图像预算，改变 pairing

| 组别 | 独立状态 | 同状态跨视角 | 显式一致性 |
|---|---:|---:|---:|
| Canonical-unique | 高 | 否 | 否 |
| Canonical-repeat | 低，每状态精确重复 | 否 | 否 |
| Broad-unpaired practical | 高 | 否 | 否 |
| Broad-unpaired state-matched | 与 paired 相同 | 否 | 否 |
| Broad-paired FM | 与 state-matched 相同 | 是 | 否 |
| Broad-paired consistency | 与 paired FM 相同 | 是 | 是 |

Extreme、Look-away 和 blackout 不进入正常动作训练。Info-pose-support 只加入可操作、任务实体
可见且不包含测试 episode/state 的 Reveal pose。

## 8. M0：可见性信息门

第一轮保持旧定义：

\[
I_{task}(v)=\operatorname{mean}_{camera,entity}
\left[\text{visible-pixel-fraction}(entity,camera,v)\right]
\]

\[
\Delta I(v)=I_{task}(v)-I_{task}(v_{canonical})
\]

每个状态冻结 Canonical、Strong-info、Medium-info、Matched-control、Blind、Look-away 和 sensor
blackout。Strong-info 与 Matched-control 位姿变化幅度相近，但只有前者的可见性增量超过预注册阈值。

M0 必须同时通过人工画面审计、逐相机可见像素审计、wrist on/off 审计和任务分布审计，才准入 M1。

## 9. M1：完整闭环信息利用

M1 从冻结中间物理状态开始，以正常 replanning 运行至任务成功或超时。主要指标：

\[
\Delta SR_{info}=SR_{info}-SR_{canonical}
\]

\[
Specificity=(SR_{info}-SR_{canonical})-(SR_{control}-SR_{canonical})
\]

同时报告 full-task success、progress、Rescue、Harm、completion steps、stage completion，分别给
wrist on/off。动作差异和 Accel 只作辅助诊断。

## 10. 统计规范

### 10.1 官方兼容分数

- 严格复现官方 success/total 聚合，报告 Camera 1,599 和 Plus 10,030 的 pooled success rate；
- 同时报告每 category、suite、difficulty 的分解，不能只给 Total；
- 原始 LIBERO 按四个 suite 分别报告并给 suite macro average。

### 10.2 方法比较

- 同一 Plus 变体在方法之间严格配对；
- 主独立单位为原始 `base task`，不是同一 base task 派生出的每个扰动变体；
- 对方法差使用 paired base-task cluster bootstrap 10,000 次，报告 95% CI；
- 对单个成功率同时给 Wilson 95% CI，作为描述性区间；
- 三个训练 seed 分别报告；主差异先在每个 base task 内跨 seed 求均值，再做 base-task bootstrap；
- 额外报告 seed range，不把三个 seed 的 episode 当成独立样本扩大显著性；
- 同一 task 的 policy flow noise、环境 seed 和执行预算在方法间共享。

### 10.3 预注册实用门槛

- 被动视角改进：Camera Full 相对 Official frozen 至少 `+5pp`，paired cluster CI 排除 0；
- canonical 保持：Original Full 下降不超过 `5pp`；
- 非相机副作用：Plus Full 的非 Camera pooled success 下降不超过 `5pp`；
- pairing 贡献：Broad paired consistency 相对相同 coverage 的 unpaired 至少 `+3pp`，并跨 seed 同向；
- 信息利用：Blind-Reveal 的 Information Specificity 为正、CI 排除 0，且 oracle/expert 有行为 headroom。

若效果小于实用门槛但 CI 很宽，结论是 `UNDERPOWERED`，不是无效或等价。

## 11. Accel 分析

候选冻结后，同一状态共享 flow noise `x0`，记录 `accel_2...accel_10`，主指标预注册为
`accel_3`。报告它与 canonical、最近 train pose、Strong-info、Reveal 和
oracle@shortlist 的关系。Accel 只有在相同观察预算下改善闭环才可称选择器；否则只称策略兼容性指标。

## 12. 正式报告的固定表格

每个百分数必须同时标注 checkpoint、训练方法、任务总体、任务数、trials/task、episode 数、
runtime commit 和闭环 horizon。

### 表 A：原始 LIBERO

| Model | Spatial | Object | Goal | LIBERO-10 | Macro Avg | Episodes |
|---|---:|---:|---:|---:|---:|---:|

### 表 B：LIBERO-Plus Camera Full

| Model/Seed | Camera-1599 | Heldout-1559 | L1 | L2 | L3 | L4 | L5 | Delta vs local Official |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

### 表 C：LIBERO-Plus Full

| Model/Seed | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

### 表 D：Exact-state 与 Blind-Reveal

| Model/Seed | Canonical | Wide | Reveal | Matched-control | Blind | Info Specificity | Wrist-off Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|

## 13. 冻结执行顺序

1. 修复并标记已有 Camera-Dev40，不再称完整官方结果；
2. Official frozen 跑 Camera Full，建立本地官方兼容主基线；
3. Broad64 seed 41 跑 Camera Full；若无基础增益，先审计训练/部署，不扩 seeds；
4. 完成 Exact-state 的 coverage、exposure、pairing quick gate；
5. 通过 gate 的模型扩展 seeds 42/43，并跑 Camera Full；
6. 构造并审核 Blind-Reveal，完成 M0/M1；
7. Official frozen、Broad64 和最终模型运行 Original Full 与 Plus Full；
8. 完成 Accel 关系分析；
9. 最后在 RoboCasa 做跨 benchmark 验证。

## 14. W&B 与归档

训练和正式评测记录超参数、seed、Git commit、数据 manifest、任务进度和聚合指标；不上传原始数据、
视频、checkpoint、模型权重或 secrets。W&B 不可用时自动离线，训练不得因此中断。

每条 episode 独立原子落盘，支持断点续跑；所有成功和失败视频使用 AV1/WebM。最终保留任务 ledger、
raw per-episode JSONL、聚合 CSV/JSON、bootstrap 输入、配置和 commit hash。
