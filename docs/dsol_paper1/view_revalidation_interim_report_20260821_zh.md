# VLA 视角泛化与 Active-Ready 感知重验证中期报告

状态：`M_A_COMPLETE_M_B_RUNNING`

日期：2026-08-21

本报告按总协议的证据顺序整理当前结果：数据构造、训练对照、被动视角泛化、M0 可见性筛选、
M1 完整闭环信息利用、Accel 候选排序，以及仍在运行的 M-B 正式确认。旧 Phase B/M0/M1 仅作为
窄视角 Legacy Anchor，不承担“图像增强等价于多视角”“pairing 无效”或“主动视角无效”等泛化结论。

## 1. 当前结论摘要

| 研究问题 | 当前证据 | 中期判断 |
|---|---|---|
| 宽多视角训练能否提高被动视角泛化 | Exact-state 与 seed-41 Camera Full 均为正 | `SUPPORTED_AT_SEED41` |
| 普通图像增强能否替代真实多视角 | Exact-state 中 Image-Aug 远弱于 Broad64 | `NOT_SUPPORTED` |
| same-state pairing 是否优于同覆盖 unpaired | M-A 中差异小且区间宽 | `UNRESOLVED`, M-B 三 seed 运行中 |
| 信息视角是否能改善完整闭环 | Broad practical 出现正信息特异性 | `DIRECTIONAL_SUPPORT`, 独立组数不足 |
| 腕部相机是否构成捷径或必要通道 | external-only 全部失败，wrist-only 明显退化 | `STRONG_DEPENDENCY_DETECTED` |
| Accel 能否直接选择闭环最优视角 | 多数选择 canonical，成功率无稳定提升 | `HOLD` |
| 视角训练是否损害原始任务能力 | Original LIBERO Full 尚未完成 | `PENDING` |

最稳妥的当前表述是：

> 在 Pi0.5 上，宽且真实的相机位姿覆盖已经显示出明显的被动视角鲁棒性收益，并使部分强信息视角
> 从“视角 OOD”转为策略可利用输入。当前仍不能确认 same-state pairing 的额外价值，也不能确认
> Accel 是有效的视角价值选择器。正式结论必须等待三 seed Camera Full、Original retention 和扩展
> Blind-Reveal 统计。

![宽视角数据与被动鲁棒性](figures/view_revalidation_data_passive_interim.png)

## 2. 研究设计与评价层级

本轮重验证回答三个递进问题：

1. **Coverage**：宽、多样的相机训练数据能否改善被动视角泛化；
2. **Pairing**：固定视角覆盖和训练预算后，same-state pairing 与显式一致性是否提供额外收益；
3. **Information use**：候选视角增加当前任务实体的可见像素后，策略能否把新证据转化为完整闭环收益。

评价分为两类，不混用：

| 类型 | 协议 | 用途 |
|---|---|---|
| 官方 benchmark | LIBERO-Plus Camera Full 1,599 episodes；Original LIBERO Full 2,000 episodes | 正式被动视角泛化和基础能力保持 |
| 严格诊断 | Exact-state；constructed M0/M1；Accel fixed-state | 控制物理状态，解释 coverage、信息、通道和选择机制 |

Exact-state 和 M0/M1 是研究诊断，不等同于官方 LIBERO 或 LIBERO-Plus 总榜分数。

## 3. 数据构造

### 3.1 Broad64 same-state 数据

数据根目录：

`/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad64_stride2`

| 项目 | 数值或定义 |
|---|---|
| 状态 | `VERIFIED`，无生成失败 |
| 原始来源 | 官方 LIBERO HDF5 simulator states 与 action |
| 任务 | 8 个，覆盖 Goal、Object、Spatial、LIBERO-10 |
| 源 episode | 400 |
| 记录数 | 38,193 |
| 划分 | Train 31,440；Validation 2,701；Test 4,052 |
| 划分单位 | episode SHA-256，80/10/10，同一 episode 不跨 split |
| 采样 | frame stride 2，action horizon 10 |
| 图像 | 224×224，JPEG quality 95 |
| 物理控制 | 同 simulator state 恢复，只改变相机位姿；paired 两视角共享动作标签 |
| 落盘规模 | 约 2.7GB |

任务组成：

| 任务 | 记录数 | 诊断作用 |
|---|---:|---|
| Open top drawer and put bowl inside | 5,115 | 抽屉内部和放置区域可见性 |
| Put cream cheese in bowl | 2,687 | 物体与容器可见性 |
| Put wine bottle on rack | 4,416 | 细目标几何与放置可见性 |
| Cream cheese to basket | 3,612 | 宽视角下物体身份 |
| Bowl from top drawer to plate | 3,754 | 抽屉遮挡和空间关系 |
| Bowl in bottom drawer and close | 6,232 | 长程抽屉交互 |
| Mug in microwave and close | 7,631 | 电器内部和长程交互 |
| Book to back compartment | 4,746 | 隔间遮挡和放置 |

### 3.2 相机支持范围

| 视角集 | 数量 | 水平环绕角 | 俯仰角 | 距离比例 | 用途 |
|---|---:|---:|---:|---:|---|
| Legacy narrow | 8 | `[-12°, +12°]` | `[-7°, +7°]` | `[0.94, 1.06]` | 旧实验锚点 |
| Broad training | 64 | 约 `[-60°, +60°]` | 约 `[-25°, +25°]` | `[0.90, 1.25]` | 正式宽视角训练 |
| Broad held-out | 32 | 约 `[-58°, +56°]` | 约 `[-24°, +24°]` | `[0.91, 1.24]` | 同范围留出评测 |
| Wide extrapolation | 24 | 约 `[-84°, +79°]` | 约 `[-37°, +39°]` | `[0.75, 1.50]` | 范围外泛化 |
| Extreme orbit | 8 | `[-180°, +180°]` | `[-60°, +45°]` | `[0.50, 2.00]` | 评测和可证伪诊断，不训练 |

Extreme、look-away、blackout 不进入正常动作训练。它们只用于证明相机信息被破坏时策略是否退化，
以及可见性和候选排序是否能识别这种退化。

### 3.3 数据设计相对旧实验的修正

1. 把旧 8 个局部 pose 扩为 64 个大范围 pose；
2. 使用完整 simulator state 恢复，保证 same-state pair 的物理状态和动作相同；
3. 把 `Canonical-unique` 与 `Canonical-repeat` 分开，避免把固定视角专化和精确重复混在一起；
4. 把 practical unpaired、state-matched unpaired、paired FM、paired+consistency 分开；
5. 把正常可操作宽视角与无信息极端视角分开，后者不作为同动作训练样本。

## 4. 训练设置

### 4.1 七个机制对照

| 方法 | 独立状态预算 | same-state 两视角 | 显式一致性 | 回答的问题 |
|---|---:|---:|---:|---|
| Canonical-unique | 高 | 否 | 否 | 普通固定视角 continuation |
| Canonical-repeat | 低，每状态重复 | 否 | 否 | 精确重复与有效 batch 降低的影响 |
| Image-Aug unique | 高 | 否 | 否 | 颜色、裁剪增强能否替代真实相机变化 |
| Broad practical | 高 | 否 | 否 | 相同图像预算下普通宽覆盖的实用上限 |
| Broad state-matched | 与 paired 相同 | 否 | 否 | 严格控制独立状态数和曝光数 |
| Broad paired FM | 与 state-matched 相同 | 是 | 否 | paired 数据本身是否有价值 |
| Broad paired consistency | 与 paired FM 相同 | 是 | 是 | 显式跨视角 action-flow 一致性是否有额外价值 |

### 4.2 模型和优化

| 项目 | 设置 |
|---|---|
| 初始化 | Pi0.5 LIBERO PyTorch checkpoint + PaliGemma 3B PT 224 |
| 总参数 | 约 4.163B |
| 冻结参数 | 约 3.465B，冻结语言模型、LM head 和 multimodal projector |
| 可训练参数 | 约 698M，包含 action expert 和视觉低秩适配 |
| 视觉适配 | SigLIP MLP `fc1/fc2`，54 层，rank 16，4,713,984 参数 |
| 输入 | external + wrist 两路有效图像，第三槽位 masked |
| 动作 | 7 维，prediction horizon 10，flow inference steps 10 |
| 优化器 | AdamW，action/base LR `1e-4`，flow head `5e-5`，VLM interface `5e-4` |
| 正式预算 | 2,000 update，global batch 32，BF16，2 GPU/run |
| 正式 seed | 41、42、43 |
| 记录 | W&B offline，原始数据、视频和 checkpoint 不上传 |

因此这不是纯 action-only，也不是只训练一个很小的 VLM LoRA。更准确的描述是：

> 冻结 PaliGemma 语言主干，在视觉编码器上训练低秩适配，同时训练 Pi0.5 action expert。

训练日志显示 OpenPI 到 AlphaBrain 权重桥接匹配 `814/935` 个目标张量，121 个框架特定张量缺失；
所有训练臂共享该初始化链路。该差异不会破坏训练臂之间的相对比较，但使 Original LIBERO retention
以及同框架 canonical 对照成为必要审计，不能只依赖 Candidate 与原生 OpenPI 的单一绝对比较。

### 4.3 M-B 正式训练

| 方法 | Seed | 状态 |
|---|---|---|
| Broad practical | 41、42、43 | `COMPLETE` |
| Broad paired consistency | 41、42、43 | `COMPLETE`，seed 41 复用同拓扑有效 checkpoint |

所有 M-B run 均采用 2 GPU、global batch 32、2,000 update。每个自包含 checkpoint 约 17.61GB。

## 5. Phase A/B：被动视角泛化结果

### 5.1 Exact-state 开发门

规模：7 方法 × 7 条件 × 24 个严格配对源 episode，共 1,176 条完整闭环 episode；跨方法物理状态
哈希不一致为 0。主统计单位为 source HDF5 episode，不把同一 episode 的派生帧当独立样本。

| 方法 | Canonical | Broad held-out | Wide extrapolation |
|---|---:|---:|---:|
| Canonical-unique | 91.7% | 33.3% | 25.0% |
| Canonical-repeat | 83.3% | 33.3% | 16.7% |
| Image-Aug unique | 91.7% | 25.0% | 16.7% |
| Broad practical | 91.7% | 87.5% | 91.7% |
| Broad state-matched | 95.8% | 83.3% | 83.3% |
| Broad paired FM | 95.8% | 79.2% | 79.2% |
| Broad paired consistency | 91.7% | 83.3% | 83.3% |

Broad practical 相对 Canonical-unique：

| 条件 | 差值 | 配对 cluster bootstrap 95% CI |
|---|---:|---:|
| Canonical | 0.0pp | `[-16.7, +16.7]` |
| Broad held-out | +54.2pp | `[+29.2, +75.0]` |
| Wide extrapolation | +66.7pp | `[+45.8, +83.3]` |

**分析：**

1. 宽位姿覆盖，而不是普通图像增强，是该开发门中视角泛化提升的主要来源；
2. `Canonical-repeat < Canonical-unique` 的趋势说明精确重复可能额外损害有效样本多样性；
3. Broad practical、state-matched、paired FM、paired consistency 的差异远小于它们与 canonical/image-Aug
   的差异，当前最强因素是 coverage；
4. 24 个独立 episode 只足够做机制筛选，不足以裁决 pairing 的 3pp 级增益。

这组结果修正了旧窄视角 Phase B 的解释：不能再声称“图像增强可以媲美多视角训练”。旧结论只适用于
小范围位姿和旧训练预算。

### 5.2 LIBERO-Plus Camera Full seed-41 锚点

Camera Full 包含 1,599 个官方相机扰动任务，来自 40 个独立 base task，每个变体一次完整闭环，
replan `K=5`。

| 模型 | LIBERO-10 | Goal | Object | Spatial | Pooled 1,599 |
|---|---:|---:|---:|---:|---:|
| Official Pi0.5 frozen | 55.6% | 79.9% | 88.6% | 80.6% | 75.9% |
| Broad64 practical, seed 41 | 64.4% | 85.8% | 91.2% | 90.4% | 82.6% |

两种聚合口径必须区分：

- episode pooled：`82.61% - 75.86% = +6.75pp`；
- 先按 40 个 base task 聚合后等权比较：`+5.23pp`，paired cluster bootstrap 95% CI
  `[+0.12, +10.69]`。

最难的 Difficulty-5 从 52.8% 提升到 69.6%，增加 16.8pp；Difficulty-1 从 97.5% 轻微下降到
95.5%。这说明收益主要来自困难视角，而不是所有变体等比例抬升。

**当前判断：**seed 41 已通过预注册的 `+5pp` 被动视角门槛，但正式结论仍等待 seeds 42/43。

## 6. M0：强信息视角构造与筛选

![M0、M1 与 Accel 中期结果](figures/view_revalidation_m0_m1_accel_interim.png)

M0 第一轮保持冻结定义：

\[
I_{task}(v)=\operatorname{mean}_{camera,entity}
[\text{visible-pixel-fraction}(entity,camera,v)]
\]

\[
\Delta I(v)=I_{task}(v)-I_{task}(v_{canonical})
\]

M0 覆盖 180 个状态、15,840 个候选记录，val/test 各包含 3 tasks × 2 source demonstrations ×
15 stages。任务为 wine-rack、bottom-drawer、mug-microwave。

关键分布：

| Test 候选组 | 平均 ΔI | 95% 分位 | 最大值 | 正增量比例 |
|---|---:|---:|---:|---:|
| Broad held-out | -0.0091 | +0.0111 | +0.0367 | 24.2% |
| Wide extrapolation | -0.0121 | +0.0226 | +0.0605 | 14.5% |
| Crossed orbit | +0.0106 | +0.1424 | +0.2085 | 26.0% |
| Extreme orbit | -0.0202 | +0.0174 | +0.0446 | 10.1% |
| Look-away | -0.0299 | -0.0119 | -0.0112 | 0% |
| Sensor controls | -0.0565 | -0.0127 | 0.0000 | 0% |

**分析：**

1. 随机换到一个更宽视角通常不会增加任务实体可见性，信息视角是候选分布中的稀疏上尾；
2. crossed-orbit 出现最高正增量，说明绕到遮挡另一侧确实能暴露额外实体像素；
3. look-away 和 sensor blackout 始终降低可见性，证明负对照方向有效；
4. 因此 M1 不能用“随机新视角”代表信息视角，必须将 Strong-info 与等位姿幅度 Matched-control 配对。

阈值只在 validation 上冻结，再一次性应用到 test；test 未用于调阈值。最终选择 21 个 test 状态：

| 任务 | 状态数 |
|---|---:|
| Bottom drawer | 10 |
| Mug microwave | 9 |
| Wine rack | 2 |

21/21 montage 通过 Codex AI-assisted visual audit，审计不是人工标注者评审。协议状态为 `PASS`，
每个状态冻结 10 个闭环条件，共 210 episode/model。

可见性诊断原图：

`/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/operational-three-task-scan-v2/analysis/visibility_diagnostics.png`

## 7. M1：完整闭环信息利用

5 个模型各运行 210 条完整闭环，共 1,050 episodes；跨模型物理状态哈希不一致为 0。每个状态从同一
冻结中间物理状态开始，策略正常 replanning 到成功或超时。

### 7.1 原始状态成功率

| 模型 | Canonical | Strong-info | Matched-control | Blind |
|---|---:|---:|---:|---:|
| Official | 61.9% | 42.9% | 19.0% | 9.5% |
| Broad practical | 57.1% | 71.4% | 52.4% | 23.8% |
| Broad state-matched | 52.4% | 47.6% | 42.9% | 33.3% |
| Broad paired FM | 52.4% | 52.4% | 52.4% | 19.0% |
| Broad paired consistency | 57.1% | 61.9% | 47.6% | 33.3% |

### 7.2 按 6 条源 demonstration 聚类后的主比较

| 模型 | Info - Canonical | Information Specificity: Info - Control | Info - Blind |
|---|---:|---:|---:|
| Official | -16.7pp `[-56.7,+13.3]` | +16.7pp `[0.0,+36.7]` | +22.5pp `[-4.2,+53.3]` |
| Broad practical | +10.0pp `[-6.7,+33.3]` | **+13.3pp `[+3.3,+26.7]`** | **+34.2pp `[+4.2,+67.5]`** |
| Broad state-matched | -3.3pp `[-20.0,+10.0]` | +4.2pp `[-6.7,+15.8]` | +10.8pp `[+3.3,+19.2]` |
| Broad paired FM | -0.8pp `[-15.8,+16.7]` | -1.7pp `[-21.7,+13.3]` | +22.5pp `[-2.5,+46.7]` |
| Broad paired consistency | +2.5pp `[-9.2,+13.3]` | +11.7pp `[-6.7,+30.0]` | +20.0pp `[0.0,+40.0]` |

状态池直接相减时，Broad practical 的 `Strong-info - Matched-control` 为 +19.0pp；预注册主统计先在
每条源 demonstration 内聚合再等权，得到 +13.3pp。后者避免状态较多的 source demo 获得更大权重。

**分析：**

1. Broad practical 是当前唯一同时具有 Camera Full 被动增益和正向、区间排除 0 的 development
   信息特异性的方法；
2. Official 的正 specificity 不能解释为“会利用信息”，因为 Strong-info 仍比 Canonical 低 19.0pp；
   其正值主要来自 Matched-control 下降得更严重；
3. Paired consistency 的点估计为正，但 95% CI 很宽，尚不能确认一致性目标优于普通宽覆盖；
4. Strong-info external-only 在五个模型中均为 0%，说明当前策略对腕部相机存在强依赖。信息视角收益
   只在 external + wrist 联合输入下成立，不能外推为外部相机单独足够；
5. 只有 6 个独立 source demonstrations，且 wine-rack 只有 2 个状态，因此这是 M-A 机制证据，
   不是论文级最终统计。

## 8. Accel 候选视角分析

Accel 在同一状态、共享 flow noise 下，对 canonical、Strong-info、Matched-control、Blind 四个候选
计算 flow prefix acceleration，并选最小 `accel_3` 的视角。

| 模型 | 21 个状态中的主要选择 | Accel 相对 Canonical 成功率 | 命中效率 oracle | 任一候选可成功 |
|---|---|---:|---:|---:|
| Broad practical | Canonical 18，Control 2，Info 1 | 0.0pp | 38.9% | 85.7% |
| Broad state-matched | Canonical 16，Control 3，Info 2 | +3.3pp | 50.0% | 76.2% |
| Broad paired FM | Canonical 15，Control 4，Info 2 | -2.5pp | 53.3% | 71.4% |
| Broad paired consistency | Canonical 18，Control 1，Blind 2 | -0.8pp | 37.5% | 76.2% |

**分析：**

1. Accel 主要选择 canonical，当前更像训练熟悉度或策略兼容性指标；
2. Broad practical 的 oracle headroom 明显：任一候选成功率 85.7%，高于 canonical 57.1%，但 Accel
   没有捕获该 headroom；
3. 因而不能得出“主动视角没有空间”，只能得出“当前 accel₃ 规则不是可靠的 view-value selector”；
4. Dynamic shortlist 保持 `HOLD`，不进入当前正式方法主张。

## 9. 当前科学判断

### 9.1 已有较强支持

1. **宽相机覆盖是有效因素。** 它在严格 exact-state 和完整 Camera Full 上均优于固定视角与普通图像增强；
2. **信息干预可以被构造并通过负对照验证。** crossed-orbit 有强正尾，look-away/blackout 稳定为负；
3. **训练 support 会影响信息视角是否可用。** Broad practical 在 Strong-info 上表现最好，而 Official
   在新视角下总体退化；
4. **主动候选存在行为 headroom。** 同一状态下确实有其他候选能把失败变成功，但现有 Accel 不会稳定选中。

### 9.2 尚未回答

1. pairing + consistency 是否相对 practical unpaired 稳定增加至少 3pp；
2. Broad64 的 Camera Full 增益能否跨 seeds 41/42/43 保持；
3. 视角训练是否令 Original LIBERO 基础能力下降超过 5pp；
4. 信息特异性是否能在更多任务和更多独立 source demonstrations 上复现；
5. 去除 wrist 后，重新训练 external-only 策略能否利用信息视角；
6. 相机与背景、布局等因素组合时，Broad64 是否仍然稳健；Plus Full 暂未运行；
7. RoboCasa 跨 benchmark 是否复现同样规律。

### 9.3 当前方法风险

| 风险 | 影响 | 处理方式 |
|---|---|---|
| M1 只有 6 个独立源 episode | CI 对任务分布敏感 | 扩到更多 source demos/tasks 后再作最终信息主张 |
| wrist 强依赖 | 外部视角价值可能被固定 wrist 捷径掩盖或补偿 | 增加 wrist-on/off 训练与评测对照 |
| Candidate 使用 AlphaBrain bridge，Official 为原生 OpenPI | 绝对比较包含运行栈差异 | Original retention、同框架 canonical baseline、严格 paired task ledger |
| 当前 Camera Full 正式锚点仅 seed 41 | 可能存在训练 seed 波动 | M-B 三 seed 正在运行 |
| Accel 只看兼容性 | 不能代表任务信息价值 | 保留为分析指标，不作为当前选择算法 |

## 10. 正在执行的 M-B 正式确认

M-B 正式训练已完成，评测控制器正在运行：

`/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/formal_evaluation_controller.log`

执行矩阵：

| 阶段 | 模型数 | 每模型 episode | 总 episode | 状态 |
|---|---:|---:|---:|---|
| Camera Full | 2 方法 × 3 seeds = 6 | 1,599 | 9,594 | `RUNNING` |
| Official Original Full | 1 | 2,000 | 2,000 | `QUEUED` |
| Candidate Original Full | 6 | 2,000 | 12,000 | `QUEUED` |
| 合计 |  |  | 23,594 | 断点续跑、逐 episode 落盘 |

截至本报告生成时，第一项 `Broad practical seed 41 Camera Full` 已由 8 个 shard 并行运行并持续逐
episode 落盘。由于官方任务顺序前段难度不均，运行中的 partial success 不作科学解释；动态进度以
评测根目录和 controller log 为准。评测完成后将输出：

1. 每 seed Camera Full 和 Original Full；
2. 跨 seed 均值与 seed range；
3. 以 40 个 base task 为独立单位的 paired cluster bootstrap 95% CI；
4. Broad practical 相对 Official 的被动视角增益；
5. Paired consistency 相对 practical 的 pairing 增益；
6. Original retention 是否满足不下降超过 5pp。

## 11. 下一步裁决

M-B 结束后按以下顺序裁决：

1. 若 Broad practical 的 Camera Full 三 seed 均为正、跨 seed 主差异至少 +5pp 且 CI 排除 0，确认
   宽视角 coverage 是稳定主因素；
2. 若 paired consistency 相对 practical 至少 +3pp 且跨 seed 同向，保留显式 consistency；否则将
   论文主线收缩为 coverage 与 information-use 的关系，不强行主张 pairing；
3. 若 Original Full 下降超过 5pp，先解决遗忘或训练部署差异，不发布视角改进结论；
4. 扩大 M1 到更多独立任务和 source demonstrations，确认信息特异性而非少数任务效应；
5. 只有固定候选 oracle headroom 继续存在，且新 selector 能稳定接近 oracle，才恢复动态主动视角；
6. Plus Full 与 RoboCasa 作为后续组合泛化和跨 benchmark 验证，不阻塞当前 M-B。

## 12. 可复现入口

| 内容 | 路径 |
|---|---|
| 总协议 | `docs/dsol_paper1/view_revalidation_master_protocol_v3_zh.md` |
| M-A 决策 | `docs/dsol_paper1/view_revalidation_m_a_results_20260820_zh.md` |
| 数据计划 | `configs/dsol_paper1/libero_pair_broad64_quick_gate_v1.json` |
| 训练配置 | `configs/experiments/dsol_libero_broad_pairing.yaml` |
| M-B 训练完成回执 | `/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/training/completion.json` |
| M0 原始汇总 | `/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/operational-three-task-scan-v2/analysis/summary.json` |
| M1 原始汇总 | `/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m1-v2/cross-model-analysis/metrics.json` |
| Camera Full 锚点 | `/share/longjunyu/alphabrain/experiments/libero-plus-camera-full-v1/` |
| Accel 联合结果 | `/share/longjunyu/alphabrain/experiments/dsol-accel-constructed-v2/m1-joins/` |
| 本报告绘图脚本 | `scripts/dsol_paper1/plot_view_revalidation_interim.py` |

报告中的两张汇总图可由以下命令从原始 JSON 重新生成：

```bash
/alphabrain/.venv/bin/python \
  scripts/dsol_paper1/plot_view_revalidation_interim.py
```
