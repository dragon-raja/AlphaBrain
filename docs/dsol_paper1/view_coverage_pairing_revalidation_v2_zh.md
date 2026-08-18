# 视角覆盖、配对与信息视角重验证协议 v2

状态：`PREFREEZE_REVIEW_REQUIRED`
适用范围：RoboCasa 主实验、LIBERO-Plus 外部验证
当前限制：Target 数据迁移通过独立回执前，不启动正式训练或论文级评测。

## 1. 修订原因

旧 Phase B/M0/M1 保留为 `LEGACY_ANCHOR_ONLY`，但不能继续支撑宽泛结论：

- 训练相机只有 8 个局部 pose cells，外部相机约为 `+/-4 cm, +/-4 deg`，腕部相机约为 `+/-4 mm, +/-1 deg`；
- `dual` 表示外部与腕部相机都受扰动，不表示同一状态同时输入两个外部视角；
- MV-paired 虽然保存了同状态、同动作的两视角曝光，但仍使用逐样本 flow-matching，模型没有显式使用 pair identity；
- Canonical-repeat 同时改变了固定视角、独立状态数和 batch 内重复率，不能单独解释固定视角继续训练的影响；
- 原 M0 的视角差异和任务信息差异偏小，原 M1 又以短 action chunk 为主，无法裁决完整闭环中的信息收益。

因此，旧结果只能表述为：

> 在 8 个小范围视角、有限 LoRA 训练和普通 flow-matching 目标下，普通图像增强、unpaired 与 paired-FM 没有显示稳定差异。

不得表述为“图像增强普遍等价于多视角训练”“pairing 没有价值”或“主动视角没有价值”。

## 2. 核心研究问题

本轮按顺序回答五个问题：

1. **Coverage：**扩大且加密正常可观测视角后，被动视角鲁棒性是否继续提高？
2. **Pairing：**在相同视角支持、图像预算和状态预算下，同状态配对是否优于 unpaired？
3. **Objective：**paired 数据只有被显式一致性目标使用时，是否产生额外收益？
4. **Information use：**模型能否利用比 canonical 包含更多任务实体可见像素的视角，而不只是容忍相机变化？
5. **Active-ready：**在明显 Blind/Reveal 差异下，推理期候选选择是否有真实闭环空间？

Coverage、pairing 与 objective 是三个独立变量，不允许再次合并解释。

## 3. 视角域分层

| 层级 | 定义 | 训练 | 评测 |
|---|---|---:|---:|
| Narrow legacy | 原 8 个小范围 cells | 是，作为锚点 | 是 |
| Broad-informative | 连续、大范围采样，任务对象与操作区域仍具有足够可见性 | 是，主训练域 | 是 |
| Reveal/info-support | 相比 canonical 显著增加任务相关实体可见像素，仍是正常图像 | 独立训练对照 | 是 |
| Extreme geometric | Look-away、强遮挡、目标离开视野 | 否 | 是，负对照/候选筛选 |
| Sensor blackout | 单路或全部相机置黑 | 否 | 是，视觉依赖与通道捷径消融 |

Broad-informative 的候选初始覆盖建议为外部相机平移约 `10-30 cm`、姿态变化约 `10-45 deg`，但最终范围必须由逐任务可见性与碰撞审计冻结，不能只按位姿数值生成。Look-away 与黑屏是两种不同干预：前者是有真实但无关的图像，后者是传感器缺失；二者必须分开报告。

## 4. 第一阶段信息视角定义

本轮 M0 先使用可解释的任务实体可见性，不引入决策信息、未来动作、策略输出或 Accel：

1. 从任务元数据解析当前任务相关实体集合；
2. 用 simulator segmentation 记录每个实体在每个部署相机中的可见像素；
3. 对预先冻结的任务实体与相机通道等权聚合，保留每个实体、每路相机的原始像素统计；
4. 对同一物理状态计算

   `Delta V(v) = V_task(v) - V_task(canonical)`；

5. `Delta V` 高于独立校准集冻结阈值的候选称为信息视角；位姿变化相近但 `Delta V` 接近零的候选称为 matched-control。

必须同时保存 `V_left`、`V_right`、`V_wrist`，防止聚合分数掩盖单路相机信息消失。阈值只能用校准任务和训练侧状态冻结，不能在正式测试结果上调参。

M0 准入 M1 的必要条件：

- Blind 与 Reveal 的差异可由图像和 segmentation 双重确认；
- Reveal 的 `Delta V` 明显高于 legacy 候选；
- matched-control 与 Reveal 的相机运动量相近，但 `Delta V` 接近零；
- wrist-on 与 wrist-off 均单独审计；
- 候选分布覆盖多个任务族，不由单一任务贡献主要样本。

如果明显的人工 Blind/Reveal 仍不能被该分数区分，结论是“可见性指标需要修订”，不是“信息视角无效”。

## 5. 正式训练矩阵

所有新训练默认使用同一个 Pi0.5 初始化、相同 LoRA 可训练参数、optimizer、更新步数、action horizon、数据顺序规则和 seeds。Pi0.5 LoRA 同时采用 `gemma_2b_lora` 与 `gemma_300m_lora` 配置；不得在不同 arm 间改变可训练模块。

| Arm | 独立物理状态 | 每状态曝光 | 视角组织 | 目标 |
|---|---:|---:|---|---|
| Official-frozen | - | - | 不继续训练 | 原始能力锚点 |
| Canonical-unique | 高 | 1 | 每状态一个 canonical | 隔离固定视角 continuation |
| Canonical-repeat | 低 | 2 个相同 RGB | 同状态精确重复 | 匹配 paired 的曝光/batch 结构 |
| ImageAug-unique | 与 Canonical-unique 相同 | 1 | canonical 加普通像素增强 | 无 repeat 混杂的主图像增强基线 |
| Legacy-ImageAug-repeat | 与 Canonical-repeat 相同 | 2 | 旧 repeat 结构加随机像素增强 | 旧 Phase B 锚点 |
| Legacy-MV8-unpaired | 旧账本 | 2 个相同 RGB | 原 8 个小 cells，状态间换视角 | 旧 unpaired 锚点 |
| Legacy-MV8-paired-FM | 旧账本 | 2 个不同视角 | 原 8 个小 cells，同状态配对，普通 FM | 旧 paired 锚点 |
| Broad-unpaired-practical | 最高 | 1 | 不同状态独立采样 broad pose | 同图像预算下的实用强基线 |
| Broad-unpaired-state-matched | 与 paired 相同 | 2 个相同 RGB | 每状态只给一个 broad pose；pair 内共享 flow noise/time | 严格隔离 same-state correspondence |
| Broad-paired-FM | 与上项相同 | 2 个不同视角 | 同状态、同动作；pair 内共享相同 flow draw，仍只用 FM | pairing 数据本身 |
| Broad-paired-consistency | 与上项完全相同 | 2 个不同视角 | 在相同 flow draw 的 FM 上增加 action-flow consistency | 显式利用 pairing |
| Info-pose-support | 与测试独立 | 1 或 2 | 正常 Reveal/info poses 进入训练支持 | 区分信息无效与视角 OOD |

`Broad-unpaired-practical` 与 paired 的 unique-state 数不同，因此只能回答“实践中相同图像预算哪种更强”；严格的 pairing 因果效应必须比较 `Broad-unpaired-state-matched`、`Broad-paired-FM` 和 `Broad-paired-consistency`。

## 6. 覆盖量与数据量必须正交

### 6.1 固定曝光预算，改变视角支持

比较 `Narrow-8 -> Broad-32 -> Broad-64`，保持 source windows、总 RGB 曝光、optimizer updates 和 seeds 一致。

### 6.2 固定视角支持，改变数据规模

比较 `1x -> 2x -> 4x` exposure，并报告 unique states、重复率和每个 pose cell 的曝光量。不得把“更多视角”和“更多训练数据”合并成一个增益。

### 6.3 数据生成审计

每个 arm 在训练前必须输出：

- episode/state/action/image identity 与 SHA-256；
- pose histogram、平移/旋转分布和训练到测试最近邻距离；
- unique states、unique RGB、重复率、每状态曝光数；
- task/scene/episode 分布与 train/calibration/test 隔离；
- task-entity 可见像素、目标面积、遮挡代理和图像边界触碰率；
- episode-fixed、window-random 或 transition 的明确标记；
- wrist/external 的独立分布。

任一 arm 的 pose marginal、状态预算或曝光预算与预期不符，先停止训练，不能事后用成功率解释。

## 7. 训练与算力协议

- 训练方式：Pi0.5 LoRA，不进行全参数微调；
- 目标 global batch：32；先在 8 x RTX 5090 上做 20-50 step 显存与吞吐 smoke，再冻结所有 arms 能共同承受的最大 batch；
- 若 batch 32 不稳定，可统一降到 16，但禁止不同 arms 使用不同有效 batch；
- seeds：至少 3 个训练 seed；pilot 单 seed 只能用于排错；
- checkpoint：用预先冻结的统一更新步选择，不允许各 arm 挑各自最优测试 checkpoint；
- W&B：只记录超参数、训练/验证指标、seed、Git commit 和 summary；不上传数据、视频、checkpoint、权重或敏感信息，网络异常自动离线。

## 8. 扩大后的评测

### 8.1 被动鲁棒性

统一报告：canonical、legacy perturbation、broad interpolation、broad extrapolation、Reveal、Look-away、third-black、all-black、external-only、wrist-only、wrist-off，以及新场景与 broad camera 的联合扰动。

### 8.2 M0：同状态候选筛选

对同一 snapshot 渲染 canonical、strong-info、medium-info、matched-control、Blind 和 Reveal。M0 只验证候选确实形成预期的可见性差异，不使用策略成功率来定义候选。

### 8.3 M1：完整闭环

从完全相同的初始状态运行完整任务，按正常频率 replanning，直至正式任务成功或超时。主要比较：

- Info vs Canonical；
- Info vs Matched-control；
- Reveal vs Blind；
- Wrist-on vs Wrist-off；
- Legacy-MV vs Broad-MV；
- Broad-MV vs Info-pose-support。

主要指标为 full-task success、stage progress、Rescue、Harm、completion steps 和信息特异性
`(SR_info - SR_canonical) - (SR_control - SR_canonical)`。动作误差仅作辅助指标。

## 9. 两个环境的职责

### RoboCasa

主实验环境，用于 Human300/Target 分布、同状态重渲染、宽范围相机、wrist shortcut、人工 Blind/Reveal 与完整闭环。正式全分布结论必须同时覆盖 pretrain 与 held-out Target，并以 task x scene seed/snapshot group 为独立统计单位。

### LIBERO-Plus

外部有效性与强覆盖基线，不替代 RoboCasa。官方 camera track 已提供 C1/C2/C3 大范围扰动；官方多样化训练结果证明宽覆盖普通训练是强基线，但没有隔离 same-state pairing。重验证优先比较：

1. broad unpaired ordinary FM；
2. same-state paired ordinary FM；
3. same-state paired + consistency；
4. nominal ID 与官方 camera track。

RoboCasa 与 LIBERO-Plus 分开报告，不把不同模型、腕部输入或训练规模的数值直接合并。

外部证据的使用边界：

- [LIBERO-Plus](https://arxiv.org/html/2510.13626) 的相机域包含距离缩放 `1.01x-2.00x`、球面位置变化 `15-75 deg` 和朝向变化 `2-10 deg`；其 22,400 条采集轨迹经筛选后保留超过 20,000 条，说明大范围数据覆盖必须作为强基线，但其主要训练模型是 OpenVLA-OFT，不能直接代替 Pi0.5/RoboCasa 结论。
- [CVC](https://arxiv.org/html/2608.06965) 在 Pi0.5 上使用 338,575 个同状态 pair；同一 paired 数据下，显式一致性从 `79.8 +/- 0.8%` 提升到 `87.2 +/- 0.4%`。这支持新增 paired-consistency arm，同时也说明 paired-FM 与 paired-consistency 必须分开。CVC 屏蔽 wrist，因此本研究仍须单独完成 wrist-on/off 控制。

## 10. Accel 的位置

Accel 仅在上述 coverage/pairing/M0/M1 通过后作为推理期候选关系分析：它可以判断模型偏好 canonical、训练支持、Reveal 还是闭环 oracle，但不能替代可见性定义、数据对照或完整闭环。若 Accel 总选 canonical，只能说明它可能偏向训练熟悉度；若与闭环 oracle 不一致，只能说明它不是完整 view value。

## 11. 样本充分性与停止门

- pilot 至少覆盖 6 个任务族、每任务 50 个 paired states；仅用于检查分布、M0 分离度和训练链路；
- 正式结论前必须进行基于 task x scene seed/snapshot group 的功效分析，并冻结样本量；
- 每个训练 arm 至少 3 seeds，报告逐 seed 和跨 seed 结果；
- 使用 paired group-level bootstrap 95% CI，不能把同组帧当成独立样本；
- Broad 目录若没有显著扩大视觉/位姿分布，停止训练并重建目录；
- Official/Canonical-unique 若学不会基础任务，标记 baseline invalid，不比较小幅方法差异；
- Extreme/black 只用于负对照，不能因其失败判定 broad multiview 无效；
- Target 迁移与 SHA-256 回执未完成时，不启动正式全分布评测。

## 12. 结果解释规则

| 结果 | 可支持的结论 |
|---|---|
| Broad-unpaired 明显优于 Legacy-MV-8 | 旧结果受窄视角支持限制，coverage 是主要因素 |
| Canonical-unique 正常、repeat 退化 | 原 canonical 退化主要来自精确重复和有效 batch 降低 |
| Canonical-unique 与 repeat 都退化 | 固定视角 continuation 造成专化/遗忘 |
| Paired-FM 约等于 state-matched unpaired | 普通 FM 下 pair identity 没被有效利用 |
| Paired-consistency 优于 paired-FM | 显式跨视角约束提供算法增益 |
| Info-pose-support 后 Reveal 才有效 | 原失败主要是视角 OOD，而非信息无价值 |
| Reveal 优于 matched-control | 收益对新增任务可见性具有特异性 |
| Wrist-off 后收益才出现 | 腕部相机是主要视觉捷径，必须作为论文控制变量 |
| Oracle Reveal 也不优于 canonical | 当前任务/候选池没有足够主动视角空间，应停止该任务设置 |

这份协议通过评审并冻结前，不生成新的论文级结论。
