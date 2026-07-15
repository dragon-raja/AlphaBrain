# 恢复支持扩展预注册

状态：等待 320-step expert-handoff Gate 完成后启动。本文先固定问题、对照和反证规则，不预设方法名称或正结果。

## 研究问题

当前需要解决的不是“如何避开近期工作的相似点”，而是：

> 当失败后果已经可见，但 Pi0.5 在自己的恢复状态分布中无法产生可完成任务的动作序列时，最小需要补充哪一段动作支持，才能让无搜索、`N=1` 策略重新进入已有的成功吸引域？

已有正式证据说明：

- 精确 suffix weighting 没有提高闭环恢复；
- receding sibling Oracle 相对 sample0 的 formal success 仅提高约 0.37 个百分点，区间跨 0，transport 反而下降；
- 当前候选对 formal success 和 transport 的 action leverage 为 0；
- continuation variation 明显大于局部 action effect，大部分候选处于并列；
- current-only 输入不能预测当前 noisy Oracle，单个 observation-action-consequence 历史也尚未可靠超过机会水平；
- Full-H 只训练了一轮完整数据，seed 42 的 attached success 为 0，baseline 稳定性仍不足。

因此停止当前 exact sibling weighting teacher，但不停止失败恢复问题。下一实验优先区分三个根因：训练不足、阶段曝光不均，以及策略诱导状态上的正确动作覆盖不足。

## 工作假设：最小充分恢复段

expert-handoff ladder 从同一 feedback state 比较：

1. policy only；
2. teacher 3 actions；
3. teacher 12 actions；
4. teacher 到稳定重抓；
5. teacher 到稳定抬升；
6. teacher 到 transport；
7. full teacher sanity。

所有 policy handoff 使用相同的反馈后 320-step 总预算、相同 policy seed schedule 和相同观测频率。full teacher 必须达到至少 90% source-cluster success，否则先修复 controller 或预算，不进入训练。

选择“最小充分段”的规则在看正式结果前固定为：找到第一个相对 policy-only 的 formal-success 提升至少 20 个百分点、source-cluster paired 95% CI 下界大于 0、且三个 checkpoint seed 同向的里程碑。若没有里程碑满足，则不训练本路线，先判定 downstream policy/controller 或任务数据不足。

这个里程碑只定义需要补充的动作支持长度，不作为部署时的 privileged signal。

## 三臂等预算实验

每个 seed 都从其现有 Full-H final checkpoint 开始，做完全相同数量的额外更新。这样 Base continuation 会直接检验原来的一轮训练是否不足，也避免把“训练更久”误归因给新数据。

### A. Base continuation

- 继续使用原始完整 episode windows；
- 原始 seeded shuffle；
- 不做阶段平衡，不加入新状态；
- 其余 optimizer、学习率、冻结模块和更新数与 B/C 相同。

### B. Stage-balanced replay

- targets 仍全部来自原有成功 teacher episodes；
- 不增加任何新 observation 或 action；
- 一半更新为原始均匀 demonstration anchor；
- 一半更新在 `regrasp / lift / transport / place` 四个阶段间等概率采样；
- 同一 snapshot group 内再均匀选 window，避免长 episode 仅凭帧数获得更高权重。

该臂只测试训练曝光与阶段不平衡，不声称增加动作支持。

### C. Policy-state recovery SFT

- anchor slots 与 B 完全相同；
- targeted slots 使用 base policy 在训练 split 上实际到达的失败/恢复偏离状态；
- 从该状态由同一 teacher 执行到预注册的最小充分里程碑；
- 只保留通过物理 milestone 和后续 policy/full-teacher 可达性审计的 correction segment；
- policy 输入仍只有当前可部署图像、语言和现有 proprioception；branch、object pose、contact、未来状态只用于 collector/audit；
- 不加入 preference loss、critic、dynamic K、horizon head 或 world model。

该臂测试的是 on-policy state coverage 和正确恢复动作支持，而不是更复杂的损失。

## 数据与计算公平性

- seeds：`41, 42, 43`；
- train/val/test 继续按 source snapshot group 隔离；
- 只在 train split 收集 correction states；
- 三臂使用相同初始化、额外更新数、optimizer、scheduler、batch size、冻结模块和 fixed K；
- B/C 的 anchor/target slot schedule 相同；
- B/C 每个 targeted slot 对应相同 source-group schedule；
- 所有方法推理时均为 `N=1`，不使用 simulator、teacher、reranker 或额外观察频率；
- 不按方法选择不同 checkpoint；
- 正式运行前先用 Base continuation 在 validation 上确认统一额外更新预算，test 只打开一次。

## 主指标与裁决

主指标只使用 fixed-K 闭环行为：

- slip full-task recovery success；
- overall full-task success；
- attached/no-intervention success；
- regrasp、lift、transport、place 子目标；
- failure continuation、premature commitment、drop 和 completion steps。

统计以 source snapshot group 为独立单位，报告每 seed、跨 seed 均值和 paired group-level bootstrap 95% CI。

### 结果 1：训练不足

若 A 已明显提高恢复与 attached success，且 B/C 没有稳定额外收益，则根因主要是原 baseline 未收敛。停止包装新方法，采用更充分的 Full-H 训练。

### 结果 2：阶段曝光不足

若 B 稳定优于 A，且 C 不优于 B，则采用简单 stage-balanced replay。该结果更接近 SARM2/VLAC-CUT 式数据利用，不主张反事实动作信用贡献。

### 结果 3：策略状态覆盖不足

若 C 相对 B 的 slip recovery 或 overall success 提高至少 10 个百分点，paired CI 下界大于 0，且 attached 退化不超过 5 个百分点，则支持“最小充分恢复段蒸馏”工作假设。

只有在随后与等交互预算的普通 full-recovery SFT、VLAC-CUT-style curation、PACT-style corrective-action alignment 和 DICE-RL-style residual improvement 比较后仍有数据效率或成功率优势，才讨论方法贡献。

### 结果 4：离线支持扩展不足

若 C 扩大候选 action coverage 但 `N=1` 无闭环收益，只允许增加一个等交互预算的 DICE-RL-style on-policy residual 对照。若三臂均不能提高闭环恢复，则停止离线 weighting/SFT 路线，转向 feedback-aware replanning 或 on-policy RL；不继续加工 Oracle 边界。

## 近邻与原创性纪律

- PACT 已使用 intervention state、corrective action 和 counterfactual credit；
- VLAC-CUT 已做 progress/failure/recovery segment curation 与 human-in-the-loop post-training；
- SARM2/SPIRAL 已做 stage reward 和 on-policy improvement；
- AFIL 已从失败 rollout 形成负指导；
- DICE-RL 已用 generative prior、residual off-policy RL、行为约束和 value-guided selection 提升长程操作。

这些工作不是需要回避的名字，而是可能已经给出正确答案的强基线。如果简单 replay、direct SFT 或 residual RL 解决了问题，就采用它们。当前项目只有在“自动找到最小充分恢复边界，并以更少 correction 数据恢复无搜索策略”这一实质效果成立时，才有额外主张；否则最有价值的交付是可信的瓶颈诊断和负结果。
